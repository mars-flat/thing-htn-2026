"""Bit-exactness tests for engine/kernels/torch_ref.py against Transformers 4.51.3 Qwen3.

Run from the project root:
    .venv/bin/python -m unittest tests/test_torch_ref.py

Norm / silu / rope twins must be bit-exact (torch.equal) with the HF modules
on CPU; the attention twin is compared to SDPA over the valid cache slice with
a tolerance, since only the reduction order differs.
"""

import ast
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from kernels import torch_ref  # noqa: E402
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

BF16 = torch.bfloat16
EPS = 1e-6
D = 128
NQ = 8
NKV = 2
THETA = 5e6


def _bf16(*shape, gen, scale=1.0):
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(BF16)


def _norm_module(n, gen):
    mod = Qwen3RMSNorm(n, eps=EPS).to(BF16)
    with torch.no_grad():
        mod.weight.copy_((1.0 + 0.5 * torch.randn(n, generator=gen)).to(BF16))
    return mod


def _config():
    return Qwen3Config(
        vocab_size=1024,
        hidden_size=NQ * D,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=D,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )


class RMSNormTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(1234)

    def test_rms_norm_bit_exact(self):
        for rows, n in [(1, 128), (6, 2560), (37, 1024), (3, 96)]:
            with self.subTest(rows=rows, n=n):
                mod = _norm_module(n, self.gen)
                x = _bf16(rows, n, gen=self.gen, scale=3.0)
                with torch.no_grad():
                    want = mod(x)
                got = torch_ref.ref_rms_norm(x, mod.weight.data, EPS)
                self.assertEqual(got.dtype, BF16)
                self.assertEqual(got.shape, x.shape)
                self.assertTrue(torch.equal(got, want))

    def test_rms_norm_flat_rows_match_3d_reference(self):
        # The engine normalises [B*T, H]; the reference normalises [B, T, H].
        b, t, n = 2, 5, 2560
        mod = _norm_module(n, self.gen)
        x3 = _bf16(b, t, n, gen=self.gen, scale=3.0)
        with torch.no_grad():
            want = mod(x3).reshape(b * t, n)
        got = torch_ref.ref_rms_norm(x3.reshape(b * t, n), mod.weight.data, EPS)
        self.assertTrue(torch.equal(got, want))

    def test_add_rms_norm_bit_exact(self):
        for rows, n in [(1, 128), (6, 2560), (37, 1024)]:
            with self.subTest(rows=rows, n=n):
                mod = _norm_module(n, self.gen)
                x = _bf16(rows, n, gen=self.gen, scale=3.0)
                r = _bf16(rows, n, gen=self.gen, scale=3.0)
                with torch.no_grad():
                    h_want = x + r
                    y_want = mod(h_want)
                h_got, y_got = torch_ref.ref_add_rms_norm(x, r, mod.weight.data, EPS)
                self.assertEqual(h_got.dtype, BF16)
                self.assertEqual(y_got.dtype, BF16)
                self.assertTrue(torch.equal(h_got, h_want))
                self.assertTrue(torch.equal(y_got, y_want))


class SiluMulTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(42)

    def test_silu_mul_bit_exact(self):
        for rows, inter in [(1, 64), (7, 3072), (13, 9728), (5, 300)]:
            with self.subTest(rows=rows, inter=inter):
                g = _bf16(rows, inter, gen=self.gen, scale=4.0)
                u = _bf16(rows, inter, gen=self.gen, scale=4.0)
                gu = torch.cat([g, u], dim=-1)
                want = F.silu(g) * u
                got = torch_ref.ref_silu_mul(gu)
                self.assertEqual(got.dtype, BF16)
                self.assertEqual(got.shape, (rows, inter))
                self.assertTrue(torch.equal(got, want))


class RopeCacheTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(7)
        self.cfg = _config()
        self.rope = Qwen3RotaryEmbedding(self.cfg)
        self.q_norm = _norm_module(D, self.gen)
        self.k_norm = _norm_module(D, self.gen)
        self.P = 256
        dummy = torch.zeros(1, dtype=BF16)
        with torch.no_grad():
            cos_tab, sin_tab = self.rope(dummy, torch.arange(self.P)[None, :])
        self.cos_tab = cos_tab[0].contiguous()
        self.sin_tab = sin_tab[0].contiguous()
        self.assertEqual(self.cos_tab.shape, (self.P, D))
        self.assertEqual(self.cos_tab.dtype, BF16)

    def test_rope_table_is_gatherable(self):
        # The engine gathers rows from a precomputed [P, D] table; HF computes
        # cos/sin for exactly the positions it needs. Both must agree.
        position_ids = torch.tensor([[5, 6, 7], [20, 21, 22]])
        dummy = torch.zeros(1, dtype=BF16)
        with torch.no_grad():
            cos_hf, sin_hf = self.rope(dummy, position_ids)
        self.assertTrue(torch.equal(cos_hf, self.cos_tab[position_ids]))
        self.assertTrue(torch.equal(sin_hf, self.sin_tab[position_ids]))
        # Spec: cos[:, :64] == cos[:, 64:].
        self.assertTrue(torch.equal(self.cos_tab[:, : D // 2], self.cos_tab[:, D // 2 :]))

    def _run(self, T, positions, B=2, CAP=64):
        width = (NQ + 2 * NKV) * D
        qkv = _bf16(B * T, width, gen=self.gen, scale=2.0)
        k_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        v_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        k_before, v_before = k_cache.clone(), v_cache.clone()
        qkv_before = qkv.clone()
        pos_t = torch.tensor(positions, dtype=torch.int32)

        # --- reference: Qwen3Attention.forward up to (and including) RoPE.
        position_ids = pos_t.to(torch.int64)[:, None] + torch.arange(T)[None, :]
        dummy = torch.zeros(1, dtype=BF16)
        with torch.no_grad():
            cos, sin = self.rope(dummy, position_ids)
            q_in = qkv[:, : NQ * D].contiguous().view(B, T, NQ, D)
            k_in = qkv[:, NQ * D : (NQ + NKV) * D].contiguous().view(B, T, NKV, D)
            v_in = qkv[:, (NQ + NKV) * D :].contiguous().view(B, T, NKV, D)
            q_hf = self.q_norm(q_in).transpose(1, 2)
            k_hf = self.k_norm(k_in).transpose(1, 2)
            v_hf = v_in.transpose(1, 2)
            q_hf, k_hf = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)

        # --- twin
        q_got = torch_ref.ref_qk_norm_rope_cache(
            qkv, self.q_norm.weight.data, self.k_norm.weight.data, EPS,
            self.cos_tab, self.sin_tab, pos_t, T, k_cache, v_cache,
        )
        self.assertEqual(q_got.shape, (B, T, NQ, D))
        self.assertEqual(q_got.dtype, BF16)
        self.assertTrue(q_got.is_contiguous())
        self.assertTrue(torch.equal(qkv, qkv_before), "input must not be modified")
        self.assertTrue(torch.equal(q_got, q_hf.transpose(1, 2)))

        # Cache writes at positions[b] + t, everything else untouched.
        touched = torch.zeros(B, CAP, dtype=torch.bool)
        for b in range(B):
            for t in range(T):
                p = positions[b] + t
                touched[b, p] = True
                self.assertTrue(torch.equal(k_cache[b, :, p, :], k_hf[b, :, t, :]))
                self.assertTrue(torch.equal(v_cache[b, :, p, :], v_hf[b, :, t, :]))
        keep = ~touched[:, None, :, None].expand_as(k_cache)
        self.assertTrue(torch.equal(k_cache[keep], k_before[keep]))
        self.assertTrue(torch.equal(v_cache[keep], v_before[keep]))

    def test_t1(self):
        self._run(T=1, positions=[5, 40])

    def test_t3(self):
        self._run(T=3, positions=[0, 17])

    def test_prefill_like_from_zero(self):
        self._run(T=9, positions=[0, 0], B=2, CAP=64)

    def test_batch_slice_view_of_cache(self):
        # Prefill hands the twin k_cache[l][b0:b1], a view; writes must land in
        # the parent tensor.
        B, T, CAP = 3, 2, 64
        width = (NQ + 2 * NKV) * D
        k_full = _bf16(B, NKV, CAP, D, gen=self.gen)
        v_full = _bf16(B, NKV, CAP, D, gen=self.gen)
        k_copy, v_copy = k_full.clone(), v_full.clone()
        qkv = _bf16(2 * T, width, gen=self.gen, scale=2.0)
        pos_t = torch.tensor([3, 9], dtype=torch.int32)
        torch_ref.ref_qk_norm_rope_cache(
            qkv, self.q_norm.weight.data, self.k_norm.weight.data, EPS,
            self.cos_tab, self.sin_tab, pos_t, T, k_full[1:3], v_full[1:3],
        )
        self.assertTrue(torch.equal(k_full[0], k_copy[0]))
        self.assertTrue(torch.equal(v_full[0], v_copy[0]))
        self.assertFalse(torch.equal(k_full[1, :, 3:5], k_copy[1, :, 3:5]))
        self.assertFalse(torch.equal(v_full[2, :, 9:11], v_copy[2, :, 9:11]))
        self.assertTrue(torch.equal(k_full[1, :, 5:], k_copy[1, :, 5:]))


class AttnDecodeTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(99)
        self.scale = D ** -0.5

    def _reference(self, q, k_cache, v_cache, lengths, T):
        """SDPA over the valid slice of each sequence, with the spec's mask."""
        B = q.shape[0]
        G = NQ // NKV
        outs = []
        for b in range(B):
            L = int(lengths[b])
            kb = repeat_kv(k_cache[b : b + 1, :, :L, :], G)   # [1, Nq, L, D]
            vb = repeat_kv(v_cache[b : b + 1, :, :L, :], G)
            qb = q[b : b + 1].transpose(1, 2)                  # [1, Nq, T, D]
            if T == 1:
                mask = None                                     # HF decode: no mask
            else:
                j = torch.arange(L)[None, :]
                t = torch.arange(T)[:, None]
                mask = j < (L - (T - 1 - t))                    # [T, L] bool, True = attend
            with torch.no_grad():
                o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, scale=self.scale)
            outs.append(o.transpose(1, 2).reshape(1, T, NQ * D))
        return torch.cat(outs, dim=0)

    def _case(self, T, lengths, B=2, CAP=64, poison=False):
        q = _bf16(B, T, NQ, D, gen=self.gen)
        k_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        v_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        len_t = torch.tensor(lengths, dtype=torch.int32)
        want = self._reference(q, k_cache, v_cache, lengths, T)
        if poison:
            for b in range(B):
                k_cache[b, :, lengths[b] :, :] = float("nan")
                v_cache[b, :, lengths[b] :, :] = float("inf")
        got = torch_ref.ref_attn_decode(q, k_cache, v_cache, len_t, self.scale, 4)
        self.assertEqual(got.shape, (B, T, NQ * D))
        self.assertEqual(got.dtype, BF16)
        self.assertTrue(got.is_contiguous())
        self.assertTrue(torch.isfinite(got.float()).all())
        self.assertTrue(
            torch.allclose(got.float(), want.float(), atol=2e-2, rtol=0),
            f"max abs diff {(got.float() - want.float()).abs().max().item():.4g}",
        )
        return q, k_cache, v_cache, got, want

    def test_t1(self):
        self._case(T=1, lengths=[13, 40])

    def test_t1_full_capacity(self):
        self._case(T=1, lengths=[64, 64])

    def test_t3_mixed_lengths(self):
        self._case(T=3, lengths=[13, 40])

    def test_t3_full_capacity(self):
        self._case(T=3, lengths=[64, 30])

    def test_stale_cache_beyond_length_is_ignored(self):
        self._case(T=3, lengths=[13, 40], poison=True)
        self._case(T=1, lengths=[1, 64], poison=True)

    def test_single_key_returns_value_exactly(self):
        # lengths = 1, T = 1: one visible key, softmax weight exactly 1.0.
        B, CAP = 2, 64
        q = _bf16(B, 1, NQ, D, gen=self.gen)
        k_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        v_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        got = torch_ref.ref_attn_decode(q, k_cache, v_cache, torch.tensor([1, 1], dtype=torch.int32), self.scale, 1)
        G = NQ // NKV
        want = v_cache[:, :, 0, :][:, :, None, :].expand(B, NKV, G, D).reshape(B, 1, NQ * D)
        self.assertTrue(torch.equal(got, want))

    def test_mask_is_load_bearing(self):
        # With T = 3 the earlier queries must NOT see the later new tokens.
        B, T, CAP = 2, 3, 64
        lengths = [13, 40]
        q = _bf16(B, T, NQ, D, gen=self.gen)
        k_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        v_cache = _bf16(B, NKV, CAP, D, gen=self.gen)
        got = torch_ref.ref_attn_decode(q, k_cache, v_cache, torch.tensor(lengths, dtype=torch.int32), self.scale, 1)
        # Wrong reference: every query sees all `lengths[b]` keys.
        G = NQ // NKV
        wrong = []
        for b in range(B):
            L = lengths[b]
            kb = repeat_kv(k_cache[b : b + 1, :, :L, :], G)
            vb = repeat_kv(v_cache[b : b + 1, :, :L, :], G)
            qb = q[b : b + 1].transpose(1, 2)
            o = F.scaled_dot_product_attention(qb, kb, vb, scale=self.scale)
            wrong.append(o.transpose(1, 2).reshape(1, T, NQ * D))
        wrong = torch.cat(wrong)
        # Last query sees everything in both formulations...
        self.assertTrue(torch.allclose(got[:, -1].float(), wrong[:, -1].float(), atol=2e-2, rtol=0))
        # ...earlier queries do not.
        self.assertFalse(torch.allclose(got[:, 0].float(), wrong[:, 0].float(), atol=2e-2, rtol=0))

    def test_kv_head_mapping(self):
        # Query head h must read KV head h // G: make KV heads distinguishable.
        B, T, CAP = 1, 1, 8
        G = NQ // NKV
        q = torch.zeros(B, T, NQ, D, dtype=BF16)
        k_cache = torch.zeros(B, NKV, CAP, D, dtype=BF16)
        v_cache = torch.zeros(B, NKV, CAP, D, dtype=BF16)
        for kvh in range(NKV):
            v_cache[0, kvh, :, :] = float(kvh + 1)
        got = torch_ref.ref_attn_decode(q, k_cache, v_cache, torch.tensor([3], dtype=torch.int32), self.scale, 1)
        got = got.view(B, T, NQ, D)
        for h in range(NQ):
            self.assertTrue(torch.equal(got[0, 0, h], torch.full((D,), float(h // G + 1), dtype=BF16)))


class CapturabilityTests(unittest.TestCase):
    """Static checks on the twins' source: no host syncs, no data-dependent shapes, no triton."""

    SYNC_ATTRS = {"item", "tolist", "cpu", "numpy", "nonzero", "masked_select", "synchronize", "unique"}

    def _module_ast(self):
        return ast.parse((ROOT / "engine" / "kernels" / "torch_ref.py").read_text())

    def test_no_host_sync_calls_in_code(self):
        offenders = []
        for node in ast.walk(self._module_ast()):
            if isinstance(node, ast.Attribute) and node.attr in self.SYNC_ATTRS:
                offenders.append(f"line {node.lineno}: .{node.attr}")
            if isinstance(node, ast.Subscript):
                # Boolean-mask indexing (x[mask]) has a data-dependent shape; the
                # twins only ever index with slices, ints, or integer index tensors.
                # Cheap proxy: forbid subscripting with a `~expr` or comparison.
                sl = node.slice
                if isinstance(sl, ast.Compare) or (isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.Invert)):
                    offenders.append(f"line {node.lineno}: boolean-mask subscript")
        self.assertEqual(offenders, [])

    def test_no_triton_import(self):
        for node in ast.walk(self._module_ast()):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                self.assertFalse(name.split(".")[0] == "triton", f"torch_ref imports {name}")


@unittest.skipUnless(
    torch.backends.mps.is_available() and os.environ.get("DRYFT_TEST_MPS", "1") != "0",
    "MPS not available",
)
class MpsSmokeTests(unittest.TestCase):
    """Device-agnosticism: the twins run on MPS and agree with CPU to tolerance."""

    def test_all_twins_on_mps(self):
        gen = torch.Generator().manual_seed(3)
        dev = torch.device("mps")
        B, T, CAP = 2, 3, 64
        width = (NQ + 2 * NKV) * D
        x = _bf16(B * T, 256, gen=gen, scale=3.0)
        r = _bf16(B * T, 256, gen=gen, scale=3.0)
        w = (1.0 + 0.5 * torch.randn(256, generator=gen)).to(BF16)
        gu = _bf16(B * T, 2 * 96, gen=gen, scale=4.0)
        qkv = _bf16(B * T, width, gen=gen, scale=2.0)
        qw = (1.0 + 0.5 * torch.randn(D, generator=gen)).to(BF16)
        kw = (1.0 + 0.5 * torch.randn(D, generator=gen)).to(BF16)
        cfg = _config()
        rope = Qwen3RotaryEmbedding(cfg)
        with torch.no_grad():
            cos_tab, sin_tab = rope(torch.zeros(1, dtype=BF16), torch.arange(128)[None, :])
        cos_tab, sin_tab = cos_tab[0].contiguous(), sin_tab[0].contiguous()
        k_cache = _bf16(B, NKV, CAP, D, gen=gen)
        v_cache = _bf16(B, NKV, CAP, D, gen=gen)
        pos = torch.tensor([5, 20], dtype=torch.int32)
        lengths = pos + T

        def run(device):
            to = lambda t: t.to(device)
            kc, vc = to(k_cache.clone()), to(v_cache.clone())
            y = torch_ref.ref_rms_norm(to(x), to(w), EPS)
            h, y2 = torch_ref.ref_add_rms_norm(to(x), to(r), to(w), EPS)
            s = torch_ref.ref_silu_mul(to(gu))
            q = torch_ref.ref_qk_norm_rope_cache(to(qkv), to(qw), to(kw), EPS, to(cos_tab), to(sin_tab), to(pos), T, kc, vc)
            a = torch_ref.ref_attn_decode(q, kc, vc, to(lengths), D ** -0.5, 2)
            return [t.float().cpu() for t in (y, h, y2, s, q, kc, vc, a)]

        cpu = run(torch.device("cpu"))
        mps = run(dev)
        names = ["rms_norm", "add_h", "add_y", "silu_mul", "q", "k_cache", "v_cache", "attn"]
        for name, a, b in zip(names, cpu, mps):
            with self.subTest(op=name):
                self.assertTrue(torch.isfinite(b).all())
                self.assertTrue(torch.allclose(a, b, atol=2e-2, rtol=2e-2), f"max abs diff {(a - b).abs().max():.4g}")


if __name__ == "__main__":
    unittest.main()
