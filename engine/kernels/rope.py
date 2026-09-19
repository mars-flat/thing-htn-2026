"""Fused per-head RMSNorm + RoPE + KV-cache write for Qwen3 (SPEC.md kernel 3).

One program per (token row, group of HB heads). For q heads the program
normalises each 128-wide head with ``q_norm_w``, applies RoPE at the token's
absolute position and writes ``q[b, t, h, :]``; for k heads it does the same
with ``k_norm_w`` and writes ``k_cache[b, kvh, pos, :]``; for v heads it copies
the raw values to ``v_cache[b, kvh, pos, :]``. Nothing else in the caches is
touched. Head groups are homogeneous (HB divides Nq and Nkv), so each program
takes exactly one of the three paths.

Numerics follow ``Qwen3RMSNorm`` and ``apply_rotary_pos_emb`` (Transformers
4.51.3) operation for operation. Every place the reference rounds to bf16, this
kernel rounds to bf16; every bf16 x bf16 product is formed in fp32 (exact) and
rounded once, exactly like PyTorch's bf16 elementwise kernels. The variance is
``sum(x*x) * (1/128)`` (torch's mean multiplies by the reciprocal; 1/128 is
exact) and rsqrt is libdevice's ``__nv_rsqrtf`` (what torch's CUDA rsqrt
compiles to) when available. Only the fp32 sum-of-squares reduction order may
differ from PyTorch, which SPEC.md allows.

The module imports without Triton (``HAS_TRITON`` is False and the launcher
raises); ``self_test`` compares the kernel with ``kernels.torch_ref``.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - local machines without CUDA
    triton = None
    tl = None
    HAS_TRITON = False

_libdevice = None
if HAS_TRITON:
    try:
        from triton.language.extra.cuda import libdevice as _libdevice
    except Exception:  # pragma: no cover
        try:
            from triton.language.extra import libdevice as _libdevice
        except Exception:
            _libdevice = None
HAS_LIBDEVICE = _libdevice is not None

#: Qwen3 head size; the kernel is specialised for it (two 64-wide halves).
HEAD_DIM = 128
#: Heads per program. Must divide Nq and Nkv (32/8 for 4B, 16/8 for 0.6B).
HEADS_PER_PROGRAM = 8
#: [HB, 64] fp32 tiles: two warps is plenty.
NUM_WARPS = 2


if HAS_TRITON:
    if HAS_LIBDEVICE:

        @triton.jit
        def _rsqrt_f32(x):
            return _libdevice.rsqrt(x)

    else:

        @triton.jit
        def _rsqrt_f32(x):
            return tl.math.rsqrt(x)

    @triton.jit
    def _norm_rope_tile(src, w_ptr, cos_row, sin_row, dst, eps, HALF: tl.constexpr):
        """RMSNorm (gain ``w_ptr``) then RoPE on an [HB, 2*HALF] tile of heads.

        ``src``/``dst`` are [HB, HALF] pointer tiles at element 0 of each head
        (bf16); ``cos_row``/``sin_row`` point at row ``pos`` of the [P, 2*HALF]
        tables. Heads are handled as two halves so rotate_half needs no
        permutation: out[:HALF] pairs x[:HALF] with -x[HALF:], out[HALF:]
        pairs x[HALF:] with x[:HALF].
        """
        d = tl.arange(0, HALF)
        x1 = tl.load(src).to(tl.float32)            # [HB, HALF]
        x2 = tl.load(src + HALF).to(tl.float32)

        # Qwen3RMSNorm: fp32 squares (individually rounded; fp fusion is
        # disabled at launch), fp32 sum, exact multiply by 1/128, fp32 rsqrt.
        var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) * (1.0 / (2 * HALF))
        r = _rsqrt_f32(var + eps)                   # [HB]

        # Cast placement: the normalised value is rounded to bf16 BEFORE the
        # gain multiply (`weight * hidden_states.to(bf16)`), product rounded once.
        n1 = (x1 * r[:, None]).to(tl.bfloat16).to(tl.float32)
        n2 = (x2 * r[:, None]).to(tl.bfloat16).to(tl.float32)
        w1 = tl.load(w_ptr + d).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
        y1 = (n1 * w1[None, :]).to(tl.bfloat16).to(tl.float32)
        y2 = (n2 * w2[None, :]).to(tl.bfloat16).to(tl.float32)

        # apply_rotary_pos_emb in bf16: (y * cos) + (rotate_half(y) * sin) with
        # rotate_half(y) = cat(-y2, y1). Three bf16 roundings per output.
        c1 = tl.load(cos_row + d).to(tl.float32)
        c2 = tl.load(cos_row + HALF + d).to(tl.float32)
        s1 = tl.load(sin_row + d).to(tl.float32)
        s2 = tl.load(sin_row + HALF + d).to(tl.float32)

        p1 = (y1 * c1[None, :]).to(tl.bfloat16).to(tl.float32)
        p2 = ((-y2) * s1[None, :]).to(tl.bfloat16).to(tl.float32)
        o1 = (p1 + p2).to(tl.bfloat16)

        p3 = (y2 * c2[None, :]).to(tl.bfloat16).to(tl.float32)
        p4 = (y1 * s2[None, :]).to(tl.bfloat16).to(tl.float32)
        o2 = (p3 + p4).to(tl.bfloat16)

        tl.store(dst, o1)
        tl.store(dst + HALF, o2)

    @triton.jit(do_not_specialize=[10, 11])  # T, CAP: runtime, one binary for prefill/decode/verify
    def _qk_norm_rope_cache_kernel(
        qkv_ptr,  # [M, (NQ + 2*NKV) * D] bf16
        qw_ptr,  # [D] bf16
        kw_ptr,  # [D] bf16
        cos_ptr,  # [P, D] bf16
        sin_ptr,  # [P, D] bf16
        pos_ptr,  # [B] int32
        q_ptr,  # [B, T, NQ, D] bf16 (out)
        k_cache_ptr,  # [B, NKV, CAP, D] bf16
        v_cache_ptr,  # [B, NKV, CAP, D] bf16
        eps,
        T,
        CAP,
        NQ: tl.constexpr,
        NKV: tl.constexpr,
        D: tl.constexpr,
        HB: tl.constexpr,
    ):
        tl.static_assert(D == 128, "qk_norm_rope_cache is specialised for head_dim 128")
        tl.static_assert(NQ % HB == 0, "HB must divide Nq")
        tl.static_assert(NKV % HB == 0, "HB must divide Nkv")
        # int64 everywhere the offsets can be large (rows x width, cache slots).
        m = tl.program_id(0).to(tl.int64)
        g = tl.program_id(1)                        # head group
        b = m // T
        t = m - b * T
        pos = tl.load(pos_ptr + b).to(tl.int64) + t
        tab = pos * D

        hh = tl.arange(0, HB)                       # [HB]
        d = tl.arange(0, D // 2)                      # [HALF]
        head0 = g * HB
        src = qkv_ptr + m * ((NQ + 2 * NKV) * D) + (head0 + hh)[:, None] * D + d[None, :]
        if g < NQ // HB:
            dst = q_ptr + (m * NQ + head0 + hh)[:, None] * D + d[None, :]
            _norm_rope_tile(src, qw_ptr, cos_ptr + tab, sin_ptr + tab, dst, eps, D // 2)
        else:
            if g < (NQ + NKV) // HB:
                kvh = head0 - NQ + hh                                   # [HB]
                dst = k_cache_ptr + ((b * NKV + kvh) * CAP + pos)[:, None] * D + d[None, :]
                _norm_rope_tile(src, kw_ptr, cos_ptr + tab, sin_ptr + tab, dst, eps, D // 2)
            else:
                kvh = head0 - NQ - NKV + hh
                dst = v_cache_ptr + ((b * NKV + kvh) * CAP + pos)[:, None] * D + d[None, :]
                tl.store(dst, tl.load(src))
                tl.store(dst + D // 2, tl.load(src + D // 2))


# Launch options beyond num_warps. ``enable_fp_fusion=False`` keeps LLVM from
# contracting the squares into the reduction's adds (the reference rounds each
# square). Dropped automatically if this Triton does not know the option.
_EXTRA_LAUNCH_OPTS = {"enable_fp_fusion": False}


def qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    eps: float,
    cos_tab: torch.Tensor,
    sin_tab: torch.Tensor,
    positions: torch.Tensor,
    T: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    """q/k RMSNorm + RoPE, K/V cache write; returns q ``[B, T, Nq, D]`` bf16.

    ``qkv`` is ``[M, (Nq + 2*Nkv) * D]`` bf16 contiguous with ``M = B*T``; row
    ``m = b*T + t`` is token ``t`` of sequence ``b`` at absolute position
    ``positions[b] + t`` (``positions`` int32 ``[B]``). Heads are laid out
    q(0..Nq-1), k(0..Nkv-1), v(0..Nkv-1) along the row. ``cos_tab``/``sin_tab``
    are ``[P, D]`` bf16 and must cover every position written; ``k_cache``/
    ``v_cache`` are ``[B, Nkv, CAP, D]`` bf16 contiguous and are written in
    place at ``[b, kvh, positions[b] + t, :]``.

    No host synchronisation, no data-dependent shapes: safe to capture in a
    CUDA graph (``q`` is allocated with ``torch.empty`` inside).
    """
    if not HAS_TRITON:
        raise RuntimeError("qk_norm_rope_cache needs Triton; use kernels.torch_ref.ref_qk_norm_rope_cache")
    B, Nkv, CAP, D = k_cache.shape
    M, W = qkv.shape
    assert D == HEAD_DIM, f"head_dim must be {HEAD_DIM}, got {D}"
    assert W % D == 0 and W // D > 2 * Nkv, f"qkv width {W} does not hold Nq + 2*{Nkv} heads of {D}"
    Nq = W // D - 2 * Nkv
    HB = HEADS_PER_PROGRAM
    assert Nq % HB == 0 and Nkv % HB == 0, f"Nq={Nq}, Nkv={Nkv} must be multiples of {HB}"
    assert M == B * T, f"qkv has {M} rows but B*T = {B}*{T}"
    assert qkv.dtype == torch.bfloat16 and qkv.is_contiguous()
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and v_cache.shape == k_cache.shape
    assert k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
    assert q_norm_w.numel() == D and k_norm_w.numel() == D
    assert q_norm_w.is_contiguous() and k_norm_w.is_contiguous()
    assert cos_tab.dim() == 2 and cos_tab.shape[1] == D and cos_tab.is_contiguous()
    assert sin_tab.shape == cos_tab.shape and sin_tab.is_contiguous()
    assert cos_tab.dtype == torch.bfloat16 and sin_tab.dtype == torch.bfloat16
    assert positions.dtype == torch.int32 and positions.numel() == B and positions.is_contiguous()

    q = torch.empty((B, T, Nq, D), dtype=qkv.dtype, device=qkv.device)
    grid = (M, (Nq + 2 * Nkv) // HB)
    args = (qkv, q_norm_w, k_norm_w, cos_tab, sin_tab, positions, q, k_cache, v_cache, float(eps), T, CAP)
    consts = dict(NQ=Nq, NKV=Nkv, D=D, HB=HB, num_warps=NUM_WARPS)
    global _EXTRA_LAUNCH_OPTS
    try:
        _qk_norm_rope_cache_kernel[grid](*args, **consts, **_EXTRA_LAUNCH_OPTS)
    except KeyError:
        if not _EXTRA_LAUNCH_OPTS:
            raise
        _EXTRA_LAUNCH_OPTS = {}
        _qk_norm_rope_cache_kernel[grid](*args, **consts)
    return q


def build_rope_tables(P: int, theta: float, device, head_dim: int = HEAD_DIM):
    """cos/sin tables ``[P, head_dim]`` bf16 as ``Qwen3RotaryEmbedding`` computes them.

    Test helper; the engine builds its own tables with the reference's exact
    op sequence (see engine._ensure_rope).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(P, device=device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]  # == the reference's K=1 matmul
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


# --------------------------------------------------------------------------
# Self-test


def _import_torch_ref():
    try:
        from .torch_ref import ref_qk_norm_rope_cache  # type: ignore

        return ref_qk_norm_rope_cache
    except Exception:
        pass
    from kernels.torch_ref import ref_qk_norm_rope_cache  # type: ignore

    return ref_qk_norm_rope_cache


def _within(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    """(max |a-b|, all elements within atol + rtol*|b|). bf16-aware: one ulp is |b|/128."""
    if a.numel() == 0:
        return 0.0, True
    diff = (a.float() - b.float()).abs()
    ok = bool((diff <= atol + rtol * b.float().abs()).all().item())
    return float(diff.max().item()), ok


def self_test(device="cuda", atol: float = 1e-2, rtol: float = 1.0 / 64, dims=None, max_T: int | None = None) -> dict:
    """Compare the Triton kernel with the torch twin on a few shapes.

    q and the written k slots must agree within ``atol + rtol*|ref|`` (rtol of
    two bf16 ulps: rsqrt / reduction-order ulps can flip one bf16 rounding),
    written v slots bit-exactly, and no other cache slot may change.
    """
    result = {"ok": True, "device": str(device), "has_triton": HAS_TRITON, "libdevice": HAS_LIBDEVICE, "cases": []}
    if not HAS_TRITON:
        result["ok"] = False
        result["error"] = "triton is not importable"
        return result
    ref_fn = _import_torch_ref()
    dev = torch.device(device)

    cases = [
        # name, B, T, Nq, Nkv, CAP, positions, theta
        ("decode_b1", 1, 1, 32, 8, 640, [517], 5e6),
        ("decode_b4", 4, 1, 32, 8, 640, [0, 3, 517, 639], 5e6),
        ("verify_b2_t5", 2, 5, 32, 8, 640, [7, 630], 5e6),
        ("prefill_b2_t64", 2, 64, 32, 8, 640, [0, 0], 5e6),
        ("small_model_b3_t2", 3, 2, 16, 8, 128, [0, 50, 100], 1e6),
    ]
    gen = torch.Generator(device="cpu").manual_seed(1234)
    for name, B, T, Nq, Nkv, CAP, pos_list, th in cases:
        if dims is not None and (Nq, Nkv) != tuple(dims):
            continue
        if max_T is not None and T > max_T:
            continue
        case = {"name": name, "B": B, "T": T, "Nq": Nq, "Nkv": Nkv, "CAP": CAP, "positions": list(pos_list)}
        try:
            M, W = B * T, (Nq + 2 * Nkv) * HEAD_DIM
            qkv = (torch.randn(M, W, generator=gen) * 3.0).to(torch.bfloat16).to(dev)
            q_w = (1.0 + 0.5 * torch.randn(HEAD_DIM, generator=gen)).to(torch.bfloat16).to(dev)
            k_w = (1.0 + 0.5 * torch.randn(HEAD_DIM, generator=gen)).to(torch.bfloat16).to(dev)
            positions = torch.tensor(pos_list, dtype=torch.int32, device=dev)
            cos_tab, sin_tab = build_rope_tables(CAP, th, dev)
            k0 = torch.randn(B, Nkv, CAP, HEAD_DIM, generator=gen).to(torch.bfloat16).to(dev)
            v0 = torch.randn(B, Nkv, CAP, HEAD_DIM, generator=gen).to(torch.bfloat16).to(dev)

            k_ref, v_ref = k0.clone(), v0.clone()
            q_ref = ref_fn(qkv, q_w, k_w, 1e-6, cos_tab, sin_tab, positions, T, k_ref, v_ref)
            k_got, v_got = k0.clone(), v0.clone()
            q_got = qk_norm_rope_cache(qkv, q_w, k_w, 1e-6, cos_tab, sin_tab, positions, T, k_got, v_got)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

            assert q_got.shape == (B, T, Nq, HEAD_DIM), q_got.shape
            assert q_got.dtype == torch.bfloat16 and q_got.is_contiguous()
            pos = positions.to(torch.int64)[:, None] + torch.arange(T, device=dev)[None, :]
            bidx = torch.arange(B, device=dev)[:, None].expand(B, T)
            written = torch.zeros(B, Nkv, CAP, dtype=torch.bool, device=dev)
            written[bidx, :, pos] = True

            case["q_max_abs_diff"], q_ok = _within(q_got, q_ref.contiguous(), atol, rtol)
            case["k_max_abs_diff"], k_ok = _within(k_got[written], k_ref[written], atol, rtol)
            case["v_exact"] = bool(torch.equal(v_got[written], v_ref[written]))
            case["k_untouched_ok"] = bool(torch.equal(k_got[~written], k0[~written]))
            case["v_untouched_ok"] = bool(torch.equal(v_got[~written], v0[~written]))
            case["ok"] = bool(
                q_ok and k_ok and case["v_exact"] and case["k_untouched_ok"] and case["v_untouched_ok"]
                and torch.isfinite(q_got.float()).all().item()
            )
        except Exception as exc:  # report, never raise: the engine decides what to do
            case["ok"] = False
            case["error"] = f"{type(exc).__name__}: {exc}"
        result["cases"].append(case)
        result["ok"] = result["ok"] and case["ok"]
    return result


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    print(json.dumps(self_test(sys.argv[1] if len(sys.argv) > 1 else "cuda"), indent=1))
