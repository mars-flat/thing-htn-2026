"""Pure-torch twins of the fused Triton kernels (see notes/SPEC.md, "Kernels").

Every Triton kernel in this package has a twin here with the SAME signature and
the SAME arithmetic, so the engine can (a) self-test each kernel against its
twin at warmup and (b) fall back to the twin, per op, when a kernel is missing
or mismatches. The twins are therefore written to be:

* device-agnostic (cpu / mps / cuda), bf16 in and out;
* bit-exact with Transformers 4.51.3 ``modeling_qwen3`` on CPU and CUDA — the
  cast placement is the reference's, and every bf16 rounding the reference
  performs is performed here, explicitly, at the same point;
* CUDA-graph capturable: no ``.item()`` / ``.tolist()`` / host syncs, and no
  data-dependent shapes (attention masks over the full cache capacity instead
  of slicing to the valid length).

Rounding points, for the record (``bf16(.)`` = round-to-nearest-even):

    RMSNorm   : xf = fp32(x); var = mean(xf^2); n = xf * rsqrt(var + eps)
                y = w * bf16(n)                     # n rounded BEFORE the gain
    add+norm  : h = bf16(fp32(x) + fp32(r)); y = RMSNorm(h)
    silu*up   : s = bf16(silu_fp32(g)); out = bf16(fp32(s) * fp32(u))
    RoPE      : p1 = bf16(x * cos); p2 = bf16(rotate_half(x) * sin)
                out = bf16(p1 + p2)                 # three roundings
    attention : scores = fp32(q) . fp32(k) * scale; m = rowmax; e = exp(s - m)
                out = bf16( (bf16(e) @ fp32(v)) / sum(e) )   # flash-style P rounding

Products and sums of two bf16 values are computed in fp32 and rounded once,
which is exactly what torch's bf16 elementwise ops do on every backend, so
``bf16(fp32(a) * fp32(b)) == a * b`` bit for bit.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "ref_rms_norm",
    "ref_add_rms_norm",
    "ref_silu_mul",
    "ref_qk_norm_rope_cache",
    "ref_attn_decode",
]


# --------------------------------------------------------------------------- norms


def ref_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``Qwen3RMSNorm.forward``: RMSNorm over the last dim of ``x`` with gain ``w``.

    ``x`` is any shape ``[..., N]`` (bf16), ``w`` is ``[N]`` in ``x``'s dtype.
    The op sequence below is the reference's, line for line: fp32 reduction,
    the normalised value rounded to bf16, THEN multiplied by the bf16 gain
    (that product is exact in fp32 and rounded once by the bf16 multiply).
    """
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(variance + eps)
    return w * normed.to(x.dtype)


def ref_add_rms_norm(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float):
    """Fused residual add + RMSNorm: ``h = bf16(x + r)``, ``y = RMSNorm(h, w)``.

    Returns ``(h, y)``; ``h`` is the new residual stream (one bf16 rounding of
    the fp32 sum, exactly what the reference's bf16 ``residual + hidden`` does).
    """
    h = (x.to(torch.float32) + r.to(torch.float32)).to(x.dtype)
    return h, ref_rms_norm(h, w, eps)


# --------------------------------------------------------------------------- mlp


def ref_silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` for the fused gate/up projection ``gu [..., 2I]``.

    gate = columns ``[0, I)``, up = columns ``[I, 2I)``. ``F.silu`` on a bf16
    tensor computes ``g / (1 + exp(-g))`` in fp32 and rounds to bf16 once (CPU
    and CUDA kernels alike), which is precisely the reference's
    ``act_fn(gate_proj(x))``; the product with ``up`` is then rounded once.
    The gate half is made contiguous so the silu kernel sees the same memory
    layout the reference sees (a contiguous ``[rows, I]`` tensor).
    """
    inter = gu.shape[-1] // 2
    gate = gu[..., :inter].contiguous()
    up = gu[..., inter:]
    s = F.silu(gate)
    return (s.to(torch.float32) * up.to(torch.float32)).to(gu.dtype)


# --------------------------------------------------------------------------- rope + cache


def _rope_bf16(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_pos_emb`` for one tensor, emulating its three bf16 roundings.

    ``x [..., D]`` bf16; ``cos``/``sin`` bf16, broadcastable to ``x``.
    """
    xf = x.to(torch.float32)
    half = xf.shape[-1] // 2
    rotated = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
    p1 = (xf * cos.to(torch.float32)).to(x.dtype)
    p2 = (rotated * sin.to(torch.float32)).to(x.dtype)
    return (p1.to(torch.float32) + p2.to(torch.float32)).to(x.dtype)


def ref_qk_norm_rope_cache(
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
    """Per-head q/k RMSNorm, RoPE, and K/V cache write for ``T`` tokens per sequence.

    ``qkv [M, (Nq + 2*Nkv) * D]`` with ``M = B * T``; row ``m = b*T + t`` is
    token ``t`` of sequence ``b`` at absolute position ``positions[b] + t``.
    Column layout: ``Nq`` q heads, then ``Nkv`` k heads, then ``Nkv`` v heads,
    ``D`` wide each (the order ``cat(q_proj, k_proj, v_proj)`` produces).

    q heads: RMSNorm over ``D`` with ``q_norm_w`` then RoPE at that position;
    returned as ``[B, T, Nq, D]`` bf16 (contiguous).
    k heads: same with ``k_norm_w``, written to ``k_cache[b, kvh, pos, :]``.
    v heads: copied to ``v_cache[b, kvh, pos, :]`` (V is not normalised).
    Caches are ``[B, Nkv, CAP, D]`` and are written in place; ``positions`` is
    ``[B]`` int32 (or int64), ``cos_tab``/``sin_tab`` are ``[P, D]`` bf16 with
    ``P > max(positions) + T - 1``. ``T`` is a Python int (static shape).
    """
    M, width = qkv.shape
    B = M // T
    D = q_norm_w.shape[0]
    Nkv = k_cache.shape[1]
    Nq = width // D - 2 * Nkv
    device = qkv.device

    q = qkv[:, : Nq * D].reshape(B, T, Nq, D)
    k = qkv[:, Nq * D : (Nq + Nkv) * D].reshape(B, T, Nkv, D)
    v = qkv[:, (Nq + Nkv) * D :].reshape(B, T, Nkv, D)

    # Absolute position of every (b, t), and the rope rows for it: [B, T, 1, D]
    # so they broadcast over the head axis.
    pos = positions.to(torch.int64)[:, None] + torch.arange(T, device=device, dtype=torch.int64)[None, :]
    cos = cos_tab[pos].unsqueeze(2)
    sin = sin_tab[pos].unsqueeze(2)

    q_rot = _rope_bf16(ref_rms_norm(q, q_norm_w, eps), cos, sin)
    k_rot = _rope_bf16(ref_rms_norm(k, k_norm_w, eps), cos, sin)

    # cache[b, :, pos[b, t], :] = new[b, t, :, :]. Indexing the [B, CAP, Nkv, D]
    # transposed view with (b, pos) index tensors of shape [B, T] puts the
    # advanced dims first, so the value shape is [B, T, Nkv, D] as produced.
    # index_put_ on the view writes through to the cache storage, launches no
    # host sync, and its shapes depend only on (B, T), never on values.
    b_idx = torch.arange(B, device=device, dtype=torch.int64)[:, None].expand(B, T)
    k_cache.transpose(1, 2).index_put_((b_idx, pos), k_rot.to(k_cache.dtype))
    v_cache.transpose(1, 2).index_put_((b_idx, pos), v.to(v_cache.dtype))
    return q_rot


# --------------------------------------------------------------------------- attention


def ref_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    num_splits: int,
) -> torch.Tensor:
    """Decode attention of ``T`` new queries per sequence over the static KV cache.

    ``q [B, T, Nq, D]`` bf16; ``k_cache``/``v_cache [B, Nkv, CAP, D]`` bf16;
    ``lengths [B]`` int32: sequence ``b`` has ``lengths[b]`` valid cached tokens,
    the last ``T`` of which are the new tokens (already written by
    ``qk_norm_rope_cache``). Query ``t`` of sequence ``b`` attends key ``j`` iff
    ``j < lengths[b] - (T - 1 - t)``. Query head ``h`` reads KV head
    ``h // (Nq // Nkv)``. Scores in fp32; the un-normalised probabilities are
    rounded to bf16 before the value matmul (as flash attention and the Triton
    kernel do), the fp32 row sum divides at the end, result rounded to bf16 once.
    Returns ``[B, T, Nq * D]`` bf16, contiguous.

    The mask covers the full capacity ``CAP`` (``arange(CAP) < limit``), so no
    shape depends on ``lengths`` and the op is CUDA-graph capturable.
    ``num_splits`` is accepted for signature parity with the Triton kernel and
    ignored (the result does not depend on the split count).
    """
    del num_splits
    B, T, Nq, D = q.shape
    Nkv, CAP = k_cache.shape[1], k_cache.shape[2]
    G = Nq // Nkv
    device = q.device

    # Query rows grouped by KV head: [B, Nkv, G*T, D], row = g*T + t.
    qf = q.to(torch.float32).reshape(B, T, Nkv, G, D).permute(0, 2, 3, 1, 4).reshape(B, Nkv, G * T, D)
    kf = k_cache.to(torch.float32)
    vf = v_cache.to(torch.float32)

    key_idx = torch.arange(CAP, device=device, dtype=torch.int64)
    t_idx = torch.arange(T, device=device, dtype=torch.int64)
    len64 = lengths.to(torch.int64)
    limit = len64[:, None] - (T - 1 - t_idx)[None, :]                 # [B, T]
    visible = key_idx[None, None, :] < limit[:, :, None]              # [B, T, CAP]
    visible_rows = visible[:, None, None, :, :].expand(B, 1, G, T, CAP).reshape(B, 1, G * T, CAP)

    # Cache slots at or beyond lengths[b] may hold stale or uninitialised data
    # (possibly NaN/Inf). Masked scores are overwritten below, but a NaN in V
    # would still poison 0 * NaN in the value matmul, so zero those slots.
    key_valid = key_idx[None, :] < len64[:, None]                     # [B, CAP]
    vf = vf.masked_fill(~key_valid[:, None, :, None], 0.0)

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale          # [B, Nkv, G*T, CAP] fp32
    scores = scores.masked_fill(~visible_rows, float("-inf"))
    # Flash-style softmax (what the reference's SDPA backend and the Triton
    # kernel do): un-normalised P = exp(s - max) rounded to bf16 for the PV
    # product, fp32 row sum l kept separately, one division at the end.
    mx = scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores - mx)                                        # masked -> 0
    l = e.sum(dim=-1, keepdim=True)
    out = torch.matmul(e.to(q.dtype).to(torch.float32), vf) / l      # [B, Nkv, G*T, D] fp32
    out = out.reshape(B, Nkv, G, T, D).permute(0, 3, 1, 2, 4)         # [B, T, Nkv, G, D]
    return out.to(q.dtype).reshape(B, T, Nq * D).contiguous()
