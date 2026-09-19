"""Qwen3's RMSNorm, and the fused residual-add + RMSNorm, in Triton, written to
match the reference exactly.

``rms_norm`` is the worked example that shipped with the template (kept as is,
apart from 64-bit pointer math). ``add_rms_norm`` fuses the decoder layer's
residual add with the norm that immediately follows it:

    h = bf16(x + r)          # fp32 add, rounded ONCE - the tensor HF stores
    y = RMSNorm(h, w, eps)   # computed FROM THE ROUNDED h, as HF reads it back

Reference (Transformers 4.51.3, ``Qwen3RMSNorm.forward``), all in fp32 until
the very end:

    xf  = x.float()
    var = mean(xf * xf, -1)
    n   = xf * rsqrt(var + eps)
    y   = w * n.to(bf16)     # bf16 * bf16, one rounding; cast BEFORE the gain

Only reduction order and rsqrt ulps may differ from the reference; every cast
sits exactly where the reference puts it.

The module imports without ``triton`` (``HAS_TRITON`` is False and the
wrappers raise); the engine then falls back to the pure-torch twins in
``torch_ref``. ``self_test`` compares each kernel against its twin.
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # local dev box without CUDA / triton
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

#: One row must fit in one block. Qwen3 4B norms 2560 columns (hidden) and 128
#: (per-head q/k norm), so both land well inside this.
MAX_BLOCK = 8192


def _num_warps(block: int) -> int:
    # Fixed per block size (no autotune): 4 warps up to 1024 columns, 16 warps
    # for the 4096-wide block that covers the 2560-column hidden state.
    return max(4, min(16, block // 256))


if HAS_TRITON:
    if HAS_LIBDEVICE:

        @triton.jit
        def _rsqrt_f32(x):
            # __nv_rsqrtf: the routine torch's CUDA rsqrt kernel compiles to.
            return _libdevice.rsqrt(x)

    else:

        @triton.jit
        def _rsqrt_f32(x):
            return tl.math.rsqrt(x)

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, row_stride, n_cols, inv_n, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offsets = row * row_stride + cols

        # The reference reduces in fp32 over the whole row. Masked lanes load as
        # zero, so they contribute nothing to the sum of squares.
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # torch's mean multiplies the fp32 sum by the fp32 reciprocal 1/N.
        variance = tl.sum(x * x, axis=0) * inv_n
        normed = x * _rsqrt_f32(variance + eps)

        # Cast placement, and the whole reason this file exists. The reference ends:
        #
        #     return self.weight * hidden_states.to(input_dtype)
        #
        # so the normalised value is rounded to bfloat16 *before* the weight
        # multiply, not after. Keeping the product in fp32 and rounding once at the
        # end is the obvious version, is strictly more accurate, and is wrong: it
        # computes a different function, and on some prompt it moves a logit further
        # than the 2.0 tie margin allows. Reorder arithmetic freely; do not
        # reformulate it.
        weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, normed.to(y_ptr.dtype.element_ty) * weight, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        x_ptr, r_ptr, w_ptr, h_ptr, y_ptr, row_stride, n_cols, inv_n, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offsets = row * row_stride + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Residual add. HF does `residual + hidden_states` on two bf16 tensors:
        # torch adds in fp32 and rounds to bf16 once (round-to-nearest-even,
        # Triton's default for fp32 -> bf16). That rounded tensor is what the
        # next norm reads, so it is both stored and reused below. Do NOT norm
        # the unrounded fp32 sum: that is a reformulation.
        h_bf16 = (x + r).to(h_ptr.dtype.element_ty)
        tl.store(h_ptr + offsets, h_bf16, mask=mask)

        # RMSNorm of the ROUNDED h, identical to _rms_norm_kernel from here on.
        h = h_bf16.to(tl.float32)
        variance = tl.sum(h * h, axis=0) * inv_n
        normed = h * _rsqrt_f32(variance + eps)

        # Same cast placement as above: round to bf16, then multiply by the gain.
        weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, normed.to(y_ptr.dtype.element_ty) * weight, mask=mask)


def _block_for(n_cols: int) -> int:
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(f"a row must fit in one block; {n_cols} columns does not")
    return block


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, matching ``Qwen3RMSNorm.forward``.

    ``x`` is any shape whose last dimension matches ``weight``; ``weight`` is
    the module's learned gain, in the same dtype as ``x``. Returns a new
    tensor of ``x``'s shape. No host syncs; CUDA-graph capturable.
    """
    if not HAS_TRITON:
        raise RuntimeError("rms_norm needs triton; use torch_ref.ref_rms_norm")
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).contiguous()
    n_rows, n_cols = rows.shape
    block = _block_for(n_cols)
    out = torch.empty_like(rows)
    if n_rows == 0:
        return out.reshape(shape)
    _rms_norm_kernel[(n_rows,)](
        rows,
        weight,
        out,
        rows.stride(0),
        n_cols,
        1.0 / n_cols,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )
    return out.reshape(shape)


def add_rms_norm(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor, eps: float):
    """Fused ``h = bf16(x + r)``, ``y = RMSNorm(h, weight, eps)``; returns ``(h, y)``.

    ``x`` and ``r`` have the same shape, whose last dimension matches
    ``weight``; all bf16. ``h`` is the residual stream the caller keeps (the
    exact bf16 tensor HF would hold after ``residual + hidden_states``) and
    ``y`` is the normalised input of the next matmul. Both are new tensors of
    ``x``'s shape. One program per row, BLOCK = next_pow2(N). No host syncs;
    CUDA-graph capturable.
    """
    if not HAS_TRITON:
        raise RuntimeError("add_rms_norm needs triton; use torch_ref.ref_add_rms_norm")
    if x.shape != r.shape:
        raise ValueError(f"x {tuple(x.shape)} and r {tuple(r.shape)} must match")
    shape = x.shape
    x_rows = x.reshape(-1, shape[-1]).contiguous()
    r_rows = r.reshape(-1, shape[-1]).contiguous()
    n_rows, n_cols = x_rows.shape
    block = _block_for(n_cols)
    h = torch.empty_like(x_rows)
    y = torch.empty_like(x_rows)
    if n_rows == 0:
        return h.reshape(shape), y.reshape(shape)
    _add_rms_norm_kernel[(n_rows,)](
        x_rows,
        r_rows,
        weight,
        h,
        y,
        x_rows.stride(0),
        n_cols,
        1.0 / n_cols,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )
    return h.reshape(shape), y.reshape(shape)


# ----------------------------------------------------------------------------
# Self-test against the pure-torch twins
# ----------------------------------------------------------------------------

_TEST_SHAPES = [(7, 2560), (3, 128)]
_TEST_EPS = 1e-6


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest |a - b| as a Python float; inf on shape/dtype mismatch, nan on nan."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    return (a.float() - b.float()).abs().max().item()


def _excess(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> float:
    """max(|a-b| - rtol*|b|): <= atol iff every element is within atol + rtol*|b|."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    return ((a.float() - b.float()).abs() - rtol * b.float().abs()).max().item()


def self_test(device: str = "cuda", atol: float = 1e-2, rtol: float = 1.0 / 64) -> dict:
    """Compare the Triton kernels with ``torch_ref`` on a few realistic shapes.

    Returns ``{'rms_norm': d, 'add_rms_norm.h': d, 'add_rms_norm.y': d, 'ok': bool}``
    where each ``d`` is the max abs difference over all shapes. ``h`` must be
    bit-exact (0.0); the norms may differ only by rsqrt / reduction-order ulps,
    far below ``atol``. On any exception ``ok`` is False and ``error`` holds
    the message, so a broken kernel degrades to the torch twin instead of
    taking the engine down. This function syncs the device; it is for warmup,
    not for the graph.
    """
    worst = {"rms_norm": 0.0, "add_rms_norm.h": 0.0, "add_rms_norm.y": 0.0}
    result = {"ok": False}
    if not HAS_TRITON:
        result.update(worst)
        result["error"] = "triton is not importable"
        return result
    try:
        try:
            from . import torch_ref as ref
        except ImportError:
            from kernels import torch_ref as ref  # archive root on sys.path

        gen = torch.Generator().manual_seed(0x5EED)
        for shape in _TEST_SHAPES:
            n = shape[-1]
            x = (torch.randn(shape, generator=gen) * 3).to(torch.bfloat16).to(device)
            r = (torch.randn(shape, generator=gen) * 3).to(torch.bfloat16).to(device)
            w = (1.0 + 0.5 * torch.randn(n, generator=gen)).to(torch.bfloat16).to(device)

            y_t = rms_norm(x, w, _TEST_EPS)
            y_r = ref.ref_rms_norm(x, w, _TEST_EPS)
            worst["rms_norm"] = max(worst["rms_norm"], _excess(y_t, y_r, atol, rtol))

            h_t, z_t = add_rms_norm(x, r, w, _TEST_EPS)
            h_r, z_r = ref.ref_add_rms_norm(x, r, w, _TEST_EPS)
            worst["add_rms_norm.h"] = max(worst["add_rms_norm.h"], _max_abs_diff(h_t, h_r))
            worst["add_rms_norm.y"] = max(worst["add_rms_norm.y"], _excess(z_t, z_r, atol, rtol))
    except Exception as exc:  # report, never raise: the engine decides the fallback
        result.update(worst)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result.update(worst)
    result["ok"] = all(math.isfinite(v) and v <= atol for v in worst.values())
    return result
