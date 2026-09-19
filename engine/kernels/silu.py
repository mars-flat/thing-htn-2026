"""SwiGLU activation ``silu(gate) * up`` in Triton, matching ``Qwen3MLP`` exactly.

The reference computes, on bf16 tensors ``g = gate_proj(x)`` and
``u = up_proj(x)``::

    s = act_fn(g)        # torch silu, bf16 in/out: fp32 math  g / (1 + expf(-g)),  rounded to bf16
    p = s * u            # bf16 * bf16: exact product in fp32, rounded to bf16 once

``silu_mul`` takes the fused ``[M, 2I]`` gate/up matmul output (gate in
columns ``[0, I)``, up in ``[I, 2I)``) and produces ``p`` ``[M, I]`` with the
same two roundings in the same places. Only the exp implementation may differ
from torch at the ulp level, and even that is avoided when libdevice is
available: ``__nv_expf`` is the accurate ``expf`` torch's CUDA silu kernel
calls, whereas ``tl.exp`` on fp32 is the ``ex2.approx`` fast path. The
division uses ``tl.math.div_rn`` (IEEE ``div.rn.f32``) to match nvcc's default
precise division rather than Triton's ``div.full.f32`` approximation.

The module imports without ``triton`` (``HAS_TRITON`` is False and the
wrapper raises); the engine then falls back to ``torch_ref.ref_silu_mul``.
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
    except Exception:  # older / different layout: try the backend-agnostic shim
        try:
            from triton.language.extra import libdevice as _libdevice
        except Exception:
            _libdevice = None
HAS_LIBDEVICE = _libdevice is not None

#: Flat 1-D tiling over the [M, I] output. Fixed, no autotune.
BLOCK = 1024
NUM_WARPS = 4


if HAS_TRITON:
    if HAS_LIBDEVICE:

        @triton.jit
        def _exp_f32(x):
            # __nv_expf: full-precision expf, the same routine torch's silu uses.
            return _libdevice.exp(x)

    else:

        @triton.jit
        def _exp_f32(x):
            # Fallback: Triton's fast exp (ex2.approx of x*log2e). A few ulp
            # worse than expf; hidden by the bf16 rounding except at ties.
            return tl.exp(x)

    @triton.jit
    def _silu_mul_kernel(gu_ptr, out_ptr, n_elements, I: tl.constexpr, BLOCK: tl.constexpr):
        # One program per BLOCK contiguous elements of the flattened [M, I]
        # output. All index math is int64; I is a compile-time constant so the
        # div/mod below become multiply-shift sequences instead of a 64-bit
        # divide per element.
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        row = offs // I
        col = offs - row * I
        gate_off = row * (2 * I) + col  # gu[row, col]
        # gu[row, I + col] is gate_off + I

        g = tl.load(gu_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + gate_off + I, mask=mask, other=0.0).to(tl.float32)

        # torch silu on bf16: x / (1 + expf(-x)) in fp32, then ONE rounding to bf16.
        e = _exp_f32(-g)
        s = tl.math.div_rn(g, 1.0 + e)
        s = s.to(tl.bfloat16).to(tl.float32)

        # bf16 * bf16 -> fp32 product is exact (8+8 significand bits); round once.
        p = (s * u).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, p, mask=mask)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """``out[m, i] = bf16( bf16(silu(gu[m, i])) * gu[m, I + i] )`` for ``gu [M, 2I]``.

    ``gu`` is the bf16 output of the fused gate/up matmul; the gate half is
    columns ``[0, I)`` and the up half ``[I, 2I)``. Returns a new contiguous
    ``[M, I]`` tensor of ``gu``'s dtype. Leading dimensions other than a single
    ``M`` are tolerated and preserved. No host syncs; CUDA-graph capturable.
    """
    if not HAS_TRITON:
        raise RuntimeError("silu_mul needs triton; use torch_ref.ref_silu_mul")
    shape = gu.shape
    two_i = shape[-1]
    if two_i % 2 != 0:
        raise ValueError(f"last dim must be 2*I, got {two_i}")
    inter = two_i // 2
    rows = gu.reshape(-1, two_i).contiguous()
    m = rows.shape[0]
    out = torch.empty((m, inter), dtype=rows.dtype, device=rows.device)
    n_elements = m * inter
    if n_elements == 0:
        return out.reshape(*shape[:-1], inter)
    grid = (triton.cdiv(n_elements, BLOCK),)
    _silu_mul_kernel[grid](
        rows,
        out,
        n_elements,
        I=inter,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
    return out.reshape(*shape[:-1], inter)


# ----------------------------------------------------------------------------
# Self-test against the pure-torch twin
# ----------------------------------------------------------------------------

_TEST_SHAPES = [(7, 2 * 9728)]


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest |a - b| as a Python float; inf on shape/dtype mismatch, nan on nan."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    return (a.float() - b.float()).abs().max().item()


def self_test(device: str = "cuda", atol: float = 1e-2, rtol: float = 1.0 / 64) -> dict:
    """Compare ``silu_mul`` with ``torch_ref.ref_silu_mul`` on realistic shapes.

    Returns ``{'silu_mul': max_abs_diff, 'ok': bool}`` (plus ``'error'`` if a
    launch or compile failed, with ``ok`` False so the engine can fall back to
    the twin). Differences, if any, are exp-ulp flips of a bf16 rounding, far
    below ``atol``. Syncs the device; warmup only, never inside the graph.
    """
    worst = {"silu_mul": 0.0}
    result = {"ok": False, "libdevice": HAS_LIBDEVICE}
    if not HAS_TRITON:
        result.update(worst)
        result["error"] = "triton is not importable"
        return result
    try:
        try:
            from . import torch_ref as ref
        except ImportError:
            from kernels import torch_ref as ref  # archive root on sys.path

        gen = torch.Generator().manual_seed(0x51D0)
        for shape in _TEST_SHAPES:
            gu = (torch.randn(shape, generator=gen) * 3).to(torch.bfloat16).to(device)
            out_t = silu_mul(gu)
            out_r = ref.ref_silu_mul(gu)
            if out_t.shape == out_r.shape and out_t.dtype == out_r.dtype:
                excess = ((out_t.float() - out_r.float()).abs() - rtol * out_r.float().abs()).max().item()
            else:
                excess = float("inf")
            worst["silu_mul"] = max(worst["silu_mul"], excess)
    except Exception as exc:  # report, never raise: the engine decides the fallback
        result.update(worst)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result.update(worst)
    result["ok"] = all(math.isfinite(v) and v <= atol for v in worst.values())
    return result
