"""Weight-streaming GEMM for decode-sized inputs (M <= 16 rows), in Triton.

``linear_small(x [M, K] bf16, w [N, K] bf16, cfg) -> [M, N] bf16`` computes
``x @ w.T`` exactly like ``torch.nn.functional.linear``: bf16 inputs, fp32
accumulation, one rounding to bf16 at the end. Decode steps are bound by
reading the 8 GB of weights once per step; cuBLAS is tuned for square-ish
GEMMs and often reaches only ~70-80% of HBM bandwidth at M <= 16. This kernel
streams each weight tile once, keeps several tiles in flight
(``num_stages``), and uses split-K so that small-N projections (o_proj,
down_proj: N = 2560) still fill the 132 SMs.

Split-K partials are fp32 and are reduced by a deterministic ``torch.sum``
over the split axis before the single bf16 rounding, so the rounding structure
matches cuBLAS (only the fp32 summation order differs, which the tie margin
absorbs).

The engine benchmarks each (N, K) shape at load time against ``F.linear`` and
uses whichever is faster (``choose_config``), so a slow config never hurts.
``self_test`` checks every config against ``F.linear``.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

#: Rows per program; inputs with M <= BLOCK_M are zero-padded.
BLOCK_M = 16

#: Candidate tile configs (BLOCK_N, BLOCK_K, num_warps, num_stages).
CONFIGS = (
    (64, 128, 4, 4),
    (64, 256, 4, 3),
)

#: Target program count for split-K selection (two waves of 132 SMs).
_TARGET_PROGRAMS = 264


if HAS_TRITON:

    @triton.jit
    def _gemm_small_kernel(
        x_ptr,        # [M, K] bf16, row stride stride_xm
        w_ptr,        # [N, K] bf16, row stride stride_wn
        out_ptr,      # SPLIT_K == 1: [M, N] bf16 ; else [SPLIT_K, M, N] fp32
        M, N, K,
        stride_xm, stride_wn, stride_om, stride_os,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # int64 [BLOCK_N]
        rk = tl.arange(0, BLOCK_K)
        m_mask = rm < M
        n_mask = rn < N
        k_per_split = K // SPLIT_K                           # host guarantees divisibility by BLOCK_K
        k0 = pid_k * k_per_split

        x_ptrs = x_ptr + rm[:, None].to(tl.int64) * stride_xm + k0 + rk[None, :]     # [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + rn[None, :] * stride_wn + k0 + rk[:, None]                  # [BLOCK_K, BLOCK_N] (transposed read)

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _ in range(0, k_per_split, BLOCK_K):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K

        out_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            ptrs = out_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :]
            tl.store(ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)
        else:
            ptrs = out_ptr + pid_k * stride_os + rm[:, None].to(tl.int64) * stride_om + rn[None, :]
            tl.store(ptrs, acc, mask=out_mask)


def split_k_for(N, K, block_n, block_k):
    """Largest power-of-two split (<= 8) that reaches the program target and divides K into BLOCK_K tiles."""
    programs = -(-N // block_n)
    split = 1
    while programs * split < _TARGET_PROGRAMS and split < 8:
        nxt = split * 2
        if K % (nxt * block_k) != 0:
            break
        split = nxt
    return split


def linear_small(x: torch.Tensor, w: torch.Tensor, cfg) -> torch.Tensor:
    """``x @ w.T`` for ``x [M<=16, K]`` and ``w [N, K]``, bf16 in/out, fp32 accumulate.

    ``cfg`` is ``(BLOCK_N, BLOCK_K, num_warps, num_stages)``. Graph-capturable.
    """
    if not HAS_TRITON:
        raise RuntimeError("linear_small needs triton; use torch.nn.functional.linear")
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw and M <= BLOCK_M, (x.shape, w.shape)
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.stride(1) == 1 and w.stride(1) == 1
    block_n, block_k, num_warps, num_stages = cfg
    assert K % block_k == 0, f"K={K} must be a multiple of BLOCK_K={block_k}"
    split = split_k_for(N, K, block_n, block_k)
    grid = (triton.cdiv(N, block_n), split)
    if split == 1:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        _gemm_small_kernel[grid](
            x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0), 0,
            BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=1,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out
    part = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    _gemm_small_kernel[grid](
        x, w, part, M, N, K, x.stride(0), w.stride(0), part.stride(1), part.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split,
        num_warps=num_warps, num_stages=num_stages,
    )
    return part.sum(dim=0).to(torch.bfloat16)


def _within(a, b, atol=2e-2, rtol=2.0 / 128):
    diff = (a.float() - b.float()).abs()
    return float(diff.max().item()), bool((diff <= atol + rtol * b.float().abs()).all().item())


def self_test(device="cuda", shapes=((6144, 2560), (2560, 9728)), rows=(1, 16)) -> dict:
    """Check every config against F.linear on the model's weight shapes. Returns {'ok', 'configs': {...}}."""
    result = {"ok": HAS_TRITON, "configs": {}}
    if not HAS_TRITON:
        result["error"] = "triton is not importable"
        return result
    gen = torch.Generator(device="cpu").manual_seed(7)
    for cfg in CONFIGS:
        rec = {"ok": True, "max_abs_diff": 0.0}
        try:
            for N, K in shapes:
                w = (torch.randn(N, K, generator=gen) * 0.02).to(torch.bfloat16).to(device)
                for M in rows:
                    x = (torch.randn(M, K, generator=gen)).to(torch.bfloat16).to(device)
                    got = linear_small(x, w, cfg)
                    ref = F.linear(x, w)
                    d, ok = _within(got, ref)
                    rec["max_abs_diff"] = max(rec["max_abs_diff"], d)
                    rec["ok"] = rec["ok"] and ok and bool(torch.isfinite(got.float()).all().item())
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(error).__name__}: {error}"
        result["configs"][str(cfg)] = rec
    result["ok"] = any(rec["ok"] for rec in result["configs"].values())
    result["good_configs"] = [cfg for cfg in CONFIGS if result["configs"][str(cfg)]["ok"]]
    return result


def choose_config(x: torch.Tensor, w: torch.Tensor, good_configs, reps: int = 10):
    """Time F.linear vs every good config for this (M, N, K); return (best_cfg_or_None, timings)."""
    if not HAS_TRITON or not good_configs:
        return None, {}

    def timeit(fn):
        fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps

    timings = {"cublas": timeit(lambda: F.linear(x, w))}
    for cfg in good_configs:
        try:
            timings[str(cfg)] = timeit(lambda: linear_small(x, w, cfg))
        except Exception as error:  # noqa: BLE001
            timings[str(cfg)] = float("inf")
            timings[f"{cfg}:error"] = repr(error)
    best_name = min((k for k in timings if not k.endswith(":error")), key=timings.get)
    if best_name == "cublas":
        return None, timings
    for cfg in good_configs:
        if str(cfg) == best_name:
            return cfg, timings
    return None, timings
