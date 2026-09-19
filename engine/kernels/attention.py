"""Flash-decoding attention over a static KV cache (SPEC.md kernel 4).

``attn_decode(q, k_cache, v_cache, lengths, scale, num_splits)`` computes, for
``T`` new query tokens per sequence, causal attention against everything that
is already in the cache, and returns ``[B, T, Nq*D]`` bf16.

Semantics (what the reference computes)
---------------------------------------
Sequence ``b`` has ``lengths[b]`` valid cached tokens ``0 .. lengths[b]-1``;
its ``T`` new tokens are the LAST ``T`` of them (kernel 3 wrote them before we
run).  Query ``t`` of sequence ``b`` therefore sits at absolute position
``lengths[b] - T + t`` and may attend key ``j`` iff

    j <= lengths[b] - T + t      <=>      j < lengths[b] - (T - 1 - t)

which is exactly what HF gets from SDPA with ``is_causal=True`` on the
prefix+new keys (and, for T == 1, from SDPA with no mask over ``lengths[b]``
keys).  Query head ``h`` reads KV head ``h // G`` with ``G = Nq // Nkv``.
Softmax is fp32; the probabilities are rounded to bf16 before the PV matmul,
which is what flash attention (the reference's CUDA backend) does; the fp32
accumulator is rounded to bf16 once at the end.

Algorithm (flash-decoding, two kernels)
---------------------------------------
Kernel 1, grid ``(B*Nkv, num_splits)``.  Program ``(bh, s)`` owns one
``(b, kvh)`` pair and one contiguous chunk of that sequence's keys.  Its query
block holds every query that reads KV head ``kvh``: ``G`` query heads x ``T``
tokens = ``G*T`` rows, padded with zero rows to ``BLOCK_M = 16`` so ``tl.dot``
is legal.

    Row layout:  r = t * G + g      (t = r // G, g = r % G, head h = kvh*G + g)
                 rows r >= G*T are padding: q = 0, fully masked, never read back.

It runs an online softmax over its chunk in 64-key tiles and writes the
running max ``m``, the running sum ``l`` and the un-normalised fp32
accumulator ``acc`` for its 16 rows.  Kernel 2, grid ``(B*Nkv, G*T)``, merges
the splits of one row with the usual log-sum-exp rescaling, divides, casts to
bf16 and scatters into ``out[b, t, kvh*G + g, :]``.

Masking uses a finite sentinel (``-1e30``) rather than ``-inf`` so the
``m_i - m_new`` and ``s - m_new`` subtractions never produce ``inf - inf =
NaN`` on rows that have not seen a visible key yet.  Empty chunks (a split
whose start is past the sequence end) and rows whose keys all fall in another
split simply keep ``m = -1e30, l = 0, acc = 0``; the reduce weights them with
``exp(-1e30 - m_true) == 0``.

Chunking (one deliberate deviation from the SPEC text): ``chunk`` is
``cdiv(len, num_splits)`` rounded UP to a multiple of ``BLOCK_N``, so split
boundaries coincide with key tiles.  Every split still does at most
``cdiv(cdiv(len, S), 64)`` tile iterations, the critical path is unchanged,
but trailing splits become empty instead of every split doing a ragged partial
tile (and for tiny ``len`` we no longer launch 64 programs that each process a
single key).  The reduce handles empty splits, so the result is identical up
to fp32 reduction order.

This module imports cleanly without Triton (``HAS_TRITON == False``); only
``attn_decode`` needs it.  ``self_test`` compares against the pure-torch twin
``kernels.torch_ref.ref_attn_decode``.
"""

from __future__ import annotations

import math

import torch

try:  # Triton is only present in the CUDA judge container.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - local macOS development
    triton = None
    tl = None
    HAS_TRITON = False

#: Query rows per program (G*T real rows, zero-padded).  tl.dot needs M >= 16.
BLOCK_M = 16
#: Keys per online-softmax iteration.
BLOCK_N = 64
#: Splits merged per unrolled step in the reduce kernel (keeps the [ST, D]
#: fp32 tile at 1024 elements regardless of num_splits).
SPLIT_TILE = 8
#: Finite stand-in for -inf.  Far below any real score (|q.k|*scale is O(10)),
#: exp(-1e30 - anything_real) underflows to exactly 0, and -1e30 - (-1e30) == 0
#: so fully-masked rows never see inf - inf.
NEG_BIG = -1e30

#: Two waves of H100 SMs; choose_num_splits targets at least this many programs.
_TARGET_PROGRAMS = 264
_MAX_SPLITS = 64


def choose_num_splits(B: int, Nkv: int, cap: int) -> int:
    """Number of KV splits for a ``(B, Nkv, CAP)`` decode shape.

    Smallest power of two ``S`` with ``B*Nkv*S >= 264`` (two waves of 132
    SMs), clamped to ``[1, 64]`` and to ``cdiv(cap, BLOCK_N)`` (never more
    splits than there are 64-key tiles).  Fixed per CUDA graph, so it is a
    ``constexpr`` of both kernels.
    """
    # Independent of ``cap`` on purpose: NUM_SPLITS is a constexpr, so a
    # cap-dependent choice would compile a new kernel for every capacity.
    # Empty splits (start >= len) cost one trivial program each.
    del cap
    bh = max(1, int(B) * int(Nkv))
    s = 1
    while bh * s < _TARGET_PROGRAMS and s < 32:
        s *= 2
    return s


if HAS_TRITON:

    @triton.jit(do_not_specialize=[11])  # T: runtime, one binary for T=1 and T=k+1
    def _attn_partial_kernel(
        q_ptr,            # [B, T, Nq, D] bf16
        k_ptr,            # [B, Nkv, CAP, D] bf16
        v_ptr,            # [B, Nkv, CAP, D] bf16
        lengths_ptr,      # [B] int32: valid tokens per sequence (includes the T new ones)
        part_ml_ptr,      # [B*Nkv, S, BLOCK_M, 2] fp32: (m, l) per row per split
        part_acc_ptr,     # [B*Nkv, S, BLOCK_M, D] fp32: un-normalised PV accumulator
        scale,            # fp32 softmax scale (D ** -0.5)
        Nkv,              # number of KV heads (runtime int)
        stride_qb,        # T * Nq * D
        stride_qt,        # Nq * D
        stride_kh,        # CAP * D   (one (b, kvh) plane of the cache)
        T,                        # new tokens per sequence (runtime, unspecialised)
        G: tl.constexpr,          # query heads per KV head
        NUM_SPLITS: tl.constexpr,
        BLOCK_M: tl.constexpr,    # 16
        BLOCK_N: tl.constexpr,    # 64
        D: tl.constexpr,          # 128
    ):

        # ---- which (b, kvh, split) am I -------------------------------------
        # int64 program ids so every base offset below is 64-bit; only the
        # tile-relative offsets (bounded by CAP*D and T*Nq*D) stay int32.
        pid_bh = tl.program_id(0).to(tl.int64)     # = b * Nkv + kvh
        pid_s = tl.program_id(1)                   # int32, used in chunk math
        b = pid_bh // Nkv
        kvh = pid_bh % Nkv

        # ---- my chunk of keys: [start, end) ----------------------------------
        seq_len = tl.load(lengths_ptr + b).to(tl.int32)
        chunk = (seq_len + NUM_SPLITS - 1) // NUM_SPLITS            # cdiv(len, S)
        chunk = (chunk + BLOCK_N - 1) // BLOCK_N * BLOCK_N          # round up to whole tiles
        start = pid_s * chunk
        end = tl.minimum(start + chunk, seq_len)
        # If start >= end the tile loop below runs zero times and we store the
        # neutral partial (m = NEG_BIG, l = 0, acc = 0).

        # ---- query block: row r = t*G + g ----------------------------------
        rows = tl.arange(0, BLOCK_M)                # int32 [BLOCK_M]
        t_of_row = rows // G
        g_of_row = rows % G
        row_valid = rows < G * T                    # padding rows are >= G*T
        head = kvh * G + g_of_row                   # int64 [BLOCK_M], query head index
        offs_d = tl.arange(0, D)                    # int32 [D]

        q_off = (b * stride_qb
                 + t_of_row.to(tl.int64) * stride_qt
                 + head * D)                        # int64 [BLOCK_M]
        q = tl.load(q_ptr + q_off[:, None] + offs_d[None, :],
                    mask=row_valid[:, None], other=0.0)          # [BLOCK_M, D] bf16

        # Key j is visible to row r (query t) iff j < len - (T-1-t).
        # For padding rows this limit is meaningless; row_valid masks them.
        row_limit = seq_len - (T - 1 - t_of_row)    # int32 [BLOCK_M]

        # ---- KV plane for (b, kvh) -----------------------------------------
        kv_base = pid_bh * stride_kh                # int64 scalar
        k_base = k_ptr + kv_base
        v_base = v_ptr + kv_base
        offs_n = tl.arange(0, BLOCK_N)              # int32 [BLOCK_N]

        # ---- online softmax state (fp32) -----------------------------------
        m_i = tl.full([BLOCK_M], -1e30, tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for n0 in range(start, end, BLOCK_N):
            n = n0 + offs_n                          # int32 [BLOCK_N] absolute key index
            n_valid = n < end                        # inside my chunk (and < len)
            # K is read straight into its transposed [D, BLOCK_N] shape (the
            # contiguous head-dim runs down dim 0), the same access pattern the
            # Triton fused-attention tutorial uses for K, so no tl.trans is
            # needed.  V is read row-major [BLOCK_N, D].
            kt_off = offs_d[:, None] + n[None, :] * D            # int32 [D, BLOCK_N]
            kv_off = n[:, None] * D + offs_d[None, :]            # int32 [BLOCK_N, D]

            kt = tl.load(k_base + kt_off, mask=n_valid[None, :], other=0.0)  # [D, BLOCK_N] bf16
            # bf16 x bf16 -> fp32 accumulate on tensor cores, then fp32 scale
            # (same order as flash: scale after the QK^T matmul).
            s = tl.dot(q, kt) * scale                            # [BLOCK_M, BLOCK_N] fp32

            visible = ((n[None, :] < row_limit[:, None])
                       & n_valid[None, :]
                       & row_valid[:, None])                     # [BLOCK_M, BLOCK_N] bool
            s = tl.where(visible, s, -1e30)

            m_new = tl.maximum(m_i, tl.max(s, axis=1))           # [BLOCK_M]
            alpha = tl.math.exp(m_i - m_new)                     # rescale old state
            p = tl.math.exp(s - m_new[:, None])                  # [BLOCK_M, BLOCK_N]
            # Explicit zero for invisible keys.  For a row that has a visible
            # key this is a no-op (exp(-1e30 - m) == 0); for a row with none
            # yet, m_new == NEG_BIG and exp(0) == 1 would otherwise leak in.
            p = tl.where(visible, p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_base + kv_off, mask=n_valid[:, None], other=0.0)   # [BLOCK_N, D] bf16
            # Flash rounds P to bf16 for the PV matmul; l keeps the fp32 sum.
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new

        # ---- write partials ------------------------------------------------
        part_row = (pid_bh * NUM_SPLITS + pid_s) * BLOCK_M + rows.to(tl.int64)   # int64 [BLOCK_M]
        tl.store(part_ml_ptr + part_row * 2, m_i)
        tl.store(part_ml_ptr + part_row * 2 + 1, l_i)
        tl.store(part_acc_ptr + part_row[:, None] * D + offs_d[None, :], acc)

    @triton.jit(do_not_specialize=[6])  # T: runtime
    def _attn_reduce_kernel(
        part_ml_ptr,      # [B*Nkv, S, BLOCK_M, 2] fp32
        part_acc_ptr,     # [B*Nkv, S, BLOCK_M, D] fp32
        out_ptr,          # [B, T, Nq, D] bf16
        Nkv,              # runtime int
        stride_ob,        # T * Nq * D
        stride_ot,        # Nq * D
        T,                # runtime, unspecialised (unused in the body; grid is G*T)
        G: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        S_POW2: tl.constexpr,     # next_power_of_2(NUM_SPLITS), for the m/l vector loads
        ST: tl.constexpr,         # splits merged per unrolled step
        BLOCK_M: tl.constexpr,
        D: tl.constexpr,
    ):
        # Grid is (B*Nkv, G*T): every program is a real row, no padding here.
        pid_bh = tl.program_id(0).to(tl.int64)
        r = tl.program_id(1)                        # int32 row, r = t*G + g
        b = pid_bh // Nkv
        kvh = pid_bh % Nkv
        t = r // G
        g = r % G
        head = kvh * G + g                          # int64

        offs_d = tl.arange(0, D)
        # Partial row index of (bh, split s, row r) is (bh*S + s)*BLOCK_M + r.
        row_base = pid_bh * NUM_SPLITS * BLOCK_M + r          # int64, split 0
        ml_base = part_ml_ptr + row_base * 2
        acc_base = part_acc_ptr + row_base * D

        # Pass 1: global max and normaliser over all splits (tiny vectors).
        offs_s = tl.arange(0, S_POW2)
        s_valid = offs_s < NUM_SPLITS
        m_all = tl.load(ml_base + offs_s * (BLOCK_M * 2), mask=s_valid, other=-1e30)
        l_all = tl.load(ml_base + offs_s * (BLOCK_M * 2) + 1, mask=s_valid, other=0.0)
        m = tl.max(m_all, axis=0)                   # fp32 scalar; finite for every real row
        w_all = tl.math.exp(m_all - m)              # empty splits: exp(-1e30 - m) == 0
        l = tl.sum(w_all * l_all, axis=0)           # fp32 scalar

        # Pass 2: weighted sum of the accumulators, ST splits at a time.
        acc = tl.zeros([D], dtype=tl.float32)
        for s0 in tl.static_range(0, NUM_SPLITS, ST):
            ss = s0 + tl.arange(0, ST)
            ss_valid = ss < NUM_SPLITS
            m_s = tl.load(ml_base + ss * (BLOCK_M * 2), mask=ss_valid, other=-1e30)
            w_s = tl.math.exp(m_s - m)                                   # [ST]
            acc_s = tl.load(acc_base + ss[:, None] * (BLOCK_M * D) + offs_d[None, :],
                            mask=ss_valid[:, None], other=0.0)           # [ST, D]
            acc += tl.sum(w_s[:, None] * acc_s, axis=0)

        o = acc / l                                 # fp32 [D]
        out_off = b * stride_ob + t.to(tl.int64) * stride_ot + head * D
        tl.store(out_ptr + out_off + offs_d, o.to(tl.bfloat16))


def attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    num_splits: int,
) -> torch.Tensor:
    """Causal decode attention of ``T`` new queries against a static KV cache.

    Args:
        q:        ``[B, T, Nq, D]`` bf16 (RoPE'd, normed).
        k_cache:  ``[B, Nkv, CAP, D]`` bf16, contiguous.
        v_cache:  ``[B, Nkv, CAP, D]`` bf16, contiguous.
        lengths:  ``[B]`` int32, valid tokens per sequence including the ``T``
                  new ones (they occupy positions ``lengths[b]-T .. lengths[b]-1``).
        scale:    softmax scale, ``D ** -0.5``.
        num_splits: KV splits per (b, kvh); see :func:`choose_num_splits`.

    Returns:
        ``[B, T, Nq*D]`` bf16; ``out[b, t, h*D:(h+1)*D]`` is query head ``h``
        (KV head ``h // G``).  Graph-capturable: only ``torch.empty`` and two
        launches, no host syncs.
    """
    if not HAS_TRITON:
        raise RuntimeError("kernels.attention.attn_decode needs triton")
    B, T, Nq, D = q.shape
    Bk, Nkv, CAP, Dk = k_cache.shape
    assert Bk == B and Dk == D and v_cache.shape == k_cache.shape, "q / cache shape mismatch"
    assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
    assert k_cache.is_contiguous() and v_cache.is_contiguous(), "KV cache must be contiguous"
    assert lengths.dtype in (torch.int32, torch.int64) and lengths.numel() == B
    assert Nq % Nkv == 0, "Nq must be a multiple of Nkv"
    assert D >= 16 and (D & (D - 1)) == 0, "head dim must be a power of two >= 16"
    G = Nq // Nkv
    BM = max(BLOCK_M, triton.next_power_of_2(G * T))
    assert G * T <= 64, f"G*T = {G * T} query rows per KV head is too many"
    S = int(num_splits)
    assert S >= 1

    q = q.contiguous()
    lengths = lengths.contiguous()
    BH = B * Nkv
    dev = q.device
    part_ml = torch.empty((BH, S, BM, 2), dtype=torch.float32, device=dev)
    part_acc = torch.empty((BH, S, BM, D), dtype=torch.float32, device=dev)
    out = torch.empty((B, T, Nq, D), dtype=torch.bfloat16, device=dev)

    _attn_partial_kernel[(BH, S)](
        q, k_cache, v_cache, lengths, part_ml, part_acc,
        float(scale),
        Nkv,
        T * Nq * D,     # stride_qb
        Nq * D,         # stride_qt
        CAP * D,        # stride_kh
        T=T, G=G, NUM_SPLITS=S,
        BLOCK_M=BM, BLOCK_N=BLOCK_N, D=D,
        num_warps=4,
    )
    _attn_reduce_kernel[(BH, G * T)](
        part_ml, part_acc, out,
        Nkv,
        T * Nq * D,     # stride_ob
        Nq * D,         # stride_ot
        T=T, G=G, NUM_SPLITS=S,
        S_POW2=max(2, triton.next_power_of_2(S)),
        ST=min(SPLIT_TILE, max(2, triton.next_power_of_2(S))),
        BLOCK_M=BM, D=D,
        num_warps=2,
    )
    return out.view(B, T, Nq * D)


# --------------------------------------------------------------------------
# Self-test against the pure-torch twin
# --------------------------------------------------------------------------

#: (name, B, T, Nq, Nkv, CAP, lengths, num_splits or None for choose_num_splits)
_SELF_TEST_CASES = (
    # Kept small: every (T, G, NUM_SPLITS, BLOCK_M) combination is a separate
    # Triton compile at load time. The tiny-length and T>1 cases are the
    # load-bearing ones for off-by-one detection; keep them.
    ("b1_t1_len513", 1, 1, 32, 8, 640, [513], None),
    ("b16_t1_len520", 16, 1, 32, 8, 640, [520] * 16, None),
    ("b2_t4_ragged", 2, 4, 32, 8, 256, [100, 37], None),
    ("b3_t1_tiny", 3, 1, 32, 8, 64, [1, 2, 64], None),
    ("b2_t8_g2_full_block", 2, 8, 16, 8, 128, [64, 9], None),
    ("b2_t5_g4_block32", 2, 5, 32, 8, 640, [600, 7], None),
)


def self_test(device: str = "cuda", atol: float = 1e-2, rtol: float = 1e-2, seed: int = 0, G: int | None = None, max_T: int | None = None) -> dict:
    """Compare ``attn_decode`` with ``kernels.torch_ref.ref_attn_decode``.

    Inputs are ``N(0, 1)`` bf16.  A case passes when the outputs are free of
    NaN/Inf and ``|out - ref| <= atol + rtol*|ref|`` everywhere (the rtol term
    only matters for |ref| > 2 where one bf16 ulp is already 1.6e-2).  Each
    case is also run with ``num_splits=1`` to check split invariance.
    Returns ``{"ok": bool, "device": str, "cases": {name: {...}}}``.
    """
    try:
        from .torch_ref import ref_attn_decode  # type: ignore[import-not-found]
    except ImportError:
        from kernels.torch_ref import ref_attn_decode  # type: ignore[no-redef]

    D = 128
    results: dict = {"ok": True, "device": str(device), "cases": {}}
    if not HAS_TRITON:
        results["ok"] = False
        results["error"] = "triton not available"
        return results

    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, B, T, Nq, Nkv, CAP, lengths, S in _SELF_TEST_CASES:
        # Skip cases whose (G, T) the loaded model / config never runs: each
        # combination is a separate Triton compile at load time.
        if G is not None and Nq // Nkv != G:
            continue
        if max_T is not None and T > max_T:
            continue
        q = torch.randn(B, T, Nq, D, generator=gen).to(device=device, dtype=torch.bfloat16)
        k = torch.randn(B, Nkv, CAP, D, generator=gen).to(device=device, dtype=torch.bfloat16)
        v = torch.randn(B, Nkv, CAP, D, generator=gen).to(device=device, dtype=torch.bfloat16)
        lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        scale = D ** -0.5
        splits = choose_num_splits(B, Nkv, CAP) if S is None else S

        rec: dict = {"num_splits": splits}
        try:
            out = attn_decode(q, k, v, lens, scale, splits)
            ref = ref_attn_decode(q, k, v, lens, scale, splits)
            out1 = attn_decode(q, k, v, lens, scale, 1) if splits != 1 and name.startswith("b2_t4") else out
            of, rf = out.float(), ref.float()
            diff = (of - rf).abs()
            finite = bool(torch.isfinite(of).all().item())
            rec["max_abs_diff"] = float(diff.max().item())
            rec["max_abs_diff_splits1"] = float((out1.float() - of).abs().max().item())
            rec["finite"] = finite
            rec["shape_ok"] = tuple(out.shape) == (B, T, Nq * D) and out.dtype == torch.bfloat16
            within = bool((diff <= atol + rtol * rf.abs()).all().item())
            splits_ok = bool(((out1.float() - of).abs() <= atol + rtol * of.abs()).all().item())
            rec["splits_ok"] = splits_ok
            rec["ok"] = bool(finite and rec["shape_ok"] and within and splits_ok)
        except Exception as e:  # noqa: BLE001 - report, do not raise, at warmup
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        results["cases"][name] = rec
        results["ok"] = results["ok"] and rec["ok"]
    return results


__all__ = [
    "HAS_TRITON",
    "BLOCK_M",
    "BLOCK_N",
    "attn_decode",
    "choose_num_splits",
    "self_test",
]
