"""Fused decode GEMMs for M <= 16 rows: RMSNorm prologue, residual / SwiGLU epilogues.

One Triton kernel family, selected by constexpr flags:

    NORM = 1 : the input rows are RMS-normalised on the fly (Qwen3RMSNorm with
               gain ``nw``), so the standalone norm kernel disappears. Every
               program recomputes the row statistics itself (x is tiny, M*K
               bf16, and stays in L2), then normalises each K tile exactly as
               the reference: fp32 sum of squares * (1/K), rsqrt, x*r rounded
               to bf16, times the bf16 gain rounded to bf16 -> the bf16 GEMM
               operand HF would have computed.
    EPI  = 0 : plain bf16 store (fp32 accumulate, one rounding).
    EPI  = 1 : residual: out = bf16(res + bf16(acc)) -- the projection output is
               rounded to bf16 first (it is a separate tensor in HF), then the
               bf16 residual add rounds once more.
    EPI  = 2 : SwiGLU pair: the program computes the gate tile (rows of ``w``)
               and the matching up tile (rows of ``w2``) and stores
               bf16( bf16(silu(bf16(g))) * bf16(u) ), i.e. Qwen3MLP's
               act_fn(gate) * up with the reference's roundings.

Everything is fp32-accumulated, with the reference's cast placements; only
reduction order differs (tie margin). No split-K here (an epilogue needs the
full sum); small-N projections use narrow BLOCK_N instead to fill the SMs.
The engine self-tests each variant against the torch composition at load and
picks per-shape configs by measurement.
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

BLOCK_M = 16

#: (BLOCK_N, BLOCK_K, num_warps, num_stages) candidates, by output width class.
CONFIGS_WIDE = ((64, 256, 4, 4), (128, 128, 4, 4))       # N >= 4096
CONFIGS_NARROW = ((16, 256, 4, 4), (32, 256, 4, 4))      # N = 2560

if HAS_TRITON:
    if HAS_LIBDEVICE:

        @triton.jit
        def _exp_f32(x):
            return _libdevice.exp(x)

        @triton.jit
        def _rsqrt_f32(x):
            return _libdevice.rsqrt(x)

    else:

        @triton.jit
        def _exp_f32(x):
            return tl.exp(x)

        @triton.jit
        def _rsqrt_f32(x):
            return tl.math.rsqrt(x)

    @triton.jit
    def _fused_gemm_kernel(
        x_ptr, w_ptr, w2_ptr, nw_ptr, res_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_wn, stride_om,
        eps, inv_k,
        NORM: tl.constexpr, EPI: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0).to(tl.int64)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        m_mask = rm < M
        n_mask = rn < N
        x_row = x_ptr + rm[:, None].to(tl.int64) * stride_xm      # [BLOCK_M, 1]

        if NORM:
            ss = tl.zeros([BLOCK_M], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                xt = tl.load(x_row + k0 + rk[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
                ss += tl.sum(xt * xt, axis=1)
            r = _rsqrt_f32(ss * inv_k + eps)                        # [BLOCK_M] fp32

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if EPI == 2:
            acc2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        w_col = w_ptr + rn[None, :] * stride_wn                     # [1, BLOCK_N]
        if EPI == 2:
            w2_col = w2_ptr + rn[None, :] * stride_wn
        for k0 in range(0, K, BLOCK_K):
            x = tl.load(x_row + k0 + rk[None, :], mask=m_mask[:, None], other=0.0)
            if NORM:
                n = (x.to(tl.float32) * r[:, None]).to(tl.bfloat16).to(tl.float32)
                nw = tl.load(nw_ptr + k0 + rk).to(tl.float32)
                x = (n * nw[None, :]).to(tl.bfloat16)
            w = tl.load(w_col + k0 + rk[:, None], mask=n_mask[None, :], other=0.0)   # [BLOCK_K, BLOCK_N]
            acc += tl.dot(x, w)
            if EPI == 2:
                w2 = tl.load(w2_col + k0 + rk[:, None], mask=n_mask[None, :], other=0.0)
                acc2 += tl.dot(x, w2)

        out_off = rm[:, None].to(tl.int64) * stride_om + rn[None, :]
        out_mask = m_mask[:, None] & n_mask[None, :]
        if EPI == 0:
            tl.store(out_ptr + out_off, acc.to(tl.bfloat16), mask=out_mask)
        elif EPI == 1:
            res = tl.load(res_ptr + out_off, mask=out_mask, other=0.0).to(tl.float32)
            y = acc.to(tl.bfloat16).to(tl.float32)
            tl.store(out_ptr + out_off, (res + y).to(tl.bfloat16), mask=out_mask)
        else:
            g = acc.to(tl.bfloat16).to(tl.float32)
            u = acc2.to(tl.bfloat16).to(tl.float32)
            s = tl.math.div_rn(g, 1.0 + _exp_f32(-g)).to(tl.bfloat16).to(tl.float32)
            tl.store(out_ptr + out_off, (s * u).to(tl.bfloat16), mask=out_mask)


def fused_linear(x, w, cfg, *, norm_w=None, eps=0.0, residual=None, silu_pair=False):
    """out = x @ w.T with optional RMSNorm prologue and residual / SwiGLU epilogue.

    x [M<=16, K] bf16; w [N, K] bf16 (for silu_pair: [2*I, K] = cat(gate, up),
    output [M, I]); norm_w [K] bf16; residual [M, N] bf16. Graph-capturable.
    """
    if not HAS_TRITON:
        raise RuntimeError("fused_linear needs triton")
    M, K = x.shape
    if silu_pair:
        assert residual is None
        N = w.shape[0] // 2
        w2 = w[N:]
    else:
        N = w.shape[0]
        w2 = w
    assert w.shape[1] == K and M <= BLOCK_M and x.stride(1) == 1 and w.stride(1) == 1
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    if silu_pair:
        assert w.shape[0] % 2 == 0
    if norm_w is not None:
        assert norm_w.numel() == K and norm_w.stride(0) == 1 and norm_w.dtype == torch.bfloat16
    block_n, block_k, num_warps, num_stages = cfg
    assert K % block_k == 0, f"K={K} not a multiple of BLOCK_K={block_k}"
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    epi = 2 if silu_pair else (1 if residual is not None else 0)
    if residual is not None:
        assert residual.shape == (M, N) and residual.stride(1) == 1 and residual.stride(0) == N
    grid = (triton.cdiv(N, block_n),)
    _fused_gemm_kernel[grid](
        x, w, w2, norm_w if norm_w is not None else w, residual if residual is not None else out, out,
        M, N, K, x.stride(0), w.stride(0), out.stride(0),
        float(eps), 1.0 / K,
        NORM=1 if norm_w is not None else 0, EPI=epi,
        BLOCK_M=BLOCK_M, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ----------------------------------------------------------------------------- reference


def ref_fused_linear(x, w, *, norm_w=None, eps=0.0, residual=None, silu_pair=False):
    """Torch composition with the same roundings (used for the self-test)."""
    if norm_w is not None:
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        x = norm_w * (xf * torch.rsqrt(var + eps)).to(x.dtype)
    if silu_pair:
        I = w.shape[0] // 2
        g = F.linear(x, w[:I])
        u = F.linear(x, w[I:])
        return (F.silu(g).to(torch.float32) * u.to(torch.float32)).to(x.dtype)
    y = F.linear(x, w)
    if residual is not None:
        y = (residual.to(torch.float32) + y.to(torch.float32)).to(x.dtype)
    return y


def _within(a, b, atol=2e-2, rtol=2.0 / 128):
    diff = (a.float() - b.float()).abs()
    return float(diff.max().item()), bool((diff <= atol + rtol * b.float().abs()).all().item())


def self_test(device="cuda", M=1, H=2560, I=9728, NQ=32, NKV=8, D=128) -> dict:
    """Check every (variant, config) the engine may use at this M. Returns good configs per variant."""
    result = {"ok": HAS_TRITON, "variants": {}}
    if not HAS_TRITON:
        result["error"] = "triton is not importable"
        return result
    gen = torch.Generator(device="cpu").manual_seed(11)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).to(device)

    x = rnd(M, H, scale=3.0)
    a = rnd(M, NQ * D, scale=1.0)
    p = rnd(M, I, scale=1.0)
    ln = (1.0 + 0.5 * torch.randn(H, generator=gen)).to(torch.bfloat16).to(device)
    w_qkv = rnd((NQ + 2 * NKV) * D, H, scale=0.02)
    w_o = rnd(H, NQ * D, scale=0.02)
    w_gu = rnd(2 * I, H, scale=0.02)
    w_down = rnd(H, I, scale=0.02)
    res = rnd(M, H, scale=3.0)
    variants = {
        "norm_qkv": (CONFIGS_WIDE, dict(norm_w=ln, eps=1e-6), x, w_qkv),
        "o_res": (CONFIGS_NARROW, dict(residual=res), a, w_o),
        "norm_gu_silu": (CONFIGS_WIDE, dict(norm_w=ln, eps=1e-6, silu_pair=True), x, w_gu),
        "down_res": (CONFIGS_NARROW, dict(residual=res), p, w_down),
    }
    # Selection weights make every accumulator hold <= 2 exactly representable
    # terms, so the expected output is bit-determined and a missing bf16 rounding
    # (a cast-placement error the tolerance test cannot see) shows up as a large
    # fraction of mismatched elements. Reduction-order / rsqrt-ulp effects can
    # still flip a rare element by one bf16 ulp, hence the small allowance.
    def two_hot(N, K, off1, off2, scale2):
        w2 = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
        j = torch.arange(min(N, K), device=device)
        w2[j, (j + off1) % K] = 1.0
        w2[j, (j + off2) % K] = scale2
        return w2

    def exact_ok(got, expect, max_frac=0.005):
        if got.shape != expect.shape or not bool(torch.isfinite(got.float()).all().item()):
            return False, 1.0
        g16 = got.view(torch.int16).to(torch.int32)
        e16 = expect.view(torch.int16).to(torch.int32)
        mism = (g16 != e16)
        frac = float(mism.float().mean().item())
        ulp_ok = bool(((g16 - e16).abs() <= 1).all().item())
        return (frac <= max_frac and ulp_ok), frac

    def ref_norm(xx, gain):
        xf = xx.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        return gain * (xf * torch.rsqrt(var + eps_val)).to(xx.dtype)

    eps_val = 1e-6
    exact_checks = {}
    x_sel = rnd(M, H, scale=3.0)
    w_norm_sel = torch.zeros(H, H, dtype=torch.bfloat16, device=device)
    w_norm_sel[torch.arange(H, device=device), torch.arange(H, device=device)] = 1.0  # one-hot
    exact_checks["norm_qkv"] = (x_sel, w_norm_sel, dict(norm_w=ln, eps=eps_val), lambda: ref_norm(x_sel, ln))
    a_sel = rnd(M, NQ * D, scale=1.0)
    w_res_sel = two_hot(H, NQ * D, 0, 1, 2.0 ** -9)
    acc_res = a_sel.float() @ w_res_sel.float().t()
    exact_checks["o_res"] = (a_sel, w_res_sel, dict(residual=res), lambda: (res.float() + acc_res.bfloat16().float()).bfloat16())
    p_sel = rnd(M, I, scale=1.0)
    w_down_sel = two_hot(H, I, 0, 1, 2.0 ** -9)
    acc_down = p_sel.float() @ w_down_sel.float().t()
    exact_checks["down_res"] = (p_sel, w_down_sel, dict(residual=res), lambda: (res.float() + acc_down.bfloat16().float()).bfloat16())
    w_g = two_hot(I, H, 0, 1, 2.0 ** -9)
    w_u = two_hot(I, H, 2, 3, 2.0 ** -9)
    w_gu_sel = torch.cat([w_g, w_u], dim=0)
    n_sel = ref_norm(x_sel, ln)
    acc_g = n_sel.float() @ w_g.float().t()
    acc_u = n_sel.float() @ w_u.float().t()
    exact_checks["norm_gu_silu"] = (
        x_sel, w_gu_sel, dict(norm_w=ln, eps=eps_val, silu_pair=True),
        lambda: (F.silu(acc_g.bfloat16()).float() * acc_u.bfloat16().float()).bfloat16(),
    )

    for name, (configs, kw, xin, w) in variants.items():
        ref = ref_fused_linear(xin, w, **kw)
        ex_x, ex_w, ex_kw, ex_expect = exact_checks[name]
        expect = ex_expect()
        good, diffs = [], {}
        for cfg in configs:
            try:
                got = fused_linear(xin, w, cfg, **kw)
                got_ex = fused_linear(ex_x, ex_w, cfg, **ex_kw)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                d, ok = _within(got, ref)
                ok = ok and bool(torch.isfinite(got.float()).all().item())
                ok_ex, frac = exact_ok(got_ex, expect)
                diffs[str(cfg)] = (round(d, 4), round(frac, 5))
                if ok and ok_ex:
                    good.append(cfg)
            except Exception as error:  # noqa: BLE001
                diffs[str(cfg)] = f"{type(error).__name__}: {error}"[:120]
        result["variants"][name] = {"good": good, "diffs": diffs}
    result["ok"] = all(v["good"] for v in result["variants"].values())
    return result


def bench(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps
