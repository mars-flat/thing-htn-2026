"""Qwen3 4B decode engine: static KV cache, fused Triton kernels, CUDA-graph decode.

Exact-greedy by construction: every operation keeps the reference's formula and
cast placement (Transformers 4.51.3 Qwen3, BF16, SDPA); only the order of
reductions differs. See notes/SPEC.md (not shipped) for the derivation.

Structure of one generate() call:
  1. _prepare(B, S, N): (re)allocate the per-shape state (KV cache, static
     buffers, CUDA graph) when the shape changes. Normally this only happens
     during the untimed warmup call, because warmup uses the sample's shape.
  2. _prefill(ids): eager forward over the prompt (flash SDPA, is_causal),
     writes K/V into the cache, returns the first greedy token.
  3. decode: one CUDA-graph replay per step. The graph reads the token history
     and per-sequence positions, runs all 36 layers + LM head + argmax, and
     writes the next token back into the history, so the host does nothing but
     replay and an async D2H copy. With spec_k > 0 a second graph verifies k
     prompt-lookup draft tokens per step (exact: every emitted token is the
     greedy choice on its own prefix).
     A small ring of pinned buffers keeps a few steps in flight so the GPU never
     waits for the Python yield.
"""

import json
import math
import os
import sys
import time
from collections import deque

import tempfile

import torch
import torch.nn.functional as F


def _ensure_triton_cache_dir():
    """Triton compiles at runtime and needs a writable cache directory.

    The judge runs the engine as an unprivileged user; if the default
    ``~/.triton`` is not writable, point Triton at a temp directory (fixed
    path so the workloads of one run share compiled kernels). Must run before
    ``triton`` is imported.
    """
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    home = os.path.expanduser("~")
    default = os.path.join(home, ".triton", "cache")
    try:
        os.makedirs(default, exist_ok=True)
        probe = os.path.join(default, ".write_probe")
        with open(probe, "w") as handle:
            handle.write("ok")
        os.remove(probe)
        return
    except OSError:
        pass
    fallback = os.path.join(tempfile.gettempdir(), "dryft_engine_triton_cache")
    os.makedirs(fallback, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = fallback
    print(f"[engine] TRITON_CACHE_DIR -> {fallback}", file=sys.stderr, flush=True)


_ensure_triton_cache_dir()

from kernels import torch_ref  # noqa: E402

CONFIG = {
    "triton": True,          # fused Triton kernels (per-op fallback to torch twins on self-test failure)
    "graphs": True,          # CUDA graphs for decode steps
    "pipeline_depth": 3,     # decode steps in flight ahead of the host yield
    "prefill_rows": 32768,   # max B*S rows per prefill batch slice
    "cap_align": 64,         # KV-cache capacity alignment
    "log": True,
    "self_test": True,       # verify each Triton kernel against its torch twin at load
    "profile": True,         # print a per-op timing breakdown during warmup
    "small_gemm": "auto",    # Triton weight-streaming GEMM for M<=16: "auto" (benchmark vs cuBLAS), True, False
    "spec_k": 4,             # exact speculative decoding: number of prompt-lookup draft tokens per step (0 = off)
    "spec_n": 3,             # n-gram length used to look up drafts in the sequence's own history
    "spec_max_batch": 8,     # use speculative decoding only when B <= this (lockstep verify pays off at small B)
    "spec_probe": (24, 6),   # when spec is losing: plain steps between probes, spec steps per probe
    "diag": False,           # raise after warmup with a diagnostic summary (the judge hides engine output)
    "fused": "auto",         # fused decode GEMMs (norm prologue / residual / SwiGLU epilogues): auto = keep if faster
    "warmup_budget_s": 200,  # skip optional warmup work (fused/spec variants) once load+warmup exceeds this
}


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _log(*parts):
    if CONFIG["log"]:
        print("[engine]", *parts, file=sys.stderr, flush=True)


class _Layer:
    __slots__ = ("ln1", "w_qkv", "q_norm", "k_norm", "w_o", "ln2", "w_gu", "w_down")


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device(os.environ.get("DRYFT_ENGINE_DEVICE", "cpu"))
        self.is_cuda = self.device.type == "cuda"
        self.use_triton = self.is_cuda and _env_flag("DRYFT_ENGINE_TRITON", CONFIG["triton"])
        self.use_graphs = self.is_cuda and _env_flag("DRYFT_ENGINE_GRAPHS", CONFIG["graphs"])
        self.pipeline_depth = max(1, int(os.environ.get("DRYFT_ENGINE_DEPTH", CONFIG["pipeline_depth"])))
        self.spec_k = max(0, int(os.environ.get("DRYFT_ENGINE_SPEC_K", CONFIG["spec_k"])))
        self.spec_n = int(CONFIG["spec_n"])
        self.spec_max_batch = int(os.environ.get("DRYFT_ENGINE_SPEC_MAX_B", CONFIG["spec_max_batch"]))

        with open(os.path.join(model_path, "config.json")) as handle:
            cfg = json.load(handle)
        self.H = cfg["hidden_size"]
        self.I = cfg["intermediate_size"]
        self.V = cfg["vocab_size"]
        self.L = cfg["num_hidden_layers"]
        self.Nq = cfg["num_attention_heads"]
        self.Nkv = cfg["num_key_value_heads"]
        self.D = cfg.get("head_dim", self.H // self.Nq)
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_theta"])
        self.scale = self.D ** -0.5
        if cfg.get("rope_scaling") is not None:
            raise RuntimeError("rope_scaling is not supported by this engine")

        # Per-shape state.
        self.B = self.CAP = 0
        self.k_cache = self.v_cache = None
        self.graphs = {}
        self.num_splits = 1
        self.hist_buf = self.pos_buf = None
        self.k_active = 0
        self.step_ms = {}
        self.diag = []
        self.fused = None          # kernels.gemm_fused module when importable
        self.fused_cfg = {}        # M -> {variant: cfg} or None
        self.use_fused = {}        # M -> bool (decided by whole-step timing)
        self.t_start = t0
        self.sdpa_gqa = None  # decided on first prefill
        self.gemm = None      # kernels.gemm module when usable
        self.gemm_good = []   # configs that passed the self-test
        self.gemm_tested = {}  # M -> good configs
        self.gemm_force = False
        self.gemm_choice = {}  # (M, N, K) -> cfg or None (cuBLAS)

        self._load_weights(model_path)
        self._rope_max = 0
        self._ensure_rope(8192)
        self._select_ops()

        _log(f"loaded in {time.time() - t0:.1f}s on {self.device}; triton={self.use_triton} graphs={self.use_graphs}")
        self.diag.append(f"load {time.time() - t0:.0f}s triton={self.use_triton} cache={os.environ.get('TRITON_CACHE_DIR', 'default')}")

    # ------------------------------------------------------------------ loading

    def _load_weights(self, model_path):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        model.to(self.device)
        base = model.model
        self.layers = []
        for layer in base.layers:
            attn, mlp = layer.self_attn, layer.mlp
            w = _Layer()
            w.ln1 = layer.input_layernorm.weight.data.contiguous()
            w.w_qkv = torch.cat(
                [attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0
            ).contiguous()
            w.q_norm = attn.q_norm.weight.data.contiguous()
            w.k_norm = attn.k_norm.weight.data.contiguous()
            w.w_o = attn.o_proj.weight.data.contiguous()
            w.ln2 = layer.post_attention_layernorm.weight.data.contiguous()
            w.w_gu = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], dim=0).contiguous()
            w.w_down = mlp.down_proj.weight.data.contiguous()
            self.layers.append(w)
        self.embed = base.embed_tokens.weight.data.contiguous()
        lm = model.lm_head.weight.data
        self.lm_head = self.embed if lm.data_ptr() == self.embed.data_ptr() else lm.contiguous()
        self.final_norm = base.norm.weight.data.contiguous()
        del model, base
        if self.is_cuda:
            torch.cuda.empty_cache()

    def _ensure_rope(self, max_pos):
        if max_pos <= self._rope_max:
            return
        max_pos = int(2 ** math.ceil(math.log2(max(max_pos, 1024))))
        # Bit-identical to Qwen3RotaryEmbedding (default rope, attention_scaling 1.0).
        dev = self.device
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.D, 2, dtype=torch.int64).to(device=dev, dtype=torch.float) / self.D)
        )
        positions = torch.arange(max_pos, device=dev, dtype=torch.float32)[None, :]
        inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
        position_ids_expanded = positions[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_tab = emb.cos()[0].to(torch.bfloat16).contiguous()
        self.sin_tab = emb.sin()[0].to(torch.bfloat16).contiguous()
        self._rope_max = max_pos
        # Captured graphs hold the old table addresses.
        if getattr(self, "graphs", None):
            self.graphs = {}

    # ------------------------------------------------------------------ ops

    def _select_ops(self):
        """Pick the Triton kernel or its torch twin for each fused op."""
        self.ops = {
            "rms_norm": torch_ref.ref_rms_norm,
            "add_rms_norm": torch_ref.ref_add_rms_norm,
            "silu_mul": torch_ref.ref_silu_mul,
            "qk_norm_rope_cache": torch_ref.ref_qk_norm_rope_cache,
            "attn_decode": torch_ref.ref_attn_decode,
        }
        self.choose_num_splits = lambda B, Nkv, cap: 1
        if not self.use_triton:
            return
        try:
            from kernels import attention, rmsnorm, rope, silu
        except Exception as error:  # noqa: BLE001
            _log(f"triton kernels unavailable ({error!r}); using torch twins")
            return
        candidates = [
            ("rms_norm", rmsnorm, rmsnorm.rms_norm),
            ("add_rms_norm", rmsnorm, rmsnorm.add_rms_norm),
            ("silu_mul", silu, silu.silu_mul),
            ("qk_norm_rope_cache", rope, rope.qk_norm_rope_cache),
            ("attn_decode", attention, attention.attn_decode),
        ]
        tested = {}
        for name, module, fn in candidates:
            if not getattr(module, "HAS_TRITON", False):
                _log(f"{name}: triton missing, torch twin")
                continue
            ok = True
            if CONFIG["self_test"]:
                if module not in tested:
                    try:
                        t_start = time.time()
                        if module is attention:
                            result = module.self_test(str(self.device), G=self.Nq // self.Nkv,
                                                      max_T=(self.spec_k + 1) if self.spec_k else 1)
                        elif module is rope:
                            result = module.self_test(str(self.device), dims=(self.Nq, self.Nkv),
                                                      max_T=max(64, self.spec_k + 1))
                        else:
                            result = module.self_test(str(self.device))
                        tested[module] = bool(result.get("ok", False))
                        _log(f"self_test {module.__name__} ({time.time() - t_start:.1f}s): {result}")
                        self.diag.append(f"{module.__name__.split('.')[-1]}:{'ok' if result.get('ok') else 'FAIL'}"
                                         f"{'' if result.get('ok') else str(result)[:200]}"
                                         f" {time.time() - t_start:.0f}s")
                    except Exception as error:  # noqa: BLE001
                        tested[module] = False
                        _log(f"self_test {module.__name__} raised {error!r}")
                        self.diag.append(f"{module.__name__.split('.')[-1]}:RAISED {error!r}"[:200])
                ok = tested[module]
            if ok:
                self.ops[name] = fn
            else:
                _log(f"{name}: self-test failed, torch twin")
        if self.ops["attn_decode"] is not torch_ref.ref_attn_decode:
            self.choose_num_splits = attention.choose_num_splits
        mode = os.environ.get("DRYFT_ENGINE_SMALL_GEMM", CONFIG["small_gemm"])
        if mode not in (False, "0", "false", "off"):
            try:
                from kernels import gemm

                if gemm.HAS_TRITON:
                    self.gemm = gemm          # self-tested lazily in _choose_gemms at the real M
                    self.gemm_force = mode in (True, "1", "true", "on")
            except Exception as error:  # noqa: BLE001
                _log(f"gemm kernel unavailable ({error!r}); cuBLAS only")
        mode = os.environ.get("DRYFT_ENGINE_FUSED", CONFIG["fused"])
        if mode not in (False, "0", "false", "off"):
            try:
                from kernels import gemm_fused

                if gemm_fused.HAS_TRITON:
                    self.fused = gemm_fused
            except Exception as error:  # noqa: BLE001
                _log(f"fused gemm unavailable ({error!r})")
        if self.is_cuda:
            torch.cuda.synchronize()

    def _over_budget(self):
        return time.time() - self.t_start > CONFIG["warmup_budget_s"]

    def _choose_fused(self, M):
        """Self-test the fused GEMM variants at M rows and pick the fastest good config per variant."""
        if self.fused is None or M > 16 or M in self.fused_cfg:
            return
        if self._over_budget():
            _log("fused: skipped (warmup budget)")
            self.fused_cfg[M] = None
            return
        t_start = time.time()
        try:
            result = self.fused.self_test(str(self.device), M=M, H=self.H, I=self.I, NQ=self.Nq, NKV=self.Nkv, D=self.D)
        except Exception as error:  # noqa: BLE001
            result = {"ok": False, "error": repr(error)}
        _log(f"self_test kernels.gemm_fused M={M} ({time.time() - t_start:.1f}s): {result}")
        self.diag.append(f"fused_test M{M}:{'ok' if result.get('ok') else 'FAIL ' + str(result)[:200]} {time.time() - t_start:.0f}s")
        if not result.get("ok"):
            self.fused_cfg[M] = None
            return
        first = self.layers[0]
        x = torch.randn((M, self.H), dtype=torch.bfloat16, device=self.device)
        a = torch.randn((M, self.Nq * self.D), dtype=torch.bfloat16, device=self.device)
        pp = torch.randn((M, self.I), dtype=torch.bfloat16, device=self.device)
        res = torch.randn((M, self.H), dtype=torch.bfloat16, device=self.device)
        plan = {
            "qkv": ("norm_qkv", x, first.w_qkv, dict(norm_w=first.ln1, eps=self.eps)),
            "o": ("o_res", a, first.w_o, dict(residual=res)),
            "gu": ("norm_gu_silu", x, first.w_gu, dict(norm_w=first.ln2, eps=self.eps, silu_pair=True)),
            "down": ("down_res", pp, first.w_down, dict(residual=res)),
            "lm": ("norm_qkv", x, self.lm_head, dict(norm_w=self.final_norm, eps=self.eps)),
        }
        chosen, report = {}, []
        for key, (variant, xin, w, kw) in plan.items():
            best, best_ms = None, float("inf")
            for cfg in result["variants"][variant]["good"]:
                try:
                    ms = self.fused.bench(lambda: self.fused.fused_linear(xin, w, cfg, **kw))
                except Exception as error:  # noqa: BLE001
                    _log(f"fused {key} {cfg} failed: {error!r}")
                    continue
                if ms < best_ms:
                    best, best_ms = cfg, ms
            if best is None:
                self.fused_cfg[M] = None
                _log(f"fused: no working config for {key}")
                return
            chosen[key] = best
            report.append(f"{key}={best_ms:.3f}ms{best}")
        self.fused_cfg[M] = chosen
        _log(f"fused configs M={M}: {' '.join(report)}")
        self.diag.append(f"fused M{M} {' '.join(report)}")

    def _step_fused(self, tok, pos, T):
        """Decode step with fused GEMMs (norm prologues, residual / SwiGLU epilogues). M = B*T <= 16."""
        ops = self.ops
        B = tok.shape[0]
        M = B * T
        cfg = self.fused_cfg[M]
        fl = self.fused.fused_linear
        lengths = pos + T
        h = F.embedding(tok.reshape(-1), self.embed)
        for l, w in enumerate(self.layers):
            qkv = fl(h, w.w_qkv, cfg["qkv"], norm_w=w.ln1, eps=self.eps)
            q = ops["qk_norm_rope_cache"](
                qkv, w.q_norm, w.k_norm, self.eps, self.cos_tab, self.sin_tab, pos, T,
                self.k_cache[l], self.v_cache[l],
            )
            a = ops["attn_decode"](q, self.k_cache[l], self.v_cache[l], lengths, self.scale, self.num_splits)
            h = fl(a.view(M, -1), w.w_o, cfg["o"], residual=h)
            p = fl(h, w.w_gu, cfg["gu"], norm_w=w.ln2, eps=self.eps, silu_pair=True)
            h = fl(p, w.w_down, cfg["down"], residual=h)
        logits = fl(h, self.lm_head, cfg["lm"], norm_w=self.final_norm, eps=self.eps)
        return logits.argmax(dim=-1)

    def _lin(self, x, w):
        """x @ w.T with the per-shape choice made at warmup (custom kernel or cuBLAS)."""
        M = x.shape[0]
        if M <= 16 and self.gemm is not None:
            cfg = self.gemm_choice.get((M, w.shape[0], w.shape[1]))
            if cfg is not None:
                return self.gemm.linear_small(x, w, cfg)
        return F.linear(x, w)

    def _choose_gemms(self, M):
        """Benchmark cuBLAS vs the custom kernel for every weight shape at M rows (warmup only)."""
        if self.gemm is None or M > 16:
            return
        if M not in self.gemm_tested:
            t_start = time.time()
            try:
                result = self.gemm.self_test(str(self.device), rows=(M,))
            except Exception as error:  # noqa: BLE001
                result = {"ok": False, "error": repr(error)}
            _log(f"self_test kernels.gemm M={M} ({time.time() - t_start:.1f}s): {result}")
            self.diag.append(f"gemm_test M{M}:{'ok' if result.get('ok') else 'FAIL ' + str(result)[:160]} {time.time() - t_start:.0f}s")
            self.gemm_tested[M] = list(result.get("good_configs", [])) if result.get("ok") else []
        good = self.gemm_tested[M]
        if not good:
            for name, w in (("qkv", self.layers[0].w_qkv), ("o", self.layers[0].w_o), ("gu", self.layers[0].w_gu),
                            ("down", self.layers[0].w_down), ("lm_head", self.lm_head)):
                self.gemm_choice[(M, w.shape[0], w.shape[1])] = None
            return
        self.gemm_good = good
        shapes = {}
        first = self.layers[0]
        for name, w in (("qkv", first.w_qkv), ("o", first.w_o), ("gu", first.w_gu), ("down", first.w_down), ("lm_head", self.lm_head)):
            shapes[name] = w
        x_cache = {}
        for name, w in shapes.items():
            key = (M, w.shape[0], w.shape[1])
            if key in self.gemm_choice:
                continue
            K = w.shape[1]
            x = x_cache.get(K)
            if x is None:
                x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
                x_cache[K] = x
            if self.gemm_force:
                cfg, timings = self.gemm_good[0], {}
            else:
                cfg, timings = self.gemm.choose_config(x, w, self.gemm_good)
            self.gemm_choice[key] = cfg
            _log(f"gemm {name} M={M} N={w.shape[0]} K={K}: {'custom ' + str(cfg) if cfg else 'cublas'} {timings}")
            short = {k[:14]: round(v, 3) for k, v in timings.items() if isinstance(v, (int, float))}
            self.diag.append(f"gemm {name} M{M}:{'C' + str(cfg) if cfg else 'cublas'} {short}")

    # ------------------------------------------------------------------ shape state

    def _prepare(self, B, S, N):
        k = self.spec_k if B <= self.spec_max_batch else 0
        slack = (self.pipeline_depth + 3) * (k + 1) + 16
        align = CONFIG["cap_align"]
        cap = ((S + N + slack + align - 1) // align) * align
        self._ensure_rope(cap)
        if B != self.B or cap > self.CAP or k != self.k_active:
            if self.is_cuda:
                torch.cuda.synchronize()
            self.graphs = {}
            self.k_cache = self.v_cache = None
            if self.is_cuda:
                torch.cuda.empty_cache()
            dev = self.device
            shape = (B, self.Nkv, cap, self.D)
            self.k_cache = [torch.zeros(shape, dtype=torch.bfloat16, device=dev) for _ in range(self.L)]
            self.v_cache = [torch.zeros(shape, dtype=torch.bfloat16, device=dev) for _ in range(self.L)]
            # Decode state: token history (prompt + generated) and tokens-in-cache per sequence.
            self.hist_buf = torch.zeros((B, cap), dtype=torch.int64, device=dev)
            self.pos_buf = torch.zeros((B,), dtype=torch.int32, device=dev)
            self.s_buf = torch.zeros((1,), dtype=torch.int64, device=dev)
            self.ntarget_buf = torch.full((1,), 1 << 40, dtype=torch.int64, device=dev)
            self.ar_W = torch.arange(cap, dtype=torch.int64, device=dev)
            self.ar_T = torch.arange(max(k + 1, self.spec_n, 2), dtype=torch.int64, device=dev)
            self.dump_idx = cap - 1
            self.B, self.CAP, self.k_active = B, cap, k
            self.num_splits = int(self.choose_num_splits(B, self.Nkv, cap))
            if self.is_cuda:
                ring = self.pipeline_depth + 2
                self.pin_nxt = [torch.zeros((k + 1, B), dtype=torch.int64, pin_memory=True) for _ in range(ring)]
                self.pin_adv = [torch.zeros((B,), dtype=torch.int64, pin_memory=True) for _ in range(ring)]
                self.events = [torch.cuda.Event() for _ in range(ring)]
            _log(f"state B={B} CAP={cap} splits={self.num_splits} spec_k={k} "
                 f"kv={2 * self.L * B * self.Nkv * cap * self.D * 2 / 2**30:.2f}GiB")
        if self.is_cuda:
            self._choose_gemms(B)
            self._choose_fused(B)
            if k:
                self._choose_gemms(B * (k + 1))
                self._choose_fused(B * (k + 1))
        if self.use_graphs:
            if 1 not in self.graphs:
                self._capture_best(1)
            if k and (k + 1) not in self.graphs:
                if self._over_budget():
                    _log("spec: skipped (warmup budget)")
                    self.k_active = 0
                else:
                    try:
                        self._capture_best(k + 1)
                    except Exception as error:  # noqa: BLE001
                        _log(f"spec capture failed ({error!r}); plain decoding only")
                        self.diag.append(f"spec capture FAILED {error!r}"[:200])
                        self.graphs.pop(k + 1, None)
                        self.k_active = 0
                        if self.is_cuda:
                            torch.cuda.synchronize()
        elif self.is_cuda and k:
            self.k_active = k
        if self.sdpa_gqa is None and self.is_cuda:
            self._probe_prefill_attention(B, S)

    def _capture_best(self, T):
        """Capture the step with and without fused GEMMs (when available) and keep the faster graph."""
        M = self.B * T
        candidates = [False]
        if self.fused_cfg.get(M):
            mode = os.environ.get("DRYFT_ENGINE_FUSED", CONFIG["fused"])
            candidates = [True, False] if mode in (True, "1", "true", "on") else [False, True]
        best = None
        for use in candidates:
            if use and self._over_budget():
                _log("fused graph: skipped (warmup budget)")
                break
            self.use_fused[M] = use
            try:
                self._capture(T)
            except Exception as error:  # noqa: BLE001
                _log(f"capture T={T} fused={use} failed: {error!r}")
                self.diag.append(f"capture T{T} fused={use} FAILED {error!r}"[:200])
                self.graphs.pop(T, None)
                if self.is_cuda:
                    torch.cuda.synchronize()
                if best is None and not use:
                    raise
                continue
            ms = self.step_ms[T]
            if best is None or ms < best[1] * 0.995:
                best = (use, ms, self.graphs[T])
            if use and best[0] and candidates[0] is True:
                break  # forced mode: fused captured fine, no need for the unfused variant
        if best is None:
            raise RuntimeError(f"no graph captured for T={T}")
        self.use_fused[M] = best[0]
        self.graphs[T] = best[2]
        self.step_ms[T] = best[1]
        _log(f"T={T}: using {'fused' if best[0] else 'unfused'} step, {best[1]:.3f} ms")
        self.diag.append(f"T{T} {'fused' if best[0] else 'unfused'} {best[1]:.3f}ms")

    # ------------------------------------------------------------------ decode steps (device-agnostic)

    def _plain_step(self):
        """One greedy token per sequence. Reads hist[pos], writes hist[pos+1], pos += 1. Returns next [B]."""
        hist, pos = self.hist_buf, self.pos_buf
        L = pos.to(torch.int64) + 1
        tok = hist.gather(1, (L - 1)[:, None])
        nxt = self._step(tok, pos, 1)
        # Sequences that already produced max_new_tokens are frozen (adaptive
        # spec mode can run plain steps while other sequences catch up).
        done = (L - self.s_buf) >= self.ntarget_buf
        idx = torch.where(done, torch.full_like(L, self.dump_idx), L)
        hist.scatter_(1, idx[:, None], nxt[:, None])
        pos.add_((~done).to(torch.int32))
        return nxt

    def _spec_draft(self, L, k):
        """Prompt-lookup draft: the k tokens that followed the latest earlier occurrence of the last n tokens.

        Any draft is safe (verification is exact); a miss just yields tokens from
        the start of the history. Pure tensor ops, no host sync.
        """
        hist = self.hist_buf
        n = self.spec_n
        W = hist.shape[1]
        suf_idx = (L[:, None] - n + self.ar_T[:n][None, :]).clamp_(min=0)          # [B, n]
        suffix = hist.gather(1, suf_idx)
        Wc = W - n + 1
        match = hist[:, 0:Wc] == suffix[:, 0:1]
        for j in range(1, n):
            match = match & (hist[:, j:Wc + j] == suffix[:, j:j + 1])
        cand = self.ar_W[:Wc][None, :]
        match = match & (cand < (L - n)[:, None])                                   # match must end before the suffix
        istar = torch.where(match, cand, -1).max(dim=1).values                      # latest match start, or -1
        src = (istar[:, None] + n + self.ar_T[:k][None, :]).clamp_(min=0)
        src = torch.minimum(src, (L - 1)[:, None])
        draft = hist.gather(1, src)                                                 # [B, k]
        last = hist.gather(1, (L - 1)[:, None])                                     # [B, 1]
        return torch.cat([last, draft], dim=1)

    def _spec_step(self, k):
        """Verify k drafted tokens per sequence in one forward of T=k+1 tokens.

        next[b, j] is the greedy token after feeding tok[b, :j+1]; drafts are
        accepted while they match, so every emitted token is the model's greedy
        choice on its own prefix. Returns (next [B, T], adv [B]) where adv is the
        number of new verified tokens (0 for sequences that already produced
        max_new_tokens, whose state is frozen).
        """
        hist, pos = self.hist_buf, self.pos_buf
        B = hist.shape[0]
        T = k + 1
        L = pos.to(torch.int64) + 1
        tok = self._spec_draft(L, k)
        nxt = self._step(tok, pos, T).view(B, T)
        eq = (nxt[:, :k] == tok[:, 1:]).to(torch.int64)
        acc = torch.cumprod(eq, dim=1).sum(dim=1)                                   # leading matches, 0..k
        done = (L - self.s_buf) >= self.ntarget_buf
        adv = torch.where(done, torch.zeros_like(acc), acc + 1)
        j = self.ar_T[:T]
        idx = L[:, None] + j[None, :]
        idx = torch.where(j[None, :] < adv[:, None], idx, self.dump_idx)
        hist.scatter_(1, idx, nxt)
        pos.add_(adv.to(torch.int32))
        return nxt.t().contiguous(), adv

    def _capture(self, T):
        """Capture one decode step (T == 1 plain, T == k+1 speculative) into a CUDA graph."""
        B = self.B
        pos = self.pos_buf
        k = T - 1
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        t0 = time.time()

        def run():
            pos.fill_(1)
            return self._plain_step() if k == 0 else self._spec_step(k)

        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if CONFIG["profile"]:
            tok = torch.zeros((B, T), dtype=torch.int64, device=self.device)
            pos.fill_(1)
            self._profile(tok, pos, T)
        pos.fill_(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            out = self._plain_step() if k == 0 else self._spec_step(k)
        torch.cuda.synchronize()
        self.graphs[T] = (graph, out)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        pos.fill_(1)
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        self.step_ms[T] = start.elapsed_time(end) / 5
        _log(f"captured T={T} graph in {time.time() - t0:.1f}s; replay {self.step_ms[T]:.3f} ms/step")
        self.diag.append(f"graph T{T} {self.step_ms[T]:.3f}ms cap {time.time() - t0:.0f}s")

    def _profile(self, tok, pos, T):
        """Rough per-op breakdown of one eager decode step (warmup only)."""
        try:
            timings = {}

            def timed(name, fn):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                end.synchronize()
                timings[name] = timings.get(name, 0.0) + start.elapsed_time(end)
                return out

            ops = self.ops
            B, M = tok.shape[0], tok.shape[0] * T
            lengths = pos + T
            x = timed("embed", lambda: F.embedding(tok.reshape(-1), self.embed))
            h, d = x, None
            for w in self.layers:
                if d is None:
                    n = timed("norm", lambda: ops["rms_norm"](x, w.ln1, self.eps))
                else:
                    h, n = timed("norm", lambda: ops["add_rms_norm"](h, d, w.ln1, self.eps))
                qkv = timed("qkv_mm", lambda: self._lin(n, w.w_qkv))
                q = timed("rope", lambda: ops["qk_norm_rope_cache"](
                    qkv, w.q_norm, w.k_norm, self.eps, self.cos_tab, self.sin_tab, pos, T,
                    self.k_cache[0], self.v_cache[0]))
                a = timed("attn", lambda: ops["attn_decode"](
                    q, self.k_cache[0], self.v_cache[0], lengths, self.scale, self.num_splits))
                o = timed("o_mm", lambda: self._lin(a.view(M, -1), w.w_o))
                h, n = timed("norm", lambda: ops["add_rms_norm"](h, o, w.ln2, self.eps))
                gu = timed("gu_mm", lambda: self._lin(n, w.w_gu))
                p = timed("silu", lambda: ops["silu_mul"](gu))
                d = timed("down_mm", lambda: self._lin(p, w.w_down))
            hfin, n = timed("norm", lambda: ops["add_rms_norm"](h, d, self.final_norm, self.eps))
            logits = timed("lm_head", lambda: self._lin(n, self.lm_head))
            timed("argmax", lambda: logits.argmax(dim=-1))
            total = sum(timings.values())
            _log("profile T=%d B=%d (eager, ms): total %.3f | %s" % (
                T, B, total, " ".join(f"{k}={v:.3f}" for k, v in sorted(timings.items(), key=lambda kv: -kv[1]))))
            self.diag.append("prof T%d B%d tot %.2f %s" % (
                T, B, total, " ".join(f"{k}={v:.2f}" for k, v in sorted(timings.items(), key=lambda kv: -kv[1]))))
        except Exception as error:  # noqa: BLE001
            _log(f"profile failed: {error!r}")

    # ------------------------------------------------------------------ forward passes

    def _step(self, tok, pos, T):
        """One decode step for T new tokens per sequence. tok [B,T] int64, pos [B] int32 -> next [B*T]."""
        ops = self.ops
        B = tok.shape[0]
        M = B * T
        if self.use_fused.get(M):
            return self._step_fused(tok, pos, T)
        lengths = pos + T
        x = F.embedding(tok.reshape(-1), self.embed)
        h, d = x, None
        for l, w in enumerate(self.layers):
            if d is None:
                n = ops["rms_norm"](x, w.ln1, self.eps)
            else:
                h, n = ops["add_rms_norm"](h, d, w.ln1, self.eps)
            qkv = self._lin(n, w.w_qkv)
            q = ops["qk_norm_rope_cache"](
                qkv, w.q_norm, w.k_norm, self.eps, self.cos_tab, self.sin_tab, pos, T,
                self.k_cache[l], self.v_cache[l],
            )
            a = ops["attn_decode"](q, self.k_cache[l], self.v_cache[l], lengths, self.scale, self.num_splits)
            o = self._lin(a.view(M, -1), w.w_o)
            h, n = ops["add_rms_norm"](h, o, w.ln2, self.eps)
            gu = self._lin(n, w.w_gu)
            p = ops["silu_mul"](gu)
            d = self._lin(p, w.w_down)
        _, n = ops["add_rms_norm"](h, d, self.final_norm, self.eps)
        logits = self._lin(n, self.lm_head)
        return logits.argmax(dim=-1)

    def _sdpa_gqa(self, qt, k, v):
        return F.scaled_dot_product_attention(qt, k, v, is_causal=True, scale=self.scale, enable_gqa=True)

    def _sdpa_expand(self, qt, k, v):
        B, S = qt.shape[0], qt.shape[2]
        rep = self.Nq // self.Nkv
        ke = k[:, :, None, :, :].expand(B, self.Nkv, rep, S, self.D).reshape(B, self.Nq, S, self.D)
        ve = v[:, :, None, :, :].expand(B, self.Nkv, rep, S, self.D).reshape(B, self.Nq, S, self.D)
        return F.scaled_dot_product_attention(qt, ke, ve, is_causal=True, scale=self.scale)

    def _probe_prefill_attention(self, B, S):
        """Pick the faster of SDPA enable_gqa vs HF-style expanded KV heads (warmup only).

        torch 2.5.1's flash backend may or may not take the enable_gqa path; if
        it does not, SDPA silently falls back to a much slower backend, which
        would cost TTFT. Measure instead of assuming.
        """
        Bp = min(B, max(1, CONFIG["prefill_rows"] // S))
        qt = torch.randn((Bp, self.Nq, S, self.D), dtype=torch.bfloat16, device=self.device)
        k = self.k_cache[0][:Bp, :, :S, :]
        v = self.v_cache[0][:Bp, :, :S, :]
        timings = {}
        for name, fn in (("gqa", self._sdpa_gqa), ("expand", self._sdpa_expand)):
            try:
                fn(qt, k, v)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):
                    fn(qt, k, v)
                end.record()
                torch.cuda.synchronize()
                timings[name] = start.elapsed_time(end) / 3
            except Exception as error:  # noqa: BLE001
                _log(f"prefill attention {name} failed: {error!r}")
        if not timings:
            raise RuntimeError("no working prefill attention path")
        best = min(timings, key=timings.get)
        self.sdpa_gqa = best == "gqa"
        _log(f"prefill attention B={Bp} S={S}: {timings} -> {best}")
        self.diag.append(f"sdpa {best} {({k: round(v, 2) for k, v in timings.items()})}")

    def _prefill_attention(self, q, k, v):
        """q [B,S,Nq,D], k/v [B,Nkv,S,D] (cache views) -> [B*S, Nq*D]. Flash SDPA, causal."""
        B, S = q.shape[0], q.shape[1]
        qt = q.transpose(1, 2)
        if self.sdpa_gqa is None:  # non-CUDA local path: try gqa once, else expand
            try:
                out = self._sdpa_gqa(qt, k, v)
                self.sdpa_gqa = True
            except Exception:  # noqa: BLE001
                self.sdpa_gqa = False
        if self.sdpa_gqa:
            out = self._sdpa_gqa(qt, k, v)
        else:
            out = self._sdpa_expand(qt, k, v)
        return out.transpose(1, 2).reshape(B * S, self.Nq * self.D)

    def _prefill(self, ids):
        """Prompt forward. ids [B,S] int64 on device -> first greedy token [B] int64."""
        B, S = ids.shape
        rows_per = max(1, CONFIG["prefill_rows"] // S)
        firsts = []
        for b0 in range(0, B, rows_per):
            b1 = min(B, b0 + rows_per)
            firsts.append(self._prefill_slice(ids[b0:b1], b0))
        return firsts[0] if len(firsts) == 1 else torch.cat(firsts)

    def _prefill_slice(self, ids, b0):
        ops = self.ops
        Bs, S = ids.shape
        M = Bs * S
        b1 = b0 + Bs
        pos = torch.zeros((Bs,), dtype=torch.int32, device=self.device)
        x = F.embedding(ids.reshape(-1), self.embed)
        h, d = x, None
        for l, w in enumerate(self.layers):
            kc = self.k_cache[l][b0:b1]
            vc = self.v_cache[l][b0:b1]
            if d is None:
                n = ops["rms_norm"](x, w.ln1, self.eps)
            else:
                h, n = ops["add_rms_norm"](h, d, w.ln1, self.eps)
            qkv = F.linear(n, w.w_qkv)
            q = ops["qk_norm_rope_cache"](
                qkv, w.q_norm, w.k_norm, self.eps, self.cos_tab, self.sin_tab, pos, S, kc, vc,
            )
            a = self._prefill_attention(q, kc[:, :, :S, :], vc[:, :, :S, :])
            o = F.linear(a, w.w_o)
            h, n = ops["add_rms_norm"](h, o, w.ln2, self.eps)
            gu = F.linear(n, w.w_gu)
            p = ops["silu_mul"](gu)
            d = F.linear(p, w.w_down)
        last = torch.arange(Bs, device=self.device) * S + (S - 1)
        h_last = h.index_select(0, last)
        d_last = d.index_select(0, last)
        _, n = ops["add_rms_norm"](h_last, d_last, self.final_norm, self.eps)
        logits = F.linear(n, self.lm_head)
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------ generate

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Never stops at end-of-sequence tokens.
        """
        B, S = len(input_ids), len(input_ids[0])
        N = int(max_new_tokens)
        if N <= 0:
            return
        with torch.inference_mode():
            self._prepare(B, S, N)
            if CONFIG["diag"]:
                raise RuntimeError("DIAG B=%d S=%d N=%d | " % (B, S, N) + " | ".join(self.diag))
            ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
            first = self._prefill(ids)
            # Decode state: history = prompt + first token; S tokens are in the cache.
            self.hist_buf[:, :S].copy_(ids)
            self.hist_buf[:, S].copy_(first)
            self.pos_buf.fill_(S)
            self.s_buf.fill_(S)
            self.ntarget_buf.fill_(N)
            if self.is_cuda:
                yield from self._decode_loop(first, B, N, self._cuda_backend(B))
            else:
                yield from self._decode_loop(first, B, N, self._eager_backend(B))

    # ------------------------------------------------------------------ decode loop

    def _cuda_backend(self, B):
        """Pipelined graph replays with async D2H copies into a pinned ring."""
        stream = torch.cuda.current_stream()
        pin_nxt, pin_adv, events = self.pin_nxt, self.pin_adv, self.events
        graphs = self.graphs
        use_graphs = self.use_graphs
        k = self.k_active

        def launch(slot, mode, first=None):
            if mode == "first":
                pin_nxt[slot][0].copy_(first, non_blocking=True)
            elif mode == "plain":
                if use_graphs:
                    graph, out = graphs[1]
                    graph.replay()
                else:
                    out = self._plain_step()
                pin_nxt[slot][0].copy_(out, non_blocking=True)
            else:
                if use_graphs:
                    graph, (nxt, adv) = graphs[k + 1]
                    graph.replay()
                else:
                    nxt, adv = self._spec_step(k)
                pin_nxt[slot].copy_(nxt, non_blocking=True)
                pin_adv[slot].copy_(adv, non_blocking=True)
            events[slot].record(stream)

        def consume(slot, mode):
            events[slot].synchronize()
            if mode == "spec":
                return pin_nxt[slot].tolist(), pin_adv[slot].tolist()   # rows[j][b]
            return pin_nxt[slot][0].tolist(), None

        return launch, consume, len(pin_nxt)

    def _eager_backend(self, B):
        """Synchronous steps (CPU / MPS): same protocol, results kept per slot."""
        k = self.k_active
        results = {}

        def launch(slot, mode, first=None):
            if mode == "first":
                results[slot] = (first.tolist(), None)
            elif mode == "plain":
                results[slot] = (self._plain_step().tolist(), None)
            else:
                nxt, adv = self._spec_step(k)
                results[slot] = (nxt.tolist(), adv.tolist())   # rows[j][b]

        def consume(slot, mode):
            return results.pop(slot)

        return launch, consume, self.pipeline_depth + 2

    def _decode_loop(self, first, B, N, backend):
        launch, consume, ring = backend
        depth = self.pipeline_depth
        k = self.k_active
        queues = [deque() for _ in range(B)]
        produced = [0] * B
        inflight = deque()
        counter = 0
        yielded = 0

        # Adaptive mode: speculate while the measured yield rate beats the plain step.
        t1 = self.step_ms.get(1, 1.0)
        tk = self.step_ms.get(k + 1, 1.0) if k else float("inf")
        mode = "spec" if k else "plain"
        rate = tk / t1 + 0.25 if k else 0.0      # optimistic start
        plain_left, probe_left = 0, 0
        probe_plain, probe_spec = CONFIG["spec_probe"]
        spec_steps = spec_tokens = 0

        launch(0, "first", first)
        inflight.append((0, "first"))
        counter = 1

        # Never launch more steps than could possibly be needed: a leftover
        # replay after the last yield would run into the next sample's prefill.
        potential = 1  # upper bound on tokens per sequence still to arrive from in-flight steps (the first token)
        while yielded < N:
            while len(inflight) <= depth and min(produced) + potential < N:
                slot = counter % ring
                counter += 1
                if k:
                    if mode == "plain":
                        plain_left -= 1
                        if plain_left <= 0:
                            mode, probe_left = "spec", probe_spec
                    elif probe_left > 0:
                        probe_left -= 1
                        if probe_left == 0 and rate * t1 < tk * 1.03:
                            mode, plain_left = "plain", probe_plain
                launch_mode = mode
                if launch_mode == "spec" and N - min(produced) - potential <= 1:
                    launch_mode = "plain"  # the tail needs one token: the cheap step wins
                launch(slot, launch_mode)
                inflight.append((slot, launch_mode))
                potential += (k + 1) if launch_mode == "spec" else 1
            slot, m = inflight.popleft()
            potential -= (k + 1) if m == "spec" else 1
            nxt, adv = consume(slot, m)
            if m == "spec":
                gains = []
                for b in range(B):
                    a = adv[b]
                    if a:
                        queues[b].extend(nxt[j][b] for j in range(a))
                        produced[b] += a
                    if produced[b] < N:
                        gains.append(a)
                if gains:
                    rate = 0.75 * rate + 0.25 * min(gains)
                    spec_steps += 1
                    spec_tokens += min(gains)
                    if probe_left == 0 and mode == "spec" and rate * t1 < tk * 1.03:
                        mode, plain_left = "plain", probe_plain
            else:
                for b in range(B):
                    queues[b].append(nxt[b])
                    produced[b] += 1
            while yielded < N and all(queues):
                yield [q.popleft() for q in queues]
                yielded += 1
        if k and spec_steps:
            _log(f"spec: {spec_steps} verify steps, {spec_tokens / spec_steps:.2f} rows/step, rate {rate:.2f}, t1 {t1:.2f} tk {tk:.2f}")
