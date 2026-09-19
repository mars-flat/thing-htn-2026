#!/usr/bin/env python
"""Local correctness harness for the Dryft Qwen3 engine (SPEC.md "Testing").

Runs on CPU or MPS against Qwen3-0.6B (no CUDA, no Triton). For every case it

  1. builds a seeded random prompt of shape [B, S] (ids uniform in [0, vocab)),
  2. runs the candidate ``Engine.generate`` and checks the yield protocol:
     exactly N yields, each a ``list`` of B Python ints inside the vocab,
  3. runs the HF baseline greedy loop (the loop of engine/engine.py) on the
     same prompt and reports the token match rate (informational),
  4. applies the JUDGE RULE: replays the candidate's tokens teacher-forced
     through HF in ONE full forward (prompt + emitted, use_cache=False) and
     requires, at every emitted position, logit[emitted] >= max_logit - margin.

State checks: two consecutive calls on the same Engine with the same shape and
different prompts (state leak), then calls with a different batch size (shape
change / rebuild). All are judged by the same rule.

The env vars DRYFT_ENGINE_DEVICE=<device> and DRYFT_ENGINE_TRITON=0 are set
BEFORE the engine module is imported; the engine is expected to honour them.

Usage:
    .venv/bin/python tests/local_check.py                        # candidate = engine/engine.py
    .venv/bin/python tests/local_check.py --baseline-as-candidate  # harness self-test
    .venv/bin/python tests/local_check.py --device mps --shapes 1x5x4,2x17x6

Exit code 0 iff every case passes (protocol + judge rule).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import itertools
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT.parent / "models" / "Qwen3-0.6B"
DEFAULT_SHAPES = "1x5x4,2x17x6,3x64x4,1x130x9"
# (name, B, S, N): same shape twice with different prompts, then B changes.
STATE_CASES = [
    ("leak-1", 2, 17, 6),
    ("leak-2", 2, 17, 6),
    ("bchange-3", 3, 17, 6),
    ("bchange-1", 1, 17, 6),
]
MAX_DETAIL_LINES = 20


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="HF checkpoint directory")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument(
        "--engine-module",
        default="engine",
        help="'engine' (default: <project>/engine/engine.py with <project>/engine on "
        "sys.path so 'kernels' resolves), a path to a .py file (its parent dir goes on "
        "sys.path), or an importable module name. Must export class Engine.",
    )
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma-separated BxSxN")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=2.0, help="judge tie margin in logits")
    p.add_argument(
        "--baseline-as-candidate",
        action="store_true",
        help="use an HF-based candidate defined in this file (engine/engine.py's loop "
        "on --device) instead of importing the engine; smoke-tests the harness",
    )
    p.add_argument("--no-state-checks", action="store_true", help="skip leak/shape-change cases")
    return p.parse_args(argv)


def parse_shapes(spec: str) -> list[tuple[int, int, int]]:
    shapes = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) != 3:
            raise SystemExit(f"bad shape {item!r}: expected BxSxN")
        b, s, n = (int(v) for v in parts)
        if min(b, s, n) < 1:
            raise SystemExit(f"bad shape {item!r}: all of B, S, N must be >= 1")
        shapes.append((b, s, n))
    if not shapes:
        raise SystemExit("no shapes given")
    return shapes


# ----------------------------------------------------------------------------
# Candidate loading
# ----------------------------------------------------------------------------
class BaselineCandidate:
    """HF-based stand-in for ``engine.Engine``.

    Mirrors engine/engine.py's greedy loop exactly, but on DRYFT_ENGINE_DEVICE
    instead of the hardcoded cuda:0, so the harness can be smoke-tested where
    there is no CUDA. It must produce a 100% token match and pass the judge.
    """

    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM

        self.device = torch.device(os.environ.get("DRYFT_ENGINE_DEVICE", "cuda:0"))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to(self.device)
        )

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        current = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        cache = None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                output = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist()


def import_engine_class(engine_module: str):
    """Import the candidate the way the judge does: archive root on sys.path,
    ``import engine``. Returns the ``Engine`` class."""
    if engine_module.endswith(".py"):
        path = Path(engine_module).resolve()
        root, name = path.parent, path.stem
    elif engine_module == "engine":
        root, name = PROJECT_ROOT / "engine", "engine"
    else:
        root, name = None, engine_module
    if root is not None:
        root_s = str(root)
        if root_s in sys.path:
            sys.path.remove(root_s)
        sys.path.insert(0, root_s)
    mod = importlib.import_module(name)
    if not hasattr(mod, "Engine"):
        raise ImportError(f"{mod.__file__} does not export class Engine")
    return mod.Engine


def load_baseline(model_path: str, device: str):
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .to(device)
    )


# ----------------------------------------------------------------------------
# Pieces of one case
# ----------------------------------------------------------------------------
def make_prompt(B: int, S: int, vocab: int, seed: int) -> list[list[int]]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab, (B, S), generator=g, dtype=torch.int64)
    return ids.tolist()


def run_candidate(engine, prompt: list[list[int]], N: int, B: int, vocab: int):
    """Drive ``engine.generate`` like the judge; validate the yield protocol.

    Returns (tokens [B][N] or None, n_steps, problems, seconds).
    """
    problems: list[str] = []
    t0 = time.perf_counter()
    # Pull at most N+1 items: the (N+1)-th pull must raise StopIteration.
    steps = list(itertools.islice(engine.generate([list(r) for r in prompt], N), N + 1))
    seconds = time.perf_counter() - t0
    n_steps = len(steps)
    if n_steps != N:
        problems.append(f"yielded {'>' if n_steps > N else ''}{n_steps} steps, expected exactly {N}")
    for i, step in enumerate(steps[:N]):
        if not isinstance(step, list):
            problems.append(f"step {i}: yielded {type(step).__name__}, expected list")
            continue
        if len(step) != B:
            problems.append(f"step {i}: list of {len(step)}, expected {B}")
            continue
        for b, t in enumerate(step):
            if type(t) is not int:  # bool / numpy / tensor scalars all fail here
                problems.append(f"step {i} seq {b}: token is {type(t).__name__}, expected int")
                break
            if not (0 <= t < vocab):
                problems.append(f"step {i} seq {b}: token {t} outside [0, {vocab})")
                break
        if len(problems) >= MAX_DETAIL_LINES:
            problems.append("... (more problems suppressed)")
            break
    if problems:
        return None, n_steps, problems, seconds
    tokens = [[steps[n][b] for n in range(N)] for b in range(B)]
    return tokens, n_steps, problems, seconds


@torch.inference_mode()
def hf_greedy(model, prompt_t: torch.Tensor, N: int) -> list[list[int]]:
    """Reference tokens: the exact loop of engine/engine.py on the baseline."""
    current = prompt_t
    cache = None
    steps = []
    for _ in range(N):
        output = model(
            input_ids=current,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        steps.append(current[:, 0].tolist())
    B = prompt_t.shape[0]
    return [[steps[n][b] for n in range(N)] for b in range(B)]


@torch.inference_mode()
def judge_replay(model, prompt_t: torch.Tensor, tokens: list[list[int]]):
    """One full teacher-forced forward over prompt + emitted tokens.

    Returns fp32 logits ``pred [B, N, V]`` (on CPU) where pred[b, n] is the
    distribution that predicts emitted token n of sequence b, i.e. the logits
    at sequence position S + n - 1.
    """
    B, S = prompt_t.shape
    N = len(tokens[0])
    tok_t = torch.tensor(tokens, dtype=torch.int64, device=prompt_t.device)
    full = torch.cat([prompt_t, tok_t], dim=1)
    out = model(input_ids=full, use_cache=False, return_dict=True)
    logits = out.logits  # [B, S+N, V] bf16, all positions
    assert logits.shape[1] == S + N, logits.shape
    pred = logits[:, S - 1 : S + N - 1, :].float().cpu()
    del out, logits, full
    return pred


@dataclass
class CaseResult:
    name: str
    shape: tuple[int, int, int]
    yields: str = "-"
    match: str = "-"
    replay_argmax: str = "-"
    max_gap: float | None = None
    headroom: float | None = None
    n_mismatch: int = 0
    n_fail: int = 0
    seconds: float = 0.0
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def result(self) -> str:
        return "PASS" if self.passed else "FAIL"


def run_case(name, B, S, N, seed, engine, model, device, vocab, margin) -> CaseResult:
    res = CaseResult(name=name, shape=(B, S, N))
    t_case = time.perf_counter()
    prompt = make_prompt(B, S, vocab, seed)

    # 1. candidate + protocol
    try:
        tokens, n_steps, problems, t_cand = run_candidate(engine, prompt, N, B, vocab)
    except Exception:
        res.notes.append("candidate raised:\n" + traceback.format_exc(limit=12).rstrip())
        res.seconds = time.perf_counter() - t_case
        return res
    res.yields = f"{n_steps}/{N}"
    if problems:
        res.notes.extend("protocol: " + p for p in problems)
        res.seconds = time.perf_counter() - t_case
        return res

    # 2. HF greedy reference (informational: mismatches are legal near-ties)
    prompt_t = torch.tensor(prompt, dtype=torch.int64, device=device)
    t0 = time.perf_counter()
    ref = hf_greedy(model, prompt_t, N)
    t_hf = time.perf_counter() - t0
    tok_t = torch.tensor(tokens, dtype=torch.int64)
    ref_t = torch.tensor(ref, dtype=torch.int64)
    matched = int((tok_t == ref_t).sum())
    res.match = f"{matched}/{B * N}"

    # 3. judge rule on the candidate's own prefix
    t0 = time.perf_counter()
    pred = judge_replay(model, prompt_t, tokens)  # [B, N, V] fp32
    t_rep = time.perf_counter() - t0
    max_logit = pred.amax(dim=-1)  # [B, N]
    argmax = pred.argmax(dim=-1)
    emitted = pred.gather(-1, tok_t.unsqueeze(-1)).squeeze(-1)
    ref_logit = pred.gather(-1, ref_t.unsqueeze(-1)).squeeze(-1)
    ok = emitted >= (max_logit - margin)
    gap = max_logit - emitted  # >= 0; must be <= margin
    gap_ref = max_logit - ref_logit
    res.max_gap = float(gap.max())
    res.headroom = margin - res.max_gap
    res.replay_argmax = f"{int((argmax == tok_t).sum())}/{B * N}"
    res.n_fail = int((~ok).sum())
    res.passed = bool(ok.all())

    mism = [(b, n) for b in range(B) for n in range(N) if tokens[b][n] != ref[b][n]]
    res.n_mismatch = len(mism)
    worst = torch.unravel_index(gap.argmax(), gap.shape)
    res.notes.append(
        f"candidate {t_cand:.2f}s | hf-greedy {t_hf:.2f}s | replay {t_rep:.2f}s | "
        f"worst gap {res.max_gap:.4f} at b={int(worst[0])} n={int(worst[1])}"
    )
    for b, n in mism[:MAX_DETAIL_LINES]:
        res.notes.append(
            f"mismatch b={b} n={n}: engine={tokens[b][n]} ref={ref[b][n]} "
            f"gap(engine)={float(gap[b, n]):.4f} gap(ref)={float(gap_ref[b, n]):.4f} "
            f"replay_argmax={int(argmax[b, n])}"
        )
    if len(mism) > MAX_DETAIL_LINES:
        res.notes.append(f"... {len(mism) - MAX_DETAIL_LINES} more mismatches")
    fails = [(b, n) for b in range(B) for n in range(N) if not bool(ok[b, n]) and (b, n) not in mism]
    for b, n in fails[:MAX_DETAIL_LINES]:
        res.notes.append(
            f"JUDGE FAIL b={b} n={n}: engine={tokens[b][n]} gap={float(gap[b, n]):.4f} "
            f"> margin {margin} (replay_argmax={int(argmax[b, n])})"
        )

    del pred, prompt_t, max_logit, argmax, emitted, ref_logit, ok, gap, gap_ref
    res.seconds = time.perf_counter() - t_case
    return res


def free_memory(device: str) -> None:
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
COLUMNS = [
    ("case", 10),
    ("shape", 10),
    ("yields", 7),
    ("match", 8),
    ("replay_argmax", 13),
    ("max_gap", 8),
    ("headroom", 9),
    ("mism", 5),
    ("nfail", 5),
    ("sec", 7),
    ("result", 6),
]


def fmt_row(values) -> str:
    return "  ".join(str(v).ljust(w) for v, (_, w) in zip(values, COLUMNS))


def row_of(r: CaseResult):
    fmt = lambda x: "-" if x is None else f"{x:.4f}"
    return [
        r.name,
        "x".join(map(str, r.shape)),
        r.yields,
        r.match,
        r.replay_argmax,
        fmt(r.max_gap),
        fmt(r.headroom),
        r.n_mismatch,
        r.n_fail,
        f"{r.seconds:.1f}",
        r.result,
    ]


def print_case(r: CaseResult) -> None:
    print(f"[{r.name}] {'x'.join(map(str, r.shape))}: {r.result}", flush=True)
    for line in r.notes:
        for sub in line.splitlines():
            print("    " + sub)
    sys.stdout.flush()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    # The engine reads these at import/construct time; set before importing it.
    os.environ["DRYFT_ENGINE_DEVICE"] = args.device
    os.environ["DRYFT_ENGINE_TRITON"] = "0"

    model_path = Path(args.model).resolve()
    missing = [f for f in ("config.json",) if not (model_path / f).exists()]
    has_weights = any(model_path.glob("*.safetensors")) or (model_path / "pytorch_model.bin").exists()
    if missing or not has_weights:
        print(f"model directory {model_path} is incomplete: missing {missing or ''}"
              f"{' weights' if not has_weights else ''}", file=sys.stderr)
        return 2
    shapes = parse_shapes(args.shapes)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    print(f"torch {torch.__version__} | device {args.device} | model {model_path}")
    print(f"env DRYFT_ENGINE_DEVICE={os.environ['DRYFT_ENGINE_DEVICE']} "
          f"DRYFT_ENGINE_TRITON={os.environ['DRYFT_ENGINE_TRITON']} | seed {args.seed} | margin {args.margin}")

    # Candidate class first (fast failure on import errors), before the heavy load.
    try:
        if args.baseline_as_candidate:
            engine_cls = BaselineCandidate
            print("candidate: BaselineCandidate (HF loop defined in this harness)")
        else:
            engine_cls = import_engine_class(args.engine_module)
            print(f"candidate: {engine_cls.__module__}.{engine_cls.__name__} "
                  f"from {sys.modules[engine_cls.__module__].__file__}")
    except Exception:
        print("failed to import the candidate engine:", file=sys.stderr)
        traceback.print_exc()
        return 2

    t0 = time.perf_counter()
    model = load_baseline(str(model_path), args.device)
    cfg = model.config
    vocab = int(cfg.vocab_size)
    print(f"baseline loaded in {time.perf_counter() - t0:.2f}s: H={cfg.hidden_size} I={cfg.intermediate_size} "
          f"L={cfg.num_hidden_layers} Nq={cfg.num_attention_heads} Nkv={cfg.num_key_value_heads} "
          f"D={cfg.head_dim} V={vocab} theta={cfg.rope_theta} eps={cfg.rms_norm_eps} "
          f"attn={cfg._attn_implementation}")

    t0 = time.perf_counter()
    try:
        engine = engine_cls(str(model_path))
    except Exception:
        print("candidate Engine.__init__ raised:", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(f"candidate loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    cases = [(f"shape-{i}", B, S, N) for i, (B, S, N) in enumerate(shapes)]
    if not args.no_state_checks:
        cases += STATE_CASES

    results: list[CaseResult] = []
    for idx, (name, B, S, N) in enumerate(cases):
        seed = args.seed * 1_000_003 + idx * 7_919 + 1
        r = run_case(name, B, S, N, seed, engine, model, args.device, vocab, args.margin)
        results.append(r)
        print_case(r)
        free_memory(args.device)

    print()
    print(fmt_row([c for c, _ in COLUMNS]))
    print(fmt_row(["-" * w for _, w in COLUMNS]))
    for r in results:
        print(fmt_row(row_of(r)))
    n_pass = sum(r.passed for r in results)
    all_pass = n_pass == len(results)
    worst = max((r.max_gap for r in results if r.max_gap is not None), default=None)
    print()
    print(f"{'ALL PASS' if all_pass else 'FAILED'}: {n_pass}/{len(results)} cases passed"
          + (f" | worst gap {worst:.4f} (margin {args.margin})" if worst is not None else ""))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
