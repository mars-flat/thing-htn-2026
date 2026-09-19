"""Invariant: the decode loop never launches a step that cannot be needed.

Leftover in-flight replays after the last yield would run into the next
sample's prefill on the GPU (TTFT gate). Count step launches on the eager
backend (same _decode_loop as the CUDA backend):

  plain mode (spec_k=0): exactly N-1 decode steps for every shape;
  spec mode: launches <= what the yields required, i.e. after the last yield
  the number of launched-but-unconsumed steps is 0.

    .venv/bin/python tests/launch_count.py --device cpu
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import local_check as lc  # noqa: E402


def count_launches(engine, prompt, N):
    counts = {"plain": 0, "spec": 0}
    orig_plain, orig_spec = engine._plain_step, engine._spec_step

    def plain():
        counts["plain"] += 1
        return orig_plain()

    def spec(k):
        counts["spec"] += 1
        return orig_spec(k)

    engine._plain_step, engine._spec_step = plain, spec
    try:
        steps = list(engine.generate(prompt, N))
    finally:
        engine._plain_step, engine._spec_step = orig_plain, orig_spec
    assert len(steps) == N, (len(steps), N)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(lc.DEFAULT_MODEL))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.environ["DRYFT_ENGINE_DEVICE"] = args.device
    os.environ["DRYFT_ENGINE_TRITON"] = "0"
    Engine = lc.import_engine_class("engine")
    ok = True

    # Plain mode: exactly N-1 steps.
    os.environ["DRYFT_ENGINE_SPEC_K"] = "0"
    engine = Engine(args.model)
    for B, S, N in ((1, 8, 1), (1, 8, 2), (2, 12, 7), (3, 5, 16)):
        prompt = [[int(x) for x in torch.randint(0, 1000, (S,), generator=torch.Generator().manual_seed(B * S + N))] for _ in range(B)]
        c = count_launches(engine, prompt, N)
        good = c["plain"] == N - 1 and c["spec"] == 0
        ok &= good
        print(f"plain B={B} S={S} N={N}: launches {c} {'OK' if good else 'BAD (expected N-1 plain)'}")

    # Spec mode: periodic prompt accepts everything; random accepts nothing; mixed batch.
    os.environ["DRYFT_ENGINE_SPEC_K"] = "4"
    engine = Engine(args.model)
    k = 4
    periodic = ([11, 220, 3, 5000, 77, 9, 42] * 10)[:56]
    random_ids = [int(x) for x in torch.randint(0, 1000, (56,), generator=torch.Generator().manual_seed(9))]
    for name, prompt, N in (("periodic", [periodic], 41), ("periodic", [periodic], 1), ("periodic", [periodic], 2),
                            ("random", [random_ids], 12), ("mixed", [periodic, random_ids], 30)):
        c = count_launches(engine, prompt, N)
        # Upper bound: never more work than the plain path would do (N-1 tokens per step-equivalent),
        # and never more spec steps than a fully-rejected run would need.
        total_potential = c["plain"] + c["spec"] * (k + 1)
        good = c["plain"] + c["spec"] <= max(N - 1, 0) and (N == 1 or total_potential >= N - 1)
        ok &= good
        print(f"spec {name} N={N}: launches {c} potential {total_potential} {'OK' if good else 'BAD'}")
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
