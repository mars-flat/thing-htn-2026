"""Exercise speculative decoding on prompts where drafts actually get accepted.

local_check.py uses random-token prompts, which almost never produce an n-gram
hit, so the acceptance path (adv > 1, hist scatter of several tokens, frozen
sequences) is barely exercised there. This script builds periodic prompts and
natural text repeated, runs the engine with DRYFT_ENGINE_SPEC_K set, and
applies the same judge rule (teacher-forced replay, 2.0 logit margin) plus a
comparison with HF greedy.

    DRYFT_ENGINE_SPEC_K=4 .venv/bin/python tests/spec_check.py --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import local_check as lc  # noqa: E402


def periodic_prompt(period_tokens, S):
    return [(period_tokens * (S // len(period_tokens) + 1))[:S]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(lc.DEFAULT_MODEL))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--N", type=int, default=40)
    args = parser.parse_args()
    os.environ["DRYFT_ENGINE_DEVICE"] = args.device
    os.environ["DRYFT_ENGINE_TRITON"] = "0"
    os.environ.setdefault("DRYFT_ENGINE_SPEC_K", "4")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = lc.load_baseline(args.model, args.device)
    vocab = model.config.vocab_size
    Engine = lc.import_engine_class("engine")
    engine = Engine(args.model)

    text = ("The quick brown fox jumps over the lazy dog. " * 12).strip()
    text_ids = tok(text)["input_ids"]
    cases = {
        "periodic7_b1": periodic_prompt([11, 220, 3, 5000, 77, 9, 42], 64),
        "text_b1": [text_ids[:96]],
        "mixed_b3": periodic_prompt([1, 2, 3, 4, 5], 48) + [text_ids[:48]] + [[int(x) for x in torch.randint(0, vocab, (48,), generator=torch.Generator().manual_seed(3))]],
    }
    all_ok = True
    for name, prompt in cases.items():
        B, S = len(prompt), len(prompt[0])
        t0 = time.time()
        tokens, n_steps, problems, secs = lc.run_candidate(engine, prompt, args.N, B, vocab)
        if problems:
            print(f"[{name}] PROTOCOL FAIL: {problems}")
            all_ok = False
            continue
        prompt_t = torch.tensor(prompt, dtype=torch.int64, device=args.device)
        ref = lc.hf_greedy(model, prompt_t, args.N)
        pred = lc.judge_replay(model, prompt_t, tokens)
        worst = 0.0
        fails = 0
        match = 0
        for b in range(B):
            for n in range(args.N):
                logits = pred[b, n]
                gap = float(logits.max() - logits[tokens[b][n]])
                worst = max(worst, gap)
                if gap > 2.0:
                    fails += 1
                match += int(tokens[b][n] == ref[b][n])
        ok = fails == 0
        all_ok &= ok
        print(f"[{name}] B={B} S={S} N={args.N}: {'PASS' if ok else 'FAIL'} match {match}/{B * args.N} "
              f"worst gap {worst:.4f} judge fails {fails} ({time.time() - t0:.1f}s)")
        print("   engine:", tokens[0][:16])
        print("   hf    :", ref[0][:16])
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
