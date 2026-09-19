"""Kernel probe: compile + self-test the Triton kernels in a CHILD process.

The judge hides every byte of engine output, so an engine that dies during a
kernel self-test (device fault, hang, compiler crash) is indistinguishable
from a platform failure. Running the first-ever launches of every kernel in a
separate process turns such failures into "this kernel is disabled": the
parent keeps only kernels the child certified, and, because the child shares
the Triton cache directory, the parent's own launches are cache hits.

Usage (from the engine, archive root on sys.path):

    python -m kernels.probe OUT.json --device cuda:0 --M 1 --H 2560 --I 9728 \
        --NQ 32 --NKV 8 --D 128 --G 4 --max-T 1 --tests rmsnorm,silu,rope,attention,gemm,fused

Writes {"ok": bool, "results": {name: {...}}, "elapsed": {...}} to OUT.json.
Every test is wrapped so one failing module does not stop the others.
"""

import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--H", type=int, default=2560)
    parser.add_argument("--I", type=int, default=9728)
    parser.add_argument("--NQ", type=int, default=32)
    parser.add_argument("--NKV", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--max-T", type=int, default=1)
    parser.add_argument("--tests", default="rmsnorm,silu,rope,attention,gemm,fused")
    args = parser.parse_args()

    results, elapsed = {}, {}
    out = {"ok": False, "results": results, "elapsed": elapsed}

    def save():
        with open(args.out, "w") as handle:
            json.dump(out, handle, default=str)

    save()  # an empty file means "crashed before finishing"
    import torch  # noqa: E402

    torch.backends.cuda.matmul.allow_tf32 = False
    G = args.NQ // args.NKV
    for name in [t for t in args.tests.split(",") if t]:
        t0 = time.time()
        try:
            if name == "rmsnorm":
                from kernels import rmsnorm

                res = rmsnorm.self_test(args.device)
            elif name == "silu":
                from kernels import silu

                res = silu.self_test(args.device)
            elif name == "rope":
                from kernels import rope

                res = rope.self_test(args.device, dims=(args.NQ, args.NKV), max_T=max(64, args.max_T))
            elif name == "attention":
                from kernels import attention

                res = attention.self_test(args.device, G=G, max_T=args.max_T)
            elif name == "gemm":
                from kernels import gemm

                res = gemm.self_test(args.device, rows=(args.M,))
            elif name == "fused":
                from kernels import gemm_fused

                res = gemm_fused.self_test(args.device, M=args.M, H=args.H, I=args.I, NQ=args.NQ, NKV=args.NKV, D=args.D)
            else:
                res = {"ok": False, "error": f"unknown test {name}"}
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException as error:  # noqa: BLE001 - report everything, keep going
            res = {"ok": False, "error": f"{type(error).__name__}: {error}"[:300]}
        results[name] = res
        elapsed[name] = round(time.time() - t0, 1)
        save()
    out["ok"] = True
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
