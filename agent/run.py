"""Package engine/, submit it, start a run, wait, and record what it measured.

    .venv/bin/python agent/run.py --mode public --note "static cache + graphs"
    .venv/bin/python agent/run.py --mode official --note "..."
    .venv/bin/python agent/run.py --result RUN_ID          # re-print a finished run
    .venv/bin/python agent/run.py --logs RUN_ID            # dump the engine's stdout/stderr tail

Reads DRYFT_TOKEN from the environment or from ../.env (KEY=VALUE lines).
Every run's JSON is stored under notes/runs/ and one line is appended to
notes/experiments.md so the research loop keeps its history.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from client import Dryft  # noqa: E402
from loop import report  # noqa: E402
from package import package  # noqa: E402

ENGINE_DIR = ROOT / "engine"
RUNS_DIR = ROOT / "notes" / "runs"
LOG_FILE = ROOT / "notes" / "experiments.md"


def load_env():
    for candidate in (ROOT / ".env", ROOT.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    os.environ.setdefault("DRYFT_API", "https://htn.dryft.ai")


def dump_logs(client, run_id, limit=2000):
    after, lines = -1, []
    while True:
        page = client.logs(run_id, after=after, limit=200)
        items = page.get("items") or []
        if not items:
            break
        for item in items:
            lines.append(item)
            after = item.get("sequence", item.get("seq", after + 1))
        if len(items) < 200 or len(lines) >= limit:
            break
    return lines


def render_log(item):
    if isinstance(item, dict):
        stream = item.get("stream") or item.get("level") or ""
        text = item.get("message") or item.get("text") or json.dumps(item)
        return f"[{stream}] {text}" if stream else text
    return str(item)


def record(detail, mode, note, ok):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = detail.get("id", "unknown")
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(detail, indent=2))
    result = detail.get("result") or {}
    shapes = result.get("shapes") or []
    cells = []
    for shape in shapes:
        tps = shape.get("tokensPerSecond")
        metrics = shape.get("modelMetrics") or {}
        ttft = tpot = None
        if metrics.get("ttftMs") and metrics.get("referenceTtftMs"):
            ttft = metrics["ttftMs"] / metrics["referenceTtftMs"]
        if metrics.get("tpotMs") and metrics.get("referenceTpotMs"):
            tpot = metrics["tpotMs"] / metrics["referenceTpotMs"]
        cells.append(
            f"{shape.get('id')}: {shape.get('caseStatus')} "
            f"{tps:.0f}tok/s" if tps else f"{shape.get('id')}: {shape.get('caseStatus')}"
        )
        if ttft and tpot:
            cells[-1] += f" ttft {ttft:.2f}x tpot {tpot:.2f}x"
    line = (
        f"- {time.strftime('%Y-%m-%d %H:%M')} {mode} run `{run_id}` "
        f"{'OK' if ok else 'FAIL'} score={result.get('score')} | {note} | " + "; ".join(cells)
    )
    failure = result.get("failureMessage") or result.get("failureCode") or detail.get("errorMessage")
    if failure:
        line += f" | failure: {failure}"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as handle:
        handle.write(line + "\n")
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="public", choices=["public", "official"])
    parser.add_argument("--note", default="")
    parser.add_argument("--timeout", type=float, default=3000)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--result", metavar="RUN_ID")
    parser.add_argument("--logs", metavar="RUN_ID")
    parser.add_argument("--submission", metavar="SUBMISSION_ID", help="reuse an uploaded archive")
    args = parser.parse_args()

    load_env()
    client = Dryft()

    if args.logs:
        for item in dump_logs(client, args.logs):
            print(render_log(item))
        return 0
    if args.result:
        detail = client.wait(args.result, timeout=args.timeout) if not args.no_wait else client.run(args.result)
        ok = report(detail)
        record(detail, detail.get("mode", "?"), args.note, ok)
        return 0 if ok else 1

    submission_id = args.submission
    if not submission_id:
        archive = package(ENGINE_DIR)
        print(f"packaged {ENGINE_DIR} -> {len(archive)} bytes")
        submission_id = client.submit(archive)
    started = client.start_run(submission_id, mode=args.mode)
    run_id = started["id"]
    print(f"submission {submission_id}, {args.mode} run {run_id}")
    if args.no_wait:
        return 0
    detail = client.wait(run_id, timeout=args.timeout)
    ok = report(detail)
    record(detail, args.mode, args.note, ok)
    if not ok:
        print("---- engine log tail ----")
        try:
            items = dump_logs(client, run_id)
            for item in items[-120:]:
                print(render_log(item))
        except Exception as error:  # noqa: BLE001
            print(f"(logs unavailable: {error})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
