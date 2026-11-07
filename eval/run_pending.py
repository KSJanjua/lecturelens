"""Run every remaining task until it is complete, waiting out free-tier daily quotas.

    python eval/run_pending.py                 # all tasks, in priority order
    python eval/run_pending.py --only exp2     # one task

Each task is a normal command whose finished work is cached; a task counts as done only when its
completion check passes. When a task stops early (usually a daily token quota), the runner waits
and tries again, so the whole list finishes as quota becomes available."""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline.utils import read_json  # noqa: E402

PY = sys.executable
RESULTS = config.RESULTS_DIR
GEN, JUDGE = "groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b"
CROSS_MODELS = ["gemini:gemini-3.1-flash-lite", "gemini:gemini-3.7-flash", "gemini:gemini-3.5-flash-lite"]


def rows(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def exp2_done():
    got = {(r["lecture"], r["strategy"]) for r in rows("exp2_prompting.csv") if r.get("unjudged", "0") in ("0", "")}
    return all((lec, s) in got for lec in ("L3", "L1", "L2") for s in ("zero_shot", "few_shot", "cot", "tot"))


def exp4_done():
    data = [r for r in rows("exp4_chunking.csv") if r["chunking"] in ("fixed-1000", "semantic")]
    answered = {(r["lecture"], r["chunking"]) for r in data if r.get("rag_accuracy")}
    scored = {(r["lecture"], r["chunking"]) for r in data if r.get("rougeL")}
    return (all((lec, c) in answered for lec in ("L1", "L2", "L3", "L4", "L5") for c in ("fixed-1000", "semantic"))
            and all((lec, c) in scored for lec in ("L1", "L2", "L3") for c in ("fixed-1000", "semantic")))


def exp3_done():
    data = rows("exp3_sampling.csv")
    return len(data) >= 24 and all(r.get("unjudged", "0") == "0" for r in data)


def crosscheck_done():
    return bool(rows("judge_crosscheck.csv"))


def kits_done():
    return bool((read_json(config.DATA_DIR / "demo_kits.json") or {}).get("complete"))


def run_eval(*args):
    return [PY, "-u", "eval/run_eval.py", *args, "--generator", GEN, "--judge", JUDGE, "--workers", "2"]


TASKS = [
    ("exp2", exp2_done, lambda attempt: run_eval("--exp", "2", "--lectures", "L3", "L1", "L2")),
    ("exp4", exp4_done, lambda attempt: run_eval("--exp", "4", "--exp4-llm", "--lectures", "L1", "L2", "L3", "L4", "L5")),
    ("crosscheck", crosscheck_done, lambda attempt: [
        PY, "-u", "eval/judge_agreement.py", "crosscheck", "--n", "30",
        "--model", CROSS_MODELS[min(attempt // 3, len(CROSS_MODELS) - 1)]]),
    ("demo-kits", kits_done, lambda attempt: [PY, "-u", "scripts/build_demo_kits.py"]),
    ("exp3", exp3_done, lambda attempt: [
        PY, "-u", "eval/run_eval.py", "--exp", "3", "--lectures", "L3", "--sampling-lectures", "1", "--repeats", "3",
        "--generator", "groq:openai/gpt-oss-20b", "--judge", JUDGE, "--workers", "2"]),
]


def log(msg, logfile):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="task names to run")
    parser.add_argument("--wait-minutes", type=float, default=20)
    parser.add_argument("--max-hours", type=float, default=72)
    args = parser.parse_args()

    logfile = RESULTS / "pending_log.txt"
    tasks = [t for t in TASKS if not args.only or t[0] in args.only]
    deadline = time.time() + args.max_hours * 3600
    for name, done, command in tasks:
        attempt = 0
        while not done():
            if time.time() > deadline:
                log(f"{name}: giving up at the time limit", logfile)
                sys.exit(1)
            cmd = command(attempt)
            log(f"{name}: attempt {attempt + 1}: {' '.join(cmd[2:])}", logfile)
            with open(RESULTS / f"{name}_run_log.txt", "a", encoding="utf-8") as out:
                code = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            attempt += 1
            if done():
                break
            log(f"{name}: incomplete (exit {code}); waiting {args.wait_minutes:.0f} min for quota", logfile)
            time.sleep(args.wait_minutes * 60)
        log(f"{name}: complete", logfile)
    log("all tasks complete", logfile)


if __name__ == "__main__":
    main()
