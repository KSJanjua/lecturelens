"""Validate the LLM judge against your own labels (section 7.4 of the plan).

    python eval/judge_agreement.py export --n 20   # writes results/judge_sample.csv to label by hand
    python eval/judge_agreement.py score           # agreement % and Cohen's kappa
    python eval/judge_agreement.py crosscheck --n 40   # second opinion from another model family

The judge's verdicts go to a separate key file so they cannot bias your labels."""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import store  # noqa: E402
from pipeline.chunk import chunk_text_with_timestamps  # noqa: E402
from pipeline.utils import read_json  # noqa: E402

SAMPLE = config.RESULTS_DIR / "judge_sample.csv"
KEY = config.RESULTS_DIR / "judge_sample_key.csv"
LABELS = ("supported", "contradicted", "not_mentioned")


def _sample(n: int, seed: int, experiments_only: bool = False) -> list[tuple[dict, str]]:
    """Stratified sample of (claim, evidence transcript) from faithfulness reports."""
    items = []
    reports = [] if experiments_only else list(config.LECTURES_DIR.glob("*/kits/*/faithfulness.json"))
    reports += list((config.RESULTS_DIR / "outputs" / "exp2").glob("*_faithfulness.json"))
    for report_path in sorted(reports):
        data = read_json(report_path)
        lecture_id = data.get("lecture_id") or report_path.parts[-4]
        chunks = {c.id: c for c in store.load_chunks(lecture_id) or []}
        for claim in data["claims"]:
            evidence = "\n".join(chunk_text_with_timestamps(chunks[i]) for i in claim["evidence_chunks"] if i in chunks)
            items.append((claim, evidence))
    if not items:
        sys.exit("No faithfulness reports found. Generate some study kits first.")
    random.Random(seed).shuffle(items)
    # Stratify so rarer verdicts are represented.
    picked, by_verdict = [], {}
    for item in items:
        by_verdict.setdefault(item[0]["verdict"], []).append(item)
    while len(picked) < min(n, len(items)):
        for verdict in list(by_verdict):
            if by_verdict[verdict] and len(picked) < n:
                picked.append(by_verdict[verdict].pop())
    return picked


def export(n: int, seed: int):
    picked = _sample(n, seed)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE, "w", newline="", encoding="utf-8") as f, open(KEY, "w", newline="", encoding="utf-8") as k:
        sample, key = csv.writer(f), csv.writer(k)
        sample.writerow(["id", "claim", "transcript_excerpt", f"your_label ({'/'.join(LABELS)})"])
        key.writerow(["id", "judge_verdict"])
        for i, (claim, evidence) in enumerate(picked, 1):
            sample.writerow([i, claim["claim"], evidence[:3000], ""])
            key.writerow([i, claim["verdict"]])
    print(f"Wrote {len(picked)} items to {SAMPLE}. Fill in the last column, then run `score`.")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(a) | set(b)) / (n * n)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def score():
    with open(SAMPLE, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    with open(KEY, encoding="utf-8") as f:
        key = {r[0]: r[1] for r in list(csv.reader(f))[1:]}
    human, judge = [], []
    for row in rows:
        label = row[3].strip().lower().replace(" ", "_")
        if label in LABELS:
            human.append(label)
            judge.append(key[row[0]])
    if not human:
        sys.exit("No labels found - fill in the last column of judge_sample.csv first.")
    agree = sum(h == j for h, j in zip(human, judge)) / len(human)
    binary = sum((h == "supported") == (j == "supported") for h, j in zip(human, judge)) / len(human)
    print(f"Labelled items: {len(human)}")
    print(f"Exact agreement: {agree:.0%} · supported-vs-not agreement: {binary:.0%} · "
          f"Cohen's kappa: {cohen_kappa(human, judge)}")


def crosscheck(n: int, seed: int, model: str):
    """Re-judge sampled claims with a model from a different family. Claims that share the same
    evidence are judged in one call, the same way the main judge saw them."""
    from pipeline import llm, prompts
    from pipeline.faithfulness import _normalise
    from pipeline.schemas import Verdicts

    provider, model_name = model.split(":", 1)
    groups: dict[str, list[dict]] = {}
    for claim, evidence in _sample(n, seed, experiments_only=True):
        groups.setdefault(evidence, []).append(claim)
    rows = []
    for evidence, claims in groups.items():
        listing = "\n".join(f"{i + 1}. {c['claim']}" for i, c in enumerate(claims))
        result = llm.generate_json(prompts.render("claims_judge", transcript=evidence, claims=listing),
                                   Verdicts, provider=provider, model=model_name, temperature=0.0,
                                   task="claims_judge")
        by_id = {v.id: _normalise(v.verdict) for v in result.verdicts}
        for i, c in enumerate(claims):
            rows.append({"claim": c["claim"], "main_judge": c["verdict"],
                         "cross_judge": by_id.get(i + 1, "not_mentioned"), "cross_model": model})
    out = config.RESULTS_DIR / "judge_crosscheck.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    a, b = [r["main_judge"] for r in rows], [r["cross_judge"] for r in rows]
    exact = sum(x == y for x, y in zip(a, b)) / len(rows)
    binary = sum((x == "supported") == (y == "supported") for x, y in zip(a, b)) / len(rows)
    print(f"Cross-check on {len(rows)} claims in {len(groups)} calls ({model}): exact agreement {exact:.0%}, "
          f"supported-vs-not {binary:.0%}, Cohen's kappa {cohen_kappa(a, b)} -> {out}")
    return {"n": len(rows), "exact": round(exact, 3), "binary": round(binary, 3), "kappa": cohen_kappa(a, b)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--n", type=int, default=20)
    ex.add_argument("--seed", type=int, default=7)
    sub.add_parser("score")
    cc = sub.add_parser("crosscheck")
    cc.add_argument("--n", type=int, default=40)
    cc.add_argument("--seed", type=int, default=7)
    cc.add_argument("--model", default="gemini:gemini-3.8-flash", help="provider:model of the second judge")
    args = parser.parse_args()
    if args.cmd == "export":
        export(args.n, args.seed)
    elif args.cmd == "crosscheck":
        crosscheck(args.n, args.seed, args.model)
    else:
        score()
