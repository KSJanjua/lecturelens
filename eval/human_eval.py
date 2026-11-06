"""Blind A/B human evaluation of two prompting strategies (run Experiment 2 first).

    python eval/human_eval.py packet --a zero_shot --b tot   # writes results/human_eval/
    python eval/human_eval.py score form_export.csv          # -> data/human_ratings.csv

Each lecture's two notes are shown as "Version 1" and "Version 2" in random order. Paste
the packet into a Google Form with one 1-5 rating question per version. Export responses
as CSV with columns: lecture, version, rating."""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.export import notes_to_markdown  # noqa: E402
from pipeline.schemas import Notes  # noqa: E402
from pipeline.utils import read_json  # noqa: E402

OUT = config.RESULTS_DIR / "human_eval"
SOURCE = config.RESULTS_DIR / "outputs" / "exp2"


def packet(a: str, b: str, seed: int):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key_rows = []
    lectures = sorted(p.stem[:-len(a) - 1] for p in SOURCE.glob(f"*_{a}.json"))
    for lecture in lectures:
        files = {s: SOURCE / f"{lecture}_{s}.json" for s in (a, b)}
        if not all(p.exists() for p in files.values()):
            continue
        order = [a, b]
        rng.shuffle(order)
        parts = [f"# Lecture {lecture}\n\nRate each version from 1 (poor) to 5 (excellent) for accuracy, "
                 "clarity and usefulness for revision.\n"]
        for n, strategy in enumerate(order, 1):
            notes = Notes.model_validate(read_json(files[strategy])["notes"])
            notes.title = f"Version {n}"
            parts.append(notes_to_markdown(notes).replace("# Version", "## Version", 1))
            key_rows.append({"lecture": lecture, "version": n, "strategy": strategy})
        (OUT / f"{lecture}.md").write_text("\n\n".join(parts), encoding="utf-8")
    with open(OUT / "key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lecture", "version", "strategy"])
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"Wrote {len(key_rows) // 2} lecture packets to {OUT} (keep key.csv away from raters).")


def score(form_csv: str):
    with open(OUT / "key.csv", encoding="utf-8") as f:
        key = {(r["lecture"], r["version"]): r["strategy"] for r in csv.DictReader(f)}
    out_rows = []
    with open(form_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            strategy = key.get((row["lecture"], str(row["version"])))
            if strategy and row.get("rating"):
                out_rows.append({"lecture": row["lecture"], "strategy": strategy, "rating": row["rating"]})
    path = config.EVAL_DATA_DIR / "human_ratings.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lecture", "strategy", "rating"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} ratings to {path}; re-run Experiment 2 to fill the human-rating column.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packet")
    p.add_argument("--a", default="zero_shot")
    p.add_argument("--b", default="tot")
    p.add_argument("--seed", type=int, default=11)
    s = sub.add_parser("score")
    s.add_argument("form_csv")
    args = parser.parse_args()
    packet(args.a, args.b, args.seed) if args.cmd == "packet" else score(args.form_csv)
