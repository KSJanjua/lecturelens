"""Pre-build complete study kits for the demo lectures, so the app loads them without API calls.

    python scripts/build_demo_kits.py

Uses the default generator and judge from .env. Safe to re-run: finished stages are cached, and a
faithfulness report with unjudged claims (API errors) is rebuilt. Exits non-zero if a kit is incomplete."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import llm, store  # noqa: E402
from pipeline.profile import PRESETS  # noqa: E402
from pipeline.run import KitOptions, build_kit, kit_key  # noqa: E402
from pipeline.utils import read_json, write_json  # noqa: E402

# (manifest id, learner profile, strategy)
DEMOS = [
    ("L3", "Beginner (first exposure)", "tot"),
    ("L5", "Exam revision (exam tomorrow)", "tot"),
]
STAGE_FILES = {"notes": "notes.json", "quiz": "quiz.json", "flashcards": "flashcards.json",
               "diagrams": "diagrams.json", "audio": "recap.mp3", "faithfulness": "faithfulness.json"}


def lecture_id(manifest_id: str) -> str:
    manifest = read_json(config.EVAL_DATA_DIR / "manifest.json")["lectures"]
    url = next(e["source"] for e in manifest if e["id"] == manifest_id)
    match = next(m for m in store.list_lectures() if m.media_path == url or m.source == url)
    return match.id


def build(manifest_id: str, profile_name: str, strategy: str) -> dict:
    lid = lecture_id(manifest_id)
    profile = PRESETS[profile_name]
    opts = KitOptions(strategy=strategy)
    folder = store.kit_dir(lid, kit_key(profile, opts))
    report = read_json(folder / "faithfulness.json")
    if report and report.get("unjudged"):
        (folder / "faithfulness.json").unlink()              # rebuild only the faithfulness stage
    kit = build_kit(lid, profile, opts, progress=lambda label, f: print(f"    {f:4.0%} {label}", flush=True))
    missing = [s for s, name in STAGE_FILES.items() if not (folder / name).exists()]
    unjudged = kit["faithfulness"].unjudged if kit.get("faithfulness") else None
    return {"lecture": manifest_id, "lecture_id": lid, "kit": kit["key"], "profile": profile_name,
            "strategy": strategy, "missing": missing, "unjudged": unjudged,
            "faithfulness": kit["faithfulness"].score if kit.get("faithfulness") else None,
            "quiz": len(kit["quiz"].questions) if kit.get("quiz") else 0,
            "generator": "/".join(llm.resolve("generator")), "judge": "/".join(llm.resolve("judge"))}


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.LLM_FALLBACK = False                  # one consistent generator per kit
    results, ok = [], True
    for demo in DEMOS:
        print(f"Building {demo}", flush=True)
        try:
            result = build(*demo)
        except llm.LLMError as exc:
            print(f"  stopped (quota/API): {str(exc)[:200]}")
            ok = False
            continue
        print(f"  {result}", flush=True)
        results.append(result)
        ok &= not result["missing"] and not result["unjudged"]
    write_json(config.DATA_DIR / "demo_kits.json", {"complete": ok, "kits": results})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
