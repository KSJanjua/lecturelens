"""Run the evaluation experiments (section 7 of the project plan) and write CSVs + plots.

    python eval/run_eval.py                       # all experiments, all lectures
    python eval/run_eval.py --exp 2 --lectures L1 L2 --strategies zero_shot cot tot

Lectures are listed in eval/data/manifest.json (see manifest.example.json). Every LLM
call is cached, so re-running only pays for configurations that changed."""
import argparse
import csv
import logging
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
from eval import metrics, plots  # noqa: E402
from pipeline import faithfulness, ingest, llm, prompts, quiz, rag, store  # noqa: E402
from pipeline.chunk import fixed_chunks, title_chunks  # noqa: E402
from pipeline.notes import STRATEGIES, generate_notes, notes_to_text  # noqa: E402
from pipeline.profile import NEUTRAL, PRESETS  # noqa: E402
from pipeline.run import prepare_lecture  # noqa: E402
from pipeline.transcribe import parse_transcript_file, transcribe  # noqa: E402
from pipeline.utils import estimate_tokens, parse_ts, read_json, write_json  # noqa: E402

RESULTS = config.RESULTS_DIR
OUTPUTS = RESULTS / "outputs"


class RagGrade(BaseModel):
    correct: bool
    reason: str = ""


# ------------------------------------------------------------------ helpers
def load_manifest() -> list[dict]:
    path = config.EVAL_DATA_DIR / "manifest.json"
    if not path.exists():
        sys.exit(f"Missing {path}. Copy eval/data/manifest.example.json to manifest.json and fill it in.")
    return read_json(path)["lectures"]


def resolve_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else config.ROOT / path


def reference_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".srt", ".vtt"):
        return " ".join(s.text for s in parse_transcript_file(raw, path.name))
    # Drop the [MM:SS] line prefixes: they are not spoken words and would count as insertions.
    return " ".join(re.sub(r"^\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*", "", line) for line in raw.splitlines())


def ensure_lecture(entry: dict) -> str:
    """Ingest a manifest entry (idempotent) and return its lecture id."""
    lang = entry.get("language", "auto")
    if entry.get("transcript_file"):
        path = resolve_path(entry["transcript_file"])
        meta = ingest.ingest_transcript(path.read_bytes(), path.name, entry.get("title"), lang)
    elif str(entry["source"]).startswith("http"):
        meta = ingest.ingest_youtube(entry["source"], lang)
    else:
        path = resolve_path(entry["source"])
        meta = ingest.ingest_file(path.read_bytes(), path.name, entry.get("title"), lang)
    if entry.get("title") and meta.title != entry["title"]:
        meta.title = entry["title"]
        store.save_meta(meta)
    return meta.id


def models_used() -> dict:
    gen, judge = llm.resolve("generator"), llm.resolve("judge")
    return {"generator": f"{gen[0]}/{gen[1]}", "judge": f"{judge[0]}/{judge[1]}"}


def write_csv(name: str, rows: list[dict], stamp: bool = True):
    if stamp:
        rows = [{**r, **models_used()} for r in rows]
    if not rows:
        print(f"  (no rows for {name})")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {RESULTS / name}")


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def human_ratings() -> dict:
    """eval/data/human_ratings.csv: lecture,strategy,rating (one row per rater per item)."""
    path = config.EVAL_DATA_DIR / "human_ratings.csv"
    if not path.exists():
        return {}
    ratings: dict = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings.setdefault((row["lecture"], row["strategy"]), []).append(float(row["rating"]))
    return {k: round(statistics.mean(v), 2) for k, v in ratings.items()}


def notes_run(lecture_id, chunks, profile, strategy, **kwargs):
    meta = store.load_meta(lecture_id)
    with llm.track_usage() as cost:
        notes, logs = generate_notes(chunks, profile, strategy, title=meta.title, language=meta.language, **kwargs)
    return notes, logs, cost


def faith(notes, chunks, language, attempts: int = 3):
    """Faithfulness check; re-run while some claims are unjudged (successful judge calls are cached)."""
    for _ in range(attempts):
        report = faithfulness.check_notes(notes, chunks, language)
        if not report.unjudged:
            break
        print(f"    {report.unjudged} claims unjudged; retrying the judge", flush=True)
    return report


# ------------------------------------------------------------------ experiments
def exp1_transcription(entries):
    print("Experiment 1 - transcription WER")
    rows = []
    for e in entries:
        ref = resolve_path(e.get("reference_transcript"))
        if not ref or e.get("transcript_file"):
            continue
        lid = ensure_lecture(e)
        meta = store.load_meta(lid)
        truth = reference_text(ref)
        for engine in ("api", "local"):
            try:
                segments, _ = transcribe(meta, engine=engine, language=e.get("language"), primary=False)
            except Exception as exc:
                print(f"  {e['id']} {engine}: failed ({exc})")
                continue
            hyp = " ".join(s.text for s in segments)
            timing = read_json(config.LECTURES_DIR / lid / f"transcribe_{engine}.json", {})
            rows.append({"lecture": e["id"], "language": e.get("language", ""), "engine": engine,
                         "reference": e.get("reference_kind", ""),
                         "model": config.WHISPER_MODEL if engine == "api" else f"faster-whisper {config.LOCAL_WHISPER_MODEL}",
                         "wer": metrics.wer(truth, hyp), "seconds": timing.get("seconds"),
                         # code-switched terms written in Latin script count as errors against a Devanagari reference
                         "latin_script_share": round(sum(bool(re.search("[A-Za-z]", w)) for w in hyp.split())
                                                     / max(1, len(hyp.split())), 3),   # None when the transcript predates timing logs
                         "audio_minutes": round(meta.duration / 60, 1)})
    write_csv("exp1_transcription.csv", rows, stamp=False)
    return rows


def exp2_prompting(entries, strategies):
    print("Experiment 2 - prompting strategies")
    rows, ratings = [], human_ratings()
    for e in entries:
        ref = resolve_path(e.get("reference_notes"))
        if not ref:
            continue
        lid = ensure_lecture(e)
        chunks = prepare_lecture(lid, language=e.get("language"))
        lang = store.load_meta(lid).language
        reference = ref.read_text(encoding="utf-8")
        for strategy in strategies:
            print(f"  {e['id']} {strategy}", flush=True)
            try:
                notes, logs, cost = notes_run(lid, chunks, NEUTRAL, strategy)
                report = faith(notes, chunks, lang)
            except llm.LLMError as exc:
                print(f"  stopped (quota/API): {str(exc)[:200]}")
                break
            write_json(OUTPUTS / "exp2" / f"{e['id']}_{strategy}.json", {"notes": notes.model_dump(), "log": logs})
            text = notes_to_text(notes)
            write_json(OUTPUTS / "exp2" / f"{e['id']}_{strategy}_faithfulness.json",
                       {"lecture_id": lid, **report.model_dump()})
            rows.append({"lecture": e["id"], "strategy": strategy, **metrics.rouge(reference, text, lang),
                         "bertscore_f1": metrics.bertscore([reference], [text], lang)[0],
                         "faithfulness": report.score, "claims": report.total, "unjudged": report.unjudged,
                         "human_rating": ratings.get((e["id"], strategy)),
                         "tokens": cost["total_tokens"], "api_calls": cost["calls"], "seconds": cost["seconds"]})
        else:
            continue
        break                                   # quota ran out: keep what is complete
    write_csv("exp2_prompting.csv", rows)
    # Compare strategies only on lectures where every strategy finished.
    complete = [lec for lec in dict.fromkeys(r["lecture"] for r in rows)
                if {r["strategy"] for r in rows if r["lecture"] == lec} >= set(strategies)]
    print(f"  lectures with all strategies: {complete}")
    summary = []
    for strategy in strategies:
        group = [r for r in rows if r["strategy"] == strategy and r["lecture"] in complete]
        if not group:
            continue
        label = STRATEGIES[strategy] + (f" ({config.TOT_BREADTH}x{config.TOT_DEPTH})" if strategy == "tot" else "")
        summary.append({"strategy": strategy, "strategy_label": label, "lectures": len(group),
                        **{k: mean(r[k] for r in group) for k in
                           ("rouge1", "rouge2", "rougeL", "bertscore_f1", "faithfulness", "human_rating", "tokens")}})
    write_csv("exp2_summary.csv", summary)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    plots.strategy_bars(summary, RESULTS / "plots" / "exp2_strategy_quality.png")
    plots.quality_vs_cost(summary, RESULTS / "plots" / "exp2_faithfulness_vs_cost.png", "faithfulness")
    plots.quality_vs_cost(summary, RESULTS / "plots" / "exp2_rougeL_vs_cost.png", "rougeL")
    return summary


def exp3_sampling(entries, strategy="zero_shot", temps=(0.2, 0.5, 0.8, 1.1), top_ps=(0.9, 1.0)):
    print("Experiment 3 - sampling (temperature x top-p)")
    rows = []
    for e in entries:
        lid = ensure_lecture(e)
        chunks = prepare_lecture(lid, language=e.get("language"))
        lang = store.load_meta(lid).language
        for top_p in top_ps:
            for t in temps:
                print(f"  {e['id']} T={t} top_p={top_p}", flush=True)
                try:
                    notes, _, cost = notes_run(lid, chunks, NEUTRAL, strategy, temperature=t, top_p=top_p, reduce=False)
                    report = faith(notes, chunks, lang)
                except llm.LLMError as exc:
                    print(f"  stopped (quota/API): {str(exc)[:200]}")
                    write_csv("exp3_sampling.csv", rows)
                    return rows
                rows.append({"lecture": e["id"], "temperature": t, "top_p": top_p,
                             "distinct2": metrics.distinct_n(notes_to_text(notes), 2),
                             "faithfulness": report.score, "unjudged": report.unjudged,
                             "tokens": cost["total_tokens"]})
    write_csv("exp3_sampling.csv", rows)
    averaged = []
    for top_p in top_ps:
        for t in temps:
            group = [r for r in rows if r["top_p"] == top_p and r["temperature"] == t]
            if group:
                averaged.append({"temperature": t, "top_p": top_p, "distinct2": mean(r["distinct2"] for r in group),
                                 "faithfulness": mean(r["faithfulness"] for r in group)})
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    plots.diversity_vs_faithfulness(averaged, RESULTS / "plots" / "exp3_diversity_vs_faithfulness.png")
    return rows


def gold_times(q: dict, reference_lines: list[str]) -> list[float]:
    """Every place the answer is taught: the labelled ts plus each reference-caption line with a keyword."""
    times = [parse_ts(q["ts"])] if q.get("ts") else []
    for line in reference_lines:
        if any(k.lower() in line.lower() for k in q.get("keywords", [])):
            sec = parse_ts(line[:10])
            if sec is not None:
                times.append(sec)
    return sorted(set(times))


def _rag_accuracy(questions, chunks, meta, answers: bool, reference_lines: list[str] = ()):
    # Embed without LLM titles so the comparison isolates where chunk boundaries fall.
    index = rag.MemoryIndex(rag.model_for(meta.language))
    index.add(meta.id, meta.title, [c.model_copy(update={"title": ""}) for c in chunks])
    correct = cited = valid = retrieved = hit1 = 0
    reciprocal = []
    for q in questions:
        hits = index.query(q["question"], config.RAG_TOP_K)
        gold = gold_times(q, reference_lines)
        if gold:
            ranked = index.query(q["question"], len(chunks))
            rank = next((i + 1 for i, h in enumerate(ranked)
                         if any(h.start - 10 <= sec <= h.end + 10 for sec in gold)), None)
            retrieved += rank is not None and rank <= config.RAG_TOP_K
            hit1 += rank == 1
            reciprocal.append(1 / rank if rank else 0.0)
        if not answers:
            continue
        result = rag.answer_from_hits(q["question"], hits)
        grade = llm.generate_json(prompts.render("rag_judge", question=q["question"], reference=q["reference"],
                                                 answer=result["answer"]),
                                  RagGrade, role="judge", temperature=0.0, task="rag_judge")
        correct += grade.correct
        cited += len(result["citations"])
        valid += sum(c["valid"] for c in result["citations"])
    n = len(questions)
    with_ts = sum(1 for q in questions if gold_times(q, reference_lines))
    return {"rag_accuracy": round(correct / n, 3) if n and answers else None,
            "citation_accuracy": round(valid / cited, 3) if cited else None,
            "hit_at_1": round(hit1 / with_ts, 3) if with_ts else None,
            "mrr": round(sum(reciprocal) / len(reciprocal), 3) if reciprocal else None,
            f"hit_at_{config.RAG_TOP_K}": round(retrieved / with_ts, 3) if with_ts else None, "rag_questions": n}


def exp4_chunking(entries, use_llm: bool = False):
    print("Experiment 4 - fixed vs semantic chunking")
    questions = read_json(config.EVAL_DATA_DIR / "rag_questions.json", {"questions": []})["questions"]
    rows = []
    for e in entries:
        lid = ensure_lecture(e)
        meta = store.load_meta(lid)
        semantic = prepare_lecture(lid, language=e.get("language"))
        fixed = store.load_chunks(lid, "fixed") or fixed_chunks(store.load_transcript(lid))
        if use_llm and not all(c.title for c in fixed):
            fixed = title_chunks(fixed)         # notes prompts use section titles, as semantic chunks have
        store.save_chunks(lid, fixed, "fixed")
        # Size-matched baseline: fixed windows with the same mean size as the semantic chunks,
        # so boundary placement is compared at equal granularity.
        mean_semantic = round(sum(estimate_tokens(c.text) for c in semantic) / len(semantic))
        matched = fixed_chunks(store.load_transcript(lid), mean_semantic)
        ref = resolve_path(e.get("reference_notes"))
        lecture_qs = [q for q in questions if q["lecture"] == e["id"]]
        ref_tx = resolve_path(e.get("reference_transcript"))
        reference_lines = ref_tx.read_text(encoding="utf-8").splitlines() if ref_tx and ref_tx.suffix == ".txt" else []
        for mode, chunks in (("fixed-1000", fixed), ("fixed-matched", matched), ("semantic", semantic)):
            if mode == "fixed-matched" and use_llm:
                continue                      # notes/answers are compared on the plan's two arms only
            print(f"  {e['id']} {mode}: {len(chunks)} chunks")
            row = {"lecture": e["id"], "chunking": mode, "chunks": len(chunks)}
            row["mean_chunk_tokens"] = round(sum(estimate_tokens(c.text) for c in chunks) / len(chunks))
            if ref and use_llm:
                notes, _, _ = notes_run(lid, chunks, NEUTRAL, "zero_shot")
                row["rougeL"] = metrics.rouge(ref.read_text(encoding="utf-8"), notes_to_text(notes), meta.language)["rougeL"]
            if lecture_qs:
                row.update(_rag_accuracy(lecture_qs, chunks, meta, answers=use_llm, reference_lines=reference_lines))
            rows.append(row)
    write_csv("exp4_chunking.csv", rows, stamp=use_llm)
    return rows


def exp5_personalisation(entries, strategy="zero_shot"):
    print("Experiment 5 - personalisation")
    profiles = {"beginner": PRESETS["Beginner (first exposure)"], "exam revision": PRESETS["Exam revision (exam tomorrow)"]}
    rows = []
    for e in entries:
        lid = ensure_lecture(e)
        chunks = prepare_lecture(lid, language=e.get("language"))
        for name, prof in profiles.items():
            print(f"  {e['id']} {name}", flush=True)
            try:
                notes, _, cost = notes_run(lid, chunks, prof, strategy)
            except llm.LLMError as exc:
                print(f"  stopped (quota/API): {str(exc)[:200]}")
                write_csv("exp5_personalisation.csv", rows)
                return rows
            write_json(OUTPUTS / "exp5" / f"{e['id']}_{name.replace(' ', '_')}.json", notes.model_dump())
            text = notes_to_text(notes)
            rows.append({"lecture": e["id"], "profile": name, "fk_grade": metrics.fk_grade(text),
                         "words": len(text.split()), "bullets": sum(len(s.bullets) for s in notes.sections),
                         "definitions": sum(len(s.definitions) for s in notes.sections), "tokens": cost["total_tokens"]})
    write_csv("exp5_personalisation.csv", rows)
    return rows


def exp6_quiz(entries, count=10):
    print("Experiment 6 - quiz quality with vs without CoT")
    rows, rating_rows = [], []
    for e in entries:
        lid = ensure_lecture(e)
        chunks = prepare_lecture(lid, language=e.get("language"))
        for use_cot in (False, True):
            label = "CoT" if use_cot else "no CoT"
            print(f"  {e['id']} {label}", flush=True)
            try:
                with llm.track_usage() as cost:
                    result = quiz.generate_quiz(chunks, NEUTRAL, count=count, use_cot=use_cot)
            except llm.LLMError as exc:
                print(f"  stopped (quota/API): {str(exc)[:200]}")
                break
            s = result.stats
            rows.append({"lecture": e["id"], "variant": label, "generated": s["generated"],
                         "passed_first_try": s["passed_first_try"], "answerability_rate": s["answerability_rate"],
                         "dropped": s["dropped"], "tokens": cost["total_tokens"]})
            for i, q in enumerate(result.questions):
                rating_rows.append({"item": f"{e['id']}-{'c' if use_cot else 'p'}{i}", "stem": q.stem,
                                    "options": " | ".join(q.options), "answer": q.options[q.answer_index],
                                    "clarity_1to5": "", "distractor_quality_1to5": ""})
    write_csv("exp6_quiz.csv", rows)
    write_csv("exp6_questions_for_rating.csv", rating_rows, stamp=False)   # variant hidden in the item code
    return rows


# ------------------------------------------------------------------ tables for the report
def write_tables():
    """Render every result CSV as a Markdown table in results/tables.md."""
    parts = []
    for path in sorted(RESULTS.glob("exp*.csv")):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "for_rating" in path.name:
            continue
        cols = list(rows[0])
        parts.append(f"### {path.stem}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
                     + "\n".join("| " + " | ".join(r[c] or "" for c in cols) + " |" for r in rows))
    if parts:
        (RESULTS / "tables.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        print(f"  wrote {RESULTS / 'tables.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--lectures", nargs="*", help="manifest ids (default: all)")
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    parser.add_argument("--sampling-lectures", type=int, default=2, help="lectures used for Experiment 3")
    parser.add_argument("--generator", help="provider:model, e.g. groq:openai/gpt-oss-120b")
    parser.add_argument("--judge", help="provider:model, e.g. groq:qwen/qwen3.8-27b")
    parser.add_argument("--exp4-llm", action="store_true", help="Experiment 4: also generate notes and RAG answers")
    parser.add_argument("--workers", type=int, help="parallel LLM calls (keep low on 8k-TPM free tiers)")
    args = parser.parse_args()
    if args.generator:
        provider, model = args.generator.split(":", 1)
        config.LLM_PROVIDER = provider
        setattr(config, "GEMINI_MODEL" if provider == "gemini" else "GROQ_MODEL", model)
    if args.judge:
        config.JUDGE_PROVIDER, config.JUDGE_MODEL = args.judge.split(":", 1)
    if args.workers:
        config.MAX_WORKERS = args.workers

    # Never switch models mid-experiment: a quota error stops the run, and the cache resumes it later.
    config.LLM_FALLBACK = False
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    entries = load_manifest()
    if args.lectures:                                # keep the order given on the command line
        by_id = {e["id"]: e for e in entries}
        entries = [by_id[i] for i in args.lectures if i in by_id]
    print(f"Generator: {llm.resolve('generator')} · judge: {llm.resolve('judge')} · {len(entries)} lectures")
    with llm.track_usage() as total:
        if 1 in args.exp:
            exp1_transcription(entries)
        if 2 in args.exp:
            exp2_prompting(entries, args.strategies)
        if 3 in args.exp:
            exp3_sampling(entries[:args.sampling_lectures])
        if 4 in args.exp:
            exp4_chunking(entries, use_llm=args.exp4_llm)
        if 5 in args.exp:
            exp5_personalisation(entries)
        if 6 in args.exp:
            exp6_quiz(entries)
    write_tables()
    print(f"Done. LLM usage this run: {total}")


if __name__ == "__main__":
    main()
