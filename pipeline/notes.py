"""Notes generation with map-reduce, under four prompting strategies:
zero-shot, few-shot, Chain-of-Thought and a hand-built Tree-of-Thought search.

Map:    each chunk -> one Section (JSON validated by Pydantic), tagged with timestamps.
Reduce: sections in order, near-duplicate bullets removed, then one call writes the
        5-line summary and the key-terms glossary."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
from pipeline import llm, prompts
from pipeline.chunk import chunk_text_with_timestamps
from pipeline.embeddings import embed
from pipeline.schemas import (Chunk, CotSection, JudgeScores, LearnerProfile, Notes,
                              ReduceOutput, Section)
from pipeline.utils import fmt_ts, parse_ts

STRATEGIES = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "Chain-of-Thought",
    "tot": "Tree-of-Thought",
}

TOT_STYLES = {
    "concept-first": "Start from the core concepts and definitions, then build up to how they connect; "
                     "examples come last as support.",
    "example-first": "Lead with the lecturer's concrete examples and worked cases, and draw out the general "
                     "idea from each one.",
    "question-driven": "Organise the notes around the questions a student would ask (what, why, how, when), "
                       "answering each from the transcript.",
}


# ------------------------------------------------------------------ helpers
def section_to_text(section: Section, index: int | None = None) -> str:
    head = f"## {index}. " if index is not None else "## "
    lines = [f"{head}{section.title} [{section.ts}]"]
    lines += [f"- {b.text} [{b.ts}]" for b in section.bullets]
    lines += [f"- Definition - {d.term}: {d.definition} [{d.ts}]" for d in section.definitions]
    lines += [f"- Formula - {f.expression}: {f.meaning} [{f.ts}]" for f in section.formulas]
    lines += [f"- Example: {e}" for e in section.examples]
    return "\n".join(lines)


def notes_to_text(notes: Notes, with_summary: bool = True) -> str:
    parts = []
    if with_summary and notes.summary:
        parts.append("Summary:\n" + "\n".join(f"- {s}" for s in notes.summary))
    parts += [section_to_text(s, i + 1) for i, s in enumerate(notes.sections)]
    return "\n\n".join(parts)


def _fix_timestamps(section: Section, chunk: Chunk) -> Section:
    """Keep every timestamp inside the chunk; replace invented ones with the nearest valid time."""
    def clamp(ts, fallback):
        sec = parse_ts(ts)
        if sec is None or sec < chunk.start - 5 or sec > chunk.end + 5:
            return fallback
        return fmt_ts(sec)

    section.ts = clamp(section.ts, fmt_ts(chunk.start))
    last = section.ts
    for item in [*section.bullets, *section.definitions, *section.formulas]:
        item.ts = clamp(item.ts, last)
        last = item.ts
    section.chunk_id = chunk.id
    if not section.title.strip():
        section.title = chunk.title or f"Part {chunk.id + 1}"
    return section


def _base_vars(chunk: Chunk, profile: LearnerProfile) -> dict:
    return dict(profile=profile.describe(), title=chunk.title or "(none)",
                transcript=chunk_text_with_timestamps(chunk), language=profile.language)


# ------------------------------------------------------------------ single-path strategies
def _simple_section(chunk, profile, strategy, temperature, top_p):
    template = {"zero_shot": "notes_zero_shot", "few_shot": "notes_few_shot"}[strategy]
    return llm.generate_json(prompts.render(template, **_base_vars(chunk, profile)), Section,
                             system=prompts.system(), temperature=temperature, top_p=top_p,
                             task=template)


def _cot_section(chunk, profile, temperature, top_p):
    result = llm.generate_json(prompts.render("notes_cot", **_base_vars(chunk, profile)), CotSection,
                               system=prompts.system(), temperature=temperature, top_p=top_p,
                               task="notes_cot")
    return result.section, {"chunk_id": chunk.id, "reasoning": result.reasoning}


# ------------------------------------------------------------------ Tree-of-Thought
def _judge(chunk, profile, candidates: list[Section]) -> list:
    labels = [chr(ord("A") + i) for i in range(len(candidates))]
    rendered = "\n\n".join(f"### Candidate {lab}\n{section_to_text(c)}" for lab, c in zip(labels, candidates))
    result = llm.generate_json(
        prompts.render("notes_tot_judge", profile=profile.describe(),
                       transcript=chunk_text_with_timestamps(chunk), candidates=rendered),
        JudgeScores, role="judge", temperature=config.JUDGE_TEMPERATURE, task="notes_tot_judge",
    )
    by_id = {s.id.strip().upper().replace("CANDIDATE ", ""): s for s in result.scores}
    scores = []
    for i, lab in enumerate(labels):
        score = by_id.get(lab) or (result.scores[i] if i < len(result.scores) else None)
        scores.append(score)
    return scores


def _score_dict(score) -> dict:
    if score is None:
        return {"total": 0.0}
    return {**score.model_dump(exclude={"id"}), "total": score.total}


def tot_section(chunk: Chunk, profile: LearnerProfile, breadth: int | None = None,
                keep: int | None = None, depth: int | None = None, top_p: float | None = None):
    """Generate -> evaluate -> prune -> expand (refine with judge feedback) -> select.
    Returns the best Section and the full search tree for the report."""
    breadth = breadth or config.TOT_BREADTH
    keep = keep or config.TOT_KEEP
    depth = depth or config.TOT_DEPTH
    styles = [list(TOT_STYLES)[i % len(TOT_STYLES)] for i in range(breadth)]
    base = _base_vars(chunk, profile)
    tree = {"chunk_id": chunk.id, "title": chunk.title, "breadth": breadth, "keep": keep,
            "depth": depth, "levels": [], "calls": 0}

    # Level 0: diverse candidates at high temperature.
    nodes = []
    for i, style in enumerate(styles):
        variant = "" if i < len(TOT_STYLES) else f" (variant {i // len(TOT_STYLES) + 1})"
        section = llm.generate_json(
            prompts.render("notes_tot_generate", strategy_name=style + variant,
                           strategy_instructions=TOT_STYLES[style], **base),
            Section, system=prompts.system(), temperature=config.TOT_TEMPERATURE, top_p=top_p,
            task="notes_tot_generate",
        )
        nodes.append({"id": f"0{chr(65 + i)}", "style": style, "parent": None,
                      "section": _fix_timestamps(section, chunk)})
    tree["calls"] += len(nodes)

    for level in range(depth):
        scores = _judge(chunk, profile, [n["section"] for n in nodes])
        tree["calls"] += 1
        for node, score in zip(nodes, scores):
            node["score"] = _score_dict(score)
            node["feedback"] = score.feedback if score else ""
        ranked = sorted(nodes, key=lambda n: (n["score"]["total"], n["score"].get("faithfulness", 0)),
                        reverse=True)
        survivors = ranked[:keep] if level < depth - 1 else ranked[:1]
        for node in nodes:
            node["kept"] = any(node is s for s in survivors)
        tree["levels"].append({"depth": level, "nodes": [
            {**{k: v for k, v in n.items() if k != "section"}, "section": n["section"].model_dump()}
            for n in nodes]})
        if level == depth - 1:
            break
        # Expand: refine each survivor with the judge's feedback.
        refined = []
        for j, node in enumerate(survivors):
            score = node["score"]
            section = llm.generate_json(
                prompts.render("notes_tot_refine", notes=section_to_text(node["section"]),
                               scores=json.dumps({k: score.get(k) for k in
                                                  ("coverage", "faithfulness", "clarity", "profile_fit")}),
                               feedback=node.get("feedback") or "none", **base),
                Section, system=prompts.system(), temperature=config.NOTES_TEMPERATURE, top_p=top_p,
                task="notes_tot_refine",
            )
            refined.append({"id": f"{level + 1}{chr(65 + j)}", "style": node["style"],
                            "parent": node["id"], "section": _fix_timestamps(section, chunk)})
        tree["calls"] += len(refined)
        nodes = refined

    best = next(n for n in nodes if n.get("kept"))
    tree["selected"] = best["id"]
    return best["section"], tree


# ------------------------------------------------------------------ map-reduce
def section_for_chunk(chunk: Chunk, profile: LearnerProfile, strategy: str,
                      temperature: float | None = None, top_p: float | None = None):
    temperature = config.NOTES_TEMPERATURE if temperature is None else temperature
    if strategy == "tot":
        section, log = tot_section(chunk, profile, top_p=top_p)
    elif strategy == "cot":
        section, log = _cot_section(chunk, profile, temperature, top_p)
    else:
        section, log = _simple_section(chunk, profile, strategy, temperature, top_p), None
    return _fix_timestamps(section, chunk), log


def dedupe_bullets(sections: list[Section], language: str, threshold: float | None = None) -> int:
    """Remove bullets that repeat an earlier bullet almost word for word (cosine >= threshold)."""
    threshold = threshold or config.BULLET_DEDUP_SIMILARITY
    refs = [(i, j) for i, s in enumerate(sections) for j in range(len(s.bullets))]
    if len(refs) < 2:
        return 0
    vectors = embed([sections[i].bullets[j].text for i, j in refs], language)
    keep, kept_vecs = set(), []
    for k, ref in enumerate(refs):
        if kept_vecs and float(np.max(np.stack(kept_vecs) @ vectors[k])) >= threshold:
            continue
        keep.add(ref)
        kept_vecs.append(vectors[k])
    for i, section in enumerate(sections):
        section.bullets = [b for j, b in enumerate(section.bullets) if (i, j) in keep]
    return len(refs) - len(keep)


def generate_notes(chunks: list[Chunk], profile: LearnerProfile, strategy: str = "tot",
                   title: str = "", language: str = "en", temperature: float | None = None,
                   top_p: float | None = None, reduce: bool = True, progress=None):
    """Returns (Notes, logs). logs holds the ToT trees / CoT reasoning per chunk."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}")
    results: list = [None] * len(chunks)
    done = 0

    def work(i):
        return i, section_for_chunk(chunks[i], profile, strategy, temperature, top_p)

    with ThreadPoolExecutor(max_workers=max(1, config.MAX_WORKERS)) as pool:
        for i, result in pool.map(work, range(len(chunks))):
            results[i] = result
            done += 1
            if progress:
                progress(done / max(1, len(chunks)))

    sections = [r[0] for r in results]
    logs = [r[1] for r in results if r[1] is not None]
    removed = dedupe_bullets(sections, language)
    notes = Notes(title=title, strategy=strategy, sections=sections)
    if reduce and sections:
        reduced = llm.generate_json(
            prompts.render("notes_reduce", title=title or "Lecture", profile=profile.describe(),
                           notes=notes_to_text(notes, with_summary=False), language=profile.language),
            ReduceOutput, system=prompts.system(), temperature=config.NOTES_TEMPERATURE, task="notes_reduce",
        )
        notes.summary = [s for s in reduced.summary if s.strip()][:5]
        notes.glossary = sorted(reduced.glossary, key=lambda g: g.term.lower())
    return notes, {"strategy": strategy, "duplicates_removed": removed, "chunks": logs}
