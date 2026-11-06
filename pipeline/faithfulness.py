"""Faithfulness checker: split generated notes into atomic claims, retrieve the closest
transcript chunk for each, and ask a judge model (different from the generator) whether
the claim is supported, contradicted or not mentioned. Score = supported / total."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config
from pipeline import llm, prompts
from pipeline.chunk import chunk_text_with_timestamps
from pipeline.embeddings import embed
from pipeline.notes import section_to_text
from pipeline.schemas import (Chunk, ClaimResult, ExtractedClaim, ExtractedClaims,
                              FaithfulnessReport, Notes, Verdicts)
from pipeline.utils import fmt_ts

SECTIONS_PER_SPLIT = 4
CLAIMS_PER_JUDGE = 12


def _normalise(verdict: str) -> str:
    v = verdict.lower()
    if "support" in v:
        return "supported"
    if "contra" in v:
        return "contradicted"
    return "not_mentioned"


def extract_claims(notes: Notes) -> list[ExtractedClaim]:
    """Section -1 holds claims from the top summary."""
    blocks = []
    if notes.summary:
        blocks.append((-1, "## Section 0 (summary)\n" + "\n".join(f"- {s}" for s in notes.summary)))
    blocks += [(i, section_to_text(s, i + 1).replace(f"## {i + 1}.", f"## Section {i + 1}:", 1))
               for i, s in enumerate(notes.sections)]
    groups = [blocks[i:i + SECTIONS_PER_SPLIT] for i in range(0, len(blocks), SECTIONS_PER_SPLIT)]

    def split(group):
        valid = {idx for idx, _ in group}
        try:
            result = llm.generate_json(
                prompts.render("claims_split", notes="\n\n".join(text for _, text in group)),
                ExtractedClaims, temperature=0.0, task="claims_split",
            )
            claims = []
            for c in result.claims:
                idx = -1 if c.section == 0 else c.section - 1
                if idx in valid and c.claim.strip():
                    claims.append(ExtractedClaim(section=idx, claim=c.claim.strip()))
            return claims
        except llm.LLMError:     # fall back to bullets as claims
            out = []
            for idx, _ in group:
                texts = notes.summary if idx == -1 else [b.text for b in notes.sections[idx].bullets]
                out += [ExtractedClaim(section=idx, claim=t) for t in texts]
            return out

    with ThreadPoolExecutor(max_workers=max(1, config.MAX_WORKERS)) as pool:
        return [c for claims in pool.map(split, groups) for c in claims]


def _judge(claims: list[ExtractedClaim], evidence: list[Chunk]) -> list[tuple[str, str]]:
    transcript = "\n\n".join(chunk_text_with_timestamps(c) for c in evidence)
    listing = "\n".join(f"{i + 1}. {c.claim}" for i, c in enumerate(claims))
    try:
        result = llm.generate_json(prompts.render("claims_judge", transcript=transcript, claims=listing),
                                   Verdicts, role="judge", temperature=0.0, task="claims_judge")
    except llm.LLMError as exc:
        return [("error", f"judge failed: {str(exc)[:200]}")] * len(claims)
    by_id = {v.id: v for v in result.verdicts}
    return [(_normalise(by_id[i + 1].verdict), by_id[i + 1].reason) if i + 1 in by_id
            else ("error", "no verdict returned") for i in range(len(claims))]


def check_notes(notes: Notes, chunks: list[Chunk], language: str = "en") -> FaithfulnessReport:
    claims = extract_claims(notes)
    if not claims or not chunks:
        return FaithfulnessReport()
    chunk_vecs = embed([c.text for c in chunks], language)
    claim_vecs = embed([c.claim for c in claims], language)
    nearest = np.argmax(claim_vecs @ chunk_vecs.T, axis=1)
    by_id = {c.id: c for c in chunks}

    # Group claims by section; evidence = the section's source chunk + the most common nearest chunk.
    groups: dict[int, list[int]] = {}
    for i, claim in enumerate(claims):
        groups.setdefault(claim.section, []).append(i)
    jobs = []
    for section, idxs in groups.items():
        common = [chunks[j].id for j, _ in Counter(int(nearest[i]) for i in idxs).most_common(3)]
        if section >= 0 and notes.sections[section].chunk_id in by_id:
            source = notes.sections[section].chunk_id
            evidence_ids = [source] + [c for c in common if c != source][:1]
        else:
            evidence_ids = common
        evidence_ids = sorted(set(evidence_ids))
        for start in range(0, len(idxs), CLAIMS_PER_JUDGE):
            jobs.append((idxs[start:start + CLAIMS_PER_JUDGE], evidence_ids))

    def run(job):
        idxs, evidence_ids = job
        return idxs, evidence_ids, _judge([claims[i] for i in idxs], [by_id[e] for e in evidence_ids])

    results: list[ClaimResult | None] = [None] * len(claims)
    with ThreadPoolExecutor(max_workers=max(1, config.MAX_WORKERS)) as pool:
        for idxs, evidence_ids, verdicts in pool.map(run, jobs):
            for i, (verdict, reason) in zip(idxs, verdicts):
                section = claims[i].section
                ts = notes.sections[section].ts if section >= 0 else fmt_ts(chunks[int(nearest[i])].start)
                results[i] = ClaimResult(claim=claims[i].claim, section=section, verdict=verdict,
                                         reason=reason, evidence_chunks=evidence_ids, ts=ts)

    counts = Counter(r.verdict for r in results)
    judged = len(results) - counts["error"]          # judge failures are excluded, not counted as unsupported
    return FaithfulnessReport(
        score=round(counts["supported"] / judged, 3) if judged else 0.0, total=judged,
        supported=counts["supported"], contradicted=counts["contradicted"],
        not_mentioned=counts["not_mentioned"], unjudged=counts["error"], claims=results,
    )
