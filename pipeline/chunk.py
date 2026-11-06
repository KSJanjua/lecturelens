"""Topic-aware (semantic) chunking, plus a fixed-size baseline for Experiment 4.

Windows of ~5 transcript segments are embedded; a new chunk starts where the cosine
similarity between neighbouring windows drops below a threshold, subject to min/max
token limits that come from the LLM context window and retrieval quality."""
import numpy as np

import config
from pipeline import llm, prompts
from pipeline.embeddings import embed
from pipeline.schemas import Chunk, ChunkTitles, LearnerProfile, Segment
from pipeline.utils import estimate_tokens, fmt_ts


def _tokens(segments: list[Segment]) -> int:
    return sum(estimate_tokens(s.text) for s in segments)


def _make_chunks(groups: list[list[Segment]]) -> list[Chunk]:
    return [Chunk(id=i, start=g[0].start, end=g[-1].end, segments=g) for i, g in enumerate(groups) if g]


def _split_oversized(group: list[Segment], max_tokens: int) -> list[list[Segment]]:
    out, cur = [], []
    for seg in group:
        if cur and _tokens(cur) + estimate_tokens(seg.text) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(seg)
    return out + ([cur] if cur else [])


def boundary_threshold(similarities: np.ndarray, threshold: float) -> float:
    """Absolute threshold, made adaptive when it would split almost everywhere
    (similarity scales vary by embedding model and speaking style)."""
    if len(similarities) and (similarities < threshold).mean() > 0.6:
        return float(np.percentile(similarities, 25))
    return threshold


def semantic_chunks(segments: list[Segment], language: str = "en", threshold: float | None = None,
                    window: int | None = None, min_tokens: int | None = None,
                    max_tokens: int | None = None) -> tuple[list[Chunk], dict]:
    threshold = config.SIM_THRESHOLD if threshold is None else threshold
    window = window or config.WINDOW_SENTENCES
    min_tokens = min_tokens or config.MIN_CHUNK_TOKENS
    max_tokens = max_tokens or config.MAX_CHUNK_TOKENS
    if not segments:
        return [], {}

    windows = [segments[i:i + window] for i in range(0, len(segments), window)]
    if len(windows) == 1:
        return _make_chunks(_split_oversized(segments, max_tokens)), {"similarities": []}

    vectors = embed([" ".join(s.text for s in w) for w in windows], language)
    sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
    cut = boundary_threshold(sims, threshold)

    groups, current = [], list(windows[0])
    for i in range(1, len(windows)):
        size = _tokens(current)
        topic_shift = sims[i - 1] < cut
        if (topic_shift and size >= min_tokens) or size + _tokens(windows[i]) > max_tokens:
            groups.append(current)
            current = []
        current.extend(windows[i])
    groups.append(current)

    if len(groups) > 1 and _tokens(groups[-1]) < min_tokens:   # tiny tail joins its neighbour
        groups[-2].extend(groups.pop())
    groups = [part for g in groups for part in _split_oversized(g, max_tokens)]
    return _make_chunks(groups), {"similarities": sims.round(3).tolist(), "threshold_used": cut}


def fixed_chunks(segments: list[Segment], size_tokens: int | None = None) -> list[Chunk]:
    return _make_chunks(_split_oversized(segments, size_tokens or config.FIXED_CHUNK_TOKENS))


def chunk_text_with_timestamps(chunk: Chunk, every_seconds: float = 30.0) -> str:
    """Render a chunk with a [MM:SS] marker at least every 30 s, so the LLM can tag points."""
    parts, last = [], -every_seconds
    for seg in chunk.segments:
        if seg.start - last >= every_seconds:
            parts.append(f"\n[{fmt_ts(seg.start)}] ")
            last = seg.start
        parts.append(seg.text + " ")
    return "".join(parts).strip()


def title_chunks(chunks: list[Chunk], profile: LearnerProfile | None = None) -> list[Chunk]:
    """Ask the LLM for a short title per chunk (one batched call); these become section headings."""
    if not chunks:
        return chunks
    excerpts = []
    for c in chunks:
        words = c.text.split()
        snippet = " ".join(words[:120]) + (" ... " + " ".join(words[-40:]) if len(words) > 160 else "")
        excerpts.append(f"Chunk {c.id + 1} [{fmt_ts(c.start)}-{fmt_ts(c.end)}]: {snippet}")
    language = profile.language if profile else "English"
    try:
        result = llm.generate_json(
            prompts.render("chunk_titles", count=len(chunks), excerpts="\n\n".join(excerpts),
                           language=language),
            ChunkTitles, system=prompts.system(), temperature=0.2, task="chunk_titles",
        )
        titles = result.titles
    except llm.LLMError:
        titles = []
    for i, chunk in enumerate(chunks):
        chunk.title = (titles[i] if i < len(titles) else "").strip() or f"Part {i + 1}"
    return chunks
