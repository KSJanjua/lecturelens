"""Ask-the-Lecture: Retrieval-Augmented Generation over chunk embeddings.

Chunks are indexed in ChromaDB (one collection per embedding model, so English and
multilingual lectures can sit side by side); a NumPy index is the fallback and is also
used for in-memory experiments. Answers come only from the top-k chunks and cite [MM:SS]."""
import logging
import re
from dataclasses import asdict, dataclass

import numpy as np

import config
from pipeline import llm, prompts, store
from pipeline.chunk import chunk_text_with_timestamps
from pipeline.embeddings import embed, model_for
from pipeline.schemas import Chunk, LearnerProfile, LectureMeta
from pipeline.utils import fmt_ts, parse_ts, read_json, sha, write_json

log = logging.getLogger(__name__)
BRACKETS = re.compile(r"\[([^\]]*\d{1,2}:\d{2}[^\]]*)\]")
TIMESTAMP = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")


def find_citations(text: str) -> list[str]:
    """All timestamps cited in square brackets: [12:40], [12:40-13:10], [Lecture 2, 12:40]."""
    found = [ts for group in BRACKETS.findall(text) for ts in TIMESTAMP.findall(group)]
    return list(dict.fromkeys(found))


@dataclass
class Hit:
    lecture_id: str
    lecture_title: str
    chunk_id: int
    title: str
    start: float
    end: float
    text: str            # chunk text with [MM:SS] markers
    score: float


def _embed_text(chunk: Chunk) -> str:
    return f"{chunk.title}. {chunk.text}" if chunk.title else chunk.text


# ------------------------------------------------------------------ in-memory index
class MemoryIndex:
    """Plain NumPy cosine search. Used for experiments and as the persistent fallback."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vectors = np.zeros((0, 384), dtype=np.float32)
        self.items: list[Hit] = []

    def add(self, lecture_id: str, lecture_title: str, chunks: list[Chunk], vectors=None):
        if vectors is None:
            vectors = embed([_embed_text(c) for c in chunks], model_name=self.model_name)
        self.vectors = vectors if not len(self.items) else np.vstack([self.vectors, vectors])
        self.items += [Hit(lecture_id, lecture_title, c.id, c.title, c.start, c.end,
                           chunk_text_with_timestamps(c), 0.0) for c in chunks]

    def query(self, question: str, k: int, lecture_ids=None) -> list[Hit]:
        if not self.items:
            return []
        q = embed([question], model_name=self.model_name)[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)
        hits = []
        for i in order:
            item = self.items[i]
            if lecture_ids and item.lecture_id not in lecture_ids:
                continue
            hits.append(Hit(**{**asdict(item), "score": float(scores[i])}))
            if len(hits) == k:
                break
        return hits


# ------------------------------------------------------------------ persistent index
_chroma = None


def _chroma_client():
    global _chroma
    if _chroma is None:
        try:
            import chromadb
            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        except Exception as exc:          # not installed / no wheel for this Python
            log.warning("ChromaDB unavailable (%s); using the NumPy index", exc)
            _chroma = False
    return _chroma or None


def _collection(model_name: str):
    client = _chroma_client()
    if client is None:
        return None
    return client.get_or_create_collection(name=f"chunks-{sha(model_name, length=10)}",
                                           metadata={"hnsw:space": "cosine"})


def index_lecture(meta: LectureMeta, chunks: list[Chunk]):
    """(Re)index one lecture's chunks with their timestamps."""
    model_name = model_for(meta.language)
    vectors = embed([_embed_text(c) for c in chunks], model_name=model_name)
    folder = store.lecture_dir(meta.id)
    np.save(folder / "index_vectors.npy", vectors)
    write_json(folder / "index_meta.json", {"model": model_name, "chunks": len(chunks)})

    collection = _collection(model_name)
    if collection is None:
        return
    collection.delete(where={"lecture_id": meta.id})
    collection.add(
        ids=[f"{meta.id}-{c.id}" for c in chunks],
        embeddings=vectors.tolist(),
        documents=[chunk_text_with_timestamps(c) for c in chunks],
        metadatas=[{"lecture_id": meta.id, "lecture_title": meta.title, "chunk_id": c.id,
                    "title": c.title, "start": c.start, "end": c.end} for c in chunks],
    )


def _query_files(question: str, lecture_ids: list[str], model_name: str, k: int) -> list[Hit]:
    index = MemoryIndex(model_name)
    for lid in lecture_ids:
        meta, chunks = store.load_meta(lid), store.load_chunks(lid)
        path = config.LECTURES_DIR / lid / "index_vectors.npy"
        if meta and chunks and path.exists():
            index.add(lid, meta.title, chunks, np.load(path))
    return index.query(question, k)


def retrieve(question: str, lecture_ids: list[str], k: int | None = None) -> list[Hit]:
    """Top-k chunks across one or more lectures (grouped by embedding model, merged by score)."""
    k = k or config.RAG_TOP_K
    groups: dict[str, list[str]] = {}
    for lid in lecture_ids:
        info = read_json(config.LECTURES_DIR / lid / "index_meta.json")
        if info:
            groups.setdefault(info["model"], []).append(lid)
    hits: list[Hit] = []
    for model_name, ids in groups.items():
        collection = _collection(model_name)
        if collection is None or collection.count() == 0:
            hits += _query_files(question, ids, model_name, k)
            continue
        q = embed([question], model_name=model_name)[0].tolist()
        where = {"lecture_id": ids[0]} if len(ids) == 1 else {"lecture_id": {"$in": ids}}
        res = collection.query(query_embeddings=[q], n_results=k, where=where)
        for doc, md, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(md["lecture_id"], md.get("lecture_title", ""), int(md["chunk_id"]),
                            md.get("title", ""), float(md["start"]), float(md["end"]), doc, 1 - float(dist)))
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]


# ------------------------------------------------------------------ answering
def answer_from_hits(question: str, hits: list[Hit], profile: LearnerProfile | None = None,
                     history: list[dict] | None = None) -> dict:
    profile = profile or LearnerProfile()
    multi = len({h.lecture_id for h in hits}) > 1
    context = "\n\n".join(
        f"[Excerpt {i + 1} | " + (f"{h.lecture_title} | " if multi else "")
        + f"{fmt_ts(h.start)}-{fmt_ts(h.end)} | {h.title}]\n{h.text}"
        for i, h in enumerate(hits)) or "(no excerpts found)"
    past = ""
    if history:
        recent = history[-4:]
        past = "Previous conversation:\n" + "\n".join(
            f"{'Student' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}" for m in recent) + "\n\n"
    text = llm.generate(
        prompts.render("rag_answer", profile=profile.describe(), context=context, history=past,
                       question=question, language=profile.language),
        system=prompts.system(), temperature=0.1, task="rag_answer",
    ).strip()

    found = not text.upper().startswith("NOT_FOUND")
    if not found:
        text = text.split(":", 1)[1].strip() if ":" in text else text
    citations = []
    for ts in find_citations(text):
        sec = parse_ts(ts)
        source = next((h for h in hits if h.start - 5 <= sec <= h.end + 5), None)
        citations.append({"ts": ts, "seconds": sec, "valid": source is not None,
                          "lecture_id": source.lecture_id if source else None})
    return {"answer": text, "found": found, "citations": citations, "hits": [asdict(h) for h in hits]}


def ask(question: str, lecture_ids: list[str], profile: LearnerProfile | None = None,
        history: list[dict] | None = None, k: int | None = None) -> dict:
    query = question
    if history:
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if len(question.split()) < 8 and last_user:
            query = f"{last_user} {question}"     # resolve follow-ups like "why is that?"
    return answer_from_hits(question, retrieve(query, lecture_ids, k), profile, history)
