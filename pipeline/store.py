"""On-disk layout of processed lectures.

data/lectures/<lecture_id>/
    meta.json, source.*, audio.wav, pieces/
    transcript_<engine>.json, transcript.json, chunks_<mode>.json
    kits/<kit_key>/notes.json, tot_log.json, quiz.json, ... , cost.json
"""
from pathlib import Path

import config
from pipeline.schemas import Chunk, LectureMeta, Segment
from pipeline.utils import read_json, write_json


def lecture_dir(lecture_id: str) -> Path:
    path = config.LECTURES_DIR / lecture_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def kit_dir(lecture_id: str, kit_key: str) -> Path:
    path = lecture_dir(lecture_id) / "kits" / kit_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_meta(meta: LectureMeta):
    write_json(lecture_dir(meta.id) / "meta.json", meta.model_dump())


def load_meta(lecture_id: str) -> LectureMeta | None:
    data = read_json(config.LECTURES_DIR / lecture_id / "meta.json")
    return LectureMeta.model_validate(data) if data else None


def list_lectures() -> list[LectureMeta]:
    if not config.LECTURES_DIR.exists():
        return []
    metas = [load_meta(p.name) for p in config.LECTURES_DIR.iterdir() if p.is_dir()]
    return sorted([m for m in metas if m], key=lambda m: m.created, reverse=True)


def load_transcript(lecture_id: str, engine: str | None = None) -> list[Segment] | None:
    name = f"transcript_{engine}.json" if engine else "transcript.json"
    data = read_json(config.LECTURES_DIR / lecture_id / name)
    return [Segment.model_validate(s) for s in data] if data else None


def save_transcript(lecture_id: str, segments: list[Segment], engine: str, primary: bool = True):
    rows = [s.model_dump() for s in segments]
    write_json(lecture_dir(lecture_id) / f"transcript_{engine}.json", rows)
    if primary:
        write_json(lecture_dir(lecture_id) / "transcript.json", rows)


def load_chunks(lecture_id: str, mode: str = "semantic") -> list[Chunk] | None:
    data = read_json(config.LECTURES_DIR / lecture_id / f"chunks_{mode}.json")
    return [Chunk.model_validate(c) for c in data] if data else None


def save_chunks(lecture_id: str, chunks: list[Chunk], mode: str = "semantic"):
    write_json(lecture_dir(lecture_id) / f"chunks_{mode}.json", [c.model_dump() for c in chunks])
