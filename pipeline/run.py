"""Orchestrator: lecture -> transcript -> chunks -> index (once per lecture), then the
study kit for a (learner profile, options) pair. Every stage is cached on disk and its
API cost (calls, tokens, seconds) is logged to cost.json."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime

import config
from pipeline import audio_recap, diagrams, faithfulness, flashcards, llm, notes, quiz, rag, store
from pipeline.chunk import fixed_chunks, semantic_chunks, title_chunks
from pipeline.schemas import (Chunk, Diagram, FaithfulnessReport, FlashcardSet, LearnerProfile,
                              Notes, Quiz, RecapScript)
from pipeline.transcribe import transcribe
from pipeline.utils import read_json, sha, write_json

OUTPUTS = ("notes", "quiz", "flashcards", "diagrams", "audio", "faithfulness")


@dataclass
class KitOptions:
    strategy: str = field(default_factory=lambda: config.DEFAULT_STRATEGY)
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    chunking: str = "semantic"         # semantic | fixed
    outputs: tuple = OUTPUTS
    podcast: bool = False
    quiz_cot: bool = True


def _noop(label, fraction):
    pass


def _pair(model):
    """A stage result plus the JSON to cache for it."""
    return model, model.model_dump()


@contextmanager
def generator_override(provider: str | None, model: str | None):
    """Temporarily switch the generator model (the UI's model picker)."""
    saved = (config.LLM_PROVIDER, config.GEMINI_MODEL, config.GROQ_MODEL)
    if provider:
        config.LLM_PROVIDER = provider
    if model:
        if config.LLM_PROVIDER == "gemini":
            config.GEMINI_MODEL = model
        elif config.LLM_PROVIDER == "groq":
            config.GROQ_MODEL = model
    try:
        yield
    finally:
        config.LLM_PROVIDER, config.GEMINI_MODEL, config.GROQ_MODEL = saved


# ------------------------------------------------------------------ per-lecture stages
def prepare_lecture(lecture_id: str, language: str | None = None, engine: str = "auto",
                    chunking: str = "semantic", progress=_noop) -> list[Chunk]:
    """Transcribe, chunk, title and index a lecture. Cached: runs once per lecture."""
    meta = store.load_meta(lecture_id)
    if meta is None:
        raise ValueError(f"unknown lecture {lecture_id}")
    cached = store.load_chunks(lecture_id, chunking)
    if cached:
        return cached

    progress("Transcribing", 0.0)
    segments, lang = transcribe(meta, engine=engine, language=language)
    if not segments:
        raise RuntimeError("transcription produced no text")
    meta = store.load_meta(lecture_id)

    progress("Chunking by topic", 0.0)
    if chunking == "fixed":
        chunks, info = fixed_chunks(segments), {}
    else:
        chunks, info = semantic_chunks(segments, lang)
    progress("Titling sections", 0.0)
    with llm.track_usage() as cost:
        chunks = title_chunks(chunks)
    store.save_chunks(lecture_id, chunks, chunking)
    write_json(store.lecture_dir(lecture_id) / f"chunking_{chunking}.json",
               {**info, "chunks": len(chunks), "cost": cost})

    if chunking == "semantic":
        progress("Indexing for Q&A", 0.0)
        rag.index_lecture(meta, chunks)
    return chunks


# ------------------------------------------------------------------ study kit
def kit_key(profile: LearnerProfile, opts: KitOptions) -> str:
    provider, model = (opts.provider or config.LLM_PROVIDER), opts.model or ""
    try:
        judge = llm.resolve("judge")
    except llm.LLMError:
        judge = None
    return sha(profile.model_dump(), opts.strategy, provider, model, opts.temperature, opts.chunking,
               opts.podcast, opts.quiz_cot, config.TOT_BREADTH, config.TOT_DEPTH, judge, length=12)


def _stage(folder, name, cost_log, fn, loader):
    """Run fn() unless its output is cached; record its usage under cost_log[name]."""
    path = folder / f"{name}.json"
    data = read_json(path)
    if data is not None:
        return loader(data)
    with llm.track_usage() as cost:
        result, raw = fn()
    write_json(path, raw)
    cost_log[name] = cost
    return result


def build_kit(lecture_id: str, profile: LearnerProfile, opts: KitOptions | None = None,
              progress=_noop) -> dict:
    opts = opts or KitOptions()
    meta = store.load_meta(lecture_id)
    chunks = prepare_lecture(lecture_id, chunking=opts.chunking, progress=progress)
    key = kit_key(profile, opts)
    folder = store.kit_dir(lecture_id, key)
    cost_log = read_json(folder / "cost.json", {})
    write_json(folder / "kit.json", {"key": key, "lecture_id": lecture_id, "profile": profile.model_dump(),
                                     "options": asdict(opts), "created": datetime.now().isoformat(timespec="seconds")})
    kit = {"key": key, "meta": meta, "profile": profile, "options": opts, "chunks": chunks}
    wanted = set(opts.outputs) | {"notes"}
    steps = [s for s in OUTPUTS if s in wanted]

    def step(i, label):
        progress(label, i / len(steps))

    with generator_override(opts.provider, opts.model):
        step(0, f"Writing notes ({notes.STRATEGIES[opts.strategy]})")

        def make_notes():
            n, logs = notes.generate_notes(chunks, profile, opts.strategy, title=meta.title,
                                           language=meta.language, temperature=opts.temperature,
                                           progress=lambda f: progress("Writing notes", f / len(steps)))
            write_json(folder / "notes_log.json", logs)
            return _pair(n)

        kit["notes"] = _stage(folder, "notes", cost_log, make_notes, Notes.model_validate)
        kit["notes_log"] = read_json(folder / "notes_log.json", {})
        done = 1

        if "quiz" in wanted:
            step(done, "Building quiz (CoT + self-check)")
            kit["quiz"] = _stage(folder, "quiz", cost_log, lambda: _pair(
                quiz.generate_quiz(chunks, profile, use_cot=opts.quiz_cot)),
                Quiz.model_validate)
            done += 1
        if "flashcards" in wanted:
            step(done, "Making flashcards")
            kit["flashcards"] = _stage(folder, "flashcards", cost_log, lambda: _pair(
                flashcards.generate_flashcards(kit["notes"], profile)),
                FlashcardSet.model_validate)
            done += 1
        if "diagrams" in wanted:
            step(done, "Drawing diagrams")

            def make_diagrams():
                items, stats = diagrams.generate_diagrams(kit["notes"], profile)
                return (items, stats), {"diagrams": [d.model_dump() for d in items], "stats": stats}

            kit["diagrams"], kit["diagram_stats"] = _stage(
                folder, "diagrams", cost_log, make_diagrams,
                lambda d: ([Diagram.model_validate(x) for x in d["diagrams"]], d["stats"]))
            done += 1
        if "audio" in wanted:
            step(done, "Recording audio recap")

            def make_audio():
                script = audio_recap.write_script(kit["notes"], profile, podcast=opts.podcast)
                audio_recap.synthesise(script, folder / "recap.mp3")
                return _pair(script)

            kit["recap"] = _stage(folder, "recap", cost_log, make_audio, RecapScript.model_validate)
            mp3 = folder / "recap.mp3"
            kit["recap_audio"] = mp3 if mp3.exists() else None
            done += 1
        if "faithfulness" in wanted:
            step(done, "Checking faithfulness")
            kit["faithfulness"] = _stage(folder, "faithfulness", cost_log, lambda: _pair(
                faithfulness.check_notes(kit["notes"], chunks, meta.language)),
                FaithfulnessReport.model_validate)

    write_json(folder / "cost.json", cost_log)
    kit["cost"] = cost_log
    progress("Done", 1.0)
    return kit


def load_kit(lecture_id: str, key: str) -> dict | None:
    """Load a previously built kit from disk (no API calls) - used for the live demo."""
    folder = config.LECTURES_DIR / lecture_id / "kits" / key
    manifest = read_json(folder / "kit.json")
    if not manifest or not (folder / "notes.json").exists():
        return None
    profile = LearnerProfile.model_validate(manifest["profile"])
    opts = KitOptions(**{**manifest["options"], "outputs": tuple(manifest["options"]["outputs"])})
    kit = {"key": key, "meta": store.load_meta(lecture_id), "profile": profile, "options": opts,
           "chunks": store.load_chunks(lecture_id, opts.chunking) or [],
           "notes": Notes.model_validate(read_json(folder / "notes.json")),
           "notes_log": read_json(folder / "notes_log.json", {}), "cost": read_json(folder / "cost.json", {})}
    if (data := read_json(folder / "quiz.json")) is not None:
        kit["quiz"] = Quiz.model_validate(data)
    if (data := read_json(folder / "flashcards.json")) is not None:
        kit["flashcards"] = FlashcardSet.model_validate(data)
    if (data := read_json(folder / "diagrams.json")) is not None:
        kit["diagrams"] = [Diagram.model_validate(x) for x in data["diagrams"]]
        kit["diagram_stats"] = data["stats"]
    if (data := read_json(folder / "recap.json")) is not None:
        kit["recap"] = RecapScript.model_validate(data)
        mp3 = folder / "recap.mp3"
        kit["recap_audio"] = mp3 if mp3.exists() else None
    if (data := read_json(folder / "faithfulness.json")) is not None:
        kit["faithfulness"] = FaithfulnessReport.model_validate(data)
    return kit


def list_kits(lecture_id: str) -> list[dict]:
    folder = config.LECTURES_DIR / lecture_id / "kits"
    if not folder.exists():
        return []
    manifests = [read_json(p / "kit.json") for p in folder.iterdir() if (p / "notes.json").exists()]
    return sorted([m for m in manifests if m], key=lambda m: m["created"], reverse=True)
