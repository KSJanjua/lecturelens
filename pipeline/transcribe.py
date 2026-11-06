"""Speech to text with segment timestamps: Groq-hosted Whisper large-v3,
falling back to faster-whisper (small, int8) on the CPU. Includes transcript cleaning."""
import logging
import re
import time

import config
from pipeline import store
from pipeline.schemas import LectureMeta, Segment
from pipeline.utils import lang_code, parse_ts, write_json

log = logging.getLogger(__name__)

FILLERS = {"um", "uh", "umm", "uhh", "hmm", "mm", "ah", "er", "erm", "okay", "ok", "so", "like",
           "you know", "right", "haan", "accha", "toh"}
NOISE = re.compile(r"^\s*[\[(].*?[\])]\s*$")          # [Music], (applause)


# ------------------------------------------------------------------ engines
def transcribe_api(pieces, language: str | None) -> tuple[list[Segment], str]:
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=config.GROQ_API_KEY)
    segments, detected = [], language or ""
    for path, offset in pieces:
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=(path.name, f.read()), model=config.WHISPER_MODEL,
                response_format="verbose_json", language=language or None, temperature=0.0,
            )
        data = resp.to_dict() if hasattr(resp, "to_dict") else resp.model_dump()
        detected = detected or data.get("language", "")
        for seg in data.get("segments") or []:
            # add each piece's offset back so timestamps match the original lecture
            segments.append(Segment(start=seg["start"] + offset, end=seg["end"] + offset,
                                    text=seg["text"].strip()))
    return segments, detected


_local_model = None


def _wav_samples(path):
    """16-bit PCM WAV -> float32 samples in [-1, 1]."""
    import wave

    import numpy as np
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_local(pieces, language: str | None) -> tuple[list[Segment], str]:
    global _local_model
    from faster_whisper import WhisperModel

    if _local_model is None:
        _local_model = WhisperModel(config.LOCAL_WHISPER_MODEL, device="cpu",
                                    compute_type=config.LOCAL_WHISPER_COMPUTE)
    segments, detected = [], language or ""
    for path, offset in pieces:
        # Pass samples, not a path: our WAV is already 16 kHz mono, and this avoids PyAV
        # (newer PyAV releases break faster-whisper's own decoder).
        result, info = _local_model.transcribe(_wav_samples(path), language=language or None, vad_filter=True)
        detected = detected or info.language
        segments.extend(Segment(start=s.start + offset, end=s.end + offset, text=s.text.strip())
                        for s in result)
    return segments, detected


# ------------------------------------------------------------------ cleaning
def collapse_repeats(text: str, min_repeats: int = 3, max_n: int = 8) -> str:
    """Fix Whisper's repeated-phrase glitch: 'the the the the' / 'thank you. thank you. ...'."""
    words = text.split()
    for n in range(1, max_n + 1):
        out, i = [], 0
        keys = [re.sub(r"\W", "", w.lower()) for w in words]
        while i < len(words):
            gram = keys[i:i + n]
            reps = 1
            while len(gram) == n and keys[i + reps * n:i + (reps + 1) * n] == gram:
                reps += 1
            if reps >= min_repeats:
                out.extend(words[i:i + n])
                i += reps * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def _is_filler(text: str) -> bool:
    if NOISE.match(text):
        return True
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    rest = " ".join(words)
    for filler in sorted(FILLERS, key=len, reverse=True):
        rest = re.sub(rf"\b{filler}\b", " ", rest)
    return not rest.strip()


def clean_segments(segments: list[Segment], min_seconds: float = 1.5, min_words: int = 4) -> list[Segment]:
    cleaned: list[Segment] = []
    for seg in segments:
        text = collapse_repeats(re.sub(r"\s+", " ", seg.text).strip())
        if not text or _is_filler(text):
            continue
        if cleaned and text.lower() == cleaned[-1].text.lower():
            cleaned[-1].end = seg.end            # duplicated segment on silence
            continue
        cleaned.append(Segment(start=seg.start, end=seg.end, text=text))

    merged: list[Segment] = []
    for seg in cleaned:
        short = (seg.end - seg.start) < min_seconds or len(seg.text.split()) < min_words
        if merged and short and seg.start - merged[-1].end < 2.0:
            merged[-1].text += " " + seg.text
            merged[-1].end = seg.end
        else:
            merged.append(seg)
    return merged


# ------------------------------------------------------------------ provided transcripts
_CUE = re.compile(r"(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _cue_seconds(h, m, s, ms) -> float:
    return int((h or "0:")[:-1]) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_transcript_file(text: str, filename: str) -> list[Segment]:
    """Parse .srt/.vtt cues, or .txt lines optionally prefixed with [MM:SS]."""
    segments: list[Segment] = []
    if _CUE.search(text):
        for block in re.split(r"\n\s*\n", text.replace("\r", "")):
            match = _CUE.search(block)
            if not match:
                continue
            g = match.groups()
            lines = block[match.end():].strip().splitlines()
            body = re.sub(r"<[^>]+>", "", " ".join(lines)).strip()
            if body:
                segments.append(Segment(start=_cue_seconds(*g[0:4]), end=_cue_seconds(*g[4:8]), text=body))
        return clean_segments(segments)

    # Plain text: honour leading timestamps, otherwise assume ~150 words per minute.
    clock = 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–:]?\s*(.*)$", line)
        if match:
            clock = parse_ts(match.group(1))
            line = match.group(2)
        duration = max(2.0, len(line.split()) / 2.5)
        if segments and segments[-1].end > clock:
            segments[-1].end = clock
        segments.append(Segment(start=clock, end=clock + duration, text=line))
        clock += duration
    return clean_segments(segments)


# ------------------------------------------------------------------ main entry
def transcribe(meta: LectureMeta, engine: str = "auto", language: str | None = None,
               primary: bool = True) -> tuple[list[Segment], str]:
    """Transcribe a lecture (cached). engine: auto | api | local | provided.
    language: None/'auto' to auto-detect, or 'en' / 'hi' to override."""
    from pipeline.ingest import split_wav

    language = None if language in (None, "", "auto") else lang_code(language)
    if engine == "provided" or meta.source_type == "transcript":
        segments = store.load_transcript(meta.id, "provided") or []
        return segments, meta.language
    for name in (["api", "local"] if engine == "auto" else [engine]):
        cached = store.load_transcript(meta.id, name)
        if cached:
            if primary:
                store.save_transcript(meta.id, cached, name, primary=True)
            return cached, meta.language

    wav = store.lecture_dir(meta.id) / "audio.wav"
    pieces = split_wav(wav)
    order = ["api", "local"] if engine == "auto" else [engine]
    errors = []
    for name in order:
        try:
            fn = transcribe_api if name == "api" else transcribe_local
            started = time.time()
            raw, detected = fn(pieces, language)
            write_json(store.lecture_dir(meta.id) / f"transcribe_{name}.json",
                       {"seconds": round(time.time() - started, 1), "pieces": len(pieces)})
            break
        except Exception as exc:
            log.warning("%s transcription failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    else:
        raise RuntimeError("All transcription engines failed. " + " | ".join(errors))

    segments = clean_segments(raw)
    store.save_transcript(meta.id, segments, name, primary=primary)
    if meta.language in ("", "auto"):
        meta.language = lang_code(detected or language)
        store.save_meta(meta)
    return segments, meta.language
