"""Small helpers shared by every pipeline stage."""
import hashlib
import json
import re
from pathlib import Path

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def sha(*parts, length=16) -> str:
    """Stable short hash of strings, bytes or JSON-able objects."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            h.update(part)
        elif isinstance(part, str):
            h.update(part.encode("utf-8"))
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def fmt_ts(seconds) -> str:
    """78.4 -> '01:18'; 3725 -> '1:02:05'."""
    seconds = max(0, int(float(seconds or 0)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_ts(value) -> float | None:
    """'12:40' / '1:02:05' / '[12:40]' / 760 -> seconds. None if unparseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(value))
    if not match:
        try:
            return float(str(value).strip())
        except ValueError:
            return None
    a, b, c = match.groups()
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 words per token for English)."""
    if not text:
        return 0
    words = len(text.split())
    return int(words * (2.0 if DEVANAGARI.search(text) else 1.33)) + 1


def lang_code(language: str | None) -> str:
    """Normalise Whisper / UI language names to 'en' or 'hi'."""
    if not language:
        return "en"
    language = language.lower()
    return "hi" if language.startswith(("hi", "hindi", "hinglish")) else "en"


def extract_json(text: str):
    """Parse JSON from an LLM reply, tolerating code fences and leading prose."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object found in model output")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1])
