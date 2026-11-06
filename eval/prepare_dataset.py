"""Build the evaluation set: reference transcripts (YouTube captions) and reference notes
(the lecturer's own written lesson pages), then write eval/data/manifest.json.

    python eval/prepare_dataset.py

Human-made captions are used where they exist. No human Hindi transcript was available for
the Hindi lecture, so its reference is YouTube's automatic captions (a "silver" reference):
WER for that lecture measures agreement with another ASR system, not true accuracy."""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

RAW = config.EVAL_DATA_DIR / "raw"          # downloaded third-party content lives here only
REFS = config.EVAL_DATA_DIR / "refs"

DATASET = [
    {"id": "L1", "video": "aircAruvnKk", "language": "en", "title": "But what is a neural network?",
     "notes_url": "https://www.3blue1brown.com/lessons/neural-networks/"},
    {"id": "L2", "video": "IHZwWFHWa-w", "language": "en", "title": "Gradient descent, how neural networks learn",
     "notes_url": "https://www.3blue1brown.com/lessons/gradient-descent/"},
    {"id": "L3", "video": "Ilg3gGewQ5U", "language": "en", "title": "Backpropagation, intuitively",
     "notes_url": "https://www.3blue1brown.com/lessons/backpropagation/"},
    {"id": "L4", "video": "O5nskjZ_GoI", "language": "en", "title": "Early Computing (Crash Course CS #1)"},
    {"id": "L5", "video": "sQK3Yr4Sc_k", "language": "en", "title": "Photosynthesis (Crash Course Biology #8)"},
    {"id": "L6", "video": "m7R8aJg5esw", "language": "hi", "title": "Probability Theory: Review (NPTEL, Hindi)",
     "silver_transcript": True},
]

NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪+|>>")


def fetch_captions(video: str, language: str, automatic: bool) -> Path | None:
    """Download one caption track as json3 (no rolling duplicates, unlike auto-caption VTT)."""
    import yt_dlp

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{video}.{language}.json3"
    if out.exists():
        return out
    opts = {"skip_download": True, "quiet": True, "no_warnings": True, "subtitleslangs": [language],
            "subtitlesformat": "json3", "outtmpl": str(RAW / "%(id)s"), "js_runtimes": {"node": {}},
            "writesubtitles": not automatic, "writeautomaticsub": automatic}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video}"])
    found = sorted(RAW.glob(f"{video}.{language}*.json3"))
    return found[0] if found else None


def json3_to_text(path: Path) -> str:
    """Caption events -> timestamped lines '[MM:SS] text', non-speech tags removed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for event in data.get("events", []):
        text = "".join(seg.get("utf8", "") for seg in event.get("segs") or [])
        text = re.sub(r"\s+", " ", NON_SPEECH.sub(" ", text)).strip()
        if text:
            start = event.get("tStartMs", 0) / 1000
            lines.append(f"[{int(start // 60):02d}:{int(start % 60):02d}] {text}")
    return "\n".join(lines)


class _LessonText(HTMLParser):
    """Collect headings, paragraphs and list items from a lesson page."""

    def __init__(self):
        super().__init__()
        self.blocks, self._current = [], None

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "p", "li"):
            self._current = [tag, ""]

    def handle_endtag(self, tag):
        if self._current and tag == self._current[0]:
            text = re.sub(r"\s+", " ", self._current[1]).strip()
            if text:
                self.blocks.append((tag, text))
            self._current = None

    def handle_data(self, data):
        if self._current:
            self._current[1] += data


def fetch_notes(url: str, out: Path):
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / (url.rstrip("/").rsplit("/", 1)[-1] + ".html")
    if not raw.exists():
        raw.write_text(requests.get(url, timeout=30).text, encoding="utf-8")
    parser = _LessonText()
    parser.feed(raw.read_text(encoding="utf-8"))
    lines = []
    for tag, text in parser.blocks:
        if tag in ("h2", "h3") and text.lower().startswith("thanks"):
            break                      # supporter credits follow
        prefix = {"h1": "# ", "h2": "## ", "h3": "### ", "li": "- "}.get(tag, "")
        lines.append(prefix + text)
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main():
    REFS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in DATASET:
        entry = {"id": item["id"], "title": item["title"], "language": item["language"],
                 "source": f"https://www.youtube.com/watch?v={item['video']}"}
        silver = item.get("silver_transcript", False)
        captions = fetch_captions(item["video"], item["language"], automatic=silver)
        if captions:
            name = f"{item['id']}_transcript{'_silver' if silver else ''}.txt"
            (REFS / name).write_text(json3_to_text(captions), encoding="utf-8")
            entry["reference_transcript"] = f"eval/data/refs/{name}"
            entry["reference_kind"] = "youtube-auto (silver)" if silver else "human captions"
        if item.get("notes_url"):
            fetch_notes(item["notes_url"], REFS / f"{item['id']}_notes.md")
            entry["reference_notes"] = f"eval/data/refs/{item['id']}_notes.md"
        manifest.append(entry)
        words = len((REFS / Path(entry.get("reference_transcript", "x")).name).read_text(encoding="utf-8").split()) \
            if "reference_transcript" in entry else 0
        print(f"{item['id']}: transcript={entry.get('reference_kind', 'none')} ({words} words), "
              f"notes={'yes' if 'reference_notes' in entry else 'no'}")
    (config.EVAL_DATA_DIR / "manifest.json").write_text(
        json.dumps({"lectures": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {config.EVAL_DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
