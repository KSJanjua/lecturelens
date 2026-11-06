"""2-3 minute spoken recap (or a teacher/student "podcast"), synthesised with edge-tts.
Hindi voices are used when the script is in Devanagari."""
import asyncio
import re
from pathlib import Path

import config
from pipeline import llm, prompts
from pipeline.notes import notes_to_text
from pipeline.schemas import LearnerProfile, Notes, RecapScript
from pipeline.utils import DEVANAGARI


def write_script(notes: Notes, profile: LearnerProfile, podcast: bool = False) -> RecapScript:
    template = "recap_podcast" if podcast else "recap_script"
    script = llm.generate_json(
        prompts.render(template, title=notes.title or "this lecture", profile=profile.describe(),
                       notes=notes_to_text(notes), language=profile.language),
        RecapScript, system=prompts.system(), temperature=config.CREATIVE_TEMPERATURE, task=template,
    )
    for line in script.lines:   # TTS reads symbols aloud; strip markdown leftovers
        line.text = re.sub(r"[*_#`>\[\]]", "", line.text).strip()
    script.lines = [l for l in script.lines if l.text]
    return script


def voices_for(script: RecapScript) -> tuple[str, str]:
    text = " ".join(l.text for l in script.lines)
    return config.TTS_VOICES["hi" if DEVANAGARI.search(text) else "en"]


async def _synthesise(script: RecapScript, out_path: Path):
    import edge_tts

    teacher, student = voices_for(script)
    audio = bytearray()
    for line in script.lines:
        voice = student if line.speaker.lower().startswith("student") else teacher
        async for chunk in edge_tts.Communicate(line.text, voice).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    out_path.write_bytes(bytes(audio))    # edge-tts emits MP3 frames; concatenation stays valid


def synthesise(script: RecapScript, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(_synthesise(script, out_path))
    except RuntimeError as exc:
        if "running event loop" not in str(exc):
            raise
        loop = asyncio.new_event_loop()       # called from inside a running loop
        try:
            loop.run_until_complete(_synthesise(script, out_path))
        finally:
            loop.close()
    return out_path


def word_count(script: RecapScript) -> int:
    return sum(len(l.text.split()) for l in script.lines)
