"""Ingestion: uploaded video/audio, YouTube URL or a ready-made transcript ->
16 kHz mono WAV (the format Whisper expects), split into ~10-minute pieces when needed."""
import re
import shutil
import subprocess
import wave
from datetime import datetime
from pathlib import Path

import config
from pipeline import store
from pipeline.schemas import LectureMeta
from pipeline.utils import sha

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
TRANSCRIPT_EXT = {".srt", ".vtt", ".txt"}


def ffmpeg_exe() -> str:
    """System ffmpeg, else the static binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from exc


def _run(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


def to_wav(src: Path, dst: Path) -> Path:
    _run([ffmpeg_exe(), "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE),
          "-c:a", "pcm_s16le", str(dst)])
    return dst


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def split_wav(wav: Path, piece_seconds: int | None = None, max_bytes: int | None = None):
    """Return [(piece_path, start_offset_seconds)]. One piece if the file fits the API limit."""
    piece_seconds = piece_seconds or config.PIECE_SECONDS
    max_bytes = max_bytes or config.API_MAX_BYTES
    if wav.stat().st_size <= max_bytes:
        return [(wav, 0.0)]
    out_dir = wav.parent / "pieces"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    _run([ffmpeg_exe(), "-y", "-i", str(wav), "-f", "segment", "-segment_time", str(piece_seconds),
          "-c", "copy", str(out_dir / "p%03d.wav")])
    pieces, offset = [], 0.0
    for path in sorted(out_dir.glob("p*.wav")):
        pieces.append((path, offset))
        offset += wav_duration(path)
    return pieces


def _new_meta(lecture_id, title, source_type, source, language, media_path="", media_kind="audio"):
    return LectureMeta(
        id=lecture_id, title=title, source_type=source_type, source=source,
        media_path=str(media_path), media_kind=media_kind, language=language,
        created=datetime.now().isoformat(timespec="seconds"),
    )


def ingest_file(data: bytes, filename: str, title: str | None = None, language: str = "auto") -> LectureMeta:
    """Store an uploaded video/audio file and convert it to WAV. Re-uploads hit the cache."""
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXT | AUDIO_EXT:
        raise ValueError(f"unsupported file type {ext}")
    lecture_id = "f" + sha(data, length=12)
    existing = store.load_meta(lecture_id)
    if existing and (store.lecture_dir(lecture_id) / "audio.wav").exists():
        return existing

    folder = store.lecture_dir(lecture_id)
    source = folder / f"source{ext}"
    source.write_bytes(data)
    wav = to_wav(source, folder / "audio.wav")
    meta = _new_meta(lecture_id, title or Path(filename).stem, "upload", filename, language,
                     media_path=source, media_kind="video" if ext in VIDEO_EXT else "audio")
    meta.duration = wav_duration(wav)
    store.save_meta(meta)
    return meta


def youtube_id(url: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    return match.group(1) if match else None


def ingest_youtube(url: str, language: str = "auto") -> LectureMeta:
    """Download audio only with yt-dlp, then convert to WAV."""
    import yt_dlp

    video_id = youtube_id(url) or sha(url, length=11)
    lecture_id = "y" + sha(video_id, length=12)
    existing = store.load_meta(lecture_id)
    if existing and (store.lecture_dir(lecture_id) / "audio.wav").exists():
        return existing

    folder = store.lecture_dir(lecture_id)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(folder / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": ffmpeg_exe(),
    }
    if shutil.which("node") and not shutil.which("deno"):
        opts["js_runtimes"] = {"node": {}}     # YouTube extraction needs a JS runtime
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    if not downloaded.exists():
        candidates = [p for p in folder.glob("source.*") if p.suffix != ".wav"]
        downloaded = candidates[0]
    wav = to_wav(downloaded, folder / "audio.wav")
    canonical = f"https://www.youtube.com/watch?v={video_id}" if youtube_id(url) else url
    if language == "auto" and info.get("language"):
        language = "hi" if str(info["language"]).startswith("hi") else language
    meta = _new_meta(lecture_id, info.get("title") or video_id, "youtube", canonical, language,
                     media_path=canonical, media_kind="youtube")
    meta.duration = wav_duration(wav)
    store.save_meta(meta)
    return meta


def ingest_transcript(data: bytes, filename: str, title: str | None = None,
                      language: str = "auto", media_url: str = "") -> LectureMeta:
    """Use an existing transcript (.srt/.vtt/.txt) instead of speech recognition."""
    from pipeline.transcribe import parse_transcript_file

    lecture_id = "t" + sha(data, length=12)
    segments = parse_transcript_file(data.decode("utf-8", errors="replace"), filename)
    if not segments:
        raise ValueError("no transcript segments found in file")
    meta = store.load_meta(lecture_id) or _new_meta(
        lecture_id, title or Path(filename).stem, "transcript", filename, language,
        media_path=media_url, media_kind="youtube" if media_url else "none",
    )
    meta.duration = segments[-1].end
    store.save_meta(meta)
    store.save_transcript(lecture_id, segments, engine="provided")
    return meta
