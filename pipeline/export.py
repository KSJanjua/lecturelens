"""Downloads: notes and quiz as Markdown or PDF."""
import re
from pathlib import Path

from pipeline.schemas import Notes, Quiz
from pipeline.utils import DEVANAGARI, parse_ts


def _ts_link(ts: str, youtube_url: str | None) -> str:
    if not ts:
        return ""
    if youtube_url:
        return f"[[{ts}]]({youtube_url}&t={int(parse_ts(ts) or 0)}s)"
    return f"[{ts}]"


def notes_to_markdown(notes: Notes, youtube_url: str | None = None) -> str:
    lines = [f"# {notes.title or 'Lecture notes'}", ""]
    if notes.summary:
        lines += ["## Summary", ""] + [f"- {s}" for s in notes.summary] + [""]
    for section in notes.sections:
        lines += [f"## {section.title} {_ts_link(section.ts, youtube_url)}", ""]
        lines += [f"- {b.text} {_ts_link(b.ts, youtube_url)}" for b in section.bullets]
        if section.definitions:
            lines += ["", "**Definitions**", ""]
            lines += [f"- **{d.term}**: {d.definition} {_ts_link(d.ts, youtube_url)}" for d in section.definitions]
        if section.formulas:
            lines += ["", "**Formulas**", ""]
            lines += [f"- `{f.expression}`: {f.meaning} {_ts_link(f.ts, youtube_url)}" for f in section.formulas]
        if section.examples:
            lines += ["", "**Examples**", ""] + [f"- {e}" for e in section.examples]
        lines.append("")
    if notes.glossary:
        lines += ["## Key terms", ""] + [f"- **{g.term}**: {g.definition}" for g in notes.glossary]
    return "\n".join(lines).strip() + "\n"


def quiz_to_markdown(quiz: Quiz, title: str = "Quiz", with_answers: bool = True) -> str:
    lines = [f"# {title}", ""]
    for i, q in enumerate(quiz.questions, 1):
        lines += [f"**{i}. {q.stem}**  _({q.bloom}, {q.difficulty})_", ""]
        lines += [f"- {'ABCD'[j]}) {o}" for j, o in enumerate(q.options)]
        lines.append("")
    if with_answers:
        lines += ["---", "", "## Answers", ""]
        for i, q in enumerate(quiz.questions, 1):
            lines.append(f"{i}. **{'ABCD'[q.answer_index]}** - {q.explanation} [{q.ts}]")
    return "\n".join(lines).strip() + "\n"


# ------------------------------------------------------------------ PDF
_FONTS = {
    "devanagari": ["C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/NirmalaUI.ttf",
                   "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                   "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"],
    "latin": ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/Library/Fonts/Arial Unicode.ttf"],
}


def _font_for(text: str) -> str | None:
    order = (["devanagari"] if DEVANAGARI.search(text) else []) + ["latin"]
    for group in order:
        for path in _FONTS[group]:
            if Path(path).exists():
                return path
    return None


def markdown_to_pdf(markdown: str) -> bytes:
    """Lightweight Markdown -> PDF (headings, bullets, paragraphs)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font_path = _font_for(markdown)
    if font_path:
        pdf.add_font("Body", "", font_path)
        family = "Body"
        if DEVANAGARI.search(markdown):
            try:
                pdf.set_text_shaping(True)       # needs uharfbuzz; plain glyphs otherwise
            except Exception:
                pass
    else:
        family = "Helvetica"

    def clean(text: str) -> str:
        text = re.sub(r"\[\[(.*?)\]\]\(.*?\)", r"[\1]", text)     # timestamp links -> [MM:SS]
        text = re.sub(r"[*_`]", "", text)
        return text if font_path else text.encode("latin-1", "replace").decode("latin-1")

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line or line == "---":
            pdf.ln(3)
            continue
        level = len(line) - len(line.lstrip("#"))
        if 0 < level <= 3 and line[level:level + 1] == " ":
            pdf.set_font(family, size={1: 18, 2: 14, 3: 12}[level])
            pdf.ln(2)
            pdf.multi_cell(0, 8, clean(line[level + 1:]), new_x="LMARGIN", new_y="NEXT")
            continue
        pdf.set_font(family, size=11)
        if line.lstrip().startswith("- "):
            line = "  \u2022 " + line.lstrip()[2:] if font_path else "  - " + line.lstrip()[2:]
        pdf.multi_cell(0, 6, clean(line), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
