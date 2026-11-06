"""Concept diagrams as generated Mermaid code (code generation used for image output).

Each diagram is validated by rendering (mermaid.ink) after a local lint; on an error the
message is sent back to the LLM for a fix, up to DIAGRAM_MAX_RETRIES times. The first-try
valid rate is recorded as a metric."""
import base64
import re

import config
from pipeline import llm, prompts
from pipeline.notes import notes_to_text
from pipeline.schemas import Diagram, DiagramDraft, DiagramPlan, LearnerProfile, MermaidCode, Notes

HEADERS = ("flowchart", "graph", "sequenceDiagram", "mindmap", "classDiagram", "stateDiagram",
           "erDiagram", "timeline")
_PAIRS = {"[": "]", "(": ")", "{": "}"}


def clean_code(code: str) -> str:
    code = (code or "").replace("\\n", "\n").strip()
    code = re.sub(r"^```(?:mermaid)?\s*|```$", "", code, flags=re.M).strip()
    if code.lower().startswith("mermaid\n"):
        code = code.split("\n", 1)[1]
    return "\n".join(line.rstrip() for line in code.splitlines() if line.strip())


def _balanced(line: str) -> bool:
    stack, in_quote = [], False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        elif in_quote:
            continue
        elif ch in _PAIRS:
            stack.append(_PAIRS[ch])
        elif ch in _PAIRS.values():
            if not stack or stack.pop() != ch:
                return False
    return not stack and not in_quote


def lint(code: str) -> str | None:
    """Catch the most common LLM Mermaid mistakes without a network call."""
    lines = code.splitlines()
    if not lines:
        return "empty diagram"
    header = lines[0].strip()
    if not header.startswith(HEADERS):
        return f"first line must declare a diagram type (e.g. 'flowchart TD'), got '{header[:40]}'"
    body = lines[1:]
    if not body:
        return "diagram has no content"
    if header.startswith(("flowchart", "graph")):
        for n, line in enumerate(body, 2):
            if not _balanced(line):
                return f"line {n}: unbalanced brackets or quotes: {line.strip()}"
            if re.search(r"\[[^\]\"]*[()][^\]\"]*\]", line):
                return f"line {n}: parentheses inside an unquoted label; wrap the label in double quotes: {line.strip()}"
    if header.startswith("mindmap"):
        indents = [len(l) - len(l.lstrip()) for l in body]
        if min(indents) == 0:
            return "mindmap nodes must be indented under 'mindmap'"
        if indents.count(indents[0]) > 1 and min(indents) == indents[0]:
            return "mindmap has more than one root node; indent everything under a single root"
        if min(indents) < indents[0]:
            return "a node is indented less than the root; mindmaps need exactly one root"
    return None


def ink_url(code: str, fmt: str = "svg") -> str:
    encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    return f"{config.MERMAID_INK_URL}/{fmt}/{encoded}"


def render_check(code: str) -> tuple[bool | None, str]:
    """Render with mermaid.ink. Returns (None, reason) when the service can't be reached."""
    import requests
    try:
        resp = requests.get(ink_url(code), timeout=20)
    except requests.RequestException as exc:
        return None, f"render service unreachable: {exc}"
    if resp.status_code == 200 and b"<svg" in resp.content[:4000] and b"Syntax error" not in resp.content:
        return True, ""
    return False, resp.text[:400] or f"HTTP {resp.status_code}"


def validate(code: str) -> tuple[bool, str]:
    error = lint(code)
    if error:
        return False, error
    if config.MERMAID_VALIDATE_REMOTE:
        ok, error = render_check(code)
        if ok is not None:
            return ok, error
    return True, ""


def _repair(draft: DiagramDraft) -> Diagram:
    code = clean_code(draft.mermaid)
    ok, error = validate(code)
    diagram = Diagram(**{**draft.model_dump(), "mermaid": code}, valid=ok, first_try_valid=ok, error=error)
    while not diagram.valid and diagram.attempts <= config.DIAGRAM_MAX_RETRIES:
        try:
            fixed = llm.generate_json(
                prompts.render("diagram_fix", kind=diagram.kind, code=diagram.mermaid, error=diagram.error),
                MermaidCode, temperature=0.0, task="diagram_fix",
            )
        except llm.LLMError as exc:
            diagram.error = str(exc)
            break
        diagram.attempts += 1
        diagram.mermaid = clean_code(fixed.mermaid)
        diagram.valid, diagram.error = validate(diagram.mermaid)
    return diagram


def _label(text: str, limit: int = 40) -> str:
    text = re.sub(r"[()\[\]{}\"':,;<>#|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit] or "item"


def outline_mindmap(notes: Notes) -> str:
    """Deterministic fallback: lecture -> sections -> key terms (always valid)."""
    lines = ["mindmap", f"  root(({_label(notes.title or 'Lecture', 30)}))"]
    for section in notes.sections:
        lines.append(f"    {_label(section.title)}")
        leaves = [d.term for d in section.definitions] or [b.text for b in section.bullets]
        lines += [f"      {_label(leaf, 32)}" for leaf in leaves[:3]]
    return "\n".join(lines)


def lecture_mindmap(notes: Notes) -> Diagram:
    outline = "\n".join(
        f"- {s.title}: " + "; ".join([d.term for d in s.definitions] + [b.text[:60] for b in s.bullets[:3]])
        for s in notes.sections)
    try:
        draft = llm.generate_json(prompts.render("mindmap", title=notes.title or "Lecture", outline=outline),
                                  MermaidCode, temperature=0.2, task="mindmap")
        diagram = _repair(DiagramDraft(title="Whole-lecture mind map", kind="mindmap", mermaid=draft.mermaid))
    except llm.LLMError as exc:
        diagram = Diagram(title="Whole-lecture mind map", kind="mindmap", mermaid="", error=str(exc))
    if not diagram.valid:
        diagram.mermaid, diagram.valid = outline_mindmap(notes), True
        diagram.error = "LLM mind map failed validation; used the outline-based fallback"
    return diagram


def generate_diagrams(notes: Notes, profile: LearnerProfile | None = None) -> tuple[list[Diagram], dict]:
    plan = llm.generate_json(
        prompts.render("diagrams_plan", notes=notes_to_text(notes, with_summary=False),
                       max_diagrams=config.DIAGRAM_MAX),
        DiagramPlan, temperature=0.2, task="diagrams_plan",
    )
    diagrams = [_repair(d) for d in plan.diagrams[:config.DIAGRAM_MAX]]
    stats = {
        "planned": len(diagrams),
        "first_try_valid": sum(d.first_try_valid for d in diagrams),
        "valid_after_retries": sum(d.valid for d in diagrams),
    }
    stats["first_try_rate"] = round(stats["first_try_valid"] / len(diagrams), 3) if diagrams else None
    kept = [d for d in diagrams if d.valid]           # skip diagrams that never became valid
    kept.insert(0, lecture_mindmap(notes))
    return kept, stats
