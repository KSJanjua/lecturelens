"""Pydantic schemas: every LLM output is validated against one of these."""
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, field_validator

from pipeline.utils import fmt_ts


def _coerce_ts(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return fmt_ts(value)
    return str(value).strip().strip("[]")


def _coerce_str(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


TS = Annotated[str, BeforeValidator(_coerce_ts)]
Text = Annotated[str, BeforeValidator(_coerce_str)]


# ------------------------------------------------------------------ lecture data
class LectureMeta(BaseModel):
    id: str
    title: str
    source_type: str                 # upload | youtube | transcript
    source: str                      # original filename or URL
    media_path: str = ""             # playable file, or the YouTube URL
    media_kind: str = "audio"        # audio | video | youtube | none
    duration: float = 0.0
    language: str = "auto"           # auto | en | hi
    created: str = ""


class Segment(BaseModel):
    start: float
    end: float
    text: str


class Chunk(BaseModel):
    id: int
    start: float
    end: float
    title: str = ""
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


# ------------------------------------------------------------------ learner profile
class LearnerProfile(BaseModel):
    level: Text = "revising"          # first exposure | revising | exam tomorrow
    goal: Text = "understand"         # understand | exam prep | interview prep
    depth: Text = "standard"          # brief | standard | detailed
    language: Text = "English"        # English | Hinglish | Hindi
    weak_areas: list[str] = Field(default_factory=list)
    background: Text = ""
    quiz_size: int = 8

    def describe(self) -> str:
        weak = ", ".join(self.weak_areas) or "none stated"
        lines = [
            f"- Level: {self.level}",
            f"- Goal: {self.goal}",
            f"- Preferred depth: {self.depth}",
            f"- Output language: {self.language}",
            f"- Weak areas: {weak}",
        ]
        if self.background:
            lines.append(f"- Background: {self.background}")
        return "\n".join(lines)


class InterviewTurn(BaseModel):
    done: bool = False
    question: Text = ""
    profile: LearnerProfile | None = None


# ------------------------------------------------------------------ notes
class Bullet(BaseModel):
    text: Text
    ts: TS = ""


class Definition(BaseModel):
    term: Text
    definition: Text
    ts: TS = ""


class Formula(BaseModel):
    expression: Text
    meaning: Text = ""
    ts: TS = ""


class Section(BaseModel):
    title: Text
    ts: TS = ""
    bullets: list[Bullet] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    examples: list[Text] = Field(default_factory=list)
    chunk_id: int | None = None

    @field_validator("bullets", mode="before")
    @classmethod
    def _bullets_from_strings(cls, value):
        return [{"text": v} if isinstance(v, str) else v for v in value or []]


class CotSection(BaseModel):
    reasoning: Text = ""
    section: Section


class GlossaryItem(BaseModel):
    term: Text
    definition: Text


class ReduceOutput(BaseModel):
    summary: list[Text]
    glossary: list[GlossaryItem] = Field(default_factory=list)


class Notes(BaseModel):
    title: str = ""
    strategy: str = ""
    summary: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    glossary: list[GlossaryItem] = Field(default_factory=list)


class ChunkTitles(BaseModel):
    titles: list[Text]


# ------------------------------------------------------------------ tree of thought
class CandidateScore(BaseModel):
    id: Text
    coverage: float
    faithfulness: float
    clarity: float
    profile_fit: float
    reason: Text = ""
    feedback: Text = ""

    @property
    def total(self) -> float:
        return round((self.coverage + self.faithfulness + self.clarity + self.profile_fit) / 4, 2)


class JudgeScores(BaseModel):
    scores: list[CandidateScore]


# ------------------------------------------------------------------ quiz
class MCQ(BaseModel):
    concept: Text
    bloom: Text = "understand"        # remember | understand | apply
    difficulty: Text = "medium"       # easy | medium | hard
    reasoning: Text = ""              # chain-of-thought used to build the question
    stem: Text
    options: list[Text]
    answer_index: int
    explanation: Text = ""
    ts: TS = ""
    verified: bool | None = None

    @field_validator("options")
    @classmethod
    def _four_options(cls, value):
        if len(value) != 4:
            raise ValueError(f"expected 4 options, got {len(value)}")
        return value

    @field_validator("answer_index")
    @classmethod
    def _index_in_range(cls, value):
        if not 0 <= value <= 3:
            raise ValueError("answer_index must be 0-3")
        return value


class Quiz(BaseModel):
    questions: list[MCQ] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


class SolverAnswer(BaseModel):
    id: int
    answer_index: int


class SolverAnswers(BaseModel):
    answers: list[SolverAnswer]


# ------------------------------------------------------------------ flashcards
class Flashcard(BaseModel):
    kind: Text = "qa"                 # qa | cloze
    front: Text
    back: Text
    ts: TS = ""


class FlashcardSet(BaseModel):
    cards: list[Flashcard] = Field(default_factory=list)


# ------------------------------------------------------------------ diagrams
class DiagramDraft(BaseModel):
    title: Text
    kind: Text = "flowchart"          # flowchart | mindmap | sequence
    concept: Text = ""
    ts: TS = ""
    mermaid: Text


class DiagramPlan(BaseModel):
    diagrams: list[DiagramDraft] = Field(default_factory=list)


class MermaidCode(BaseModel):
    mermaid: Text


class Diagram(DiagramDraft):
    valid: bool = False
    first_try_valid: bool = False
    attempts: int = 1
    error: str = ""


# ------------------------------------------------------------------ audio recap
class RecapLine(BaseModel):
    speaker: Text = "teacher"         # teacher | student
    text: Text


class RecapScript(BaseModel):
    lines: list[RecapLine]


# ------------------------------------------------------------------ faithfulness
class ExtractedClaim(BaseModel):
    section: int
    claim: Text


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class Verdict(BaseModel):
    id: int
    verdict: Text                     # supported | contradicted | not_mentioned
    reason: Text = ""


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class ClaimResult(BaseModel):
    claim: str
    section: int                      # -1 = top summary
    verdict: str
    reason: str = ""
    evidence_chunks: list[int] = Field(default_factory=list)
    ts: str = ""


class FaithfulnessReport(BaseModel):
    score: float = 0.0
    total: int = 0
    supported: int = 0
    contradicted: int = 0
    not_mentioned: int = 0
    unjudged: int = 0                 # claims the judge could not assess (API errors); not in the score
    claims: list[ClaimResult] = Field(default_factory=list)
