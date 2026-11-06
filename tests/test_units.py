from eval import metrics
from pipeline import diagrams, llm
from pipeline.quiz import _allocate, bloom_mix, shuffle_options
from pipeline.rag import find_citations
from pipeline.schemas import MCQ, LearnerProfile, Section, Segment
from pipeline.transcribe import clean_segments, collapse_repeats, parse_transcript_file
from pipeline.utils import extract_json, fmt_ts, parse_ts


def test_timestamps_round_trip():
    assert fmt_ts(78.4) == "01:18"
    assert fmt_ts(3725) == "1:02:05"
    assert parse_ts("[12:40]") == 760
    assert parse_ts("1:02:05") == 3725
    assert parse_ts("nonsense") is None


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_validate_wraps_bare_list():
    from pipeline.schemas import ChunkTitles
    assert llm._validate('["a", "b"]', ChunkTitles).titles == ["a", "b"]


def test_section_coerces_numeric_ts_and_string_bullets():
    s = Section.model_validate({"title": "t", "ts": 75, "bullets": ["plain bullet"]})
    assert s.ts == "01:15" and s.bullets[0].text == "plain bullet"


def test_collapse_repeats_fixes_whisper_loop():
    assert collapse_repeats("thank you thank you thank you thank you so much") == "thank you so much"
    assert collapse_repeats("the the the cat") == "the cat"
    assert collapse_repeats("a normal sentence stays") == "a normal sentence stays"


def test_clean_segments_drops_fillers_and_merges_short():
    segs = [Segment(start=0, end=5, text="Welcome to the lecture on graphs."),
            Segment(start=5, end=6, text="Um, uh."),
            Segment(start=6, end=6.5, text="Okay."),
            Segment(start=6.5, end=7, text="right now"),
            Segment(start=7, end=12, text="A graph has vertices and edges between them.")]
    out = clean_segments(segs)
    assert [s.text for s in out] == ["Welcome to the lecture on graphs. right now",
                                     "A graph has vertices and edges between them."]


def test_parse_srt_and_timestamped_txt():
    srt = "1\n00:00:01,000 --> 00:00:04,000\nHello class, today is sorting.\n\n2\n00:01:00,500 --> 00:01:04,000\nMerge sort splits the array."
    segs = parse_transcript_file(srt, "x.srt")
    assert segs[1].start == 60.5 and "Merge sort" in segs[1].text
    txt = "[00:10] First we define a tree data structure.\n[02:00] Then we traverse the tree in order."
    segs = parse_transcript_file(txt, "x.txt")
    assert segs[0].start == 10 and segs[1].start == 120


def test_mermaid_lint():
    assert diagrams.lint('flowchart TD\nA["ok"] --> B["fine"]') is None
    assert "parentheses" in diagrams.lint("flowchart TD\nA[bad (label)] --> B")
    assert "diagram type" in diagrams.lint("A --> B")
    assert diagrams.lint("mindmap\n  root((X))\n    a\n    b") is None
    assert "root" in diagrams.lint("mindmap\n  root((X))\n  second root")


def test_quiz_helpers():
    assert sum(_allocate([100, 300, 600], 7)) == 7
    assert bloom_mix(LearnerProfile(level="exam tomorrow"), 10) == "2 remember, 4 understand, 4 apply"
    q = MCQ(concept="c", stem="Which is right?", options=["right", "a", "b", "c"], answer_index=0)
    q = shuffle_options(q)
    assert q.options[q.answer_index] == "right"
    keep = MCQ(concept="c", stem="s", options=["x", "y", "z", "All of the above"], answer_index=3)
    assert shuffle_options(keep).options[3] == "All of the above"


def test_find_citations():
    assert find_citations("A [12:40] and B [01:02:03] and [Lecture 2, 05:00-06:00]") == \
        ["12:40", "01:02:03", "05:00", "06:00"]


def test_metrics():
    assert metrics.wer("the cat sat", "the cat sat") == 0
    assert metrics.wer("the cat sat", "the dog sat") == round(1 / 3, 4)
    assert metrics.distinct_n("a b a b", 2) == round(2 / 3, 4)
    assert metrics.rouge("नमस्ते दुनिया", "नमस्ते दुनिया", "hi")["rougeL"] == 1.0
    assert metrics.normalise("नमस्ते, दुनिया!") == "नमस्ते दुनिया"


def test_judge_uses_other_provider(monkeypatch):
    import config
    monkeypatch.setattr(llm, "_PROVIDERS", {k: v for k, v in llm._PROVIDERS.items() if k in ("gemini", "groq")})
    monkeypatch.setattr(config, "GEMINI_API_KEY", "x")
    monkeypatch.setattr(config, "GROQ_API_KEY", "y")
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "auto")
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    assert llm.resolve("judge")[0] == "groq"
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert llm.resolve("judge") == ("gemini", config.GEMINI_SMALL_MODEL)
