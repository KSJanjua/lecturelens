"""End-to-end: transcript -> chunks -> full study kit -> RAG, through the fake LLM."""
from tests.conftest import SRT

from pipeline import ingest, llm, profile, rag, store
from pipeline.export import markdown_to_pdf, notes_to_markdown, quiz_to_markdown
from pipeline.flashcards import to_anki_csv
from pipeline.quiz import generate_followup
from pipeline.run import KitOptions, build_kit, list_kits, load_kit, prepare_lecture


def _lecture():
    meta = ingest.ingest_transcript(SRT.encode(), "bio.srt", "Biology 101", "en")
    return meta, prepare_lecture(meta.id)


def test_semantic_chunks_follow_topics(env):
    meta, chunks = _lecture()
    assert len(chunks) >= 2
    assert all(c.title.startswith("Topic") for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == store.load_transcript(meta.id)[-1].end


def test_full_kit_with_tot(env):
    meta, chunks = _lecture()
    learner = profile.PRESETS["Beginner (first exposure)"]
    with llm.track_usage() as cost:
        kit = build_kit(meta.id, learner, KitOptions(strategy="tot"))
    assert cost["calls"] > 0

    notes = kit["notes"]
    assert len(notes.sections) == len(chunks) and len(notes.summary) == 5
    assert notes.sections[0].bullets, "duplicate removal must keep the first copy"
    trees = kit["notes_log"]["chunks"]
    assert len(trees) == len(chunks)
    assert [len(level["nodes"]) for level in trees[0]["levels"]] == [3, 2]   # breadth 3 -> keep 2
    assert trees[0]["selected"].startswith("1")

    quiz = kit["quiz"]
    assert len(quiz.questions) == learner.quiz_size
    assert quiz.stats["answerability_rate"] == 1.0
    assert all(q.options[q.answer_index] == "right" for q in quiz.questions)

    d = kit["diagrams"]
    assert d[0].kind == "mindmap"
    broken = next(x for x in d if x.title == "Broken")
    assert broken.valid and not broken.first_try_valid and broken.attempts == 2
    assert kit["diagram_stats"]["first_try_rate"] == 0.5

    assert kit["recap_audio"].read_bytes() == b"ID3fake"
    assert "**" not in kit["recap"].lines[0].text

    report = kit["faithfulness"]
    assert report.total > 0 and 0 < report.score < 1
    assert report.supported + report.not_mentioned + report.contradicted == report.total

    assert "What is chlorophyll?,A green pigment [00:05]" in to_anki_csv(kit["flashcards"].cards)
    assert "_____" in kit["flashcards"].cards[1].front
    md = notes_to_markdown(notes)
    assert md.startswith("# Biology 101")
    assert markdown_to_pdf(md)[:4] == b"%PDF"
    assert "## Answers" in quiz_to_markdown(quiz)

    follow = generate_followup(quiz.questions[0], 1, chunks, learner)
    assert follow.options[follow.answer_index] == "right"

    # Cached reload: no new API calls, same content.
    saved = list_kits(meta.id)
    assert saved and saved[0]["key"] == kit["key"]
    with llm.track_usage() as cost:
        again = load_kit(meta.id, kit["key"])
    assert cost["calls"] == 0 and again["notes"] == notes


def test_other_strategies_and_profiles_get_separate_kits(env):
    meta, _ = _lecture()
    keys = set()
    for strategy in ("zero_shot", "few_shot", "cot"):
        kit = build_kit(meta.id, profile.NEUTRAL, KitOptions(strategy=strategy, outputs=("notes",)))
        assert kit["notes"].strategy == strategy
        keys.add(kit["key"])
    kit = build_kit(meta.id, profile.PRESETS["Exam revision (exam tomorrow)"], KitOptions(strategy="cot", outputs=("notes",)))
    keys.add(kit["key"])
    assert len(keys) == 4
    assert "reasoning" in kit["notes_log"]["chunks"][0]


def test_rag_answers_with_valid_citations(env):
    meta, _ = _lecture()
    result = rag.ask("What does chlorophyll absorb?", [meta.id])
    assert result["found"] and len(result["hits"]) <= 4
    assert result["citations"] and all(c["valid"] for c in result["citations"])


def test_interview_returns_profile(env):
    turn = profile.next_turn([{"role": "assistant", "content": "Hi?"}, {"role": "user", "content": "exam tomorrow"}],
                             "Biology", ["Topic 1"])
    assert turn.done and turn.profile.goal == "exam prep"
