"""Interview Pattern: the LLM interviews the student (one question at a time, 3-5 turns)
and produces a LearnerProfile that conditions every later prompt."""
from pipeline import llm, prompts
from pipeline.schemas import InterviewTurn, LearnerProfile

MAX_TURNS = 5

PRESETS = {
    "Beginner (first exposure)": LearnerProfile(
        level="first exposure", goal="understand", depth="detailed", language="English",
        background="new to the subject", quiz_size=6),
    "Exam revision (exam tomorrow)": LearnerProfile(
        level="exam tomorrow", goal="exam prep", depth="brief", language="English",
        background="has studied the topic, needs fast revision", quiz_size=12),
    "Interview prep": LearnerProfile(
        level="revising", goal="interview prep", depth="standard", language="English",
        background="preparing for technical interviews", quiz_size=10),
    "Hinglish learner": LearnerProfile(
        level="revising", goal="understand", depth="standard", language="Hinglish", quiz_size=8),
}
NEUTRAL = LearnerProfile()            # used by experiments that are not about personalisation


def _history_text(history: list[dict]) -> str:
    if not history:
        return "(no messages yet)"
    return "\n".join(f"{'Interviewer' if m['role'] == 'assistant' else 'Student'}: {m['content']}"
                     for m in history)


def next_turn(history: list[dict], lecture_title: str, topics: list[str]) -> InterviewTurn:
    """history: [{"role": "assistant"|"user", "content": str}]. Returns the next question,
    or done=True with the profile once the LLM decides it knows enough."""
    asked = sum(1 for m in history if m["role"] == "assistant")
    prompt = prompts.render(
        "interview", lecture_title=lecture_title, topics=", ".join(topics[:12]) or "unknown",
        max_turns=MAX_TURNS, turns=asked, history=_history_text(history),
    )
    turn = llm.generate_json(prompt, InterviewTurn, temperature=0.3, task="interview", cache=False)
    if turn.done and turn.profile is None:
        turn.done = False
    if not turn.done and (asked >= MAX_TURNS or not turn.question):
        # Hard stop: force the model to summarise what it has.
        forced = prompt + "\n\nYou have asked enough questions. Reply now with done=true and the profile."
        turn = llm.generate_json(forced, InterviewTurn, temperature=0.0, task="interview", cache=False)
        if turn.profile is None:
            turn = InterviewTurn(done=True, profile=LearnerProfile())
        turn.done = True
    if turn.profile is not None:
        turn.profile.quiz_size = max(3, min(15, turn.profile.quiz_size))
    return turn
