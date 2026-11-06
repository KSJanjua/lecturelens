"""MCQ quiz with Chain-of-Thought generation, a self-verification pass and adaptive follow-ups.

CoT: for each question the model identifies the concept, writes the answer, lists common
misconceptions and turns them into distractors. Self-verification: a different model answers
each question WITHOUT the key; disagreements are regenerated once, then dropped."""
import random
import re
from concurrent.futures import ThreadPoolExecutor

import config
from pipeline import llm, prompts
from pipeline.chunk import chunk_text_with_timestamps
from pipeline.schemas import MCQ, Chunk, LearnerProfile, Quiz, SolverAnswers
from pipeline.utils import estimate_tokens, fmt_ts, parse_ts, sha

BLOOM_MIX = {                     # share of remember / understand / apply
    "first exposure": (0.5, 0.35, 0.15),
    "revising": (0.3, 0.4, 0.3),
    "exam tomorrow": (0.2, 0.4, 0.4),
}
_NO_SHUFFLE = re.compile(r"\b(all|none|both) of the above\b|\bboth\b.*\band\b", re.I)


def _level_key(profile: LearnerProfile) -> str:
    level = profile.level.lower()
    return next((k for k in BLOOM_MIX if k.split()[0] in level), "revising")


def bloom_mix(profile: LearnerProfile, count: int) -> str:
    shares = BLOOM_MIX[_level_key(profile)]
    counts = [int(round(count * s)) for s in shares]
    counts[1] += count - sum(counts)
    return ", ".join(f"{n} {name}" for n, name in zip(counts, ("remember", "understand", "apply")) if n > 0)


def difficulty(profile: LearnerProfile) -> str:
    return {"first exposure": "mostly easy, a few medium", "revising": "medium",
            "exam tomorrow": "medium to hard, exam style"}[_level_key(profile)]


def _batches(chunks: list[Chunk], max_tokens: int) -> list[list[Chunk]]:
    batches, cur = [], []
    for c in chunks:
        if cur and sum(estimate_tokens(x.text) for x in cur) + estimate_tokens(c.text) > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(c)
    return batches + ([cur] if cur else [])


def _allocate(weights: list[int], total: int) -> list[int]:
    """Largest-remainder split of `total` questions across batches by transcript size."""
    s = sum(weights) or 1
    raw = [w * total / s for w in weights]
    counts = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[:total - sum(counts)]:
        counts[i] += 1
    return counts


def shuffle_options(q: MCQ) -> MCQ:
    """LLMs favour certain answer positions; shuffle deterministically unless options refer to each other."""
    if any(_NO_SHUFFLE.search(o) for o in q.options):
        return q
    order = list(range(4))
    random.Random(sha(q.stem)).shuffle(order)
    q.options = [q.options[i] for i in order]
    q.answer_index = order.index(q.answer_index)
    return q


def _batch_text(batch: list[Chunk]) -> str:
    return "\n\n".join(chunk_text_with_timestamps(c) for c in batch)


def _fix_ts(q: MCQ, batch: list[Chunk]) -> MCQ:
    sec = parse_ts(q.ts)
    if sec is None or not (batch[0].start - 5 <= sec <= batch[-1].end + 5):
        q.ts = fmt_ts(batch[0].start)
    return q


def _generate(batch, profile, count, use_cot, extra="") -> list[MCQ]:
    template = "quiz_cot" if use_cot else "quiz_plain"
    prompt = prompts.render(template, count=count, profile=profile.describe(),
                            bloom_mix=bloom_mix(profile, count), difficulty=difficulty(profile),
                            transcript=_batch_text(batch), language=profile.language) + extra
    result = llm.generate_json(prompt, Quiz, system=prompts.system(),
                               temperature=config.QUIZ_TEMPERATURE, task=template)
    return [shuffle_options(_fix_ts(q, batch)) for q in result.questions[:count]]


def verify(questions: list[MCQ], batch: list[Chunk]) -> list[bool | None]:
    """A second model answers without seeing the key. True where it agrees with the key,
    None when the verifier could not be reached (the question is kept but not counted as passed)."""
    if not questions:
        return []
    listing = "\n\n".join(
        f"{i + 1}. {q.stem}\n" + "\n".join(f"   {j}) {o}" for j, o in enumerate(q.options))
        for i, q in enumerate(questions))
    try:
        result = llm.generate_json(prompts.render("quiz_verify", transcript=_batch_text(batch), questions=listing),
                                   SolverAnswers, role="judge", temperature=0.0, task="quiz_verify")
    except llm.LLMError:
        return [None] * len(questions)
    answers = {a.id: a.answer_index for a in result.answers}
    return [answers.get(i + 1) == q.answer_index for i, q in enumerate(questions)]


def _process_batch(batch, profile, count, use_cot, do_verify):
    stats = {"generated": 0, "passed_first_try": 0, "regenerated": 0, "dropped": 0, "unverified": 0}
    questions = _generate(batch, profile, count, use_cot)
    stats["generated"] = len(questions)
    if not do_verify:
        return questions, stats
    kept = []
    for q, ok in zip(questions, verify(questions, batch)):
        if ok is None:
            stats["unverified"] += 1
            kept.append(q)
            continue
        if ok:
            q.verified = True
            stats["passed_first_try"] += 1
            kept.append(q)
            continue
        stats["regenerated"] += 1
        hint = (f"\n\nWrite exactly 1 question about the concept \"{q.concept}\". A previous question on it "
                f"was ambiguous (\"{q.stem}\"): make sure only one option can be defended from the transcript.")
        try:
            retry = _generate(batch, profile, 1, use_cot, extra=hint)
        except llm.LLMError:
            retry = []
        if retry and verify(retry, batch)[0]:
            retry[0].verified = True
            kept.append(retry[0])
        else:
            stats["dropped"] += 1
    return kept, stats


def generate_quiz(chunks: list[Chunk], profile: LearnerProfile, count: int | None = None,
                  use_cot: bool = True, do_verify: bool = True) -> Quiz:
    count = count or profile.quiz_size or config.QUIZ_DEFAULT_COUNT
    batches = _batches(chunks, config.QUIZ_BATCH_TOKENS)
    counts = _allocate([sum(estimate_tokens(c.text) for c in b) for b in batches], count)
    jobs = [(b, n) for b, n in zip(batches, counts) if n > 0]

    with ThreadPoolExecutor(max_workers=max(1, config.MAX_WORKERS)) as pool:
        results = list(pool.map(lambda job: _process_batch(job[0], profile, job[1], use_cot, do_verify), jobs))

    questions, stats = [], {"generated": 0, "passed_first_try": 0, "regenerated": 0, "dropped": 0, "unverified": 0}
    for qs, st in results:
        questions.extend(qs)
        for k in stats:
            stats[k] += st[k]
    checked = stats["generated"] - stats["unverified"]
    stats["answerability_rate"] = round(stats["passed_first_try"] / checked, 3) if checked else None
    stats["use_cot"] = use_cot
    return Quiz(questions=questions, stats=stats)


def chunk_for(ts: str, chunks: list[Chunk]) -> Chunk:
    sec = parse_ts(ts) or 0
    return next((c for c in chunks if c.start - 1 <= sec <= c.end + 1), chunks[0])


def generate_followup(question: MCQ, chosen: int, chunks: list[Chunk], profile: LearnerProfile) -> MCQ:
    """Adaptive mode: after a wrong answer, an easier question on the same concept."""
    chunk = chunk_for(question.ts, chunks)
    follow = llm.generate_json(
        prompts.render("quiz_followup", stem=question.stem, options=" | ".join(question.options),
                       correct=question.options[question.answer_index], chosen=question.options[chosen],
                       concept=question.concept, profile=profile.describe(),
                       transcript=chunk_text_with_timestamps(chunk), language=profile.language),
        MCQ, system=prompts.system(), temperature=config.QUIZ_TEMPERATURE, task="quiz_followup",
    )
    return shuffle_options(_fix_ts(follow, [chunk]))
