"""Test fixtures: an isolated data folder, a deterministic fake LLM and fake embeddings,
so the whole pipeline runs offline without API keys."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import audio_recap, chunk, embeddings, faithfulness, llm, notes, rag  # noqa: E402


def fake_embed(texts, language=None, model_name=None):
    """Bag-of-words hashed into 64 dims: similar wording -> similar vectors."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def _first_ts(prompt):
    m = re.search(r"\[(\d{1,2}:\d{2})\]", prompt)
    return m.group(1) if m else "00:00"


def _section(prompt, title="Section"):
    ts = _first_ts(prompt)
    return {"title": title, "ts": ts,
            "bullets": [{"text": "Photosynthesis converts light energy into chemical energy.", "ts": ts},
                        {"text": "Chlorophyll absorbs mostly red and blue light.", "ts": ts}],
            "definitions": [{"term": "Chlorophyll", "definition": "The green pigment in plants.", "ts": ts}],
            "formulas": [], "examples": ["Leaves of a plant"]}


def fake_llm(model, system, prompt, temperature, top_p, json_mode, max_tokens, task):
    def n(pattern, default=1):
        match = re.search(pattern, prompt)
        return int(match.group(1)) if match else default

    if task == "chunk_titles":
        data = {"titles": [f"Topic {i + 1}" for i in range(n(r"Below are (\d+)"))]}
    elif task in ("notes_zero_shot", "notes_few_shot", "notes_tot_generate", "notes_tot_refine"):
        data = _section(prompt)
    elif task == "notes_cot":
        data = {"reasoning": "1. concepts ... 4. all supported", "section": _section(prompt)}
    elif task == "notes_tot_judge":
        ids = re.findall(r"### Candidate ([A-Z])", prompt)
        data = {"scores": [{"id": c, "coverage": 7 + i, "faithfulness": 8, "clarity": 7, "profile_fit": 6,
                            "reason": "ok", "feedback": "add an example"} for i, c in enumerate(ids)]}
    elif task == "notes_reduce":
        data = {"summary": [f"Line {i}" for i in range(5)],
                "glossary": [{"term": "Chlorophyll", "definition": "green pigment"}]}
    elif task == "interview":
        data = {"done": True, "question": "", "profile": {"level": "revising", "goal": "exam prep",
                "depth": "brief", "language": "English", "weak_areas": [], "quiz_size": 4}}
    elif task in ("quiz_cot", "quiz_plain"):
        count = n(r"Create (\d+) multiple-choice")
        data = {"questions": [{"concept": f"concept {i}", "bloom": "remember", "difficulty": "easy",
                               "reasoning": "concept, answer, misconceptions", "stem": f"Question {i}?",
                               "options": ["right", "wrong a", "wrong b", "wrong c"], "answer_index": 0,
                               "explanation": "because", "ts": _first_ts(prompt)} for i in range(count)]}
    elif task == "quiz_verify":
        # Answer by finding the option text "right" in each listed question.
        answers, qid = [], 0
        for line in prompt.splitlines():
            if re.match(r"^\d+\. ", line):
                qid += 1
            m = re.match(r"^\s+(\d)\) right$", line)
            if m:
                answers.append({"id": qid, "answer_index": int(m.group(1))})
        data = {"answers": answers}
    elif task == "quiz_followup":
        data = {"concept": "c", "bloom": "remember", "difficulty": "easy", "stem": "Easier?",
                "options": ["right", "x", "y", "z"], "answer_index": 0, "explanation": "e", "ts": "00:05"}
    elif task == "flashcards":
        data = {"cards": [{"kind": "qa", "front": "What is chlorophyll?", "back": "A green pigment", "ts": "00:05"},
                          {"kind": "cloze", "front": "Plants use light", "back": "light", "ts": "00:05"}]}
    elif task == "diagrams_plan":
        data = {"diagrams": [
            {"title": "Broken", "kind": "flowchart", "mermaid": "flowchart TD\nA[Light (sun)] --> B[Leaf]"},
            {"title": "Good", "kind": "flowchart", "mermaid": 'flowchart TD\nA["Light"] --> B["Leaf"]'}]}
    elif task == "diagram_fix":
        data = {"mermaid": 'flowchart TD\nA["Light sun"] --> B["Leaf"]'}
    elif task == "mindmap":
        data = {"mermaid": "mindmap\n  root((Plants))\n    Light\n    Water"}
    elif task in ("recap_script", "recap_podcast"):
        data = {"lines": [{"speaker": "teacher", "text": "Today we learned about **photosynthesis**."}]}
    elif task == "claims_split":
        sections = [int(s) for s in re.findall(r"## Section (\d+)", prompt)]
        data = {"claims": [{"section": s, "claim": f"Claim {k} from section {s}"} for s in sections for k in (1, 2)]}
    elif task == "claims_judge":
        count = len(re.findall(r"^\d+\. ", prompt, flags=re.M))
        data = {"verdicts": [{"id": i + 1, "verdict": "supported" if i % 2 == 0 else "not_mentioned",
                              "reason": "r"} for i in range(count)]}
    elif task == "rag_answer":
        return f"Chlorophyll absorbs light [{_first_ts(prompt)}].", 100, 20
    elif task == "rag_judge":
        data = {"correct": True, "reason": "matches"}
    else:
        raise AssertionError(f"fake LLM got unexpected task {task!r}")
    return json.dumps(data), 100, 50


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LECTURES_DIR", tmp_path / "lectures")
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "MERMAID_VALIDATE_REMOTE", False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "fake")
    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 40)
    monkeypatch.setattr(config, "MAX_CHUNK_TOKENS", 120)
    llm.register_provider("fake", fake_llm)
    monkeypatch.setattr(rag, "_chroma", False)            # NumPy index: fast and dependency-free
    for module in (embeddings, chunk, notes, faithfulness, rag):
        monkeypatch.setattr(module, "embed", fake_embed)

    async def fake_tts(script, out_path):
        Path(out_path).write_bytes(b"ID3fake")
    monkeypatch.setattr(audio_recap, "_synthesise", fake_tts)
    return tmp_path


SRT = "\n\n".join(
    f"{i + 1}\n00:{i // 6:02d}:{(i % 6) * 10:02d},000 --> 00:{i // 6:02d}:{(i % 6) * 10 + 9:02d},000\n{text} Point {i}."
    for i, text in enumerate(
        ["Today we study photosynthesis in green plants and how leaves capture light energy."] * 12
        + ["Now a different topic: cellular respiration releases energy from glucose in mitochondria."] * 12))
