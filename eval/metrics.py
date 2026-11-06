"""Evaluation metrics: WER, ROUGE, BERTScore, Flesch-Kincaid, Distinct-n."""
import re
import unicodedata

from pipeline.utils import lang_code

_bertscore_cache: dict = {}


def normalise(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace (keeps Devanagari letters)."""
    text = unicodedata.normalize("NFC", text.lower())
    # Drop punctuation/symbols by Unicode category; a [^\w] regex would also strip Devanagari vowel signs.
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def wer(reference: str, hypothesis: str) -> float:
    """WER = (S + D + I) / N on normalised words."""
    import jiwer
    return round(jiwer.wer(normalise(reference), normalise(hypothesis)), 4)


class _UnicodeTokenizer:
    """rouge-score's default tokenizer drops non-Latin text, which zeroes Hindi scores."""

    def tokenize(self, text):
        return normalise(text).split()


def rouge(reference: str, hypothesis: str, language: str = "en") -> dict:
    from rouge_score import rouge_scorer
    kwargs = {"use_stemmer": True} if lang_code(language) == "en" else {"tokenizer": _UnicodeTokenizer()}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], **kwargs)
    scores = scorer.score(reference, hypothesis)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def bertscore(references: list[str], hypotheses: list[str], language: str = "en") -> list[float]:
    """BERTScore F1. Uses a small model by default so it runs on a laptop CPU;
    set BERTSCORE_MODEL=roberta-large for the paper-standard (1.4 GB) model."""
    import os

    from bert_score import score
    model = os.getenv("BERTSCORE_MODEL") or (
        "distilbert-base-uncased" if lang_code(language) == "en" else "bert-base-multilingual-cased")
    _, _, f1 = score(hypotheses, references, model_type=model, lang=lang_code(language), verbose=False)
    return [round(float(x), 4) for x in f1]


def fk_grade(text: str) -> float:
    import textstat
    return round(textstat.flesch_kincaid_grade(text), 2)


def distinct_n(text: str, n: int = 2) -> float:
    """Unique n-grams / total n-grams: a simple lexical diversity measure."""
    words = normalise(text).split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return round(len(set(grams)) / len(grams), 4) if grams else 0.0
