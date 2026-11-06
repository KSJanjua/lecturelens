"""Sentence embeddings on CPU (MiniLM; multilingual MiniLM for Hindi/Hinglish)."""
import os
import threading

import numpy as np

import config
from pipeline.utils import lang_code

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows without developer mode
_models: dict = {}
_lock = threading.Lock()


def model_for(language: str | None) -> str:
    return config.EMBED_MODEL_MULTILINGUAL if lang_code(language) == "hi" else config.EMBED_MODEL


def _load(name: str):
    with _lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer
            _models[name] = SentenceTransformer(name, device="cpu")
        return _models[name]


def embed(texts: list[str], language: str | None = None, model_name: str | None = None) -> np.ndarray:
    """Return L2-normalised embeddings, so a dot product is cosine similarity."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _load(model_name or model_for(language))
    vectors = model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float32)
