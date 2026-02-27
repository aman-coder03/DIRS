"""
embedding.py
------------
Generates text embeddings using Sentence Transformers.
Models are loaded once and cached in memory for the lifetime of the process.
"""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_MAP = {
    "BGE-small": "BAAI/bge-small-en",
    "MiniLM":    "sentence-transformers/all-MiniLM-L6-v2",
    "E5-small":  "intfloat/e5-small-v2",
}

# In-process model cache — avoids reloading on every call
_loaded_models: dict[str, SentenceTransformer] = {}


def embed(texts: list[str], model_name: str = "BGE-small") -> list:
    """
    Embed a list of texts using the specified model.

    Args:
        texts:      List of strings to embed.
        model_name: Key from MODEL_MAP. Defaults to 'BGE-small'.

    Returns:
        List of embedding vectors (numpy arrays).

    Raises:
        ValueError: If model_name is not in MODEL_MAP.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unsupported embedding model: '{model_name}'. "
            f"Choose from: {list(MODEL_MAP.keys())}"
        )

    hf_name = MODEL_MAP[model_name]

    if hf_name not in _loaded_models:
        logger.info("Loading embedding model: %s", hf_name)
        _loaded_models[hf_name] = SentenceTransformer(hf_name)

    model = _loaded_models[hf_name]
    return model.encode(texts)