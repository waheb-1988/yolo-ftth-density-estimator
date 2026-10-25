"""Embedding router: dispatches to sentence-transformers (local, CPU) or
Ollama's /api/embed (for the Qwen3-Embedding GGUF models), per the model
roster in core.config.EMBEDDING_MODELS. Every vector store calls through
here rather than loading a model directly, so adding a new embedding model
only means adding a roster entry."""
from __future__ import annotations

from functools import lru_cache

import httpx
import numpy as np

from nbx.core import config

DEFAULT_TIMEOUT = 120.0


@lru_cache(maxsize=None)
def _get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _resolve(model_key: str | None) -> dict:
    key = model_key or config.DEFAULT_EMBEDDING_MODEL
    info = config.get_embedding_info(key)
    if info is None:
        raise ValueError(f"Unknown embedding model '{key}'")
    return info


def get_dimension(model_key: str | None = None) -> int:
    """Returns the vector dimension for a model without loading it —
    vector stores need this at collection-creation time."""
    return _resolve(model_key)["dimension"]


def _embed_ollama(texts: list[str], model_name: str) -> np.ndarray:
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.post(f"{config.OLLAMA_URL}/api/embed", json={"model": model_name, "input": texts})
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as e:
        raise ValueError(f"Ollama is not reachable at {config.OLLAMA_URL} (needed for embedding model '{model_name}').") from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"Embedding model '{model_name}' is not pulled in Ollama yet. Run: ollama pull {model_name}"
            ) from e
        raise ValueError(f"Ollama embedding call failed for '{model_name}': {e}") from e
    return np.array(data["embeddings"], dtype="float32")


def _embed_sentence_transformers(texts: list[str], model_name: str) -> np.ndarray:
    model = _get_sentence_transformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_texts(texts: list[str], model_key: str | None = None) -> np.ndarray:
    info = _resolve(model_key)
    if info["backend"] == "ollama":
        if not config.OLLAMA_ENABLED:
            raise ValueError(f"Embedding model '{info['key']}' needs Ollama, which is disabled (OLLAMA_ENABLED=0). "
                             "Use all-MiniLM-L6-v2 or set OLLAMA_ENABLED=1.")
        return _embed_ollama(texts, info["model_name"])
    return _embed_sentence_transformers(texts, info["model_name"])
