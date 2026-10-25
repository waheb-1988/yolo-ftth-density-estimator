"""Vector store package: exposes a cached factory for the 5 backends."""
from __future__ import annotations

import importlib
from functools import lru_cache

from nbx.core import config


@lru_cache(maxsize=len(config.VECTOR_STORES) * max(len(config.EMBEDDING_MODELS), 1))
def get_store(key: str, embedding_model: str | None = None):
    """Returns a cached instance of the requested vector store backend for a
    given embedding model (one collection per (store, embedding_model) pair).
    Raises the backend's own init error if the store is unavailable
    (caller is expected to handle graceful degradation)."""
    store_cfg = next((v for v in config.VECTOR_STORES if v["key"] == key), None)
    if store_cfg is None:
        raise ValueError(f"Unknown vector store: {key}")
    module = importlib.import_module(store_cfg["module"])
    cls = getattr(module, store_cfg["class_name"])
    return cls(embedding_model=embedding_model or config.DEFAULT_EMBEDDING_MODEL)


def try_get_store(key: str, embedding_model: str | None = None):
    """Same as get_store but returns None instead of raising on failure."""
    try:
        return get_store(key, embedding_model)
    except Exception:
        return None
