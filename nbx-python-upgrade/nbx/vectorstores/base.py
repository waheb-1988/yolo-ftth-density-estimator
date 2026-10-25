"""Abstract vector store interface implemented by all 5 backends."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from nbx.core import config
from nbx.llms.embeddings import embed_texts, get_dimension

_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def sanitize_embedding_key(embedding_model: str) -> str:
    """Turns an embedding model key into a safe collection/table/file suffix,
    e.g. 'qwen3-embedding:4b' -> 'qwen3_embedding_4b'."""
    return _SANITIZE_RE.sub("_", embedding_model.lower()).strip("_")


class BaseVectorStore(ABC):
    """All vector store backends must implement this interface.

    Each store keeps one collection/table/index per embedding model (vector
    dimensions differ across models), so switching embedding_model never
    corrupts an existing index — it indexes a parallel collection instead.
    """

    key: str = "base"
    display: str = "Base"

    def __init__(self, embedding_model: str | None = None):
        self.embedding_model = embedding_model or config.DEFAULT_EMBEDDING_MODEL
        self._dim = get_dimension(self.embedding_model)
        self._last_index_time_ms: int = 0
        self._last_index_count: int = 0

    def _embed(self, texts: list[str]):
        return embed_texts(texts, self.embedding_model)

    @abstractmethod
    def index_documents(self, documents: list[dict]) -> int:
        """Embeds and indexes a list of {id, text, type, metadata} documents.
        Returns the number of documents indexed."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Returns up to top_k results, each shaped as:
        {"id": str, "text": str, "score": float, "type": str, "metadata": dict}"""

    @abstractmethod
    def delete_collection(self) -> bool:
        """Drops the collection/index. Returns True on success."""

    @abstractmethod
    def get_stats(self) -> dict:
        """Returns {"count": int, "size_bytes": int, "index_time_ms": int, "status": str}"""
