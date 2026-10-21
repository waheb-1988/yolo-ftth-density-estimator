"""FAISS local flat-index vector store implementation (cosine similarity via
L2-normalized inner product). Metadata is kept in a parallel JSON sidecar
file since FAISS itself only stores vectors."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from nbx.core import config
from nbx.vectorstores.base import BaseVectorStore, sanitize_embedding_key


class FaissStore(BaseVectorStore):
    key = "faiss"
    display = "FAISS"

    def __init__(self, embedding_model: str | None = None):
        super().__init__(embedding_model)
        import faiss

        self._faiss = faiss
        os.makedirs(config.FAISS_DIR, exist_ok=True)
        suffix = sanitize_embedding_key(self.embedding_model)
        self._index_file = os.path.join(config.FAISS_DIR, f"index__{suffix}.faiss")
        self._docs_file = os.path.join(config.FAISS_DIR, f"docs__{suffix}.json")
        self._docs: list[dict] = []
        self._index = self._faiss.IndexFlatIP(self._dim)

        if os.path.exists(self._index_file) and os.path.exists(self._docs_file):
            try:
                self._index = self._faiss.read_index(self._index_file)
                self._docs = json.loads(open(self._docs_file, "r", encoding="utf-8").read())
            except Exception:
                self._index = self._faiss.IndexFlatIP(self._dim)
                self._docs = []

    def _save(self):
        self._faiss.write_index(self._index, self._index_file)
        with open(self._docs_file, "w", encoding="utf-8") as f:
            json.dump(self._docs, f)

    def index_documents(self, documents: list[dict]) -> int:
        start = time.time()
        if not documents:
            return 0

        existing_ids = {d["id"] for d in documents}
        self._docs = [d for d in self._docs if d["id"] not in existing_ids]
        self._docs.extend(documents)

        texts = [d["text"] for d in self._docs]
        embeddings = self._embed(texts).astype("float32")
        self._faiss.normalize_L2(embeddings)

        self._index = self._faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        self._save()

        self._last_index_time_ms = int((time.time() - start) * 1000)
        self._last_index_count = len(self._docs)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._index.ntotal == 0:
            return []
        query_vec = self._embed([query]).astype("float32")
        self._faiss.normalize_L2(query_vec)
        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[idx]
            out.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(score),
                    "type": doc.get("type", "unknown"),
                    "metadata": doc.get("metadata", {}),
                }
            )
        return out

    def delete_collection(self) -> bool:
        try:
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._docs = []
            for path in (self._index_file, self._docs_file):
                if os.path.exists(path):
                    os.remove(path)
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        size_bytes = 0
        for path in (self._index_file, self._docs_file):
            if os.path.exists(path):
                size_bytes += os.path.getsize(path)
        return {
            "count": self._index.ntotal,
            "size_bytes": size_bytes,
            "index_time_ms": self._last_index_time_ms,
            "status": "ok",
        }
