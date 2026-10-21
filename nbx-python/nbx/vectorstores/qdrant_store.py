"""Qdrant vector store implementation running in embedded local mode
(file-backed, no external server process required)."""
from __future__ import annotations

import os
import time
import uuid

from nbx.core import config
from nbx.vectorstores.base import BaseVectorStore, sanitize_embedding_key

QDRANT_DIR = os.path.join(str(config.DATA_DIR), "qdrant_local")


class QdrantStore(BaseVectorStore):
    key = "qdrant"
    display = "Qdrant (local)"

    def __init__(self, embedding_model: str | None = None):
        super().__init__(embedding_model)
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self._collection_name = f"nbo_docs__{sanitize_embedding_key(self.embedding_model)}"
        os.makedirs(QDRANT_DIR, exist_ok=True)
        self._client = QdrantClient(path=QDRANT_DIR)

        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(size=self._dim, distance=qmodels.Distance.COSINE),
            )

    @staticmethod
    def _point_id(doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    def index_documents(self, documents: list[dict]) -> int:
        start = time.time()
        if not documents:
            return 0
        texts = [d["text"] for d in documents]
        embeddings = self._embed(texts)

        points = []
        for doc, emb in zip(documents, embeddings):
            payload = {**doc.get("metadata", {}), "doc_id": doc["id"], "text": doc["text"], "type": doc.get("type", "unknown")}
            points.append(
                self._qmodels.PointStruct(id=self._point_id(doc["id"]), vector=emb.tolist(), payload=payload)
            )

        self._client.upsert(collection_name=self._collection_name, points=points)
        self._last_index_time_ms = int((time.time() - start) * 1000)
        self._last_index_count = len(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed([query])[0]
        results = self._client.search(
            collection_name=self._collection_name, query_vector=embedding.tolist(), limit=top_k
        )
        out = []
        for r in results:
            payload = r.payload or {}
            out.append(
                {
                    "id": payload.get("doc_id", str(r.id)),
                    "text": payload.get("text", ""),
                    "score": float(r.score),
                    "type": payload.get("type", "unknown"),
                    "metadata": {k: v for k, v in payload.items() if k not in ("text", "doc_id", "type")},
                }
            )
        return out

    def delete_collection(self) -> bool:
        try:
            self._client.delete_collection(collection_name=self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=self._qmodels.VectorParams(size=self._dim, distance=self._qmodels.Distance.COSINE),
            )
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        try:
            info = self._client.get_collection(self._collection_name)
            count = info.points_count or 0
        except Exception:
            count = 0
        size_bytes = 0
        if os.path.isdir(QDRANT_DIR):
            for root, _, files in os.walk(QDRANT_DIR):
                for f in files:
                    try:
                        size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        return {
            "count": count,
            "size_bytes": size_bytes,
            "index_time_ms": self._last_index_time_ms,
            "status": "ok",
        }
