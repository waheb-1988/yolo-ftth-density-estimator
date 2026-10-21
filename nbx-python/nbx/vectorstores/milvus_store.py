"""Milvus Lite (embedded, file-backed) vector store implementation."""
from __future__ import annotations

import binascii
import os
import time

from nbx.core import config
from nbx.vectorstores.base import BaseVectorStore, sanitize_embedding_key


def _stable_int_id(doc_id: str) -> int:
    return binascii.crc32(doc_id.encode("utf-8"))


class MilvusStore(BaseVectorStore):
    key = "milvus"
    display = "Milvus Lite"

    def __init__(self, embedding_model: str | None = None):
        super().__init__(embedding_model)
        from pymilvus import MilvusClient

        self._collection_name = f"nbo_docs__{sanitize_embedding_key(self.embedding_model)}"
        self._client = MilvusClient(uri=config.MILVUS_DB_PATH)

        if not self._client.has_collection(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=self._dim,
                auto_id=False,
                metric_type="COSINE",
            )

    def index_documents(self, documents: list[dict]) -> int:
        start = time.time()
        if not documents:
            return 0
        texts = [d["text"] for d in documents]
        embeddings = self._embed(texts)

        rows = []
        for doc, emb in zip(documents, embeddings):
            flat_meta = {k: v for k, v in doc.get("metadata", {}).items() if isinstance(v, (str, int, float, bool))}
            rows.append(
                {
                    "id": _stable_int_id(doc["id"]),
                    "vector": emb.tolist(),
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    "type": doc.get("type", "unknown"),
                    **flat_meta,
                }
            )

        try:
            self._client.upsert(collection_name=self._collection_name, data=rows)
        except AttributeError:
            self._client.insert(collection_name=self._collection_name, data=rows)

        self._last_index_time_ms = int((time.time() - start) * 1000)
        self._last_index_count = len(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed([query])[0]
        results = self._client.search(
            collection_name=self._collection_name,
            data=[embedding.tolist()],
            limit=top_k,
            output_fields=["doc_id", "text", "type"],
        )
        out = []
        hits = results[0] if results else []
        for hit in hits:
            entity = hit.get("entity", hit)
            out.append(
                {
                    "id": entity.get("doc_id", str(hit.get("id"))),
                    "text": entity.get("text", ""),
                    "score": float(hit.get("distance", 0.0)),
                    "type": entity.get("type", "unknown"),
                    "metadata": {k: v for k, v in entity.items() if k not in ("doc_id", "text", "type")},
                }
            )
        return out

    def delete_collection(self) -> bool:
        try:
            self._client.drop_collection(collection_name=self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=self._dim,
                auto_id=False,
                metric_type="COSINE",
            )
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        count = 0
        try:
            stats = self._client.get_collection_stats(collection_name=self._collection_name)
            count = int(stats.get("row_count", 0))
        except Exception:
            pass
        size_bytes = 0
        if os.path.exists(config.MILVUS_DB_PATH):
            size_bytes = os.path.getsize(config.MILVUS_DB_PATH)
        return {
            "count": count,
            "size_bytes": size_bytes,
            "index_time_ms": self._last_index_time_ms,
            "status": "ok",
        }
