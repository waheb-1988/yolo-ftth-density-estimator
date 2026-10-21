"""ChromaDB local persistent vector store implementation."""
from __future__ import annotations

import os
import time

from nbx.core import config
from nbx.vectorstores.base import BaseVectorStore, sanitize_embedding_key


class ChromaStore(BaseVectorStore):
    key = "chroma"
    display = "ChromaDB"

    def __init__(self, embedding_model: str | None = None):
        super().__init__(embedding_model)
        import chromadb

        self._collection_name = f"nbo_docs__{sanitize_embedding_key(self.embedding_model)}"
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index_documents(self, documents: list[dict]) -> int:
        start = time.time()
        if not documents:
            return 0
        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [{**d.get("metadata", {}), "doc_type": d.get("type", "unknown")} for d in documents]
        embeddings = self._embed(texts)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        self._last_index_time_ms = int((time.time() - start) * 1000)
        self._last_index_count = len(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed([query])[0]
        result = self._collection.query(query_embeddings=[embedding.tolist()], n_results=top_k)
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            distance = dists[i] if i < len(dists) else 1.0
            score = max(0.0, 1.0 - distance)
            meta = metas[i] if i < len(metas) else {}
            out.append(
                {
                    "id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "score": float(score),
                    "type": meta.get("doc_type", "unknown"),
                    "metadata": meta,
                }
            )
        return out

    def delete_collection(self) -> bool:
        try:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        count = self._collection.count()
        size_bytes = 0
        if os.path.isdir(config.CHROMA_DIR):
            for root, _, files in os.walk(config.CHROMA_DIR):
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
