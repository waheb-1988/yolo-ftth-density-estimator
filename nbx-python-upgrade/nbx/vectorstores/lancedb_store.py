"""LanceDB local (embedded, file-backed) vector store implementation."""
from __future__ import annotations

import json
import os
import time

from nbx.core import config
from nbx.vectorstores.base import BaseVectorStore, sanitize_embedding_key


class LanceDBStore(BaseVectorStore):
    key = "lancedb"
    display = "LanceDB"

    def __init__(self, embedding_model: str | None = None):
        super().__init__(embedding_model)
        import lancedb

        self._table_name = f"nbo_docs__{sanitize_embedding_key(self.embedding_model)}"
        os.makedirs(config.LANCEDB_DIR, exist_ok=True)
        self._db = lancedb.connect(config.LANCEDB_DIR)

    def _existing_docs(self) -> list[dict]:
        if self._table_name not in self._db.table_names():
            return []
        tbl = self._db.open_table(self._table_name)
        docs = []
        for row in tbl.to_pandas().to_dict(orient="records"):
            docs.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "type": row.get("type", "unknown"),
                    "metadata": json.loads(row.get("metadata_json", "{}")),
                }
            )
        return docs

    def index_documents(self, documents: list[dict]) -> int:
        start = time.time()
        if not documents:
            return 0

        existing = self._existing_docs()
        new_ids = {d["id"] for d in documents}
        merged = [d for d in existing if d["id"] not in new_ids] + documents

        texts = [d["text"] for d in merged]
        embeddings = self._embed(texts)

        rows = [
            {
                "id": d["id"],
                "text": d["text"],
                "type": d.get("type", "unknown"),
                "metadata_json": json.dumps(d.get("metadata", {})),
                "vector": emb.tolist(),
            }
            for d, emb in zip(merged, embeddings)
        ]

        self._db.create_table(self._table_name, data=rows, mode="overwrite")
        self._last_index_time_ms = int((time.time() - start) * 1000)
        self._last_index_count = len(merged)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._table_name not in self._db.table_names():
            return []
        tbl = self._db.open_table(self._table_name)
        embedding = self._embed([query])[0]
        results = tbl.search(embedding.tolist()).metric("cosine").limit(top_k).to_list()

        out = []
        for r in results:
            distance = r.get("_distance", 1.0)
            score = max(0.0, 1.0 - distance)
            out.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "score": float(score),
                    "type": r.get("type", "unknown"),
                    "metadata": json.loads(r.get("metadata_json", "{}")),
                }
            )
        return out

    def delete_collection(self) -> bool:
        try:
            if self._table_name in self._db.table_names():
                self._db.drop_table(self._table_name)
            return True
        except Exception:
            return False

    def get_stats(self) -> dict:
        count = 0
        if self._table_name in self._db.table_names():
            count = self._db.open_table(self._table_name).count_rows()
        size_bytes = 0
        if os.path.isdir(config.LANCEDB_DIR):
            for root, _, files in os.walk(config.LANCEDB_DIR):
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
