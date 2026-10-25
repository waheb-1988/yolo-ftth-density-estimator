"""Hybrid (dense + sparse) retrieval over a vector store backend."""
from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sparse_score(query: str, text: str) -> float:
    """Lightweight term-overlap sparse score (no external BM25 dependency)."""
    q_tokens = set(_tokenize(query))
    d_tokens = _tokenize(text)
    if not q_tokens or not d_tokens:
        return 0.0
    d_counts = Counter(d_tokens)
    overlap = sum(d_counts[t] for t in q_tokens if t in d_counts)
    score = overlap / (len(d_tokens) ** 0.5)
    return min(score, 1.0)


class HybridRetriever:
    """Runs multiple queries against a dense vector store, blends each hit's
    dense similarity with a sparse term-overlap score, and returns the
    deduplicated top-k results ranked by the blended score.

    Segment/pattern documents are short and keyword-dense (they literally
    contain the dv_segment string used in the queries), so a plain global
    top-k tends to crowd out actual offer documents entirely. A minimum
    quota of offer-type hits is reserved so the downstream LLM always has
    real offers to choose from and the reranker has retrieval scores to
    backfill with.
    """

    def __init__(self, store, dense_weight: float = 0.7):
        self.store = store
        self.dense_weight = dense_weight

    def retrieve(self, queries: list[str], top_k: int = 10, min_offer_slots: int = 6) -> list[dict]:
        merged: dict[str, dict] = {}
        search_k = max(top_k * 3, 15)
        for query in queries:
            for hit in self.store.search(query, top_k=search_k):
                sparse = sparse_score(query, hit["text"])
                hybrid = self.dense_weight * hit["score"] + (1 - self.dense_weight) * sparse
                key = hit["id"]
                if key not in merged or hybrid > merged[key]["retrieval_score"]:
                    merged[key] = {**hit, "retrieval_score": hybrid}

        ranked = sorted(merged.values(), key=lambda r: r["retrieval_score"], reverse=True)
        offers = [r for r in ranked if r["type"] == "offer"]
        others = [r for r in ranked if r["type"] != "offer"]

        selected = offers[:min_offer_slots]
        selected_ids = {r["id"] for r in selected}

        rest_pool = sorted(offers[min_offer_slots:] + others, key=lambda r: r["retrieval_score"], reverse=True)
        for hit in rest_pool:
            if len(selected) >= top_k:
                break
            if hit["id"] in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(hit["id"])

        selected.sort(key=lambda r: r["retrieval_score"], reverse=True)
        return selected[:top_k]
