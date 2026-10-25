"""Cross-scoring reranker: blends LLM judgment, retrieval relevance, and
price fit into a single final_score per offer (Step 4 of the pipeline)."""
from __future__ import annotations

from nbx.core.config import RERANK_WEIGHTS

FALLBACK_RETRIEVAL_SCORE = 0.4
TOP_N = 4


def _price_fit(offer_price: float, avg_offer_price: float) -> float:
    denom = max(avg_offer_price, offer_price, 1.0)
    return max(0.0, 1.0 - abs(offer_price - avg_offer_price) / denom)


def _match_offer(offer_name: str, offers_by_name: dict[str, dict]) -> dict | None:
    if not offer_name:
        return None
    if offer_name in offers_by_name:
        return offers_by_name[offer_name]
    normalized = offer_name.strip().upper()
    for name, offer in offers_by_name.items():
        if name.upper() == normalized:
            return offer
    for name, offer in offers_by_name.items():
        if normalized in name.upper() or name.upper() in normalized:
            return offer
    return None


def rerank(
    llm_offers: list[dict],
    retrieved: list[dict],
    customer: dict,
    offers_by_name: dict[str, dict],
    graph_scores: dict[str, float] | None = None,
    cf_scores: dict[str, float] | None = None,
) -> list[dict]:
    retrieval_score_by_offer: dict[str, float] = {}
    for hit in retrieved:
        if hit.get("type") == "offer":
            program_rule = hit.get("metadata", {}).get("program_rule")
            if program_rule:
                retrieval_score_by_offer[program_rule] = max(
                    retrieval_score_by_offer.get(program_rule, 0.0), hit["retrieval_score"]
                )

    graph_scores = graph_scores or {}
    cf_scores = cf_scores or {}
    avg_offer_price = customer.get("avg_offer_price", 5.0)
    candidates: list[dict] = []
    used_offers: set[str] = set()

    def _final_score(llm_score: float, cf_score: float, retrieval_score: float, graph_score: float, price_fit: float) -> float:
        return (
            RERANK_WEIGHTS["llm_score"] * llm_score
            + RERANK_WEIGHTS["cf_score"] * cf_score
            + RERANK_WEIGHTS["retrieval_score"] * retrieval_score
            + RERANK_WEIGHTS["graph_score"] * graph_score
            + RERANK_WEIGHTS["price_fit"] * price_fit
        )

    def _make(offer: dict, llm_score: float, cf_score: float, retrieval_score: float, graph_score: float, why: str) -> dict:
        price_fit = _price_fit(offer["price_omr"], avg_offer_price)
        final_score = _final_score(llm_score, cf_score, retrieval_score, graph_score, price_fit)
        return {
            "offer_name": offer["offer_name"],
            "category": offer["category"],
            "price_omr": offer["price_omr"],
            "data_gb": offer["data_gb"],
            "voice_mins": offer["voice_mins"],
            "validity_days": offer["validity_days"],
            "final_score": round(final_score, 4),
            "llm_score": round(llm_score, 4),
            "cf_score": round(cf_score, 4),
            "retrieval_score": round(retrieval_score, 4),
            "graph_score": round(graph_score, 4),
            "why": why,
        }

    for entry in llm_offers:
        offer = _match_offer(entry.get("offer_name", ""), offers_by_name)
        if offer is None or offer["program_rule"] in used_offers:
            continue
        used_offers.add(offer["program_rule"])

        llm_score = min(max(float(entry.get("llm_score", 0.7)), 0.0), 1.0)
        cf_score = cf_scores.get(offer["program_rule"], 0.0)
        retrieval_score = retrieval_score_by_offer.get(offer["program_rule"], FALLBACK_RETRIEVAL_SCORE)
        graph_score = graph_scores.get(offer["program_rule"], 0.0)
        why = entry.get("why") or f"Selected by LLM reasoning for {offer['category']} need."
        candidates.append(_make(offer, llm_score, cf_score, retrieval_score, graph_score, why))

    # Backfill order when the LLM produced fewer than TOP_N valid offers:
    # collaborative filtering first (it's the strongest non-LLM behavioral
    # signal), then retrieval, then the cohort graph.
    if len(candidates) < TOP_N:
        cf_ranked = sorted(cf_scores.items(), key=lambda kv: kv[1], reverse=True)
        for program_rule, cf_score in cf_ranked:
            if len(candidates) >= TOP_N:
                break
            if program_rule in used_offers:
                continue
            offer = offers_by_name.get(program_rule)
            if offer is None:
                continue
            used_offers.add(program_rule)
            retrieval_score = retrieval_score_by_offer.get(program_rule, 0.0)
            graph_score = graph_scores.get(program_rule, 0.0)
            candidates.append(_make(
                offer, llm_score=cf_score, cf_score=cf_score, retrieval_score=retrieval_score, graph_score=graph_score,
                why="Backfilled via collaborative-filtering model (LLM output incomplete or invalid).",
            ))

    if len(candidates) < TOP_N:
        retrieval_ranked = sorted(retrieval_score_by_offer.items(), key=lambda kv: kv[1], reverse=True)
        for program_rule, retrieval_score in retrieval_ranked:
            if len(candidates) >= TOP_N:
                break
            if program_rule in used_offers:
                continue
            offer = offers_by_name.get(program_rule)
            if offer is None:
                continue
            used_offers.add(program_rule)
            graph_score = graph_scores.get(program_rule, 0.0)
            candidates.append(_make(
                offer, llm_score=retrieval_score, cf_score=cf_scores.get(program_rule, 0.0),
                retrieval_score=retrieval_score, graph_score=graph_score,
                why="Backfilled via retrieval ranking (LLM and collaborative filtering both incomplete).",
            ))

    if len(candidates) < TOP_N:
        graph_ranked = sorted(graph_scores.items(), key=lambda kv: kv[1], reverse=True)
        for program_rule, graph_score in graph_ranked:
            if len(candidates) >= TOP_N:
                break
            if program_rule in used_offers:
                continue
            offer = offers_by_name.get(program_rule)
            if offer is None:
                continue
            used_offers.add(program_rule)
            retrieval_score = retrieval_score_by_offer.get(program_rule, 0.0)
            candidates.append(_make(
                offer, llm_score=graph_score, cf_score=cf_scores.get(program_rule, 0.0),
                retrieval_score=retrieval_score, graph_score=graph_score,
                why="Backfilled via cohort purchase graph (LLM, CF, and retrieval all incomplete).",
            ))

    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    top = candidates[:TOP_N]
    for i, c in enumerate(top, start=1):
        c["rank"] = i
    return top
