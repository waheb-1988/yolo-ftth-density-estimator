"""In-process knowledge graph over the customer/offer/purchase/segment data.

Vector search answers "offers that read like this text." This graph answers
the relational question the JSON data actually encodes: "what did customers
in this cohort buy" — a signal vector similarity can't see because segment
membership and purchase history aren't textual, they're edges. Built with
networkx (in-memory, no server), rebuilt whenever the data is reseeded.
"""
from __future__ import annotations

import json
from functools import lru_cache

import networkx as nx

from nbx.core import config


def _load_json(path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _customer_node(msisdn: int) -> str:
    return f"customer:{msisdn}"


def _offer_node(program_rule: str) -> str:
    return f"offer:{program_rule}"


def _segment_node(dv_segment: str) -> str:
    return f"segment:{dv_segment}"


def _category_node(category: str) -> str:
    return f"category:{category}"


def build_graph() -> nx.DiGraph:
    """Builds the full graph fresh from the JSON data files. Call get_graph()
    for the cached singleton — this is exposed separately so seed_data can
    force a rebuild after regenerating the underlying data."""
    customers = _load_json(config.CUSTOMERS_FILE)
    offers = _load_json(config.OFFERS_FILE)
    purchases = _load_json(config.PURCHASES_FILE)
    segments = _load_json(config.SEGMENTS_FILE)

    g = nx.DiGraph()

    for seg in segments:
        g.add_node(_segment_node(seg["segment"]), kind="segment", typical_behavior=seg["typical_behavior"])

    for offer in offers:
        g.add_node(
            _offer_node(offer["program_rule"]),
            kind="offer",
            category=offer["category"],
            price_omr=offer["price_omr"],
            offer_name=offer["offer_name"],
        )
        g.add_node(_category_node(offer["category"]), kind="category")
        g.add_edge(_offer_node(offer["program_rule"]), _category_node(offer["category"]), kind="IN_CATEGORY")

    for c in customers:
        cid = _customer_node(c["msisdn"])
        g.add_node(
            cid,
            kind="customer",
            dv_segment=c["dv_segment"],
            revenue_segment=c["revenue_segment"],
            usage_type=c["usage_type"],
            favorite_category=c.get("favorite_category"),
        )
        seg_node = _segment_node(c["dv_segment"])
        if seg_node in g:
            g.add_edge(cid, seg_node, kind="IN_SEGMENT")

    for p in purchases:
        cid = _customer_node(p["msisdn"])
        oid = _offer_node(p["program_rule"])
        if cid not in g or oid not in g:
            continue
        if g.has_edge(cid, oid):
            g[cid][oid]["weight"] += p.get("purchase_count", 1)
        else:
            g.add_edge(cid, oid, kind="PURCHASED", weight=p.get("purchase_count", 1))

    # Segment -> category affinity, derived from how often each segment's
    # customers purchase each category. Same signal seed_data's pattern
    # documents encode for the vector store, kept here as graph edges too.
    affinity: dict[tuple[str, str], int] = {}
    customer_by_msisdn = {c["msisdn"]: c for c in customers}
    for p in purchases:
        c = customer_by_msisdn.get(p["msisdn"])
        if not c:
            continue
        key = (c["dv_segment"], p["category"])
        affinity[key] = affinity.get(key, 0) + 1
    for (dv_segment, category), count in affinity.items():
        seg_node, cat_node = _segment_node(dv_segment), _category_node(category)
        if seg_node in g and cat_node in g:
            g.add_edge(seg_node, cat_node, kind="AFFINITY", weight=count)

    return g


_graph_cache: nx.DiGraph | None = None


def get_graph(force_rebuild: bool = False) -> nx.DiGraph:
    global _graph_cache
    if force_rebuild or _graph_cache is None:
        _graph_cache = build_graph()
    return _graph_cache


def get_stats(graph: nx.DiGraph | None = None) -> dict:
    g = graph or get_graph()
    kinds = {}
    for _, data in g.nodes(data=True):
        kinds[data.get("kind", "unknown")] = kinds.get(data.get("kind", "unknown"), 0) + 1
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "by_kind": kinds}


def cohort_offers(customer: dict, top_k: int = 10, graph: nx.DiGraph | None = None) -> list[dict]:
    """'Customers in this cohort bought these offers' — aggregates purchases
    of peer customers who share this customer's dv_segment, weighted by how
    many peers (and how often) bought each offer. Returns
    [{"program_rule", "graph_score" (0-1), "peer_count", "reason"}]."""
    g = graph or get_graph()
    seg_node = _segment_node(customer.get("dv_segment", ""))
    if seg_node not in g:
        return []

    self_node = _customer_node(customer.get("msisdn", -1))
    peers = [u for u, _, d in g.in_edges(seg_node, data=True) if d.get("kind") == "IN_SEGMENT" and u != self_node]

    offer_weight: dict[str, float] = {}
    offer_peer_count: dict[str, int] = {}
    for peer in peers:
        for _, offer_node, edata in g.out_edges(peer, data=True):
            if edata.get("kind") != "PURCHASED":
                continue
            offer_weight[offer_node] = offer_weight.get(offer_node, 0.0) + edata.get("weight", 1)
            offer_peer_count[offer_node] = offer_peer_count.get(offer_node, 0) + 1

    if not offer_weight:
        return []

    max_weight = max(offer_weight.values())
    ranked = sorted(offer_weight.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    out = []
    for offer_node, weight in ranked:
        program_rule = offer_node.split(":", 1)[1]
        peer_count = offer_peer_count[offer_node]
        out.append(
            {
                "program_rule": program_rule,
                "graph_score": round(weight / max_weight, 4),
                "peer_count": peer_count,
                "reason": f"Bought by {peer_count} customer(s) in the '{customer.get('dv_segment')}' cohort.",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Graph-only need retrieval -- the relational substitute for vector search.
#
# Vector search answers "what offer reads like this need" by embedding text.
# Nothing in the production telco data (msisdn, item_id, price, revenue) is
# text, so that job is answered here the same way cohort_offers() answers
# "what did this cohort buy": by walking real edges, not comparing embeddings.
#   need -> category           (fixed mapping, mirrors offer.category)
#   category -> IN_CATEGORY -> candidate offers
#   customer -> IN_SEGMENT -> segment -> AFFINITY -> category  (fallback weight)
#   customer -> PURCHASED -> offer                              (primary weight)
# ---------------------------------------------------------------------------
NEED_CATEGORY_MAP = {
    "REACTIVATION": ["NBO", "Bundle"],
    "RENEWAL": ["NBO", "Bundle"],
    "INTERNATIONAL": ["International"],
    "ROAMING": ["Roaming"],
    "DATA_UPSELL": ["Data"],
    "VOICE_UPSELL": ["Voice"],
    "CHANNEL_MIGRATION": ["NBO", "Bundle"],
    "GENERAL_NBO": ["NBO", "Bundle", "Data"],
}


def need_based_offers(
    customer: dict, detected_need: str, top_k: int = 10, graph: nx.DiGraph | None = None
) -> list[dict]:
    """Returns up to top_k offers whose category matches the detected need,
    scored by real peer behavior instead of text similarity:
      - offers this segment's peers actually purchased -> cohort_offers()'
        own weight (direct evidence)
      - offers with no direct peer purchase -> segment->category AFFINITY
        edge weight, damped 0.6x to mark it as the weaker, less specific
        signal it is
    Catalog attributes (price, data_gb, ...) come from offers.json, same as
    every other agent -- the graph supplies the score, not the attributes."""
    g = graph or get_graph()
    categories = NEED_CATEGORY_MAP.get(detected_need, [])
    if not categories:
        return []

    offers = _load_json(config.OFFERS_FILE)
    candidates = [o for o in offers if o.get("category") in categories]
    if not candidates:
        return []

    cohort = {h["program_rule"]: h["graph_score"] for h in cohort_offers(customer, top_k=len(candidates), graph=g)}

    seg_node = _segment_node(customer.get("dv_segment", ""))
    affinity_raw = {
        v.split(":", 1)[1]: d.get("weight", 0)
        for _, v, d in g.out_edges(seg_node, data=True)
        if d.get("kind") == "AFFINITY"
    } if seg_node in g else {}
    max_affinity = max(affinity_raw.values()) if affinity_raw else 0

    scored = []
    for offer in candidates:
        rule = offer["program_rule"]
        if rule in cohort:
            score, basis = cohort[rule], "purchased by peers in this segment"
        else:
            aff = affinity_raw.get(offer["category"], 0)
            score = round(0.6 * (aff / max_affinity), 4) if max_affinity else 0.0
            basis = f"segment affinity for '{offer['category']}' (no direct peer purchases)"
        scored.append({**offer, "graph_retrieval_score": score, "basis": basis})

    scored.sort(key=lambda r: r["graph_retrieval_score"], reverse=True)
    return scored[:top_k]


def similar_customers(msisdn: int, top_k: int = 5, graph: nx.DiGraph | None = None) -> list[dict]:
    """Customers who share this customer's segment, ranked by how many
    purchased offers they have in common with the target customer."""
    g = graph or get_graph()
    cid = _customer_node(msisdn)
    if cid not in g:
        return []

    my_offers = {v for _, v, d in g.out_edges(cid, data=True) if d.get("kind") == "PURCHASED"}
    seg_nodes = [v for _, v, d in g.out_edges(cid, data=True) if d.get("kind") == "IN_SEGMENT"]
    if not seg_nodes:
        return []
    seg_node = seg_nodes[0]

    peers = [u for u, _, d in g.in_edges(seg_node, data=True) if d.get("kind") == "IN_SEGMENT" and u != cid]

    scored = []
    for peer in peers:
        peer_offers = {v for _, v, d in g.out_edges(peer, data=True) if d.get("kind") == "PURCHASED"}
        overlap = len(my_offers & peer_offers)
        scored.append({"msisdn": int(peer.split(":", 1)[1]), "shared_segment": g.nodes[seg_node].get("kind") and seg_node.split(":", 1)[1], "shared_offer_count": overlap})

    scored.sort(key=lambda r: r["shared_offer_count"], reverse=True)
    return scored[:top_k]
