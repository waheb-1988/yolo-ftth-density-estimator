"""In-process knowledge graph (networkx): lines -> device segment, lines -> BTLs bought, BTL -> service.

The FSDR offer agent's `cohort_offers` tool builds it from the run's own tables (device segments, the 270-day
BTL history, the catalogue) and asks "what did lines of this cohort buy" -- a relational signal that needs
no embeddings.
"""
from __future__ import annotations

import networkx as nx

def _customer_node(msisdn: int) -> str:
    return f"customer:{msisdn}"


def _offer_node(program_rule: str) -> str:
    return f"offer:{program_rule}"


def _segment_node(dv_segment: str) -> str:
    return f"segment:{dv_segment}"


def _category_node(category: str) -> str:
    return f"category:{category}"



def build_graph_from(customers: list[dict], offers: list[dict], purchases: list[dict], segments: list[dict]) -> nx.DiGraph:
    """The same graph from records already in memory (e.g. the FSDR tables): customers {msisdn, dv_segment,
    revenue_segment, usage_type}, offers {program_rule, category, price_omr, offer_name}, purchases {msisdn,
    program_rule, category, purchase_count}, segments {segment, typical_behavior}."""
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


def get_stats(g: nx.DiGraph) -> dict:
    kinds = {}
    for _, data in g.nodes(data=True):
        kinds[data.get("kind", "unknown")] = kinds.get(data.get("kind", "unknown"), 0) + 1
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "by_kind": kinds}


def cohort_offers(customer: dict, graph: nx.DiGraph, top_k: int = 10) -> list[dict]:
    """'Customers in this cohort bought these offers' — aggregates purchases
    of peer customers who share this customer's dv_segment, weighted by how
    many peers (and how often) bought each offer. Returns
    [{"program_rule", "graph_score" (0-1), "peer_count", "reason"}]."""
    g = graph
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
