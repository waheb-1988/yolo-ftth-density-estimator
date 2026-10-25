"""Tools the agentic NBA/NBO agents can call.

Every tool wraps an existing, already-tested signal in this repo -- the VW
contextual bandit (core/nba_bandit.py), the cohort graph
(core/knowledge_graph.py), BPR collaborative filtering
(core/collab_filter.py), the production business rules
(agents/rules_engine.py) -- or a new, explicitly heuristic one
(simulate_expected_value, contact_policy). Nothing here calls an LLM: the
agents decide WHICH evidence to gather and how to weigh it; the tools only
report facts.

Observations are deliberately compact (rounded, few keys): they go straight
into a small local model's context window on every following step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from nbx.agents.agentic import languages
from nbx.core import collab_filter, config, knowledge_graph, nba_bandit
from nbx.core.data_simulator import detect_need

# ---------------------------------------------------------------------------
# Data access (small JSON files; re-read so uploads are picked up live)
# ---------------------------------------------------------------------------


def _load(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_customer(msisdn: int) -> dict | None:
    return next((c for c in _load(config.CUSTOMERS_FILE) if c["msisdn"] == msisdn), None)


def load_offers() -> dict[str, dict]:
    return {o["program_rule"]: o for o in _load(config.OFFERS_FILE)}


def load_revenue(msisdn: int) -> dict:
    return next((r for r in _load(config.REVENUE_HISTORY_FILE) if r["msisdn"] == msisdn), {})


class ToolError(Exception):
    """Raised for bad arguments; the message goes back to the agent as an
    observation so it can correct itself instead of the run failing."""


@dataclass
class RunContext:
    """Per-run state shared by every tool call of every agent in one run."""

    msisdn: int
    customer: dict
    score_row: dict | None
    revenue: dict
    offers: dict[str, dict]
    # Set by the orchestrator as agents finish -- tools read it to tailor
    # their answers (e.g. eligibility depends on the chosen action).
    action: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, msisdn: int) -> "RunContext":
        customer = load_customer(msisdn)
        if customer is None:
            raise LookupError(f"Unknown customer msisdn={msisdn}")
        return cls(
            msisdn=msisdn,
            customer=customer,
            score_row=nba_bandit.get_score_row(msisdn),
            revenue=load_revenue(msisdn),
            offers=load_offers(),
        )

    @property
    def avg_rev_monthly(self) -> float:
        return float(self.revenue.get("avg_rev_monthly") or self.customer.get("avg_offer_price", 5.0))


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def _resolve_offer(ctx: RunContext, name: str) -> dict | None:
    """Exact program_rule first, then case-insensitive -- small models often
    change the case or drop a suffix of long offer codes."""
    if name in ctx.offers:
        return ctx.offers[name]
    low = str(name).strip().lower()
    return next((o for k, o in ctx.offers.items() if k.lower() == low), None)


def _offer_list_arg(args: dict) -> list[str]:
    raw = args.get("offers", args.get("offer_names", args.get("offer")))
    if isinstance(raw, str):
        raw = [s.strip() for s in raw.split(",") if s.strip()]
    if not isinstance(raw, list) or not raw:
        raise ToolError('args must include "offers": a list of offer program_rule codes, e.g. ["DT_7D_4R_5GB"]')
    return [str(x) for x in raw][:6]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def get_customer_profile(ctx: RunContext, args: dict) -> dict:
    c = ctx.customer
    return {
        "msisdn": c["msisdn"],
        "tenure_days": c["tenure_days"],
        "arpu_omr": c["arpu"],
        "avg_monthly_revenue_omr": _r(ctx.avg_rev_monthly, 2),
        "device_segment": c["dv_segment"],
        "value_segment": c["revenue_segment"],
        "usage_type": c["usage_type"],
        "data_tier": c["data_usage_tier"],
        "activity_level": c["activity_level"],
        "international_caller": c["is_intl_caller"],
        "roamer": c["is_roamer"],
        "days_since_last_purchase": c["days_since_last_purchase"],
        "days_since_last_network_activity": c["days_since_last_network_activity"],
        "preferred_channel": c["preferred_recharge_channel"],
        "favorite_category": c["favorite_category"],
        "last_product": c["last_product_name"],
        "need_signal": detect_need(c),
    }


def risk_level(row: dict | None) -> str:
    if not row:
        return "UNKNOWN"
    churn, rev = row.get("proba_churn", 0), row.get("proba_rev_decr", 0)
    if churn >= 0.5 or row.get("proba_churn_1d", 0) >= 0.5:
        return "HIGH"
    if churn >= 0.3 or rev >= 0.6:
        return "MEDIUM"
    return "LOW"


def get_risk_scores(ctx: RunContext, args: dict) -> dict:
    row = ctx.score_row
    if row is None:
        return {"available": False, "note": "No propensity scores for this customer."}
    return {
        "nba_segment": row["nba_segment"],
        "excluded_from_nba": row["excluded"],
        "risk_level": risk_level(row),
        "churn": _r(row["proba_churn"]),
        "churn_next_day": _r(row["proba_churn_1d"]),
        "reactivation_30d": _r(row["proba_react_30"]),
        "data_appetency": _r(row["proba_app_data"]),
        "voice_appetency": _r(row["proba_app_voice"]),
        "data_fragility": _r(row["proba_frag_data"]),
        "voice_fragility": _r(row["proba_frag_voice"]),
        "revenue_decrease": _r(row["proba_rev_decr"]),
    }


def _trend(pct: float) -> str:
    return "rising" if pct > 10 else "falling" if pct < -10 else "stable"


def get_usage_trends(ctx: RunContext, args: dict) -> dict:
    rev, row = ctx.revenue, ctx.score_row or {}
    rev_mom = row.get("var_rev_m1_m2", 0.0)
    data_mom = row.get("var_data_m1_m2", 0.0)
    voice_mom = row.get("var_voice_m1_m2", 0.0)
    return {
        "revenue_last_3_months_omr": [rev.get("rev_m3"), rev.get("rev_m2"), rev.get("rev_m1")],
        "revenue_last_4_weeks_omr": [rev.get("rev_w4"), rev.get("rev_w3"), rev.get("rev_w2"), rev.get("rev_w1")],
        "revenue_change_month_pct": _r(rev_mom, 1),
        "data_usage_change_month_pct": _r(data_mom, 1),
        "voice_usage_change_month_pct": _r(voice_mom, 1),
        "revenue_trend": _trend(rev_mom),
        "data_trend": _trend(data_mom),
        "voice_trend": _trend(voice_mom),
        "data_share_of_revenue": _r(rev.get("avg_rev_data_monthly", 0) / max(ctx.avg_rev_monthly, 1e-9), 2),
    }


def bandit_recommendation(ctx: RunContext, args: dict) -> dict:
    row = ctx.score_row
    if row is None:
        return {"available": False, "note": "No bandit context for this customer."}
    if row["excluded"]:
        return {"excluded": True, "note": f"Segment {row['nba_segment']} is excluded from NBA campaigns."}
    if "bandit" not in ctx.cache:
        ctx_feats = {f: row.get(f, 0) for f in config.NBA_CONTEXT_FEATURES}
        ctx.cache["bandit"] = nba_bandit.get_action(ctx_feats)
    action, prob = ctx.cache["bandit"]
    return {
        "bandit_action": action,
        "bandit_probability": _r(prob),
        "segment_policy_action": config.NBA_SEGMENT_ACTION_MAP.get(row["nba_segment"]),
        "historical_action": row.get("action"),
        "population_base_rate": _r(config.NBA_ACTION_PROPENSITY.get(action, 0)),
        "note": "Vowpal Wabbit contextual bandit (epsilon-greedy, 20% exploration).",
    }


_CHANNEL_MAP = {
    "APP": "APP",
    "WEBSITE": "APP",
    "USSD": "SMS",
    "RETAIL_STORE": "SMS",
    "CALL_CENTER": "OUTBOUND_CALL",
}
_SEND_WINDOW = {
    "HIGH_ACTIVITY": "12:00-14:00 or 19:00-21:00",
    "MEDIUM_ACTIVITY": "18:00-21:00",
    "LOW_ACTIVITY": "19:00-21:00 (weekend preferred)",
}


def contact_policy(ctx: RunContext, args: dict) -> dict:
    """Contact-fatigue and channel policy. Heuristic stand-in for a real
    contact-history table, which this dataset doesn't have."""
    c = ctx.customer
    recent_purchase = c["days_since_last_purchase"] <= 3
    dormant = c["days_since_last_network_activity"] >= 60
    channel = _CHANNEL_MAP.get(c["preferred_recharge_channel"], "SMS")
    if dormant:
        channel = "OUTBOUND_CALL" if c["revenue_segment"] in ("High Value", "Premium") else "SMS"
    return {
        "push_offer_allowed": not recent_purchase,
        "cooldown_reason": "Purchased within the last 3 days -- avoid a sales push." if recent_purchase else None,
        "dormant": dormant,
        "recommended_channel": channel,
        "send_window_local": _SEND_WINDOW.get(c["activity_level"], "18:00-21:00"),
        "quiet_hours": "22:00-08:00 and Friday 11:30-13:30",
        "nationality_segment": languages.plan(c.get("nationality"))["segment"],
        "message_languages": languages.plan(c.get("nationality"))["languages"],
        "lead_language": languages.plan(c.get("nationality"))["lead_language"],
    }


def peer_insights(ctx: RunContext, args: dict) -> dict:
    peers = knowledge_graph.similar_customers(ctx.msisdn, top_k=3)
    cohort = knowledge_graph.cohort_offers(ctx.customer, top_k=5)
    return {
        "segment": ctx.customer["dv_segment"],
        "similar_customers": [{"msisdn": p["msisdn"], "shared_offers": p["shared_offer_count"]} for p in peers],
        "cohort_top_offers": [
            {"offer": h["program_rule"], "peer_buyers": h["peer_count"], "graph_score": _r(h["graph_score"], 2)}
            for h in cohort
        ],
    }


def cf_recommendations(ctx: RunContext, args: dict) -> dict:
    hits = collab_filter.score_customer(ctx.customer, top_k=5)
    return {
        "basis": hits[0]["basis"] if hits else "none",
        "offers": [{"offer": h["program_rule"], "cf_score": _r(h["cf_score"], 2)} for h in hits],
    }


def _offer_brief(o: dict) -> dict:
    return {
        "offer": o["program_rule"],
        "category": o["category"],
        "price_omr": o["price_omr"],
        "validity_days": o["validity_days"],
        "data_gb": o["data_gb"],
        "voice_mins": o["voice_mins"],
    }


def search_offers(ctx: RunContext, args: dict) -> dict:
    raw = args.get("category") or args.get("categories")
    max_price = args.get("max_price")
    min_price = args.get("min_price")
    cats = {o["category"] for o in ctx.offers.values()}
    # Small models pass "Data", ["NBO","Bundle"] or "NBO, Bundle" -- accept all.
    wanted = raw if isinstance(raw, list) else [s.strip() for s in str(raw).split(",")] if raw else []
    chosen = set()
    for w in wanted:
        match = next((c for c in cats if c.lower() == str(w).lower()), None)
        if match is None:
            raise ToolError(f"Unknown category '{w}'. Valid: {sorted(cats)}")
        chosen.add(match)
    out = []
    for o in ctx.offers.values():
        if chosen and o["category"] not in chosen:
            continue
        if max_price is not None and o["price_omr"] > float(max_price):
            continue
        if min_price is not None and o["price_omr"] < float(min_price):
            continue
        out.append(o)
    # Closest to the customer's affordability sweet spot first.
    target = ctx.avg_rev_monthly + config.AFFORDABILITY_BAND_OMR / 2
    out.sort(key=lambda o: abs(_monthly_equiv(o) - target))
    return {"count": len(out), "offers": [_offer_brief(o) for o in out[:8]], "categories": sorted(cats)}


def _monthly_equiv(o: dict) -> float:
    return o["price_omr"] * 30 / max(o["validity_days"], 1)


def _affordability_tier(ctx: RunContext, o: dict) -> int:
    avg, band, price = ctx.avg_rev_monthly, config.AFFORDABILITY_BAND_OMR, o["price_omr"]
    if avg < price <= avg + band:
        return 0
    if price > avg:
        return 1
    return 2


def eligibility(ctx: RunContext, name: str) -> dict:
    o = _resolve_offer(ctx, name)
    if o is None:
        return {"offer": name, "eligible": False, "reasons": ["not in the catalog"]}
    reasons = []
    if o["category"] == "Voice":
        reasons.append("voice bundles are excluded by business rule (very low take-up)")
    if o["price_omr"] > 2.5 * max(ctx.avg_rev_monthly, 1.0):
        reasons.append(f"price {o['price_omr']} OMR is > 2.5x average monthly spend ({ctx.avg_rev_monthly:.1f} OMR)")
    if o["category"] == "Roaming" and not ctx.customer["is_roamer"]:
        reasons.append("roaming pack for a customer with no roaming activity")
    if o["category"] == "International" and not ctx.customer["is_intl_caller"]:
        reasons.append("international pack for a customer with no international calls")
    tier = _affordability_tier(ctx, o)
    return {
        "offer": o["program_rule"],
        "eligible": not reasons,
        "reasons": reasons,
        "affordability": ["sweet-spot upsell", "upsell", "at/below current spend"][tier],
        "already_owned": o["program_rule"] == ctx.customer["last_product_name"],
    }


def check_eligibility(ctx: RunContext, args: dict) -> dict:
    results = [eligibility(ctx, n) for n in _offer_list_arg(args)]
    ctx.cache.setdefault("checked", {}).update({r["offer"]: r for r in results})
    return {"results": results}


def expected_value(ctx: RunContext, name: str) -> dict | None:
    """Heuristic uplift simulator -- NOT a trained uplift model. Assumptions
    (documented so nobody mistakes the numbers for a forecast):
      p_accept = 0.12 base take-rate
                 x (1 + cf_score) x (1 + graph_score)       behavioral fit
                 x {1.3 sweet-spot, 0.9 upsell, 0.6 downsell} affordability
                 x (1 - 0.5 x churn)                         at-risk dampener
                 capped at 0.85
      EV (OMR/month) = p_accept x monthly-equivalent price
                     + p_accept x churn x avg_monthly_rev x 0.25
                       (an accepted offer assumed to cut churn by 25%)."""
    o = _resolve_offer(ctx, name)
    if o is None:
        return None
    if "cf_map" not in ctx.cache:
        ctx.cache["cf_map"] = {h["program_rule"]: h["cf_score"] for h in collab_filter.score_customer(ctx.customer, top_k=50)}
        ctx.cache["graph_map"] = {h["program_rule"]: h["graph_score"] for h in knowledge_graph.cohort_offers(ctx.customer, top_k=50)}
    cf = ctx.cache["cf_map"].get(o["program_rule"], 0.0)
    graph = ctx.cache["graph_map"].get(o["program_rule"], 0.0)
    churn = (ctx.score_row or {}).get("proba_churn", 0.0)
    tier = _affordability_tier(ctx, o)
    p = 0.12 * (1 + cf) * (1 + graph) * (1.3, 0.9, 0.6)[tier] * (1 - 0.5 * churn)
    p = min(p, 0.85)
    monthly = _monthly_equiv(o)
    retention = p * churn * ctx.avg_rev_monthly * 0.25
    return {
        "offer": o["program_rule"],
        "p_accept": _r(p),
        "monthly_equiv_omr": _r(monthly, 2),
        "expected_revenue_omr": _r(p * monthly, 2),
        "retention_value_omr": _r(retention, 2),
        "expected_value_omr": _r(p * monthly + retention, 2),
    }


def simulate_expected_value(ctx: RunContext, args: dict) -> dict:
    sims = [s for s in (expected_value(ctx, n) for n in _offer_list_arg(args)) if s]
    sims.sort(key=lambda s: s["expected_value_omr"], reverse=True)
    ctx.cache.setdefault("simulated", {}).update({s["offer"]: s for s in sims})
    return {"ranked": sims, "model": "heuristic uplift simulator (see tool docs)"}


def market_overview(ctx: RunContext, limit: int = 4) -> dict:
    """Mass-market competitor plans priced around what this customer already
    spends -- the alternatives a switching customer would actually compare."""
    from nbx.core import competitors

    spend = ctx.avg_rev_monthly
    plans = [c for c in competitors.load_offers() if c.get("segment", "mass") == "mass" and c.get("monthly_equiv_omr")
             and abs(c["monthly_equiv_omr"] - spend) <= max(3.0, spend * 0.3)]
    plans.sort(key=lambda c: (c.get("data_gb") or 0) / max(c["monthly_equiv_omr"], 0.01), reverse=True)
    return {
        "customer_monthly_spend_omr": round(spend, 2),
        "competitor_plans_at_this_spend": [
            {"operator": c["operator"], "plan": c["plan"], "monthly_equiv_omr": c["monthly_equiv_omr"],
             "data_gb": c.get("data_gb"), "minutes": c.get("minutes")} for c in plans[:limit]
        ],
        "count": len(plans),
    }


def competitor_benchmark(ctx: RunContext, args: dict) -> dict:
    from nbx.core import competitors

    if not competitors.load_offers():
        return {"available": False, "note": "No competitor snapshot yet (run: python -m nbx.scripts.scrape_competitors)."}
    raw = args.get("offers", args.get("offer_names"))
    if not raw:
        overview = market_overview(ctx)
        ctx.cache["market"] = overview
        return overview
    results = []
    for name in _offer_list_arg(args)[:4]:
        o = _resolve_offer(ctx, name)
        if o is None:
            results.append({"offer": name, "error": "not in the catalog"})
            continue
        cmp = competitors.compare_offer(o)
        ctx.cache.setdefault("competition", {})[o["program_rule"]] = cmp
        results.append({
            "offer": o["program_rule"],
            "position": cmp["position"],
            "our_monthly_equiv_omr": cmp["our_monthly_equiv_omr"],
            "closest": [f"{c['operator']} {c['plan']}: {c['monthly_equiv_omr']} OMR/month, "
                        f"{c['data_gb'] if c['data_gb'] is not None else 'unlimited/n.a.'} GB -> ours is {c['verdict']}"
                        for c in cmp["closest"]],
        })
    return {"results": results, "snapshot": "public competitor prices, see source_url in the competitor store"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: str  # human/LLM-readable signature, e.g. '{"category"?: str}'
    fn: Callable[[RunContext, dict], dict]


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("get_customer_profile", "Tenure, ARPU, segments, usage type, channel, recency and the rule-based need signal.", "{}", get_customer_profile),
        Tool("get_risk_scores", "Churn, reactivation, appetency, fragility and revenue-decrease propensities + NBA segment.", "{}", get_risk_scores),
        Tool("get_usage_trends", "Revenue by month/week and month-on-month change of revenue, data and voice usage.", "{}", get_usage_trends),
        Tool("bandit_recommendation", "Contextual-bandit (Vowpal Wabbit) action pick, its probability, and the segment policy action.", "{}", bandit_recommendation),
        Tool("contact_policy", "Contact fatigue/cooldown, recommended channel, send window, nationality segment and message languages.", "{}", contact_policy),
        Tool("peer_insights", "Similar customers from the cohort graph and what their cohort buys.", "{}", peer_insights),
        Tool("cf_recommendations", "Top offers from the collaborative-filtering (BPR) model for this customer.", "{}", cf_recommendations),
        Tool("search_offers", "Search the offer catalog. Categories: NBO, Data, Voice, Bundle, International, Roaming.", '{"category"?: str | [str], "max_price"?: number, "min_price"?: number}', search_offers),
        Tool("check_eligibility", "Business-rule eligibility and affordability for specific offers.", '{"offers": [offer codes]}', check_eligibility),
        Tool("simulate_expected_value", "Simulated acceptance probability and expected value (OMR/month) for specific offers.", '{"offers": [offer codes]}', simulate_expected_value),
        Tool("competitor_benchmark", "Compare our offers with COMPETITOR_A and COMPETITOR_B plans (scraped prices), or without offers: competitor plans at this customer's spend level.", '{"offers"?: [offer codes]}', competitor_benchmark),
    ]
}


def call_tool(ctx: RunContext, name: str, args: dict, registry: dict | None = None) -> dict:
    """registry: tool registry to resolve `name` in (default: TOOLS); the FSDR agents pass their own."""
    tool = (registry or TOOLS)[name]
    try:
        return tool.fn(ctx, args or {})
    except ToolError as e:
        return {"error": str(e)}
    except (TypeError, ValueError) as e:
        return {"error": f"bad arguments for {name}: {e}. Expected {tool.args}"}
