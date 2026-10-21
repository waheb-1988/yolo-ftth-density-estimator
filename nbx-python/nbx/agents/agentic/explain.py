"""Decision explanations: WHY this action, WHY these offers.

Every statement here is computed from evidence (scores, trends, bandit,
contact policy, eligibility, expected value, competitor prices) -- never
generated -- so the explanation is auditable line by line. The LLM only
turns these facts into a short business narrative (agents/agentic/council.py
`run_messenger`), and a template stands in when it can't.
"""
from __future__ import annotations

from nbx.agents.agentic import languages
from nbx.core import competitors, config
from nbx.agents.agentic.tools import (
    RunContext,
    bandit_recommendation,
    contact_policy,
    eligibility,
    expected_value,
    get_usage_trends,
    market_overview,
    risk_level,
)

ACTION_GOALS = {
    "RETAIN": "keep a customer who shows churn signals",
    "MAINTAIN": "protect the relationship without a sales push",
    "UPSELL": "grow spend with a bigger pack",
    "UPSELL DATA": "grow data spend",
    "UPSELL VOICE": "grow voice spend",
    "CROSS SELL DATA": "add data to a voice-led customer",
    "CROSS SELL VOICE": "add voice to a data-led customer",
    "NO_ACTION": "leave the customer alone",
}


def _pct(x) -> str:
    return f"{round((x or 0) * 100)}%"


def action_drivers(ctx: RunContext) -> list[dict]:
    """Signals ranked by how strongly they push the decision."""
    row = ctx.score_row or {}
    trends = get_usage_trends(ctx, {})
    c = ctx.customer
    drivers = []

    def add(signal, value, effect, weight):
        drivers.append({"signal": signal, "value": value, "effect": effect, "weight": weight})

    churn = row.get("proba_churn", 0)
    add("Churn propensity", _pct(churn), "pushes toward retention" if churn >= 0.5 else "low, leaves room to sell" if churn < 0.3 else "moderate, sell carefully", abs(churn - 0.4) * 2)
    rev = row.get("proba_rev_decr", 0)
    if rev >= 0.4:
        add("Revenue-decrease propensity", _pct(rev), "spend likely to drop next month", rev)
    add("Revenue trend (month on month)", f"{trends['revenue_change_month_pct']:+.0f}%", f"revenue is {trends['revenue_trend']}", min(abs(trends["revenue_change_month_pct"]) / 50, 1))
    if trends["data_trend"] != "stable":
        add("Data usage trend", f"{trends['data_usage_change_month_pct']:+.0f}%", f"data use is {trends['data_trend']}", min(abs(trends["data_usage_change_month_pct"]) / 60, 1))
    if trends["voice_trend"] != "stable":
        add("Voice usage trend", f"{trends['voice_usage_change_month_pct']:+.0f}%", f"voice use is {trends['voice_trend']}", min(abs(trends["voice_usage_change_month_pct"]) / 60, 1))
    for key, label in (("proba_app_data", "Data appetency"), ("proba_app_voice", "Voice appetency")):
        if row.get(key, 0) >= 0.4:
            add(label, _pct(row[key]), "ready for more of this service", row[key])
    if c["days_since_last_network_activity"] >= 30:
        add("Inactivity", f"{c['days_since_last_network_activity']} days", "dormant line, reactivate first", 0.8)
    if c["days_since_last_purchase"] <= 3:
        add("Recent purchase", f"{c['days_since_last_purchase']} days ago", "contact cooldown, no push", 0.9)
    add("Tenure / value", f"{c['tenure_days'] // 30} months, {c['revenue_segment']}", "worth protecting" if c["revenue_segment"] in ("High Value", "Premium") else "standard value", 0.3)
    drivers.sort(key=lambda d: d["weight"], reverse=True)
    return drivers[:6]


def explain_action(ctx: RunContext, nba: dict) -> dict:
    action = nba["action"]
    bandit = bandit_recommendation(ctx, {})
    contact = contact_policy(ctx, {})
    drivers = action_drivers(ctx)
    row = ctx.score_row or {}

    if bandit.get("excluded"):
        prior = f"Segment {row.get('nba_segment')} is excluded from NBA campaigns, so no action is taken."
    elif bandit.get("bandit_action") == action:
        prior = f"The contextual bandit also proposed {action} (p = {bandit.get('bandit_probability')}): model and evidence agree."
    elif bandit.get("bandit_action"):
        prior = (f"The contextual bandit proposed {bandit['bandit_action']} (p = {bandit.get('bandit_probability')}), "
                 f"but the council chose {action} because {drivers[0]['signal'].lower()} is {drivers[0]['value']} ({drivers[0]['effect']}).")
    else:
        prior = "No bandit context for this customer; the decision rests on the propensity evidence."

    rejected = []
    if action != "RETAIN" and row.get("proba_churn", 0) >= 0.5:
        rejected.append("RETAIN was considered but churn is not the dominant risk after review.")
    if action in ("RETAIN", "MAINTAIN") and bandit.get("bandit_action") not in (None, action):
        rejected.append(f"{bandit['bandit_action']} rejected: a sales push now could accelerate churn or break contact policy.")
    if action not in ("RETAIN", "MAINTAIN", "NO_ACTION") and risk_level(row) == "LOW":
        rejected.append("RETAIN not needed: churn risk is low.")

    market = market_overview(ctx, limit=3)
    pressure = None
    if market["competitor_plans_at_this_spend"]:
        best = market["competitor_plans_at_this_spend"][0]
        pressure = (f"{market['count']} competitor plans sit at this customer's spend (~{market['customer_monthly_spend_omr']} OMR/month); "
                    f"the richest is {best['operator']} {best['plan']} with {best['data_gb'] or 'unlimited'} GB for "
                    f"{best['monthly_equiv_omr']} OMR/month.")

    return {
        "headline": f"{action}: {ACTION_GOALS.get(action, 'best next step')}.",
        "drivers": drivers,
        "bandit_comparison": prior,
        "alternatives_rejected": rejected,
        "channel_and_timing": (f"{nba.get('channel') or contact.get('recommended_channel')} because the customer prefers "
                               f"{ctx.customer['preferred_recharge_channel'].replace('_', ' ').lower()}; send "
                               f"{nba.get('send_window') or contact.get('send_window_local')}, outside quiet hours "
                               f"({contact.get('quiet_hours')})."),
        "competitive_pressure": pressure,
        "market": market,
        "language": languages.plan(ctx.customer.get("nationality"))["reason"],
    }


def explain_offer(ctx: RunContext, code: str, action: str) -> dict:
    o = ctx.offers[code]
    elig = eligibility(ctx, code)
    ev = ctx.cache.get("simulated", {}).get(code) or expected_value(ctx, code) or {}
    cf = ctx.cache.get("cf_map", {}).get(code, 0.0)
    graph = ctx.cache.get("graph_map", {}).get(code, 0.0)
    cmp = ctx.cache.get("competition", {}).get(code) or competitors.compare_offer(o)
    spend = ctx.avg_rev_monthly

    reasons = []
    cats = {"RETAIN": "retention-friendly value packs", "MAINTAIN": "low-pressure value packs"}.get(action, "the chosen action")
    reasons.append({"kind": "fit", "text": f"{o['category']} pack serving {cats}; customer need signal is "
                                          f"{_need(ctx).replace('_', ' ').lower()} and favourite category is {ctx.customer['favorite_category']}."})
    if cf > 0 or graph > 0:
        reasons.append({"kind": "behaviour", "text": f"Collaborative filtering score {cf:.2f} and cohort-graph score {graph:.2f}: "
                                                     f"customers like this one buy this pack."})
    reasons.append({"kind": "affordability", "text": f"{o['price_omr']} OMR for {o['validity_days']} days vs ~{spend:.1f} OMR average monthly spend "
                                                     f"({elig['affordability']})."})
    if ev:
        reasons.append({"kind": "economics", "text": f"Simulated acceptance {_pct(ev.get('p_accept'))}, expected value "
                                                     f"{ev.get('expected_value_omr')} OMR/month (revenue {ev.get('expected_revenue_omr')} + retention "
                                                     f"{ev.get('retention_value_omr')})."})
    comp_lines = [f"{c['operator']} {c['plan']} ({c['monthly_equiv_omr']} OMR/month, "
                  f"{c['data_gb'] if c['data_gb'] is not None else 'unlimited'} GB): ours is {c['verdict']}" for c in cmp["closest"]]
    reasons.append({"kind": "competition", "text": f"Market position {cmp['position']}." + (" " + "; ".join(comp_lines) + "." if comp_lines else "")})

    watch = []
    if "weak" in cmp["position"]:
        watch.append("A competitor gives more data for less: lead with service quality or add a loyalty bonus.")
    if elig["affordability"] == "at/below current spend":
        watch.append("Priced at or below current spend: protects the relationship but does not grow ARPU.")
    if elig["already_owned"]:
        watch.append("Customer already holds this pack: position it as a renewal.")
    return {
        "offer": code,
        "headline": f"{code}: {o['description']}",
        "reasons": reasons,
        "competition": cmp,
        "watch_outs": watch,
    }


def _need(ctx: RunContext) -> str:
    from nbx.core.data_simulator import detect_need

    return detect_need(ctx.customer)


def build(ctx: RunContext, nba: dict, offers: list[dict]) -> dict:
    return {
        "action": explain_action(ctx, nba),
        "offers": [explain_offer(ctx, o["offer"], nba["action"]) for o in offers],
        "data_sources": [
            "propensity scores (churn, appetency, fragility, revenue decrease)",
            "revenue and usage history",
            "Vowpal Wabbit contextual bandit",
            "BPR collaborative filtering + cohort graph",
            "business rules (eligibility, affordability)",
            f"competitor price snapshot ({len(competitors.load_offers())} Omantel / Vodafone Oman plans)",
        ],
    }


def template_narrative(expl: dict) -> str:
    a = expl["action"]
    top = a["drivers"][0] if a["drivers"] else None
    parts = [a["headline"]]
    if top:
        parts.append(f"The strongest signal is {top['signal'].lower()} at {top['value']} ({top['effect']}).")
    parts.append(a["bandit_comparison"])
    if expl["offers"]:
        o = expl["offers"][0]
        econ = next((r["text"] for r in o["reasons"] if r["kind"] == "economics"), "")
        comp = next((r["text"] for r in o["reasons"] if r["kind"] == "competition"), "")
        parts.append(f"Lead offer {o['offer']}. {econ} {comp}")
    if a.get("competitive_pressure"):
        parts.append(a["competitive_pressure"])
    return " ".join(p.strip() for p in parts if p)


def narrative_prompt(expl: dict) -> str:
    a = expl["action"]
    lines = [f"DECISION: {a['headline']}", f"BANDIT: {a['bandit_comparison']}", "DRIVERS:"]
    lines += [f"- {d['signal']}: {d['value']} ({d['effect']})" for d in a["drivers"]]
    if a.get("competitive_pressure"):
        lines.append(f"MARKET: {a['competitive_pressure']}")
    for o in expl["offers"][:2]:
        lines.append(f"OFFER {o['offer']}:")
        lines += [f"- {r['kind']}: {r['text']}" for r in o["reasons"]]
    if expl["offers"]:
        ask = ("explaining WHY this action and WHY the lead offer, citing the numbers above exactly (OMR, %). "
               "Mention how the lead offer compares with the competitor plans.")
    else:
        ask = ("explaining WHY this action, citing the numbers above exactly (OMR, %). There is no offer in this "
               "decision: do not mention offers; use the MARKET line only as competitive pressure.")
    lines.append(f"Write 4-6 sentences for a CVM manager {ask} No new facts.\n" + 'Reply: {"narrative": "<the explanation>"}')
    return "\n".join(lines)
