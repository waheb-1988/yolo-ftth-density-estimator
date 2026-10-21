"""AI Copilot: a ReAct agent that answers marketing / CVM questions about
the portfolio ("who should we retain this week?", "why is 15 at risk?")
with portfolio-level tools, reusing the policy-enforced loop in react.py."""
from __future__ import annotations

import dataclasses
import time

from nbx.core import config, insights
from nbx.agents.agentic import council
from nbx.agents.agentic.react import AgentSpec, LLMUnavailable, run_react
from nbx.agents.agentic.tools import TOOLS, RunContext, Tool, ToolError, load_offers


def _limit(args: dict, default: int = 5) -> int:
    try:
        return max(1, min(int(args.get("limit", default)), 10))
    except (TypeError, ValueError):
        return default


def portfolio_summary(ctx: RunContext, args: dict) -> dict:
    s = insights.portfolio_summary()
    return {
        "customers": s["customers"],
        "high_risk_customers": s["high_risk"],
        "medium_risk_customers": s["medium_risk"],
        "average_arpu": _omr(s["avg_arpu"]),
        "average_churn_probability": _pct(s["avg_churn"]),
        "monthly_revenue_at_risk": _omr(s["revenue_at_risk"]),
        "portfolio_revenue_last_3_months": [_omr(r["omr"]) for r in s["monthly_revenue"]],
        "customers_by_value_segment": s["by_value_segment"],
        "customers_by_need": s["by_need"],
        "customers_by_recommended_action": s["by_policy_action"],
        "top_offers": s["top_offers"],
        "note": "Currency is OMR (Omani Rial). Values are pre-formatted; quote them exactly.",
    }


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _omr(x: float) -> str:
    return f"{x:.2f} OMR"


def _fmt_row(r: dict) -> dict:
    """Small models mis-convert units (0.846 -> "8.46%", OMR -> "$"), so
    tools hand them ready-to-quote strings instead of raw numbers."""
    return {
        "msisdn": r["msisdn"],
        "value_segment": r["value_segment"],
        "arpu": _omr(r["arpu"]),
        "churn_probability": _pct(r["churn"]),
        "risk_level": r["risk_level"],
        "need": r["need_signal"],
        "recommended_action": r["policy_action"],
    }


def find_customers(ctx: RunContext, args: dict) -> dict:
    rows = insights.customers_list()
    for key, field in (("risk_level", "risk_level"), ("need", "need_signal"), ("value_segment", "value_segment"),
                       ("policy_action", "policy_action"), ("channel", "preferred_channel")):
        if args.get(key):
            rows = [r for r in rows if str(r[field]).lower() == str(args[key]).lower()]
    sort = args.get("sort_by", "churn")
    if sort not in ("churn", "arpu", "avg_monthly_revenue", "revenue_decrease", "data_appetency"):
        sort = "churn"
    rows.sort(key=lambda r: r[sort], reverse=True)
    return {"matches": len(rows), "customers": [_fmt_row(r) for r in rows[: _limit(args)]],
            "note": "Values are pre-formatted; quote them exactly."}


def customer_snapshot(ctx: RunContext, args: dict) -> dict:
    try:
        msisdn = int(args.get("msisdn"))
    except (TypeError, ValueError):
        raise ToolError('args must include "msisdn": <integer>')
    c = insights.customer_360(msisdn)
    if c is None:
        raise ToolError(f"No customer with msisdn {msisdn}")
    return {
        **_fmt_row(c),
        "device_segment": c["device_segment"],
        "avg_monthly_revenue": _omr(c["avg_monthly_revenue"]),
        "tenure": f"{c['tenure_months']} months",
        "usage_type": c["usage_type"],
        "nba_segment": c["nba_segment"],
        "revenue_decrease_probability": _pct(c["revenue_decrease"]),
        "preferred_channel": c["preferred_channel"],
        "days_since_last_purchase": c["days_since_last_purchase"],
        "days_inactive": c["days_inactive"],
        "revenue_last_3_months": [_omr(r["omr"] or 0) for r in c["revenue_monthly"]],
    }


def policy_preview(ctx: RunContext, args: dict) -> dict:
    """Fast deterministic NBA+NBO preview (no LLM) -- the full agentic run
    is available from the Customer 360 / NBA / NBO pages."""
    try:
        sub = RunContext.load(int(args.get("msisdn")))
    except (TypeError, ValueError, LookupError):
        raise ToolError('args must include a valid "msisdn"')
    sink = lambda ev: None  # noqa: E731
    hist: list[dict] = []
    nba = council.policy_nba(sub, hist, sink)
    nbo = council.policy_nbo(sub, nba["action"], [], sink)
    return {"msisdn": sub.msisdn, "action": nba["action"], "why": nba["rationale"], "channel": nba["channel"],
            "offers": [o["offer"] for o in nbo["offers"]]}


def catalog(ctx: RunContext, args: dict) -> dict:
    offers = list(load_offers().values())
    if args.get("category"):
        offers = [o for o in offers if o["category"].lower() == str(args["category"]).lower()]
    return {"offers": [{"offer": o["program_rule"], "category": o["category"], "price_omr": o["price_omr"],
                        "validity_days": o["validity_days"]} for o in offers[:12]]}


COPILOT_TOOLS = [
    Tool("portfolio_summary", "Portfolio KPIs: customers, risk counts, ARPU, revenue trend, revenue at risk, segment/need/action mix.", "{}", portfolio_summary),
    Tool("find_customers", "Filter and rank customers.", '{"risk_level"?: "HIGH|MEDIUM|LOW", "need"?: str, "value_segment"?: str, "policy_action"?: str, "sort_by"?: "churn|arpu|revenue_decrease|data_appetency", "limit"?: int}', find_customers),
    Tool("customer_snapshot", "Profile, risk scores and revenue of one customer.", '{"msisdn": int}', customer_snapshot),
    Tool("policy_preview", "Quick recommended action and offers for one customer.", '{"msisdn": int}', policy_preview),
    Tool("catalog", "List offers, optionally by category (NBO, Data, Voice, Bundle, International, Roaming).", '{"category"?: str}', catalog),
]
for _t in COPILOT_TOOLS:
    TOOLS[_t.name] = _t


def _validate(final, ctx):
    if isinstance(final, dict) and final.get("answer"):
        return {"answer": str(final["answer"]).strip()}, None
    if isinstance(final, str) and final.strip():
        return {"answer": final.strip()}, None
    return None, 'final must be {"answer": "<markdown answer>"}'


COPILOT_SPEC = AgentSpec(
    key="copilot",
    name="NBX Copilot",
    role=(
        "You are NBX Copilot, an analyst assistant for Ooredoo Oman's customer value management team. "
        "Answer using tool evidence only -- never invent or convert numbers; quote the pre-formatted values exactly "
        "(currency is OMR, never $). Cite msisdns and figures. "
        "Keep answers short (max 6 lines or a short list). Reply with JSON only."
    ),
    goal="",
    tools=[t.name for t in COPILOT_TOOLS],
    required_tools=[],
    final_schema='{"answer": "<short markdown answer citing the evidence>"}',
    validate_final=_validate,
    max_steps=5,
)


def _fallback_answer(question: str) -> str:
    s = insights.portfolio_summary()
    risky = find_customers(None, {"risk_level": "HIGH", "limit": 5})["customers"]
    lines = [
        "_LLM unavailable -- portfolio facts only:_",
        f"- {s['customers']} customers, {s['high_risk']} high-risk, avg ARPU {s['avg_arpu']} OMR",
        f"- Monthly revenue at risk: {s['revenue_at_risk']} OMR",
        "- Highest churn risk: " + ", ".join(f"{c['msisdn']} ({c['churn_probability']})" for c in risky),
    ]
    return "\n".join(lines)


async def ask(question: str, llm: str | None = None, history: list[dict] | None = None) -> dict:
    start = time.time()
    events: list[dict] = []
    emit = events.append
    model = await council._resolve_model(llm)
    convo = ""
    if history:
        convo = " Previous turns: " + " | ".join(f"{h.get('role')}: {str(h.get('content'))[:200]}" for h in history[-4:])
    spec = dataclasses.replace(COPILOT_SPEC, goal=f'Answer the user question: "{question}".{convo}')
    ctx = RunContext(msisdn=0, customer={}, score_row=None, revenue={}, offers=load_offers())
    answer, mode = None, "fallback"
    if model:
        try:
            res = await run_react(spec, ctx, emit, model)
            if res.final:
                answer, mode = res.final["answer"], "agentic"
            else:
                # Budget exhausted: answer from what was gathered.
                answer, mode = _fallback_answer(question), "budget"
        except LLMUnavailable:
            pass
    if answer is None:
        answer = _fallback_answer(question)
    return {
        "answer": answer,
        "mode": mode,
        "model": model or None,
        "trace": events,
        "elapsed_ms": int((time.time() - start) * 1000),
        "default_llm": config.DEFAULT_LLM,
    }
