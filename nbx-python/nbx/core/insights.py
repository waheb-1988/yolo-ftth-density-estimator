"""Portfolio / customer read-models for the Ooredoo UI and the copilot.

Everything is computed from the same JSON data the agents use -- no mock
numbers. Monetary values are OMR.
"""
from __future__ import annotations

import collections
import json

from nbx.core import config, nba_bandit
from nbx.core.data_simulator import detect_need
from nbx.agents.agentic import languages
from nbx.agents.agentic.tools import risk_level


def _load(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _rows():
    customers = _load(config.CUSTOMERS_FILE)
    scores = {r["msisdn"]: r for r in nba_bandit._load_scores()}
    revenue = {r["msisdn"]: r for r in _load(config.REVENUE_HISTORY_FILE)}
    return customers, scores, revenue


def customer_row(c: dict, s: dict | None, rev: dict) -> dict:
    s = s or {}
    return {
        "msisdn": c["msisdn"],
        "device_segment": c["dv_segment"],
        "value_segment": c["revenue_segment"],
        "arpu": c["arpu"],
        "avg_monthly_revenue": rev.get("avg_rev_monthly", 0.0),
        "tenure_months": c["tenure_days"] // 30,
        "usage_type": c["usage_type"],
        "activity_level": c["activity_level"],
        "preferred_channel": c["preferred_recharge_channel"],
        "favorite_category": c["favorite_category"],
        "nationality": c["nationality"],
        "nationality_segment": languages.plan(c["nationality"])["segment"],
        "message_languages": languages.plan(c["nationality"])["languages"],
        "lead_language": languages.plan(c["nationality"])["lead_language"],
        "network": c["nwk_type"],
        "need_signal": detect_need(c),
        "nba_segment": s.get("nba_segment"),
        "churn": round(s.get("proba_churn", 0.0), 3),
        "revenue_decrease": round(s.get("proba_rev_decr", 0.0), 3),
        "data_appetency": round(s.get("proba_app_data", 0.0), 3),
        "risk_level": risk_level(s or None),
        "policy_action": config.NBA_SEGMENT_ACTION_MAP.get(s.get("nba_segment", ""), "NO_ACTION"),
        "days_since_last_purchase": c["days_since_last_purchase"],
        "days_inactive": c["days_since_last_network_activity"],
    }


def customers_list() -> list[dict]:
    customers, scores, revenue = _rows()
    return [customer_row(c, scores.get(c["msisdn"]), revenue.get(c["msisdn"], {})) for c in customers]


def customer_360(msisdn: int) -> dict | None:
    customers, scores, revenue = _rows()
    c = next((x for x in customers if x["msisdn"] == msisdn), None)
    if c is None:
        return None
    s, rev = scores.get(msisdn) or {}, revenue.get(msisdn, {})
    purchases = [p for p in _load(config.PURCHASES_FILE) if p["msisdn"] == msisdn]
    purchases.sort(key=lambda p: p["purchase_date"], reverse=True)
    return {
        **customer_row(c, s or None, rev),
        "profile": c,
        "scores": {k: round(v, 3) for k, v in s.items() if k.startswith("proba_")},
        "revenue_monthly": [{"period": p, "omr": rev.get(f"rev_{p}")} for p in ("m3", "m2", "m1")],
        "revenue_weekly": [{"period": p, "omr": rev.get(f"rev_{p}")} for p in ("w4", "w3", "w2", "w1")],
        "purchases": purchases[:10],
    }


def portfolio_summary() -> dict:
    rows = customers_list()
    revenue = _load(config.REVENUE_HISTORY_FILE)
    purchases = _load(config.PURCHASES_FILE)
    n = len(rows) or 1
    by = lambda key: dict(collections.Counter(r[key] for r in rows).most_common())  # noqa: E731
    top_offers = collections.Counter(p["program_rule"] for p in purchases).most_common(6)
    return {
        "customers": len(rows),
        "high_risk": sum(r["risk_level"] == "HIGH" for r in rows),
        "medium_risk": sum(r["risk_level"] == "MEDIUM" for r in rows),
        "avg_arpu": round(sum(r["arpu"] for r in rows) / n, 2),
        "avg_churn": round(sum(r["churn"] for r in rows) / n, 3),
        "monthly_revenue": [
            {"period": p, "omr": round(sum(r.get(f"rev_{p}", 0) for r in revenue), 2)} for p in ("m3", "m2", "m1")
        ],
        "weekly_revenue": [
            {"period": p, "omr": round(sum(r.get(f"rev_{p}", 0) for r in revenue), 2)} for p in ("w4", "w3", "w2", "w1")
        ],
        "revenue_at_risk": round(sum(r["avg_monthly_revenue"] * r["churn"] for r in rows), 2),
        "by_value_segment": by("value_segment"),
        "by_device_segment": by("device_segment"),
        "by_risk": by("risk_level"),
        "by_need": by("need_signal"),
        "by_policy_action": by("policy_action"),
        "by_channel": by("preferred_channel"),
        "purchases": len(purchases),
        "purchase_revenue": round(sum(p["price_omr"] * p.get("purchase_count", 1) for p in purchases), 2),
        "top_offers": [{"offer": o, "purchases": k} for o, k in top_offers],
    }
