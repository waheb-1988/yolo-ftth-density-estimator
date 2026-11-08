"""Batch flow, one call per batch of customers:

    1. action             llm arm: the action agent (offline = bandit sampled action, as cstr);
                          cstr: the scored action; free: shown, not used (BR-60)
    2. top 4 offers       llm arm: offer + competitor agents; cstr / free: the 3-pass rules (BR-66..82)
    3. top 2 roaming      add-on family, ranked on the customer's roaming history (nbx/fsdr/addons.py)
    4. international      add-on family, only if the customer calls abroad

NO_CONTACT (or a customer outside the scored sample) gets no offer and no add-on.
"""
from __future__ import annotations

import asyncio

from nbx.fsdr import config


def _offer_rows(ws, names: list[str]) -> list[dict]:
    cat = ws.catalog
    return [{"offer": n, "service": cat.at[n, "SERVICE"], "generosity": cat.at[n, "GENEROSITY"],
             "validity": int(cat.at[n, "VALIDITY"]), "price": float(cat.at[n, "PRICE"])} for n in names if n in cat.index]


async def recommend_one(ws, msisdn: str, arm: str = "llm", llm: str | None = None) -> dict:
    v = str(msisdn).strip()
    if v not in ws.scored.index:
        return {"msisdn": v, "error": "not in the scored sample of this run"}
    if arm == "llm":
        from nbx.fsdr.agents.recommender_agent import decide

        d = (await decide(ws, v, "llm", llm))["decision"]
        row = {"msisdn": v, "arm": arm, "mode": d["mode"], "action": d["action"], "policy_source": d["policy_source"],
               "propensity": d["propensity"], "offers": d["offers"], "competitor": d["competitor"]["verdict"],
               "critic": d["critic"]["verdict"], "overrides": d["override_count"]}
    elif arm in ("cstr", "free"):
        action = ws.scored.at[v, "ACTION"]
        picks = ws.engine.select(v, action if arm == "cstr" else None, arm)
        row = {"msisdn": v, "arm": arm, "mode": "rules", "action": action, "policy_source": "bandit",
               "propensity": float(ws.scored.at[v, "PROBA"]), "offers": _offer_rows(ws, [p.btl_name for p in picks]),
               "competitor": None, "critic": None, "overrides": 0}
    else:
        raise ValueError("arm must be llm, cstr or free")
    main = [o["offer"] for o in row["offers"]]
    if row["action"] == config.NO_CONTACT:
        add = {"roaming": [], "roaming_basis": None, "international": [], "international_reason": "no contact this week",
               "international_calls_m1": None}
    else:
        add = ws.addons.recommend(v, main)
    return {**row, **add}


async def recommend_batch(ws, msisdns: list[str], arm: str = "llm", llm: str | None = None, concurrency: int | None = None) -> list[dict]:
    if len(msisdns) > config.BATCH_MAX:
        raise ValueError(f"at most {config.BATCH_MAX} customers per call")
    sem = asyncio.Semaphore(max(1, concurrency or config.AGENT_CONCURRENCY))

    async def one(v):
        async with sem:
            return await recommend_one(ws, v, arm, llm)

    return list(await asyncio.gather(*(one(v) for v in msisdns)))


def flat_row(r: dict) -> dict:
    """One CSV / table row per customer."""
    def names(items, k):
        return [f"{o['offer']} ({o['price']:g})" for o in (items or [])][:k]

    offers, roaming, intl = names(r.get("offers"), 4), names(r.get("roaming"), 2), names(r.get("international"), 2)
    return {"msisdn": r["msisdn"], "arm": r.get("arm"), "action": r.get("action", ""), "error": r.get("error", ""),
            **{f"offer_{i + 1}": offers[i] if i < len(offers) else "" for i in range(4)},
            **{f"roaming_{i + 1}": roaming[i] if i < len(roaming) else "" for i in range(2)},
            "international": ", ".join(intl) or (r.get("international_reason") or ""),
            "competitor": r.get("competitor") or "", "critic": r.get("critic") or "", "mode": r.get("mode", ""),
            "propensity": r.get("propensity")}
