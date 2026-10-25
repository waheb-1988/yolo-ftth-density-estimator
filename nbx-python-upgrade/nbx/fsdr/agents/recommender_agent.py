"""FSDR llm-arm recommender: three agents on the shared ReAct loop, the LLM decides.

    Action agent  ->  Offer agent  ->  Competitor agent  ->  critic (code)  ->  messenger (scripts)
    (any of 7 actions    (any eligible       (compares the offers with
     or NO_CONTACT)       catalogue offer)     competitor plans, may swap)

Evidence, not orders: the bandit pmf, the segment policy, the ranker and the
3-pass price band are tools the agents call; the LLM may disagree with them,
but every override needs a written reason (validated) and is logged.

Hard rules (code-enforced, the LLM cannot override): eligible customer (BR-02,
BR-10, BR-36), minimum gap since the last contact, no upsell-family action
when PROBA_CHURN >= threshold, offers exist / price > 0 / <= 2 per validity /
<= 4 in total / no conflicting pair. A violating answer is rejected back to
the agent; if it still violates when the agent stops, the code repairs it.

Off-policy logging: with probability EPSILON_LLM the final action is a bandit
sample (propensity = pmf[a], policy_source "bandit"); otherwise the LLM's
choice with propensity = 1 - eps + eps * pmf[a] (policy_source "llm").

Offline (no LLM): every agent takes its rule-based path with the same tools,
which is exactly the constraint arm (scored action + 3-pass offers).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from nbx.agents.agentic import council
from nbx.agents.agentic.react import AgentSpec, LLMUnavailable, call_tool_traced, run_react
from nbx.fsdr import campaign_log, config
from nbx.fsdr.agents.tools import FSDR_TOOLS, FsdrContext, FsdrWorkspace
from nbx.fsdr.offer_rules import check_offers
from nbx.fsdr.pipelines import outputs_dir, write_report, write_table

ALL_ACTIONS = config.ACTIONS + [config.NO_CONTACT]
VERDICTS = {"competitive", "mixed", "uncompetitive"}


def _norm_action(a) -> str | None:
    s = str(a or "").upper().replace("_", " ").strip()
    return config.NO_CONTACT if s in ("NO CONTACT", "NONE", "NO ACTION") else s


def _clip01(x, default: float = 0.5) -> float:
    try:
        return float(min(max(float(x), 0.0), 1.0))
    except (TypeError, ValueError):
        return default


def _resolve_names(ctx: FsdrContext, names) -> list[str]:
    if isinstance(names, str):
        names = [s.strip() for s in names.split(",")]
    lookup = {n.lower(): n for n in ctx.ws.catalog.index}
    return [lookup.get(str(n).strip().lower(), str(n).strip()) for n in (names or []) if str(n).strip()]


# ---------------------------------------------------------------------------
# Final-answer validators (rejections go back to the agent as observations)
# ---------------------------------------------------------------------------
def validate_action(final, ctx: FsdrContext):
    if not isinstance(final, dict):
        return None, "final must be a JSON object"
    a = _norm_action(final.get("action"))
    if a not in ALL_ACTIONS:
        return None, f"action must be one of {ALL_ACTIONS}"
    top = ctx.bandit_top()
    reason = str(final.get("override_reason") or "").strip()
    if a != top and len(reason) < 8:
        return None, f"you chose {a} but the bandit's top action is {top}: explain why in override_reason"
    signals = final.get("signals_used")
    return {"action": a, "signals_used": [str(s) for s in signals][:8] if isinstance(signals, list) else [],
            "override_reason": reason, "confidence": _clip01(final.get("confidence")),
            "expected_acceptance": _clip01(final.get("expected_acceptance"), 0.1)}, None


def validate_offers(final, ctx: FsdrContext):
    if not isinstance(final, dict):
        return None, "final must be a JSON object"
    offers = _resolve_names(ctx, final.get("offers"))
    if not offers:
        return None, "offers must be a non-empty list of catalogue codes"
    violations = check_offers(offers, ctx.ws.engine.catalog, ctx.ws.engine.conflicts)
    if violations:
        return None, "hard rules violated: " + "; ".join(violations[:4])
    default = [p.btl_name for p in ctx.deterministic_offers()]
    reason = str(final.get("override_reason") or "").strip()
    if set(offers) != set(default) and len(reason) < 8:
        return None, "your offers differ from price_band_candidates: explain why in override_reason"
    return {"offers": offers, "rationale": str(final.get("rationale") or "").strip(), "override_reason": reason}, None


def validate_competitor(final, ctx: FsdrContext):
    if not isinstance(final, dict):
        return None, "final must be a JSON object"
    verdict = str(final.get("verdict") or "").lower().strip()
    if verdict not in VERDICTS:
        return None, f"verdict must be one of {sorted(VERDICTS)}"
    swaps = []
    for s in final.get("swap") or []:
        if not isinstance(s, dict):
            continue
        out, new = _resolve_names(ctx, [s.get("out")]), _resolve_names(ctx, [s.get("in")])
        if out and new and out[0] in ctx.offers and new[0] in ctx.ws.catalog.index:
            swaps.append({"out": out[0], "in": new[0], "reason": str(s.get("reason") or "").strip()})
    return {"verdict": verdict, "summary": str(final.get("summary") or "").strip(), "swap": swaps}, None


ACTION_SPEC = AgentSpec(
    key="fsdr_action", name="Best Action agent",
    role=(f"You are the {config.OPERATOR_NAME} Next-Best-Action strategist for prepaid customers. Choose ONE action for this week: "
          + ", ".join(ALL_ACTIONS) + ". The bandit, the segment policy and the scores are evidence, not orders: you may "
          "disagree, but then explain why. Respect contact_policy (blocked actions, eligibility). Reply with JSON only."),
    goal="Pick the best action for this customer this week and list the signals you used.",
    tools=["customer_360", "bandit_pmf", "segment_policy", "campaign_history", "contact_policy"],
    required_tools=["customer_360", "bandit_pmf", "contact_policy"],
    final_schema='{"action": str, "signals_used": [str], "override_reason": str (required if not the bandit top action), '
                 '"confidence": 0-1, "expected_acceptance": 0-1}',
    validate_final=validate_action, max_steps=6)

OFFER_SPEC = AgentSpec(
    key="fsdr_offer", name="Offer agent",
    role=(f"You are the {config.OPERATOR_NAME} offer specialist. For the chosen action, pick 1 to 4 BTL offers (at most 2 per "
          "validity, never two conflicting offers). price_band_candidates is the rule-based answer; the ARPU price band and the "
          "generosity mapping are guidance you may override with a reason. Any eligible catalogue offer is allowed. "
          "Always call check_offers on your final list. Reply with JSON only."),
    goal="Choose the offers that maximise acceptance and value for this customer and action.",
    tools=["price_band_candidates", "ranker_top_offers", "cohort_offers", "arpu_anchors", "search_catalog", "check_offers",
           "expected_value"],
    required_tools=["price_band_candidates", "check_offers"],
    final_schema='{"offers": [codes], "rationale": str, "override_reason": str (required if offers differ from price_band_candidates)}',
    validate_final=validate_offers, max_steps=7)

COMPETITOR_SPEC = AgentSpec(
    key="fsdr_competitor", name="Competitor agent",
    role=(f"You compare {config.OPERATOR_NAME}'s recommended offers with competitor plans (shown as COMPETITOR_A / COMPETITOR_B). "
          "Judge whether each offer is competitive at the same validity. If one is clearly beaten, you may propose a swap for "
          "another catalogue offer; the code checks every swap against the hard rules. Reply with JSON only."),
    goal="Give a verdict on the offers versus the market and propose swaps only when an offer is clearly uncompetitive.",
    tools=["competitor_compare", "competitor_market", "search_catalog"],
    required_tools=["competitor_compare"],
    final_schema='{"verdict": "competitive|mixed|uncompetitive", "summary": str, "swap": [{"out": code, "in": code, "reason": str}]}',
    validate_final=validate_competitor, max_steps=5)


# ---------------------------------------------------------------------------
# Hard rules on the action, repair of the offers, exploration
# ---------------------------------------------------------------------------
def guard_action(ctx: FsdrContext, action: str) -> tuple[str, list[str]]:
    findings = []
    contact = FSDR_TOOLS["contact_policy"].fn(ctx, {})
    if action != config.NO_CONTACT and not contact["can_contact"]:
        findings.append(f"{action} -> NO_CONTACT: " + "; ".join(contact["blocked_reasons"]))
        action = config.NO_CONTACT
    if action in config.UPSELL_FAMILY and ctx.churn() >= config.CHURN_BLOCK_THRESHOLD:
        findings.append(f"{action} -> RETAIN: PROBA_CHURN {ctx.churn():.2f} >= {config.CHURN_BLOCK_THRESHOLD} blocks upsell")
        action = "RETAIN"
    return action, findings


def repair_offers(ctx: FsdrContext, offers: list[str]) -> tuple[list[str], list[str]]:
    """Keeps the offers (in order) that pass every hard rule together; falls back to the 3-pass picks."""
    kept, dropped = [], []
    for o in offers:
        if not check_offers(kept + [o], ctx.ws.engine.catalog, ctx.ws.engine.conflicts):
            kept.append(o)
        else:
            dropped.append(o)
    findings = [f"dropped {o}: breaks a hard rule with the other offers" for o in dropped]
    if not kept:
        kept = [p.btl_name for p in ctx.deterministic_offers()]
        findings.append("no valid offer left: used the rule-based price-band offers")
    return kept, findings


def logged_propensity(action: str, llm_action: str, pmf: dict[str, float], eps: float) -> float:
    """P(action) under the llm-arm policy: (1 - eps) x 1[action = LLM action] + eps x PMF[action].
    The same value whichever branch produced the action (exploration may sample the LLM's own action)."""
    return float((1 - eps) * (action == llm_action) + eps * pmf.get(action, 0.0))


def explore(ctx: FsdrContext, llm_action: str, rng: np.random.Generator) -> tuple[str, str, float]:
    """(action, policy_source, propensity) -- keeps the llm-arm logs usable for off-policy training."""
    pmf = ctx.pmf()
    eps = config.EPSILON_LLM
    if rng.random() < eps:
        acts = list(pmf)
        p = np.array([pmf[a] for a in acts])
        a = acts[int(rng.choice(len(acts), p=p / p.sum()))]
        return a, "bandit", logged_propensity(a, llm_action, pmf, eps)
    return llm_action, "llm", logged_propensity(llm_action, llm_action, pmf, eps)


def _cache_key(ctx: FsdrContext, model: str | None) -> str:
    row = ctx.row
    feats = "|".join(f"{k}={round(float(row.get(k, 0) or 0), 4)}" for k in sorted(config.proba_features()))
    key = f"{ctx.msisdn}|{feats}|{ctx.ws.engine.version}|{ctx.ws.model_version}|{model}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _messages(ctx: FsdrContext, offers: list[str]) -> dict:
    en = [ctx.ws.scripts[o][0] for o in offers if o in ctx.ws.scripts]
    ar = [ctx.ws.scripts[o][1] for o in offers if o in ctx.ws.scripts]
    return {"sms_en": en[:2], "sms_ar": ar[:2], "source": "BTL_script_english_arabic_ok"}


def _competition_verdict(results: list[dict]) -> str:
    risky = [r for r in results if r.get("at_risk")]
    return "competitive" if not risky else "uncompetitive" if len(risky) == len(results) else "mixed"


# ---------------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------------
def _prefetch(agent: str, ctx: FsdrContext, calls: list[tuple[str, dict]], emit) -> list[dict] | None:
    """Runs an agent's required evidence tools in code (config.AGENT_PREFETCH); None = the LLM calls them."""
    if not config.AGENT_PREFETCH:
        return None
    hist: list[dict] = []
    for tool, args in calls:
        call_tool_traced(agent, ctx, tool, args, emit, hist, FSDR_TOOLS)
    return hist


async def decide(ws: FsdrWorkspace, msisdn: str, arm: str = "llm", llm: str | None = None, emit=None) -> dict:
    events: list[dict] = []
    start = time.time()

    def _emit(ev: dict) -> None:
        ev = {"t_ms": int((time.time() - start) * 1000), **ev}
        events.append(ev)
        if emit:
            emit(ev)

    ctx = ws.context(msisdn, arm)
    model = await council._resolve_model(llm)
    mode = "agentic" if model else "fallback"
    rng = np.random.default_rng([ws.run_date.toordinal(), int(msisdn)])
    llm_calls = 0
    findings: list[str] = []
    overrides: list[dict] = []
    _emit({"type": "run_start", "msisdn": msisdn, "arm": arm, "mode": mode, "model": model,
           "agents": ["fsdr_action", "fsdr_offer", "fsdr_competitor", "critic", "messenger"]})

    # --- 1. Action agent -----------------------------------------------------------------
    _emit({"type": "agent_start", "agent": "fsdr_action", "name": ACTION_SPEC.name})
    final_a = None
    if model:
        try:
            res = await run_react(ACTION_SPEC, ctx, _emit, model, registry=FSDR_TOOLS,
                                  prior=_prefetch("fsdr_action", ctx, [(t, {}) for t in ACTION_SPEC.required_tools], _emit))
            llm_calls += res.llm_calls
            final_a = res.final
        except LLMUnavailable as e:
            _emit({"type": "policy", "agent": "fsdr_action", "text": f"LLM unavailable ({e}); rule-based path."})
            model, mode = None, "partial_fallback"
    top = ctx.bandit_top()
    if final_a is None:
        hist: list[dict] = []
        for t in ACTION_SPEC.required_tools:
            call_tool_traced("fsdr_action", ctx, t, {}, _emit, hist, FSDR_TOOLS)
        s = ctx.scored
        action, policy_source, propensity = s["ACTION"], "fallback", float(s["PROBA"])
        justification = {"signals_used": ["bandit sampled action (constraint-arm logic)"], "override_reason": "",
                         "expected_acceptance": None}
        _emit({"type": "agent_final", "agent": "fsdr_action", "output": {"action": action}, "finalized_by": "policy"})
    else:
        guarded, f = guard_action(ctx, final_a["action"])
        findings += f
        action, policy_source, propensity = explore(ctx, guarded, rng)
        if policy_source == "bandit":
            _emit({"type": "policy", "agent": "fsdr_action",
                   "text": f"Exploration (eps={config.EPSILON_LLM}): bandit sample {action} replaces {guarded}."})
        if final_a["action"] != top:
            overrides.append({"type": "action", "model": top, "agent": final_a["action"], "reason": final_a["override_reason"]})
        justification = {k: final_a[k] for k in ("signals_used", "override_reason", "expected_acceptance")}
        _emit({"type": "agent_final", "agent": "fsdr_action", "output": final_a, "finalized_by": "llm"})
    ctx.action = action

    # --- 2. Offer agent --------------------------------------------------------------------
    _emit({"type": "agent_start", "agent": "fsdr_offer", "name": OFFER_SPEC.name})
    offer_rationale, offers = "", []
    picks = {} if action == config.NO_CONTACT else {p.btl_name: p for p in ctx.deterministic_offers()}
    if action == config.NO_CONTACT:
        _emit({"type": "policy", "agent": "fsdr_offer", "text": "NO_CONTACT: no offer this week."})
    else:
        final_o = None
        if model:
            try:
                extra = f"Chosen action: {action}. Rule-based offers: {', '.join(picks) or 'none'}."
                prior = _prefetch("fsdr_offer", ctx, [("price_band_candidates", {"action": action}),
                                                      ("check_offers", {"offers": list(picks)})], _emit)
                res = await run_react(OFFER_SPEC, ctx, _emit, model, extra_context=extra, registry=FSDR_TOOLS, prior=prior)
                llm_calls += res.llm_calls
                final_o = res.final
            except LLMUnavailable as e:
                _emit({"type": "policy", "agent": "fsdr_offer", "text": f"LLM unavailable ({e}); rule-based offers."})
                model, mode = None, "partial_fallback"
        if final_o is None:
            hist = []
            call_tool_traced("fsdr_offer", ctx, "price_band_candidates", {"action": action}, _emit, hist, FSDR_TOOLS)
            offers = list(picks)
        else:
            offers, f = repair_offers(ctx, final_o["offers"])
            findings += f
            offer_rationale = final_o["rationale"]
            if set(offers) != set(picks):
                overrides.append({"type": "offers", "model": list(picks), "agent": offers, "reason": final_o["override_reason"]})
        _emit({"type": "agent_final", "agent": "fsdr_offer", "output": {"offers": offers}, "finalized_by": "llm" if final_o else "policy"})
    ctx.offers = offers

    # --- 3. Competitor agent -----------------------------------------------------------------
    competitor = {"verdict": None, "summary": "", "comparisons": [], "swaps_applied": []}
    if offers:
        _emit({"type": "agent_start", "agent": "fsdr_competitor", "name": COMPETITOR_SPEC.name})
        hist = []
        cmp = call_tool_traced("fsdr_competitor", ctx, "competitor_compare", {"offers": offers}, _emit, hist, FSDR_TOOLS)
        competitor["comparisons"] = cmp.get("results", [])
        competitor["verdict"] = _competition_verdict(competitor["comparisons"]) if competitor["comparisons"] else None
        final_c = None
        if model:
            try:
                res = await run_react(COMPETITOR_SPEC, ctx, _emit, model, prior=hist,
                                      extra_context=f"Offers to compare: {', '.join(offers)}.", registry=FSDR_TOOLS)
                llm_calls += res.llm_calls
                final_c = res.final
            except LLMUnavailable as e:
                _emit({"type": "policy", "agent": "fsdr_competitor", "text": f"LLM unavailable ({e}); rule-based verdict."})
                model, mode = None, "partial_fallback"
        if final_c:
            competitor.update({"verdict": final_c["verdict"], "summary": final_c["summary"]})
            for s in final_c["swap"]:
                trial = [s["in"] if o == s["out"] else o for o in offers]
                bad = check_offers(trial, ctx.ws.engine.catalog, ctx.ws.engine.conflicts)
                if bad:
                    findings.append(f"swap {s['out']} -> {s['in']} refused: {bad[0]}")
                    continue
                offers = trial
                competitor["swaps_applied"].append(s)
                overrides.append({"type": "competitor_swap", "model": s["out"], "agent": s["in"], "reason": s["reason"]})
        elif competitor["verdict"]:
            competitor["summary"] = f"{sum(1 for r in competitor['comparisons'] if r.get('at_risk'))} of " \
                                    f"{len(competitor['comparisons'])} offer(s) priced above the closest competitor plan"
        _emit({"type": "agent_final", "agent": "fsdr_competitor", "output": {k: competitor[k] for k in ("verdict", "summary", "swaps_applied")},
               "finalized_by": "llm" if final_c else "policy"})
    ctx.offers = offers

    # --- 4. Critic (code): every hard rule on the final decision ------------------------------
    final_violations = check_offers(offers, ctx.ws.engine.catalog, ctx.ws.engine.conflicts) if offers else []
    if final_violations:                                   # cannot happen after repair; kept as the last gate
        offers, f = repair_offers(ctx, offers)
        findings += f
    critic = {"verdict": "revised" if findings else "approved", "findings": findings}
    _emit({"type": "guardrails", "agent": "critic", "findings": findings, "verdict": critic["verdict"]})

    # --- 5. Messenger ---------------------------------------------------------------------------
    message = _messages(ctx, offers)
    cat = ctx.ws.catalog
    detailed = [{"offer": o, "service": cat.at[o, "SERVICE"], "generosity": cat.at[o, "GENEROSITY"],
                 "validity": int(cat.at[o, "VALIDITY"]), "price": float(cat.at[o, "PRICE"]),
                 "pass": picks[o].pass_no if o in picks else None, "anchor": picks[o].anchor if o in picks else None}
                for o in offers]
    decision = {
        "msisdn": msisdn, "arm": arm, "run_date": ws.run_date.isoformat(), "mode": mode, "model": model or None,
        "model_version": ws.model_version, "catalog_version": ws.engine.version, "action": action,
        "policy_source": policy_source, "propensity": round(propensity, 6), "bandit_top_action": top,
        "bandit_pmf": {a: round(p, 4) for a, p in ctx.pmf().items()}, "offers": detailed, "overrides": overrides,
        "override_count": len(overrides), "critic": critic, "competitor": competitor, "message": message,
        "justification": {**justification, "offer_rationale": offer_rationale},
        "stats": {"llm_calls": llm_calls, "tool_calls": sum(1 for e in events if e["type"] == "tool_call"),
                  "elapsed_ms": int((time.time() - start) * 1000)},
        "cache_key": _cache_key(ctx, model),
    }
    _emit({"type": "run_end", "decision": {k: decision[k] for k in ("action", "policy_source", "propensity", "override_count")}})
    return {"decision": decision, "trace": events}


# ---------------------------------------------------------------------------
# Batch over the llm-arm sample (concurrency, resume, cache)
# ---------------------------------------------------------------------------
DECISIONS_FILE = "llm_decisions.jsonl"


async def run_batch(data_dir: str | Path, llm: str | None = None, concurrency: int | None = None, limit: int | None = None,
                    resume: bool = True, ws: FsdrWorkspace | None = None, progress=None) -> dict:
    """Decides for every line of the llm arm. Each decision is appended to outputs/llm_decisions.jsonl
    as soon as it is made, so an interrupted run resumes where it stopped (same cache key = reused)."""
    t0 = time.time()
    ws = ws or FsdrWorkspace(data_dir)
    vals = ws.lines("llm")[: limit or None]
    path = outputs_dir(data_dir) / DECISIONS_FILE
    done: dict[str, dict] = {}
    if resume and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["msisdn"]] = d
    else:
        path.write_text("", encoding="utf-8")
    model = await council._resolve_model(llm)
    todo = [v for v in vals if v not in done or done[v].get("cache_key") != _cache_key(ws.context(v), model)]
    sem = asyncio.Semaphore(max(1, concurrency or config.AGENT_CONCURRENCY))
    lock = asyncio.Lock()
    count = {"n": 0}

    async def one(v: str):
        async with sem:
            d = (await decide(ws, v, "llm", llm))["decision"]
        async with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
            done[v] = d
            count["n"] += 1
            if progress:
                progress(count["n"], len(todo))

    await asyncio.gather(*(one(v) for v in todo))
    decisions = [done[v] for v in vals if v in done]
    rows = [{"ACCESS_METHOD_VAL": d["msisdn"], "ARM": "llm", "ACTION": d["action"], "BTL_NAME": o["offer"], "SERVICE": o["service"],
             "GENEROSITY": o["generosity"], "VALIDITY": o["validity"], "PRICE": o["price"], "PASS_NO": o["pass"],
             "ANCHOR": o["anchor"], "POLICY_SOURCE": d["policy_source"], "PROPENSITY": d["propensity"]}
            for d in decisions for o in d["offers"]]
    import pandas as pd

    recos = pd.DataFrame(rows)
    for validity, name in ((30, "monthly"), (7, "weekly")):
        write_table(recos[recos["VALIDITY"] == validity] if len(recos) else recos, data_dir, f"recos_{name}_llm")
    report = {
        "arm": "llm", "customers": len(vals), "decided_now": len(todo), "reused_from_checkpoint": len(vals) - len(todo),
        "modes": pd.Series([d["mode"] for d in decisions]).value_counts().to_dict() if decisions else {},
        "policy_source": pd.Series([d["policy_source"] for d in decisions]).value_counts().to_dict() if decisions else {},
        "actions": pd.Series([d["action"] for d in decisions]).value_counts().to_dict() if decisions else {},
        "override_rate": round(sum(1 for d in decisions if d["override_count"]) / len(decisions), 4) if decisions else 0.0,
        "competitor_verdicts": pd.Series([d["competitor"]["verdict"] for d in decisions]).value_counts().to_dict() if decisions else {},
        "offers": len(rows), "seconds": round(time.time() - t0, 1),
    }
    report["campaign_log_rows"] = campaign_log.append_log(campaign_log.root_for(data_dir),
                                                          campaign_log.rows_from_decisions(decisions, ws.engine.anchors))   # M1
    write_report(report, data_dir, "nba_recommend_report_llm")
    return report
