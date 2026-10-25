"""Agentic Next-Best-Action / Next-Best-Offer: a small "decision council".

    NBA Strategist  --action-->  NBO Specialist  --offers-->  Critic  --approved-->  Messenger
         (ReAct + tools)            (ReAct + tools)        (guardrails +      (bilingual SMS +
                                                            LLM review,        call script)
                                                            1 revision loop)

What makes it agentic rather than a fixed pipeline:
  - each specialist decides which evidence to gather (tools), in what
    order, and when it has enough -- the old pipelines always ran the same
    steps for every customer;
  - offers are chosen on simulated expected value (acceptance probability
    x revenue + retention value), not on a similarity score;
  - a critic reviews the joint decision against hard business guardrails
    and can send it back once with feedback (self-correction); anything
    still violating a hard guardrail is auto-fixed by code, never shipped;
  - the run streams every thought / tool call / observation as events, so
    the UI shows the reasoning live.

Every agent has a deterministic fallback using the same tools, so a run
completes (mode="fallback") when no LLM is reachable or a call fails.
"""
from __future__ import annotations

import collections
import re
import time
from typing import Any

from nbx.core import config
from nbx.llms import llm_client
from nbx.agents.agentic.react import (
    AgentResult,
    AgentSpec,
    Emit,
    LLMUnavailable,
    call_tool_traced,
    evidence_of,
    run_react,
)
from nbx.agents.agentic import explain, languages
from nbx.agents.agentic.tools import RunContext, eligibility, expected_value, market_overview, risk_level

NBA_ACTIONS = config.NBA_ACTIONS + ["NO_ACTION"]
CHANNELS = ["APP", "SMS", "OUTBOUND_CALL", "USSD", "CALL_CENTER"]
PUSH_ACTIONS = {a for a in config.NBA_ACTIONS if a not in config.NBA_NON_PUSH_ACTIONS}

# Which catalog categories serve each action. Voice offers are excluded by
# business rule, so voice-flavored actions are served by mixed Bundles.
ACTION_CATEGORIES = {
    "UPSELL DATA": ["Data", "Bundle"],
    "CROSS SELL DATA": ["Data", "Bundle"],
    "UPSELL VOICE": ["Bundle"],
    "CROSS SELL VOICE": ["Bundle"],
    "UPSELL": ["NBO", "Data", "Bundle"],
    "RETAIN": ["NBO", "Bundle"],
    "MAINTAIN": ["NBO", "Bundle"],
}

# Recent runs, newest first -- feeds the dashboard's "recent decisions".
RECENT_RUNS: collections.deque = collections.deque(maxlen=50)

# Last agentic NBA decision per customer, so the offer step can be run on the
# action the user just reviewed ("Step 2" in the UI) without re-running NBA.
NBA_HANDOFF_TTL_S = 30 * 60
_NBA_HANDOFF: dict[int, tuple[float, dict]] = {}


def last_nba_decision(msisdn: int) -> dict | None:
    hit = _NBA_HANDOFF.get(msisdn)
    if hit and time.time() - hit[0] <= NBA_HANDOFF_TTL_S:
        return hit[1]
    return None


# ---------------------------------------------------------------------------
# Final-answer validators
# ---------------------------------------------------------------------------


def _norm_action(a: Any) -> str | None:
    s = str(a or "").strip().upper().replace("_", " ")
    if s == "NO ACTION":
        return "NO_ACTION"
    return s if s in NBA_ACTIONS else None


def validate_nba(final: Any, ctx: RunContext) -> tuple[dict | None, str | None]:
    if not isinstance(final, dict):
        return None, "final must be a JSON object"
    action = _norm_action(final.get("action"))
    if action is None:
        return None, f"action must be one of {NBA_ACTIONS}"
    channel = str(final.get("channel") or "").upper().replace(" ", "_")
    try:
        confidence = min(max(float(final.get("confidence", 0.6)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.6
    return {
        "action": action,
        "channel": channel if channel in CHANNELS else None,
        "send_window": final.get("send_window"),
        "confidence": round(confidence, 2),
        "rationale": str(final.get("rationale") or "").strip(),
    }, None


def validate_nbo(final: Any, ctx: RunContext) -> tuple[dict | None, str | None]:
    if not isinstance(final, dict) or not isinstance(final.get("offers"), list):
        return None, 'final must be {"offers": [{"offer": "<code>", "why": "..."}], "rationale": "..."}'
    picked, unknown = [], []
    for item in final["offers"][:3]:
        code = item.get("offer") if isinstance(item, dict) else item
        e = eligibility(ctx, str(code))
        if "not in the catalog" in e["reasons"]:
            unknown.append(str(code))
            continue
        if e["offer"] not in {p["offer"] for p in picked}:
            picked.append({"offer": e["offer"], "why": (item.get("why") if isinstance(item, dict) else "") or ""})
    if unknown:
        return None, f"unknown offer codes {unknown}; use exact codes from search_offers/cf_recommendations"
    if not picked:
        return None, "pick at least one offer"
    return {"offers": picked, "rationale": str(final.get("rationale") or "").strip()}, None


# ---------------------------------------------------------------------------
# Agent specs
# ---------------------------------------------------------------------------

NBA_SPEC = AgentSpec(
    key="nba",
    name="NBA Strategist",
    role=(
        f"You are the Next-Best-Action strategist of {config.OPERATOR_NAME}'s decision council. "
        "You gather evidence with tools, then choose ONE action for the customer. "
        "Protect at-risk customers (retain before you sell), respect contact fatigue, "
        "and treat the bandit as a strong prior you may override with evidence. "
        "Reply with JSON only."
    ),
    goal="Decide the single best action for this customer, the channel and the send window.",
    tools=["get_customer_profile", "get_risk_scores", "get_usage_trends", "bandit_recommendation", "contact_policy", "competitor_benchmark"],
    required_tools=["get_risk_scores", "bandit_recommendation"],
    final_schema=(
        '{"action": "<one of: ' + ", ".join(NBA_ACTIONS) + '>", '
        '"channel": "APP|SMS|OUTBOUND_CALL", "send_window": "<hh:mm-hh:mm>", '
        '"confidence": <0-1>, "rationale": "<2 sentences citing the evidence>"}'
    ),
    validate_final=validate_nba,
    max_steps=6,
)

NBO_SPEC = AgentSpec(
    key="nbo",
    name="NBO Specialist",
    role=(
        f"You are the Next-Best-Offer specialist of {config.OPERATOR_NAME}'s decision council. "
        "Given the chosen action, find candidate offers (catalog, collaborative filtering, peers), "
        "check their eligibility, simulate their expected value, then pick up to 3 eligible offers "
        "with the highest expected value that fit the action. Use exact offer codes. Reply with JSON only."
    ),
    goal=(
        "Pick up to 3 offers that serve the chosen action and maximize expected value. "
        "Efficient order: search_offers (and/or cf_recommendations) -> check_eligibility on 3-6 candidates "
        "-> simulate_expected_value on the eligible ones -> competitor_benchmark on your best 1-3 -> final. "
        "Prefer offers that beat COMPETITOR_A / COMPETITOR_B on price or data."
    ),
    tools=["cf_recommendations", "peer_insights", "search_offers", "check_eligibility", "simulate_expected_value", "competitor_benchmark"],
    required_tools=["check_eligibility", "simulate_expected_value"],
    final_schema='{"offers": [{"offer": "<exact offer code>", "why": "<1 sentence>"}], "rationale": "<2 sentences>"}',
    validate_final=validate_nbo,
    max_steps=7,
)


# ---------------------------------------------------------------------------
# Deterministic policies (fallback + safety net)
# ---------------------------------------------------------------------------


def _ensure(spec_key: str, ctx: RunContext, history: list[dict], tools: list[str], emit: Emit) -> None:
    for t in tools:
        if evidence_of(history, t) is None:
            call_tool_traced(spec_key, ctx, t, {}, emit, history)


def policy_nba(ctx: RunContext, history: list[dict], emit: Emit) -> dict:
    _ensure("nba", ctx, history, ["get_risk_scores", "bandit_recommendation", "contact_policy"], emit)
    risk = evidence_of(history, "get_risk_scores") or {}
    bandit = evidence_of(history, "bandit_recommendation") or {}
    contact = evidence_of(history, "contact_policy") or {}
    if risk.get("excluded_from_nba") or bandit.get("excluded"):
        action, why = "NO_ACTION", "Customer's segment is excluded from NBA campaigns."
    elif risk.get("risk_level") == "HIGH":
        action, why = "RETAIN", f"High churn risk ({risk.get('churn')}) -- retention comes before any sales push."
    else:
        action = bandit.get("segment_policy_action") or bandit.get("bandit_action") or "MAINTAIN"
        why = f"Segment policy / bandit favour {action} (bandit p={bandit.get('bandit_probability')})."
    if action in PUSH_ACTIONS and contact.get("push_offer_allowed") is False:
        action, why = "MAINTAIN", contact.get("cooldown_reason") or "Contact cooldown."
    return {
        "action": action,
        "channel": contact.get("recommended_channel"),
        "send_window": contact.get("send_window_local"),
        "confidence": 0.55,
        "rationale": why,
    }


def candidate_offers(ctx: RunContext, action: str) -> list[str]:
    cats = ACTION_CATEGORIES.get(action, [])
    return [code for code, o in ctx.offers.items() if o["category"] in cats]


def policy_nbo(ctx: RunContext, action: str, history: list[dict], emit: Emit, exclude: set[str] | None = None) -> dict:
    if action == "NO_ACTION":
        return {"offers": [], "rationale": "No action -- no offer."}
    exclude = exclude or set()
    cats = ACTION_CATEGORIES.get(action, [])
    checked = ctx.cache.get("checked", {})
    # Prefer what the agent already shortlisted and verified, if it fits the action.
    shortlist = [c for c, r in checked.items() if r.get("eligible") and c not in exclude
                 and ctx.offers.get(c, {}).get("category") in cats]
    if shortlist:
        eligible = shortlist
        emit({"type": "policy", "agent": "nbo", "text": f"Finalizing from the agent's own eligible shortlist: {', '.join(shortlist)}."})
    else:
        cands = [c for c in candidate_offers(ctx, action) if c not in exclude]
        call_tool_traced("nbo", ctx, "check_eligibility", {"offers": cands}, emit, history)
        eligible = [c for c in cands if ctx.cache.get("checked", {}).get(c, {}).get("eligible")]
    if not eligible:
        return {"offers": [], "rationale": "No eligible offer serves this action."}
    sims = call_tool_traced("nbo", ctx, "simulate_expected_value", {"offers": eligible}, emit, history)["ranked"]
    top = sims[:3]
    return {
        "offers": [{"offer": s["offer"], "why": f"Highest simulated expected value ({s['expected_value_omr']} OMR/month)."} for s in top],
        "rationale": f"Ranked {len(eligible)} eligible {', '.join(ACTION_CATEGORIES.get(action, []))} offers by expected value.",
    }


# ---------------------------------------------------------------------------
# Guardrails + critic
# ---------------------------------------------------------------------------


def guardrails(ctx: RunContext, nba: dict | None, nbo: dict | None) -> list[dict]:
    out = []
    row = ctx.score_row or {}
    action = (nba or {}).get("action") or ctx.action
    if action:
        if row.get("excluded") and action != "NO_ACTION":
            out.append({"code": "EXCLUDED_SEGMENT", "severity": "hard", "message": f"Segment {row.get('nba_segment')} is excluded from NBA; action must be NO_ACTION."})
        if risk_level(row) == "HIGH" and action in PUSH_ACTIONS:
            out.append({"code": "PUSH_TO_AT_RISK", "severity": "hard", "message": f"Churn risk is HIGH ({row.get('proba_churn'):.2f}) but the action pushes a sale ({action}); retain first."})
        if ctx.customer["days_since_last_purchase"] <= 3 and action in PUSH_ACTIONS:
            out.append({"code": "CONTACT_COOLDOWN", "severity": "hard", "message": "Customer purchased within 3 days; a sales push now breaks contact policy."})
    for o in (nbo or {}).get("offers", []):
        e = eligibility(ctx, o["offer"])
        if not e["eligible"]:
            out.append({"code": "INELIGIBLE_OFFER", "severity": "hard", "offer": o["offer"], "message": f"{o['offer']}: {'; '.join(e['reasons'])}"})
        cats = ACTION_CATEGORIES.get(action or "", [])
        cat = ctx.offers.get(o["offer"], {}).get("category")
        if cats and cat not in cats:
            out.append({"code": "OFFER_ACTION_MISMATCH", "severity": "soft", "offer": o["offer"], "message": f"{o['offer']} ({cat}) does not serve action {action} (expects {cats})."})
    return out


CRITIC_ROLE = (
    f"You are the risk & compliance critic of {config.OPERATOR_NAME}'s decision council. "
    "Review a proposed decision against the evidence and the guardrail findings. "
    "Approve only if it is safe, consistent and customer-centric. Reply with JSON only."
)


async def run_critic(ctx: RunContext, model: str | None, nba: dict | None, nbo: dict | None, findings: list[dict], emit: Emit) -> dict:
    hard = [f for f in findings if f["severity"] == "hard"]
    if model is None:
        verdict = {"verdict": "revise" if hard else "approve", "issues": [f["message"] for f in findings], "feedback": "; ".join(f["message"] for f in hard)}
        emit({"type": "thought", "agent": "critic", "text": f"Rule-based review: {len(hard)} hard / {len(findings) - len(hard)} soft finding(s)."})
        return verdict
    risk = ctx.score_row or {}
    prompt = (
        f"PROPOSED DECISION for customer {ctx.msisdn}:\n"
        f"action={(nba or {}).get('action') or ctx.action}, channel={(nba or {}).get('channel')}, "
        f"offers={[o['offer'] for o in (nbo or {}).get('offers', [])]}\n"
        f"NBA rationale: {(nba or {}).get('rationale', '-')}\nNBO rationale: {(nbo or {}).get('rationale', '-')}\n\n"
        f"KEY EVIDENCE: risk_level={risk_level(risk)}, churn={risk.get('proba_churn')}, "
        f"segment={risk.get('nba_segment')}, days_since_last_purchase={ctx.customer['days_since_last_purchase']}, "
        f"avg_monthly_revenue={ctx.avg_rev_monthly:.1f} OMR\n"
        f"COMPETITIVE POSITION OF CHOSEN OFFERS: {_competition_line(ctx, nbo)}\n"
        f"GUARDRAIL FINDINGS (computed by code): {findings or 'none'}\n\n"
        'Reply: {"verdict": "approve" | "revise", "issues": ["<short>"], "feedback": "<one instruction to fix, or empty>"}'
    )
    t0 = time.time()
    res = await llm_client.generate(model=model, prompt=prompt, system=CRITIC_ROLE, force_json=True)
    if res.get("error"):
        raise LLMUnavailable(res["error"])
    parsed = llm_client.extract_json(res["content"]) or {}
    verdict = str(parsed.get("verdict", "approve")).lower()
    out = {
        "verdict": "revise" if hard else ("revise" if verdict == "revise" else "approve"),
        "issues": [str(i) for i in (parsed.get("issues") or [])][:5] or [f["message"] for f in findings],
        "feedback": str(parsed.get("feedback") or "; ".join(f["message"] for f in hard)),
        "llm_verdict": verdict,
    }
    emit({"type": "thought", "agent": "critic", "text": f"Verdict: {out['verdict']}. " + (out["feedback"] or "Decision is consistent with the evidence."), "llm_ms": int((time.time() - t0) * 1000)})
    return out


def _competition_line(ctx: RunContext, nbo: dict | None) -> str:
    from nbx.core import competitors

    out = []
    for o in (nbo or {}).get("offers", [])[:3]:
        cmp = ctx.cache.get("competition", {}).get(o["offer"]) or competitors.compare_offer(ctx.offers[o["offer"]])
        ctx.cache.setdefault("competition", {})[o["offer"]] = cmp
        out.append(f"{o['offer']}: {cmp['position']}")
    return "; ".join(out) or "no offers"


def auto_fix(ctx: RunContext, nba: dict, nbo: dict | None, findings: list[dict], history: list[dict], emit: Emit) -> tuple[dict, dict | None, list[str]]:
    fixes = []
    codes = {f["code"] for f in findings if f["severity"] == "hard"}
    if "EXCLUDED_SEGMENT" in codes:
        nba = {**nba, "action": "NO_ACTION"}
        fixes.append("Excluded segment -> NO_ACTION.")
    elif "PUSH_TO_AT_RISK" in codes:
        nba = {**nba, "action": "RETAIN"}
        fixes.append("At-risk customer: push action replaced by RETAIN.")
    elif "CONTACT_COOLDOWN" in codes:
        nba = {**nba, "action": "MAINTAIN"}
        fixes.append("Contact cooldown: push action replaced by MAINTAIN.")
    if nbo is not None:
        action_changed = bool(codes & {"EXCLUDED_SEGMENT", "PUSH_TO_AT_RISK", "CONTACT_COOLDOWN"})
        bad = {f["offer"] for f in findings if f["code"] == "INELIGIBLE_OFFER"}
        if action_changed or bad:
            ctx.action = nba["action"]
            nbo = policy_nbo(ctx, nba["action"], history, emit, exclude=bad)
            fixes.append(f"Offers re-selected for {nba['action']} by expected value" + (f" (dropped {sorted(bad)})" if bad else "") + ".")
    for f in fixes:
        emit({"type": "auto_fix", "agent": "critic", "text": f})
    return nba, nbo, fixes


# ---------------------------------------------------------------------------
# Messenger
# ---------------------------------------------------------------------------

MESSENGER_ROLE = (
    f"You write short, warm, compliant customer messages for {config.OPERATOR_NAME} in English and Modern Standard Arabic. "
    "No pressure tactics, no false urgency. The currency is the Omani Rial: write 'OMR' in English and "
    "'ر.ع.' or 'ريال عماني' in Arabic -- never dirham or another country's riyal. Reply with JSON only."
)

# Currency words a model may wrongly use for Oman (UAE dirham, Saudi/Qatari riyal...).
_WRONG_CURRENCY = ["درهم إماراتي", "دراهم", "درهم", "ريال سعودي", "ريال قطري", "AED", "SAR", "QAR", "dirhams", "dirham", "Dirham"]


def _real_text(v) -> bool:
    """Small models sometimes echo the reply template ("...", "<the English SMS text>")."""
    s = str(v or "").strip()
    return len(s) >= 15 and not s.startswith("<") and s.strip(". ") != ""


# Internal vocabulary that must never reach a customer.
_INTERNAL_TERMS = re.compile(r"\b(NBO|NBA|churn|risk|segment|propensity|upsell|up-sell|cross[- ]sell|retain|retention|bandit|CVM)\b", re.I)
_ARABIC_LETTER = re.compile(r"[؀-ۿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def customer_safe(text: str, arabic: bool = False) -> tuple[str, str | None]:
    """Returns (cleaned_text, problem). Hashtags are stripped; internal jargon,
    or an 'Arabic' SMS that isn't mostly Arabic script (stray English words,
    other scripts), is a problem and the caller falls back to the template."""
    cleaned = re.sub(r"\s*#\w+", "", text or "").strip()
    if _INTERNAL_TERMS.search(cleaned):
        return cleaned, f"internal term '{_INTERNAL_TERMS.search(cleaned).group(0)}'"
    if arabic and cleaned:
        letters = [ch for ch in cleaned if ch.isalpha()]
        arabic_n = sum(1 for ch in letters if _ARABIC_LETTER.match(ch))
        other = [ch for ch in letters if not _ARABIC_LETTER.match(ch) and not _LATIN_LETTER.match(ch)]
        latin_words = [w for w in re.findall(r"[A-Za-z]{2,}", cleaned) if w not in (config.OPERATOR_NAME, "OMR", "GB", "MB", "YES", "HELP", "SMS")]
        if other or latin_words or (letters and arabic_n / len(letters) < 0.6):
            return cleaned, "Arabic SMS mixes in other words or scripts"
    return cleaned, None


def compliance_fix(msg: dict) -> tuple[dict, list[str]]:
    """Code-level check on the model's message: the currency must be OMR."""
    notes = []
    for key in ("sms_en", "sms_ar", "agent_script"):
        text = msg.get(key, "")
        for bad in _WRONG_CURRENCY:
            if bad in text:
                good = "OMR" if bad.isascii() else "ر.ع."
                text = text.replace(bad, good)
                notes.append(f"{key}: replaced wrong currency '{bad}' with '{good}'")
        msg[key] = text
    return msg, notes


def template_message(ctx: RunContext, action: str, offers: list[dict]) -> dict:
    if action == "NO_ACTION":
        lp = languages.plan(ctx.customer.get("nationality"))
        return {"sms_en": "", "sms_ar": "", "agent_script": "No outreach for this customer.", "sms_native": "",
                "native_language": None, "native_rtl": False, "languages": lp["languages"],
                "lead_language": lp["lead_language"], "nationality_segment": lp["segment"]}
    if offers:
        o = ctx.offers[offers[0]["offer"]]
        # "NBO" is an internal catalog category -- never show it to the customer.
        kind = {"NBO": "special"}.get(o["category"], o["category"].lower())
        en = f"{config.OPERATOR_NAME}: {'an' if kind[0] in 'aeiou' else 'a'} {kind} pack picked for you - {o['data_gb']} data, {o['voice_mins']} min for {o['price_omr']} OMR ({o['validity_days']} days). Reply YES to activate."
        ar = f"{config.OPERATOR_NAME}: باقة مختارة لك - {o['data_gb']} بيانات و{o['voice_mins']} دقيقة بسعر {o['price_omr']} ر.ع لمدة {o['validity_days']} يوم. أرسل نعم للتفعيل."
    else:
        en = f"{config.OPERATOR_NAME}: thank you for being with us. Need help with your plan? Reply HELP and we'll call you."
        ar = f"{config.OPERATOR_NAME}: شكراً لبقائك معنا. هل تحتاج مساعدة في باقتك؟ أرسل مساعدة وسنتصل بك."
    script = f"Thank the customer for {ctx.customer['tenure_days'] // 30} months with {config.OPERATOR_NAME}; action {action}; listen first, then present the offer."
    lp = languages.plan(ctx.customer.get("nationality"))
    native = lp["native"]
    if native:
        script += f" Offer to continue in {native['language']}."
    return {
        "sms_en": en, "sms_ar": ar, "agent_script": script,
        "sms_native": languages.template(native["code"], ctx.offers[offers[0]["offer"]] if offers else None) if native else "",
        "native_language": native["language"] if native else None,
        "native_rtl": bool(native and native["rtl"]),
        "languages": lp["languages"], "lead_language": lp["lead_language"], "nationality_segment": lp["segment"],
    }


async def run_messenger(ctx: RunContext, model: str | None, nba: dict, offers: list[dict], emit: Emit) -> dict:
    fallback = template_message(ctx, nba["action"], offers)
    if model is None or nba["action"] == "NO_ACTION":
        return fallback
    lp = languages.plan(ctx.customer.get("nationality"))
    native = lp["native"]
    native_llm = bool(native) and config.MESSENGER_NATIVE_SOURCE == "llm"
    offer_lines = [
        f"{ctx.offers[o['offer']]['description']} Price {ctx.offers[o['offer']]['price_omr']} OMR, "
        f"valid {ctx.offers[o['offer']]['validity_days']} days, data {ctx.offers[o['offer']]['data_gb']}, "
        f"voice {ctx.offers[o['offer']]['voice_mins']} min."
        for o in offers
    ]
    prompt = (
        f"ACTION: {nba['action']} via {nba.get('channel') or 'SMS'}.\n"
        f"CUSTOMER: tenure {ctx.customer['tenure_days'] // 30} months, {ctx.customer['revenue_segment']}, "
        f"likes {ctx.customer['favorite_category']}, risk {risk_level(ctx.score_row)}.\n"
        f"OFFER: {offer_lines[0] if offer_lines else 'none (relationship / retention message)'}\n"
        "Never mention internal offer codes. Write: an English SMS (max 160 characters), the same SMS in Arabic, and a 2-sentence call-center script in English.\n"
        + (f"The customer is {lp['segment']}: also write the same SMS in {native['language']} ({native['script']} script), "
           f"keeping the price in OMR.\n" if native_llm else "")
        + 'Reply: {"sms_en": "<the English SMS text>", "sms_ar": "<the Arabic SMS text>", '
        + (f'"sms_native": "<the {native["language"]} SMS text>", ' if native_llm else "")
        + '"agent_script": "<the English call script>"}'
    )
    t0 = time.time()
    res = await llm_client.generate(model=model, prompt=prompt, system=MESSENGER_ROLE, force_json=True)
    parsed = llm_client.extract_json(res.get("content") or "") if not res.get("error") else None
    if not isinstance(parsed, dict) or not all(_real_text(parsed.get(k)) for k in ("sms_en", "sms_ar")):
        emit({"type": "policy", "agent": "messenger", "text": "Model message unusable; used the compliant template."})
        return fallback
    msg = {**fallback, **{k: str(parsed.get(k) or fallback[k]).strip() for k in ("sms_en", "sms_ar", "agent_script")}}
    if len(msg["sms_en"]) > 200:
        msg["sms_en"] = msg["sms_en"][:197] + "..."
    msg, notes = compliance_fix(msg)
    for key in ("sms_en", "sms_ar", "agent_script"):
        cleaned, problem = customer_safe(msg[key], arabic=key == "sms_ar")
        if problem:
            msg[key] = fallback[key]
            emit({"type": "policy", "agent": "messenger", "text": f"{key} rejected ({problem}); used the compliant template."})
        elif cleaned != msg[key]:
            msg[key] = cleaned
            notes.append(f"{key}: removed hashtags")
    if native_llm:
        text = str(parsed.get("sms_native") or "").strip()
        lead = ctx.offers[offers[0]["offer"]] if offers else None
        price_ok = lead is None or str(lead["price_omr"]) in text
        if _real_text(text) and languages.script_ok(text, native["code"]) and price_ok:
            text, cur = languages.fix_currency(text)
            notes += [f"sms_native: {n}" for n in cur]
            if "OMR" in text or lead is None:
                msg["sms_native"] = text
            else:
                emit({"type": "policy", "agent": "messenger", "text": f"{native['language']} SMS lost the OMR price; used the {native['language']} template."})
        else:
            emit({"type": "policy", "agent": "messenger",
                  "text": f"{native['language']} SMS failed the script/price check; used the {native['language']} template."})
    elif native:
        emit({"type": "thought", "agent": "messenger", "text": f"{native['language']} SMS taken from the reviewed template (MESSENGER_NATIVE_SOURCE=template)."})
    for n in notes:
        emit({"type": "auto_fix", "agent": "messenger", "text": f"Compliance: {n}"})
    langs = " + ".join(lp["languages"])
    emit({"type": "thought", "agent": "messenger", "text": f"Drafted SMS in {langs} (lead: {lp['lead_language']}) and the call script.",
          "llm_ms": int((time.time() - t0) * 1000)})
    return msg


EXPLAINER_ROLE = (
    "You explain AI decisions to a telecom customer-value-management manager in Oman. "
    "Use only the facts given, quote numbers exactly, currency is OMR. Reply with JSON only."
)


async def run_explainer(model: str | None, expl: dict, emit: Emit) -> tuple[str, bool]:
    """Turns the computed explanation into a short narrative; template fallback."""
    fallback = explain.template_narrative(expl)
    if model is None:
        return fallback, False
    t0 = time.time()
    res = await llm_client.generate(model=model, prompt=explain.narrative_prompt(expl), system=EXPLAINER_ROLE, force_json=True)
    parsed = llm_client.extract_json(res.get("content") or "") if not res.get("error") else None
    text = str((parsed or {}).get("narrative") or "").strip()
    if not _real_text(text) or len(text) < 80:
        emit({"type": "policy", "agent": "messenger", "text": "Narrative unusable; used the evidence template."})
        return fallback, True
    fixed, _ = compliance_fix({"sms_en": text, "sms_ar": "", "agent_script": ""})
    text = fixed["sms_en"]
    emit({"type": "thought", "agent": "messenger", "text": "Wrote the decision rationale (why this action, why this offer, vs competitors).",
          "llm_ms": int((time.time() - t0) * 1000)})
    return text, True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _resolve_model(llm: str | None) -> str | None:
    if llm_client.is_offline() or not await llm_client.is_available():
        return None
    return await llm_client.resolve_model(llm or config.DEFAULT_LLM)


async def _run_agent(spec: AgentSpec, ctx: RunContext, model: str | None, emit: Emit, extra_context: str = "",
                     feedback: str | None = None, prior: list[dict] | None = None, max_steps: int | None = None) -> tuple[AgentResult, str | None]:
    """Runs one specialist; returns (result, model) where model becomes None
    if the LLM failed mid-run so later agents go straight to fallback."""
    if model is not None:
        try:
            return await run_react(spec, ctx, emit, model, extra_context, feedback, prior, max_steps), model
        except LLMUnavailable as e:
            emit({"type": "policy", "agent": spec.key, "text": f"LLM unavailable ({e}); switching to deterministic policy."})
            return AgentResult(final=None, evidence=list(prior or []), finalized_by="fallback"), None
    return AgentResult(final=None, evidence=list(prior or []), finalized_by="fallback"), None


def _nba_context(ctx: RunContext, nba: dict) -> str:
    cats = ACTION_CATEGORIES.get(nba["action"], [])
    c = ctx.customer
    row = ctx.score_row or {}
    return (
        f"STEP 1 RESULT -- the NBA strategist chose action {nba['action']} (channel {nba.get('channel')}, "
        f"send {nba.get('send_window')}, confidence {nba.get('confidence')}). Its reasoning: {nba.get('rationale') or '-'} "
        f"Risk level {risk_level(row)}, churn {row.get('proba_churn', 0):.2f}, revenue-decrease {row.get('proba_rev_decr', 0):.2f}. "
        f"STEP 2 (your job): serve this action with offers from categories {cats}. Customer facts: average monthly spend {ctx.avg_rev_monthly:.1f} OMR, "
        f"ARPU {c['arpu']} OMR, {c['revenue_segment']}, {c['usage_type']}, data tier {c['data_usage_tier']}, "
        f"favorite category {c['favorite_category']}, last product {c['last_product_name']}, "
        f"international caller {c['is_intl_caller']}, roamer {c['is_roamer']}."
        + _market_line(ctx)
    )


def _market_line(ctx: RunContext) -> str:
    """Competitor plans at this customer's spend, retrieved for every NBO run so
    the specialist reasons with market prices even if it never calls the tool."""
    m = market_overview(ctx, limit=3)
    if not m["competitor_plans_at_this_spend"]:
        return ""
    plans = "; ".join(f"{p['operator']} {p['plan']} {p['monthly_equiv_omr']} OMR/month {p['data_gb'] or 'unlimited'} GB"
                      for p in m["competitor_plans_at_this_spend"])
    return f" COMPETITOR PLANS AT THIS SPEND: {plans}."


async def run_council(msisdn: int, scope: str = "full", llm: str | None = None, emit: Emit | None = None,
                      use_last_nba: bool = False) -> dict:
    """Agentic NBA always comes first; the offer step builds on its result.

    scope "nba":  NBA strategist -> critic -> messenger (action only).
    scope "nbo" / "full": NBA strategist -> NBO specialist (offers for that
                  action) -> critic -> messenger.
    use_last_nba: reuse this customer's last agentic NBA decision (made in the
                  last 30 min) instead of running the strategist again."""
    events: list[dict] = []

    def _emit(ev: dict) -> None:
        ev = {"t_ms": int((time.time() - start) * 1000), **ev}
        events.append(ev)
        if emit:
            emit(ev)

    start = time.time()
    ctx = RunContext.load(msisdn)
    model = await _resolve_model(llm)
    mode = "agentic" if model else "fallback"
    agents = ["nba", "critic", "messenger"] if scope == "nba" else ["nba", "nbo", "critic", "messenger"]
    handoff = last_nba_decision(msisdn) if use_last_nba and scope != "nba" else None
    _emit({"type": "run_start", "msisdn": msisdn, "scope": scope, "mode": mode, "model": model, "agents": agents})

    nba_hist: list[dict] = []
    nbo_hist: list[dict] = []
    nba: dict | None = None
    nbo: dict | None = None
    llm_calls = 0

    # --- Step 1: NBA strategist (always first) --------------------------------
    if handoff is not None:
        _emit({"type": "agent_start", "agent": "nba", "name": NBA_SPEC.name})
        nba = dict(handoff)
        _emit({"type": "policy", "agent": "nba", "text": f"Reusing the NBA decision reviewed in step 1: {nba['action']}."})
        _emit({"type": "agent_final", "agent": "nba", "output": nba, "finalized_by": "step 1 (reused)"})
    else:
        _emit({"type": "agent_start", "agent": "nba", "name": NBA_SPEC.name})
        res, model = await _run_agent(NBA_SPEC, ctx, model, _emit)
        nba_hist, llm_calls = res.evidence, llm_calls + res.llm_calls
        nba = res.final or policy_nba(ctx, nba_hist, _emit)
        contact = evidence_of(nba_hist, "contact_policy")
        if contact:
            nba["channel"] = nba.get("channel") or contact.get("recommended_channel")
            nba["send_window"] = nba.get("send_window") or contact.get("send_window_local")
        _emit({"type": "agent_final", "agent": "nba", "output": nba, "finalized_by": res.finalized_by if res.final else "policy"})
        _NBA_HANDOFF[msisdn] = (time.time(), dict(nba))
    ctx.action = nba["action"]

    # --- Step 2: NBO specialist, built on the NBA result -----------------------
    if scope in ("nbo", "full"):
        _emit({"type": "agent_start", "agent": "nbo", "name": NBO_SPEC.name})
        _emit({"type": "policy", "agent": "nbo",
               "text": f"Handoff from NBA: action {nba['action']} via {nba.get('channel')}; searching "
                       f"{', '.join(ACTION_CATEGORIES.get(nba['action'], [])) or 'no'} offers for it."})
        if nba["action"] == "NO_ACTION":
            nbo = {"offers": [], "rationale": "No action -- no offer."}
            _emit({"type": "policy", "agent": "nbo", "text": "Action is NO_ACTION; offer search skipped."})
            fin = "policy"
        else:
            res, model = await _run_agent(NBO_SPEC, ctx, model, _emit, extra_context=_nba_context(ctx, nba))
            nbo_hist, llm_calls = res.evidence, llm_calls + res.llm_calls
            nbo = res.final or policy_nbo(ctx, nba["action"], nbo_hist, _emit)
            fin = res.finalized_by if res.final else "policy"
        _emit({"type": "agent_final", "agent": "nbo", "output": nbo, "finalized_by": fin})

    # --- Critic + one revision round -----------------------------------------
    _emit({"type": "agent_start", "agent": "critic", "name": "Critic"})
    findings = guardrails(ctx, nba, nbo)
    _emit({"type": "guardrails", "agent": "critic", "findings": findings})
    try:
        verdict = await run_critic(ctx, model, nba, nbo, findings, _emit)
        llm_calls += 1 if model else 0
    except LLMUnavailable:
        model = None
        verdict = await run_critic(ctx, None, nba, nbo, findings, _emit)
    revised, fixes = False, []
    if verdict["verdict"] == "revise":
        revised = True
        hard_codes = {f["code"] for f in findings if f["severity"] == "hard"}
        feedback = verdict["feedback"] or "; ".join(f["message"] for f in findings)
        _emit({"type": "revision", "agent": "critic", "text": f"Sent back for revision: {feedback}"})
        if hard_codes & {"EXCLUDED_SEGMENT", "PUSH_TO_AT_RISK", "CONTACT_COOLDOWN"}:
            res, model = await _run_agent(NBA_SPEC, ctx, model, _emit, feedback=feedback, prior=nba_hist, max_steps=3)
            llm_calls += res.llm_calls
            if res.final:
                nba = {**nba, **res.final}
                ctx.action = nba["action"]
                _emit({"type": "agent_final", "agent": "nba", "output": nba, "finalized_by": "revision"})
        if nbo is not None and nba["action"] != "NO_ACTION":
            res, model = await _run_agent(NBO_SPEC, ctx, model, _emit, extra_context=_nba_context(ctx, nba), feedback=feedback, prior=nbo_hist, max_steps=3)
            llm_calls += res.llm_calls
            if res.final:
                nbo = res.final
                _emit({"type": "agent_final", "agent": "nbo", "output": nbo, "finalized_by": "revision"})
        remaining = guardrails(ctx, nba, nbo)
        if any(f["severity"] == "hard" for f in remaining):
            nba, nbo, fixes = auto_fix(ctx, nba, nbo, remaining, nbo_hist, _emit)
    elif nbo is not None and nba["action"] == "NO_ACTION":
        nbo = {"offers": [], "rationale": "No action -- no offer."}

    _emit({"type": "agent_final", "agent": "critic",
           "output": {"verdict": verdict["verdict"], "revised": revised, "auto_fixes": fixes},
           "finalized_by": "llm" if verdict.get("llm_verdict") else "rules"})

    # --- Messenger -------------------------------------------------------------
    offers = (nbo or {}).get("offers", [])
    _emit({"type": "agent_start", "agent": "messenger", "name": "Messenger"})
    message = await run_messenger(ctx, model, nba, offers, _emit)
    llm_calls += 1 if model and nba["action"] != "NO_ACTION" else 0
    explanation = explain.build(ctx, nba, offers)
    explanation["narrative"], used_llm = await run_explainer(model, explanation, _emit)
    llm_calls += 1 if used_llm else 0
    _emit({"type": "agent_final", "agent": "messenger", "output": message, "finalized_by": "llm" if model else "template"})

    # --- Assemble ------------------------------------------------------------
    detailed = []
    for o in offers:
        cat = ctx.offers[o["offer"]]
        ev = (ctx.cache.get("simulated", {}).get(o["offer"]) or expected_value(ctx, o["offer"]) or {})
        detailed.append({
            "offer": o["offer"], "offer_name": cat["offer_name"], "category": cat["category"],
            "price_omr": cat["price_omr"], "validity_days": cat["validity_days"],
            "data_gb": cat["data_gb"], "voice_mins": cat["voice_mins"], "why": o.get("why", ""),
            "p_accept": ev.get("p_accept"), "expected_value_omr": ev.get("expected_value_omr"),
        })
    tool_calls = sum(1 for e in events if e["type"] == "tool_call")
    decision = {
        "msisdn": msisdn,
        "scope": scope,
        "mode": mode if model or mode == "fallback" else "partial_fallback",
        "model": model,
        "action": nba["action"],
        "action_type": nba["action"].replace(" ", "_"),
        "is_push_action": nba["action"] in PUSH_ACTIONS,
        "channel": nba.get("channel"),
        "send_window": nba.get("send_window"),
        "confidence": nba.get("confidence"),
        "nba_rationale": nba.get("rationale", ""),
        "offers": detailed,
        "nbo_rationale": (nbo or {}).get("rationale", ""),
        "expected_value_omr": round(sum(d["expected_value_omr"] or 0 for d in detailed[:1]), 2),
        "risk_level": risk_level(ctx.score_row),
        "nba_segment": (ctx.score_row or {}).get("nba_segment"),
        "critic": {"verdict": verdict["verdict"], "issues": verdict["issues"], "revised": revised, "auto_fixes": fixes, "guardrail_findings": findings},
        "message": message,
        "explanation": explanation,
        "stats": {"llm_calls": llm_calls, "tool_calls": tool_calls, "elapsed_ms": int((time.time() - start) * 1000)},
    }
    _emit({"type": "run_end", "decision": decision})
    RECENT_RUNS.appendleft({k: decision[k] for k in ("msisdn", "scope", "mode", "action", "channel", "expected_value_omr", "risk_level")}
                           | {"offer": detailed[0]["offer"] if detailed else None, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {"decision": decision, "trace": events}
