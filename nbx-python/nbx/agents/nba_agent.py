"""NBA (Next Best Action) agent -- pure NBA, symmetric to agents/nbo_agent.py:
a statistical signal (the contextual bandit, core/nba_bandit.py) followed by
an LLM reasoning step that can confirm or override it, exactly the pattern
NBO already uses (cf_score -> LLM). Self-contained -- does NOT call NBO.
Composition with NBO and the business-rules layer now lives one level up,
in agents/final_agent.py.

Flagging this plainly: the production Vowpal Wabbit script this bandit is
ported from has NO LLM anywhere -- it's pure bandit arithmetic. The
confirm-or-override LLM step below is a deliberate enhancement on top of
that source, not something extracted from it, applied here for the same
reason it was added to NBO's collaborative-filtering signal: a statistical
model's pick is evidence an LLM can reason over and override, not a verdict.

Customers with no simulated NBA score, or excluded ones (Nursering_* --
tenure < 45 days, ported straight from the production script's own
exclusion), get NO_ACTION and never reach the bandit or the LLM.
"""
from __future__ import annotations

import time

from nbx.core import config, nba_bandit
from nbx.llms import llm_client

SYSTEM_PROMPT = (
    "You are an NBA (Next Best Action) decision engine for Ooredoo Oman.\n"
    "A contextual bandit has already scored 7 candidate actions on real\n"
    "historical behavior and picked one. Sanity-check that pick against the\n"
    "customer's actual profile below and either confirm it or override it\n"
    "with a better action from the same list.\n"
    "Return ONLY valid JSON — no prose before or after."
)

OUTPUT_SCHEMA_HINT = """Return exactly this JSON shape (no other text):
{
  "action": "RETAIN" | "MAINTAIN" | "UPSELL" | "UPSELL VOICE" | "UPSELL DATA" | "CROSS SELL VOICE" | "CROSS SELL DATA",
  "reasoning": "<2-4 sentences explaining your decision, including whether you agreed with or overrode the bandit>"
}"""


def _build_prompt(ctx_row: dict, bandit_action: str, bandit_prob: float) -> str:
    return (
        f"CUSTOMER NBA CONTEXT:\n"
        f"  Segment: {ctx_row.get('nba_segment')}\n"
        f"  Historical action on record: {ctx_row.get('action')}\n"
        f"  Nationality group: {ctx_row.get('nationality_group')}\n"
        f"  Churn probability: {ctx_row.get('proba_churn', 0):.2f} (1-day-active: {ctx_row.get('proba_churn_1d', 0):.2f})\n"
        f"  Reactivation probability — 30d/15d: {ctx_row.get('proba_react_30', 0):.2f} / {ctx_row.get('proba_react_15', 0):.2f}\n"
        f"  Data/voice appetency: {ctx_row.get('proba_app_data', 0):.2f} / {ctx_row.get('proba_app_voice', 0):.2f}\n"
        f"  Data/voice fragility: {ctx_row.get('proba_frag_data', 0):.2f} / {ctx_row.get('proba_frag_voice', 0):.2f}\n"
        f"  Revenue-decrease probability: {ctx_row.get('proba_rev_decr', 0):.2f}\n"
        f"  Revenue trend (older→newer week, %): {ctx_row.get('var_rev_w1_w2', 0):.1f}\n\n"
        f"BANDIT'S CANDIDATE ACTION: {bandit_action} (confidence {bandit_prob:.2f})\n\n"
        f"{OUTPUT_SCHEMA_HINT}"
    )


def _action_type(action: str) -> str:
    return action.replace(" ", "_")


async def decide_action(msisdn: int, llm: str | None = None) -> dict:
    """Returns:
    {
        "msisdn", "action_type" (underscored, LLM-final), "bandit_action" (raw,
        space-separated, pre-LLM), "bandit_action_probability", "llm_reasoning",
        "llm_overrode_bandit", "is_push_action", "bandit_context", "step_times",
    }
    action_type == "NO_ACTION" for excluded/unsimulated customers -- the
    bandit and LLM never run for those."""
    llm_name = llm or config.DEFAULT_LLM
    ctx_row = nba_bandit.get_score_row(msisdn)

    if ctx_row is None or ctx_row.get("excluded"):
        return {
            "msisdn": msisdn,
            "action_type": "NO_ACTION",
            "bandit_action": None,
            "bandit_action_probability": None,
            "llm_reasoning": "Excluded from the NBA bandit (no simulated score, or Nursering segment).",
            "llm_overrode_bandit": False,
            "is_push_action": False,
            "bandit_context": None,
            "step_times": {"bandit_ms": 0, "llm_ms": 0},
        }

    t0 = time.time()
    ctx = {f: ctx_row.get(f, 0) for f in config.NBA_CONTEXT_FEATURES}
    bandit_action, bandit_prob = nba_bandit.get_action(ctx)
    bandit_ms = int((time.time() - t0) * 1000)

    t0 = time.time()
    resolved_model = await llm_client.resolve_model(llm_name)
    prompt = _build_prompt(ctx_row, bandit_action, bandit_prob)

    async def _call(p: str) -> tuple[dict, dict | None]:
        result = await llm_client.generate(model=resolved_model, prompt=p, system=SYSTEM_PROMPT, force_json=True)
        return result, llm_client.extract_json(result["content"])

    llm_result, parsed = await _call(prompt)
    llm_action = parsed.get("action") if isinstance(parsed, dict) else None
    retried = False

    # Same one-retry policy as agents/nbo_agent.py: format=json guarantees
    # syntax, not our schema -- one nudge before falling back to the
    # bandit's own pick.
    if not llm_result.get("error") and (not llm_action or llm_action not in config.NBA_ACTIONS):
        retried = True
        retry_prompt = (
            f"{prompt}\n\nYour previous response did not include a valid \"action\" from the exact list "
            f"above. Return ONLY the JSON object described above, with a valid \"action\"."
        )
        llm_result, parsed = await _call(retry_prompt)
        llm_action = parsed.get("action") if isinstance(parsed, dict) else None

    llm_reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
    final_action = llm_action if llm_action in config.NBA_ACTIONS else bandit_action

    if llm_result.get("error"):
        llm_reasoning = f"LLM call failed ({llm_result['error']}); using the bandit's own pick unconfirmed."
    elif final_action == bandit_action and not llm_reasoning:
        why_empty = "returned valid JSON with no valid \"action\"" if parsed is not None else "did not return valid JSON"
        llm_reasoning = (
            f"LLM {why_empty} after {'a retry' if retried else '1 attempt'}; using the bandit's own pick unconfirmed."
        )
    llm_ms = int((time.time() - t0) * 1000)

    return {
        "msisdn": msisdn,
        "action_type": _action_type(final_action),
        "final_action": final_action,  # space-separated form, for agents/final_agent.py's need_override lookup
        "bandit_action": bandit_action,
        "bandit_action_probability": round(bandit_prob, 4),
        "llm_reasoning": llm_reasoning,
        "llm_overrode_bandit": final_action != bandit_action,
        "is_push_action": final_action not in config.NBA_NON_PUSH_ACTIONS,
        "bandit_context": {
            "nba_segment": ctx_row.get("nba_segment"),
            "historical_action": ctx_row.get("action"),
        },
        "step_times": {"bandit_ms": bandit_ms, "llm_ms": llm_ms},
    }
