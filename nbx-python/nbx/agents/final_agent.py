"""Final Agent -- composes NBA and NBO, the only thing that knows about
both. Each of the two is a self-contained peer (its own statistical signal,
its own LLM reasoning step, callable and testable alone):

    agents/nba_agent.py  -- bandit + LLM -> action
    agents/nbo_agent.py  -- retrieval/CF/graph + LLM -> ranked offers

Sequential composition, decided explicitly (not the alternative of running
them independently and merely comparing notes): NBA runs first, and when its
final action names a category (UPSELL DATA / CROSS SELL VOICE / ...), that
steers NBO's need signal for this one request (agents/nbo_agent.py's
need_override) -- the same category-steering behavior the old
agents/nba_agent.py had when it called NBO directly, just relocated here now
that NBA no longer calls NBO itself. NBO's recommendations then go through
the business-rules layer (agents/rules_engine.py) -- unchanged from before.
"""
from __future__ import annotations

import json

from nbx.core import config
from nbx.agents import nba_agent
from nbx.agents.nbo_agent import run_pipeline
from nbx.agents.rules_engine import apply_rules

# Actions that name a specific category steer NBO's need signal for this
# request. RETAIN / MAINTAIN / UPSELL name no category -- "upsell something"
# without a stated category isn't NBA's call to make, so NBO's own 8-signal
# detect_need() is left alone for those.
ACTION_NEED_OVERRIDE = {
    "UPSELL DATA": "DATA_UPSELL",
    "CROSS SELL DATA": "DATA_UPSELL",
    "UPSELL VOICE": "VOICE_UPSELL",
    "CROSS SELL VOICE": "VOICE_UPSELL",
}


def _get_customer(msisdn: int) -> dict:
    customers = json.loads(config.CUSTOMERS_FILE.read_text(encoding="utf-8"))
    return next((c for c in customers if c["msisdn"] == msisdn), {})


async def decide_next_best_action(
    msisdn: int,
    llm: str | None = None,
    vectorstore: str | None = None,
    embedding_model: str | None = None,
    retrieval_mode: str | None = None,
) -> dict:
    nba_result = await nba_agent.decide_action(msisdn, llm=llm)

    if nba_result["action_type"] == "NO_ACTION":
        nbo_result = await run_pipeline(
            msisdn=msisdn, llm=llm, vectorstore=vectorstore, embedding_model=embedding_model,
            retrieval_mode=retrieval_mode,
        )
        return {
            "msisdn": msisdn,
            "action_type": "NO_ACTION",
            "is_push_action": False,
            "bandit_action_probability": None,
            "nba_llm_reasoning": nba_result["llm_reasoning"],
            "bandit_context": None,
            "rules_applied": [],
            "nbo": nbo_result,
        }

    need_override = ACTION_NEED_OVERRIDE.get(nba_result["final_action"])

    nbo_result = await run_pipeline(
        msisdn=msisdn, llm=llm, vectorstore=vectorstore, embedding_model=embedding_model,
        retrieval_mode=retrieval_mode, need_override=need_override,
    )

    context = {
        "detected_need": nbo_result["detected_need"],
        "channel_suggestion": nbo_result["channel_suggestion"],
        "bandit_action": nba_result["final_action"],
    }
    customer = _get_customer(msisdn)

    final_recommendations, rules_applied = apply_rules(nbo_result["recommendations"], customer, context)
    nbo_result["recommendations"] = final_recommendations

    return {
        "msisdn": msisdn,
        "action_type": nba_result["action_type"],
        "is_push_action": nba_result["is_push_action"],
        "bandit_action_probability": nba_result["bandit_action_probability"],
        "nba_llm_reasoning": nba_result["llm_reasoning"],
        "bandit_context": {
            "nba_segment": nba_result["bandit_context"]["nba_segment"],
            "historical_action": nba_result["bandit_context"]["historical_action"],
            "need_override_applied": need_override,
            "llm_overrode_bandit": nba_result["llm_overrode_bandit"],
        },
        "rules_applied": rules_applied,
        "nbo": nbo_result,
    }
