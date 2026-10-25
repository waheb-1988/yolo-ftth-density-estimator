"""Pluggable business-rules layer that sits between NBO's raw recommendations
and the NBA decision that goes out to the customer.

A rule receives the NBO recommendation list plus the customer/context and
returns a (possibly filtered/reordered/annotated) recommendation list. Return
the input unchanged for a no-op. Each applied rule should report what it did
via RuleApplication so the NBA response stays auditable.

The two rules registered below are ported from the production Dataiku
LightFM NBO batch job (the monthly-upsell campaign script), not invented
here -- see their docstrings for the source logic:
  - _exclude_voice_for_upsell:    "Voice BTL will not be recommended" filter
  - _affordable_upsell_priority:  the 3-pass price-affordability cascade

Register further rules by appending Rule instances to RULES (or calling
register_rule at import time from wherever the next rule gets defined).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from nbx.core import config

RuleFn = Callable[[list[dict], dict, dict], list[dict]]


@dataclass
class Rule:
    name: str
    fn: RuleFn
    description: str = ""


def register_rule(rule: Rule) -> None:
    RULES.append(rule)


# ---------------------------------------------------------------------------
# Rule 1: voice bundles are never recommended.
#
# Ported verbatim from the production job:
#   interaction = interaction[~interaction.btl_name.isin(
#       items_feat[items_feat.service == "VOICE"].btl_name.unique())]
# There, it's applied once, upstream, to the training interactions (voice
# never even enters the model). Here NBO can legitimately detect a
# VOICE_UPSELL need and retrieve voice offers as context, so the exclusion is
# applied at the same point production applies its business rule: on the
# final recommendation list, not earlier in the pipeline.
# ---------------------------------------------------------------------------
def _exclude_voice_for_upsell(recommendations: list[dict], customer: dict, context: dict) -> list[dict]:
    return [r for r in recommendations if r["category"] != "Voice"]


# ---------------------------------------------------------------------------
# Rule 2: affordability-band upsell priority.
#
# Ported from the production job's calculate_rev() + 3-pass selection:
#   avg_rev_monthly = mean(rev_m1..m3) + std(rev_m1..m3)          (deliberately
#                                                                   generous estimate)
#   Pass 1 (tightest): avg_rev < price < avg_rev + 3   OMR
#   Pass 3 (loosest):  avg_rev < price                  (no ceiling)
#   anything <= avg_rev is never preferred
# Production collapses this into 1-2 final rows per customer (DATA/COMBO
# only, monthly only) via 3 literal cascading passes with drop_duplicates.
# This API's contract is a top-4, multi-category ranked list, so the same
# preference order is applied as a *reorder* of the existing 4 candidates
# (tightest band first, then any upsell, then non-upsells last) rather than
# a collapse to 1-2 rows -- same priority, adapted to this response shape.
# ---------------------------------------------------------------------------
_revenue_cache: dict[int, dict] | None = None


def _load_revenue() -> dict[int, dict]:
    global _revenue_cache
    if _revenue_cache is None:
        _revenue_cache = {}
        if config.REVENUE_HISTORY_FILE.exists():
            rows = json.loads(config.REVENUE_HISTORY_FILE.read_text(encoding="utf-8"))
            _revenue_cache = {row["msisdn"]: row for row in rows}
    return _revenue_cache


def _affordable_upsell_priority(recommendations: list[dict], customer: dict, context: dict) -> list[dict]:
    revenue = _load_revenue().get(customer.get("msisdn"), {})
    avg_rev = revenue.get("avg_rev_monthly", customer.get("avg_offer_price", 5.0))
    band = config.AFFORDABILITY_BAND_OMR

    def tier(r: dict) -> int:
        price = r["price_omr"]
        if avg_rev < price <= avg_rev + band:
            return 0  # production pass 1: tightest affordability band
        if price > avg_rev:
            return 1  # production pass 3: any upsell, no ceiling
        return 2  # at or below current spend -- never preferred

    reordered = sorted(recommendations, key=lambda r: (tier(r), -r["final_score"]))
    for i, r in enumerate(reordered, start=1):
        r["rank"] = i
    return reordered


RULES: list[Rule] = [
    Rule(
        "exclude_voice_for_upsell",
        _exclude_voice_for_upsell,
        "Voice bundles are never recommended (very low take-up rate) — ported from the production LightFM NBO job.",
    ),
    Rule(
        "affordable_upsell_priority",
        _affordable_upsell_priority,
        "Prioritizes offers priced just above the customer's estimated average spend "
        f"(avg_rev_monthly to avg_rev_monthly + {config.AFFORDABILITY_BAND_OMR} OMR), then any upsell, "
        "over anything at or below current spend — ported from the production job's 3-pass price-affordability cascade.",
    ),
]


def apply_rules(recommendations: list[dict], customer: dict, context: dict) -> tuple[list[dict], list[dict]]:
    """Runs every registered rule in order. Returns (final_recommendations,
    rules_applied) where rules_applied is [{"rule_name", "effect"}] — built
    from comparing offer_name sets before/after each rule so the effect is
    always accurate to what actually happened, not just the rule's own claim."""
    current = recommendations
    applied: list[dict] = []

    for rule in RULES:
        before_names = [r["offer_name"] for r in current]
        current = rule.fn(current, customer, context)
        after_names = [r["offer_name"] for r in current]

        if after_names == before_names:
            effect = "no change"
        else:
            removed = set(before_names) - set(after_names)
            effect_parts = []
            if removed:
                effect_parts.append(f"removed {', '.join(sorted(removed))}")
            if after_names != before_names:
                effect_parts.append("reordered/modified")
            effect = "; ".join(effect_parts) or "modified"

        applied.append({"rule_name": rule.name, "effect": effect})

    return current, applied
