"""Level-2 LLM queue and the weekly budget (master §19.1a, update §6).

    q = llm_queue(decisions)                          # priority tier first, the 4 groups interleaved inside a tier
    q = apply_budget(q, seconds_per_customer=0.4)     # LEVEL = level2_llm | cell_policy_timebox

Each contacted customer costs one NBA + one NBO call (for T1F / T2F both run in the same wave: the offers do not
depend on the action). Past the budget (FSDR_LLM_WEEKLY_BUDGET_H, default 10 h) the rest get their cell policy
applied by code, logged per group. Interleaving keeps the coverage of the 4 groups equal (+-2 %).
"""
from __future__ import annotations

import pandas as pd

from nbx.fsdr import config

PRIORITY_TIERS = [   # master §19.1a
    ["Retention_RGS_At_Risk", "Retention_High_Probability_To_Reconnect", "Retention_Med_Probability_To_Reconnect"],
    ["Retention_RGS_Fragile", "Onboarding_Early_Churn_At_Risk", "Onboarding_Early_Churn_Not_At_Risk"],
    ["Data_Only_Voice_Appetency", "Data_Only_No_Voice_Appetency", "Voice_Only_Data_Appetency", "Voice_Only_No_Data_Appetency"],
    ["RGS_Not_At_Risk_DualPlay_Upgrade", "Retention_Low_Probability_To_Reconnect"],
]
TIER = {s: i + 1 for i, tier in enumerate(PRIORITY_TIERS) for s in tier}


def llm_queue(decisions: pd.DataFrame) -> pd.DataFrame:
    """Order of the level-2 calls: inside each group, priority tier first; across groups, round-robin (the k-th
    customer of every group, then the k+1-th), so a budget cut leaves the 4 groups with the same coverage."""
    q = decisions[decisions["GROUP_KEY"] != config.CONTROL_GROUP].copy()
    q["TIER"] = q["SEGMENT"].map(TIER).fillna(len(PRIORITY_TIERS) + 1).astype(int)
    q = q.sort_values(["TIER", "SEGMENT", "ACCESS_METHOD_VAL"], kind="mergesort")
    q["_K"] = q.groupby("GROUP_KEY").cumcount()
    order = {k: i for i, k in enumerate(config.GROUPS)}
    q = q.sort_values(["_K", "GROUP_KEY"], key=lambda s: s.map(order) if s.name == "GROUP_KEY" else s,
                      kind="mergesort")
    return q.drop(columns="_K").reset_index(drop=True)


def apply_budget(q: pd.DataFrame, seconds_per_customer: float, budget_h: float | None = None) -> pd.DataFrame:
    budget_s = 3600.0 * (config.LLM_WEEKLY_BUDGET_H if budget_h is None else budget_h)
    n = int(budget_s // max(seconds_per_customer, 1e-9))
    q = q.copy()
    q["LEVEL"] = "cell_policy_timebox"
    q.loc[q.index[:n], "LEVEL"] = "level2_llm"
    return q


def coverage(q: pd.DataFrame) -> pd.Series:
    """Share of each group decided by the level-2 LLM."""
    return q.groupby("GROUP_KEY")["LEVEL"].apply(lambda s: round(float((s == "level2_llm").mean()), 4))
