"""⑧ weekly report of the 2 x 2 design (update §7): each group vs control, and the effects with 95 % CIs.

    effects = two_by_two(outcomes, "reward")    # outcomes: GROUP_KEY + one row per line (control included)

rules effect     T2C - T1C (constraint universe), T2F - T1F (free universe)
universe effect  T1C - T1F (production rules), T2C - T2F (challenger rules)
interaction      (T2C - T1C) - (T2F - T1F)
CIs: normal approximation of a difference of means, SE = sqrt(sum var / n).
"""
from __future__ import annotations

import math

import pandas as pd

from nbx.fsdr import config

Z = 1.959964


def _stats(outcomes: pd.DataFrame, metric: str) -> dict[str, tuple[float, float, int]]:
    g = outcomes.groupby("GROUP_KEY")[metric]
    return {k: (float(s.mean()), float(s.var(ddof=1)) if len(s) > 1 else 0.0, int(len(s))) for k, s in g}


def _effect(name: str, terms: list[tuple[int, str]], st: dict) -> dict:
    if any(k not in st for _, k in terms):
        return {"effect": name, "estimate": None, "ci_low": None, "ci_high": None}
    est = sum(sign * st[k][0] for sign, k in terms)
    se = math.sqrt(sum(st[k][1] / max(st[k][2], 1) for _, k in terms))
    return {"effect": name, "estimate": round(est, 6), "ci_low": round(est - Z * se, 6), "ci_high": round(est + Z * se, 6),
            "significant": (est - Z * se) > 0 or (est + Z * se) < 0}


def two_by_two(outcomes: pd.DataFrame, metric: str) -> pd.DataFrame:
    st = _stats(outcomes, metric)
    c = config.CONTROL_GROUP
    rows = [_effect(f"{k} vs control", [(1, k), (-1, c)], st) for k in config.GROUPS]
    rows += [_effect("rules effect, constraint (T2C - T1C)", [(1, "t2c"), (-1, "t1c")], st),
             _effect("rules effect, free (T2F - T1F)", [(1, "t2f"), (-1, "t1f")], st),
             _effect("universe effect, production (T1C - T1F)", [(1, "t1c"), (-1, "t1f")], st),
             _effect("universe effect, challenger (T2C - T2F)", [(1, "t2c"), (-1, "t2f")], st),
             _effect("interaction (T2C - T1C) - (T2F - T1F)", [(1, "t2c"), (-1, "t1c"), (-1, "t2f"), (1, "t1f")], st)]
    return pd.DataFrame(rows).assign(metric=metric)
