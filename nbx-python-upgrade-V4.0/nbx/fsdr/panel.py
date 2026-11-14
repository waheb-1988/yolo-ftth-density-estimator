"""Panel (update §3, master §19.2): the eligible base assigned at random to T1C / T1F / T2C / T2F / control.

    panel, report = assign_panel(pool, sizes=config.group_sizes(), seed=7)

Stratified random, never by score:
  1. the pool (eligible, not Nursering, not in another campaign this week, not cooling down) is sorted by the
     `Final` stratum (nationality group x value band x usage type x plan), then age band, tenure band, segment,
     then a random key: neighbours in that order are alike on every balancing variable
  2. systematic sample of the 120k: one customer every pool / 120k positions, random start
  3. the sample is cut into blocks of 24 (= 5 : 5 : 5 : 5 : 4 for 25k x 4 + 20k); every block gets a random
     permutation of the 24 labels. So every stratum is split in the target proportions up to one block.
The provided test sample (sample_cg_tg_new_7oct2026) is used when given: cstr -> T1C, free -> T1F, cg -> control
(topped up from the same strata); T2C / T2F come from the rest of the pool the same way.
A pool smaller than the sizes is scaled down proportionally (reported).
"""
from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pandas as pd

from nbx.fsdr import config

VAL = "ACCESS_METHOD_VAL"
STRATUM_FIELDS = ["NATIONALITY_GROUP", "VALUE_BAND", "USAGE_TYPE", "CURRENT_PLAN"]     # the `Final` stratum
BALANCE_FIELDS = ["AGE_BAND", "TENURE_BAND", "SEGMENT"]
REPORT_FIELDS = ["NATIONALITY_GROUP", "VALUE_BAND", "USAGE_TYPE", "AGE_BAND", "TENURE_BAND", "SEGMENT"]


def panel_group(key: str) -> str:
    return config.CONTROL_GROUP if key == config.CONTROL_GROUP else config.GROUPS[key]["panel_group"]


def eligible_pool(segments: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """H-01 / B-02 / H-05: not excluded by the S-rules (Nursering), not in `exclude` (other campaigns, cool-down)."""
    pool = segments[segments["EXCLUDED"] == 0]
    if exclude:
        pool = pool[~pool[VAL].isin(exclude)]
    return pool.drop_duplicates(VAL).reset_index(drop=True)


def stratum(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in STRATUM_FIELDS if c in df.columns]
    return reduce(lambda a, b: a + "|" + b, [df[c].astype("string").fillna("-") for c in cols])


def _ordered(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    keys = ["STRATUM"] + [c for c in BALANCE_FIELDS if c in df.columns]
    return df.assign(_R=rng.random(len(df))).sort_values(keys + ["_R"], kind="mergesort").drop(columns="_R")


def _labels(sizes: dict[str, int], n: int, rng: np.random.Generator) -> np.ndarray:
    """n labels in the proportions of sizes: blocks of the smallest pattern, each block randomly permuted."""
    g = reduce(math.gcd, [v for v in sizes.values() if v > 0])
    pattern = np.repeat(list(sizes), [v // g for v in sizes.values()])
    blocks = math.ceil(n / len(pattern))
    out = np.concatenate([rng.permutation(pattern) for _ in range(blocks)])[:n]
    return out


def _draw(pool: pd.DataFrame, sizes: dict[str, int], rng: np.random.Generator) -> pd.DataFrame:
    """Systematic stratified sample of sum(sizes) from the ordered pool, labelled in proportion."""
    need = sum(sizes.values())
    ordered = _ordered(pool, rng)
    if need >= len(ordered):
        sample = ordered
    else:
        step = len(ordered) / need
        idx = (rng.random() * step + step * np.arange(need)).astype(int)
        sample = ordered.iloc[np.minimum(idx, len(ordered) - 1)]
    return sample.assign(GROUP_KEY=_labels(sizes, len(sample), rng))


def scale_sizes(sizes: dict[str, int], available: int) -> tuple[dict[str, int], float]:
    """A pool smaller than the panel: every group scaled by the same factor (multiples of the label pattern)."""
    total = sum(sizes.values())
    if available >= total:
        return dict(sizes), 1.0
    f = available / total
    return {k: int(v * f) for k, v in sizes.items()}, round(f, 6)


def assign_panel(pool: pd.DataFrame, sizes: dict[str, int] | None = None, seed: int = config.GENERATOR_SEED,
                 sample: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """pool: eligible lines with ACCESS_METHOD_VAL and the stratum / balance columns.
    sample: optional (ACCESS_METHOD_VAL, group) of the provided test sample."""
    rng = np.random.default_rng(seed)
    sizes, factor = scale_sizes(dict(sizes or config.group_sizes()), len(pool))
    pool = pool.assign(STRATUM=stratum(pool))
    fixed = pd.DataFrame(columns=[VAL, "GROUP_KEY"])
    if sample is not None and len(sample):
        s = sample.assign(GROUP_KEY=sample["group"].astype(str).str.lower().map(config.SAMPLE_GROUP_MAP))
        s = s.dropna(subset=["GROUP_KEY"])
        s = s[s[VAL].isin(pool[VAL]) & s["GROUP_KEY"].isin(list(sizes))].drop_duplicates(VAL)
        # never more than the configured size: a random subset of an over-full sample group
        s = s.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        s = s[s.groupby("GROUP_KEY").cumcount() < s["GROUP_KEY"].map(sizes)]
        fixed = s[[VAL, "GROUP_KEY"]]
    left = {k: v - int((fixed["GROUP_KEY"] == k).sum()) for k, v in sizes.items()}
    left = {k: v for k, v in left.items() if v > 0}
    drawn = _draw(pool[~pool[VAL].isin(fixed[VAL])], left, rng) if left else pool.iloc[:0].assign(GROUP_KEY=None)
    chosen = pd.concat([fixed.merge(pool, on=VAL, how="left"), drawn], ignore_index=True)
    chosen["PANEL_GROUP"] = chosen["GROUP_KEY"].map(panel_group)
    chosen["TARGET"] = chosen["GROUP_KEY"].map(lambda k: "control" if k == config.CONTROL_GROUP else config.GROUPS[k]["target"])
    chosen["SOURCE"] = np.where(chosen[VAL].isin(fixed[VAL]), "sample", "drawn")
    report = {"pool": len(pool), "requested": dict(sizes), "scale_factor": factor,
              "sizes": chosen["GROUP_KEY"].value_counts().reindex(list(sizes), fill_value=0).astype(int).to_dict(),
              "from_sample": int((chosen["SOURCE"] == "sample").sum()), "seed": seed,
              "t2_split": config.T2_SPLIT}
    return chosen.reset_index(drop=True), report


def balance(panel: pd.DataFrame, fields: list[str] | None = None) -> pd.DataFrame:
    """A-31: per field and value, the share (%) of each group, the max gap in points and a chi-square p-value."""
    from scipy.stats import chi2_contingency

    rows = []
    for f in fields or [c for c in REPORT_FIELDS if c in panel.columns]:
        tab = pd.crosstab(panel[f].astype("string").fillna("-"), panel["GROUP_KEY"])
        share = tab / tab.sum() * 100
        try:
            p = float(chi2_contingency(tab.to_numpy())[1]) if tab.shape[0] > 1 and tab.shape[1] > 1 else 1.0
        except ValueError:
            p = 1.0
        for value, r in share.iterrows():
            rows.append({"field": f, "value": value, **{f"share_{g}": round(v, 2) for g, v in r.items()},
                         "max_gap_points": round(float(r.max() - r.min()), 2), "chi2_p": round(p, 4)})
    return pd.DataFrame(rows)


def check_sizes(panel: pd.DataFrame, sizes: dict[str, int], tol: float = config.GROUP_SIZE_TOLERANCE) -> list[str]:
    """A-30: every group within +-tol of its size, no MSISDN in two groups."""
    problems = []
    got = panel["GROUP_KEY"].value_counts()
    for k, v in sizes.items():
        n = int(got.get(k, 0))
        if v and abs(n - v) > tol * v:
            problems.append(f"{k}: {n:,} lines, expected {v:,} (+-{tol:.0%})")
    dup = int(panel[VAL].duplicated().sum())
    if dup:
        problems.append(f"{dup} MSISDN(s) in two groups")
    return problems
