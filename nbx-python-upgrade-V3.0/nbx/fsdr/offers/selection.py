"""The selections of the four groups (update §4). Input: one row per customer x candidate of ONE validity, with
ACCESS_METHOD_VAL, BTL_NAME, SERVICE, GENEROSITY, VALIDITY, PRICE, SCORE (the LLM NBO score, or its fallback),
ACTION, and ANCHOR (B-27, anchors.anchor_for). Output: the selected rows, at most 1 per type (service) per customer,
with PASS and SELECTED_BY. Sorting is stable (mergesort), so a tie keeps the candidate order.

select_code3      T1C, code 3 reco_btl3: k = 2 (MAINTAIN, RETAIN) / 3 (UPSELL and the voice / data actions)
select_code4      T1F, code 4 reco_btl2: k = 3, the action is not used
select_challenger T2C (CC-v1) / T2F (CF-v1): relative band on the challenger anchor (CC-03), score x price,
                  widened once upwards, then the matching T1 passes as fallback (logged)

price passes (B-28 … B-30 / FR-09 … FR-11):
  pass 1  anchor < price < anchor + k, per type: score desc, price desc
  pass 2  customers with one type: the missing type, price > anchor (no cap): score desc, price asc
  pass 3  customers with nothing: both types, price > anchor (no cap): score desc, price asc
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nbx.fsdr import config

VAL = "ACCESS_METHOD_VAL"


def _best(df: pd.DataFrame, price_asc: bool, by: list[str] | None = None) -> pd.DataFrame:
    """Best row per (customer, type): score desc, then price desc (pass 1) or asc (passes 2 and 3)."""
    s = df.sort_values([VAL, "SCORE", "PRICE"], ascending=[True, False, price_asc], kind="mergesort")
    return s.drop_duplicates(by or [VAL, "SERVICE"])


def price_passes(c: pd.DataFrame, k: pd.Series | float, rule_ids: tuple[str, str, str]) -> pd.DataFrame:
    """The 3 passes on all customers of c at once. k: per row (code 3, by action) or one value (code 4)."""
    c = c.assign(_K=k)
    p1 = _best(c[(c["PRICE"] > c["ANCHOR"]) & (c["PRICE"] < c["ANCHOR"] + c["_K"])], price_asc=False)
    types = p1.groupby(VAL)["SERVICE"].nunique()
    one = types[types == 1].index
    found = p1.drop_duplicates(VAL).set_index(VAL)["SERVICE"]
    miss = c[c[VAL].isin(one) & (c["SERVICE"] != c[VAL].map(found)) & (c["PRICE"] > c["ANCHOR"])]
    p2 = _best(miss, price_asc=True, by=[VAL])
    p3 = _best(c[~c[VAL].isin(p1[VAL]) & (c["PRICE"] > c["ANCHOR"])], price_asc=True)
    out = pd.concat([p1.assign(PASS=1, SELECTED_BY=rule_ids[0]), p2.assign(PASS=2, SELECTED_BY=rule_ids[1]),
                     p3.assign(PASS=3, SELECTED_BY=rule_ids[2])], ignore_index=True)
    return out.drop(columns="_K").sort_values([VAL, "PASS"], kind="mergesort").reset_index(drop=True)


def select_code3(c: pd.DataFrame) -> pd.DataFrame:
    """T1C: code 3 passes, k by the customer's action (B-28)."""
    k = c["ACTION"].map(config.PRICE_BAND_K).astype(float)
    return price_passes(c, k, ("B-28", "B-29", "B-30"))


def select_code4(c: pd.DataFrame) -> pd.DataFrame:
    """T1F: code 4 passes, k = 3 whatever the action (FR-09 … FR-11)."""
    return price_passes(c, config.FREE_PRICE_K, ("FR-09", "FR-10", "FR-11"))


def challenger_band(c: pd.DataFrame, band: tuple[float, float]) -> tuple[pd.Series, pd.Series]:
    """CC-02 / CF-02: [max(lo x anchor, anchor + 0.5 for UPSELL actions), hi x anchor] on the challenger anchor;
    never below the anchor (no downgrade); MAINTAIN, RETAIN and the cross-sell actions may sit at the anchor."""
    a = c["CH_ANCHOR"]
    step = np.where(c["ACTION"].isin(config.UPSELL_ACTIONS), config.CHALLENGER_UPSELL_STEP, 0.0) \
        if "ACTION" in c.columns else 0.0
    lo = np.maximum(band[0] * a, a + step)
    return pd.Series(lo, index=c.index), band[1] * a


def select_challenger(c: pd.DataFrame, fallback: str, prefix: str = "CC") -> pd.DataFrame:
    """T2C (prefix CC, fallback "code3") or T2F (prefix CF, fallback "code4"). c also needs CH_ANCHOR."""
    c = c.assign(EV=c["SCORE"] * c["PRICE"])                                   # C-04: expected value
    lo, hi = challenger_band(c, config.CHALLENGER_BAND)

    def best_ev(df: pd.DataFrame) -> pd.DataFrame:
        s = df.sort_values([VAL, "EV", "SCORE"], ascending=[True, False, False], kind="mergesort")
        return s.drop_duplicates([VAL, "SERVICE"])

    p1 = best_ev(c[(c["PRICE"] >= lo) & (c["PRICE"] <= hi)]).assign(PASS=1, SELECTED_BY=f"{prefix}-04")
    have = set(zip(p1[VAL], p1["SERVICE"]))
    _, hi_w = challenger_band(c, config.CHALLENGER_BAND_WIDENED)
    missing = ~pd.Series([k in have for k in zip(c[VAL], c["SERVICE"])], index=c.index)
    p2 = best_ev(c[missing & (c["PRICE"] >= lo) & (c["PRICE"] <= hi_w)]).assign(PASS=2, SELECTED_BY=f"{prefix}-04 widened")
    sel = pd.concat([p1, p2], ignore_index=True)
    rest = c[~c[VAL].isin(sel[VAL])]                                           # nothing even widened: T1 passes
    fb = (select_code3 if fallback == "code3" else select_code4)(rest.drop(columns=["EV"])) if len(rest) else rest.iloc[:0]
    if len(fb):
        fb = fb.assign(PASS=fb["PASS"] + 2, SELECTED_BY="fallback " + fb["SELECTED_BY"])
    out = pd.concat([sel.drop(columns=["EV"]), fb], ignore_index=True)
    return out.sort_values([VAL, "PASS"], kind="mergesort").reset_index(drop=True)


def select(group: str, c: pd.DataFrame) -> pd.DataFrame:
    """The selection of a group key (t1c / t1f / t2c / t2f)."""
    rules = config.GROUPS[group]["rules"]
    if rules == "code3":
        return select_code3(c)
    if rules == "code4":
        return select_code4(c)
    return select_challenger(c, "code3" if rules == "CC" else "code4", prefix=rules)
