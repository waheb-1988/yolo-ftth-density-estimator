"""S-rules (master prompt §7): the current segment of every line, first match wins, thresholds in config.

    seg = apply_s_rules(features)      # adds RULE_ID, SEGMENT, RULE_ACTION, EXCLUDED and the trigger values

B-05: a missing score never triggers "at risk" (nor reconnect, fragility or appetency): the triggers read a score
only where its *_MISSING flag is 0, so the imputed median cannot move a line into a risk segment.
An unknown tenure is not "new": it never matches S-1 … S-4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nbx.fsdr import config

TRIGGERS = ["S_REV_M1", "S_RGS", "S_VOICE_USER", "S_DATA_USER", "S_AT_RISK", "S_RECONNECT_P", "S_FRAGILITY",
            "S_REV_M1_M2_PCT"]


def _observed(df: pd.DataFrame, feature: str) -> pd.Series:
    """The score where it was observed, 0 where it was missing (or absent)."""
    if feature not in df.columns:
        return pd.Series(0.0, index=df.index)
    s = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    flag = f"{feature}{config.MISSING_FLAG_SUFFIX}"
    return s.where(df[flag] == 0, 0.0) if flag in df.columns else s


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)


def apply_s_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a frame aligned on df with the segment columns and the values that triggered them."""
    c = config
    tenure = pd.to_numeric(df["TENURE_DAYS"], errors="coerce")
    churn, churn_1d = _observed(df, "PROBA_CHURN"), _observed(df, "PROBA_CHURN_1D")
    rev_m1 = _col(df, "ARPU_M1")
    rgs = (rev_m1 > 0) & (_col(df, "ACTIVE_DAYS_COUNT_M1") > 0)
    voice = _col(df, "LOCAL_VOICE_MIN_M1") >= c.S_VOICE_USER_MIN
    data = _col(df, "DATA_VOLUME_MB_M1") >= c.S_DATA_USER_MB
    reconnect = np.maximum(_observed(df, "PROBA_REACT_30"), _observed(df, "PROBA_REACT_15"))
    fragility = np.maximum.reduce([_observed(df, "PROBA_FRAG_DATA"), _observed(df, "PROBA_FRAG_VOICE"),
                                   _observed(df, "PROBA_REV_DECR")])
    rev_change = _col(df, "VAR_REV_M1_M2")
    at_risk = churn >= c.S_CHURN_AT_RISK
    new_45 = (tenure < c.S_TENURE_NURSERING).fillna(False)
    new_90 = (tenure < c.S_TENURE_ONBOARDING).fillna(False)
    conds = [
        new_45 & at_risk, new_45,                                                       # S-1, S-2
        new_90 & at_risk, new_90,                                                       # S-3, S-4
        ~rgs & (reconnect >= c.S_RECONNECT_HIGH), ~rgs & (reconnect >= c.S_RECONNECT_MED), ~rgs,   # S-5 … S-7
        at_risk | (churn_1d >= c.S_CHURN_1D_AT_RISK),                                   # S-8 (RGS from here)
        (fragility >= c.S_FRAGILITY) | (rev_change <= c.S_REV_DROP_PCT),                # S-9
        voice & data,                                                                   # S-10
        data & (_observed(df, "PROBA_APP_VOICE") >= c.S_APPETENCY), data,               # S-11, S-12
        _observed(df, "PROBA_APP_DATA") >= c.S_APPETENCY,                               # S-13 (voice only or none)
    ]
    idx = np.select([np.asarray(x, dtype=bool) for x in conds], list(range(len(conds))), default=len(conds))  # S-14
    rules = c.S_RULES
    out = pd.DataFrame(index=df.index)
    out["RULE_ID"] = np.array([r[0] for r in rules], dtype=object)[idx]
    out["SEGMENT"] = np.array([r[1] for r in rules], dtype=object)[idx]
    out["RULE_ACTION"] = np.array([r[2] for r in rules], dtype=object)[idx]
    out["EXCLUDED"] = out["RULE_ACTION"].isna().astype("int8")
    out["SEGMENT_SOURCE"] = "s_rules"
    out["S_REV_M1"] = rev_m1.round(4)
    out["S_RGS"] = rgs.astype("int8")
    out["S_VOICE_USER"] = voice.astype("int8")
    out["S_DATA_USER"] = data.astype("int8")
    out["S_AT_RISK"] = at_risk.astype("int8")
    out["S_RECONNECT_P"] = np.round(reconnect, 6)
    out["S_FRAGILITY"] = np.round(fragility, 6)
    out["S_REV_M1_M2_PCT"] = rev_change.round(4)
    return out


def segment_family(segment: pd.Series) -> pd.Series:
    """Nursering / Onboarding / Retention / RGS (for the tenure-agreement report)."""
    s = segment.astype("string")
    return pd.Series(np.select([s.str.startswith("Nursering"), s.str.startswith("Onboarding"),
                                s.str.startswith("Retention")], ["Nursering", "Onboarding", "Retention"], "RGS"),
                     index=segment.index)
