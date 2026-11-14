"""ARPU anchors (master §11 B-25 … B-27, update CC-03 / CF-03).

B-26   anchor = mean + std (ddof 1) of (Total_Sub_Amt + PG_Total_Amount), monthly M1-M3 / weekly W1-W4;
       data = 0.9 x, voice = 0.1 x  (offer_rules.arpu_anchors, shared with V1.0)
B-27   COMBO -> total anchor, DATA -> data anchor, VOICE -> voice anchor; weekly offers use the weekly anchors
CC-03  challenger split: data / voice from the customer's own data vs voice subscription spend (no history: 90 / 10)
C-02   a customer with no spend gets the median anchor of his micro-segment (challenger bands only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.calendar import MONTHS
from nbx.fsdr.features import upper
from nbx.fsdr.offer_rules import arpu_anchors

b26_anchors = arpu_anchors


def data_share(src) -> pd.Series:
    """ACCESS_METHOD_VAL -> share of data in the customer's data + voice subscription spend (M1-M3)."""
    if not src.has("subscriptions_tg"):
        return pd.Series(dtype=float)
    df = upper(src.read("subscriptions_tg"))
    d = sum(pd.to_numeric(df.get(f"DATA_SUB_AMT_{m}"), errors="coerce").fillna(0.0) for m in MONTHS)
    v = sum(pd.to_numeric(df.get(f"VOICE_SUB_AMT_{m}"), errors="coerce").fillna(0.0) for m in MONTHS)
    share = pd.Series(np.where((d + v) > 0, d / (d + v).where((d + v) > 0, 1.0), np.nan), index=df["ACCESS_METHOD_VAL"])
    return share[~share.index.duplicated(keep="first")]


def _period(validity) -> np.ndarray:
    return np.where(pd.to_numeric(validity) == 7, "WEEKLY", "MONTHLY")


def anchor_for(cands: pd.DataFrame, anchors: pd.DataFrame) -> pd.Series:
    """B-27 anchor of every candidate row (by its service and validity)."""
    period = _period(cands["VALIDITY"])
    svc = cands["SERVICE"].astype(str).str.upper().to_numpy()
    kind = np.select([svc == "DATA", svc == "VOICE"], ["DATA_", "VOICE_"], "")
    col = np.array([f"AVG_REV_{k}{p}" for k, p in zip(kind, period)], dtype=object)
    out = np.zeros(len(cands))
    vals = cands["ACCESS_METHOD_VAL"].to_numpy()
    for c in set(col):
        m = col == c
        if c in anchors.columns:
            out[m] = anchors[c].reindex(vals[m]).fillna(0.0).to_numpy()
    return pd.Series(out, index=cands.index)


def challenger_anchor(cands: pd.DataFrame, anchors: pd.DataFrame, share: pd.Series,
                      micro_median: pd.Series | None = None) -> pd.Series:
    """CC-03 / CF-03: total anchor for COMBO, own data share x total for DATA, (1 - share) x total for VOICE.
    C-02: a zero anchor is replaced by the micro-segment median (micro_median: ACCESS_METHOD_VAL -> median total)."""
    period = _period(cands["VALIDITY"])
    vals = cands["ACCESS_METHOD_VAL"].to_numpy()
    total = np.where(period == "WEEKLY", anchors["AVG_REV_WEEKLY"].reindex(vals).fillna(0.0).to_numpy(),
                     anchors["AVG_REV_MONTHLY"].reindex(vals).fillna(0.0).to_numpy())
    if micro_median is not None:
        med = micro_median.reindex(vals).to_numpy(dtype=float)
        total = np.where((total <= 0) & ~np.isnan(med), med, total)
    s = share.reindex(vals).fillna(config.DEFAULT_DATA_SHARE).to_numpy(dtype=float)
    svc = cands["SERVICE"].astype(str).str.upper().to_numpy()
    return pd.Series(np.select([svc == "DATA", svc == "VOICE"], [s * total, (1 - s) * total], total), index=cands.index)
