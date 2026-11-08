"""Offer universes (master §11 B-20b, B-21; update §4.1, §4.2 FR-01 … FR-06).

recommendable   in the current btl_catalog AND bought by at least one line of the base in 270 days (sub_btl_3months)
constraint      code 3: the action's two families (service, generosity), active mapping = L2 for all; CROSS SELL
                VOICE has its own filter (the code-3 typo that left it unfiltered is fixed)
free            code 4: L2 only, DATA and COMBO (no VOICE), the action is not used
"""
from __future__ import annotations

import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.features import norm_val
from nbx.fsdr.offer_rules import families

CAND_COLS = ["BTL_NAME", "TYPE", "SERVICE", "GENEROSITY", "VALIDITY", "PRICE", "GB", "MIN"]


def purchases(src) -> pd.DataFrame:
    """sub_btl_3months (270 days, no date): ACCESS_METHOD_VAL, BTL_NAME, NB_SUB."""
    if not src.has("sub_btl_3months"):
        return pd.DataFrame(columns=["ACCESS_METHOD_VAL", "BTL_NAME", "NB_SUB"])
    df = src.read("sub_btl_3months")
    out = pd.DataFrame({"ACCESS_METHOD_VAL": norm_val(df["access_method_val"]),
                        "BTL_NAME": df["PRODUCTNAME"].astype(str).str.strip(),
                        "NB_SUB": pd.to_numeric(df.get("nb_sub"), errors="coerce").fillna(1)})
    return out.dropna(subset=["ACCESS_METHOD_VAL"])


def recommendable(catalog: pd.DataFrame, bought: pd.DataFrame) -> pd.DataFrame:
    """B-20b / FR-05: catalogue offers bought by someone in the base in the last 270 days."""
    sold = set(bought.loc[bought["NB_SUB"] > 0, "BTL_NAME"])
    cols = [c for c in CAND_COLS if c in catalog.columns]
    return catalog.loc[catalog["BTL_NAME"].isin(sold), cols].reset_index(drop=True)


def constraint_universe(rec: pd.DataFrame, action: str, validity: int) -> pd.DataFrame:
    """T1C / T2C (CC-01): the action's families and generosity, one validity."""
    mask = pd.Series(False, index=rec.index)
    for svc, gen in families(action, "cstr"):
        mask |= (rec["SERVICE"] == svc) & (rec["GENEROSITY"] == gen)
    return rec[mask & (rec["VALIDITY"] == validity)]


def free_universe(rec: pd.DataFrame, validity: int) -> pd.DataFrame:
    """T1F / T2F (FR-01 … FR-06, CF-01): L2, DATA / COMBO, whatever the action."""
    m = ((rec["GENEROSITY"] == config.FREE_UNIVERSE_GENEROSITY) & rec["SERVICE"].isin(config.FREE_UNIVERSE_SERVICES)
         & (rec["VALIDITY"] == validity))
    return rec[m]


def _own_filter(cands: pd.DataFrame, own: pd.DataFrame | None) -> pd.DataFrame:
    """B-20b switch only_own_past_offers: slots 1-4 limited to the customer's own past offers."""
    if not config.ONLY_OWN_PAST_OFFERS or own is None:
        return cands
    keys = own[["ACCESS_METHOD_VAL", "BTL_NAME"]].drop_duplicates()
    return cands.merge(keys, on=["ACCESS_METHOD_VAL", "BTL_NAME"], how="inner")


def constraint_candidates(customers: pd.DataFrame, rec: pd.DataFrame, validity: int,
                          own: pd.DataFrame | None = None) -> pd.DataFrame:
    """customers: ACCESS_METHOD_VAL, ACTION -> one row per customer x offer of the action's universe."""
    parts = []
    for action, grp in customers.groupby("ACTION", sort=False):
        uni = constraint_universe(rec, action, validity)
        if len(uni):
            parts.append(grp[["ACCESS_METHOD_VAL", "ACTION"]].merge(uni, how="cross"))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ACCESS_METHOD_VAL", "ACTION", *CAND_COLS])
    return _own_filter(out, own)


def free_candidates(customers: pd.DataFrame, rec: pd.DataFrame, validity: int,
                    own: pd.DataFrame | None = None) -> pd.DataFrame:
    """customers: ACCESS_METHOD_VAL (+ ACTION, carried but not used) -> customer x free-universe offer."""
    cols = ["ACCESS_METHOD_VAL"] + (["ACTION"] if "ACTION" in customers.columns else [])
    out = customers[cols].merge(free_universe(rec, validity), how="cross")
    return _own_filter(out, own)
