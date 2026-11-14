"""⑦ Final output agent: code 5 (master §12, F-01 … F-19) -> input_flytxt, vectorised over all customers.

    flytxt, slots = build_input_flytxt(monthly, weekly, scored, src, run_date)

monthly / weekly  the reco tables of the groups (B-32 structure: access_method_val, btl_name, price, …)
scored            results_scoring_nba (access_method_val, action, proba, tg_flag): the population (F-01)
flytxt            the 38 lower-case columns of F-19, one row per MSISDN
slots             per customer and slot 1-6: offer, reason (recommended / favourite / forced) and the rule that
                  cleared it (F-09 / F-11 / F-12): the detail behind every row of the file

Code-5 issues, fixed (master §12): (a) a cleared slot also loses its script and opportunity name, (b) an empty slot
is an empty field (never nbo_LFM_nan), (c) favourites are one row per offer, by price then purchase count,
(d) the F-05 anchor is separate from B-26. The bug-reproduction replay for the parity test A-25 comes with Phase 7.
"""
from __future__ import annotations

from datetime import date, timedelta
from itertools import combinations

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.calendar import MONTHS, WEEKS
from nbx.fsdr.features import norm_val, upper
from nbx.fsdr.offer_rules import load_catalog

VAL = "ACCESS_METHOD_VAL"
SLOTS = range(1, 7)


def f05_anchors(src, vals: pd.Series | None = None) -> pd.DataFrame:
    """F-05: revenue = Total_Sub_Amt + PG_Total_Amount, zero periods treated as missing, anchor = min(mean + std,
    max); data 0.9 x. No non-zero period -> no anchor (NaN). Index: ACCESS_METHOD_VAL."""
    sub = upper(src.read("subscriptions_tg")).drop_duplicates(VAL).set_index(VAL)
    use = upper(src.read("usage_data_voice_payg_tg")).drop_duplicates(VAL).set_index(VAL)
    idx = sub.index.union(use.index) if vals is None else pd.Index(vals.unique())
    out = pd.DataFrame(index=idx)
    for name, periods in (("MONTHLY", MONTHS), ("WEEKLY", WEEKS)):
        rev = pd.DataFrame({p: pd.to_numeric(sub.get(f"TOTAL_SUB_AMT_{p}"), errors="coerce").reindex(idx).fillna(0.0)
                            + pd.to_numeric(use.get(f"PG_TOTAL_AMOUNT_{p}"), errors="coerce").reindex(idx).fillna(0.0)
                            for p in periods}).replace(0.0, np.nan)
        avg = rev.mean(axis=1) + rev.std(axis=1, ddof=1)
        out[f"F05_{name}"] = pd.concat([avg, rev.max(axis=1)], axis=1).min(axis=1)
        out[f"F05_DATA_{name}"] = config.ARPU_DATA_SHARE * out[f"F05_{name}"]
    return out


# F-02 / F-03 slots of the recommended offers by price rank. V4.0 with 6 offers: the third monthly offer goes to
# slot 5, the third weekly to slot 6 (slots 1-4 keep their V3.0 meaning).
MONTHLY_SLOTS = {1: 1, 2: 2, 3: 5}
WEEKLY_SLOTS = {1: 3, 2: 4, 3: 6}


def recommended_slots() -> list[int]:
    n = config.RECOS_PER_VALIDITY
    return sorted([MONTHLY_SLOTS[k] for k in range(1, n + 1)] + [WEEKLY_SLOTS[k] for k in range(1, n + 1)])


def _slots_from_recos(recos: pd.DataFrame, slots: dict[int, int]) -> pd.DataFrame:
    """F-02 / F-03: price rank (highest first, ties by first occurrence) -> slot of that rank."""
    slots = {k: s for k, s in slots.items() if k <= config.RECOS_PER_VALIDITY}
    if not len(recos):
        return pd.DataFrame(columns=[VAL] + [f"S{s}" for s in slots.values()]).set_index(VAL)
    r = recos.assign(_RANK=recos.groupby("access_method_val")["price"].rank(method="first", ascending=False))
    out = {}
    for k, s in slots.items():
        top = r[r["_RANK"] == k].drop_duplicates("access_method_val")
        out[f"S{s}"] = top.set_index("access_method_val")["btl_name"]
    return pd.DataFrame(out).rename_axis(VAL)


def favourites(src, vals: pd.Index, catalog: pd.DataFrame) -> pd.DataFrame:
    """F-06 (issue c fixed): the customer's 270-day catalogue purchases, one row per offer, price desc then count."""
    b = src.read("sub_btl_3months")
    df = pd.DataFrame({VAL: norm_val(b["access_method_val"]), "BTL_NAME": b["PRODUCTNAME"].astype(str).str.strip(),
                       "NB": pd.to_numeric(b.get("nb_sub"), errors="coerce").fillna(1)})
    df = df[df[VAL].isin(vals) & df["BTL_NAME"].isin(set(catalog["BTL_NAME"]))]
    df = df.groupby([VAL, "BTL_NAME"], as_index=False)["NB"].sum()
    df = df.merge(catalog[["BTL_NAME", "PRICE", "VALIDITY"]], on="BTL_NAME", how="left")
    return df.sort_values([VAL, "PRICE", "NB"], ascending=[True, False, False], kind="mergesort")


def build_input_flytxt(monthly: pd.DataFrame, weekly: pd.DataFrame, scored: pd.DataFrame, src, run_date: date
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = load_catalog(src)
    pop = pd.Index(scored["access_method_val"].astype(str).unique(), name=VAL)                       # F-01
    s = pd.DataFrame(index=pop).join(_slots_from_recos(monthly, MONTHLY_SLOTS)).join(_slots_from_recos(weekly, WEEKLY_SLOTS))
    monthly_slots = set(MONTHLY_SLOTS.values())
    for i in SLOTS:
        s[f"S{i}"] = s.get(f"S{i}")
        s[f"REASON_{i}"] = np.where(s[f"S{i}"].notna(), "recommended monthly" if i in monthly_slots
                                    else "recommended weekly", None)
    keep = s[[f"S{i}" for i in recommended_slots()]].notna().any(axis=1)
    dropped = s.index[~keep]                                                                           # F-04
    s = s[keep].copy()

    anchors = f05_anchors(src, pd.Series(s.index))
    fav = favourites(src, s.index, catalog)                                                             # F-06
    if config.FAVOURITE_ARPU_FILTER:                                                                    # F-07 (off)
        a = fav[VAL].map(anchors["F05_MONTHLY"]).where(fav["VALIDITY"] == 30, fav[VAL].map(anchors["F05_WEEKLY"]))
        fav = fav[fav["PRICE"] >= a]
    fav = fav.assign(_K=fav.groupby(VAL).cumcount() + 1)
    f1, f2 = (fav[fav["_K"] == k].set_index(VAL)["BTL_NAME"].reindex(s.index) for k in (1, 2))
    empty5, empty6 = s["S5"].isna(), s["S6"].isna()           # V4.0: a favourite only fills an EMPTY slot 5 / 6
    fill5 = empty5 & f1.notna()
    s.loc[fill5, "S5"], s.loc[fill5, "REASON_5"] = f1[fill5], "favourite"
    six = f2.where(fill5, f1)                                 # slot 5 took the first favourite -> the second here
    fill6 = empty6 & six.notna()
    s.loc[fill6, "S6"], s.loc[fill6, "REASON_6"] = six[fill6], "favourite"

    in_catalog = set(catalog["BTL_NAME"])
    s = s.join(anchors)
    for col, lo, hi, offer, popular in config.FORCED_BUNDLES:                                            # F-08
        if offer not in in_catalog:
            continue
        holds = s[[f"S{i}" for i in SLOTS]].isin(popular).any(axis=1)
        m = s[col].ge(lo) & s[col].lt(hi) & s["S6"].isna() & ~holds
        s.loc[m, "S6"] = offer
        s.loc[m, "REASON_6"] = f"forced F-08 {col} [{lo}, {hi})"

    for i in SLOTS:
        s[f"REMOVED_{i}"] = None
    info = catalog.set_index("BTL_NAME")[["TYPE", "PRICE"]]
    key = {i: s[f"S{i}"].map(info["TYPE"]).astype(str) + "|" + s[f"S{i}"].map(info["PRICE"]).astype(str) for i in SLOTS}
    has_type = {i: s[f"S{i}"].map(info["TYPE"]).notna() for i in SLOTS}
    for i in SLOTS:                                                                                    # F-09
        dup = pd.Series(False, index=s.index)
        for j in range(1, i):
            dup |= has_type[j] & has_type[i] & (key[j] == key[i]) & s[f"S{j}"].notna()
        _clear(s, i, dup & s[f"S{i}"].notna(), "F-09")
    for i in range(2, 7):                                                                              # F-11
        dup = pd.Series(False, index=s.index)
        for j in range(1, i):
            dup |= s[f"S{i}"].notna() & (s[f"S{i}"] == s[f"S{j}"])
        _clear(s, i, dup, "F-11")
    pairs = _conflict_pairs(src)                                                                       # F-12
    for i, j in combinations(SLOTS, 2):
        a, b = s[f"S{i}"], s[f"S{j}"]
        offer_a = (a.astype(str) + "|" + b.astype(str)).map(pairs)
        both = a.notna() & b.notna() & offer_a.notna()
        _clear(s, i, both & (offer_a == a), "F-12")
        _clear(s, j, both & (offer_a != a) & (offer_a == b), "F-12")

    flytxt = _format(s, scored, src, catalog, run_date)
    slots = s.reset_index().assign(IN_INPUT_FLYTXT=True)
    if len(dropped):
        slots = pd.concat([slots, pd.DataFrame({VAL: dropped, "IN_INPUT_FLYTXT": False, "DROP_RULE": "F-04"})],
                          ignore_index=True)
    return flytxt, slots


def _clear(s: pd.DataFrame, i: int, mask: pd.Series, rule: str) -> None:
    if mask.any():
        s.loc[mask, f"REMOVED_{i}"] = rule + ": " + s.loc[mask, f"S{i}"].astype(str)
        s.loc[mask, f"S{i}"] = None


def _conflict_pairs(src) -> pd.Series:
    """F-12: "a|b" and "b|a" -> Offer_A of the row; rows with Offer_Type_A = VOICE ignored (as code 5)."""
    if not src.has("conflicting_offers_h"):
        return pd.Series(dtype=object)
    c = src.read("conflicting_offers_h")
    c.columns = [str(x).lower() for x in c.columns]
    c = c[c["offer_type_a"].astype(str) != "VOICE"]
    a, b = c["offer_a"].astype(str), c["offer_b"].astype(str)
    pairs = pd.concat([pd.Series(a.to_numpy(), index=(a + "|" + b).to_numpy()),
                       pd.Series(a.to_numpy(), index=(b + "|" + a).to_numpy())])
    return pairs[~pairs.index.duplicated(keep="last")]           # later rows win, as the code-5 dict does


def _format(s: pd.DataFrame, scored: pd.DataFrame, src, catalog: pd.DataFrame, run_date: date) -> pd.DataFrame:
    """F-10, F-13 … F-19: scripts, segment, BTL taker flags, dates, opportunity names, the 38 columns."""
    scripts = src.read("BTL_script_english_arabic_ok")
    scripts.columns = [str(c).upper() for c in scripts.columns]
    scripts = scripts.drop_duplicates("DESTINATION_OFFER_NAME").set_index("DESTINATION_OFFER_NAME")
    action = scored.drop_duplicates("access_method_val").set_index("access_method_val")["action"]
    bought = src.read("sub_btl_3months")
    takers = set(norm_val(bought.loc[bought["PRODUCTNAME"].astype(str).str.strip().isin(set(catalog["BTL_NAME"])),
                                     "access_method_val"]).dropna())
    out = pd.DataFrame({"date": run_date.isoformat(), "start_date": (run_date + timedelta(days=1)).isoformat(),
                        "end_date": "", "msisdn": s.index.astype(str)}, index=s.index)
    for i in SLOTS:
        out[f"channel_{i}"] = ""
    for i in SLOTS:
        out[f"opportunity_name_{i}"] = (config.OPPORTUNITY_PREFIX + s[f"S{i}"].astype(str)).where(s[f"S{i}"].notna(), "")
    for i in SLOTS:
        out[f"destination_offer_name_{i}"] = s[f"S{i}"].fillna("")
    out["flag_splio"] = 1
    for lang, col in (("en", "ENGLISH SCRIPT"), ("ar", "ARABIC SCRIPT")):
        for i in SLOTS:
            out[f"offer_script_{lang}_{i}"] = s[f"S{i}"].map(scripts[col]).fillna("") if col in scripts.columns else ""
    out["segment"] = action.reindex(s.index).fillna("").to_numpy()                                   # F-14
    out["segment_btl_taker"] = np.where(s.index.isin(takers), "BTL_Taker", "No_BTL_Taker")            # F-15
    out["segment_btl_taker_upsell"] = np.where(out["segment"] == "UPSELL", "BTL_upsell", "")          # F-16
    return out[config.INPUT_FLYTXT_COLUMNS].drop_duplicates("msisdn").reset_index(drop=True)          # F-13, F-19
