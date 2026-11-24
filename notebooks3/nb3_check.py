"""Cells of notebooks3/05_check_flytxt_results.ipynb: every check of the final file (input_flytxt) and of the scored
360 view of a run, each one as a DataFrame, and a scorecard at the end. Read only: nothing is written, the engine is
not used (pandas only). No MSISDN is printed: the examples show the row number of the final file (LINE)."""

TITLE = r'''# 05 · Check the results: the final file (input_flytxt) and the scored 360 view, rule by rule

Run it after `04_agents_nba_fast_ollama_real25k` (or 03). It **only reads** the run's files and shows every check as a
table (a DataFrame you can filter):

| Part | What is checked |
|---|---|
| 1 | what was scored: target lines, lines in the file, why the others were dropped |
| 2 | the file itself: 38 columns, one row per MSISDN, dates, 6 offers per line |
| 3 | every offer: in the catalogue, both scripts, the opportunity name |
| 4 | the code-5 rules: no repeated offer (F-11), no two offers of the same type and price (F-09), no conflict pair (F-12) |
| 5 | daily / weekly / monthly: the slots (1, 2, 5 = 30 days; 3, 4, 6 = 7 days) and the mix per line |
| 6 | why each offer is there: recommended, favourite, forced bundle, top-up; and the favourites really bought |
| 7 | **same price, different allocation** (e.g. 5 R for 15 GB and 5 R for 20 GB in one line) and **dominated offers** (the same money or more for less) |
| 8 | the price order of the slots and the price against the customer's usual spend (cannibalization) |
| 8b | **no downgrade**: every offer against the customer's ARPU, recomputed from the raw tables (total and per service) |
| 9 | the actions: per segment, upsell to an at-risk segment (H-06), upsell with churn >= 0.60, the action's offer families |
| 10 | the 360 view against the final file, the most recommended offers |
| 12 | **NBX against the last 5 campaigns** (`targeting_dataiku_auto`): the share of each action and of each offer, side by side |
| 11 | **the scorecard**: every check, its value, what is expected, OK / CHECK |

**No MSISDN is printed.** The examples show `LINE`, the row number in the final file. The files hold MSISDNs: keep
this notebook's outputs on this laptop, do not open the files in Excel.'''

SETTINGS = r'''# --- settings -------------------------------------------------------------------
RUN_ROOT = None               # None = the most recent run of "data/nbx_llm" (04 final), "data/real25k" (04 real), "data" (03)
RUN_DATE = None               # None = the latest run in <RUN_ROOT>/output/; or "2026-11-17", or a saved copy "2026-11-17_25k"
TARGET = "target2"
EXAMPLES = 10                 # rows shown for each example table
HISTORY_DATES = 5             # part 12: compare NBX with the last N campaign dates of targeting_dataiku_auto
HISTORY_FILE = None           # None = the full export data/data_raw/targeting_dataiku_auto* (all its dates),
                              #   else the week folder (latest campaign only); or a path "data/data_raw/my_export.csv.gz"'''

SETUP = r'''import re
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 60, "display.width", 220, "display.max_colwidth", 80, "display.max_rows", 200)
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
def _latest(base):
    runs = sorted(p for p in (base / "output").glob("20??-??-??") if (p / TARGET).is_dir())
    return runs


if RUN_ROOT is None:
    _cands = [(max(r.stat().st_mtime for r in _latest(ROOT / b)), b) for b in ("data/nbx_llm", "data/real25k", "data") if _latest(ROOT / b)]
    RUN_ROOT = max(_cands)[1] if _cands else "data/nbx_llm"
BASE = ROOT / RUN_ROOT
_runs = _latest(BASE)
if not _runs and not RUN_DATE:
    raise SystemExit(f"no run in {BASE / 'output'}: set RUN_ROOT (\"data/real25k\" or \"data\")")
OUT = BASE / "output" / RUN_DATE if RUN_DATE else _runs[-1]
D = OUT.name[:10]                                  # "2026-11-17_25k" (a saved copy) -> the week 2026-11-17
WEEK = next((p for p in (BASE / "weekly" / D, ROOT / "data" / "weekly" / D) if p.is_dir()), None)
if WEEK is None:
    raise SystemExit(f"no week folder for {D} (the catalogue, scripts and conflict pairs are read there)")
CHECKS = []
SLOTS = range(1, 7)
MONTHLY_SLOTS, WEEKLY_SLOTS = (1, 2, 5), (3, 4, 6)
ACTION_FAMILIES = {"MAINTAIN": ("COMBO", "DATA"), "UPSELL": ("COMBO", "DATA"), "RETAIN": ("COMBO", "DATA"),
                   "UPSELL VOICE": ("DATA", "COMBO"), "CROSS SELL VOICE": ("DATA", "COMBO"),
                   "CROSS SELL DATA": ("VOICE", "COMBO"), "UPSELL DATA": ("VOICE", "COMBO")}
UPSELL_ACTIONS = {"UPSELL", "UPSELL DATA", "UPSELL VOICE"}
AT_RISK_SEGMENTS = {"Nursering_Early_Churn_At_Risk", "Onboarding_Early_Churn_At_Risk", "Retention_RGS_At_Risk"}
H06_EXEMPT = {"Onboarding_Early_Churn_At_Risk": "UPSELL"}   # its production action (S-3) is UPSELL: allowed


def rd(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def table(name):
    p = next(WEEK.rglob(f"{name}.csv.gz"), None) or next(WEEK.rglob(f"{name}.csv"), None)
    if p is None:
        print(f"note: {name} not found in {WEEK}")
    return None if p is None else rd(p)


def check(part, name, value, expected, ok, note=""):
    CHECKS.append({"part": part, "check": name, "value": value, "expected": expected,
                   "status": "OK" if ok else ("INFO" if ok is None else "CHECK"), "note": note})


def show(df, title, n=None):
    display(Markdown(f"**{title}**"))
    display(df if n is None else df.head(n))


def period(v):
    v = pd.to_numeric(v, errors="coerce")
    return pd.Series(np.select([v <= 1, v <= 7, v >= 28], ["daily", "weekly", "monthly"], "other"), index=v.index).where(v.notna(), "not in catalogue")


print(f"run root {RUN_ROOT}, run {OUT.name}: {OUT}\nweek (catalogue, scripts, conflicts): {WEEK}")'''

LOAD = r'''_fp = OUT / "final" / "input_flytxt.csv.gz"
if not _fp.exists():
    _fp = OUT / TARGET / f"input_flytxt_{TARGET}.csv.gz"
fly = rd(_fp)
fly.index.name = "LINE"
s360 = pd.read_parquet(OUT / TARGET / f"{TARGET}_scored_360.parquet")
a360 = pd.read_parquet(OUT / TARGET / f"{TARGET}_360.parquet")

cat = table("btl_catalog")
cat.columns = [c.strip().upper() for c in cat.columns]
cat["BTL_NAME"] = cat["BTL_NAME"].str.strip()
for _c in ("PRICE", "GB", "MIN", "VALIDITY"):
    cat[_c] = pd.to_numeric(cat[_c], errors="coerce") if _c in cat.columns else np.nan
for _c in ("TYPE", "SERVICE", "GENEROSITY"):
    cat[_c] = cat[_c].astype(str).str.strip().str.upper() if _c in cat.columns else ""
cat = cat.drop_duplicates("BTL_NAME").set_index("BTL_NAME")

scr = table("BTL_script_english_arabic_ok")
scr.columns = [c.strip().upper() for c in scr.columns]
scr = scr.assign(DESTINATION_OFFER_NAME=scr["DESTINATION_OFFER_NAME"].str.strip()).drop_duplicates("DESTINATION_OFFER_NAME").set_index("DESTINATION_OFFER_NAME")

_cf = table("conflicting_offers_h")
CONFLICTS = set()
if _cf is not None:
    _cf.columns = [c.strip().lower() for c in _cf.columns]
    _cf = _cf[_cf.get("offer_type_a", pd.Series("", index=_cf.index)).astype(str) != "VOICE"]     # as code 5 (F-12)
    CONFLICTS = {frozenset((a.strip(), b.strip())) for a, b in zip(_cf["offer_a"], _cf["offer_b"])}

_h = table("sub_btl_3months")
HIST = set()
if _h is not None:
    _vcol = next(c for c in _h.columns if c.strip().lower() == "access_method_val")
    _pcol = next(c for c in _h.columns if c.strip().upper() == "PRODUCTNAME")
    HIST = set(zip(_h[_vcol].str.strip().str[-8:], _h[_pcol].str.strip()))

# one row per (line, slot) of the final file, with the catalogue facts and, from the 360 view, why it is there
fly["_V"] = fly["msisdn"].str.strip().str[-8:]
_r = s360.assign(_V=s360["ACCESS_METHOD_VAL"].astype(str).str.strip().str[-8:]).drop_duplicates("_V").set_index("_V")
long = pd.concat([pd.DataFrame({"LINE": fly.index, "_V": fly["_V"], "SLOT": i, "ACTION": fly["segment"],
                                "OFFER": fly[f"destination_offer_name_{i}"].str.strip(),
                                "OPPORTUNITY": fly[f"opportunity_name_{i}"].str.strip(),
                                "SCRIPT_EN": fly[f"offer_script_en_{i}"].str.strip(),
                                "SCRIPT_AR": fly[f"offer_script_ar_{i}"].str.strip()}) for i in SLOTS], ignore_index=True)
long["REASON"] = [(_r.at[v, f"REASON_{i}"] if v in _r.index and f"REASON_{i}" in _r.columns else None)
                  for v, i in zip(long["_V"], long["SLOT"])]
long["SELECTED_BY"] = [(_r.at[v, f"SLOT_{i}_SELECTED_BY"] if v in _r.index and f"SLOT_{i}_SELECTED_BY" in _r.columns else None)
                       for v, i in zip(long["_V"], long["SLOT"])]
long["VS_ANCHOR"] = [(_r.at[v, f"SLOT_{i}_VS_ANCHOR"] if v in _r.index and f"SLOT_{i}_VS_ANCHOR" in _r.columns else np.nan)
                     for v, i in zip(long["_V"], long["SLOT"])]
long["SEGMENT"] = long["_V"].map(_r["SEGMENT"]) if "SEGMENT" in _r.columns else None
long["GROUP_KEY"] = long["_V"].map(_r["GROUP_KEY"]) if "GROUP_KEY" in _r.columns else None
EMPTY = long[long["OFFER"].eq("")]
long = long[long["OFFER"].ne("")].join(cat[["TYPE", "SERVICE", "GENEROSITY", "VALIDITY", "PRICE", "GB", "MIN"]], on="OFFER")
long["PERIOD"] = period(long["VALIDITY"])
long["REASON"] = long["REASON"].fillna("-").astype(str)
print(f"final file {_fp.relative_to(ROOT)}: {len(fly):,} rows x {fly.shape[1] - 1} columns | {len(long):,} offers | "
      f"360 scored view {len(s360):,} rows, 360 all target lines {len(a360):,} | catalogue {len(cat):,} offers")'''

PART1 = r'''_in = a360["IN_INPUT_FLYTXT"].astype(bool) if "IN_INPUT_FLYTXT" in a360.columns else pd.Series(True, index=a360.index)
scope = a360.groupby("GROUP_KEY").agg(target_lines=("ACCESS_METHOD_VAL", "size"), in_final_file=("IN_INPUT_FLYTXT", "sum")) \
    if "GROUP_KEY" in a360.columns else pd.DataFrame({"target_lines": [len(a360)], "in_final_file": [int(_in.sum())]})
scope["not_in_file"] = scope["target_lines"] - scope["in_final_file"]
scope["share_in_file"] = (scope["in_final_file"] / scope["target_lines"]).round(4)
show(scope.reset_index(), "1a. Target lines and lines in the final file, per group")
drops = (a360.loc[~_in, "DROP_RULE"].fillna("(no rule given)").value_counts().rename_axis("DROP_RULE").reset_index(name="lines")
         if "DROP_RULE" in a360.columns else pd.DataFrame())
show(drops, "1b. Why the other lines are not in the file (DROP_RULE)")
check(1, "target lines in the final file", f"{int(_in.sum()):,} of {len(a360):,}", "most of them", None,
      f"{len(a360) - int(_in.sum()):,} dropped, see 1b")'''

PART2 = r'''_dupl = int(fly["msisdn"].duplicated().sum())
_n = long.groupby("LINE").size().reindex(fly.index, fill_value=0)
per_line = _n.value_counts().sort_index().rename_axis("offers in the line").reset_index(name="lines")
show(pd.DataFrame({"what": ["rows", "distinct MSISDN", "duplicated MSISDN", "columns", "date", "start_date", "end_date"],
                   "value": [len(fly), fly["msisdn"].nunique(), _dupl, fly.shape[1] - 1, ", ".join(sorted(fly["date"].unique())),
                             ", ".join(sorted(fly["start_date"].unique())), ", ".join(sorted(fly["end_date"].unique())) or "(empty)"]}),
     "2a. The file")
show(per_line, "2b. Offers per line")
show(EMPTY.groupby("SLOT").size().reindex(list(SLOTS), fill_value=0).rename_axis("slot").reset_index(name="empty"),
     "2c. Empty slots, per slot")
show(fly["segment"].value_counts().rename_axis("segment (= action)").reset_index(name="lines"), "2d. The segment column (the action sent)")
show(fly["segment_btl_taker"].value_counts().rename_axis("segment_btl_taker").reset_index(name="lines"), "2e. BTL takers (270 days)")
check(2, "one row per MSISDN", _dupl, 0, _dupl == 0)
check(2, "38 columns", fly.shape[1] - 1, 38, fly.shape[1] - 1 == 38)
check(2, "lines with 6 offers", f"{int((_n == 6).sum()):,} of {len(fly):,}", "all", bool((_n == 6).all()))'''

PART3 = r'''_bad_cat = long[long["PRICE"].isna()]
_no_en, _no_ar = long[long["SCRIPT_EN"].eq("")], long[long["SCRIPT_AR"].eq("")]
_exp_en = long["OFFER"].map(scr["ENGLISH SCRIPT"].str.strip()) if "ENGLISH SCRIPT" in scr.columns else long["SCRIPT_EN"]
_exp_ar = long["OFFER"].map(scr["ARABIC SCRIPT"].str.strip()) if "ARABIC SCRIPT" in scr.columns else long["SCRIPT_AR"]
_diff_en, _diff_ar = long[long["SCRIPT_EN"].ne(_exp_en.fillna(""))], long[long["SCRIPT_AR"].ne(_exp_ar.fillna(""))]
_opp = long[long["OPPORTUNITY"].ne("nbo_LFM_" + long["OFFER"])]
_opp_empty = EMPTY[EMPTY["OPPORTUNITY"].ne("") | EMPTY["SCRIPT_EN"].ne("") | EMPTY["SCRIPT_AR"].ne("")]
show(pd.DataFrame({"check": ["offer not in the catalogue", "no English script", "no Arabic script",
                             "English script differs from the script table", "Arabic script differs from the script table",
                             "opportunity name is not nbo_LFM_<offer>", "empty slot with an opportunity name or a script"],
                   "offers": [len(_bad_cat), len(_no_en), len(_no_ar), len(_diff_en), len(_diff_ar), len(_opp), len(_opp_empty)]}),
     "3a. Every offer of the file (expected 0 everywhere)")
if len(_bad_cat):
    show(_bad_cat["OFFER"].value_counts().rename_axis("offer").reset_index(name="times"), "3b. Offers not in the catalogue", EXAMPLES)
for _name, _df in (("offer in the catalogue", _bad_cat), ("English script", _no_en), ("Arabic script", _no_ar),
                   ("scripts = the script table", pd.concat([_diff_en, _diff_ar])), ("opportunity name", _opp)):
    check(3, _name, len(_df), 0, len(_df) == 0)'''

PART4 = r'''_k = long[["LINE", "SLOT", "OFFER", "TYPE", "PRICE", "VALIDITY", "SERVICE", "GB", "MIN", "REASON"]]
pairs = _k.merge(_k, on="LINE", suffixes=("_A", "_B"))
pairs = pairs[pairs["SLOT_A"] < pairs["SLOT_B"]]
f11 = pairs[pairs["OFFER_A"] == pairs["OFFER_B"]]
f09 = pairs[(pairs["OFFER_A"] != pairs["OFFER_B"]) & (pairs["TYPE_A"] == pairs["TYPE_B"]) & (pairs["PRICE_A"] == pairs["PRICE_B"])]
f12 = pairs[[frozenset((a, b)) in CONFLICTS for a, b in zip(pairs["OFFER_A"], pairs["OFFER_B"])]]
show(pd.DataFrame({"rule": ["F-11 the same offer twice in a line", "F-09 two offers of the same TYPE and PRICE",
                            "F-12 a conflict pair (conflicting_offers_h) in a line"],
                   "lines": [f11["LINE"].nunique(), f09["LINE"].nunique(), f12["LINE"].nunique()],
                   "pairs": [len(f11), len(f09), len(f12)]}), "4a. The code-5 rules (expected 0)")
for _t, _df in (("F-11", f11), ("F-09", f09), ("F-12", f12)):
    if len(_df):
        show(_df[["LINE", "SLOT_A", "OFFER_A", "SLOT_B", "OFFER_B"]], f"4b. {_t} examples", EXAMPLES)
    check(4, _t, int(_df["LINE"].nunique()), 0, len(_df) == 0)'''

PART5 = r'''long["SLOT_KIND"] = np.where(long["SLOT"].isin(MONTHLY_SLOTS), "monthly slot (1, 2, 5)", "weekly slot (3, 4, 6)")
show(pd.crosstab(long["SLOT"], long["PERIOD"], margins=True, margins_name="total"), "5a. Slot x validity of the offer")
show(long["PERIOD"].value_counts().rename_axis("validity").reset_index(name="offers")
     .assign(share=lambda d: (d["offers"] / d["offers"].sum()).round(4)), "5b. Daily / weekly / monthly offers")
_mix = long.pivot_table(index="LINE", columns="PERIOD", values="OFFER", aggfunc="size", fill_value=0)
for _p in ("monthly", "weekly", "daily"):
    if _p not in _mix.columns:
        _mix[_p] = 0
_pattern = (_mix["monthly"].astype(str) + " monthly + " + _mix["weekly"].astype(str) + " weekly"
            + np.where(_mix["daily"] > 0, " + " + _mix["daily"].astype(str) + " daily", ""))
show(_pattern.value_counts().rename_axis("mix in the line").reset_index(name="lines"), "5c. The mix per line")
_wrong = long[((long["SLOT"].isin(MONTHLY_SLOTS)) & long["PERIOD"].ne("monthly")) | ((long["SLOT"].isin(WEEKLY_SLOTS)) & long["PERIOD"].ne("weekly"))]
show(_wrong.groupby(["SLOT_KIND", "PERIOD", "REASON"]).size().rename("offers").reset_index(),
     "5d. Offers whose validity does not match their slot, by reason (a favourite or a forced bundle may do this)")
_wrong_rec = _wrong[_wrong["REASON"].str.startswith("recommended")]
check(5, "recommended offers in a slot of another validity", len(_wrong_rec), 0, len(_wrong_rec) == 0)
check(5, "lines with 3 monthly + 3 weekly", f"{int(((_mix['monthly'] == 3) & (_mix['weekly'] == 3)).sum()):,} of {len(fly):,}",
      "most of them", None, "the rest: favourites / forced bundles of another validity, see 5d")'''

PART6 = r'''_reason = long["REASON"].str.replace(r"forced F-08 .*", "forced F-08 (bundle)", regex=True)
show(_reason.value_counts().rename_axis("why the offer is there").reset_index(name="offers")
     .assign(share=lambda d: (d["offers"] / d["offers"].sum()).round(4)), "6a. Why each offer is there")
show(pd.crosstab(long["SLOT"], _reason), "6b. Per slot")
if long["SELECTED_BY"].notna().any():
    show(long["SELECTED_BY"].fillna("-").value_counts().rename_axis("selected by (rule / pass)").reset_index(name="offers"),
         "6c. Which rule selected the recommended offers")
fav = long[long["REASON"].eq("favourite")]
_real = pd.Series([(v, o) in HIST for v, o in zip(fav["_V"], fav["OFFER"])], index=fav.index)
show(pd.DataFrame({"what": ["lines with at least 1 favourite", "favourite offers", "favourites in slot 5", "favourites in slot 6",
                            "favourites really bought in the 270 days (sub_btl_3months)", "favourites NOT found in the history"],
                   "value": [fav["LINE"].nunique(), len(fav), int((fav["SLOT"] == 5).sum()), int((fav["SLOT"] == 6).sum()),
                             int(_real.sum()), int((~_real).sum())]}), "6d. Favourites")
show(fav.groupby("PERIOD").size().rename_axis("validity").reset_index(name="favourites"), "6e. Favourites by validity")
check(6, "favourites found in the customer's purchase history", f"{int(_real.sum()):,} of {len(fav):,}", "all", bool(_real.all()))
check(6, "lines with a favourite", f"{fav['LINE'].nunique():,} of {len(fly):,}", "-", None)'''

PART7 = r'''same = pairs[(pairs["OFFER_A"] != pairs["OFFER_B"]) & (pairs["PRICE_A"] == pairs["PRICE_B"])].copy()
same["SAME_VALIDITY"] = same["VALIDITY_A"] == same["VALIDITY_B"]
same["ALLOCATION_DIFFERS"] = (same["GB_A"].fillna(-1) != same["GB_B"].fillna(-1)) | (same["MIN_A"].fillna(-1) != same["MIN_B"].fillna(-1))
alloc = same[same["SAME_VALIDITY"] & same["ALLOCATION_DIFFERS"]]
show(pd.DataFrame({"case": ["two offers with the same price in a line (any validity)",
                            "same price AND same validity, different GB / minutes",
                            "of which the same service (e.g. two DATA offers)"],
                   "lines": [same["LINE"].nunique(), alloc["LINE"].nunique(), alloc[alloc["SERVICE_A"] == alloc["SERVICE_B"]]["LINE"].nunique()],
                   "pairs": [len(same), len(alloc), int((alloc["SERVICE_A"] == alloc["SERVICE_B"]).sum())]}),
     "7a. Same price, different allocation in one line")
_top = (alloc.assign(PAIR=lambda d: d["OFFER_A"] + "  |  " + d["OFFER_B"])
        .groupby(["PAIR", "PRICE_A", "VALIDITY_A"]).agg(lines=("LINE", "nunique"), GB_A=("GB_A", "first"), GB_B=("GB_B", "first"),
                                                       MIN_A=("MIN_A", "first"), MIN_B=("MIN_B", "first"),
                                                       reasons=("REASON_B", lambda s: ", ".join(sorted(set(s)))))
        .sort_values("lines", ascending=False).reset_index())
show(_top, "7b. The pairs, most frequent first (same price, same validity, different allocation)", EXAMPLES)

# dominated: same validity and service, B costs the same or more and gives less or equal GB AND minutes (one strictly)
_d = pairs[(pairs["OFFER_A"] != pairs["OFFER_B"]) & (pairs["VALIDITY_A"] == pairs["VALIDITY_B"]) & (pairs["SERVICE_A"] == pairs["SERVICE_B"])].copy()
_ga, _gb, _ma, _mb = (_d[c].fillna(0) for c in ("GB_A", "GB_B", "MIN_A", "MIN_B"))
_a_worse = (_d["PRICE_A"] >= _d["PRICE_B"]) & (_ga <= _gb) & (_ma <= _mb) & ((_d["PRICE_A"] > _d["PRICE_B"]) | (_ga < _gb) | (_ma < _mb))
_b_worse = (_d["PRICE_B"] >= _d["PRICE_A"]) & (_gb <= _ga) & (_mb <= _ma) & ((_d["PRICE_B"] > _d["PRICE_A"]) | (_gb < _ga) | (_mb < _ma))
dom = _d[_a_worse | _b_worse].assign(WORSE=np.where(_a_worse[_a_worse | _b_worse], "A", "B"))
dom["WORSE_OFFER"] = np.where(dom["WORSE"] == "A", dom["OFFER_A"], dom["OFFER_B"])
dom["BETTER_OFFER"] = np.where(dom["WORSE"] == "A", dom["OFFER_B"], dom["OFFER_A"])
dom["WORSE_REASON"] = np.where(dom["WORSE"] == "A", dom["REASON_A"], dom["REASON_B"])
show(pd.DataFrame({"case": ["a line holds an offer that is dominated by another offer of the same line"],
                   "lines": [dom["LINE"].nunique()], "pairs": [len(dom)]}),
     "7c. Dominated offers (same validity and service: the same price or more for less or equal GB and minutes)")
show(dom.groupby(["WORSE_OFFER", "BETTER_OFFER"]).agg(lines=("LINE", "nunique"), worse_is=("WORSE_REASON", lambda s: ", ".join(sorted(set(s)))))
     .sort_values("lines", ascending=False).reset_index(), "7d. Dominated pairs, most frequent first", EXAMPLES)
show(dom["WORSE_REASON"].value_counts().rename_axis("the dominated offer is there because").reset_index(name="pairs"),
     "7e. Where the dominated offers come from")
check(7, "lines with 2 offers of the same price and validity but a different allocation", f"{alloc['LINE'].nunique():,}", 0,
      alloc.empty, "see 7b: F-09 only clears the same TYPE and price")
check(7, "lines with a dominated offer", f"{dom['LINE'].nunique():,}", 0, dom.empty, "see 7d / 7e")'''

PART8 = r'''_rec = long[long["REASON"].str.startswith("recommended")]
_order = []
for _slots, _name in ((MONTHLY_SLOTS, "monthly"), (WEEKLY_SLOTS, "weekly")):
    _p = _rec[_rec["SLOT"].isin(_slots)].pivot_table(index="LINE", columns="SLOT", values="PRICE", aggfunc="first")
    _p = _p.reindex(columns=list(_slots))
    _bad = ((_p[_slots[0]] < _p[_slots[1]]) | (_p[_slots[1]] < _p[_slots[2]])).sum()
    _order.append({"slots": f"{_name} {_slots}", "lines with 3 recommended": int(_p.notna().all(axis=1).sum()),
                   "lines not in price order (highest first)": int(_bad)})
show(pd.DataFrame(_order), "8a. Price order of the recommended offers (F-02 / F-03: highest price in the first slot)")
_va = pd.to_numeric(long["VS_ANCHOR"], errors="coerce")
_band = pd.cut(_va, [-np.inf, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, np.inf], right=False,
               labels=["< 0.5", "0.5-0.8", "0.8-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">= 3.0"])
show(pd.crosstab(_band.cat.add_categories("no anchor").fillna("no anchor"), long["REASON"].str.replace(r"forced F-08 .*", "forced F-08", regex=True),
                 margins=True, margins_name="total"),
     "8b. Price / the customer's usual spend (anchor), by reason: < 1.0 = below the usual spend (cannibalization risk)")
_below = long[_va < 1.0]
show(_below.groupby(["REASON", "PERIOD"]).size().rename("offers").reset_index(), "8c. Offers below the usual spend")
show(long.groupby("PERIOD")["PRICE"].describe().round(2), "8d. Prices per validity")
_cols = [c for c in ("CHK_OFFERS_BELOW_ANCHOR", "CHK_FAVOURITE_BELOW_ANCHOR", "CHK_FALLBACK_PASS_BELOW_ANCHOR", "CHK_TOPUP_BELOW_ANCHOR") if c in s360.columns]
show(pd.DataFrame({c: [int((pd.to_numeric(s360[c], errors="coerce") > 0).sum())] for c in _cols}, index=["lines"]),
     "8e. The flags of the 360 view (lines with at least one)")
_rec_below = _rec[(_va.reindex(_rec.index) < 1.0) & _rec["SELECTED_BY"].fillna("").astype(str).str.startswith("challenger")]
check(8, "recommended offers not in price order", int(sum(r["lines not in price order (highest first)"] for r in _order)), 0,
      all(r["lines not in price order (highest first)"] == 0 for r in _order))
check(8, "lines with an offer below the usual spend", f"{_below['LINE'].nunique():,} of {len(fly):,}", "few",
      None, "favourites / fallback passes / top-up may be below it: see 8c")
check(8, "challenger-band offers below the usual spend", len(_rec_below), 0, _rec_below.empty)'''

PART8B = r'''# --- the customer's ARPU, recomputed here from the raw tables (independent of the engine's anchors) -----------------
# revenue of a period = TOTAL_SUB_AMT (subscriptions_tg) + PG_TOTAL_AMOUNT (pay-as-you-go, usage_data_voice_payg_tg)
# ARPU = the mean of the periods with spend (a period at 0 = no spend that period, as F-05); LAST = the latest period
WEEKS, MONTHS = ("W1", "W2", "W3", "W4"), ("M1", "M2", "M3")
_vals = set(long["_V"])


def _read_cols(name, want):
    p = next(WEEK.rglob(f"{name}.csv.gz"), None)
    if p is None:
        print(f"note: {name} not found: its part of the revenue is 0")
        return pd.DataFrame(columns=["_V"])
    hdr = pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns
    cols = {c: c.strip().upper() for c in hdr if c.strip().upper() in want}
    parts = []
    for ch in pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=list(cols), chunksize=500_000):
        ch = ch.rename(columns=cols)
        ch["_V"] = ch["ACCESS_METHOD_VAL"].str.strip().str.replace(r"\.0+$", "", regex=True).str[-8:]
        parts.append(ch[ch["_V"].isin(_vals)])
    df = pd.concat(parts, ignore_index=True).drop_duplicates("_V")             # one row per line (02 step 4b already)
    for c in df.columns:
        if c not in ("_V", "ACCESS_METHOD_VAL"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df.drop(columns=["ACCESS_METHOD_VAL"], errors="ignore").set_index("_V")


_P = WEEKS + MONTHS
_sub = _read_cols("subscriptions_tg", {"ACCESS_METHOD_VAL"} | {f"{k}_SUB_AMT_{p}" for k in ("TOTAL", "DATA", "VOICE") for p in _P})
_pg = _read_cols("usage_data_voice_payg_tg", {"ACCESS_METHOD_VAL"} | {f"PG_TOTAL_AMOUNT_{p}" for p in _P})
L = pd.DataFrame(index=pd.Index(sorted(_vals), name="_V"))


def _col(df, c):
    return df[c].reindex(L.index).fillna(0.0) if c in df.columns else pd.Series(0.0, index=L.index)


for _name, _per in (("WEEKLY", WEEKS), ("MONTHLY", MONTHS)):
    for _svc, _k, _pgw in (("TOTAL", "TOTAL", 1.0), ("DATA", "DATA", 0.9), ("VOICE", "VOICE", 0.1)):
        rev = pd.DataFrame({p: _col(_sub, f"{_k}_SUB_AMT_{p}") + _pgw * _col(_pg, f"PG_TOTAL_AMOUNT_{p}") for p in _per})
        L[f"ARPU_{_svc}_{_name}"] = rev.replace(0.0, np.nan).mean(axis=1).fillna(0.0).round(3)
    L[f"LAST_TOTAL_{_name}"] = (_col(_sub, f"TOTAL_SUB_AMT_{_per[0]}") + _col(_pg, f"PG_TOTAL_AMOUNT_{_per[0]}")).round(3)

_per_name = long["PERIOD"].map({"weekly": "WEEKLY", "monthly": "MONTHLY"})
_svc = long["SERVICE"].where(long["SERVICE"].isin(["DATA", "VOICE"]), "TOTAL")            # COMBO and the rest: total spend
long["ARPU_TOTAL"] = [L.at[v, f"ARPU_TOTAL_{p}"] if isinstance(p, str) else np.nan for v, p in zip(long["_V"], _per_name)]
long["ARPU_SERVICE"] = [L.at[v, f"ARPU_{s}_{p}"] if isinstance(p, str) else np.nan for v, s, p in zip(long["_V"], _svc, _per_name)]
long["VS_ARPU_TOTAL"] = (long["PRICE"] / long["ARPU_TOTAL"].where(long["ARPU_TOTAL"] > 0)).round(2)
long["VS_ARPU_SERVICE"] = (long["PRICE"] / long["ARPU_SERVICE"].where(long["ARPU_SERVICE"] > 0)).round(2)
long["DOWNGRADE_TOTAL"] = long["VS_ARPU_TOTAL"] < 1.0
long["DOWNGRADE_SERVICE"] = long["VS_ARPU_SERVICE"] < 1.0

_lines = pd.DataFrame({"monthly ARPU > 0": [int((L["ARPU_TOTAL_MONTHLY"] > 0).sum())],
                       "weekly ARPU > 0": [int((L["ARPU_TOTAL_WEEKLY"] > 0).sum())],
                       "no spend at all (ARPU 0: nothing to downgrade)": [int(((L["ARPU_TOTAL_MONTHLY"] == 0) & (L["ARPU_TOTAL_WEEKLY"] == 0)).sum())],
                       "lines": [len(L)]}, index=["lines"])
show(_lines, "8f. The customers' ARPU (recomputed from subscriptions_tg + usage_data_voice_payg_tg)")
show(L.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T.round(2), "8g. ARPU per period and service (R)")
_bands = [-np.inf, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, np.inf]
_lab = ["< 0.5", "0.5-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", ">= 3.0"]
for _c, _t in (("VS_ARPU_TOTAL", "the customer's TOTAL spend of that period"),
               ("VS_ARPU_SERVICE", "the customer's spend on that SERVICE (data offer vs data spend, voice vs voice, combo vs total)")):
    _b = pd.cut(long[_c], _bands, right=False, labels=_lab).cat.add_categories("ARPU 0").fillna("ARPU 0")
    show(pd.crosstab(_b, long["REASON"], margins=True, margins_name="total"),
         f"8h. Offer price / {_t}: < 1.0 = DOWNGRADE (the offer is cheaper than what the customer spends)")
_dg = long[long["DOWNGRADE_SERVICE"]]
show(pd.DataFrame({"what": ["offers below the customer's service spend", "lines with at least one such offer",
                            "lines whose 3 monthly offers are ALL below the monthly service spend",
                            "lines whose 3 weekly offers are ALL below the weekly service spend",
                            "offers below the TOTAL spend (stricter)", "lines with at least one"],
                   "value": [len(_dg), _dg["LINE"].nunique(),
                             int(long[long["PERIOD"].eq("monthly")].groupby("LINE")["DOWNGRADE_SERVICE"].all().sum()),
                             int(long[long["PERIOD"].eq("weekly")].groupby("LINE")["DOWNGRADE_SERVICE"].all().sum()),
                             int(long["DOWNGRADE_TOTAL"].sum()), long.loc[long["DOWNGRADE_TOTAL"], "LINE"].nunique()]}),
     "8i. Downgrades")
show(_dg.groupby(["REASON", "PERIOD", "SERVICE"]).size().rename("offers").reset_index().sort_values("offers", ascending=False),
     "8j. Where the downgrades come from (by reason, validity, service)")
show(_dg.groupby(["OFFER", "PRICE"]).agg(times=("LINE", "size"), median_service_arpu=("ARPU_SERVICE", "median"),
                                         median_ratio=("VS_ARPU_SERVICE", "median")).sort_values("times", ascending=False).reset_index(),
     "8k. The offers most often below the customer's spend", EXAMPLES)
show(long.groupby("SLOT")[["DOWNGRADE_SERVICE", "DOWNGRADE_TOTAL"]].mean().round(4), "8l. Share of downgrades per slot")
_dg_rec = _dg[_dg["REASON"].str.startswith("recommended")]
check("8b", "recommended offers below the customer's service spend (downgrade)", f"{len(_dg_rec):,} in {_dg_rec['LINE'].nunique():,} lines",
      0, _dg_rec.empty, "see 8j / 8k")
check("8b", "favourite / top-up offers below the customer's service spend", f"{len(_dg) - len(_dg_rec):,}", "few", None,
      "a favourite is an offer the customer bought; the top-up fills a slot the rules left empty")
check("8b", "lines with ALL monthly offers below the monthly service spend",
      int(long[long["PERIOD"].eq("monthly")].groupby("LINE")["DOWNGRADE_SERVICE"].all().sum()), 0,
      int(long[long["PERIOD"].eq("monthly")].groupby("LINE")["DOWNGRADE_SERVICE"].all().sum()) == 0)'''

PART9 = r'''_l = s360.copy()
_act = _l["ACTION_SENT"].astype(str) if "ACTION_SENT" in _l.columns else _l["ACTION_AGENT"].astype(str)
show(pd.crosstab(_l["SEGMENT"], _act, margins=True, margins_name="total"), "9a. Segment x action sent")
if "DECIDED_BY" in _l.columns:
    show(_l["DECIDED_BY"].value_counts().rename_axis("decided by").reset_index(name="lines"), "9b. Who decided the action")
_h06 = _l[_act.isin(UPSELL_ACTIONS) & _l["SEGMENT"].isin(AT_RISK_SEGMENTS)].assign(ACTION=_act)
_h06["EXEMPT"] = [H06_EXEMPT.get(s) == a for s, a in zip(_h06["SEGMENT"], _h06["ACTION"])]
show(_h06.groupby(["SEGMENT", "ACTION", "EXEMPT"]).size().rename("lines").reset_index(),
     "9c. Upsell to an at-risk segment (H-06). EXEMPT = the segment's own production action (Onboarding at risk = UPSELL)")
_churn = pd.to_numeric(_l.get("PROBA_CHURN"), errors="coerce")
_up60 = int((_act.isin(UPSELL_ACTIONS) & (_churn >= 0.6)).sum())
_nurs = int(_l["SEGMENT"].astype(str).str.lower().str.startswith("nursering").sum())
_fam = long[long["GROUP_KEY"].eq("t2c") & long["ACTION"].isin(ACTION_FAMILIES)].copy()
_fam["IN_FAMILY"] = [s in ACTION_FAMILIES[a] for s, a in zip(_fam["SERVICE"], _fam["ACTION"])]
show(_fam.groupby(["REASON", "IN_FAMILY"]).size().unstack(fill_value=0), "9d. Group a: is the offer's service one of the action's families (BR-66)?")
_t2f = long[long["GROUP_KEY"].eq("t2f")]
_t2f_bad = _t2f[_t2f["REASON"].str.startswith("recommended") & (~_t2f["SERVICE"].isin(["DATA", "COMBO"]) | _t2f["GENEROSITY"].ne("L2"))]
_out_rec = _fam[~_fam["IN_FAMILY"] & _fam["REASON"].str.startswith("recommended")]
check(9, "upsell to an at-risk segment, not exempt (H-06)", int((~_h06["EXEMPT"]).sum()), 0, bool(_h06["EXEMPT"].all()),
      "see 9c" + (f"; {int(_h06['EXEMPT'].sum()):,} exempt lines (Onboarding at risk = UPSELL in production)" if _h06["EXEMPT"].any() else ""))
check(9, "upsell with churn >= 0.60", _up60, "few", None, "information (plan Q9)")
check(9, "Nursering lines in the file", _nurs, 0, _nurs == 0)
check(9, "group a: recommended offers outside the action's families", len(_out_rec), 0, _out_rec.empty, "see 9d")
if len(_t2f):
    check(9, "group b: recommended offers outside L2 DATA / COMBO", len(_t2f_bad), 0, _t2f_bad.empty)'''

PART10 = r'''_c = [c for c in ("CHK_NOT_IN_FLYTXT", "CHK_SLOTS_DIFFER_FROM_FLYTXT") if c in s360.columns]
_v = {c: int(s360[c].astype(bool).sum()) for c in _c}
_extra = len(set(fly["_V"]) - set(s360["ACCESS_METHOD_VAL"].astype(str).str[-8:]))
show(pd.DataFrame({"check": [*_c, "final-file lines not in the scored 360 view"], "lines": [*_v.values(), _extra]}),
     "10a. The 360 view against the final file (expected 0)")
for _k, _x in _v.items():
    check(10, _k, _x, 0, _x == 0)
check(10, "final-file lines not in the 360 view", _extra, 0, _extra == 0)
top = (long.groupby(["OFFER", "PERIOD", "SERVICE", "GENEROSITY", "PRICE", "GB", "MIN"], dropna=False)
       .agg(times=("LINE", "size"), lines=("LINE", "nunique"), as_favourite=("REASON", lambda s: int((s == "favourite").sum())))
       .sort_values("times", ascending=False).reset_index())
top["share"] = (top["times"] / top["times"].sum()).round(4)
show(top, "10b. The most recommended offers", 25)
show(long.groupby(["PERIOD", "SERVICE"]).size().unstack(fill_value=0), "10c. Offers per validity and service")'''

PART12 = r'''# --- NBX (this run) against the last HISTORY_DATES campaign dates of targeting_dataiku_auto -----------------------------
from collections import Counter

ACTIONS7 = ["RETAIN", "MAINTAIN", "UPSELL", "UPSELL DATA", "UPSELL VOICE", "CROSS SELL DATA", "CROSS SELL VOICE"]
SEG2ACT = {"retention_low_probability_to_reconnect": "RETAIN", "retention_med_probability_to_reconnect": "RETAIN",
           "retention_high_probability_to_reconnect": "RETAIN", "retention_rgs_at_risk": "RETAIN",
           "retention_rgs_fragile": "UPSELL", "rgs_not_at_risk_dualplay_upgrade": "MAINTAIN",
           "rgs_not_at_risk_data_only_voice_appetency": "UPSELL VOICE",
           "rgs_not_at_risk_data_only_no_voice_appetency": "CROSS SELL VOICE",
           "rgs_not_at_risk_voice_only_data_appetency": "UPSELL DATA",
           "rgs_not_at_risk_voice_only_no_data_appetency": "CROSS SELL DATA",
           "onboarding_early_churn_at_risk": "UPSELL", "onboarding_early_churn_not_at_risk": "UPSELL"}   # BR-39


def _act_of(seg):
    s = str(seg).strip()
    if s.upper() in ACTIONS7:                       # the segment column already holds an action (an NBX campaign)
        return s.upper()
    if s.lower().startswith("nursering"):
        return "(Nursering: excluded)"
    return SEG2ACT.get(s.lower(), "(other segment)")


if HISTORY_FILE:
    _hp = ROOT / HISTORY_FILE
else:
    _hp = next(iter(sorted((ROOT / "data" / "data_raw").glob("targeting_dataiku_auto*.csv*"))), None) \
        or next(WEEK.rglob("targeting_dataiku_auto.csv*"), None)
if _hp is None or not Path(_hp).exists():
    raise SystemExit("no targeting_dataiku_auto found (data/data_raw/ or the week folder): set HISTORY_FILE")
_hdr = pd.read_csv(_hp, nrows=0, encoding="utf-8-sig").columns
_low = {c: c.strip().lower() for c in _hdr}
_dcol = next(c for c in _hdr if _low[c] == "date1")
_idcol = next((c for c in _hdr if _low[c] == "access_method_id"), None) \
    or next((c for c in _hdr if _low[c] in ("access_method_val", "msisdn")), None)
if _idcol is None:
    raise SystemExit(f"no access_method_id column in {_hp}: the campaign lines cannot be made distinct")
_use = [c for c in _hdr if _low[c] in {"date1", "segment", *[f"destination_offer_name_{i}" for i in SLOTS]}]
if _idcol not in _use:
    _use.append(_idcol)

_dates = Counter()                                   # pass 1: the campaign dates
for _ch in pd.read_csv(_hp, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=[_dcol], chunksize=1_000_000):
    _dates.update(pd.to_datetime(_ch[_dcol], errors="coerce").dt.date.dropna().astype(str))
LAST = sorted(_dates)[-HISTORY_DATES:][::-1]          # newest first
print(f"{Path(_hp).relative_to(ROOT)}: {len(_dates)} campaign date(s) in the file; compared: {', '.join(LAST)}")
if len(_dates) < HISTORY_DATES:
    print(f"note: only {len(_dates)} date(s) in this file. For {HISTORY_DATES}, export targeting_dataiku_auto with its "
          "history into data/data_raw (notebook 02 keeps only the latest campaign in the week folder)")

# pass 2: lines, actions and offers per date, ONE ROW PER DISTINCT access_method_id PER DATE (the first row seen is kept)
_lines, _acts, _offs, _raw = Counter(), Counter(), Counter(), Counter()
_ocols = [c for c in _use if _low[c].startswith("destination_offer_name_")]
_seen = np.array([], dtype=np.uint64)                # hashes of the (date, id) pairs already counted, across chunks
for _ch in pd.read_csv(_hp, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=_use, chunksize=1_000_000):
    _ch["_D"] = pd.to_datetime(_ch[_dcol], errors="coerce").dt.date.astype(str)
    _ch = _ch[_ch["_D"].isin(LAST)]
    if _ch.empty:
        continue
    _raw.update(_ch["_D"].value_counts().to_dict())
    _ch["_ID"] = _ch[_idcol].str.strip().str.replace(r"\.0+$", "", regex=True)
    _ch = _ch[_ch["_ID"].ne("")].drop_duplicates(["_D", "_ID"])
    _key = pd.util.hash_pandas_object(_ch[["_D", "_ID"]], index=False).to_numpy()
    _keep = ~np.isin(_key, _seen)
    _ch = _ch[_keep]
    _seen = np.union1d(_seen, _key[_keep])
    if _ch.empty:
        continue
    _lines.update(_ch["_D"].value_counts().to_dict())
    _seg = next((c for c in _use if _low[c] == "segment"), None)
    _a = _ch[_seg].map({s: _act_of(s) for s in _ch[_seg].unique()}) if _seg else pd.Series("(no segment column)", index=_ch.index)
    _acts.update(pd.Series(list(zip(_ch["_D"], _a))).value_counts().to_dict())
    for c in _ocols:
        o = _ch[c].str.strip()
        _offs.update(pd.Series(list(zip(_ch.loc[o.ne(""), "_D"], o[o.ne("")]))).value_counts().to_dict())

show(pd.DataFrame({"rows in the file": [_raw.get(d, 0) for d in LAST],
                   f"distinct {_idcol}": [_lines.get(d, 0) for d in LAST],
                   "rows dropped (duplicate or empty id)": [_raw.get(d, 0) - _lines.get(d, 0) for d in LAST]},
                  index=pd.Index(LAST, name="campaign date")),
     f"12-0. targeting_dataiku_auto made distinct on {_idcol} (each id counted once per date, before every count below)")

# --- 12a. the actions: share of lines -------------------------------------------------------------------------------
NBX = "NBX (this run)"
_nbx_act = fly["segment"].str.strip().str.upper().value_counts()
_rows = sorted({*ACTIONS7, *(a for _, a in _acts)}, key=lambda a: (a not in ACTIONS7, ACTIONS7.index(a) if a in ACTIONS7 else 0, a))
ACT_CMP = pd.DataFrame(index=pd.Index(_rows, name="action"))
ACT_CMP[NBX] = (_nbx_act.reindex(ACT_CMP.index).fillna(0) / len(fly) * 100).round(1)
for d in LAST:
    ACT_CMP[d] = (pd.Series({a: n for (dd, a), n in _acts.items() if dd == d}).reindex(ACT_CMP.index).fillna(0)
                  / max(_lines.get(d, 0), 1) * 100).round(1)
if LAST:
    ACT_CMP[f"NBX - {LAST[0]} (pts)"] = (ACT_CMP[NBX] - ACT_CMP[LAST[0]]).round(1)
    ACT_CMP[f"avg of {len(LAST)} dates"] = ACT_CMP[LAST].mean(axis=1).round(1)
_shown = ACT_CMP.copy()
_shown.loc["(lines)"] = [float(len(fly)), *[float(_lines.get(d, 0)) for d in LAST]] + ([np.nan, np.nan] if LAST else [])
show(_shown, "12a. Actions: % of the lines, NBX against each campaign date (newest first)")

# --- 12b. the offers: share of all offers sent, and of the lines that got it ----------------------------------------
_nbx_off = long["OFFER"].value_counts()
_names = sorted(set(_nbx_off.index) | {o for _, o in _offs})
OFF_CMP = pd.DataFrame(index=pd.Index(_names, name="offer"))
OFF_CMP = OFF_CMP.join(cat[["SERVICE", "VALIDITY", "PRICE", "GB", "MIN"]], how="left")
OFF_CMP[f"{NBX} % of offers"] = (_nbx_off.reindex(OFF_CMP.index).fillna(0) / max(len(long), 1) * 100).round(2)
OFF_CMP[f"{NBX} % of lines"] = (long.groupby("OFFER")["LINE"].nunique().reindex(OFF_CMP.index).fillna(0) / len(fly) * 100).round(2)
for d in LAST:
    tot = sum(n for (dd, _), n in _offs.items() if dd == d)
    OFF_CMP[f"{d} % of offers"] = (pd.Series({o: n for (dd, o), n in _offs.items() if dd == d}).reindex(OFF_CMP.index).fillna(0)
                                   / max(tot, 1) * 100).round(2)
if LAST:
    OFF_CMP[f"NBX - {LAST[0]} (pts)"] = (OFF_CMP[f"{NBX} % of offers"] - OFF_CMP[f"{LAST[0]} % of offers"]).round(2)
OFF_CMP = OFF_CMP.sort_values(f"{NBX} % of offers", ascending=False)
show(OFF_CMP.reset_index(), f"12b. Offers: % of all offers sent, NBX against each campaign date (top 30; the full table is OFF_CMP, "
     f"{len(OFF_CMP):,} offers)", 30)
_new = OFF_CMP[(OFF_CMP[f"{NBX} % of offers"] > 0) & (OFF_CMP[[f"{d} % of offers" for d in LAST]].sum(axis=1) == 0)] if LAST else OFF_CMP.iloc[:0]
_gone = OFF_CMP[(OFF_CMP[f"{NBX} % of offers"] == 0)] if LAST else OFF_CMP.iloc[:0]
show(_new.reset_index()[["offer", "SERVICE", "VALIDITY", "PRICE", f"{NBX} % of offers"]],
     "12c. Offers NBX sends that none of the compared campaigns sent", EXAMPLES)
show(_gone.reset_index()[["offer", "SERVICE", "VALIDITY", "PRICE", *[f"{d} % of offers" for d in LAST]]]
     .sort_values(f"{LAST[0]} % of offers" if LAST else "offer", ascending=False),
     "12d. Offers the campaigns sent that NBX does not send", EXAMPLES)

# --- 12e. the mix: service x validity, price, offers per line -------------------------------------------------------
def _mix(counts, total_lines):
    s = pd.Series(counts, dtype=float)
    if s.empty:
        return {}
    info = cat.reindex(s.index)
    per = period(info["VALIDITY"])
    out = {f"{sv} {p}": round(float(s[(info["SERVICE"] == sv).to_numpy() & (per == p).to_numpy()].sum() / s.sum() * 100), 1)
           for sv in ("COMBO", "DATA", "VOICE") for p in ("monthly", "weekly")}
    for p in ("monthly", "weekly"):
        m = (per == p).to_numpy() & info["PRICE"].notna().to_numpy()
        out[f"avg price {p}"] = round(float((s[m] * info["PRICE"][m]).sum() / max(s[m].sum(), 1)), 2)
    out["offers per line"] = round(float(s.sum() / max(total_lines, 1)), 2)
    out["distinct offers"] = int((s > 0).sum())
    return out


MIX_CMP = pd.DataFrame({NBX: _mix(_nbx_off.to_dict(), len(fly)),
                        **{d: _mix({o: n for (dd, o), n in _offs.items() if dd == d}, _lines.get(d, 0)) for d in LAST}})
show(MIX_CMP, "12e. The mix: % of offers per service and validity, average price, offers per line, distinct offers")
check(12, f"compared with the last {len(LAST)} campaign date(s)", ", ".join(LAST) or "-", f"{HISTORY_DATES} dates", None,
      f"distinct on {_idcol}; 12a actions, 12b offers, 12e mix; ACT_CMP / OFF_CMP / MIX_CMP hold the full tables")'''

PART12F = r'''# --- 12f. the NBX target joined to the last campaign: the old action of the SAME lines -----------------------------------
# needs Part 12 (LAST, _hp, _use, _idcol, _act_of, ACT_CMP, MIX_CMP, _mix). The old campaign of LAST[0] is read again and
# kept only for the lines of this run, one row per access_method_id (the first row, as 12-0).
if not LAST:
    raise SystemExit("no campaign date in targeting_dataiku_auto: run Part 12 first")
D0 = LAST[0]


def _nid(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# this run's lines -> their id in the campaign table
if _low[_idcol] == "access_method_id":
    if "ACCESS_METHOD_ID" in s360.columns:
        _map = pd.DataFrame({"_V": s360["ACCESS_METHOD_VAL"].astype(str).str.strip().str[-8:], "_ID": _nid(s360["ACCESS_METHOD_ID"])})
    else:                                               # the week's access_method_id_auto: id -> MSISDN, latest activation
        _am = table("access_method_id_auto")
        if _am is None:
            raise SystemExit("no ACCESS_METHOD_ID in the 360 view and no access_method_id_auto in the week folder")
        _am.columns = [c.strip().upper() for c in _am.columns]
        _am = _am.sort_values("ACTIVATION_DATE").drop_duplicates("ACCESS_METHOD_ID", keep="last")
        _map = pd.DataFrame({"_V": _am["MSISDN"].str.strip().str[-8:], "_ID": _nid(_am["ACCESS_METHOD_ID"])})
else:                                                   # the campaign table holds the MSISDN itself
    _map = pd.DataFrame({"_V": fly["_V"], "_ID": None})
_map = _map[~_map["_ID"].isin(["", "nan", "<NA>", "None"])].drop_duplicates("_V")
J = fly[["_V", "segment"]].rename(columns={"segment": "NBX_ACTION"}).merge(_map, on="_V", how="left")
J["NBX_ACTION"] = J["NBX_ACTION"].str.strip().str.upper()
_key = "_V" if _low[_idcol] != "access_method_id" else "_ID"
_want = set(J[_key].dropna())

_segc = next(c for c in _use if _low[c] == "segment")
_parts = []
for _ch in pd.read_csv(_hp, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=_use, chunksize=1_000_000):
    _ch = _ch[pd.to_datetime(_ch[_dcol], errors="coerce").dt.date.astype(str).eq(D0)]
    _ch = _ch.assign(_ID=_nid(_ch[_idcol]))
    if _key == "_V":
        _ch["_V"] = _ch["_ID"].str[-8:]
    _parts.append(_ch[_ch[_key].isin(_want)])
_rows_t = pd.concat(_parts, ignore_index=True) if _parts else pd.DataFrame(columns=[*_use, "_ID", "_V"])
_two_seg = int((_rows_t.groupby(_key)[_segc].nunique() > 1).sum())     # ids whose rows carry different segments
OLD = _rows_t.drop_duplicates(_key).copy()
OLD["OLD_ACTION"] = OLD[_segc].map({s: _act_of(s) for s in OLD[_segc].unique()})
J = J.merge(OLD[[_key, "OLD_ACTION"]], on=_key, how="left")
M = J[J["OLD_ACTION"].notna()]
if M.empty:
    raise SystemExit(f"none of the {len(J):,} NBX lines is in the {D0} campaign: check the id ({_idcol}) and HISTORY_FILE")

show(pd.DataFrame({"what": ["NBX target lines (this run)", f"with an {_idcol}", f"found in the {D0} campaign",
                            "NOT in that campaign", f"rows of the campaign for these lines", "lines whose rows carry 2+ segments"],
                   "lines": [len(J), int(J[_key].notna().sum()), len(M), len(J) - len(M), len(_rows_t), _two_seg]}),
     f"12f-0. The NBX target joined to the {D0} campaign on {_idcol}")

# --- the action distribution: the whole campaign, the campaign on the NBX target, NBX ---------------------------------
_ix = ACT_CMP.index
T12 = pd.DataFrame(index=_ix)
T12[f"old {D0}: all lines"] = ACT_CMP[D0]
T12[f"old {D0}: NBX target lines"] = (M["OLD_ACTION"].value_counts().reindex(_ix).fillna(0) / max(len(M), 1) * 100).round(1)
T12["NBX: the same lines"] = (M["NBX_ACTION"].value_counts().reindex(_ix).fillna(0) / max(len(M), 1) * 100).round(1)
T12["target - all (pts)"] = (T12[f"old {D0}: NBX target lines"] - T12[f"old {D0}: all lines"]).round(1)
T12["NBX - old, same lines (pts)"] = (T12["NBX: the same lines"] - T12[f"old {D0}: NBX target lines"]).round(1)
_t = T12.copy()
_t.loc["(lines)"] = [float(_lines.get(D0, 0)), float(len(M)), float(len(M)), np.nan, np.nan]
show(_t, f"12f. Actions, % of the lines: the old campaign on ALL its lines, the old campaign on the NBX TARGET lines (join), "
         f"NBX on the same lines")

# --- line by line: the old action against the NBX action ------------------------------------------------------------
show(pd.crosstab(M["OLD_ACTION"], M["NBX_ACTION"], margins=True, margins_name="total")
     .rename_axis(index=f"old action ({D0})", columns="NBX action"), "12g. The same lines: old action (rows) x NBX action (columns)")
_same = M["OLD_ACTION"].eq(M["NBX_ACTION"])
AGREE = (M.assign(SAME=_same).groupby("OLD_ACTION").agg(lines=("SAME", "size"), same_action_in_NBX=("SAME", "sum"))
         .assign(share=lambda d: (d["same_action_in_NBX"] / d["lines"] * 100).round(1)).sort_values("lines", ascending=False))
show(AGREE, "12h. Per old action: how many of its lines NBX gave the same action (%)")
print(f"NBX gives the old campaign's action to {_same.mean():.1%} of the {len(M):,} common lines "
      f"(a random choice among 7 actions: 14.3 %)")

# --- the offer mix of the old campaign on the NBX target ------------------------------------------------------------
_oc = [c for c in _use if _low[c].startswith("destination_offer_name_")]
_om = OLD[OLD[_key].isin(set(M[_key]))]
_cnt = pd.Series(_om[_oc].to_numpy().ravel()).str.strip()
_cnt = _cnt[_cnt.ne("")].value_counts().to_dict()
MIX_T = pd.DataFrame({f"old {D0}: all lines": MIX_CMP[D0], f"old {D0}: NBX target lines": pd.Series(_mix(_cnt, len(_om))),
                      "NBX (this run)": MIX_CMP[NBX]})
show(MIX_T, "12i. The offer mix: the old campaign on all its lines, on the NBX target lines, and NBX")
check(12, f"NBX target lines found in the {D0} campaign", f"{len(M):,} of {len(J):,}", "most of them", None, "see 12f-0")
check(12, "same action as the old campaign, on the same lines", f"{_same.mean():.1%}", "-", None,
      "see 12g / 12h; a random choice among 7 actions = 14.3 %")'''

SCORECARD = r'''card = pd.DataFrame(CHECKS)
show(card, "11. Scorecard: every check (CHECK = look at the table of that part; INFO = a figure, no rule)")
print(f"{int((card['status'] == 'OK').sum())} OK, {int((card['status'] == 'CHECK').sum())} CHECK, "
      f"{int((card['status'] == 'INFO').sum())} INFO")'''

CELLS = [
    ("md", TITLE), ("code", SETTINGS), ("code", SETUP),
    ("md", "## Load the run\n\nThe final file, the two 360 views, and from the week folder the catalogue, the SMS scripts, the conflict pairs and the 270-day purchases (for the favourites)."),
    ("code", LOAD),
    ("md", "## Part 1 · What was scored"), ("code", PART1),
    ("md", "## Part 2 · The final file"), ("code", PART2),
    ("md", "## Part 3 · Every offer: catalogue, scripts, opportunity name"), ("code", PART3),
    ("md", "## Part 4 · The code-5 rules: F-11, F-09, F-12"), ("code", PART4),
    ("md", "## Part 5 · Daily, weekly, monthly\n\nSlots 1, 2, 5 hold the 30-day offers and 3, 4, 6 the 7-day offers (F-02 / F-03). A favourite or a forced bundle fills an empty slot 5 / 6 with whatever validity it has."),
    ("code", PART5),
    ("md", "## Part 6 · Why each offer is there, and the favourites"), ("code", PART6),
    ("md", ("## Part 7 · Same price, different allocation, and dominated offers\n\n"
           "- **Same price, different allocation:** two offers of one line with the same price and validity but other GB or minutes, "
           "e.g. 5 R for 15 GB and 5 R for 20 GB. Code 5 (F-09) only clears two offers of the same **TYPE** and price, so this can pass.\n"
           "- **Dominated:** in one line, same validity and service, one offer costs the same or more and gives less or equal GB "
           "**and** minutes. The customer would never choose it: it wastes a slot.")),
    ("code", PART7),
    ("md", "## Part 8 · Price order and price against the usual spend"), ("code", PART8),
    ("md", ("## Part 8b · No downgrade: every offer against the customer's ARPU\n\n"
           "The ARPU is recomputed here from the raw tables of the week, independently of the engine: revenue of a period = "
           "`TOTAL_SUB_AMT` (subscriptions_tg) + `PG_TOTAL_AMOUNT` (pay-as-you-go); ARPU = the mean of the periods with spend "
           "(weeks W1-W4 for a 7-day offer, months M1-M3 for a 30-day offer).\n\n"
           "- **Total:** the offer price against everything the customer spends in that period (strict).\n"
           "- **Service:** a DATA offer against the data spend (`DATA_SUB_AMT` + 90 % of pay-as-you-go), a VOICE offer "
           "against the voice spend (`VOICE_SUB_AMT` + 10 %), a COMBO against the total. This is the fair test of a downgrade.\n\n"
           "**Price / ARPU < 1.0 = a downgrade**: the offer is cheaper than what the customer already spends.")),
    ("code", PART8B),
    ("md", "## Part 9 · The actions and their rules"), ("code", PART9),
    ("md", "## Part 10 · The 360 view against the file, the most recommended offers"), ("code", PART10),
    ("md", ("## Part 12 · NBX against the last campaigns (targeting_dataiku_auto)\n\n"
           "The share of each **action** (% of the lines) and of each **offer** (% of all offers sent, and % of the lines "
           "that got it) in this run, side by side with the last `HISTORY_DATES` campaign dates of `targeting_dataiku_auto`.\n\n"
           "- **The source:** the full export in `data/data_raw/` (notebook 02 keeps only the latest campaign in the week "
           "folder), or `HISTORY_FILE`. Export the table **with its history** to compare 5 dates.\n"
           "- **Distinct lines:** the campaign table is made distinct on `access_method_id` first (one row per id per date, "
           "the first one kept; 12-0 shows the rows dropped). Every line, action and offer count below uses the distinct lines.\n"
           "- **The action of a past campaign** comes from its `segment` (BR-39: segment → action); a segment that already "
           "holds an action name (an NBX campaign) is taken as it is.\n"
           "- The full tables stay in `ACT_CMP`, `OFF_CMP` and `MIX_CMP` for your own filters.")),
    ("code", PART12),
    ("md", ("## Part 12f · The NBX target in the old campaign (join on access_method_id)\n\n"
           "Part 12 compares **different customers** (the old campaign's whole base against the NBX target). Here the last "
           "campaign is **joined on the lines of this run**, so the same customers are compared:\n\n"
           "- **12f:** the action distribution of the old campaign on **all** its lines, of the old campaign on the **NBX "
           "target** lines, and of NBX on those same lines. `target - all` shows how different the target is from the base; "
           "`NBX - old` is the real change NBX makes.\n"
           "- **12g / 12h:** line by line, the old action against the NBX action, and the share of lines that keep their action.\n"
           "- **12i:** the offer mix of the old campaign on the same lines, against NBX.\n\n"
           "The id comes from the 360 view (`ACCESS_METHOD_ID`), else from `access_method_id_auto` of the week. A line "
           "with two rows in the campaign keeps the first one (as 12-0); 12f-0 counts the lines whose rows carry two segments.")),
    ("code", PART12F),
    ("md", "## Part 11 · Scorecard"), ("code", SCORECARD),
]
