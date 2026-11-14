"""⑨ auditor checks of the four groups (update §7): A-12 per group, A-12b, A-17, A-30, A-31. Deterministic; any FAIL
blocks the push (H-08). The other A / T / R checks come with Phase 7.

    rep = audit_groups(panel, selected, recommendable, sizes)
    rep["status"]  "PASS" | "WARN" | "FAIL"      rep["push_blocked"]  True on any FAIL
"""
from __future__ import annotations

import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.offer_rules import families
from nbx.fsdr.offers.selection import challenger_band
from nbx.fsdr.panel import balance, check_sizes

VAL = "ACCESS_METHOD_VAL"
EPS = 1e-9


def _check(cid: str, rule: str, bad: pd.DataFrame | list, what: str, warn: bool = False) -> dict:
    n = len(bad)
    examples = (bad[[c for c in (VAL, "GROUP_KEY", "BTL_NAME", "PRICE") if c in bad.columns]].head(5).to_dict("records")
                if isinstance(bad, pd.DataFrame) else list(bad)[:5])
    status = "PASS" if n == 0 else ("WARN" if warn else "FAIL")
    return {"id": cid, "rule": rule, "status": status, "count": n, "what": what, "examples": examples}


def check_universe(sel: pd.DataFrame) -> list[dict]:
    """A-12: every offer inside its group's universe."""
    free = sel[sel["GROUP_KEY"].map(lambda k: config.GROUPS[k]["universe"]) == "free"]
    bad_free = free[(free["SERVICE"] == "VOICE") | (free["GENEROSITY"] != config.FREE_UNIVERSE_GENEROSITY)]
    cons = sel[sel["GROUP_KEY"].map(lambda k: config.GROUPS[k]["universe"]) == "constraint"]
    ok = [(s, g) in families(a, "cstr") for a, s, g in zip(cons["ACTION"], cons["SERVICE"], cons["GENEROSITY"])]
    bad_cons = cons[[not x for x in ok]]
    return [_check("A-12", "FR-02 / FR-03", bad_free, "T1F / T2F offers are L2 and never VOICE"),
            _check("A-12", "B-21 / CC-01", bad_cons, "T1C / T2C offers in the action's families and generosity")]


def check_passes(sel: pd.DataFrame) -> list[dict]:
    """A-11 / A-12: pass-1 offers of T1C / T1F inside anchor < price < anchor + k; T2 band offers inside C-02."""
    t1 = sel[sel["GROUP_KEY"].isin(["t1c", "t1f"]) & (sel["PASS"] == 1)]
    k = t1.apply(lambda r: config.FREE_PRICE_K if r["GROUP_KEY"] == "t1f" else config.PRICE_BAND_K[r["ACTION"]], axis=1) \
        if len(t1) else pd.Series(dtype=float)
    bad1 = t1[~((t1["PRICE"] > t1["ANCHOR"] + EPS) & (t1["PRICE"] < t1["ANCHOR"] + k - EPS))] if len(t1) else t1
    t2 = sel[sel["GROUP_KEY"].isin(["t2c", "t2f"]) & sel["SELECTED_BY"].str.match(r"C[CF]-04")]
    bad2 = t2.iloc[:0]
    if len(t2):
        lo, hi = challenger_band(t2, config.CHALLENGER_BAND)
        _, hi_w = challenger_band(t2, config.CHALLENGER_BAND_WIDENED)
        top = hi.where(~t2["SELECTED_BY"].str.endswith("widened"), hi_w)
        bad2 = t2[(t2["PRICE"] < lo - EPS) | (t2["PRICE"] > top + EPS)]
    return [_check("A-12", "B-28 / FR-09", bad1, "Target 1 pass-1 offers inside anchor < price < anchor + k"),
            _check("A-12", "CC-02 / CF-02", bad2, "Target 2 offers inside the relative ARPU band")]


def audit_groups(panel: pd.DataFrame, selected: pd.DataFrame, recommendable: pd.DataFrame,
                 sizes: dict[str, int]) -> dict:
    checks = [_check("A-30", "sizes", check_sizes(panel, sizes), "group sizes within +-2 %, no MSISDN in two groups")]
    bal = balance(panel)
    imbalanced = bal[(bal["max_gap_points"] > config.BALANCE_MAX_POINTS) & (bal["chi2_p"] < 0.01)]
    checks.append(_check("A-31", "balance", [f"{r.field}={r.value}: gap {r.max_gap_points} pts (p={r.chi2_p})"
                                             for r in imbalanced.itertuples()],
                         "stratum shares within 2 points between groups (significant gaps only)"))
    ctrl = set(panel.loc[panel["GROUP_KEY"] == config.CONTROL_GROUP, VAL])
    if len(selected):
        checks += check_universe(selected) + check_passes(selected)
        checks.append(_check("A-12b", "B-20b", selected[~selected["BTL_NAME"].isin(set(recommendable["BTL_NAME"]))],
                             "every offer is in the catalogue and bought in the base in 270 days"))
        wrong = selected[selected["RULES_VERSION"] != selected["GROUP_KEY"].map(lambda k: config.GROUPS[k]["rules_version"])]
        checks.append(_check("A-17", "rules version", wrong, "each group uses only its own rules / approved version"))
        checks.append(_check("A-21", "H-01", selected[selected[VAL].isin(ctrl)], "no control line gets an offer"))
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else ("WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    return {"status": status, "push_blocked": status == "FAIL", "checks": checks}
