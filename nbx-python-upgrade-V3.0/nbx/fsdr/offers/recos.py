"""The reco tables of the four groups (update §5): code-2 tables + 8 reco tables in the original structure (B-32).

    out = build_group_recos(panel, cache)
    out["results_scoring_nba"], out["TG_40K_constraint_scored_nba"], out["recos_monthly_40k_constraint"], ...

Until the LLM agents exist (master Phases 5-6), the decisions are their documented deterministic fallbacks, and
every row says so:
  action  the group's rule action: action_original (Target 1) or action_agent (Target 2); DECIDED_BY = fallback_rule_action
  score   the NBO fallback order without helpers yet: E-O3 own past purchase, then E-O1 buyer share of the offer
          in the 270-day base; SCORE_SOURCE = fallback:E-O3+E-O1
  proba   the logged propensity: 1.0 for a deterministic fallback
10.1 consistency (constraint groups only): a customer with no offer at all for his action is sent back once with
his second action (action_agent rank 2, else the other dictionary's action, else the next B-10 prior); final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.offer_rules import load_catalog
from nbx.fsdr.offers import anchors as anc
from nbx.fsdr.offers import selection, universe

VAL = "ACCESS_METHOD_VAL"
VALIDITY_TABLE = {30: "recos_monthly", 7: "recos_weekly"}
RECO_COLUMNS = ["access_method_val", "item_id", "rank", "score", "btl_name", "type", "generosity", "service",
                "validity", "price", "action", "anchor", "pass", "selected_by", "score_source"]


def group_actions(panel: pd.DataFrame) -> pd.DataFrame:
    """ACTION of every contacted line = its group's rule action (fallback until ③ decides)."""
    out = panel[panel["GROUP_KEY"] != config.CONTROL_GROUP].copy()
    rule_col = out["GROUP_KEY"].map(lambda k: config.GROUPS[k]["rule_action"])
    picked = np.where(rule_col == "ACTION_AGENT", out["ACTION_AGENT"], out["ACTION_ORIGINAL"])
    out["ACTION"] = pd.Series(picked, index=out.index).where(pd.notna(picked), out["RULE_ACTION"])
    out["DECIDED_BY"] = "fallback_rule_action"
    return out


def second_action(row: pd.Series) -> str | None:
    for a in (row.get("ACTION_AGENT_RANK2"), row.get("ACTION_AGENT"), row.get("ACTION_ORIGINAL")):
        if isinstance(a, str) and a != row["ACTION"]:
            return a
    return next(a for a in sorted(config.ACTION_PRIOR, key=config.ACTION_PRIOR.get, reverse=True) if a != row["ACTION"])


def fallback_scores(cands: pd.DataFrame, bought: pd.DataFrame, base_lines: int) -> pd.Series:
    """0.5 x (the customer bought it in 270 days) + 0.5 x (buyer share of the offer / the best buyer share)."""
    share = bought.groupby("BTL_NAME")[VAL].nunique() / max(base_lines, 1)
    share = share / share.max() if len(share) else share
    own = pd.MultiIndex.from_frame(cands[[VAL, "BTL_NAME"]]).isin(
        pd.MultiIndex.from_frame(bought[[VAL, "BTL_NAME"]].drop_duplicates()))
    return (0.5 * own + 0.5 * cands["BTL_NAME"].map(share).fillna(0.0)).round(6)


def _candidates(key: str, cust: pd.DataFrame, rec: pd.DataFrame, validity: int, ctx: dict) -> pd.DataFrame:
    g = config.GROUPS[key]
    if g["universe"] == "constraint":
        c = universe.constraint_candidates(cust, rec, validity, ctx["bought"])
    else:
        c = universe.free_candidates(cust, rec, validity, ctx["bought"])
    if not len(c):
        return c
    c = c.assign(SCORE=fallback_scores(c, ctx["bought"], ctx["base_lines"]), SCORE_SOURCE="fallback:E-O3+E-O1")
    c = c.sort_values([VAL, "SCORE", "PRICE"], ascending=[True, False, False], kind="mergesort")
    c = c.assign(RANK=c.groupby(VAL).cumcount() + 1)
    c = c[c["RANK"] <= config.MAX_CANDIDATES_PER_VALIDITY]                 # §10: at most 25 per validity
    c = c.assign(ANCHOR=anc.anchor_for(c, ctx["anchors"]))
    if g["rules"] in ("CC", "CF"):
        c = c.assign(CH_ANCHOR=anc.challenger_anchor(c, ctx["anchors"], ctx["share"], ctx["micro_median"]))
    return c.reset_index(drop=True)


def _select_group(key: str, cust: pd.DataFrame, rec: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    parts = []
    for validity in (30, 7):
        c = _candidates(key, cust, rec, validity, ctx)
        if len(c):
            sel = selection.select(key, c)
            parts.append(sel.assign(N_CANDIDATES=sel[VAL].map(c.groupby(VAL).size())))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[VAL])


def _micro_median(panel: pd.DataFrame, anchors: pd.DataFrame) -> pd.Series:
    """C-02: a customer with no spend gets the median anchor of his micro-segment (positive anchors only)."""
    a = anchors["AVG_REV_MONTHLY"].reindex(panel[VAL]).to_numpy()
    df = pd.DataFrame({VAL: panel[VAL].to_numpy(), "MS": panel["MICRO_SEGMENT"].to_numpy(), "A": a})
    med = df[df["A"] > 0].groupby("MS")["A"].median()
    return pd.Series(df["MS"].map(med).to_numpy(), index=df[VAL])


def build_group_recos(panel: pd.DataFrame, src, no_contact: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """panel: assign_panel output joined with the T3 segment columns. src: the run's cache (CacheSource)."""
    catalog = load_catalog(src)
    bought = universe.purchases(src)
    bought = bought[bought["BTL_NAME"].isin(set(catalog["BTL_NAME"]))]          # B-20: catalogue offers only
    rec = universe.recommendable(catalog, bought)
    item_id = {b: i for i, b in enumerate(rec["BTL_NAME"])}
    a = anc.b26_anchors(src)
    ctx = {"bought": bought, "base_lines": int(src.read("base_prepaid").shape[0]) if src.has("base_prepaid") else len(panel),
           "anchors": a, "share": anc.data_share(src), "micro_median": _micro_median(panel, a)}
    contacted = group_actions(panel)
    if no_contact:                                                             # C-05: do-not-disturb (Target 2)
        dnd = contacted[VAL].isin(no_contact) & contacted["TARGET"].eq("target2")
        contacted.loc[dnd, "DECIDED_BY"] = "no_contact_C-05"
    contacted["CONSISTENCY_RETRY"] = False
    out: dict[str, pd.DataFrame] = {}
    selected = []
    for key in config.GROUPS:
        cust = contacted[(contacted["GROUP_KEY"] == key) & (contacted["DECIDED_BY"] != "no_contact_C-05")]
        if not len(cust):
            continue
        sel = _select_group(key, cust, rec, ctx)
        if config.GROUPS[key]["consistency_loop"]:                              # 10.1: once, the second choice is final
            lost = cust[~cust[VAL].isin(sel[VAL] if len(sel) else [])]
            if len(lost):
                retry = lost.assign(ACTION=lost.apply(second_action, axis=1))
                sel = pd.concat([sel, _select_group(key, retry, rec, ctx)], ignore_index=True)
                second = retry.set_index(VAL)["ACTION"]
                m = contacted[VAL].isin(second.index)
                contacted.loc[m, "ACTION"] = contacted.loc[m, VAL].map(second)
                contacted.loc[m, "CONSISTENCY_RETRY"] = True
        if len(sel):
            selected.append(sel.assign(GROUP_KEY=key, RULES_VERSION=config.GROUPS[key]["rules_version"]))
        g = config.GROUPS[key]
        for validity, prefix in VALIDITY_TABLE.items():
            part = sel[sel["VALIDITY"] == validity] if len(sel) else sel
            out[g[prefix]] = reco_table(part, g, item_id)
    out["results_scoring_nba"] = results_scoring(contacted)
    for key, g in config.GROUPS.items():
        out[g["scored_table"]] = out["results_scoring_nba"][out["results_scoring_nba"]["tg_flag"] == g["tg_flag"]].reset_index(drop=True)
    out["_selected"] = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    out["_decisions"] = contacted
    out["_recommendable"] = rec
    return out


def reco_table(sel: pd.DataFrame, group: dict, item_id: dict) -> pd.DataFrame:
    """B-32 structure, lower-case as code 3 / code 4; the action column only in the constraint tables."""
    if not len(sel):
        cols = RECO_COLUMNS if group["action_column"] else [c for c in RECO_COLUMNS if c != "action"]
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame({"access_method_val": sel[VAL], "item_id": sel["BTL_NAME"].map(item_id), "rank": sel["RANK"],
                      "score": sel["SCORE"], "btl_name": sel["BTL_NAME"], "type": sel.get("TYPE"),
                      "generosity": sel["GENEROSITY"], "service": sel["SERVICE"], "validity": sel["VALIDITY"],
                      "price": sel["PRICE"], "action": sel["ACTION"],
                      "anchor": (sel["CH_ANCHOR"] if "CH_ANCHOR" in sel.columns and sel["CH_ANCHOR"].notna().any()
                                 else sel["ANCHOR"]).round(6),
                      "pass": sel["PASS"], "selected_by": sel["SELECTED_BY"], "score_source": sel["SCORE_SOURCE"]})
    if not group["action_column"]:
        t = t.drop(columns="action")
    return t.reset_index(drop=True)


def results_scoring(contacted: pd.DataFrame) -> pd.DataFrame:
    """B-12 results_scoring_nba: access_method_val, action, proba (= logged propensity), tg_flag."""
    c = contacted[contacted["DECIDED_BY"] != "no_contact_C-05"]
    return pd.DataFrame({"access_method_val": c[VAL].to_numpy(), "action": c["ACTION"].to_numpy(),
                         "proba": 1.0, "tg_flag": c["GROUP_KEY"].map(lambda k: config.GROUPS[k]["tg_flag"]).to_numpy()})
