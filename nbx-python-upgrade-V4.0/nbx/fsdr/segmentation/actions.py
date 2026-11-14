"""②b The two actions of every line (master prompt §7.1), plus the week-0 history of code 1 (B-02, B-03, B-09, B-10).

action_original   code 1's dict_segm_actions (case-sensitive, last value wins) applied to the line's segment:
                  the segment-engine segment of targeting_dataiku_auto when the line has one this week, otherwise
                  the S-rule segment translated by the crosswalk (config.SEGMENT_CROSSWALK). Deterministic, no LLM.
action_agent      the agent dictionary dict_segm_actions_agent_vN: per S-segment x nationality group, ranked actions.
                  Week 0 it is a copy of the original dictionary (v0); the LLM changes cells from Phase 4 on, and
                  every version is validated here before it is applied (invalid -> the previous cell is kept).
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.features import norm_id, norm_val

ENGINE, CROSSWALK = "segment_engine", "crosswalk"
AT_RISK_SEGMENTS = {s for _, s, _ in config.S_RULES if s.endswith("_At_Risk") and "Not_At_Risk" not in s}
# H-06 for the dictionary: an at-risk segment may not be moved TO an upsell action. Its original action is kept
# allowed (Onboarding_Early_Churn_At_Risk is UPSELL in code 1). CHECK item (plan §11).
UPSELL_ACTIONS = {"UPSELL", "UPSELL DATA", "UPSELL VOICE"}


def is_nursering(segment: pd.Series) -> pd.Series:
    return segment.astype("string").str.lower().str.startswith("nursering").fillna(False)


# ---------------------------------------------------------------------------
# Crosswalk and action_original
# ---------------------------------------------------------------------------
def crosswalk_table() -> pd.DataFrame:
    """One row per S-rule: the original segment name it maps to, both actions, and the CHECK note."""
    rows = []
    for rid, seg, rule_action in config.S_RULES:
        orig, note = config.SEGMENT_CROSSWALK[seg]
        orig_action = config.DICT_SEGM_ACTIONS.get(orig)
        rows.append({"rule_id": rid, "s_segment": seg, "original_segment": orig, "rule_action": rule_action or "excluded",
                     "original_action": orig_action or "excluded", "same_action": (rule_action == orig_action),
                     "check": "CHECK", "note": note})
    return pd.DataFrame(rows)


def week0_targeting(tg: pd.DataFrame | None, fulfil: pd.DataFrame | None, catalog: pd.DataFrame | None,
                    run_date: date, lines: pd.DataFrame) -> pd.DataFrame:
    """Per line in targeting_dataiku_auto: the latest segment (W0_SEGMENT), its original action (B-03; rows without
    a segment or in Nursering get none, B-02) and the B-09 reward: a catalogue BTL fulfilled in D-7 … D-1.
    lines: ACCESS_METHOD_ID -> ACCESS_METHOD_VAL of the base (targeting carries the id, fulfilments the MSISDN)."""
    cols = ["ACCESS_METHOD_ID", "W0_SEGMENT", "W0_DATE", "W0_ACTION", "W0_REWARD", "W0_BTL_REVENUE"]
    if tg is None or not len(tg):
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame({"ACCESS_METHOD_ID": norm_id(tg["ACCESS_METHOD_ID"]),
                      "W0_SEGMENT": tg["SEGMENT"].astype("string").str.strip().astype(object),
                      "W0_DATE": pd.to_datetime(tg["DATE1"], errors="coerce")})
    t = t.dropna(subset=["ACCESS_METHOD_ID"]).sort_values("W0_DATE", kind="stable").drop_duplicates("ACCESS_METHOD_ID", keep="last")
    t["W0_SEGMENT"] = t["W0_SEGMENT"].where(t["W0_SEGMENT"].notna(), None)
    t["W0_ACTION"] = t["W0_SEGMENT"].map(lambda s: config.DICT_SEGM_ACTIONS.get(s) if isinstance(s, str) else None)
    t = t.merge(lines[["ACCESS_METHOD_ID", "ACCESS_METHOD_VAL"]].drop_duplicates("ACCESS_METHOD_ID"),
                on="ACCESS_METHOD_ID", how="left")
    t["W0_REWARD"] = 0
    t["W0_BTL_REVENUE"] = 0.0
    return t[cols].reset_index(drop=True) if fulfil is None else _rewards(t, fulfil, catalog, run_date)[cols]


def _rewards(t: pd.DataFrame, fulfil: pd.DataFrame, catalog: pd.DataFrame | None, run_date: date) -> pd.DataFrame:
    from nbx.fsdr import calendar as cal

    w = cal.campaign_windows(run_date)["reward"]
    f = pd.DataFrame({"ACCESS_METHOD_VAL": norm_val(fulfil["ACCESS_METHOD_VAL"]),
                      "PRODUCTNAME": fulfil["PRODUCTNAME"].astype(str).str.strip(),
                      "DAY": pd.to_datetime(fulfil["FULFILMENTSUCCESSDATE_TIME"], errors="coerce").dt.date})
    f = f.dropna(subset=["ACCESS_METHOD_VAL", "DAY"])
    f = f[(f["DAY"] >= w.start) & (f["DAY"] <= w.end)]
    if catalog is not None:
        price = catalog.set_index("BTL_NAME")["PRICE"]
        f = f[f["PRODUCTNAME"].isin(price.index)]
        f = f.assign(PRICE=f["PRODUCTNAME"].map(price).astype("float64").fillna(0.0))
    else:
        f = f.assign(PRICE=0.0)
    by_line = f.groupby("ACCESS_METHOD_VAL")["PRICE"].agg(["size", "sum"])
    t = t.reset_index(drop=True)
    vals = t["ACCESS_METHOD_VAL"]
    t["W0_REWARD"] = vals.map(by_line["size"]).fillna(0).gt(0).astype("int8")
    t["W0_BTL_REVENUE"] = vals.map(by_line["sum"]).fillna(0.0).round(3)
    return t


def apply_original(seg: pd.DataFrame) -> pd.DataFrame:
    """ORIGINAL_SEGMENT, ACTION_ORIGINAL, ACTION_ORIGINAL_SOURCE from W0_SEGMENT (engine) or the crosswalk.
    An engine segment that the dictionary does not know (spelling) falls back to the crosswalk; an engine Nursering
    segment is kept (the line gets no original action, as in code 1)."""
    w0 = seg["W0_SEGMENT"] if "W0_SEGMENT" in seg.columns else pd.Series(None, index=seg.index, dtype=object)
    known = w0.map(lambda s: isinstance(s, str) and s in config.DICT_SEGM_ACTIONS) | is_nursering(w0)
    cross = seg["SEGMENT"].map(lambda s: config.SEGMENT_CROSSWALK[s][0])
    seg["ORIGINAL_SEGMENT"] = np.where(known, w0, cross).astype(object)
    seg["ACTION_ORIGINAL"] = seg["ORIGINAL_SEGMENT"].map(config.DICT_SEGM_ACTIONS).astype(object)
    seg["ACTION_ORIGINAL"] = seg["ACTION_ORIGINAL"].where(seg["ACTION_ORIGINAL"].notna(), None)
    seg["ACTION_ORIGINAL_SOURCE"] = np.where(known, ENGINE, CROSSWALK).astype(object)
    return seg


# ---------------------------------------------------------------------------
# Agent dictionary (dict_segm_actions_agent_vN)
# ---------------------------------------------------------------------------
def initial_agent_dictionary(run_date: date) -> dict:
    """v0 = the original dictionary through the crosswalk, for every S-segment x nationality group (70 cells)."""
    cells = []
    for _, seg, _ in config.S_RULES:
        orig = config.DICT_SEGM_ACTIONS.get(config.SEGMENT_CROSSWALK[seg][0])
        for nat in config.NATIONALITY_GROUPS:
            cells.append({"segment": seg, "nationality_group": nat, "excluded": orig is None,
                          "actions": [] if orig is None else
                          [{"action": orig, "rank": 1, "score": 1.0, "evidence_ids": ["B-03"],
                            "reason": "week 0: copy of the original dictionary"}]})
    return {"version": 0, "name": f"{config.AGENT_DICTIONARY_PREFIX}0", "created": run_date.isoformat(),
            "parent": None, "made_by": "code (copy of dict_segm_actions)", "cells": cells}


def validate_agent_dictionary(d: dict, previous: dict | None = None) -> tuple[dict, list[dict]]:
    """⑥ check of every cell: known segment and nationality, allowed actions only, excluded segments stay excluded,
    no aggressive upsell in an at-risk segment (H-06). An invalid cell is replaced by the previous version's cell
    (or v0's). Returns the valid dictionary and the problems found."""
    base = {(c["segment"], c["nationality_group"]): c
            for c in (previous or initial_agent_dictionary(date.fromisoformat(d.get("created", "2000-01-01"))))["cells"]}
    problems, cells, seen = [], [], set()
    for c in d.get("cells", []):
        key = (c.get("segment"), c.get("nationality_group"))
        why = []
        if key[0] not in config.S_SEGMENTS or key[1] not in config.NATIONALITY_GROUPS:
            why.append("unknown segment or nationality group")
        acts = [a.get("action") for a in c.get("actions", [])]
        if any(a not in config.ACTIONS for a in acts):
            why.append(f"action not allowed: {[a for a in acts if a not in config.ACTIONS]}")
        original = config.DICT_SEGM_ACTIONS.get(config.SEGMENT_CROSSWALK.get(key[0], (None,))[0])
        excluded = original is None
        if excluded and acts:
            why.append("excluded segment got an action (B-02)")
        if not excluded and not acts:
            why.append("no action for a targetable segment")
        if key[0] in AT_RISK_SEGMENTS and acts[:1] and acts[0] in UPSELL_ACTIONS and acts[0] != original:
            why.append("upsell as top action of an at-risk segment (H-06)")
        if len(acts) != len(set(acts)) or len(acts) > 3:
            why.append("ranked actions must be 1-3 distinct ids")
        if why:
            problems.append({"segment": key[0], "nationality_group": key[1], "problems": why})
            if key in base and key not in seen:
                cells.append(base[key])
                seen.add(key)
        elif key not in seen:
            cells.append(c)
            seen.add(key)
    for key, c in base.items():                                   # a missing cell keeps its previous value
        if key not in seen:
            cells.append(c)
            problems.append({"segment": key[0], "nationality_group": key[1], "problems": ["missing cell (kept previous)"]})
    return {**d, "cells": cells}, problems


def agent_lookup(d: dict) -> pd.DataFrame:
    rows = [{"SEGMENT": c["segment"], "NATIONALITY_GROUP": c["nationality_group"],
             "ACTION_AGENT": (c["actions"][0]["action"] if c["actions"] else None),
             "ACTION_AGENT_RANK2": (c["actions"][1]["action"] if len(c["actions"]) > 1 else None)} for c in d["cells"]]
    return pd.DataFrame(rows)


def apply_agent_dictionary(seg: pd.DataFrame, d: dict) -> pd.DataFrame:
    """ACTION_AGENT, ACTION_AGENT_RANK2, ACTION_AGENT_VERSION of every line, by (S-segment, nationality group)."""
    lk = agent_lookup(d)
    out = seg[["SEGMENT", "NATIONALITY_GROUP"]].merge(lk, on=["SEGMENT", "NATIONALITY_GROUP"], how="left")
    seg["ACTION_AGENT"] = out["ACTION_AGENT"].where(out["ACTION_AGENT"].notna(), None).to_numpy()
    seg["ACTION_AGENT_RANK2"] = out["ACTION_AGENT_RANK2"].where(out["ACTION_AGENT_RANK2"].notna(), None).to_numpy()
    seg["ACTION_AGENT_VERSION"] = d["name"]
    return seg


def dictionary_files(step_dir: Path) -> list[Path]:
    rx = re.compile(re.escape(config.AGENT_DICTIONARY_PREFIX) + r"(\d+)\.json$")
    files = [p for p in Path(step_dir).glob(f"{config.AGENT_DICTIONARY_PREFIX}*.json") if rx.search(p.name)]
    return sorted(files, key=lambda p: int(rx.search(p.name).group(1)))


def previous_agent_dictionary(runs_root: Path, run_date: date) -> tuple[dict | None, Path | None]:
    """The latest agent dictionary of an EARLIER run date (data/runs/<D'>/train/T3_segments/), if any."""
    root = Path(runs_root)
    dates = sorted(p.name for p in root.glob("20??-??-??") if p.is_dir() and p.name < run_date.isoformat())
    for d in reversed(dates):
        files = dictionary_files(root / d / "train" / "T3_segments")
        if files:
            return json.loads(files[-1].read_text(encoding="utf-8")), files[-1]
    return None, None


def action_agreement(original: pd.Series, agent: pd.Series, sent: pd.Series) -> np.ndarray:
    """§7.1 log field: all_same / sent_original / sent_agent / other."""
    o, a, s = (x.astype("string") for x in (original, agent, sent))
    so, sa = (s == o).fillna(False).to_numpy(bool), (s == a).fillna(False).to_numpy(bool)
    return np.select([so & sa, so, sa], ["all_same", "sent_original", "sent_agent"], "other").astype(object)
