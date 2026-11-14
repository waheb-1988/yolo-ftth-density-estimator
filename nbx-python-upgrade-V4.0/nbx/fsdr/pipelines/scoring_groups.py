"""Scoring steps of the four groups (plan §14, update §3 - §5), run after training T2 + T3:

    run_p1_panel(D)    score/P1_panel/    panel (T1C / T1F / T2C / T2F / control), group sizes, balance (A-30 / A-31)
    run_p6_offers(D)   score/P6_offers/   results_scoring_nba, the 4 code-2 tables, the 8 reco tables (FSDR CSV
                                          standard .csv.gz + schema + _outputs_manifest.json), traces, group audit

V4.0, each target scored separately (its own steps, its own output folder):

    run_p6_offers(D, target="target1")   score/P6_offers_target1/   the tables of T1C (a) and T1F (b) only
    run_target_final(D, "target1")       data/output/<D>/target1/   input_flytxt_target1 (on PASS), a_constraint/ and
                                                                    b_free/ files, details, checks, summary
    run_push(D)                          data/output/<D>/push/      both targets merged when both PASS; control/
    run_targets(D)                       P1, then P6 + final of each target, then the push

P6 uses the documented fallbacks of the LLM agents until Phases 5-6 build them (recos.py says which); the rules,
universes, anchors, passes and bands are the final ones.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import audit_groups, config, csvstd, llm_budget
from nbx.fsdr import targets as targets_mod
from nbx.fsdr.contract.cache import CacheSource
from nbx.fsdr.features import norm_val
from nbx.fsdr.offers.recos import build_group_recos
from nbx.fsdr.panel import assign_panel, balance, check_sizes, eligible_pool, panel_group, stratum
from nbx.fsdr.pipelines.training import T2, T3, StageBlocked
from nbx.fsdr.steps import Step, fingerprint, read_step, runs_dir

P1, P6 = "P1_panel", "P6_offers"
VAL = "ACCESS_METHOD_VAL"
SEG_COLS = ["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "SEGMENT", "RULE_ACTION", "EXCLUDED", "NATIONALITY_GROUP",
            "AGE_BAND", "TENURE_BAND", "MICRO_SEGMENT", "VALUE_BAND", "USAGE_TYPE", "ACTION_ORIGINAL", "ACTION_AGENT",
            "ACTION_AGENT_RANK2"]


def _parts(folder: Path, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(Path(folder).glob("part-*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True) if files else pd.DataFrame()


def base_segments(run_date: date, runs_root: Path | None = None) -> pd.DataFrame:
    train = runs_dir(run_date, runs_root) / "train"
    for st in (T2, T3):
        if read_step(train / st).get("status") != "done":
            raise StageBlocked(f"{st} of {run_date} is not done: run fsdr-features first")
    seg = _parts(train / T3 / "segments", SEG_COLS)
    plan = _parts(train / T2 / "base_features", [VAL, "CURRENT_PLAN"])
    return seg.merge(plan.drop_duplicates(VAL), on=VAL, how="left")


def other_campaigns(cache: CacheSource, run_date: date) -> set[str]:
    """H-05: lines in another campaign this week (targeting_auto START_DATE in D-7 … D)."""
    if not cache.has("targeting_auto"):
        return set()
    t = cache.read("targeting_auto")
    start = pd.to_datetime(t.get("START_DATE"), errors="coerce").dt.date
    lo, hi = run_date - timedelta(days=7), run_date
    return set(norm_val(t.loc[(start >= lo) & (start <= hi), "MSISDN"]).dropna())


def _sample(cache: CacheSource) -> pd.DataFrame | None:
    if not cache.has(config.TARGET_SAMPLE_TABLE):
        return None
    s = cache.read(config.TARGET_SAMPLE_TABLE)
    return pd.DataFrame({VAL: norm_val(s["access_method_val"]), "group": s["group"].astype(str)})


def input_folder(run_date: date, runs_root: Path | None = None) -> Path:
    """The folder of the target lists: the scoring twin T0 loaded (V4.0 data/scoring/<D>/), else the single input
    folder of a V3.0 run (data/input/<D>/), else data/scoring/<D>/."""
    from nbx.fsdr.contract import data_agent

    t0 = data_agent.load_summary(run_date, runs_root)
    src = t0.get("scoring_source") or t0.get("source")
    return Path(src) if src else config.scoring_folder(run_date.isoformat())


def generate_targets(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None,
                     fmt: str = "csv.gz", seed: int = config.GENERATOR_SEED, folder: Path | None = None) -> dict:
    """Dummy target lists for tests: a stratified draw of the eligible base, written to <input>/targets/."""
    seg = base_segments(run_date, runs_root)
    cache = CacheSource(run_date, cache_root)
    busy = other_campaigns(cache, run_date)
    pool = eligible_pool(seg, exclude=busy)
    panel, rep = assign_panel(pool, config.group_sizes(), seed)
    folder = Path(folder or input_folder(run_date, runs_root))
    files = targets_mod.write_targets(panel, folder, fmt, {"run_date": run_date.isoformat(), "seed": seed,
                                                            "scale_factor": rep["scale_factor"]})
    return {"folder": str(targets_mod.targets_dir(folder)), "files": [str(f) for f in files], **rep}


def _panel_from_targets(folder: Path, seg: pd.DataFrame, busy: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    t, issues = targets_mod.read_targets(folder)
    kept, rejected, more = targets_mod.check_targets(t, seg, busy, targets_mod.expected_sizes(folder))
    issues += more
    panel = kept.merge(seg, on=VAL, how="left")
    panel = panel.assign(STRATUM=stratum(panel), SOURCE="uploaded",
                         PANEL_GROUP=panel["GROUP_KEY"].map(panel_group),
                         TARGET=panel["GROUP_KEY"].map(lambda k: "control" if k == config.CONTROL_GROUP
                                                       else config.GROUPS[k]["target"]))
    sizes = panel["GROUP_KEY"].value_counts().reindex(list(config.group_sizes()), fill_value=0).astype(int).to_dict()
    rep = {"mode": "uploaded", "pool": int(len(t)), "requested": sizes, "configured": config.group_sizes(),
           "scale_factor": 1.0, "sizes": sizes, "from_sample": 0, "rejected": int(len(rejected))}
    return panel, rejected, issues, rep


def run_p1_panel(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None,
                 use_sample: bool = True, seed: int = config.GENERATOR_SEED, force: bool = False) -> dict:
    """P1: the uploaded target lists when <input>/targets/ holds them, otherwise a stratified random draw."""
    seg = base_segments(run_date, runs_root)
    cache = CacheSource(run_date, cache_root)
    t3 = read_step(runs_dir(run_date, runs_root) / "train" / T3)
    folder = input_folder(run_date, runs_root)
    uploaded = targets_mod.has_targets(folder)
    tfiles = targets_mod.list_files(folder) if uploaded else []
    inputs = fingerprint({"t3": t3.get("started"), "sizes": config.group_sizes(), "seed": seed, "sample": use_sample,
                          "t2_split": config.T2_SPLIT, "targets": fingerprint(tfiles) if tfiles else ""})
    with Step(runs_dir(run_date, runs_root) / "score", P1, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        busy = other_campaigns(cache, run_date)
        issues: list = []
        if uploaded:
            panel, rejected, issues, rep = _panel_from_targets(folder, seg, busy)
            st.write_csv(rejected, "targets_rejected.csv")
            st.write_json("targets_check.json", issues)
            pool_n = rep["pool"]
        else:
            pool = eligible_pool(seg, exclude=busy)
            panel, rep = assign_panel(pool, config.group_sizes(), seed, _sample(cache) if use_sample else None)
            rep["mode"] = "drawn"
            pool_n = len(pool)
        st.write_table(panel, "panel")
        counts = pd.DataFrame([{"group_key": k, "panel_group": panel_group(k), "requested": v,
                                "assigned": int((panel["GROUP_KEY"] == k).sum())} for k, v in rep["requested"].items()])
        st.write_csv(counts, "group_sizes.csv")
        bal = balance(panel) if len(panel) else pd.DataFrame(columns=["max_gap_points"])
        st.write_csv(bal, "balance.csv")
        blocking = [i for i in issues if i["level"] == "blocking"]
        summary = {"run_date": run_date.isoformat(), "stage": P1, **rep, "eligible_base": int(len(seg)),
                   "excluded_nursering": int((seg["EXCLUDED"] == 1).sum()), "in_other_campaigns": len(busy),
                   "panel": len(panel), "size_problems": check_sizes(panel, rep["requested"]),
                   "target_issues": issues, "status": "blocked" if blocking else "done",
                   "max_balance_gap_points": float(bal["max_gap_points"].max()) if len(bal) else 0.0,
                   "targets_folder": str(targets_mod.targets_dir(folder)) if uploaded else None, "step_dir": str(st.dir)}
        if blocking:
            st.block("; ".join(i["message"] for i in blocking))
        st.rows(rows_in=pool_n, rows_out=len(panel))
        st.write_json("summary.json", summary)
        return summary


def _traces(out: dict, panel: pd.DataFrame, per_group: int = 2) -> pd.DataFrame:
    """Update §12: 2 decisions per group for the SAME type of customer (segment x nationality x value band)."""
    dec, sel = out["_decisions"], out["_selected"]
    if not len(sel):
        return pd.DataFrame()
    typ = dec.assign(T=dec["SEGMENT"].astype(str) + " | " + dec["NATIONALITY_GROUP"].astype(str) + " | "
                     + dec["VALUE_BAND"].astype(str))
    with_offer = typ[typ[VAL].isin(sel[VAL])]
    present = with_offer.groupby("T")["GROUP_KEY"].nunique()
    common = present[present == present.max()].index
    t = with_offer[with_offer["T"].isin(common)]["T"].value_counts().index[0]
    rows = []
    for key in config.GROUPS:
        for v in with_offer.loc[(with_offer["T"] == t) & (with_offer["GROUP_KEY"] == key), VAL].head(per_group):
            d = dec[dec[VAL] == v].iloc[0]
            for r in sel[sel[VAL] == v].itertuples():
                rows.append({"group": key, "customer_type": t, VAL: v, "rule_action_source": config.GROUPS[key]["rule_action"],
                             "action": d["ACTION"], "decided_by": d["DECIDED_BY"], "consistency_retry": d["CONSISTENCY_RETRY"],
                             "validity": r.VALIDITY, "candidates": r.N_CANDIDATES, "btl_name": r.BTL_NAME,
                             "service": r.SERVICE, "price": r.PRICE, "score": r.SCORE,
                             "anchor": getattr(r, "CH_ANCHOR", None) if key in ("t2c", "t2f") else r.ANCHOR,
                             "pass": r.PASS, "selected_by": r.SELECTED_BY, "rules_version": r.RULES_VERSION})
    return pd.DataFrame(rows)


def target_keys(target: str | None) -> list[str]:
    """The group keys of a target ("target1" -> t1c, t1f), all four for None."""
    if target is None:
        return list(config.GROUPS)
    if target not in config.TARGETS:
        raise ValueError(f"target must be one of {list(config.TARGETS)}")
    return [k for k in config.TARGETS[target] if k in config.group_sizes()]


def p6_name(target: str | None) -> str:
    return P6 if target is None else f"{P6}_{target}"


def run_p6_offers(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None,
                  seconds_per_customer: float = 0.36, force: bool = False, target: str | None = None) -> dict:
    """P6 of the four groups (target None), or of one target only (V4.0: target1 = T1C + T1F, target2 = T2C + T2F)."""
    score = runs_dir(run_date, runs_root) / "score"
    p1 = read_step(score / P1)
    if p1.get("status") != "done":
        raise StageBlocked(f"{P1} of {run_date} is {p1.get('status', 'missing')}: run fsdr-panel (and fix the "
                           f"target lists if it is blocked)")
    keys = target_keys(target)
    inputs = fingerprint({"p1": p1.get("started"), "groups": config.GROUPS, "band": config.CHALLENGER_BAND,
                          "keys": keys, "reco_offers": config.RECO_OFFERS})
    with Step(score, p6_name(target), inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        panel = pd.read_parquet(score / P1 / "panel.parquet")
        panel = panel[panel["GROUP_KEY"].isin(keys + [config.CONTROL_GROUP])].reset_index(drop=True)
        cache = CacheSource(run_date, cache_root)
        out = build_group_recos(panel, cache)
        own = {"results_scoring_nba"} | {config.GROUPS[k][t] for k in keys
                                         for t in ("scored_table", "recos_monthly", "recos_weekly")}
        files = []
        for name, df in out.items():
            if not name.startswith("_") and name in own:
                files.append(csvstd.write_csv(df, st.output(f"{name}{csvstd.SUFFIX}")))
                st.outputs.append(csvstd.schema_path(files[-1]).name)
        st.outputs.append(csvstd.write_outputs_manifest(st.dir, files, run_id=f"score_{run_date.isoformat()}"
                                                        + (f"_{target}" if target else ""),
                                                        versions={k: config.GROUPS[k]["rules_version"] for k in keys}).name)
        st.write_table(out["_decisions"], "decisions", csv="none")
        tr = _traces(out, panel)
        st.write_csv(tr, "traces.csv")
        sizes = json.loads((score / P1 / "summary.json").read_text(encoding="utf-8"))["requested"]   # scaled if small
        sizes = {k: v for k, v in sizes.items() if k in keys or k == config.CONTROL_GROUP}
        audit = audit_groups.audit_groups(panel, out["_selected"], out["_recommendable"], sizes)
        st.write_json("audit_groups.json", audit)
        q = llm_budget.apply_budget(llm_budget.llm_queue(out["_decisions"]), seconds_per_customer)
        sel = out["_selected"]
        per_group = []
        for key in keys:
            g = config.GROUPS[key]
            d = out["_decisions"][out["_decisions"]["GROUP_KEY"] == key]
            s = sel[sel["GROUP_KEY"] == key] if len(sel) else sel
            per_group.append({"group": key, "rules": g["rules_version"], "contacted": len(d),
                              "with_offers": int(s[VAL].nunique()) if len(s) else 0,
                              "offers": len(s), "consistency_retry": int(d["CONSISTENCY_RETRY"].sum()),
                              "by_rule": s["SELECTED_BY"].value_counts().to_dict() if len(s) else {},
                              "llm_coverage_at_budget": float(llm_budget.coverage(q).get(key, 0.0))})
        st.write_csv(pd.DataFrame(per_group).assign(by_rule=lambda d: d["by_rule"].map(json.dumps)), "group_report.csv")
        summary = {"run_date": run_date.isoformat(), "stage": p6_name(target), "target": target or "all",
                   "decisions": "fallback (LLM agents: Phases 5-6)", "reco_offers": config.RECO_OFFERS,
                   "tables": [n for n in out if not n.startswith("_") and n in own], "groups": per_group,
                   "audit_status": audit["status"], "push_blocked": audit["push_blocked"],
                   "step_dir": str(st.dir)}
        st.rows(rows_in=len(panel), rows_out=len(sel))
        st.write_json("summary.json", summary)
        return summary


# ---------------------------------------------------------------------------
# P7 final: input_flytxt (push) + per-target files + details (the "big data" behind every row)
# ---------------------------------------------------------------------------
FINAL = "final"
TARGET_DIRS = {"t1c": "target1", "t1f": "target1", "t2c": "target2", "t2f": "target2", "control": "control"}
RECO_FIELDS = ["btl_name", "service", "generosity", "price", "score", "anchor", "pass", "selected_by"]


def _read_reco(p6: Path, table: str) -> pd.DataFrame:
    p = p6 / f"{table}{csvstd.SUFFIX}"
    df = csvstd.read_csv(p) if p.exists() else pd.DataFrame(columns=["access_method_val", "btl_name", "price"])
    for c in ("price", "score", "anchor", "pass"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _flat_recos(recos: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Up to RECOS_PER_VALIDITY offers per customer (2, or 3 with 6 offers), price desc -> reco_<prefix>1_* …"""
    if not len(recos):
        return pd.DataFrame(index=pd.Index([], name=VAL))
    r = recos.assign(_K=recos.groupby("access_method_val")["price"].rank(method="first", ascending=False).astype(int))
    parts = []
    for k in range(1, config.RECOS_PER_VALIDITY + 1):
        top = r[r["_K"] == k].set_index("access_method_val")[[c for c in RECO_FIELDS if c in r.columns]]
        parts.append(top.add_prefix(f"reco_{prefix}{k}_"))
    return pd.concat(parts, axis=1).rename_axis(VAL)


def output_checks(flytxt: pd.DataFrame, control: set[str], catalog_names: set[str], run_date: date) -> list[dict]:
    """A-20 … A-24 on the file to push (the full set comes with Phase 7)."""
    offers = pd.unique(flytxt[[f"destination_offer_name_{i}" for i in range(1, 7)]].to_numpy().ravel())
    not_cat = sorted(o for o in offers if o and o not in catalog_names)
    checks = [("A-20", "38 columns in the F-19 order", list(flytxt.columns) == config.INPUT_FLYTXT_COLUMNS),
              ("A-21", "one row per msisdn, 8-digit text", bool(flytxt["msisdn"].is_unique
               and flytxt["msisdn"].astype(str).str.fullmatch(r"\d{8}").all())),
              ("A-21", "no control msisdn in the file", not (set(flytxt["msisdn"]) & control)),
              ("A-22", "date / start_date / end_date / channels / flag_splio",
               bool((flytxt["date"] == run_date.isoformat()).all()
                    and (flytxt["start_date"] == (run_date + timedelta(days=1)).isoformat()).all()
                    and (flytxt["end_date"] == "").all() and (flytxt["flag_splio"] == 1).all())),
              ("A-23", f"every offer in the catalogue ({len(not_cat)} not)", not not_cat),
              ("A-24", "segment is one of the 7 actions", bool(flytxt["segment"].isin(config.ACTIONS).all()))]
    return [{"id": i, "check": what, "status": "PASS" if ok else "FAIL"} for i, what, ok in checks]


def _count_removed(det: pd.DataFrame, rule: str) -> int:
    cols = [f"REMOVED_{i}" for i in range(1, 7) if f"REMOVED_{i}" in det.columns]
    return int(sum(det[c].astype(str).str.startswith(rule).sum() for c in cols))


def _details(run_date: date, runs_root: Path | None, dec: pd.DataFrame, panel: pd.DataFrame, recos: dict,
             slots: pd.DataFrame, anchors: pd.DataFrame, f05: pd.DataFrame) -> pd.DataFrame:
    """One row per panel line: profile, segment, both actions, the action sent, the features, both anchors, the
    recommended offers with their pass / rule, the 6 slots with their reason and what removed them."""
    from nbx.fsdr.segmentation.actions import action_agreement

    train = runs_dir(run_date, runs_root) / "train"
    vals = set(panel[VAL])
    feats = pd.concat([d[d[VAL].isin(vals)] for d in (pd.read_parquet(f) for f in
                       sorted((train / T2 / "base_features").glob("part-*.parquet")))], ignore_index=True)
    feats = feats.drop(columns=[c for c in feats.columns if c in panel.columns and c != VAL]).drop_duplicates(VAL)
    d = panel.merge(dec[[VAL, "ACTION", "DECIDED_BY", "CONSISTENCY_RETRY"]], on=VAL, how="left")
    d = d.rename(columns={"ACTION": "ACTION_SENT"})
    d["RULES_VERSION"] = d["GROUP_KEY"].map(lambda k: config.GROUPS[k]["rules_version"] if k in config.GROUPS else "")
    d["PROPENSITY"] = np.where(d["ACTION_SENT"].notna(), 1.0, np.nan)
    d["ACTION_AGREEMENT"] = np.where(d["ACTION_SENT"].notna(),
                                     action_agreement(d["ACTION_ORIGINAL"], d["ACTION_AGENT"], d["ACTION_SENT"]), None)
    b26 = anchors.reindex(d[VAL]).add_prefix("B26_").reset_index(drop=True)
    f5 = f05.reindex(d[VAL]).reset_index(drop=True)
    d = pd.concat([d.reset_index(drop=True), b26, f5], axis=1)
    for prefix, table in (("m", "monthly"), ("w", "weekly")):
        d = d.merge(_flat_recos(recos[table], prefix), left_on=VAL, right_index=True, how="left")
    s = slots.rename(columns={f"S{i}": f"SLOT_{i}" for i in range(1, 7)})
    s = s[[c for c in s.columns if c.startswith(("SLOT_", "REASON_", "REMOVED_", "IN_INPUT", "DROP_")) or c == VAL]]
    d = d.merge(s, on=VAL, how="left")
    d["IN_INPUT_FLYTXT"] = d["IN_INPUT_FLYTXT"].astype("boolean").fillna(False).astype(bool)
    d["DROP_RULE"] = np.select([d["GROUP_KEY"] == config.CONTROL_GROUP, d["ACTION_SENT"].isna(), ~d["IN_INPUT_FLYTXT"]],
                               ["control (H-01)", "no decision", "F-04 no offer in slots 1-4"], "")
    return d.merge(feats, on=VAL, how="left")


def _final_core(run_date: date, runs_root: Path | None, cache_root: Path | None, d6: Path, keys: list[str]) -> dict:
    """Code 5 on the P6 tables of some groups: input_flytxt, slots, output checks, the P6 audit and the details."""
    from nbx.fsdr.final_output import build_input_flytxt, f05_anchors
    from nbx.fsdr.offer_rules import arpu_anchors, load_catalog

    run = runs_dir(run_date, runs_root)
    cache = CacheSource(run_date, cache_root)
    panel = pd.read_parquet(run / "score" / P1 / "panel.parquet")
    panel = panel[panel["GROUP_KEY"].isin(keys + [config.CONTROL_GROUP])].reset_index(drop=True)
    dec = pd.read_parquet(d6 / "decisions.parquet")
    scored = csvstd.read_csv(d6 / f"results_scoring_nba{csvstd.SUFFIX}")
    groups = [k for k in keys if (panel["GROUP_KEY"] == k).any()]
    recos = {t: pd.concat([_read_reco(d6, config.GROUPS[k][f"recos_{t}"]).assign(GROUP_KEY=k) for k in groups],
                          ignore_index=True) if groups else _read_reco(d6, "_none_") for t in ("monthly", "weekly")}
    flytxt, slots = build_input_flytxt(recos["monthly"], recos["weekly"], scored, cache, run_date)       # F-01
    catalog = load_catalog(cache)
    control = set(panel.loc[panel["GROUP_KEY"] == config.CONTROL_GROUP, VAL])
    checks = output_checks(flytxt, control, set(catalog["BTL_NAME"]), run_date)
    audit6 = json.loads((d6 / "audit_groups.json").read_text(encoding="utf-8"))
    ok = all(c["status"] == "PASS" for c in checks) and not audit6["push_blocked"]
    details = _details(run_date, runs_root, dec, panel, recos, slots, arpu_anchors(cache),
                       f05_anchors(cache, panel[VAL]))
    return {"panel": panel, "groups": groups, "flytxt": flytxt, "checks": checks, "audit6": audit6, "ok": ok,
            "details": details, "group_of": panel.set_index(VAL)["GROUP_KEY"]}


def _group_report(key: str, det: pd.DataFrame, part: pd.DataFrame) -> dict:
    return {"group": key, "rules": config.GROUPS[key]["rules_version"], "lines": len(det),
            "in_input_flytxt": len(part),
            "dropped_F04": int((det["DROP_RULE"] == "F-04 no offer in slots 1-4").sum()),
            "favourites": int(det[["REASON_5", "REASON_6"]].eq("favourite").sum().sum()),
            "forced_F08": int(det["REASON_6"].astype(str).str.startswith("forced").sum()),
            "cleared_F09": _count_removed(det, "F-09"), "cleared_F11": _count_removed(det, "F-11"),
            "cleared_F12": _count_removed(det, "F-12"),
            "offers_per_line": round(float(part[[f"destination_offer_name_{i}" for i in range(1, 7)]].ne("").sum(axis=1)
                                           .mean()), 3) if len(part) else 0.0}


def _blocked_text(what: str, checks: list[dict], audit6: dict) -> str:
    return (f"{what} NOT written (H-08). Failed checks:\n"
            + "\n".join(f"- {c['id']} {c['check']}" for c in checks if c["status"] != "PASS")
            + ("\n- P6 group audit FAIL (audit_groups.json)" if audit6["push_blocked"] else ""))


def run_p7_final(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None,
                 force: bool = False) -> dict:
    """V3.0 final of the four groups together: data/runs/<D>/final/ (push/, target1/, target2/, control/)."""
    run = runs_dir(run_date, runs_root)
    p6 = read_step(run / "score" / P6)
    if p6.get("status") != "done":
        raise StageBlocked(f"{P6} of {run_date} is {p6.get('status', 'missing')}: run fsdr-offers first")
    inputs = fingerprint({"p6": p6.get("started"), "prefix": config.OPPORTUNITY_PREFIX, "forced": config.FORCED_BUNDLES,
                          "reco_offers": config.RECO_OFFERS})
    with Step(run, FINAL, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        c = _final_core(run_date, runs_root, cache_root, run / "score" / P6, list(config.GROUPS))
        flytxt, details, group_of, checks, audit6, ok = (c["flytxt"], c["details"], c["group_of"], c["checks"],
                                                         c["audit6"], c["ok"])
        files = []
        if (st.dir / "push" / "BLOCKED.txt").exists():
            (st.dir / "push" / "BLOCKED.txt").unlink()
        if ok:                                                                    # H-08: push file only on PASS
            files.append(csvstd.write_csv(flytxt, st.output(f"push/input_flytxt{csvstd.SUFFIX}")))
            if config.PUSH_COMPRESSION == "none":
                files.append(csvstd.write_csv(flytxt, st.output("push/input_flytxt.csv"), compress=False))
        else:
            for stale in (st.dir / "push").glob("input_flytxt*"):              # never leave an older file to push
                stale.unlink()
            st.output("push/BLOCKED.txt").write_text(_blocked_text("input_flytxt", checks, audit6), encoding="utf-8")
        per_target = []
        for key in c["groups"] + [config.CONTROL_GROUP]:
            folder = TARGET_DIRS[key]
            det = details[details["GROUP_KEY"] == key]
            files.append(csvstd.write_csv(det, st.output(f"{folder}/details_{key}{csvstd.SUFFIX}")))
            if key == config.CONTROL_GROUP:
                files.append(csvstd.write_csv(det[[VAL, "GROUP_KEY", "STRATUM", "SEGMENT", "MICRO_SEGMENT"]],
                                              st.output(f"control/control_{run_date.isoformat()}{csvstd.SUFFIX}")))
                per_target.append({"group": key, "rules": "never contacted", "lines": len(det), "in_input_flytxt": 0})
                continue
            part = flytxt[flytxt["msisdn"].map(group_of) == key]
            files.append(csvstd.write_csv(part, st.output(f"{folder}/input_flytxt_{key}{csvstd.SUFFIX}")))
            per_target.append(_group_report(key, det, part))
        for tgt, keys in (("target1", ["t1c", "t1f"]), ("target2", ["t2c", "t2f"])):
            part = flytxt[flytxt["msisdn"].map(group_of).isin(keys)]
            files.append(csvstd.write_csv(part, st.output(f"{tgt}/input_flytxt_{tgt}{csvstd.SUFFIX}")))
        st.write_table(details, "details_all", csv="none")
        st.write_csv(pd.DataFrame(per_target), "final_rules_report.csv")
        st.output(csvstd.OUTPUTS_MANIFEST)
        csvstd.write_outputs_manifest(st.dir, files, run_id=f"final_{run_date.isoformat()}",
                                      versions={k: g["rules_version"] for k, g in config.GROUPS.items()})
        summary = {"run_date": run_date.isoformat(), "stage": FINAL, "push_ready": ok,
                   "input_flytxt_rows": len(flytxt), "output_checks": checks, "p6_group_audit": audit6["status"],
                   "per_target": per_target, "details_rows": len(details), "details_columns": int(details.shape[1]),
                   "reco_offers": config.RECO_OFFERS, "step_dir": str(st.dir)}
        if not ok:
            st.block("push blocked: see push/BLOCKED.txt")
        st.rows(rows_in=len(c["panel"]), rows_out=len(flytxt))
        st.write_json("summary.json", summary)
        return summary


# ---------------------------------------------------------------------------
# V4.0: each target scored separately into data/output/<D>/<target>/, then one push file
# ---------------------------------------------------------------------------
PUSH, CONTROL_DIR = "push", "control"


def run_target_final(run_date: date, target: str, runs_root: Path | None = None, cache_root: Path | None = None,
                     out_root: Path | None = None, force: bool = False) -> dict:
    """Code 5 on ONE target -> data/output/<D>/<target>/:

        input_flytxt_<target>.csv.gz        the target's file to push (only on PASS, else BLOCKED.txt)
        a_constraint/input_flytxt_<t?c>.csv.gz   details_<t?c>.csv.gz     group a (constraint universe)
        b_free/input_flytxt_<t?f>.csv.gz         details_<t?f>.csv.gz     group b (free universe)
        details_<target>.parquet            every line of the target with every column (the big file to validate)
        final_rules_report.csv  output_checks.json  audit_groups.json  summary.json  _outputs_manifest.json
    """
    keys = target_keys(target)
    run = runs_dir(run_date, runs_root)
    d6 = run / "score" / p6_name(target)
    p6 = read_step(d6)
    if p6.get("status") != "done":
        raise StageBlocked(f"{p6_name(target)} of {run_date} is {p6.get('status', 'missing')}: "
                           f"run fsdr-score-target --target {target} first")
    inputs = fingerprint({"p6": p6.get("started"), "prefix": config.OPPORTUNITY_PREFIX, "forced": config.FORCED_BUNDLES,
                          "reco_offers": config.RECO_OFFERS})
    with Step(config.output_folder(run_date.isoformat(), out_root), target, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        c = _final_core(run_date, runs_root, cache_root, d6, keys)
        flytxt, details, group_of = c["flytxt"], c["details"], c["group_of"]
        for stale in [*st.dir.glob(f"input_flytxt_{target}*"), st.dir / "BLOCKED.txt"]:   # never an older file to push
            if stale.exists():
                stale.unlink()
        files = []
        if c["ok"]:
            files.append(csvstd.write_csv(flytxt, st.output(f"input_flytxt_{target}{csvstd.SUFFIX}")))
        else:
            st.output("BLOCKED.txt").write_text(_blocked_text(f"input_flytxt_{target}", c["checks"], c["audit6"]),
                                                encoding="utf-8")
        per_group = []
        for key in c["groups"]:
            sub = config.GROUP_SUBFOLDER[key]
            det = details[details["GROUP_KEY"] == key]
            part = flytxt[flytxt["msisdn"].map(group_of) == key]
            files.append(csvstd.write_csv(part, st.output(f"{sub}/input_flytxt_{key}{csvstd.SUFFIX}")))
            files.append(csvstd.write_csv(det, st.output(f"{sub}/details_{key}{csvstd.SUFFIX}")))
            per_group.append({**_group_report(key, det, part), "folder": sub})
        st.write_table(details[details["GROUP_KEY"] != config.CONTROL_GROUP], f"details_{target}", csv="none")
        st.write_csv(pd.DataFrame(per_group), "final_rules_report.csv")
        st.write_json("output_checks.json", c["checks"])
        st.write_json("audit_groups.json", c["audit6"])
        st.output(csvstd.OUTPUTS_MANIFEST)
        csvstd.write_outputs_manifest(st.dir, files, run_id=f"final_{run_date.isoformat()}_{target}",
                                      versions={k: config.GROUPS[k]["rules_version"] for k in keys})
        summary = {"run_date": run_date.isoformat(), "stage": f"final_{target}", "target": target, "groups": keys,
                   "push_ready": c["ok"], "input_flytxt_rows": len(flytxt), "output_checks": c["checks"],
                   "p6_group_audit": c["audit6"]["status"], "per_group": per_group, "reco_offers": config.RECO_OFFERS,
                   "details_rows": int((details["GROUP_KEY"] != config.CONTROL_GROUP).sum()),
                   "details_columns": int(details.shape[1]), "step_dir": str(st.dir)}
        if not c["ok"]:
            st.block(f"{target} blocked: see {target}/BLOCKED.txt")
        st.rows(rows_in=int((c["panel"]["GROUP_KEY"] != config.CONTROL_GROUP).sum()), rows_out=len(flytxt))
        st.write_json("summary.json", summary)
        return summary


def write_control(run_date: date, runs_root: Path | None = None, out_root: Path | None = None) -> dict:
    """data/output/<D>/control/control_<D>.csv.gz: the control lines (never contacted, H-01) and their strata."""
    panel = pd.read_parquet(runs_dir(run_date, runs_root) / "score" / P1 / "panel.parquet")
    ctrl = panel[panel["GROUP_KEY"] == config.CONTROL_GROUP]
    cols = [VAL, "GROUP_KEY", "STRATUM", "SEGMENT", "MICRO_SEGMENT", "NATIONALITY_GROUP", "VALUE_BAND", "USAGE_TYPE"]
    cols += [c for c in ctrl.columns if c.startswith("T_")]
    path = config.output_folder(run_date.isoformat(), out_root) / CONTROL_DIR / f"control_{run_date.isoformat()}{csvstd.SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    csvstd.write_csv(ctrl[[c for c in cols if c in ctrl.columns]], path)
    return {"file": str(path), "lines": len(ctrl)}


def run_push(run_date: date, out_root: Path | None = None, force: bool = True) -> dict:
    """data/output/<D>/push/input_flytxt.csv.gz: the target files merged, only when EVERY target passed (H-08) and no
    MSISDN is in two targets (A-30); else push/BLOCKED.txt and no file."""
    out = config.output_folder(run_date.isoformat(), out_root)
    states = {t: read_step(out / t) for t in config.TARGETS}
    inputs = fingerprint({t: s.get("started") for t, s in states.items()})
    with Step(out, PUSH, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        for stale in [*st.dir.glob("input_flytxt*"), st.dir / "BLOCKED.txt"]:
            if stale.exists():
                stale.unlink()
        reasons, parts, rows = [], [], {}
        for t, s in states.items():
            f = out / t / f"input_flytxt_{t}{csvstd.SUFFIX}"
            if s.get("status") != "done" or not f.exists():
                reasons.append(f"{t}: {s.get('status', 'not scored')} (see {t}/BLOCKED.txt or run it)")
                continue
            parts.append(csvstd.read_csv(f))
            rows[t] = len(parts[-1])
        flytxt = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=config.INPUT_FLYTXT_COLUMNS)
        twice = int(flytxt["msisdn"].duplicated().sum()) if len(flytxt) else 0
        if twice:
            reasons.append(f"A-30: {twice} MSISDN in two targets")
        ok = not reasons
        files = []
        if ok:
            flytxt = flytxt[config.INPUT_FLYTXT_COLUMNS]
            files.append(csvstd.write_csv(flytxt, st.output(f"input_flytxt{csvstd.SUFFIX}")))
            if config.PUSH_COMPRESSION == "none":
                files.append(csvstd.write_csv(flytxt, st.output("input_flytxt.csv"), compress=False))
            st.output(csvstd.OUTPUTS_MANIFEST)
            csvstd.write_outputs_manifest(st.dir, files, run_id=f"push_{run_date.isoformat()}",
                                          versions={k: g["rules_version"] for k, g in config.GROUPS.items()})
        else:
            st.output("BLOCKED.txt").write_text("input_flytxt NOT written (H-08):\n" + "\n".join(f"- {r}" for r in reasons),
                                                encoding="utf-8")
            st.block("; ".join(reasons))
        summary = {"run_date": run_date.isoformat(), "stage": PUSH, "push_ready": ok, "rows_per_target": rows,
                   "input_flytxt_rows": len(flytxt) if ok else 0, "reasons": reasons, "step_dir": str(st.dir)}
        st.rows(rows_out=summary["input_flytxt_rows"])
        st.write_json("summary.json", summary)
        return summary


def run_targets(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None,
                out_root: Path | None = None, targets: list[str] | None = None, force: bool = True) -> dict:
    """V4.0 scoring in one go: P1 panel, then each target on its own (P6 + final), the control file, the push."""
    p1 = run_p1_panel(run_date, runs_root, cache_root, force=force)
    res = {"P1_panel": p1}
    if p1["status"] != "done":
        return res
    for t in targets or list(config.TARGETS):
        res[f"P6_{t}"] = run_p6_offers(run_date, runs_root, cache_root, force=force, target=t)
        res[t] = run_target_final(run_date, t, runs_root, cache_root, out_root, force=force)
    res["control"] = write_control(run_date, runs_root, out_root)
    res["push"] = run_push(run_date, out_root)
    return res
