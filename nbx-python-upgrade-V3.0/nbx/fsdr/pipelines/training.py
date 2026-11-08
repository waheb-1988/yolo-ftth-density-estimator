"""V2.0 training pipeline, stages T2 and T3 (plan §14, §15; master prompt §6 - §8, §18).

    run_t2(D)        T2_features   base_features/part-NN.parquet (B-01 … B-12) for the full base
    run_t3(D)        T3_segments   segments/part-NN.parquet: S-rule segment, bands, micro-segment, week-0 history,
                                   action_original + crosswalk, action_agent (agent dictionary), and the reports
    run_features(D)  T2 then T3 (fsdr-features)

Both stages read T0's Parquet cache and run partition by partition, `workers` partitions in parallel
(FSDR_FEATURE_WORKERS: 2 on the dev laptop, 8 on the P16). Each worker writes its own partition file and returns
small count tables, which the parent sums: nothing of size 1M is ever gathered in one process except the few
columns T2 needs for the B-05 medians and the ARPU tertiles.

Resume: a stage whose _step.json is done with the same inputs (the cache manifest, the previous stage, the rule
constants) is skipped; force=True re-runs it.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from nbx.fsdr import base_features as bf
from nbx.fsdr import config
from nbx.fsdr.contract import data_agent
from nbx.fsdr.contract.cache import CACHE_MANIFEST, CacheSource, cache_dir
from nbx.fsdr.offer_rules import load_catalog
from nbx.fsdr.segmentation import actions, bands, rules
from nbx.fsdr.steps import Step, fingerprint, read_step, runs_dir, sha256_file

T2, T3 = "T2_features", "T3_segments"
SEGMENT_COLUMNS = ["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "RULE_ID", "SEGMENT", "RULE_ACTION", "EXCLUDED",
                   "SEGMENT_SOURCE", *rules.TRIGGERS, "NATIONALITY_GROUP", "AGE", "AGE_BAND", "YOUNG_SPLIT",
                   "TENURE_DAYS", "TENURE_BAND", "MICRO_SEGMENT", "ARPU_M1", "VALUE_BAND", "USAGE_TYPE", "REGION",
                   "DV_SEGMENT", "W0_SEGMENT", "W0_DATE", "W0_TARGETED", "W0_ACTION", "W0_REWARD", "W0_BTL_REVENUE",
                   "ORIGINAL_SEGMENT", "ACTION_ORIGINAL", "ACTION_ORIGINAL_SOURCE", "ACTION_AGENT",
                   "ACTION_AGENT_RANK2", "ACTION_AGENT_VERSION"]


class StageBlocked(RuntimeError):
    """A stage cannot start (its previous stage is missing or blocked)."""


def _iso(d: date | str) -> str:
    return d.isoformat() if isinstance(d, date) else str(d)


def _run(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return list(ex.map(fn, jobs))


def rules_version() -> str:
    """Hash of every constant the S-rules, bands and dictionaries depend on (a change re-runs T3)."""
    keys = [k for k in dir(config) if k.startswith(("S_", "AGE_", "TENURE_", "YOUNG_", "VALUE_", "USAGE_", "NATIONALITY_"))]
    keys += ["DICT_SEGM_ACTIONS_SOURCE", "SEGMENT_CROSSWALK", "ACTIONS"]
    blob = json.dumps({k: getattr(config, k) for k in sorted(keys)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _t0(run_date: date, runs_root: Path | None) -> dict:
    t0 = data_agent.load_summary(run_date, runs_root)
    if not t0:
        raise StageBlocked(f"no T0_load for {run_date}: run fsdr-load first")
    if t0.get("status") != "done":
        raise StageBlocked(f"T0_load of {run_date} is {t0.get('status')}: fix the data, re-run fsdr-load")
    return t0


def _sum_counts(frames: list[pd.DataFrame], by: list[str], value: list[str] | None = None) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=by + (value or ["lines"]))
    return pd.concat(frames, ignore_index=True).groupby(by, dropna=False, as_index=False)[value or ["lines"]].sum()


# ---------------------------------------------------------------------------
# T2 features
# ---------------------------------------------------------------------------
def _t2_pass1(job: tuple) -> dict:
    cache_root, run_date, part, anchor, out = job
    cache = CacheSource(run_date, Path(cache_root))
    df, stats = bf.partition_features(cache, part, date.fromisoformat(run_date), anchor)
    df.to_parquet(out, index=False)
    return stats


def _t2_pass2(job: tuple) -> pd.DataFrame:
    path, medians = job
    df = pd.read_parquet(path)
    bf.impute(df, medians)
    df.to_parquet(path, index=False)
    return bf.column_stats(df)


def run_t2(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None, workers: int | None = None,
           force: bool = False) -> dict:
    t0 = _t0(run_date, runs_root)
    anchor = bf.anchor_of(t0["pipeline"])
    cache = CacheSource(run_date, cache_root)
    workers = int(workers or config.FEATURE_WORKERS)
    inputs = fingerprint({"cache": sha256_file(cache_dir(run_date, cache_root) / CACHE_MANIFEST), "anchor": anchor,
                          "registry": [m.table for m in config.MODEL_SCORE_REGISTRY], "trend": config.TREND_PAIRS})
    with Step(runs_dir(run_date, runs_root) / "train", T2, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        parts = list(range(cache.partitions))
        outs = [st.output(f"base_features/part-{k:02d}.parquet") for k in parts]
        stats = _run(_t2_pass1, [(str(cache_dir(run_date, cache_root).parent), _iso(run_date), k, anchor, str(outs[k]))
                                 for k in parts], workers)
        medians = bf.score_medians(outs)                                       # B-05, whole base
        cutoffs = bf.arpu_cutoffs(outs)
        col_stats = _run(_t2_pass2, [(str(o), medians) for o in outs], workers)

        cs = pd.concat(col_stats).groupby(level=0).agg({"count": "sum", "nulls": "sum", "sum": "sum", "min": "min", "max": "max"})
        cs["mean"] = (cs["sum"] / cs["count"].where(cs["count"] > 0)).round(6)
        st.write_csv(cs.drop(columns=["sum"]).reset_index(names="feature"), "feature_summary.csv")
        st.write_csv(pd.read_parquet(outs[0]).head(config.STEP_PREVIEW_ROWS), "base_features_preview.csv")
        vals = pd.concat([pd.read_parquet(o, columns=["ACCESS_METHOD_VAL"]) for o in outs])["ACCESS_METHOD_VAL"]
        lines = int(sum(s["lines"] for s in stats))
        missing = {f: int(sum(s["missing_scores"][f] for s in stats)) for f in config.proba_features()}
        src_counts = {k: _merge_counts([s[k] for s in stats]) for k in ("age_source", "tenure_source")}
        stats_doc = {"medians_by_nationality": medians, "arpu_tertiles": cutoffs, "anchor": anchor,
                     "party_table": bf.party_table(anchor, cache)}
        st.write_json("stats.json", stats_doc)
        summary = {"run_date": _iso(run_date), "stage": T2, "pipeline": t0["pipeline"], "anchor": anchor,
                   "tables": "*_auto (week 0)" if anchor == "T" else "*_score (week n)",
                   "base_rows": int(sum(s["base_rows"] for s in stats)), "lines": lines,
                   "eligible_filters": {k: int(sum(s[k] for s in stats)) for k in
                                        ("null_keys", "excluded_plan", "bad_msisdn", "duplicates_removed")},
                   "msisdn_in_two_partitions": int(vals.duplicated().sum()),
                   "features": int(len(cs)), "missing_scores": missing,
                   "missing_share": {f: round(n / lines, 4) if lines else 0.0 for f, n in missing.items()},
                   **src_counts, "no_nationality": int(sum(s["no_nationality"] for s in stats)),
                   "arpu_tertiles": cutoffs, "partitions": len(parts), "workers": workers, "step_dir": str(st.dir)}
        st.rows(rows_in=summary["base_rows"], rows_out=lines)
        st.note(anchor=anchor, workers=workers)
        st.write_json("summary.json", summary)
        return summary


def _merge_counts(dicts: list[dict]) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


# ---------------------------------------------------------------------------
# T3 segments and the two actions
# ---------------------------------------------------------------------------
def segment_partition(feat: pd.DataFrame, cache, part: int, run_date: date, cutoffs: list[float],
                      agent_dict: dict, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Every T3 column of one partition (pure function of the features, the cache partition and the dictionary)."""
    seg = pd.concat([feat[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "NATIONALITY_GROUP", "AGE", "TENURE_DAYS",
                           "ARPU_M1", "REGION", "DV_SEGMENT"]], rules.apply_s_rules(feat)], axis=1)
    seg["AGE_BAND"] = bands.age_band(seg["AGE"])
    seg["YOUNG_SPLIT"] = bands.young_split(seg["AGE"])
    seg["TENURE_BAND"] = bands.tenure_band(seg["TENURE_DAYS"])
    seg["MICRO_SEGMENT"] = bands.micro_segment(seg["NATIONALITY_GROUP"], seg["AGE_BAND"], seg["TENURE_BAND"])
    seg["VALUE_BAND"] = bands.value_band(seg["ARPU_M1"], cutoffs)
    seg["USAGE_TYPE"] = bands.usage_type(seg["S_VOICE_USER"], seg["S_DATA_USER"])

    tg = bf._part(cache, "targeting_dataiku_auto", part)
    ful = bf._part(cache, "subs_btl_auto", part)
    w0 = actions.week0_targeting(tg, ful, catalog, run_date, seg)
    seg = seg.merge(w0.drop(columns=["ACCESS_METHOD_VAL"], errors="ignore"), on="ACCESS_METHOD_ID", how="left")
    seg["W0_TARGETED"] = seg["W0_DATE"].notna().astype("int8")
    seg["W0_REWARD"] = seg["W0_REWARD"].fillna(0).astype("int8")
    seg["W0_BTL_REVENUE"] = seg["W0_BTL_REVENUE"].fillna(0.0)
    seg = actions.apply_original(seg)
    seg = actions.apply_agent_dictionary(seg, agent_dict)
    return seg[SEGMENT_COLUMNS]


def _counts(seg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def g(by, **agg):
        d = seg.assign(lines=1, **{k: seg[v] for k, v in agg.items()})
        return d.groupby(by, dropna=False, as_index=False)[["lines", *agg]].sum()
    s = seg.assign(W0_SEGMENT=seg["W0_SEGMENT"].fillna("(not targeted)"),
                   ACTION_ORIGINAL=seg["ACTION_ORIGINAL"].fillna("(none)"), ACTION_AGENT=seg["ACTION_AGENT"].fillna("(none)"),
                   RULE_ACTION=seg["RULE_ACTION"].fillna("excluded"), W0_ACTION=seg["W0_ACTION"].fillna("(none)"),
                   YOUNG_SPLIT=seg["YOUNG_SPLIT"].fillna("-"), FAMILY=rules.segment_family(seg["SEGMENT"]))
    seg = s
    return {
        "segments": g(["RULE_ID", "SEGMENT", "RULE_ACTION"], arpu_sum="ARPU_M1", w0_targeted="W0_TARGETED", w0_takers="W0_REWARD"),
        "micro": g(["NATIONALITY_GROUP", "AGE_BAND", "TENURE_BAND"]),
        "agreement": g(["SEGMENT", "W0_SEGMENT"]),
        "tenure": g(["FAMILY", "TENURE_BAND"]),
        "actions": g(["ACTION_ORIGINAL_SOURCE", "ACTION_ORIGINAL", "ACTION_AGENT"]),
        "context": g(["VALUE_BAND", "USAGE_TYPE"]),
        "young": g(["NATIONALITY_GROUP", "YOUNG_SPLIT"], takers="W0_REWARD", targeted="W0_TARGETED"),
        "w0": g(["W0_ACTION"], targeted="W0_TARGETED", takers="W0_REWARD", btl_revenue="W0_BTL_REVENUE"),
    }


def _t3_part(job: tuple) -> dict:
    feat_path, cache_root, run_date, part, cutoffs, agent_dict, out = job
    cache = CacheSource(run_date, Path(cache_root))
    catalog = load_catalog(cache) if cache.has("btl_catalog") else None
    seg = segment_partition(pd.read_parquet(feat_path), cache, part, date.fromisoformat(run_date), cutoffs, agent_dict, catalog)
    seg.to_parquet(out, index=False)
    return _counts(seg)


def run_t3(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None, workers: int | None = None,
           force: bool = False) -> dict:
    _t0(run_date, runs_root)
    train = runs_dir(run_date, runs_root) / "train"
    t2 = read_step(train / T2)
    if t2.get("status") != "done":
        raise StageBlocked(f"T2_features of {run_date} is {t2.get('status', 'missing')}: run fsdr-features")
    workers = int(workers or config.FEATURE_WORKERS)
    root = Path(runs_root or config.FSDR_RUNS_DIR)
    prev_dict, prev_path = actions.previous_agent_dictionary(root, run_date)
    inputs = fingerprint({"t2": t2.get("inputs"), "t2_started": t2.get("started"), "rules": rules_version(),
                          "previous_dictionary": str(prev_path or "")})
    with Step(train, T3, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        cache = CacheSource(run_date, cache_root)
        cutoffs = json.loads((train / T2 / "stats.json").read_text(encoding="utf-8"))["arpu_tertiles"]
        agent_dict, problems = actions.validate_agent_dictionary(prev_dict or actions.initial_agent_dictionary(run_date), prev_dict)
        st.write_json(f"{agent_dict['name']}.json", agent_dict)
        st.write_json("agent_dictionary_check.json", {"version": agent_dict["name"], "source": str(prev_path or "v0 (week 0)"),
                                                      "problems": problems})
        st.write_json("dict_segm_actions_original.json", {"source_order": config.DICT_SEGM_ACTIONS_SOURCE,
                                                          "effective (last value wins)": config.DICT_SEGM_ACTIONS})
        crosswalk = actions.crosswalk_table()
        st.write_csv(crosswalk, "crosswalk.csv")

        parts = list(range(cache.partitions))
        outs = [st.output(f"segments/part-{k:02d}.parquet") for k in parts]
        jobs = [(str(train / T2 / "base_features" / f"part-{k:02d}.parquet"), str(cache_dir(run_date, cache_root).parent),
                 _iso(run_date), k, cutoffs, agent_dict, str(outs[k])) for k in parts]
        counts = _run(_t3_part, jobs, workers)
        summary = _t3_reports(st, counts, run_date, crosswalk, agent_dict)
        st.write_csv(pd.read_parquet(outs[0]).head(config.STEP_PREVIEW_ROWS), "segments_preview.csv")
        summary.update({"run_date": _iso(run_date), "stage": T3, "rules_version": rules_version(),
                        "agent_dictionary": agent_dict["name"], "agent_dictionary_problems": len(problems),
                        "partitions": len(parts), "workers": workers, "step_dir": str(st.dir)})
        st.rows(rows_in=summary["lines"], rows_out=summary["lines"])
        st.note(rules_version=rules_version(), agent_dictionary=agent_dict["name"])
        st.write_json("summary.json", summary)
        return summary


def _t3_reports(st: Step, counts: list[dict], run_date: date, crosswalk: pd.DataFrame, agent_dict: dict) -> dict:
    get = lambda k: [c[k] for c in counts]                                        # noqa: E731
    seg = _sum_counts(get("segments"), ["RULE_ID", "SEGMENT", "RULE_ACTION"], ["lines", "arpu_sum", "w0_targeted", "w0_takers"])
    order = {r: i for i, (r, _, _) in enumerate(config.S_RULES)}
    seg = seg.sort_values("RULE_ID", key=lambda s: s.map(order)).reset_index(drop=True)
    lines = int(seg["lines"].sum())
    seg["share"] = (seg["lines"] / max(lines, 1)).round(4)
    seg["avg_arpu_m1"] = (seg["arpu_sum"] / seg["lines"].where(seg["lines"] > 0)).round(3)
    st.write_csv(seg.drop(columns=["arpu_sum"]), "segment_counts.csv")

    micro = _sum_counts(get("micro"), ["NATIONALITY_GROUP", "AGE_BAND", "TENURE_BAND"])
    grid = pd.DataFrame(bands.all_micro_segments(), columns=["NATIONALITY_GROUP", "AGE_BAND", "TENURE_BAND"])
    micro = grid.merge(micro, how="left").fillna({"lines": 0})
    micro["lines"] = micro["lines"].astype(int)
    micro["MICRO_SEGMENT"] = micro["NATIONALITY_GROUP"] + "|" + micro["AGE_BAND"] + "|" + micro["TENURE_BAND"]
    st.write_csv(micro[["MICRO_SEGMENT", "NATIONALITY_GROUP", "AGE_BAND", "TENURE_BAND", "lines"]], "microsegment_counts.csv")

    agree = _sum_counts(get("agreement"), ["SEGMENT", "W0_SEGMENT"])
    cw = {s: config.SEGMENT_CROSSWALK[s][0].lower() for s in config.S_SEGMENTS}
    targeted = agree[agree["W0_SEGMENT"] != "(not targeted)"]
    match = targeted["SEGMENT"].map(cw) == targeted["W0_SEGMENT"].astype(str).str.lower()
    agree["match"] = (agree["SEGMENT"].map(cw) == agree["W0_SEGMENT"].astype(str).str.lower())
    st.write_csv(agree.sort_values(["SEGMENT", "lines"], ascending=[True, False]), "segment_agreement.csv")

    ten = _sum_counts(get("tenure"), ["FAMILY", "TENURE_BAND"])
    st.write_csv(ten.pivot_table(index="FAMILY", columns="TENURE_BAND", values="lines", aggfunc="sum", fill_value=0)
                   .astype(int).reset_index(), "tenure_agreement.csv")
    acts = _sum_counts(get("actions"), ["ACTION_ORIGINAL_SOURCE", "ACTION_ORIGINAL", "ACTION_AGENT"])
    st.write_csv(acts, "action_counts.csv")
    st.write_csv(_sum_counts(get("context"), ["VALUE_BAND", "USAGE_TYPE"]), "value_usage.csv")
    young = _sum_counts(get("young"), ["NATIONALITY_GROUP", "YOUNG_SPLIT"], ["lines", "takers", "targeted"])
    st.write_csv(young[young["YOUNG_SPLIT"] != "-"], "young_split.csv")

    w0 = _sum_counts(get("w0"), ["W0_ACTION"], ["lines", "targeted", "takers", "btl_revenue"])
    acted = w0[w0["W0_ACTION"].isin(config.ACTIONS)]
    n = int(acted["targeted"].sum())
    recomputed = {a: round(float(acted.loc[acted["W0_ACTION"] == a, "targeted"].sum()) / n, 7) if n else None
                  for a in config.ACTIONS}
    take = {a: (round(float(r["takers"]) / r["targeted"], 4) if r["targeted"] else None)
            for a, r in acted.set_index("W0_ACTION").iterrows()}
    priors = {"rule": "B-10", "configured": config.ACTION_PRIOR, "recomputed_week0": recomputed,
              "targeted_with_action": n, "take_rate_week0": take, "order": config.ACTIONS,
              "used": "recomputed_week0" if n else "configured"}
    st.write_json("priors.json", priors)

    excluded = int(seg.loc[seg["RULE_ACTION"] == "excluded", "lines"].sum())
    src = acts.groupby("ACTION_ORIGINAL_SOURCE")["lines"].sum().to_dict()
    same = acts[acts["ACTION_ORIGINAL"] == acts["ACTION_AGENT"]]["lines"].sum()
    non_empty = int((micro["lines"] > 0).sum())
    return {"lines": lines, "excluded_nursering": excluded, "targetable": lines - excluded,
            "segments_non_empty": int((seg["lines"] > 0).sum()),
            "segment_lines": dict(zip(seg["SEGMENT"], seg["lines"].astype(int))),
            "micro_segments_non_empty": non_empty, "micro_segments_total": len(micro),
            "micro_segments_ge_300": int((micro["lines"] >= 300).sum()),
            "engine_targeted_lines": int(targeted["lines"].sum()),
            "engine_vs_s_rules_agreement": round(float(targeted.loc[match, "lines"].sum() / max(targeted["lines"].sum(), 1)), 4),
            "action_original_source": {k: int(v) for k, v in src.items()},
            "action_original_eq_agent": round(float(same / max(lines, 1)), 4),
            "crosswalk_same_action": int(crosswalk["same_action"].sum()), "crosswalk_lines": len(crosswalk),
            "priors_used": priors["used"], "week0_targeted_with_action": n}


def run_features(run_date: date, runs_root: Path | None = None, cache_root: Path | None = None,
                 workers: int | None = None, force: bool = False) -> dict:
    """fsdr-features: T2 then T3 (force re-runs both)."""
    t2 = run_t2(run_date, runs_root, cache_root, workers, force)
    t3 = run_t3(run_date, runs_root, cache_root, workers, force)
    return {"T2_features": t2, "T3_segments": t3}
