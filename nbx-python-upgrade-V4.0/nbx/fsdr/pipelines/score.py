"""Scoring pipeline -- port of docs/fsdr_reference/02_nba_score.py on the JSON data contract.

    1 load_model          champion from the registry; refuses an incompatible or stale model
    2 population          'target': the test sample, arms free / cstr / llm (cg = control, not scored);
                          'base': every eligible line
    3 build_features      D-anchored tables (*_score, party_scoring) -- same function as training
    4 score               policy 'sample' (production) or 'greedy'; seed = run date by default
    5 write               results_scoring_nba + TG_free / TG_constraint / TG_llm _scored_nba
    6 drift               PSI per feature vs the training data, action-mix shift
"""
from __future__ import annotations

from nbx.core import tracing

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import bandit, config, registry
from nbx.fsdr.datasource import DataSource
from nbx.fsdr import campaign_log
from nbx.fsdr.features import build_feature_frame, eligible_base, model_features, norm_id, norm_val
from nbx.fsdr.pipelines import write_report, write_table
from nbx.fsdr.pipelines.train import load_and_validate

ARM_OUTPUTS = {"free": "TG_free_scored_nba", "with_constraint": "TG_constraint_scored_nba", "llm": "TG_llm_scored_nba"}


class StaleModel(RuntimeError):
    pass


def target_population(src: DataSource) -> tuple[pd.DataFrame, dict]:
    s = src.read(config.TARGET_SAMPLE_TABLE)
    group = s["group"].astype(str).str.strip().str.lower()
    pop = pd.DataFrame({"ACCESS_METHOD_VAL": norm_val(s["access_method_val"]), "ACCESS_METHOD_ID": norm_id(s["access_method_id"]),
                        "ACCOUNT_ID": norm_id(s["account_id"]), "GROUP": group, "TG_FLAG": group.map(config.GROUP_TO_FLAG)})
    counts = group.value_counts().to_dict()
    pop = pop.dropna(subset=["TG_FLAG"])                                  # cg (control) is not scored
    acct = src.read("consumer_aact_account")
    acct = pd.DataFrame({"ACCOUNT_ID": norm_id(acct["ACCOUNT_ID"]), "PARTY_ID": norm_id(acct["PARTY_ID"])})
    pop = pop.merge(acct.dropna().drop_duplicates("ACCOUNT_ID"), on="ACCOUNT_ID", how="left")
    return pop, {"sample_groups": counts, "scored_lines": len(pop)}


def _psi(expected: list[float], actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected), 1e-4, None)
    a = np.clip(actual, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(feat: pd.DataFrame, meta: dict, scored: pd.DataFrame) -> dict:
    psi = {}
    for f, b in (meta.get("psi_bins") or {}).items():
        if f not in feat.columns:
            continue
        v = pd.to_numeric(feat[f], errors="coerce").dropna().to_numpy()
        if not len(v):
            continue
        counts = np.bincount(np.searchsorted(np.asarray(b["edges"]), v, side="right"), minlength=len(b["p"]))
        psi[f] = round(_psi(b["p"], counts / counts.sum()), 4)
    alerts = {f: v for f, v in psi.items() if v > config.PSI_ALERT}
    train_mix = meta.get("holdout_action_distribution") or {}
    now_mix = scored["ACTION_GREEDY"].value_counts(normalize=True).to_dict() if len(scored) else {}
    shift = round(0.5 * sum(abs(now_mix.get(a, 0) - train_mix.get(a, 0)) for a in config.ACTIONS), 4)
    return {"psi": psi, "psi_alerts": alerts, "max_psi": max(psi.values()) if psi else 0.0,
            "action_mix_shift_tv": shift, "action_mix_now": {k: round(v, 4) for k, v in now_mix.items()}}


@tracing.traced_pipeline("fsdr_score")
def run_scoring(data_dir: str | Path, run_date: date | None = None, population: str = "target",
                policy: str = config.SCORING_POLICY, seed: int | None = None, version: str | None = None,
                root: Path | None = None, allow_stale: bool = False) -> dict:
    t0 = datetime.now()
    src, run_date, val_rep = load_and_validate(data_dir, run_date)
    features = model_features()
    vw, meta = registry.load(version, root, features)                     # refuses incompatible models
    age = (run_date - date.fromisoformat(meta["run_date"])).days
    if age > config.MAX_MODEL_AGE_DAYS and not allow_stale:
        raise StaleModel(f"champion {meta['version']} is {age} days old (max {config.MAX_MODEL_AGE_DAYS}): retrain first")

    if population == "target":
        pop, pop_rep = target_population(src)
    elif population == "base":
        base, pop_rep = eligible_base(src)
        pop = base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]].assign(TG_FLAG="base")
    else:
        raise ValueError("population must be 'target' or 'base'")
    seed = seed if seed is not None else run_date.toordinal()
    feat, feat_rep = build_feature_frame(src, pop, "D", seed, log_root=campaign_log.root_for(data_dir), context_date=run_date)
    res = bandit.score_population(vw, feat, features, policy=policy, id_cols=("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "TG_FLAG"),
                                  seed=seed)
    res["SCORE_DATE"] = run_date.isoformat()
    res["MODEL_VERSION"] = meta["version"]
    res["MODEL_TRAINED_AT"] = meta["trained_at"]
    write_table(res, data_dir, "results_scoring_nba")
    if population == "target":
        for flag, name in ARM_OUTPUTS.items():
            write_table(res[res["TG_FLAG"] == flag], data_dir, name)
        # M1: the control group is logged as not contacted, so feedback and the report can compare the arms with it
        s = src.read(config.TARGET_SAMPLE_TABLE)
        cg = norm_val(s.loc[s["group"].astype(str).str.strip().str.lower() == "cg", "access_method_val"]).dropna()
        campaign_log.append_log(campaign_log.root_for(data_dir), campaign_log.rows_control(cg.tolist(), run_date))

    mix = pd.crosstab(res["ACTION"], res["TG_FLAG"], normalize="columns").round(4) if len(res) else pd.DataFrame()
    report = {
        "pipeline": "score", "model_version": meta["version"], "model_age_days": age, "population": population, "policy": policy,
        "seed": seed, "validation": val_rep, "population_report": pop_rep, "features": feat_rep,
        "action_distribution_per_arm": {c: mix[c].to_dict() for c in mix.columns},
        "drift": drift_report(feat, meta, res), "seconds": round((datetime.now() - t0).total_seconds(), 1),
    }
    write_report(report, data_dir, "nba_scoring_report")
    return report
