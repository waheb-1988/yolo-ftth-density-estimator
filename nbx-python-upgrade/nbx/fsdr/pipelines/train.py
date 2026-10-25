"""Training pipeline -- port of docs/fsdr_reference/01_nba_train.py on the JSON data contract.

    1 load_and_validate   read the run folder, refuse blocked data
    2 build_features      T-anchored tables (*_auto, party_auto), whole eligible base
    3 build_labels        targeting D-8..D-2 -> action, reward D-7..D-1, propensity
    4 train_bandit        VW cb_explore_adf, logged (default) or simulation mode
    5 evaluate            20 % holdout: logged take rate, IPS / SNIPS of the model policy,
                          takers' action reproduced, action distribution
    6 register_model      versioned folder + metadata (features, actions, dates, metrics, drift bins)
    7 score_full_base     greedy action for every eligible line -> nba_scored_base

Outputs (data/fsdr/<D>/outputs/): nba_training_report, nba_training_set, nba_scored_base.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import bandit, config, registry
from nbx.fsdr.calendar import anchor_date
from nbx.fsdr.datasource import DataSource, JsonSource
from nbx.fsdr import campaign_log
from nbx.fsdr.features import build_feature_frame, eligible_base, model_features
from nbx.fsdr.labels import build_labels, build_labels_from_log
from nbx.fsdr.pipelines import write_report, write_table
from nbx.fsdr.validation import validate


class DataBlocked(RuntimeError):
    pass


_VALIDATED: dict = {}


def _folder_signature(folder: Path) -> tuple:
    """Input files of a run folder with size and mtime: any change re-validates (outputs/ is a subfolder)."""
    return tuple(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file()))


def load_and_validate(data_dir: str | Path, run_date: date | None = None) -> tuple[DataSource, date, dict]:
    """Validation report of a run folder, cached while its input files are unchanged (each pipeline step calls
    this; the weekly cycle and the simulation would otherwise validate the same folder 6 times)."""
    src = JsonSource(data_dir)
    key = (str(Path(data_dir).resolve()), run_date, _folder_signature(Path(data_dir)))
    rep = _VALIDATED.get(key)
    if rep is None:
        if len(_VALIDATED) >= 32:                                   # bounded: a long-running API sees many folders
            _VALIDATED.pop(next(iter(_VALIDATED)))
        rep = _VALIDATED[key] = validate(src, run_date)
    if rep.status == "blocked":
        blocking = [f"{i.table}: {i.message}" for i in rep.issues if i.level == "blocking"]
        raise DataBlocked("data is blocked by validation: " + "; ".join(blocking[:5]))
    run_date = run_date or date.fromisoformat(rep.run_date)
    return src, run_date, {"status": rep.status, "issues": rep.counts(), "run_date": run_date.isoformat(),
                           "anchor_date": anchor_date(run_date).isoformat()}


def psi_bins(feat: pd.DataFrame, features: list[str], bins: int = config.PSI_BINS) -> dict:
    """Quantile edges + proportions per feature, stored with the model for the scoring drift check."""
    out = {}
    for f in features:
        v = pd.to_numeric(feat[f], errors="coerce").dropna().to_numpy()
        if not len(v):
            continue
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
        out[f] = {"edges": edges.round(6).tolist(), "p": (counts / counts.sum()).round(6).tolist()}
    return out


def evaluate(vw, hold: pd.DataFrame, features: list[str]) -> dict:
    if not len(hold):
        return {"n_holdout": 0}
    pmf = bandit.predict_pmfs(vw, hold, features)
    greedy = pmf.argmax(axis=1)
    match = greedy == hold["ACTION_IDX"].to_numpy()
    r, p = hold["REWARD"].to_numpy().astype(float), hold["PROPENSITY"].to_numpy().astype(float)
    w = match / p
    takers = r == 1
    acts = np.array(config.ACTIONS)
    return {
        "n_holdout": int(len(hold)),
        "holdout_take_rate": round(float(r.mean()), 4),                                       # current (logged) policy
        "holdout_ips_take_rate": round(float(np.mean(r * w)), 4),                              # model policy, IPS
        "holdout_snips_take_rate": round(float((r * w).sum() / w.sum()), 4) if w.sum() else 0.0,
        "holdout_taker_match": round(float(match[takers].mean()), 4) if takers.any() else None,
        "holdout_action_distribution": pd.Series(acts[greedy]).value_counts(normalize=True).round(4).to_dict(),
    }


def labels_for_week(src: DataSource, feat: pd.DataFrame, run_date: date, reward_def: str, log_root: Path,
                    label_source: str = config.LABEL_SOURCE) -> tuple[pd.DataFrame, dict]:
    """H5: last week's campaign log when it exists (action sent + logged propensity), else the targeting tables.
    The campaign is the one logged in the targeting window D-8..D-2 (BR-21)."""
    if label_source not in ("auto", "targeting", "campaign_log"):
        raise ValueError("label_source must be auto, targeting or campaign_log")
    if label_source != "targeting":
        lo, hi = config.TARGETING_WINDOW
        log = campaign_log.read_log(log_root, run_date + timedelta(days=lo), run_date + timedelta(days=hi))
        sent = log[(log["arm"] != "cg") & log["action"].isin(config.ACTIONS)] if len(log) else log
        if len(sent):
            return build_labels_from_log(src, feat, run_date, log, reward_def)
        if label_source == "campaign_log":
            raise RuntimeError(f"no campaign logged between D{lo} and D{hi} in {log_root}")
    train, rep = build_labels(src, feat, run_date, reward_def)
    return train, {"label_source": "targeting", **rep}


def run_training(data_dir: str | Path, run_date: date | None = None, mode: str = config.TRAIN_MODE,
                 reward_def: str = config.REWARD_DEFINITION, passes: int = config.TRAIN_PASSES,
                 epsilon: float = config.VW_EPSILON, holdout: float = config.HOLDOUT_SHARE, seed: int = config.GENERATOR_SEED,
                 promote: str = "auto", root: Path | None = None, score_base: bool = True,
                 label_source: str = config.LABEL_SOURCE) -> dict:
    """promote: 'auto' (code rule: IPS >= champion), 'never' (challenger only) or 'always' (forced)."""
    t0 = datetime.now()
    src, run_date, val_rep = load_and_validate(data_dir, run_date)
    features = model_features()
    log_root = campaign_log.root_for(data_dir)

    base, base_rep = eligible_base(src)
    # last-campaign context as it was when last week's campaign was decided (one week before D)
    feat, feat_rep = build_feature_frame(src, base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]], "T", seed,
                                         log_root=log_root, context_date=run_date - timedelta(days=7))
    train_all, lab_rep = labels_for_week(src, feat, run_date, reward_def, log_root, label_source)
    if len(train_all) < 20:
        raise RuntimeError(f"only {len(train_all)} labelled rows: not enough to train")

    rng = np.random.default_rng(seed)
    train_all["HOLDOUT"] = rng.random(len(train_all)) < holdout
    tr = train_all[~train_all["HOLDOUT"]].reset_index(drop=True)
    ho = train_all[train_all["HOLDOUT"]].reset_index(drop=True)
    vw, train_rep = bandit.train(tr, features, mode, passes, epsilon, seed)
    eval_rep = evaluate(vw, ho, features)

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"), "run_date": run_date.isoformat(),
        "anchor_date": anchor_date(run_date).isoformat(), "data_dir": str(data_dir), "vw_args": bandit.vw_args(epsilon),
        "train_mode": mode, "reward_definition": reward_def, "label_source": lab_rep["label_source"], "epsilon": epsilon, "passes": passes, "actions": config.ACTIONS,
        "features": features, "categorical": config.CATEGORICAL_FEATURES,
        "registry": [m.table for m in config.MODEL_SCORE_REGISTRY],
        "n_train": int(len(tr)), "n_takers_train": int(tr["REWARD"].sum()), **eval_rep,
        "psi_bins": psi_bins(feat, features),
    }
    version = registry.register(vw, meta, root)
    ok, why = registry.can_promote(version, root)
    promoted = False
    if promote == "always" or (promote == "auto" and ok):
        registry.promote(version, root, reason=why, force=promote == "always")
        promoted = True

    scored = None
    if score_base:
        scored = bandit.score_population(vw, feat, features, policy=config.BASE_SCORING_POLICY,
                                         id_cols=("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID"), seed=seed)
        scored["SCORE_DATE"] = run_date.isoformat()
        scored["MODEL_VERSION"] = version
        write_table(scored, data_dir, "nba_scored_base")

    keep = ["ACCESS_METHOD_VAL", "ACTION", "REWARD", "REWARD_ANY_BTL", "REWARD_SAME_SERVICE", "REWARD_EXACT_OFFER",
            "PROPENSITY", "HOLDOUT"] + features
    write_table(train_all[keep], data_dir, "nba_training_set")
    report = {
        "pipeline": "train", "version": version, "promoted": promoted, "promotion_rule": why,
        "champion": registry.champion(root), "validation": val_rep, "base": base_rep, "features": feat_rep,
        "labels": lab_rep, "training": train_rep, "evaluation": eval_rep,
        "scored_base_action_distribution": scored["ACTION"].value_counts(normalize=True).round(4).to_dict() if scored is not None else None,
        "seconds": round((datetime.now() - t0).total_seconds(), 1),
    }
    write_report(report, data_dir, "nba_training_report")
    return report
