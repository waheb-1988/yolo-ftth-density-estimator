"""FSDR NBA contextual bandit (Vowpal Wabbit cb_explore_adf) -- port of nba_lib / 01_nba_train.

  train()             logged mode (default: every targeted line, takers AND non-takers,
                      real logged propensity) or simulation mode (production original:
                      takers only, cost -1 when the sampled action = the logged action)
  predict_pmfs()      probability of each of the 7 actions, in config.ACTIONS order
  score_population()  sample (production, keeps exploration) or greedy action per line
  save() / load()     model + metadata; load refuses a model whose features or actions differ

The text format and the learning / sampling loops live in nbx/core/vw_format.py, shared with
core/nba_bandit.py: numeric features name:value, action tokens without spaces, fixed order.
The workspace is never finish()ed before scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import vowpalwabbit

from nbx.core import vw_format
from nbx.fsdr import config
from nbx.fsdr.features import model_features

MODEL_FILE = "nba_cb_adf.vw"
META_FILE = "nba_meta.json"


class IncompatibleModel(ValueError):
    pass


def vw_args(epsilon: float = config.VW_EPSILON) -> str:
    return config.VW_ARGS_TEMPLATE.format(epsilon=epsilon)


def shared(df: pd.DataFrame, features: list[str]) -> list[str]:
    return vw_format.shared_lines(df, features, config.CATEGORICAL_FEATURES)


def train(train_df: pd.DataFrame, features: list[str], mode: str = config.TRAIN_MODE, passes: int = config.TRAIN_PASSES,
          epsilon: float = config.VW_EPSILON, seed: int = config.GENERATOR_SEED,
          sim_iterations: int = config.SIM_ITERATIONS) -> tuple[vowpalwabbit.Workspace, dict]:
    """Returns (workspace, report). train_df needs ACTION_IDX, COST, PROPENSITY, REWARD + the features."""
    if mode not in ("logged", "simulation"):
        raise ValueError("mode must be 'logged' or 'simulation'")
    lo, hi = config.VW_EPSILON_RANGE
    if not lo <= epsilon <= hi:
        raise ValueError(f"epsilon {epsilon} outside the allowed range {config.VW_EPSILON_RANGE}")
    rng = np.random.default_rng(seed)
    vw = vowpalwabbit.Workspace(arg_list=vw_args(epsilon).split())
    actions = config.ACTIONS
    report = {"mode": mode, "epsilon": epsilon, "rows": len(train_df)}
    if mode == "logged":
        vw_format.learn_logged(vw, shared(train_df, features), actions, train_df["ACTION_IDX"].to_numpy(),
                               train_df["COST"].to_numpy(), train_df["PROPENSITY"].to_numpy(), passes, rng)
        report["passes"] = passes
    else:
        tk = train_df[train_df["REWARD"] == 1].reset_index(drop=True)
        sim = vw_format.learn_simulation(vw, shared(tk, features), actions, tk["ACTION_IDX"].to_numpy(), sim_iterations, rng)
        report.update({"iterations": sim_iterations, "takers": len(tk), "match_rate": sim["match_rate"]})
    return vw, report


def predict_pmfs(vw: vowpalwabbit.Workspace, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    sh = shared(df, features)
    out = np.zeros((len(sh), len(config.ACTIONS)))
    for i, s in enumerate(sh):
        out[i] = vw_format.predict_pmf(vw, s, config.ACTIONS)
    return out


def score_population(vw: vowpalwabbit.Workspace, feat: pd.DataFrame, features: list[str], policy: str = config.SCORING_POLICY,
                     id_cols: tuple[str, ...] = ("ACCESS_METHOD_VAL",), seed: int = config.GENERATOR_SEED) -> pd.DataFrame:
    """policy 'sample' draws the action from the pmf (production, BR-57); 'greedy' takes the most probable."""
    if policy not in ("sample", "greedy"):
        raise ValueError("policy must be 'sample' or 'greedy'")
    pmfs = predict_pmfs(vw, feat, features)
    n = len(pmfs)
    greedy = pmfs.argmax(axis=1) if n else np.zeros(0, int)
    u = np.random.default_rng(seed).random(n)[:, None]
    sampled = np.minimum((pmfs.cumsum(axis=1) < u).sum(axis=1), len(config.ACTIONS) - 1) if n else greedy
    chosen = sampled if policy == "sample" else greedy
    acts = np.array(config.ACTIONS)
    out = feat[list(id_cols)].copy().reset_index(drop=True)
    out["ACTION"] = acts[chosen] if n else []
    out["PROBA"] = pmfs[np.arange(n), chosen] if n else []
    out["ACTION_GREEDY"] = acts[greedy] if n else []
    out["PROBA_GREEDY"] = pmfs[np.arange(n), greedy] if n else []
    for i, a in enumerate(config.ACTIONS):
        out[f"P_{vw_format.action_token(a)}"] = pmfs[:, i] if n else []
    out["POLICY"] = policy
    return out.drop_duplicates("ACCESS_METHOD_VAL").reset_index(drop=True)                      # BR-58


def save(vw: vowpalwabbit.Workspace, meta: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vw.save(str(folder / MODEL_FILE))       # --save_resume is not needed: training restarts from the logged data each week
    meta = {"categorical": config.CATEGORICAL_FEATURES, **meta}     # VW shared-namespace tokens the model was trained with
    (folder / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return folder


def check_compatible(meta: dict, features: list[str] | None = None) -> None:
    features = features if features is not None else model_features()
    if meta.get("actions") != config.ACTIONS:
        raise IncompatibleModel("model was trained with a different action list or order -- retrain it")
    if meta.get("categorical", ["NATIONALITY_GROUP"]) != config.CATEGORICAL_FEATURES:
        raise IncompatibleModel(f"model was trained with categorical features {meta.get('categorical')}, "
                                f"this config uses {config.CATEGORICAL_FEATURES} -- retrain it")
    if meta.get("features") != features:
        missing = sorted(set(features) - set(meta.get("features") or []))
        extra = sorted(set(meta.get("features") or []) - set(features))
        raise IncompatibleModel(f"model was trained with a different feature list (missing {missing}, extra {extra}) -- retrain it")


def load(folder: Path, features: list[str] | None = None) -> tuple[vowpalwabbit.Workspace, dict]:
    folder = Path(folder)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    check_compatible(meta, features)
    # arg_list keeps the path whole: VW splits arg_str on whitespace and model paths can contain spaces
    vw = vowpalwabbit.Workspace(arg_list=meta["vw_args"].split() + ["-i", str(folder / MODEL_FILE)])
    return vw, meta
