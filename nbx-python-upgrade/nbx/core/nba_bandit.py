"""NBA contextual bandit -- ported from the production Vowpal Wabbit script.

Real `vowpalwabbit` (not a substitute) installs cleanly on this Python
3.12/Windows setup, so this is the actual --cb_explore_adf -q UA
--epsilon 0.2 model, trained the same way production trains it: an online
"replay" simulation over takers (customers with REWARD==1), where the
bandit's exploratory pick is scored against that customer's real historical
action and learned from live -- see run_simulation() / get_action() /
to_vw_example_format() in the source script for the pattern this mirrors.

Two corrections vs. the source, documented in core/config.py's NBA_* block:
  1. context_ list deduped/completed (proba_frag_voice wasn't duplicated,
     proba_app_voice wasn't dropped).
  2. Action names underscored inside VW example strings only -- VW's text
     format splits features on whitespace, so a literal "CROSS SELL VOICE"
     value would parse as three separate tokens, not one categorical value.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import vowpalwabbit

from nbx.core import config, vw_format


def _load_json(path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_change(newer: float, older: float, epsilon: float | None = None) -> float:
    """(newer-older)/older*100, with the production script's zero-baseline
    handling: both periods zero -> 0; baseline zero but newer nonzero ->
    substitute epsilon for the baseline (a large-but-finite % instead of a
    NaN/inf from dividing by zero). Moved here from core/data_simulator.py --
    production runs calcul_evol() on the already-merged dataframe, after the
    per-source datasets below are joined, not before."""
    eps = epsilon if epsilon is not None else config.NBA_PCT_CHANGE_EPSILON
    if older == 0 and newer == 0:
        return 0.0
    if older == 0:
        return round((newer - eps) / eps * 100, 2)
    return round((newer - older) / older * 100, 2)


def _load_scores() -> list[dict]:
    """Merges the 7 NBA source files (core.config.NBA_SOURCE_FILE_ATTRS) plus
    revenue_history.json (shared with NBO's affordability rule) into one
    per-customer row -- the same flat context shape get_score_row() /
    context_for_msisdn() / training always expected, so nothing downstream
    of this function changed when the single nba_scores.json split into 7.
    targeting.json is the spine (it's the only file with excluded/action/
    reward/cost/propensity); a msisdn missing from the other 6 files gets
    0.0 for those features rather than dropping the row -- same "impute
    rather than fail" policy the source script uses for missing model
    scores (np.random.uniform(0, 0.02) there; 0.0 here since this is a
    read-time default, not a training-time simulation choice)."""
    targeting = _load_json(config.NBA_TARGETING_FILE)
    if not targeting:
        return []

    def _by_msisdn(path):
        return {row["msisdn"]: row for row in _load_json(path)}

    churn = _by_msisdn(config.NBA_CHURN_FILE)
    reactivation = _by_msisdn(config.NBA_REACTIVATION_FILE)
    appetency = _by_msisdn(config.NBA_APPETENCY_FILE)
    fragility = _by_msisdn(config.NBA_FRAGILITY_FILE)
    rev_decrease = _by_msisdn(config.NBA_REV_DECREASE_FILE)
    usage = _by_msisdn(config.NBA_ACTIVITY_USAGE_FILE)
    revenue = _by_msisdn(config.REVENUE_HISTORY_FILE)

    merged = []
    for row in targeting:
        msisdn = row["msisdn"]
        c, r, a, f = churn.get(msisdn, {}), reactivation.get(msisdn, {}), appetency.get(msisdn, {}), fragility.get(msisdn, {})
        rd, u, rev = rev_decrease.get(msisdn, {}), usage.get(msisdn, {}), revenue.get(msisdn, {})

        merged.append(
            {
                **row,
                "proba_churn": c.get("proba_churn", 0.0),
                "proba_churn_1d": c.get("proba_churn_1d", 0.0),
                "proba_react_30": r.get("proba_react_30", 0.0),
                "proba_react_15": r.get("proba_react_15", 0.0),
                "proba_app_data": a.get("proba_app_data", 0.0),
                "proba_app_voice": a.get("proba_app_voice", 0.0),
                "proba_frag_data": f.get("proba_frag_data", 0.0),
                "proba_frag_voice": f.get("proba_frag_voice", 0.0),
                "proba_rev_decr": rd.get("proba_rev_decr", 0.0),
                "var_rev_w1_w2": _pct_change(rev.get("rev_w1", 0), rev.get("rev_w2", 0)),
                "var_rev_w2_w3": _pct_change(rev.get("rev_w2", 0), rev.get("rev_w3", 0)),
                "var_rev_w3_w4": _pct_change(rev.get("rev_w3", 0), rev.get("rev_w4", 0)),
                "var_rev_m1_m2": _pct_change(rev.get("rev_m1", 0), rev.get("rev_m2", 0)),
                "var_rev_m2_m3": _pct_change(rev.get("rev_m2", 0), rev.get("rev_m3", 0)),
                "var_voice_w1_w2": _pct_change(u.get("local_voice_min_w1", 0), u.get("local_voice_min_w2", 0)),
                "var_voice_w2_w3": _pct_change(u.get("local_voice_min_w2", 0), u.get("local_voice_min_w3", 0)),
                "var_voice_w3_w4": _pct_change(u.get("local_voice_min_w3", 0), u.get("local_voice_min_w4", 0)),
                "var_voice_m1_m2": _pct_change(u.get("local_voice_min_m1", 0), u.get("local_voice_min_m2", 0)),
                "var_voice_m2_m3": _pct_change(u.get("local_voice_min_m2", 0), u.get("local_voice_min_m3", 0)),
                "var_data_w1_w2": _pct_change(u.get("data_volume_mb_w1", 0), u.get("data_volume_mb_w2", 0)),
                "var_data_w2_w3": _pct_change(u.get("data_volume_mb_w2", 0), u.get("data_volume_mb_w3", 0)),
                "var_data_w3_w4": _pct_change(u.get("data_volume_mb_w3", 0), u.get("data_volume_mb_w4", 0)),
                "var_data_m1_m2": _pct_change(u.get("data_volume_mb_m1", 0), u.get("data_volume_mb_m2", 0)),
                "var_data_m2_m3": _pct_change(u.get("data_volume_mb_m2", 0), u.get("data_volume_mb_m3", 0)),
            }
        )
    return merged


def get_score_row(msisdn: int, scores: list[dict] | None = None) -> dict | None:
    """The full simulated NBA row for one customer -- context features plus
    nba_segment/excluded/action/reward/propensity. Used by agents/nba_agent.py
    both to build the bandit's context and to report bandit_context in the
    response (nba_segment, historical_action)."""
    scores = scores if scores is not None else _load_scores()
    return next((s for s in scores if s["msisdn"] == msisdn), None)


def context_for_msisdn(msisdn: int, scores: list[dict] | None = None) -> dict | None:
    row = get_score_row(msisdn, scores)
    if row is None:
        return None
    return {f: row.get(f, 0) for f in config.NBA_CONTEXT_FEATURES}


# One VW text format for the whole app (core/vw_format.py, shared with nbx/fsdr/bandit.py): numeric
# features as name:value (the old "feat=value" made VW read every number as a category), action tokens
# without spaces, the action list in fixed order.
_CATEGORICAL = ["nationality_group"]
_NUMERIC = [f for f in config.NBA_CONTEXT_FEATURES if f not in _CATEGORICAL]


def _shared_line(ctx: dict) -> str:
    return vw_format.shared_lines(pd.DataFrame([{f: ctx.get(f, 0) for f in config.NBA_CONTEXT_FEATURES}]), _NUMERIC, _CATEGORICAL)[0]


_RNG = np.random.default_rng()                  # unseeded, as the source: live exploration differs run to run


def get_action(ctx: dict, vw: "vowpalwabbit.Workspace" = None) -> tuple[str, float]:
    """Predicts + samples one action for a single customer's context.
    Mirrors the source's get_action(): predict the PMF, sample an index,
    return (action_name, its_probability)."""
    vw = vw or get_bandit()
    idx, prob = vw_format.sample_action(vw_format.predict_pmf(vw, _shared_line(ctx), config.NBA_ACTIONS), _RNG)
    return config.NBA_ACTIONS[idx], prob


def _run_simulation(vw: "vowpalwabbit.Workspace", takers: list[dict], iterations: int) -> dict:
    """Mirrors the source's run_simulation(): repeatedly pick a random
    taker, let the current policy choose an action, score it against that
    taker's real historical action (cost -1 if it matches, else 0), and
    learn from it live. Tracks cumulative reward and the action distribution
    actually explored, for core/nba_bandit.py::get_stats()."""
    shared = [_shared_line({f: row.get(f, 0) for f in config.NBA_CONTEXT_FEATURES}) for row in takers]
    true_idx = [config.NBA_ACTIONS.index(row["action"]) for row in takers]
    sim = vw_format.learn_simulation(vw, shared, config.NBA_ACTIONS, true_idx, iterations, _RNG)
    return {"avg_reward": sim["match_rate"], "action_counts": sim["action_counts"], "iterations": iterations,
            "n_takers": len(takers)}


class _BanditModel:
    def __init__(self, vw, stats: dict):
        self.vw = vw
        self.stats = stats


_bandit_cache: _BanditModel | None = None


def _train() -> _BanditModel:
    scores = _load_scores()
    takers = [s for s in scores if not s.get("excluded") and s.get("action") and s.get("reward") == 1]

    vw = vowpalwabbit.Workspace(config.NBA_VW_ARGS)
    if len(takers) < 2:
        # Not enough labeled takers to learn anything meaningful yet (e.g.
        # right after a fresh seed with an unlucky draw) -- ship an
        # untrained (uniform-exploration) policy rather than crash.
        return _BanditModel(vw, {"avg_reward": 0.0, "action_counts": {a: 0 for a in config.NBA_ACTIONS}, "iterations": 0, "n_takers": len(takers)})

    stats = _run_simulation(vw, takers, config.NBA_TRAIN_ITERATIONS)
    return _BanditModel(vw, stats)


def get_bandit(force_rebuild: bool = False) -> "vowpalwabbit.Workspace":
    global _bandit_cache
    if force_rebuild or _bandit_cache is None:
        if _bandit_cache is not None:
            _bandit_cache.vw.finish()
        _bandit_cache = _train()
    return _bandit_cache.vw


def get_stats() -> dict:
    get_bandit()  # ensures _bandit_cache is populated
    return {
        **_bandit_cache.stats,
        "actions": config.NBA_ACTIONS,
        "vw_args": config.NBA_VW_ARGS,
    }
