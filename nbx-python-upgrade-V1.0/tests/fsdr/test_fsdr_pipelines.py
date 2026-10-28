"""Training / scoring pipelines on the JSON folder, model registry rules."""
from __future__ import annotations

import json

import pytest

from nbx.fsdr import config, registry
from nbx.fsdr.datasource import SyntheticSource
from nbx.fsdr.pipelines import read_table
from nbx.fsdr.pipelines.score import StaleModel, run_scoring
from nbx.fsdr.pipelines.train import DataBlocked, run_training

from .conftest import RUN_DATE


def test_training_report_and_outputs(scored_run):
    rep, folder = scored_run["train"], scored_run["folder"]
    assert rep["promoted"] and rep["champion"] == rep["version"] == "v0001"
    ev = rep["evaluation"]
    assert ev["n_holdout"] > 30 and 0 <= ev["holdout_ips_take_rate"] and 0 <= ev["holdout_take_rate"] <= 1
    assert set(ev["holdout_action_distribution"]) <= set(config.ACTIONS)
    for name in ("nba_training_report", "nba_training_set", "nba_scored_base"):
        assert (folder / "outputs" / f"{name}.json").exists()
    meta = registry.meta("v0001", scored_run["root"])
    assert meta["actions"] == config.ACTIONS and meta["psi_bins"] and meta["run_date"] == RUN_DATE.isoformat()


def test_scoring_scores_the_three_arms_but_not_the_control_group(scored_run):
    rep, folder = scored_run["score"], scored_run["folder"]
    assert set(rep["action_distribution_per_arm"]) == {"free", "with_constraint", "llm"}
    res = read_table(folder, "results_scoring_nba")
    sample = SyntheticSource(n=1200, run_date=RUN_DATE, seed=13).read(config.TARGET_SAMPLE_TABLE)
    cg = set(sample.loc[sample["group"] == "cg", "access_method_val"])
    assert not set(res["ACCESS_METHOD_VAL"].astype(str)) & cg
    assert {f"P_{a.replace(' ', '_')}" for a in config.ACTIONS} <= set(res.columns)
    for name in ("TG_free_scored_nba", "TG_constraint_scored_nba", "TG_llm_scored_nba"):
        assert len(read_table(folder, name)) > 0
    assert rep["drift"]["max_psi"] < config.PSI_ALERT                     # T vs D windows of the same world


def test_scoring_refuses_a_stale_model(scored_run, monkeypatch):
    monkeypatch.setattr(config, "MAX_MODEL_AGE_DAYS", -1)
    with pytest.raises(StaleModel):
        run_scoring(scored_run["folder"], root=scored_run["root"])


def test_blocked_data_is_refused(tmp_path):
    folder = SyntheticSource(n=300, run_date=RUN_DATE, scenarios="data_quality_issues", seed=2).write(tmp_path / "bad")
    with pytest.raises(DataBlocked):
        run_training(folder, root=tmp_path / "models")


def test_promotion_is_code_enforced_and_rollback_works(scored_run, tmp_path):
    root = tmp_path / "models"
    first = run_training(scored_run["folder"], root=root, score_base=False)
    assert first["promoted"]
    idx = json.loads((root / "registry.json").read_text())
    # a challenger with a worse IPS cannot be promoted, unless forced
    second = run_training(scored_run["folder"], root=root, score_base=False, promote="never")
    idx = json.loads((root / "registry.json").read_text())
    idx["models"][second["version"]]["holdout_ips_take_rate"] = -1.0
    (root / "registry.json").write_text(json.dumps(idx))
    ok, why = registry.can_promote(second["version"], root)
    assert not ok and "<" in why
    with pytest.raises(ValueError):
        registry.promote(second["version"], root)
    registry.promote(second["version"], root, force=True)
    assert registry.champion(root) == second["version"]
    registry.rollback(root)
    assert registry.champion(root) == first["version"]
