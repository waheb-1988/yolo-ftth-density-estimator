"""V1.0 improvements from docs/FSDR_REVIEW_2026-10.md, one block per finding (#)."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nbx.fsdr import campaign_log, config, registry, stats
from nbx.fsdr.datasource import JsonSource


# --- #2 / #3 off-policy evaluation and promotion ---------------------------------------------------------------
def test_ips_snips_of_a_policy_equal_to_the_logger_is_the_logged_mean():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 500)
    r = (rng.random(500) < 0.3).astype(float)
    ips, snips = stats.ips_snips(p, r, p)                    # pi = logging policy -> weights 1
    assert ips == pytest.approx(r.mean()) and snips == pytest.approx(r.mean())


def test_paired_bootstrap_ci_brackets_the_difference_and_detects_a_better_policy():
    rng = np.random.default_rng(1)
    n = 2000
    a = rng.integers(0, 2, n)                                 # 2 actions, uniform logging
    r = (rng.random(n) < np.where(a == 1, 0.30, 0.10)).astype(float)
    p = np.full(n, 0.5)
    good, bad = np.where(a == 1, 0.9, 0.1), np.where(a == 1, 0.1, 0.9)
    res = stats.paired_bootstrap(good, bad, r, p, seed=3)
    assert res["ci"][0] <= res["delta"] <= res["ci"][1] and res["ci"][0] > 0.05


def _reg(tmp_path, models: dict, champion="v0001") -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "registry.json").write_text(json.dumps({"champion": champion, "history": [], "models": models}))


def test_paired_rule_refuses_only_a_significantly_worse_challenger(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROMOTION_RULE", "paired_ci")
    base = {"holdout_ips_take_rate": 0.10, "ope_valid": True}

    def vs(d, lo, hi):
        return {"vs_champion": {"champion": "v0001", "n": 400, "delta": d, "ci": [lo, hi]}}

    _reg(tmp_path, {"v0001": base,
                    "v0002": {**base, "holdout_ips_take_rate": 0.05, **vs(0.0, -0.01, 0.02)},
                    "v0003": {**base, "holdout_ips_take_rate": 0.20, **vs(-0.05, -0.08, -0.03)},
                    "v0004": {**base, **vs(0.04, 0.01, 0.07)}})
    ok, why = registry.can_promote("v0002", tmp_path)
    assert ok and "inconclusive" in why                       # point IPS lower (V0 would refuse): noise
    ok, why = registry.can_promote("v0003", tmp_path)
    assert not ok and "significantly worse" in why            # point IPS higher (V0 would promote), worse on the same rows
    ok, why = registry.can_promote("v0004", tmp_path)
    assert ok and "significantly better" in why


def test_paired_rule_falls_back_to_point_rule_without_valid_ope(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROMOTION_RULE", "paired_ci")
    _reg(tmp_path, {"v0001": {"holdout_ips_take_rate": 0.10}, "v0002": {"holdout_ips_take_rate": 0.12, "ope_valid": False}})
    ok, why = registry.can_promote("v0002", tmp_path)
    assert ok and "point rule" in why


def test_registry_lock_blocks_a_second_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "LOCK_TIMEOUT_S", 0.2)
    with registry._locked(tmp_path):
        with pytest.raises(TimeoutError):
            with registry._locked(tmp_path):
                pass
    with registry._locked(tmp_path):                          # released
        pass


def test_training_reports_deployed_policy_ope_and_champion_comparison(scored_run):
    ev = scored_run["train"]["evaluation"]
    assert ev["ope_policy"] == "sample" and ev["ope_valid"] is False          # legacy targeting labels
    lo, hi = ev["holdout_snips_ci"]
    assert lo <= ev["holdout_snips_take_rate"] <= hi
    assert "holdout_ips_greedy_take_rate" in ev


# --- #1 propensity of the legacy policy --------------------------------------------------------------------------
def test_legacy_targeting_labels_have_deterministic_propensity(small_source):
    from nbx.fsdr.features import build_feature_frame, eligible_base
    from nbx.fsdr.labels import build_labels

    base, _ = eligible_base(small_source)
    feat, _ = build_feature_frame(small_source, base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]], "T")
    train, rep = build_labels(small_source, feat, date(2026, 10, 21))
    assert (train["PROPENSITY"] == 1.0).all() and rep["targeting_propensity"] == "deterministic"


# --- #4 uplift reward ----------------------------------------------------------------------------------------------
def test_uplift_adjust_subtracts_the_control_baseline(small_source, monkeypatch):
    from nbx.fsdr import labels

    rng = np.random.default_rng(0)
    vals = [str(70_000_000 + i) for i in range(400)]
    feat = pd.DataFrame({"ACCESS_METHOD_VAL": vals, "X": rng.random(400)})
    log = pd.DataFrame({"msisdn": vals[:200], "arm": "cg", "run_date": "2026-10-14", "offers": [[]] * 200,
                        "action": config.NO_CONTACT, "propensity": 1.0, "policy_source": "control"})
    monkeypatch.setattr(campaign_log, "compute_rewards", lambda log, f, c: [
        {"msisdn": m, **{f"reward_{d}": int(i % 10 == 0) for d in config.REWARD_DEFINITIONS}} for i, m in enumerate(log["msisdn"])])
    train = pd.DataFrame({"ACCESS_METHOD_VAL": vals[200:], "ACTION": "RETAIN", "REWARD": 1, "COST": -1.0,
                          "X": feat["X"].iloc[200:].to_numpy()})
    out, rep = labels.uplift_adjust(train, small_source, feat, date(2026, 10, 21), log, ["X"])
    assert rep["applied"] and rep["baseline_model"] == "logistic" and rep["control_take_rate"] == pytest.approx(0.1)
    assert out["COST"].between(-1.0, -0.8).all()               # -(1 - ~0.1)
    out, rep = labels.uplift_adjust(train, small_source, feat, date(2026, 10, 21), log.head(60), ["X"])
    assert rep["baseline_model"] == "constant" and out["BASELINE"].nunique() == 1   # 6 control takers: no model


# --- #6 consent ----------------------------------------------------------------------------------------------------
def test_consent_table_is_generated_and_blocks_every_arm(scored_run, workspace, monkeypatch):
    from nbx.fsdr.agents.tools import contact_policy
    from nbx.fsdr.features import consent_blocked
    from nbx.fsdr.pipelines.recommend import run_recommend
    from nbx.fsdr.pipelines import read_table

    folder = scored_run["folder"]
    blocked = consent_blocked(JsonSource(folder))
    assert 0 < len(blocked) < 0.1 * 1200
    run_recommend(folder, "cstr")
    recos = pd.concat([read_table(folder, "recos_monthly_cstr"), read_table(folder, "recos_weekly_cstr")])
    assert not set(recos["ACCESS_METHOD_VAL"].astype(str)) & blocked
    llm_blocked = [v for v in workspace.lines("llm") if v in blocked]
    if llm_blocked:
        res = contact_policy(workspace.context(llm_blocked[0]), {})
        assert not res["can_contact"] and any("consent" in r for r in res["blocked_reasons"])


# --- #10 / #14 acceptance model and cold start -------------------------------------------------------------------
def test_acceptance_model_scores_an_offer_never_sold(workspace):
    acc = workspace.engine.acceptance
    assert acc is not None and acc.report["positives"] >= config.ACCEPTANCE_MIN_POSITIVES
    new = pd.DataFrame({"SERVICE": ["DATA", "DATA"], "GENEROSITY": ["L2", "L2"], "VALIDITY": [30, 30], "PRICE": [8.0, 30.0]})
    val = workspace.lines("llm")[0]
    p = acc.p_accept(val, "UPSELL DATA", new)
    assert ((p > 0) & (p < 1)).all() and p[0] > p[1]           # same offer far above the spend: less likely


def test_expected_value_tool_uses_the_acceptance_model(workspace):
    from nbx.fsdr.agents.tools import expected_value

    ctx = workspace.context(workspace.lines("llm")[0])
    names = workspace.catalog["BTL_NAME"].head(3).tolist()
    out = expected_value(ctx, {"offers": names})
    assert out["model"].startswith("acceptance model") and len(out["ranked"]) == 3


# --- #8 PII --------------------------------------------------------------------------------------------------------
def test_prompts_carry_a_pseudonym_not_the_msisdn(workspace):
    from nbx.agents.agentic.react import build_prompt
    from nbx.core.privacy import pseudonym
    from nbx.fsdr.agents.recommender_agent import ACTION_SPEC
    from nbx.fsdr.agents.tools import FSDR_TOOLS, customer_360

    val = workspace.lines("llm")[0]
    ctx = workspace.context(val)
    prompt = build_prompt(ACTION_SPEC, ctx, [], "", None, FSDR_TOOLS)
    assert val not in prompt and pseudonym(val) in prompt
    assert val not in json.dumps(customer_360(ctx, {}), default=str)
    assert pseudonym(val) == pseudonym(val) and pseudonym(val) != pseudonym(str(int(val) + 1))


# --- #9 anchor guard ------------------------------------------------------------------------------------------------
def test_anchor_growth_guard_is_on_by_default():
    assert config.ANCHOR_MAX_WEEKLY_GROWTH == pytest.approx(0.10)


# --- #11 / #12 features ---------------------------------------------------------------------------------------------
def test_extended_features_add_missing_flags_and_tenure(monkeypatch, small_source):
    from nbx.fsdr.features import build_feature_frame, eligible_base, model_features

    assert len(model_features()) == 48 + 3 + 10
    base, _ = eligible_base(small_source)
    feat, _ = build_feature_frame(small_source, base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]], "D")
    assert set(model_features()) <= set(feat.columns)
    m = feat["PROBA_CHURN_MISSING"] == 1
    assert m.any() and feat.loc[m, "PROBA_CHURN"].between(*config.MISSING_SCORE_RANGE).all()
    assert (feat["LOG_TENURE"] > 0).mean() > 0.9
    monkeypatch.setattr(config, "EXTENDED_FEATURES", False)
    assert len(model_features()) == 48 + 3                     # production parity


# --- #7 / #13 report ---------------------------------------------------------------------------------------------------
def test_feedback_summary_has_uplift_ci_power_and_fairness():
    from nbx.fsdr.pipelines.feedback import summarise

    rng = np.random.default_rng(0)
    n = 800
    arm = np.where(np.arange(n) < 400, "cg", "cstr")
    rew = pd.DataFrame({"msisdn": [str(70_000_000 + i) for i in range(n)], "arm": arm, "action": "UPSELL",
                        **{f"reward_{d}": (rng.random(n) < np.where(arm == "cg", 0.03, 0.12)).astype(int) for d in config.REWARD_DEFINITIONS}})
    rew["revenue"] = rew["reward_any_btl"] * 5.0
    groups = pd.Series(np.where(np.arange(n) % 2, "Oman", "India"), index=rew["msisdn"])
    out = summarise(rew, pd.DataFrame(columns=campaign_log.LOG_FIELDS), groups)
    t = out["arms"]["cstr"]
    assert t["uplift_test"]["ci"][0] > 0 and t["uplift_test"]["p_value"] < 0.01
    assert 0 < t["detectable_uplift_80pct_power"] < 0.05
    assert set(out["fairness"]["cstr"]["groups"]) == {"Oman", "India"}
    assert stats.control_size_for(0.03, 0.02, 400) is None             # 400 treated lines alone cannot detect 2 pts
    assert 500 < stats.control_size_for(0.03, 0.02, 4000) < 1000


# --- #15 campaign log cache ----------------------------------------------------------------------------------------------
def test_campaign_log_appends_incrementally_and_sees_outside_rewrites(tmp_path):
    rows = [{"run_date": "2026-10-21", "arm": "cstr", "msisdn": str(70_000_000 + i), "action": "UPSELL"} for i in range(5)]
    assert campaign_log.append_log(tmp_path, rows) == 5
    assert campaign_log.append_log(tmp_path, rows + rows[:1]) == 0                    # every key already there
    assert campaign_log.append_log(tmp_path, [{**rows[0], "msisdn": "79999999"}]) == 1
    assert len(campaign_log.read_log(tmp_path)) == 6
    (tmp_path / campaign_log.LOG_FILE).write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")   # rewritten outside
    assert len(campaign_log.read_log(tmp_path)) == 1


# --- uploaded exports (no _manifest.json): the run date comes from the folder name --------------------------------
def test_a_run_without_manifest_trains_like_an_upload(small_folder, tmp_path):
    import shutil

    from nbx.fsdr.pipelines.train import DataBlocked, load_and_validate

    up = tmp_path / small_folder.name
    shutil.copytree(small_folder, up)
    (up / config.MANIFEST_FILE).unlink()                         # what /fsdr/upload leaves behind
    _, run_date, rep = load_and_validate(up)
    assert run_date == date.fromisoformat(small_folder.name) and rep["anchor_date"] == "2026-10-15"
    bad = tmp_path / "exports"
    shutil.copytree(up, bad)
    with pytest.raises(DataBlocked, match="no run date"):
        load_and_validate(bad)
