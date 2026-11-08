"""Campaign log -> feedback -> next week's training (audit M1..M4, H5, L4): two consecutive weeks."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pandas as pd
import pytest

from nbx.fsdr import campaign_log, config
from nbx.fsdr.offer_rules import cap_anchor_growth

W1 = date(2026, 10, 8)
W2 = W1 + timedelta(days=7)


@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    from nbx.fsdr.agents.recommender_agent import run_batch
    from nbx.fsdr.agents.tools import FsdrWorkspace
    from nbx.fsdr.pipelines.feedback import run_feedback
    from nbx.fsdr.pipelines.recommend import run_recommend
    from nbx.fsdr.pipelines.score import run_scoring
    from nbx.fsdr.pipelines.train import run_training
    from nbx.fsdr.synth.generator import generate_weeks

    base = tmp_path_factory.mktemp("fsdr loop")
    generate_weeks(1500, W1, 2, "baseline", seed=31, out_root=base)
    d1, d2, models = base / W1.isoformat(), base / W2.isoformat(), base / "models"
    t1 = run_training(d1, root=models)
    run_scoring(d1, root=models)
    rec = {arm: run_recommend(d1, arm) for arm in ("cstr", "free")}
    llm = asyncio.run(run_batch(d1, limit=40, ws=FsdrWorkspace(d1)))
    fb = run_feedback(d2)
    t2 = run_training(d2, root=models, promote="never")
    return {"root": campaign_log.root_for(d1), "t1": t1, "rec": rec, "llm": llm, "fb": fb, "t2": t2, "d2": d2}


def test_campaign_log_holds_every_arm_with_the_required_fields(loop):
    log = campaign_log.read_log(loop["root"])
    assert set(log["arm"]) == {"cg", "free", "cstr", "llm"}
    assert set(campaign_log.LOG_FIELDS) <= set(log.columns)
    sent = log[log["arm"] != "cg"]
    assert sent["propensity"].between(0, 1).all() and sent["bandit_pmf"].map(lambda p: len(p) == 7).all()
    with_offers = sent[sent["offers"].map(len) > 0]
    assert len(with_offers) and with_offers["offers"].map(lambda os: all({"code", "price", "validity"} <= set(o) for o in os)).all()
    assert log[log["arm"] == "llm"]["justification_id"].notna().all()


def test_campaign_log_is_append_only_and_idempotent(loop):
    root = loop["root"]
    before = (root / campaign_log.LOG_FILE).read_text(encoding="utf-8")
    row = campaign_log.read_log(root).iloc[0].to_dict()
    assert campaign_log.append_log(root, [row]) == 0                         # same key: skipped
    assert (root / campaign_log.LOG_FILE).read_text(encoding="utf-8") == before


def test_feedback_writes_the_three_rewards_per_logged_line(loop):
    rew = campaign_log.read_rewards(loop["root"])
    log = campaign_log.read_log(loop["root"])
    assert len(rew) == len(log) and loop["fb"]["rows"] == len(log)
    assert (rew["reward_exact_offer"] <= rew["reward_same_service"]).all()
    assert (rew["reward_same_service"] <= rew["reward_any_btl"]).all()
    arms = loop["fb"]["arms"]
    assert set(arms) == {"cg", "free", "cstr", "llm"} and "uplift_vs_control" in arms["cstr"]


def test_next_week_trains_on_the_campaign_log_with_the_logged_propensity(loop):
    """H5: label source = campaign log; the propensity is the logged one, not a share."""
    assert loop["t1"]["labels"]["label_source"] == "targeting"
    lab = loop["t2"]["labels"]
    assert lab["label_source"] == "campaign_log" and lab["labelled_rows"] > 50
    train = pd.read_json(loop["d2"] / "outputs" / "nba_training_set.json")
    log = campaign_log.read_log(loop["root"])
    logged = dict(zip(log["msisdn"].astype(str), log["propensity"]))
    got = train.set_index(train["access_method_val"].astype(str))["propensity"]
    common = [v for v in got.index if v in logged]
    assert common and all(abs(got[v] - max(min(logged[v], 1.0), config.PROPENSITY_CLIP[0])) < 1e-6 for v in common)


def test_last_campaign_features_are_model_features_and_reflect_the_log(loop):
    from nbx.fsdr.features import model_features
    feats = model_features()
    assert set(config.LAST_CAMPAIGN_NUMERIC) <= set(feats) and "PREV_ACTION" in config.CATEGORICAL_FEATURES
    lc = campaign_log.last_campaign_features(loop["root"], pd.Series(campaign_log.read_log(loop["root"])["msisdn"]), W2)
    sent = set(campaign_log.read_log(loop["root"], arm="cstr")["msisdn"].astype(str))
    hit = lc[lc["ACCESS_METHOD_VAL"].isin(sent)]
    assert (hit["NB_CONTACTS_LAST_4W"] >= 1).all() and (hit["PREV_ACTION"] != config.PREV_ACTION_NONE).all()
    assert lc["DAYS_SINCE_LAST_BTL"].between(0, -config.INTERACTION_WINDOW[0]).all()


def test_contacts_in_the_last_28_days_are_counted_from_the_log(tmp_path):
    """L4 input: 3 contacts logged in D-28..D-1 (control rows and older contacts are not counted)."""
    root = tmp_path / "state" / "campaign"
    rows = [{"run_date": (W2 - timedelta(days=7 * k)).isoformat(), "arm": "cstr", "msisdn": "70000001", "action": "UPSELL",
             "offers": [], "policy_source": "bandit", "propensity": 0.8} for k in (1, 2, 3, 5)]
    rows += [{"run_date": (W2 - timedelta(days=7)).isoformat(), "arm": "cg", "msisdn": "70000002", "action": config.NO_CONTACT}]
    campaign_log.append_log(root, rows)
    lc = campaign_log.last_campaign_features(root, pd.Series(["70000001", "70000002"]), W2).set_index("ACCESS_METHOD_VAL")
    assert lc.at["70000001", "NB_CONTACTS_LAST_4W"] == 3 and lc.at["70000001", "PREV_ACTION"] == "UPSELL"
    assert lc.at["70000002", "NB_CONTACTS_LAST_4W"] == 0 and lc.at["70000002", "PREV_ACTION"] == config.PREV_ACTION_NONE


def test_contact_policy_blocks_at_the_cap(workspace):
    """L4: a line at the 28-day cap cannot be contacted, whatever the LLM proposes."""
    from nbx.fsdr.agents.recommender_agent import guard_action
    from nbx.fsdr.agents.tools import FSDR_TOOLS
    v = next(m for m in workspace.lines("llm") if FSDR_TOOLS["contact_policy"].fn(workspace.context(m), {})["can_contact"])
    ctx = workspace.context(v)
    workspace.features.loc[v, "NB_CONTACTS_LAST_4W"] = config.CONTACT_CAP_28D
    try:
        cp = FSDR_TOOLS["contact_policy"].fn(ctx, {})
        assert not cp["can_contact"] and any("28 days" in r for r in cp["blocked_reasons"])
        action, findings = guard_action(ctx, "UPSELL DATA")
        assert action == config.NO_CONTACT and findings
    finally:
        workspace.features.loc[v, "NB_CONTACTS_LAST_4W"] = 0.0


def test_anchor_growth_guard_caps_week_over_week_growth():
    """M4: anchor at most (1 + g) x last logged anchor; data / voice follow; unlogged lines untouched."""
    a = pd.DataFrame({"AVG_REV_MONTHLY": [10.0, 5.0, 8.0], "AVG_REV_WEEKLY": [3.0, 1.0, 2.0]}, index=["a", "b", "c"])
    for p in ("MONTHLY", "WEEKLY"):
        a[f"AVG_REV_DATA_{p}"], a[f"AVG_REV_VOICE_{p}"] = 0.9 * a[f"AVG_REV_{p}"], 0.1 * a[f"AVG_REV_{p}"]
    out = cap_anchor_growth(a, {"a": {"monthly": 6.0, "weekly": 3.0}, "b": {"monthly": 6.0, "weekly": 0.5}}, 0.10)
    assert out.at["a", "AVG_REV_MONTHLY"] == pytest.approx(6.6) and out.at["a", "AVG_REV_DATA_MONTHLY"] == pytest.approx(5.94)
    assert out.at["a", "AVG_REV_WEEKLY"] == 3.0                               # below the cap: unchanged
    assert out.at["b", "AVG_REV_MONTHLY"] == 5.0 and out.at["b", "AVG_REV_WEEKLY"] == pytest.approx(0.55)
    assert out.at["c", "AVG_REV_MONTHLY"] == 8.0 and not out.at["c", "ANCHOR_CAPPED"]
    assert bool(out.at["a", "ANCHOR_CAPPED"]) and bool(out.at["b", "ANCHOR_CAPPED"])
