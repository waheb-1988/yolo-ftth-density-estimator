"""Labels (BR-36..45) and the VW bandit (order, save/load, compatibility, modes)."""
from __future__ import annotations

import numpy as np
import pytest

from nbx.fsdr import bandit, config
from nbx.fsdr.features import build_feature_frame, eligible_base, numeric_features
from nbx.fsdr.labels import build_labels, segment_action

from .conftest import RUN_DATE


@pytest.fixture(scope="module")
def labelled(small_source):
    base, _ = eligible_base(small_source)
    feat, _ = build_feature_frame(small_source, base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]], "T")
    train, rep = build_labels(small_source, feat, RUN_DATE)
    return feat, train, rep


def test_segment_mapping_is_case_insensitive_and_follows_br39():
    assert segment_action("Rgs_Not_At_Risk_Dualplay_Upgrade") == segment_action("RGS_Not_At_Risk_DualPlay_Upgrade") == "MAINTAIN"
    assert segment_action("Retention_High_Probability_To_Reconnect") == "RETAIN"          # plan Q1 default
    assert segment_action("RGS_Not_At_Risk_Data_Only_Voice_Appetency") == "UPSELL VOICE"  # plan Q1b default
    assert segment_action("Nursering_Early_Churn_At_Risk") is None and segment_action(None) is None


def test_labels_exclude_nursering_and_keep_one_row_per_line(labelled, small_source):
    _, train, rep = labelled
    assert train["ACCESS_METHOD_VAL"].is_unique
    assert set(train["ACTION"]) <= set(config.ACTIONS)
    assert not train["SEGMENT"].str.lower().isin(config.EXCLUDED_SEGMENTS).any()
    raw = small_source.read("targeting_dataiku_auto")
    assert rep["no_segment_dropped"] == int(raw["segment"].isna().sum())
    assert rep["nursering_excluded"] > 0


def test_reward_definitions_are_nested_and_cost_is_minus_reward(labelled):
    _, train, rep = labelled
    assert (train["REWARD_EXACT_OFFER"] <= train["REWARD_SAME_SERVICE"]).all()
    assert (train["REWARD_SAME_SERVICE"] <= train["REWARD_ANY_BTL"]).all()
    assert (train["COST"] == -train["REWARD"]).all() and train["REWARD"].equals(train["REWARD_ANY_BTL"])
    assert train["PROPENSITY"].between(*config.PROPENSITY_CLIP).all()
    assert 0 < rep["take_rate"]["any_btl"] < 0.5


def test_default_reward_counts_any_btl_in_the_window_and_the_option_only_removes_takers(small_source, labelled):
    """BR-40: the default counts any catalogue BTL in D-7..D-1, whatever the targeting date (production)."""
    assert config.REWARD_AFTER_TARGETING is False
    feat, train, _ = labelled
    loose, _ = build_labels(small_source, feat, RUN_DATE, reward_after_targeting=False)
    strict, _ = build_labels(small_source, feat, RUN_DATE, reward_after_targeting=True)
    assert train["REWARD"].equals(loose["REWARD"])
    assert strict["REWARD"].sum() <= loose["REWARD"].sum()


def test_bandit_trains_scores_and_round_trips(labelled, tmp_path):
    feat, train, _ = labelled
    features = numeric_features()
    vw, rep = bandit.train(train, features, passes=1)
    pmf = bandit.predict_pmfs(vw, feat.head(30), features)
    assert pmf.shape == (30, len(config.ACTIONS)) and np.allclose(pmf.sum(axis=1), 1)
    folder = tmp_path / "model with space"
    bandit.save(vw, {"vw_args": bandit.vw_args(), "actions": config.ACTIONS, "features": features}, folder)
    vw2, _ = bandit.load(folder, features)
    assert np.allclose(pmf, bandit.predict_pmfs(vw2, feat.head(30), features))
    s = bandit.score_population(vw, feat.head(50), features, policy="sample", seed=1)
    g = bandit.score_population(vw, feat.head(50), features, policy="greedy", seed=1)
    assert (g["ACTION"] == g["ACTION_GREEDY"]).all() and set(s["ACTION"]) <= set(config.ACTIONS)
    assert s.equals(bandit.score_population(vw, feat.head(50), features, policy="sample", seed=1))


def test_incompatible_models_are_refused():
    features = numeric_features()
    with pytest.raises(bandit.IncompatibleModel):
        bandit.check_compatible({"actions": list(reversed(config.ACTIONS)), "features": features}, features)
    with pytest.raises(bandit.IncompatibleModel):
        bandit.check_compatible({"actions": config.ACTIONS, "features": features[:-1]}, features)


def test_simulation_mode_and_epsilon_range(labelled):
    _, train, _ = labelled
    vw, rep = bandit.train(train, numeric_features(), mode="simulation", sim_iterations=500)
    assert rep["iterations"] == 500 and 0 <= rep["match_rate"] <= 1
    with pytest.raises(ValueError):
        bandit.train(train, numeric_features(), epsilon=0.9)


def test_access_method_map_uses_the_msisdn_as_is_contract_has_no_prefix(small_source):
    """Production SQL builds access_method_id_auto with `ACCESS_METHOD_VAL as MSISDN` (no country prefix)."""
    from nbx.fsdr.labels import access_method_map
    am = small_source.read("access_method_id_auto")
    m = access_method_map(small_source)
    assert set(m["ACCESS_METHOD_VAL"]) == set(am["MSISDN"].astype(str))
    assert m["ACCESS_METHOD_VAL"].str.fullmatch(r"\d{8}").all()
