"""Features: VW format, parity with the reference nba_lib, train = score path, registry-driven."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbx.core import vw_format
from nbx.fsdr import config, schemas
from nbx.fsdr.features import (build_feature_frame, build_features, eligible_base, load_model_scores, load_party,
                               load_subs, load_usage, norm_val, numeric_features)

REF = Path(__file__).resolve().parents[2] / "docs" / "fsdr_reference" / "nba_lib.py"


def _nba_lib():
    """Imports the reference implementation with Dataiku stubbed out."""
    sys.modules.setdefault("dataiku", types.ModuleType("dataiku"))
    spec = importlib.util.spec_from_file_location("nba_lib_reference", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vw_format_writes_numbers_as_name_colon_value_and_tokens_without_spaces():
    df = pd.DataFrame({"PROBA_CHURN": [0.25], "VAR_REV_W1_W2": [0.0], "NATIONALITY_GROUP": ["Sri Lanka"]})
    line = vw_format.shared_lines(df, ["PROBA_CHURN", "VAR_REV_W1_W2"], ["NATIONALITY_GROUP"])[0]
    assert "proba_churn:0.25" in line and "=0.25" not in line
    assert "var_rev_w1_w2" not in line                         # zero omitted
    assert "nationality_group=Sri_Lanka" in line
    ex = vw_format.make_example(line, config.ACTIONS, (2, -1, 0.5))
    assert ex[1:] [2].startswith("0:-1:0.500000 |Action product=RETAIN")
    assert all(" " not in l.split("product=")[1] for l in ex[1:])
    assert [l.split("product=")[1].split()[0] for l in ex[1:]] == [vw_format.action_token(a) for a in config.ACTIONS]


def test_action_order_is_the_reference_order():
    assert config.ACTIONS == _nba_lib().ACTIONS


def test_feature_names_and_values_match_the_reference_nba_lib(small_source):
    ref = _nba_lib()
    assert numeric_features() == ref.NUMERIC_FEATURES
    base, _ = eligible_base(small_source)
    pop = base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]]
    party, (scores, _) = load_party(small_source, "party_auto"), load_model_scores(small_source)
    usage, subs = load_usage(small_source, "T"), load_subs(small_source, "T")
    ours = build_features(pop, party, scores, usage, subs, seed=5)

    def as_int(df):
        df = df.copy()
        df["ACCESS_METHOD_VAL"] = pd.to_numeric(df["ACCESS_METHOD_VAL"]).astype("Int64")
        return df

    theirs = ref.build_features(as_int(pop), party, as_int(scores), as_int(usage), as_int(subs), seed=5)
    theirs["ACCESS_METHOD_VAL"] = theirs["ACCESS_METHOD_VAL"].astype(str)
    theirs = theirs.set_index("ACCESS_METHOD_VAL").loc[ours["ACCESS_METHOD_VAL"]]
    for f in ref.NUMERIC_FEATURES:
        assert np.allclose(ours[f].to_numpy(float), theirs[f].to_numpy(float), equal_nan=True), f
    assert (ours["NATIONALITY_GROUP"].to_numpy() == theirs["NATIONALITY_GROUP"].to_numpy()).all()


def test_train_and_score_paths_use_the_same_function(small_source):
    base, _ = eligible_base(small_source)
    pop = base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]].head(100)
    a, _ = build_feature_frame(small_source, pop, "T", seed=1)
    b, _ = build_feature_frame(small_source, pop, "T", seed=1)
    assert a.equals(b)
    d, rep = build_feature_frame(small_source, pop, "D", seed=1)
    assert list(d.columns) == list(a.columns) and rep["anchor"] == "D"
    assert not np.allclose(a["VAR_REV_W1_W2"], d["VAR_REV_W1_W2"])      # different windows, same code


def test_missing_scores_get_one_random_value_per_row(small_source):
    base, _ = eligible_base(small_source)
    feat, _ = build_feature_frame(small_source, base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]], "T", seed=2)
    scored = set(norm_val(small_source.read("CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring")["access_method_val"]))
    filled = feat.loc[~feat["ACCESS_METHOD_VAL"].isin(scored), "PROBA_CHURN_1D"]
    assert len(filled) > 10 and filled.between(0, 0.02).all() and filled.nunique() == len(filled)


def test_trends_go_from_older_to_recent():
    pop = pd.DataFrame({"ACCESS_METHOD_VAL": ["70000001"], "PARTY_ID": [1]})
    usage = pd.DataFrame({"ACCESS_METHOD_VAL": ["70000001"], "LOCAL_VOICE_MIN_W1": [150.0], "LOCAL_VOICE_MIN_W2": [100.0]})
    f = build_features(pop, pd.DataFrame({"PARTY_ID": [1], "NATIONALITY": ["Oman"]}), pd.DataFrame(columns=["ACCESS_METHOD_VAL"]),
                       usage, pd.DataFrame(columns=["ACCESS_METHOD_VAL"]))
    assert f["VAR_VOICE_W1_W2"].iat[0] == pytest.approx(50.0)           # +50 %: W1 (recent) vs W2 (older)
    assert f["VOICE_W2_ZERO"].iat[0] == 0 and f["VOICE_W3_ZERO"].iat[0] == 1


def test_eligible_base_applies_br02_br08_br10():
    from nbx.fsdr.datasource import SyntheticSource

    src = SyntheticSource(n=600, scenarios="data_quality_issues", seed=3)
    base, rep = eligible_base(src)
    assert base["ACCESS_METHOD_VAL"].is_unique and rep["duplicates_removed"] > 0
    assert not base["CURRENT_PLAN"].isin(config.EXCLUDED_PLANS).any()


DUMMY = config.ModelScore("DAIM_DUMMY_TWELFTH_MODEL.prediction", "PROBA_DUMMY", "max", "dummy_signal")


def test_a_twelfth_model_reaches_the_features_the_vw_line_and_the_compatibility_check(monkeypatch, small_source):
    from nbx.fsdr import bandit
    from nbx.fsdr.datasource import SyntheticSource

    old_features = numeric_features()
    monkeypatch.setattr(config, "MODEL_SCORE_REGISTRY", config.MODEL_SCORE_REGISTRY + [DUMMY])
    schemas.refresh()
    try:
        feats = numeric_features()
        assert "PROBA_DUMMY" in feats and len(feats) == len(old_features) + 1
        src = SyntheticSource(n=300, seed=4)
        base, _ = eligible_base(src)
        feat, _ = build_feature_frame(src, base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]], "T")
        assert feat["PROBA_DUMMY"].notna().all()
        assert any("proba_dummy:" in s for s in bandit.shared(feat, feats))
        with pytest.raises(bandit.IncompatibleModel):
            bandit.check_compatible({"actions": config.ACTIONS, "features": old_features}, feats)
    finally:
        monkeypatch.undo()
        schemas.refresh()
