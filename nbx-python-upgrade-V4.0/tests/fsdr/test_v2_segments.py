"""V2.0 Phase 3: features B-01 … B-12 (T2), S-rules, bands, micro-segments, action_original / action_agent (T3).
Golden cases of master prompt §23 (Features; Segmentation, age and tenure; Segment-action agent)."""
from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbx.fsdr import base_features as bf
from nbx.fsdr import config
from nbx.fsdr.contract import data_agent
from nbx.fsdr.features import add_trends
from nbx.fsdr.pipelines import training
from nbx.fsdr.segmentation import actions, bands, rules
from nbx.fsdr.steps import read_step

from .conftest import RUN_DATE


class FakeCache:
    """Just enough of CacheSource for one partition: has / read / dir."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables
        self.dir = Path(f"fake-{uuid.uuid4().hex}")

    def has(self, t):
        return t in self.tables

    def read(self, t, part=None):
        return self.tables[t].copy()


def _base(n=2):
    return pd.DataFrame({"ACCESS_METHOD_VAL": [f"9100000{i}" for i in range(n)], "ACCESS_METHOD_ID": list(range(1, n + 1)),
                         "ACCOUNT_ID": list(range(11, 11 + n)), "PARTY_ID": list(range(21, 21 + n)),
                         "access_method_subs_start_dt": ["2026-01-01"] * n, "CURRENT_PLAN": ["FSDR Prepaid"] * n})


# ---------------------------------------------------------------------------
# Features (B-04 … B-08)
# ---------------------------------------------------------------------------
def test_trend_golden_case_older_to_recent():
    df = pd.DataFrame({"REV_W1": [2.1], "REV_W2": [3.5], "DATA_VOLUME_MB_W1": [0.0], "DATA_VOLUME_MB_W2": [850.0],
                       "LOCAL_VOICE_MIN_W1": [40.0], "LOCAL_VOICE_MIN_W2": [0.0]})
    for k in ("REV", "DATA_VOLUME_MB", "LOCAL_VOICE_MIN"):
        for p in ("W3", "W4", "M1", "M2", "M3"):
            df[f"{k}_{p}"] = 1.0
    add_trends(df)
    r = df.iloc[0]
    assert r["VAR_REV_W1_W2"] == pytest.approx(-40.0)
    assert r["LOG_REV_W1_W2"] == pytest.approx(-0.3726753, abs=1e-7)
    assert r["VAR_DATA_W1_W2"] == pytest.approx(-100.0)
    assert r["LOG_DATA_W1_W2"] == pytest.approx(-6.7464121, abs=1e-7)
    assert r["VOICE_W2_ZERO"] == 1 and r["VAR_VOICE_W1_W2"] == 0.0
    assert r["LOG_VOICE_W1_W2"] == pytest.approx(3.7135721, abs=1e-7)


def test_churn_is_the_first_row_of_the_stack_and_other_scores_the_max():
    reg = config.MODEL_SCORE_REGISTRY
    churn = [m.table for m in reg if m.feature == "PROBA_CHURN"]
    react = [m.table for m in reg if m.feature == "PROBA_REACT_30"][0]
    t = {churn[0]: pd.DataFrame({"access_method_val": ["91000000"], "proba_1": [0.2]}),
         churn[1]: pd.DataFrame({"access_method_val": ["91000000", "91000001"], "proba_1": [0.9, 0.7]}),
         react: pd.DataFrame({"access_method_val": ["91000000", "91000000"], "proba_1": [0.1, 0.4]})}
    out = bf.model_scores(FakeCache(t), 0).set_index("ACCESS_METHOD_VAL")
    assert out.loc["91000000", "PROBA_CHURN"] == 0.2              # global table first (V1.0 took the max: 0.9)
    assert out.loc["91000001", "PROBA_CHURN"] == 0.7
    assert out.loc["91000000", "PROBA_REACT_30"] == 0.4


def test_profile_overrides_party_age_and_activation_tenure():
    party = pd.DataFrame({"party_id": [21, 22], "AGE": [50, 33], "NATIONALITY": ["Oman", "Kenya"]})
    prof = pd.DataFrame({"msisdn": ["91000000"], "age": [19], "tenure": [40]})
    cache = FakeCache({"base_prepaid": _base(), "party_auto": party, config.PROFILE_TABLE: prof})
    df, stats = bf.partition_features(cache, 0, date(2026, 10, 22), "T")
    df = df.set_index("ACCESS_METHOD_VAL")
    assert df.loc["91000000", "AGE"] == 19 and df.loc["91000000", "AGE_SOURCE"] == "profile"
    assert df.loc["91000000", "TENURE_DAYS"] == 40 and df.loc["91000000", "TENURE_SOURCE"] == "profile"
    assert df.loc["91000001", "AGE"] == 33 and df.loc["91000001", "AGE_SOURCE"] == "party"
    assert df.loc["91000001", "TENURE_DAYS"] == (date(2026, 10, 22) - date(2026, 1, 1)).days
    assert df.loc["91000001", "NATIONALITY_GROUP"] == "others"                     # B-08
    assert stats["age_source"] == {"profile": 1, "party": 1}
    for p in ("W1", "M1"):                                                         # B-05 usage 0, B-06 revenue
        assert df.loc["91000000", f"REV_{p}"] == 0.0


def test_missing_score_is_flagged_and_imputed_with_the_nationality_median():
    df = pd.DataFrame({"NATIONALITY_GROUP": ["Oman", "Oman", "India", "India"], "PROBA_CHURN": [0.1, np.nan, 0.8, np.nan]})
    for f in config.proba_features():
        if f != "PROBA_CHURN":
            df[f] = 0.5
    med = {"Oman": {"PROBA_CHURN": 0.3}, "India": {"PROBA_CHURN": None}, "_global": {"PROBA_CHURN": 0.45}}
    bf.impute(df, med)
    assert df["PROBA_CHURN"].tolist() == [0.1, 0.3, 0.8, 0.45]                     # India: no median -> global


# ---------------------------------------------------------------------------
# S-rules (§7)
# ---------------------------------------------------------------------------
def _feat(**over) -> dict:
    row = {"TENURE_DAYS": 400, "ARPU_M1": 5.0, "ACTIVE_DAYS_COUNT_M1": 20, "LOCAL_VOICE_MIN_M1": 30.0,
           "DATA_VOLUME_MB_M1": 500.0, "VAR_REV_M1_M2": 0.0}
    for f in config.proba_features():
        row[f] = 0.1
        row[f + config.MISSING_FLAG_SUFFIX] = 0
    row.update(over)
    return row


S_CASES = [
    ("S-1", _feat(TENURE_DAYS=40, PROBA_CHURN=0.7)), ("S-2", _feat(TENURE_DAYS=44)),
    ("S-3", _feat(TENURE_DAYS=60, PROBA_CHURN=0.5)), ("S-4", _feat(TENURE_DAYS=89)),
    ("S-5", _feat(ARPU_M1=0.0, PROBA_REACT_15=0.6)), ("S-6", _feat(ACTIVE_DAYS_COUNT_M1=0, PROBA_REACT_30=0.3)),
    ("S-7", _feat(ARPU_M1=0.0)), ("S-8", _feat(PROBA_CHURN_1D=0.55)), ("S-9", _feat(VAR_REV_M1_M2=-30.0)),
    ("S-9", _feat(PROBA_FRAG_VOICE=0.5)), ("S-10", _feat()),
    ("S-11", _feat(LOCAL_VOICE_MIN_M1=0.5, PROBA_APP_VOICE=0.5)), ("S-12", _feat(LOCAL_VOICE_MIN_M1=0.0)),
    ("S-13", _feat(DATA_VOLUME_MB_M1=9.9, PROBA_APP_DATA=0.5)), ("S-14", _feat(DATA_VOLUME_MB_M1=0.0)),
    ("S-14", _feat(DATA_VOLUME_MB_M1=0.0, LOCAL_VOICE_MIN_M1=0.0)),                # no usage -> voice-only branch
    ("S-10", _feat(TENURE_DAYS=90)), ("S-10", _feat(TENURE_DAYS=np.nan)),          # unknown tenure is not "new"
]


@pytest.mark.parametrize("rule_id,row", S_CASES)
def test_s_rule_cases_first_match_wins(rule_id, row):
    out = rules.apply_s_rules(pd.DataFrame([row]))
    assert out["RULE_ID"].iloc[0] == rule_id
    expected = dict((r, (s, a)) for r, s, a in config.S_RULES)[rule_id]
    assert (out["SEGMENT"].iloc[0], out["RULE_ACTION"].iloc[0]) == expected
    assert out["EXCLUDED"].iloc[0] == int(rule_id in ("S-1", "S-2"))


def test_a_missing_score_never_triggers_at_risk():
    imputed_high = _feat(TENURE_DAYS=30, PROBA_CHURN=0.9, PROBA_CHURN_MISSING=1)
    assert rules.apply_s_rules(pd.DataFrame([imputed_high]))["RULE_ID"].iloc[0] == "S-2"
    rgs = _feat(PROBA_CHURN=0.9, PROBA_CHURN_MISSING=1, PROBA_FRAG_DATA=0.8, PROBA_FRAG_DATA_MISSING=1)
    assert rules.apply_s_rules(pd.DataFrame([rgs]))["RULE_ID"].iloc[0] == "S-10"


# ---------------------------------------------------------------------------
# Bands and micro-segments (§8)
# ---------------------------------------------------------------------------
def test_age_bands_golden():
    ages = pd.Series([19, 40, 59, 60, None, 120, 14, 100])
    assert list(bands.age_band(ages)) == ["young", "middle", "middle", "senior", "unknown", "unknown", "unknown", "senior"]
    assert list(bands.young_split(pd.Series([19, 24, 25, 39, 40]))) == ["lt25", "lt25", "25_39", "25_39", None]


def test_tenure_bands_golden():
    days = pd.Series([40, 89, 90, 182, 183, 365, 366, 730, 731, -5, None])
    assert list(bands.tenure_band(days)) == ["t0_lt3m", "t0_lt3m", "t1_3_6m", "t1_3_6m", "t2_6_12m", "t2_6_12m",
                                             "t3_1_2y", "t3_1_2y", "t4_gt2y", "unknown", "unknown"]
    assert rules.apply_s_rules(pd.DataFrame([_feat(TENURE_DAYS=40)]))["SEGMENT"].iloc[0].startswith("Nursering")


def test_micro_segment_id_and_the_120_cells():
    nat = pd.Series(["Oman", "India"])
    ms = bands.micro_segment(nat, bands.age_band(pd.Series([30, 70])), bands.tenure_band(pd.Series([100, 800])))
    assert list(ms) == ["Oman|young|t1_3_6m", "India|senior|t4_gt2y"]
    assert len(bands.all_micro_segments()) == 120 == 5 * 4 * 6


def test_value_band_and_usage_type():
    assert list(bands.value_band(pd.Series([0, 1, 5, 9]), [2.0, 6.0])) == ["zero", "low", "mid", "high"]
    assert list(bands.usage_type(pd.Series([1, 0, 1, 0]), pd.Series([1, 1, 0, 0]))) == ["dual", "data_only", "voice_only", "none"]


# ---------------------------------------------------------------------------
# action_original, crosswalk, agent dictionary (§7.1)
# ---------------------------------------------------------------------------
def test_original_dictionary_last_value_wins_and_is_case_sensitive():
    d = config.DICT_SEGM_ACTIONS
    assert len(config.DICT_SEGM_ACTIONS_SOURCE) == 30 and len(d) == 19
    assert d["Retention_High_Probability_To_Reconnect"] == "RETAIN"                # first UPSELL, last RETAIN
    assert d["RGS_Not_At_Risk_Data_Only_Voice_Appetency"] == "UPSELL VOICE"
    assert d["Rgs_Not_At_Risk_Data_Only_Voice_Appetency"] == "CROSS SELL VOICE"
    assert d["Retention_RGS_At_Risk"] == "RETAIN" and "Nursering_Early_Churn_At_Risk" not in d
    assert set(d.values()) == set(config.ACTIONS)


def test_crosswalk_covers_every_s_segment_and_is_marked_check():
    cw = actions.crosswalk_table()
    assert list(cw["s_segment"]) == config.S_SEGMENTS and (cw["check"] == "CHECK").all()
    assert set(cw["original_segment"]) - {"Nursering_Early_Churn_At_Risk", "Nursering_Early_Churn_Not_At_Risk"} <= set(config.DICT_SEGM_ACTIONS)


def test_engine_segment_overrides_the_crosswalk_and_the_source_is_logged():
    seg = pd.DataFrame({"SEGMENT": ["Retention_RGS_At_Risk", "Data_Only_Voice_Appetency", "RGS_Not_At_Risk_DualPlay_Upgrade",
                                    "RGS_Not_At_Risk_DualPlay_Upgrade", "Retention_RGS_Fragile"],
                        "W0_SEGMENT": [None, "Rgs_Not_At_Risk_Data_Only_Voice_Appetency", "unknown_spelling",
                                       "Nursering_Early_Churn_At_Risk", "Retention_Rgs_At_Risk"]})
    out = actions.apply_original(seg)
    assert list(out["ACTION_ORIGINAL"]) == ["RETAIN", "CROSS SELL VOICE", "MAINTAIN", None, "RETAIN"]
    assert list(out["ACTION_ORIGINAL_SOURCE"]) == ["crosswalk", "segment_engine", "crosswalk", "segment_engine", "segment_engine"]


def test_week0_agent_dictionary_equals_the_original_for_every_cell():
    d = actions.initial_agent_dictionary(RUN_DATE)
    assert len(d["cells"]) == 70 and d["name"] == "dict_segm_actions_agent_v0"
    seg = pd.DataFrame([(s, n) for s in config.S_SEGMENTS for n in config.NATIONALITY_GROUPS],
                       columns=["SEGMENT", "NATIONALITY_GROUP"]).assign(W0_SEGMENT=None)
    out = actions.apply_agent_dictionary(actions.apply_original(seg), d)
    assert out["ACTION_AGENT"].equals(out["ACTION_ORIGINAL"])
    valid, problems = actions.validate_agent_dictionary(d)
    assert problems == [] and valid["cells"] == d["cells"]


def test_invalid_llm_dictionary_cells_are_rejected_and_the_previous_kept():
    prev = actions.initial_agent_dictionary(RUN_DATE)
    fake = json.loads(json.dumps(prev))
    fake.update(version=1, name="dict_segm_actions_agent_v1")
    edits = {("Retention_RGS_At_Risk", "Oman"): [{"action": "UPSELL", "rank": 1}],               # H-06
             ("Nursering_Early_Churn_At_Risk", "India"): [{"action": "RETAIN", "rank": 1}],      # B-02
             ("Voice_Only_Data_Appetency", "others"): [{"action": "FREE MONEY", "rank": 1}],     # invented id
             ("RGS_Not_At_Risk_DualPlay_Upgrade", "Pakistan"): [{"action": "UPSELL", "rank": 1},
                                                                {"action": "RETAIN", "rank": 2}]}  # valid change
    for c in fake["cells"]:
        if (c["segment"], c["nationality_group"]) in edits:
            c["actions"] = edits[(c["segment"], c["nationality_group"])]
    valid, problems = actions.validate_agent_dictionary(fake, prev)
    assert {(p["segment"], p["nationality_group"]) for p in problems} == set(list(edits)[:3])
    lk = actions.agent_lookup(valid).set_index(["SEGMENT", "NATIONALITY_GROUP"])
    assert lk.loc[("Retention_RGS_At_Risk", "Oman"), "ACTION_AGENT"] == "RETAIN"
    assert lk.loc[("Nursering_Early_Churn_At_Risk", "India"), "ACTION_AGENT"] is None
    assert lk.loc[("Voice_Only_Data_Appetency", "others"), "ACTION_AGENT"] == "UPSELL DATA"
    assert tuple(lk.loc[("RGS_Not_At_Risk_DualPlay_Upgrade", "Pakistan")]) == ("UPSELL", "RETAIN")
    assert lk.loc[("Onboarding_Early_Churn_At_Risk", "Oman"), "ACTION_AGENT"] == "UPSELL"   # its original: allowed


def test_action_agreement_values():
    o = pd.Series(["RETAIN", "RETAIN", "UPSELL", "MAINTAIN"])
    a = pd.Series(["RETAIN", "UPSELL", "UPSELL", "RETAIN"])
    s = pd.Series(["RETAIN", "RETAIN", "MAINTAIN", "RETAIN"])
    assert list(actions.action_agreement(o, a, s)) == ["all_same", "sent_original", "other", "sent_agent"]


# ---------------------------------------------------------------------------
# T2 + T3 end to end on a generated run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def t3_run(small_folder, tmp_path_factory):
    root = tmp_path_factory.mktemp("v2 runs")
    runs, cache = root / "runs", root / "cache"
    t0 = data_agent.run_t0(small_folder, runs_root=runs, cache_root=cache)
    assert t0["status"] == "done"
    rep = training.run_features(RUN_DATE, runs_root=runs, cache_root=cache, workers=1)
    return {"runs": runs, "cache": cache, "rep": rep, "train": runs / RUN_DATE.isoformat() / "train"}


def test_t2_writes_one_feature_row_per_eligible_line(t3_run):
    t2 = t3_run["rep"]["T2_features"]
    d = t3_run["train"] / "T2_features"
    feat = pd.concat([pd.read_parquet(p) for p in sorted((d / "base_features").glob("part-*.parquet"))])
    assert len(feat) == t2["lines"] == feat["ACCESS_METHOD_VAL"].nunique() and t2["anchor"] == "T"
    assert not feat[config.proba_features()].isna().any().any()                    # imputed
    flags = [f + config.MISSING_FLAG_SUFFIX for f in config.proba_features()]
    assert feat[flags].sum().sum() == sum(t2["missing_scores"].values())
    for f in ("feature_summary.csv", "base_features_preview.csv", "stats.json", "summary.json"):
        assert (d / f).exists(), f
    assert read_step(d)["status"] == "done"


def test_t3_segments_reports_and_two_actions(t3_run):
    t3 = t3_run["rep"]["T3_segments"]
    d = t3_run["train"] / "T3_segments"
    seg = pd.concat([pd.read_parquet(p) for p in sorted((d / "segments").glob("part-*.parquet"))])
    assert len(seg) == t3["lines"] == t3_run["rep"]["T2_features"]["lines"]
    assert list(seg.columns) == training.SEGMENT_COLUMNS
    assert set(seg["SEGMENT"]) <= set(config.S_SEGMENTS)
    assert seg.loc[seg["EXCLUDED"] == 1, "SEGMENT"].str.startswith("Nursering").all()
    assert (seg["MICRO_SEGMENT"] == seg["NATIONALITY_GROUP"] + "|" + seg["AGE_BAND"] + "|" + seg["TENURE_BAND"]).all()
    micro = pd.read_csv(d / "microsegment_counts.csv")
    assert len(micro) == 120 and micro["lines"].sum() == len(seg)
    assert pd.read_csv(d / "segment_counts.csv")["lines"].sum() == len(seg)
    engine = seg["ACTION_ORIGINAL_SOURCE"] == "segment_engine"
    assert engine.sum() == seg["W0_SEGMENT"].isin(list(config.DICT_SEGM_ACTIONS)).sum() + \
        seg["W0_SEGMENT"].astype("string").str.lower().str.startswith("nursering").fillna(False).sum()
    assert (seg["ACTION_AGENT_VERSION"] == "dict_segm_actions_agent_v0").all()
    priors = json.loads((d / "priors.json").read_text(encoding="utf-8"))
    assert sum(priors["recomputed_week0"].values()) == pytest.approx(1.0, abs=1e-5)
    assert list(priors["recomputed_week0"]) == config.ACTIONS
    assert len(pd.read_csv(d / "crosswalk.csv")) == 14


def test_features_resume_and_rerun_when_forced(t3_run):
    again = training.run_features(RUN_DATE, runs_root=t3_run["runs"], cache_root=t3_run["cache"], workers=1)
    assert again["T2_features"].get("skipped") and again["T3_segments"].get("skipped")
    forced = training.run_t3(RUN_DATE, runs_root=t3_run["runs"], cache_root=t3_run["cache"], workers=1, force=True)
    assert not forced.get("skipped") and forced["lines"] == t3_run["rep"]["T3_segments"]["lines"]


def test_the_next_week_carries_the_previous_agent_dictionary(t3_run):
    d, path = actions.previous_agent_dictionary(t3_run["runs"], date(2026, 10, 28))
    assert d["name"] == "dict_segm_actions_agent_v0" and path.parent.name == "T3_segments"
    assert actions.previous_agent_dictionary(t3_run["runs"], RUN_DATE) == (None, None)


def test_t3_refuses_to_start_without_t2(tmp_path, t3_run):
    with pytest.raises(training.StageBlocked):
        training.run_t3(date(2026, 10, 29), runs_root=tmp_path, cache_root=t3_run["cache"])


def test_partitions_in_parallel_give_the_same_segments(t3_run, tmp_path):
    """2 worker processes write exactly the files 1 process wrote (determinism across the pool)."""
    import shutil

    runs = tmp_path / "runs"
    shutil.copytree(t3_run["runs"] / RUN_DATE.isoformat() / "train" / "T0_load", runs / RUN_DATE.isoformat() / "train" / "T0_load")
    training.run_features(RUN_DATE, runs_root=runs, cache_root=t3_run["cache"], workers=2)
    for stage, sub in (("T2_features", "base_features"), ("T3_segments", "segments")):
        for p in sorted((t3_run["train"] / stage / sub).glob("part-*.parquet")):
            a, b = pd.read_parquet(p), pd.read_parquet(runs / RUN_DATE.isoformat() / "train" / stage / sub / p.name)
            pd.testing.assert_frame_equal(a, b, obj=f"{stage}/{p.name}")
