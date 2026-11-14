"""Generator realism and window consistency."""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.synth import generator
from nbx.fsdr.synth.world import World


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def test_activities_m1_equals_the_four_weeks(small_source):
    for anchor in ("auto", "score"):
        name = lambda p: "activities_auto_m2" if (p == "m2" and anchor == "auto") else f"activities_{p}_{anchor}"  # noqa: E731
        m1 = small_source.read(name("m1")).set_index("ACCESS_METHOD_VAL")
        for metric in ("Local_Voice_min", "Data_Volume_MB", "Call_Count", "Active_Days_Count"):
            weeks = None
            for i in range(1, 5):
                df = small_source.read(name(f"w{i}")).set_index("ACCESS_METHOD_VAL")
                s = _num(df, f"{metric}_W{i}")
                weeks = s if weeks is None else weeks.add(s, fill_value=0)
            diff = (_num(m1, f"{metric}_M1") - weeks.reindex(m1.index).fillna(0)).abs()
            assert diff.max() < 0.01, (anchor, metric, diff.max())


def test_subscription_month_contains_its_weeks(small_source):
    df = small_source.read("subs_amt_auto")
    weeks = sum(_num(df, f"Sub_Amt_w{i}") for i in range(1, 5))
    assert (_num(df, "Sub_Amt_m1") + 0.01 >= weeks).all()


def test_w2_of_d_equals_w1_of_d_minus_7():
    d = date(2026, 10, 21)
    world = World.create(300, seed=5, as_of=d)
    now, before = generator.generate_run(world, d), generator.generate_run(world, d - timedelta(7))
    a = now.tables["activities_w2_score"].set_index("ACCESS_METHOD_VAL")
    b = before.tables["activities_w1_score"].set_index("ACCESS_METHOD_VAL")
    common = a.index.intersection(b.index)
    assert len(common) > 200
    for metric in ("Local_Voice_min", "Data_Volume_MB", "PG_Total_Amount"):
        assert np.allclose(_num(a.loc[common], f"{metric}_W2"), _num(b.loc[common], f"{metric}_W1"), atol=1e-6)


def test_model_scores_track_the_hidden_truth_with_partial_coverage(small_source):
    world = small_source.run.world
    base = set(world.c["msisdn"].astype("int64"))
    for m in config.MODEL_SCORE_REGISTRY:
        df = small_source.read(m.table)
        vals = df["access_method_val"].astype("int64")
        p1 = df["proba_1"].astype(float).to_numpy()
        assert np.allclose(p1 + df["proba_0"].astype(float).to_numpy(), 1.0, atol=1e-5)
        assert set(df["prediction"]) <= {"0", "1"}
        if m.tenure_range:
            assert df["tenure"].between(*m.tenure_range).all()
            continue
        cov = len(set(vals) & base) / len(base)
        assert 0.65 <= cov <= 0.95, (m.table, cov)
    churn = small_source.read(config.MODEL_SCORE_REGISTRY[0].table)
    idx = pd.Index(world.c["msisdn"].astype("int64")).get_indexer(churn["access_method_val"].astype("int64"))
    assert np.corrcoef(churn["proba_1"].astype(float), world.signal("churn")[idx])[0, 1] > 0.6


def test_keys_formats_and_excluded_plans(small_source):
    base = small_source.read("base_prepaid")
    assert base["ACCESS_METHOD_VAL"].str.fullmatch(r"\d{8}").all()
    assert (base["ACCESS_METHOD_VAL"] > config.MSISDN_MIN_EXCLUSIVE).all()            # BR-02
    am = small_source.read("access_method_id_auto")
    assert set(am["MSISDN"]) == set(base["ACCESS_METHOD_VAL"])     # contract: MSISDN = ACCESS_METHOD_VAL, no prefix
    world = World.create(3000, seed=1)
    assert world.c["plan"].isin(config.EXCLUDED_PLANS).sum() > 0                       # BR-10 exercised


def test_catalogue_generosity_is_monotonic_with_price(small_source):
    cat = small_source.read("btl_catalog").assign(price=lambda d: d["price"].astype(float))
    assert 50 <= len(cat) <= 70
    assert cat["price"].between(0.5, 15).all()
    for _, g in cat.groupby(["service", "validity"]):
        top = g.groupby("generosity")["price"].agg(["min", "max"]).loc[["L0", "L1", "L2"]]
        assert top["max"]["L0"] < top["min"]["L1"] and top["max"]["L1"] < top["min"]["L2"]


def test_campaign_targeting_rewards_and_history(small_source):
    tg = small_source.read("targeting_dataiku_auto")
    assert set(tg["segment"].dropna()) & {"RGS_Not_At_Risk_DualPlay_Upgrade", "Rgs_Not_At_Risk_Dualplay_Upgrade"}
    assert all(s.lower() in config.SEGMENT_TO_ACTION or s.lower() in config.EXCLUDED_SEGMENTS for s in tg["segment"].dropna())
    cat = set(small_source.read("btl_catalog")["btl_name"])
    f = small_source.read("subs_btl_auto")
    assert (~f["PRODUCTNAME"].isin(cat)).any(), "non-BTL noise products must exist (filtered by BR-40/63)"
    am = small_source.read("access_method_id_auto")
    to_val = dict(zip(am["ACCESS_METHOD_ID"], am["MSISDN"]))
    targeted = {to_val[i] for i in tg["access_method_id"]}
    takers = set(f.loc[f["PRODUCTNAME"].isin(cat), "access_method_val"])
    rate_t = len(takers & targeted) / len(targeted)
    rate_o = len(takers - targeted) / (len(small_source.read("base_prepaid")) - len(targeted))
    assert 0.03 < rate_t < 0.35 and rate_t > rate_o
    hist = small_source.read("sub_btl_3months")
    assert hist["PRODUCTNAME"].isin(cat).mean() > 0.8


def test_sample_arms_are_balanced_and_stratified(small_source):
    s = small_source.read(config.TARGET_SAMPLE_TABLE)
    shares = s["group"].value_counts(normalize=True)
    assert set(shares.index) == set(config.ARMS)
    assert (shares - 0.25).abs().max() < 0.08
    assert (s["strata"].str.count(r"\|") == 4).all()
    sizes = s.groupby("strata").size()
    assert (s["counts"].astype(int) == s["strata"].map(sizes)).all()
    assert not s["plan_name"].isin(config.EXCLUDED_PLANS).any()


def test_same_seed_same_data_and_5k_under_a_minute():
    d = date(2026, 10, 21)
    a = generator.generate(n=300, run_date=d, seed=3)
    b = generator.generate(n=300, run_date=d, seed=3)
    assert all(a.tables[t].equals(b.tables[t]) for t in a.tables)
    t0 = time.time()
    generator.generate(n=5000, run_date=d, seed=3)
    assert time.time() - t0 < 60


def test_pandas_and_pyarrow_load_safely_with_the_api():
    # vowpalwabbit must load before pyarrow on Windows; nbx/__init__.py guarantees it.
    code = "import nbx, pandas, pyarrow; from nbx.api import main; print('ok')"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-500:]
