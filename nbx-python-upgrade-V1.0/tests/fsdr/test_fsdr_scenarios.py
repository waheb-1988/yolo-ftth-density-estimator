"""Each scenario moves the signal it is supposed to move."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nbx.fsdr import config
from nbx.fsdr.datasource import SyntheticSource
from nbx.fsdr.synth import scenarios
from nbx.fsdr.synth.world import WAVE_REGIONS, World
from nbx.fsdr.validation import validate

D = date(2026, 10, 21)
N = 1500


def _score_by_line(src, table):
    df = src.read(table)
    return pd.Series(df["proba_1"].astype(float).to_numpy(), index=df["access_method_val"].astype(str))


def test_parse_scenarios():
    assert scenarios.parse("churn_wave, data_growth,churn_wave") == ["churn_wave", "data_growth"]
    assert scenarios.parse(None) == ["baseline"]
    with pytest.raises(ValueError):
        scenarios.parse("rain")
    assert set(scenarios.SCENARIOS) == set(config.SCENARIOS)


def test_churn_wave_raises_churn_and_cuts_usage_in_the_wave_regions():
    base = SyntheticSource(n=N, run_date=D, scenarios="baseline", seed=21)
    wave = SyntheticSource(n=N, run_date=D, scenarios="churn_wave", seed=21)
    w = wave.run.world.c
    in_wave = set(w.loc[w["region"].isin(WAVE_REGIONS), "msisdn"])
    t = config.MODEL_SCORE_REGISTRY[0].table
    sb, sw = _score_by_line(base, t), _score_by_line(wave, t)
    common = sorted(set(sb.index) & set(sw.index) & in_wave)
    assert sw[common].mean() > sb[common].mean() + 0.08
    def active_days(src):
        df = src.read("activities_w1_score")
        return df.loc[df["ACCESS_METHOD_VAL"].isin(in_wave), "Active_Days_Count_W1"].sum()

    assert active_days(wave) < 0.9 * active_days(base)                  # usage drops in the wave regions


def test_data_growth_shows_in_recent_weeks_and_in_the_appetency_score():
    src = SyntheticSource(n=N, run_date=D, scenarios="data_growth", seed=22)
    c = src.run.world.c
    growers = set(c.loc[c["growth"], "msisdn"])
    assert len(growers) > 50
    w1 = src.read("activities_w1_score").set_index("ACCESS_METHOD_VAL")["Data_Volume_MB_W1"].astype(float)
    w4 = src.read("activities_w4_score").set_index("ACCESS_METHOD_VAL")["Data_Volume_MB_W4"].astype(float)
    g = list(growers)
    assert w1.reindex(g).fillna(0).sum() > 1.5 * w4.reindex(g).fillna(0).sum()
    base = SyntheticSource(n=N, run_date=D, scenarios="baseline", seed=22)
    t = "APPETENCY_DATA_204_SCORING_TD.prediction"
    sb, sg = _score_by_line(base, t), _score_by_line(src, t)
    common = sorted(set(sb.index) & set(sg.index) & growers)
    assert sg[common].mean() > sb[common].mean()


def _accept(world, price, contacts=None):
    idx = np.arange(world.n)
    if contacts is not None:
        world.c["contacts_last_4w"] = contacts
    n = world.n
    return world.accept_probability(idx, np.full(n, "UPSELL DATA"), np.full(n, "DATA"), np.full(n, "L2"),
                                    np.full(n, price), np.full(n, 30))


def test_price_sensitive_makes_acceptance_drop_faster_above_spend():
    base, sens = World.create(2000, seed=3), World.create(2000, seed=3, scenarios=["price_sensitive"])
    drop_base = _accept(base, 2.0).mean() - _accept(base, 15.0).mean()
    drop_sens = _accept(sens, 2.0).mean() - _accept(sens, 15.0).mean()
    assert drop_sens > drop_base


def test_campaign_fatigue_lowers_acceptance_with_contacts():
    base, tired = World.create(2000, seed=4), World.create(2000, seed=4, scenarios=["campaign_fatigue"])
    zero, four = np.zeros(2000, int), np.full(2000, 4)
    assert (_accept(tired, 5.0, zero).mean() - _accept(tired, 5.0, four).mean()
            > _accept(base, 5.0, zero).mean() - _accept(base, 5.0, four).mean())


def test_new_offers_have_no_purchase_history():
    src = SyntheticSource(n=600, run_date=D, scenarios="new_offers", seed=6)
    cat = src.read("btl_catalog")["btl_name"]
    new = set(cat[cat.str.endswith("_NEW")])
    assert len(new) == 3
    assert not (set(src.read("sub_btl_3months")["PRODUCTNAME"]) & new)


def test_data_quality_issues_are_caught_by_validation():
    src = SyntheticSource(n=800, run_date=D, scenarios="data_quality_issues", seed=8)
    rep = validate(src)
    assert rep.status == "blocked"
    found = {(i.check, i.table) for i in rep.issues}
    churn = config.MODEL_SCORE_REGISTRY[0].table
    assert ("duplicates", "base_prepaid") in found                      # duplicate MSISDNs
    assert ("null_key", "base_prepaid") in found and ("null_key", "activities_w1_auto") in found
    nulls = [i for i in rep.issues if i.check == "null_key" and i.table == "base_prepaid"]
    assert any(i.column == "ACCESS_METHOD_VAL" for i in nulls) and any(i.column == "PARTY_ID" for i in nulls)
    assert ("duplicates", churn) in found                               # stacked churn duplicates (warning)
    assert ("raw_type", "subs_amt_auto") in found and ("raw_type", churn) in found   # keys as strings
    assert {i.issue for i in [type("x", (), d) for d in src.manifest()["quality_issues_injected"]]} >= {
        "duplicate_msisdn", "null_key", "stacked_duplicates", "key_as_string"}


def test_scenarios_combine():
    src = SyntheticSource(n=400, run_date=D, scenarios="churn_wave,data_growth,price_sensitive", seed=9)
    assert src.manifest()["generator"]["scenarios"] == ["churn_wave", "data_growth", "price_sensitive"]
    assert validate(src).status == "ok"


# -- consecutive weeks from one world (generator --weeks) ---------------------------------------
@pytest.fixture(scope="module")
def four_weeks():
    from nbx.fsdr.synth.generator import generate_weeks
    from nbx.fsdr.datasource import _raw_frame
    runs = generate_weeks(1200, date(2026, 10, 1), 4, "churn_wave,data_growth", seed=21)
    return [{t: _raw_frame(df.to_dict("records"), t) for t, df in r.tables.items()} for r in runs]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def test_consecutive_weeks_share_their_common_days(four_weeks):
    """C2: W2 of D+7 = W1 of D and W3 of D+7 = W2 of D, for activities and subscriptions."""
    for prev, nxt in zip(four_weeks, four_weeks[1:]):
        for (a, pa), (b, pb) in [(("activities_w1_score", "W1"), ("activities_w2_score", "W2")),
                                 (("activities_w2_score", "W2"), ("activities_w3_score", "W3"))]:
            m = prev[a].merge(nxt[b], on="ACCESS_METHOD_VAL")
            assert len(m) > 1000
            for metric in ("Call_Count", "Data_Volume_MB", "Local_Voice_min", "PG_Total_Amount", "Active_Days_Count"):
                assert np.allclose(_num(m, f"{metric}_{pa}"), _num(m, f"{metric}_{pb}"), atol=1e-6), (a, b, metric)
        s = prev["subs_amt_score"].merge(nxt["subs_amt_score"], on="ACCESS_METHOD_VAL", suffixes=("_d", "_d7"))
        assert np.allclose(_num(s, "Sub_Amt_w1_d"), _num(s, "Sub_Amt_w2_d7"), atol=1e-6)


def test_churn_wave_rises_every_week_in_north_batinah_only(four_weeks):
    """C3: mean proba_1 of the global churn file, by region via location_auto."""
    churn_t = config.MODEL_SCORE_REGISTRY[0].table
    assert WAVE_REGIONS == {"North Batinah"}
    wave, other = [], []
    for wk in four_weeks:
        loc = wk["location_auto"].drop_duplicates("Access_Method_id")
        sc = wk[churn_t].merge(loc, left_on="access_method_id", right_on="Access_Method_id")
        p = _num(sc, "proba_1")
        wave.append(p[sc["REGION"] == "North Batinah"].mean())
        other.append(p[sc["REGION"] != "North Batinah"].mean())
    assert all(b > a for a, b in zip(wave, wave[1:])), wave
    assert max(other) - min(other) <= 0.02, other


def test_data_growth_share_with_data_rises_every_week(four_weeks):
    share = [(_num(wk["activities_w1_score"], "Data_Volume_MB_W1") > 0).mean() for wk in four_weeks]
    assert all(b > a for a, b in zip(share, share[1:])), share
