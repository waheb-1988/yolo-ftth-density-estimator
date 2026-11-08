"""Offer business rules (BR-66..82 + conflicts), the ranker, and the free / cstr arms."""
from __future__ import annotations

import pandas as pd
import pytest

from nbx.fsdr import config
from nbx.fsdr.offer_ranker import ImplicitBPRRanker
from nbx.fsdr.offer_rules import anchor_column, check_offers, families, family_items, select_offers
from nbx.fsdr.pipelines import read_table
from nbx.fsdr.pipelines.recommend import OfferEngine, run_recommend


def test_anchor_columns_follow_br77():
    assert anchor_column("COMBO", 30) == "AVG_REV_MONTHLY" and anchor_column("DATA", 7) == "AVG_REV_DATA_WEEKLY"
    assert anchor_column("VOICE", 30) == "AVG_REV_VOICE_MONTHLY"


def test_arpu_anchor_is_mean_plus_std_with_90_10_split(small_source):
    eng = OfferEngine(small_source)
    a = eng.anchors
    rev = a[["REV_M1", "REV_M2", "REV_M3"]]
    assert (a["AVG_REV_MONTHLY"] - (rev.mean(axis=1) + rev.std(axis=1, ddof=1))).abs().max() < 1e-9
    assert (a["AVG_REV_DATA_MONTHLY"] - 0.9 * a["AVG_REV_MONTHLY"]).abs().max() < 1e-9
    assert (a["AVG_REV_VOICE_WEEKLY"] - 0.1 * a["AVG_REV_WEEKLY"]).abs().max() < 1e-9


def test_families_including_the_cross_sell_voice_fix():
    assert families("CROSS SELL VOICE", "cstr", "all_L2") == [("DATA", "L2"), ("COMBO", "L2")]
    assert families("CROSS SELL DATA", "cstr", "all_L2") == [("VOICE", "L2"), ("COMBO", "L2")]
    assert families("UPSELL VOICE", "cstr", "normal") == [("DATA", "L0"), ("COMBO", "L1")]
    assert families("MAINTAIN", "cstr", "normal") == [("COMBO", "L0"), ("DATA", "L0")]
    assert families(None, "free") == [("COMBO", "L2"), ("DATA", "L2")]


def _cands(rows):
    return pd.DataFrame([{"ACCESS_METHOD_VAL": "70000001", "VALIDITY": 30, "GENEROSITY": "L2", "RANK": i + 1, **r}
                         for i, r in enumerate(rows)])


def _anchors(total=5.0):
    return pd.DataFrame({"AVG_REV_MONTHLY": [total], "AVG_REV_DATA_MONTHLY": [0.9 * total], "AVG_REV_VOICE_MONTHLY": [0.1 * total]},
                        index=["70000001"])


FAMS = [("COMBO", "L2"), ("DATA", "L2")]


def test_pass_1_picks_inside_the_band_by_score_then_price_desc():
    c = _cands([{"BTL_NAME": "C1", "SERVICE": "COMBO", "PRICE": 6.0, "SCORE": 0.9},
                {"BTL_NAME": "C2", "SERVICE": "COMBO", "PRICE": 7.0, "SCORE": 0.9},
                {"BTL_NAME": "C3", "SERVICE": "COMBO", "PRICE": 20.0, "SCORE": 1.0},     # outside band
                {"BTL_NAME": "D1", "SERVICE": "DATA", "PRICE": 6.0, "SCORE": 0.5}])
    picks = select_offers(c, _anchors(5.0), "70000001", FAMS, 3.0, 30, set())
    assert [(p.btl_name, p.pass_no) for p in picks] == [("C2", 1), ("D1", 1)]


def test_pass_2_fills_the_missing_type_without_upper_cap():
    c = _cands([{"BTL_NAME": "C1", "SERVICE": "COMBO", "PRICE": 6.0, "SCORE": 0.9},
                {"BTL_NAME": "D9", "SERVICE": "DATA", "PRICE": 14.0, "SCORE": 0.2},
                {"BTL_NAME": "D8", "SERVICE": "DATA", "PRICE": 12.0, "SCORE": 0.2}])
    picks = select_offers(c, _anchors(5.0), "70000001", FAMS, 3.0, 30, set())
    assert [(p.btl_name, p.pass_no) for p in picks] == [("C1", 1), ("D8", 2)]           # score tie -> cheaper


def test_pass_3_when_nothing_fits_and_no_offer_when_anchor_is_above_every_price():
    c = _cands([{"BTL_NAME": "C9", "SERVICE": "COMBO", "PRICE": 15.0, "SCORE": 0.9},
                {"BTL_NAME": "D9", "SERVICE": "DATA", "PRICE": 14.0, "SCORE": 0.8}])
    picks = select_offers(c, _anchors(2.0), "70000001", FAMS, 3.0, 30, set())
    assert {p.pass_no for p in picks} == {3} and len(picks) == 2
    assert select_offers(c, _anchors(50.0), "70000001", FAMS, 3.0, 30, set()) == []     # BR-82


def test_conflicting_offers_are_never_picked_together():
    c = _cands([{"BTL_NAME": "C1", "SERVICE": "COMBO", "PRICE": 6.0, "SCORE": 0.9},
                {"BTL_NAME": "D1", "SERVICE": "DATA", "PRICE": 6.0, "SCORE": 0.9},
                {"BTL_NAME": "D2", "SERVICE": "DATA", "PRICE": 6.5, "SCORE": 0.5}])
    picks = select_offers(c, _anchors(5.0), "70000001", FAMS, 3.0, 30, {frozenset(("C1", "D1"))})
    assert [p.btl_name for p in picks] == ["C1", "D2"]


def test_check_offers_hard_rules(small_source):
    eng = OfferEngine(small_source)
    cat = eng.catalog
    monthly = cat[cat["VALIDITY"] == 30]["BTL_NAME"].tolist()
    a, b = next(iter(eng.conflicts))
    v = check_offers(["NOPE", a, b] + monthly[:3], cat, eng.conflicts)
    assert any("not in the catalogue" in x for x in v) and any("conflict" in x for x in v)
    assert any("at most 4" in x for x in v) and any("validity 30" in x for x in v)
    assert check_offers(monthly[:1], cat, eng.conflicts) == []


@pytest.mark.parametrize("action,arm", [("MAINTAIN", "cstr"), ("CROSS SELL VOICE", "cstr"), (None, "free")])
def test_ranker_only_proposes_offers_bought_in_the_filtered_history(small_source, monkeypatch, action, arm):
    """Production Codes 3/4: item_id = interaction1.BTL_NAME.unique() -- a family offer never bought is no candidate."""
    eng = OfferEngine(small_source)
    bought = set(eng.interactions["BTL_NAME"])
    for validity in (30, 7):
        family = set(family_items(eng.catalog, families(action, arm, eng.mapping), validity)["BTL_NAME"])
        assert set(eng.items(action, arm, validity)["BTL_NAME"]) == family & bought
    monkeypatch.setattr(config, "RANKER_ITEMS", "catalogue")
    assert set(eng.items(action, arm, 30)["BTL_NAME"]) == set(
        family_items(eng.catalog, families(action, arm, eng.mapping), 30)["BTL_NAME"])


def test_a_family_without_history_gives_no_candidate_and_no_offer(small_source):
    eng = OfferEngine(small_source)
    eng.interactions = eng.interactions.iloc[0:0]
    val = str(eng.anchors.index[0])
    assert eng.candidates([val], "MAINTAIN", "cstr", 30).empty
    assert eng.select(val, "MAINTAIN", "cstr") == []


def test_ranker_scores_by_the_customers_own_id_and_cold_starts_with_popularity():
    inter = pd.DataFrame({"ACCESS_METHOD_VAL": ["u1", "u1", "u2", "u3", "u3", "u4"], "BTL_NAME": ["A", "B", "B", "C", "A", "A"],
                          "WEIGHT": 1.0})
    r = ImplicitBPRRanker().fit(inter, ["A", "B", "C"])
    s12, s21 = r.score(["u1", "u2"]), r.score(["u2", "u1"])
    assert (s12[0] == s21[1]).all() and (s12[1] == s21[0]).all()        # no loop-index bug
    cold = r.score(["new"])[0]
    assert cold.argmax() == 0                                            # A is the most popular


@pytest.mark.parametrize("policy", ["price_band", "value"])
@pytest.mark.parametrize("arm", ["cstr", "free"])
def test_arm_recommendations_respect_every_rule(scored_run, arm, policy, monkeypatch):
    monkeypatch.setattr(config, "OFFER_POLICY", policy)
    folder = scored_run["folder"]
    rep = run_recommend(folder, arm)
    recos = pd.concat([read_table(folder, f"recos_monthly_{arm}"), read_table(folder, f"recos_weekly_{arm}")])
    assert rep["with_offers"] > 0.5 * rep["customers"] and len(recos) == rep["offers"]
    per_cust = recos.groupby("ACCESS_METHOD_VAL")
    assert per_cust.size().max() <= config.MAX_OFFERS_TOTAL
    assert recos.groupby(["ACCESS_METHOD_VAL", "VALIDITY"]).size().max() <= config.MAX_OFFERS_PER_VALIDITY
    if policy == "price_band":                                  # production's 3 passes (BR-78..81)
        assert rep["offer_policy"] == "price_band"
        assert (recos["PRICE"] > recos["ANCHOR"]).all() and set(recos["PASS_NO"]) <= {1, 2, 3}
    else:                                                       # V1.0 value arbitration (pass 4, 5 = exploration)
        assert rep["offer_policy"] == "value" and rep["acceptance_model"]["trained"]
        assert set(recos["PASS_NO"]) <= {4, 5} and recos["P_ACCEPT"].between(0, 1).all()
        assert (recos["PRICE"] > recos["ANCHOR"]).all()           # default floor = production's (BR-78..80)
        assert (recos["PASS_NO"] == 4).mean() > 0.8
    eng = OfferEngine.__new__(OfferEngine)
    from nbx.fsdr.datasource import JsonSource
    from nbx.fsdr.offer_rules import load_conflicts

    conflicts = load_conflicts(JsonSource(folder))
    for _, g in per_cust:
        names = g["BTL_NAME"].tolist()
        assert not any(frozenset((a, b)) in conflicts for i, a in enumerate(names) for b in names[i + 1:])
    if arm == "free":
        assert set(recos["GENEROSITY"]) == {"L2"} and set(recos["SERVICE"]) <= {"DATA", "COMBO"}
    else:
        for action, g in recos.groupby("ACTION"):
            allowed = {s for s, _ in families(action, "cstr")}
            assert set(g["SERVICE"]) <= allowed, action
