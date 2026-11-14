"""V3.0: the FSDR CSV standard (.csv.gz, master §2 / §20) and the 4-group design of the update prompt (§8 tests)."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from nbx.fsdr import audit_groups, config, csvstd, llm_budget, panel, report_groups, rules
from nbx.fsdr.contract import data_agent
from nbx.fsdr.datasource import JsonSource, read_table_file
from nbx.fsdr.offers import anchors as anc
from nbx.fsdr.offers import replay, selection, universe
from nbx.fsdr.pipelines import scoring_groups, training
from nbx.fsdr.validation import validate

from .conftest import RUN_DATE

VAL = "ACCESS_METHOD_VAL"


# ---------------------------------------------------------------------------
# FSDR CSV standard
# ---------------------------------------------------------------------------
def test_csv_standard_values_ids_missing_decimals_and_arabic(tmp_path):
    df = pd.DataFrame({"ACCESS_METHOD_VAL": ["09123456", "91234567"], "price": [12.5, 7.0], "amount": [0.0001, None],
                       "n": pd.array([3, None], dtype="Int64"), "script": ["عرض خاص, 5 ريال", 'say "yes"'],
                       "day": [pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-23")]})
    p = csvstd.write_csv(df, tmp_path / "t.csv.gz")
    raw = gzip.decompress(p.read_bytes()).decode("utf-8")
    assert not raw.startswith("﻿") and "\r" not in raw
    lines = raw.split("\n")
    assert lines[0] == "ACCESS_METHOD_VAL,price,amount,n,script,day"
    assert lines[1] == '09123456,12.5,0.0001,3,"عرض خاص, 5 ريال",2026-10-22'
    assert lines[2] == '91234567,7.0,,,"say ""yes""",2026-10-23'
    assert "e-0" not in raw and "nan" not in raw.lower() and "None" not in raw
    back = csvstd.read_csv(p)
    assert back.loc[0, "ACCESS_METHOD_VAL"] == "09123456" and back.loc[1, "amount"] == ""
    assert back.loc[0, "script"] == "عرض خاص, 5 ريال"
    schema = json.loads(csvstd.schema_path(p).read_text(encoding="utf-8"))
    assert {c["name"]: c["type"] for c in schema["columns"]}["ACCESS_METHOD_VAL"] == "string"
    assert csvstd.write_csv(df, tmp_path / "u.csv.gz").read_bytes() == p.read_bytes()      # deterministic gzip


def test_csv_gz_id_with_leading_zero_survives_the_loader(tmp_path):
    base = pd.DataFrame({"UBSCRIPTION_ID": ["0400001"], "ACCESS_METHOD_VAL": ["09123456"], "ACCESS_METHOD_ID": [1],
                         "ACCOUNT_ID": [2], "PARTY_ID": [3], "access_method_subs_start_dt": ["2025-01-01"],
                         "CURRENT_PLAN": ["FSDR Prepaid"]})
    df = read_table_file(csvstd.write_csv(base, tmp_path / "base_prepaid.csv.gz"), "base_prepaid")
    assert df.loc[0, "ACCESS_METHOD_VAL"] == "09123456" and df.loc[0, "UBSCRIPTION_ID"] == "0400001"


def test_generated_run_round_trips_through_csv_gz(small_source, small_folder, tmp_path):
    folder = small_source.write(tmp_path / RUN_DATE.isoformat(), fmt="csv.gz")
    ref, got = JsonSource(small_folder), JsonSource(folder)
    assert set(got.tables()) == set(ref.tables()) and not got.unknown_files()
    assert all(csvstd.schema_path(got.path(t)).exists() for t in got.tables())
    for t in ref.tables():
        a, b = ref.read(t), got.read(t)
        norm = lambda df: df.map(lambda v: None if v is None or v == "" else round(float(v), 6)       # noqa: E731
                                 if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        pd.testing.assert_frame_equal(norm(a), norm(b), check_dtype=False, obj=t)
    assert validate(got, RUN_DATE).status == "ok"
    t = "btl_catalog"                                     # .csv.gz -> Parquet-free re-write gives the same bytes
    again = csvstd.write_csv(csvstd.read_csv(got.path(t)), tmp_path / f"{t}.csv.gz", schema=False)
    assert gzip.decompress(again.read_bytes()) == gzip.decompress(got.path(t).read_bytes())


def test_production_run_blocks_on_a_test_format(small_folder, tmp_path, monkeypatch):
    import shutil

    folder = tmp_path / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    man = json.loads((folder / config.MANIFEST_FILE).read_text(encoding="utf-8"))
    man["source"] = "dataiku export"
    (folder / config.MANIFEST_FILE).write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(config, "ALLOW_TEST_FORMATS", False)
    rep = validate(JsonSource(folder), RUN_DATE)
    assert any(i.check == "format" and i.level == "blocking" for i in rep.issues)
    monkeypatch.setattr(config, "ALLOW_TEST_FORMATS", True)
    assert not any(i.check == "format" and i.level == "blocking" for i in validate(JsonSource(folder), RUN_DATE).issues)


# ---------------------------------------------------------------------------
# Panel (test 1, 14)
# ---------------------------------------------------------------------------
def _pool(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({VAL: [f"9{i:07d}" for i in range(n)],
                         "NATIONALITY_GROUP": rng.choice(config.NATIONALITY_GROUPS, n, p=[.5, .1, .2, .1, .1]),
                         "VALUE_BAND": rng.choice(config.VALUE_BANDS, n), "USAGE_TYPE": rng.choice(config.USAGE_TYPES, n),
                         "CURRENT_PLAN": rng.choice(["A", "B", "C"], n), "AGE_BAND": rng.choice(config.AGE_BAND_NAMES, n),
                         "TENURE_BAND": rng.choice(config.TENURE_BAND_NAMES, n),
                         "SEGMENT": rng.choice(config.S_SEGMENTS[2:], n), "EXCLUDED": 0})


def test_assignment_100k_plus_20k_stratified_no_overlap():
    pool = _pool(400_000)
    p, rep = panel.assign_panel(pool, config.group_sizes(), seed=3)
    assert rep["sizes"] == {"t1c": 25_000, "t1f": 25_000, "t2c": 25_000, "t2f": 25_000, "control": 20_000}
    assert panel.check_sizes(p, config.group_sizes()) == [] and p[VAL].is_unique
    bal = panel.balance(p)
    big = bal[bal[[c for c in bal.columns if c.startswith("share_")]].min(axis=1) > 5]
    assert big["max_gap_points"].max() <= config.BALANCE_MAX_POINTS
    assert set(p["PANEL_GROUP"]) == {"t1_constraint", "t1_free", "t2_constraint", "t2_free", "control"}


def test_sample_mapping_cstr_free_cg_and_top_up():
    pool = _pool(20_000)
    sample = pd.DataFrame({VAL: pool[VAL].iloc[:600], "group": ["cstr"] * 200 + ["free"] * 200 + ["cg"] * 200})
    sizes = {"t1c": 200, "t1f": 200, "t2c": 200, "t2f": 200, "control": 300}
    p, rep = panel.assign_panel(pool, sizes, seed=1, sample=sample)
    by = p.set_index(VAL)["GROUP_KEY"]
    assert (by.reindex(sample[VAL].iloc[:200]) == "t1c").all() and (by.reindex(sample[VAL].iloc[200:400]) == "t1f").all()
    assert (by.reindex(sample[VAL].iloc[400:]) == "control").all()
    assert rep["sizes"] == sizes and rep["from_sample"] == 600            # control topped up by 100 drawn lines


def test_t2_split_false_gives_one_50k_t2_with_cc_rules(monkeypatch):
    monkeypatch.setattr(config, "T2_SPLIT", False)
    sizes = config.group_sizes()
    assert sizes == {"t1c": 25_000, "t1f": 25_000, "t2c": 50_000, "control": 20_000}
    assert config.GROUPS["t2c"]["rules"] == "CC"


# ---------------------------------------------------------------------------
# Universes and selections (tests 2 - 8)
# ---------------------------------------------------------------------------
def _catalog() -> pd.DataFrame:
    rows = []
    for svc in ("DATA", "VOICE", "COMBO"):
        for gen in ("L0", "L1", "L2"):
            for v in (7, 30):
                rows.append({"BTL_NAME": f"{svc}_{gen}_{v}", "TYPE": svc, "SERVICE": svc, "GENEROSITY": gen,
                             "VALIDITY": v, "PRICE": 5.0, "GB": 1.0, "MIN": 0.0})
    return pd.DataFrame(rows)


def test_constraint_universe_by_action_cross_sell_voice_fixed():
    rec = _catalog()
    fam = lambda a: set(zip(*universe.constraint_universe(rec, a, 30)[["SERVICE", "GENEROSITY"]].T.values))  # noqa: E731
    assert fam("UPSELL DATA") == {("VOICE", "L2"), ("COMBO", "L2")}
    assert fam("MAINTAIN") == {("COMBO", "L2"), ("DATA", "L2")}
    assert fam("CROSS SELL VOICE") == {("DATA", "L2"), ("COMBO", "L2")}


def test_free_universe_ignores_the_action_and_the_unsold():
    cat = _catalog()
    bought = pd.DataFrame({VAL: ["9"] * (len(cat) - 1), "BTL_NAME": cat["BTL_NAME"].iloc[1:].tolist(), "NB_SUB": 1})
    rec = universe.recommendable(cat.assign(BTL_NAME=cat["BTL_NAME"]), bought)
    assert cat["BTL_NAME"].iloc[0] not in set(rec["BTL_NAME"])                  # nobody bought it: never a candidate
    a = universe.free_candidates(pd.DataFrame({VAL: ["1"], "ACTION": ["UPSELL DATA"]}), rec, 30)
    b = universe.free_candidates(pd.DataFrame({VAL: ["1"], "ACTION": ["RETAIN"]}), rec, 30)
    assert list(a["BTL_NAME"]) == list(b["BTL_NAME"])
    assert set(a["SERVICE"]) <= {"DATA", "COMBO"} and set(a["GENEROSITY"]) == {"L2"}


GOLD = [("C7", "COMBO", 7.0, 0.9), ("C8", "COMBO", 8.0, 0.95), ("D55", "DATA", 5.5, 0.7), ("D74", "DATA", 7.4, 0.8),
        ("D73", "DATA", 7.3, 0.8)]


def _cands(rows, action="MAINTAIN", anchors=(6.0, 5.4), val="91000000"):
    return pd.DataFrame([{VAL: val, "BTL_NAME": n, "SERVICE": s, "GENEROSITY": "L2", "VALIDITY": 30, "PRICE": p,
                          "SCORE": sc, "ACTION": action, "ANCHOR": anchors[0] if s == "COMBO" else anchors[1]}
                         for n, s, p, sc in rows])


def test_t1c_golden_passes_k2():
    assert selection.select_code3(_cands(GOLD))["BTL_NAME"].tolist() == ["C7", "D73"]


def test_t1f_golden_passes_k3_boundaries_and_pass2_tie():
    assert selection.select_code4(_cands(GOLD))["BTL_NAME"].tolist() == ["C8", "D74"]
    edge = _cands([("CA", "COMBO", 6.0, 0.9), ("CB", "COMBO", 9.0, 0.9), ("CC", "COMBO", 7.0, 0.1),
                   ("DX", "DATA", 9.0, 0.5), ("DY", "DATA", 8.5, 0.5)])
    sel = selection.select_code4(edge)
    assert sel[["BTL_NAME", "PASS"]].values.tolist() == [["CC", 1], ["DY", 2]]   # 6.0 / 9.0 not in pass 1; tie -> cheaper


def _band(prices, action="UPSELL"):
    return pd.DataFrame([{VAL: "9", "BTL_NAME": f"O{p}", "SERVICE": "COMBO", "GENEROSITY": "L2", "VALIDITY": 30,
                          "PRICE": p, "SCORE": 0.5, "ACTION": action, "ANCHOR": 6.0, "CH_ANCHOR": 6.0} for p in prices])


def test_challenger_band_widen_then_t1_fallback():
    c = _band([5.0, 6.4, 7.0, 8.5, 10.0])
    lo, hi = selection.challenger_band(c, config.CHALLENGER_BAND)
    assert (lo.iloc[0], hi.iloc[0]) == (6.5, 9.0)
    inside = c[(c["PRICE"] >= lo) & (c["PRICE"] <= hi)]["PRICE"].tolist()
    assert inside == [7.0, 8.5]
    assert selection.select_challenger(c, "code3")[["BTL_NAME", "SELECTED_BY"]].values.tolist() == [["O8.5", "CC-04"]]
    assert selection.select_challenger(_band([5.0, 10.0]), "code3")["SELECTED_BY"].tolist() == ["CC-04 widened"]
    fb = _band([13.0]).assign(ANCHOR=6.0)                      # beyond 2x: the T1C passes (pass 3, no cap)
    assert selection.select_challenger(fb, "code3")["SELECTED_BY"].tolist() == ["fallback B-30"]


def test_anchor_split_from_own_spend():
    anchors = pd.DataFrame({"AVG_REV_MONTHLY": [10.0], "AVG_REV_WEEKLY": [2.0]}, index=["9"])
    c = pd.DataFrame({VAL: ["9", "9", "9"], "SERVICE": ["DATA", "VOICE", "COMBO"], "VALIDITY": [30, 30, 7]})
    got = anc.challenger_anchor(c, anchors, pd.Series({"9": 0.7})).round(6).tolist()
    assert got == [7.0, 3.0, 2.0]
    assert anc.challenger_anchor(c, anchors, pd.Series(dtype=float)).round(6).tolist() == [9.0, 1.0, 2.0]   # 90 / 10


def _random_frame(rng, n_cust, services):
    rows = []
    for i in range(n_cust):
        a_tot, a_data = rng.uniform(0, 8), None
        a_data = 0.9 * a_tot
        for j in range(rng.integers(2, 9)):
            svc = services[rng.integers(0, 2)]
            rows.append({"access_method_val": f"9{i:07d}", "btl_name": f"B{j}_{svc}", "service": svc, "generosity": "L2",
                         "validity": 30, "price": round(rng.uniform(0.5, 14), 1), "score": float(rng.random()),
                         "avg_rev": a_tot, "avg_rev_data": a_data, "avg_rev_voice": 0.1 * a_tot})
    return pd.DataFrame(rows).drop_duplicates(["access_method_val", "btl_name"])


@pytest.mark.parametrize("action", ["MAINTAIN", "UPSELL DATA", "UPSELL VOICE"])
def test_select_code3_equals_the_reco_btl3_replay(action):
    rng = np.random.default_rng(5)
    svcs = replay.MAP_ACTION_OFFERS[action]["service"]
    m = _random_frame(rng, 300, svcs)
    rev_1, rev_2 = replay.REV_ARGS[action]
    ref = replay.replay_reco_btl3(m, action, 30, rev_1, rev_2)
    c = pd.DataFrame({VAL: m["access_method_val"], "BTL_NAME": m["btl_name"], "SERVICE": m["service"],
                      "GENEROSITY": "L2", "VALIDITY": 30, "PRICE": m["price"], "SCORE": m["score"], "ACTION": action,
                      "ANCHOR": np.select([m["service"] == "COMBO", m["service"] == "DATA"],
                                          [m["avg_rev"], m["avg_rev_data"]], m["avg_rev_voice"])})
    got = selection.select_code3(c)
    assert sorted(zip(ref["access_method_val"], ref["btl_name"])) == sorted(zip(got[VAL], got["BTL_NAME"]))


def test_select_code4_equals_the_reco_btl2_replay():
    rng = np.random.default_rng(9)
    m = _random_frame(rng, 300, ["COMBO", "DATA"])
    ref = replay.replay_reco_btl2(m, 30, "avg_rev", "avg_rev_data")
    c = pd.DataFrame({VAL: m["access_method_val"], "BTL_NAME": m["btl_name"], "SERVICE": m["service"],
                      "GENEROSITY": "L2", "VALIDITY": 30, "PRICE": m["price"], "SCORE": m["score"], "ACTION": "RETAIN",
                      "ANCHOR": np.where(m["service"] == "COMBO", m["avg_rev"], m["avg_rev_data"])})
    got = selection.select_code4(c)
    assert sorted(zip(ref["access_method_val"], ref["btl_name"])) == sorted(zip(got[VAL], got["BTL_NAME"]))


# ---------------------------------------------------------------------------
# End to end on a generated run (test 9, audit test 11)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def p6_run(small_folder, tmp_path_factory):
    root = tmp_path_factory.mktemp("v3 runs")
    runs, cache = root / "runs", root / "cache"
    assert data_agent.run_t0(small_folder, runs_root=runs, cache_root=cache)["status"] == "done"
    training.run_features(RUN_DATE, runs_root=runs, cache_root=cache, workers=1)
    p1 = scoring_groups.run_p1_panel(RUN_DATE, runs_root=runs, cache_root=cache)
    p6 = scoring_groups.run_p6_offers(RUN_DATE, runs_root=runs, cache_root=cache)
    return {"p1": p1, "p6": p6, "dir": runs / RUN_DATE.isoformat() / "score"}


def test_p6_writes_the_4_code2_tables_and_8_reco_tables(p6_run):
    d = p6_run["dir"] / "P6_offers"
    for key, g in config.GROUPS.items():
        for t in (g["scored_table"], g["recos_monthly"], g["recos_weekly"]):
            assert (d / f"{t}{csvstd.SUFFIX}").exists(), t
        cols = list(csvstd.read_csv(d / f"{g['recos_monthly']}{csvstd.SUFFIX}").columns)
        assert ("action" in cols) == g["action_column"]
        scored = csvstd.read_csv(d / f"{g['scored_table']}{csvstd.SUFFIX}")
        assert list(scored.columns) == ["access_method_val", "action", "proba", "tg_flag"]
        assert set(scored["tg_flag"]) <= {g["tg_flag"]}
    assert not any((d / f"{t}{csvstd.SUFFIX}").exists() for t in config.SUPERSEDED_TABLES)
    man = json.loads((d / csvstd.OUTPUTS_MANIFEST).read_text(encoding="utf-8"))
    assert len(man["files"]) == 13 and all(f["sha256"] and f["schema"] for f in man["files"])
    res = csvstd.read_csv(d / f"results_scoring_nba{csvstd.SUFFIX}")
    panel_df = pd.read_parquet(p6_run["dir"] / "P1_panel" / "panel.parquet")
    assert not set(res["access_method_val"]) & set(panel_df.loc[panel_df["GROUP_KEY"] == "control", VAL])
    assert p6_run["p6"]["audit_status"] in ("PASS", "WARN") and not p6_run["p6"]["push_blocked"]
    tr = pd.read_csv(d / "traces.csv")
    assert tr["customer_type"].nunique() == 1 and set(tr["group"]) == set(config.GROUPS)


def test_t1f_and_t2f_never_get_voice_or_non_l2(p6_run):
    d = p6_run["dir"] / "P6_offers"
    for g in ("t1f", "t2f"):
        for t in ("recos_monthly", "recos_weekly"):
            r = csvstd.read_csv(d / f"{config.GROUPS[g][t]}{csvstd.SUFFIX}")
            assert "VOICE" not in set(r["service"]) and set(r["generosity"]) <= {"L2"}


def _sel_row(**kw):
    base = {VAL: "91000000", "GROUP_KEY": "t1c", "BTL_NAME": "X", "SERVICE": "COMBO", "GENEROSITY": "L2",
            "VALIDITY": 30, "PRICE": 7.0, "SCORE": 0.5, "ACTION": "MAINTAIN", "ANCHOR": 6.0, "CH_ANCHOR": np.nan,
            "PASS": 1, "SELECTED_BY": "B-28", "RULES_VERSION": "production_constraint"}
    return {**base, **kw}


@pytest.mark.parametrize("bad", [
    _sel_row(GROUP_KEY="t1f", SERVICE="VOICE", ACTION="RETAIN", PRICE=7.0, SELECTED_BY="FR-09", RULES_VERSION="production_free"),
    _sel_row(SERVICE="VOICE"),                                               # T1C offer outside its B-21 family
    _sel_row(PRICE=8.5),                                                     # MAINTAIN pass 1 at anchor + 2.5
    _sel_row(GROUP_KEY="t2f", ACTION="RETAIN", CH_ANCHOR=6.0, SELECTED_BY="CF-04", RULES_VERSION="CC-v1"),
    _sel_row(BTL_NAME="NOT_RECOMMENDABLE"),
])
def test_auditor_fails_and_blocks_the_push(bad):
    pool = _pool(2_000)
    sizes = {"t1c": 300, "t1f": 300, "t2c": 300, "t2f": 300, "control": 200}
    p, _ = panel.assign_panel(pool, sizes, seed=2)
    rec = pd.DataFrame({"BTL_NAME": ["X"]})
    ok = audit_groups.audit_groups(p, pd.DataFrame([_sel_row()]), rec, sizes)
    assert ok["status"] == "PASS"
    rep = audit_groups.audit_groups(p, pd.DataFrame([bad]), rec, sizes)
    assert rep["status"] == "FAIL" and rep["push_blocked"]


def test_auditor_fails_on_group_sizes_30k_20k():
    p = pd.DataFrame({VAL: [f"9{i:07d}" for i in range(50)], "GROUP_KEY": ["t1c"] * 30 + ["t1f"] * 20,
                      "NATIONALITY_GROUP": "Oman"})
    rep = audit_groups.audit_groups(p, pd.DataFrame(), pd.DataFrame({"BTL_NAME": []}), {"t1c": 25, "t1f": 25})
    assert rep["push_blocked"] and next(c for c in rep["checks"] if c["id"] == "A-30")["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Budget, 2 x 2 report, registries (tests 12, 13)
# ---------------------------------------------------------------------------
def test_budget_interleaves_the_4_groups_equally():
    rng = np.random.default_rng(0)
    n = 4_000
    dec = pd.DataFrame({VAL: [f"9{i:07d}" for i in range(n)], "GROUP_KEY": np.repeat(list(config.GROUPS), n // 4),
                        "SEGMENT": rng.choice(config.S_SEGMENTS[2:], n)})
    q = llm_budget.apply_budget(llm_budget.llm_queue(dec), seconds_per_customer=1.0, budget_h=0.5)   # 1,800 of 4,000
    cov = llm_budget.coverage(q)
    assert (q["LEVEL"] == "level2_llm").sum() == 1800 and cov.max() - cov.min() <= 0.02
    for _, g in q.groupby("GROUP_KEY"):                                            # at-risk tiers first, per group
        assert g.loc[g["LEVEL"] == "level2_llm", "TIER"].max() <= g.loc[g["LEVEL"] != "level2_llm", "TIER"].min()


def test_two_by_two_effects_with_cis():
    rng = np.random.default_rng(1)
    means = {"t1c": 0.10, "t1f": 0.08, "t2c": 0.14, "t2f": 0.09, "control": 0.05}
    out = pd.concat([pd.DataFrame({"GROUP_KEY": k, "reward": rng.binomial(1, m, 20_000)}) for k, m in means.items()])
    eff = report_groups.two_by_two(out, "reward").set_index("effect")
    assert eff.loc["rules effect, constraint (T2C - T1C)", "estimate"] == pytest.approx(0.04, abs=0.01)
    assert eff.loc["universe effect, production (T1C - T1F)", "estimate"] == pytest.approx(0.02, abs=0.01)
    inter = eff.loc["interaction (T2C - T1C) - (T2F - T1F)"]
    assert inter["ci_low"] < 0.03 < inter["ci_high"] and inter["significant"]


def test_group_registries_list_every_rule_id():
    ids = {k: [r["id"] for r in rules.group_registry(k)["rules"]] for k in rules.GROUP_REGISTRIES}
    assert ids["t1f"] == [f"FR-{i:02d}" for i in range(1, 15)]
    assert ids["t2c"] == [f"CC-0{i}" for i in range(1, 7)] and ids["t2f"] == [f"CF-0{i}" for i in range(1, 7)]
    assert {"B-21", "B-28", "B-29", "B-30", "B-32"} <= set(ids["t1c"])
    assert all(rules.group_registry(k)["version"] == config.GROUPS[k]["rules_version"] for k in rules.GROUP_REGISTRIES)
