"""V4.0: training / scoring twin folders, target lists in scoring/<D>/target1|2/a_constraint|b_free/, each target scored
separately into data/output/<D>/<target>/, the push of both targets, and the 6 recommended offers."""
from __future__ import annotations

import pandas as pd
import pytest

from nbx.fsdr import config, csvstd, targets
from nbx.fsdr.contract import data_agent
from nbx.fsdr.datasource import JsonSource
from nbx.fsdr.final_output import recommended_slots
from nbx.fsdr.offers import selection
from nbx.fsdr.pipelines import scoring_groups as sg
from nbx.fsdr.pipelines import training
from nbx.fsdr.synth import generator

from .conftest import RUN_DATE

VAL = "ACCESS_METHOD_VAL"


@pytest.fixture(scope="module")
def split_run(small_source, tmp_path_factory):
    root = tmp_path_factory.mktemp("v4 split")
    tr = root / "data" / "training" / RUN_DATE.isoformat()
    generator.write_run(small_source.run, tr, "csv.gz", "split")
    sc = root / "data" / "scoring" / RUN_DATE.isoformat()
    runs, cache, out = root / "data" / "runs", root / "data" / "cache", root / "data" / "output"
    t0 = data_agent.run_t0(tr, runs_root=runs, cache_root=cache)
    training.run_features(RUN_DATE, runs_root=runs, cache_root=cache, workers=1)
    gen = sg.generate_targets(RUN_DATE, runs_root=runs, cache_root=cache)
    res = sg.run_targets(RUN_DATE, runs_root=runs, cache_root=cache, out_root=out)
    return {"training": tr, "scoring": sc, "runs": runs, "cache": cache, "t0": t0, "gen": gen, "res": res,
            "out": out / RUN_DATE.isoformat()}


def test_the_generator_splits_training_and_scoring_with_the_same_folders(split_run):
    tr, sc = split_run["training"], split_run["scoring"]
    for root in (tr, sc):
        assert all((root / f).is_dir() for f in config.INPUT_FOLDERS)
        assert (root / config.MANIFEST_FILE).exists()
    assert (sc / "subscription" / "subscriptions_tg.csv.gz").exists()
    assert not (tr / "subscription" / "subscriptions_tg.csv.gz").exists()
    assert (tr / "subscription" / "base_prepaid.csv.gz").exists() and not (sc / "subscription" / "base_prepaid.csv.gz").exists()
    assert (tr / "catalog" / "btl_catalog.csv.gz").exists() and (sc / "catalog" / "btl_catalog.csv.gz").exists()
    src = JsonSource(tr)
    assert src.side_of["btl_catalog"] == "scoring" and src.side_of["base_prepaid"] == "training"
    assert not src.duplicate_files and not src.misfiled()


def test_t0_reads_both_twins_as_one_run(split_run):
    t0 = split_run["t0"]
    assert t0["status"] == "done"
    assert t0["training_source"] == str(split_run["training"]) and t0["scoring_source"] == str(split_run["scoring"])
    assert sg.input_folder(RUN_DATE, split_run["runs"]) == split_run["scoring"]


def test_a_scoring_table_filed_under_training_is_a_warning(small_source, tmp_path):
    tr = tmp_path / "training" / RUN_DATE.isoformat()
    generator.write_run(small_source.run, tr, "json", "split")
    sc = tmp_path / "scoring" / RUN_DATE.isoformat()
    (sc / "others" / "targeting_auto.json").rename(tr / "others" / "targeting_auto.json")
    src = JsonSource(tr)
    assert src.misfiled() == {"targeting_auto": ("training", "scoring")} and src.has("targeting_auto")


def test_target_lists_live_in_target_and_group_folders(split_run):
    sc = split_run["scoring"]
    for key, rel in config.TARGET_FILES.items():
        assert (sc / f"{rel}.csv.gz").exists(), rel
    assert (sc / "target1" / "a_constraint" / "target1_a.csv.gz").exists()
    assert (sc / "target2" / "b_free" / "target2_b.csv.gz").exists()
    assert targets.has_targets(sc) and split_run["res"]["P1_panel"]["mode"] == "uploaded"


def test_each_target_is_scored_separately(split_run):
    res, out = split_run["res"], split_run["out"]
    for t, keys in config.TARGETS.items():
        p6 = res[f"P6_{t}"]
        assert p6["stage"] == f"P6_offers_{t}" and [g["group"] for g in p6["groups"]] == keys
        own = {config.GROUPS[k][x] for k in keys for x in ("recos_monthly", "recos_weekly", "scored_table")}
        assert set(p6["tables"]) == own | {"results_scoring_nba"}
        f = res[t]
        assert f["push_ready"], f["output_checks"]
        flytxt = csvstd.read_csv(out / t / f"input_flytxt_{t}{csvstd.SUFFIX}")
        assert list(flytxt.columns) == config.INPUT_FLYTXT_COLUMNS
        parts = []
        for k in keys:
            sub = config.GROUP_SUBFOLDER[k]
            parts.append(csvstd.read_csv(out / t / sub / f"input_flytxt_{k}{csvstd.SUFFIX}"))
            det = csvstd.read_csv(out / t / sub / f"details_{k}{csvstd.SUFFIX}")
            assert set(det["GROUP_KEY"]) == {k}
        assert sorted(pd.concat(parts)["msisdn"]) == sorted(flytxt["msisdn"])
        det = pd.read_parquet(out / t / f"details_{t}.parquet")
        assert set(det["GROUP_KEY"]) == set(keys) and det[VAL].is_unique
        assert (out / t / "summary.json").exists() and (out / t / "output_checks.json").exists()


def test_the_push_merges_both_targets_and_the_control_is_apart(split_run):
    res, out = split_run["res"], split_run["out"]
    assert res["push"]["push_ready"]
    push = csvstd.read_csv(out / "push" / f"input_flytxt{csvstd.SUFFIX}")
    t1 = csvstd.read_csv(out / "target1" / f"input_flytxt_target1{csvstd.SUFFIX}")
    t2 = csvstd.read_csv(out / "target2" / f"input_flytxt_target2{csvstd.SUFFIX}")
    assert len(push) == len(t1) + len(t2) and push["msisdn"].is_unique
    ctrl = csvstd.read_csv(out / "control" / f"control_{RUN_DATE.isoformat()}{csvstd.SUFFIX}")
    assert len(ctrl) == res["control"]["lines"] > 0 and not set(ctrl[VAL]) & set(push["msisdn"])


def test_a_blocked_target_blocks_the_push(split_run, tmp_path):
    import shutil

    out = tmp_path / "output" / RUN_DATE.isoformat()
    shutil.copytree(split_run["out"], out)
    (out / "target2" / f"input_flytxt_target2{csvstd.SUFFIX}").unlink()
    r = sg.run_push(RUN_DATE, out_root=tmp_path / "output")
    assert not r["push_ready"] and "target2" in r["reasons"][0]
    assert (out / "push" / "BLOCKED.txt").exists() and not list((out / "push").glob("input_flytxt*"))


def test_six_recommended_offers(split_run):
    assert config.RECO_OFFERS == 6 and recommended_slots() == [1, 2, 3, 4, 5, 6]
    out = split_run["out"]
    det = pd.concat([pd.read_parquet(out / t / f"details_{t}.parquet") for t in config.TARGETS])
    third = det[det["REASON_5"].eq("recommended monthly") | det["REASON_6"].eq("recommended weekly")]
    assert len(third), "no customer got a third offer of a validity"
    assert "reco_m3_btl_name" in det.columns and "reco_w3_btl_name" in det.columns
    sel = det.dropna(subset=["reco_m3_btl_name"])
    assert (sel["reco_m3_selected_by"] == selection.EXTENSION_RULE).any()
    flytxt = csvstd.read_csv(out / "push" / f"input_flytxt{csvstd.SUFFIX}")
    n = flytxt[[f"destination_offer_name_{i}" for i in range(1, 7)]].ne("").sum(axis=1)
    assert n.max() == 6 and n.min() >= 1


def test_extension_never_downgrades_and_never_repeats():
    c = pd.DataFrame({VAL: ["1"] * 5, "BTL_NAME": list("ABCDE"), "SERVICE": ["DATA", "COMBO", "DATA", "DATA", "COMBO"],
                      "PRICE": [5.0, 6.0, 5.0, 2.0, 7.0], "SCORE": [0.9, 0.8, 0.7, 0.99, 0.6], "ANCHOR": 4.0})
    sel = c.iloc[:2].assign(PASS=1, SELECTED_BY="B-28")
    ext = selection.extend(c, sel, 3)
    added = ext[ext["PASS"] == selection.EXTENSION_PASS]
    # C: same service + price as A (F-09 would clear it); D: below the anchor; E is the third offer
    assert added["BTL_NAME"].tolist() == ["E"] and len(ext) == 3
    assert selection.extend(c, sel.iloc[:0], 3).empty


def test_four_offers_mode_keeps_the_v3_file(monkeypatch):
    monkeypatch.setattr(config, "RECOS_PER_VALIDITY", 2)
    assert recommended_slots() == [1, 2, 3, 4]


def test_v4_api_routes_end_to_end(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from nbx.api.main import app

    for name, sub in (("FSDR_TRAINING_DIR", "training"), ("FSDR_SCORING_DIR", "scoring"), ("FSDR_OUTPUT_DIR", "output"),
                      ("FSDR_RUNS_DIR", "runs"), ("FSDR_CACHE_DIR", "cache"), ("FSDR_INPUT_DIR", "input")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    c = TestClient(app)
    run = RUN_DATE.isoformat()
    g = c.post(f"/fsdr/v4/{run}/generate", json={"n": 400}).json()
    from pathlib import Path

    assert Path(g["training"]) == tmp_path / "training" / run and Path(g["scoring"]) == tmp_path / "scoring" / run
    prep = c.post(f"/fsdr/v4/{run}/prepare").json()
    assert prep["t0"]["status"] == "done" and prep["features"]["lines"] > 0
    assert c.post(f"/fsdr/v4/{run}/targets/generate").status_code == 200
    assert c.post(f"/fsdr/v4/{run}/score/target9").status_code == 400
    assert c.post(f"/fsdr/v4/{run}/score/target1").status_code == 409            # no panel yet
    assert c.post(f"/fsdr/v4/{run}/panel").json()["status"] == "done"
    t1 = c.post(f"/fsdr/v4/{run}/score/target1").json()
    assert t1["final"]["push_ready"] and {g["group"] for g in t1["final"]["per_group"]} == {"t1c", "t1f"}
    early = c.post(f"/fsdr/v4/{run}/push").json()
    assert not early["push_ready"] and "target2" in early["reasons"][0]          # target 2 not scored yet
    allr = c.post(f"/fsdr/v4/{run}/run").json()
    assert allr["push"]["push_ready"] and allr["target2"]["push_ready"]
    o = c.get(f"/fsdr/v4/{run}/output").json()
    assert "push/input_flytxt.csv.gz" in o["files"] and "target1/a_constraint/input_flytxt_t1c.csv.gz" in o["files"]
    assert c.get("/fsdr/v4/2031-01-01/output").status_code == 404
