"""V4.0: training / scoring twin folders, target lists in scoring/<D>/target1|2/a_constraint|b_free/, each target scored
separately into data/output/<D>/<target>/, the final file of both targets, and the 6 recommended offers."""
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
        assert f["ready"], f["output_checks"]
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
    assert res["final"]["ready"]
    final_df = csvstd.read_csv(out / "final" / f"input_flytxt{csvstd.SUFFIX}")
    t1 = csvstd.read_csv(out / "target1" / f"input_flytxt_target1{csvstd.SUFFIX}")
    t2 = csvstd.read_csv(out / "target2" / f"input_flytxt_target2{csvstd.SUFFIX}")
    assert len(final_df) == len(t1) + len(t2) and final_df["msisdn"].is_unique
    ctrl = csvstd.read_csv(out / "control" / f"control_{RUN_DATE.isoformat()}{csvstd.SUFFIX}")
    assert len(ctrl) == res["control"]["lines"] > 0 and not set(ctrl[VAL]) & set(final_df["msisdn"])


def test_a_blocked_target_blocks_the_push(split_run, tmp_path):
    import shutil

    out = tmp_path / "output" / RUN_DATE.isoformat()
    shutil.copytree(split_run["out"], out)
    (out / "target2" / f"input_flytxt_target2{csvstd.SUFFIX}").unlink()
    r = sg.run_final_file(RUN_DATE, out_root=tmp_path / "output")
    assert not r["ready"] and "target2" in r["reasons"][0]
    assert (out / "final" / "BLOCKED.txt").exists() and not list((out / "final").glob("input_flytxt*"))


def test_six_recommended_offers(split_run):
    assert config.RECO_OFFERS == 6 and recommended_slots() == [1, 2, 3, 4, 5, 6]
    out = split_run["out"]
    det = pd.concat([pd.read_parquet(out / t / f"details_{t}.parquet") for t in config.TARGETS])
    third = det[det["REASON_5"].eq("recommended monthly") | det["REASON_6"].eq("recommended weekly")]
    assert len(third), "no customer got a third offer of a validity"
    assert "reco_m3_btl_name" in det.columns and "reco_w3_btl_name" in det.columns
    sel = det.dropna(subset=["reco_m3_btl_name"])
    assert (sel["reco_m3_selected_by"] == selection.EXTENSION_RULE).any()
    flytxt = csvstd.read_csv(out / "final" / f"input_flytxt{csvstd.SUFFIX}")
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
    assert t1["final"]["ready"] and {g["group"] for g in t1["final"]["per_group"]} == {"t1c", "t1f"}
    early = c.post(f"/fsdr/v4/{run}/final-file").json()
    assert not early["ready"] and "target2" in early["reasons"][0]          # target 2 not scored yet
    allr = c.post(f"/fsdr/v4/{run}/run").json()
    assert allr["final"]["ready"] and allr["target2"]["ready"]
    o = c.get(f"/fsdr/v4/{run}/output").json()
    assert "final/input_flytxt.csv.gz" in o["files"] and "target1/a_constraint/input_flytxt_t1c.csv.gz" in o["files"]
    assert c.get("/fsdr/v4/2031-01-01/output").status_code == 404


def test_dummy_lists_never_overwrite_real_lists(split_run, tmp_path):
    """A real list copied over a dummy one (the old generator manifest left behind) is detected by its checksum:
    generate_targets refuses, and the lists are checked against the configured sizes, not the dummy ones."""
    import shutil

    from nbx.fsdr.pipelines.training import StageBlocked

    runs = split_run["runs"]
    sc = tmp_path / "data" / "scoring" / RUN_DATE.isoformat()
    shutil.copytree(split_run["scoring"], sc)
    assert targets.dummy_lists(sc) and not sg.real_targets(sc)
    assert targets.expected_sizes(sc) != config.group_sizes()          # dummy lists: the generator's own sizes
    real = targets.group_file(sc, "t1c")
    df = csvstd.read_csv(real)
    csvstd.write_csv(df.iloc[: len(df) // 2], real)                     # "your" list, same file name
    assert not targets.dummy_lists(sc) and sg.real_targets(sc)
    assert targets.expected_sizes(sc) == config.group_sizes()
    before = real.read_bytes()
    with pytest.raises(StageBlocked, match="real lists"):
        sg.generate_targets(RUN_DATE, runs_root=runs, cache_root=split_run["cache"], folder=sc)
    assert real.read_bytes() == before                                  # untouched
    sg.generate_targets(RUN_DATE, runs_root=runs, cache_root=split_run["cache"], folder=sc, overwrite=True)
    assert targets.dummy_lists(sc)


# ---------------------------------------------------------------------------
# V4.0 GPU notebook: one weekly folder of *_auto tables, agent 2b, agents 3 / 4 in P6
# ---------------------------------------------------------------------------
def test_week0_needs_no_score_tables(small_folder, tmp_path):
    """A weekly export of fresh *_auto tables (no *_score, no party_scoring) is a complete week-0 run."""
    import shutil

    folder = tmp_path / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    gone = {"party_scoring", "subs_amt_score"} | {f"activities_{p}_score" for p in ("w1", "w2", "w3", "w4", "m1", "m2", "m3")}
    for f in folder.rglob("*"):
        if f.is_file() and f.name.split(".")[0] in gone:
            f.unlink()
    rep = data_agent.run_t0(folder, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache")
    assert rep["pipeline"] == "train_week0" and rep["status"] == "done", rep["explanation"]
    t2 = training.run_t2(RUN_DATE, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache", workers=1)
    assert t2["anchor"] == "T" and t2["lines"] > 0


def test_agent_2b_writes_and_applies_a_new_dictionary(split_run):
    runs = split_run["runs"]
    cur = training.current_agent_dictionary(RUN_DATE, runs)
    seg = next(c for c in cur["cells"] if not c["excluded"] and c["segment"] not in
               __import__("nbx.fsdr.segmentation.actions", fromlist=["x"]).AT_RISK_SEGMENTS)
    new_action = next(a for a in config.ACTIONS if a != seg["actions"][0]["action"] and "UPSELL" not in a)
    cells = [dict(c) for c in cur["cells"]]
    for c in cells:
        if (c["segment"], c["nationality_group"]) == (seg["segment"], seg["nationality_group"]):
            c["actions"] = [{"action": new_action, "rank": 1, "score": 0.9, "reason": "test"}]
    cells.append({"segment": "Not_A_Segment", "nationality_group": "Oman", "excluded": False,
                  "actions": [{"action": "RETAIN", "rank": 1}]})                     # rejected by the check
    r = training.update_agent_dictionary(RUN_DATE, {"cells": cells}, runs)
    assert r["version"].endswith(str(cur["version"] + 1)) and r["problems"]
    assert [c["after"] for c in r["changed_cells"]] == [[new_action]]
    parts = pd.concat(pd.read_parquet(p) for p in (runs / RUN_DATE.isoformat() / "train" / "T3_segments" / "segments").glob("*.parquet"))
    cell = parts[(parts["SEGMENT"] == seg["segment"]) & (parts["NATIONALITY_GROUP"] == seg["nationality_group"])]
    assert (parts["ACTION_AGENT_VERSION"] == r["version"]).all()
    assert len(cell) == 0 or (cell["ACTION_AGENT"] == new_action).all()
    assert training.current_agent_dictionary(RUN_DATE, runs)["name"] == r["version"]


def test_agents_3_and_4_decide_target2_while_the_rules_still_filter(split_run):
    from nbx.fsdr.audit_groups import audit_groups
    from nbx.fsdr.contract.cache import CacheSource
    from nbx.fsdr.offers.recos import build_group_recos

    runs, cache = split_run["runs"], split_run["cache"]
    panel = pd.read_parquet(runs / RUN_DATE.isoformat() / "score" / "P1_panel" / "panel.parquet")
    src = CacheSource(RUN_DATE, cache)
    t2c = panel.loc[panel["GROUP_KEY"] == "t2c", VAL].tolist()
    acts = pd.DataFrame({VAL: t2c[:5] + ["99999999"], "ACTION": ["RETAIN"] * 4 + ["INVENTED", "RETAIN"],
                         "PROBA": 0.91, "DECIDED_BY": "llm_nba"})
    seen = []

    def nbo(key, cands):                                  # cheapest first, to prove the score is used
        seen.append((key, len(cands), "ANCHOR" in cands.columns))
        return 1.0 - cands["PRICE"].rank(pct=True)

    out = build_group_recos(panel, src, agents={"actions": acts, "nbo": nbo, "groups": {"t2c", "t2f"}, "label": "fake"})
    dec = out["_decisions"].set_index(VAL)
    assert (dec.loc[t2c[:4], "DECIDED_BY"] == "llm_nba").all() and (dec.loc[t2c[:4], "PROBA"] == 0.91).all()
    assert dec.loc[t2c[4], "DECIDED_BY"] != "llm_nba"                              # an invented action is refused
    assert {k for k, _, _ in seen} == {"t2c", "t2f"} and all(a for _, _, a in seen)
    sel = out["_selected"]
    t2 = sel[sel["GROUP_KEY"].isin(["t2c", "t2f"])]
    t1 = sel[sel["GROUP_KEY"].isin(["t1c", "t1f"])]
    assert t2["SCORE_SOURCE"].eq("llm:fake").all() and not t1["SCORE_SOURCE"].str.startswith("llm").any()
    assert audit_groups(panel, sel, out["_recommendable"], panel["GROUP_KEY"].value_counts().to_dict())["status"] != "FAIL"   # the rules still hold


def test_target2_only_is_the_default_deployment(split_run, tmp_path, monkeypatch):
    """V4.0 default (FSDR_TARGETS=target2, FSDR_CONTROL_SIZE=0): only T2C (a) + T2F (b), 50k each, no control;
    Target 1 and the control are never read, scored nor written, and the final file needs Target 2 only."""
    import shutil

    monkeypatch.setattr(config, "T2_CONSTRAINT_SIZE", 50_000)
    monkeypatch.setattr(config, "T2_FREE_SIZE", 50_000)
    monkeypatch.setattr(config, "CONTROL_SIZE", 0)
    config.set_targets(["target2"])
    try:
        assert config.group_sizes() == {"t2c": 50_000, "t2f": 50_000}
        assert set(config.TARGET_FILES) == {"t2c", "t2f"} and list(config.TARGETS) == ["target2"]
        runs = tmp_path / "runs"
        shutil.copytree(split_run["runs"], runs)
        sc = tmp_path / "scoring" / RUN_DATE.isoformat()
        shutil.copytree(split_run["scoring"], sc)
        for d in ("target1", "target2", "control"):
            shutil.rmtree(sc / d, ignore_errors=True)
        (sc / config.TARGETS_MANIFEST).unlink(missing_ok=True)
        gen = sg.generate_targets(RUN_DATE, runs_root=runs, cache_root=split_run["cache"], folder=sc)
        assert set(gen["sizes"]) == {"t2c", "t2f"} and gen["sizes"]["t2c"] == gen["sizes"]["t2f"]
        assert not (sc / "target1").exists() and not (sc / "control").exists()
        import json
        t0 = runs / RUN_DATE.isoformat() / "train" / "T0_load" / "summary.json"
        summ = json.loads(t0.read_text(encoding="utf-8"))
        t0.write_text(json.dumps({**summ, "scoring_source": str(sc)}), encoding="utf-8")      # the lists are here
        out = tmp_path / "output"
        res = sg.run_targets(RUN_DATE, runs_root=runs, cache_root=split_run["cache"], out_root=out)
        assert set(res) == {"P1_panel", "P6_target2", "target2", "control", "final"}
        assert res["final"]["ready"] and set(res["final"]["rows_per_target"]) == {"target2"}
        d = out / RUN_DATE.isoformat()
        assert (d / "final" / f"input_flytxt{csvstd.SUFFIX}").exists() and not (d / "target1").exists()
        assert res["control"]["lines"] == 0 and not (d / "control").exists()
        assert not (runs / RUN_DATE.isoformat() / "score" / "P6_offers_target1").exists() or \
            (split_run["runs"] / RUN_DATE.isoformat() / "score" / "P6_offers_target1").exists()   # copied, not re-made
        with pytest.raises(ValueError):
            sg.run_p6_offers(RUN_DATE, runs_root=runs, cache_root=split_run["cache"], target="target1")
    finally:
        monkeypatch.undo()
        config.set_targets(["target1", "target2"])


def test_fsdr_folders_plans_each_layout():
    """run.py fsdr-folders: the weekly folder (notebooks 03 / 04) has the *_auto history and no *_score table; the
    split layout (01 / 02) puts each table on its side; the lists follow the active targets."""
    from datetime import date

    from nbx.fsdr.cli import folder_plan

    wk = folder_plan(date(2026, 11, 19), "weekly")
    files = {r["file"] for r in wk["tables"]}
    assert "activities_w1_auto.csv.gz" in files and "btl_catalog.csv.gz" in files
    assert not any(f.startswith(("activities_w1_score", "subs_amt_score", "party_scoring")) for f in files)
    assert all(r["folder"].startswith(wk["roots"]["weekly"]) for r in wk["tables"])
    assert set(wk["lists"]) == set(config.TARGET_FILES)
    sp = folder_plan(date(2026, 11, 19), "split")
    side = {(r["file"], "training" if r["folder"].startswith(sp["roots"]["training"]) else "scoring") for r in sp["tables"]}
    assert ("base_prepaid.csv.gz", "training") in side and ("subscriptions_tg.csv.gz", "scoring") in side
    assert ("btl_catalog.csv.gz", "training") in side and ("btl_catalog.csv.gz", "scoring") in side
