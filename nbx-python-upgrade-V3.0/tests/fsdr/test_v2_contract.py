"""V2.0 Phase 2: 5 input folders, 4 file formats, customer_profile, completeness per pipeline, Parquet cache, ① T0."""
from __future__ import annotations

import json
import shutil
from datetime import date

import pandas as pd
import pytest

from nbx.fsdr import config, schemas
from nbx.fsdr.contract import completeness, data_agent
from nbx.fsdr.contract.cache import CacheSource, build_cache, line_key, line_map, partition_of, verify_cache
from nbx.fsdr.datasource import JsonSource, SyntheticSource
from nbx.fsdr.steps import Step, list_steps, read_step
from nbx.fsdr.validation import validate

from .conftest import RUN_DATE


def test_every_table_has_one_of_the_5_folders_and_the_split_of_the_plan():
    folders = {t: s.folder for t, s in schemas.TABLES.items()}
    assert set(folders.values()) == set(config.INPUT_FOLDERS)
    count = pd.Series(folders).value_counts().to_dict()
    assert count == {"usage": 18, "models": 11, "subscription": 6, "catalog": 3, "others": 9}
    assert len(schemas.TABLES) == 47 and folders[config.PROFILE_TABLE] == "others"
    listed = [t for names in config.TABLE_FOLDERS.values() for t in names]
    assert len(listed) == len(set(listed)) and set(listed) <= set(schemas.TABLES)


def test_generator_writes_the_5_folders(small_folder):
    for t, s in schemas.TABLES.items():
        assert (small_folder / s.folder / f"{t}.json").exists(), t
    assert not list(small_folder.glob("*.json*")) or [p.name for p in small_folder.glob("*.json*")] == [config.MANIFEST_FILE]
    man = json.loads((small_folder / config.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert man["layout"] == "folders" and man["folders"]["base_prepaid"] == "subscription"
    src = JsonSource(small_folder)
    assert src.layout == "folders" and not src.misplaced() and not src.duplicate_files


@pytest.mark.parametrize("fmt", ["jsonl", "csv", "parquet"])
def test_every_format_reads_back_the_same_raw_values(small_source, tmp_path, fmt, small_folder):
    ref = JsonSource(small_folder)
    other = JsonSource(small_source.write(tmp_path / RUN_DATE.isoformat(), fmt=fmt))
    assert set(other.tables()) == set(ref.tables())
    for t in ref.tables():
        a, b = ref.read(t), other.read(t)
        assert list(a.columns) == list(b.columns), t
        # JSON may hold 5 where CSV/Parquet give 5.0 in a double field, and "" where CSV gives null
        norm = lambda df: df.map(lambda v: None if v is None or v == "" else round(float(v), 6)
                                 if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        pd.testing.assert_frame_equal(norm(a), norm(b), check_dtype=False, obj=t)
    assert validate(other, RUN_DATE).status == "ok"


def test_flat_v1_layout_still_loads_with_one_warning(small_source, tmp_path):
    folder = small_source.write(tmp_path / RUN_DATE.isoformat(), layout="flat")
    src = JsonSource(folder)
    rep = validate(src, RUN_DATE)
    assert src.layout == "flat" and rep.usable and rep.layout == "flat"
    assert [i.check for i in rep.issues] == ["layout"]


def test_misplaced_table_warns_and_a_table_found_twice_blocks(small_folder, tmp_path):
    folder = tmp_path / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    (folder / "usage" / "btl_catalog.json").write_bytes((folder / "catalog" / "btl_catalog.json").read_bytes())
    shutil.move(folder / "models" / f"{config.MODEL_SCORE_REGISTRY[0].table}.json", folder / "others")
    src = JsonSource(folder)
    assert src.misplaced() == {config.MODEL_SCORE_REGISTRY[0].table: ("others", "models")}
    rep = validate(src, RUN_DATE)
    checks = {(i.check, i.table, i.level) for i in rep.issues}
    assert ("folder", config.MODEL_SCORE_REGISTRY[0].table, "warning") in checks
    assert ("file_duplicate", "btl_catalog", "blocking") in checks and not rep.usable


def test_customer_profile_agrees_with_party_and_activation_date(small_source, small_folder):
    prof = small_source.read(config.PROFILE_TABLE)
    base = small_source.read("base_prepaid")
    party = small_source.read("party_scoring").set_index("party_id")["AGE"]
    assert list(prof["msisdn"]) == list(base["ACCESS_METHOD_VAL"])
    assert list(prof["age"]) == list(base["PARTY_ID"].map(party))
    start = pd.to_datetime(base["access_method_subs_start_dt"])
    assert ((pd.Timestamp(RUN_DATE) - start).dt.days == prof["tenure"]).all()
    rep = validate(JsonSource(small_folder), RUN_DATE)
    assert rep.profile["coverage"] == 1.0
    assert all(rep.profile[k] == 0 for k in ("age_out_of_range", "negative_tenure", "age_vs_party", "tenure_vs_activation"))


def test_profile_defects_are_counted_as_warnings():
    src = SyntheticSource(n=400, run_date=RUN_DATE, scenarios="data_quality_issues", seed=11)
    inj = next(i for i in src.manifest()["quality_issues_injected"] if i["issue"] == "profile_disagreement")
    rep = validate(src, RUN_DATE)
    p = rep.profile
    assert p["age_out_of_range"] == inj["age_out_of_range"] and p["negative_tenure"] == inj["negative_tenure"]
    assert p["age_vs_party"] >= inj["age_off"] and p["tenure_vs_activation"] >= inj["tenure_off"]
    lv = {i.check: i.level for i in rep.issues if i.table == config.PROFILE_TABLE}
    assert set(lv.values()) == {"warning"} and "profile_age_out_of_range" in lv


def test_profile_is_optional_and_its_absence_is_a_warning(small_folder, tmp_path):
    folder = tmp_path / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    (folder / "others" / f"{config.PROFILE_TABLE}.json").unlink()
    src = JsonSource(folder)
    rep = validate(src, RUN_DATE)
    assert rep.usable and any(i.table == config.PROFILE_TABLE and i.level == "warning" for i in rep.issues)
    comp = completeness.check(src, "train_weekly", RUN_DATE)
    assert comp["complete"] and config.PROFILE_TABLE in comp["warnings"]


def test_completeness_per_pipeline(small_folder, tmp_path):
    src = JsonSource(small_folder)
    assert completeness.detect_pipeline(src) == "train_week0"
    for p in completeness.PIPELINES:
        assert completeness.check(src, p, RUN_DATE)["complete"], p
    # week n does not need the _auto tables; scoring does not need the base (it comes from the bundle)
    folder = tmp_path / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    for t in ("activities_w1_auto", "subs_amt_auto", "base_prepaid"):
        (folder / schemas.TABLES[t].folder / f"{t}.json").unlink()
    src = JsonSource(folder)
    assert completeness.detect_pipeline(src) == "train_weekly"
    assert completeness.check(src, "train_week0", RUN_DATE)["blocking"] == ["base_prepaid", "subs_amt_auto", "activities_w1_auto"]
    assert completeness.check(src, "train_weekly", RUN_DATE)["blocking"] == ["base_prepaid"]
    assert completeness.check(src, "score", RUN_DATE)["complete"]
    stale = completeness.check(JsonSource(small_folder), "score", date(2026, 10, 28))
    assert not stale["complete"] and not stale["fresh"] and "stale" in stale["reasons"][0]


def test_cache_partitions_keep_every_line_in_one_partition(small_source, tmp_path):
    man = build_cache(small_source, RUN_DATE, tmp_path, parts=4)
    c = CacheSource(RUN_DATE, tmp_path)
    assert c.partitions == 4 and verify_cache(RUN_DATE, tmp_path) == []
    assert not c.partitioned("btl_catalog") and not c.partitioned("party_scoring") and c.partitioned("targeting_auto")
    lines = line_map(small_source)
    where = {}
    for t in ("base_prepaid", "activities_w1_score", "subs_btl_auto", config.PROFILE_TABLE,
              config.MODEL_SCORE_REGISTRY[0].table, "location_auto"):
        assert len(c.read(t)) == man["tables"][t]["rows"] == len(small_source.read(t))
        for k in range(4):
            key = line_key(c.read(t, part=k), lines)
            for v in key.dropna().unique():
                assert where.setdefault(v, k) == k, (t, v)
    keys = pd.Series(["1001", "1002", "1001"])
    assert list(partition_of(keys, 16)) == list(partition_of(keys, 16)) and partition_of(keys, 16)[0] == partition_of(keys, 16)[2]


def test_cache_detects_a_changed_file(small_source, tmp_path):
    build_cache(small_source, RUN_DATE, tmp_path, parts=2, tables=["btl_catalog", "base_prepaid"])
    p = tmp_path / RUN_DATE.isoformat() / "btl_catalog" / "part-all.parquet"
    p.write_bytes(p.read_bytes() + b"x")
    assert verify_cache(RUN_DATE, tmp_path) == [str(p.relative_to(tmp_path / RUN_DATE.isoformat()))]


def test_t0_saves_its_outputs_and_resumes(small_folder, tmp_path):
    rep = data_agent.run_t0(small_folder, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache")
    assert rep["status"] == "done" and rep["pipeline"] == "train_week0" and rep["tables"] == 47
    step = tmp_path / "runs" / RUN_DATE.isoformat() / "train" / "T0_load"
    meta = read_step(step)
    assert meta["status"] == "done" and meta["rows"]["tables"] == 47
    for f in ("validation_report.json", "validation_report.md", "completeness.csv", "completeness.json",
              "cache_manifest.json", "summary.json"):
        assert (step / f).exists() and f in meta["outputs"], f
    assert rep["cache"]["partitions"] == config.CACHE_PARTITIONS
    again = data_agent.run_t0(small_folder, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache")
    assert again["skipped"] and read_step(step)["started"] == meta["started"]
    forced = data_agent.run_t0(small_folder, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache", force=True)
    assert "skipped" not in forced
    assert [s["step"] for s in list_steps(tmp_path / "runs" / RUN_DATE.isoformat())] == ["T0_load"]


def test_t0_blocks_on_a_missing_required_table_and_builds_no_cache(small_folder, tmp_path):
    folder = tmp_path / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    (folder / "catalog" / "btl_catalog.json").unlink()
    rep = data_agent.run_t0(folder, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache")
    assert rep["status"] == "blocked" and "BLOCKED" in rep["explanation"] and "btl_catalog" in rep["explanation"]
    meta = read_step(tmp_path / "runs" / RUN_DATE.isoformat() / "train" / "T0_load")
    assert meta["status"] == "blocked" and "btl_catalog" in meta["reason"]
    assert not (tmp_path / "cache").exists()


def test_step_marks_a_failure_and_reruns_it(tmp_path):
    with pytest.raises(RuntimeError):
        with Step(tmp_path, "T2_features", inputs="a") as st:
            st.write_json("x.json", {})
            raise RuntimeError("boom")
    assert read_step(tmp_path / "T2_features")["status"] == "failed"
    with Step(tmp_path, "T2_features", inputs="a") as st:
        assert not st.done
        st.write_table(pd.DataFrame({"a": range(5)}), "base_features")
    assert Step(tmp_path, "T2_features", inputs="a").done and not Step(tmp_path, "T2_features", inputs="b").done
    assert (tmp_path / "T2_features" / "base_features.csv").exists()
    with Step(tmp_path, "T3_segments", inputs="a") as st:
        big = pd.DataFrame({"a": range(config.STEP_FULL_CSV_MAX_ROWS + 1)})
        st.write_table(big, "segments")
    assert len(pd.read_csv(tmp_path / "T3_segments" / "segments_preview.csv")) == config.STEP_PREVIEW_ROWS


def test_msisdns_stay_8_digits_at_1m_and_2m():
    import numpy as np

    from nbx.fsdr.synth.world import msisdns

    for n in (2_000, 1_000_000, 2_000_000):
        m = msisdns(np.random.default_rng(1), n).astype(np.int64)
        assert len(np.unique(m)) == n and m.min() > 69_999_999 and m.max() <= 99_999_999, n
