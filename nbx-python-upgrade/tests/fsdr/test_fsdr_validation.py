"""Validation report: clean data passes, each kind of contract break is graded correctly."""
from __future__ import annotations

import json
import shutil
from datetime import date

import pytest

from nbx.fsdr import config
from nbx.fsdr.datasource import JsonSource
from nbx.fsdr.validation import validate, validate_folder


@pytest.fixture
def folder(small_folder, tmp_path):
    dst = tmp_path / small_folder.name
    shutil.copytree(small_folder, dst)
    return dst


def _rewrite(folder, table, fn):
    p = folder / f"{table}.json"
    recs = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(fn(recs)), encoding="utf-8")


def test_clean_generated_folder_is_ok_and_reports_are_written(folder):
    rep = validate_folder(folder)
    assert rep.status == "ok" and rep.usable, [i.message for i in rep.issues]
    assert (folder / config.VALIDATION_JSON).exists() and (folder / config.VALIDATION_MD).exists()
    assert json.loads((folder / config.VALIDATION_JSON).read_text(encoding="utf-8"))["status"] == "ok"
    assert len(rep.coverage) == len(config.MODEL_SCORE_REGISTRY)
    assert all(r["orphans"] == 0 for r in rep.relations)


def test_missing_model_table_blocks_but_missing_unused_table_only_warns(folder):
    (folder / "subs_amt_auto.json").unlink()
    (folder / "location_auto.json").unlink()
    (folder / "REV_DECREASE_204_SCORING_TD.predictions.json").unlink()
    rep = validate_folder(folder)
    lv = {i.table: i.level for i in rep.issues if i.check == "file_present"}
    assert lv == {"subs_amt_auto": "blocking", "location_auto": "warning", "REV_DECREASE_204_SCORING_TD.predictions": "warning"}
    assert rep.status == "blocked"


def test_missing_field_blocks_and_unknown_field_warns(folder):
    _rewrite(folder, "subs_amt_score", lambda rs: [{k: v for k, v in r.items() if k != "Sub_Amt_w2"} for r in rs])
    _rewrite(folder, "party_auto", lambda rs: [{**r, "MARITAL_STATUS": "x"} for r in rs])
    rep = validate_folder(folder)
    levels = {(i.check, i.table): i.level for i in rep.issues}
    assert levels[("fields_missing", "subs_amt_score")] == "blocking"
    assert levels[("fields_unknown", "party_auto")] == "warning"


def test_wrong_type_outside_keys_is_only_a_warning(folder):
    _rewrite(folder, "subs_amt_score", lambda rs: [{**r, "Sub_Amt_w1": str(r["Sub_Amt_w1"])} for r in rs])
    rep = validate_folder(folder)
    issue = next(i for i in rep.issues if i.check == "raw_type")
    assert issue.level == "warning" and issue.column == "Sub_Amt_w1"


def test_orphan_keys_across_tables_block(folder):
    _rewrite(folder, "targeting_dataiku_auto", lambda rs: [{**r, "access_method_id": r["access_method_id"] + 10**9} for r in rs])
    rep = validate_folder(folder)
    assert any(i.check == "cross_table" and i.table == "targeting_dataiku_auto" and i.level == "blocking" for i in rep.issues)


def test_fulfilment_lines_unknown_to_the_base_are_reported(folder):
    _rewrite(folder, "subs_btl_auto", lambda rs: [{**r, "access_method_val": "79999999"} if i < 3 else r for i, r in enumerate(rs)])
    rep = validate_folder(folder)
    hit = [i for i in rep.issues if i.check == "cross_table" and i.table == "subs_btl_auto"]
    assert hit and hit[0].level == "warning" and not any(i.level == "blocking" and i.table == "subs_btl_auto" for i in rep.issues)   # reported, not blocking


def test_manifest_anchor_mismatch_blocks_and_missing_manifest_warns(folder):
    m = json.loads((folder / config.MANIFEST_FILE).read_text(encoding="utf-8"))
    m["anchor_date"] = "2026-10-21"
    (folder / config.MANIFEST_FILE).write_text(json.dumps(m), encoding="utf-8")
    assert any(i.check == "manifest" and i.level == "blocking" for i in validate_folder(folder).issues)
    (folder / config.MANIFEST_FILE).unlink()
    rep = validate_folder(folder)                       # run date then comes from the folder name
    assert rep.run_date == folder.name and any(i.check == "manifest" and i.level == "warning" for i in rep.issues)


def test_dates_outside_their_business_windows_warn(folder):
    _rewrite(folder, "targeting_dataiku_auto", lambda rs: [{**r, "date1": "2026-01-01"} for r in rs])
    rep = validate(JsonSource(folder), date.fromisoformat(folder.name))
    assert any(i.check == "date_window" and i.table == "targeting_dataiku_auto" for i in rep.issues)


def test_markdown_report_lists_issues(folder):
    _rewrite(folder, "party_auto", lambda rs: [{**r, "EXTRA": 1} for r in rs])
    md = validate_folder(folder).to_markdown()
    assert "FSDR validation report" in md and "fields_unknown" in md and "Model-score coverage" in md
