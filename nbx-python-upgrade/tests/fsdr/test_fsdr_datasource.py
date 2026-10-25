"""JsonSource reads exactly what the in-memory generator produced, in both file formats."""
from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from nbx.fsdr import schemas
from nbx.fsdr.datasource import JsonSource


@pytest.mark.parametrize("fmt", ["json", "jsonl"])
def test_round_trip_is_exact(small_source, tmp_path, fmt):
    folder = small_source.write(tmp_path / "run", fmt)
    src = JsonSource(folder)
    assert set(src.tables()) == set(schemas.TABLES)
    for t in schemas.TABLES:
        assert small_source.read(t).equals(src.read(t)), t
    assert src.manifest()["format"] == fmt and src.manifest()["source"] == "generator"


def test_format_is_detected_from_content_and_aliases_are_accepted(small_folder, tmp_path):
    folder = tmp_path / "mixed"
    shutil.copytree(small_folder, folder)
    # JSON Lines content inside a .json file, and the production typo file name
    recs = json.loads((folder / "activities_w1_score.json").read_text(encoding="utf-8"))
    (folder / "activities_w1_score.json").unlink()
    (folder / "activities_w1_score_hfds.json").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    src = JsonSource(folder)
    assert src.has("activities_w1_score") and len(src.read("activities_w1_score")) == len(recs)


def test_ints_with_nulls_stay_ints(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "consumer_aact_account.json").write_text('[{"ACCOUNT_ID": 7, "PARTY_ID": null}, {"ACCOUNT_ID": 8, "PARTY_ID": 3}]')
    df = JsonSource(folder).read("consumer_aact_account")
    assert df["PARTY_ID"].tolist() == [None, 3] and type(df["ACCOUNT_ID"].iloc[0]) is int


def test_coerce_gives_typed_columns(small_source):
    typed = schemas.coerce(small_source.read("subs_amt_auto"), "subs_amt_auto")
    assert str(typed["ACCESS_METHOD_ID"].dtype) == "Int64"
    assert typed["Sub_Amt_m1"].dtype == float
    assert pd.api.types.is_string_dtype(typed["ACCESS_METHOD_VAL"])


def test_missing_folder_and_table_raise(tmp_path, small_folder):
    with pytest.raises(FileNotFoundError):
        JsonSource(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        JsonSource(small_folder).read("no_such_table")
