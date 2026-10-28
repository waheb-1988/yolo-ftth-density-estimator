"""Step-by-step journey (nbx/fsdr/pipelines/journey.py): ingest the dummy data, then every step for one line and a batch."""
from __future__ import annotations

import json

import pytest

from nbx.fsdr import config
from nbx.fsdr.pipelines import journey


@pytest.fixture(scope="module")
def ingested(small_folder, tmp_path_factory):
    """The small generated run used as the dummy-data source, ingested into a private data root."""
    root = tmp_path_factory.mktemp("journey data")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "FSDR_DATA_DIR", root / "fsdr")
    mp.setattr(config, "FSDR_STATE_DIR", root / "fsdr" / "state")
    res = journey.ingest(small_folder)
    yield res
    mp.undo()


@pytest.fixture(scope="module")
def jws(ingested):
    from pathlib import Path

    from nbx.fsdr.agents.tools import FsdrWorkspace

    return FsdrWorkspace(Path(ingested["destination"]), Path(ingested["destination"]).parent / "state" / "models")


def test_ingest_copies_the_same_bytes_and_every_table_is_green(ingested, small_folder):
    assert ingested["same_data"] and ingested["usable"]
    assert ingested["tables_copied"] == ingested["tables_in_contract"] == 46
    assert {t["status"] for t in ingested["tables"]} == {"green"}
    for t in ingested["tables"]:
        assert (small_folder / f"{t['table']}.json").exists() and t["identical_copy"]


def test_joins_and_recomputed_calculations_are_green(ingested):
    assert all(j["status"] != "red" for j in ingested["joins"])
    keys = [j for j in ingested["joins"] if "->" in j["join"] and ("ACCOUNT_ID" in j["join"] or "PARTY_ID" in j["join"])]
    assert keys and all(j["match_rate"] == 1.0 for j in keys)
    assert len(ingested["calculations"]) == 7
    assert all(c["status"] == "green" and c["mismatches"] == 0 and c["lines_checked"] > 0 for c in ingested["calculations"])
    assert ingested["training"]["features"] == 61 and ingested["scoring"]["scored_lines"] > 0


def test_a_reloaded_page_continues_on_the_last_upload(ingested, tmp_path):
    from pathlib import Path

    rep = journey.last_ingest(Path(ingested["destination"]).parent)
    assert rep["run"] == ingested["run"] and rep["lines"] == ingested["lines"]
    assert journey.last_ingest(tmp_path) is None


def test_a_missing_optional_table_is_grey_not_red(small_folder, tmp_path, monkeypatch):
    import shutil

    src = tmp_path / small_folder.name
    shutil.copytree(small_folder, src)
    (src / f"{config.CONSENT_TABLE}.json").unlink()
    monkeypatch.setattr(config, "FSDR_DATA_DIR", tmp_path / "fsdr")
    res = journey.ingest(src, prepare=False)
    st = {t["table"]: t["status"] for t in res["tables"]}
    assert st[config.CONSENT_TABLE] == "grey" and res["usable"]


async def test_every_step_for_one_line_continues_from_the_previous(jws, ingested):
    v = ingested["lines"]["llm"][0]
    act = (await journey.run_step(jws, "action", [v]))["results"][0]
    assert act["action"] in config.ACTIONS + [config.NO_CONTACT] and act["explanation"] and "llm_answer" in act
    off = (await journey.run_step(jws, "offers", [v]))["results"][0]
    assert off["action"] == act["action"] and len(off["offers"]) <= 4
    add = (await journey.run_step(jws, "addons", [v]))["results"][0]
    assert len(add["roaming"]) <= config.ADDON_TOP_ROAMING
    comp = (await journey.run_step(jws, "competitors", [v]))["results"][0]
    assert comp["final_offers"] == [o["offer"] for o in off["offers"]] or comp["swaps_applied"]
    out = (await journey.run_step(jws, "output", [v]))["results"][0]
    saved = json.loads(open(out["file"], encoding="utf-8").read())
    assert saved["decision"]["action"] == act["action"] and set(saved["steps"]) >= {"action", "offers", "addons", "competitors"}
    assert "base_prepaid" in saved["context"]["raw_tables"] and saved["context"]["engine"]["features"]["ACCESS_METHOD_VAL"] == v
    assert out["summary"]["action"] == act["action"] and any(k.startswith("feat_") for k in out["summary"])


async def test_a_later_step_runs_the_missing_earlier_ones_and_batch_reports_bad_lines(jws, ingested):
    vals = ingested["lines"]["cstr"][:2] + ["99999999"]
    res = await journey.run_step(jws, "competitors", vals)
    assert res["count"] == 3 and res["errors"] == 1 and "not in the scored sample" in res["results"][2]["error"]
    st = journey.load_state(jws, vals[0])
    assert set(st["steps"]) == {"action", "offers", "addons", "competitors"}
    assert res["files"]["csv"].endswith(".csv")


def test_parse_msisdns_from_csv_or_plain_list():
    assert journey.parse_msisdns("msisdn,x\n70000001,a\n70000002,b\n70000001,c\n") == ["70000001", "70000002"]
    assert journey.parse_msisdns("70000003\r\n70000004.0\n\nabc\n") == ["70000003", "70000004"]
    assert journey.parse_msisdns("name,access_method_val\nx,70000005\n") == ["70000005"]


def test_journey_api_refuses_a_source_outside_the_dummy_folder():
    from fastapi.testclient import TestClient

    from nbx.api.main import app

    c = TestClient(app)
    assert c.post("/fsdr/journey/ingest", json={"source": "C:/Windows"}).status_code == 400
    assert c.post("/fsdr/journey/2026-10-21/nope", json={"msisdns": ["1"]}).status_code == 400
