"""FSDR API routes end to end (temporary data folder and model registry)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nbx.api import fsdr as fsdr_api
from nbx.api.main import app
from nbx.fsdr import config

from .conftest import RUN_DATE


@pytest.fixture
def client(scored_run, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FSDR_DATA_DIR", scored_run["folder"].parent)
    monkeypatch.setattr(config, "FSDR_STATE_DIR", tmp_path / "state")
    fsdr_api._workspaces.clear()
    return TestClient(app)


RUN = RUN_DATE.isoformat()


def test_runs_tables_and_validation(client):
    runs = client.get("/fsdr/runs").json()
    assert runs[0]["run"] == RUN and runs[0]["scored"]
    assert len(client.get(f"/fsdr/tables/{RUN}").json()) == 46          # V1.0: + consent_contact
    prev = client.get(f"/fsdr/tables/{RUN}/btl_catalog", params={"limit": 3}).json()
    assert len(prev["records"]) == 3 and "Price data" in prev["columns"]
    v = client.post(f"/fsdr/validate/{RUN}").json()
    assert v["status"] == "ok" and "FSDR validation report" in v["markdown"]


def test_errors_are_clean(client):
    assert client.get("/fsdr/tables/not-a-date").status_code == 400
    assert client.get("/fsdr/tables/2020-01-01").status_code == 404
    assert client.post(f"/fsdr/recommend/{RUN}", json={"arm": "nope"}).status_code == 400


def test_train_score_recommend_and_registry(client):
    t = client.post(f"/fsdr/train/{RUN}", json={"passes": 1}).json()
    assert t["version"] == "v0001" and t["promoted"]
    s = client.post(f"/fsdr/score/{RUN}", json={}).json()
    assert set(s["action_distribution_per_arm"]) == {"free", "with_constraint", "llm"}
    assert client.get("/fsdr/models").json()["champion"] == "v0001"
    assert client.post("/fsdr/models/rollback").status_code == 409            # nothing to roll back to
    r = client.post(f"/fsdr/recommend/{RUN}", json={"arm": "cstr"}).json()
    assert r["with_offers"] > 0
    assert client.get(f"/fsdr/recos/{RUN}/cstr", params={"limit": 5}).json()["rows"] >= r["offers"] > 0


def test_agentic_decision_and_stream(client):
    customers = client.get(f"/fsdr/customers/{RUN}", params={"limit": 5}).json()
    v = customers[0]["msisdn"]
    d = client.post(f"/fsdr/decide/{RUN}", json={"msisdn": v}).json()["decision"]
    assert d["action"] in config.ACTIONS + [config.NO_CONTACT] and d["mode"] == "fallback"
    with client.stream("GET", f"/fsdr/stream/{RUN}", params={"msisdn": v}) as s:
        text = "".join(s.iter_text())
    assert '"type": "run_start"' in text and '"type": "decision"' in text
    assert client.post(f"/fsdr/decide/{RUN}", json={"msisdn": "1"}).status_code == 404


def test_campaign_log_feedback_and_report_routes(client):
    client.post(f"/fsdr/recommend/{RUN}", json={"arm": "cstr"})
    log = client.get("/fsdr/campaign-log", params={"arm": "cstr", "limit": 5}).json()
    assert log["total"] > 0 and len(log["rows"]) == 5 and {"propensity", "offers", "bandit_pmf"} <= set(log["rows"][0])
    assert client.get("/fsdr/campaign-log", params={"start": "bad"}).status_code == 400
    rep = client.get(f"/fsdr/report/{RUN}").json()
    assert rep["logged_per_arm"]["cstr"] > 0 and "cg" in rep["logged_per_arm"]
    assert client.get("/fsdr/report/2020-01-01").status_code == 404
    assert client.post(f"/fsdr/feedback/{RUN}").status_code == 409          # no campaign logged one week earlier


def test_upload_real_json_files_then_validate(client, scored_run):
    folder = scored_run["folder"]
    files = [("files", (p.name, p.read_bytes(), "application/json")) for p in sorted(folder.glob("*.json"))
             if not p.name.startswith("_")]
    r = client.post("/fsdr/upload/2026-11-05", files=files + [("files", ("notes.json", b"[]", "application/json"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["saved"]) == 46 and body["unknown_files"] == ["notes.json"]
    assert body["validation"]["status"] in ("ok", "warnings") and "FSDR validation report" in body["validation"]["markdown"]
    assert client.post("/fsdr/upload/2026-11-05", files=[("files", ("x.csv", b"a", "text/csv"))]).status_code == 400
    r = client.post("/fsdr/upload/2026-12-03", files=[("files", ("x.csv", b"a", "text/csv"))])
    assert r.status_code == 400 and not (config.FSDR_DATA_DIR / "2026-12-03").exists()   # nothing half-written


def test_simulate_route(client):
    r = client.post("/fsdr/simulate", json={"weeks": 1, "n": 600, "start": "2026-09-03", "arms": ["cstr"]})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["arms"] == ["cg", "cstr"] and set(rep["weekly"][0]["results"]["arms"]) == {"cg", "cstr"}


def test_operations_agent_route(client):
    r = client.post(f"/fsdr/agents/data/{RUN}")
    assert r.status_code == 200 and r.json()["final"]["decision"] == "proceed" and r.json()["mode"] == "fallback"
    assert client.post(f"/fsdr/agents/nope/{RUN}").status_code == 400
