"""End-to-end smoke test of the FastAPI app on the offline backend: every
read endpoint answers, and the recommendation endpoints return complete
decisions. Endpoints that rewrite the data (/seed, /upload) or scrape the web
(/competitors/refresh) are covered by their own tests, not here."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nbx.api import main
from nbx.core import config, insights

client = TestClient(main.app)
MSISDN = insights.customers_list()[0]["msisdn"]
OFFER = main._load_json(config.OFFERS_FILE)[0]["program_rule"]


@pytest.mark.parametrize("path", [
    "/health", "/models", "/embeddings", "/vectorstores", "/offers", "/customers",
    f"/customer/{MSISDN}", f"/customers/{MSISDN}/360", "/portfolio/summary",
    "/graph/stats", f"/graph/cohort-offers/{MSISDN}", f"/graph/similar-customers/{MSISDN}",
    "/cf/stats", f"/cf/score/{MSISDN}", "/nba/bandit/stats", f"/nba/bandit/context/{MSISDN}",
    "/rules", "/logs", "/agentic/recent", "/competitors/offers", f"/competitors/compare/{OFFER}",
])
def test_read_endpoints_answer(path):
    resp = client.get(path)
    assert resp.status_code == 200, resp.text


def test_health_is_ok_offline():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_backend"] == "offline"


def test_unknown_customer_is_404():
    assert client.get("/customer/999999999").status_code == 404


def test_recommend_returns_offers():
    body = client.post("/recommend", json={"msisdn": MSISDN, "retrieval_mode": "graph"}).json()
    assert body["recommendations"], body


def test_nba_decide_returns_an_action():
    resp = client.post("/nba/decide", json={"msisdn": MSISDN, "retrieval_mode": "graph"})
    assert resp.status_code == 200, resp.text


def test_nba_action_uses_the_bandit():
    body = client.post("/nba/action", json={"msisdn": MSISDN}).json()
    assert body["final_action"] in config.NBA_ACTIONS


@pytest.mark.parametrize("scope", ["nba", "full"])
def test_agentic_decide(scope):
    body = client.post("/agentic/decide", json={"msisdn": MSISDN, "scope": scope}).json()
    decision = body["decision"]
    assert decision["mode"] == "fallback"
    assert decision["action"] in config.NBA_ACTIONS
    assert decision["explanation"]


def test_agentic_stream_ends_with_a_decision():
    with client.stream("GET", "/agentic/stream", params={"msisdn": MSISDN, "scope": "nba"}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "run_end" in text


def test_copilot_chat():
    body = client.post("/copilot/chat", json={"question": "How many customers are at high risk?"}).json()
    assert body["answer"] and body["mode"] == "fallback"
