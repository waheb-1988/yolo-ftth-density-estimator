"""Tests for the offline LLM backend (llms/offline_client.py): the client
dispatches to it, and every agent completes on its deterministic path."""
from __future__ import annotations

from nbx.agents import nba_agent
from nbx.agents.agentic import copilot, council
from nbx.agents.nbo_agent import run_pipeline
from nbx.core import config, insights, readiness
from nbx.llms import llm_client, offline_client


def _first_msisdn() -> int:
    return insights.customers_list()[0]["msisdn"]


def test_offline_is_the_default_backend():
    assert config.LLM_BACKENDS[0] == "offline"
    assert llm_client.backend_name() == "offline"
    assert llm_client.is_offline()
    assert llm_client.backend_url() == "in-process"


async def test_offline_backend_is_available_but_never_generates():
    assert await llm_client.is_available()
    assert await llm_client.list_installed_models() == []
    assert not llm_client.model_is_installed([], config.DEFAULT_LLM)
    assert await llm_client.resolve_model("mistral:7b") == "mistral:7b"
    res = await llm_client.generate(model="mistral:7b", prompt="hi")
    assert res["content"] == "" and res["error"] == offline_client.OFFLINE_NOTE


async def test_health_reports_ok_on_offline_backend():
    health = await readiness.compute_health()
    assert health["status"] == "ok"
    assert health["ollama_connected"] is True
    assert health["llm_backend"] == "offline"


async def test_council_runs_in_fallback_mode_without_calling_the_llm(monkeypatch):
    async def must_not_be_called(**kwargs):
        raise AssertionError("offline mode must not call generate()")

    monkeypatch.setattr(llm_client, "generate", must_not_be_called)
    decision = (await council.run_council(_first_msisdn(), "full"))["decision"]
    assert decision["mode"] == "fallback"
    assert decision["stats"]["llm_calls"] == 0
    assert decision["action"] in config.NBA_ACTIONS
    assert decision["message"]["sms_en"], "template messages still render"


async def test_nbo_pipeline_falls_back_to_cf_graph_ranking():
    result = await run_pipeline(msisdn=_first_msisdn(), retrieval_mode="graph")
    assert result["recommendations"]
    assert "offline mode" in result["llm_reasoning"]


async def test_nba_agent_uses_the_bandit_pick():
    msisdn = next(r["msisdn"] for r in insights.customers_list() if r["nba_segment"])
    out = await nba_agent.decide_action(msisdn)
    assert out["final_action"] == out["bandit_action"]


async def test_copilot_answers_in_fallback_mode():
    out = await copilot.ask("Which customers are most at risk?")
    assert out["mode"] == "fallback" and out["answer"]
