"""Data, training, scoring and monitor agents (audit L1): fallback paths and code-enforced guards."""
from __future__ import annotations

import asyncio
import json
import shutil

import pytest

from nbx.fsdr import config, registry
from nbx.fsdr.agents.ops_agents import OPS_AGENTS, run_ops_agent
from nbx.llms import llm_client

MARKERS = {"data": "data steward", "training": "model owner", "scoring": "scoring operator", "monitor": "campaign analyst"}


class FakeLLM:
    def __init__(self, scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}

    async def generate(self, model, prompt, system=None, force_json=False, timeout=120.0):
        kind = next((k for k, m in MARKERS.items() if m in (system or "")), "other")
        q = self.scripts.get(kind) or [{"thought": "done", "final": {}}]
        reply = q.pop(0) if len(q) > 1 else q[0]
        return {"content": json.dumps(reply), "duration_ms": 1, "model": model, "error": None}


@pytest.fixture
def fake_llm(monkeypatch):
    def install(scripts):
        from nbx.core import config as nbx_config

        async def yes():
            return True

        async def resolve(model):
            return "fake-model"

        monkeypatch.setattr(nbx_config, "LLM_BACKEND", "ollama")
        monkeypatch.setattr(llm_client, "is_available", yes)
        monkeypatch.setattr(llm_client, "resolve_model", resolve)
        monkeypatch.setattr(llm_client, "generate", FakeLLM(scripts).generate)
    return install


@pytest.fixture
def run(scored_run, tmp_path):
    """A private copy of the scored run and its model registry (the agents train and score)."""
    folder = tmp_path / scored_run["folder"].name
    shutil.copytree(scored_run["folder"], folder)
    models = tmp_path / "models"
    shutil.copytree(scored_run["root"], models)
    return folder, models


def _run(kind, folder, models):
    return asyncio.run(run_ops_agent(kind, folder, models_root=models))


def _step(tool, args=None):
    return {"thought": f"call {tool}", "tool": tool, "args": args or {}}


@pytest.mark.parametrize("kind", OPS_AGENTS)
def test_every_agent_has_a_rule_based_fallback(run, kind):
    folder, models = run
    out = _run(kind, folder, models)
    assert out["mode"] == "fallback" and out["finalized_by"] == "policy" and out["final"]
    tools = [e["tool"] for e in out["trace"] if e["type"] == "tool_call"]
    assert tools                                                     # the fallback uses the same tools
    if kind == "data":
        assert out["final"]["decision"] == "proceed" and "validate_run" in tools
    if kind == "training":
        ok, _ = registry.can_promote(out["final"]["challenger"], models)
        assert out["final"]["decision"] == ("promote" if ok else "keep") and out["final"]["justification"]
    if kind == "scoring":
        assert out["final"]["decision"] == "scored"
    if kind == "monitor":
        assert out["final"]["summary"].startswith("No rewards yet") or out["final"]["summary"].startswith("Week")


def test_data_agent_cannot_proceed_on_blocked_data(run, fake_llm):
    folder, models = run
    (folder / "subs_amt_auto.json").unlink()                                  # a required input table missing: blocked
    fake_llm({"data": [_step("validate_run"), {"thought": "looks fine", "final": {"decision": "proceed", "reason": "ok"}}]})
    out = _run("data", folder, models)
    assert out["mode"] == "agentic" and out["final"]["decision"] == "stop" and "blocked" in out["findings"][0]


def test_training_agent_stays_in_range_and_cannot_force_a_promotion(run, fake_llm, monkeypatch):
    folder, models = run
    monkeypatch.setattr(registry, "can_promote", lambda v, root=None: (False, "IPS 0.01 < champion 0.05"))
    fake_llm({"training": [_step("training_options"), _step("train_challenger", {"epsilon": 0.9}),
                           _step("train_challenger", {"epsilon": 0.2, "passes": 2}),
                           {"thought": "better", "final": {"decision": "promote", "justification": "looks better"}}]})
    before = registry.champion(models)
    out = _run("training", folder, models)
    results = [e for e in out["trace"] if e["type"] == "tool_result" and e["tool"] == "train_challenger"]
    assert "error" in results[0]["result"] and "epsilon" in results[0]["result"]["error"]    # out of range: refused
    assert out["final"]["decision"] == "keep" and out["findings"] and registry.champion(models) == before


def test_training_agent_promotes_when_the_rule_allows_and_writes_why(run, fake_llm, monkeypatch):
    folder, models = run
    monkeypatch.setattr(registry, "can_promote", lambda v, root=None: (True, "IPS 0.06 >= champion 0.05"))
    fake_llm({"training": [_step("training_options"), _step("train_challenger"),
                           {"thought": "ok", "final": {"decision": "promote", "justification": "IPS above the champion"}}]})
    out = _run("training", folder, models)
    assert out["final"]["decision"] == "promote" and registry.champion(models) == out["final"]["challenger"]
    assert "IPS above the champion" in registry.models(models)["history"][-1]["reason"]


def test_scoring_agent_cannot_score_with_a_stale_model(run, fake_llm, monkeypatch):
    folder, models = run
    monkeypatch.setattr(config, "MAX_MODEL_AGE_DAYS", -1)                       # every model is stale
    fake_llm({"scoring": [_step("model_status"), _step("score"),
                          {"thought": "done", "final": {"decision": "scored", "reason": "fine", "drift_alerts": []}}]})
    out = _run("scoring", folder, models)
    res = [e for e in out["trace"] if e["type"] == "tool_result" and e["tool"] == "score"]
    assert "refused" in res[0]["result"]["error"]
    assert out["final"]["decision"] == "refused" and out["findings"]


def test_monitor_agent_reports_per_arm_once_rewards_exist(run):
    from nbx.fsdr import campaign_log
    folder, models = run
    root = campaign_log.root_for(folder)
    week = "2026-10-15"
    campaign_log.append_rewards(root, [
        {"run_date": week, "arm": a, "msisdn": f"7{i:07d}", "action": "RETAIN", "propensity": 0.5, "policy_source": "bandit",
         "reward_any_btl": r, "reward_same_service": r, "reward_exact_offer": 0, "revenue": 3.0 * r, "last_btl_date": None}
        for i, (a, r) in enumerate([("cg", 0), ("cg", 1), ("cstr", 1), ("cstr", 1)])])
    campaign_log.append_log(root, [{"run_date": week, "arm": "cstr", "msisdn": "70000002", "action": "RETAIN"}])
    from datetime import date
    out = asyncio.run(run_ops_agent("monitor", folder, models_root=models))
    # the default week is D - 7; ask for the seeded week explicitly through the tool
    from nbx.fsdr.agents.ops_agents import OPS_TOOLS, _context
    rep = OPS_TOOLS["weekly_report"].fn(_context(folder, models), {"week": week})
    assert rep["arms"]["cstr"]["uplift_vs_control"] == pytest.approx(0.5)
    assert out["mode"] == "fallback" and isinstance(out["final"]["alerts"], list)
