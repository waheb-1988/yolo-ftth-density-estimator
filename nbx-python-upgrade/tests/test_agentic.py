"""Tests for the agentic NBA/NBO council (agents/agentic/).

The LLM is replaced by a scripted fake keyed on the agent's role, so these
test the controller's policy enforcement -- required evidence, loop guard,
guardrails, critic revision, auto-fix, compliance -- not model quality.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nbx.agents.agentic import council, react
from nbx.agents.agentic.tools import RunContext, call_tool, risk_level
from nbx.core import config, insights
from nbx.llms import llm_client


def _customers_by(pred):
    return [r["msisdn"] for r in insights.customers_list() if pred(r)]


HIGH_RISK = _customers_by(lambda r: r["risk_level"] == "HIGH" and r["nba_segment"] and "Nursering" not in r["nba_segment"])[0]
LOW_RISK = _customers_by(lambda r: r["risk_level"] == "LOW" and r["days_since_last_purchase"] > 3 and r["policy_action"] == "UPSELL")[0]


@pytest.fixture
def no_llm(monkeypatch):
    async def unavailable():
        return False

    monkeypatch.setattr(llm_client, "is_available", unavailable)


class FakeLLM:
    """Returns scripted JSON replies per agent; records every prompt."""

    def __init__(self, scripts: dict[str, list[dict]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.prompts: list[tuple[str, str]] = []

    def _agent(self, system: str) -> str:
        for key, marker in (("nba", "Next-Best-Action strategist"), ("nbo", "Next-Best-Offer specialist"),
                            ("critic", "compliance critic"), ("messenger", "customer messages")):
            if marker in (system or ""):
                return key
        return "other"

    async def generate(self, model, prompt, system=None, force_json=False, timeout=120.0):
        agent = self._agent(system)
        self.prompts.append((agent, prompt))
        queue = self.scripts.get(agent) or [{"thought": "done", "final": {}}]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return {"content": json.dumps(reply, ensure_ascii=False), "duration_ms": 1, "model": model, "error": None}


@pytest.fixture
def fake_llm(monkeypatch):
    def install(scripts):
        fake = FakeLLM(scripts)

        async def available():
            return True

        async def resolve(model):
            return "fake-model"

        # A fake *server* LLM: the offline backend would skip the LLM entirely.
        monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
        monkeypatch.setattr(llm_client, "is_available", available)
        monkeypatch.setattr(llm_client, "resolve_model", resolve)
        monkeypatch.setattr(llm_client, "generate", fake.generate)
        return fake

    return install


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tools_return_errors_as_observations_not_exceptions():
    ctx = RunContext.load(LOW_RISK)
    assert "error" in call_tool(ctx, "check_eligibility", {})
    assert "error" in call_tool(ctx, "search_offers", {"category": "Spaceships"})
    assert call_tool(ctx, "search_offers", {"category": ["data", "Bundle"]})["count"] > 0  # case + list tolerant


def test_voice_offers_are_never_eligible():
    ctx = RunContext.load(LOW_RISK)
    res = call_tool(ctx, "check_eligibility", {"offers": ["VC_30D_8R_UNLIMITED"]})["results"][0]
    assert not res["eligible"] and "voice" in res["reasons"][0]


# ---------------------------------------------------------------------------
# Fallback mode (no LLM)
# ---------------------------------------------------------------------------


async def test_fallback_retains_high_risk_customer(no_llm):
    out = await council.run_council(HIGH_RISK, "full")
    d = out["decision"]
    assert d["mode"] == "fallback"
    assert d["action"] == "RETAIN"
    assert all(o["category"] in council.ACTION_CATEGORIES["RETAIN"] for o in d["offers"])
    assert d["message"]["sms_en"]
    assert out["trace"][0]["type"] == "run_start" and out["trace"][-1]["type"] == "run_end"


async def test_every_decision_respects_hard_guardrails(no_llm):
    for msisdn in [r["msisdn"] for r in insights.customers_list()][:20]:
        d = (await council.run_council(msisdn, "full"))["decision"]
        ctx = RunContext.load(msisdn)
        hard = [f for f in council.guardrails(ctx, {"action": d["action"]}, {"offers": d["offers"]}) if f["severity"] == "hard"]
        assert not hard, (msisdn, hard)


# ---------------------------------------------------------------------------
# Agentic mode: controller policy
# ---------------------------------------------------------------------------


async def test_finalize_is_blocked_until_required_evidence(fake_llm):
    fake = fake_llm({
        "nba": [
            {"thought": "I already know", "final": {"action": "MAINTAIN"}},  # blocked: no evidence
            {"thought": "risk", "tool": "get_risk_scores", "args": {}},
            {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
            {"thought": "ok", "final": {"action": "UPSELL", "channel": "APP", "confidence": 0.7, "rationale": "fit"}},
        ],
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "Hi", "sms_ar": "مرحبا", "agent_script": "Hello"}],
    })
    out = await council.run_council(LOW_RISK, "nba")
    policies = [e["text"] for e in out["trace"] if e["type"] == "policy"]
    assert any("Finalize blocked" in p for p in policies)
    assert out["decision"]["action"] == "UPSELL"
    assert out["decision"]["mode"] == "agentic"
    # The blocked agent was told what it was missing.
    assert "Not allowed to finalize yet" in fake.prompts[1][1]


async def test_critic_revision_then_auto_fix_for_push_to_at_risk(fake_llm):
    # The strategist insists on a sales push for a high-risk customer, even after feedback.
    push = {"thought": "sell", "final": {"action": "UPSELL DATA", "channel": "APP", "confidence": 0.9, "rationale": "data fit"}}
    fake_llm({
        "nba": [
            {"thought": "risk", "tool": "get_risk_scores", "args": {}},
            {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
            push,
        ],
        "nbo": [{"thought": "pick", "final": {"offers": [{"offer": "DT_7D_4R_5GB", "why": "data"}], "rationale": "r"}}],
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],  # even a lenient critic can't wave it through
        "messenger": [{"sms_en": "Hi", "sms_ar": "مرحبا", "agent_script": "Hello"}],
    })
    out = await council.run_council(HIGH_RISK, "full")
    d = out["decision"]
    assert d["critic"]["verdict"] == "revise" and d["critic"]["revised"]
    assert d["action"] == "RETAIN"
    assert any("RETAIN" in f for f in d["critic"]["auto_fixes"])
    assert any(e["type"] == "revision" for e in out["trace"])


async def test_loop_guard_stops_repeated_calls(fake_llm):
    same = {"thought": "search again", "tool": "search_offers", "args": {"category": "Data"}}
    fake_llm({
        "nba": [{"thought": "risk", "tool": "get_risk_scores", "args": {}},
                {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
                {"thought": "ok", "final": {"action": "UPSELL", "channel": "APP", "confidence": 0.7, "rationale": "r"}}],
        "nbo": [same],  # repeats forever
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "Hi", "sms_ar": "مرحبا", "agent_script": "Hello"}],
    })
    out = await council.run_council(LOW_RISK, "full")
    texts = [e["text"] for e in out["trace"] if e["type"] == "policy" and e["agent"] == "nbo"]
    assert any("Duplicate" in t for t in texts) and any("Loop detected" in t for t in texts)
    assert out["decision"]["offers"]  # finalized from evidence by policy


async def test_unknown_offer_codes_are_rejected(fake_llm):
    fake_llm({
        "nbo": [
            {"thought": "elig", "tool": "check_eligibility", "args": {"offers": ["BN_7D_7R_MIX"]}},
            {"thought": "ev", "tool": "simulate_expected_value", "args": {"offers": ["BN_7D_7R_MIX"]}},
            {"thought": "pick", "final": {"offers": [{"offer": "MADE_UP_OFFER"}]}},
            {"thought": "pick", "final": {"offers": [{"offer": "BN_7D_7R_MIX", "why": "fit"}], "rationale": "r"}},
        ],
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "Hi", "sms_ar": "مرحبا", "agent_script": "Hello"}],
    })
    out = await council.run_council(LOW_RISK, "nbo")
    assert any("unknown offer codes" in e["text"] for e in out["trace"] if e["type"] == "policy")
    assert [o["offer"] for o in out["decision"]["offers"]] == ["BN_7D_7R_MIX"]


async def test_messenger_currency_is_forced_to_omr(fake_llm):
    fake_llm({
        "nba": [{"thought": "risk", "tool": "get_risk_scores", "args": {}},
                {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
                {"thought": "ok", "final": {"action": "UPSELL", "channel": "SMS", "confidence": 0.7, "rationale": "r"}}],
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "Enjoy 5GB for only 5 AED this week!", "sms_ar": "استمتع بـ 5 جيجابايت مقابل 5 درهم إماراتي فقط", "agent_script": "Offer the pack at 5 dirham."}],
    })
    d = (await council.run_council(LOW_RISK, "nba"))["decision"]
    assert "AED" not in d["message"]["sms_en"] and "OMR" in d["message"]["sms_en"]
    assert "درهم" not in d["message"]["sms_ar"] and "ر.ع." in d["message"]["sms_ar"]


async def test_messenger_template_echo_falls_back_to_compliant_template(fake_llm):
    fake_llm({
        "nba": [{"thought": "risk", "tool": "get_risk_scores", "args": {}},
                {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
                {"thought": "ok", "final": {"action": "UPSELL", "channel": "SMS", "confidence": 0.7, "rationale": "r"}}],
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "...", "sms_ar": "...", "agent_script": "..."}],
    })
    out = await council.run_council(LOW_RISK, "nba")
    msg = out["decision"]["message"]
    assert msg["sms_en"].startswith(f"{config.OPERATOR_NAME}:") and "..." != msg["sms_ar"]
    assert any("template" in e["text"] for e in out["trace"] if e["type"] == "policy")


async def test_copilot_tools_return_preformatted_units():
    from nbx.agents.agentic import copilot

    row = copilot.find_customers(None, {"risk_level": "HIGH", "limit": 1})["customers"][0]
    assert row["churn_probability"].endswith("%") and row["arpu"].endswith(" OMR")


async def test_llm_failure_mid_run_falls_back_and_completes(fake_llm, monkeypatch):
    fake_llm({})

    async def broken(**kw):
        return {"content": "", "duration_ms": 0, "model": "fake-model", "error": "connection refused"}

    monkeypatch.setattr(llm_client, "generate", broken)
    d = (await council.run_council(HIGH_RISK, "full"))["decision"]
    assert d["mode"] == "partial_fallback" and d["action"] == "RETAIN"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_api_endpoints(no_llm):
    from nbx.api import main

    client = TestClient(main.app)
    assert len(client.get("/customers").json()) == len(insights.customers_list())
    summary = client.get("/portfolio/summary").json()
    assert summary["customers"] > 0 and "revenue_at_risk" in summary
    assert client.get(f"/customers/{HIGH_RISK}/360").json()["risk_level"] == risk_level(RunContext.load(HIGH_RISK).score_row)
    assert client.post("/agentic/decide", json={"msisdn": HIGH_RISK, "scope": "bad"}).status_code == 400
    assert client.post("/agentic/decide", json={"msisdn": 999999}).status_code == 404

    with client.stream("GET", f"/agentic/stream?msisdn={HIGH_RISK}&scope=full") as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    assert client.get("/agentic/recent").json()[0]["msisdn"] == HIGH_RISK

    chat = client.post("/copilot/chat", json={"question": "Who is at risk?"}).json()
    assert chat["mode"] == "fallback" and "high-risk" in chat["answer"]


# ---------------------------------------------------------------------------
# NBA first, then NBO on the NBA result
# ---------------------------------------------------------------------------
NBA_SCRIPT = [
    {"thought": "risk", "tool": "get_risk_scores", "args": {}},
    {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
    {"thought": "ok", "final": {"action": "UPSELL", "channel": "APP", "confidence": 0.7, "rationale": "Low churn and rising data use."}},
]


async def test_offer_run_starts_with_the_nba_agent_and_hands_its_decision_to_nbo(fake_llm):
    fake = fake_llm({
        "nba": list(NBA_SCRIPT),
        "nbo": [{"thought": "elig", "tool": "check_eligibility", "args": {"offers": ["BN_7D_7R_MIX"]}},
                {"thought": "ev", "tool": "simulate_expected_value", "args": {"offers": ["BN_7D_7R_MIX"]}},
                {"thought": "pick", "final": {"offers": [{"offer": "BN_7D_7R_MIX", "why": "fit"}], "rationale": "r"}}],
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "A pack picked for you for 7 OMR", "sms_ar": "باقة مختارة لك بسعر 7 ر.ع", "agent_script": "Greet."}],
    })
    out = await council.run_council(LOW_RISK, "nbo")
    starts = [e["agent"] for e in out["trace"] if e["type"] == "agent_start"]
    assert starts[:2] == ["nba", "nbo"] and "orchestrator" not in starts
    assert out["trace"][0]["agents"] == ["nba", "nbo", "critic", "messenger"]
    nbo_prompts = [p for a, p in fake.prompts if a == "nbo"]
    assert "STEP 1 RESULT" in nbo_prompts[0] and "UPSELL" in nbo_prompts[0] and "Low churn and rising data use." in nbo_prompts[0]
    assert out["decision"]["action"] == "UPSELL" and out["decision"]["offers"]


async def test_step2_reuses_the_last_nba_decision_without_rerunning_nba(fake_llm):
    fake = fake_llm({
        "nba": list(NBA_SCRIPT),
        "critic": [{"verdict": "approve", "issues": [], "feedback": ""}],
        "messenger": [{"sms_en": "A pack picked for you for 7 OMR", "sms_ar": "باقة مختارة لك بسعر 7 ر.ع", "agent_script": "Greet."}],
    })
    first = await council.run_council(LOW_RISK, "nba")
    assert first["decision"]["action"] == "UPSELL"
    nba_calls_before = sum(1 for a, _ in fake.prompts if a == "nba")
    second = await council.run_council(LOW_RISK, "nbo", use_last_nba=True)
    assert sum(1 for a, _ in fake.prompts if a == "nba") == nba_calls_before  # strategist not called again
    assert second["decision"]["action"] == "UPSELL"
    assert any(e["type"] == "agent_final" and e["agent"] == "nba" and e["finalized_by"] == "step 1 (reused)" for e in second["trace"])


# ---------------------------------------------------------------------------
# Small-model near-misses seen live with mistral:7b
# ---------------------------------------------------------------------------
async def test_several_tools_in_one_step_run_one_by_one_instead_of_being_rejected(fake_llm):
    fake_llm({"nba": [
        {"thought": "both", "tool": "get_risk_scores{msisdn: 13}, bandit_recommendation{msisdn: 13}", "args": {}},
        {"thought": "both", "tool": "get_risk_scores{msisdn: 13}, bandit_recommendation{msisdn: 13}", "args": {}},
        {"thought": "ok", "final": {"action": "UPSELL", "channel": "APP", "confidence": 0.7, "rationale": "r"}},
    ]})
    events = []
    res = await react.run_react(council.NBA_SPEC, RunContext.load(LOW_RISK), events.append, "fake-model")
    assert [h["tool"] for h in res.evidence] == ["get_risk_scores", "bandit_recommendation"]
    assert not any("Rejected unknown tool" in e.get("text", "") for e in events)
    assert res.final["action"] == "UPSELL" and res.llm_calls == 3


async def test_tool_named_final_is_read_as_the_final_answer(fake_llm):
    fake_llm({"nba": [
        {"thought": "risk", "tool": "get_risk_scores", "args": {}},
        {"thought": "bandit", "tool": "bandit_recommendation", "args": {}},
        {"thought": "done", "tool": "final", "args": {"action": "UPSELL", "channel": "SMS", "confidence": 0.6, "rationale": "r"}},
    ]})
    res = await react.run_react(council.NBA_SPEC, RunContext.load(LOW_RISK), lambda e: None, "fake-model")
    assert res.final and res.final["action"] == "UPSELL" and res.finalized_by == "llm"
