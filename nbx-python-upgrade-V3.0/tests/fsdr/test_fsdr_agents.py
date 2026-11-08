"""Action, offer and competitor agents: offline = constraint arm, LLM freedom, hard rules, logging."""
from __future__ import annotations

import asyncio
import json

import pytest

from nbx.fsdr import config
from nbx.fsdr.agents import recommender_agent as ra
from nbx.llms import llm_client


def _decide(ws, msisdn, llm=None):
    return asyncio.run(ra.decide(ws, msisdn, "llm", llm))


class FakeLLM:
    """Scripted JSON replies per agent, recognised from the agent's system role."""

    def __init__(self, scripts: dict[str, list[dict]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.prompts: list[tuple[str, str]] = []

    @staticmethod
    def agent(system: str) -> str:
        for key, marker in (("action", "Next-Best-Action strategist"), ("offer", "offer specialist"),
                            ("competitor", "recommended offers with competitor plans")):
            if marker in (system or ""):
                return key
        return "other"

    async def generate(self, model, prompt, system=None, force_json=False, timeout=120.0):
        a = self.agent(system)
        self.prompts.append((a, prompt))
        q = self.scripts.get(a) or [{"thought": "done", "final": {}}]
        reply = q.pop(0) if len(q) > 1 else q[0]
        return {"content": json.dumps(reply), "duration_ms": 1, "model": model, "error": None}


@pytest.fixture
def fake_llm(monkeypatch):
    def install(scripts, epsilon=0.0):
        fake = FakeLLM(scripts)

        async def yes():
            return True

        async def resolve(model):
            return "fake-model"

        monkeypatch.setattr(config, "EPSILON_LLM", epsilon)
        from nbx.core import config as nbx_config

        monkeypatch.setattr(nbx_config, "LLM_BACKEND", "vllm")      # a live server (every call is faked)
        monkeypatch.setattr(llm_client, "is_available", yes)
        monkeypatch.setattr(llm_client, "resolve_model", resolve)
        monkeypatch.setattr(llm_client, "generate", fake.generate)
        return fake

    return install


def _tools(*names, args=None):
    return [{"thought": f"check {n}", "tool": n, "args": (args or {}).get(n, {})} for n in names]


def _customer(ws, low_churn=True, with_offers=True):
    for v in ws.lines("llm"):
        ctx = ws.context(v)
        churn_ok = ctx.churn() < config.CHURN_BLOCK_THRESHOLD if low_churn else ctx.churn() >= config.CHURN_BLOCK_THRESHOLD
        if churn_ok and not ctx.eligibility_issues() and (not with_offers or ctx.deterministic_offers(ctx.scored["ACTION"])):
            return v, ctx
    pytest.skip("no suitable customer in the fixture")


def test_offline_llm_arm_equals_the_constraint_arm(workspace):
    for v in workspace.lines("llm")[:40]:
        d = _decide(workspace, v)["decision"]
        expected_action = workspace.scored.loc[v, "ACTION"]
        expected = [p.btl_name for p in workspace.engine.select(v, expected_action, "cstr")]
        assert d["mode"] == "fallback" and d["policy_source"] == "fallback"
        assert d["action"] == expected_action and [o["offer"] for o in d["offers"]] == expected
        assert d["propensity"] == pytest.approx(float(workspace.scored.loc[v, "PROBA"]), abs=1e-6)
        assert d["stats"]["llm_calls"] == 0


def test_competitor_names_are_replaced_by_labels(workspace):
    v, _ = _customer(workspace)
    d = _decide(workspace, v)["decision"]
    ops = {c["operator"] for r in d["competitor"]["comparisons"] for c in r.get("closest", [])}
    assert ops and ops <= {"COMPETITOR_A", "COMPETITOR_B"}
    assert d["competitor"]["verdict"] in ra.VERDICTS


def test_a_justified_llm_override_is_accepted_and_logged(workspace, fake_llm):
    v, ctx = _customer(workspace)
    top = ctx.bandit_top()
    other = next(a for a in ("RETAIN", "MAINTAIN", "CROSS SELL DATA") if a != top)
    picks = [p.btl_name for p in ctx.deterministic_offers(other)]
    if not picks:
        pytest.skip("no offers for the override action")
    fake = fake_llm({
        "action": _tools("customer_360", "bandit_pmf", "contact_policy") + [
            {"thought": "pick", "final": {"action": other, "signals_used": ["churn trend"], "confidence": 0.7,
                                          "override_reason": "falling revenue and rising churn score"}}],
        "offer": _tools("price_band_candidates", "check_offers", args={"check_offers": {"offers": picks[:1]}}) + [
            {"thought": "one offer", "final": {"offers": picks[:1], "rationale": "best fit",
                                                 "override_reason": "a single offer avoids over-contacting"}}],
        "competitor": [{"thought": "ok", "final": {"verdict": "competitive", "summary": "fine", "swap": []}}],
    })
    d = _decide(workspace, v, "fake")["decision"]
    assert d["mode"] == "agentic" and d["action"] == other and d["policy_source"] == "llm"
    assert d["propensity"] == pytest.approx(1.0)                        # eps = 0: 1 - 0 + 0 * pmf
    kinds = {o["type"] for o in d["overrides"]}
    assert {"action", "offers"} <= kinds and d["override_count"] >= 2
    assert [o["offer"] for o in d["offers"]] == picks[:1]
    assert any(a == "competitor" for a, _ in fake.prompts)


def test_an_override_without_reason_is_sent_back(workspace, fake_llm):
    v, ctx = _customer(workspace)
    other = next(a for a in config.ACTIONS if a != ctx.bandit_top() and a not in config.UPSELL_FAMILY)
    fake = fake_llm({"action": _tools("customer_360", "bandit_pmf", "contact_policy") + [
        {"thought": "no reason", "final": {"action": other, "signals_used": []}},
        {"thought": "with reason", "final": {"action": other, "signals_used": ["x"], "override_reason": "segment shows rising data use"}}]})
    d = _decide(workspace, v, "fake")["decision"]
    assert d["action"] == other
    rejected = [p for a, p in fake.prompts if a == "action" and "override_reason" in p and "Invalid final answer" in p]
    assert rejected, "the first final answer must have been rejected"


def test_exploration_logs_the_bandit_propensity(workspace, fake_llm):
    v, ctx = _customer(workspace)
    fake_llm({"action": _tools("customer_360", "bandit_pmf", "contact_policy") + [
        {"thought": "follow", "final": {"action": ctx.bandit_top(), "signals_used": ["pmf"]}}]}, epsilon=1.0)
    d = _decide(workspace, v, "fake")["decision"]
    assert d["policy_source"] == "bandit"
    assert d["propensity"] == pytest.approx(ctx.pmf()[d["action"]], abs=1e-4)


def test_hard_rule_churn_block_and_conflicting_offers_are_repaired(workspace, fake_llm):
    v, ctx = _customer(workspace, low_churn=False, with_offers=False)
    a, b = sorted(sorted(p) for p in workspace.engine.conflicts)[0]          # deterministic: sets have no order
    fake_llm({
        "action": _tools("customer_360", "bandit_pmf", "contact_policy") + [
            {"thought": "push", "final": {"action": "UPSELL", "signals_used": ["spend"], "override_reason": "high spend customer"}}],
        "offer": _tools("price_band_candidates", "check_offers", args={"check_offers": {"offers": [a, b]}}) + [
            {"thought": "both", "final": {"offers": [a, b], "rationale": "x", "override_reason": "two packs"}}],
    })
    d = _decide(workspace, v, "fake")["decision"]
    assert d["action"] == "RETAIN"                                       # upsell blocked by churn >= threshold
    assert any("blocks upsell" in f for f in d["critic"]["findings"]) and d["critic"]["verdict"] == "revised"
    names = [o["offer"] for o in d["offers"]]
    assert not (a in names and b in names)                               # the conflicting pair never ships


def test_competitor_swap_is_applied_only_when_it_passes_the_hard_rules(workspace, fake_llm):
    v, ctx = _customer(workspace)
    action = ctx.scored["ACTION"]
    picks = [p.btl_name for p in ctx.deterministic_offers(action)]
    out = picks[0]
    validity = int(workspace.catalog.at[out, "VALIDITY"])
    same_validity = [n for n in workspace.catalog.index if int(workspace.catalog.at[n, "VALIDITY"]) == validity and n not in picks]
    good = next(n for n in same_validity if not ra.check_offers([n if o == out else o for o in picks], workspace.engine.catalog,
                                                                  workspace.engine.conflicts))
    fake_llm({
        "action": _tools("customer_360", "bandit_pmf", "contact_policy") + [
            {"thought": "follow", "final": {"action": action, "signals_used": ["pmf"],
                                            "override_reason": "agree with the sampled action"}}],
        "offer": _tools("price_band_candidates", "check_offers", args={"check_offers": {"offers": picks}}) + [
            {"thought": "rules", "final": {"offers": picks, "rationale": "price band"}}],
        "competitor": [{"thought": "swap", "final": {"verdict": "mixed", "summary": "one beaten",
                                                     "swap": [{"out": out, "in": good, "reason": "cheaper per GB than the closest plan"},
                                                              {"out": picks[-1], "in": "NOT_A_REAL_OFFER", "reason": "x"}]}}],
    })
    d = _decide(workspace, v, "fake")["decision"]
    names = [o["offer"] for o in d["offers"]]
    assert good in names and out not in names
    assert d["competitor"]["swaps_applied"] == [{"out": out, "in": good, "reason": "cheaper per GB than the closest plan"}]


def test_no_contact_means_no_offer(workspace, fake_llm):
    v, _ = _customer(workspace)
    fake_llm({"action": _tools("customer_360", "bandit_pmf", "contact_policy") + [
        {"thought": "rest", "final": {"action": "NO_CONTACT", "signals_used": ["fatigue"], "override_reason": "contacted too often lately"}}]})
    d = _decide(workspace, v, "fake")["decision"]
    assert d["action"] == config.NO_CONTACT and d["offers"] == [] and d["competitor"]["verdict"] is None


def test_batch_writes_decisions_resumes_and_writes_llm_recos(scored_run, workspace):
    folder = scored_run["folder"]
    first = asyncio.run(ra.run_batch(folder, ws=workspace, limit=20, resume=False))
    assert first["decided_now"] == 20 and first["modes"] == {"fallback": 20}
    again = asyncio.run(ra.run_batch(folder, ws=workspace, limit=20))
    assert again["decided_now"] == 0 and again["reused_from_checkpoint"] == 20
    lines = (folder / "outputs" / ra.DECISIONS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20 and all({"propensity", "policy_source", "bandit_pmf", "cache_key"} <= set(json.loads(l)) for l in lines)
    assert (folder / "outputs" / "recos_monthly_llm.json").exists()


def test_logged_propensity_golden_values():
    """L6: (1 - eps) x 1[a = LLM action] + eps x PMF[a], eps = 0.1, for every branch."""
    from nbx.fsdr.agents.recommender_agent import logged_propensity
    top, other = 0.8 + 0.2 / 7, 0.2 / 7
    pmf = {"RETAIN": top, **{a: other for a in config.ACTIONS if a != "RETAIN"}}
    assert logged_propensity("RETAIN", "RETAIN", pmf, 0.1) == pytest.approx(0.9828571, abs=1e-6)    # LLM picks the top action
    assert logged_propensity("UPSELL", "UPSELL", pmf, 0.1) == pytest.approx(0.9028571, abs=1e-6)    # LLM picks a low-PMF action
    assert logged_propensity("UPSELL", "RETAIN", pmf, 0.1) == pytest.approx(0.0028571, abs=1e-6)    # exploration, another action
    # exploration drawing the LLM's own action logs the same probability as the LLM branch (was eps x PMF)
    assert logged_propensity("RETAIN", "RETAIN", pmf, 0.1) != pytest.approx(0.1 * top)


def test_cohort_tool_reuses_the_knowledge_graph_on_fsdr_data(workspace):
    """A3 / L2: core/knowledge_graph built from this run's device segments and 270-day BTL history."""
    from nbx.fsdr.agents.tools import FSDR_TOOLS
    g = workspace.cohort_graph
    assert any(d.get("kind") == "PURCHASED" for _, _, d in g.edges(data=True))
    v = next(m for m in workspace.lines("llm") if m in workspace.base.index)
    out = FSDR_TOOLS["cohort_offers"].fn(workspace.context(v), {"top_k": 3})
    assert out["dv_segment"] and 0 < len(out["offers"]) <= 3
    assert all(o["offer"] in workspace.catalog.index and 0 < o["graph_score"] <= 1 for o in out["offers"])
    data = FSDR_TOOLS["cohort_offers"].fn(workspace.context(v), {"service": "DATA"})
    assert all(o["service"] == "DATA" for o in data["offers"])


def test_offer_tools_handle_add_on_services_without_an_arpu_anchor(workspace):
    """Roaming / international offers have no BR-77 anchor: the tools compare them with the total anchor
    instead of raising KeyError (which crashed the whole decision)."""
    from nbx.fsdr.agents.tools import FSDR_TOOLS
    from nbx.fsdr.offer_rules import anchor_column
    assert anchor_column("INTERNATIONAL", 7) == "AVG_REV_WEEKLY" and anchor_column("ROAMING", 30) == "AVG_REV_MONTHLY"
    v, ctx = _customer(workspace)
    ctx.action = "RETAIN"
    addon = next(n for n, s in zip(workspace.catalog["BTL_NAME"], workspace.catalog["SERVICE"]) if s in ("INTERNATIONAL", "ROAMING"))
    out = FSDR_TOOLS["check_offers"].fn(ctx, {"offers": [addon]})
    assert any(addon in note for note in out["soft_notes"])                  # outside the action's families: a soft note
    assert "offers" in FSDR_TOOLS["search_catalog"].fn(ctx, {"service": "ROAMING"})
    assert FSDR_TOOLS["expected_value"].fn(ctx, {"offers": [addon]})
