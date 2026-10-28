"""Batch flow: action -> top 4 offers -> top 2 roaming -> international (if the customer calls abroad)."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from nbx.fsdr import config
from nbx.fsdr.pipelines import batch as batch_mod
from nbx.fsdr.pipelines.batch import flat_row, recommend_batch, recommend_one


def _run(coro):
    return asyncio.run(coro)


def test_catalogue_has_roaming_and_international_add_ons(workspace):
    services = set(workspace.catalog["SERVICE"])
    assert {"ROAMING", "INTERNATIONAL"} <= services
    assert len(workspace.addons.families["roaming"]) == 6 and len(workspace.addons.families["international"]) == 6


def test_each_customer_gets_action_offers_roaming_and_conditional_international(workspace):
    lines = workspace.lines("llm")[:60]
    rows = _run(recommend_batch(workspace, lines, "llm"))
    assert len(rows) == len(lines)
    saw_intl = saw_no_intl = False
    for r in rows:
        assert r["action"] in config.ACTIONS + [config.NO_CONTACT]
        main = [o["offer"] for o in r["offers"]]
        assert len(main) <= config.MAX_OFFERS_TOTAL
        assert len(r["roaming"]) <= config.ADDON_TOP_ROAMING
        assert all(o["service"] in config.ROAMING_SERVICES for o in r["roaming"])
        assert all(o["service"] in config.INTERNATIONAL_SERVICES for o in r["international"])
        everything = main + [o["offer"] for o in r["roaming"] + r["international"]]
        assert len(everything) == len(set(everything))
        assert not any(frozenset((a, b)) in workspace.engine.conflicts for i, a in enumerate(everything) for b in everything[i + 1:])
        if r["international"]:
            saw_intl = True
            assert r["international_calls_m1"] >= config.INTL_MIN_CALLS_M1
        elif r["action"] != config.NO_CONTACT:
            saw_no_intl = True
            assert r["international_calls_m1"] < config.INTL_MIN_CALLS_M1 and "international call" in r["international_reason"]
    assert saw_intl and saw_no_intl
    assert any(r["roaming"] for r in rows)


def test_roaming_is_personalised_for_cold_start_customers(workspace):
    rows = _run(recommend_batch(workspace, workspace.lines("llm")[:80], "llm"))
    pairs = {tuple(o["offer"] for o in r["roaming"]) for r in rows if r["roaming"]}
    assert len(pairs) > 1


def test_cstr_arm_batch_uses_the_rule_based_offers(workspace):
    v = workspace.lines("cstr")[0]
    r = _run(recommend_one(workspace, v, "cstr"))
    expected = [p.btl_name for p in workspace.engine.select(v, workspace.scored.at[v, "ACTION"], "cstr")]
    assert [o["offer"] for o in r["offers"]] == expected and r["mode"] == "rules"


def test_no_contact_gets_nothing(workspace, monkeypatch):
    from nbx.fsdr.agents import recommender_agent

    async def no_contact(ws, msisdn, arm="llm", llm=None, emit=None):
        return {"decision": {"mode": "agentic", "action": config.NO_CONTACT, "policy_source": "llm", "propensity": 0.9,
                             "offers": [], "competitor": {"verdict": None}, "critic": {"verdict": "approved"}, "override_count": 1}}

    monkeypatch.setattr(recommender_agent, "decide", no_contact)
    r = _run(recommend_one(workspace, workspace.lines("llm")[0], "llm"))
    assert r["offers"] == [] and r["roaming"] == [] and r["international"] == []


def test_unknown_customer_and_flat_row(workspace):
    r = _run(recommend_one(workspace, "1", "llm"))
    assert "error" in r and flat_row(r)["error"]
    good = flat_row(_run(recommend_one(workspace, workspace.lines("llm")[1], "llm")))
    assert {"offer_1", "offer_4", "roaming_1", "roaming_2", "international"} <= set(good)


def test_batch_endpoint(scored_run, monkeypatch, tmp_path):
    from nbx.api import fsdr as fsdr_api
    from nbx.api.main import app

    monkeypatch.setattr(config, "FSDR_DATA_DIR", scored_run["folder"].parent)
    fsdr_api._workspaces.clear()
    c = TestClient(app)
    run = scored_run["folder"].name
    res = c.post(f"/fsdr/batch/{run}", json={"arm": "llm", "limit": 10}).json()
    assert res["count"] == 10 and len(res["table"]) == 10
    assert c.post(f"/fsdr/batch/{run}", json={"arm": "x"}).status_code == 400
    with pytest.raises(ValueError):
        _run(recommend_batch(None, ["1"] * (config.BATCH_MAX + 1)))
