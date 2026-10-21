"""Tests for competitor intelligence (core/competitors.py) and the decision
explanations that use it (agents/agentic/explain.py). Parsers run on small
fixture documents shaped like the real pages -- no network access."""
from __future__ import annotations

import json

import pytest

from nbx.agents.agentic import council
from nbx.agents.agentic.tools import RunContext, call_tool
from nbx.core import competitors, insights
from nbx.llms import llm_client

VODAFONE_HTML = """
<html><body><nav>menu</nav>
<div class="plan-revamp"><span>Best Seller</span><div>6GB</div><div>RED 5</div><div>5</div><div>/</div><div>28 Days</div>
  <b>6 GB Local Data</b><div>30 Flexi &amp; 120 Local Minutes</div><div>Plan Benefits</div><div>Data Roll-over</div></div>
<div class="plan-revamp"><div>30GB</div><div>RED 13</div><div>13</div><div>/</div><div>28 Days</div>
  <b>15 GB Local Data</b><div>150 Flexi &amp; 350 Local minutes</div><div>Roaming pass included</div></div>
</body></html>"""

OMANTEL_API = [
    {"categoryCode": "WEB_NEW_HAYYAK_PLANS", "categoryEntries": [
        {"shortDescription": "Hayyak 11", "price": ["11", "11"], "attributes": [
            {"name": "DURATION_LABEL", "value": "4 Weeks"}, {"name": "LOC_DATA_PACKAGE_SIZE", "value": "12 GB + 4 GB"},
            {"name": "INCLUDED_MINUTES", "value": "250 Flexi Min"}]},
        {"shortDescription": "No price", "price": [], "attributes": []},
    ]},
    {"categoryCode": "WEB_ERADA_PLANS", "categoryEntries": [  # special-needs segment: not mapped, skipped
        {"shortDescription": "5 OMR Plan", "price": ["5"], "attributes": []}]},
    {"categoryCode": "WEB_BAQATI_POSTPAID_12_MONTHS", "categoryEntries": [
        {"shortDescription": "Baqati 18", "price": ["18", "18.9"], "attributes": [
            {"name": "INCLUDED_DATA", "value": "30 GB"}, {"name": "INCLUDED_MINUTES", "value": "Unlimited"},
            {"name": "CONTRACT_DURATION", "value": "12"}]}]},
]

OMANTEL_TABLE_HTML = """<html><body><div>Daily</div><div>Weekly</div>
<p>Data</p><p>Subscription Fee</p><p>Activation</p><p>Validity</p>
<p>1 GB</p><p>1 OMR</p><p>Dial: *141*24#</p><p>1 Day</p>
<p>200 MB</p><p>100 BZ</p><p>Dial: *141*35#</p><p>1 Day</p>
<p>Data</p><p>Subscription Fee</p><p>Activation</p>
<p>6 GB</p><p>4 OMR</p><p>Dial: *666*40#</p>
</body></html>"""


def test_normalization_helpers():
    assert competitors.parse_data_gb("40 GB + 20 GB") == 60.0
    assert competitors.parse_data_gb("500 MB") == pytest.approx(0.488, abs=1e-3)
    assert competitors.parse_price_omr("100 BZ") == 0.1
    assert competitors.parse_price_omr("OMR 9") == 9.0
    assert competitors.parse_days("4 Weeks") == 28
    assert competitors.parse_days("28 Days") == 28
    assert competitors.parse_days("No commitment") == 30


def test_vodafone_cards_parse_to_bundles():
    offers = competitors.parse_vodafone_cards(VODAFONE_HTML, "https://vf.test/prepaid", "prepaid")
    assert [o["plan"] for o in offers] == ["RED 5", "RED 13"]
    red5 = offers[0]
    assert (red5["price_omr"], red5["validity_days"], red5["data_gb"]) == (5.0, 28, 6.0)
    assert red5["category"] == "Bundle" and red5["segment"] == "mass"
    assert red5["monthly_equiv_omr"] == pytest.approx(5.36, abs=0.01)
    assert offers[1]["category"] == "Bundle"  # a roaming pass inside the card doesn't make it a roaming plan


def test_omantel_api_maps_mass_market_categories_only():
    offers = competitors.parse_omantel_api(OMANTEL_API, "https://om.test/api")
    by_plan = {o["plan"]: o for o in offers}
    assert set(by_plan) == {"Hayyak 11", "Baqati 18"}  # unpriced entry and special-needs plans skipped
    h = by_plan["Hayyak 11"]
    assert (h["price_omr"], h["validity_days"], h["data_gb"]) == (11.0, 28, 16.0)
    assert any("bonus" in f for f in h["features"])
    assert by_plan["Baqati 18"]["validity_days"] == 30


def test_omantel_tables_use_tab_order_for_validity():
    offers = competitors.parse_omantel_data_tables(OMANTEL_TABLE_HTML, "https://om.test/data", "data")
    by = {o["plan"]: o for o in offers}
    assert by["1 GB (Daily)"]["price_omr"] == 1.0 and by["1 GB (Daily)"]["validity_days"] == 1
    assert by["200 MB (Daily)"]["price_omr"] == 0.1
    assert by["6 GB (Weekly)"]["validity_days"] == 7


def test_scrape_survives_a_failing_page():
    def fetch(url):
        if "broken" in url:
            raise ConnectionError("down")
        return VODAFONE_HTML

    res = competitors.scrape(
        sources=[("Vodafone Oman", "prepaid", "https://vf.test/ok"), ("Vodafone Oman", "prepaid", "https://vf.test/broken")],
        fetch=fetch, fetch_json=lambda u: OMANTEL_API, api_sources=[("Omantel", "https://om.test/api")], delay_s=0,
    )
    assert len(res["errors"]) == 1 and "broken" in res["errors"][0]["url"]
    assert {o["operator"] for o in res["offers"]} == {"Vodafone Oman", "Omantel"}
    assert all(o["id"] and o["scraped_at"] for o in res["offers"])


def test_compare_offer_verdicts_and_segment_filter():
    comp = [
        competitors.enrich({"operator": "Rival", "plan": "Big", "category": "Bundle", "segment": "mass", "price_omr": 10.0,
                            "validity_days": 30, "data_gb": 50.0, "source_url": "x"}),
        competitors.enrich({"operator": "Rival", "plan": "Tourist", "category": "Bundle", "segment": "visitors", "price_omr": 12.0,
                            "validity_days": 30, "data_gb": 1.0, "source_url": "x"}),
    ]
    ours = {"category": "Bundle", "price_omr": 12.0, "validity_days": 30, "data_gb": "20GB"}
    res = competitors.compare_offer(ours, comp)
    assert res["closest"][0]["plan"] == "Big"  # visitor plans never used as the comparison
    assert "more expensive" in res["closest"][0]["verdict"] and "less data" in res["closest"][0]["verdict"]
    assert res["position"].startswith("weak")


@pytest.fixture
def snapshot(monkeypatch):
    offers = competitors.parse_vodafone_cards(VODAFONE_HTML, "https://vf.test", "prepaid") + \
        competitors.parse_omantel_api(OMANTEL_API, "https://om.test")
    monkeypatch.setattr(competitors, "load_offers", lambda: offers)
    return offers


def test_competitor_benchmark_tool(snapshot):
    msisdn = insights.customers_list()[0]["msisdn"]
    ctx = RunContext.load(msisdn)
    overview = call_tool(ctx, "competitor_benchmark", {})
    assert "customer_monthly_spend_omr" in overview
    res = call_tool(ctx, "competitor_benchmark", {"offers": ["BN_30D_18R_MIX"]})["results"][0]
    assert res["offer"] == "BN_30D_18R_MIX" and res["closest"]


async def test_decision_carries_a_full_explanation(snapshot, monkeypatch):
    async def unavailable():
        return False

    monkeypatch.setattr(llm_client, "is_available", unavailable)
    high = next(r["msisdn"] for r in insights.customers_list() if r["risk_level"] == "HIGH" and "Nursering" not in (r["nba_segment"] or ""))
    d = (await council.run_council(high, "full"))["decision"]
    e = d["explanation"]
    assert e["action"]["drivers"] and e["action"]["bandit_comparison"] and e["action"]["channel_and_timing"]
    assert e["narrative"] and d["action"] in e["narrative"]
    assert len(e["offers"]) == len(d["offers"])
    kinds = {r["kind"] for r in e["offers"][0]["reasons"]}
    assert {"fit", "affordability", "economics", "competition"} <= kinds
    json.dumps(d)  # the whole decision stays JSON-serializable for the API / SSE


def test_compare_prefers_same_validity_band():
    comp = [
        competitors.enrich({"operator": "Rival", "plan": "Monthly 20", "category": "Bundle", "segment": "mass", "price_omr": 20.0,
                            "validity_days": 30, "data_gb": 60.0, "source_url": "x"}),
        competitors.enrich({"operator": "Rival", "plan": "Daily 1", "category": "Bundle", "segment": "mass", "price_omr": 0.5,
                            "validity_days": 1, "data_gb": 1.0, "source_url": "x"}),
    ]
    ours = {"category": "Bundle", "price_omr": 1.0, "validity_days": 1, "data_gb": "500MB"}
    row = competitors.compare_offer(ours, comp)["closest"][0]
    assert row["plan"] == "Daily 1" and row["same_validity_band"]
