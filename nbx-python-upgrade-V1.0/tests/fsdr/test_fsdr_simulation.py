"""Closed-loop simulation smoke test (audit N1..N3) and the test-sample arms option."""
from __future__ import annotations

import json
from datetime import date

import pytest

from nbx.fsdr import config
from nbx.fsdr.datasource import SyntheticSource


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from nbx.fsdr.pipelines.simulate import run_simulation
    root = tmp_path_factory.mktemp("fsdr sim")
    return root, run_simulation(root, date(2026, 10, 1), weeks=2, n=1500, scenarios="baseline", seed=7)


def test_simulation_runs_the_weekly_cycle_and_writes_its_report(sim):
    root, rep = sim
    assert (root / "simulation_report.json").exists() and json.loads((root / "simulation_report.json").read_text())["weeks"] == 2
    assert [w["week"] for w in rep["weekly"]] == ["2026-10-01", "2026-10-08"]
    assert sorted(p.name for p in root.iterdir() if p.name.startswith("2026")) == ["2026-10-01", "2026-10-08", "2026-10-15"]
    assert rep["weekly"][0]["training"]["label_source"] == "targeting"
    assert rep["weekly"][1]["training"]["label_source"] == "campaign_log"         # learns from its own campaign
    for w in rep["weekly"]:
        assert set(w["results"]["arms"]) == {"cg", "free", "cstr", "llm"}


def test_contacted_arms_are_not_below_control(sim):
    """N3: the campaign responses come on top of organic buying, so the bandit / llm arms beat the control."""
    s = sim[1]["summary"]
    assert s["cstr"]["take_rate"] >= s["cg"]["take_rate"] and s["llm"]["take_rate"] >= s["cg"]["take_rate"]
    assert s["cstr"]["uplift_vs_control"] > 0


def test_the_responses_of_week_k_are_the_fulfilments_of_week_k_plus_1(sim):
    from nbx.fsdr import campaign_log
    root, _ = sim
    rew = campaign_log.read_rewards(campaign_log.root_for(root / "2026-10-08"), date(2026, 10, 1), date(2026, 10, 1))
    took = rew[(rew["arm"] != "cg") & (rew["reward_exact_offer"] == 1)]
    assert len(took) > 0                                     # sent offers were accepted and seen at D+7


def test_test_sample_can_hold_a_subset_of_arms_balanced():
    src = SyntheticSource(n=1500, run_date=date(2026, 10, 1), seed=5)
    from nbx.fsdr.synth.world import World
    w = World.create(1500, seed=5, as_of=date(2026, 10, 1), arms=["cg", "cstr"])
    from nbx.fsdr.synth.generator import generate_run
    groups = generate_run(w, date(2026, 10, 1)).tables[config.TARGET_SAMPLE_TABLE]["group"].value_counts()
    assert set(groups.index) == {"cg", "cstr"} and abs(groups["cg"] - groups["cstr"]) / groups.mean() <= 0.25
    full = src.read(config.TARGET_SAMPLE_TABLE)["group"].value_counts()
    assert set(full.index) == set(config.ARMS) and ((full - full.mean()).abs() / full.mean() <= 0.25).all()
