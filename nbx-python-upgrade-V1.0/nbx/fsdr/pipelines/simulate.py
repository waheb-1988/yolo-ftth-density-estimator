"""Closed-loop simulation (audit N1..N3): the weekly cycle on the hidden world, week after week.

For week k (D = start + 7k), one world for the whole run:
  1. generate the JSON tables of D; from k >= 1 last week's logged campaign is the targeting, and its
     accept / ignore draws (hidden response function) are this week's fulfilments
  2. feedback of last week's campaign (rewards per arm, uplift vs control)
  3. train (labels from the campaign log from week 1 on), score, recommend for the arms (free / cstr / llm)
     -- each arm is logged; cg is the control
  4. D += 7; behaviour evolves with the calendar (scenario drifts, churned lines)
One extra generation after the last campaign collects its rewards. Output: <root>/simulation_report.json.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from nbx.fsdr import campaign_log, config
from nbx.fsdr.synth import generator, scenarios as scen
from nbx.fsdr.synth.world import World

SIM_ARMS = ["free", "cstr", "llm"]


def run_simulation(root: str | Path, start: date, weeks: int = 3, n: int = 2000, scenarios="baseline",
                   arms: list[str] | None = None, seed: int = config.GENERATOR_SEED, llm: str | None = None) -> dict:
    from nbx.fsdr.agents.recommender_agent import run_batch
    from nbx.fsdr.agents.tools import FsdrWorkspace
    from nbx.fsdr.pipelines.feedback import run_feedback
    from nbx.fsdr.pipelines.recommend import run_recommend
    from nbx.fsdr.pipelines.score import run_scoring
    from nbx.fsdr.pipelines.train import run_training

    t0 = datetime.now()
    arms = [a for a in (arms or SIM_ARMS) if a in SIM_ARMS]
    root = Path(root)
    models = root / config.FSDR_STATE_DIR.name / "models"
    dates = [start + timedelta(days=7 * k) for k in range(weeks + 1)]          # + 1: rewards of the last campaign
    world = World.create(n, seed=seed, scenarios=scen.parse(scenarios), as_of=dates[-1], arms=["cg"] + arms)
    weekly = []
    for k, d in enumerate(dates):
        folder = root / d.isoformat()
        log_root = campaign_log.root_for(folder)
        prev = campaign_log.read_log(log_root, d - timedelta(days=7), d - timedelta(days=1)) if k else None
        generator.write_run(generator.generate_run(world, d, prev if k else None), folder)
        week = {"week": d.isoformat()}
        if k:
            fb = run_feedback(folder)
            weekly[-1]["results"] = {"arms": fb["arms"], **({"llm": fb["llm"]} if "llm" in fb else {})}
        if k == weeks:
            break
        tr = run_training(folder, root=models)
        week["training"] = {"version": tr["version"], "promoted": tr["promoted"], "label_source": tr["labels"]["label_source"],
                            "labelled_rows": tr["labels"]["labelled_rows"]}
        run_scoring(folder, root=models)
        for arm in arms:
            if arm == "llm":
                rep = asyncio.run(run_batch(folder, llm=llm, ws=FsdrWorkspace(folder, models)))
                week["llm"] = {"modes": rep.get("modes"), "override_rate": rep.get("override_rate")}
            else:
                run_recommend(folder, arm)
        weekly.append(week)

    report = {"pipeline": "simulate", "start": start.isoformat(), "weeks": weeks, "n": n, "seed": seed,
              "scenarios": scen.parse(scenarios), "arms": ["cg"] + arms, "weekly": weekly,
              "summary": _summary(weekly), "pooled": pooled(campaign_log.read_rewards(campaign_log.root_of(root))),
              "seconds": round((datetime.now() - t0).total_seconds(), 1)}
    (root / "simulation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def pooled(rew) -> dict:
    """V1.0 review #13: all simulated weeks pooled per arm, uplift vs control with a 95 % CI and p-value."""
    from nbx.fsdr import stats

    if not len(rew):
        return {}
    cg = rew[rew["arm"] == "cg"]
    out = {}
    for arm, g in rew.groupby("arm"):
        m = {"lines": int(len(g)), "take_rate": round(float(g["reward_any_btl"].mean()), 4),
             "revenue_per_line": round(float(g["revenue"].mean()), 4),
             "exact_offer_take_rate": round(float(g["reward_exact_offer"].mean()), 4)}
        if arm != "cg" and len(cg):
            m["uplift"] = stats.diff_ci(g["reward_any_btl"], cg["reward_any_btl"])
            m["revenue_uplift"] = stats.diff_ci(g["revenue"], cg["revenue"])
        out[arm] = m
    return out


def _summary(weekly: list[dict]) -> dict:
    """Mean take rate / revenue per arm over the simulated weeks, and the uplift vs control."""
    acc: dict[str, dict[str, list]] = {}
    for w in weekly:
        for arm, m in (w.get("results") or {}).get("arms", {}).items():
            a = acc.setdefault(arm, {"take": [], "rev": []})
            a["take"].append(m["take_rate_any_btl"])
            a["rev"].append(m["revenue_per_line"])
    out = {arm: {"take_rate": round(sum(v["take"]) / len(v["take"]), 4), "revenue_per_line": round(sum(v["rev"]) / len(v["rev"]), 4)}
           for arm, v in acc.items() if v["take"]}
    if "cg" in out:
        for arm, m in out.items():
            if arm != "cg":
                m["uplift_vs_control"] = round(m["take_rate"] - out["cg"]["take_rate"], 4)
    return out
