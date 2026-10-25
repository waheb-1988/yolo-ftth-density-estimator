"""Feedback at D+7 and the weekly report (audit M2, N2).

    run_feedback(<folder of D>)   the campaign logged in D-8..D-2 (BR-21) meets the fulfilments of D-7..D-1
                                  (BR-23): rewards for the 3 definitions -> state/campaign/nba_rewards.jsonl
                                  (append-only) + outputs/nba_rewards.json + outputs/nba_feedback_report.json
    weekly_report(root, D)        per arm (cg / free / cstr / llm): lines, take rate (3 definitions), revenue per
                                  line, uplift vs control, override rate, bandit-vs-LLM agreement

The reward rows keep the logged propensity, so training can use them as labels (labels.build_labels_from_log).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nbx.fsdr import campaign_log, config
from nbx.fsdr.labels import load_fulfilments
from nbx.fsdr.offer_rules import load_catalog
from nbx.fsdr.pipelines import write_report, write_table
from nbx.fsdr.pipelines.train import load_and_validate


def campaign_window(run_date: date) -> tuple[date, date]:
    lo, hi = config.TARGETING_WINDOW
    return run_date + timedelta(days=lo), run_date + timedelta(days=hi)


def run_feedback(data_dir: str | Path, run_date: date | None = None) -> dict:
    t0 = datetime.now()
    src, run_date, _ = load_and_validate(data_dir, run_date)
    root = campaign_log.root_for(data_dir)
    start, end = campaign_window(run_date)
    log = campaign_log.read_log(root, start, end)
    if not len(log):
        raise RuntimeError(f"no campaign logged between {start} and {end} in {root}: run recommend for that week first")
    rows = campaign_log.compute_rewards(log, load_fulfilments(src, run_date), load_catalog(src))
    added = campaign_log.append_rewards(root, rows)
    write_table(pd.DataFrame(rows), data_dir, "nba_rewards")
    report = {"pipeline": "feedback", "run_date": run_date.isoformat(), "campaign_window": [start.isoformat(), end.isoformat()],
              "reward_window": [(run_date + timedelta(days=config.REWARD_WINDOW[0])).isoformat(),
                                (run_date + timedelta(days=config.REWARD_WINDOW[1])).isoformat()],
              "rows": len(rows), "new_reward_rows": added,
              **summarise(pd.DataFrame(rows), log),
              "seconds": round((datetime.now() - t0).total_seconds(), 1)}
    write_report(report, data_dir, "nba_feedback_report")
    return report


def summarise(rew: pd.DataFrame, log: pd.DataFrame) -> dict:
    """Per-arm take rates, revenue and uplift vs control; llm-arm override rate and bandit agreement."""
    if not len(rew):
        return {"arms": {}}
    arms = {}
    for arm, g in rew.groupby("arm"):
        arms[arm] = {"lines": int(len(g)),
                     **{f"take_rate_{d}": round(float(g[f"reward_{d}"].mean()), 4) for d in config.REWARD_DEFINITIONS},
                     "revenue_per_line": round(float(g["revenue"].mean()), 4)}
    cg = arms.get("cg")
    for arm, m in arms.items():
        if cg and arm != "cg":
            m["uplift_vs_control"] = round(m["take_rate_any_btl"] - cg["take_rate_any_btl"], 4)
            m["revenue_uplift_vs_control"] = round(m["revenue_per_line"] - cg["revenue_per_line"], 4)
    out = {"arms": arms}
    llm = log[log["arm"] == "llm"] if len(log) else log
    if len(llm):
        top = llm["bandit_pmf"].map(lambda p: max(p, key=p.get) if isinstance(p, dict) and p else None)
        out["llm"] = {"lines": int(len(llm)),
                      "bandit_agreement": round(float((llm["action"] == top).mean()), 4),
                      "override_rate": round(float(((llm["policy_source"] == "llm") & (llm["action"] != top)).mean()), 4),
                      "policy_source": llm["policy_source"].value_counts().to_dict()}
    return out


def weekly_report(root: Path, week: date) -> dict:
    """Report of the campaign run in the week starting on `week` (its run date D). Rewards appear once the
    feedback of D+7 has run."""
    start, end = week, week + timedelta(days=6)
    log = campaign_log.read_log(root, start, end)
    rew = campaign_log.read_rewards(root, start, end)
    return {"week": week.isoformat(), "campaign_window": [start.isoformat(), end.isoformat()],
            "logged_lines": int(len(log)), "rewarded_lines": int(len(rew)),
            "logged_per_arm": log["arm"].value_counts().to_dict() if len(log) else {},
            **summarise(rew, log)}
