"""Feedback at D+7 and the weekly report (audit M2, N2).

    run_feedback(<folder of D>)   the campaign logged in D-8..D-2 (BR-21) meets the fulfilments of D-7..D-1
                                  (BR-23): rewards for the 3 definitions -> state/campaign/nba_rewards.jsonl
                                  (append-only) + outputs/nba_rewards.json + outputs/nba_feedback_report.json
    weekly_report(root, D)        per arm (cg / free / cstr / llm): lines, take rate (3 definitions), revenue per
                                  line, uplift vs control, override rate, bandit-vs-LLM agreement

V1.0: every uplift comes with a 95 % CI and p-value, each arm with the smallest uplift its size can detect
(power analysis, review #13), and the feedback report has a fairness view per nationality group (review #7).

The reward rows keep the logged propensity, so training can use them as labels (labels.build_labels_from_log).
"""
from __future__ import annotations

from nbx.core import tracing

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nbx.fsdr import campaign_log, config, stats
from nbx.fsdr.features import nationality_groups
from nbx.fsdr.labels import load_fulfilments
from nbx.fsdr.offer_rules import load_catalog
from nbx.fsdr.pipelines import write_report, write_table
from nbx.fsdr.pipelines.train import load_and_validate


def campaign_window(run_date: date) -> tuple[date, date]:
    lo, hi = config.TARGETING_WINDOW
    return run_date + timedelta(days=lo), run_date + timedelta(days=hi)


@tracing.traced_pipeline("fsdr_feedback")
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
              **summarise(pd.DataFrame(rows), log, nationality_groups(src)),
              "seconds": round((datetime.now() - t0).total_seconds(), 1)}
    write_report(report, data_dir, "nba_feedback_report")
    return report


def summarise(rew: pd.DataFrame, log: pd.DataFrame, groups: pd.Series | None = None) -> dict:
    """Per-arm take rates, revenue and uplift vs control (with 95 % CI, p-value and the detectable uplift);
    llm-arm override rate and bandit agreement; fairness per nationality group when `groups` is given."""
    if not len(rew):
        return {"arms": {}}
    arms = {}
    for arm, g in rew.groupby("arm"):
        arms[arm] = {"lines": int(len(g)),
                     **{f"take_rate_{d}": round(float(g[f"reward_{d}"].mean()), 4) for d in config.REWARD_DEFINITIONS},
                     "revenue_per_line": round(float(g["revenue"].mean()), 4)}
    cg = arms.get("cg")
    cg_rows = rew[rew["arm"] == "cg"]
    for arm, m in arms.items():
        if cg and arm != "cg":
            t = rew[rew["arm"] == arm]
            m["uplift_vs_control"] = round(m["take_rate_any_btl"] - cg["take_rate_any_btl"], 4)
            m["revenue_uplift_vs_control"] = round(m["revenue_per_line"] - cg["revenue_per_line"], 4)
            m["uplift_test"] = stats.diff_ci(t["reward_any_btl"], cg_rows["reward_any_btl"])
            m["revenue_uplift_test"] = stats.diff_ci(t["revenue"], cg_rows["revenue"])
            m["detectable_uplift_80pct_power"] = stats.mde(cg["take_rate_any_btl"], len(t), len(cg_rows))
    out = {"arms": arms}
    if groups is not None and len(groups):
        out["fairness"] = fairness(rew, log, groups)
    llm = log[log["arm"] == "llm"] if len(log) else log
    if len(llm):
        top = llm["bandit_pmf"].map(lambda p: max(p, key=p.get) if isinstance(p, dict) and p else None)
        out["llm"] = {"lines": int(len(llm)),
                      "bandit_agreement": round(float((llm["action"] == top).mean()), 4),
                      "override_rate": round(float(((llm["policy_source"] == "llm") & (llm["action"] != top)).mean()), 4),
                      "policy_source": llm["policy_source"].value_counts().to_dict()}
    return out


def fairness(rew: pd.DataFrame, log: pd.DataFrame, groups: pd.Series) -> dict:
    """Review #7 (Ascarza & Israeli 2022): per arm and nationality group -- lines, take rate, revenue per line,
    mean price of the offers sent and the action mix; plus the min / max ratios across groups."""
    g = rew.assign(GROUP=rew["msisdn"].astype(str).map(groups).fillna("unknown"))
    price = {}
    if len(log):
        for r in log.itertuples(index=False):
            ps = [o["price"] for o in (r.offers or []) if o.get("price") is not None]
            if ps:
                price[(r.arm, str(r.msisdn))] = sum(ps) / len(ps)
    g["OFFER_PRICE"] = [price.get((a, str(m))) for a, m in zip(g["arm"], g["msisdn"])]
    out = {}
    for arm, ga in g.groupby("arm"):
        rows = {}
        for grp, gg in ga.groupby("GROUP"):
            mix = gg["action"].value_counts(normalize=True).round(3).to_dict()
            rows[grp] = {"lines": int(len(gg)), "take_rate": round(float(gg["reward_any_btl"].mean()), 4),
                         "revenue_per_line": round(float(gg["revenue"].mean()), 4),
                         "mean_offer_price": round(float(gg["OFFER_PRICE"].dropna().mean()), 3) if gg["OFFER_PRICE"].notna().any() else None,
                         "action_mix": mix}
        big = {k: v for k, v in rows.items() if v["lines"] >= 30}          # small groups make noisy ratios
        tr = [v["take_rate"] for v in big.values()]
        pr = [v["mean_offer_price"] for v in big.values() if v["mean_offer_price"]]
        out[arm] = {"groups": rows,
                    "take_rate_ratio_min_max": round(min(tr) / max(tr), 3) if len(tr) > 1 and max(tr) > 0 else None,
                    "offer_price_ratio_min_max": round(min(pr) / max(pr), 3) if len(pr) > 1 else None}
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
