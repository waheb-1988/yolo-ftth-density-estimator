"""Analysis of experiment.py: per run and arm, pooled over the 4 campaigns; across seeds, mean and paired
differences (same world per seed) with a t 95 % CI.   python analyze.py <exp_dir> <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ["free", "cstr", "llm"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def run_metrics(folder: Path) -> dict | None:
    if not (folder / "simulation_report.json").exists():
        return None
    camp = folder / "state" / "campaign"
    rew, log = jsonl(camp / "nba_rewards.jsonl"), jsonl(camp / "nba_campaign_log.jsonl")
    sent = {(r["run_date"], r["arm"], str(r["msisdn"])): r.get("offers") or [] for r in log}
    out = {}
    by_arm: dict[str, list] = {}
    for r in rew:
        by_arm.setdefault(r["arm"], []).append(r)
    cg = by_arm.get("cg", [])
    cg_take = np.mean([r["reward_any_btl"] for r in cg]) if cg else np.nan
    cg_rev = np.mean([r["revenue"] for r in cg]) if cg else np.nan
    for arm, rows in by_arm.items():
        take = np.array([r["reward_any_btl"] for r in rows], float)
        rev = np.array([r["revenue"] for r in rows], float)
        exact = np.array([r["reward_exact_offer"] for r in rows], float)
        offers = [sent.get((r["run_date"], arm, str(r["msisdn"])), []) for r in rows]
        contacted = [o for o in offers if o]
        prices = [x["price"] for o in contacted for x in o]
        out[arm] = {"lines": len(rows), "take_rate": take.mean(), "revenue_per_line": rev.mean(),
                    "exact_offer_take_rate": exact.mean(),
                    "uplift_take": take.mean() - cg_take, "uplift_revenue": rev.mean() - cg_rev,
                    "contacted_share": len(contacted) / len(rows) if rows else np.nan,
                    "offers_per_contacted": np.mean([len(o) for o in contacted]) if contacted else np.nan,
                    "mean_offer_price": np.mean(prices) if prices else np.nan,
                    "revenue_per_taker": rev[take == 1].mean() if take.any() else np.nan}
    rep = json.loads((folder / "simulation_report.json").read_text(encoding="utf-8"))
    out["_promotions"] = [w.get("training", {}).get("promoted") for w in rep.get("weekly", [])]
    return out


def ci(x) -> dict:
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "ci": [None, None], "n": len(x)}
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(x.mean()), "ci": [float(x.mean() - h), float(x.mean() + h)], "n": len(x),
            "p_value": float(stats.ttest_1samp(x, 0).pvalue) if x.std() > 0 else None}


def main(exp: Path, out_path: Path) -> None:
    runs = {}
    for cfg_dir in sorted(p for p in exp.iterdir() if p.is_dir()):
        for scen_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(scen_dir.iterdir()):
                m = run_metrics(seed_dir)
                runs[(cfg_dir.name, scen_dir.name, seed_dir.name)] = m
    failed = [k for k, v in runs.items() if v is None]
    metrics = ["take_rate", "revenue_per_line", "uplift_take", "uplift_revenue", "exact_offer_take_rate",
               "mean_offer_price", "revenue_per_taker", "contacted_share", "offers_per_contacted"]
    summary: dict = {"failed_runs": ["/".join(k) for k in failed], "levels": {}, "paired": {}}
    cfgs = sorted({k[0] for k in runs})
    scens = sorted({k[1] for k in runs})
    for cfg in cfgs:
        for scen in scens:
            rs = [v for k, v in runs.items() if k[0] == cfg and k[1] == scen and v]
            if not rs:
                continue
            summary["levels"][f"{cfg}|{scen}"] = {
                arm: {m: ci([r[arm][m] for r in rs if arm in r]) for m in metrics} for arm in ["cg"] + ARMS}
            summary["levels"][f"{cfg}|{scen}"]["_promotions"] = [r["_promotions"] for r in rs]
    for new in [c for c in cfgs if c != "V0"]:
        for scen in scens:
            seeds = sorted({k[2] for k in runs if k[1] == scen})
            pairs = [(runs.get(("V0", scen, s)), runs.get((new, scen, s))) for s in seeds]
            pairs = [(a, b) for a, b in pairs if a and b]
            if not pairs:
                continue
            summary["paired"][f"{new} - V0|{scen}"] = {
                arm: {m: ci([b[arm][m] - a[arm][m] for a, b in pairs]) for m in metrics} for arm in ARMS}
            # relative change of the uplift (incremental take / revenue)
            for arm in ARMS:
                for m in ("uplift_take", "uplift_revenue"):
                    v0 = np.mean([a[arm][m] for a, _ in pairs])
                    v1 = np.mean([b[arm][m] for _, b in pairs])
                    summary["paired"][f"{new} - V0|{scen}"][arm][f"{m}_relative"] = (v1 - v0) / abs(v0) if v0 else None
    out_path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"{len(runs)} runs, {len(failed)} failed: {summary['failed_runs']}")
    for key, arms in summary["paired"].items():
        print(f"\n== {key}")
        for arm in ARMS:
            a = arms[arm]
            print(f"  {arm:5s} take {a['take_rate']['mean']:+.4f} [{a['take_rate']['ci'][0]:+.4f},{a['take_rate']['ci'][1]:+.4f}]"
                  f"  rev/line {a['revenue_per_line']['mean']:+.4f} [{a['revenue_per_line']['ci'][0]:+.4f},{a['revenue_per_line']['ci'][1]:+.4f}]"
                  f"  uplift_rev rel {a['uplift_revenue_relative']:+.1%}  price {a['mean_offer_price']['mean']:+.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
