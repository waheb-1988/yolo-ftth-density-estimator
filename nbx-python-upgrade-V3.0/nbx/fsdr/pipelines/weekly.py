"""The weekly cycle in one call (CLI `fsdr-weekly`, run by a scheduler every week on the new run folder D).

    1 validate    contract, keys, joins, windows; a blocked run stops here
    2 feedback    rewards of the campaign logged in D-8..D-2 (skipped the first week: nothing logged yet)
    3 train       bandit on last week's campaign (legacy targeting tables the first week) + the whole base scored
    4 score       the target sample (test-sample table): an action per line, arms free / cstr / llm
    5 offers      cstr and free arms (rules), llm arm (agents; --llm-limit caps the lines decided)
    6 report      last week's campaign per arm: take-up, revenue, uplift vs control with 95 % CI

Each step's status, duration and summary go to outputs/weekly_run_report.json; the run stops at the first failure
and returns it (exit code 1 from the CLI). Steps already done can be skipped with `steps`.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

from nbx.core import tracing
from nbx.fsdr import campaign_log, config
from nbx.fsdr.pipelines import outputs_dir

STEPS = ["validate", "feedback", "train", "score", "offers", "report"]


def _summary(rep: dict, keys: tuple) -> dict:
    return {k: rep.get(k) for k in keys if k in rep}


def run_weekly(data_dir: str | Path, arms: tuple[str, ...] = ("cstr", "free", "llm"), llm: str | None = None,
               llm_limit: int | None = None, steps: list[str] | None = None) -> dict:
    from nbx.fsdr.agents.recommender_agent import run_batch
    from nbx.fsdr.pipelines.feedback import campaign_window, run_feedback, weekly_report
    from nbx.fsdr.pipelines.recommend import run_recommend
    from nbx.fsdr.pipelines.score import run_scoring
    from nbx.fsdr.pipelines.train import load_and_validate, run_training

    data_dir = Path(data_dir)
    run_date = date.fromisoformat(data_dir.name) if data_dir.name[:2] == "20" else None
    todo = steps or STEPS
    report = {"pipeline": "weekly", "run": data_dir.name, "started_at": datetime.now().isoformat(timespec="seconds"),
              "steps": {}, "status": "running"}
    log_root = campaign_log.root_for(data_dir)

    def save():
        (outputs_dir(data_dir) / "weekly_run_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def step(name: str, fn):
        if name not in todo:
            report["steps"][name] = {"status": "skipped", "reason": "not requested"}
            return None
        t0 = time.time()
        print(f"[weekly] {name} ...", flush=True)
        try:
            with tracing.span(f"fsdr_weekly:{name}", "chain", inputs={"run": data_dir.name}) as sp:
                out = fn()
                sp.end(outputs=out if isinstance(out, dict) else {"result": str(out)})
        except Exception as e:  # noqa: BLE001  (a failed step stops the run; the report says which and why)
            report["steps"][name] = {"status": "failed", "seconds": round(time.time() - t0, 1),
                                     "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
            report["status"] = f"failed at {name}"
            save()
            raise
        report["steps"][name] = {"status": out.pop("_status", "done") if isinstance(out, dict) else "done",
                                 "seconds": round(time.time() - t0, 1), **(out if isinstance(out, dict) else {})}
        print(f"[weekly] {name}: {report['steps'][name]['status']} in {report['steps'][name]['seconds']} s", flush=True)
        save()
        return out

    def do_validate():
        _, d, rep = load_and_validate(data_dir, run_date)
        return {"run_date": d.isoformat(), **rep}

    def do_feedback():
        d = run_date or date.fromisoformat(load_and_validate(data_dir)[2]["run_date"])
        lo, hi = campaign_window(d)
        if not len(campaign_log.read_log(log_root, lo, hi)):
            return {"_status": "skipped", "reason": f"no campaign logged in {lo}..{hi} (first week)"}
        rep = run_feedback(data_dir, d)
        return {"rows": rep["rows"], "arms": {a: _summary(m, ("lines", "take_rate_any_btl", "revenue_per_line", "uplift_vs_control"))
                                              for a, m in rep["arms"].items()}}

    def do_train():
        rep = run_training(data_dir, run_date)
        return {"version": rep["version"], "promoted": rep["promoted"], "promotion_rule": rep["promotion_rule"],
                "champion": rep["champion"], "label_source": rep["labels"]["label_source"],
                "labelled_rows": rep["labels"]["labelled_rows"], "eligible_base": rep["base"]["eligible"],
                "base_scored": rep["base"]["eligible"], "features": rep["features"]["model_features"]}

    def do_score():
        rep = run_scoring(data_dir, run_date)
        return {"model": rep["model_version"], "scored_lines": rep["population_report"]["scored_lines"],
                "groups": rep["population_report"]["sample_groups"], "max_psi": rep["drift"]["max_psi"],
                "psi_alerts": rep["drift"]["psi_alerts"]}

    def do_offers():
        out = {}
        for arm in arms:
            t = time.time()
            if arm == "llm":
                if llm_limit == 0:
                    out[arm] = {"status": "skipped", "reason": "--llm-limit 0"}
                    continue
                rep = asyncio.run(run_batch(data_dir, llm=llm, limit=llm_limit))
                out[arm] = _summary(rep, ("customers", "decided_now", "modes", "actions", "override_rate", "offers"))
            else:
                rep = run_recommend(data_dir, arm)
                out[arm] = _summary(rep, ("customers", "with_offers", "without_offer", "offers", "offer_policy", "consent_blocked",
                                          "mean_price"))
            out[arm]["seconds"] = round(time.time() - t, 1)
        return out

    def do_report():
        d = run_date or date.fromisoformat(load_and_validate(data_dir)[2]["run_date"])
        prev = d - timedelta(days=7)
        rep = weekly_report(log_root, prev)
        if not rep["rewarded_lines"]:
            return {"_status": "skipped", "reason": f"no rewards yet for the campaign of {prev} (first week)"}
        return {"week": rep["week"], "arms": {a: _summary(m, ("lines", "take_rate_any_btl", "revenue_per_line", "uplift_vs_control",
                                                             "uplift_test", "detectable_uplift_80pct_power"))
                                              for a, m in rep["arms"].items()}}

    with tracing.span("fsdr_weekly", "chain", inputs={"run": data_dir.name, "arms": list(arms), "llm_limit": llm_limit}) as top:
        for name, fn in (("validate", do_validate), ("feedback", do_feedback), ("train", do_train), ("score", do_score),
                         ("offers", do_offers), ("report", do_report)):
            step(name, fn)
        report["status"] = "done"
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        report["seconds"] = round(sum(s.get("seconds", 0) for s in report["steps"].values()), 1)
        top.end(outputs={"status": report["status"], "seconds": report["seconds"]})
    save()
    return report
