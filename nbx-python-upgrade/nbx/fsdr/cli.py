"""FSDR command line (called by run.py: `python run.py fsdr-<command> ...`).

    fsdr-generate  --date 2026-10-22 --n 5000 --scenario baseline[,churn_wave,...] [--seed 42] [--format json|jsonl] [--out DIR] [--force]
    fsdr-validate  --data-dir data/fsdr/2026-10-22 [--date 2026-10-22]
    fsdr-schemas   [--out docs/fsdr_json_schemas]
    fsdr-train     [--data-dir DIR] [--mode logged|simulation] [--reward any_btl|same_service|exact_offer] [--promote auto|never|always]
    fsdr-score     [--data-dir DIR] [--population target|base] [--policy sample|greedy]
    fsdr-recommend --arm free|cstr|llm [--data-dir DIR] [--mapping all_L2|normal] [--llm MODEL] [--concurrency N (default FSDR_AGENT_CONCURRENCY)]
    fsdr-decide    [--data-dir DIR] [--msisdn N] [--llm MODEL] [--trace]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from nbx.fsdr import config


def _date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got '{s}'") from e


def _guard_overwrite(out: Path, force: bool) -> None:
    """Never silently overwrite real exports with dummy data."""
    if not out.exists() or not any(out.iterdir()) or force:
        return
    manifest = out / config.MANIFEST_FILE
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("source") == "generator":
        return
    raise SystemExit(f"{out} already holds data that was not produced by the generator. "
                     "Use another --out, or --force to overwrite it.")


def cmd_generate(a) -> int:
    from nbx.fsdr.synth import generator, scenarios
    from nbx.fsdr.synth.world import World
    from nbx.fsdr.calendar import anchor_date

    run_date = a.date or date.today()
    sc = scenarios.parse(a.scenario)
    t0 = time.time()
    if a.weeks > 1:                                   # one world, consecutive weeks: <root>/<D>, <root>/<D+7>, ...
        root = Path(a.out) if a.out else config.FSDR_DATA_DIR
        runs = generator.generate_weeks(a.n, run_date, a.weeks, sc, a.seed, arms=_arms(a.arms))
        for r in runs:
            _guard_overwrite(root / r.run_date.isoformat(), a.force)
            generator.write_run(r, root / r.run_date.isoformat(), a.format)
        print(f"Generated {a.weeks} weekly runs from one world: {', '.join(r.run_date.isoformat() for r in runs)} -> {root}")
        run, out = runs[-1], root / runs[-1].run_date.isoformat()
    else:
        out = Path(a.out) if a.out else config.FSDR_DATA_DIR / run_date.isoformat()
        _guard_overwrite(out, a.force)
        world = World.create(a.n, seed=a.seed, scenarios=sc, as_of=run_date, arms=_arms(a.arms))
        run = generator.generate_run(world, run_date)
        generator.write_run(run, out, a.format)
    rows = run.manifest["row_counts"]
    print(f"Generated {len(rows)} tables for D={run_date} (T={anchor_date(run_date)}), n={a.n}, scenarios={','.join(sc)}, "
          f"seed={a.seed} in {time.time() - t0:.1f}s -> {out}")
    width = max(len(t) for t in rows)
    for t, r in rows.items():
        print(f"  {t:<{width}}  {r:>8,}")
    if run.manifest["quality_issues_injected"]:
        print("Injected data-quality issues:", json.dumps(run.manifest["quality_issues_injected"]))
    return 0


def cmd_validate(a) -> int:
    from nbx.fsdr.validation import validate_folder

    rep = validate_folder(a.data_dir, a.date)
    c = rep.counts()
    print(f"Validation of {a.data_dir}: {rep.status.upper()}  (blocking {c['blocking']}, warnings {c['warning']}, info {c['info']})")
    for i in rep.issues:
        print(f"  [{i.level:8s}] {i.check:18s} {i.table}{'.' + i.column if i.column else ''}: {i.message}")
    print(f"Report: {Path(a.data_dir) / config.VALIDATION_MD}")
    return 1 if rep.status == "blocked" else 0


def _arms(text: str | None) -> list[str] | None:
    return [s.strip().lower() for s in text.split(",") if s.strip()] if text else None


def cmd_feedback(a) -> int:
    from nbx.fsdr.pipelines.feedback import run_feedback

    d = _data_dir(a)
    rep = run_feedback(d)
    print(f"Feedback of the campaign {rep['campaign_window'][0]}..{rep['campaign_window'][1]} with the fulfilments of "
          f"{rep['reward_window'][0]}..{rep['reward_window'][1]}: {rep['rows']} rows ({rep['new_reward_rows']} new)")
    _print_arms(rep)
    return 0


def _print_arms(rep: dict) -> None:
    for arm, m in rep.get("arms", {}).items():
        up = f"  uplift vs cg {m['uplift_vs_control']:+.4f}" if "uplift_vs_control" in m else ""
        print(f"  {arm:5s} lines {m['lines']:5d}  take rate {m['take_rate_any_btl']:.4f}  revenue/line {m['revenue_per_line']:.3f}{up}")
    if "llm" in rep:
        print(f"  llm   override rate {rep['llm']['override_rate']:.4f}  bandit agreement {rep['llm']['bandit_agreement']:.4f}")


def cmd_simulate(a) -> int:
    from nbx.fsdr.pipelines.simulate import run_simulation

    root = Path(a.out) if a.out else config.FSDR_DATA_DIR / "simulation"
    rep = run_simulation(root, a.date or date.today(), a.weeks, a.n, a.scenario,
                         [x for x in (_arms(a.arms) or []) if x != "cg"] or None, a.seed, a.llm)
    print(f"Simulated {a.weeks} week(s) on {a.n} lines in {rep['seconds']}s -> {root / 'simulation_report.json'}")
    for w in rep["weekly"]:
        print(f" week {w['week']}: model {w['training']['version']} ({w['training']['label_source']} labels)")
        _print_arms(w.get("results", {}))
    print(" mean over the weeks:", json.dumps(rep["summary"]))
    return 0


def cmd_agent(a) -> int:
    import asyncio

    from nbx.fsdr.agents.ops_agents import run_ops_agent

    out = asyncio.run(run_ops_agent(a.agent, _data_dir(a), llm=a.llm))
    print(f"{a.agent} agent on {out['run']}: mode {out['mode']} ({out['finalized_by']}), {out['seconds']}s")
    if a.trace:
        for e in out["trace"]:
            txt = e.get("text") or e.get("tool") or ""
            print(f"  {e['t_ms']:>7} ms  {e['type']:<12} {txt}")
    print(json.dumps(out["final"], indent=2, ensure_ascii=False, default=str))
    for f in out["findings"]:
        print("  guard:", f)
    return 0


def cmd_report(a) -> int:
    from nbx.fsdr import campaign_log
    from nbx.fsdr.pipelines.feedback import weekly_report

    d = _data_dir(a)
    rep = weekly_report(campaign_log.root_for(d), a.week or date.fromisoformat(d.name))
    print(f"Week {rep['week']}: {rep['logged_lines']} lines logged {rep['logged_per_arm']}, {rep['rewarded_lines']} rewarded")
    _print_arms(rep)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    return 0


def _data_dir(a) -> Path:
    if a.data_dir:
        return Path(a.data_dir)
    runs = sorted(p for p in config.FSDR_DATA_DIR.glob("20??-??-??") if p.is_dir()) if config.FSDR_DATA_DIR.exists() else []
    if not runs:
        raise SystemExit("no run folder: pass --data-dir or run fsdr-generate first")
    return runs[-1]


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_train(a) -> int:
    from nbx.fsdr.pipelines.train import run_training

    d = _data_dir(a)
    rep = run_training(d, mode=a.mode, reward_def=a.reward, passes=a.passes, epsilon=a.epsilon, promote=a.promote)
    print(f"Trained {rep['version']} on {d} in {rep['seconds']}s -- promoted: {rep['promoted']} ({rep['promotion_rule']}); "
          f"champion: {rep['champion']}")
    _print_json({"labels": rep["labels"], "training": rep["training"], "evaluation": rep["evaluation"],
                 "scored_base_action_distribution": rep["scored_base_action_distribution"]})
    print(f"Report: {d / 'outputs' / 'nba_training_report.json'}")
    return 0


def cmd_score(a) -> int:
    from nbx.fsdr.pipelines.score import run_scoring

    d = _data_dir(a)
    rep = run_scoring(d, population=a.population, policy=a.policy, allow_stale=a.allow_stale)
    print(f"Scored {rep['population_report'].get('scored_lines', rep['population_report'].get('eligible'))} lines with "
          f"{rep['model_version']} (policy={rep['policy']}) in {rep['seconds']}s")
    _print_json({"action_distribution_per_arm": rep["action_distribution_per_arm"],
                 "drift": {k: rep["drift"][k] for k in ("max_psi", "psi_alerts", "action_mix_shift_tv")}})
    return 0


def cmd_recommend(a) -> int:
    d = _data_dir(a)
    if a.arm == "llm":
        import asyncio

        from nbx.fsdr.agents.recommender_agent import run_batch

        rep = asyncio.run(run_batch(d, llm=a.llm, concurrency=a.concurrency, limit=a.limit, resume=not a.restart,
                                    progress=lambda i, n: print(f"  {i}/{n}", end="\r")))
    else:
        from nbx.fsdr.pipelines.recommend import run_recommend

        rep = run_recommend(d, a.arm, mapping=a.mapping)
    _print_json(rep)
    print(f"Outputs: {d / 'outputs'} (recos_monthly_{a.arm}.json, recos_weekly_{a.arm}.json)")
    return 0


def cmd_decide(a) -> int:
    import asyncio

    from nbx.fsdr.agents.recommender_agent import decide
    from nbx.fsdr.agents.tools import FsdrWorkspace

    ws = FsdrWorkspace(_data_dir(a))
    msisdn = a.msisdn or ws.lines("llm")[0]
    out = asyncio.run(decide(ws, str(msisdn), "llm", a.llm))
    if a.trace:
        for e in out["trace"]:
            what = e.get("tool") or e.get("text") or e.get("name") or ""
            print(f"  {e['t_ms']:>6} ms  {e['type']:<12} {e.get('agent', ''):<16} {str(what)[:110]}")
    d = out["decision"]
    _print_json({k: d[k] for k in ("msisdn", "mode", "action", "policy_source", "propensity", "bandit_top_action", "offers",
                                   "overrides", "critic", "competitor", "message", "justification")})
    return 0


def cmd_batch(a) -> int:
    import asyncio
    import csv

    from nbx.fsdr.agents.tools import FsdrWorkspace
    from nbx.fsdr.pipelines import outputs_dir
    from nbx.fsdr.pipelines.batch import flat_row, recommend_batch

    d = _data_dir(a)
    ws = FsdrWorkspace(d)
    msisdns = [m.strip() for m in (a.msisdns or "").replace(";", ",").split(",") if m.strip()] or ws.lines(a.arm)[: a.limit]
    rows = []
    for i in range(0, len(msisdns), a.batch_size):
        chunk = msisdns[i:i + a.batch_size]
        rows += [flat_row(r) for r in asyncio.run(recommend_batch(ws, chunk, a.arm, a.llm))]
        print(f"  batch {i // a.batch_size + 1}: {len(rows)}/{len(msisdns)} customers")
    path = outputs_dir(d) / f"batch_recos_{a.arm}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["msisdn"])
        w.writeheader()
        w.writerows(rows)
    for r in rows[: a.show]:
        print(f"{r['msisdn']}  {r['action']:<16} offers: {', '.join(x for x in (r['offer_1'], r['offer_2'], r['offer_3'], r['offer_4']) if x) or '-'}")
        print(f"{'':10}roaming: {', '.join(x for x in (r['roaming_1'], r['roaming_2']) if x) or '-'} | international: {r['international']}")
    print(f"{len(rows)} customers -> {path}")
    return 0


def cmd_schemas(a) -> int:
    from nbx.fsdr import schemas

    paths = schemas.export_json_schemas(Path(a.out) if a.out else None)
    print(f"Wrote {len(paths)} JSON Schemas (+ _index.json) to {paths[0].parent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run.py fsdr", description="FSDR data commands")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("fsdr-generate", help="write dummy JSON files for a run date")
    g.add_argument("--date", type=_date, help="run date D (default: today)")
    g.add_argument("--n", type=int, default=config.GENERATOR_DEFAULT_N, help=f"lines (default {config.GENERATOR_DEFAULT_N}, max {config.GENERATOR_MAX_N})")
    g.add_argument("--scenario", default="baseline", help=f"comma-separated: {','.join(config.SCENARIOS)}")
    g.add_argument("--seed", type=int, default=config.GENERATOR_SEED)
    g.add_argument("--format", choices=["json", "jsonl"], default="json")
    g.add_argument("--weeks", type=int, default=1, help="consecutive weekly runs from one world (default 1)")
    g.add_argument("--arms", help=f"test-sample groups, comma-separated (default {','.join(config.ARMS)})")
    g.add_argument("--out", help="output folder (default data/fsdr/<date>; with --weeks > 1, the root of the <date> folders)")
    g.add_argument("--force", action="store_true", help="overwrite a folder that holds non-generated data")
    g.set_defaults(fn=cmd_generate)

    v = sub.add_parser("fsdr-validate", help="validate a folder of JSON files")
    v.add_argument("--data-dir", required=True)
    v.add_argument("--date", type=_date, help="run date (default: manifest, then folder name)")
    v.set_defaults(fn=cmd_validate)

    s = sub.add_parser("fsdr-schemas", help="export one JSON Schema per table")
    s.add_argument("--out", help=f"output folder (default {config.JSON_SCHEMA_DIR})")
    s.set_defaults(fn=cmd_schemas)

    t = sub.add_parser("fsdr-train", help="train the NBA bandit on a run folder")
    t.add_argument("--data-dir", help="run folder (default: latest data/fsdr/<date>)")
    t.add_argument("--mode", choices=["logged", "simulation"], default=config.TRAIN_MODE)
    t.add_argument("--reward", choices=config.REWARD_DEFINITIONS, default=config.REWARD_DEFINITION)
    t.add_argument("--passes", type=int, default=config.TRAIN_PASSES)
    t.add_argument("--epsilon", type=float, default=config.VW_EPSILON)
    t.add_argument("--promote", choices=["auto", "never", "always"], default="auto")
    t.set_defaults(fn=cmd_train)

    sc = sub.add_parser("fsdr-score", help="score the target sample (or the base) with the champion")
    sc.add_argument("--data-dir")
    sc.add_argument("--population", choices=["target", "base"], default="target")
    sc.add_argument("--policy", choices=["sample", "greedy"], default=config.SCORING_POLICY)
    sc.add_argument("--allow-stale", action="store_true", help=f"score even if the model is older than {config.MAX_MODEL_AGE_DAYS} days")
    sc.set_defaults(fn=cmd_score)

    r = sub.add_parser("fsdr-recommend", help="offers for one arm (free / cstr rules, llm = agents)")
    r.add_argument("--data-dir")
    r.add_argument("--arm", choices=["free", "cstr", "llm"], required=True)
    r.add_argument("--mapping", choices=["all_L2", "normal"], help="generosity mapping (default: config)")
    r.add_argument("--llm", help="LLM model for the llm arm (default: DEFAULT_LLM; offline backend = rule-based)")
    r.add_argument("--concurrency", type=int, default=config.AGENT_CONCURRENCY)
    r.add_argument("--limit", type=int)
    r.add_argument("--restart", action="store_true", help="ignore the llm-arm checkpoint")
    r.set_defaults(fn=cmd_recommend)

    dc = sub.add_parser("fsdr-decide", help="one llm-arm decision with the full agent trace")
    dc.add_argument("--data-dir")
    dc.add_argument("--msisdn")
    dc.add_argument("--llm")
    dc.add_argument("--trace", action="store_true")
    dc.set_defaults(fn=cmd_decide)

    b = sub.add_parser("fsdr-batch", help="batch flow: action -> top 4 offers -> top 2 roaming -> international")
    b.add_argument("--data-dir")
    b.add_argument("--msisdns", help="comma-separated list (default: the first --limit customers of the arm)")
    b.add_argument("--arm", choices=["llm", "cstr", "free"], default="llm")
    b.add_argument("--llm")
    b.add_argument("--limit", type=int, default=100)
    b.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    b.add_argument("--show", type=int, default=5, help="customers printed to the console")
    b.set_defaults(fn=cmd_batch)

    fb = sub.add_parser("fsdr-feedback", help="rewards of last week's campaign (D-8..D-2) from this run's fulfilments")
    fb.add_argument("--data-dir", help="run folder D (default: latest data/fsdr/<date>)")
    fb.set_defaults(fn=cmd_feedback)

    sm = sub.add_parser("fsdr-simulate", help="closed-loop simulation over N weeks on the hidden world")
    sm.add_argument("--weeks", type=int, default=3)
    sm.add_argument("--n", type=int, default=2000)
    sm.add_argument("--date", type=_date, help="first run date (default: today)")
    sm.add_argument("--scenario", default="baseline")
    sm.add_argument("--arms", help="arms to run besides the control cg, comma-separated (default free,cstr,llm)")
    sm.add_argument("--seed", type=int, default=config.GENERATOR_SEED)
    sm.add_argument("--llm", help="LLM model for the llm arm (offline backend = rule-based)")
    sm.add_argument("--out", help="simulation root (default data/fsdr/simulation)")
    sm.set_defaults(fn=cmd_simulate)

    ag = sub.add_parser("fsdr-agent", help="data / training / scoring / monitor agent on a run")
    ag.add_argument("--agent", choices=["data", "training", "scoring", "monitor"], required=True)
    ag.add_argument("--data-dir")
    ag.add_argument("--llm", help="LLM model (offline backend = rule-based path)")
    ag.add_argument("--trace", action="store_true")
    ag.set_defaults(fn=cmd_agent)

    rp = sub.add_parser("fsdr-report", help="weekly report: take rate and revenue per arm, uplift vs control")
    rp.add_argument("--data-dir", help="a run folder of the data root (default: latest)")
    rp.add_argument("--week", type=_date, help="campaign week (run date D; default: the folder's date)")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
