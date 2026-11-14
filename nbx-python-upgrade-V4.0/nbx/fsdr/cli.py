"""FSDR command line (called by run.py: `python run.py fsdr-<command> ...`).

    fsdr-generate  --date 2026-10-22 [--n 5000 | --mode small|medium|full] --scenario baseline[,churn_wave,...] [--seed 42]
                   [--format csv.gz|json|jsonl|csv|parquet] [--layout folders|flat] [--weeks N] [--out DIR] [--force]
    fsdr-load      --data-dir data/input/2026-10-22 [--pipeline train_week0|train_weekly] [--force]   step T0 (①)
    fsdr-validate  --data-dir data/input/2026-10-22 [--date 2026-10-22]
    fsdr-schemas   [--out docs/fsdr_json_schemas]
    fsdr-train     [--data-dir DIR] [--mode logged|simulation] [--reward any_btl|same_service|exact_offer] [--promote auto|never|always]
    fsdr-score     [--data-dir DIR] [--population target|base] [--policy sample|greedy]
    fsdr-recommend --arm free|cstr|llm [--data-dir DIR] [--mapping all_L2|normal] [--llm MODEL] [--concurrency N (default FSDR_AGENT_CONCURRENCY)]
    fsdr-decide    [--data-dir DIR] [--msisdn N] [--llm MODEL] [--trace]
    fsdr-weekly    [--data-dir DIR] [--arms cstr,free,llm] [--llm-limit N] [--steps validate,...]   the whole weekly cycle
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
    if a.mode:
        a.n = config.GENERATOR_MODES[a.mode]
    if not 1 <= a.n <= config.GENERATOR_MAX_N:
        raise SystemExit(f"--n must be between 1 and {config.GENERATOR_MAX_N:,}")
    t0 = time.time()
    if a.layout == "split" and a.out:                 # split needs <data>/training/<D>: any other --out = one folder
        parent = Path(a.out) if a.weeks > 1 else Path(a.out).parent
        if parent.name != config.TRAINING_FOLDER:
            print(f"--out {a.out} is not under a '{config.TRAINING_FOLDER}/' folder: writing one folder (layout folders)")
            a.layout = "folders"
    default_root = config.FSDR_TRAINING_DIR if a.layout == "split" else config.FSDR_INPUT_DIR
    if a.weeks > 1:                                   # one world, consecutive weeks: <root>/<D>, <root>/<D+7>, ...
        root = Path(a.out) if a.out else default_root
        runs = generator.generate_weeks(a.n, run_date, a.weeks, sc, a.seed, arms=_arms(a.arms))
        for r in runs:
            _guard_overwrite(root / r.run_date.isoformat(), a.force)
            generator.write_run(r, root / r.run_date.isoformat(), a.format, a.layout)
        print(f"Generated {a.weeks} weekly runs from one world: {', '.join(r.run_date.isoformat() for r in runs)} -> {root}")
        run, out = runs[-1], root / runs[-1].run_date.isoformat()
    else:
        out = Path(a.out) if a.out else default_root / run_date.isoformat()
        _guard_overwrite(out, a.force)
        if a.layout == "split":
            _guard_overwrite(generator.split_twins(out)[1], a.force)
        world = World.create(a.n, seed=a.seed, scenarios=sc, as_of=run_date, arms=_arms(a.arms))
        run = generator.generate_run(world, run_date)
        generator.write_run(run, out, a.format, a.layout)
    rows = run.manifest["row_counts"]
    outs = list(generator.split_twins(out)) if a.layout == "split" else [out]
    size = sum(p.stat().st_size for o in outs for p in o.rglob("*") if p.is_file())
    print(f"Generated {len(rows)} tables for D={run_date} (T={anchor_date(run_date)}), n={a.n:,}, scenarios={','.join(sc)}, "
          f"seed={a.seed}, {a.format} in {a.layout}, {size / 1e6:,.1f} MB in {time.time() - t0:.1f}s -> "
          f"{' + '.join(str(o) for o in outs)}")
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


def cmd_load(a) -> int:
    from nbx.fsdr.contract import data_agent

    folder = Path(a.data_dir) if a.data_dir else config.training_folder((a.date or date.today()).isoformat())
    rep = data_agent.run_t0(folder, a.pipeline, a.date, force=a.force)
    tag = " (skipped: same inputs, already done)" if rep.get("skipped") else ""
    print(f"T0_load {rep['status'].upper()}{tag} -- {rep['pipeline']}, {rep['tables']} tables, {rep['rows_in']:,} rows, "
          f"layout {rep['layout']}, validation {rep['validation']} {rep['issue_counts']}")
    print(rep["explanation"])
    if rep.get("cache"):
        c = rep["cache"]
        print(f"Parquet cache: {c['dir']} ({c['bytes'] / 1e6:,.1f} MB, {c['partitions']} partitions, {c['seconds']}s)")
    if rep.get("scoring_source") and rep["scoring_source"] != rep["source"]:
        print(f"Read as one run: {rep['training_source']}  +  {rep['scoring_source']}"
              + (f"  ({rep['misfiled_tables']} table(s) in the wrong twin, see misfiled_tables.json)"
                 if rep.get("misfiled_tables") else ""))
    print(f"Step outputs: {rep['step_dir']}")
    return 1 if rep["status"] == "blocked" else 0


def cmd_features(a) -> int:
    from nbx.fsdr.pipelines import training

    run_date = a.date or _latest_run_date()
    if run_date is None:
        print("No run yet: run fsdr-load first.")
        return 1
    try:
        rep = training.run_features(run_date, workers=a.workers, force=a.force)
    except training.StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    t2, t3 = rep["T2_features"], rep["T3_segments"]
    skip = lambda r: " (skipped: same inputs, already done)" if r.get("skipped") else ""      # noqa: E731
    print(f"T2_features DONE{skip(t2)} -- {t2['lines']:,} lines of {t2['base_rows']:,} ({t2['tables']}), "
          f"{t2['features']} features, ARPU tertiles {t2['arpu_tertiles']}")
    print("  missing scores (imputed with the nationality median, flagged): "
          + ", ".join(f"{k} {v:.1%}" for k, v in t2["missing_share"].items()))
    print(f"T3_segments DONE{skip(t3)} -- {t3['segments_non_empty']}/14 segments, "
          f"{t3['micro_segments_non_empty']}/{t3['micro_segments_total']} micro-segments "
          f"({t3['micro_segments_ge_300']} with >= 300 lines), {t3['excluded_nursering']:,} Nursering excluded")
    for seg, n in t3["segment_lines"].items():
        print(f"  {seg:45s} {n:9,d}")
    print(f"  action_original from {t3['action_original_source']}; segment engine vs S-rules agreement "
          f"{t3['engine_vs_s_rules_agreement']:.1%}; action_agent = {t3['agent_dictionary']} "
          f"(equal to original on {t3['action_original_eq_agent']:.1%} of lines)")
    print(f"Step outputs: {t2['step_dir']}")
    print(f"              {t3['step_dir']}")
    return 0


def cmd_panel(a) -> int:
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    try:
        r = sg.run_p1_panel(run_date, use_sample=not a.no_sample, seed=a.seed, force=a.force)
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    skip = " (skipped: same inputs, already done)" if r.get("skipped") else ""
    scaled = f", scaled x{r['scale_factor']} to the pool" if r["scale_factor"] < 1 else ""
    print(f"P1_panel DONE{skip} -- pool {r['pool']:,} eligible lines ({r['in_other_campaigns']:,} in other campaigns "
          f"left out){scaled}; {r['from_sample']:,} from the provided sample")
    for k, n in r["sizes"].items():
        print(f"  {k:8s} {n:9,d}  (requested {r['requested'][k]:,})")
    print(f"  size checks (A-30): {r['size_problems'] or 'OK'}; largest stratum gap {r['max_balance_gap_points']} points")
    print(f"Step outputs: {r['step_dir']}")
    return 1 if r["size_problems"] else 0


def cmd_offers(a) -> int:
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    try:
        r = sg.run_p6_offers(run_date, force=a.force)
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    skip = " (skipped: same inputs, already done)" if r.get("skipped") else ""
    print(f"P6_offers DONE{skip} -- decisions: {r['decisions']}; group audit {r['audit_status']}"
          f"{' (push blocked)' if r['push_blocked'] else ''}")
    for g in r["groups"]:
        print(f"  {g['group']:4s} {g['rules']:22s} contacted {g['contacted']:7,d}  with offers {g['with_offers']:7,d}  "
              f"offers {g['offers']:8,d}  10.1 retries {g['consistency_retry']:6,d}")
    print(f"  tables (.csv.gz): {', '.join(r['tables'])}")
    print(f"Step outputs: {r['step_dir']}")
    return 1 if r["push_blocked"] else 0


def cmd_targets(a) -> int:
    from nbx.fsdr import targets as tg
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    folder = sg.input_folder(run_date)
    try:
        if a.generate:
            r = sg.generate_targets(run_date, fmt=a.format, seed=a.seed)
            scaled = f" (scaled x{r['scale_factor']} to the base)" if r["scale_factor"] < 1 else ""
            print(f"Dummy target lists written to {r['folder']}{scaled}:")
            for f in r["files"]:
                print(f"  {Path(f).relative_to(r['folder'])}")
            print("  sizes:", r["sizes"])
            print("Next: fsdr-run-targets (panel from these lists -> target 1 -> target 2 -> push)")
            return 0
        if not tg.has_targets(folder):
            print(f"No target lists in {tg.targets_dir(folder)} (P1 will draw the panel at random).")
            print("Upload them there, or create dummy ones with: fsdr-targets --generate")
            return 0
        seg = sg.base_segments(run_date)
        t, issues = tg.read_targets(folder)
        kept, rejected, more = tg.check_targets(t, seg, sg.other_campaigns(sg.CacheSource(run_date), run_date),
                                                tg.expected_sizes(folder))
        issues += more
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    print(f"Target lists in {tg.targets_dir(folder)}: {len(t):,} lines read, {len(kept):,} kept, {len(rejected):,} dropped")
    print("  kept per group:", kept["GROUP_KEY"].value_counts().to_dict())
    for i in issues:
        print(f"  {i['level'].upper():8s} {i['check']}: {i['message']} ({i.get('count', 0):,})")
    return 1 if any(i["level"] == "blocking" for i in issues) else 0


def cmd_final(a) -> int:
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    try:
        r = sg.run_p7_final(run_date, force=a.force)
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    _print_final(r)
    return 0 if r["push_ready"] else 1


def _print_final(r: dict) -> None:
    skip = " (skipped: same inputs, already done)" if r.get("skipped") else ""
    state = "READY TO PUSH" if r["push_ready"] else "PUSH BLOCKED (see push/BLOCKED.txt)"
    print(f"final DONE{skip} -- {state}; input_flytxt {r['input_flytxt_rows']:,} rows; details {r['details_rows']:,} "
          f"lines x {r['details_columns']} columns")
    print("  checks:", ", ".join(f"{c['id']} {c['status']}" for c in r["output_checks"]), "| group audit", r["p6_group_audit"])
    for t in r["per_target"]:
        extra = "" if t["group"] == "control" else (f"  F-04 dropped {t['dropped_F04']:,}  favourites {t['favourites']:,}  "
                                                    f"forced {t['forced_F08']:,}  cleared F-09/F-11/F-12 "
                                                    f"{t['cleared_F09']}/{t['cleared_F11']}/{t['cleared_F12']}")
        print(f"  {t['group']:8s} {t['rules']:22s} lines {t['lines']:7,d}  in file {t['in_input_flytxt']:7,d}{extra}")
    print(f"Files: {r['step_dir']}  (push/, target1/, target2/, control/)")


def cmd_run_groups(a) -> int:
    """P1 panel -> P6 offers -> final, re-run from the panel."""
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    try:
        p1 = sg.run_p1_panel(run_date, force=True)
        print(f"P1_panel {p1['status'].upper()} -- mode {p1['mode']}, sizes {p1['sizes']}"
              + (f", {p1.get('rejected', 0):,} target lines dropped" if p1["mode"] == "uploaded" else ""))
        for i in p1.get("target_issues", []):
            if i["level"] == "blocking":
                print(f"  BLOCKING {i['check']}: {i['message']}")
        if p1["status"] != "done":
            return 1
        p6 = sg.run_p6_offers(run_date, force=True)
        print(f"P6_offers DONE -- group audit {p6['audit_status']}")
        r = sg.run_p7_final(run_date, force=True)
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    _print_final(r)
    return 0 if r["push_ready"] else 1


def _print_target(r: dict) -> None:
    skip = " (skipped: same inputs, already done)" if r.get("skipped") else ""
    state = "READY" if r["push_ready"] else f"BLOCKED (see {r['target']}/BLOCKED.txt)"
    print(f"{r['target']} final DONE{skip} -- {state}; input_flytxt_{r['target']} {r['input_flytxt_rows']:,} rows; "
          f"{r['reco_offers']} offers per line max; details {r['details_rows']:,} lines x {r['details_columns']} columns")
    print("  checks:", ", ".join(f"{c['id']} {c['status']}" for c in r["output_checks"]), "| group audit", r["p6_group_audit"])
    for g in r["per_group"]:
        print(f"  {g['group']:4s} {g['folder']:12s} {g['rules']:22s} lines {g['lines']:7,d}  in file {g['in_input_flytxt']:7,d}"
              f"  offers/line {g['offers_per_line']:.2f}  F-04 dropped {g['dropped_F04']:,}  favourites {g['favourites']:,}")
    print(f"  files: {r['step_dir']}")


def cmd_score_target(a) -> int:
    """V4.0: one target on its own: P6 offers of its two groups, then its final files in data/output/<D>/<target>/."""
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    try:
        if a.panel:
            p1 = sg.run_p1_panel(run_date, force=True)
            print(f"P1_panel {p1['status'].upper()} -- mode {p1['mode']}, sizes {p1['sizes']}")
            if p1["status"] != "done":
                return 1
        p6 = sg.run_p6_offers(run_date, force=a.force, target=a.target)
        print(f"{p6['stage']} DONE -- group audit {p6['audit_status']}; groups "
              + ", ".join(f"{g['group']} {g['with_offers']:,}/{g['contacted']:,} with offers" for g in p6["groups"]))
        r = sg.run_target_final(run_date, a.target, force=a.force)
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    _print_target(r)
    return 0 if r["push_ready"] else 1


def cmd_push(a) -> int:
    from nbx.fsdr.pipelines import scoring_groups as sg

    run_date = a.date or _latest_run_date()
    r = sg.run_push(run_date)
    if r["push_ready"]:
        print(f"push READY -- input_flytxt {r['input_flytxt_rows']:,} rows {r['rows_per_target']} -> {r['step_dir']}")
        print("Human approval (H-08) before it is sent to the campaign platform.")
        return 0
    print("push BLOCKED -- " + "; ".join(r["reasons"]) + f"  ({r['step_dir']}/BLOCKED.txt)")
    return 1


def cmd_run_targets(a) -> int:
    """V4.0: P1 panel, then target 1 and target 2 scored separately, the control file and the push."""
    from nbx.fsdr.pipelines import scoring_groups as sg
    from nbx.fsdr.pipelines.training import StageBlocked

    run_date = a.date or _latest_run_date()
    try:
        res = sg.run_targets(run_date, targets=[a.target] if a.target else None)
    except StageBlocked as e:
        print(f"BLOCKED: {e}")
        return 1
    p1 = res["P1_panel"]
    print(f"P1_panel {p1['status'].upper()} -- mode {p1['mode']}, sizes {p1['sizes']}")
    if p1["status"] != "done":
        for i in p1.get("target_issues", []):
            if i["level"] == "blocking":
                print(f"  BLOCKING {i['check']}: {i['message']}")
        return 1
    for t in config.TARGETS:
        if t in res:
            _print_target(res[t])
    print(f"control: {res['control']['lines']:,} lines -> {res['control']['file']}")
    r = res["push"]
    print(("push READY -- " + f"{r['input_flytxt_rows']:,} rows" if r["push_ready"] else "push BLOCKED -- "
           + "; ".join(r["reasons"])) + f"  ({r['step_dir']})")
    return 0 if r["push_ready"] else 1


def _latest_run_date():
    from datetime import date

    root = config.FSDR_RUNS_DIR
    dates = sorted(p.name for p in root.glob("20??-??-??") if (p / "train" / "T0_load").is_dir()) if root.is_dir() else []
    return date.fromisoformat(dates[-1]) if dates else None


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


def cmd_weekly(a) -> int:
    from nbx.fsdr.pipelines.weekly import STEPS, run_weekly

    d = _data_dir(a)
    steps = [s.strip() for s in a.steps.split(",")] if a.steps else None
    if steps and set(steps) - set(STEPS):
        raise SystemExit(f"--steps must be among {','.join(STEPS)}")
    try:
        rep = run_weekly(d, tuple(x.strip() for x in a.arms.split(",") if x.strip()), a.llm, a.llm_limit, steps)
    except Exception as e:  # noqa: BLE001
        print(f"Weekly run of {d.name} FAILED: {type(e).__name__}: {e}  (details: {d / 'outputs' / 'weekly_run_report.json'})")
        return 1
    print(f"Weekly run of {d.name}: {rep['status']} in {rep['seconds']} s")
    for name, s in rep["steps"].items():
        print(f"  {name:9s} {s['status']:8s} {s.get('seconds', '')!s:>7} s  {s.get('reason', '')}")
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
    if not runs and config.FSDR_INPUT_DIR.exists():          # V2.0 layout: data/input/<date>
        runs = sorted(p for p in config.FSDR_INPUT_DIR.glob("20??-??-??") if p.is_dir())
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
                                   "overrides", "critic", "message", "justification")})
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

    g = sub.add_parser("fsdr-generate", help="write dummy data files for a run date (5 input folders)")
    g.add_argument("--date", type=_date, help="run date D (default: today)")
    g.add_argument("--n", type=int, default=config.GENERATOR_DEFAULT_N, help=f"lines (default {config.GENERATOR_DEFAULT_N}, max {config.GENERATOR_MAX_N:,})")
    g.add_argument("--mode", choices=list(config.GENERATOR_MODES),
                   help=", ".join(f"{k} = {v:,} lines" for k, v in config.GENERATOR_MODES.items()) + " (overrides --n)")
    g.add_argument("--scenario", default="baseline", help=f"comma-separated: {','.join(config.SCENARIOS)}")
    g.add_argument("--seed", type=int, default=config.GENERATOR_SEED)
    g.add_argument("--format", choices=["csv.gz", "json", "jsonl", "csv", "parquet"], default="csv.gz",
                   help="parquet: ~10x smaller, for the 1M mode")
    g.add_argument("--layout", choices=["split", "folders", "flat"], default="split",
                   help="split (V4.0): data/training/<D>/ + data/scoring/<D>/, each with usage/ models/ subscription/ "
                        "catalog/ others/; folders: one folder data/input/<D>/ (V3.0); flat: V1.0")
    g.add_argument("--weeks", type=int, default=1, help="consecutive weekly runs from one world (default 1)")
    g.add_argument("--arms", help=f"test-sample groups, comma-separated (default {','.join(config.ARMS)})")
    g.add_argument("--out", help="output folder (default data/training/<date>, its twin data/scoring/<date>; "
                                 "with --weeks > 1, the root of the <date> folders)")
    g.add_argument("--force", action="store_true", help="overwrite a folder that holds non-generated data")
    g.set_defaults(fn=cmd_generate)

    ld = sub.add_parser("fsdr-load", help="step T0: validate + completeness + Parquet cache (① data agent)")
    ld.add_argument("--data-dir", help="input run folder (default data/training/<--date>; its twin "
                                       "data/scoring/<date> is read with it)")
    ld.add_argument("--pipeline", choices=["train_week0", "train_weekly"], help="default: detected from the tables")
    ld.add_argument("--date", type=_date, help="run date (default: manifest, then folder name)")
    ld.add_argument("--force", action="store_true", help="re-run even if done with the same inputs")
    ld.set_defaults(fn=cmd_load)

    ft = sub.add_parser("fsdr-features", help="steps T2 + T3: features B-01..B-12, S-rule segments, micro-segments, "
                                              "action_original / action_agent (full base, partition by partition)")
    ft.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run with a T0_load)")
    ft.add_argument("--workers", type=int, help=f"partitions in parallel (default {config.FEATURE_WORKERS}; 8 on the P16)")
    ft.add_argument("--force", action="store_true", help="re-run even if the inputs did not change")
    ft.set_defaults(fn=cmd_features)

    pn = sub.add_parser("fsdr-panel", help="step P1: panel T1C / T1F / T2C / T2F / control (stratified random)")
    pn.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    pn.add_argument("--seed", type=int, default=config.GENERATOR_SEED)
    pn.add_argument("--no-sample", action="store_true", help="ignore the provided sample_cg_tg table")
    pn.add_argument("--force", action="store_true")
    pn.set_defaults(fn=cmd_panel)

    of = sub.add_parser("fsdr-offers", help="step P6: code-2 tables + the 8 reco tables of the 4 groups (.csv.gz)")
    of.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    of.add_argument("--force", action="store_true")
    of.set_defaults(fn=cmd_offers)

    tg = sub.add_parser("fsdr-targets", help="the target lists of data/scoring/<D>/target1|2/a_constraint|b_free/: "
                                             "check them, or --generate dummy ones")
    tg.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    tg.add_argument("--generate", action="store_true", help="write dummy lists (stratified draw of the eligible base)")
    tg.add_argument("--format", choices=["csv.gz", "csv", "json"], default="csv.gz")
    tg.add_argument("--seed", type=int, default=config.GENERATOR_SEED)
    tg.set_defaults(fn=cmd_targets)

    fi = sub.add_parser("fsdr-final", help="final step: input_flytxt to push + per-target files + details")
    fi.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    fi.add_argument("--force", action="store_true")
    fi.set_defaults(fn=cmd_final)

    rg = sub.add_parser("fsdr-run-groups", help="panel (uploaded targets or drawn) -> offers -> final, in one go")
    rg.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    rg.set_defaults(fn=cmd_run_groups)

    stg = sub.add_parser("fsdr-score-target", help="V4.0: score ONE target (P6 of its a / b groups + final files) "
                                                   "into data/output/<D>/<target>/")
    stg.add_argument("--target", required=True, choices=list(config.TARGETS))
    stg.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    stg.add_argument("--panel", action="store_true", help="re-run P1 (the panel from the target lists) first")
    stg.add_argument("--force", action="store_true")
    stg.set_defaults(fn=cmd_score_target)

    ps = sub.add_parser("fsdr-push", help="V4.0: data/output/<D>/push/input_flytxt.csv.gz from both targets (both PASS)")
    ps.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    ps.set_defaults(fn=cmd_push)

    rt = sub.add_parser("fsdr-run-targets", help="V4.0: panel -> target 1 -> target 2 (each scored separately) -> "
                                                 "control -> push, into data/output/<D>/")
    rt.add_argument("--date", type=date.fromisoformat, help="run date (default: the latest run)")
    rt.add_argument("--target", choices=list(config.TARGETS), help="only this target (the push then waits for the other)")
    rt.set_defaults(fn=cmd_run_targets)

    v = sub.add_parser("fsdr-validate", help="validate a run folder (5 input folders or flat)")
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

    wk = sub.add_parser("fsdr-weekly", help="the whole weekly cycle: validate, feedback, train, score, offers, report")
    wk.add_argument("--data-dir", help="run folder D (default: latest data/fsdr/<date>)")
    wk.add_argument("--arms", default="cstr,free,llm", help="arms to build offers for (default cstr,free,llm)")
    wk.add_argument("--llm", help="LLM model for the llm arm (offline backend = rule-based)")
    wk.add_argument("--llm-limit", type=int, help="llm-arm lines decided by the agents (0 = skip the llm arm)")
    wk.add_argument("--steps", help="only these steps, comma-separated (validate,feedback,train,score,offers,report)")
    wk.set_defaults(fn=cmd_weekly)

    rp = sub.add_parser("fsdr-report", help="weekly report: take rate and revenue per arm, uplift vs control")
    rp.add_argument("--data-dir", help="a run folder of the data root (default: latest)")
    rp.add_argument("--week", type=_date, help="campaign week (run date D; default: the folder's date)")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
