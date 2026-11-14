"""① Data agent (master prompt §5): step T0_load of the training pipeline.

    rep = run_t0(Path("data/input/2026-10-21"))            # pipeline detected: train_week0 or train_weekly
    rep["status"]     "done" (the cache is built) or "blocked" (the run stops: rep["explanation"] says why)

What it does, in order, each result saved in data/runs/<D>/train/T0_load/ (plan §14):
  1. validate every table against the contract (fields, raw types, keys, duplicates, cross-table orphans, model
     coverage, date windows, the 5-folder layout, customer_profile)    -> validation_report.json / .md
  2. weekly completeness for the pipeline about to run (§17.9)          -> completeness.csv / .json
  3. if nothing blocks: convert every table to the partitioned Parquet cache (plan §15) -> cache_manifest.json
  4. a plain-language summary of what blocks or warns                   -> summary.json

The checks are code; the agent only reports. Blocking issues stop the run here, before any feature is computed.
A re-run with the same input files is skipped (resume), unless force=True.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.contract import cache as cache_mod
from nbx.fsdr.contract import completeness
from nbx.fsdr.datasource import JsonSource
from nbx.fsdr.steps import Step, fingerprint, runs_dir
from nbx.fsdr.validation import ValidationReport, validate

STEP = "T0_load"


def run_date_of(folder: Path, src: JsonSource, run_date: date | None = None) -> date | None:
    if run_date:
        return run_date
    man = src.manifest()
    if man.get("run_date"):
        return date.fromisoformat(man["run_date"])
    try:
        return date.fromisoformat(Path(folder).name)
    except ValueError:
        return None


def explain(rep: ValidationReport, comp: dict) -> str:
    """Plain-language summary of the run's data: what blocks first, then the main warnings (deterministic)."""
    c = rep.counts()
    lines = []
    if rep.status == "blocked" or not comp["complete"]:
        lines.append("The run is BLOCKED: no feature is computed until these are fixed.")
        for i in [i for i in rep.issues if i.level == "blocking"][:10]:
            lines.append(f"- {i.table}{'.' + i.column if i.column else ''}: {i.message}")
        for r in comp["reasons"]:
            lines.append(f"- {r}")
    else:
        lines.append(f"The data of {rep.run_date} is usable for {comp['pipeline']}: "
                     f"{len(rep.tables)} tables checked, {c['warning']} warning(s), {c['info']} note(s).")
    warns = [i for i in rep.issues if i.level == "warning"]
    if warns:
        lines.append("Warnings (the run continues):")
        lines += [f"- {i.table}{'.' + i.column if i.column else ''}: {i.message}" for i in warns[:10]]
        if len(warns) > 10:
            lines.append(f"- ... and {len(warns) - 10} more in validation_report.md")
    if comp["warnings"]:
        lines.append(f"Optional tables missing (fallback used): {', '.join(comp['warnings'])}.")
    return "\n".join(lines)


def run_t0(folder: Path | str, pipeline: str | None = None, run_date: date | None = None,
           runs_root: Path | None = None, cache_root: Path | None = None, force: bool = False) -> dict:
    folder = Path(folder)
    src = JsonSource(folder)
    run_date = run_date_of(folder, src, run_date)
    if run_date is None:
        raise ValueError(f"{folder}: no run date (no _manifest.json, folder name is not YYYY-MM-DD, none given)")
    pipeline = pipeline or completeness.detect_pipeline(src)
    files = [src.path(t) for t in src.tables()] + src.manifest_files()
    inputs = fingerprint(files, {"pipeline": pipeline, "parts": config.CACHE_PARTITIONS, "cache": str(cache_root or "")})
    with Step(runs_dir(run_date, runs_root) / "train", STEP, inputs=inputs, force=force) as st:
        if st.done:
            return {**st.read_json("summary.json"), "skipped": True}
        rep = validate(src, run_date)
        st.write_json("validation_report.json", rep.to_dict())
        st.write_text("validation_report.md", rep.to_markdown())
        comp = completeness.check(src, pipeline, run_date)
        st.write_json("completeness.json", comp)
        st.write_csv(pd.DataFrame(comp["tables"]), "completeness.csv")
        rows_in = sum(t.rows for t in rep.tables.values())
        blocked = rep.status == "blocked" or not comp["complete"]
        twins = config.run_twins(folder)
        if twins and src.misfiled():
            st.write_json("misfiled_tables.json", {t: {"found": f, "expected": e} for t, (f, e) in src.misfiled().items()})
        summary = {"run_date": run_date.isoformat(), "pipeline": pipeline, "source": str(folder),
                   "training_source": str(twins.get(config.TRAINING_FOLDER, folder)),
                   "scoring_source": str(twins.get(config.SCORING_FOLDER, folder)),
                   "misfiled_tables": len(src.misfiled()),
                   "layout": rep.layout, "status": "blocked" if blocked else "done",
                   "validation": rep.status, "issue_counts": rep.counts(), "complete": comp["complete"],
                   "tables": len(src.tables()), "rows_in": rows_in, "profile": rep.profile,
                   "explanation": explain(rep, comp), "step_dir": str(st.dir)}
        if blocked:
            st.block("; ".join(comp["reasons"] + [i.message for i in rep.issues if i.level == "blocking"][:5]))
        else:
            man = cache_mod.build_cache(src, run_date, cache_root)
            st.write_json("cache_manifest.json", man)
            summary["cache"] = {"dir": str(cache_mod.cache_dir(run_date, cache_root)), "bytes": man["bytes"],
                                "partitions": man["partitions"], "seconds": man["seconds"]}
        st.rows(rows_in=rows_in, tables=len(src.tables()))
        st.note(pipeline=pipeline, validation=rep.status)
        st.write_json("summary.json", summary)
        return summary


def load_summary(run_date: date, runs_root: Path | None = None) -> dict:
    p = runs_dir(run_date, runs_root) / "train" / STEP / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
