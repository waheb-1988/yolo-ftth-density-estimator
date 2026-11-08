"""Weekly completeness (master prompt §5, §17.9): every table the pipeline about to run needs is present, has
rows and belongs to this week's run.

    rep = completeness.check(src, "train_week0", date(2026, 10, 21))
    rep["complete"]        False -> the run is blocked (a required table is missing, empty or stale)

Pipelines: train_week0 (_auto + _score + history), train_weekly (_score + purchases), score (the tables scoring reads
directly; the rest comes from the validated bundle). The lists live in config.PIPELINE_TABLES.
"""
from __future__ import annotations

from datetime import date

from nbx.fsdr import config, schemas
from nbx.fsdr.datasource import DataSource

PIPELINES = tuple(config.PIPELINE_TABLES)


def detect_pipeline(src: DataSource) -> str:
    """Week 0 when the history tables of the segment engine are there (_auto + targeting_dataiku_auto)."""
    week0 = ["activities_w1_auto", "subs_amt_auto", "targeting_dataiku_auto"]
    return "train_week0" if all(src.has(t) for t in week0) else "train_weekly"


def check(src: DataSource, pipeline: str, run_date: date | None = None) -> dict:
    if pipeline not in config.PIPELINE_TABLES:
        raise ValueError(f"pipeline must be one of {PIPELINES}")
    spec = config.PIPELINE_TABLES[pipeline]
    man = src.manifest()
    man_date = man.get("run_date")
    fresh = run_date is None or man_date is None or man_date == run_date.isoformat()
    rows = []
    for need in ("required", "optional"):
        for t in spec[need]:
            present = src.has(t)
            n = len(src.read(t)) if present else 0
            status = "ok" if present and n else "empty" if present else "missing"
            level = "ok" if status == "ok" else "blocking" if need == "required" else "warning"
            rows.append({"table": t, "folder": schemas.folder_of(t, schemas.TABLES[t].family if t in schemas.TABLES else None),
                         "need": need, "present": present, "rows": n, "status": status, "level": level})
    blocking = [r["table"] for r in rows if r["level"] == "blocking"]
    reasons = [f"required table {r['table']} is {r['status']}" for r in rows if r["level"] == "blocking"]
    if not fresh:
        reasons.append(f"stale data: the manifest is for {man_date}, the run is for {run_date}")
    return {"pipeline": pipeline, "run_date": run_date.isoformat() if run_date else None, "manifest_run_date": man_date,
            "fresh": fresh, "complete": not blocking and fresh, "blocking": blocking,
            "warnings": [r["table"] for r in rows if r["level"] == "warning"], "reasons": reasons, "tables": rows}
