"""Deterministic FSDR pipelines (train, score, recommend). Each step returns a JSON report,
so the agents can call the steps as tools."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"


def outputs_dir(data_dir: str | Path) -> Path:
    p = Path(data_dir) / OUTPUT_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 6)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, date):
        return v.isoformat()
    return v


def write_table(df: pd.DataFrame, data_dir: str | Path, name: str) -> Path:
    """Writes an output table as a JSON array of records (lower-case columns, as the production outputs)."""
    path = outputs_dir(data_dir) / f"{name}.json"
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    records = [{k: _clean(v) for k, v in r.items()} for r in out.to_dict("records")]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def read_table(data_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(data_dir) / OUTPUT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    df.columns = [c.upper() for c in df.columns]
    return df


def write_report(report: dict, data_dir: str | Path, name: str) -> Path:
    path = outputs_dir(data_dir) / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_clean), encoding="utf-8")
    return path
