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
    cols = [str(c).lower() for c in df.columns]
    values = [_column_values(df.iloc[:, i]) for i in range(df.shape[1])]
    records = [dict(zip(cols, row)) for row in zip(*values)] if cols else [{} for _ in range(len(df))]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def _column_values(s: pd.Series) -> list:
    """The JSON values of one column, as the per-value _clean of df.to_dict("records") gave them: NaN -> None,
    numpy numbers -> Python numbers (unrounded: to_dict already boxes them), dates -> ISO strings. Number and
    string columns are converted in one go (review #15: _clean ran once per value, 5 M calls at 100 k lines)."""
    k = s.dtype.kind
    if k in "iub" and not pd.api.types.is_extension_array_dtype(s.dtype):
        return s.tolist()
    if k == "f" and not pd.api.types.is_extension_array_dtype(s.dtype):
        return [None if v != v else v for v in s.tolist()]
    if s.dtype == object and pd.api.types.infer_dtype(s, skipna=True) in ("string", "empty"):
        return [v if isinstance(v, str) else None for v in s.tolist()]
    return [_clean(v) for v in s.to_frame().to_dict("list")[s.name if s.name is not None else 0]]


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
