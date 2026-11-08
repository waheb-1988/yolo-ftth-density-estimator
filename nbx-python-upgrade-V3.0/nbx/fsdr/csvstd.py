"""The FSDR CSV standard (master prompt §2, §20): the one format of every input AND output table file.

    write_csv(df, path)                 <name>.csv.gz (gzip level 6), + <name>.schema.json next to it
    read_csv(path)                      every column as text (IDs keep their leading zeros), "" for missing
    write_outputs_manifest(dir, files)  _outputs_manifest.json: file list, row counts, SHA-256, run id, versions

| Compression | gzip, level 6 (mtime 0: the same table gives the same bytes)  |
| Encoding    | UTF-8, no BOM                                                 |
| Separator   | ,   quoting only when needed (comma, quote, line break), "" inside |
| Line ending | \\n                                                            |
| Decimal     | dot, no thousands separator, never scientific notation        |
| Dates       | YYYY-MM-DD / YYYY-MM-DD HH:MM:SS                              |
| Missing     | empty field (never nan / None / NULL)                         |
| Index       | none                                                          |
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config

SUFFIX = ".csv.gz"
SCHEMA_SUFFIX = ".schema.json"
OUTPUTS_MANIFEST = "_outputs_manifest.json"


def table_name(path: Path) -> str:
    """activities_w1_score.csv.gz -> activities_w1_score (also .csv / .json / .jsonl / .parquet)."""
    name = Path(path).name
    for suffix in (SUFFIX, *config.INPUT_FORMATS):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def file_format(path: Path) -> str:
    """".csv.gz", ".csv", ".json", ".jsonl" or ".parquet" ("" for anything else)."""
    name = Path(path).name.lower()
    return next((f for f in (SUFFIX, *config.INPUT_FORMATS) if name.endswith(f)), "")


def format_value(v) -> str:
    """One cell -> its text in the FSDR CSV standard."""
    if v is None or v is pd.NA or v is pd.NaT:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return ""
        return np.format_float_positional(f, trim="0")              # 7.0, 12.5, 0.0001 -- never 1e-04
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, datetime):                                     # pd.Timestamp included
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def to_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every cell formatted as the standard writes it (also what read_csv gives back)."""
    out = {}
    for c in df.columns:
        out[c] = _text_column(df[c])
    return pd.DataFrame(out, columns=df.columns, index=df.index)


def _float_text(s: pd.Series) -> pd.Series:
    """Vectorised repr of floats ("7.0", "12.5"); the rare exponent forms are rewritten positionally."""
    f = s.astype("float64")
    t = f.astype(str)
    exp = t.str.contains("e", regex=False) & f.notna() & np.isfinite(f)
    if exp.any():
        t[exp] = [np.format_float_positional(v, trim="0") for v in f[exp]]
    return t.where(f.notna() & np.isfinite(f), "")


def _text_column(s: pd.Series) -> pd.Series:
    """One column -> text, by dtype first (fast paths for 1M-row tables), value by value only for mixed columns."""
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s.dropna()
        fmt = "%Y-%m-%d" if bool((d.dt.normalize() == d).all()) else "%Y-%m-%d %H:%M:%S"
        return s.dt.strftime(fmt).fillna("")
    if pd.api.types.is_bool_dtype(s):
        return s.map(format_value)
    if pd.api.types.is_integer_dtype(s):
        return s.astype("string").fillna("").astype(object)
    if pd.api.types.is_float_dtype(s):
        return _float_text(s)
    kind = pd.api.types.infer_dtype(s, skipna=True)
    if kind in ("string", "empty"):
        return s.where(s.notna(), "").astype(str)
    if kind == "integer":
        return s.astype("Int64").astype("string").fillna("").astype(object)
    if kind == "floating":
        return _float_text(s)
    return s.map(format_value)


def column_types(df: pd.DataFrame) -> list[dict]:
    """Column names and types for <name>.schema.json (identifier columns are always text)."""
    cols = []
    for c in df.columns:
        s = df[c]
        if str(c).upper() in config.ID_TEXT_COLUMNS:
            t = "string"
        elif pd.api.types.is_bool_dtype(s):
            t = "boolean"
        elif pd.api.types.is_integer_dtype(s):
            t = "bigint"
        elif pd.api.types.is_float_dtype(s):
            t = "double"
        elif pd.api.types.is_datetime64_any_dtype(s):
            t = "timestamp"
        else:
            t = "string"
        cols.append({"name": str(c), "type": t})
    return cols


def write_csv(df: pd.DataFrame, path: Path, schema: bool = True, compress: bool = True) -> Path:
    """Writes df in the FSDR CSV standard. path may end in .csv.gz (default) or .csv (compress=False)."""
    path = Path(path)
    if compress and not path.name.endswith(SUFFIX):
        path = path.with_name(table_name(path) + SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow([str(c) for c in df.columns])
    w.writerows(to_text_frame(df).itertuples(index=False, name=None))
    data = buf.getvalue().encode("utf-8")
    if compress:
        with path.open("wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", compresslevel=config.CSV_GZIP_LEVEL, mtime=0,
                                                  filename="") as gz:
            gz.write(data)
    else:
        path.write_bytes(data)
    if schema:
        schema_path(path).write_text(json.dumps({"table": table_name(path), "columns": column_types(df)}, indent=2),
                                     encoding="utf-8")
    return path


def schema_path(path: Path) -> Path:
    return Path(path).with_name(table_name(path) + SCHEMA_SUFFIX)


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """A .csv.gz / .csv of the standard -> text frame ("" stays "", so a missing value is never "nan")."""
    path = Path(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return pd.read_csv(fh, dtype=str, keep_default_na=False, na_filter=False, usecols=columns)


def write_outputs_manifest(folder: Path, files: list[Path], run_id: str = "", versions: dict | None = None) -> Path:
    from nbx.fsdr.steps import sha256_file

    entries = []
    for f in files:
        f = Path(f)
        rows = len(read_csv(f)) if file_format(f) in (SUFFIX, ".csv") else None
        entries.append({"file": f.name, "rows": rows, "sha256": sha256_file(f),
                        "schema": schema_path(f).name if schema_path(f).exists() else None})
    p = Path(folder) / OUTPUTS_MANIFEST
    p.write_text(json.dumps({"run_id": run_id, "versions": versions or {}, "files": entries}, indent=2), encoding="utf-8")
    return p
