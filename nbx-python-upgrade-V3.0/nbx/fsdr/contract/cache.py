"""Parquet cache of a run (plan §4, §15): each input table converted once, typed, split into partitions.

    data/cache/<D>/<table>/part-00.parquet ... part-15.parquet    tables with a line (MSISDN or ACCESS_METHOD_ID)
    data/cache/<D>/<table>/part-all.parquet                        reference tables (catalogue, party, accounts, ...)
    data/cache/<D>/_cache_manifest.json                            rows and SHA-256 of every file

Every per-line table is partitioned on the SAME key: the line's ACCESS_METHOD_ID (taken from base_prepaid when the
table only has the MSISDN), canonicalised, then hashed. So all rows of one line, whatever the table, land in the
same partition number: features and segments run partition by partition (8 in parallel on the P16) and every join
stays inside a partition. A line outside the base is hashed on its own key (it joins nothing anyway).

    cache = build_cache(src, date(2026, 10, 21))
    CacheSource(date(2026, 10, 21)).read("activities_w1_score", part=3)
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config, schemas
from nbx.fsdr.datasource import DataSource, compact_frame
from nbx.fsdr.steps import _parquet_safe, sha256_file

CACHE_MANIFEST = "_cache_manifest.json"
LINE_ID_COLS = ("ACCESS_METHOD_ID", "access_method_id", "Access_Method_id")
LINE_VAL_COLS = ("ACCESS_METHOD_VAL", "access_method_val", "MSISDN", "msisdn")
ALL = "all"


def cache_dir(run_date: date | str, root: Path | None = None) -> Path:
    d = run_date.isoformat() if isinstance(run_date, date) else str(run_date)
    return Path(root or config.FSDR_CACHE_DIR) / d


def _canon(s: pd.Series) -> pd.Series:
    from nbx.fsdr.validation import norm_keys

    return norm_keys(s)


def line_map(src: DataSource) -> pd.Series:
    """Canonical MSISDN -> canonical ACCESS_METHOD_ID, from base_prepaid (first row wins on a duplicate MSISDN)."""
    if not src.has("base_prepaid"):
        return pd.Series(dtype=object)
    base = src.read("base_prepaid")
    if not {"ACCESS_METHOD_VAL", "ACCESS_METHOD_ID"} <= set(base.columns):
        return pd.Series(dtype=object)
    m = pd.Series(_canon(base["ACCESS_METHOD_ID"]).to_numpy(), index=_canon(base["ACCESS_METHOD_VAL"]).to_numpy())
    m = m[m.index.notna() & m.notna()]
    return m[~m.index.duplicated(keep="first")]


def line_key(df: pd.DataFrame, lines: pd.Series) -> pd.Series | None:
    """The partition key of every row: canonical ACCESS_METHOD_ID, or None for a table without a line column."""
    for c in LINE_ID_COLS:
        if c in df.columns:
            return _canon(df[c])
    for c in LINE_VAL_COLS:
        if c in df.columns:
            val = _canon(df[c])
            return val.map(lines).fillna("v:" + val.fillna(""))        # MSISDN outside the base: its own hash
    return None


def partition_of(keys: pd.Series, parts: int) -> np.ndarray:
    """Stable partition number of canonical keys (pandas' fixed-key hash: same key -> same partition, every run)."""
    h = pd.util.hash_pandas_object(keys.fillna("").astype(str), index=False).to_numpy()
    return (h % np.uint64(parts)).astype(np.int16)


def build_cache(src: DataSource, run_date: date, root: Path | None = None, parts: int | None = None,
                tables: list[str] | None = None) -> dict:
    """Writes every table of the source (or `tables`) to the cache; returns and writes the manifest."""
    parts = int(parts or config.CACHE_PARTITIONS)
    out = cache_dir(run_date, root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = line_map(src)
    man = {"run_date": run_date.isoformat(), "partitions": parts, "key": "ACCESS_METHOD_ID (via base_prepaid for MSISDN)",
           "built_at": datetime.now().isoformat(timespec="seconds"), "tables": {}}
    for t in tables or [t for t in src.tables() if t in schemas.TABLES]:
        df = compact_frame(src.read(t))
        tdir = out / t
        tdir.mkdir(exist_ok=True)
        for old in tdir.glob("part-*.parquet"):
            old.unlink()
        key = line_key(df, lines)
        files = {}
        if key is None:
            p = tdir / f"part-{ALL}.parquet"
            _parquet_safe(df).to_parquet(p, index=False)
            files[ALL] = {"rows": len(df), "sha256": sha256_file(p)}
        else:
            part = partition_of(key, parts)
            safe = _parquet_safe(df)
            for k in range(parts):
                p = tdir / f"part-{k:02d}.parquet"
                sub = safe[part == k]
                sub.to_parquet(p, index=False)
                files[f"{k:02d}"] = {"rows": len(sub), "sha256": sha256_file(p)}
        man["tables"][t] = {"rows": len(df), "partitioned": key is not None, "folder": schemas.TABLES[t].folder,
                            "files": files}
    man["seconds"] = round(time.time() - t0, 2)
    man["bytes"] = sum(p.stat().st_size for p in out.rglob("*.parquet"))
    (out / CACHE_MANIFEST).write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def verify_cache(run_date: date, root: Path | None = None) -> list[str]:
    """Files whose SHA-256 no longer matches the manifest (empty list = intact)."""
    d = cache_dir(run_date, root)
    man = json.loads((d / CACHE_MANIFEST).read_text(encoding="utf-8"))
    bad = []
    for t, info in man["tables"].items():
        for k, f in info["files"].items():
            p = d / t / f"part-{k}.parquet"
            if not p.exists() or sha256_file(p) != f["sha256"]:
                bad.append(str(p.relative_to(d)))
    return bad


class CacheSource:
    """Reads the cache: a whole table, or one partition of a per-line table (reference tables have no partitions:
    any part returns the whole table)."""

    def __init__(self, run_date: date | str, root: Path | None = None):
        self.dir = cache_dir(run_date, root)
        p = self.dir / CACHE_MANIFEST
        if not p.exists():
            raise FileNotFoundError(f"No cache for {run_date}: run fsdr-load first ({p})")
        self._man = json.loads(p.read_text(encoding="utf-8"))
        self.partitions = int(self._man["partitions"])

    def tables(self) -> list[str]:
        return list(self._man["tables"])

    def has(self, table: str) -> bool:
        return table in self._man["tables"]

    def partitioned(self, table: str) -> bool:
        return bool(self._man["tables"][table]["partitioned"])

    def read(self, table: str, part: int | None = None) -> pd.DataFrame:
        if not self.has(table):
            raise FileNotFoundError(f"{table} is not in the cache {self.dir}")
        tdir = self.dir / table
        if not self.partitioned(table):
            return pd.read_parquet(tdir / f"part-{ALL}.parquet")
        if part is not None:
            return pd.read_parquet(tdir / f"part-{int(part):02d}.parquet")
        return pd.concat([pd.read_parquet(tdir / f"part-{k:02d}.parquet") for k in range(self.partitions)],
                         ignore_index=True)

    def manifest(self) -> dict:
        return dict(self._man)
