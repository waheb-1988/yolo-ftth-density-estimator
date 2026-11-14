"""Where the FSDR pipelines get their tables.

    JsonSource("data/input/2026-10-22")   a run folder, one file per table (real exports or generator output)
    SyntheticSource(n=5000, ...)          the same tables generated in memory

Both return the same thing for a table: a raw frame (python values, None =
null, schema columns first) exactly as a JSON file holds it, so swapping
dummy data for real data is a path change, not a code change. Use
schemas.coerce() to get typed columns.

Layout (plan §13): the tables sit in the 5 input folders (usage/, models/,
subscription/, catalog/, others/) or, as in V1.0, all in the run folder itself.

Formats: "<table>.json" (array of records; JSON Lines is detected from the
content), "<table>.jsonl", "<table>.csv" (text: numbers are parsed with the
raw type of the contract) or "<table>.parquet".
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from nbx.fsdr import config, schemas


class DataSource(Protocol):
    def tables(self) -> list[str]: ...
    def has(self, table: str) -> bool: ...
    def read(self, table: str) -> pd.DataFrame: ...
    def manifest(self) -> dict: ...


def _raw_frame(records: list[dict], table: str) -> pd.DataFrame:
    """Records -> object frame: schema columns first (in schema order), then any extra fields."""
    schema = schemas.TABLES.get(table)
    # dtype=object: no type inference, so an int column with nulls stays ints (from_records would make 7 -> 7.0)
    df = pd.DataFrame(records, dtype=object) if records else pd.DataFrame()
    df = df.where(pd.notna(df), None) if len(df) else df
    if schema is None:
        return df
    extra = [c for c in df.columns if c not in schema.field_names]
    present = [c for c in schema.field_names if c in df.columns]
    return df[present + extra] if len(df.columns) else pd.DataFrame(columns=schema.field_names)


def compact_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw frame -> numeric columns as numpy int64 (no null) / float64 (nulls, or ints mixed with floats); text
    stays object. For the pipelines only, once validation has seen the raw values (review #15: a column of Python
    objects costs ~4x the memory of a numpy column, ~30 GB for a 1.7 M-line run)."""
    out = {}
    for c in df.columns:
        s = df[c]
        kind = pd.api.types.infer_dtype(s, skipna=True) if s.dtype == object else ""
        try:
            if kind == "integer":
                out[c] = s.astype("int64") if s.notna().all() else s.astype("float64")
            elif kind in ("floating", "mixed-integer-float"):
                out[c] = s.astype("float64")
            else:
                out[c] = s
        except (OverflowError, TypeError, ValueError):     # e.g. an int beyond int64: keep the raw values
            out[c] = s
    return pd.DataFrame(out, index=df.index)


def read_json_records(path: Path) -> list[dict]:
    """Reads a JSON array or JSON Lines file (auto-detected from the first non-blank character)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array of records")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


_INT_TEXT = re.compile(r"^-?\d+(\.0+)?$")


def _conform(df: pd.DataFrame, table: str, from_text: bool) -> pd.DataFrame:
    """CSV / Parquet frame -> the raw values a JSON file would hold: None for nulls, python ints for integer
    fields (a Parquet int column with nulls arrives as float), floats for doubles, ISO strings for dates. CSV
    text that does not parse keeps its string, so validation reports it as a type error."""
    schema = schemas.TABLES.get(table)
    types = schema.types() if schema else {}
    out = {}
    for c in df.columns:
        s = df[c]
        rt = types.get(c)
        if pd.api.types.is_datetime64_any_dtype(s):
            fmt = "%Y-%m-%d" if rt == "date" else "%Y-%m-%dT%H:%M:%S"
            vals = s.dt.strftime(fmt).astype(object)
        elif from_text:
            vals = s.astype(object).where(s.astype(str) != "", None)
            if rt in schemas.INT_TYPES or rt == "double":
                num = pd.to_numeric(vals, errors="coerce")
                ok = num.notna()
                if rt in schemas.INT_TYPES:
                    ok &= vals.astype(str).str.match(_INT_TEXT.pattern)
                    conv = [int(v) for v in num[ok]]
                else:
                    conv = [float(v) for v in num[ok]]
                vals = vals.copy()
                vals[ok] = pd.Series(conv, index=num[ok].index, dtype=object)
        else:
            vals = s.astype(object)
            if rt in schemas.INT_TYPES and pd.api.types.is_float_dtype(s):
                whole = s.notna() & (s % 1 == 0)
                vals = vals.copy()
                vals[whole] = pd.Series([int(v) for v in s[whole]], index=s[whole].index, dtype=object)
            elif pd.api.types.is_numeric_dtype(s):
                vals = pd.Series(s.tolist(), index=s.index, dtype=object)       # numpy scalars -> python
        out[c] = vals.where(pd.notna(vals), None)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _frame_raw(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Conformed frame -> schema columns first (in schema order), then any extra fields."""
    schema = schemas.TABLES.get(table)
    if schema is None:
        return df
    extra = [c for c in df.columns if c not in schema.field_names]
    present = [c for c in schema.field_names if c in df.columns]
    return df[present + extra] if len(df.columns) else pd.DataFrame(columns=schema.field_names)


def read_table_file(path: Path, table: str) -> pd.DataFrame:
    """One table file, any supported format -> raw frame."""
    from nbx.fsdr import csvstd

    suffix = csvstd.file_format(path)
    if suffix in (csvstd.SUFFIX, ".csv"):                  # V3.0: .csv.gz (FSDR CSV standard), text in, typed here
        return _frame_raw(_conform(csvstd.read_csv(path), table, from_text=True), table)
    if suffix == ".parquet":
        return _frame_raw(_conform(pd.read_parquet(path), table, from_text=False), table)
    return _raw_frame(read_json_records(path), table)


class JsonSource:
    """A run folder: one file per table (file name = exact table name), in the 5 input folders or flat."""

    def __init__(self, folder: str | Path, compact: bool = False):
        """compact=True: numeric columns typed (compact_frame), for the pipelines; False: raw values, for validation."""
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"No such data folder: {self.folder}")
        self.compact = compact
        self._cache: dict[str, pd.DataFrame] = {}
        self._files = self._index()

    def _index(self) -> dict[str, Path]:
        """Table -> file. Fills found_in (table -> input folder, "" = the run folder itself) and duplicate_files
        (a table found more than once: validation blocks, since nobody can tell which file is the right one)."""
        from nbx.fsdr import csvstd

        # V4.0: data/training/<D> and data/scoring/<D> are read as one run (config.run_twins); single folder otherwise
        self.twins = config.run_twins(self.folder) or {"": self.folder}
        found: dict[str, dict[str, list[tuple[str, Path]]]] = {}
        for side, root in self.twins.items():
            dirs = [(root, "")] + [(root / f, f) for f in config.INPUT_FOLDERS if (root / f).is_dir()]
            for d, sub in dirs:
                for p in sorted(d.iterdir()):
                    if (not p.is_file() or not csvstd.file_format(p) or p.name.startswith("_")
                            or p.name.endswith(csvstd.SCHEMA_SUFFIX)):
                        continue
                    name = csvstd.table_name(p)
                    found.setdefault(schemas.TABLE_ALIASES.get(name, name), {}).setdefault(side, []).append((sub, p))
        files: dict[str, Path] = {}
        self.found_in: dict[str, str] = {}
        self.side_of: dict[str, str] = {}
        self.duplicate_files = {}
        for t, by_side in found.items():
            side = config.table_side(t) if config.table_side(t) in by_side else next(iter(by_side))
            cands = by_side[side]                                                  # a table in both twins: its own side
            if len(cands) > 1:
                self.duplicate_files[t] = [str(p) for _, p in cands]
            home = schemas.TABLES[t].folder if t in schemas.TABLES else None
            sub, p = next(((s, p) for s, p in cands if s == home), cands[0])     # its own folder first
            files[t], self.found_in[t], self.side_of[t] = p, sub, side
        return files

    def misfiled(self) -> dict[str, tuple[str, str]]:
        """V4.0: tables in the wrong twin (a scoring table under training/ or the reverse): table -> (found, expected).
        A warning only: the run still reads them."""
        if set(self.twins) == {""}:
            return {}
        return {t: (s, config.table_side(t)) for t, s in self.side_of.items()
                if t in schemas.TABLES and s != config.table_side(t)}

    @property
    def layout(self) -> str:
        """"folders" (plan §13), "flat" (V1.0: every file in the run folder) or "mixed"."""
        subs = set(self.found_in.values())
        return "flat" if subs <= {""} else "folders" if "" not in subs else "mixed"

    def misplaced(self) -> dict[str, tuple[str, str]]:
        """Contract tables filed in a folder that is not theirs: table -> (found in, expected)."""
        return {t: (sub, schemas.TABLES[t].folder) for t, sub in self.found_in.items()
                if sub and t in schemas.TABLES and sub != schemas.TABLES[t].folder}

    def tables(self) -> list[str]:
        return list(self._files)

    def unknown_files(self) -> list[str]:
        return [t for t in self._files if t not in schemas.TABLES]

    def path(self, table: str) -> Path | None:
        return self._files.get(table)

    def has(self, table: str) -> bool:
        return table in self._files

    def read(self, table: str) -> pd.DataFrame:
        if table not in self._cache:
            if table not in self._files:
                raise FileNotFoundError(f"{table}: no {table}.csv.gz (or a test format) in {self.folder}")
            df = read_table_file(self._files[table], table)
            self._cache[table] = compact_frame(df) if self.compact else df
        return self._cache[table]

    def manifest(self) -> dict:
        for root in [self.folder, *self.twins.values()]:
            p = root / config.MANIFEST_FILE
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def manifest_files(self) -> list[Path]:
        return [r / config.MANIFEST_FILE for r in dict.fromkeys([self.folder, *self.twins.values()])]

    def compact_table(self, table: str) -> None:
        """Replaces a cached raw table by its typed version (validation, once its raw checks are done)."""
        if table in self._cache:
            self._cache[table] = compact_frame(self._cache[table])


class SyntheticSource:
    """Generates the same tables in memory (no files) -- tests, demos, simulation."""

    def __init__(self, n: int = config.GENERATOR_DEFAULT_N, run_date: date | None = None, scenarios=None,
                 seed: int = config.GENERATOR_SEED, world=None, as_of: date | None = None):
        from nbx.fsdr.synth import generator
        from nbx.fsdr.synth import scenarios as scen
        from nbx.fsdr.synth.world import World

        run_date = run_date or date.today()
        world = world or World.create(n, seed=seed, scenarios=scen.parse(scenarios), as_of=as_of or run_date)
        self.run = generator.generate_run(world, run_date)
        self._cache: dict[str, pd.DataFrame] = {}

    def tables(self) -> list[str]:
        return list(self.run.tables)

    def has(self, table: str) -> bool:
        return table in self.run.tables

    def read(self, table: str) -> pd.DataFrame:
        if table not in self._cache:
            # same path as JsonSource: records -> raw frame
            self._cache[table] = _raw_frame(self.run.tables[table].to_dict("records"), table)
        return self._cache[table]

    def manifest(self) -> dict:
        return dict(self.run.manifest)

    def write(self, folder: str | Path, fmt: str = "json", layout: str = "folders") -> Path:
        from nbx.fsdr.synth import generator

        return generator.write_run(self.run, Path(folder), fmt, layout)


FolderSource = JsonSource          # V2.0 name: the folder holds JSON, JSONL, CSV or Parquet files
