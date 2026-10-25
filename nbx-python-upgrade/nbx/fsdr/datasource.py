"""Where the FSDR pipelines get their tables.

    JsonSource("data/fsdr/2026-10-22")    a folder of JSON files, one per table (real exports or generator output)
    SyntheticSource(n=5000, ...)          the same tables generated in memory

Both return the same thing for a table: a raw frame (python values, None =
null, schema columns first) exactly as the JSON file holds it, so swapping
dummy data for real data is a path change, not a code change. Use
schemas.coerce() to get typed columns.

Formats: "<table>.json" (array of records) or "<table>.jsonl" (one record per
line). The format is detected from the content, not the extension.
"""
from __future__ import annotations

import json
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


class JsonSource:
    """A folder of JSON files, one per table (file name = exact table name)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"No such data folder: {self.folder}")
        self._cache: dict[str, pd.DataFrame] = {}
        self._files = self._index()

    def _index(self) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for p in sorted(self.folder.iterdir()):
            if p.suffix.lower() not in (".json", ".jsonl") or p.name.startswith("_"):
                continue
            name = p.name[: -len(p.suffix)]
            files[schemas.TABLE_ALIASES.get(name, name)] = p
        return files

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
                raise FileNotFoundError(f"{table}: no {table}.json / .jsonl in {self.folder}")
            self._cache[table] = _raw_frame(read_json_records(self._files[table]), table)
        return self._cache[table]

    def manifest(self) -> dict:
        p = self.folder / config.MANIFEST_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


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

    def write(self, folder: str | Path, fmt: str = "json") -> Path:
        from nbx.fsdr.synth import generator

        return generator.write_run(self.run, Path(folder), fmt)
