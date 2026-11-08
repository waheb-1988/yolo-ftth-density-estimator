"""Step folders (plan §14): every pipeline step saves its output before the next one starts.

    data/runs/<D>/train/T0_load/  ...  data/runs/<D>/score/P10_report/  ...  data/runs/<D>/final/

    with steps.Step(runs_dir(D) / "train", "T0_load", inputs=fingerprint(files)) as st:
        if st.done:                       # same inputs, finished before: skipped (resume)
            return st.read_json("summary.json")
        st.write_table(df, "base_features")         # Parquet + CSV (full or preview)
        st.write_json("summary.json", {...})
        st.rows(rows_in=..., rows_out=...)

Each step folder holds its files and _step.json: status (running / done / failed), start, duration, rows in / out,
the fingerprint of its inputs and the list of its outputs. A step is skipped when its _step.json says done with
the same inputs; force=True re-runs it (fsdr-* --from <step>).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nbx.fsdr import config

STATUSES = ("running", "done", "failed", "blocked")


def runs_dir(run_date: date | str, root: Path | None = None) -> Path:
    d = run_date.isoformat() if isinstance(run_date, date) else str(run_date)
    return Path(root or config.FSDR_RUNS_DIR) / d


def fingerprint(paths: list[Path] | dict, extra: dict | None = None) -> str:
    """Cheap, stable id of a step's inputs: name, size and modification time of each file (hashing 12 GB of JSON
    every Thursday would cost minutes), plus any parameters. A dict is hashed as it is."""
    if isinstance(paths, dict):
        payload = paths
    else:
        payload = {str(p): [p.stat().st_size, int(p.stat().st_mtime)] for p in sorted(map(Path, paths)) if p.exists()}
    blob = json.dumps({"inputs": payload, "extra": extra or {}}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Step:
    """One step folder. Context manager: status running -> done, or failed (with the error) on an exception."""

    def __init__(self, parent: Path, name: str, inputs: str = "", force: bool = False):
        self.dir = Path(parent) / name
        self.name = name
        self.inputs = inputs
        self.force = force
        self.outputs: list[str] = []
        self._rows: dict = {}
        self._extra: dict = {}
        self._t0 = 0.0
        prev = read_step(self.dir)
        self.done = (not force and prev.get("status") == "done" and prev.get("inputs") == inputs
                     and all((self.dir / f).exists() for f in prev.get("outputs", [])))
        self.previous = prev

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> "Step":
        self._t0 = time.time()
        if not self.done:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._write_meta("running")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.done:
            return False
        if exc is not None:
            self._write_meta("failed", error=f"{exc_type.__name__}: {exc}")
        else:
            self._write_meta(self._extra.pop("_status", "done"))
        return False

    def block(self, reason: str) -> None:
        """Ends the step as blocked (the run stops here; the outputs explain why)."""
        self._extra["_status"] = "blocked"
        self._extra["reason"] = reason

    def rows(self, **counts) -> None:
        self._rows.update({k: int(v) for k, v in counts.items()})

    def note(self, **info) -> None:
        self._extra.update(info)

    def _write_meta(self, status: str, error: str | None = None) -> None:
        meta = {"step": self.name, "status": status, "inputs": self.inputs,
                "started": datetime.fromtimestamp(self._t0).isoformat(timespec="seconds"),
                "seconds": round(time.time() - self._t0, 2), "rows": self._rows, "outputs": self.outputs,
                **{k: v for k, v in self._extra.items() if not k.startswith("_")}}
        if error:
            meta["error"] = error
        tmp = self.dir / (config.STEP_FILE + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.dir / config.STEP_FILE)

    # -- outputs ---------------------------------------------------------------
    def _add(self, name: str) -> Path:
        if name not in self.outputs:
            self.outputs.append(name)
        return self.dir / name

    def output(self, name: str) -> Path:
        """Registers an output written by other code (e.g. one partition file "base_features/part-03.parquet")."""
        p = self._add(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, name: str, obj) -> Path:
        p = self._add(name)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self._add(name)
        p.write_text(text, encoding="utf-8")
        return p

    def write_csv(self, df: pd.DataFrame, name: str) -> Path:
        p = self._add(name)
        df.to_csv(p, index=False, encoding="utf-8")
        return p

    def write_table(self, df: pd.DataFrame, stem: str, csv: str = "auto") -> list[Path]:
        """<stem>.parquet always, plus <stem>.csv in full (small tables, e.g. the 50k of scoring) or a
        preview.csv of the first rows (1M-row training tables). csv: "auto" | "full" | "preview" | "none"."""
        out = [self._add(f"{stem}.parquet")]
        _parquet_safe(df).to_parquet(out[0], index=False)
        mode = csv if csv != "auto" else ("full" if len(df) <= config.STEP_FULL_CSV_MAX_ROWS else "preview")
        if mode == "full":
            out.append(self._add(f"{stem}.csv"))
            df.to_csv(out[-1], index=False, encoding="utf-8")
        elif mode == "preview":
            out.append(self._add(f"{stem}_preview.csv"))
            df.head(config.STEP_PREVIEW_ROWS).to_csv(out[-1], index=False, encoding="utf-8")
        return out

    def read_json(self, name: str):
        return json.loads((self.dir / name).read_text(encoding="utf-8"))


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns mixing types (raw input values) cannot go to Parquet as they are: written as text."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            kinds = {type(v) for v in out[c].dropna().head(10_000)}
            if len(kinds) > 1:
                out[c] = out[c].map(lambda v: None if v is None else str(v)).astype("string")
    return out


def read_step(step_dir: Path) -> dict:
    p = Path(step_dir) / config.STEP_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_steps(run_dir: Path) -> list[dict]:
    """Every step of a run (train / score / final), in order, with its status and files: for the UI and the API."""
    out = []
    for phase in ("train", "score", "final"):
        d = Path(run_dir) / phase
        if not d.is_dir():
            continue
        for sd in sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: _step_order(p.name)):
            meta = read_step(sd)
            out.append({"phase": phase, "step": sd.name, "status": meta.get("status", "unknown"),
                        "seconds": meta.get("seconds"), "rows": meta.get("rows", {}),
                        "files": sorted(f.name for f in sd.iterdir() if f.is_file() and f.name != config.STEP_FILE)})
    return out


def _step_order(name: str) -> tuple:
    """T0 < T1 < ... < T8; P0 < P1 < ... < P6 < P6b < P7 < P7b < ... < P10."""
    head = name.split("_", 1)[0]
    digits = "".join(ch for ch in head[1:] if ch.isdigit())
    return (head[:1], int(digits) if digits else 0, head)
