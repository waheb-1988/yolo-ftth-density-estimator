"""Target upload folders (V4.0): the lists of the four groups and the control, chosen outside the engine. They sit
in the scoring folder of the run, one folder per target and one sub-folder per group (a = constraint, b = free):

    data/scoring/<D>/
    ├── target1/a_constraint/target1_a.csv.gz    T1C  production code 3
    ├── target1/b_free/target1_b.csv.gz          T1F  production code 4
    ├── target2/a_constraint/target2_a.csv.gz    T2C  challenger CC-v1
    ├── target2/b_free/target2_b.csv.gz          T2F  challenger CF-v1
    ├── control/control.csv.gz                   never contacted
    └── _targets_manifest.json                   written by the generator (source, sizes, seed)

One row per line, at least `access_method_val` (or `msisdn`); any other column (the sample_cg_tg fields: stratum
`Final`, plan, nationality, …) is kept and carried to the details file. `.csv.gz` in production; `.csv`, `.json`,
`.jsonl`, `.parquet` for tests. When the folder holds lists, P1 uses them instead of drawing the panel.

Checks (read_targets + check_targets), each with a count and examples:
  blocking  a line in two groups; no list at all for a contacted group; no access_method_val column
  dropped   not an 8-digit MSISDN; not in the eligible base; Nursering (B-02); in another campaign this week (H-05)
  warning   a duplicate inside one list; a size off the configured size by more than 2 %
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nbx.fsdr import config, csvstd
from nbx.fsdr.features import norm_val

VAL = "ACCESS_METHOD_VAL"
FORMATS = (csvstd.SUFFIX, ".csv", ".json", ".jsonl", ".parquet")


def targets_dir(input_folder: Path) -> Path:
    return Path(input_folder) / config.TARGETS_FOLDER


def group_file(folder: Path, key: str) -> Path | None:
    stem = targets_dir(folder) / config.TARGET_FILES[key]
    return next((stem.with_name(stem.name + f) for f in FORMATS if stem.with_name(stem.name + f).exists()), None)


def list_files(input_folder: Path) -> list[Path]:
    """The list files present (+ the manifest): what P1 fingerprints (never the whole scoring folder)."""
    files = [group_file(input_folder, k) for k in config.TARGET_FILES]
    man = targets_dir(input_folder) / config.TARGETS_MANIFEST
    return sorted([f for f in files if f is not None] + ([man] if man.exists() else []))


def has_targets(input_folder: Path) -> bool:
    return any(group_file(input_folder, k) for k in config.TARGET_FILES)


def expected_sizes(input_folder: Path) -> dict[str, int]:
    """Sizes to check the lists against: the generator's own sizes for dummy lists, else the configured sizes."""
    m = targets_dir(input_folder) / config.TARGETS_MANIFEST
    man = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
    return {k: int(v) for k, v in man["sizes"].items()} if man.get("source") == "generator" else config.group_sizes()


def _read(path: Path) -> pd.DataFrame:
    fmt = csvstd.file_format(path)
    if fmt in (csvstd.SUFFIX, ".csv"):
        return csvstd.read_csv(path)
    if fmt == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, lines=fmt == ".jsonl", dtype=False)


def read_targets(input_folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    """All the lists -> one frame (ACCESS_METHOD_VAL, GROUP_KEY, + the file's own columns, prefixed T_)."""
    frames, issues = [], []
    keys = list(config.group_sizes())
    for key in keys:
        p = group_file(input_folder, key)
        if p is None:
            issues.append({"level": "blocking", "group": key, "check": "missing_list",
                           "message": f"no list {config.TARGET_FILES[key]}.csv.gz", "count": 0})
            continue
        df = _read(p)
        cols = {c.lower(): c for c in df.columns}
        idc = cols.get("access_method_val") or cols.get("msisdn")
        if idc is None:
            issues.append({"level": "blocking", "group": key, "check": "no_id_column",
                           "message": f"{p.name}: no access_method_val / msisdn column", "count": 0})
            continue
        out = df.rename(columns={c: f"T_{c}" for c in df.columns if c != idc}).rename(columns={idc: VAL})
        out[VAL] = norm_val(out[VAL])
        frames.append(out.assign(GROUP_KEY=key, TARGET_FILE=str(p.relative_to(targets_dir(input_folder)))))
    t = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[VAL, "GROUP_KEY"])
    return t, issues


def check_targets(t: pd.DataFrame, segments: pd.DataFrame, busy: set[str] | None = None,
                  sizes: dict[str, int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """-> (kept lines, rejected lines with their reason, issues)."""
    issues: list[dict] = []
    sizes = sizes or config.group_sizes()

    def add(level, check, msg, rows):
        if len(rows):
            issues.append({"level": level, "check": check, "message": msg, "count": int(len(rows)),
                           "examples": rows[VAL].astype(str).head(5).tolist(),
                           "groups": rows["GROUP_KEY"].value_counts().to_dict()})

    dup_in = t[t.duplicated([VAL, "GROUP_KEY"])]
    add("warning", "duplicate_in_list", "a line listed twice in one group (kept once)", dup_in)
    t = t.drop_duplicates([VAL, "GROUP_KEY"])
    two = t[t.duplicated(VAL, keep=False)]
    add("blocking", "line_in_two_groups", "a line in two groups (A-30): fix the lists", two)
    rejected = []

    def reject(mask, reason):
        nonlocal t
        rows = t[mask]
        if len(rows):
            rejected.append(rows.assign(REJECT_REASON=reason))
            add("warning", "dropped", f"dropped: {reason}", rows)
            t = t[~mask]

    reject(~t[VAL].astype(str).str.fullmatch(r"\d{8}").fillna(False), "not an 8-digit MSISDN")
    base = segments.set_index(VAL)
    reject(~t[VAL].isin(base.index), "not in the eligible base (D-07)")
    reject(t[VAL].map(base["EXCLUDED"]).eq(1) & t["GROUP_KEY"].ne(config.CONTROL_GROUP), "Nursering segment (B-02)")
    if busy:
        reject(t[VAL].isin(busy) & t["GROUP_KEY"].ne(config.CONTROL_GROUP), "in another campaign this week (H-05)")
    for k, v in sizes.items():
        n = int((t["GROUP_KEY"] == k).sum())
        if v and abs(n - v) > config.GROUP_SIZE_TOLERANCE * v:
            issues.append({"level": "warning", "check": "size", "group": k, "count": n,
                           "message": f"{k}: {n:,} lines kept, configured {v:,} (+-2 %)"})
    rej = pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame(columns=[VAL, "GROUP_KEY", "REJECT_REASON"])
    return t.reset_index(drop=True), rej, issues


def write_targets(panel: pd.DataFrame, input_folder: Path, fmt: str = "csv.gz", meta: dict | None = None) -> list[Path]:
    """Writes one list per group (dummy targets for tests, or a drawn panel to hand over)."""
    cols = [VAL, "ACCESS_METHOD_ID", "NATIONALITY_GROUP", "SEGMENT", "MICRO_SEGMENT", "VALUE_BAND", "USAGE_TYPE",
            "CURRENT_PLAN", "STRATUM"]
    written = []
    for key, rel in config.TARGET_FILES.items():
        part = panel.loc[panel["GROUP_KEY"] == key, [c for c in cols if c in panel.columns]]
        part = part.rename(columns={VAL: "access_method_val", "STRATUM": "Final"})
        part.columns = [c.lower() if c != "Final" else c for c in part.columns]
        p = targets_dir(input_folder) / f"{rel}.{fmt}"
        p.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv.gz":
            csvstd.write_csv(part, p)
        elif fmt == "csv":
            csvstd.write_csv(part, p, compress=False)
        else:
            part.to_json(p, orient="records", lines=fmt == "jsonl")
        written.append(p)
    man = {"source": "generator", "sizes": panel["GROUP_KEY"].value_counts().to_dict(), "format": fmt, **(meta or {})}
    (targets_dir(input_folder) / config.TARGETS_MANIFEST).write_text(json.dumps(man, indent=2), encoding="utf-8")
    return written
