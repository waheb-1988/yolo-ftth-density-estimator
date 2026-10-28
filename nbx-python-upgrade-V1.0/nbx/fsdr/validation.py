"""Validation of a run folder (or any DataSource) against the data contract.

Checks, each graded blocking / warning / info:
  - files: every table present (missing model-used table = blocking)
  - fields: missing fields (blocking for model-used tables), unknown extra fields (warning)
  - raw types: wrong type in a key field = blocking, elsewhere = warning
  - keys: null keys and duplicates on the primary key (blocking above 1 % unless
    duplicates are expected, e.g. stacked model scores resolved by max)
  - cross-table: child keys missing from the parent table (BR joins)
  - coverage: share of the base scored by each model (< 70 % = warning)
  - consistency: activities M1 = W1+..+W4, Sub_Amt_m1 >= its weeks, dates inside
    their BR windows, manifest anchor date = calendar anchor

The report is JSON-serialisable and also rendered as Markdown; both are
written next to the data as _validation.json / _validation.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import calendar as cal
from nbx.fsdr import config, schemas
from nbx.fsdr.datasource import DataSource, JsonSource

LEVELS = ("blocking", "warning", "info")


@dataclass
class Issue:
    level: str
    check: str
    table: str
    message: str
    count: int = 0
    column: str | None = None
    examples: list = field(default_factory=list)


@dataclass
class TableReport:
    table: str
    present: bool
    rows: int = 0
    missing_fields: list[str] = field(default_factory=list)
    unknown_fields: list[str] = field(default_factory=list)
    type_errors: dict[str, int] = field(default_factory=dict)
    null_keys: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0


@dataclass
class ValidationReport:
    run_date: str | None
    anchor_date: str | None
    source: str
    tables: dict[str, TableReport] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    relations: list[dict] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def add(self, level: str, check: str, table: str, message: str, count: int = 0, column: str | None = None,
            examples: list | None = None) -> None:
        self.issues.append(Issue(level, check, table, message, int(count), column, list(examples or [])))

    @property
    def status(self) -> str:
        if any(i.level == "blocking" for i in self.issues):
            return "blocked"
        return "warnings" if any(i.level == "warning" for i in self.issues) else "ok"

    @property
    def usable(self) -> bool:
        return self.status != "blocked"

    def counts(self) -> dict[str, int]:
        return {lv: sum(1 for i in self.issues if i.level == lv) for lv in LEVELS}

    def to_dict(self) -> dict:
        return {"status": self.status, "usable": self.usable, "run_date": self.run_date, "anchor_date": self.anchor_date,
                "source": self.source, "checked_at": self.checked_at, "issue_counts": self.counts(),
                "issues": [asdict(i) for i in self.issues], "coverage": self.coverage, "relations": self.relations,
                "tables": {k: asdict(v) for k, v in self.tables.items()}}

    def to_markdown(self) -> str:
        c = self.counts()
        icon = {"ok": "✅", "warnings": "⚠️", "blocked": "⛔"}[self.status]
        lines = [f"# FSDR validation report — {self.run_date or '?'}", "",
                 f"**Status:** {icon} {self.status}  ·  blocking {c['blocking']} · warnings {c['warning']} · info {c['info']}",
                 f"**Anchor date (T):** {self.anchor_date or '?'}  ·  **Source:** {self.source}  ·  checked {self.checked_at}", ""]
        for lv in LEVELS:
            items = [i for i in self.issues if i.level == lv]
            if not items:
                continue
            lines += [f"## {lv.capitalize()} ({len(items)})", "", "| Check | Table | Column | Count | Message |", "|---|---|---|---|---|"]
            for i in items:
                ex = f" e.g. {', '.join(map(str, i.examples[:3]))}" if i.examples else ""
                lines.append(f"| {i.check} | `{i.table}` | {i.column or ''} | {i.count} | {i.message}{ex} |")
            lines.append("")
        if self.coverage:
            lines += ["## Model-score coverage of the base", "", "| Model file | Coverage |", "|---|---|"]
            lines += [f"| `{k}` | {v:.1%} |" for k, v in self.coverage.items()]
            lines.append("")
        lines += ["## Tables", "", "| Table | Present | Rows | Missing fields | Unknown fields | Type errors | Null keys | Duplicates |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in self.tables.values():
            lines.append(f"| `{t.table}` | {'yes' if t.present else '**no**'} | {t.rows} | {len(t.missing_fields)} | "
                         f"{len(t.unknown_fields)} | {sum(t.type_errors.values())} | {sum(t.null_keys.values())} | {t.duplicate_rows} |")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm_keys(s: pd.Series) -> pd.Series:
    """_norm_key on a whole column, vectorised (review #15)."""
    from nbx.fsdr.features import canonical_keys

    return canonical_keys(s, strip_zero=False, empty_none=False)


def _key_set(s: pd.Series, drop_empty: bool = False) -> set:
    k = norm_keys(s).dropna()
    return set(k[k != ""]) if drop_empty else set(k)


def _norm_key(v) -> str | None:
    """Key value -> canonical string ('91234567', 1001, 1001.0 and '1001' compare equal)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        s = str(int(v))
    elif isinstance(v, (float, np.floating)):
        s = str(int(v)) if float(v).is_integer() else str(v)
    else:
        s = str(v).strip()
    return s


def _share(count: int, rows: int) -> float:
    return count / rows if rows else 0.0


def _over(count: int, rows: int) -> bool:
    return _share(count, rows) > config.BLOCKING_KEY_ISSUE_SHARE


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _check_table(src: DataSource, schema: schemas.TableSchema, rep: ValidationReport) -> pd.DataFrame | None:
    name = schema.name
    if not src.has(name):
        level = "blocking" if schema.used_by_model and schema.family != "model_score" else "warning"
        why = ("missing model file: its score falls back to random 0-0.02 (BR-47)" if schema.family == "model_score"
               else "required table missing" if level == "blocking" else "table missing (not used by the model)")
        rep.add(level, "file_present", name, why)
        rep.tables[name] = TableReport(name, present=False)
        return None
    df = src.read(name)
    tr = TableReport(name, present=True, rows=len(df))
    rep.tables[name] = tr
    if not len(df):
        rep.add("warning" if not schema.used_by_model else "blocking", "rows", name, "file has no rows")
        return df

    tr.missing_fields = [f for f in schema.field_names if f not in df.columns]
    tr.unknown_fields = [c for c in df.columns if c not in schema.field_names]
    if tr.missing_fields:
        rep.add("blocking" if schema.used_by_model else "warning", "fields_missing", name,
                f"{len(tr.missing_fields)} field(s) missing", len(tr.missing_fields), examples=tr.missing_fields)
    if tr.unknown_fields:
        rep.add("warning", "fields_unknown", name, "extra field(s) not in the contract (ignored)",
                len(tr.unknown_fields), examples=tr.unknown_fields)

    for f in schema.fields:
        if f.name not in df.columns:
            continue
        bad, examples = schemas.type_errors(df[f.name], f.raw_type)
        if bad:
            tr.type_errors[f.name] = bad
            rep.add("blocking" if f.key else "warning", "raw_type", name,
                    f"{bad} value(s) are not {f.raw_type}", bad, f.name, examples)
        if f.key:
            nulls = int(df[f.name].isna().sum())
            if nulls:
                tr.null_keys[f.name] = nulls
                rep.add("blocking" if _over(nulls, len(df)) else "warning", "null_key", name,
                        f"{nulls} null key value(s) ({_share(nulls, len(df)):.1%})", nulls, f.name)

    pk = [c for c in schema.primary_key if c in df.columns]
    if pk:
        keyed = df[pk].apply(norm_keys)
        dups = int(keyed.duplicated().sum())
        tr.duplicate_rows = dups
        if dups:
            blocking = schema.duplicates_blocking and _over(dups, len(df))
            note = "" if schema.duplicates_blocking else " (expected: resolved downstream, e.g. max of stacked scores)"
            rep.add("blocking" if blocking else "warning", "duplicates", name,
                    f"{dups} duplicate row(s) on ({', '.join(pk)}){note}", dups, ",".join(pk))
    return df


def _check_relations(src: DataSource, rep: ValidationReport) -> None:
    for r in schemas.relations():
        if not (src.has(r.child) and src.has(r.parent)):
            continue
        child, parent = src.read(r.child), src.read(r.parent)
        if r.child_col not in child.columns or r.parent_col not in parent.columns:
            continue
        parent_keys = _key_set(parent[r.parent_col])
        ck = norm_keys(child[r.child_col]).dropna().tolist()
        if not ck:
            continue
        orphans = sum(1 for k in ck if k not in parent_keys)
        rep.relations.append({"child": f"{r.child}.{r.child_col}", "parent": f"{r.parent}.{r.parent_col}",
                              "child_keys": len(ck), "orphans": orphans, "orphan_share": round(_share(orphans, len(ck)), 4)})
        if orphans:
            level = "blocking" if r.blocking and _over(orphans, len(ck)) else "warning"
            rep.add(level, "cross_table", r.child, f"{orphans} {r.child_col} value(s) not found in {r.parent}.{r.parent_col}"
                    + (f" -- {r.note}" if r.note else ""), orphans, r.child_col)


def _check_coverage(src: DataSource, rep: ValidationReport) -> None:
    if not src.has("base_prepaid"):
        return
    base = _key_set(src.read("base_prepaid")["ACCESS_METHOD_VAL"], drop_empty=True)
    for m in config.MODEL_SCORE_REGISTRY:
        if not src.has(m.table) or "access_method_val" not in src.read(m.table).columns:
            continue
        scored = _key_set(src.read(m.table)["access_method_val"], drop_empty=True)
        cov = _share(len(scored & base), len(base))
        rep.coverage[m.table] = round(cov, 4)
        if m.tenure_range is None and cov < config.MIN_SCORE_COVERAGE:
            rep.add("warning", "coverage", m.table, f"scores only {cov:.1%} of the base (missing -> random 0-0.02, BR-47)")


def _check_consistency(src: DataSource, rep: ValidationReport, run_date: date | None) -> None:
    # activities: M1 window = W1..W4 exactly (BR 4.2)
    for anchor in ("T", "D"):
        names = {p: schemas.activity_table_name(p, anchor) for p in ("W1", "W2", "W3", "W4", "M1")}
        if not all(src.has(t) for t in names.values()):
            continue
        m1 = src.read(names["M1"])
        if "ACCESS_METHOD_VAL" not in m1.columns or "Local_Voice_min_M1" not in m1.columns:
            continue
        tot = pd.Series(0.0, index=norm_keys(m1["ACCESS_METHOD_VAL"]))
        weeks = pd.Series(dtype=float)
        for w in ("W1", "W2", "W3", "W4"):
            df = src.read(names[w])
            col = f"Local_Voice_min_{w}"
            if col in df.columns:
                s = pd.Series(_num(df, col).to_numpy(), index=norm_keys(df["ACCESS_METHOD_VAL"]))
                weeks = weeks.add(s.groupby(level=0).sum(), fill_value=0)
        m1v = pd.Series(_num(m1, "Local_Voice_min_M1").to_numpy(), index=tot.index).groupby(level=0).sum()
        diff = (m1v - weeks.reindex(m1v.index).fillna(0)).abs()
        off = int((diff > 0.05 + 0.001 * m1v.abs()).sum())
        if off:
            rep.add("warning", "window_consistency", names["M1"], f"{off} line(s) where Local_Voice_min_M1 != W1+W2+W3+W4", off)

    # subscriptions: M1 (30 days incl. anchor) contains W1..W4 (28 days)
    for t in ("subs_amt_auto", "subs_amt_score"):
        if not src.has(t):
            continue
        df = src.read(t)
        cols = [f"Sub_Amt_w{i}" for i in range(1, 5)]
        if not set(cols + ["Sub_Amt_m1"]) <= set(df.columns):
            continue
        off = int((_num(df, "Sub_Amt_m1") + 0.01 < sum(_num(df, c).fillna(0) for c in cols)).sum())
        if off:
            rep.add("warning", "window_consistency", t, f"{off} line(s) where Sub_Amt_m1 < Sub_Amt_w1..w4", off)

    if run_date is None:
        return
    win = cal.campaign_windows(run_date)
    if src.has("targeting_dataiku_auto") and "date1" in src.read("targeting_dataiku_auto").columns:
        d1 = pd.to_datetime(src.read("targeting_dataiku_auto")["date1"], errors="coerce").dt.date
        out = int(((d1 < win["targeting"].start) | (d1 > win["targeting"].end)).sum())
        if out:
            rep.add("warning", "date_window", "targeting_dataiku_auto",
                    f"{out} date1 value(s) outside {win['targeting'].start}..{win['targeting'].end} (BR-21)", out, "date1")
    if src.has("subs_btl_auto") and "FULFILMENTSUCCESSDATE_TIME" in src.read("subs_btl_auto").columns:
        ts = pd.to_datetime(src.read("subs_btl_auto")["FULFILMENTSUCCESSDATE_TIME"], errors="coerce").dt.date
        out = int(((ts < win["reward"].start) | (ts > win["reward"].end)).sum())
        if out:
            rep.add("warning", "date_window", "subs_btl_auto",
                    f"{out} fulfilment(s) outside {win['reward'].start}..{win['reward'].end} (BR-23)", out,
                    "FULFILMENTSUCCESSDATE_TIME")
    if src.has("base_prepaid") and "ACCESS_METHOD_VAL" in src.read("base_prepaid").columns:
        vals = norm_keys(src.read("base_prepaid")["ACCESS_METHOD_VAL"]).dropna()
        bad = int((~vals.str.fullmatch(r"\d{8}") | (vals <= config.MSISDN_MIN_EXCLUSIVE)).sum())
        if bad:
            rep.add("warning", "msisdn_format", "base_prepaid",
                    f"{bad} MSISDN(s) not 8 digits > '{config.MSISDN_MIN_EXCLUSIVE}' (BR-02: filtered out downstream)", bad)


def validate(src: DataSource, run_date: date | None = None) -> ValidationReport:
    manifest = src.manifest()
    if run_date is None and manifest.get("run_date"):
        run_date = date.fromisoformat(manifest["run_date"])
    anchor = cal.anchor_date(run_date) if run_date else None
    rep = ValidationReport(run_date.isoformat() if run_date else None, anchor.isoformat() if anchor else None,
                           manifest.get("source", "unknown"))
    if not manifest:
        rep.add("warning", "manifest", config.MANIFEST_FILE, "no _manifest.json: run date taken from the folder name or argument")
    elif manifest.get("anchor_date") and anchor and manifest["anchor_date"] != anchor.isoformat():
        rep.add("blocking", "manifest", config.MANIFEST_FILE,
                f"manifest anchor_date {manifest['anchor_date']} != calendar anchor {anchor} for run date {run_date}")
    if run_date is None:
        rep.add("warning", "run_date", "-", "run date unknown: date-window checks skipped")

    for schema in schemas.TABLES.values():
        _check_table(src, schema, rep)
        if isinstance(src, JsonSource):      # raw values are only needed by the per-table checks above (review #15)
            src.compact_table(schema.name)
    if isinstance(src, JsonSource):
        for t in src.unknown_files():
            rep.add("info", "file_unknown", t, "file not in the data contract (ignored)")
    _check_relations(src, rep)
    _check_coverage(src, rep)
    _check_consistency(src, rep, run_date)
    return rep


def validate_folder(folder: str | Path, run_date: date | None = None, write: bool = True) -> ValidationReport:
    """Validates a run folder; the run date comes from the manifest, the argument or the folder name."""
    folder = Path(folder)
    src = JsonSource(folder)
    if run_date is None and not src.manifest():
        try:
            run_date = date.fromisoformat(folder.name)
        except ValueError:
            pass
    rep = validate(src, run_date)
    if write:
        (folder / config.VALIDATION_JSON).write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False, default=str),
                                                     encoding="utf-8")
        (folder / config.VALIDATION_MD).write_text(rep.to_markdown(), encoding="utf-8")
    return rep
