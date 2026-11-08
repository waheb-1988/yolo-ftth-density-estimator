"""Builds notebooks/fsdr_results.ipynb (run it again after editing the cells below):

    uv run python notebooks/build_results_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# FSDR V2.0: check the results of a run

This notebook reads what the terminal commands saved and shows it. It computes nothing new.

| Where | What |
|---|---|
| `data/input/<D>/` | the input: 5 folders `usage/ models/ subscription/ catalog/ others/` |
| `data/runs/<D>/train/T0_load … T8_*/` | training steps on the full base (≈ 1M lines) |
| `data/runs/<D>/score/P0_* … P10_*/` | scoring steps on the 40k panel + 10k control |
| `data/runs/<D>/final/` | `input_flytxt_<D>.csv` (38 columns), control, summary |
| `data/cache/<D>/` | Parquet cache, 16 partitions per table |

**How to use:**
1. In a terminal, run the commands of `docs/RUN_FROM_TERMINAL.md` (for example `fsdr-generate`, then `fsdr-load`).
2. Here, set `RUN_DATE` in the next cell (or leave it as `None` for the latest run), then **Run All**.

Sections 1–8 cover what exists today (Phases 2–3 and the V3.0 groups). Section 9 lists **every** step folder and previews its files, so the steps added in later phases (features, actions, offers, `input_flytxt`, audit) show up here without changing the notebook."""),
    code("""# --- settings -------------------------------------------------------------------
RUN_DATE = None          # e.g. "2026-10-22"; None = the latest run in data/runs (or data/input)
import os
RUN_DATE = RUN_DATE or os.environ.get("FSDR_NOTEBOOK_RUN_DATE")    # set by: run.py notebook-html --date D
PREVIEW_ROWS = 10

import json, os, sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:      # works from notebooks/ or the repo root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from nbx.fsdr import config, csvstd, schemas, steps
from nbx.fsdr.contract import completeness
from nbx.fsdr.contract.cache import CacheSource, verify_cache
from nbx.fsdr.datasource import JsonSource

pd.set_option("display.max_columns", 60, "display.width", 200, "display.max_colwidth", 80)

def _dates(d):
    return sorted(p.name for p in Path(d).glob("20??-??-??") if p.is_dir()) if Path(d).is_dir() else []

RUN_DATE = RUN_DATE or (_dates(config.FSDR_RUNS_DIR) or _dates(config.FSDR_INPUT_DIR) or [None])[-1]
INPUT, RUN, CACHE = (config.FSDR_INPUT_DIR / str(RUN_DATE), config.FSDR_RUNS_DIR / str(RUN_DATE),
                     config.FSDR_CACHE_DIR / str(RUN_DATE))
print(f"run date : {RUN_DATE}")
print(f"input    : {INPUT}  ({'found' if INPUT.is_dir() else 'MISSING'})")
print(f"run      : {RUN}  ({'found' if RUN.is_dir() else 'MISSING: run fsdr-load first'})")
print(f"cache    : {CACHE}  ({'found' if CACHE.is_dir() else 'missing'})")
print("other runs:", ", ".join(_dates(config.FSDR_RUNS_DIR)) or "none")"""),

    md("## 1. Input: the 5 folders"),
    code("""src = JsonSource(INPUT)
man = src.manifest()
rows = []
for t, s in schemas.TABLES.items():
    p = src.path(t)
    rows.append({"folder": s.folder, "table": t, "present": p is not None,
                 "found_in": src.found_in.get(t, ""), "format": p.suffix[1:] if p else "",
                 "MB": round(p.stat().st_size / 1e6, 2) if p else 0,
                 "rows (manifest)": (man.get("row_counts") or {}).get(t)})
inputs = pd.DataFrame(rows)
print(f"layout: {src.layout} | tables found: {inputs.present.sum()} / {len(inputs)} | "
      f"misplaced: {src.misplaced() or 'none'} | duplicates: {list(src.duplicate_files) or 'none'}")
print(f"manifest: run_date={man.get('run_date')} anchor_date={man.get('anchor_date')} source={man.get('source')} "
      f"format={man.get('format')} n={(man.get('generator') or {}).get('n')}")
display(inputs.groupby("folder").agg(tables=("table", "count"), present=("present", "sum"), MB=("MB", "sum")))
inputs"""),

    md("## 2. Every step of the run: status, time, rows, files"),
    code("""st = pd.DataFrame(steps.list_steps(RUN))
if st.empty:
    print("No step yet: run  uv run python run.py fsdr-load --data-dir", INPUT)
else:
    st["files"] = st["files"].map(lambda f: ", ".join(f))
    display(st)"""),

    md("## 3. T0 — data agent: summary and validation"),
    code("""T0 = RUN / "train" / "T0_load"
summary = json.loads((T0 / "summary.json").read_text(encoding="utf-8"))
display(Markdown(f"**Status: {summary['status'].upper()}** · pipeline `{summary['pipeline']}` · "
                 f"validation `{summary['validation']}` · {summary['tables']} tables · {summary['rows_in']:,} rows"))
print(summary["explanation"])
step_meta = steps.read_step(T0)
print(f"\\nT0 took {step_meta.get('seconds')} s; started {step_meta.get('started')}")"""),
    code("""val = json.loads((T0 / "validation_report.json").read_text(encoding="utf-8"))
print("issue counts:", val["issue_counts"], "| layout:", val.get("layout"))
issues = pd.DataFrame(val["issues"])
display(issues if len(issues) else Markdown("_No issue: every check passed._"))"""),
    code("""tables = pd.DataFrame(val["tables"]).T
tables["type_errors"] = tables["type_errors"].map(lambda d: sum(d.values()) if isinstance(d, dict) else 0)
tables["null_keys"] = tables["null_keys"].map(lambda d: sum(d.values()) if isinstance(d, dict) else 0)
tables["folder"] = [schemas.TABLES[t].folder if t in schemas.TABLES else "" for t in tables.index]
tables[["folder", "present", "rows", "type_errors", "null_keys", "duplicate_rows", "missing_fields", "unknown_fields"]]"""),
    code("""cov = pd.Series(val.get("coverage", {})).sort_values()
if len(cov):
    ax = (cov * 100).plot.barh(figsize=(9, 4), title="Model-score coverage of the base (%)")
    ax.axvline(config.MIN_SCORE_COVERAGE * 100, ls="--", c="grey")
    ax.set_xlabel("% of base lines scored (dashed = warning threshold)")
    ax.set_yticklabels([t.split(".")[0][:45] for t in cov.index])
    plt.tight_layout(); plt.show()
pd.DataFrame(val.get("relations", []))"""),

    md("## 4. Weekly completeness per pipeline"),
    code("""comp = pd.read_csv(T0 / "completeness.csv")
print("pipeline checked at T0:", summary["pipeline"])
display(comp[comp.level != "ok"] if (comp.level != "ok").any() else Markdown("_Every table this pipeline needs is present._"))
# the same check for the 3 pipelines on this input
pd.DataFrame([{k: v for k, v in completeness.check(src, p, pd.Timestamp(RUN_DATE).date()).items()
               if k in ("pipeline", "complete", "fresh", "blocking", "warnings")} for p in completeness.PIPELINES])"""),

    md("## 5. customer_profile: age and tenure"),
    code("""print("checks:", json.dumps(val.get("profile", {}), indent=1))
if src.has(config.PROFILE_TABLE):
    prof = src.read(config.PROFILE_TABLE)
    age, ten = pd.to_numeric(prof["age"], errors="coerce"), pd.to_numeric(prof["tenure"], errors="coerce")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    age.plot.hist(bins=40, ax=ax[0], title="age"); ten.plot.hist(bins=40, ax=ax[1], title=f"tenure ({config.PROFILE_TENURE_UNIT})")
    plt.tight_layout(); plt.show()
    display(pd.DataFrame({"age": age.describe(), "tenure": ten.describe()}).T.round(1))"""),

    md("## 6. Parquet cache: partitions and checksums"),
    code("""cache = CacheSource(RUN_DATE)
cm = cache.manifest()
bad = verify_cache(RUN_DATE)
print(f"{len(cm['tables'])} tables, {cm['partitions']} partitions, {cm['bytes'] / 1e6:,.1f} MB, built in {cm['seconds']} s; "
      f"checksums: {'all OK' if not bad else f'{len(bad)} file(s) CHANGED: {bad[:5]}'}")
parts = pd.DataFrame({t: {k: f["rows"] for k, f in info["files"].items()}
                      for t, info in cm["tables"].items() if info["partitioned"]}).T
display(parts.describe().T[["mean", "min", "max"]].round(0).T)
ax = parts.loc[["base_prepaid", "activities_w1_score", "subs_btl_auto"]].T.plot.bar(figsize=(11, 3.5),
                                                                                    title="rows per partition")
plt.tight_layout(); plt.show()"""),
    code("""# Look at any table from the cache: the whole table, or one partition of a per-line table
TABLE, PART = "activities_w1_score", 0
df = cache.read(TABLE, part=PART if cache.partitioned(TABLE) else None)
print(TABLE, "partition", PART if cache.partitioned(TABLE) else "-", df.shape)
df.head(PREVIEW_ROWS)"""),

    md("""## 7. T2 features and T3 segments (Phase 3: `fsdr-features`)

T2 builds B-01 … B-12 for the full base, partition by partition; T3 adds the S-rule segment, the bands, the
micro-segment (nationality × age × tenure), the week-0 history, `action_original` (code 1's dictionary, via the
segment engine or the crosswalk) and `action_agent` (the agent dictionary; v0 = copy of the original)."""),
    code("""T2, T3 = RUN / "train" / "T2_features", RUN / "train" / "T3_segments"
if not (T3 / "summary.json").exists():
    print("No T3 yet: run  uv run python run.py fsdr-features --date", RUN_DATE)
else:
    s2 = json.loads((T2 / "summary.json").read_text(encoding="utf-8"))
    s3 = json.loads((T3 / "summary.json").read_text(encoding="utf-8"))
    display(Markdown(f"**{s2['lines']:,} lines** ({s2['tables']}), {s2['features']} features · "
                     f"T2 {steps.read_step(T2).get('seconds')} s, T3 {steps.read_step(T3).get('seconds')} s · "
                     f"{s3['segments_non_empty']}/14 segments · {s3['micro_segments_non_empty']}/120 micro-segments "
                     f"({s3['micro_segments_ge_300']} with ≥ 300 lines) · agent dictionary `{s3['agent_dictionary']}`"))
    print("missing scores (flagged, imputed with the nationality-group median):",
          {k: f"{v:.1%}" for k, v in s2["missing_share"].items()})
    print("action_original source:", s3["action_original_source"],
          f"| segment engine vs S-rules agreement {s3['engine_vs_s_rules_agreement']:.1%}")
    display(pd.read_csv(T3 / "segment_counts.csv"))"""),
    code("""if (T3 / "microsegment_counts.csv").exists():
    micro = pd.read_csv(T3 / "microsegment_counts.csv")
    heat = micro.pivot_table(index=["NATIONALITY_GROUP", "AGE_BAND"], columns="TENURE_BAND", values="lines", aggfunc="sum")
    display(heat.style.background_gradient(cmap="Blues", axis=None).format("{:,.0f}"))
    display(pd.read_csv(T3 / "crosswalk.csv"))
    display(pd.read_csv(T3 / "action_counts.csv").sort_values("lines", ascending=False).head(15))
    display(pd.read_csv(T3 / "tenure_agreement.csv"))
    pri = json.loads((T3 / "priors.json").read_text(encoding="utf-8"))
    display(pd.DataFrame({"B-10 configured": pri["configured"], "recomputed week 0": pri["recomputed_week0"],
                          "take rate week 0": pri["take_rate_week0"]}).loc[pri["order"]])"""),

    md("""## 8. V3.0: the panel and the four groups (`fsdr-panel`, `fsdr-offers`)

T1C (production code 3) · T1F (production code 4) · T2C (challenger CC-v1) · T2F (challenger CF-v1) · control.
P6 uses the documented fallbacks of the LLM agents until Phases 5-6; the universes, anchors, passes and bands are final."""),
    code("""P1, P6 = RUN / "score" / "P1_panel", RUN / "score" / "P6_offers"
if not (P1 / "summary.json").exists():
    print("No panel yet: run  uv run python run.py fsdr-panel --date", RUN_DATE)
else:
    s1 = json.loads((P1 / "summary.json").read_text(encoding="utf-8"))
    print(f"pool {s1['pool']:,} eligible lines, scale x{s1['scale_factor']}, {s1['from_sample']:,} from the provided sample; "
          f"size checks: {s1['size_problems'] or 'OK'}")
    display(pd.read_csv(P1 / "group_sizes.csv"))
    bal = pd.read_csv(P1 / "balance.csv")
    display(bal.sort_values("max_gap_points", ascending=False).head(10))
if (P6 / "summary.json").exists():
    s6 = json.loads((P6 / "summary.json").read_text(encoding="utf-8"))
    print("decisions:", s6["decisions"], "| group audit:", s6["audit_status"])
    display(pd.read_csv(P6 / "group_report.csv"))
    display(pd.read_csv(P6 / "traces.csv"))
    audit = json.loads((P6 / "audit_groups.json").read_text(encoding="utf-8"))
    display(pd.DataFrame(audit["checks"])[["id", "rule", "status", "count", "what"]])"""),

    md("""## 9. Every step folder: files and previews

This section grows by itself as the next phases add steps (T2_features, T3_segments, P3_nba, P7_code5, …).
Every CSV / Parquet output is previewed; JSON files are summarised."""),
    code("""def preview(path, n=PREVIEW_ROWS):
    if path.suffix == ".parquet":
        d = pd.read_parquet(path); print(f"  {path.name}: {d.shape[0]:,} rows x {d.shape[1]} cols"); display(d.head(n))
    elif path.name.endswith(".csv.gz"):
        d = csvstd.read_csv(path); print(f"  {path.name}: {len(d):,} rows x {d.shape[1]} cols (FSDR CSV standard)"); display(d.head(n))
    elif path.suffix == ".csv" and not (path.with_suffix(".parquet")).exists():
        d = pd.read_csv(path, nrows=n); print(f"  {path.name}: first {len(d)} rows"); display(d)
    elif path.suffix == ".json" and path.name != config.STEP_FILE:
        obj = json.loads(path.read_text(encoding="utf-8"))
        keys = list(obj)[:12] if isinstance(obj, dict) else f"list of {len(obj)}"
        print(f"  {path.name}: {keys}")

for phase in ("train", "score", "final"):
    for sd in sorted((RUN / phase).glob("*")) if (RUN / phase).is_dir() else []:
        if sd.is_dir():
            meta = steps.read_step(sd)
            display(Markdown(f"### {phase} / {sd.name} — `{meta.get('status', '?')}` ({meta.get('seconds', '?')} s)"))
            for f in sorted(sd.iterdir()):
                if f.is_file() and f.name != "cache_manifest.json":
                    preview(f)
                elif f.is_dir():
                    print(f"  {f.name}/: {len(list(f.glob('*.parquet')))} partition files (see the _preview.csv)")"""),

    md("""## 10. Final output: the file to push, per target, and the details

`final/push/input_flytxt.csv.gz` (38 columns, all groups) is written only when every check passes; `final/target1/`,
`final/target2/` hold the same rows per group plus `details_<group>.csv.gz`, the full detail behind every line."""),
    code("""FIN = RUN / "final"
if not (FIN / "summary.json").exists():
    print("No final output yet: run  uv run python run.py fsdr-run-groups --date", RUN_DATE)
else:
    sf = json.loads((FIN / "summary.json").read_text(encoding="utf-8"))
    print("push ready:", sf["push_ready"], "| rows:", sf["input_flytxt_rows"], "| checks:",
          ", ".join(f"{c['id']} {c['status']}" for c in sf["output_checks"]))
    if (FIN / "push" / "BLOCKED.txt").exists():
        print((FIN / "push" / "BLOCKED.txt").read_text(encoding="utf-8"))
    display(pd.read_csv(FIN / "final_rules_report.csv"))
    push = FIN / "push" / "input_flytxt.csv.gz"
    if push.exists():
        display(csvstd.read_csv(push).head(PREVIEW_ROWS))
    det = pd.read_parquet(FIN / "details_all.parquet")
    print(f"details: {len(det):,} lines x {det.shape[1]} columns")
    display(det.groupby(["GROUP_KEY", "DROP_RULE"]).size().rename("lines").reset_index())
    display(det.head(PREVIEW_ROWS))"""),
]


def build(path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (FSDR .venv)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    print("written:", build(Path(__file__).with_name("fsdr_results.ipynb")))
