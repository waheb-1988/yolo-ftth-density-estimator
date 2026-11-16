"""Builds the two STANDALONE step-by-step notebooks of V4.0 (run it again after editing the cells below or the engine):

    uv run python notebooks/build_step_notebooks.py

    notebooks/01_training_step_by_step.ipynb   data -> T0 validation + cache -> T2 features -> T3 segments and actions
    notebooks/02_scoring_step_by_step.ipynb    target lists -> P1 panel -> Target 2 (P6 + final, groups a and b) -> final file

Standalone: the code of every engine module the steps use is copied into the notebook (Part A, one cell per
module, `%%engine <module>`), so a notebook imports nothing from the project folder and never loads vowpalwabbit.
Both call the same functions as the terminal commands (fsdr-generate, fsdr-load, fsdr-features, fsdr-targets,
fsdr-score-target, fsdr-final-file), one step per cell, and show what each step wrote. They are stored WITHOUT outputs:
outputs hold MSISDNs (tests/test_notebook.py checks it).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
HERE = Path(__file__).resolve().parent

SETUP = '''# --- paths and helpers (do not edit) ---------------------------------------------
import json, os, sys
from datetime import date
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:      # works from notebooks/ or the repo root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from nbx.fsdr import config, csvstd, steps

pd.set_option("display.max_columns", 80, "display.width", 220, "display.max_colwidth", 60)
D = date.fromisoformat(RUN_DATE)
TRAINING, SCORING = config.training_folder(RUN_DATE), config.scoring_folder(RUN_DATE)
RUN, CACHE, OUTPUT = config.FSDR_RUNS_DIR / RUN_DATE, config.FSDR_CACHE_DIR / RUN_DATE, config.output_folder(RUN_DATE)


def show(obj, n=PREVIEW_ROWS, title=None):
    """A table (DataFrame, list of dicts, dict or CSV / Parquet / csv.gz path), first n rows."""
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        title = title or (str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        if not p.exists() or p.stat().st_size == 0:
            print(f"{title}: empty or not written")
            return
        try:
            obj = (csvstd.read_csv(p) if p.name.endswith(".csv.gz") else pd.read_parquet(p) if p.suffix == ".parquet"
                   else pd.read_csv(p) if p.suffix == ".csv" else json.loads(p.read_text(encoding="utf-8")))
        except pd.errors.EmptyDataError:
            print(f"{title}: no rows")
            return
    if isinstance(obj, dict):
        obj = pd.DataFrame([{"key": k, "value": v if not isinstance(v, (dict, list)) else json.dumps(v, default=str)[:150]}
                            for k, v in obj.items()])
    if isinstance(obj, list):
        obj = pd.DataFrame(obj)
    if title:
        display(Markdown(f"**{title}** · {len(obj):,} rows x {obj.shape[1]} columns"))
    display(obj.head(n))


def tree(folder, depth=3, skip_schemas=True):
    """The files of a folder, as a tree with sizes."""
    folder = Path(folder)
    if not folder.exists():
        print(f"{folder}: does not exist yet")
        return
    print(folder)
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if len(rel.parts) > depth or (skip_schemas and p.name.endswith(".schema.json")):
            continue
        pad = "    " * (len(rel.parts) - 1)
        print(f"{pad}{p.name}{'/' if p.is_dir() else f'  ({p.stat().st_size / 1e3:,.1f} kB)'}")


def step_status(phase_dir, name):
    meta = steps.read_step(Path(phase_dir) / name)
    print(f"{name}: {meta.get('status', 'not run')}  rows {meta.get('rows', {})}  {meta.get('seconds', '-')} s"
          + (f"  reason: {meta['reason']}" if meta.get("reason") else ""))
    return meta


print(f"run date : {RUN_DATE}")
for name, p in (("training", TRAINING), ("scoring", SCORING), ("runs", RUN), ("cache", CACHE), ("output", OUTPUT)):
    print(f"{name:9s}: {p}  ({'found' if p.exists() else 'not yet'})")
print(f"offers per line: {config.RECO_OFFERS}  (FSDR_RECO_OFFERS)")'''

# ---------------------------------------------------------------------------
# 01 training
# ---------------------------------------------------------------------------
TRAINING_CELLS = [
    md("""# 01 · Training, step by step (FSDR V4.0)

**Training** prepares the full customer base of one run date *D*. It validates the data, builds the features,
assigns the segments, and decides the two actions of every line. The scoring notebook (`02_scoring_step_by_step.ipynb`) then reads those results.

| Step | What it does | Writes |
|---|---|---|
| 0 | (optional) dummy data for tests | `data/training/<D>/`, `data/scoring/<D>/` |
| 1 | look at the two input folders (5 sub-folders each) | — |
| 2 | **T0** validation, completeness, Parquet cache | `data/runs/<D>/train/T0_load/`, `data/cache/<D>/` |
| 3 | look at the cache (16 partitions per table) | — |
| 4 | **T2** features B-01 … B-12 for the whole base | `data/runs/<D>/train/T2_features/` |
| 5 | **T3** S-rule segments, bands, micro-segments, week-0 history, `action_original` + `action_agent` | `data/runs/<D>/train/T3_segments/` |
| 6 | summary of the training steps | — |

**Folders of a run date (V4.0):**

```
data/training/<D>/   usage/ models/ subscription/ catalog/ others/     the base, history, model scores
data/scoring/<D>/    usage/ models/ subscription/ catalog/ others/     this week's scoring tables
                     target2/a_constraint/ target2/b_free/                the target lists (Target 2 only)
```

T0 reads the two folders **as one run**. A table filed on the wrong side is still loaded, with a warning.

**How to use:** set the run date in the next cell, then run the cells one by one (Shift+Enter). Read the explanation above each step.
Every step is **resumable**: a step already done with the same inputs is skipped, and `FORCE = True` re-runs it."""),
    code("""# --- settings -------------------------------------------------------------------
RUN_DATE = "2026-11-19"     # the run date D of your real exports (a Thursday in production)
GENERATE = False            # False: your real exports. True: dummy data for a test (real exports are never overwritten)
N_LINES = 3000              # dummy lines (2000 = 30 s; 1,000,000 = the production size, ~16 min to generate)
WORKERS = 2                 # partitions in parallel for T2 / T3 (2 on a laptop, 8 on a big machine)
FORCE = False               # True: re-run every step even if its inputs did not change
PREVIEW_ROWS = 8
""" + SETUP),

    md("""## Step 0 · Dummy data (optional)

The generator writes the 47 tables of run date *D* as `.csv.gz`, the FSDR CSV standard: gzip, UTF-8, IDs kept as text.

- The training tables (base, usage history, model scores) go to `data/training/<D>/`.
- The scoring tables of the week go to `data/scoring/<D>/`: `subscriptions_tg`, `usage_data_voice_payg_tg`, `sub_btl_3months`, `targeting_auto`, the catalogue, the scripts, the conflicts, consent, and the `*_score` tables.
- `btl_catalog` is written to both sides.

The manifest says `source: generator`, so real exports are never overwritten: with real files in the folder and `GENERATE = True`, this cell stops. With real data, set `GENERATE = False` and copy your files into the folders (see **How to add new data** above)."""),
    code("""if GENERATE:
    from nbx.fsdr.synth import generator

    man = TRAINING / config.MANIFEST_FILE
    if TRAINING.exists() and any(TRAINING.iterdir()) and (not man.exists() or json.loads(man.read_text())["source"] != "generator"):
        raise SystemExit(f"{TRAINING} holds real data: set GENERATE = False")
    run = generator.generate(N_LINES, D, ["baseline"], config.GENERATOR_SEED, TRAINING, "csv.gz", "split")
    print(f"{len(run.tables)} tables generated for {N_LINES:,} lines")
else:
    if not any(p.is_file() for p in TRAINING.rglob("*.csv*")) if TRAINING.exists() else True:
        raise SystemExit(f"no exports in {TRAINING}: create the folders with `uv run python run.py fsdr-folders "
                         f"--date {RUN_DATE} --layout split`, copy the files, then run again (or GENERATE = True for a test)")
    print("GENERATE = False: using the files already in", TRAINING, "and", SCORING)"""),

    md("""## Step 1 · The two input folders

Every table lives in one of the 5 sub-folders. Its side (training or scoring) is set by `config.SCORING_TABLES`.
The table below lists, for each table of the data contract:
- the side and folder it was found in;
- its format and size;
- its row count from the manifest.

**Check:** `misfiled` should be empty, and so should `duplicates` (two files for one table block T0)."""),
    code("""from nbx.fsdr import schemas
from nbx.fsdr.datasource import JsonSource

src = JsonSource(TRAINING)
man = src.manifest()
rows = []
for t, s in schemas.TABLES.items():
    p = src.path(t)
    rows.append({"side": src.side_of.get(t, "-"), "folder": s.folder, "table": t, "present": p is not None,
                 "expected side": config.table_side(t), "MB": round(p.stat().st_size / 1e6, 3) if p else 0,
                 "rows (manifest)": (man.get("row_counts") or {}).get(t)})
inputs = pd.DataFrame(rows).sort_values(["side", "folder", "table"])
print(f"twins: {', '.join(f'{k} = {v}' for k, v in src.twins.items())}")
print(f"tables found: {inputs.present.sum()} / {len(inputs)} | misfiled: {src.misfiled() or 'none'} | "
      f"duplicates: {src.duplicate_files or 'none'}")
display(inputs.groupby(["side", "folder"]).agg(tables=("table", "count"), MB=("MB", "sum")).round(2))
show(inputs, n=60)"""),

    md("""## Step 2 · T0: validation, completeness, Parquet cache · Agent 1 (①) data agent

T0 runs before anything else and does three things.

1. **Validation** against the data contract: columns, types, keys, value ranges, date windows, the `.csv.gz` format and the folders.
   - A *blocking* issue stops the run.
   - A *warning* is reported and the run continues.
2. **Completeness**: are the tables the pipeline needs all present?
   - T0 chooses the mode from the files. **Week 0** (`train_week0`): `activities_w1_auto`, `subs_amt_auto` and `targeting_dataiku_auto` are present, and the features come from the `*_auto` history.
   - **Every next week** (`train_weekly`): no `*_auto` history; the features come from this week's `*_score` tables. The missing history tables are then warnings ("not needed by the train_weekly pipeline"), not blocking issues.
   - A missing optional table (sample, consent) falls back to a documented default.
3. **Cache**: every table is written to Parquet in 16 partitions, by a hash of `ACCESS_METHOD_ID`.
   - Every later step reads this cache, never the `.csv.gz` again, so 1M lines fit in memory partition by partition.

Outputs: `validation_report.md / .json`, `completeness.csv`, `cache_manifest.json`, `summary.json`."""),
    code("""from nbx.fsdr.contract import data_agent

t0 = data_agent.run_t0(TRAINING, run_date=D, force=FORCE)
print(f"T0_load {t0['status'].upper()}{' (skipped: same inputs)' if t0.get('skipped') else ''} -- pipeline {t0['pipeline']}, "
      f"{t0['tables']} tables, {t0['rows_in']:,} rows, validation {t0['validation']} {t0['issue_counts']}")
print("read as one run:", t0.get("training_source"), "+", t0.get("scoring_source"))
print()
print(t0["explanation"])
if t0["pipeline"] == "train_weekly":
    print()
    print("weekly run: features from this week's *_score tables; the agent dictionary comes from the previous run")"""),
    code("""T0 = RUN / "train" / "T0_load"
rep = json.loads((T0 / "validation_report.json").read_text(encoding="utf-8"))
issues = pd.DataFrame(rep["issues"])
if len(issues):
    display(issues.groupby(["level", "check"]).size().rename("issues").reset_index())
    show(issues[["level", "check", "table", "column", "message"]], n=20, title="validation issues")
else:
    print("validation: no issue")
show(T0 / "completeness.csv", n=50)"""),

    md("""## Step 3 · The Parquet cache

Each table is split into 16 partitions. A line is always in the same partition across tables, so T2 / T3 can
join the tables of one partition without reading the whole base. **Check:** `verify_cache` returns no problem."""),
    code("""from nbx.fsdr.contract.cache import CacheSource, verify_cache

cache = CacheSource(D)
cm = cache.manifest()
tables = pd.DataFrame([{"table": t, **{k: v for k, v in m.items() if k in ("rows", "partitioned", "bytes")}}
                       for t, m in cm["tables"].items()])
print(f"cache {CACHE}: {cm['partitions']} partitions, {cm['bytes'] / 1e6:,.1f} MB, built in {cm['seconds']} s")
print("verify_cache:", verify_cache(D) or "OK")
show(tables.sort_values("rows", ascending=False), n=50)"""),

    md("""## Step 4 · T2: features of the whole base (B-01 … B-12)

T2 works partition by partition, with `WORKERS` partitions in parallel. For every eligible line it builds:

- **Eligibility** (B-01 … B-03):
  - active prepaid lines;
  - valid 8-digit MSISDN;
  - not an internal / employee plan;
  - duplicates removed.
- **Profile**: age, tenure, nationality group (Oman, India, Bangladesh, Pakistan, others), region, device segment.
- **Usage and revenue**: weekly (w1 … w4) and monthly (m1 … m3) activity, BTL and pay-as-you-go spend, the trends between periods.
- **Model scores** (B-05): the 11 model tables (churn, appetency, …). A missing score gets the median of the line's nationality group.

The anchor decides which tables are read: week 0 reads the `*_auto` tables (`anchor = T`), a weekly run reads the `*_score` tables.

Outputs:
- `base_features/part-NN.parquet` (1 file per partition);
- `feature_summary.csv` (count, nulls, mean, min, max per feature);
- `stats.json` (the medians and the ARPU tertiles that T3 uses);
- `base_features_preview.csv`."""),
    code("""from nbx.fsdr.pipelines import training

t2 = training.run_t2(D, workers=WORKERS, force=FORCE)
print(f"T2_features DONE{' (skipped)' if t2.get('skipped') else ''}: {t2['lines']:,} eligible lines of {t2['base_rows']:,}, "
      f"{t2['features']} features, anchor {t2['anchor']} ({t2['tables']})")
show({"eligibility filters": t2["eligible_filters"], "missing model scores": t2["missing_share"],
      "age source": t2["age_source"], "tenure source": t2["tenure_source"], "ARPU tertiles": t2["arpu_tertiles"],
      "msisdn in two partitions (must be 0)": t2["msisdn_in_two_partitions"]}, title="T2 summary")"""),
    code("""T2 = RUN / "train" / "T2_features"
fs = pd.read_csv(T2 / "feature_summary.csv")
fs["null_share"] = (fs["nulls"] / fs["count"].where(fs["count"] > 0)).round(4)
show(fs.sort_values("null_share", ascending=False), n=15, title="features with the most missing values")
stats = json.loads((T2 / "stats.json").read_text(encoding="utf-8"))
show(pd.DataFrame(stats["medians_by_nationality"]).T.round(4), n=10, title="B-05 medians by nationality (missing scores)")
show(T2 / "base_features_preview.csv", n=5)"""),

    md("""## Step 5 · T3: segments, micro-segments and the two actions · Agents 2 (②) segmentation and 2b (②b) agent dictionary

For every line of T2:

- **Segment**: the S-rules (in order, the first that matches wins) give one of the 14 segments (Retention, RGS, Onboarding, Nursering).
  - The Nursering lines are **excluded** (B-02): they are never contacted.
- **Bands**:
  - age (young / middle / senior);
  - tenure;
  - value band (ARPU tertiles from T2);
  - usage type (voice / data / both).
- **Micro-segment** = nationality | age band | tenure band. It is used for the balance of the panel and the median anchor.
- **Week-0 history**: what the production engine targeted last week (`W0_*`), and who took an offer.
- **Two actions per line** (one of the 7 actions):
  - `ACTION_ORIGINAL`: the production dictionary (segment -> action), kept for reference (Target 1 is not scored in V4.0).
  - `ACTION_AGENT`: the agent dictionary, versioned and checked against the previous version. Target 2 uses it.

Outputs:
- `segments/part-NN.parquet`;
- `segment_counts.csv`, `microsegment_counts.csv`, `action_counts.csv`, `crosswalk.csv`;
- `priors.json` (B-10 action priors) and the agent dictionary `.json`."""),
    code("""t3 = training.run_t3(D, workers=WORKERS, force=FORCE)
print(f"T3_segments DONE{' (skipped)' if t3.get('skipped') else ''}: {t3['lines']:,} lines, {t3['targetable']:,} targetable, "
      f"{t3['excluded_nursering']:,} Nursering excluded; {t3['segments_non_empty']} segments, "
      f"{t3['micro_segments_non_empty']} / {t3['micro_segments_total']} micro-segments; agent dictionary {t3['agent_dictionary']}")
T3 = RUN / "train" / "T3_segments"
seg = pd.read_csv(T3 / "segment_counts.csv")
show(seg, n=20)
ax = seg.set_index("SEGMENT")["lines"].sort_values().plot.barh(figsize=(8, 5), title="lines per segment")
plt.tight_layout(); plt.show()"""),
    code("""acts = pd.read_csv(T3 / "action_counts.csv")
pivot = acts.pivot_table(index="ACTION_ORIGINAL", columns="ACTION_AGENT", values="lines", aggfunc="sum", fill_value=0)
display(Markdown(f"**action_original (rows) x action_agent (columns)** · same action on "
                 f"{t3['action_original_eq_agent']:.1%} of the lines"))
display(pivot)
show(T3 / "microsegment_counts.csv", n=10)
show(json.loads((T3 / "priors.json").read_text(encoding="utf-8")), title="B-10 action priors")
show(T3 / "segments_preview.csv", n=5)"""),

    md("""## Step 6 · Training summary

Every step folder holds a `_step.json` with its status (done / blocked / failed), its rows, its duration and its
outputs. When all three steps are **done**, open `02_scoring_step_by_step.ipynb`."""),
    code("""rows = [{"step": s["step"], "status": s["status"], "seconds": s["seconds"], "rows": s["rows"], "files": len(s["files"])}
        for s in steps.list_steps(RUN) if s["phase"] == "train"]
show(rows, n=20, title=f"training steps of {RUN_DATE}")
tree(RUN / "train", depth=1)"""),
]

# ---------------------------------------------------------------------------
# 02 scoring
# ---------------------------------------------------------------------------
TARGET_P6 = '''TARGET = "{t}"
p6 = scoring_groups.run_p6_offers(D, force=FORCE, target=TARGET)
P6 = RUN / "score" / f"P6_offers_{{TARGET}}"
print(f"{{p6['stage']}} DONE{{' (skipped)' if p6.get('skipped') else ''}} -- group audit {{p6['audit_status']}}"
      f"{{' (FINAL_FILE BLOCKED)' if p6['blocked'] else ''}}, up to {{p6['reco_offers']}} offers per line")
show(pd.read_csv(P6 / "group_report.csv"), title="per group")
audit = json.loads((P6 / "audit_groups.json").read_text(encoding="utf-8"))
show([{{k: c[k] for k in ("id", "rule", "status", "count", "what")}} for c in audit["checks"]], n=20, title="group audit")'''

TARGET_P6_DETAIL = '''monthly = csvstd.read_csv(P6 / f"{{config.GROUPS['{a}']['recos_monthly']}}{{csvstd.SUFFIX}}")
display(Markdown("**How the offers were chosen** (`selected_by`: B-28/B-29/B-30 code 3 passes, FR-09/10/11 code 4, "
                 "CC-04 / CF-04 challenger band, EXT-6 = the third offer of V4.0)"))
sel = pd.concat([csvstd.read_csv(P6 / f"{{config.GROUPS[k][f'recos_{{v}}']}}{{csvstd.SUFFIX}}").assign(group=k, validity=v)
                 for k in config.TARGETS[TARGET] for v in ("monthly", "weekly")])
display(sel.pivot_table(index="selected_by", columns=["group", "validity"], values="access_method_val",
                        aggfunc="count", fill_value=0))
per_line = sel.groupby(["group", "validity", "access_method_val"]).size().rename("offers").reset_index()
display(per_line.groupby(["group", "validity"])["offers"].value_counts().unstack(fill_value=0))
show(monthly, title="recos_monthly of {a}")
show(P6 / "traces.csv", n=6)'''

TARGET_FINAL = '''fin = scoring_groups.run_target_final(D, TARGET, force=FORCE)
OUT_T = OUTPUT / TARGET
print(f"{{TARGET}} final {{'READY' if fin['ready'] else 'BLOCKED (see BLOCKED.txt)'}} -- input_flytxt_{{TARGET}} "
      f"{{fin['input_flytxt_rows']:,}} rows; details {{fin['details_rows']:,}} lines x {{fin['details_columns']}} columns")
show(fin["output_checks"], title="output checks A-20 ... A-24")
show(fin["per_group"], title="per group a / b")
tree(OUT_T, depth=2)'''

TARGET_DETAILS = '''det = pd.read_parquet(OUT_T / f"details_{{TARGET}}.parquet")
reasons = pd.concat([det[f"REASON_{{i}}"].fillna("(empty)").value_counts().rename(f"slot {{i}}") for i in range(1, 7)], axis=1)
display(Markdown("**What fills each slot** (V4.0: 5 = 3rd monthly, 6 = 3rd weekly; a favourite or a forced offer only "
                 "fills a slot 5 / 6 left empty)"))
display(reasons.fillna(0).astype(int))
flytxt = csvstd.read_csv(OUT_T / f"input_flytxt_{{TARGET}}{{csvstd.SUFFIX}}") if fin["ready"] else None
if flytxt is not None:
    n = flytxt[[f"destination_offer_name_{{i}}" for i in range(1, 7)]].ne("").sum(axis=1)
    n.value_counts().sort_index().plot.bar(title=f"{{TARGET}}: offers per line in the file", figsize=(6, 3))
    plt.tight_layout(); plt.show()
one = det[det["IN_INPUT_FLYTXT"]].iloc[0]
cols = ["GROUP_KEY", "SEGMENT", "ACTION_ORIGINAL", "ACTION_AGENT", "ACTION_SENT", "B26_AVG_REV_MONTHLY", "F05_MONTHLY"]
cols += [c for i in range(1, 7) for c in (f"SLOT_{{i}}", f"REASON_{{i}}", f"REMOVED_{{i}}")]
display(Markdown(f"**One line, end to end** (`details_{{TARGET}}.parquet` has {{det.shape[1]}} columns per line)"))
display(one[[c for c in cols if c in one.index]].to_frame("value"))'''

SCORING_CELLS = [
    md("""# 02 · Scoring of Target 2, step by step (FSDR V4.0)

Scoring runs after training (`01_training_step_by_step.ipynb`) on the **same run date**: the whole base is trained,
then the target lists are joined with it and **Target 2** is scored, group a and group b, into `data/output/<D>/`.
It then saves the final file (`input_flytxt.csv.gz`, the campaign format). Nothing is sent anywhere.

**V4.0: Target 2 only** (the target where the LLM agents decide, notebooks 03 / 04). Target 1 (the production rules)
is not scored: its lists are not read and its folders are not written (`FSDR_TARGETS=target2`, the default;
`FSDR_TARGETS=target1,target2` brings it back).

| Group | Name | Lines | Universe of offers | Action | Rules |
|---|---|---|---|---|---|
| **a** `a_constraint` | **T2C** = Target 2, Constraint | 50,000 | the offer families of the customer's action (code 3 universe) | `action_agent` (the agent dictionary) | CC-v1: price 1.0–1.5x the customer's own spend |
| **b** `b_free` | **T2F** = Target 2, Free | 50,000 | L2 DATA / COMBO offers, no voice, whatever the action (code 4 universe) | not used | CF-v1: price 1.0–1.5x the customer's own spend |
| control | — | 0 | no control group by default (`FSDR_CONTROL_SIZE=0`); set `FSDR_CONTROL_SIZE=20000` and add `control/control.csv.gz` to measure the uplift against never-contacted lines | — | — |

| Step | What it does | Writes |
|---|---|---|
| 1 | check that training is done | — |
| 2 | the target lists (upload, or dummy lists), their checks | `data/scoring/<D>/target2/…` |
| 3 | **P1** panel from the lists, sizes, balance | `data/runs/<D>/score/P1_panel/` |
| 4 | **Target 2** P6: the offers of T2C and T2F | `data/runs/<D>/score/P6_offers_target2/` |
| 5 | **Target 2** final files (code 5) | `data/output/<D>/target2/` |
| 6 | the **final file** (Target 2 PASS) | `data/output/<D>/final/` |
| 7 | validate the final file | — |

**Up to 6 offers per line (V4.0).** Slots 1–2 are the 2 best monthly offers and slots 3–4 the 2 best weekly ones. Slot 5 is the 3rd monthly offer and slot 6 the 3rd weekly one (rule EXT-6: same universe, above the anchor, not the same service and price). Favourites and forced bundles only fill a slot 5 / 6 left empty. Set `FSDR_RECO_OFFERS=4` in `.env` for the V3.0 file (4 recommended offers).

Decisions use the documented fallbacks of the LLM agents (rule action, NBO order E-O3 + E-O1) until Phases 5–6 build them, and every row says so (`DECIDED_BY`, `score_source`)."""),
    code("""# --- settings -------------------------------------------------------------------
RUN_DATE = "2026-11-19"         # the same run date as the training notebook
GENERATE_TARGETS = False        # False: your real lists. True: dummy lists for a test (real lists are never overwritten)
FORCE = True                    # scoring is quick: re-run every step by default
PREVIEW_ROWS = 8
""" + SETUP + """

from nbx.fsdr import targets
from nbx.fsdr.pipelines import scoring_groups"""),

    md("""## Step 1 · Training must be done

P1 reads the T3 segments and the T2 features of the same run date. If a step below is not **done**, run the training notebook first."""),
    code("""for name in ("T0_load", "T2_features", "T3_segments"):
    meta = step_status(RUN / "train", name)
    assert meta.get("status") == "done", f"{name} is not done: run 01_training_step_by_step.ipynb first"
print("training OK")"""),

    md("""## Step 2 · The target lists

You get one list per group, chosen outside the engine. Each list has one row per line and at least `access_method_val`; any other column is kept and carried to the details.

```
data/scoring/<D>/target2/a_constraint/target2_a.csv.gz    T2C (group a), 50,000 lines
data/scoring/<D>/target2/b_free/target2_b.csv.gz          T2F (group b), 50,000 lines
```

**Checks:**
- **Blocking**:
  - a line in two groups;
  - a missing list;
  - no `access_method_val` column.
- **Dropped**, with the reason in `targets_rejected.csv`:
  - not an 8-digit MSISDN;
  - not in the eligible base;
  - Nursering;
  - in another campaign this week.
- **Warning**: a size more than 2 % off the configured size."""),
    code("""from nbx.fsdr.contract.cache import CacheSource

folder = scoring_groups.input_folder(D)
if GENERATE_TARGETS and scoring_groups.real_targets(folder):
    print("GENERATE_TARGETS = True, but the folder holds YOUR lists (not dummy lists): they are kept and used.",
          "Set GENERATE_TARGETS = False to remove this message.")
elif GENERATE_TARGETS:
    gen = scoring_groups.generate_targets(D)
    print(f"dummy lists written (scaled x{gen['scale_factor']} to the base): {gen['sizes']}")
print("lists are", "DUMMY (generator)" if targets.dummy_lists(folder) else "YOURS (checked against the configured sizes)")
print("lists:", {k: str(targets.group_file(folder, k).relative_to(folder)) if targets.group_file(folder, k) else "MISSING"
                 for k in config.TARGET_FILES})
seg = scoring_groups.base_segments(D)
t, issues = targets.read_targets(folder)
kept, rejected, more = targets.check_targets(t, seg, scoring_groups.other_campaigns(CacheSource(D), D), targets.expected_sizes(folder))
print(f"{len(t):,} lines read, {len(kept):,} kept, {len(rejected):,} dropped")
show(issues + more or [{"level": "-", "message": "no issue"}], title="checks")
show(kept.groupby("GROUP_KEY").size().rename("lines").reset_index(), title="kept per group")"""),

    md("""## Step 3 · P1 panel

The panel is the uploaded lists, joined with the T3 segment columns of each line. Without lists, it is a stratified random draw.

**Balance (A-31):** the strata shares should be within 2 points between the groups (only significant gaps count).
**Sizes (A-30):** within ±2 % of the configured sizes (25k per group + 20k control in production)."""),
    code("""p1 = scoring_groups.run_p1_panel(D, force=FORCE)
print(f"P1_panel {p1['status'].upper()} -- mode {p1['mode']}, panel {p1['panel']:,} lines, sizes {p1['sizes']}, "
      f"max balance gap {p1['max_balance_gap_points']} points")
assert p1["status"] == "done", "fix the target lists (see the checks above)"
P1 = RUN / "score" / "P1_panel"
show(P1 / "group_sizes.csv")
bal = pd.read_csv(P1 / "balance.csv")
show(bal.sort_values("max_gap_points", ascending=False), n=10, title="balance: largest gaps between groups")"""),

    md("""## Step 4 · Target 2, scoring (P6): T2C (a) and T2F (b) · Agents 3 (③), 4 (④), 6B (⑥B) challenger rules, 9 (⑨) auditor

**The difference between a and b:** group a (T2C) gets offers of the families of its **action** (a retention action
gets retention offers, an upsell action bigger bundles); group b (T2F) gets the best L2 DATA / COMBO offers for its
spend, **whatever the action**. Both use the customer's own spend as the price anchor.

These rules apply to the two groups:

1. **Action** = `action_agent`, the agent dictionary.
2. **Challenger anchor** (CC-03 / CF-03): the customer's spend, split data / voice. A customer with no spend gets the median of his micro-segment.
3. **Relative band** (CC-02 / CF-02): price between 1.0× and 1.5× the anchor. UPSELL actions need at least +0.5 OMR, and no offer is ever a downgrade.
   - The pick is the best `score × price` (C-04).
   - If nothing fits, the band is widened once, to 2.0×.
   - Then the production (code 3 / code 4) passes apply as a fallback (logged as `fallback …`).
4. **EXT-6**: the third offer per validity.
5. **10.1 consistency**: T2C only."""),
    code(TARGET_P6.format(t="target2")),
    code(TARGET_P6_DETAIL.format(a="t2c")),

    md("""## Step 5 · Target 2, final files (code 5) → `data/output/<D>/target2/` · Agent 7 (⑦) code 5

Code 5 turns the recommended offers into the campaign file `input_flytxt`, which has 38 columns:

1. Slots 1–4 come from the price rank of the monthly and weekly offers (F-02 / F-03), and slots 5–6 from the 3rd monthly and the 3rd weekly offer.
2. A line with no recommended offer is dropped (F-04).
3. Favourites (F-06) and forced bundles (F-08) only fill a slot 5 / 6 left empty.
4. Slots are cleared for:
   - the same type and price (F-09);
   - the same offer (F-11);
   - conflicting offers (F-12).
5. Scripts in English and Arabic, segment, BTL-taker flags, dates (F-13 … F-19).

The **output checks** A-20 … A-24 run on the file: 38 columns in order, one row per MSISDN, no control line, the dates, every offer in the catalogue, and the segment being one of the 7 actions.
`input_flytxt_target2.csv.gz` is written **only when everything passes** (H-08); otherwise you get `BLOCKED.txt` with the reasons.

| File | Content |
|---|---|
| `input_flytxt_target2.csv.gz` | the file of Target 2 (both groups) |
| `a_constraint/input_flytxt_t2c.csv.gz`, `b_free/input_flytxt_t2f.csv.gz` | the same, per group |
| `a_constraint/details_t2c.csv.gz`, `b_free/details_t2f.csv.gz` | every line of the group with every column |
| `details_target2.parquet` | the big file to validate: profile, segment, both actions, features, anchors, offers, slots, reasons, removals |
| `final_rules_report.csv`, `output_checks.json`, `audit_groups.json`, `summary.json` | the checks and counts |"""),
    code(TARGET_FINAL.format()),
    code(TARGET_DETAILS.format()),

    md("""## Step 6 · The final file · Agent 9 (⑨) auditor: no final file unless Target 2 PASSES

- Control: no control group by default (`FSDR_CONTROL_SIZE=0`); set `FSDR_CONTROL_SIZE=20000` and add `control/control.csv.gz` to measure the uplift against never-contacted lines (then `data/output/<D>/control/` holds them, never contacted).
- `data/output/<D>/final/input_flytxt.csv.gz` holds the Target 2 file (groups a and b). It is written **only if Target 2 passed** and no MSISDN is in two groups (A-30); otherwise you get `final/BLOCKED.txt`.

The engine only **saves** these files in the campaign format (38 columns, `.csv.gz`); it sends nothing. You hand
`final/input_flytxt.csv.gz` over yourself, once you have checked it (Step 7)."""),
    code("""ctrl = scoring_groups.write_control(D)
print(f"control: {ctrl['lines']:,} lines -> {ctrl['file']}" if ctrl["file"] else "control: none (FSDR_CONTROL_SIZE=0)")
fin = scoring_groups.run_final_file(D)
print(f"final file {'READY' if fin['ready'] else 'BLOCKED'} -- {fin['input_flytxt_rows']:,} rows {fin['rows_per_target']}"
      + (f" -- reasons: {fin['reasons']}" if fin["reasons"] else ""))
tree(OUTPUT, depth=3)"""),

    md("""## Step 7 · Validate the final file

These are the last checks before handing the file over:
- the 38 columns in order;
- one row per MSISDN, as 8-digit text;
- no control line in the file;
- every line found in the details with `IN_INPUT_FLYTXT = True`;
- offers per line between 1 and 6."""),
    code("""final_file = OUTPUT / "final" / f"input_flytxt{csvstd.SUFFIX}"
if not final_file.exists():
    blocked = OUTPUT / "final" / "BLOCKED.txt"
    print("NO FINAL FILE: it is blocked. Fix the reason below, then run Steps 2-6 again.")
    print()
    print(blocked.read_text(encoding="utf-8") if blocked.exists() else "final/ is empty: run Step 8 first")
    for t in config.TARGETS:
        s = OUTPUT / t / "summary.json"
        if s.exists():
            r = json.loads(s.read_text(encoding="utf-8"))
            print(f"{t}: {'READY' if r.get('ready') else 'BLOCKED'}, group audit {r.get('p6_group_audit')}, "
                  f"failed checks {[c['id'] for c in r.get('output_checks', []) if c.get('status') != 'PASS']}")
        if (OUTPUT / t / "BLOCKED.txt").exists():
            print("   ", (OUTPUT / t / "BLOCKED.txt").read_text(encoding="utf-8").strip().replace("\\n", "\\n    "))
else:
    f = csvstd.read_csv(final_file)
    details = pd.concat([pd.read_parquet(OUTPUT / t / f"details_{t}.parquet") for t in config.TARGETS])
    cfile = OUTPUT / "control" / f"control_{RUN_DATE}{csvstd.SUFFIX}"
    control = csvstd.read_csv(cfile) if cfile.exists() else pd.DataFrame({"ACCESS_METHOD_VAL": []})   # no control group
    n = f[[f"destination_offer_name_{i}" for i in range(1, 7)]].ne("").sum(axis=1)
    checks = {
        "38 columns in order": list(f.columns) == config.INPUT_FLYTXT_COLUMNS,
        "one row per msisdn": bool(f["msisdn"].is_unique),
        "8-digit text msisdn": bool(f["msisdn"].astype(str).str.fullmatch(r"\\d{8}").all()),
        "no control line": not (set(f["msisdn"]) & set(control["ACCESS_METHOD_VAL"])),
        "every line in the details": set(f["msisdn"]) == set(details.loc[details["IN_INPUT_FLYTXT"], "ACCESS_METHOD_VAL"]),
        "offers per line 1..6 (recommended + favourites / forced)": bool(n.between(1, 6).all()),
    }
    show([{"check": k, "status": "PASS" if v else "FAIL"} for k, v in checks.items()], title="final file")
    print(f"{len(f):,} lines, {n.mean():.2f} offers per line on average; segment (action) mix:")
    display(f["segment"].value_counts().to_frame("lines"))
    show(f, n=5, title="input_flytxt.csv.gz (first rows)")"""),
]

NOTEBOOKS = {"01_training_step_by_step.ipynb": TRAINING_CELLS, "02_scoring_step_by_step.ipynb": SCORING_CELLS}


# ---------------------------------------------------------------------------
# Standalone notebooks (V4.0): the engine code is copied INTO the notebook (Part A), nothing is imported from the
# project folder, and vowpalwabbit is never loaded (it crashed Jupyter kernels on Windows: msvcp140.dll conflict).
# ---------------------------------------------------------------------------
import ast  # noqa: E402
import re  # noqa: E402

ROOT = HERE.parent
ROOTS = {"01_training_step_by_step.ipynb": ["nbx.fsdr.synth.generator", "nbx.fsdr.datasource", "nbx.fsdr.schemas",
                                            "nbx.fsdr.contract.data_agent", "nbx.fsdr.contract.cache",
                                            "nbx.fsdr.pipelines.training", "nbx.fsdr.steps", "nbx.fsdr.csvstd"],
         "02_scoring_step_by_step.ipynb": ["nbx.fsdr.pipelines.scoring_groups", "nbx.fsdr.targets",
                                           "nbx.fsdr.contract.cache", "nbx.fsdr.steps", "nbx.fsdr.csvstd"]}
# the V1.0 bandit modules: only reached by V1.0 code paths the V4.0 steps never call (a call would fail loudly)
EXCLUDED = {"nbx.core.vw_format", "nbx.fsdr.bandit", "nbx.fsdr.registry"}
FORBIDDEN = {"vowpalwabbit", "implicit", "gradio", "fastapi", "uvicorn", "httpx", "langsmith", "openai", "anthropic"}
# the package root of the notebook: telemetry off, and NO vowpalwabbit (the V1.0 bandit is not part of V4.0)
NBX_INIT = '''"""NBX engine, standalone in this notebook (V4.0 training / scoring). vowpalwabbit is never loaded."""
import os

for _k, _v in {"GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
               "VLLM_NO_USAGE_STATS": "1", "ANONYMIZED_TELEMETRY": "False"}.items():
    os.environ.setdefault(_k, _v)

__version__ = "4.0.0"
'''

LOADER = '''# --- Part A loader: the cells below hold the engine code; %%engine registers each one as a module -------------
# Nothing is imported from the project folder: a cell "%%engine nbx.fsdr.csvstd" runs ITS OWN code and makes it
# available to the other cells under that name. Only pip packages are imported (pandas, numpy, pyarrow, pyyaml,
# python-dotenv, matplotlib). vowpalwabbit is never loaded.
import linecache, os, sys, types
from pathlib import Path

from IPython.core.magic import register_cell_magic

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:      # the project folder (for data/ and .env)
    ROOT = ROOT.parent
os.chdir(ROOT)
for _name in [m for m in sys.modules if m == "nbx" or m.startswith("nbx.")]:
    del sys.modules[_name]                                                  # a re-run starts clean


def _module(name, package=False):
    mod = types.ModuleType(name)
    mod.__fsdr_notebook__ = True                                            # proof: defined in this notebook
    if package:
        mod.__path__ = []          # its sub-modules are cells of this notebook, never files on disk
    mod.__file__ = str(ROOT / (name.replace(".", "/") + ("/__init__.py" if package else ".py")))  # paths only
    mod.__package__ = name if package else name.rpartition(".")[0]
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_package(parent), child, mod)
    return mod


def _package(name):
    m = sys.modules.get(name)
    return m if getattr(m, "__fsdr_notebook__", False) else _module(name, package=True)


@register_cell_magic
def engine(line, cell):
    """%%engine <module> [package]: run this cell's code as that module."""
    name, *flags = line.split()
    mod = _package(name) if "package" in flags else _module(name)
    filename = f"<notebook: {name}>"
    linecache.cache[filename] = (len(cell), None, cell.splitlines(True), filename)   # readable tracebacks
    exec(compile(cell, filename, "exec"), mod.__dict__)


def use(module, *names):
    """The objects of an engine module (defined by a cell below), e.g. config, csvstd = use("nbx.fsdr", "config", "csvstd")."""
    mod = sys.modules[module]
    out = [getattr(mod, n) if hasattr(mod, n) else sys.modules[f"{module}.{n}"] for n in names]
    return out[0] if len(out) == 1 else tuple(out)


print("engine loader ready: the next cells define the engine modules")'''

PROOF = '''# --- proof: everything ran from this notebook ------------------------------------------------------------
mods = sorted(m for m in sys.modules if m == "nbx" or m.startswith("nbx."))
from_disk = [m for m in mods if not getattr(sys.modules[m], "__fsdr_notebook__", False)]
print(f"{len(mods)} engine modules, all defined in this notebook: {not from_disk}")
print("vowpalwabbit loaded:", "vowpalwabbit" in sys.modules)
assert not from_disk, f"loaded from the project folder: {from_disk}"
assert "vowpalwabbit" not in sys.modules'''


def _file(mod: str) -> Path:
    p = ROOT / mod.replace(".", "/")
    return p / "__init__.py" if p.is_dir() else p.with_suffix(".py")


def _resolve(mod: str, name: str | None = None) -> str:
    """'from nbx.fsdr import config' -> nbx.fsdr.config (a module file) or nbx.fsdr (an attribute of the package)."""
    if name and (ROOT / (mod + "." + name).replace(".", "/")).with_suffix(".py").exists():
        return f"{mod}.{name}"
    if name and (ROOT / (mod + "." + name).replace(".", "/")).is_dir():
        return f"{mod}.{name}"
    return mod


def _deps(mod: str, top_only: bool) -> set[str]:
    tree = ast.parse(_file(mod).read_text(encoding="utf-8"))
    nodes = tree.body if top_only else list(ast.walk(tree))
    out: set[str] = set()
    for n in nodes:
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == "nbx":
            out |= {_resolve(n.module, a.name) for a in n.names}
        elif isinstance(n, ast.Import):
            out |= {_resolve(a.name) for a in n.names if a.name.split(".")[0] == "nbx"}
    parts = mod.split(".")
    out |= {".".join(parts[:i]) for i in range(1, len(parts))}               # the parent packages run first
    return out - {mod}


def _third_party(mod: str) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(_file(mod).read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


def engine_modules(roots: list[str]) -> list[str]:
    """Every engine module the roots need (lazy imports included), parents and dependencies first."""
    need, todo = set(), list(roots)
    while todo:
        m = todo.pop()
        if m not in need and m not in EXCLUDED:
            need.add(m)
            todo += sorted(_deps(m, top_only=False))
    bad = {m: _third_party(m) & FORBIDDEN for m in need if m != "nbx" and _third_party(m) & FORBIDDEN}
    if bad:
        raise SystemExit(f"these engine modules import forbidden packages: {bad}")
    order: list[str] = []

    def visit(m, stack=()):
        if m in order:
            return
        for d in sorted(_deps(m, top_only=True) & need):
            if d not in stack:
                visit(d, (*stack, m))
        order.append(m)
    for m in sorted(need):
        visit(m)
    return order


_IMPORT = re.compile(r"^(?P<ind>\s*)from (?P<mod>nbx[\w.]*) import (?P<names>[\w, ]+)$", re.M)


def _use_imports(src: str) -> str:
    """Step cells: 'from nbx.fsdr import config, csvstd' -> 'config, csvstd = use("nbx.fsdr", "config", "csvstd")'."""
    def sub(m):
        names = [n.strip() for n in m["names"].split(",") if n.strip()]
        return f'{m["ind"]}{", ".join(names)} = use("{m["mod"]}", {", ".join(repr(n) for n in names)})'.replace("'", '"')
    return _IMPORT.sub(sub, src)


def _engine_cells(mods: list[str]) -> list:
    rows = []
    for m in mods:
        src = _file(m).read_text(encoding="utf-8")
        doc = (ast.get_docstring(ast.parse(src)) or "").strip().split("\n")[0]
        rows.append((m, str(_file(m).relative_to(ROOT)).replace("\\", "/"), len(src.splitlines()), doc))
    table = "\n".join(f"| `{m}` | `{f}` | {n} | {d.replace('|', '/')} |" for m, f, n, d in rows)
    cells = [md(f"""## Part A · The engine code ({len(mods)} modules, {sum(r[2] for r in rows):,} lines)

This notebook is **standalone**: every function the steps call is defined in the cells of this part. Nothing is
imported from the project folder (`nbx/`), and the only imports are pip packages (pandas, numpy, pyarrow, pyyaml,
python-dotenv, matplotlib). **vowpalwabbit is never loaded**: it is the V1.0 bandit only, and loaded after pandas it
crashes the Jupyter kernel on Windows (two copies of `msvcp140.dll`).

Each cell starts with `%%engine <module>`: the cell runs its own code and registers it under that name, so the
cells can call each other (an `import` of `nbx.…` inside these cells resolves to the cell above, never to a file).
The code is copied from the project by `notebooks/build_step_notebooks.py`, so it is **exactly** the code of the
terminal commands. Run this part once (it takes a few seconds), then go to Part B.

T2 / T3 run in this process (`WORKERS = 1`): worker processes would have to import the code from files.

| Module | Copied from | Lines | What it does |
|---|---|---|---|
{table}"""), code(LOADER)]
    for m in mods:
        src = NBX_INIT if m == "nbx" else _file(m).read_text(encoding="utf-8")
        is_pkg = _file(m).name == "__init__.py"
        cells.append(code(f"%%engine {m}{' package' if is_pkg else ''}\n" + src.rstrip() + "\n"))
    return cells


def standalone(cells: list, name: str) -> list:
    """The step cells, with the engine (Part A) inserted after the settings and the nbx imports turned into use()."""
    title, settings, *steps = cells
    head, _, helpers = settings.source.partition("# --- paths and helpers")
    head = re.sub(r"^WORKERS = .*$", "WORKERS = 1                 # standalone notebook: T2 / T3 run in this process "
                  "(the engine code lives here, not in files)", head, flags=re.M)
    helpers = "# --- paths and helpers" + helpers
    helpers = helpers.replace("sys.path.insert(0, str(ROOT))\n", "")
    part_b = md("## Part B · The steps\n\nThe engine is loaded. Each step below calls it and shows what it wrote.")
    out = [title, MACHINE_MD, code(MACHINE_CHECK), AGENTS_MD, NEW_DATA_MD, code(head.rstrip() + "\n"),
           *_engine_cells(engine_modules(ROOTS[name])), part_b,
           code(_use_imports(helpers))]
    for c in steps:
        out.append(code(_use_imports(c.source)) if c.cell_type == "code" else c)
    out += [md("## Proof: nothing came from the project folder\n\nEvery engine module was defined by a cell of "
               "this notebook, and vowpalwabbit was never loaded."), code(PROOF)]
    return out


MACHINE_MD = md("""## Your machine: Python, GPU and the local LLM

Run this cell first. It only **reads** (nothing is installed, nothing leaves the machine) and prints:
- the Python of the kernel: it must be the project's `.venv` (`…\\.venv\\Scripts\\python.exe`);
- the NVIDIA GPU of the laptop (`nvidia-smi`): name, driver, memory, how busy it is now, and which programs use it;
- the local LLM server of `.env` (`LLM_BACKEND`): is it listening?

**Who uses the GPU.** The steps of this notebook (pandas, Parquet) run on the **CPU**. The GPU is used only by the
local LLM server (vLLM, Ollama, llama.cpp or LM Studio), when an agent calls it. In V4.0 the decisions use the
agents' rule fallbacks (see the agent map below), so the GPU stays idle during these steps; that is expected.

**To see the GPU working:** start your LLM server (for example `ollama serve`), run this cell again, and look at
`programs on the GPU`. Or, in a terminal, run `nvidia-smi -l 1` (it refreshes every second) while the LLM answers:
`GPU-Util` goes up and the server process (`ollama.exe`, `python`, `llama-server`) is listed.""")

MACHINE_CHECK = '''# --- your machine: Python, GPU, local LLM (read only) -------------------------------------------------------
import os, platform, shutil, socket, subprocess, sys
from pathlib import Path
from urllib.parse import urlparse

print("Python  :", sys.version.split()[0], "|", sys.executable)
_venv = Path(sys.executable).parent.parent
print("kernel  :", "OK, the project's .venv" if _venv.name == ".venv" else
      f"NOT the project's .venv ({_venv}): choose the .venv kernel (top right)")
print("machine :", platform.platform(), "|", os.cpu_count(), "CPU threads")

_smi = shutil.which("nvidia-smi")
if not _smi:
    print("GPU     : no NVIDIA GPU found (nvidia-smi is not on the PATH: no NVIDIA driver, or another GPU brand)")
else:
    def _q(args):
        try:
            r = subprocess.run([_smi, *args, "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            return [l.strip() for l in r.stdout.splitlines() if l.strip()]
        except (OSError, subprocess.SubprocessError) as e:
            return [f"nvidia-smi failed: {e}"]
    for i, row in enumerate(_q(["--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu"])):
        print(f"GPU {i}   : {row}   (name, driver, memory total, memory used, busy now)")
    apps = _q(["--query-compute-apps=process_name,used_memory"])
    print("programs on the GPU:", "; ".join(apps) if apps else "none right now (the LLM server is not running or idle)")

try:                                              # optional: only if PyTorch is installed in this venv
    import importlib.util
    if importlib.util.find_spec("torch"):
        import torch
        print("PyTorch : CUDA available =", torch.cuda.is_available(),
              f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
except Exception as e:                            # noqa: BLE001 - a check, never a failure
    print("PyTorch : check failed:", e)

# the local LLM server of .env (the key is never printed)
_root = Path.cwd()
while not (_root / "pyproject.toml").exists() and _root != _root.parent:
    _root = _root.parent
try:
    from dotenv import dotenv_values
    _env = {**dotenv_values(_root / ".env"), **os.environ}
except ImportError:
    _env = dict(os.environ)
_backend = (_env.get("LLM_BACKEND") or "offline").lower()
_urls = {"vllm": ("VLLM_URL", "http://127.0.0.1:8001"), "ollama": ("OLLAMA_URL", "http://127.0.0.1:11434"),
         "llamacpp": ("LLAMACPP_URL", "http://127.0.0.1:8080"), "lmstudio": ("LMSTUDIO_URL", "http://127.0.0.1:1234")}
if _backend not in _urls:
    print(f"LLM     : LLM_BACKEND = {_backend}: no LLM server, the agents use their rule fallbacks (GPU not needed)")
else:
    _u = urlparse(_env.get(_urls[_backend][0]) or _urls[_backend][1])
    try:
        socket.create_connection((_u.hostname, _u.port or 80), timeout=2).close()
        _up = "listening"
    except OSError:
        _up = "NOT running (start it, or set LLM_BACKEND=offline)"
    print(f"LLM     : {_backend} at {_u.scheme}://{_u.hostname}:{_u.port} -> {_up}")
'''

AGENTS_MD = md("""## The agents: Agent 1 … Agent 9

The engine is organised as numbered agents (the numbers of the plan, `docs/fsdr_plan.md` §5). The step titles below
say which agent works in each step.

| Agent | Name | Step (notebook) | What it does | How it decides in V4.0 |
|---|---|---|---|---|
| **Agent 1** ① | Data agent | 01 · Step 2 (T0) | validates the 47 tables, completeness, cache; **blocks the run** on bad data | rules (contract checks); its summary is fixed text |
| **Agent 2** ② | Segmentation | 01 · Step 5 (T3) | S-rules → segment, bands, micro-segment | rules (first rule that matches wins) |
| **Agent 2b** ②b | Agent dictionary | 01 · Step 5 (T3) | segment → `action_agent` (used by Target 2) | week 0: a copy of the production dictionary (v0) |
| **Agent 3** ③ | NBA, next best action | 02 · Step 4 | the action of each T2C line | fallback: the rule action `action_agent` (the LLM decides in notebooks 03 / 04) |
| **Agent 4** ④ | NBO, next best offer | 02 · Step 4 | scores the allowed offers of each T2C / T2F line | fallback: the order E-O3 + E-O1 (`score_source` says it) |
| **Agent 6A** ⑥A | Production rules | — | Target 1 (not scored in V4.0); its code 3 / code 4 passes remain the fallback of 6B | rules (YAML in `nbx/fsdr/rules/`) |
| **Agent 6B** ⑥B | Challenger rules | 02 · Step 4 | Target 2: CC-v1 (group a) and CF-v1 (group b), relative ARPU band | rules |
| **Agent 7** ⑦ | Code 5 | 02 · Step 5 | the 38-column `input_flytxt` file, F-01 … F-19 | rules |
| **Agent 9** ⑨ | Auditor | 02 · Steps 4 and 6 | A-12, A-12b, A-17, A-30, A-31; **blocks the final file** on any FAIL | rules |

Not in these notebooks: Agent 5 ⑤ (market intelligence, removed in V4.0), Agent 8 ⑧ (strategy report),
Agents 10 / 11 (⑩ bandit, ⑪ ranker: evidence helpers), Agent 12 ⑫ (ledger), Agent 13 ⑬ (revenue guard) and the
orchestrator ★. They belong to later phases or to the web interface.

**Does an agent create the actions by itself? Not in V4.0.** Every action and every offer of these notebooks comes
from fixed, versioned rules, and each row says so: `DECIDED_BY = fallback_rule_action`. When the LLM agents ③ / ④
are switched on (plan Phases 5–6), the local LLM will only **score** the actions and offers that the rules allow;
the code then picks the action (with ε exploration and a logged propensity), rejects any action or offer id the LLM
invents, and ⑨ can still block the final file. The LLM proposes; the rules decide.""")


NEW_DATA_MD = md("""## How to add new data (real data, or the next week)

Everything is driven by the **run date `D`** (`YYYY-MM-DD`, the extraction date). Each week has its own folders;
the old weeks stay. The full guide, with every table, is `docs/DATA_UPDATE.md`.

1. **Create the folders of `D`** (5 sub-folders on each side, plus the target folders):
   ```
   data/training/<D>/  usage/ models/ subscription/ catalog/ others/
   data/scoring/<D>/   usage/ models/ subscription/ catalog/ others/
                       target2/a_constraint/ target2/b_free/
   ```
2. **Copy the exports**: file name = the exact table name + `.csv.gz` (the only format accepted for real data), e.g. `activities_w1_auto.csv.gz`.
   - **Training side**: `base_prepaid`, `consumer_aact_account`, `subs_btl_auto`, the 11 model tables, `btl_catalog`.
     **Week 0 only**: the history `activities_*_auto`, `subs_amt_auto`, `party_auto`, `access_method_id_auto`,
     `targeting_dataiku_auto`. **Leave them out from week 1 on**, or T0 runs week 0 again on the old history.
   - **Scoring side** (every week): `subscriptions_tg`, `usage_data_voice_payg_tg`, `sub_btl_3months`,
     `activities_w1…w4_score`, `activities_m1…m3_score`, `subs_amt_score`, `party_scoring`, `targeting_auto`,
     `btl_catalog`, `BTL_script_english_arabic_ok`, `conflicting_offers_h`, `consent_contact`.
3. **Copy the 3 target lists** (column `access_method_val`, 8 digits): `target2/a_constraint/target2_a.csv.gz`
   (group a, 50,000 lines) and `target2/b_free/target2_b.csv.gz` (group b, 50,000 lines). No control list by default.
4. **Settings**: `RUN_DATE = "<D>"` in both notebooks, `GENERATE = False` (training) and `GENERATE_TARGETS = False`
   (scoring). Real exports and real lists are never overwritten by dummy data, even if you forget.
5. **Run** training (T0 → T2 → T3), then scoring. Check: T0 **DONE** (`train_week0` the first week,
   `train_weekly` after), both target audits **PASS**, and `data/output/<D>/final/input_flytxt.csv.gz` exists.
6. **Hand over** `data/output/<D>/final/input_flytxt.csv.gz` once it is checked. The engine only saves files, it sends nothing. Keep `data/runs/<D>/` and `data/output/<D>/`.""")


def build(out_dir: Path = HERE) -> list[Path]:
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = standalone(cells, name)
        nb.metadata = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        p = Path(out_dir) / name
        nbf.write(nb, p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in build():
        print("written", p)
