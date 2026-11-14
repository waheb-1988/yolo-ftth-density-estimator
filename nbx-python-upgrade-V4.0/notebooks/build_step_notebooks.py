"""Builds the two step-by-step notebooks of V4.0 (run it again after editing the cells below):

    uv run python notebooks/build_step_notebooks.py

    notebooks/01_training_step_by_step.ipynb   data -> T0 validation + cache -> T2 features -> T3 segments and actions
    notebooks/02_scoring_step_by_step.ipynb    target lists -> P1 panel -> Target 1 (P6 + final) -> Target 2 -> push

Both call the same functions as the terminal commands (fsdr-generate, fsdr-load, fsdr-features, fsdr-targets,
fsdr-score-target, fsdr-push), one step per cell, and show what each step wrote. They are stored WITHOUT outputs:
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
                     target1/a_constraint/ target1/b_free/ target2/a_constraint/ target2/b_free/ control/
```

T0 reads the two folders **as one run**. A table filed on the wrong side is still loaded, with a warning.

**How to use:** set the run date in the next cell, then run the cells one by one (Shift+Enter). Read the explanation above each step.
Every step is **resumable**: a step already done with the same inputs is skipped, and `FORCE = True` re-runs it."""),
    code("""# --- settings -------------------------------------------------------------------
RUN_DATE = "2026-10-22"     # the run date D (a Thursday in production)
GENERATE = True             # True: write dummy data for D first (tests); False: use the exports you copied
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

The manifest says `source: generator`, so real exports are never overwritten silently. With real data, set `GENERATE = False` and copy your files into the same folders."""),
    code("""if GENERATE:
    from nbx.fsdr.synth import generator

    man = TRAINING / config.MANIFEST_FILE
    if TRAINING.exists() and any(TRAINING.iterdir()) and (not man.exists() or json.loads(man.read_text())["source"] != "generator"):
        raise SystemExit(f"{TRAINING} holds real data: set GENERATE = False")
    run = generator.generate(N_LINES, D, ["baseline"], config.GENERATOR_SEED, TRAINING, "csv.gz", "split")
    print(f"{len(run.tables)} tables generated for {N_LINES:,} lines")
else:
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

    md("""## Step 2 · T0: validation, completeness, Parquet cache (① data agent)

T0 runs before anything else and does three things.

1. **Validation** against the data contract: columns, types, keys, value ranges, date windows, the `.csv.gz` format and the folders.
   - A *blocking* issue stops the run.
   - A *warning* is reported and the run continues.
2. **Completeness**: are the tables the pipeline needs all present?
   - Week 0 needs the `*_auto` tables; a weekly run needs the `*_score` tables.
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
print(t0["explanation"])"""),
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

    md("""## Step 5 · T3: segments, micro-segments and the two actions

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
  - `ACTION_ORIGINAL`: the production dictionary (segment -> action). Target 1 uses it.
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
      f"{{' (PUSH BLOCKED)' if p6['push_blocked'] else ''}}, up to {{p6['reco_offers']}} offers per line")
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
print(f"{{TARGET}} final {{'READY' if fin['push_ready'] else 'BLOCKED (see BLOCKED.txt)'}} -- input_flytxt_{{TARGET}} "
      f"{{fin['input_flytxt_rows']:,}} rows; details {{fin['details_rows']:,}} lines x {{fin['details_columns']}} columns")
show(fin["output_checks"], title="output checks A-20 ... A-24")
show(fin["per_group"], title="per group a / b")
tree(OUT_T, depth=2)'''

TARGET_DETAILS = '''det = pd.read_parquet(OUT_T / f"details_{{TARGET}}.parquet")
reasons = pd.concat([det[f"REASON_{{i}}"].fillna("(empty)").value_counts().rename(f"slot {{i}}") for i in range(1, 7)], axis=1)
display(Markdown("**What fills each slot** (V4.0: 5 = 3rd monthly, 6 = 3rd weekly; a favourite or a forced offer only "
                 "fills a slot 5 / 6 left empty)"))
display(reasons.fillna(0).astype(int))
flytxt = csvstd.read_csv(OUT_T / f"input_flytxt_{{TARGET}}{{csvstd.SUFFIX}}") if fin["push_ready"] else None
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
    md("""# 02 · Scoring, step by step (FSDR V4.0), each target separately

Scoring runs after training (`01_training_step_by_step.ipynb`) on the **same run date**. It takes the target lists, builds the panel, and scores **Target 1** and **Target 2 separately**, each into its own output folder.
It then builds the single file to push.

| Target | Group a (`a_constraint`) | Group b (`b_free`) | Action used |
|---|---|---|---|
| **Target 1** (production) | T1C · code 3, the action's offer families | T1F · code 4, L2 DATA / COMBO | `action_original` |
| **Target 2** (challenger) | T2C · CC-v1, relative ARPU band | T2F · CF-v1, relative band, L2 | `action_agent` |
| **Control** | never contacted | | |

| Step | What it does | Writes |
|---|---|---|
| 1 | check that training is done | — |
| 2 | the target lists (upload, or dummy lists), their checks | `data/scoring/<D>/target1/…`, `target2/…`, `control/` |
| 3 | **P1** panel from the lists, sizes, balance | `data/runs/<D>/score/P1_panel/` |
| 4–5 | **Target 1**: P6 offers, then the final files | `data/runs/<D>/score/P6_offers_target1/`, `data/output/<D>/target1/` |
| 6–7 | **Target 2**: the same | `…/P6_offers_target2/`, `data/output/<D>/target2/` |
| 8 | control file and **push** (both targets PASS) | `data/output/<D>/control/`, `data/output/<D>/push/` |
| 9 | validate the file to push | — |

**Up to 6 offers per line (V4.0).** Slots 1–2 are the 2 best monthly offers and slots 3–4 the 2 best weekly ones. Slot 5 is the 3rd monthly offer and slot 6 the 3rd weekly one (rule EXT-6: same universe, above the anchor, not the same service and price). Favourites and forced bundles only fill a slot 5 / 6 left empty. Set `FSDR_RECO_OFFERS=4` in `.env` for the V3.0 file (4 recommended offers).

Decisions use the documented fallbacks of the LLM agents (rule action, NBO order E-O3 + E-O1) until Phases 5–6 build them, and every row says so (`DECIDED_BY`, `score_source`)."""),
    code("""# --- settings -------------------------------------------------------------------
RUN_DATE = "2026-10-22"         # the same run date as the training notebook
GENERATE_TARGETS = True         # True: dummy lists (stratified draw of the eligible base); False: your uploaded lists
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
data/scoring/<D>/target1/a_constraint/target1_a.csv.gz    T1C
data/scoring/<D>/target1/b_free/target1_b.csv.gz          T1F
data/scoring/<D>/target2/a_constraint/target2_a.csv.gz    T2C
data/scoring/<D>/target2/b_free/target2_b.csv.gz          T2F
data/scoring/<D>/control/control.csv.gz                   control
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

if GENERATE_TARGETS:
    gen = scoring_groups.generate_targets(D)
    print(f"dummy lists written (scaled x{gen['scale_factor']} to the base): {gen['sizes']}")
folder = scoring_groups.input_folder(D)
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

    md("""## Step 4 · Target 1, scoring (P6): T1C (a) and T1F (b)

These rules apply only to the two groups of Target 1:

1. **Action** = `action_original`, the production dictionary.
2. **Candidates** per validity (30 days, then 7 days):
   - T1C: the offer families of the action (B-21);
   - T1F: the L2 DATA / COMBO offers, without VOICE (FR-02 / FR-03).
   - Only offers bought in the base in the last 270 days count (B-20b), at most 25 per validity.
3. **Price passes** around the anchor (the customer's average spend, B-26 / B-27):
   - pass 1: anchor < price < anchor + k;
   - pass 2: the missing type;
   - pass 3: nothing found yet.
   - k = 2 or 3 by action for code 3, k = 3 for code 4.
4. **EXT-6 (V4.0)**: a third offer per validity.
5. **10.1 consistency** (T1C only): a customer with no offer is retried once with his second action.

The **group audit** (A-12 universe, A-12b catalogue, A-17 rules version, A-21 no control line, A-30 / A-31) must PASS, or the push is blocked."""),
    code(TARGET_P6.format(t="target1")),
    code(TARGET_P6_DETAIL.format(a="t1c")),

    md("""## Step 5 · Target 1, final files (code 5) → `data/output/<D>/target1/`

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
`input_flytxt_target1.csv.gz` is written **only when everything passes** (H-08); otherwise you get `BLOCKED.txt` with the reasons.

| File | Content |
|---|---|
| `input_flytxt_target1.csv.gz` | the file of Target 1 (both groups) |
| `a_constraint/input_flytxt_t1c.csv.gz`, `b_free/input_flytxt_t1f.csv.gz` | the same, per group |
| `a_constraint/details_t1c.csv.gz`, `b_free/details_t1f.csv.gz` | every line of the group with every column |
| `details_target1.parquet` | the big file to validate: profile, segment, both actions, features, anchors, offers, slots, reasons, removals |
| `final_rules_report.csv`, `output_checks.json`, `audit_groups.json`, `summary.json` | the checks and counts |"""),
    code(TARGET_FINAL.format()),
    code(TARGET_DETAILS.format()),

    md("""## Step 6 · Target 2, scoring (P6): T2C (a) and T2F (b), the challenger

These rules apply only to the two groups of Target 2:

1. **Action** = `action_agent`, the agent dictionary.
2. **Challenger anchor** (CC-03 / CF-03): the customer's spend, split data / voice. A customer with no spend gets the median of his micro-segment.
3. **Relative band** (CC-02 / CF-02): price between 1.0× and 1.5× the anchor. UPSELL actions need at least +0.5 OMR, and no offer is ever a downgrade.
   - The pick is the best `score × price` (C-04).
   - If nothing fits, the band is widened once, to 2.0×.
   - Then the Target 1 passes apply as a fallback (logged as `fallback …`).
4. **EXT-6**: the third offer per validity.
5. **10.1 consistency**: T2C only."""),
    code(TARGET_P6.format(t="target2")),
    code(TARGET_P6_DETAIL.format(a="t2c")),

    md("""## Step 7 · Target 2, final files → `data/output/<D>/target2/`

Same code 5 and the same checks as Target 1, on the Target 2 lines only. The files are the same, with `t2c` / `t2f` in their names."""),
    code(TARGET_FINAL.format()),
    code(TARGET_DETAILS.format()),

    md("""## Step 8 · Control file and the push

- `data/output/<D>/control/control_<D>.csv.gz` holds the control lines, which are never contacted (H-01), with their strata. They are the baseline for measuring uplift.
- `data/output/<D>/push/input_flytxt.csv.gz` holds both target files merged. It is written **only if both targets passed** and no MSISDN is in two targets (A-30); otherwise you get `push/BLOCKED.txt`.

A **human approval** (H-08) is still needed before the file goes to the campaign platform."""),
    code("""ctrl = scoring_groups.write_control(D)
print(f"control: {ctrl['lines']:,} lines -> {ctrl['file']}")
push = scoring_groups.run_push(D)
print(f"push {'READY' if push['push_ready'] else 'BLOCKED'} -- {push['input_flytxt_rows']:,} rows {push['rows_per_target']}"
      + (f" -- reasons: {push['reasons']}" if push["reasons"] else ""))
tree(OUTPUT, depth=3)"""),

    md("""## Step 9 · Validate the file to push

These are the last checks before handing the file over:
- the 38 columns in order;
- one row per MSISDN, as 8-digit text;
- no control line in the file;
- every line found in the details with `IN_INPUT_FLYTXT = True`;
- offers per line between 1 and 6."""),
    code("""f = csvstd.read_csv(OUTPUT / "push" / f"input_flytxt{csvstd.SUFFIX}")
details = pd.concat([pd.read_parquet(OUTPUT / t / f"details_{t}.parquet") for t in config.TARGETS])
control = csvstd.read_csv(OUTPUT / "control" / f"control_{RUN_DATE}{csvstd.SUFFIX}")
n = f[[f"destination_offer_name_{i}" for i in range(1, 7)]].ne("").sum(axis=1)
checks = {
    "38 columns in order": list(f.columns) == config.INPUT_FLYTXT_COLUMNS,
    "one row per msisdn": bool(f["msisdn"].is_unique),
    "8-digit text msisdn": bool(f["msisdn"].astype(str).str.fullmatch(r"\\d{8}").all()),
    "no control line": not (set(f["msisdn"]) & set(control["ACCESS_METHOD_VAL"])),
    "every line in the details": set(f["msisdn"]) == set(details.loc[details["IN_INPUT_FLYTXT"], "ACCESS_METHOD_VAL"]),
    "offers per line 1..6 (recommended + favourites / forced)": bool(n.between(1, 6).all()),
}
show([{"check": k, "status": "PASS" if v else "FAIL"} for k, v in checks.items()], title="push file")
print(f"{len(f):,} lines, {n.mean():.2f} offers per line on average; segment (action) mix:")
display(f["segment"].value_counts().to_frame("lines"))
show(f, n=5, title="input_flytxt.csv.gz (first rows)")"""),
]

NOTEBOOKS = {"01_training_step_by_step.ipynb": TRAINING_CELLS, "02_scoring_step_by_step.ipynb": SCORING_CELLS}


def build(out_dir: Path = HERE) -> list[Path]:
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        p = Path(out_dir) / name
        nbf.write(nb, p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in build():
        print("written", p)
