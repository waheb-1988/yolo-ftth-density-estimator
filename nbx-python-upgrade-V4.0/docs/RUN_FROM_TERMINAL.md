# FSDR V4.0: run it from the terminal, check the results in the notebooks

The same flow in the web interface, tab by tab: [USER_GUIDE_INTERFACE.md](USER_GUIDE_INTERFACE.md). A new laptop, from
`pip install uv` to the first run: [SETUP_NEW_LAPTOP.md](SETUP_NEW_LAPTOP.md).

Every command runs from the project folder. They work the same in PowerShell, Git Bash or the VS Code terminal.

```powershell
cd <path>\nbx-python-upgrade-V4.0
```

Status: **Phases 1–3 are built**, plus the **V3.0 changes** (the `.csv.gz` file contract, the four test groups) and the **V4.0 changes**:
- training and scoring input in separate folders;
- each target scored separately into `data/output/<D>/<target>/`;
- up to 6 offers per line;
- the step-by-step notebooks;
- no market-intelligence module.

 The commands marked *(Phase N)* arrive with that phase. Section 7 lists them so you can see the full flow.

---

## 1. Once: install

```powershell
pip install uv                   # once per laptop (or the standalone installer, SETUP_NEW_LAPTOP.md §2)
uv sync                          # creates .venv from uv.lock (Python 3.12, every package incl. Jupyter)
uv run python run.py check       # Python, packages, data files OK?
uv run python run.py test        # the whole test suite (~4 min, no network, no GPU)
```

`.env` holds your local settings, including the vLLM key. It is never committed; copy `.env.example` to start one.

## 2. Get the input data of a week

The data of run date D is split in two folders with the same 5 sub-folders, one file per table. T0 reads them as one run:

```
data/training/2026-10-22/          the base, the history, the model scores (training T0 - T3)
├── usage/          subs_amt_auto, activities_*_auto, paygo_activity_auto
├── models/         the 11 model-score tables
├── subscription/   base_prepaid, consumer_aact_account, access_method_id_auto, subs_btl_auto
├── catalog/        btl_catalog
├── others/         party_auto, customer_profile, location_auto, device segment, targeting_dataiku_auto
└── _manifest.json
data/scoring/2026-10-22/           this week's tables + the target lists (scoring)
├── usage/          usage_data_voice_payg_tg (+ the *_score tables of a weekly run)
├── models/         (empty when unused)
├── subscription/   subscriptions_tg, sub_btl_3months
├── catalog/        btl_catalog, BTL_script_english_arabic_ok, conflicting_offers_h
├── others/         targeting_auto, consent_contact, party_scoring, sample_cg_tg_* (optional)
├── target1/a_constraint/  target1/b_free/  target2/a_constraint/  target2/b_free/  control/   (section 5a)
└── _manifest.json
```

The side of each table is `config.SCORING_TABLES` (everything else is training). A table on the wrong side still loads,
with a warning; two files of one table on the same side block T0. A single folder `data/input/<D>/` (the V3.0 layout)
still works: pass it to `fsdr-load --data-dir`.

Format: **`<table>.csv.gz`** (the Dataiku compressed CSV export, *FSDR CSV standard*), optionally with `<table>.schema.json` next to it. The file name is the exact table name. `.csv`, `.json`, `.jsonl` and `.parquet` are read too, but only for generated data (its `_manifest.json` says `source: generator`) or with `FSDR_ALLOW_TEST_FORMATS=true` in `.env`; otherwise T0 blocks.

**Real data:** copy the exported `.csv.gz` files into those folders.

**Dummy data:** use the generator.

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --mode small                    # 2,000 lines -> data/training + data/scoring
uv run python run.py fsdr-generate --date 2026-10-22 --mode medium                   # 100,000 lines, .csv.gz (~80 s, 93 MB)
uv run python run.py fsdr-generate --date 2026-10-22 --mode full                     # 1,000,000 lines, .csv.gz
uv run python run.py fsdr-generate --date 2026-10-22 --mode full --format parquet    # same in Parquet (faster to write)
uv run python run.py fsdr-generate --date 2026-10-22 --weeks 3 --mode small          # 3 consecutive weeks from one world
uv run python run.py fsdr-generate --date 2026-10-22 --mode small --scenario data_quality_issues   # data with defects
```

The default format is `csv.gz`, as production, and the default layout `split` (training + scoring folders; `--layout folders` writes one `data/input/<D>/` folder). `--format json` would be about 12 GB at 1M lines. Add `--force` to overwrite a folder that holds non-generated data.

## 3. Step T0: validate, check completeness, build the cache

```powershell
uv run python run.py fsdr-load --date 2026-10-22          # = --data-dir data/training/2026-10-22 (+ its scoring twin)
```

It prints `DONE` or `BLOCKED` with a plain explanation (and `Read as one run: …training… + …scoring…`), and saves everything in `data/runs/2026-10-22/train/T0_load/`:

| File | Content |
|---|---|
| `summary.json` | status, pipeline, tables, rows, explanation |
| `validation_report.md` / `.json` | every check: fields, types, keys, duplicates, joins, coverage, windows, folders, customer_profile |
| `completeness.csv` | each table the pipeline needs: present, rows, ok / missing |
| `cache_manifest.json` | the Parquet cache: rows and SHA-256 of each of the 16 partitions |
| `_step.json` | status, start, duration, rows, the inputs it ran on |

- **Running it again with the same files** skips it and prints `skipped`. Add `--force` to re-run it.
- **Exit code:** 0 when done, 1 when blocked, so a scheduler can stop the chain.
- **Pipeline:** detected automatically. Week 0 needs the `_auto` history tables; a normal week needs only `_score`. Force it with `--pipeline train_week0` or `--pipeline train_weekly`.
- **Validation only**, without the cache: `uv run python run.py fsdr-validate --data-dir data/training/2026-10-22`.
- **Tables on the wrong side** are listed in `misfiled_tables.json` (warning only).

## 4. Steps T2 + T3: features, segments, the two actions

```powershell
uv run python run.py fsdr-features                       # the latest run that has a T0_load
uv run python run.py fsdr-features --date 2026-10-22 --workers 8    # 8 partitions in parallel (P16); default 2
```

It needs a `done` T0_load for that date (otherwise it prints `BLOCKED` and exits with 1). It runs on the Parquet cache,
partition by partition, and prints the segment counts. Same inputs again → `skipped`; `--force` re-runs both steps.

`data/runs/<D>/train/T2_features/`, the features B-01 … B-12 of every eligible line:

| File | Content |
|---|---|
| `base_features/part-NN.parquet` | one file per partition: scores (+ `*_MISSING` flags), usage, spend, REV, trends, nationality group, AGE / TENURE_DAYS and their source, region, device |
| `base_features_preview.csv` | the first 1,000 rows, to open in Excel |
| `feature_summary.csv` | per feature: count, nulls, mean, min, max |
| `stats.json` | the B-05 medians per nationality group (missing scores), the ARPU tertiles |

`data/runs/<D>/train/T3_segments/`:

| File | Content |
|---|---|
| `segments/part-NN.parquet` | per line: S-rule, segment, rule action, trigger values, age / tenure / value bands, usage type, micro-segment, week-0 segment and reward, `action_original` (+ source), `action_agent` (+ rank 2, version) |
| `segment_counts.csv` | lines, share, average ARPU, week-0 targeted and takers per segment |
| `microsegment_counts.csv` | the 120 cells nationality × age × tenure, with their lines |
| `crosswalk.csv` | S-rule segment → original segment name → action; every line is a CHECK item |
| `segment_agreement.csv`, `tenure_agreement.csv` | S-rules vs the segment engine; segment family vs tenure band |
| `action_counts.csv` | `action_original` (by source) × `action_agent` |
| `dict_segm_actions_original.json`, `dict_segm_actions_agent_v0.json` | code 1's dictionary (last value wins) and the agent dictionary (week 0 = a copy; the LLM edits it from Phase 4) |
| `priors.json` | B-10 priors: configured and recomputed from week 0, with the week-0 take rate per action |
| `value_usage.csv`, `young_split.csv` | context bands; the under-25 / 25–39 split to test in Phase 4 |

## 5. The panel and the four groups (two targets)

**V4.0: each target is scored separately** (section 5a):

```powershell
uv run python run.py fsdr-score-target --target target1 --panel   # P1, then P6 + final of Target 1 (T1C a, T1F b)
uv run python run.py fsdr-score-target --target target2           # P6 + final of Target 2 (T2C a, T2F b)
uv run python run.py fsdr-push                                     # both targets merged (both PASS) -> data/output/<D>/push/
uv run python run.py fsdr-run-targets                              # the three above in one go
```

The V3.0 commands still score the four groups together (`data/runs/<D>/score/P6_offers/`, `data/runs/<D>/final/`):

```powershell
uv run python run.py fsdr-panel                 # P1: T1C / T1F / T2C / T2F / control (needs fsdr-features)
uv run python run.py fsdr-offers                # P6 of the 4 groups together: code-2 tables + the 8 reco tables, .csv.gz
```

| Group | Rules | Offer universe | Rule action (NBA hint) | Tables |
|---|---|---|---|---|
| T1C (25k) | production code 3 (B-20 … B-32) | families + generosity by action | `action_original` | `TG_40K_constraint_scored_nba`, `recos_monthly/weekly_40k_constraint` |
| T1F (25k) | production code 4 (FR-01 … FR-14) | L2, DATA / COMBO, action not used | `action_original` | `TG_40k_free_scored_nba`, `recos_monthly/weekly_tg_free` |
| T2C (25k) | challenger CC-v1 | as T1C | `action_agent` | `TG_challenger_constraint_scored_nba`, `recos_*_challenger_constraint` |
| T2F (25k) | challenger CF-v1 | as T1F | `action_agent` | `TG_challenger_free_scored_nba`, `recos_*_challenger_free` |
| control (20k) | never contacted | — | — | — |

`data/runs/<D>/score/P1_panel/`: `panel.parquet` (+ preview), `group_sizes.csv`, `balance.csv` (share of every
stratum value per group, max gap in points, chi-square p), `summary.json`. The provided sample
`sample_cg_tg_new_7oct2026` is used when present (cstr → T1C, free → T1F, cg → control); `--no-sample` ignores it.
A base smaller than 120k eligible lines gets every group scaled down by the same factor (printed).

`data/runs/<D>/score/P6_offers_target1/` and `P6_offers_target2/` (V4.0, one per target, with that target's tables
only) or `P6_offers/` (V3.0, the 4 groups): the tables as `.csv.gz` with their `.schema.json` and `_outputs_manifest.json`,
`group_report.csv` (offers per rule / pass, 10.1 retries, LLM coverage at the 10 h budget), `traces.csv` (2 customers
of the same type per group: action → candidates → score → pass / rule → offers), `audit_groups.json` (A-12, A-12b,
A-17, A-21, A-30, A-31). Exit code 1 if the audit fails (push blocked).

**6 offers (V4.0, `FSDR_RECO_OFFERS=6`):** after the passes of its group, a customer gets a third offer per validity:
- same universe;
- priced above the anchor (never a downgrade);
- not the same service + price as a selected offer.

Its `selected_by` is `EXT-6` (`pass` 6). `FSDR_RECO_OFFERS=4` restores the V3.0 selection.

Until the LLM agents (Phases 5–6), the action is the group's rule action and the offer score is the NBO fallback
(own past purchase, then buyer share); every row says so in `decided_by` / `score_source`.

Sizes and switches (`.env`): `FSDR_T1_CONSTRAINT_SIZE`, `FSDR_T1_FREE_SIZE`, `FSDR_T2_CONSTRAINT_SIZE`,
`FSDR_T2_FREE_SIZE`, `FSDR_CONTROL_SIZE`, `FSDR_T2_SPLIT` (false = one 50k T2 with the CC rules),
`FSDR_LLM_WEEKLY_BUDGET_H`, `FSDR_GUARD_MODE_TARGET1`.

## 5a. Your own target lists, each target scored separately, and the file to push

**Upload folder.** Put one list per group in the run's **scoring** folder, one folder per target and one sub-folder per
group (any extra columns are kept in the details):

```
data/scoring/2026-10-22/
├── target1/a_constraint/target1_a.csv.gz     T1C  production code 3
├── target1/b_free/target1_b.csv.gz           T1F  production code 4
├── target2/a_constraint/target2_a.csv.gz     T2C  challenger CC-v1
├── target2/b_free/target2_b.csv.gz           T2F  challenger CF-v1
└── control/control.csv.gz                    never contacted
```

Each file needs an `access_method_val` (or `msisdn`) column. `.csv.gz` in production; `.csv` / `.json` for tests.
When the folder holds lists, the panel uses them; otherwise it draws the groups at random (section 5).

```powershell
uv run python run.py fsdr-targets --generate      # dummy lists for testing (stratified draw of the eligible base)
uv run python run.py fsdr-targets                 # check the lists that are there
uv run python run.py fsdr-run-targets             # panel (your lists) -> Target 1 -> Target 2 -> control -> push
uv run python run.py fsdr-score-target --target target2   # re-score one target only (after fixing its list: add --panel)
uv run python run.py fsdr-push                    # rebuild the push file from the two target folders
```

Through the app (API): `POST /fsdr/targets/<D>/<group>` (file upload, group = t1c / t1f / t2c / t2f / control),
`GET /fsdr/targets/<D>` (lists + check), `POST /fsdr/v4/<D>/panel`, `POST /fsdr/v4/<D>/score/<target1|target2>`,
`POST /fsdr/v4/<D>/push`, `POST /fsdr/v4/<D>/run`, `GET /fsdr/v4/<D>/output`.

**Checks of the lists:** a line in two groups or a missing list **blocks**; a line that is not an 8-digit MSISDN, not in
the eligible base, in Nursering (B-02) or in another campaign this week (H-05) is **dropped** and listed in
`score/P1_panel/targets_rejected.csv` with its reason; a duplicate inside a list or a size off the configured size by
more than 2 % is a warning.

**What is saved every run** in `data/output/<D>/` (V4.0):

| File | What |
|---|---|
| `push/input_flytxt.csv.gz` | **the file to push**: the 38 columns of code 5, both targets, one row per MSISDN. Written only when both targets passed and no MSISDN is in two targets; otherwise `push/BLOCKED.txt` says why (H-08) |
| `target1/input_flytxt_target1.csv.gz` | Target 1's file (only when Target 1 passed, else `target1/BLOCKED.txt`) |
| `target1/a_constraint/input_flytxt_t1c.csv.gz`, `target1/b_free/input_flytxt_t1f.csv.gz` | the same rows, per group |
| `target2/…` | the same for Target 2 (`t2c`, `t2f`) |
| `target1/a_constraint/details_t1c.csv.gz`, `target1/b_free/details_t1f.csv.gz`, `target2/…/details_t2c`, `details_t2f` | **the details ("big data")**: one row per line of the group, ≈ 240 columns: group, stratum, profile and bands, segment and its triggers, `action_original` / `action_agent` / action sent / agreement / 10.1 retry, every feature (scores, missing flags, revenue, trends), B-26 and F-05 anchors, the recommended offers per validity with their pass or rule, the 6 slots with their reason (recommended / favourite / forced) and the rule that cleared them (F-09 / F-11 / F-12), in the file or not and why |
| `target1/details_target1.parquet`, `target2/details_target2.parquet` | all the details of the target in one table (for the notebook / Excel via Power Query) |
| `control/control_<D>.csv.gz` | the control list kept for measurement |
| `<target>/final_rules_report.csv` | per group: lines, rows in the file, offers per line, F-04 drops, favourites, forced bundles, F-09 / F-11 / F-12 clearings |
| `<target>/output_checks.json`, `audit_groups.json`, `_outputs_manifest.json`, `*.schema.json`, `summary.json` | the checks A-20 … A-24, the P6 group audit, rows and SHA-256 of every file, the column types |

The scoring files of each target stay in `data/runs/<D>/score/P6_offers_<target>/` (the code-2 tables and the reco
tables of its two groups). The V3.0 `fsdr-run-groups` / `fsdr-final` still write the four groups together to
`data/runs/<D>/final/`.

**The 6 slots of `input_flytxt` (V4.0):**

| Slot | Content |
|---|---|
| 1, 2 | the 1st and 2nd monthly offers |
| 3, 4 | the 1st and 2nd weekly offers |
| 5 | the 3rd monthly offer (EXT-6); when empty, a favourite |
| 6 | the 3rd weekly offer; when empty, a favourite or a forced bundle (F-08) |

## 5b. The notebooks: run every step, or check the results

**Step by step (V4.0).** These two notebooks run the steps one cell at a time, with the explanation of each step and its details:

```powershell
uv run python run.py notebook --name training    # 01_training_step_by_step.ipynb: data -> T0 -> cache -> T2 -> T3
uv run python run.py notebook --name scoring     # 02_scoring_step_by_step.ipynb: lists -> P1 -> Target 1 -> Target 2 -> push
```

They call the same functions as the commands above and write the same files. Rebuild them after editing
`notebooks/build_step_notebooks.py` with `uv run python notebooks/build_step_notebooks.py`.

**Read-only view.** The notebook `notebooks/fsdr_results.ipynb` reads what the commands saved; it computes nothing new. Its sections:

1. the input folders (training + scoring): tables, formats, sizes, manifest
2. every step of the run: status, seconds, rows, files
3. T0 summary, validation issues, table checks, model-score coverage chart
4. completeness for the 3 pipelines (training week 0, week n, scoring)
5. customer_profile: age and tenure histograms
6. the Parquet cache: partition balance, checksum check, and any table or partition shown in full
7. T2 / T3: segment counts, the micro-segment heat map, the crosswalk, both actions, the priors
8. **every step folder with a preview of its files.** The steps of the next phases (evidence, actions, offers, audit) appear here without changing the notebook.
9. the final `input_flytxt` (38 columns), once Phase 7 is built

There are three ways to open it:

```powershell
uv run python run.py notebook            # opens it in VS Code: choose the kernel ".venv (Python 3.12)", then Run All
uv run python run.py notebook --lab      # JupyterLab in the browser
uv run python run.py notebook-html       # no browser: runs it and saves data/runs/<D>/fsdr_results_<D>.html
uv run python run.py notebook-html --date 2026-10-22
```

In the notebook, the first cell holds `RUN_DATE = None`, which means the latest run. Put a date there to look at another week.

To change the notebook's cells, edit `notebooks/build_results_notebook.py`, then run `uv run python notebooks/build_results_notebook.py`. The notebook in git has no outputs: outputs would contain MSISDNs, so clear them before committing.

## 6. The app (UI + API), the LLM

```powershell
uv run python run.py start        # UI http://localhost:7860  +  API http://localhost:8000/docs   (Ctrl+C stops both)
uv run python run.py vllm-serve   # the local LLM in WSL2 on http://127.0.0.1:8001 (otherwise the agents use their fallback)
```

The **Journey** tab (step-by-step for one line or a batch) ingests a dummy source from `data/fsdr_dummy/<D>/`. To create one:

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --mode small --layout folders --out data/fsdr_dummy/2026-10-22
```

## 7. The full weekly flow (built phase by phase)

| Order | Command | Saves in `data/runs/<D>/` | Phase |
|---|---|---|---|
| 1 | `fsdr-load --date <D>` (data/training/<D> + data/scoring/<D>) | `train/T0_load/` | **built** |
| 2 | `fsdr-features` (T2 features, T3 segments) | `train/T2_features/`, `train/T3_segments/` | **built** |
| 2b | `fsdr-targets` (your lists or `--generate`), then `fsdr-run-targets` = panel + each target (P6 + final) + push (V4.0, LLM fallbacks) | `score/P1_panel/`, `score/P6_offers_target1/`, `…_target2/`, and `data/output/<D>/` (target1 / target2 / control / push) | **built** |
| 3 | `fsdr-train --weekly` (T1 feedback, T4 evidence … T8 bundle) | `train/T1_feedback/`, `T4_evidence/ … T8_validate/`, `models/bundle_<D>/` | 4–5, 9 |
| 4 | `fsdr-score --bundle <D>` (P0 … P10, batches of 1,000) | `score/P0_bundle/ … P10_report/` | 5–8 |
| 5 | — (written by scoring on audit PASS) | `final/input_flytxt.csv.gz` (≈100k rows, 38 columns), `input_flytxt_t1c/_t1f/_t2c/_t2f`, control, `summary.json` | 7 |
| 6 | `fsdr-approve` then `fsdr-push` | `final/approval.json`, the export | 9 |
| — | `fsdr-run-week` = 1 to 4, stops before approval | everything above | 9 |
| any time | `notebook` / `notebook-html` | `fsdr_results_<D>.html` | **built** |

The V1.0 commands (`fsdr-train`, `fsdr-score`, `fsdr-recommend`, `fsdr-weekly`, `fsdr-simulate`, …) still work on a run folder until the V2.0 pipelines replace them. `uv run python run.py --help` lists every command.

## 8. Where everything is, what is never committed

| Folder | What | In git? |
|---|---|---|
| `data/training/<D>/`, `data/scoring/<D>/` | input tables (5 folders each), the target lists | no |
| `data/output/<D>/` | each target's files, the control, the file to push | no |
| `data/input/<D>/` | V3.0 single input folder (still accepted) | no |
| `data/cache/<D>/` | Parquet cache, 16 partitions | no |
| `data/runs/<D>/` | every step's outputs, the final file, the HTML report | no |
| `data/fsdr/`, `data/fsdr_dummy/` | V1.0 run folders, journey sources | no |
| `.env` | local settings and the vLLM key | no |
| `docs/fsdr_reference/fsdr_original_*.txt` | raw extracts of the original code and SQL | no |
| `notebooks/` | the step-by-step and results notebooks (without outputs) and their builders | yes |

## 9. When something goes wrong

| Symptom | What to do |
|---|---|
| `T0_load BLOCKED` | read the explanation, or `validation_report.md` (blocking issues come first); fix the file, re-run `fsdr-load` |
| `N table(s) are not .csv.gz` (T0 blocked) | real inputs must be the Dataiku `.csv.gz` export; for hand-made test files set `FSDR_ALLOW_TEST_FORMATS=true` in `.env` |
| `fsdr-offers` exits with 1, `audit_groups.json` FAIL | a group rule was broken (A-12 / A-17 / A-30 / A-31): the failing check lists examples; nothing may be pushed |
| `found in usage/, expected in catalog/` | the table is loaded anyway (warning); move the file to keep the folders clean |
| `found under training/, expected under scoring/` | the table is on the wrong side (warning, `misfiled_tables.json`); move it |
| `BLOCKED: P6_offers_target2 … is missing` | score that target first: `fsdr-score-target --target target2` |
| `push BLOCKED` | a target is blocked or not scored: `data/output/<D>/push/BLOCKED.txt` names it |
| `N files for one table` | two copies of a table (for example `.json` and `.csv`): delete one |
| `stale data: the manifest is for …` | `_manifest.json` is for another week: wrong folder or an old manifest |
| `checksums: N file(s) CHANGED` in the notebook | a cache file was changed after T0: `fsdr-load --force` rebuilds it |
| the notebook says `MISSING: run fsdr-load first` | no step yet for that date, or `RUN_DATE` points to another week |
| `BLOCKED: no T0_load for <D>` (fsdr-features) | run `fsdr-load` for that date first, or pass `--date` |
| many `*_MISSING` scores in T2 | a model-score file covers few lines: check its coverage in `validation_report.md` (the missing scores are imputed, flagged, and never trigger "at risk") |
| `API status degraded` | vLLM is not running: start `vllm-serve`, or keep the rule-based fallback |
