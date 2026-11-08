# FSDR V3.0: run it from the terminal, check the results in the notebook

The same flow in the web interface, tab by tab: [USER_GUIDE_INTERFACE.md](USER_GUIDE_INTERFACE.md).

Every command runs from the project folder. They work the same in PowerShell, Git Bash or the VS Code terminal.

```powershell
cd C:\Abdelouaheb\perso\Data_science_2024_projects\2026\llm_recommendation_engin_telco\nbx-python-upgrade-V3.0
```

Status: **Phases 1–3 are built**, plus the **V3.0 changes**: the `.csv.gz` file contract and the four test groups (panel P1, offers P6 with the final rules and the LLM agents' fallbacks). The commands marked *(Phase N)* arrive with that phase. Section 7 lists them so you can see the full flow.

---

## 1. Once: install

```powershell
uv sync                          # creates .venv from uv.lock (Python 3.12, every package incl. Jupyter)
uv run python run.py check       # Python, packages, data files OK?
uv run python run.py test        # the whole test suite (~4 min, no network, no GPU)
```

`.env` holds your local settings, including the vLLM key. It is never committed; copy `.env.example` to start one.

## 2. Get the input data of a week

The data of run date D goes to `data/input/<D>/`, one file per table, in 5 folders:

```
data/input/2026-10-22/
├── usage/          18 tables  activities_*, subs_amt_*, paygo, usage_data_voice_payg_tg
├── models/         11 tables  the model scores
├── subscription/    6 tables  base_prepaid, accounts, MSISDN map, subscriptions_tg, purchases
├── catalog/         3 tables  btl_catalog, scripts, conflicting offers
├── others/          9 tables  party, customer_profile, location, consent, device, targeting, sample
└── _manifest.json
```

Format: **`<table>.csv.gz`** (the Dataiku compressed CSV export, *FSDR CSV standard*), optionally with `<table>.schema.json` next to it. The file name is the exact table name. `.csv`, `.json`, `.jsonl` and `.parquet` are read too, but only for generated data (its `_manifest.json` says `source: generator`) or with `FSDR_ALLOW_TEST_FORMATS=true` in `.env`; otherwise T0 blocks.

**Real data:** copy the exported `.csv.gz` files into those folders.

**Dummy data:** use the generator.

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --mode small                    # 2,000 lines, .csv.gz (~10 s)
uv run python run.py fsdr-generate --date 2026-10-22 --mode medium                   # 100,000 lines, .csv.gz (~80 s, 93 MB)
uv run python run.py fsdr-generate --date 2026-10-22 --mode full                     # 1,000,000 lines, .csv.gz
uv run python run.py fsdr-generate --date 2026-10-22 --mode full --format parquet    # same in Parquet (faster to write)
uv run python run.py fsdr-generate --date 2026-10-22 --weeks 3 --mode small          # 3 consecutive weeks from one world
uv run python run.py fsdr-generate --date 2026-10-22 --mode small --scenario data_quality_issues   # data with defects
```

The default format is `csv.gz`, as production. `--format json` would be about 12 GB at 1M lines. Add `--force` to overwrite a folder that holds non-generated data.

## 3. Step T0: validate, check completeness, build the cache

```powershell
uv run python run.py fsdr-load --data-dir data/input/2026-10-22
```

It prints `DONE` or `BLOCKED` with a plain explanation, and saves everything in `data/runs/2026-10-22/train/T0_load/`:

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
- **Validation only**, without the cache: `uv run python run.py fsdr-validate --data-dir data/input/2026-10-22`.

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

## 5. V3.0: the panel and the four groups

```powershell
uv run python run.py fsdr-panel                 # P1: T1C / T1F / T2C / T2F / control (needs fsdr-features)
uv run python run.py fsdr-offers                # P6: code-2 tables + the 8 reco tables, .csv.gz
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

`data/runs/<D>/score/P6_offers/`: the 13 tables as `.csv.gz` with their `.schema.json` and `_outputs_manifest.json`,
`group_report.csv` (offers per rule / pass, 10.1 retries, LLM coverage at the 10 h budget), `traces.csv` (2 customers
of the same type per group: action → candidates → score → pass / rule → offers), `audit_groups.json` (A-12, A-12b,
A-17, A-21, A-30, A-31). Exit code 1 if the audit fails (push blocked).

Until the LLM agents (Phases 5–6), the action is the group's rule action and the offer score is the NBO fallback
(own past purchase, then buyer share); every row says so in `decided_by` / `score_source`.

Sizes and switches (`.env`): `FSDR_T1_CONSTRAINT_SIZE`, `FSDR_T1_FREE_SIZE`, `FSDR_T2_CONSTRAINT_SIZE`,
`FSDR_T2_FREE_SIZE`, `FSDR_CONTROL_SIZE`, `FSDR_T2_SPLIT` (false = one 50k T2 with the CC rules),
`FSDR_LLM_WEEKLY_BUDGET_H`, `FSDR_GUARD_MODE_TARGET1`.

## 5a. Your own target lists, and the files to push

**Upload folder.** Put one list per group in the run's input folder (any extra columns are kept in the details):

```
data/input/2026-10-22/targets/
├── target1/t1_constraint.csv.gz     T1C  production code 3
├── target1/t1_free.csv.gz           T1F  production code 4
├── target2/t2_constraint.csv.gz     T2C  challenger CC-v1
├── target2/t2_free.csv.gz           T2F  challenger CF-v1
└── control/control.csv.gz           never contacted
```

Each file needs an `access_method_val` (or `msisdn`) column. `.csv.gz` in production; `.csv` / `.json` for tests.
When the folder holds lists, the panel uses them; otherwise it draws the groups at random (section 5).

```powershell
uv run python run.py fsdr-targets --generate      # dummy lists for testing (stratified draw of the eligible base)
uv run python run.py fsdr-targets                 # check the lists that are there
uv run python run.py fsdr-run-groups              # panel (your lists) -> scoring per group -> final files
uv run python run.py fsdr-final                   # only the final step again
```

Through the app (API): `POST /fsdr/targets/<D>/<group>` (file upload, group = t1c / t1f / t2c / t2f / control),
`GET /fsdr/targets/<D>` (lists + check), `POST /fsdr/groups/<D>/run`, `GET /fsdr/final/<D>`.

**Checks of the lists:** a line in two groups or a missing list **blocks**; a line that is not an 8-digit MSISDN, not in
the eligible base, in Nursering (B-02) or in another campaign this week (H-05) is **dropped** and listed in
`score/P1_panel/targets_rejected.csv` with its reason; a duplicate inside a list or a size off the configured size by
more than 2 % is a warning.

**What is saved every run** in `data/runs/<D>/final/`:

| File | What |
|---|---|
| `push/input_flytxt.csv.gz` | **the file to push**: the 38 columns of code 5, all 4 groups, one row per MSISDN. Written only when every check passes; otherwise `push/BLOCKED.txt` says why (H-08) |
| `target1/input_flytxt_t1c.csv.gz`, `…_t1f`, `input_flytxt_target1.csv.gz` | the same rows, per group and for Target 1 |
| `target2/input_flytxt_t2c.csv.gz`, `…_t2f`, `input_flytxt_target2.csv.gz` | the same, Target 2 |
| `target1/details_t1c.csv.gz`, `details_t1f`, `target2/details_t2c`, `details_t2f`, `control/details_control` | **the details ("big data")**: one row per line of the group, ≈ 220 columns: group, stratum, profile and bands, segment and its triggers, `action_original` / `action_agent` / action sent / agreement / 10.1 retry, every feature (scores, missing flags, revenue, trends), B-26 and F-05 anchors, the recommended offers per validity with their pass or rule, the 6 slots with their reason (recommended / favourite / forced) and the rule that cleared them (F-09 / F-11 / F-12), in the file or not and why |
| `details_all.parquet` | all the details in one table (for the notebook / Excel via Power Query) |
| `control/control_<D>.csv.gz` | the control list kept for measurement |
| `final_rules_report.csv` | per group: lines, rows in the file, F-04 drops, favourites, forced bundles, F-09 / F-11 / F-12 clearings |
| `_outputs_manifest.json`, `*.schema.json`, `summary.json` | rows and SHA-256 of every file, the column types, the checks (A-20 … A-24) |

The scoring files of each group stay in `score/P6_offers/` (the code-2 tables and the 8 reco tables).

## 5b. Check the results in the notebook

The notebook `notebooks/fsdr_results.ipynb` reads what the commands saved; it computes nothing new. Its sections:

1. the 5 input folders: tables, formats, sizes, manifest
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
uv run python run.py fsdr-generate --date 2026-10-22 --mode small --out data/fsdr_dummy/2026-10-22
```

## 7. The full weekly flow (built phase by phase)

| Order | Command | Saves in `data/runs/<D>/` | Phase |
|---|---|---|---|
| 1 | `fsdr-load --data-dir data/input/<D>` | `train/T0_load/` | **built** |
| 2 | `fsdr-features` (T2 features, T3 segments) | `train/T2_features/`, `train/T3_segments/` | **built** |
| 2b | `fsdr-targets` (your lists or `--generate`), then `fsdr-run-groups` = panel + offers + final (V3.0, LLM fallbacks) | `score/P1_panel/`, `score/P6_offers/`, `final/` (push / target1 / target2 / control) | **built** |
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
| `data/input/<D>/` | input tables (5 folders) | no |
| `data/cache/<D>/` | Parquet cache, 16 partitions | no |
| `data/runs/<D>/` | every step's outputs, the final file, the HTML report | no |
| `data/fsdr/`, `data/fsdr_dummy/` | V1.0 run folders, journey sources | no |
| `.env` | local settings and the vLLM key | no |
| `docs/fsdr_reference/fsdr_original_*.txt` | raw extracts of the original code and SQL | no |
| `notebooks/` | the results notebook (without outputs) and its builder | yes |

## 9. When something goes wrong

| Symptom | What to do |
|---|---|
| `T0_load BLOCKED` | read the explanation, or `validation_report.md` (blocking issues come first); fix the file, re-run `fsdr-load` |
| `N table(s) are not .csv.gz` (T0 blocked) | real inputs must be the Dataiku `.csv.gz` export; for hand-made test files set `FSDR_ALLOW_TEST_FORMATS=true` in `.env` |
| `fsdr-offers` exits with 1, `audit_groups.json` FAIL | a group rule was broken (A-12 / A-17 / A-30 / A-31): the failing check lists examples; nothing may be pushed |
| `found in usage/, expected in catalog/` | the table is loaded anyway (warning); move the file to keep the folders clean |
| `N files for one table` | two copies of a table (for example `.json` and `.csv`): delete one |
| `stale data: the manifest is for …` | `_manifest.json` is for another week: wrong folder or an old manifest |
| `checksums: N file(s) CHANGED` in the notebook | a cache file was changed after T0: `fsdr-load --force` rebuilds it |
| the notebook says `MISSING: run fsdr-load first` | no step yet for that date, or `RUN_DATE` points to another week |
| `BLOCKED: no T0_load for <D>` (fsdr-features) | run `fsdr-load` for that date first, or pass `--date` |
| many `*_MISSING` scores in T2 | a model-score file covers few lines: check its coverage in `validation_report.md` (the missing scores are imputed, flagged, and never trigger "at risk") |
| `API status degraded` | vLLM is not running: start `vllm-serve`, or keep the rule-based fallback |
