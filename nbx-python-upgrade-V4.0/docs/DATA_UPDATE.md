# Data update: running FSDR V4.0 on real data

This guide takes you from the real exports to the final file saved in `data/output/<D>/final/` (the engine only saves files; it sends nothing). Follow the steps in order the first time. After that, use the short checklist at the end (§10) each week.

The steps assume the project is in `D:\production\nbx-python-upgrade-V4.0` and the environment is already set up ([SETUP_NEW_LAPTOP.md](SETUP_NEW_LAPTOP.md) §1–6). Every command runs from the project folder in PowerShell.

---

> **Simpler: one weekly folder (the GPU notebook).** If you export fresh `*_auto` tables every week, you need neither the `*_score` tables nor a separate scoring folder. Put everything for the week in `data/weekly/<D>/` and run `notebooks/03_gpu_agents_step_by_step.ipynb`. It runs every agent, with the LLM on your GPU. Every week is then a week-0 run on fresh history. The tables to copy are in [GPU_AGENTS_NOTEBOOK.md](GPU_AGENTS_NOTEBOOK.md) §2. The rest of this guide describes the two-folder layout of notebooks 01 and 02.

## 0. The full flow, week by week

```
WEEK 0 · run date D (e.g. 2026-11-19)
  data/training/D/  (all tables, with the *_auto history)  +  data/scoring/D/  (this week's tables + 5 lists)
     │
     ├─ 1. TRAINING on the whole base ............ notebook 01  (or fsdr-load + fsdr-features)
     │     T0  validate the 2 folders, completeness, Parquet cache   → data/runs/D/train/T0_load/, data/cache/D/
     │     T2  features of EVERY line (B-01 … B-12)                  → data/runs/D/train/T2_features/
     │     T3  segment + action_original + action_agent of EVERY line → data/runs/D/train/T3_segments/
     │
     ├─ 2. THE WHOLE BASE IS SCORED at the action level: after T3, every eligible line has its segment,
     │     micro-segment, value band and its two actions (segments/part-NN.parquet). Offers are not built
     │     for the whole base, only for the lines of the target lists (step 3).
     │
     └─ 3. SCORING OF THE TARGETS .................. notebook 02  (or fsdr-run-targets)
           P1  panel = your 5 lists joined with T3                   → data/runs/D/score/P1_panel/
           P6 + code 5 Target 2 (T2C, T2F)                           → data/output/D/target2/
           final file (Target 2 PASS)                                → data/output/D/final/
           → check it, then hand over final/input_flytxt.csv.gz yourself

WEEK 1 · run date D+7 (e.g. 2026-11-26): NEW folders, same 3 steps
  data/training/D+7/  (14 refreshed tables, NO *_auto history)  +  data/scoring/D+7/  (17 tables + 5 new lists)
     ├─ 1. TRAINING again (pipeline train_weekly: features from the *_score tables)
     │     T3 takes the agent dictionary of week 0 from data/runs/D/ (keep the old runs)
     ├─ 2. whole base: segments and actions of this week
     └─ 3. targets: P1 → Target 2 (a + b) → final file          → data/output/D+7/…

WEEK 2 · D+14, and so on: same as week 1.
```

**What changes each week:**

| | Week 0 (D) | Every next week (D+7, D+14, …) |
|---|---|---|
| Folders | `data/training/D/`, `data/scoring/D/` | **new** folders with the new date; the old ones stay |
| Training tables | all of §3.1, **with** the `*_auto` history tables (`activities_*_auto`, `subs_amt_auto`, `party_auto`, `access_method_id_auto`, `targeting_dataiku_auto`) | 14 refreshed tables: `base_prepaid`, `consumer_aact_account`, `subs_btl_auto`, the 11 model tables, plus `btl_catalog`. **Leave the `*_auto` history out** |
| Scoring tables | the 17 of §3.2 | the same 17, this week's extract |
| Target lists | 5 lists | 5 new lists |
| Mode found by T0 | `pipeline train_week0` | `pipeline train_weekly` |
| Features from | the `*_auto` tables | the `*_score` tables |
| Agent dictionary | v0, a copy of the production dictionary | the latest version, from the previous run in `data/runs/` |
| Retrain? | yes (T0, T2, T3) | **yes, every week**: scoring reads only the training of its own date |
| Output | `data/output/D/` | `data/output/D+7/` |

The mode is chosen by the files: if `activities_w1_auto`, `subs_amt_auto` and `targeting_dataiku_auto` are present, T0 runs week 0 again and builds the features from those old history tables. So leave them out from week 1 on, and check that the T0 cell prints `pipeline train_weekly`. The missing history tables then show as warnings ("not needed by the train_weekly pipeline"), not as blocking issues.

In V4.0 nothing learns from last week's results yet. The weekly feedback (`fsdr-feedback`, rewards at D+7) is a V1.0 terminal command, not part of these notebooks. Keep every run's `data/runs/<D>/` and `data/output/<D>/` anyway: they are needed to measure uplift against the control group later.

---

## 1. Before you start

1. **Choose the run date `D`.** It is the date the exports were extracted for (a Thursday in production), written `YYYY-MM-DD`, for example `2026-11-19`.
   - The folder names carry this date, and the engine reads the run date from them.
   - The weeks W1…W4 and months M1…M3 are counted back from it. A wrong date gives "date window" warnings.
2. **Use a new date, not the dummy one.** The dummy runs used `2026-10-22`. If you must reuse a date that already has dummy runs, delete its dummy folders first (§9).
3. **Set the secret salt.** Open `.env` and set `FSDR_PII_SALT` to a long random secret, which pseudonymises the customers. Keep `FSDR_LOCAL_ONLY=true`.
4. **Real data stays on this laptop.** Never put `data/` or a notebook with outputs in an e-mail, a zip or a shared drive: they hold MSISDNs.

---

## 2. Create the folders of the run date

```powershell
cd D:\production\nbx-python-upgrade-V4.0
$D = "2026-11-19"                        # your run date
foreach ($side in "training", "scoring") {
  foreach ($f in "usage", "models", "subscription", "catalog", "others") {
    New-Item -ItemType Directory -Force "data\$side\$D\$f" | Out-Null } }
foreach ($t in "target2\a_constraint", "target2\b_free") {
  New-Item -ItemType Directory -Force "data\scoring\$D\$t" | Out-Null }
```

The result:

```
data\training\2026-11-19\   usage\  models\  subscription\  catalog\  others\
data\scoring\2026-11-19\    usage\  models\  subscription\  catalog\  others\
                            target2\a_constraint\  target2\b_free\
```

---

## 3. Copy the exports

**File name = the exact table name** (case included) + the format. For example, `activities_w1_auto.csv.gz` and `DAIM_CHURN_GLOBAL_SCORING_TD.prediction.csv.gz`.

**Formats:**
- `.csv.gz` is the production format.
- **Only `.csv.gz` for real data:** validation blocks `.csv`, `.parquet`, `.json` and `.jsonl` (test formats), unless `FSDR_ALLOW_TEST_FORMATS=true` is set in `.env`.
- Use one file per table: two files for the same table (for example `.csv` and `.csv.gz`) block T0.

**CSV rules:** UTF-8, comma separator, one header row, a dot as the decimal point, dates `YYYY-MM-DD`, and an empty field for missing values. IDs are read as text, so leading zeros are kept.

**Columns:** the columns of each table are in [docs/fsdr_json_schemas/](fsdr_json_schemas/) (`<table>.json`). Key columns must be present and never empty.

### 3.1 Training side: `data\training\<D>\`

| Folder | Table | Needed |
|---|---|---|
| `subscription\` | `base_prepaid` | yes |
| `subscription\` | `consumer_aact_account` | yes |
| `subscription\` | `access_method_id_auto` | week 0 |
| `subscription\` | `subs_btl_auto` | yes |
| `usage\` | `activities_w1_auto`, `activities_w2_auto`, `activities_w3_auto`, `activities_w4_auto` | week 0 |
| `usage\` | `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto` (**`auto_m2`**: the name of the real table) | week 0 |
| `usage\` | `subs_amt_auto` | week 0 |
| `usage\` | `paygo_activity_auto` | optional |
| `models\` | the 11 model tables: `DAIM_CHURN_GLOBAL_SCORING_TD.prediction`, `DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction`, `DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored`, `DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored`, `PROPENSITY_REACTIVATION_15D_401_SCORING.Scored`, `APPETENCY_DATA_204_SCORING_TD.prediction`, `APPETENCY_VOICE_204_SCORING_TD.Scored`, `CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring`, `REV_DECREASE_204_SCORING_TD.predictions`, `DAIM_FRAGILITY_DATA_SCORING_TD.prediction`, `DAIM_FRAGILITY_VOICE_SCORING_TD.prediction` | yes |
| `others\` | `party_auto` | week 0 |
| `others\` | `targeting_dataiku_auto` | week 0 |
| `others\` | `location_auto` | optional |
| `others\` | `DEVICEBEHAVIORSEGMENTATION.dv_segment` | optional |
| `others\` | `customer_profile` | optional |

### 3.2 Scoring side: `data\scoring\<D>\`

| Folder | Table | Needed |
|---|---|---|
| `subscription\` | `subscriptions_tg` | yes |
| `subscription\` | `sub_btl_3months` (the purchases of the last 270 days: the offer universe and the offer score) | yes |
| `usage\` | `usage_data_voice_payg_tg` | yes |
| `usage\` | `activities_w1_score` … `activities_w4_score`, `activities_m1_score` … `activities_m3_score` | yes |
| `usage\` | `subs_amt_score` | yes |
| `catalog\` | `btl_catalog` (also copy it to `data\training\<D>\catalog\`) | yes |
| `catalog\` | `BTL_script_english_arabic_ok` | yes |
| `catalog\` | `conflicting_offers_h` | yes |
| `others\` | `targeting_auto` (the campaigns of the operator: a line in a campaign that started in D-7 … D is dropped, H-05) | yes |
| `others\` | `party_scoring` | yes |
| `others\` | `consent_contact` | optional, but use it for real runs: a line with `dnd = 1` or `marketing_opt_in = 0` is never contacted; without the table, no consent filter is applied |
| `others\` | `sample_cg_tg_new_7oct2026` (the sample table; change its name with `FSDR_TARGET_SAMPLE_TABLE` in `.env`) | optional |

**Week 0 or a weekly run.** The engine chooses by itself:
- If `activities_w1_auto`, `subs_amt_auto` and `targeting_dataiku_auto` are present, it is **week 0**: the features come from the `*_auto` tables.
- Otherwise it is a **weekly run**: the features come from the `*_score` tables.

The first real run is normally week 0, so copy all the "week 0" tables.

A table put on the wrong side is still read, with a warning. Move it to the right folder anyway.

---

## 4. Put the target lists

You can copy one file per group:

| File | Group | Rules |
|---|---|---|
| `data\scoring\<D>\target2\a_constraint\target2_a.csv.gz` | T2C | challenger CC-v1, `action_agent` |
| `data\scoring\<D>\target2\b_free\target2_b.csv.gz` | T2F | challenger CF-v1 |

Or upload the lists in the web interface, tab "FSDR Targets & Output", step 2a.

**Rules for the lists:**
- One row per line and at least the column `access_method_val` (the 8-digit MSISDN, without a country prefix).
  - Any other column (stratum `Final`, plan, nationality, …) is kept and copied to the details file.
- A line must be in only one list.
- Configured sizes: 50,000 for group a (T2C) and 50,000 for group b (T2F); no control group and no Target 1 in V4.0 (`FSDR_CONTROL_SIZE=20000` adds a control list `control\control.csv.gz`). A list more than 2% off its size only gets a warning.
- **Real lists are recognised by themselves.** The dummy generator records a checksum of every list it writes in `_targets_manifest.json`. A list you copy over a dummy one has another checksum, so it is treated as yours: it is never overwritten, and it is checked against the configured sizes. Deleting `_targets_manifest.json` is no longer needed (it does no harm).

**Without lists:** if you have no lists for this run, you can let the engine draw them from the eligible base (dummy lists). Do this only for a test, never for a real campaign.

---

## 5. Check the files before running

```powershell
uv run python run.py check
```

It must show `[ok]` for `data\training`, `data\scoring` and `data\output`.

Count the files:

```powershell
(Get-ChildItem -Recurse -File "data\training\$D").Count    # 26 for week 0, btl_catalog included (+ the optional ones)
(Get-ChildItem -Recurse -File "data\scoring\$D").Count     # 17 tables + up to 5 lists (+ the optional ones)
```

---

## 6. Run the training notebook

1. Start JupyterLab with `uv run jupyter lab`, then open `notebooks/01_training_step_by_step.ipynb`.
2. **Cell "Your machine":** check it prints `kernel  : OK, the project's .venv`.
3. **Settings cell:** change only these two values:
   ```python
   RUN_DATE = "2026-11-19"     # your D
   GENERATE = False            # REAL DATA: never True
   ```
   With `GENERATE = True`, the notebook stops ("holds real data"), so your exports cannot be overwritten. Still, set it to `False`.
4. **Run Part A** (the engine cells), then the steps of Part B one by one.
5. **Step 1 (input folders):** `misfiled` and `duplicates` must say `none`, and `tables found` must count every table you copied.
6. **Step 2 (T0) is the important check:**
   - `T0_load DONE` means the data is usable. `BLOCKED` means read the explanation printed below it.
   - The full report is in `data\runs\<D>\train\T0_load\validation_report.md`: blocking issues first, then warnings.
   - `completeness.csv` lists every needed table as ok / missing / empty.
   - Fix the files, then run Step 2 again. It re-runs by itself when the files have changed.
7. **Step 4 (T2):**
   - `eligible lines` should be close to your active prepaid base.
   - Look at `missing model scores`. A high share means a model table does not match the base: check its MSISDN / ACCESS_METHOD_ID.
8. **Step 5 (T3):** look at the lines per segment. A segment with 0 lines, or with almost every line, means a column is empty or has the wrong units.
9. **Step 6:** T0, T2 and T3 must all be **done**.

**Large base (1M lines):** the notebook runs T2 / T3 with one worker. For speed, run training in the terminal, then use the notebooks to look at the results (steps already done are skipped):

```powershell
uv run python run.py fsdr-load --date $D         # T0, about 14 min for 1M lines
uv run python run.py fsdr-features --date $D     # T2 + T3, about 1-2 min
```

---

## 7. Run the scoring notebook

1. Open `notebooks/02_scoring_step_by_step.ipynb`.
2. **Settings cell:**
   ```python
   RUN_DATE = "2026-11-19"
   GENERATE_TARGETS = False    # REAL LISTS (with True they are kept anyway, and the cell says so)
   ```
   With `GENERATE_TARGETS = True` left on by mistake, your lists are **kept**: the cell prints `the folder holds YOUR lists … they are kept and used`, then `lists are YOURS`. Dummy lists only replace dummy lists. To replace your lists with dummy ones on purpose, delete them first (or `fsdr-targets --generate --force`).
3. **Run Part A**, then the steps.
4. **Step 2 (target lists):**
   - every group shows a file name, never `MISSING`;
   - `lines read / kept / dropped`;
   - the checks: **blocking** stops the run. Each **dropped** line has its reason in `targets_rejected.csv`: not an 8-digit MSISDN, not in the eligible base, Nursering, or in another campaign this week.
   - Many lines dropped as "not in the eligible base" usually means the MSISDN format differs between the lists and `base_prepaid`, for example a `968` country prefix or a leading zero.
5. **Step 3 (P1):** sizes per group, the balance of the panel, and `SOURCE = uploaded`.
6. **Target 2, group a and group b (Steps 4–5):**
   - the group audit must be `PASS`;
   - the final files are written to `data\output\<D>\target2\`;
   - `output_checks.json` must have every check passed.
7. **Steps 6–7 (Target 2):** the same checks, written to `data\output\<D>\target2\`.
8. **Step 8 (final file):**
   - `data\output\<D>\final\input_flytxt.csv.gz` is written only if both targets PASS and no MSISDN is in two targets;
   - otherwise you get `final\BLOCKED.txt` with the reason.
9. **Step 9:** the last checks of the final file: 38 columns in order, 8-digit MSISDNs, no control line, the offers per line.
10. **Cell "One line, end to end":** pick a few lines and check with the business that the segment, action, anchor and the 6 offers make sense.

**Or in the terminal:**

```powershell
uv run python run.py fsdr-targets --date $D        # checks the lists only (NO --generate)
uv run python run.py fsdr-run-targets --date $D    # P1, Target 2 (a + b), final file
```

---

## 8. Validate and hand over

1. **Read the reports:**
   - `data\output\<D>\target2\`: `output_checks.json`, `audit_groups.json`, and `details_<target>.parquet` (every line with all its columns);
2. **Get the business to validate** a sample of `details_*`.
3. **Once checked**, hand over `data\output\<D>\final\input_flytxt.csv.gz` yourself. The engine saves it in the campaign format; it sends nothing.
4. **Keep the run** (`data\runs\<D>\`, `data\output\<D>\`). Next week's feedback and uplift are measured against it.

---

## 9. Cleaning up dummy data (only if you reuse a dummy date)

Delete only the folders of that date, and check the date before you delete:

```powershell
$D = "2026-10-22"
Remove-Item -Recurse -Force "data\training\$D", "data\scoring\$D", "data\cache\$D", "data\runs\$D", "data\output\$D"
```

Never delete the folders of a real run you still need.

---

## 10. Weekly checklist

1. Create the folders of the new `D` (§2).
2. Copy the exports with their exact table names (§3), then the 5 lists (§4).
3. Run `uv run python run.py check`.
4. Training notebook: `RUN_DATE = D` and `GENERATE = False`. T0 must be DONE; read `validation_report.md`.
5. Scoring notebook: `RUN_DATE = D` and `GENERATE_TARGETS = False`. Both audits must PASS, and `final\input_flytxt.csv.gz` must exist.
6. Business validation of `details_*`, then you hand over `final\input_flytxt.csv.gz`.

---

## 11. Common problems

| What you see | Cause and fix |
|---|---|
| `T0_load BLOCKED`, "missing" in `completeness.csv` | A needed table is missing or has a different file name. Rename the file to the exact table name (§3). |
| `duplicates` in Step 1 | Two files for one table, for example `.csv` and `.csv.gz`. Keep one. |
| "date window" warnings | `RUN_DATE` does not match the extraction date of the exports. Use the right `D` (folder name and `RUN_DATE`). |
| Many "missing model scores" in T2 | The model tables use other IDs than `base_prepaid`. Check `access_method_val` / `access_method_id` in the model files. |
| Almost every list line "not in the eligible base" | The MSISDN format differs between tables (country prefix, leading zero, `.0` added by Excel). Export the IDs as text, 8 digits. |
| Wrong numbers after opening a CSV in Excel | Excel changes IDs and dates. Never save an export from Excel; take the file straight from the database. |
| Step 2 of scoring: `MISSING` for a group | The list file name or folder is wrong. It must be exactly `target2\a_constraint\target2_a.csv.gz` (and so on, §4). |
| Size warnings on real lists | Your lists are more than 2 % off the configured sizes (50,000 per group: `.env`). A group far below its size can make the group audit FAIL and block the final file; Step 9 then prints the reason. |
| `final\BLOCKED.txt` | One target did not PASS, or one MSISDN is in two targets. The file says which. Fix the lists and run Steps 2–8 again. |
| The kernel dies or "access denied" | See [SETUP_NEW_LAPTOP.md](SETUP_NEW_LAPTOP.md) §13. |
