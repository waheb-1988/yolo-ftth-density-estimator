# Run the real week, step by step: from `data_raw/` to 50k scored customers

Three notebooks, all in `notebooks3/`:

| Notebook | What it does |
|---|---|
| `02_prepare_week.ipynb` | `data/data_raw/` → `data/weekly/<D>/`, and a test copy of 1k / 5k / 10k lines |
| `03_agents_nba_fast_ollama_gpu.ipynb` | every agent on the GPU with Ollama, and the final file |
| `04_agents_nba_fast_vllm_gpu.ipynb` | the same with vLLM (faster); use 03 **or** 04 |

The example scores **group a only (T2C, `target2_a`, ~50k)**. Step 7 explains how to score group b, or both groups.
Paths are relative to the project folder `nbx-python-upgrade-V4.0`.

**The order:** run 02 and 03 at `RUN_SIZE = 1_000` → `5_000` / `10_000` (optional) → `"all"`. Use **the same
`RUN_SIZE` in 02 and 03**.

---

## Step 0 · Once: the environment and the LLM

1. **The Python environment** (`.venv` of V4, Python 3.12): see [docs/UV_ENV_SETUP.md](../docs/UV_ENV_SETUP.md).
2. **Ollama** (notebook 03):
   - check that `ollama list` shows `qwen2.5:7b-instruct-q8_0`;
   - once, then restart Ollama:

     ```powershell
     setx OLLAMA_NUM_PARALLEL 8
     setx OLLAMA_FLASH_ATTENTION 1
     ```
3. **Or vLLM** (notebook 04): see [docs/VLLM_AGENTS_NOTEBOOK.md](../docs/VLLM_AGENTS_NOTEBOOK.md). Start the server
   with `uv run python run.py vllm-serve` before the notebook.

---

## Step 1 · Paste the exports into `data/data_raw/`

Copy every file **as exported**, with its Dataiku name: `_hdfs` and the new names are recognised, so rename
nothing. `.csv` and `.csv.gz` both work.

| Group | Your export names |
|---|---|
| base | `base_prepaid_hdfs`, `consumer_aact_account_hdfs`, `access_method_id_auto_hdfs`, `customer_profile` |
| training (`*_auto`, at t2) | `party_auto_hdfs`, `subs_amt_auto_hdfs`, `activities_w1_auto_hdfs` … `activities_w4_auto_hdfs`, `activities_m1_auto_hdfs`, `activities_auto_m2_hdfs`, `activities_m3_auto_hdfs` |
| scoring (`*_score`, at t1) | `party_scoring_hdfs`, `subs_amt_score_hdfs`, `activities_w1_score_hdfs` … `activities_w4_score_hdfs`, `activities_m1_score_hdfs`, `activities_m2_score_hdfs`, `activities_m3_score_hdfs` |
| last campaign | `targeting_dataiku_auto_hdfs` (the history: step 4c keeps the latest campaign), `subs_btl_test_hdfs` (the purchases), `targeting_auto_hdfs` (may be empty) |
| the 11 models | `DAIM_CHURN_GLOBAL_SCORING_prediction`, `DAIM_EARLY_CHURN_0_30_SCORING_Early_Churn_0_30_Prediction`, `DAIM_EARLY_CHURN_30_60_SCORING_scoring_data_scored`, `DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_Scored`, `PROPENSITY_REACTIVATION_15D_401_Scored`, `APPETENCY_DATA_204_SCORING_prediction`, `APPETENCY_VOICE_204_SCORING_Scored`, `CHURN_ONE_DAY_ACTIVE_107_Churn_One_Day_Scoring`, `REV_DECREASE_204_SCORING_predictions`, `DAIM_FRAGILITY_DATA_SCORING_prediction`, `DAIM_FRAGILITY_VOICE_SCORING_prediction` |
| offers and final file | `btl_catalog_hdfs`, `SEGMENT_Conflicting_Offers_h`, `sub_btl_3months_hdfs`, `subscriptions_tg_hdfs`, `usage_data_voice_payg_tg_hdfs`, **`BTL_script_english_arabic_ok`** |
| optional | `paygo_activity_auto_hdfs`, `location_auto_hdfs`, `dv_segment`, `consent_contact` |
| **the lists** | `target2_a` (group a, ~50k), and `target2_b` (group b) only if you score it |

- **`BTL_script_english_arabic_ok`** (the offer scripts) is not in your list of 2026-11-18, but **the final file
  needs it**. Without it, step 2 of 02 says `required table(s) missing`, and the final file is blocked.
- These exports are **ignored** (not used): `target_new_april26_hdfs` (the lists are `target2_a` / `target2_b`),
  `tenure_hdfs` (tenure comes from `customer_profile`), and
  `DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_filterd_nationality` (the `_Scored` file is used).
- **Export the `activities_*_score` tables without the MSISDN filter**, for the whole base.
- These files hold MSISDNs: keep them on this laptop, and do not open them in Excel.

---

## Step 2 · Start Jupyter

```powershell
cd "C:\…\nbx-python-upgrade-V4.0"
.\.venv\Scripts\Activate.ps1
python -m jupyterlab
```

---

## Step 3 · Notebook 02: prepare the week

Open `notebooks3/02_prepare_week.ipynb`. In the **first code cell**:

```python
RUN_DATE = None              # D = t1 = the most recent day of the data (the last purchase of subs_btl); or "2026-11-19"
FIX_HEADERS = True           # a file without a header gets the contract header
RUN_SIZE = 1_000             # 1_000 | 5_000 | 10_000 = a test copy of that many target lines | "all" = the full week only
SCORE_GROUP = "t2c"          # the group of the test copy (group a)
```

**Kernel → Restart & Run All.** Notebook 02 always writes **the full week** to `data/weekly/<D>/`. With a
`RUN_SIZE` number it **also** writes the test copy to `data/weekly_sample/<D>/`.

**The dates:**

| Date | What | How it is set |
|---|---|---|
| **t1 = D** | the scoring date, the anchor of the `*_score` tables | the most recent purchase day in `subs_btl_test` (or `RUN_DATE`) |
| **t2 = T** | the training date, the anchor of the `*_auto` tables; the model trains here | **t1 − 7** (`ANCHOR_LAG_DAYS = 7`) |
| last week's campaign | `targeting_dataiku_auto.date1` | the latest campaign: from t2 to t1 − 1 |
| last week's purchases (the reward) | `subs_btl_test` | from t2 to t1 |

**Check:**

| Step of 02 | You want to see |
|---|---|
| 1 | `47 contract tables, 2 target lists`; the 3 exports listed above as unknown (ignored) |
| 2 | no `required table(s) missing` (`BTL_script_english_arabic_ok` must be there) |
| 3 | `D (run date) = t1` and `T = t1 − 7`; `targeting_dataiku_auto … the latest campaign = N rows`; `*_score` vs `*_auto`: `OK (differ)` |
| 4 | `no header -> contract header added (sanity OK)` for any file without a header |
| 4b | the duplicates removed (thousands of rows, not ~1 million) |
| 4c | `latest campaign … N lines kept` |
| 5 | (test copy) `nothing blocks` |

---

## Step 4 · Notebook 03 at 1k, then 5k / 10k

Open `notebooks3/03_agents_nba_fast_ollama_gpu.ipynb` (or 04). In the **first code cell**:

```python
RUN_DATE = None              # the latest week prepared by notebook 02
RUN_SIZE = 1_000             # the SAME as in notebook 02
SCORE_GROUPS = ["t2c"]       # group a
PROFILE_LEVEL = "auto"
```

**Restart & Run All.** A test runs apart from the real run, in `data/test10k/`.

**Check, top to bottom:**

| Cell | You want to see |
|---|---|
| settings | `RUN_DATE = …: the latest week prepared by notebook 02`, `groups scored: group a` |
| G1 / G2 | the model 100 % in GPU memory; no failed test call |
| groups / dates | `dates: D = t1 = …, T = t2 = … (D − 7); last week's purchases counted t2 .. t1` |
| Step 2 | `T0 DONE`; `last campaign … 100% inside the expected window` |
| Step 4b | the quality gate without `FALLBACK`, once the `*_score` tables are exported in full |
| Step 5 / 5b | `P1 DONE`; the level table |
| Step 6 / 7 | quality check ≥ 70 %; fallback < 20 % |
| Step 11 / 12 | `final file READY`; `flytxt check … 0 … 0`; `OK: 0 lines with another offer below the band anchor` |
| G3 | `VERDICT: LLM + GPU run` |

Then repeat 02 and 03 with `RUN_SIZE = 5_000` or `10_000` if you want a bigger test.

---

## Step 5 · The real run: 50k of group a

1. **Notebook 02** with `RUN_SIZE = "all"` (if the full week of this D is not prepared yet).
2. **Notebook 03 / 04** with:

   ```python
   RUN_SIZE = "all"
   SCORE_GROUPS = ["t2c"]
   TIME_BUDGET_MIN = 45       # GPU minutes for agents 3 + 4; customers past it get the cell policy
   ```

   then **Restart & Run All**.

   **Before agents 3 and 4 start, step 5b prints the GPU minutes for your real ~50k.** If you want every customer
   decided by the LLM, set `TIME_BUDGET_MIN` to that figure and run again from step 5b.

---

## Step 5b · The real run with `04_agents_nba_fast_ollama_real25k.ipynb` (recommended)

This notebook is 03 for the real run. After notebook 02 has written the real week (`RUN_SIZE = "all"` or any size: it
always writes `data/weekly/<D>/`):

1. **Settings (first cell):** `RUN_SIZE` = the target lines scored (`1_000`, `10_000`, `25_000`, `50_000` or `"all"`,
   split between the groups of `SCORE_GROUPS`), `SCORE_GROUPS` (`["t2c"]` = group a, both groups), `TRAIN_LINES = "all"`
   (training on the whole prepaid base; a number = a random sample, for a faster test), `MODEL`, `PARALLEL` (the
   same as `OLLAMA_NUM_PARALLEL`). Nothing else to type: there is no `RUN_DATE`. To go from a 1k test to the 25k run,
   change only `RUN_SIZE`: step 0 rebuilds the copy.
2. **Run it**, from Jupyter (Restart & Run All) or headless, with no cell timeout:

   ```powershell
   python notebooks3\run_notebook04.py
   ```

   The executed copy goes to `data/real25k/executed/`. Follow the LLM in another window:
   `Get-Content data\real25k\runs\<D>\agents\llm_progress.log -Wait`.
3. **What it does differently from 03:**
   - **Step 0** reads the dates from the data: D = the most recent purchase day (`subs_btl_auto`), T = the campaign day
     (`date1`). Takers are counted from T to D. It **stops** when there is more than one `date1`, when fewer than 95 %
     of the purchases fall between T and D, or when an `*_auto` window goes past T.
   - **Step 0** builds `data/real25k/weekly/<D>/`: the whole eligible prepaid base to train (outside the lists; the
     tables are hard-linked, not copied) + `RUN_SIZE` random lines of the lists to score.
   - **No budget:** every customer gets an LLM answer. Retries, Ollama restart, the cache (a re-run resumes), Windows
     kept awake, the pre-flight check, a progress line every 100 answers.
   - **Step 7b / 8b:** the top-up to 6 offers, and why lines had fewer; **A-25** blocks the file unless every row
     has 6 distinct, valid offers.
   - **Final report:** dates, takers and taker rate, agent 3 propensities, offers before / after, LLM answers
     (fresh / cached / retried / failed), the cell-policy share (must be 0), customers per minute
     (`data/real25k/output/<D>/real_report_<RUN_SIZE>.json`).
4. **Time:** with no budget, step 5b prints the GPU minutes of the level `auto` picks (with no budget it picks
   `fine`, about one answer per customer). To go faster, set `PROFILE_LEVEL = "coarse"` or `"minimal"` and check the
   quality line of step 6.

## Step 6 · Follow the time of every step

Every step prints `[time]` lines:

```
[time] Step 3: estimate 25m00s | elapsed 32m10s | left about 1h42m (end about 14:35)
...
[time] Step 3 done in 21m48s (estimate 25m00s)
```

- **estimate:** the time this step should take;
- **elapsed:** the time since the notebook started;
- **left:** the sum of the estimates of the remaining steps, and the clock time when the notebook should end.

**Where the estimates come from:**

1. **The first run of a `RUN_SIZE`** uses rough defaults.
2. **Agents 3 and 4 (steps 6 / 7)** use the real figure from step 5b: profiles × seconds per call measured by G2.
3. **From the second run of the same `RUN_SIZE` on**, every estimate is the time measured on your laptop. The times
   are kept in `data/runs/_step_times.json` (step names and seconds only).

The last cell, **Time per step**, gives the table of every step: minutes, against the estimate.

**Rough times on the real data** (estimates, not measurements):

| Notebook | Step | `RUN_SIZE = 1_000` | `10_000` | `"all"` (2M base, 50k) |
|---|---|---|---|---|
| 02 | 1–4c (read, rename, dates, write, duplicates, last campaign) | 20–35 min | 20–35 min | 20–35 min |
| 02 | 5 (the test copy) | ~5 min | ~5 min | – |
| 03 | Part A, G1, G2 | ~2 min | ~2 min | ~2 min |
| 03 | 1–5b (T0, features and segments, dictionary, scoring view, panel) | ~5 min | ~5–10 min | ~1–1.5 h |
| 03 | 6–7 (agents 3 + 4) | minutes | ~10–30 min | **step 5b says it**; capped by `TIME_BUDGET_MIN` |
| 03 | 8–12, G3 | ~2 min | ~3 min | ~10–15 min |

Notebook 02 takes about the same time at every `RUN_SIZE`, because it always writes the full week.

---

## Step 7 · Score group b, or both groups

| You want | `SCORE_GROUPS` | Lists needed | Customers |
|---|---|---|---|
| group a only | `["t2c"]` | `target2_a` | ~50k |
| group b only | `["t2f"]` | `target2_b` | ~50k |
| both groups | `["t2c", "t2f"]` | both | ~100k |

For a test copy, `SCORE_GROUP` of 02 must be one of the groups in `SCORE_GROUPS`.

---

## Step 8 · The outputs and your checks

In `data/output/<D>/target2/` (for a test copy: `data/test10k/output/<D>/target2/`):

| File | What |
|---|---|
| `../final/input_flytxt.csv.gz` | **the final file**, only if the audit passed |
| **`target2_scored_360.parquet`** / `.csv.gz` | **the scored MSISDNs** with all the data and the check flags |
| `target2_360.parquet` / `.csv.gz` | every target line, scored or not (`DROP_RULE` says why not) |

**The flags:**

| Flag | Expected |
|---|---|
| `CHK_NOT_IN_FLYTXT`, `CHK_SLOTS_DIFFER_FROM_FLYTXT` | 0 |
| `CHK_UPSELL_AT_RISK_SEGMENT` | 0 |
| `CHK_OFFERS_BELOW_ANCHOR` | only favourites (`CHK_FAVOURITE_BELOW_ANCHOR`) and fallback passes (`CHK_FALLBACK_PASS_BELOW_ANCHOR`) |
| `CHK_UPSELL_CHURN_GE_060` | information |

**Read it with pandas**, in a new notebook on this laptop:

```python
import pandas as pd
d = pd.read_parquet("data/output/2026-11-19/target2/target2_scored_360.parquet")
d.filter(like="CHK_").sum()
d["ACTION_SENT"].value_counts()
```

---

## Step 9 · Hand over, and next week

1. **Hand over.** The engine sends nothing: check `data/output/<D>/final/input_flytxt.csv.gz` and hand it over
   yourself.
2. **Next week:**
   - replace the files in `data/data_raw/` with the new exports;
   - keep `RUN_DATE = None`: 02 reads the new t1, and 03 takes the latest week;
   - run 02, then 03.

   Agent 2b starts from last week's dictionary, and the step times of this week become the estimates.

---

## If something goes wrong

| Message | What to do |
|---|---|
| `required table(s) missing` (02, step 2) | export the table it names (for example `BTL_script_english_arabic_ok`) |
| `copied WITHOUT header (fix it)` | the file's columns do not match the contract: re-export it with a header |
| `no week prepared in data/weekly_sample` | run 02 with the same `RUN_SIZE` (a number) |
| `the 10k copy holds only t2c …` | put that group in `SCORE_GROUPS`, or change `SCORE_GROUP` in 02 |
| `note: the test copy … holds N target lines, RUN_SIZE here is M` | the same `RUN_SIZE` in 02 and 03 |
| `T0 is BLOCKED` | `data/runs/<D>/train/T0_load/validation_report.md`, or `notebooks2/05_check_columns.ipynb` |
| `FALLBACK` in step 4b | a `*_score` table covers too few target lines: export it for the whole base |
| `every test call failed` | start Ollama, or the vLLM server |
