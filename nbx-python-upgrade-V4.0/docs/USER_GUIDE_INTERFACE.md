# FSDR Recommender V4.0 — user guide for the interface

This guide explains, step by step, how to use the web interface:
- what every tab, field and button does;
- what you see after each click;
- which files are written;
- what to do when something goes wrong.

To install the app on a new laptop first, see [SETUP_NEW_LAPTOP.md](SETUP_NEW_LAPTOP.md).

**Contents**

1. [What the interface is](#1-what-the-interface-is)
2. [Start and stop the app](#2-start-and-stop-the-app)
3. [The tabs at a glance](#3-the-tabs-at-a-glance)
4. [Main workflow: two targets, scored separately, and the file to push (tab *FSDR Targets & Output*)](#4-main-workflow-two-targets-scored-separately-and-the-file-to-push)
5. [Demo per customer (tab *FSDR Journey*)](#5-demo-per-customer-tab-fsdr-journey)
6. [Tab *FSDR Data*](#6-tab-fsdr-data)
7. [Tab *FSDR Training & Scoring*](#7-tab-fsdr-training--scoring)
8. [Tab *FSDR Offers*](#8-tab-fsdr-offers)
9. [Tab *FSDR Agentic Recommender*](#9-tab-fsdr-agentic-recommender)
10. [Tab *FSDR Simulation*](#10-tab-fsdr-simulation)
11. [Using the API directly](#11-using-the-api-directly)
12. [Where every file goes](#12-where-every-file-goes)
13. [Checking the results in the notebooks](#13-checking-the-results-in-the-notebooks)
14. [Troubleshooting](#14-troubleshooting)
15. [Glossary](#15-glossary)

---

## 1. What the interface is

The app has two parts that run together:

| Part | Address | What it is |
|---|---|---|
| **The interface (UI)** | http://localhost:7860 | The web screens: tabs, fields, buttons, tables, downloads |
| **The API** | http://localhost:8000/docs | The engine behind the screens. Every button calls it. You can also use it directly (section 11) |

**Everything runs on your computer.** No data leaves it. The only optional external piece is the local LLM server (vLLM), which also runs on your computer.

**Two data areas.** The tabs work on two kinds of runs. Keep them apart:

| Data area | Used by | What it holds |
|---|---|---|
| `data/training/<D>/` + `data/scoring/<D>/` → `data/output/<D>/` (+ `data/runs/<D>/`) | **FSDR Targets & Output** (V4.0) | The training and the scoring input of run date D (5 folders each), your target lists, the files of each target and the file to push |
| `data/fsdr/<D>/` | FSDR Data, Training & Scoring, Offers, Agentic Recommender, Journey, Simulation (V1.0 cycle) | A run folder of the earlier version: the bandit training, the arms cstr / free / llm |

**For the campaign file, use the tab *FSDR Targets & Output* (section 4).** The other tabs remain for demos, the per-customer journey and the V1.0 cycle.

**V4.0 has no market-intelligence module.** The competitor agent, the competitor snapshot and the *Competitors* tab are removed.

**What is final and what is a placeholder (V4.0 today):**

| Final (the business rules) | Placeholder until the LLM agents are built |
|---|---|
| Data contract and validation, features, segments, micro-segments, both actions (`action_original`, `action_agent`) | The action sent = the group's rule action (`decided_by = fallback_rule_action`) |
| The four groups and their rules: code 3, code 4, CC-v1, CF-v1, anchors, price passes and bands, the 6-offer extension EXT-6 | The offer score = own past purchase, then buyer share (`score_source = fallback:E-O3+E-O1`) |
| Code 5 (the 38-column `input_flytxt`), the checks, the push block | |

Every row of every output says which one it used.

---

## 2. Start and stop the app

### 2.1 Once: install

Open PowerShell (or the VS Code terminal) in the project folder. The full guide is [SETUP_NEW_LAPTOP.md](SETUP_NEW_LAPTOP.md).

```powershell
cd <path>\nbx-python-upgrade-V4.0
pip install uv                       # once per laptop
uv sync                              # installs everything from uv.lock (Python 3.12)
copy .env.example .env               # your local settings (never committed)
uv run python run.py check           # Python, packages, data folders OK?
```

### 2.2 Start

```powershell
uv run python run.py start
```

Wait for the two lines `Uvicorn running on http://127.0.0.1:8000` and `Running on local URL: http://127.0.0.1:7860` (about 10–20 seconds). Then open **http://localhost:7860** in your browser.

**Optional: the local LLM.** In a second terminal run `uv run python run.py vllm-serve`. vLLM runs in WSL2 on port 8001.
- Without it, the API health says `degraded` and the agents use their rule-based fallbacks.
- Everything still works.

### 2.3 Stop

In the terminal where the app runs, press **Ctrl + C**. This stops both the UI and the API.

If a window was closed and the ports stay busy, check them:

```powershell
netstat -ano | findstr ":7860 :8000"                 # shows the PIDs still listening
taskkill /PID <pid> /F                               # stop that process
```

### 2.4 Check that it runs

Open http://localhost:8000/health. Expected: `{"status": "ok" or "degraded", "operator": "FSDR", ...}`. `degraded` only means that vLLM is off.

---

## 3. The tabs at a glance

| Tab | Use it to | Section |
|---|---|---|
| **Home** | See what the engine does, which LLM is connected, the version (v4.0) | — |
| **FSDR Journey** | Follow one customer (or a batch) through every step: action → offers → add-ons → output | 5 |
| **FSDR Data** | Generate, upload, validate and preview a V1.0 run folder | 6 |
| **FSDR Training & Scoring** | Train the bandit, score, manage model versions (V1.0 cycle) | 7 |
| **FSDR Offers** | Offers per arm (cstr / free / llm), download as CSV | 8 |
| **FSDR Agentic Recommender** | Run the agents for one customer with the full trace, or for a batch | 9 |
| **FSDR Simulation** | Closed-loop simulation over several weeks; feedback, operations agents, weekly report, campaign log | 10 |
| **FSDR Targets & Output** | **V4.0:** training of a run date, the target lists, Target 1 and Target 2 scored separately, the file to push and the details | **4** |

---

## 4. Main workflow: two targets, scored separately, and the file to push

Tab **FSDR Targets & Output**. One run date, four steps. It takes about 1 minute on 3,000 lines and a few minutes on 1M lines, after a 15-minute T0.

### 4.0 The two targets and their groups

| Target | Group (sub-folder) | Rules | Offer universe | Action used as hint / fallback |
|---|---|---|---|---|
| **Target 1** | **T1C** (`a_constraint`) | production code 3 (B-20 … B-32) | the action's offer families, L2 | `action_original` (your segment → action dictionary) |
| **Target 1** | **T1F** (`b_free`) | production code 4 (FR-01 … FR-14) | L2, DATA / COMBO, action not used | `action_original` |
| **Target 2** | **T2C** (`a_constraint`) | challenger CC-v1 (price band 1.0–1.5 × the anchor, best score × price) | as T1C | `action_agent` (the agent's dictionary) |
| **Target 2** | **T2F** (`b_free`) | challenger CF-v1 | as T1F | `action_agent` |
| **Control** | `control` | never contacted, kept to measure the uplift | — | — |

The configured sizes are 25,000 per group and 20,000 for control. They are set in `.env`: `FSDR_T1_CONSTRAINT_SIZE`, `FSDR_T1_FREE_SIZE`, `FSDR_T2_CONSTRAINT_SIZE`, `FSDR_T2_FREE_SIZE`, `FSDR_CONTROL_SIZE`.

**Each target is scored on its own.** Each target has its own P6 step, its own group audit, its own code-5 run and checks, and its own folder `data/output/<D>/<target>/`. The file to push merges both targets, and only when both passed.

**Up to 6 offers per line** (`FSDR_RECO_OFFERS=6`, the default): 3 monthly + 3 weekly. Set `4` for the V3.0 file.

### 4.1 Field at the top: the run date

**Run date D (YYYY-MM-DD).** This is the Thursday of the campaign, for example `2026-10-22`. Every button of the tab works on this date. The date must be valid, otherwise you get *"Run date must be YYYY-MM-DD"*.

### 4.2 Step 1 — data and training

**What the input is.** One file per table, named exactly like the table. The files are split in two folders with the same 5 sub-folders:

```
data/training/2026-10-22/                       the base, the history, the model scores (training T0 - T3)
├── usage/          subs_amt_auto, activities_*_auto, paygo_activity_auto
├── models/         the 11 model-score tables
├── subscription/   base_prepaid, consumer_aact_account, access_method_id_auto, subs_btl_auto
├── catalog/        btl_catalog
├── others/         party_auto, customer_profile, location_auto, device segment, targeting_dataiku_auto
└── _manifest.json
data/scoring/2026-10-22/                        this week's tables + the target lists (scoring P1 - P7)
├── usage/          usage_data_voice_payg_tg (+ the *_score tables of a weekly run)
├── models/         (empty when unused)
├── subscription/   subscriptions_tg, sub_btl_3months
├── catalog/        btl_catalog, BTL_script_english_arabic_ok, conflicting_offers_h
├── others/         targeting_auto, consent_contact, party_scoring, sample_cg_tg_* (optional)
├── target1/ target2/ control/                  the lists (step 2)
└── _manifest.json
```

The two folders are read **as one run**. A table on the wrong side still loads, with a warning in the validation report.

**Production format:** `<table>.csv.gz`, the Dataiku compressed CSV export, optionally with `<table>.schema.json`.

**Two ways to fill it:**

| You have | Do |
|---|---|
| **Real exports** | Copy the `.csv.gz` files into the folders with Windows Explorer, then go to button **1b** |
| **Nothing yet (test)** | Set **Dummy lines** (e.g. `2000`, or `100000` for a bigger test), then click **1a · Generate dummy data**. Tick **Replace existing data** to overwrite generated data that is already there |

**Button 1a · Generate dummy data**
- Writes 47 tables of fake but consistent data (`.csv.gz`) into both folders.
- About 3 s for 2,000 lines, 80 s for 100,000 lines.
- It refuses to overwrite a folder holding real data, unless you tick **Replace existing data**.

**Button 1b · Training (validate + features + segments)** runs, in order:
1. **T0 — validation and Parquet cache.**
   - Every table of both folders is checked: fields, types, keys, duplicates, joins, coverage, folders, side, format, `customer_profile`.
   - A table that is not `.csv.gz` (and not generated) blocks the run.
2. **T2 + T3 — features and segments:**
   - features B-01 … B-12;
   - the S-rule segment;
   - age / tenure / value bands and the micro-segment (nationality × age × tenure);
   - both actions of every line.

**What you see:**
- *T0*: `status` (done / blocked), `pipeline` (train_week0 or train_weekly), the tables and rows, and a plain-language `explanation`.
- *Training T2 + T3*: the number of lines, the number of micro-segments, and a table of lines per segment.

**If T0 is blocked:** the explanation says why. Fix the file and click **1b** again, ticking **Re-run even if done**.

A second click with the same files is instant: the steps are skipped. Tick **Re-run even if done** to force them.

### 4.3 Step 2 — the target lists

**Where they go:** in the scoring folder of the date, one folder per target and one sub-folder per group:

```
data/scoring/<D>/target1/a_constraint/target1_a.csv.gz     T1C
data/scoring/<D>/target1/b_free/target1_b.csv.gz           T1F
data/scoring/<D>/target2/a_constraint/target2_a.csv.gz     T2C
data/scoring/<D>/target2/b_free/target2_b.csv.gz           T2F
data/scoring/<D>/control/control.csv.gz                    control
```

**What a list contains:**
- One line per customer, with an **`access_method_val`** column (or **`msisdn`**): the 8-digit number.
- Any other columns (plan, stratum `Final`, region, …) are kept and appear in the details file.
- Production format: `.csv.gz`; `.csv` / `.json` / `.jsonl` are accepted for tests.

Example (`target1_a.csv`, before compressing):

```
access_method_val,plan,Final
70012510,FSDR Prepaid Data,Oman|mid|dual|FSDR Prepaid Data
70046318,FSDR Shababiah,India|low|data_only|FSDR Shababiah
```

**Three buttons:**

| Button | What it does |
|---|---|
| **2a · Upload this list** | Choose the **Group** in the dropdown, drop the file in **List**, click. The file is saved as that group's list (it replaces an earlier one) and all the lists are checked again |
| **2b · Generate dummy lists (all groups)** | Writes test lists for the 5 groups: a stratified random draw of the eligible base, scaled to its size (e.g. 465 × 4 + 372 on a 3,000-line base) |
| **2c · Check the lists** | Shows which lists are there and checks them, without changing anything |

You can also copy the files into the folders yourself, then click **2c**.

**What the check shows:**
- The file of each group (or *— missing —*).
- The result, **OK** or **BLOCKED**, with the lines read, dropped, and kept per group.
- A table of issues:

| Level | Check | Meaning | What happens |
|---|---|---|---|
| **blocking** | `line_in_two_groups` | the same MSISDN is in two lists | the panel (3a) is blocked: fix the lists |
| **blocking** | `missing_list` | a group has no list | the panel is blocked |
| warning | `dropped: not an 8-digit MSISDN` | a malformed number | the line is dropped |
| warning | `dropped: not in the eligible base (D-07)` | not an active prepaid line of the base | dropped |
| warning | `dropped: Nursering segment (B-02)` | tenure < 45 days: never targeted | dropped (contacted groups only) |
| warning | `dropped: in another campaign this week (H-05)` | already in `targeting_auto` this week | dropped (contacted groups only) |
| warning | `duplicate_in_list` | listed twice in the same group | kept once |
| warning | `size` | more than 2 % off the configured size | runs anyway, reported |

Every dropped line and its reason is saved in `data/runs/<D>/score/P1_panel/targets_rejected.csv`.

**No list at all?** The panel then draws the five groups at random from the eligible base:
- stratified on nationality × value band × usage type × plan, balanced on age band, tenure band and segment;
- the provided sample table `sample_cg_tg_new_7oct2026`, if present, maps cstr → T1C, free → T1F, cg → control.

### 4.4 Step 3 — scoring, one target at a time

| Button | What it does | Output folder |
|---|---|---|
| **3a · Panel (from the lists)** | **P1**: takes your lists (mode `uploaded`) or draws the groups (mode `drawn`); sizes (A-30) and balance (A-31) | `data/runs/<D>/score/P1_panel/` |
| **3b · Score Target 1** | **P6** for T1C + T1F only (rule action `action_original`; code 3 / code 4 passes; EXT-6 third offer; 10.1 retry for T1C), then **code 5** on Target 1 → its files and checks | `data/runs/<D>/score/P6_offers_target1/`, **`data/output/<D>/target1/`** |
| **3c · Score Target 2** | The same for T2C + T2F (rule action `action_agent`; CC-v1 / CF-v1 bands) | `…/P6_offers_target2/`, **`data/output/<D>/target2/`** |
| **3d · Build the push file** | The control file, then both targets merged into the file to push, only when **both** passed and no MSISDN is in two targets | **`data/output/<D>/control/`**, **`data/output/<D>/push/`** |
| **3a → 3d in one go** | The four buttons above, in order | all of the above |
| **Show the last output** | Shows the result of the last run of that date again, without re-running | — |

You can score Target 1 and Target 2 in any order, and re-score one target without touching the other: for example, after correcting a Target 2 list, click **3a** then **3c** and **3d**.

**What you see after 3b / 3c:**
1. **P6 of the target:**
   - the group audit (**PASS** / FAIL);
   - per group: contacted, customers with offers, offers, 10.1 retries.
2. **The target's state:**
   - **READY**, or **BLOCKED — see BLOCKED.txt**;
   - the rows in `input_flytxt_<target>`, up to 6 offers, the size of the details, and the group audit;
   - the output checks:

   | Check | Rule |
   |---|---|
   | A-20 | exactly the 38 columns of F-19, in order |
   | A-21 | one row per MSISDN, 8-digit; no control MSISDN in the file |
   | A-22 | date = D, start_date = D + 1, end_date and channels empty, flag_splio = 1 |
   | A-23 | every offer in the catalogue |
   | A-24 | segment is one of the 7 actions |

3. **Groups a / b:**
   - the folder, rules, lines and rows in the file;
   - **offers per line**;
   - customers dropped by F-04 (no recommended offer);
   - favourites added, forced bundles (F-08);
   - slots cleared by F-09 (same type and price), F-11 (same offer twice) and F-12 (conflicting pair).

**After 3d:** **Push READY** with the rows per target, or **Push BLOCKED** with the reasons, plus the control file.

### 4.5 Step 4 — download

The box **4 · Download** lists the files of the run. Click a file to download it:

| File | What it is | Who uses it |
|---|---|---|
| `push/input_flytxt.csv.gz` | **The file to push**: the 38 lower-case columns of code 5, both targets, one row per MSISDN. Written **only when both targets passed** | The campaign platform (after approval) |
| `target1/input_flytxt_target1.csv.gz` | Target 1's file (T1C + T1F), written only when Target 1 passed | Target 1 review |
| `target1/a_constraint/input_flytxt_t1c.csv.gz`, `target1/b_free/input_flytxt_t1f.csv.gz` | The same rows, per group | Audit, per-group review |
| `target2/…` | The same for Target 2 (`t2c`, `t2f`) | Target 2 review |
| `target1/a_constraint/details_t1c.csv.gz` … `target2/b_free/details_t2f.csv.gz` | **The details ("big data")**: one row per customer of the group, about 240 columns (below) | Validation, analysis |
| `target1/details_target1.parquet`, `target2/details_target2.parquet` | The details of the whole target in one table (notebook, Power BI) | Validation |
| `control/control_<D>.csv.gz` | The control lines (never contacted) and their strata | Measuring the uplift next week |
| `target1/final_rules_report.csv`, `target2/…` | Per group: lines, rows in the file, offers per line, F-04 drops, favourites, forced bundles, F-09 / F-11 / F-12 clearings | Quick review |

Each target folder also holds `output_checks.json`, `audit_groups.json`, `summary.json` and `_outputs_manifest.json` (rows + SHA-256 of every file).

**The 38 columns of `input_flytxt`, in order:** `date`, `start_date`, `end_date`, `msisdn`, `channel_1…6`, `opportunity_name_1…6`, `destination_offer_name_1…6`, `flag_splio`, `offer_script_en_1…6`, `offer_script_ar_1…6`, `segment`, `segment_btl_taker`, `segment_btl_taker_upsell`.

**The 6 slots (V4.0, 6 offers):**

| Slot | Content |
|---|---|
| 1, 2 | the 1st and 2nd monthly offers (highest price first) |
| 3, 4 | the 1st and 2nd weekly offers |
| 5 | the 3rd monthly offer (EXT-6); if empty: the customer's favourite (his own most expensive 270-day purchase) |
| 6 | the 3rd weekly offer (EXT-6); if empty: the next favourite, or a forced popular bundle (F-08) |

**EXT-6**, the third offer of a validity, is chosen in the group's own universe with three conditions:
- priced above the customer's anchor (never a downgrade);
- not already selected;
- not the same service and price as a selected offer.

Its `selected_by` is `EXT-6`. With `FSDR_RECO_OFFERS=4`, slots 5 / 6 are only the favourites and forced bundles, as in V3.0.

An empty slot is an empty field, never `nbo_LFM_nan`.

**What the details file holds**, column groups in order:

| Group of columns | Examples |
|---|---|
| Group and source | `GROUP_KEY`, `PANEL_GROUP`, `TARGET`, `SOURCE` (uploaded / drawn), `STRATUM`, `T_*` (the columns of your list), `TARGET_FILE` |
| Profile and bands | `NATIONALITY_GROUP`, `AGE`, `AGE_BAND`, `TENURE_DAYS`, `TENURE_BAND`, `MICRO_SEGMENT`, `VALUE_BAND`, `USAGE_TYPE`, `REGION`, `DV_SEGMENT`, `CURRENT_PLAN` |
| Segment | `SEGMENT`, `RULE_ACTION`, `EXCLUDED` |
| Actions | `ACTION_ORIGINAL`, `ACTION_AGENT`, `ACTION_AGENT_RANK2`, `ACTION_SENT`, `ACTION_AGREEMENT` (all_same / sent_original / sent_agent / other), `DECIDED_BY`, `CONSISTENCY_RETRY`, `PROPENSITY`, `RULES_VERSION` |
| Anchors | `B26_AVG_REV_MONTHLY`, `B26_AVG_REV_DATA_MONTHLY`, … (code 3 / 4 anchors), `F05_MONTHLY`, `F05_DATA_MONTHLY`, … (code 5 anchors) |
| Recommended offers | `reco_m1_*` … `reco_m3_*` (monthly), `reco_w1_*` … `reco_w3_*` (weekly): `btl_name`, `service`, `price`, `score`, `anchor`, `pass`, `selected_by` (B-28 / B-29 / B-30, FR-09 …, CC-04, CC-04 widened, fallback …, EXT-6) |
| Slots | `SLOT_1` … `SLOT_6`, `REASON_1` … `REASON_6` (recommended monthly / weekly, favourite, forced F-08 …), `REMOVED_1` … `REMOVED_6` (e.g. `F-12: BTL_…`) |
| Result | `IN_INPUT_FLYTXT` (1 / 0), `DROP_RULE` (F-04 no offer, no decision) |
| Features | the 9 model scores `PROBA_*` and their `*_MISSING` flags, `REV_W1…M3`, `VAR_*`, `LOG_*`, usage and spend per period |

**Open the details in Excel:**
1. Choose *Data → Get Data → From File → From Text/CSV*.
2. Excel reads the `.csv.gz` directly in recent versions. Otherwise unzip it with 7-Zip first.

### 4.6 When a target or the push is blocked

If a check of a target fails:
- `<target>/input_flytxt_<target>.csv.gz` is **not** written, and an older one is deleted, so nothing stale can be pushed by mistake.
- `<target>/BLOCKED.txt` lists the failed checks.
- The per-group and details files are still written, so you can investigate.
- The push (3d) is then blocked too: `push/BLOCKED.txt` names the target.

Fix the cause (usually the lists, or the input data), then score that target again (**3a** if the lists changed, then **3b** or **3c**, then **3d**).

**Rule H-08:** nothing is pushed without every check passing and a human approval. The SFTP push itself is not automated yet: it comes with a later phase.

### 4.7 Run again

- **Same run date:** each scoring button always recomputes from the current lists. Upload a corrected list with **2a**, then **3a** and the target(s) concerned, then **3d**.
- **Next week:** change the run date, then do steps 1 → 3 again. Each date keeps its own folders, so past weeks stay available.

---

## 5. Demo per customer (tab FSDR Journey)

Use it to show, for one customer, how the decision is made step by step. It works on the dummy folders `data/fsdr_dummy/<date>/`.

**Create a dummy source** in a terminal, once per date:

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --mode small --layout folders --out data/fsdr_dummy/2026-10-22
```

**Steps:**

| # | Control | What it does / what you see |
|---|---|---|
| 1 | **Dummy data folder** + **1 · Upload data** | Copies the source into `data/fsdr/<date>/`, checks every table (green / amber / red with the reason), trains and scores. Shows 4 sections: the tables of the data contract, the joins the NBA agent relies on, the training and scoring, and the calculations recomputed from the raw tables. **Ingestion report (JSON)** is downloadable |
| — | **Mode** | *Single MSISDN*: pick one scored line in **MSISDN** (or type it). *Batch*: upload a `.csv` / `.txt` with an `msisdn` column, or paste MSISDNs (one per line or comma-separated) |
| — | **LLM model** | Empty = the default vLLM model. Without vLLM, the agents use their fallbacks |
| 2 | **2 · Best action recommender** | The action, its probability (propensity), the bandit's top action and its probabilities, and who decided (agent or fallback) |
| 3 | **3 · Best offer recommender (top 4)** | Up to 4 offers with service, generosity, validity, price, the price pass and the anchor |
| 4 | **4 · Roaming & international recommender** | Top 2 roaming offers and international packs |
| 5 | **5 · Full output (save)** | Everything for the customer in one file: `data/fsdr/<date>/outputs/journey/<msisdn>_output.json` (+ a CSV row) |

Each step continues from the previous one; a missing previous step is run first. Each step has a **Download** box. After a page reload the tab continues on the last upload.

---

## 6. Tab FSDR Data

This tab works on the V1.0 runs in `data/fsdr/<D>/`.

| Control | What it does |
|---|---|
| **Run** (dropdown) + **Refresh runs** | Picks a run folder; Refresh reloads the list |
| **Validate** | Checks every table of the run against the data contract: fields, types, keys, duplicates, joins, coverage, windows, format. Shows the report (blocking issues first) |
| Accordion **Generate dummy data**: **Run date D**, **Lines**, **Seed**, **Scenarios**, **Generate** | Writes a dummy run in `data/fsdr/<D>/`. The scenarios combine: `baseline`, `churn_wave`, `data_growth`, `price_sensitive`, `new_offers`, `campaign_fatigue`, `data_quality_issues` (defects that validation must catch) |
| Accordion **Upload real exports (.csv.gz)**: **Run date D**, **Replace generated data**, **Table files**, **Upload and validate** | Saves your files in `data/fsdr/<D>/` and validates them. Accepts `.csv.gz` (+ `.schema.json`); `.json` / `.jsonl` / `.csv` are test formats, blocked by validation unless `FSDR_ALLOW_TEST_FORMATS=true`. Any other file (e.g. `.xlsx`) is refused before anything is written |
| Accordion **Preview a table**: **Table**, **Rows**, **Preview** | Shows the first rows of a table and its field information |

---

## 7. Tab FSDR Training & Scoring

This tab runs the V1.0 cycle on `data/fsdr/<D>/`: the Vowpal Wabbit bandit.

| Control | What it does |
|---|---|
| **Run** | The run folder |
| **Training mode** | `logged` (all targeted lines, recommended) or `simulation` (the original draws) |
| **Reward** | `any_btl` (production), `same_service`, `exact_offer` |
| **Promote** | `auto` (if not worse than the champion), `never`, `always` |
| **Passes**, **Exploration epsilon** | Training passes; exploration rate |
| **Train** | Trains a new model version; shows the version, whether it was promoted, and why |
| **Population** | `target` (the test sample) or `base` |
| **Policy** | `sample` (production) or `greedy` |
| **Allow a stale model** | Score even with a model trained for another week |
| **Score** | Scores with the champion model |
| **Refresh registry**, **Version to promote** + **Promote**, **Roll back** | Manages the model versions (champion, history) |

---

## 8. Tab FSDR Offers

| Control | What it does |
|---|---|
| **Run** | The run folder (scored) |
| **Arm** | `cstr`: the action restricts services and generosity. `free`: L2 DATA + COMBO only. `llm`: the agents decide (rule-based without vLLM) |
| **Generosity mapping** | `all_L2` (production) or `normal` (L0 / L1 / L2 by action) |
| **Recommend** | Builds the offers of every customer of the arm; shows a table |
| **Download offers (CSV)** | The offers as a CSV file |

---

## 9. Tab FSDR Agentic Recommender

**One customer:**
- Pick a customer in **…llm-arm customer (sorted by churn)**, or type an **MSISDN**.
- Click **Run the agents**.
- You see:
  - the pipeline;
  - the trace of the two agents (action → offer), with the tools they called;
  - the decision with its reasons;
  - the SMS message.

**Batch** (accordion *Batch: action → top 4 offers → top 2 roaming → international offers*):

| Control | What it does |
|---|---|
| **MSISDN list (optional)** | Paste MSISDNs; empty = the customers of the arm |
| **Arm** | `llm`, `cstr` or `free` |
| **Batch size**, **Max customers** | Customers per batch; total cap |
| **Recommend for the batch** | Runs batch by batch; a status line and a result table |
| **Download batch (CSV)** | The results as CSV |

---

## 10. Tab FSDR Simulation

**Closed-loop simulation**, in `data/fsdr/simulation/`, apart from your runs. Each week it:
1. generates the data;
2. collects last week's rewards;
3. trains;
4. scores;
5. recommends for every arm and logs it.

| Control | What it does |
|---|---|
| **First week D**, **Weeks** (1–8), **Lines**, **Seed** | The simulated period and size |
| **Scenarios**, **Arms** | The world's drifts; the arms besides the control `cg` |
| **Run the simulation** | Shows per week and per arm: lines, take rate, revenue per line, uplift vs control |

**Weekly cycle on your runs:**

| Control | What it does |
|---|---|
| **Run** + **Collect feedback** | Pick the run of the week *after* a campaign: its fulfilments hold the responses; the rewards are added to the campaign log |
| **Operations agent** (`data`, `training`, `scoring`, `monitor`) + **Run the agent** | An agent reviews that stage of the run and reports |
| **Campaign week** + **Weekly report** | Take rate, revenue, uplift vs control per arm for that week |
| **Campaign log: arm**, **Rows**, **Show the campaign log** | What was sent to whom, with the logged propensity (newest first) |

---

## 11. Using the API directly

Open **http://localhost:8000/docs**: a local page (no CDN) that lists every route, with a **Try it** button and a filter. `FSDR_SWAGGER_CDN=true` in `.env` gives the standard Swagger UI instead (it loads from the internet).

**The V4.0 routes (tab FSDR Targets & Output):**

| Route | What it does |
|---|---|
| `POST /fsdr/v4/{run}/generate` | Dummy data of the date in `data/training/<D>/` + `data/scoring/<D>/` (button 1a). Body: `{"n": 2000, "seed": 42, "force": false}` |
| `POST /fsdr/v4/{run}/prepare?force=false` | Training T0 + T2 + T3 (button 1b) |
| `POST /fsdr/v4/{run}/targets/generate` | Dummy lists (button 2b) |
| `POST /fsdr/targets/{run}/{group}` | Upload one list (`group` = `t1c`, `t1f`, `t2c`, `t2f`, `control`; form field `file`) |
| `GET /fsdr/targets/{run}` | The lists and their check (button 2c) |
| `POST /fsdr/v4/{run}/panel` | P1 (button 3a) |
| `POST /fsdr/v4/{run}/score/{target}` | One target, `target1` or `target2`: P6 + final (buttons 3b / 3c). HTTP 409 without a panel |
| `POST /fsdr/v4/{run}/push` | Control file + push (button 3d) |
| `POST /fsdr/v4/{run}/run` | 3a → 3d in one call. HTTP 409 when the lists block |
| `GET /fsdr/v4/{run}/output` | Each target's summary, the push state and the files of `data/output/<D>/` |

The V3.0 routes (`/fsdr/v3/…`, `POST /fsdr/groups/{run}/run`, `GET /fsdr/final/{run}`) still work. They score the four groups together into `data/runs/<D>/final/`.

**Example in PowerShell:** upload the T2C list, then score Target 2 and build the push.

```powershell
curl.exe -F "file=@C:\lists\target2_a.csv.gz" http://localhost:8000/fsdr/targets/2026-10-22/t2c
curl.exe -X POST http://localhost:8000/fsdr/v4/2026-10-22/panel
curl.exe -X POST http://localhost:8000/fsdr/v4/2026-10-22/score/target2
curl.exe -X POST http://localhost:8000/fsdr/v4/2026-10-22/push
```

If `FSDR_API_KEY` is set in `.env`, add the header `-H "X-API-Key: <key>"`; the UI sends it itself.

**The same flow from the terminal**, without the UI:

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --n 3000
uv run python run.py fsdr-load --date 2026-10-22
uv run python run.py fsdr-features --date 2026-10-22
uv run python run.py fsdr-targets --date 2026-10-22 --generate     # or copy your lists into data/scoring/<D>/target1|2/
uv run python run.py fsdr-run-targets --date 2026-10-22            # or fsdr-score-target --target target1|target2, fsdr-push
```

See `docs/RUN_FROM_TERMINAL.md` for every command.

---

## 12. Where every file goes

| Folder | What | Committed to git? |
|---|---|---|
| `data/training/<D>/` | The training input of the date (5 folders) | no |
| `data/scoring/<D>/` | The scoring input (5 folders) and the lists `target1/`, `target2/` (`a_constraint/`, `b_free/`), `control/` | no |
| `data/cache/<D>/` | The Parquet cache (16 partitions per table) | no |
| `data/runs/<D>/train/T0_load … T3_segments/` | Validation report, completeness, features, segments, both dictionaries, crosswalk, priors | no |
| `data/runs/<D>/score/P1_panel/` | `panel.parquet`, `group_sizes.csv`, `balance.csv`, `targets_rejected.csv`, `targets_check.json` | no |
| `data/runs/<D>/score/P6_offers_target1/`, `P6_offers_target2/` | Per target: the code-2 tables of its groups (`TG_*_scored_nba`, `results_scoring_nba`), its 4 reco tables (`recos_monthly/weekly_…`), `traces.csv`, `group_report.csv`, `audit_groups.json`, `decisions.parquet` | no |
| `data/output/<D>/target1/`, `target2/` | `input_flytxt_<target>.csv.gz`, `a_constraint/`, `b_free/` (per-group file + details), `details_<target>.parquet`, `final_rules_report.csv`, `output_checks.json`, `audit_groups.json`, `_outputs_manifest.json`, `summary.json` | no |
| `data/output/<D>/control/`, `push/` | The control file; the file to push (or `BLOCKED.txt`) | no |
| `data/fsdr/<D>/` | The V1.0 run folders (tabs of sections 5–10) | no |
| `.env` | Your settings and keys | no |

Every step folder has a `_step.json`: status, start, duration, rows, input fingerprint. A step whose inputs did not change is skipped on the next run.

**Customer data never goes to git.** The folders above are git-ignored, because they hold MSISDNs.

---

## 13. Checking the results in the notebooks

```powershell
uv run python run.py notebook --name training    # 01_training_step_by_step.ipynb: run T0 -> T2 -> T3, cell by cell
uv run python run.py notebook --name scoring     # 02_scoring_step_by_step.ipynb: lists -> P1 -> Target 1 -> Target 2 -> push
uv run python run.py notebook                    # fsdr_results.ipynb: read-only view of a run (then Run All)
uv run python run.py notebook-html --date 2026-10-22   # no browser: an HTML report of fsdr_results in data/runs/<D>/
```

The two step-by-step notebooks run every step with the explanation of its rules and show its details:
- **training**: validation issues, completeness, the cache, feature statistics, segments, the action cross-table;
- **scoring**: the list checks, the panel balance, each target's audit and `selected_by` rules, the offers per line, what fills each of the 6 slots, one line end to end, and the final checks of the file to push.

Set `RUN_DATE` in the first code cell. They are stored without outputs; do not commit them with outputs (MSISDNs).

---

## 14. Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| The page does not open | The app is not running, or still starting | Run `uv run python run.py start`, wait for both "Running" lines, reload |
| Health says `degraded` | vLLM is off | Normal: the agents use their fallbacks. Start `run.py vllm-serve` for the LLM |
| 1b: `T0 … BLOCKED` | A blocking data issue | Read the explanation, or `data/runs/<D>/train/T0_load/validation_report.md`; fix the file; 1b again with *Re-run even if done* |
| 1b: `N table(s) are not .csv.gz` | Real input in another format | Use the Dataiku `.csv.gz` export, or set `FSDR_ALLOW_TEST_FORMATS=true` in `.env` for tests (restart the app) |
| Validation warning `found under training/, expected under scoring/` | A table is in the wrong folder | It still loads; move it to its side to silence the warning (section 4.2) |
| 1a: `holds data that was not generated` | The folder holds real data | Tick **Replace existing data** only if you really want to overwrite it |
| 2c / 3a: `BLOCKED`, `line_in_two_groups` | The same MSISDN is in two lists | Remove it from one list, upload it again (2a), click 3a |
| 3b / 3c: `P1_panel … is missing` (HTTP 409) | The panel was not built | Click **3a** first |
| 3a: `fsdr-features first` / `not done` | Step 1b was not run for this date | Click **1b** |
| 3b / 3c: **BLOCKED** | An output check or the target's group audit failed | Open `data/output/<D>/<target>/BLOCKED.txt`; the per-group and details files show the lines concerned |
| 3d: **Push BLOCKED** | A target is blocked or not scored yet | Read `push/BLOCKED.txt`: it names the target; score or fix it, then 3d again |
| Many customers with `DROP_RULE = F-04` | Their spend anchor is above every offer price, so no offer fits (code 3 / 4 rule BR-82); frequent on dummy data, whose catalogue is small | Normal on dummy data; on real data, check the catalogue and the anchors in the details file |
| `forced_F08` is always 0 | The forced bundles `CM_*` / `DM_*` are not in the catalogue, or slot 6 is already taken by the 3rd weekly offer | Normal on dummy data; on real data they apply when they are in `btl_catalog` and slot 6 is empty |
| A size warning (`configured 25,000`) | Your lists are smaller or larger than the configured sizes | Information only; change the sizes in `.env` if your design differs |
| A port is busy at start | An earlier app is still running | Section 2.3 (`netstat`, `taskkill`) |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Run date D** | The Thursday of the campaign; every file of the week belongs to it |
| **Training / scoring folder** | `data/training/<D>/` (the base, the history, the model scores) and `data/scoring/<D>/` (this week's tables, the lists); read as one run |
| **Target 1 / Target 2** | Production (T1C a + T1F b) and challenger (T2C a + T2F b), each scored separately; **control** is never contacted |
| **T1C / T1F / T2C / T2F** | The four contacted groups (section 4.0) |
| **Action** | One of 7: CROSS SELL DATA, UPSELL DATA, RETAIN, MAINTAIN, UPSELL, CROSS SELL VOICE, UPSELL VOICE |
| **`action_original`** | Your original segment → action dictionary (code 1), via the segment engine or the crosswalk |
| **`action_agent`** | The agent's dictionary (week 0: a copy of the original; the LLM will update it weekly) |
| **Segment, micro-segment** | The S-rule segment (14); nationality × age band × tenure band (120 cells) |
| **Anchor (B-26 / F-05)** | The customer's spend reference: mean + std of his revenue (codes 3–4), and the code-5 version min(mean + std, max) |
| **Pass 1 / 2 / 3** | Code 3 / 4 price passes: in the band above the anchor; the missing type above the anchor; anything above the anchor |
| **CC-04 / CF-04** | Challenger selection: best score × price inside 1.0–1.5 × the anchor, widened once to 2.0 ×, else the production passes |
| **EXT-6** | V4.0: the third offer of a validity (6 offers per line), above the anchor, in the group's universe |
| **Slots 1–6** | The 6 offers of `input_flytxt`: monthly 1–2, weekly 1–2, then the 3rd monthly (5) and 3rd weekly (6), else favourites / forced bundle |
| **F-04, F-09, F-11, F-12** | Code-5 rules: drop without offer; same type and price; same offer twice; conflicting pair |
| **H-08** | No push without every check passing and a human approval |
| **Fallback** | The deterministic decision used when the LLM is not available (or not built yet); always marked |
