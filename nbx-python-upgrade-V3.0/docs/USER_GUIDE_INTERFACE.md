# FSDR Recommender V3.0 — user guide for the interface

This guide explains, step by step, how to use the web interface: what every tab, field and button does, what you see after each click, which files are written, and what to do when something goes wrong.

**Contents**

1. [What the interface is](#1-what-the-interface-is)
2. [Start and stop the app](#2-start-and-stop-the-app)
3. [The tabs at a glance](#3-the-tabs-at-a-glance)
4. [Main workflow: the four groups and the file to push (tab *FSDR Targets & Final*)](#4-main-workflow-the-four-groups-and-the-file-to-push)
5. [Demo per customer (tab *FSDR Journey*)](#5-demo-per-customer-tab-fsdr-journey)
6. [Tab *FSDR Data*](#6-tab-fsdr-data)
7. [Tab *FSDR Training & Scoring*](#7-tab-fsdr-training--scoring)
8. [Tab *FSDR Offers*](#8-tab-fsdr-offers)
9. [Tab *FSDR Agentic Recommender*](#9-tab-fsdr-agentic-recommender)
10. [Tab *FSDR Simulation*](#10-tab-fsdr-simulation)
11. [Tab *Competitors*](#11-tab-competitors)
12. [Using the API directly](#12-using-the-api-directly)
13. [Where every file goes](#13-where-every-file-goes)
14. [Checking the results in the notebook](#14-checking-the-results-in-the-notebook)
15. [Troubleshooting](#15-troubleshooting)
16. [Glossary](#16-glossary)

---

## 1. What the interface is

The app has two parts that run together:

| Part | Address | What it is |
|---|---|---|
| **The interface (UI)** | http://localhost:7860 | The web screens: tabs, fields, buttons, tables, downloads |
| **The API** | http://localhost:8000/docs | The engine behind the screens. Every button calls it. You can also use it directly (section 12) |

**Everything runs on your computer.** No data leaves it. The only optional external piece is the local LLM server (vLLM), which also runs on your computer.

**Two data areas.** The tabs work on two kinds of runs. Keep them apart:

| Data area | Used by | What it holds |
|---|---|---|
| `data/input/<D>/` (+ `data/runs/<D>/`) | **FSDR Targets & Final** (V3.0) | The input of run date D in 5 folders, your target lists, and every step's output |
| `data/fsdr/<D>/` | FSDR Data, Training & Scoring, Offers, Agentic Recommender, Journey, Simulation (V1.0 cycle) | A run folder of the earlier version: the bandit training, the arms cstr / free / llm |

**For the campaign file of the four groups, use the tab *FSDR Targets & Final* (section 4).** The other tabs remain for demos, the per-customer journey and the V1.0 cycle.

**What is final and what is a placeholder (V3.0 today):**

| Final (the business rules) | Placeholder until the LLM agents are built |
|---|---|
| Data contract and validation, features, segments, micro-segments, both actions (`action_original`, `action_agent`) | The action sent = the group's rule action (`decided_by = fallback_rule_action`) |
| The four groups and their rules: code 3, code 4, CC-v1, CF-v1, anchors, price passes and bands | The offer score = own past purchase, then buyer share (`score_source = fallback:E-O3+E-O1`) |
| Code 5 (the 38-column `input_flytxt`), the checks, the push block | |

Every row of every output says which one it used.

---

## 2. Start and stop the app

### 2.1 Once: install

Open PowerShell (or the VS Code terminal):

```powershell
cd C:\Abdelouaheb\perso\Data_science_2024_projects\2026\llm_recommendation_engin_telco\nbx-python-upgrade-V3.0
uv sync                              # installs everything from uv.lock
copy .env.example .env               # your local settings (never committed)
uv run python run.py check           # Python, packages, data folders OK?
```

### 2.2 Start

```powershell
uv run python run.py start
```

Wait for the two lines `Uvicorn running on http://0.0.0.0:8000` and `Running on local URL: http://0.0.0.0:7860`, which take about 10–20 seconds. Then open **http://localhost:7860** in your browser.

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
| **Home** | See what the engine does, which LLM is connected, the version (v3.0) | — |
| **FSDR Journey** | Follow one customer (or a batch) through every step: action → offers → add-ons → competitors → output | 5 |
| **FSDR Data** | Generate, upload, validate and preview a V1.0 run folder | 6 |
| **FSDR Training & Scoring** | Train the bandit, score, manage model versions (V1.0 cycle) | 7 |
| **FSDR Offers** | Offers per arm (cstr / free / llm), download as CSV | 8 |
| **FSDR Agentic Recommender** | Run the agents for one customer with the full trace, or for a batch | 9 |
| **FSDR Simulation** | Closed-loop simulation over several weeks; feedback, operations agents, weekly report, campaign log | 10 |
| **FSDR Targets & Final** | **V3.0:** prepare a run date, upload the target lists of the 4 groups, score them, get the file to push and the details | **4** |
| **Competitors** | Browse and search the competitor plans, compare an FSDR offer | 11 |

---

## 4. Main workflow: the four groups and the file to push

Tab **FSDR Targets & Final**. One run date, four steps, about 30 seconds on 2,000 lines and a few minutes on 1M lines.

### 4.0 The four groups

| Group | Target | Rules | Offer universe | Action used as hint / fallback |
|---|---|---|---|---|
| **T1C** | Target 1 | production code 3 (B-20 … B-32) | the action's offer families, L2 | `action_original` (your segment → action dictionary) |
| **T1F** | Target 1 | production code 4 (FR-01 … FR-14) | L2, DATA / COMBO, action not used | `action_original` |
| **T2C** | Target 2 | challenger CC-v1 (price band 1.0–1.5 × the anchor, best score × price) | as T1C | `action_agent` (the agent's dictionary) |
| **T2F** | Target 2 | challenger CF-v1 | as T1F | `action_agent` |
| **Control** | — | never contacted, kept to measure the uplift | — | — |

The configured sizes are 25,000 per group and 20,000 for control. They are set in `.env`: `FSDR_T1_CONSTRAINT_SIZE`, `FSDR_T1_FREE_SIZE`, `FSDR_T2_CONSTRAINT_SIZE`, `FSDR_T2_FREE_SIZE`, `FSDR_CONTROL_SIZE`.

### 4.1 Field at the top: the run date

**Run date D (YYYY-MM-DD).** This is the Thursday of the campaign, for example `2026-10-22`. Every button of the tab works on this date. The date must be valid, otherwise you get *"Run date must be YYYY-MM-DD"*.

### 4.2 Step 1 — the input of the run date

**What the input is.** `data/input/<D>/` holds one file per table, named exactly like the table, in 5 folders:

```
data/input/2026-10-22/
├── usage/          18 tables  activities_*, subs_amt_*, paygo, usage_data_voice_payg_tg
├── models/         11 tables  the model scores
├── subscription/    6 tables  base_prepaid, accounts, MSISDN map, subscriptions_tg, purchases
├── catalog/         3 tables  btl_catalog, scripts, conflicting offers
├── others/          9 tables  party, customer_profile, location, consent, device, targeting, sample
└── _manifest.json
```

**Production format:** `<table>.csv.gz`, the Dataiku compressed CSV export, optionally with `<table>.schema.json`.

**Two ways to fill it:**

| You have | Do |
|---|---|
| **Real exports** | Copy the `.csv.gz` files into the 5 folders with Windows Explorer, then go to button **1b** |
| **Nothing yet (test)** | Set **Dummy lines** (e.g. `2000`, or `100000` for a bigger test), then click **1a · Generate dummy input**. Tick **Replace existing data** to overwrite generated data that is already there |

**Button 1a · Generate dummy input**
- Writes 47 tables of fake but consistent data (`.csv.gz`) for the date.
- About 3 s for 2,000 lines, 80 s for 100,000 lines.
- It refuses to overwrite a folder holding real data, unless you tick **Replace existing data**.

**Button 1b · Prepare (validate + features)** runs, in order:
1. **T0 — validation and Parquet cache.** Every table is checked: fields, types, keys, duplicates, joins, coverage, folders, format, `customer_profile`. A table that is not `.csv.gz` (and not generated) blocks the run.
2. **T2 + T3 — features and segments.** Features B-01 … B-12, the S-rule segment, age / tenure / value bands, the micro-segment (nationality × age × tenure), and both actions of every line.

**What you see:**
- *T0 · validation and Parquet cache*: `status` (done / blocked), `pipeline` (train_week0 or train_weekly), the tables and rows, and a plain-language `explanation`.
- *T2 + T3*: the number of lines, the number of micro-segments, and a table of lines per segment.

**If T0 is blocked:** the explanation says why. Fix the file and click **1b** again, ticking **Re-run even if done**.

A second click with the same files is instant: the steps are skipped. Tick **Re-run even if done** to force them.

### 4.3 Step 2 — the target lists

**Where they go:** `data/input/<D>/targets/`, one list per group:

```
targets/target1/t1_constraint.csv.gz      T1C
targets/target1/t1_free.csv.gz            T1F
targets/target2/t2_constraint.csv.gz      T2C
targets/target2/t2_free.csv.gz            T2F
targets/control/control.csv.gz            control
```

**What a list contains:**
- One line per customer, with an **`access_method_val`** column (or **`msisdn`**): the 8-digit number.
- Any other columns (plan, stratum `Final`, region, …) are kept and appear in the details file.
- Production format: `.csv.gz`; `.csv` / `.json` / `.jsonl` are accepted for tests.

Example (`t1_constraint.csv`, before compressing):

```
access_method_val,plan,Final
70012510,FSDR Prepaid Data,Oman|mid|dual|FSDR Prepaid Data
70046318,FSDR Shababiah,India|low|data_only|FSDR Shababiah
```

**Three buttons:**

| Button | What it does |
|---|---|
| **2a · Upload this list** | Choose the **Group** in the dropdown, drop the file in **List**, click. The file is saved as that group's list (it replaces an earlier one) and all the lists are checked again |
| **2b · Generate dummy lists (all groups)** | Writes test lists for the 5 groups: a stratified random draw of the eligible base, scaled to its size (e.g. 307 × 4 + 245 on a 2,000-line base) |
| **2c · Check the lists** | Shows which lists are there and checks them, without changing anything |

You can also copy the files into the folders yourself, then click **2c**.

**What the check shows:**
- The file of each group (or *— missing —*).
- The result, **OK** or **BLOCKED**, with the lines read, dropped, and kept per group.
- A table of issues:

| Level | Check | Meaning | What happens |
|---|---|---|---|
| **blocking** | `line_in_two_groups` | the same MSISDN is in two lists | step 3 refuses to run (HTTP 409): fix the lists |
| **blocking** | `missing_list` | a group has no list | step 3 refuses to run |
| warning | `dropped: not an 8-digit MSISDN` | a malformed number | the line is dropped |
| warning | `dropped: not in the eligible base (D-07)` | not an active prepaid line of the base | dropped |
| warning | `dropped: Nursering segment (B-02)` | tenure < 45 days: never targeted | dropped (contacted groups only) |
| warning | `dropped: in another campaign this week (H-05)` | already in `targeting_auto` this week | dropped (contacted groups only) |
| warning | `duplicate_in_list` | listed twice in the same group | kept once |
| warning | `size` | more than 2 % off the configured size | runs anyway, reported |

Every dropped line and its reason is saved in `data/runs/<D>/score/P1_panel/targets_rejected.csv`.

**No list at all?** Step 3 then draws the five groups at random from the eligible base:
- stratified on nationality × value band × usage type × plan, balanced on age band, tenure band and segment;
- the provided sample table `sample_cg_tg_new_7oct2026`, if present, maps cstr → T1C, free → T1F, cg → control.

### 4.4 Step 3 — run

**Button 3 · Run panel + scoring + final** runs three steps:

| Step | What it does | Output folder |
|---|---|---|
| **P1 panel** | Takes your lists (mode `uploaded`) or draws the groups (mode `drawn`) | `data/runs/<D>/score/P1_panel/` |
| **P6 scoring per group** | For each group: the action, the candidate offers of its universe, the scores, and the group's selection rule (code 3 / code 4 / CC-v1 / CF-v1); 10.1 retry with a second action when an action has no offer | `data/runs/<D>/score/P6_offers/` |
| **Final** | Code 5 on the 4 groups → `input_flytxt` (38 columns); output checks; per-target files; details | `data/runs/<D>/final/` |

**What you see, from top to bottom:**
1. **P1 panel:** the mode and the lines per group.
2. **P6 scoring per group:**
   - the group audit (**PASS** / FAIL);
   - per group: contacted, customers with offers, offers, 10.1 retries.
3. **The final state:**
   - **READY TO PUSH**, or **PUSH BLOCKED — see push/BLOCKED.txt**;
   - the rows in `input_flytxt` and the size of the details;
   - the output checks:

   | Check | Rule |
   |---|---|
   | A-20 | exactly the 38 columns of F-19, in order |
   | A-21 | one row per MSISDN, 8-digit; no control MSISDN in the file |
   | A-22 | date = D, start_date = D + 1, end_date and channels empty, flag_splio = 1 |
   | A-23 | every offer in the catalogue |
   | A-24 | segment is one of the 7 actions |

4. **Per target:**
   - lines, rows in the file, customers dropped by F-04 (no offer in slots 1–4);
   - favourites added (slots 5–6), forced bundles (F-08);
   - slots cleared by F-09 (same type and price), F-11 (same offer twice) and F-12 (conflicting pair).

**Button Show the last final output** shows the result of the last run of that date again, without re-running.

### 4.5 Step 4 — download

The box **4 · Download** lists the files of the run. Click a file to download it:

| File | What it is | Who uses it |
|---|---|---|
| `push/input_flytxt.csv.gz` | **The file to push**: the 38 lower-case columns of code 5, all 4 groups, one row per MSISDN. Written **only when every check passes** | The campaign platform (after approval) |
| `target1/input_flytxt_t1c.csv.gz`, `input_flytxt_t1f.csv.gz` | The same rows, per group | Audit, per-group review |
| `target1/input_flytxt_target1.csv.gz` | T1C + T1F | Target 1 review |
| `target2/input_flytxt_t2c.csv.gz`, `…_t2f`, `input_flytxt_target2.csv.gz` | Same for Target 2 | Target 2 review |
| `target1/details_t1c.csv.gz` … `target2/details_t2f.csv.gz` | **The details ("big data")**: one row per customer of the group, about 220 columns (below) | Validation, analysis |
| `control/control_<D>.csv.gz`, `control/details_control.csv.gz` | The control lines (never contacted) and their details | Measuring the uplift next week |
| `final_rules_report.csv` | Per group: lines, rows in the file, F-04 drops, favourites, forced bundles, F-09 / F-11 / F-12 clearings | Quick review |

**The 38 columns of `input_flytxt`, in order:** `date`, `start_date`, `end_date`, `msisdn`, `channel_1…6`, `opportunity_name_1…6`, `destination_offer_name_1…6`, `flag_splio`, `offer_script_en_1…6`, `offer_script_ar_1…6`, `segment`, `segment_btl_taker`, `segment_btl_taker_upsell`.

**The 6 slots:**

| Slots | Content |
|---|---|
| 1, 2 | monthly offers, highest price first |
| 3, 4 | weekly offers |
| 5 | the customer's favourite (his own most expensive 270-day purchase) |
| 6 | the second favourite, or a forced popular bundle (F-08) |

An empty slot is an empty field, never `nbo_LFM_nan`.

**What the details file holds**, column groups in order:

| Group of columns | Examples |
|---|---|
| Group and source | `GROUP_KEY`, `PANEL_GROUP`, `TARGET`, `SOURCE` (uploaded / drawn), `STRATUM`, `T_*` (the columns of your list), `TARGET_FILE` |
| Profile and bands | `NATIONALITY_GROUP`, `AGE`, `AGE_BAND`, `TENURE_DAYS`, `TENURE_BAND`, `MICRO_SEGMENT`, `VALUE_BAND`, `USAGE_TYPE`, `REGION`, `DV_SEGMENT`, `CURRENT_PLAN` |
| Segment | `SEGMENT`, `RULE_ACTION`, `EXCLUDED` |
| Actions | `ACTION_ORIGINAL`, `ACTION_AGENT`, `ACTION_AGENT_RANK2`, `ACTION_SENT`, `ACTION_AGREEMENT` (all_same / sent_original / sent_agent / other), `DECIDED_BY`, `CONSISTENCY_RETRY`, `PROPENSITY`, `RULES_VERSION` |
| Anchors | `B26_AVG_REV_MONTHLY`, `B26_AVG_REV_DATA_MONTHLY`, … (code 3 / 4 anchors), `F05_MONTHLY`, `F05_DATA_MONTHLY`, … (code 5 anchors) |
| Recommended offers | `reco_m1_*`, `reco_m2_*` (monthly), `reco_w1_*`, `reco_w2_*` (weekly): `btl_name`, `service`, `price`, `score`, `anchor`, `pass`, `selected_by` (B-28 / B-29 / B-30, FR-09 …, CC-04, CC-04 widened, fallback …) |
| Slots | `SLOT_1` … `SLOT_6`, `REASON_1` … `REASON_6` (recommended monthly / weekly, favourite, forced F-08 …), `REMOVED_1` … `REMOVED_6` (e.g. `F-12: BTL_…`) |
| Result | `IN_INPUT_FLYTXT` (1 / 0), `DROP_RULE` (control (H-01), F-04 no offer in slots 1-4, no decision) |
| Features | the 9 model scores `PROBA_*` and their `*_MISSING` flags, `REV_W1…M3`, `VAR_*`, `LOG_*`, usage and spend per period |

**Open the details in Excel:**
1. Choose *Data → Get Data → From File → From Text/CSV*.
2. Excel reads the `.csv.gz` directly in recent versions. Otherwise unzip it with 7-Zip first.

`details_all.parquet` (in `data/runs/<D>/final/`) holds every group in one table, for the notebook or Power BI.

### 4.6 When the push is blocked

If a check fails:
- `push/input_flytxt.csv.gz` is **not** written, and an older one is deleted, so nothing stale can be pushed by mistake.
- `push/BLOCKED.txt` lists the failed checks.
- The per-target and details files are still written, so you can investigate.

Fix the cause (usually the lists, or the input data) and click **3** again.

**Rule H-08:** nothing is pushed without every check passing and a human approval. The SFTP push itself is not automated yet: it comes with a later phase.

### 4.7 Run again

- **Same run date:** step 3 always recomputes P1 → P6 → Final from the current lists. Upload a corrected list with **2a**, then click **3**.
- **Next week:** change the run date, then do steps 1 → 3 again. Each date keeps its own folders, so past weeks stay available.

---

## 5. Demo per customer (tab FSDR Journey)

Use it to show, for one customer, how the decision is made step by step. It works on the dummy folders `data/fsdr_dummy/<date>/`.

**Create a dummy source** in a terminal, once per date:

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --mode small --out data/fsdr_dummy/2026-10-22
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
| 5 | **5 · Competitor comparison** | Each offer against the closest COMPETITOR_A / COMPETITOR_B plan and a verdict |
| 6 | **6 · Full output (save)** | Everything for the customer in one file: `data/fsdr/<date>/outputs/journey/<msisdn>_output.json` (+ a CSV row) |

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
  - the trace of the three agents (action → offer → competitor), with the tools they called;
  - the decision with its reasons;
  - the competitor comparison;
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

## 11. Tab Competitors

| Control | What it does |
|---|---|
| **Operator** (All / COMPETITOR_A / COMPETITOR_B), **Segment**, **Show plans** | The anonymised snapshot of the competitors' public plans |
| **Search the snapshot** + **Search** | Free text search, e.g. `unlimited data weekly` |
| **Compare one FSDR offer** + **Compare** | One FSDR offer (e.g. `BTL_DATA_L2_30D_14.4GB`) against the closest competitor plans |

---

## 12. Using the API directly

Open **http://localhost:8000/docs**. Each route there has a **Try it out** button.

**The V3.0 routes:**

| Route | What it does |
|---|---|
| `POST /fsdr/v3/{run}/generate` | Dummy input of the date in `data/input/<D>/`. Body: `{"n": 2000, "seed": 42, "force": false}` |
| `POST /fsdr/v3/{run}/prepare?force=false` | T0 + T2 + T3 (button 1b) |
| `POST /fsdr/v3/{run}/targets/generate` | Dummy lists (button 2b) |
| `POST /fsdr/targets/{run}/{group}` | Upload one list (`group` = `t1c`, `t1f`, `t2c`, `t2f`, `control`; form field `file`) |
| `GET /fsdr/targets/{run}` | The lists and their check (button 2c) |
| `POST /fsdr/groups/{run}/run` | Panel → scoring → final (button 3). HTTP 409 when the lists block |
| `GET /fsdr/final/{run}` | The last final summary |

**Example in PowerShell:** upload the T2C list and run.

```powershell
curl.exe -F "file=@C:\lists\t2_constraint.csv.gz" http://localhost:8000/fsdr/targets/2026-10-22/t2c
curl.exe -X POST http://localhost:8000/fsdr/groups/2026-10-22/run
```

If `FSDR_API_KEY` is set in `.env`, add the header `-H "X-API-Key: <key>"`; the UI sends it itself.

**The same flow from the terminal**, without the UI:

```powershell
uv run python run.py fsdr-generate --date 2026-10-22 --mode small
uv run python run.py fsdr-load --data-dir data/input/2026-10-22
uv run python run.py fsdr-features --date 2026-10-22
uv run python run.py fsdr-targets --date 2026-10-22 --generate     # or copy your lists into targets/
uv run python run.py fsdr-run-groups --date 2026-10-22
```

See `docs/RUN_FROM_TERMINAL.md` for every command.

---

## 13. Where every file goes

| Folder | What | Committed to git? |
|---|---|---|
| `data/input/<D>/` | The input of the date (5 folders) and `targets/` | no |
| `data/cache/<D>/` | The Parquet cache (16 partitions per table) | no |
| `data/runs/<D>/train/T0_load … T3_segments/` | Validation report, completeness, features, segments, both dictionaries, crosswalk, priors | no |
| `data/runs/<D>/score/P1_panel/` | `panel.parquet`, `group_sizes.csv`, `balance.csv`, `targets_rejected.csv`, `targets_check.json` | no |
| `data/runs/<D>/score/P6_offers/` | The code-2 tables (`TG_40K_constraint_scored_nba`, `TG_40k_free_scored_nba`, `TG_challenger_constraint_scored_nba`, `TG_challenger_free_scored_nba`, `results_scoring_nba`), the 8 reco tables (`recos_monthly/weekly_…`), `traces.csv`, `group_report.csv`, `audit_groups.json`, `decisions.parquet` | no |
| `data/runs/<D>/final/` | `push/`, `target1/`, `target2/`, `control/`, `details_all.parquet`, `final_rules_report.csv`, `_outputs_manifest.json` (rows + SHA-256 of every file), `summary.json` | no |
| `data/fsdr/<D>/` | The V1.0 run folders (tabs of sections 5–10) | no |
| `.env` | Your settings and keys | no |

Every step folder has a `_step.json`: status, start, duration, rows, input fingerprint. A step whose inputs did not change is skipped on the next run.

**Customer data never goes to git.** The folders above are git-ignored, because they hold MSISDNs.

---

## 14. Checking the results in the notebook

```powershell
uv run python run.py notebook                    # opens notebooks/fsdr_results.ipynb in VS Code (then Run All)
uv run python run.py notebook-html --date 2026-10-22   # no browser: an HTML report in data/runs/<D>/
```

Sections of the notebook:
1. the inputs;
2. the steps;
3. validation;
4. completeness;
5. `customer_profile`;
6. the cache;
7. features and segments;
8. **the panel and the four groups**;
9. every step folder;
10. **the final output**: push file, per-target report, details.

Set `RUN_DATE` in the first cell to look at another week.

---

## 15. Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| The page does not open | The app is not running, or still starting | Run `uv run python run.py start`, wait for both "Running" lines, reload |
| Health says `degraded` | vLLM is off | Normal: the agents use their fallbacks. Start `run.py vllm-serve` for the LLM |
| 1b: `T0 … BLOCKED` | A blocking data issue | Read the explanation, or `data/runs/<D>/train/T0_load/validation_report.md`; fix the file; 1b again with *Re-run even if done* |
| 1b: `N table(s) are not .csv.gz` | Real input in another format | Use the Dataiku `.csv.gz` export, or set `FSDR_ALLOW_TEST_FORMATS=true` in `.env` for tests (restart the app) |
| 1a: `holds data that was not generated` | The folder holds real data | Tick **Replace existing data** only if you really want to overwrite it |
| 2c / 3: `BLOCKED`, `line_in_two_groups` | The same MSISDN is in two lists | Remove it from one list, upload it again (2a), run 3 |
| 3: `fsdr-features first` / `not done` | Step 1b was not run for this date | Click **1b** |
| 3: **PUSH BLOCKED** | An output check or the group audit failed | Open `final/push/BLOCKED.txt`; the per-target and details files show the lines concerned |
| Many customers with `DROP_RULE = F-04` | Their spend anchor is above every offer price, so no offer fits (code 3 / 4 rule BR-82); frequent on dummy data, whose catalogue is small | Normal on dummy data; on real data, check the catalogue and the anchors in the details file |
| `forced_F08` is always 0 | The forced bundles `CM_*` / `DM_*` are not in the catalogue | Normal on dummy data; on real data they apply when they are in `btl_catalog` |
| A size warning (`configured 25,000`) | Your lists are smaller or larger than the configured sizes | Information only; change the sizes in `.env` if your design differs |
| A port is busy at start | An earlier app is still running | Section 2.3 (`netstat`, `taskkill`) |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Run date D** | The Thursday of the campaign; every file of the week belongs to it |
| **T1C / T1F / T2C / T2F** | The four contacted groups (section 4.0); **control** is never contacted |
| **Action** | One of 7: CROSS SELL DATA, UPSELL DATA, RETAIN, MAINTAIN, UPSELL, CROSS SELL VOICE, UPSELL VOICE |
| **`action_original`** | Your original segment → action dictionary (code 1), via the segment engine or the crosswalk |
| **`action_agent`** | The agent's dictionary (week 0: a copy of the original; the LLM will update it weekly) |
| **Segment, micro-segment** | The S-rule segment (14); nationality × age band × tenure band (120 cells) |
| **Anchor (B-26 / F-05)** | The customer's spend reference: mean + std of his revenue (codes 3–4), and the code-5 version min(mean + std, max) |
| **Pass 1 / 2 / 3** | Code 3 / 4 price passes: in the band above the anchor; the missing type above the anchor; anything above the anchor |
| **CC-04 / CF-04** | Challenger selection: best score × price inside 1.0–1.5 × the anchor, widened once to 2.0 ×, else the production passes |
| **Slots 1–6** | The 6 offers of `input_flytxt`: 2 monthly, 2 weekly, favourite, second favourite or forced bundle |
| **F-04, F-09, F-11, F-12** | Code-5 rules: drop without offer; same type and price; same offer twice; conflicting pair |
| **H-08** | No push without every check passing and a human approval |
| **Fallback** | The deterministic decision used when the LLM is not available (or not built yet); always marked |
