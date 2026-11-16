# FSDR V4.0: the complete guide, every step one by one

This file follows the whole process in order: installing the laptop, preparing the week's data, running the notebooks step by step (what each step does, which agent works in it, what to check) and handing over the final file. Read it top to bottom the first time; after that, the checklist in §13 is enough each week.

**First real run:** run date **D = 2026-11-19**.
**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU, 24 GB, driver 597.13 (CUDA 13.2), Windows 11.
**Project folder:** `D:\production\nbx-python-upgrade-V4.0` (change the path in the commands if yours is different).

---

## Contents

1. [What the engine does](#1-what-the-engine-does)
2. [What runs where](#2-what-runs-where)
3. [Install the laptop (once)](#3-install-the-laptop-once)
4. [`.env`: the settings (once)](#4-env-the-settings-once)
5. [The LLM server: Ollama or vLLM (once)](#5-the-llm-server-ollama-or-vllm-once)
6. [Prepare the week's data](#6-prepare-the-weeks-data)
7. [Which notebook to run](#7-which-notebook-to-run)
8. [Notebook 03 / 04: every step one by one (recommended)](#8-notebook-03--04-every-step-one-by-one-recommended)
9. [Notebooks 01 + 02: every step one by one (rules only)](#9-notebooks-01--02-every-step-one-by-one-rules-only)
10. [Notebook `fsdr_results`: look at a run](#10-notebook-fsdr_results-look-at-a-run)
11. [The output files](#11-the-output-files)
12. [How the action and the offers are chosen](#12-how-the-action-and-the-offers-are-chosen)
13. [Time, and the weekly checklist](#13-time-and-the-weekly-checklist)
14. [Problems and fixes](#14-problems-and-fixes)
15. [Glossary](#15-glossary)
16. [The other documents](#16-the-other-documents)

---

## 1. What the engine does

Every week, for the prepaid customers of your target lists, the engine:
1. **trains the whole base** (about 1.2M lines). The rules check the data, build the features, and give every line a segment and an action;
2. **scores Target 2**, in two groups of **50,000** customers each:
   - **group a: T2C (Target 2, Constraint).** The LLM chooses the **action** of each customer, then the **offers** of that action's families.
   - **group b: T2F (Target 2, Free).** The LLM chooses the best **DATA / COMBO offers** for the customer; there is no action.
3. **saves** the campaign file `input_flytxt.csv.gz` (38 columns, up to 6 offers per line). The engine sends nothing: you check the file and hand it over yourself.

```
   weekly exports (.csv.gz)          your 2 lists (50k + 50k)
            │                                  │
            ▼                                  │
 ┌─────────────────────────────┐               │
 │ TRAINING  (whole base, CPU)  │               │
 │ T0 validate + cache          │  Agent 1      │
 │ T2 features                  │               │
 │ T3 segments + actions        │  Agents 2, 2b │
 └──────────────┬──────────────┘               │
                ▼                               ▼
 ┌──────────────────────────────────────────────────────┐
 │ SCORING  (Target 2: group a T2C + group b T2F)         │
 │ P1 panel = lists joined with the trained base          │
 │ action per customer (group a)      Agent 3 (LLM, GPU)  │
 │ offers allowed by the rules        Agent 6B (rules)    │
 │ offer scores per customer          Agent 4 (LLM, GPU)  │
 │ the campaign file (code 5)         Agent 7 (rules)     │
 │ checks, can block                  Agent 9 (rules)     │
 │ notes for next week                Agent 8 (LLM)       │
 └──────────────┬───────────────────────────────────────┘
                ▼
     data/output/<D>/final/input_flytxt.csv.gz   (you hand it over)
```

**Not scored:** Target 1 (the production rules) and a control group. They can come back with `FSDR_TARGETS=target1,target2` and `FSDR_CONTROL_SIZE=20000`.

---

## 2. What runs where

| Part | Runs on | Time with 1.2M lines |
|---|---|---|
| Training: T0, T2, T3 | **CPU** (pandas, Parquet) | about 20–25 min |
| Rules: panel, offer rules, code 5, checks | **CPU** | about 3 min |
| LLM agents 1, 2, 2b, 8, 9 (explanations and the dictionary) | **GPU** | about 1–2 min |
| **LLM agents 3 + 4** (one call per customer of Target 2) | **GPU** | **8–14 h with Ollama, 3–6 h with vLLM** (`qwen3:8b`, estimates) |

The GPU is used **only by the LLM**. The data steps are pandas on the CPU: that is normal, and a GPU would not make them faster.

---

## 3. Install the laptop (once)

Full detail: [../docs/UV_ENV_SETUP.md](../docs/UV_ENV_SETUP.md).

**3.1 Copy the folder** `nbx-python-upgrade-V4.0` to `D:\production\`, **without** `.venv\`. Leave out `__pycache__\` and `.ipynb_checkpoints\` too. A copied `.venv` causes "access denied" errors and dead kernels.

**3.2 Install uv, Python 3.12 and the environment**, in PowerShell:

```powershell
pip install uv                                  # or: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
cd D:\production\nbx-python-upgrade-V4.0
uv python install 3.12
uv sync --locked                                # creates .venv from uv.lock (Jupyter, pandas, pytest, …)
```

**3.3 Check:**

```powershell
uv run python -c "import sys; print(sys.version); print(sys.executable)"   # 3.12.x and …\.venv\Scripts\python.exe
uv run python run.py check                                                 # every line [ok]
nvidia-smi                                                                 # NVIDIA RTX PRO 5000, 24463 MiB
```

If a warning says `Failed to inspect Python interpreter … Access is denied`, ignore it: it is about another Python on the laptop.

---

## 4. `.env`: the settings (once)

```powershell
copy .env.example .env
notepad .env
```

Check these lines (the values shown are the ones you need):

```ini
# --- the targets: Target 2 only, group a 50k + group b 50k, no control ---
FSDR_TARGETS=target2
FSDR_T2_CONSTRAINT_SIZE=50000
FSDR_T2_FREE_SIZE=50000
FSDR_CONTROL_SIZE=0

# --- privacy, local only ---
FSDR_PII_SALT=<a long random secret>
FSDR_LOCAL_ONLY=true

# --- vLLM only (notebook 04) ---
VLLM_API_KEY=<a long random secret>
# VLLM_HF_MODEL=Qwen/Qwen3-8B-AWQ      # or Qwen/Qwen3-14B-AWQ
# VLLM_SERVED_NAME=qwen3:8b            # or qwen3:14b
# VLLM_OFFLINE=1                       # once the model is downloaded
```

- **An older `.env`** (from an older `.env.example`) may say 25000 / 25000 / 20000. Those values override the new defaults: change them.
- **Never** send or commit `.env`.

---

## 5. The LLM server: Ollama or vLLM (once)

Choose one. Both run the same agents; vLLM is faster, but it needs WSL2. See [../docs/VLLM_VS_OLLAMA_SECURITY.md](../docs/VLLM_VS_OLLAMA_SECURITY.md) for the comparison and the email for the security team.

### 5.1 Ollama (notebook 03): native Windows, already installed

```powershell
ollama --version
ollama pull qwen3:8b                 # about 5 GB, the only download (qwen3:14b: better, about 2x slower)
setx OLLAMA_NUM_PARALLEL 8           # several customers at once
setx OLLAMA_FLASH_ATTENTION 1
```

Then quit Ollama (tray icon → Quit) and start it again. **At each session:** start the Ollama app.

### 5.2 vLLM (notebook 04): in WSL2, after the security team's approval

```powershell
wsl --install -d Ubuntu              # PowerShell as administrator; restart; create the Ubuntu user
wsl -e nvidia-smi                    # the RTX PRO 5000 must appear (never install a driver inside Ubuntu)
cd D:\production\nbx-python-upgrade-V4.0
uv run python run.py vllm-install    # must print: compute capability 12.0 -> vLLM profile: modern … vllm 0.32.0
```

**At each session**, in a second terminal that you keep open:

```powershell
uv run python run.py vllm-serve      # wait for: Application startup complete
```

The first start downloads the model (about 6 GB). After that, set `VLLM_OFFLINE=1` in `.env`.

---

## 6. Prepare the week's data

### 6.1 Create the folders, and get the list of files

```powershell
uv run python run.py fsdr-folders --date 2026-11-19                   # notebooks 03 / 04: data\weekly\2026-11-19\
uv run python run.py fsdr-folders --date 2026-11-19 --layout split    # notebooks 01 + 02: data\training + data\scoring
```

The command creates every folder and prints the exact file name of every table, marked `required` or `optional`.

### 6.2 Copy the exports

**Rules for every file:**
- the file name is the exact table name + **`.csv.gz`** (the only format accepted for real data);
- gzip, UTF-8, comma separator, one header row;
- IDs exported as **text**;
- **never** open and save an export in Excel (it changes IDs and dates).

**The weekly folder `data\weekly\2026-11-19\`** (notebooks 03 / 04):

| Sub-folder | Files |
|---|---|
| `usage\` | `activities_w1_auto`, `activities_w2_auto`, `activities_w3_auto`, `activities_w4_auto`, `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto`, `subs_amt_auto`, `usage_data_voice_payg_tg`; optional: `paygo_activity_auto` |
| `models\` | the 11 model tables: `DAIM_CHURN_GLOBAL_SCORING_TD.prediction`, `DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction`, `DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored`, `DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored`, `PROPENSITY_REACTIVATION_15D_401_SCORING.Scored`, `APPETENCY_DATA_204_SCORING_TD.prediction`, `APPETENCY_VOICE_204_SCORING_TD.Scored`, `CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring`, `REV_DECREASE_204_SCORING_TD.predictions`, `DAIM_FRAGILITY_DATA_SCORING_TD.prediction`, `DAIM_FRAGILITY_VOICE_SCORING_TD.prediction` |
| `subscription\` | `base_prepaid`, `consumer_aact_account`, `access_method_id_auto`, `subs_btl_auto`, `sub_btl_3months`, `subscriptions_tg` |
| `catalog\` | `btl_catalog`, `BTL_script_english_arabic_ok`, `conflicting_offers_h` |
| `others\` | `party_auto`, `targeting_auto`, `targeting_dataiku_auto`; optional: `consent_contact` (recommended: do-not-contact), `customer_profile`, `location_auto`, `DEVICEBEHAVIORSEGMENTATION.dv_segment` |

**Not needed** in the weekly folder: the `*_score` tables and `party_scoring`. A week with fresh `*_auto` history is a "week-0" run, which reads the `*_auto` tables.

### 6.3 Copy the two target lists

| List | Path (weekly folder) | Lines |
|---|---|---|
| group a (T2C) | `data\weekly\2026-11-19\target2\a_constraint\target2_a.csv.gz` | 50,000 |
| group b (T2F) | `data\weekly\2026-11-19\target2\b_free\target2_b.csv.gz` | 50,000 |

- **Columns:** at least `access_method_val`, the 8-digit MSISDN as text, without a `968` prefix. Any other column is kept and copied into the details file.
- A line must be in **only one** list.
- (With notebooks 01 + 02, the lists go in `data\scoring\2026-11-19\target2\…`.)

---

## 7. Which notebook to run

| You want | Notebook | Folder |
|---|---|---|
| **Every agent, with the LLM on the GPU (recommended)** | **`03_gpu_agents_step_by_step`** (Ollama) | `data\weekly\<D>\` |
| The same, faster (vLLM in WSL2) | **`04_vllm_agents_step_by_step`** | `data\weekly\<D>\` |
| Rules only, no LLM | `01_training_step_by_step`, then `02_scoring_step_by_step` | `data\training\<D>\` + `data\scoring\<D>\` |
| Look at a finished run | `fsdr_results` | read-only |

**Open a notebook:**

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab                   # the browser opens at http://localhost:8888
```

Open the notebook, then click **Kernel → Restart Kernel and Run All Cells**, or run the cells one by one with Shift+Enter.

**How every notebook is built:**
- **The settings cell** comes first. The defaults are already your real run: `RUN_DATE = "2026-11-19"`, `GENERATE = False`, `GENERATE_TARGETS = False`.
- **Part A** holds the engine code, one cell per module. It runs in a few seconds; you never need to read it.
- **Part B** holds the steps described below.
- **The last cell** is a proof check: `all defined in this notebook: True`, `vowpalwabbit loaded: False`.

---

## 8. Notebook 03 / 04: every step one by one (recommended)

Notebook 03 (Ollama) and notebook 04 (vLLM) have **the same steps**. Only the LLM server and three settings differ.

### Settings (first code cell)

| Setting | Default | What it does |
|---|---|---|
| `RUN_DATE` | `"2026-11-19"` | the week: the folder `data\weekly\<D>\` |
| `GENERATE`, `GENERATE_TARGETS` | `False`, `False` | your real exports and lists (`True` = dummy data for a test; real data is never overwritten anyway) |
| `OLLAMA_URL` (03) / `LLM_URL` (04) | `127.0.0.1:11434` / `127.0.0.1:8001` | the local LLM server; any other address is refused |
| `MODEL` | `"auto"` | the best model installed (03) or the model the server serves (04) |
| `PARALLEL` | `4` (03; use **8** with `OLLAMA_NUM_PARALLEL=8`) / `32` (04) | customers sent to the LLM at the same time |
| **`TIME_BUDGET_MIN`** | `20` | minutes for agents 3 + 4. Customers are taken in priority order (retention first, a and b interleaved); those past the budget get Agent 2b's cell policy. **Set `600` to give every customer the LLM (overnight).** |
| `MAX_LLM_LINES` | `None` | or a number: at most that many customers for agents 3 + 4 |
| `AGENT_TARGET` | `"target2"` | the agents decide group a and group b |
| `EPSILON` | `0.1` | Agent 3's exploration share (§12) |
| `NBO_SHORTLIST` | `8` | offers per validity that Agent 4 scores per customer (the best by the rules) |
| `SEED`, `FORCE`, `PREVIEW_ROWS` | `7`, `True`, `8` | reproducibility, re-run every step, rows shown |

### The LLM client cell

It connects to the local server only, and keeps every answer in `data\runs\<D>\agents\llm_cache.jsonl`, so a re-run of the same week costs nothing. If a call fails, that customer gets the rule decision instead.

### G1 · The GPU and the LLM server

| | |
|---|---|
| **Does** | reads `nvidia-smi`; checks that the LLM server answers; lists the models; picks `MODEL`; loads it |
| **Check** | `GPU : NVIDIA RTX PRO 5000 …`; `MODEL : qwen3:8b`; with Ollama, **`in GPU memory: 100%`**; with vLLM, `first answer in … s: {'ok': True}` |
| **If not** | 03: start the Ollama app or `ollama pull qwen3:8b`. 04: start `run.py vllm-serve` |

### G2 · Speed test and the time estimate

| | |
|---|---|
| **Does** | 10 customer-sized NBA prompts and 10 NBO prompts, `PARALLEL` at a time |
| **Check** | `… s per NBA call, … s per NBO call`, **`0 failed`**, and `production week (50k T2C + 50k T2F): about … h` |
| **Use it** | to choose `TIME_BUDGET_MIN`. If every call fails, the notebook stops: restart the server |

### Step 1 · The weekly folder

| | |
|---|---|
| **Does** | lists the tables found in `data\weekly\<D>\`, per sub-folder |
| **Check** | every required table is present (§6.2) |
| **If not** | `no exports in …`: run `fsdr-folders` and copy the files |

### Step 2 · Agent 1, the data agent (T0)

| | |
|---|---|
| **Rules** | check every table against the data contract (columns, types, keys, value ranges, date windows) and the week's completeness, then write the Parquet cache (16 partitions). A **blocking** issue stops the run |
| **LLM** | explains the report in plain words and says what to fix first. **It cannot unblock anything** |
| **Check** | `T0 DONE -- pipeline train_week0`; `blocking: 0`. The missing `*_score` tables are warnings ("not needed by the pipeline"): that is expected |
| **Writes** | `data\runs\<D>\train\T0_load\validation_report.md`, `completeness.csv`, `data\cache\<D>\` |

### Step 3 · Agent 2, segmentation (T2 + T3)

| | |
|---|---|
| **Rules** | **T2** builds 113 features for every line: eligibility, profile, weekly and monthly usage and revenue, trends, and the 11 model scores. A missing score gets the median of the nationality group. **T3** applies the S-rules (first match wins) and gives every line one of 14 **segments**, the bands (age, tenure, value, usage), the **micro-segment** (nationality \| age \| tenure) and the two actions. Lines under 45 days of tenure (Nursering) are excluded |
| **LLM** | one sentence per segment: who these customers are and what to watch (14 calls) |
| **Check** | about 1.2M lines; every segment non-empty; the excluded Nursering count |
| **Writes** | `data\runs\<D>\train\T2_features\`, `T3_segments\` (`segment_counts.csv`, `microsegment_counts.csv`, …) |

### Step 4 · Agent 2b, the agent dictionary

| | |
|---|---|
| **Does** | for each **segment × nationality cell** (up to 60 cells with lines), the LLM proposes 1–2 actions. It uses the cell's evidence: size, ARPU, at-risk / fragile / data / voice shares, and **last week's reward rate per action** |
| **Rules** | the check refuses an unknown action, an action for an excluded segment, and **an upsell as the top action of an at-risk segment (H-06)**; a refused cell keeps its current action |
| **Check** | `dict_segm_actions_agent_v<N>: … cells with a NEW top action …, 0 refused`; the table of changed cells and their reasons |
| **Writes** | `T3_segments\dict_segm_actions_agent_v<N>.json` and `agent_dictionary_check_v<N>.json`. The new version is applied to every line, and **next week starts from it** |

### Step 5 · Target lists and the panel (P1)

| | |
|---|---|
| **Does** | reads your 2 lists, checks them, and joins each line with its training results: segment, actions, bands, features |
| **Checks of the lists** | a line in both lists, or a missing list, **blocks**. A line is **dropped** (listed in `targets_rejected.csv`) if it is not 8 digits, not in the eligible base, in Nursering, or in another campaign this week. A list more than 2% off 50,000 gets a **warning** |
| **Check** | `lists are YOURS`; `P1 DONE`, about 50,000 + 50,000 lines |
| **Writes** | `data\runs\<D>\score\P1_panel\` |

### Step 6 · Agent 3, NBA: the action of each group-a customer

| | |
|---|---|
| **Does** | for each **T2C** customer within the budget, the LLM scores the allowed actions from the customer's facts: segment, value, usage, tenure, revenue trend, churn, appetency, last week's targeting. **No MSISDN, nationality or age is sent** |
| **Rules** | allowed actions = the 7 actions minus an upsell for an at-risk segment (H-06). **The code draws the action**: p = (1 − ε)·score/Σscores + ε/k; the propensity p is logged. Past the budget, or after an invalid answer: the cell policy of step 4 |
| **Check** | `llm_nba` (the agent's top action) / `llm_nba_sampled` (another action drawn from its scores) / `fallback_cell_policy`; the cross-table of cell policy against the agent's action |
| **Writes** | `data\runs\<D>\agents\agent3_nba.csv` |

### Step 7 · Agent 4, NBO, with the challenger rules (Agent 6B)

| | |
|---|---|
| **Rules first** | for every Target-2 customer, **Agent 6B** gives the allowed offers. Group a (CC-v1): the families of the customer's action. Group b (CF-v1): L2 DATA / COMBO. Only offers bought in the base in the last 270 days count |
| **LLM** | for each customer within the budget, it scores the best `NBO_SHORTLIST` offers per validity (0–100) |
| **Rules after** | the passes still choose: price **1.0–1.5x the customer's own spend** (+0.5 OMR for an upsell, widened once to 2.0x), one offer per family, 2 monthly + 2 weekly, then the 3rd monthly and 3rd weekly (EXT-6). Group a retries once with the second action if no offer fits |
| **Check** | `group audit PASS`; per group `llm_scored_offers` > 0 and `decided_by` |
| **Writes** | `data\runs\<D>\score\P6_offers_target2\` (`group_report.csv`, `audit_groups.json`, `traces.csv`, the reco tables) |

### Step 8 · Agent 7, code 5: the final file of Target 2

| | |
|---|---|
| **Rules** | the 38-column campaign file: slots 1–2 monthly, 3–4 weekly, 5–6 the third offers; favourites and forced bundles fill an empty slot 5 / 6; slots cleared for the same type and price (F-09), the same offer (F-11) or conflicts (F-12); a line with no offer is dropped (F-04); English / Arabic scripts, dates, channels. Checks A-20 … A-24 |
| **Check** | `target2: final READY, … rows`; the details preview |
| **Writes** | `data\output\<D>\target2\` (the file, per-group files, `details_target2.parquet` of about 240 columns per line, the checks) |

### Step 9 · Agent 9, the auditor

| | |
|---|---|
| **Rules** | the group audit (A-12 universe, A-12b catalogue, A-17 rules version, A-30 / A-31 sizes and balance) and the file checks. **Any FAIL and the final file is not written** |
| **LLM** | explains the verdict; if it disagrees, the rules win |
| **Check** | `Verdict of the rules: READY` |

### Step 10 · Agent 8, strategy

| | |
|---|---|
| **Does** | compares group a with group b, and the customers the agents decided with those on the cell policy: lines, lines in the file, offers per line, action mix. The LLM writes 3 points for next week; **a proposal only**, nothing changes by itself |

### Step 11 · The final file

| | |
|---|---|
| **Does** | writes `data\output\<D>\final\input_flytxt.csv.gz` (Target 2, groups a and b), **only if** Target 2 passed and no MSISDN is in both groups; otherwise `final\BLOCKED.txt` with the reason |
| **Check** | `final file READY -- … rows` |
| **Then** | check `details_target2.parquet` with the business, and **hand over the final file yourself**. The engine sends nothing |

### Time per agent, and the proof

- **Time per agent:** the calls, tokens and seconds of each agent, plus the measured time per customer extrapolated to 100,000 customers. Saved in `data\runs\<D>\agents\agent_times.json`.
- **Proof:** `all defined in this notebook: True`, `vowpalwabbit loaded: False`, `api / gradio loaded: False`.

---

## 9. Notebooks 01 + 02: every step one by one (rules only)

Use these without an LLM: the decisions are the rule fallbacks (`DECIDED_BY = fallback_rule_action`). The data goes in two folders (`fsdr-folders --layout split`).

### Notebook 01 · Training

| Step | Does | Check |
|---|---|---|
| Your machine | Python, kernel, GPU (for information) | `kernel : OK, the project's .venv` |
| Settings | `RUN_DATE = "2026-11-19"`, `GENERATE = False` | — |
| 0 · Dummy data | only with `GENERATE = True` | — |
| 1 · The two input folders | every table, its side and folder | `misfiled: none`, `duplicates: none` |
| 2 · T0, Agent 1 | validation, completeness, cache | `T0_load DONE` |
| 3 · The Parquet cache | 16 partitions per table | `verify_cache: OK` |
| 4 · T2 features | 113 features of the whole base | eligible lines; the missing-score shares |
| 5 · T3, Agents 2 / 2b | segments, bands, micro-segments, `action_original` + `action_agent` (dictionary v0, or the last version) | segment chart; action cross-table |
| 6 · Summary | the status of each step | T0, T2, T3 all `done` |

### Notebook 02 · Scoring of Target 2

| Step | Does | Check |
|---|---|---|
| Settings | `RUN_DATE = "2026-11-19"`, `GENERATE_TARGETS = False` | — |
| 1 · Training must be done | T0 / T2 / T3 of the same date | `training OK` |
| 2 · The target lists | your 2 lists, their checks | `lists are YOURS`, kept / dropped |
| 3 · P1 panel | lists joined with the base | sizes, balance |
| 4 · Target 2 P6 (Agents 3, 4 as rule fallbacks; 6B; 9) | the offers of T2C and T2F | `group audit PASS` |
| 5 · Target 2 final (Agent 7, code 5) | the 38-column file and details | `target2 final READY` |
| 6 · The final file (Agent 9) | `final\input_flytxt.csv.gz` | `final file READY` |
| 7 · Validate | 38 columns, 8-digit MSISDNs, offers per line 1–6 | every check `PASS` (if blocked, the reason is printed) |

---

## 10. Notebook `fsdr_results`: look at a run

A read-only view of a run date: data and validation, completeness, cache, features, segments, panel, scoring, and the output files. Set `RUN_DATE` (`None` = the latest run), then Run All. Or, as an HTML report:

```powershell
uv run python run.py notebook-html --date 2026-11-19
```

---

## 11. The output files

```
data\
├── weekly\2026-11-19\              YOUR INPUT (notebooks 03 / 04): the exports + target2\a_constraint, target2\b_free
├── runs\2026-11-19\
│   ├── train\T0_load\              validation_report.md, completeness.csv, summary.json
│   ├── train\T2_features\          base_features\part-NN.parquet, feature_summary.csv, stats.json
│   ├── train\T3_segments\          segments\part-NN.parquet, segment_counts.csv, microsegment_counts.csv,
│   │                               dict_segm_actions_agent_v<N>.json (agent 2b; next week starts from it)
│   ├── score\P1_panel\             panel.parquet, balance.csv, targets_rejected.csv
│   ├── score\P6_offers_target2\    the reco tables, group_report.csv, audit_groups.json, traces.csv
│   └── agents\                     agent3_nba.csv, llm_cache.jsonl, agent_times.json
├── cache\2026-11-19\               the Parquet cache (16 partitions per table)
└── output\2026-11-19\
    ├── target2\                    input_flytxt_target2.csv.gz, a_constraint\ and b_free\ (per-group file + details),
    │                               details_target2.parquet (≈ 240 columns per line), output_checks.json, summary.json
    └── final\input_flytxt.csv.gz   ← THE FILE YOU HAND OVER (or BLOCKED.txt with the reason)
```

**Never** send or commit `data\`, and never commit a notebook saved with its outputs: they contain MSISDNs.

---

## 12. How the action and the offers are chosen

**The LLM is never trained.** Its weights (Qwen3) stay fixed. What changes from week to week is the **agent dictionary** (Agent 2b), which uses last week's reward evidence.

| Layer | Who decides | From what | Group |
|---|---|---|---|
| 1. Segment | rules (S-rules) | churn, appetency, revenue, usage, tenure | a and b |
| 2. Cell policy (segment × nationality) | **Agent 2b (LLM)**, checked by H-06 | the cell's size, value, risk shares, last week's reward rate per action | a |
| 3. Customer action | **Agent 3 (LLM)** + the ε draw by code (propensity logged) | the customer's facts (no MSISDN, nationality or age) | a |
| 4. Allowed offers | **Agent 6B (rules)** | the action's families (a) or L2 DATA / COMBO (b); bought in the last 270 days | a and b |
| 5. Offer scores | **Agent 4 (LLM)** | the customer's facts, the action, the price anchor, each offer | a and b |
| 6. Chosen offers | **rules** (passes, EXT-6, code 5) | price 1.0–1.5x the customer's own spend, one offer per family, the 6 slots | a and b |
| 7. Can the file go out? | **Agent 9 (rules)** | the audit and the checks | a and b |

- **The micro-segment** (nationality \| age \| tenure) is computed for every line. It is used for the price anchor of a customer with no spend history, for balancing the lists, and in the reports. **It does not choose the action.**
- **Exploration ε = 0.1:** a small share of customers get an action other than the agent's top choice. Together with the logged propensity, this lets next week's results show which action really works, without bias.

---

## 13. Time, and the weekly checklist

**One week with a 1.2M base** (estimates; G2 measures the real speed):

| | Ollama (`qwen3:8b`) | vLLM (`Qwen3-8B-AWQ`) |
|---|---|---|
| Training (T0, T2, T3, CPU) | 20–25 min | 20–25 min |
| Agents 1, 2, 2b | 1–2 min | < 1 min |
| Lists, panel, rules, code 5, final file | 3 min | 3 min |
| **Agents 3 + 4 (100k customers)** | **8–14 h** | **3–6 h** |
| **Total** | **≈ 8.5–14.5 h** | **≈ 3.5–6.5 h** |
| Total with the 14B model | ≈ 16.5–28.5 h | ≈ 6.5–12.5 h |

To shorten the week, use `TIME_BUDGET_MIN`: customers are taken by priority, and the rest get the cell policy.

**Every week** (next: 2026-11-26):

1. `uv run python run.py fsdr-folders --date 2026-11-26`
2. Copy the new exports (`.csv.gz`) and the 2 new lists.
3. Start Ollama (or `run.py vllm-serve`), then `uv run jupyter lab`.
4. Notebook 03 / 04: set `RUN_DATE = "2026-11-26"`, then Run All.
5. Check: T0 `DONE`, G2 `0 failed`, `group audit PASS`, `final file READY`.
6. Business check of `details_target2.parquet`, then hand over `data\output\2026-11-26\final\input_flytxt.csv.gz`.
7. **Keep** `data\runs\2026-11-19\`: Agent 2b starts from its dictionary.

---

## 14. Problems and fixes

| What you see | Fix |
|---|---|
| `kernel : NOT the project's .venv` / `ModuleNotFoundError: pandas` | Start JupyterLab with `uv run jupyter lab` from the project folder |
| The kernel dies, "access denied", `msvcp140.dll` | Delete `.venv\Lib\site-packages\00_vw_preload.pth` if it exists; rebuild `.venv` (`Remove-Item -Recurse -Force .venv; uv sync --locked`) |
| `no exports in …` | `run.py fsdr-folders --date <D>`, then copy the files |
| `T0 BLOCKED` | Read `validation_report.md`: a missing or misnamed table, or a file that is not `.csv.gz` |
| A list group `MISSING` | Wrong file name or folder (§6.3) |
| Many lines "not in the eligible base" | MSISDN format: `968` prefix, lost leading zero, or Excel damage; export the IDs as 8-digit text |
| G1: Ollama / vLLM does not answer | Start the Ollama app / `run.py vllm-serve` (wait for `Application startup complete`) |
| G1: `in GPU memory` below 100% | Close the other GPU programs, or use `qwen3:8b` |
| G2: `every test call failed` | The server stopped: restart it, then G1 and G2 again |
| vLLM: `HTTP Error 401` | `VLLM_API_KEY` changed: restart the server |
| vLLM: `Engine process … died` | WSL2 ran out of RAM: lower `VLLM_MAX_SEQS` and `PARALLEL` to 16 |
| `NO FINAL FILE: it is blocked` | The reason is printed (audit FAIL or a failed check); fix it and run the scoring steps again |
| The group sizes are 25,000 / a control group appears | `.env` still has the old values (§4) |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **D**, run date | the date of the week's exports (`YYYY-MM-DD`), e.g. 2026-11-19 |
| **T2C / group a** | **T**arget **2**, **C**onstraint: offers from the families of the customer's action; 50,000 lines |
| **T2F / group b** | **T**arget **2**, **F**ree: the best L2 DATA / COMBO offers, whatever the action; 50,000 lines |
| T1C / T1F, control | Target 1 (production rules) and the never-contacted group: **off** in V4.0 |
| T0, T2, T3 | training steps: validate and cache; features; segments and actions |
| P1, P6, code 5 | scoring steps: panel; offers; the 38-column campaign file |
| Segment | one of 14 S-rule groups (Retention, RGS, Onboarding, …); Nursering is excluded |
| Micro-segment | nationality \| age band \| tenure band |
| Cell policy | the action of a segment × nationality cell, from Agent 2b's dictionary |
| ε, propensity | the exploration share (0.1), and the logged probability of the action sent |
| EXT-6 | the rule that adds the 3rd monthly and 3rd weekly offers (slots 5 and 6) |
| H-06 | no upsell as the top action of an at-risk segment |
| `DECIDED_BY` / `score_source` | who decided each action / offer score: `llm_nba`, `fallback_cell_policy`, `llm:<model>`, `fallback:E-O3+E-O1` |

---

## 16. The other documents

| Document | For |
|---|---|
| [../docs/REAL_DATA_2026-11-19.md](../docs/REAL_DATA_2026-11-19.md) | the short checklist of the first real run |
| [../docs/UV_ENV_SETUP.md](../docs/UV_ENV_SETUP.md) | the uv environment, in detail |
| [../docs/GPU_AGENTS_NOTEBOOK.md](../docs/GPU_AGENTS_NOTEBOOK.md), [../docs/VLLM_AGENTS_NOTEBOOK.md](../docs/VLLM_AGENTS_NOTEBOOK.md) | notebooks 03 and 04 in detail |
| [../docs/VLLM_VS_OLLAMA_SECURITY.md](../docs/VLLM_VS_OLLAMA_SECURITY.md) | Ollama against vLLM, and the WSL2 email for the security team |
| [01_training_step_by_step.md](01_training_step_by_step.md), [02_scoring_step_by_step.md](02_scoring_step_by_step.md), [03_gpu_agents_step_by_step.md](03_gpu_agents_step_by_step.md), [04_vllm_agents_step_by_step.md](04_vllm_agents_step_by_step.md), [fsdr_results.md](fsdr_results.md) | the short run guide of each notebook |
| [../docs/SETUP_NEW_LAPTOP.md](../docs/SETUP_NEW_LAPTOP.md), [../docs/RUN_FROM_TERMINAL.md](../docs/RUN_FROM_TERMINAL.md) | the whole project, and every terminal command |
