# The weekly run, explained step by step

This document explains **everything the weekly run does**, from the exports you paste to the final file: the data,
the dates, how we **train**, how we **score**, what each **agent** decides, and **how long** each part takes.

It follows the last notebook, `notebooks3/03_agents_nba_fast_ollama_gpu.ipynb` (and `04_…_vllm_gpu`, which is the
same with another LLM server), plus the data notebook it needs, `notebooks3/02_prepare_week.ipynb`.

Contents:

1. [The big picture](#1-the-big-picture)
2. [The two dates: D and T](#2-the-two-dates-d-and-t)
3. [The data](#3-the-data)
4. [Notebook 02: preparing the week](#4-notebook-02-preparing-the-week)
5. [Notebook 03: the settings and Part A](#5-notebook-03-the-settings-and-part-a)
6. [How the LLM is used (all agents)](#6-how-the-llm-is-used-all-agents)
7. [G1 and G2: the GPU and the speed test](#7-g1-and-g2-the-gpu-and-the-speed-test)
8. [Training: steps 1 to 4](#8-training-steps-1-to-4)
9. [Scoring: steps 4b to 7](#9-scoring-steps-4b-to-7)
10. [The final file: steps 8 to 12](#10-the-final-file-steps-8-to-11)
11. [The reports: time table, FAST_MODE figures, G3](#11-the-reports-time-table-fast_mode-figures-g3)
12. [Time](#12-time)
13. [What you check, step by step](#13-what-you-check-step-by-step)
14. [Where everything is written](#14-where-everything-is-written)
15. [Glossary](#15-glossary)

---

## 1. The big picture

Every week we choose, for each customer of **Target 2**, **one action** (for example `UPSELL DATA`) and **up to 6
offers** (3 monthly + 3 weekly). The result is one file, `input_flytxt.csv.gz`, which you check and hand over.

Target 2 has two groups of about 50,000 customers each:

| Group | File | What is decided | Offer rules |
|---|---|---|---|
| **a = T2C** (constraint) | `target2/a_constraint/target2_a.csv.gz` | the action (agent 3), then the offers that fit it (agent 4) | CC-v1 |
| **b = T2F** (free) | `target2/b_free/target2_b.csv.gz` | the offers only (agent 4); the action is not used | CF-v1 |

There is no control group by default, and Target 1 is not scored.

The run has two halves, like the classic NBA recipe:

```
              TRAINING (learn from last week)                     SCORING (decide this week)
   *_auto tables + party_auto, anchored on T              *_score tables + party_scoring, anchored on D
   ───────────────────────────────────────────            ──────────────────────────────────────────────
   T0  check the data, build the cache                     4b  describe the target customers at D
   T2  features of the whole base (2M lines)               5   the panel: the target lists + their profile
   T3  segments, bands, last week's reward                 5b  the customer profiles (FAST_MODE)
   2b  agent dictionary: best action per cell              6   agent 3: the action of each T2C customer
                                                           7   agent 4 + rules: the offers of everyone
                                                           8-11 final file, audit, strategy
```

**Who decides what:**

- **The rules (code) always decide what is allowed and what blocks.** The data contract, the S-rules for the
  segments, the price bands, the audits.
- **The LLM chooses inside what the rules allow.** It picks the action per cell (agent 2b) and per customer (agent
  3), and it scores the offers (agent 4). Agents 1, 2, 8 and 9 only **explain**; they cannot change anything.
- **If the LLM fails or is too slow, the rule answer is used**, and the notebook counts it.

Everything runs on this laptop. The LLM is a local server (Ollama on `127.0.0.1:11434`, or vLLM on
`127.0.0.1:8001`). No MSISDN is ever put in a prompt.

---

## 2. The two dates: D and T

| Date | Name | How it is found | Used for |
|---|---|---|---|
| **D** | run date | `RUN_DATE` in the notebooks, or read from the data by notebook 02 | scoring: the `*_score` tables describe the customer **now** |
| **T** | anchor | **D − 7** (t2 = t1 − 7, `ANCHOR_LAG_DAYS`) | training: the `*_auto` tables describe the customer **before last week's campaign** |

**Why two dates?** To learn from last week's campaign, we must look at the customer **as they were before** the
campaign (at T). To decide this week, we look at the customer **as they are now** (at D). Mixing the two would let
the model "see" the reward in its own inputs.

**t1 and t2.** In notebooks3, **D = t1** = the most recent day of the data (the last purchase day of `subs_btl`),
unless `RUN_DATE` is set, and **T = t2 = t1 − 7**. Example: t1 = 2026-11-19 gives t2 = 2026-11-12.

**The windows** (all inclusive):

| Window | Relative to | Days | Example (D = t1 = 2026-11-19, T = t2 = 2026-11-12) |
|---|---|---|---|
| last week's targeting (`targeting_dataiku_auto.date1`, the latest campaign) | D | D−7 … D−1 (`TARGETING_WINDOW`) | 2026-11-12 … 2026-11-18 |
| last week's reward (BTL bought, `subs_btl_auto`) | D | D−7 … D (`REWARD_WINDOW`) | 2026-11-12 … 2026-11-19 |
| arbitration (`targeting_auto.start_date`) | D | D−7 … D | 2026-11-12 … 2026-11-19 |
| past interactions (`sub_btl_3months`) | D | D−270 … D−1 | |
| subscriptions W1 / W2 / W3 / W4 (`subs_amt_*`) | anchor | A−6…A, A−13…A−7, A−20…A−14, A−27…A−21 | training (A = T): W1 = 11-06 … 11-12 |
| subscriptions M1 / M2 / M3 | anchor | A−29…A, A−59…A−30, A−89…A−60 | |
| activities W1 … W4 (`activities_*`) | anchor, **anchor day excluded** | A−7…A−1, A−14…A−8, … | training: W1 = 11-05 … 11-11 |
| activities M1 / M2 / M3 | anchor, anchor day excluded | A−28…A−1, A−56…A−29, A−85…A−57 | |

For **training** the anchor A is **T**. For the **scoring view** (step 4b) the same offsets are used with A = **D**.

---

## 3. The data

You paste the Dataiku exports into `data/data_raw/` (`.csv` or `.csv.gz`, with or without `_hdfs` / `_hfds`).
Notebook 02 gives each file its contract name and puts it in the week folder.

| Group | Tables | Used by |
|---|---|---|
| **base** | `base_prepaid`, `consumer_aact_account`, `access_method_id_auto` | both halves: who is a customer, their account and line |
| **training** (anchor T) | `party_auto`, `subs_amt_auto`, `activities_w1…w4_auto`, `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto` | T2 / T3 |
| **last campaign** | `targeting_dataiku_auto` (who was targeted, with which segment), `subs_btl_auto` (who bought), `targeting_auto` (other campaigns) | T3 (the reward), agent 2b, P1 |
| **scoring** (anchor D) | `party_scoring`, `subs_amt_score`, `activities_w1…m3_score` | step 4b, then agents 3 / 4 |
| **models** | the 11 model score files (churn, reactivation, appetency data / voice, fragility, `REV_DECREASE_204_SCORING_TD.predictions`, …) | the features and the S-rules |
| **offers and final file** | `btl_catalog`, `BTL_script_english_arabic_ok`, and the reference tables | P6 and code 5 |
| **target lists** | `target2_a` (group a), `target2_b` (group b) | P1: who we score |

**Model scores** are probabilities between 0 and 1 (for example `PROBA_CHURN`). A missing score gets a **flag** and
the **median of its nationality group** (computed on the whole base). The flag matters: **a missing score never puts a
customer in a risk segment**.

---

## 4. Notebook 02: preparing the week

Runs on the CPU. It changes nothing in `data_raw/`.

| Step | What it does | What you look at |
|---|---|---|
| 1 · What is in `data_raw/` | lists every file, gives it its contract name (`base_prepaid_hdfs` → `base_prepaid`; typos and case fixed), checks the header | unknown files, files without a header |
| 2 · The tables of the NBA logic | which of the 41 tables are there, which are missing, which are optional | **missing** required tables |
| 3 · The two dates | reads D = t1 from the data (the most recent purchase day of `subs_btl`), T = t1 − 7, and every window; checks that each `*_score` table is **not a copy** of its `*_auto` twin | D and T right, D − T = 7, `OK (differ)` |
| 4 · Write `data/weekly/<D>/` | copies each table to its sub-folder as `.csv.gz`, puts the target lists in `target2/a_constraint/` and `target2/b_free/` (a list without a header is read as `access_method_val`; the `968` prefix is dropped), writes `_manifest.json` with D and T | the target list sizes |
| 4b · Duplicates (NBA rule) | one row per key in every table; the kept row is the one matching the line in `base_prepaid`, otherwise the first one in the file | how many rows were removed per table |
| 5 · `TEST_10K` | writes a small copy in `data/weekly_sample/<D>/`: 10,000 training lines and 10,000 lines of one target group | the sizes |

**Why the duplicates matter.** The NBA recipe joins each table on the line and keeps the **first** row. Our engine
**adds up** the rows of a line (revenue, minutes, MB). A duplicated row would **double** the customer's revenue and
change their segment and their price band. Step 4b gives the engine exactly what the recipe sees.

If an existing week folder is there, it is moved to `data/backup/` first.

---

## 5. Notebook 03: the settings and Part A

**The settings cell** (the first code cell) is the only cell you edit. The ones that matter:

| Setting | Default | Meaning |
|---|---|---|
| `RUN_DATE` | `"2026-11-17"` | D: **must be the same as in notebook 02** |
| `TEST_10K` | `True` | `True` = the 10k copy (runs under `data/test10k/`); `False` = the real week |
| `SCORE_GROUPS` | `["t2c", "t2f"]` | the lists to score: `["t2c"]` = group a only (~50k), `["t2f"]` = group b only, both = ~100k. The other list is not read |
| `MODEL` / `PARALLEL` | qwen2.5 7B / 8 (vLLM: the server's model / 32) | the LLM and how many requests at once |
| `TIME_BUDGET_MIN` | 45 | GPU minutes allowed for agents 3 + 4 (in `TEST_10K`: `TEST_BUDGET_MIN` = 90) |
| `PROFILE_LEVEL` | `"auto"` | `fine` / `medium` / `coarse` / `minimal` / `off` (see step 5b) |
| `FAST_CHECK_N` | 100 | customers also asked with their exact facts, to measure the quality of the profiles |
| `SHARED_LLM_CACHE` | `True` | keep the LLM answers from week to week |
| `EPSILON` | 0.1 | agent 3: share of exploration |
| `NBO_SHORTLIST` | 8 | agent 4: offers per validity the LLM scores |
| `SCORE_VIEW` / `SCORE_MIN_COVERAGE` | `True` / 0.5 | score on `*_score`; replace a `*_score` table that covers less than 50 % |
| `RELAX_UNUSED_CONTRACT`, `TARGETING_AUTO_OPTIONAL`, `REV_DECREASE_OPTIONAL` | `True`, `True`, `False` | accept columns / tables that no step needs |
| `MAX_FALLBACK_SHARE` / `STOP_ON_FALLBACK` | 0.2 / `False` | warn (or stop) when more than 20 % of the LLM answers fail |

With `TEST_10K = True` the settings cell also points every folder to `data/test10k/`, so a test never touches the
real runs.

**Part A** holds the **whole engine code**, copied into the notebook (40 modules). Nothing is imported from the
project folder. That is why the last cell can prove that every engine module was defined by the notebook itself.

**Part B, the first two cells:**

- **The contract relaxation.** The relaxed columns (`tenure`, `subscription_id`, `(S)UBSCRIPTION_ID`, …) are not read
  by any step, so their absence does not block T0.
- **The helpers.** The paths, `show()` and `tree()` to display tables and folders, and the LLM client (next section).

---

## 6. How the LLM is used (all agents)

The same rules hold for all agents:

1. **Local only.** The client refuses any address that is not this machine.
2. **The answer is JSON with a fixed shape.** Each call sends a schema, for example "a list of exactly 7 integers
   from 0 to 100". The server is forced to follow it. An answer with the wrong length is rejected.
3. **Temperature 0 and a fixed seed.** The same prompt gives the same answer.
4. **Cache.** Every answer is saved, keyed by (model, prompt), in `agents/llm_cache.jsonl`, and in notebooks3 also in
   the shared `data/runs/_llm_cache.jsonl`. A re-run, or a profile already seen in an earlier week, costs nothing.
5. **Parallel calls.** `PARALLEL` requests are in flight at the same time. The GPU serves them together.
6. **Fallback.** A failed call returns nothing, and the step uses the rule answer instead (the cell policy for an
   action, the rule score for an offer). The `STATS` table counts calls, cached answers, errors, tokens and seconds
   per agent.
7. **No MSISDN in prompts.** Prompts hold segment names, bands and numbers only.

---

## 7. G1 and G2: the GPU and the speed test

**G1** checks that the server answers, that the model is installed, and (Ollama) that the model is **100 % in GPU
memory**. A model that does not fit runs partly on the CPU and becomes many times slower.

**G2** sends 10 prompts shaped like agent 3's and 10 shaped like agent 4's (no cache), `PARALLEL` at a time, and
measures:

- `s_nba` = seconds per agent 3 call, and `s_nbo` = seconds per agent 4 call (wall time, with the parallel calls);
- tokens per second;
- the per-customer estimate (one call per customer). In notebooks3 the real estimate comes later, in step 5b.

If every test call fails, the notebook stops here: the server is not working.

---

## 8. Training: steps 1 to 4

### Step 1 · The weekly folder

Reads `data/weekly/<D>/` (or the 10k copy). It checks:

- that notebook 02 prepared it **for the same D**;
- how many duplicates 02 removed;
- which `*_score` tables are missing (step 4b will then use their `*_auto` twin).

### Step 2 · Agent 1, the data agent (T0)

- **The rules decide (T0).** Every table is checked against the data contract: columns, types, keys, value ranges,
  date windows, and the completeness of the week. A **blocking** issue stops the run. Otherwise T0 writes the
  **Parquet cache**: every table split into partitions, with all the rows of one line in the same partition, so the
  next steps can work partition by partition.
- **The LLM explains.** It reads the issues and says in plain words whether the data is usable and what to fix
  first. It cannot unblock anything.
- **The targeting week.** It checks that last week's campaign (`date1` of `targeting_dataiku_auto`) is inside the
  window D−8 … D−2. If not, the rewards will be measured on the wrong week, and the notebook warns.

If T0 is BLOCKED: read `validation_report.md`, or run notebook 05, fix the data, and run again.

### Step 3 · Agent 2, segmentation (T2 and T3, `*_auto`, anchor T)

**T2, the features (CPU, all 2M lines of the base).** For every line, from the `*_auto` tables at T:

- **Revenue, voice minutes, data MB and active days** per period (W1 … W4, M1 … M3).
- **The trends**, for example the revenue change M1 against M2 in %.
- **The model scores.** A missing score gets a flag plus the median of the nationality group, computed on the whole
  base. That is why T2 has two passes:
  - pass 1 builds the features, partition by partition;
  - the medians and the **ARPU tertiles** (the value bands) are then computed on the whole base;
  - pass 2 fills in the missing scores.
- **The profile**: age, tenure, nationality group, region.

**T3, the segments (CPU).** The **S-rules** are applied in order. The first one that matches gives the segment:

| Rule | Segment | Condition (in order) | Action of the rule |
|---|---|---|---|
| S-1 | Nursering_Early_Churn_At_Risk | tenure < 45 days and churn ≥ 0.5 | **excluded** |
| S-2 | Nursering_Early_Churn_Not_At_Risk | tenure < 45 days | **excluded** |
| S-3 | Onboarding_Early_Churn_At_Risk | tenure < 90 days and churn ≥ 0.5 | UPSELL |
| S-4 | Onboarding_Early_Churn_Not_At_Risk | tenure < 90 days | UPSELL |
| S-5 | Retention_High_Probability_To_Reconnect | not a revenue-generating subscriber (RGS) and reconnect ≥ 0.6 | RETAIN |
| S-6 | Retention_Med_Probability_To_Reconnect | not RGS and reconnect ≥ 0.3 | RETAIN |
| S-7 | Retention_Low_Probability_To_Reconnect | not RGS | RETAIN |
| S-8 | Retention_RGS_At_Risk | churn ≥ 0.5 or churn 1-day ≥ 0.5 | RETAIN |
| S-9 | Retention_RGS_Fragile | fragility ≥ 0.5 or revenue M1 vs M2 ≤ −30 % | UPSELL |
| S-10 | RGS_Not_At_Risk_DualPlay_Upgrade | voice user and data user | MAINTAIN |
| S-11 | Data_Only_Voice_Appetency | data user and voice appetency ≥ 0.5 | UPSELL VOICE |
| S-12 | Data_Only_No_Voice_Appetency | data user | CROSS SELL VOICE |
| S-13 | Voice_Only_Data_Appetency | data appetency ≥ 0.5 | UPSELL DATA |
| S-14 | Voice_Only_No_Data_Appetency | everything else | CROSS SELL DATA |

The terms in the table:

- **RGS:** revenue M1 > 0 and at least one active day in M1.
- **Voice user:** at least 1 local minute in M1.
- **Data user:** at least 10 MB in M1.
- **Reconnect:** the higher of the 30-day and 15-day reactivation scores.
- **Fragility:** the highest of the data fragility, voice fragility and revenue-decrease scores.
- An unknown tenure never counts as "new". A missing score never triggers a rule.

T3 also gives every line:

- the **value band** (`low` / `mid` / `high`, from the ARPU tertiles) and the **usage type** (voice only, data only,
  dual, none);
- the age band, tenure band and micro-segment;
- **last week's facts**, from `targeting_dataiku_auto` and `subs_btl_auto`:
  - `W0_TARGETED` = was the line targeted in D−8 … D−2;
  - `W0_SEGMENT` / `W0_ACTION` = with which segment and action;
  - `W0_REWARD` = **1 if the line bought a catalogue BTL in D−7 … D−1**;
  - `W0_BTL_REVENUE`;
- **the two actions:** `ACTION_ORIGINAL` (the production dictionary) and `ACTION_AGENT` (the agent dictionary, step
  4), plus its second choice `ACTION_AGENT_RANK2`.

**The LLM explains** each segment in one sentence (one call per segment, about 14 calls) for the business.

### Step 4 · Agent 2b, the agent dictionary (the learning step)

This is where the system **learns from last week**.

- **A cell** = one segment × one nationality group (50 to 100 cells: 50 in the dummy test; the excluded segments are skipped).
- **For each cell the LLM gets the evidence:**
  - the number of lines and the average ARPU;
  - the share at risk, fragile, data users and voice users;
  - the current action;
  - and **last week's reward rate per action**: of the lines targeted with each action, the share that bought.
  For example: `UPSELL DATA: 7% of 1,200; RETAIN: 3% of 800`.
- **It ranks 1 or 2 actions** among the allowed ones, with a reason.
- **The check** (`validate_agent_dictionary`) **refuses**:
  - an unknown action;
  - an action for an excluded segment;
  - **an upsell as the top action of an at-risk segment (H-06)**.
  A refused cell keeps its current action.
- **The new version is saved** as `dict_segm_actions_agent_v<N>.json`, with its parent version, and applied to every
  line. **Next week starts from it**, so the dictionary improves week after week as the rewards come in.

The result for each line is its **cell policy** (`ACTION_AGENT`): the action the dictionary gives its cell. It is the
**fallback** of agent 3, and the action of every customer the LLM does not reach.

---

## 9. Scoring: steps 4b to 7

### Step 4b · The scoring view (`*_score`, anchor D)

Training described the customers **at T**. To decide this week, we describe the **target customers at D** with the
freshest data. Step 4b re-runs **the same T2 and T3 code** with anchor D on `*_score` + `party_scoring`, into
`T2_features_score` and `T3_segments_score`.

- **It keeps the training scale:** the medians and ARPU tertiles **of training**, and the agent dictionary that agent
  2b just wrote. A customer's segment and value band at D are then comparable with what was learnt.
- **The quality gate.** For each `*_score` table, the step counts how many **target lines** it covers, against its
  `*_auto` twin:
  - if the `*_score` table is missing, or covers less than 50 % (`SCORE_MIN_COVERAGE`), the twin is used for that
    period only, and the row says `FALLBACK`;
  - if the `*_score` file is byte-identical to the `*_auto` one, it is flagged (a probable copy).
- **The segment moves.** The step counts how many lines changed segment between T and D (a normal share is a few %).
- **The NBA tables**, with the NBA variable names (`proba_churn`, `rev_w1_zero`, `log_var_rev_w1_w2`, …, the same
  formulas as `calcul_evol`), written to `data/runs/<D>/nba/`:
  - `nba_train_context`: the lines targeted last week (on `*_auto`), with `action`, `reward` and `cost`;
  - `nba_score_context`: the target lines (on `*_score`), with `tg_flag`.

From here on, everything (P1, agents 3 and 4, P6, the final file) reads the scoring view.

### Step 5 · The panel (P1)

P1 reads the two target lists and joins each line with its scoring view: segment, value band, usage, the two
actions, the group key (`t2c` / `t2f`). The result is `panel.parquet`, one row per target line. The cell says whether
the lists are **yours** or dummy ones.

### Step 5b · The customer profiles (FAST_MODE, before any LLM call)

**The idea:** many customers look the same once their facts are rounded. They can share one LLM answer.

- **Each target customer's facts are rounded into bands.**
  - Revenue, data and voice: quantile bands over the target customers, shown as ranges (`revenue M1 2.10-3.40`).
  - The probabilities: fixed bands (`churn 0.2-0.4`).
  - Tenure and active days: fixed ranges.
  - The revenue trend: `down` / `flat` / `up` (M1 against the mean of M2 and M3, ±10 %).
- **The four levels:**

  | Level | The prompt keeps |
  |---|---|
  | `fine` | segment, value band, usage, revenue (5 bands) + trend, data, voice, active days, tenure, churn + appetencies (5 bands), last week with its action |
  | `medium` | the same with 3 bands |
  | `coarse` | 3 bands; without data, voice, active days, tenure |
  | `minimal` | as coarse, without the appetencies; last week = took an offer / no offer / not targeted |
  | `off` | the exact figures: one call per customer (= notebooks2) |

- **A unit of work** = one profile × one group (× the cell policy for T2C). A T2C unit costs one agent 3 call + one
  agent 4 call; a T2F unit costs one agent 4 call.
- **The table**, printed **before any call**, gives per level: the number of profiles, the customers per profile and
  the **estimated GPU minutes** (with the speed of G2). `auto` takes the **finest level that fits
  `TIME_BUDGET_MIN`**.
- **The budget.** Units are taken in priority order until the budget is used:
  - first the priority tier of the segment (retention first, then the fragile and onboarding segments, then the
    others);
  - inside a tier, the biggest profiles first, with the two groups interleaved.
  Customers of the units past the budget get the cell policy.
- **The shared cache** of the earlier weeks is loaded here.

**What the dummy test showed (20k target lines):**

- `fine`: 1.1 customers per profile, almost no saving.
- `minimal`: 6.6 customers per T2C profile; agents 3 + 4 needed 77 % fewer calls.

On real data, read the table: it is the real figure for your week.

### Step 6 · Agent 3, NBA: the action of each T2C customer

- **One LLM call per T2C profile.** The prompt gives:
  - the profile's facts;
  - the cell policy;
  - **the allowed actions**: the 7 actions, minus the upsells for an at-risk segment (H-06).
  The LLM returns a score from 0 to 100 per allowed action, and a short reason.
- **The action is drawn per customer**, from the profile's scores:
  `p(action) = (1 − ε) × score / sum of scores + ε / k`, with ε = 0.1 and k = the number of allowed actions.
  - Each customer has their own random seed, so two customers of one profile can get different actions.
  - **The propensity p of the chosen action is logged** (`PROBA`). This is what makes the uplift measurable next
    week (the same idea as Vowpal Wabbit's exploration).
  - `DECIDED_BY` = `llm_nba` (the LLM's top action), `llm_nba_sampled` (another action, drawn) or
    `fallback_cell_policy` (an invalid answer).
- **The quality check.** 100 customers (`FAST_CHECK_N`) are asked again **with their exact figures**. The cell
  prints how often the profile and the exact figures give **the same top action**, and warns under 70 %. Then choose
  a finer level.
- **The guard.** If more than 20 % of the answers failed, it warns (or stops, with `STOP_ON_FALLBACK`).
- **The outputs:** `agents/agent3_nba.csv` (line, action, propensity, who decided, profile), and a table of cell
  policy × agent 3 action.

T2F customers get no action: the free universe does not use it.

### Step 7 · Agent 4, NBO, and the offer rules (P6)

**First the rules (CPU).** For each customer, the universe rules give the **allowed offers**, per validity (30-day
monthly, 7-day weekly):

- **CC-v1, for T2C:** the offers that fit the customer's action (its family), from the catalogue, near the
  customer's price;
- **CF-v1, for T2F:** the free universe; the action is not used.

Each allowed offer gets a **rule score** (the customer's own past purchase first, then the buyer share of the offer
in the base). The best `NBO_SHORTLIST` = 8 per validity make the **short list**.

**Then the LLM (GPU), one call per (profile, action):**

- **The grouping.** The customers of one profile with the same action (T2F: "any") are grouped. The LLM scores the 8
  offers per validity that are **most often** on their short lists, and returns a score of 0–100 per offer.
- **Each customer of the group gets those scores** for the offers they are allowed. Their other allowed offers rank
  after them (rule score × 0.01).
- **Why this grouping.** Each customer's own short list depends on their own rule scores, so almost no two customers
  share one. Grouping by short list saved almost nothing in the test (1.2 customers per call).
- **The prompt shows no price anchor.** The rules have already kept only the offers in the customer's price band,
  and the profile's revenue band stands for the price level.
- The cell prints the customers per call, and the share of each customer's own short-listed offers that got an LLM
  score (the **coverage**).

**Then the passes choose (CPU), per customer.** For T2C and T2F (the challenger rules):

1. **The expected value** EV = score × price.
2. **The band.** The best EV per type (data / voice / combo), with the price between **1.0× and 1.5×** the
   customer's challenger anchor (their usual spend). For an upsell, at least +0.5 OMR above it.
3. **The widened band.** A customer still missing a type: the band is widened once, upwards only, to 2.0×.
4. **The fallback.** A customer with nothing yet gets the production passes (code 3 for T2C, code 4 for T2F), and
   it is logged.
5. **EXT-6**, with 6 offers: a third offer per validity.
6. **The consistency loop (T2C).** A customer with no offer for their action is sent back once with their second
   action.

The results are in `score/P6_offers_target2/`. `group_report.csv` gives, per group, how many customers were
contacted, how many have offers, how many offers were scored by the LLM, and who decided. `SCORE_SOURCE` =
`llm:<model>` marks the LLM scores.

---

## 10. The final file: steps 8 to 11

### Step 8 · Agent 7, code 5: the Target 2 file

The production code 5 (F-01 … F-19) builds the **38-column file**:

- the 6 offer slots and their reasons;
- favourites and forced bundles in the empty slots;
- the conflicts, and the scripts in English and Arabic.

`details_target2.parquet` keeps every column per line (segment, both actions, the action sent, who decided, the
slots). `input_flytxt_target2.csv.gz` is written **only if the audit passes**.

### Step 9 · Agent 9, the auditor

- **The rules block:**
  - the group audit: A-12 universe, A-12b catalogue, A-17 rules version, A-21 no control line, A-30 / A-31;
  - the file checks A-20 … A-24 (for example A-24: the segment column holds one of the 7 actions).
  Any FAIL and the file is not written.
- **The LLM explains** the result for the business. If it disagrees with the rules, **the rules win**, and the cell
  says so.

### Step 10 · Agent 8, strategy

The LLM compares two pairs:

- what group **a** (T2C) got with what group **b** (T2F) got;
- the customers the **agents** decided with those who got the **cell policy** (past the budget).

The comparison covers the lines, the lines in the file, the offers per line and the mix of actions. It then gives 3
things to watch next week. **This is a proposal only.** The real test is the reward at **D+7** (who bought), which
next week's run reads as `W0_REWARD`.

### Step 11 · The final file

`data/output/<D>/final/input_flytxt.csv.gz` holds Target 2 (groups a and b), only if it passed. **The engine only
saves files; it sends nothing.** You check the file and hand it over yourself.

### Step 12 · The 360 view (for your checks)

Two files next to the final file, in `data/output/<D>/target2/`:

- **`target2_scored_360`** (Parquet + `.csv.gz`): **the scored MSISDNs only**, with all the data. It also has
  `MSISDN_FLYTXT` and two checks against the flytxt file (`CHK_NOT_IN_FLYTXT`, `CHK_SLOTS_DIFFER_FROM_FLYTXT`; both
  must be 0).
- **`target2_360`**: every target line, scored or not (`DROP_RULE` says why a line is not in the file).

One row per line:

- the segment at T and at D, and both actions;
- agent 3's scores, reason and **real logged propensity**, and the FAST_MODE profile;
- the 6 offers, with the rule that picked each (`SLOT_i_SELECTED_BY`), its price, the anchor of the price band, and
  price ÷ anchor;
- last week, and every feature.

**The flags:**

- `CHK_OFFERS_BELOW_ANCHOR`, split into `CHK_FAVOURITE_BELOW_ANCHOR` and `CHK_FALLBACK_PASS_BELOW_ANCHOR`. Any other
  offer below the anchor is printed as a WARNING, and is expected to be 0.
- `CHK_UPSELL_AT_RISK_SEGMENT`: must be 0.
- `CHK_UPSELL_CHURN_GE_060`: for information.

The cell prints the summary per group. The fallback passes (production code 3 / 4) price with the fixed 90 / 10
data / voice split, so a voice or data offer can sit under the customer's own split. The flag shows how often this
happens.

**How next week learns from this week:**

- The file you send becomes next week's `targeting_dataiku_auto`.
- The purchases become `subs_btl_auto`.
- Next week, T3 turns them into `W0_REWARD`, and agent 2b updates the dictionary from them. That closes the loop.

---

## 11. The reports: time table, FAST_MODE figures, G3

- **Time per agent:** calls, cached answers, errors, tokens, `seconds` (summed over the parallel calls) and `wall`
  (the time you waited), per agent. Saved in `agents/agent_times.json`.
- **FAST_MODE in figures:**
  - the level used;
  - the customers per answer for agent 3 and for agent 4;
  - the share of the per-customer calls saved;
  - the answers that came from the cache;
  - the GPU minutes, against the per-customer estimate.
  Saved in `agents/fast_mode.json`.
- **G3, the verdict:** reads the GPU with `nvidia-smi`, the share of the model in GPU memory, and who decided (fresh
  answers, cached ones, failures). It prints:
  - **`VERDICT: LLM + GPU run`**: everything as expected;
  - **"model NOT fully on the GPU"**: slow; choose a smaller model or close other GPU programs;
  - **"NOT an LLM run"**: most decisions fell back to the rules; see the warnings of agents 3 and 4.
  Saved in `agents/llm_gpu_report.json`, with the FAST_MODE figures.
- **The proof cell:** every engine module was defined by the notebook, Vowpal Wabbit was never loaded, and no API
  package was imported.

---

## 12. Time

> **These are estimates, not measurements.** Nothing has been timed yet on the real 2M base or on your GPU. Section
> 12.3 says how to get the real figures in one run.

### 12.1 What takes the time

| Part | Runs on | What drives the time | Estimate, full week (2M base, 50k + 50k targets) |
|---|---|---|---|
| Notebook 02 (read, rename, dedup, write ~60M rows) | CPU | the size of the exports | ~15–35 min |
| Part A, G1, G2 | CPU / GPU | fixed | ~1–2 min |
| Step 2: T0 check + cache | CPU | the size of the exports | ~10–20 min |
| Step 3: T2 / T3 on the 2M lines | CPU | the base size | ~15–40 min |
| Step 4: agent 2b (50–100 cells) + agent 2 (~14 segments) | GPU | ~65–115 calls | ~1–3 min |
| Step 4b: the scoring view | CPU | the base size | ~10–30 min |
| Steps 5, 5b: panel and profiles | CPU | the 100k targets | ~1–3 min |
| **Steps 6 + 7: agents 3 + 4** | **GPU** | **number of calls × seconds per call** | see 12.2; capped by `TIME_BUDGET_MIN` |
| Step 7: the P6 rules and passes | CPU | the 100k targets × the catalogue | ~3–10 min |
| Steps 8–11: code 5, audit, strategy, final file | CPU | the 100k targets | ~3–8 min |
| **CPU total** | | | **about 1–3 h** |

The CPU range comes from the dummy test (a 30k-line week: notebook 02 took 16 s, notebook 03 under 2 min), scaled to
2M lines with denser real tables. It is a rough figure.

### 12.2 The GPU part: agents 3 and 4

For 100k customers, one call per customer means about **150,000 calls**: 50k agent 3 + 50k agent 4 for T2C, and 50k
agent 4 for T2F. With the `minimal` profiles, about **25,000–35,000 calls** (extrapolated from the dummy test).

| Seconds per call (G2) | `off` (~150k calls) | `minimal` (~25–35k calls) |
|---|---|---|
| 0.1 s (vLLM range) | ~4 h | ~45–60 min |
| 0.25 s | ~10 h | ~2–2.5 h |
| 0.5 s (Ollama range) | ~21 h | ~3.5–5 h |

**The budget caps this part.** `TIME_BUDGET_MIN` (45 min by default) is the GPU time agents 3 + 4 may use. The run
never takes longer, but the customers past the budget get the cell policy instead of the LLM. So the choice is
between **time** and **LLM coverage**.

**Rough total:** about **2–4 h with vLLM + `minimal`**, and **5–8 h with Ollama + `minimal`**, if every customer is
sent to the LLM.

The following weeks are faster: the shared cache answers the profiles already seen.

### 12.3 How to get the real figures (one run)

1. **Run notebook 02** with `TEST_10K = False`.
2. **Run notebook 03 (or 04)** with `TEST_10K = False` and **`MAX_LLM_LINES = 1000`**. All the CPU steps then run on
   the full 2M base, and the GPU part stays small.
3. **Read three figures:**
   - **G3, `notebook N min`:** the real CPU time (plus a few GPU minutes).
   - **G2:** the real seconds per call on your GPU.
   - **The step 5b table:** for your real 100k targets, the number of profiles and the GPU minutes of each level,
     before any call.
4. **Add them up.** CPU time + the step 5b minutes of the level you choose = the real duration of a full week.

---

## 13. What you check, step by step

| Where | You want to see | If not |
|---|---|---|
| 02, step 2 | no missing required table | export it, or set the `…_OPTIONAL` setting it names |
| 02, step 3 | D and T right, D − T = 7, `OK (differ)` | check the exports' dates; a copy means a wrong `*_score` export |
| 03, G1 | model 100 % in GPU memory | a smaller model, or close other GPU programs |
| 03, G2 | no failed test call | check the server, lower `PARALLEL` |
| 03, step 1 | the folder prepared for the same D | the same `RUN_DATE` in 02 and 03 |
| 03, step 2 | `T0 DONE`; targeting inside its window | `validation_report.md` or notebook 05 |
| 03, step 4 | some cells answered, few refused | refused = H-06 / unknown action: normal if few |
| 03, step 4b | no `FALLBACK` (or only accepted ones); a few % of segment moves | ask Dataiku for complete `*_score` exports |
| 03, step 5b | the level table; enough customers per profile | raise `TIME_BUDGET_MIN` or choose the level yourself |
| 03, step 6 | quality check ≥ 70 % same top action; fallback < 20 % | a finer level; check the server |
| 03, step 7 | customers per answer; high coverage; fallback < 20 % | as above |
| 03, step 9 / 11 | audit PASS, `final file READY` | read the failing check |
| 03, G3 | `VERDICT: LLM + GPU run` | follow the message |

---

## 14. Where everything is written

| What | Full week | With `TEST_10K` |
|---|---|---|
| prepared week | `data/weekly/<D>/` | `data/weekly_sample/<D>/` |
| T0 report and cache | `data/runs/<D>/train/T0_load/`, `data/cache/<D>/` | under `data/test10k/` |
| training features / segments | `data/runs/<D>/train/T2_features/`, `T3_segments/` | same, under `data/test10k/` |
| agent dictionary | `dict_segm_actions_agent_v<N>.json` (in the runs folder) | |
| scoring view | `data/runs/<D>/train/T2_features_score/`, `T3_segments_score/` | |
| NBA tables | `data/runs/<D>/nba/nba_train_context.parquet`, `nba_score_context.parquet` | |
| panel, offers | `data/runs/<D>/score/P1_panel/`, `P6_offers_target2/` | |
| agents | `data/runs/<D>/agents/`: `agent3_nba.csv`, `llm_cache.jsonl`, `agent_times.json`, `fast_mode.json`, `llm_gpu_report.json` | |
| shared LLM cache | `data/runs/_llm_cache.jsonl` | `data/test10k/runs/_llm_cache.jsonl` |
| final file | `data/output/<D>/final/input_flytxt.csv.gz`; `data/output/<D>/target2/` (details, audit) | `data/test10k/output/<D>/…` |

**Everything under `data/` holds MSISDNs. So do the executed notebooks.** Keep them on this laptop: no e-mail, no
zip, no shared drive, and do not open them in Excel. Never commit a notebook with its outputs.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **D** | the run date; scoring is anchored on it |
| **T** | t2 = D − 7; training is anchored on it |
| **`*_auto` / `*_score`** | the same tables exported for T (history, training) and for D (now, scoring) |
| **T0 / T2 / T3** | data check + cache / features / segments |
| **P1 / P6** | the panel of target lines / the offers |
| **S-rules** | the 14 ordered rules giving each line its segment |
| **cell** | one segment × one nationality group |
| **agent dictionary** | the action(s) of each cell, chosen by agent 2b from last week's rewards; versioned |
| **cell policy** | the action the dictionary gives a customer's cell (`ACTION_AGENT`); the fallback of agent 3 |
| **H-06** | no upsell as the top action of an at-risk segment |
| **W0_REWARD** | 1 if a line targeted last week bought a catalogue BTL in D−7 … D−1 |
| **ε (EPSILON)** | the share of exploration in agent 3's draw |
| **propensity** | the probability of the chosen action, logged to measure the uplift |
| **profile** | a customer's facts rounded into bands; customers with the same profile share one LLM answer |
| **short list** | the best 8 allowed offers per validity, by rule score |
| **CC-v1 / CF-v1** | the challenger offer rules of T2C / T2F |
| **challenger anchor** | the customer's usual spend; the price band is 1.0×–1.5× of it (widened to 2.0×) |
| **EV** | expected value = offer score × price |
| **EXT-6** | the third offer per validity (6 offers in total) |
| **code 5** | the production rules that build the 38-column final file |
| **fallback** | the rule answer used when the LLM fails or is past the budget |
| **TEST_10K** | the 10k test copy, run apart from the real week |
