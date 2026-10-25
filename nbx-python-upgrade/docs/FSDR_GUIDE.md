# FSDR agentic NBA / NBO: how the app works, and every business rule

This guide explains, step by step, what the FSDR part of the app does each week and how to run it, from the web UI
or the command line. It then lists **every business rule (BR-01 … BR-86)** with where and how the app applies it,
the rules the app adds, the defaults that were chosen, and the known differences from production.

- Plan and phase status: [fsdr_plan.md](fsdr_plan.md)
- Reference material: [fsdr_reference/](fsdr_reference/) (business-rule catalogue, `nba_lib.py`, training / scoring scripts)
- JSON data contract: [fsdr_json_schemas/](fsdr_json_schemas/) (one JSON Schema per table)

---

## Contents

1. [What the app does](#1-what-the-app-does) (and [which data it uses](#which-data-fsdr-uses-and-which-it-ignores))
2. [Start and stop](#2-start-and-stop)
3. [The weekly cycle, step by step](#3-the-weekly-cycle-step-by-step)
4. [The three agents of the llm arm](#4-the-three-agents-of-the-llm-arm)
5. [Business rules BR-01 … BR-86](#5-business-rules-br-01--br-86)
6. [Rules the app adds](#6-rules-the-app-adds)
7. [Defaults chosen where the rules were ambiguous](#7-defaults-chosen-where-the-rules-were-ambiguous)
8. [Known differences from production](#8-known-differences-from-production)
9. [Output files](#9-output-files)
10. [Switching to real data](#10-switching-to-real-data)

The weekly cycle now also has [step 8: campaign log and feedback](#step-8-campaign-log-and-feedback-at-d7) and
[step 9: simulation and the operations agents](#step-9-closed-loop-simulation-and-the-operations-agents).

---

## 1. What the app does

Every week, for each prepaid customer in the test sample, the app chooses **one action** (Next-Best-Action, NBA)
and then **the offers for that action** (Next-Best-Offer, NBO). It does this in four arms, so their results can
be compared:

| Arm | Who decides the action | Who decides the offers |
|---|---|---|
| `cg` (control) | nobody: no contact | none |
| `free` | the bandit scores it, but it is **not used** (BR-60) | the ranker, L2 DATA + COMBO offers only (BR-67) |
| `cstr` (constraint) | the bandit | the ranker, restricted to the action's offer families (BR-66) |
| `llm` (new) | the **action agent** (the LLM), using the bandit as evidence | the **offer agent** (the LLM), then the **competitor agent** checks them against the market |

```
 data/fsdr/<D>/*.json  ──► validate ──► train bandit (T-anchored tables) ──► score sample (D-anchored tables)
   (45 tables, real          │              │                                     │
    export or dummy)         │              ▼                                     ▼
                             │       model registry (champion)        action + probabilities per customer
                             │                                                    │
                             │                       ┌────────────────────────────┼──────────────────────────────┐
                             │                       ▼                            ▼                              ▼
                             │                 cstr arm: rules             free arm: rules            llm arm: 3 agents
                             │                 (families + price           (L2 DATA/COMBO +           action → offers →
                             │                  bands + conflicts)          price bands)               competitor check
                             ▼                       └────────────► outputs/recos_<monthly|weekly>_<arm>.json ◄─┘
                   _validation.md                                           + llm_decisions.jsonl (with propensities)
```

**Two dates drive everything** (BR notation):

- **D**, the run date. It is the folder name, e.g. `data/fsdr/2026-10-23/`.
- **T**, the campaign anchor: the most recent Thursday on or before D, `T = D - (((D - 2026-03-12) MOD 7) + 7) MOD 7`.
  Training uses tables anchored on T (`*_auto`); scoring uses tables anchored on D (`*_score`).

**Layers:** web UI (Gradio, port 7860) → HTTP → FastAPI (port 8000, routes `/fsdr/*`) → FSDR modules in
`nbx/fsdr/`. The UI holds no logic: every button calls an API endpoint, and every endpoint has a command-line
equivalent in `run.py`.

---

### Which data FSDR uses (and which it ignores)

| Data | Used by FSDR? |
|---|---|
| `data/fsdr/<D>/*.json`: the 45 tables of the run date (real exports or generated dummy data) | **Yes, the only customer / offer / usage input** |
| `data/fsdr/<D>/outputs/`: results of the previous steps of the same run | Yes (scoring reads training's model; offers read scoring) |
| `data/fsdr/state/models/`: model registry | Yes (champion bandit) |
| `data/competitors/competitor_offers.json`: public competitor price snapshot | Yes, only for the competitor agent (market prices, no customer data) |
| `data/customers.json`, `offers.json`, `purchases.json`, `segments.json`, `revenue_history.json`, `nba_scores_*.json`, `nba_activity_usage.json`, `nba_targeting.json`: the old NBX demo data | **No.** Only the 12 original NBX tabs use it |

Changing the run folder (or dropping real exports into a new one) changes everything FSDR computes. Nothing from
the old NBX data leaks into the FSDR results.

---

## 2. Start and stop

From the `nbx-python-upgrade` folder, with the virtual environment active:

```powershell
python run.py check        # packages, data files, search index
python run.py setup        # once: builds the search indexes (competitor search uses them)
python run.py start        # API + web UI   -> http://localhost:7860  (API docs: http://localhost:8000/docs)
```

Press **Ctrl+C** in that terminal to stop both servers. `python run.py test` runs the test suite (218 tests).

The FSDR tabs come after the 12 NBX tabs: **FSDR Data**, **FSDR Training & Scoring**, **FSDR Offers**,
**FSDR Agentic Recommender**.

The complete reference (every step, rule, setting, command and API route) is **[FSDR_HANDBOOK.md](FSDR_HANDBOOK.md)**.

**LLM:** on this laptop the agents run on **vLLM**. Start the server with `python run.py vllm-serve` (WSL2, GTX 1650,
model `qwen2.5:3b`), then run `python run.py start`. `.env` sets `LLM_BACKEND=vllm`. Decisions show `mode: agentic`
and take 25–150 s each. Without a server, `python run.py --backend offline start` uses the agents' rule-based path,
which is exactly the constraint-arm logic, and every decision shows `mode: fallback`. Setup, timings and limits are
in [FSDR_AGENTIC_RECOMMENDER.md §1](FSDR_AGENTIC_RECOMMENDER.md#1-is-an-llm-used-offline-ollama-vllm) and
[GPU_SETUP.md §14](GPU_SETUP.md#14-small-gpu-on-windows-wsl2-gtx-1650-4-gb).

---

## 3. The weekly cycle, step by step

Run the steps in this order for each run date. Each step writes its results to `data/fsdr/<D>/outputs/`.

### Step 1: get the data for date D

**Web UI.** In **FSDR Data**, open *Generate dummy data* and choose the run date, the number of lines and the
scenarios. Click **Generate**, then **Refresh runs**.
**Command line.** `python run.py fsdr-generate --date 2026-10-23 --n 2000 --scenario baseline`

What happens:

- The app writes **45 JSON files**, one per table, plus `_manifest.json`, into `data/fsdr/<D>/`. The files have
  exactly the field names, case and types of the production tables.
- Behind the dummy data is a hidden "world" of customers. Each one has a churn risk, data and voice appetite, price
  sensitivity, preferred validity and spend level. Their daily usage is simulated over 97 days and then summed with
  the exact BR §4 windows, so every weekly and monthly value is consistent with the others.
- The 11 model scores are noisy readings of that hidden truth. Each score file covers 70–90 % of the base, and the
  early-churn files cover only young lines.
- Last week's campaign is simulated too: targeting, offers sent, and which customers took them.
- Scenarios change the hidden world: `baseline`, `churn_wave`, `data_growth`, `price_sensitive`, `new_offers`,
  `campaign_fatigue`, `data_quality_issues`. They can be combined.
- `--weeks N` writes N consecutive weekly folders (D, D+7, …) from **one** world. The days the weeks share agree
  exactly (W2 of D+7 = W1 of D), and a scenario drift grows week after week (e.g. `churn_wave` in North Batinah):
  `python run.py fsdr-generate --date 2026-10-01 --weeks 4 --scenario churn_wave`.

With real data you skip generation: put your JSON exports in `data/fsdr/<D>/` (section 10).

### Step 2: validate

**Web UI.** In **FSDR Data**, pick the run and click **Validate**.
**Command line.** `python run.py fsdr-validate --data-dir data/fsdr/2026-10-23`

The validation checks:

- every table is present;
- every field is present, with the right raw type (string / int / double / date / timestamp);
- no null keys and no duplicate keys;
- keys match across tables (for example, every targeted line exists in `access_method_id_auto`);
- model-score coverage of the base;
- window consistency (activities M1 = W1+W2+W3+W4; `Sub_Amt_m1` ≥ its weeks);
- dates inside their BR windows;
- the manifest's anchor date.

Each issue is **blocking**, **warning** or **info**. Training and scoring **refuse blocked data**. The report is
written to `_validation.md` and `_validation.json` in the run folder. *Preview a table* shows any file's first rows.

### Step 3: train the bandit

**Web UI.** In **FSDR Training & Scoring**, pick the run, keep the defaults, and click **Train**.
**Command line.** `python run.py fsdr-train --data-dir data/fsdr/2026-10-23`

This is a port of `01_nba_train.py`:

1. **Eligible base:** MSISDN range, excluded plans, one line per MSISDN (BR-02, BR-08, BR-10).
2. **Features** for the whole base, from the T-anchored tables. There are 48 numeric features: the 9 model scores
   plus 39 trend features for revenue, voice and data, and one categorical feature, the nationality group.
3. **Labels:** last week's targeting gives each line its segment, and the segment gives the action (BR-39). The
   reward is "took a catalogue BTL in D-7…D-1"; three reward definitions are computed. Each row also gets a
   propensity.
4. **Training:** the Vowpal Wabbit contextual bandit (`--cb_explore_adf -q UA --epsilon 0.2`). In `logged` mode
   (default) it learns from every targeted line, takers and non-takers. `simulation` mode is the original
   production approach: takers only.
5. **Evaluation on a 20 % holdout:**
   - take rate of the current (logged) policy;
   - IPS and SNIPS estimates of the new model's take rate;
   - share of takers whose action the model reproduces;
   - action mix.
6. **Registration:** a new version in `data/fsdr/state/models/`. It becomes champion **only if** its IPS is at least
   the champion's (code-enforced).
7. **Scoring of the whole base** with the greedy action (`nba_scored_base`).

The tab shows the labels per action, the holdout metrics and the action mix. Use **Refresh registry**,
**Promote** and **Roll back** to manage models.

### Step 4: score the test sample

**Web UI.** In **FSDR Training & Scoring**, click **Score**.
**Command line.** `python run.py fsdr-score --data-dir data/fsdr/2026-10-23`

This is a port of `02_nba_score.py`:

1. **Load the champion model.** It is refused if it was trained with other features or actions, or if it is older
   than 14 days (*Allow a stale model* overrides the age check).
2. **Population:** the test sample `sample_cg_tg_new_7oct2026`, arms `free`, `cstr` and `llm`. The control group
   `cg` is not scored. Account → party comes from `consumer_aact_account`.
3. **Fresh features** from the D-anchored tables, built by the same function as training.
4. **Scoring:** policy `sample` (production: the action is drawn from the model's probabilities) or `greedy`. Every
   line gets an action, its probability, the greedy action and the probability of all 7 actions.
5. **Outputs:** `results_scoring_nba` and one file per arm (`TG_free_scored_nba`, `TG_constraint_scored_nba`,
   `TG_llm_scored_nba`).
6. **Drift report:** PSI per feature against the training data (alert above 0.25), and the action-mix shift.

### Step 5: offers for the `cstr` and `free` arms

**Web UI.** In **FSDR Offers**, pick the arm (`cstr` or `free`) and the generosity mapping, then click
**Recommend**. You can download the result as CSV.
**Command line.** `python run.py fsdr-recommend --arm cstr` (or `--arm free`)

This is a port of production Codes 3 and 4. For each customer and each validity (30 days, then 7 days):

1. **Offer families** (section 5.8):
   - cstr: the action's two services, e.g. RETAIN → COMBO + DATA, with generosity L2 (all-L2 mapping, active today)
     or the "normal" mapping (BR-66);
   - free: L2 COMBO + L2 DATA (BR-67).
2. **Ranker:** a BPR model trained on the 270-day BTL history of those families scores every offer for the
   customer; the top 100 are kept (BR-83 … BR-86).
3. **ARPU anchor:** the mean + standard deviation of the customer's revenue periods. COMBO offers compare with the
   total anchor, DATA with 90 % of it, VOICE with 10 % of it (BR-68 … BR-77).
4. **Three passes** (BR-78 … BR-81):
   - pass 1: one offer of each family priced between the anchor and anchor + k;
   - pass 2: the missing family, priced above the anchor with no upper limit;
   - pass 3: for customers with nothing yet, both families above the anchor.
5. **Limits:** at most 2 offers per validity and 4 in total (BR-82), and never two conflicting offers together
   (`conflicting_offers_h`, rule added by the app). A customer whose anchor is above every allowed price gets no
   offer.

Each offer row records the action, the offer, price, anchor, anchor type and **the pass that selected it**.

### Step 6: the llm arm (the three agents)

**Web UI, one customer.** In **FSDR Agentic Recommender**, pick the run and a customer (the list is sorted by churn
score), then click **Run the agents**. You see each agent's tool calls live, then the decision, the overrides, the
critic's findings, the competitor comparison and the SMS.
**Web UI, the whole arm.** In **FSDR Offers**, choose arm `llm`.
**Command line.** `python run.py fsdr-decide --msisdn 70074996 --trace`, or `python run.py fsdr-recommend --arm llm`
for the whole arm.

See section 4 for how the agents work. In batch, every decision is appended to `outputs/llm_decisions.jsonl` as soon
as it is made. An interrupted run resumes where it stopped, and a decision is reused when the customer's features,
the catalogue, the model and the LLM are unchanged.

### Step 6b: batch flow (action → top 4 offers → top 2 roaming → international)

**Web UI.** In **FSDR Agentic Recommender**, use the **Batch** section. Paste a list of MSISDNs (or leave it empty
to take the first customers of the arm), choose the arm (`llm`, `cstr` or `free`), the batch size and the maximum
number of customers, then click **Recommend for the batch**. The table grows batch by batch, and a CSV is ready at the
end.
**Command line.** `python run.py fsdr-batch --arm llm --limit 100 --batch-size 25`, which writes
`outputs/batch_recos_<arm>.csv`.
**API.** `POST /fsdr/batch/{run}` with `{"msisdns": [...], "arm": "llm"}`, one call per batch.

For every customer, in one pass:

1. **Action.** llm arm: the action agent (offline: the bandit's sampled action, as cstr). cstr: the scored action.
   free: shown but not used (BR-60).
2. **Top 4 offers** for that action. llm arm: offer agent + competitor agent. cstr / free: the 3-pass rules. All
   BR-66 … BR-82 rules apply.
3. **Top 2 roaming packs:** offers whose service is `ROAMING`. They are ranked by a BPR model on the customer's
   270-day roaming purchases. A customer without roaming history gets popularity blended with the price fit to their
   own ARPU anchor, so each customer gets packs that match their spend.
4. **International packs, only if available:** offered only when the customer made **at least 3 international calls in
   M1** (`Call_Count_M1 − Local_Call_Count_M1`; local calls exclude international ones). The top 2 international packs
   are ranked the same way. Otherwise the row says why ("0 international calls in M1").

A NO_CONTACT customer gets no offer and no add-on.

### Step 7: read the results

- **FSDR Offers** shows the summary and the offers of each arm; download them as CSV.
- The files in `data/fsdr/<D>/outputs/` are listed in [section 9](#9-output-files).

### Step 8: campaign log and feedback at D+7

Every arm is logged when its offers are made, in `data/fsdr/state/campaign/nba_campaign_log.jsonl`. The file is
append-only: re-running a step skips the rows already there. Each row holds:
- run date, arm and line;
- action and offers (code, price, validity);
- policy source, the **propensity** of the action sent, the bandit PMF;
- model version, LLM model, justification id, and the line's ARPU anchors.

The control group (cg) is logged as not contacted.

One week later, run the feedback on the **next** run folder: `python run.py fsdr-feedback --data-dir data/fsdr/<D+7>`.
It joins the campaign logged in D-8..D-2 with the fulfilments of D-7..D-1 and writes `nba_rewards.jsonl`, with the
3 reward definitions (any BTL, same service, exact offer) and the revenue. It also prints take rate, revenue and
uplift vs control per arm. `python run.py fsdr-report --week <D>` shows the same report for a campaign week.

From then on, training uses the log by default (`FSDR_LABEL_SOURCE=auto`):
- the action actually sent and its **logged** propensity, not a weekly share;
- the four last-campaign features (`PREV_ACTION`, `PREV_OFFER_TAKEN`, `DAYS_SINCE_LAST_BTL`,
  `NB_CONTACTS_LAST_4W`), which are part of the model's feature list and of the compatibility check.

### Step 9: closed-loop simulation and the operations agents

`python run.py fsdr-simulate --weeks 3 --n 2000` (or the **FSDR Simulation** tab) runs the whole cycle on the hidden
world, week after week:
- generate the week's data, with last week's logged campaign as the targeting and its accept / ignore draws as this
  week's fulfilments;
- feedback, then train (campaign-log labels), score, and recommend for free / cstr / llm;
- report take rate, revenue and uplift vs control per arm.

It runs in `data/fsdr/simulation/`, apart from your runs; 3 weeks on 2,000 lines take about 45 s.

`python run.py fsdr-agent --agent data|training|scoring|monitor` runs one operations agent on a run (same ReAct
pattern as the recommender, rule-based without an LLM). The LLM proposes; the code decides what is allowed:
- **data:** never "proceed" on blocked data, and never generate over real data;
- **training:** hyper-parameters only inside the allowed ranges; promotion only if the challenger's holdout IPS is
  at least the champion's;
- **scoring:** a stale or incompatible model is refused;
- **monitor:** report and alerts only.

---

## 4. The three agents of the llm arm

In-depth document, with a real decision explained field by field and the LLM backends (offline / Ollama / vLLM):
**[FSDR_AGENTIC_RECOMMENDER.md](FSDR_AGENTIC_RECOMMENDER.md)**.

All three run on the same tool loop as the NBX council (`nbx/agents/agentic/react.py`):

- the model replies with one JSON object per step, either a tool call or a final answer;
- it cannot finish before calling its required tools;
- repeated identical calls are refused;
- each agent has a step limit;
- answers that break a rule are sent back with the reason.

Code: `nbx/fsdr/agents/recommender_agent.py` and `nbx/fsdr/agents/tools.py`.

```
Action agent ──action──► Offer agent ──offers──► Competitor agent ──► critic (code) ──► messenger (scripts)
```

### 4.1 Action agent

- **Tools (evidence, not orders):**
  - `customer_360`: scores, revenue W1…M3, trends, usage type, eligibility;
  - `bandit_pmf`: the probability of each of the 7 actions, the top action and the sampled action;
  - `segment_policy`: the segment and its BR-39 action;
  - `campaign_history`: last week's offers, what the customer took, top BTLs over 270 days;
  - `contact_policy`: whether we can contact, and the actions blocked by churn risk.
- **Required before answering:** `customer_360`, `bandit_pmf`, `contact_policy`.
- **Freedom:** any of the 7 actions, or **NO_CONTACT**.
- **Rule:** if the chosen action is not the bandit's top action, `override_reason` is required. An answer without it
  is sent back.

### 4.2 Offer agent

- **Tools:**
  - `price_band_candidates`: the rule-based (cstr) offers for the action;
  - `ranker_top_offers`;
  - `arpu_anchors`: anchors and soft price bands;
  - `search_catalog`: any offer in the catalogue;
  - `check_offers`: hard rules plus soft-band notes;
  - `expected_value`: a heuristic acceptance estimate.
- **Required:** `price_band_candidates`, `check_offers`.
- **Freedom:** 1 to 4 offers from the whole catalogue.
- **Rules:** a list that breaks a hard rule is sent back. A list that differs from the rule-based offers needs
  `override_reason`.

### 4.3 Competitor agent

- **Tools:**
  - `competitor_compare`: for each offer, the closest competitor plan at the same validity, its monthly price,
    data, the monthly price difference and a verdict;
  - `competitor_market`: competitor plans at the customer's spend;
  - `search_catalog`.
- **Output:** a verdict (`competitive`, `mixed` or `uncompetitive`), a summary, and optional **swaps** (replace offer X
  by offer Y). A swap is applied only if the new list passes every hard rule.
- **Labels:** competitor names are always shown as **COMPETITOR_A** and **COMPETITOR_B**, also inside plan names.

### 4.4 Hard rules, soft rules, critic

| Type | Rules | What happens if broken |
|---|---|---|
| **Hard (code-enforced)** | customer eligible (BR-02, BR-10, BR-36); at least 2 days since the last contact; **no UPSELL / UPSELL DATA / UPSELL VOICE when PROBA_CHURN ≥ 0.60**; offer in the catalogue; price > 0; at most 2 per validity, 4 in total; no conflicting pair | action → NO_CONTACT, or upsell → RETAIN; offers that break a rule are dropped (in order), and if none is left the rule-based offers are used. Every repair is a critic finding, and the critic verdict becomes `revised` |
| **Soft (guidance)** | ARPU price band; action → generosity mapping; bandit's top action | may be overridden with a written reason; every override is logged and counted |

### 4.5 Learning-compatible logging (propensity)

With probability **ε_llm = 0.1**, the final action is a bandit sample instead of the LLM's choice. Each decision
logs its source and propensity:

| `policy_source` | When | Propensity |
|---|---|---|
| `bandit` | the 10 % exploration draw | ε × pmf[action] |
| `llm` | the LLM's choice | 1 − ε + ε × pmf[action] |
| `fallback` | no LLM (offline) | the bandit's probability of the sampled action |

Every decision also logs the bandit's pmf, the model version, the catalogue version, the overrides and the critic
findings, so later training can learn from the LLM's choices without bias (off-policy).

### 4.6 Without an LLM

Each agent takes its rule-based path with the same tools:

- the action is the bandit's sampled action;
- the offers are the 3-pass offers;
- the competitor verdict is computed, with no swap.

That is **exactly the constraint arm**. A test checks it customer by customer, and on the 2,000-line run all 196
llm-arm decisions matched.

---

## 5. Business rules BR-01 … BR-86

Source: [fsdr_reference/FSDR_NBA_business_rules.md](fsdr_reference/FSDR_NBA_business_rules.md). Every constant is in
`nbx/fsdr/config.py`, with its BR number in a comment.

**How to read the "In the app" column**

- **Applied:** the app enforces the rule on whatever data it reads.
- **Upstream:** the rule belongs to the SQL extraction that produces the JSON files. The app does not run SQL; it
  trusts the export, and the dummy-data generator reproduces the rule.
- **Not used:** production loads the table or value but does not use it, and neither does the app.

### 5.1 Customer eligibility (`base_prepaid`)

| # | Rule | In the app |
|---|---|---|
| BR-01 | Snapshot `ACTIVITY_DATE = D-2` | Upstream. `calendar.snapshot_date()`; the generator builds the base as of D-2 |
| BR-02 | `ACCESS_METHOD_VAL > '69999999'` | **Applied.** `features.eligible_base()` drops lines outside the range; the hard rule `customer_eligibility()` blocks contact; validation warns |
| BR-03 … BR-07 | Consumer, mobile, prepaid, active lifecycle, line not deactivated | Upstream (the export only contains such lines) |
| BR-08 | One line per MSISDN, latest activation | **Applied.** `eligible_base()` keeps the latest `access_method_subs_start_dt`; validation flags duplicates |
| BR-09 | Plan name = product commercial name if `OFFERING_ID >= 300000` | Upstream (`CURRENT_PLAN` arrives already resolved) |
| BR-10 | 13 excluded internal / employee / business plans | **Applied.** `config.EXCLUDED_PLANS`; removed in `eligible_base()`, blocked by `customer_eligibility()`; excluded from the generated test sample |
| BR-11 | Active consumer accounts (not `TERM`) | Upstream; `consumer_aact_account` gives account → party for scoring |
| BR-12 | Usage / fulfilment MSISDN without the 3-character country prefix | **Upstream.** The production SQL already applies `SUBSTR(MSISDN, 4)`, and `access_method_id_auto` is built as `ACCESS_METHOD_VAL as MSISDN`, so every MSISDN in the JSON files is the 8-digit line number. The app strips nothing. `labels.access_method_map()` keeps the latest activation per line |

### 5.2 Location and demographics

| # | Rule | In the app |
|---|---|---|
| BR-13 | Home location = previous calendar month of D-2 (January → December) | Upstream; `calendar.location_month()`; generated `location_auto` uses it |
| BR-14 | Several locations → `MIN(REGION/WILAYAT/AREA)` | Not used by the model; the generator includes ~3 % multi-location lines |
| BR-15 | Nationality group: Oman, India, Bangladesh, Pakistan, else `others` | **Applied.** Feature `NATIONALITY_GROUP` in `features.build_features()` |

### 5.3 Revenue scope

| # | Rule | In the app |
|---|---|---|
| BR-16 | 14 product types count as subscription revenue | Upstream (the amounts arrive already filtered) |
| BR-17 / BR-18 | Data and voice subscription classes | Upstream (`Data_Sub_Amt_*`, `Voice_Sub_Amt_*` in `subscriptions_tg`) |
| BR-19 | NBA tables use `SUBSCRIPTION_TOT_AMT`; offer pricing uses `SUBSCRIPTION_AMT` | Upstream; the generator writes TOT_AMT (incl. tax) to `subs_amt_*` and AMT to `subscriptions_tg` |
| BR-20 | Missing amount = 0 | **Applied** (BR-48, BR-76) |

### 5.4 Time windows (BR §4)

All windows are defined once in `nbx/fsdr/calendar.py`. The generator aggregates with them, and the validation and
features rely on them.

| Table group | Anchor | Windows |
|---|---|---|
| `subs_amt_auto` / `subs_amt_score` | T / D, **includes** the anchor day | W1 a-6…a, W2 a-13…a-7, W3 a-20…a-14, W4 a-27…a-21, M1 a-29…a, M2 a-59…a-30, M3 a-89…a-60 |
| `activities_*_auto` / `activities_*_score` | T / D, **excludes** the anchor day | W1 a-7…a-1, W2 a-14…a-8, W3 a-21…a-15, W4 a-28…a-22, M1 a-28…a-1, M2 a-56…a-29, M3 a-85…a-57 |
| `subscriptions_tg` | D (2-day lag) | W1 D-8…D-2 … W4 D-29…D-23; M1 D-31…D-2, M2 D-61…D-32, M3 D-91…D-62 |
| `usage_data_voice_payg_tg` voice & data | D | p1 D-31…D-2, p2 D-61…D-32, p3 D-91…D-62 |
| `usage_data_voice_payg_tg` PAYG | D | M1 D-30…D-1 … M3 D-90…D-61; W1 D-7…D-1 … W4 D-28…D-22 |
| `paygo_activity_auto` | T | same as PAYG above (not used) |

| # | Rule | In the app |
|---|---|---|
| BR-21 | Targeting for learning: `date1` in D-8…D-2 | **Applied** in validation (warning outside); labels use the targeting file as exported |
| BR-22 | Arbitration targeting `start_date` in D-7…D | Not used (loaded and validated) |
| BR-23 | Reward window: fulfilment D-7…D-1 | **Applied.** `labels.load_fulfilments()` |
| BR-24 | Interaction history D-270…D-1 | **Applied.** `sub_btl_3months` is the ranker's history |
| BR-25 | Successful fulfilments only | **Applied.** Rows without a success timestamp are dropped |

### 5.5 Usage metric definitions

| # | Rule | In the app |
|---|---|---|
| BR-26 … BR-32 | Voice / data records, minutes, MB, on-net / off-net / local voice | Upstream; the generator respects them (local excludes international, on-net + off-net = local) |
| BR-33 | PAYG = total charged; `_tg` tables count positive charges only | Upstream; the generator writes all charges to `activities_*` and positive charges only to `*_tg` / `paygo` |
| BR-34 | `_tg` data volume: records with volume > 0 | Upstream; generator |
| BR-35 | Active days = distinct activity dates | Upstream; generator |

### 5.6 NBA model: segments, actions, rewards (BR-36 … BR-45)

| # | Rule | In the app |
|---|---|---|
| BR-36 | Nursering segments (tenure < 45 days) excluded | **Applied.** Dropped from training labels; hard rule in the llm arm |
| BR-37 | Targeted rows without segment removed | **Applied** (`labels.build_labels`, counted in the report) |
| BR-38 | 7 actions, fixed order: CROSS SELL DATA, UPSELL DATA, RETAIN, MAINTAIN, UPSELL, CROSS SELL VOICE, UPSELL VOICE | **Applied.** `config.ACTIONS`, same order as `nba_lib`; never shuffled |
| BR-39 | Segment → action map (12 segments) | **Applied**, case-insensitive (`RGS_…` and `Rgs_…` are the same). The two ambiguous entries use the defaults of section 7 |
| BR-40 | Taker = any catalogue BTL in the reward window | **Applied** as reward `any_btl` (default); `same_service` and `exact_offer` are also computed |
| BR-41 | REWARD 1 / 0, COST = −REWARD | **Applied** |
| BR-42 | One row per customer | **Applied:** the latest targeting row per line is kept (as `nba_lib`) |
| BR-43 | Training population = takers only | Available as `--mode simulation`. Default `logged` mode trains on takers **and** non-takers (the reference library's recommendation) |
| BR-44 | Propensity = action prior + uniform(0, 0.02) | Replaced: the logged propensity when available, else this week's action share, else the prior; clipped to 0.01…1 |
| BR-45 | Action priors (RETAIN 0.509 … CROSS SELL DATA 0.008) | **Applied** as the last-resort propensity (`config.ACTION_PRIOR`) |

### 5.7 Features, learning and scoring (BR-46 … BR-58)

| # | Rule | In the app |
|---|---|---|
| BR-46 | Model scores used: churn (max of 3 files), reactivation 30/15, appetency data/voice, churn 1-day, revenue decrease, fragility data/voice | **Applied** through `config.MODEL_SCORE_REGISTRY`: one shared schema for the 11 files; the 3 churn files are merged by max; adding a model is one line of config |
| BR-47 | Missing score → random value in [0, 0.02] | **Applied, per row** (the original drew one value for everyone) |
| BR-48 | Missing usage / subscription → 0 | **Applied** |
| BR-49 | NBA revenue = SUB_AMT + PG_DATA_AMOUNT | **Applied** (`REV_W1…REV_M3`) |
| BR-50 | % variation, 0 if the previous period is 0 | **Applied**, from the **older** period to the **recent** one (W1 = most recent); clipped to −100…500 |
| BR-51 | Zero flag when the previous period is 0 | **Applied** (`*_ZERO`) |
| BR-52 | Log variation `log(1+cur) − log(1+prev)` | **Applied** (`LOG_*`) |
| BR-53 | Pairs: revenue W1-W2, W2-W3, W3-W4, M1-M2, M2-M3; voice and data W1-W2, W2-W3, W3-W4, M1-M2 | **Applied:** 39 trend features |
| BR-54 | VW `--cb_explore_adf -q UA --epsilon 0.2` | **Applied**; epsilon configurable within 0.05…0.3 |
| BR-55 | 20,000 simulated draws | **Applied** in `simulation` mode |
| BR-56 | Simulated cost −1 when the sampled action = the logged action | **Applied** in `simulation` mode |
| BR-57 | Scoring samples the action from the probabilities | **Applied** (`--policy sample`, default); `greedy` also available |
| BR-58 | One action per customer | **Applied** (duplicates removed after scoring) |

### 5.8 Experiment design and offer families (BR-59 … BR-67)

| # | Rule | In the app |
|---|---|---|
| BR-59 | Arms from the sample's `group`: `free`, `cstr` | **Applied**, plus the new arm `llm`; `cg` is the control and is not scored |
| BR-60 | Free arm ignores the NBA action | **Applied** |
| BR-61 | Constraint arm: the action restricts services and generosity | **Applied** |
| BR-62 | Sample stratified by nationality group × 3-month revenue × voice group × data group × plan | Upstream; the generator builds `strata` / `counts` and balances the 4 arms inside each stratum |
| BR-63 | Only catalogue products | **Applied** (ranker items, rewards, hard rule) |
| BR-64 | Monthly (30 d) and weekly (7 d) offers computed separately | **Applied** (one ranker and one selection per validity) |
| BR-65 | Generosity L0 < L1 < L2 | **Applied** (catalogue; generated prices rise with generosity) |
| BR-66 | Action → families (service A, service B) and generosity | **Applied** (`offer_rules.families()`), see table below; switch `FSDR_GENEROSITY_MAPPING = all_L2` (default, active in production) or `normal` |
| BR-67 | Free arm: L2 only, DATA and COMBO only | **Applied** |

BR-66 families:

| Action | Service A | Service B | Generosity, normal | Generosity, all_L2 (default) |
|---|---|---|---|---|
| MAINTAIN | COMBO | DATA | L0 / L0 | L2 / L2 |
| UPSELL | COMBO | DATA | L1 / L1 | L2 / L2 |
| RETAIN | COMBO | DATA | L2 / L2 | L2 / L2 |
| UPSELL VOICE | DATA | COMBO | L0 / L1 | L2 / L2 |
| CROSS SELL VOICE | DATA | COMBO | L0 / L2 | L2 / L2 |
| CROSS SELL DATA | VOICE | COMBO | L0 / L2 | L2 / L2 |
| UPSELL DATA | VOICE | COMBO | L0 / L2 | L2 / L2 |

CROSS SELL VOICE has its own entry. The original code repeated "CROSS SELL DATA" there, so CROSS SELL VOICE was never
filtered.

### 5.9 ARPU / spending anchor (BR-68 … BR-77)

| # | Rule | In the app |
|---|---|---|
| BR-68 | Revenue per period = `Total_Sub_Amt` + `PG_Total_Amount` (subscriptions + **total** PAYG) | **Applied** (`offer_rules.arpu_anchors()`) |
| BR-69 / BR-70 | Monthly periods M1…M3; weekly W1…W4 | **Applied** |
| BR-71 | **Anchor = mean(periods) + std(periods)** | **Applied** (sample standard deviation, as pandas) |
| BR-72 | Max revenue computed, not used | Same (`MAX_REV_*` computed, not used) |
| BR-73 | Old method `min(mean, max)` or `max + 1` | Not implemented (documented only) |
| BR-74 | Data anchor = 0.9 × anchor | **Applied** |
| BR-75 | Voice anchor = 0.1 × anchor | **Applied** |
| BR-76 | Missing revenue = 0 | **Applied** |
| BR-77 | COMBO vs total anchor, DATA vs data anchor, VOICE vs voice anchor; weekly offers vs weekly anchors | **Applied** (`offer_rules.anchor_column()`) |

### 5.10 Price limits and selection (BR-78 … BR-82)

| # | Rule | In the app |
|---|---|---|
| BR-78 | Pass 1: `anchor < price < anchor + k`, k = 3 for UPSELL, 2 for MAINTAIN / RETAIN, 3 for the voice / data actions, 3 for the free arm | **Applied** (`config.PRICE_BAND_K`) |
| BR-79 | Pass 2: the missing type only, `price > anchor`, no cap | **Applied** |
| BR-80 | Pass 3: customers with no offer, both types, `price > anchor`, no cap | **Applied** |
| BR-81 | Sort: pass 1 by score ↓ then price ↓; passes 2-3 by score ↓ then price ↑ | **Applied** |
| BR-82 | At most 2 offers per validity (one per type), 4 in total; no offer if the anchor is above every price | **Applied**, and also a hard rule in the llm arm |

### 5.11 Ranker (BR-83 … BR-86)

| # | Rule | In the app |
|---|---|---|
| BR-83 | Interaction = 1 per (customer, BTL) subscribed in the last 270 days | **Applied** (`sub_flag` summed, catalogue BTLs only) |
| BR-84 | LightFM, WARP, 400 components, 30 epochs, k 15 | `LightFMRanker` keeps these parameters but `lightfm` does not install on Python 3.12, so the default is **BPR** (`implicit`), the same family of model |
| BR-85 | Top 100 offers per customer | **Applied** (`config.RANKER_TOP_N`). As in production, only offers **bought in the filtered history** (action family, validity) can be proposed; `FSDR_RANKER_ITEMS=catalogue` allows every family offer (brand-new offers have no history) |
| BR-86 | Customers without history scored from existing users | Replaced by **popularity** for cold-start customers. The original reused other users' rows by loop index, which was a bug |

### 5.12 Reference tables

| Table | In the app |
|---|---|
| `conflicting_offers_h` | **Now applied:** two conflicting offers are never sent together (all arms) |
| `BTL_script_english_arabic_ok` | **Now used:** the messenger takes each offer's English and Arabic script |
| `paygo_activity_auto`, `targeting_auto`, `location_auto`, `DEVICEBEHAVIORSEGMENTATION.dv_segment` | Loaded and validated, not used by the model (as production) |

### 5.13 Inconsistencies listed in the catalogue, and how the app handles them

| # | Inconsistency | Handling |
|---|---|---|
| 1 | NBA revenue uses PAYG data, offer pricing uses PAYG total | Both kept exactly (BR-49 for features, BR-68 for anchors) |
| 2 | 30-day vs 28/29-day months, 2-day lag in pricing | Kept; every window is in `calendar.py` |
| 3 | Reward counts any BTL; targeting and reward windows overlap | `any_btl` stays the default, and any catalogue BTL fulfilled in D-7..D-1 counts, as in production (BR-40). `FSDR_REWARD_AFTER_TARGETING=true` also requires the fulfilment on or after the customer's targeting date |
| 4 | Two ambiguous segment mappings | Defaults of section 7, configurable |
| 5 | Generosity forced to L2 for every action | Default `all_L2`; switch to `normal` in the Offers tab or with `FSDR_GENEROSITY_MAPPING` |
| 6 | Prior counts do not add up | Priors used only as last-resort propensity |
| 7 | Anchor = mean + std pushes above usual spend | Kept (BR-71); the llm arm may go below the band with a written reason (soft rule) |
| 8 | Fixed 90 / 10 data-voice split | Kept |

---

## 6. Rules the app adds

| Rule | Where | Value |
|---|---|---|
| Conflicting offers never sent together | all arms (`offer_rules.select_offers`, `check_offers`) | pairs of `conflicting_offers_h` |
| Contact cap | llm arm, hard | at least 2 days since the last contact, and at most 3 contacts in 28 days (campaign log) |
| No upsell-family action at high churn risk | llm arm, hard | PROBA_CHURN ≥ 0.60 blocks UPSELL, UPSELL DATA, UPSELL VOICE (→ RETAIN) |
| Override needs a written reason | llm arm | action ≠ bandit top, offers ≠ rule-based offers, competitor swap |
| Exploration share in the llm arm | llm arm | ε_llm = 0.1, logged propensity |
| Data is refused when validation blocks | train, score | any blocking issue |
| Model promotion only if not worse | registry | challenger IPS ≥ champion IPS |
| Model age limit for scoring | score | 14 days |
| Drift alert | score | PSI > 0.25 on a feature |
| Fulfilment counts from the targeting date (option) | labels | off by default (production rule); `FSDR_REWARD_AFTER_TARGETING=true` |
| Roaming and international add-ons | batch flow | services `ROAMING` / `INTERNATIONAL` (`FSDR_ROAMING_SERVICES`, `FSDR_INTERNATIONAL_SERVICES`); top 2 each |
| International only for customers who call abroad | batch flow | ≥ 3 international calls in M1 (`INTL_MIN_CALLS_M1`) |
| Add-ons beside the main offers | batch flow | BR-82's limit of 4 applies to the main offers; an add-on never duplicates or conflicts with a main offer or another add-on |

All values are in `nbx/fsdr/config.py`, and several can be set in `.env` (see `.env.example`).

---

## 7. Defaults chosen where the rules were ambiguous

| Question | Default |
|---|---|
| `Retention_High_Probability_To_Reconnect` → RETAIN or UPSELL? | **RETAIN** (the value production's dictionary ends up with) |
| `RGS_Not_At_Risk_Data_Only_Voice_Appetency` → UPSELL VOICE or CROSS SELL VOICE? | **UPSELL VOICE** |
| Reward definition for training | `any_btl` (production); `same_service` and `exact_offer` computed too |
| Generosity mapping | `all_L2` (active in production) |
| Contact cap | minimum 2-day gap and at most 3 contacts in 28 days (from the campaign log). "1 per 7 days" would block almost everyone, because last week's targeting is always D-8…D-2 |
| Churn threshold for blocking upsell | 0.60 |
| Ranker | BPR (`implicit`); LightFM only if installable |
| Target sample table | `sample_cg_tg_new_7oct2026` (`FSDR_TARGET_SAMPLE_TABLE`) |
| Currency and nationality "Oman" | kept: they are data values, not operator names |

---

## 8. Known differences from production

- **No SQL.** The app reads already-aggregated JSON, one file per table. Rules applied by the SQL (marked
  *Upstream*) are trusted, not re-checked, except where the validation can see them.
- **Training mode.** The default is `logged` (all targeted lines, real or weekly-share propensity), not production's
  takers-only simulation. The original mode is still available.
- **Bug fixes** from the reference list:
  - numeric VW features are written `name:value`;
  - action tokens have no spaces;
  - the action order is fixed;
  - the model is saved and loaded, not finished before scoring;
  - missing scores get one random value per row;
  - trends go from the older to the recent period;
  - one key type everywhere;
  - the ranker predicts by the customer's own id;
  - CROSS SELL VOICE has its own filter.
- **Ranker:** BPR instead of LightFM; popularity instead of the batch reuse for cold-start customers.
- **Conflicts and scripts** are used (production loaded them but did not apply them).
- **Competitor names** are shown as labels COMPETITOR_A / COMPETITOR_B; the operator is `OPERATOR_NAME` (FSDR).
- **Reward:** any catalogue BTL fulfilled in D-7..D-1 counts, as in production (BR-40); the option
  `FSDR_REWARD_AFTER_TARGETING=true` also requires it on or after the targeting date.
- **Contact cap** (llm arm, hard rule): at most 3 contacts logged in 28 days, and 2 days since the last one.
- **Price-escalation guard** (option): `FSDR_ANCHOR_MAX_GROWTH=0.1` caps the ARPU anchor at +10 % over the anchor logged
  for the line the week before.
- **Not built:** the rename of the 12 NBX tabs to FSDR.

---

## 9. Output files

Everything is written to `data/fsdr/<D>/outputs/`, as JSON arrays of records with lower-case column names.

| File | Written by | Content |
|---|---|---|
| `nba_training_report.json` | train | validation, base, features, labels, training and holdout metrics, promotion decision |
| `nba_training_set.json` | train | labelled rows: action, the 3 rewards, propensity, holdout flag, features |
| `nba_scored_base.json` | train | greedy action for every eligible line |
| `results_scoring_nba.json` | score | action, probability, greedy action, the 7 probabilities, arm, model version |
| `TG_free_scored_nba.json`, `TG_constraint_scored_nba.json`, `TG_llm_scored_nba.json` | score | the same, per arm |
| `nba_scoring_report.json` | score | action mix per arm, drift (PSI, action-mix shift) |
| `recos_monthly_<arm>.json`, `recos_weekly_<arm>.json` | recommend | one row per offer: customer, action, offer, service, generosity, validity, price, anchor, anchor type, pass (+ policy source and propensity for `llm`) |
| `nba_recommend_report_<arm>.json` | recommend | customers with / without offers, offers per pass, mean price (llm: modes, policy sources, override rate, competitor verdicts) |
| `llm_decisions.jsonl` | llm arm | one full decision per line: action, offers, propensity, pmf, overrides, critic, competitor, message, cache key |
| `batch_recos_<arm>.csv` | `fsdr-batch` | one row per customer: action, offer 1-4, roaming 1-2, international (or the reason there is none), competitor verdict, critic, mode, propensity |

Elsewhere: `data/fsdr/<D>/_validation.md` and `_validation.json` (validation report), and
`data/fsdr/state/models/` (model registry, `registry.json`).

---

## 10. Switching to real data

1. Create `data/fsdr/<YYYY-MM-DD>/` (the run date D) and put one JSON file per table in it. Each file is named after
   its table, e.g. `base_prepaid.json` or `DAIM_CHURN_GLOBAL_SCORING_TD.prediction.json`. It can be a JSON array or
   JSON Lines (`.jsonl`); the format is detected automatically. The 45 tables and their fields are in
   [fsdr_json_schemas/](fsdr_json_schemas/).
2. Validate the folder: `python run.py fsdr-validate --data-dir data/fsdr/<date>`. Fix anything **blocking**, and
   check the warnings.
3. Run the same steps 3 to 6 (train, score, recommend), with the web UI or with `--data-dir data/fsdr/<date>`.
   No code changes.

`data/fsdr/` is git-ignored, so real customer data is never committed.
