# FSDR agentic NBA / NBO upgrade of NBX – how to run it in Claude Code (VS Code)

## Part A – Steps for you (do these in order)

**Step 1 – Back up the project**
```powershell
cd nbx-python
git init            # skip if it is already a git repo
git add -A; git commit -m "NBX before FSDR upgrade"
git checkout -b fsdr-agentic-nba
```
Each phase gets its own commit, so you can always go back.

**Step 2 – Add the reference files**
Unzip `fsdr_reference.zip` into `nbx-python/docs/fsdr_reference/`. You should see:

| File | What it is |
|---|---|
| `nba_lib.py`, `01_nba_train.py`, `02_nba_score.py` | Cleaned FSDR NBA library, training and scoring pipelines (Dataiku version, tested) |
| `README_NBA_pipelines.md` | Step map of the two pipelines + bug list |
| `FSDR_NBA_business_rules.md` | 86 business rules (filters, windows, ARPU anchor, price bands, offer families) |
| `fsdr_original_sql_and_schemas.txt` | Table schemas + original SQL. For column meaning only: no SQL is run |
| `fsdr_original_python_codes.txt` | The original 4 Python codes (NBA, split, LightFM constraint, LightFM free) |

If you already have some real JSON files, put a few in `nbx-python/docs/fsdr_reference/json_samples/` (10–20 records per file is enough, with dummy MSISDNs). Claude Code will check its schemas against them.

**Step 3 – Check the project still runs**
```powershell
.venv\Scripts\Activate.ps1
python run.py test        # the 104 tests must pass before you start
```

**Step 4 – Open Claude Code**
In VS Code: open the `nbx-python` folder, then open the Claude Code panel (Claude icon in the sidebar, or run `claude` in the VS Code terminal). Switch to **plan mode** (Shift+Tab until "plan mode" shows), so Claude reads and plans before it changes anything.

**Step 5 – Paste the prompt**
Copy everything in **Part B** below and paste it. Claude Code will read the repo and the reference files, then reply with a plan.

**Step 6 – Review the plan**
Check these points, then answer "approved" or ask for changes:
- the existing NBX folders, UI layout, theme and agent pattern are kept (Part B, section 0, rule 3)
- one generic model-score schema for the 11 models (section 2)
- the list of open questions and the default chosen for each

**Step 7 – Let it run phase by phase**
Leave plan mode (Shift+Tab) so it can edit files. After each phase it stops and reports. At each stop:
1. Run `python run.py test` yourself.
2. Commit: `git add -A; git commit -m "FSDR phase N"`.
3. Paste the next message from the table below.

| After phase | Message to send |
|---|---|
| 1 (plan) | `Approved. Start phase 2. Stop at the end of the phase and report.` |
| 2 (data) | `Run fsdr-generate --n 2000 --scenario baseline and fsdr-validate, show me the validation report and one example record per JSON file, then start phase 3.` |
| 3 (ML pipelines) | `Run fsdr-train then fsdr-score on the dummy data, show the training report and the action distribution per arm, then start phase 4.` |
| 4 (offers) | `Show the offers for 5 customers per arm with the ARPU anchor and which price pass selected each offer, then start phase 5.` |
| 5 (agents) | `Run the llm-arm recommender in offline mode and with the fake LLM for 3 customers, show the full agent trace and any override, then start phase 6.` |
| 6 (feedback + simulation) | `Run fsdr-simulate --weeks 4 --scenario churn_wave,data_growth and show take rate per arm per week, then start phase 7.` |
| 7 (API/UI/docs) | `Run the full test suite, start the app, and give me the list of new endpoints, new UI tabs, and changed files.` |

**Step 8 – Final check on your side**
```powershell
python run.py test
python run.py fsdr-generate --n 5000 --scenario baseline
python run.py fsdr-simulate --weeks 4
python run.py start       # UI http://localhost:7860, API http://localhost:8000/docs
```
In the UI, check that the new FSDR tabs look like the existing tabs (same theme, layout and components).

**Step 9 – Switch to your real data**
Put your real JSON files in `data/fsdr/<YYYY-MM-DD>/` (one file per table, exact table names), then run `python run.py fsdr-validate --data-dir data/fsdr/<date>` and fix anything it reports. Train and score with `--data-dir` pointing to that folder.

**Tips**
- If Claude Code tries to do several phases at once, say: `Stop. Finish only the current phase and report.`
- If the conversation gets long, start a new Claude Code session and say: `Read docs/fsdr_plan.md and continue from phase N.` (Part B asks it to keep that plan file up to date.)

---

## Part B – Prompt to paste into Claude Code

---

You are working in my existing project **NBX** (`nbx-python/`). It is a recommendation engine for a prepaid mobile operator. It has an agent council (`nbx/agents/agentic/council.py`: NBA strategist → NBO specialist → critic → messenger → explanation), a Vowpal Wabbit bandit (`nbx/core/nba_bandit.py`), collaborative filtering (`nbx/core/collab_filter.py`, implicit BPR), a cohort knowledge graph, 5 vector stores, a competitor module, FastAPI (`nbx/api/main.py`), a Gradio UI with 12 tabs, an offline / ollama / vllm LLM layer, and 104 passing tests.

I want to upgrade NBX so it reproduces my **real production NBA / NBO project** end to end, with the same structure of input data, and make every stage **agentic**. Most important: the **LLM must be free to act as the recommender agent**. Today's ML models should give it evidence, not dictate the answer. The code must still enforce hard business rules and log every decision so the system can learn from it.

## 0. Ground rules

1. **Start in plan mode.** First read the whole repo (`docs/`, `council.py`, `nba_bandit.py`, `collab_filter.py`, `rules_engine.py`, `api/main.py`, `ui/`, the tests) and every file in `docs/fsdr_reference/`, including `json_samples/` if it exists. Then give me a written plan: modules, file list, data flow, API and UI changes, test plan, phases, and open questions with your proposed default. Save it as `docs/fsdr_plan.md` and keep it updated at the end of each phase (what is done, what is next). Wait for my approval before writing code.
2. **Operator name = "FSDR".** Do not use any real operator name anywhere: code, UI texts, SMS templates, docs, test fixtures. Replace only the **name, logo text and wording** with "FSDR". In the competitor module, keep the logic but make competitor display names configurable labels (`COMPETITOR_A`, `COMPETITOR_B` by default).
3. **Keep the existing NBX design exactly as it is.** This is an extension, not a rewrite.
   - **Architecture:** same layers (UI → HTTP → FastAPI → agents → core models / vector stores → llm_client). The UI keeps no logic: every new button calls an API endpoint.
   - **Package layout:** new code goes in `nbx/fsdr/` and reuses the existing modules (`core/settings`, `core/schemas`, `core/nba_bandit.py`, `core/collab_filter.py`, `core/knowledge_graph.py`, `agents/rules_engine.py`, `agents/reranker.py`, `llms/llm_client`). Extend them; don't create parallel copies. Absolute imports, same code style, same type hints and docstring style.
   - **Agents:** use the same tool-loop pattern, JSON reply format, step caps, evidence checks, repeated-call refusal, rule-based fallback and `mode` values (`fallback` / `agentic` / `partial_fallback`) as `council.py`.
   - **UI:** new tabs reuse the existing Gradio theme (colours, fonts, spacing), the same components and the same layout patterns as the current tabs. Keep the 12 existing tabs and their order; add the FSDR tabs after them.
   - **API:** same style as `api/main.py` (routers, schemas, error format, upload steps that can be retried from the failed step).
   - **Launcher and config:** new commands go in `run.py`; new settings in `.env.example` with defaults; data under `data/`; docs under `docs/`; tests under `tests/` with the same `conftest.py` offline forcing.
   - **Existing features keep working unchanged:** offline / ollama / vllm modes, the 5 vector stores, competitor intelligence, uploads, batch scoring, LLM benchmark.
4. **Don't break what exists.**
   - The 104 tests must still pass.
   - Offline mode stays the default: every new agent needs a rule-based fallback that uses the same tools.
   - Still pip-only, no external servers.
   - Windows, macOS and Linux all supported.
5. **Port, don't re-invent.** `docs/fsdr_reference/nba_lib.py` + `01_nba_train.py` + `02_nba_score.py` are the reference implementation for features, VW format, training and scoring. Port them into NBX, removing the Dataiku dependency behind a data-source interface. Do **not** reintroduce the bugs listed in `README_NBA_pipelines.md`:
   - `vw.finish()` before scoring
   - shuffled action list
   - `name=value` for numeric features in VW
   - spaces in VW action tokens
   - inverted trend direction (W1 is the most recent week)
   - one random value for all missing scores
   - LightFM predicting with the loop index instead of the user id
   - the CROSS SELL VOICE filter typo
   - string vs integer join keys
6. **Business rules come from `FSDR_NBA_business_rules.md`** (BR-01 … BR-86). Put every constant (windows, priors, k margins, ARPU formula, generosity mapping, excluded plans/segments…) in one config module, with the BR number in a comment. Never hard-code them inside logic.
7. **Work one phase at a time** (section 10). At the end of each phase: run the full test suite, update `docs/fsdr_plan.md`, give me a short summary (files changed, tests, what to check) and **stop until I say continue**.

## 1. Target architecture (new package `nbx/fsdr/`, plugged into the existing NBX layers)

```
nbx/fsdr/
├── config.py            all business rules / constants (BR-xx), anchor-date logic, paths
├── schemas.py           exact raw schemas of every input JSON file (section 2), one shared MODEL_SCORE_SCHEMA, output tables
├── calendar.py          T = last Thursday ≤ D (reference Thursday 2026-03-12), all windows of BR section 4
├── datasource.py        DataSource interface: JsonSource (reads a folder of JSON files, one per table) | SyntheticSource (generates the same JSON files with dummy data)
├── synth/
│   ├── world.py         hidden "ground truth" customer world + weekly behaviour evolution (section 3)
│   ├── scenarios.py     use-case presets (section 2.2)
│   └── generator.py     writes every table as a JSON file with the exact field names and types, for a given date D
├── features.py          port of nba_lib feature engineering (identical for train and score)
├── labels.py            segment → action, reward, propensity (BR-36 … BR-45) + campaign-log rewards
├── bandit.py            VW contextual bandit (reuse / refactor core/nba_bandit.py; logged + simulation modes)
├── offer_ranker.py      OfferRanker interface: ImplicitBPRRanker (default, already in repo) | LightFMRanker (optional, only if installable)
├── offer_rules.py       ARPU anchor, data/voice split, action → offer families, 3-pass price bands, conflicts, eligibility
├── registry.py          model registry: versioned folder, metadata, champion / challenger, rollback
├── campaign_log.py      append-only log of every decision sent + rewards
├── pipelines/
│   ├── train.py         deterministic training pipeline (callable as tools)
│   ├── score.py         deterministic scoring pipeline
│   ├── recommend.py     NBO for constraint / free / llm arms
│   ├── feedback.py      reward computation D+7, feedback into training set
│   └── simulate.py      multi-week closed-loop simulation
└── agents/
    ├── data_agent.py        generate / load / validate data
    ├── training_agent.py    runs the training pipeline through tools, decides promote or not
    ├── scoring_agent.py     runs scoring at campaign start, drift checks
    ├── recommender_agent.py the LLM recommender (extends the existing council)
    └── monitor_agent.py     weekly report: take rate per arm, drift, model comparison
```

## 2. Input data: JSON files, plus a dummy-data generator with the same structure

**There is no SQL and no database in this project.** I will provide the input data as **JSON files, one file per table**. The data is already extracted and aggregated for each window, so the project only reads JSON. Do not implement or translate any SQL. The SQL in `fsdr_original_sql_and_schemas.txt` is only there to explain what each column means and which window it covers.

### 2.1 JSON data contract
- **Folder layout:** one folder per run date:
  ```
  data/fsdr/<YYYY-MM-DD>/
  ├── _manifest.json                  run_date (D), anchor_date (T), generator/source, row counts, schema_version
  ├── base_prepaid.json
  ├── subs_amt_auto.json
  ├── DAIM_CHURN_GLOBAL_SCORING_TD.prediction.json
  └── ...                             file name = exact table name from the tables below + ".json"
  ```
- **File format:** a JSON array of records, e.g. `[{"ACCESS_METHOD_VAL": "91234567", "ACCESS_METHOD_ID": 1001, ...}, ...]`. Also accept JSON Lines (`.jsonl`) for big tables. Detect the format automatically.
- **Field names and case** are exactly as in the tables below. Keep production quirks: `UBSCRIPTION_ID`, `activities_auto_m2`, mixed case.
- **Types:** use JSON types matching the raw types below. "string" fields are JSON strings, even when they hold numbers (e.g. `"price": "3.5"`). Dates are `"YYYY-MM-DD"` strings and timestamps ISO 8601. Missing values are `null`.
- `schemas.py` defines the contract. Write a JSON Schema per table to `docs/fsdr_json_schemas/`, so I can check my real exports before loading them.
- The loader validates every file (fields, types, nulls in keys, duplicates, cross-table key consistency) and produces a validation report. Unknown extra fields are reported but do not block the run.

### 2.2 Dummy-data generator (simulates the use case)
The generator writes **exactly the same JSON files** as the real data, with **dummy variables**:
- fake but valid MSISDNs, ids, ages, regions and wilayats
- dummy offer names (e.g. `BTL_DATA_L2_30D_10GB`) and dummy plan names
- dummy segment and script texts

That means the whole project runs on dummy data first and on my real JSON later, with no code change: just point `JsonSource` to another folder.

It must support **use-case scenarios** (`--scenario`, combinable). Each one changes the hidden world of section 3:
| Scenario | What it simulates |
|---|---|
| `baseline` | Normal weekly behaviour |
| `churn_wave` | A segment's churn risk rises over the weeks; RETAIN should win for it |
| `data_growth` | Voice-only customers gradually gain data appetite; UPSELL DATA / CROSS SELL DATA should grow |
| `price_sensitive` | Acceptance drops fast when the price is above the ARPU anchor |
| `new_offers` | New BTL offers appear in the catalogue mid-simulation (cold start for the ranker) |
| `campaign_fatigue` | Acceptance falls with the number of contacts in the last 4 weeks |
| `data_quality_issues` | Duplicate MSISDNs, null keys, stacked churn duplicates, keys as strings; validation must catch them |

The tables below describe each JSON file. Production types are messy, e.g. strings for numbers in `btl_catalog` and in the sample table. Keep them messy, so the loading and cleaning code is exercised. Keys must be consistent across all files.

**Keys and formats**
- `ACCESS_METHOD_VAL` is the MSISDN: an 8-digit string > `'69999999'` (BR-02).
- `ACCESS_METHOD_ID`, `ACCOUNT_ID` and `PARTY_ID` are integers.
- `MSISDN` in `access_method_id_auto.json` keeps a 3-character country prefix (e.g. `"96891234567"`); the loader strips it (BR-12). In `subs_btl_auto.json` and `sub_btl_3months.json`, `access_method_val` already comes without the prefix.

**Base & reference**
| Table | Columns (raw type) |
|---|---|
| `base_prepaid` | UBSCRIPTION_ID:string (sic, keep the typo), ACCESS_METHOD_VAL:string, ACCESS_METHOD_ID:int, ACCOUNT_ID:int, PARTY_ID:int, access_method_subs_start_dt:string, CURRENT_PLAN:string |
| `consumer_aact_account` | ACCOUNT_ID:bigint, PARTY_ID |
| `access_method_id_auto` | ACCESS_METHOD_ID:int, MSISDN:string, ACTIVATION_DATE:string |
| `party_auto`, `party_scoring` | party_id:int, AGE:int, GENDER:string, NATIONALITY:string, nationality_identifier:string |
| `location_auto` | Access_Method_id:int, Calling_Month:int, Calling_Year:int, REGION:string, WILAYAT:string, AREA:string |
| `btl_catalog` | btl_name:string, type:string, generosity:string (L0/L1/L2), gb:string, min:string, price:string, "Price data":string, "Price voice":string, service:string (DATA/VOICE/COMBO), validity:string (7/30) |
| `BTL_script_english_arabic_ok` | "Sl No.":string, destination_offer_name:string, "English Script":string, "Arabic Script":string |
| `conflicting_offers_h` | Offer_A, Gen_Level_A, Price_A, Data_A, Min_A, Validity_A, Offer_Type_A, Offer_B, Gen_Level_B, Price_B, Data_B, Min_B, Validity_B, Offer_Type_B, Price_Difference, Data_Difference, SAME_VALIDITY, SAME_OFFER_TYPE (all string) |
| `DEVICEBEHAVIORSEGMENTATION.dv_segment` | ACCESS_METHOD_ID:int, DV_SEGMENT:string |

**Revenue & usage.** Generate an `_auto` version (anchored on T) and a `_score` version (anchored on D) of each.
| Table | Columns |
|---|---|
| `subs_amt_auto`, `subs_amt_score` | ACCESS_METHOD_VAL:string, ACCESS_METHOD_ID:int, ACCOUNT_ID:double, Sub_Amt_w1…w4:double, Sub_Amt_m1…m3:double |
| `activities_{w1,w2,w3,w4,m1,m3}_auto`, `activities_auto_m2` (note the different name), `activities_{w1…m3}_score` | ACCESS_METHOD_VAL:string, ACCESS_METHOD_ID:bigint, ACCOUNT_ID:int, then, for period suffix P ∈ {W1…W4, M1…M3}: Call_Count_P:int, Data_Session_Count_P:int, Voice_Traffic_min_P:double, Data_Volume_MB_P:double, PG_Total_Amount_P:double, PG_Voice_Amount_P:double, PG_Data_Amount_P:double, Onnet_Call_Count_P:int, Onnet_Voice_min_P:double, Offnet_Call_Count_P:int, Offnet_Voice_min_P:double, Local_Call_Count_P:int, Local_Voice_min_P:double, Active_Days_Count_P:int, Active_Days_Voice_Count_P:int, Active_Days_Data_Count_P:int |
| `paygo_activity_auto` | ACCESS_METHOD_ID:bigint, ACCOUNT_ID:int, ACCESS_METHOD_VAL:string, PG_Total_Amount_m1…m3:double, PG_Total_Amount_w1…w4:double |
| `subscriptions_tg` | ACCESS_METHOD_VAL:string, ACCESS_METHOD_ID:int, ACCOUNT_ID:double, Total_/Data_/Voice_Sub_Amt_w1…w4:double, Sub_Count_M1…M3:int, Total_/Data_/Voice_Sub_Amt_M1…M3:double |
| `usage_data_voice_payg_tg` | ACCESS_METHOD_ID:bigint, ACCOUNT_ID:int, ACCESS_METHOD_VAL:string, local_Voice_min_p1…p3:double, data_volume_mb_p1…p3:double, PG_Total_Amount_m1…m3:double, PG_Total_Amount_w1…w4:double |

**Campaign targeting & outcomes**
| Table | Columns |
|---|---|
| `targeting_dataiku_auto` | date1:string, access_method_id:int, account_id:int, segment:string (the 12 segments of BR-39 + the 2 Nursering ones, with the real mixed case, e.g. `RGS_…` and `Rgs_…`), destination_offer_name_1…6:string |
| `targeting_auto` | DATE1, START_DATE, END_DATE:date, MSISDN:string, CHANNEL_1…6, OPPORTUNITY_NAME_1…6, DESTINATION_OFFER_NAME_1…6, FLAG_SPLIO, OFFER_SCRIPT_EN_1…6, OFFER_SCRIPT_AR_1…6, SEGMENT, FILENAME:string, ETL_CREATE_DT:date, JOB_ID:int, SEGMENT_BTL_TAKER, SEGMENT_BTL_TAKER_UPSELL:string |
| `subs_btl_auto` (fulfillment D-7 … D-1) | access_method_val:string, PRODUCTNAME:string, FULFILMENTSUCCESSDATE_TIME:timestamp, sub_flag:tinyint, nb_sub:int |
| `sub_btl_3months` (fulfillment D-270 … D-1, despite the name) | access_method_val:string, PRODUCTNAME:string, sub_flag:tinyint, nb_sub:int |

**The 11 model-score files: all share ONE identical structure**

All model-score JSON files have exactly the same fields and types:

| Field | Type | Meaning |
|---|---|---|
| subscription_id | string | Subscription id |
| access_method_val | bigint | MSISDN (as a number in these files) |
| access_method_id | bigint | Line id |
| tenure | bigint | Days since activation |
| life_in_nw_range | string | Tenure band label |
| proba_0 | double | Probability of class 0 (= 1 − proba_1) |
| proba_1 | double | **The score used by the NBA** |
| prediction | string | Predicted class ("0"/"1") |

So implement them **generically**:
- **One schema** (`MODEL_SCORE_SCHEMA`) in `schemas.py`, used by every model file.
- **One registry** in `config.py` mapping file name → feature name → how to combine:

  | File (table name) | Feature | Combine |
  |---|---|---|
  | `DAIM_CHURN_GLOBAL_SCORING_TD.prediction` | PROBA_CHURN | max with the 2 early-churn files |
  | `DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction` | PROBA_CHURN | max |
  | `DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored` | PROBA_CHURN | max |
  | `DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored` | PROBA_REACT_30 | – |
  | `PROPENSITY_REACTIVATION_15D_401_SCORING.Scored` | PROBA_REACT_15 | – |
  | `APPETENCY_DATA_204_SCORING_TD.prediction` | PROBA_APP_DATA | – |
  | `APPETENCY_VOICE_204_SCORING_TD.Scored` | PROBA_APP_VOICE | – |
  | `CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring` | PROBA_CHURN_1D | – |
  | `REV_DECREASE_204_SCORING_TD.predictions` | PROBA_REV_DECR | – |
  | `DAIM_FRAGILITY_DATA_SCORING_TD.prediction` | PROBA_FRAG_DATA | – |
  | `DAIM_FRAGILITY_VOICE_SCORING_TD.prediction` | PROBA_FRAG_VOICE | – |

- **One loader** (`load_model_scores(registry)`): reads each file, keeps `proba_1`, renames it to the feature, resolves duplicates per line (max), merges on `access_method_val` + `access_method_id`, and reports the coverage of each model.
- **One generator function** (`generate_model_score(name, hidden_signal, coverage, tenure_filter)`): writes any model file in that schema. The score is driven by the matching hidden signal of the world (section 3) plus noise. Coverage is partial (70–90 % of the base), so the "missing score → random 0–0.02" rule is exercised. The early-churn files only cover young tenure (0–30 and 30–60 days).
- **Adding or removing a model is a config change only:** add a line in the registry and the loader, generator, features, VW format, model-compatibility check, UI and docs pick it up. Add a test that a 12th dummy model added through config flows into the features.

**Test sample** `sample_cg_tg_new_7oct2026`. Every column is a string: access_method_val, access_method_id, account_id, tenure, activation_date, plan_name, payg_revenue, subscription_revenue, total_revenue, total_revenue_3m, voice_traffic, data_volume, nationality, region, ARPU, ARPU_3m, voice_trf_group, data_trf_group, Final, channel, group.
- `group` ∈ {`cg` control, `free`, `cstr`} plus a new arm **`llm`** (section 5).
- Stratify on nationality group × revenue band × voice group × data group × plan (BR-62), and add `strata` and `counts` like `new_tg_nba_16jun26_*`.

**Realism requirements**
- Default 5,000 lines, configurable up to 200k. Fixed seed. Generation of 5k must take under a minute.
- Nationality mix: mostly Oman, India, Bangladesh, Pakistan, plus a long tail. Plans include the 13 excluded employee/internal plans (BR-10), so the filter is exercised.
- Heavy-tailed revenue and usage, many zeros, voice-only, data-only and dual users.
- Catalogue of ~60 BTL offers across DATA / VOICE / COMBO × L0 / L1 / L2 × 7 / 30 days, priced roughly 0.5–15 currency units. Generosity must be monotonic with price within a family.
- Model scores must be **correlated with the hidden truth**: churn score with real churn risk, appetency with real appetite. They should not be pure noise.
- The window values in each file must be consistent with each other and with the run date. For example, `Sub_Amt_m1` ≈ the sum of the weekly amounts in the same 30 days, and the W2 of date D equals the W1 of date D-7. The generator computes them from a daily dummy history using the windows of BR section 4. The pipelines then only read the aggregated JSON, like with the real data.

## 3. Hidden customer world + weekly evolution (needed to test learning)

`synth/world.py` holds a hidden state per customer: churn risk, data appetite, voice appetite, price sensitivity, preferred validity, and spend level. It also holds a **hidden response function** P(accept offer | customer, action, offer, price vs ARPU anchor). Some examples of what it should encode:
- RETAIN works for high churn risk.
- UPSELL DATA works for voice-only customers with data appetite.
- Acceptance drops when the price is far above the customer's spend.
- L2 offers are accepted more often but cost more.

Then `simulate.py --weeks N` runs a closed loop, week after week:
1. Generate the JSON files for date D in `data/fsdr/<D>/` (all windows exactly as BR section 4).
2. Run training, scoring and recommendation for every arm.
3. Draw accept / ignore from the hidden response function. Write the fulfillments into the **next** week's `subs_btl_auto` / `sub_btl_3months`.
4. Evolve behaviour: accepted offers raise revenue/usage in the next windows, churners go inactive, appetites drift slowly, and new customers arrive.
5. Advance D by 7 days and repeat.

This must show that a new scoring detects new behaviour through the W1/W2 trends and the refreshed scores. It must also show whether the model, and the LLM, learn from last week's responses.

## 4. Deterministic ML pipelines (exposed as tools)

**Training** (port of `01_nba_train.py`). Steps, each one a callable tool with typed input/output and a JSON report:
1. `load_and_validate`: read the JSON folder, check schemas, key types, duplicates and coverage per table.
2. `build_features`: identical function for training and scoring.
3. `build_labels`: targeting-based (BR-36…45) **and/or** campaign-log-based. The reward is "took the recommended offer", "took one of the same service" or "any BTL" (configurable). Use the real logged propensity when available.
4. `train_bandit`: `--cb_explore_adf -q UA --epsilon 0.2`, `logged` mode (default) or `simulation` mode (original).
5. `evaluate`: holdout, IPS / SNIPS estimate of the policy vs the logged policy, action distribution, comparison vs the hidden-truth optimal policy (synthetic only).
6. `register_model`: version, metadata (features, actions, data dates, metrics), champion / challenger.
7. `score_full_base`.

**Scoring** (port of `02_nba_score.py`):
1. Load the champion; refuse to run if the feature list or action list differs.
2. Build fresh features (D-anchored).
3. Score the target or the base with policy `sample` / `greedy`. Seed configurable, default = date.
4. Write `results_scoring_nba` and per-arm outputs.
5. Drift report vs training data (PSI per feature, action-mix shift).

**Recommendation (NBO)**, port of Codes 3 and 4:
- **constraint arm**: action → offer families and generosity (BR-66; config switch between the "normal" mapping and the currently active "all L2"); one ranker per action × validity.
- **free arm**: L2, DATA + COMBO only (BR-67).
- ARPU anchor = mean + std of periods; revenue = subscriptions + total PAYG (BR-68…77).
- 3-pass price bands with k per action (BR-78…82), at most 2 offers per validity and 4 in total.
- Outputs `recos_monthly_*` / `recos_weekly_*`.
- Add the missing rule: **conflicting offers** (`conflicting_offers_h`) must never be sent together.

## 5. Agentic layer: give the LLM real freedom as recommender

### 5.1 Pipeline agents (training / scoring / monitoring)
Each agent runs the same tool-loop pattern already used in `council.py`: one JSON reply at a time, tool call or final answer, code-enforced step caps, and repeated calls refused. Each agent also has a rule-based fallback.
- **Data agent** generates or loads the data and runs validation. It decides whether data is usable and explains the blocking issues.
- **Training agent** runs the training tools. It may choose the training mode, the reward definition and the hyper-parameters within allowed ranges. It reads the evaluation report and decides **promote / keep champion / ask human**, with a written justification. Promotion requires metrics ≥ champion (code-enforced).
- **Scoring agent** runs at campaign start. It checks drift and model age, and refuses to score with a stale or incompatible model.
- **Monitor agent** writes the weekly report: take rate and revenue per arm (cg / free / cstr / llm), uplift vs control, bandit vs LLM agreement, drift, and recommendations for next week.

### 5.2 Recommender agent (the core change)
Extend the existing council so that in the new **`llm` arm** the LLM is the decision-maker:
- **Tools it can call (evidence, not orders):**
  - customer 360: raw features, W/M trends, the 11 scores, last campaigns and responses
  - bandit pmf over the 7 actions
  - ranker top-N offers per action / validity, with scores
  - ARPU anchors and price-band results
  - eligibility and conflict check
  - cohort graph / similar customers
  - competitor comparison
  - campaign history ("what did we send last week and did they take it?")
- **Freedom:** the LLM may pick **any of the 7 actions** and **any eligible catalogue offer**, even when it disagrees with the bandit or the ranker. It may also propose "no contact this week". It must return a structured justification: signals used, why it overrode the model (if it did), and expected acceptance and value.
- **Hard rules (code-enforced, the LLM cannot override):**
  - offer exists in the catalogue
  - customer eligible (BR-01…12, excluded plans and segments)
  - no conflicting offers together
  - at most 2 offers per validity and 4 in total
  - contact-frequency cap
  - price > 0
  - no aggressive upsell to customers above a churn threshold (configurable)
- **Soft rules (shown to the LLM as guidance, overridable with a justification):** the ARPU price band, the action → generosity mapping, the bandit's top action. Every override is logged and counted.
- **Critic** checks hard rules and the quality of the justification, then approves, revises or rejects. **Messenger** and **explanation** stay as they are.
- **Learning-compatible logging:** for each decision log `policy_source` (bandit / ranker / llm / fallback), the bandit pmf, the chosen action, and a **propensity**. To keep off-policy learning valid, the final decision in the llm arm is: with probability ε_llm (config, default 0.1) take a bandit-sampled action (known propensity); otherwise take the LLM decision (propensity = 1 − ε_llm + ε_llm × pmf[action]). Rewards from all arms feed the next training, so the bandit learns from the LLM's choices too.
- **Offline mode:** the recommender fallback reproduces the deterministic constraint-arm logic, so `llm` arm = `cstr` arm when no LLM is connected.
- **Batch mode:** the llm arm must handle the whole llm-arm sample in batch, with concurrency for vLLM and progress plus resume for Ollama. Add a cache keyed on (customer feature hash, catalogue version, model version).

## 6. Feedback loop & campaign log
- `nba_campaign_log` (append-only, parquet): run_date, arm, msisdn, action, offers sent (codes, prices, validity), policy_source, propensity, bandit pmf, model version, LLM model name, justification id.
- `feedback.py` runs at D+7: joins fulfillments, computes reward per decision for the 3 reward definitions, and writes `nba_rewards`.
- New "last campaign" features for the next scoring: `PREV_ACTION`, `PREV_OFFER_TAKEN`, `DAYS_SINCE_LAST_BTL`, `NB_CONTACTS_LAST_4W`. Add them to the feature list (and the model-compatibility check).
- Guard against price escalation: option to cap the ARPU anchor growth week over week, or to exclude campaign purchases from the anchor (config).

## 7. API, UI, CLI
- **CLI** (`run.py`): `fsdr-generate --date --n --scenario`, `fsdr-validate --data-dir`, `fsdr-train`, `fsdr-score --population target|base --policy sample|greedy`, `fsdr-recommend --arm free|cstr|llm`, `fsdr-feedback`, `fsdr-simulate --weeks N --arms cg,free,cstr,llm`, `fsdr-report`.
- **API**: endpoints for each step + `GET /fsdr/runs`, `GET /fsdr/models` (registry, champion), `GET /fsdr/campaign-log`, `GET /fsdr/report/{week}`. Plus a streaming endpoint for agent steps (reuse the `/agentic/stream` pattern).
- **UI** – new tabs, added after the 12 existing ones, built with the existing theme and components (see section 0, rule 3):
  - **FSDR Data**: generate dummy data (choose size, date, scenarios), upload a folder or zip of real JSON files, preview each table, validation report.
  - **FSDR Training**: run with agent trace, metrics, registry, promote / rollback.
  - **FSDR Scoring & Offers**: per arm, downloadable outputs.
  - **FSDR LLM Recommender**: single customer with full trace and overrides; batch.
  - **FSDR Simulation**: weekly take rate / revenue per arm, uplift vs control, override rate, drift.
- Real data: `JsonSource` reads a folder of JSON files named after the tables. I can drop my real JSON exports in `data/fsdr/<date>/`, or upload them, and run the same pipeline. Validate them against `schemas.py`, including cross-table key consistency.
- Outputs (`results_scoring_nba`, per-arm files, `recos_*`, reports) are written as JSON too, plus CSV export from the UI. The campaign log and rewards can stay parquet internally, with JSON export.

## 8. Tests (all offline, fast)
- Schema test: every generated JSON file matches `schemas.py` and the JSON Schemas exactly (names, case, raw types).
- Round-trip test: generate dummy JSON → load with `JsonSource` → the same dataframes as the in-memory generator. Both array and JSON Lines formats are read.
- Scenario tests: each scenario changes the expected signal (e.g. `churn_wave` raises PROBA_CHURN and RETAIN takes in the affected segment), and `data_quality_issues` is caught by validation.
- Calendar test: T and every window of BR section 4 for several dates, including a Thursday, a Wednesday and a January rollover.
- Feature parity test: features built from the same raw data are identical for train and score paths, and match the reference `nba_lib.py` on a small fixture.
- Business-rule tests: one test per rule group (eligibility, segment mapping, reward, ARPU anchor, each price-band pass, offer families, conflicts, max offers).
- Bandit tests: action order is stable, save/load round trip gives identical pmf, and the compatibility check refuses a mismatched model.
- Agent tests with the existing fake LLM: an LLM override is accepted when justified, a hard-rule violation is rejected and repaired by the critic, propensity is logged correctly, and the fallback works.
- Simulation smoke test: 3 weeks × 2,000 customers in < 60 s. The bandit / llm arm take rate is not below control on the synthetic world with a known signal.
- Keep all 104 existing tests green.

## 9. Docs
Update `docs/` with: architecture diagram, JSON data dictionary (every file and field, with an example record), the scenario list, business-rule ↔ code mapping (BR-xx → function), how to plug real data, how training/scoring/feedback run weekly, and how the LLM freedom and guardrails work. Add a "known differences vs production" section.

## 10. Phases (stop and report after each)
1. Plan (no code) → my approval.
2. `config`, `schemas` + JSON Schemas, `calendar`, `JsonSource`, dummy world + generator + scenarios, validation + tests.
3. Features, labels, bandit, registry, training & scoring pipelines (deterministic) + tests.
4. Offer ranker, offer rules, recommendation for free / cstr arms + tests.
5. Agents (data, training, scoring, monitor) + recommender agent with freedom, hard/soft rules, logging + tests.
6. Campaign log, feedback, closed-loop simulation + tests.
7. API, CLI, UI tabs, FSDR rebranding, docs.

If something in my reference files is ambiguous (e.g. the two segment mappings marked CHECK, the overlap of the targeting and reward windows, the forced L2 generosity), list it in the plan with your proposed default instead of guessing silently.
