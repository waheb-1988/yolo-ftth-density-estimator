# FSDR NBX Recommender – MASTER PROMPT (final; fully agentic LLM recommenders; replaces every earlier prompt)

> One complete specification for Claude Code (VS Code). The recommenders are **LLM agents**: an **LLM NBA agent** chooses the action and an **LLM NBO agent** ranks the offers. Two **ML helper agents** give them extra evidence: **⑩ Action-bandit agent** (the strength of a contextual bandit like Vowpal Wabbit) and **⑪ Offer-pattern agent** (the strength of a collaborative-filtering ranker like LightFM). **The helpers never decide**: they only score, and the LLM decides inside the business rules. My original codes 1–5 are given **so you understand the data and the business rules**, which stay in force.
>
> **Training** runs on the **full base (~1,000,000 lines)**. **Scoring** runs on the **40k target**: **Target 1 (20k)** + **Target 2 (20k)**, plus a control group that is never contacted. The final campaign file **`input_flytxt`** must match code 5 exactly. A **⑨ Flow & rules auditor** checks every flow, every rule and the output, and blocks the push on failure.

## Part A – Steps for you
1. **Commit:** `git add -A; git commit -m "before FSDR master build"`.
2. **Reference files:** unzip `fsdr_reference.zip` into `nbx-python/docs/fsdr_reference/`. It contains:
   - `fsdr_original_python_codes.txt` (codes 1–4: **rules reference only**)
   - `fsdr_original_code5_final_output.txt` (code 5: **source of truth for `input_flytxt`**)
   - `fsdr_original_sql_and_schemas.txt`
   - `FSDR_NBA_business_rules.md`
   - `fsdr_generate_dummy.py`
   - this file
3. **Real data (optional):** put 10–20 anonymised rows per table in `docs/fsdr_reference/data_samples/`. Include one real `input_flytxt` if possible: it is the format reference.
4. **Paste:** open Claude Code in `nbx-python`, switch to **plan mode**, and paste **Part B**.
5. **Review the plan.** Check:
   - the input contract (section 2)
   - the two targets and the control group (section 19.2)
   - the two pipelines, training and production scoring, and which table each one uses (section 17.9, then 18–19)
   - the auditor checks (section 17)
   - the open items (section 25)

   Then answer `Approved`. After each phase: run `python run.py test`, commit, and reply `continue`.

---

## Part B – Master prompt (paste everything below)

You are working in my existing project **NBX** (`nbx-python/`): a local, offline recommendation engine. It has an agent council (`nbx/agents/agentic/council.py`), a cohort knowledge graph (`core/knowledge_graph.py`), a competitor module (`core/competitors.py`), a rules engine (`agents/rules_engine.py`), FastAPI (`nbx/api/main.py`), a Gradio UI, and offline / ollama / vllm LLM modes. It also contains ML modules (`core/nba_bandit.py`, `core/collab_filter.py`): **leave them and their tests untouched, but do not use them for FSDR decisions**. A partial FSDR extension may exist in `nbx/fsdr/`: reuse what is correct and replace what is not.

Build the **complete FSDR NBX Recommender**, a **fully agentic LLM recommender**:
- **(1) LLM NBA agent:** does the job a contextual bandit would do. It reads the customer context and the evidence, scores every allowed action, explores (ε), logs the propensity, and learns every week from rewards.
- **(2) LLM NBO agent:** does the job a collaborative-filtering ranker would do. It scores the rule-filtered offer candidates using purchase evidence from the whole base; the business rules then select the offers.
- **(3) Business rules:** every rule is applied **in code** around the LLM; the LLM can never break a rule. Target 1 uses the production rules of codes 1–5 (data, features, offer families, generosity, ARPU anchors, price passes, slots, favourites, forced bundles, duplicates, conflicts, the output format). Target 2 uses the same data rules, guardrails and code-5 output rules, with the **challenger offer and ARPU rules** (section 13) in place of B-21 … B-31.
- **(3b) ML helper agents (section 10.2):** ⑩ learns fine-grained action effects from the logged campaigns (bandit); ⑪ learns hidden co-purchase patterns from the whole base (matrix factorisation). Their scores are evidence E-A5 / E-O6 for the LLM agents and the first fallback when the LLM fails.
- **(4) Training** on the **full base (~1M lines)**: features, segments, micro-segments, **evidence tables**, the **two helper models** and the **weekly LLM policies**.
- **(5) Weekly scoring and production** on the **40k**:
  - **Target 1 (20k), production rules:** the LLM agents decide inside **my production business rules and ARPU policy** (codes 1–5: action → offer families and generosity, ARPU anchors, +2 / +3 price passes, code 5). Applied by ⑥A.
  - **Target 2 (20k), challenger rules (B):** the same LLM agents decide inside the **challenger rules** of ⑥B (a better rule set proposed, tested and approved by me; first version C-v1 from week 0). Applied by ⑥B.
  - **Control (~10k):** never contacted
  - Then audit, approval, push, feedback, learning and rollback.
- **(6) ⑨ Flow & rules auditor** blocks the push on any hard failure.

### 0. Ground rules
1. **Plan mode first.**
   - Read the whole repo and every reference file.
   - Write `docs/fsdr_plan.md`: modules, files, input contract, agents, tools, rule registries, evidence tables, prompts, flows, tests, phases, and **open questions with your proposed default**.
   - Wait for my approval.
   - Keep the plan updated after each phase.
2. **Keep the NBX design:**
   - layers UI → API → agents → core; the UI holds no logic
   - the `council.py` agent pattern: one JSON reply per step, tool call or final answer, step cap, evidence required before finishing, repeated calls refused, rule-based fallback, `mode` = fallback / agentic / partial_fallback
   - same Gradio theme; new tabs after the existing ones
   - pip-only
   - **local LLM only** (vLLM / Ollama on 127.0.0.1)
   - **no cloud, no external API, no internet at runtime**
   - all existing tests still pass
3. **LLM decides, ML helps.**
   - The **LLM agents decide** the action and the offer scores.
   - ML models exist **only inside ⑩ and ⑪** (section 10.2). They produce scores that are passed to the LLM as evidence; they never write a decision, an action or an offer to the outputs. Do not reuse the original VW / LightFM code: build ⑩ and ⑪ clean, with the bugs of codes 1–4 fixed.
   - Every number, segment, evidence row, candidate list, rule check and file row is produced **in code**.
   - Every LLM call has a **deterministic fallback** (evidence ranking).
   - The same inputs, evidence version, prompt version, model and seed give the same outputs.
4. **Constants in config.** Every constant, threshold, band, list, offer name and target size goes in `nbx/fsdr/config.py` or the YAML rule registries, each with its rule id.
5. **Names:** the operator is **"FSDR"**; competitors are `COMPETITOR_A` / `COMPETITOR_B`. Dataset, file and column names from my codes are kept **exactly** (technical identifiers, including `input_flytxt`, `results_scoring_nba`, `TG_40K_constraint_scored_nba`, `recos_monthly_40k_constraint`, and the `nbo_LFM_` opportunity prefix, which the campaign platform expects).
6. **Original bugs:** a flag `reproduce_original_bugs` (default **false**) exists only for the **code-5 parity test**. Bugs found in codes 1–4 are fixed and listed in `docs/fsdr_known_differences.md`.
7. **Work one phase at a time** (section 26). Stop after each phase and show the evidence listed there.

### 1. Reference files
| File | Role |
|---|---|
| `fsdr_original_python_codes.txt` | Codes 1–4: **reference for data preparation and business rules** (features, segments → actions, offer families, generosity, anchors, price passes). Their VW and LightFM parts are **not** to be rebuilt |
| `fsdr_original_code5_final_output.txt` | Code 5: **source of truth** for `input_flytxt` |
| `fsdr_original_sql_and_schemas.txt` | Schemas and meaning and windows of every column. **Do not run or translate SQL**: data arrives already extracted |
| `FSDR_NBA_business_rules.md` | Rules BR-01 … BR-86 (cross-check with the codes; ignore the VW / LightFM model settings) |
| `fsdr_generate_dummy.py` | Tested dummy generator. Extend it to every table of section 2, to a **1M-line base**, and to the weekly loop |

### 2. Input data contract (all tables used by codes 1–5)
- **Format (production): `.csv.gz` only** – the Dataiku "compressed CSV" export. One file per table, named **exactly** as the table (`<table>.csv.gz`, e.g. `activities_w1_score.csv.gz`), in `data/in/<run_date>/`, plus `_manifest.json` (run_date D, anchor_date T, file list, row counts, schema_version). A missing `.csv.gz` for a required table is a blocking error. Plain `.csv`, `.json` / `.jsonl` are accepted **only** for tests and the dummy generator's small mode (flag `allow_test_formats`, default false in production). Excel and `.geojson` are never accepted.
- **CSV format (the "FSDR CSV standard", used for every input AND every output file):**
  | Setting | Value |
  |---|---|
  | Compression | gzip (`.csv.gz`), level ⚙6 |
  | Encoding | UTF-8, no BOM |
  | Separator | comma `,` |
  | Header | first row, column names exactly as the table (inputs: as exported by Dataiku; `input_flytxt`: the 38 lower-case names of F-19) |
  | Quoting | double quotes `"` only when a value contains a comma, quote or line break; inner quotes doubled |
  | Line ending | `
` |
  | Decimal | dot `.`; no thousands separator; no scientific notation for prices and amounts |
  | Dates / timestamps | `YYYY-MM-DD` / `YYYY-MM-DD HH:MM:SS` |
  | Missing value | empty field (never `nan`, `None`, `NULL`) |
  | Index column | none |
  | Arabic scripts (`offer_script_ar_*`) | UTF-8 text, kept exactly as in `BTL_script_english_arabic_ok` |
- **Types:** next to each input, an optional `<table>.schema.json` (Dataiku schema export: column names and types) is used when present; otherwise the types come from this input contract. **Identifier columns (`ACCESS_METHOD_VAL`, `MSISDN`, `ACCESS_METHOD_ID`, `ACCOUNT_ID`, `PARTY_ID`, `SUBSCRIPTION_ID`, `UBSCRIPTION_ID`) are always read and written as text** (no lost leading zeros, no `9.68e+07`).
- **Reading:** polars (or duckdb) reads the `.csv.gz` directly, only the columns listed in this contract (`columns=`), with the types above, in chunks for the large tables.
- **On load:** each file is validated (row count vs `_manifest.json`, SHA-256 checksum logged, header vs contract) and converted **once** to Parquet in `data/cache/<run_date>/` (section 19.5). The pipelines read only that Parquet cache; Parquet is internal and never delivered.
- **Keys:** normalise `ACCESS_METHOD_VAL`, `ACCESS_METHOD_ID`, `ACCOUNT_ID`, `PARTY_ID` to one type before any join. `MSISDN` in `access_method_id_auto` / `targeting_auto` = `ACCESS_METHOD_VAL`, with no prefix.
- **Dummy generator:** produces **every** table below as `<table>.csv.gz` in the FSDR CSV standard, with its `_manifest.json` and `<table>.schema.json`, for any run date and for consecutive weeks, at 1M lines (and a small 2,000-line mode for tests).

**2.1 Base and reference**
| Table | Fields | Used by |
|---|---|---|
| `base_prepaid` | UBSCRIPTION_ID (sic), ACCESS_METHOD_VAL, ACCESS_METHOD_ID, ACCOUNT_ID, PARTY_ID, access_method_subs_start_dt, CURRENT_PLAN | Population (full base), eligibility, tenure |
| `consumer_aact_account` | ACCOUNT_ID, PARTY_ID | Account → party |
| `access_method_id_auto` | ACCESS_METHOD_ID, MSISDN, ACTIVATION_DATE | Targeting id → MSISDN (week 0) |
| `party_auto` (week 0), `party_scoring` (weekly) | party_id, AGE, GENDER, NATIONALITY, nationality_identifier | Nationality group (and age if the profile table is missing) |
| **`customer_profile`** (provided by me weekly) | **msisdn, age, tenure** (tenure in **days** by default; unit set in config) | **Age band and tenure band (micro-segments).** Overrides `party_*.AGE` and the tenure computed from `access_method_subs_start_dt` when present. The validator reports coverage and disagreements |
| `location_auto` | Access_Method_id, Calling_Month, Calling_Year, REGION, WILAYAT, AREA | Agent context (region) |
| `DEVICEBEHAVIORSEGMENTATION.dv_segment` | ACCESS_METHOD_ID, DV_SEGMENT | Agent context (device) |
| `btl_catalog` | btl_name, type, generosity, gb, min, price, Price data, Price voice, service, validity | Offers, candidates, slots, F-09 (type + price) |
| `BTL_script_english_arabic_ok` | Sl No., destination_offer_name, English Script, Arabic Script | F-10 scripts |
| `conflicting_offers_h` | Offer_A, Gen_Level_A, Price_A, Data_A, Min_A, Validity_A, Offer_Type_A, Offer_B, Gen_Level_B, Price_B, Data_B, Min_B, Validity_B, Offer_Type_B, Price_Difference, Data_Difference, SAME_VALIDITY, SAME_OFFER_TYPE | F-12 conflicts (**required**) |

**2.2 Usage and spending.** `_auto` = anchored on Thursday T (**week-0 training only**); `_score` = anchored on run date D (**every week, training on the full base and scoring**).
| Table | Fields | Used by |
|---|---|---|
| `subs_amt_auto`, `subs_amt_score` | ACCESS_METHOD_VAL, ACCESS_METHOD_ID, ACCOUNT_ID, Sub_Amt_w1…w4, Sub_Amt_m1…m3 | Revenue features |
| `activities_w1_auto` … `activities_w4_auto`, `activities_m1_auto`, **`activities_auto_m2`**, `activities_m3_auto`; `activities_w1_score` … `activities_m3_score` | ACCESS_METHOD_VAL, ACCESS_METHOD_ID, ACCOUNT_ID + per period P: Call_Count, Data_Session_Count, Voice_Traffic_min, Data_Volume_MB, PG_Total_Amount, PG_Voice_Amount, PG_Data_Amount, Onnet_Call_Count, Onnet_Voice_min, Offnet_Call_Count, Offnet_Voice_min, Local_Call_Count, Local_Voice_min, Active_Days_Count, Active_Days_Voice_Count, Active_Days_Data_Count | Features, segmentation |
| `paygo_activity_auto` | ACCESS_METHOD_ID, ACCOUNT_ID, ACCESS_METHOD_VAL, PG_Total_Amount_m1…m3, w1…w4 | Validate and keep available |
| `subscriptions_tg` | ACCESS_METHOD_VAL, ACCESS_METHOD_ID, ACCOUNT_ID, Total_/Data_/Voice_Sub_Amt_w1…w4, Sub_Count_M1…M3, Total_/Data_/Voice_Sub_Amt_M1…M3 | ARPU anchors (B-26, F-05), revenue reward |
| `usage_data_voice_payg_tg` | ACCESS_METHOD_ID, ACCOUNT_ID, ACCESS_METHOD_VAL, local_Voice_min_p1…p3, data_volume_mb_p1…p3, PG_Total_Amount_m1…m3, w1…w4 | ARPU anchors, usage mix |

**2.3 The 11 model-score tables.** One shared schema: subscription_id, access_method_val, access_method_id, tenure, life_in_nw_range, proba_0, proba_1, prediction. One schema object, one registry, one loader, one generator; adding a model is a config change only.

Files → features:
- PROBA_CHURN: stack of the 3 churn tables (DAIM_CHURN_GLOBAL_SCORING_TD.prediction, DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction, DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored)
- PROBA_REACT_30: DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored
- PROBA_REACT_15: PROPENSITY_REACTIVATION_15D_401_SCORING.Scored
- PROBA_APP_DATA: APPETENCY_DATA_204_SCORING_TD.prediction
- PROBA_APP_VOICE: APPETENCY_VOICE_204_SCORING_TD.Scored
- PROBA_CHURN_1D: CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring
- PROBA_REV_DECR: REV_DECREASE_204_SCORING_TD.predictions
- PROBA_FRAG_DATA: DAIM_FRAGILITY_DATA_SCORING_TD.prediction
- PROBA_FRAG_VOICE: DAIM_FRAGILITY_VOICE_SCORING_TD.prediction

Accept `…REACTIVATION_30D_404_SCORING_TD.filterd_nationality` if present (not used). The scores are **one version shared** by training and scoring.

**2.4 Campaign history, purchases, target**
| Table | Fields | Used by |
|---|---|---|
| `targeting_dataiku_auto` | date1, access_method_id, account_id, segment, destination_offer_name_1…6 | **Week-0 action evidence**: past segment → action contacts (with `subs_btl_auto` for their results) |
| `targeting_auto` | DATE1, START_DATE, END_DATE, MSISDN, CHANNEL_1…6, OPPORTUNITY_NAME_1…6, DESTINATION_OFFER_NAME_1…6, FLAG_SPLIO, OFFER_SCRIPT_EN/AR_1…6, SEGMENT, FILENAME, ETL_CREATE_DT, JOB_ID, SEGMENT_BTL_TAKER, SEGMENT_BTL_TAKER_UPSELL | Other campaigns sent (contact conflicts, clean control); week-0 offer evidence (offered vs taken); format reference for `input_flytxt` |
| `subs_btl_auto` | access_method_val, PRODUCTNAME, **FULFILMENTSUCCESSDATE_TIME**, sub_flag, nb_sub | **Dated purchases**: week-0 evidence and **weekly rewards** (both targets **and** control) |
| `sub_btl_3months` | access_method_val, PRODUCTNAME, sub_flag, nb_sub (no date) | 270-day history of the **full base**: offer evidence (E-O1, E-O2, E-O3), favourites (F-06), BTL taker flag (F-15). **Not usable for rewards** |
| `sample_cg_tg_new_7oct2026` (optional; all fields string) | access_method_val, access_method_id, account_id, tenure, activation_date, plan_name, payg_revenue, subscription_revenue, total_revenue, total_revenue_3m, voice_traffic, data_volume, nationality, region, ARPU, ARPU_3m, voice_trf_group, data_trf_group, Final, channel, group | If provided, the panel source (cstr → Target 1, free → Target 2, cg → control; the groups are only used as random stratified samples); otherwise we build the panel (section 19.2) |

**2.5 Internal tables we produce** (inputs to the next week)
- `base_features_<run_date>` (full base, Parquet)
- `evidence_<version>` (all evidence tables E-A* / E-O*)
- `policy_<version>` (level-1 cell policies)
- `feature_snapshot_<run_id>` (the 40k + control at push time)
- `nba_campaign_log`, `nba_rewards`
- `panel_<T>` (Target 1, Target 2, control)
- the registry (evidence, policy, prompt and model versions)
- the rule registries
- `audit_report_<run_id>`

### 3. Calendar and windows (D-xx)
- **D-01** T = most recent Thursday on or before D: `T = D − (((D − 2026-03-12) mod 7) + 7) mod 7`.
- **D-02** Spend (`subs_amt`) windows **include** the anchor: W1 A-6…A, W2 A-13…A-7, W3 A-20…A-14, W4 A-27…A-21, M1 A-29…A, M2 A-59…A-30, M3 A-89…A-60.
- **D-03** Activity windows **exclude** the anchor: W1 A-7…A-1, W2 A-14…A-8, W3 A-21…A-15, W4 A-28…A-22, M1 A-28…A-1, M2 A-56…A-29, M3 A-85…A-57.
- **D-04** `subscriptions_tg`: W1 D-8…D-2, W2 D-15…D-9, W3 D-22…D-16, W4 D-29…D-23; M1 D-31…D-2, M2 D-61…D-32, M3 D-91…D-62.
- **D-05** `usage_data_voice_payg_tg`:
  - voice / data p1 D-31…D-2, p2 D-61…D-32, p3 D-91…D-62
  - PAYG m1 D-30…D-1, m2 D-60…D-31, m3 D-90…D-61, w1 D-7…D-1, w2 D-14…D-8, w3 D-21…D-15, w4 D-28…D-22
- **D-06** Week-0 targeting date1 D-8…D-2 · week-0 reward D-7…D-1 · history D-270…D-1 · location = previous calendar month of D-2 (January rolls back to December).
- **D-07** Base: active prepaid consumer lines on D-2, MSISDN > '69999999', 13 internal / employee plans excluded (list in config from the SQL file), latest activation per MSISDN.
- **D-08** Spend scope: CVM, DATA, Other Base Plans, DIGITAL SERVICE, EASY CREDIT, EXPIRED CREDIT, Flexi, Hala, INTERNATIONAL VOICE, New Shababiah, Roaming, SMS, VOICE, Hala Flex. Data subscriptions = Daily / Weekly / Monthly Data; voice = Local Voice.
- **D-09** **Production reward window:** from START_DATE (push date + 1) to the day before the next run. Run every Thursday D → START_DATE Friday → next run D+7 sees purchases up to D+6 = **6 days (Friday–Wednesday)**. Window length is config. Monthly offers are also tracked over 30 days in a lagged report.

### 4. Agents and flow
| # | Agent | Role | LLM use |
|---|---|---|---|
| ① | Data agent | Load and validate every table; weekly completeness | Explains blocking issues |
| ② | Segmentation agent | Segment + rule action (S-rules), nationality, age band, tenure band, micro-segment, for the **full base** | One-line explanation |
| ②b | **Segment-action agent** (new) | Gives every line **two actions**: `action_original` (my original dictionary) and `action_agent` (the agent's own segment → action map, written weekly by the LLM from the evidence) | **Writes the agent dictionary** weekly (per segment × nationality) |
| ③ | **LLM NBA agent** | Scores the allowed actions from the evidence; code samples the action (ε) and logs the propensity | **Decides the action** |
| ④ | **LLM NBO agent** | Scores the rule-filtered offer candidates from the purchase evidence | **Ranks the offers** |
| ⑩ | **Action-bandit agent** (ML helper) | Contextual bandit trained on the logged campaigns: probability and expected reward per allowed action → evidence E-A5 for ③ | None (model only) |
| ⑪ | **Offer-pattern agent** (ML helper) | Implicit-feedback matrix factorisation on the 1M base's purchases: pattern score per candidate offer → evidence E-O6 for ④ | None (model only) |
| ⑤ | Market intelligence | Offline competitor comparison (index per offer) | Report narrative |
| ⑥A | Production rules agent | **Target 1:** the business rules and ARPU policy of codes 1–4 exactly (B-20 … B-32) | None |
| ⑥B | Challenger rules agent | **Target 2:** applies the approved challenger rule version (C-v1 from week 0); every week proposes, tests and documents the next version | Hypotheses, memo |
| ⑦ | Final output agent | Code-5 rules → `input_flytxt` (both targets, one file) | Summary |
| ⑧ | Strategy & monitor | Weekly playbook, panel rules, weekly report, drift | Narrative |
| ★ | Chief recommender (orchestrator) | Runs ③ → ④ → ⑤ → ⑥ → ⑦ per customer, resolves conflicts, runs the weekly level-1 policy update | **Orchestrator** (RL + marketing expert) |
| ⑨ | **Flow & rules auditor** | Checks every flow step, every applicable rule, target integrity and the exact `input_flytxt` format; **blocks the push on hard failures** | Audit narrative only |
| ⑬ | **Revenue guard agent** (ARPU and cannibalisation) | Before push: checks every recommended offer of **both targets** against the customer's ARPU and current spending, repairs or removes any offer that would lower his revenue. At T+7: measures real cannibalisation vs control and blocks cannibalising offer × segment pairs (section 14.1) | Narrative only |
| ⑫ | **Customer ledger agent** (backup) | Writes **one table, one row per MSISDN per weekly run**, with every calculation, step and decision (not the raw data); adds the rewards at T+7 (section 20.1) | None (copies LLM justifications only) |

**Flow:**
```
TRAINING (full base ~1M, no LLM per customer):
① Data → ② Segments + micro-segments (1M) → features (1M) → evidence tables E-A*/E-O*
  → ②b agent dictionary (LLM, ~70 calls) → action_original + action_agent for all 1M (code)
  → ★ level-1 LLM cell policies

SCORING (40k, every Thursday):
For each customer of Target 1 and Target 2:
  ★ → ③ LLM NBA (action, ε, propensity) → ④ LLM NBO (scores on candidates)
    ├─ Target 1 (20k): ⑥A production rules (families + generosity by action, ARPU mean + std, +2 / +3 passes)
    └─ Target 2 (20k): ⑥B challenger rules (approved version C-vN: relative ARPU band, usage-based split, LLM freedom on families)
  → ⑦ code 5 → one input_flytxt (≈40k rows, 38 columns)
→ ⑨ Auditor → human approval → push → campaign log
⑧ Strategy: weekly playbook before scoring, weekly report after feedback
Control (~10k): never contacted, measured every week
```

**Rule registries** in `nbx/fsdr/rules/`:
- `data_rules.yaml` (D)
- `segmentation_rules.yaml` (S)
- `production_rules_offers.yaml` (B)
- `final_output_rules.yaml` (F)
- `guardrails.yaml` (H)
- `production_flow_rules.yaml` (P)
- `challenger_rules_vN.yaml` (C)

Each entry has: id, description, source (file and line), stage, target(s) where applicable, hard / soft, parameters (config), check function, repair strategy.

### 5. ① Data agent
- **Tools:** `load_tables`, `validate_tables`, `table_report`, `weekly_completeness`, `generate_dummy`.
- **Report:**
  - per table: rows, missing / extra fields, type errors, null keys, duplicates, window checks (D-xx), model coverage
  - cross-table checks: e.g. a panel MSISDN not in the base, a purchase of an unknown offer
- **`customer_profile` checks:** coverage of the base and panel; age outside 15–100; negative tenure; disagreement with `party_*.AGE` (> ⚙2 years) or with activation-date tenure (> ⚙30 days), reported with counts.
- **Weekly completeness** (section 17.9 table): every table required by the pipeline about to run (training week 0, training week n, or scoring) is present and fresh.
- Blocking errors stop the run.

### 6. Data preparation and features (rules from code 1: B-01 … B-12)
- **B-01** **Population:**
  - **Training:** the **full eligible base (~1M)** with location, demographics, device segment, churn, the 7 activity windows, 8 other scores and spend, joined by **left joins** in code-1 order (`_auto` tables in week 0, `_score` tables from week 1).
  - **Scoring:** Target 1 + Target 2 (+ control for measurement); party through `consumer_aact_account` → `party_scoring`.
- **B-02** Nursering segments (`Nursering_Early_Churn_At_Risk`, `Nursering_Early_Churn_Not_At_Risk`) are never targeted. Targeting rows without a segment are excluded from week-0 evidence.
- **B-03** Original `dict_segm_actions` (12 segments → 7 actions; case-sensitive; last value wins). It maps `targeting_dataiku_auto` segments to actions for the week-0 evidence:
  - `Retention_High_Probability_To_Reconnect` → RETAIN (**CHECK**: also UPSELL)
  - `RGS_Not_At_Risk_Data_Only_Voice_Appetency` → UPSELL VOICE vs `Rgs_…` → CROSS SELL VOICE (**CHECK**)
- **B-04** Churn = the 3 churn tables stacked; first row per line.
- **B-05** Missing model score → flagged as missing (`*_missing = 1`) and imputed with the median of the customer's nationality group (the original uniform [0, 0.02] draw is documented as replaced). A missing score never triggers "at risk". Missing usage and spend → 0.
- **B-06** `REV_P = SUB_AMT_P + PG_DATA_AMOUNT_P` (W1…W4, M1…M3).
- **B-07** Trends (`calcul_evol`): % change (0 if the base = 0), zero flag, log change, from older to recent, clipped to [−100, 500] % (the original inverted direction is documented as fixed).
  - Revenue: W1-W2, W2-W3, W3-W4, M1-M2, M2-M3.
  - Local voice minutes and data MB: W1-W2, W2-W3, W3-W4, M1-M2.
- **B-08** Nationality group: Oman, India, Bangladesh, Pakistan, others (null → others).
- **B-09** Week-0 reward = the targeted line bought a **catalogue** BTL between D-7 and D-1.
- **B-10** **Cold-start action priors** (share of each action in past targeting): CROSS SELL DATA 0.0083128, CROSS SELL VOICE 0.0528180, MAINTAIN 0.1015183, RETAIN 0.5086577, UPSELL 0.2624428, UPSELL DATA 0.0422645, UPSELL VOICE 0.0239859. They are recomputed from `targeting_dataiku_auto` in week 0 and are used only when the evidence for an action is too thin (section 8).
- **B-11** **Fixed action list and order** (all logs, evidence and prompts): CROSS SELL DATA, UPSELL DATA, RETAIN, MAINTAIN, UPSELL, CROSS SELL VOICE, UPSELL VOICE. The code-1 context (nationality group, the scores, VAR_REV ×5, VAR_VOICE ×4, VAR_DATA ×4, voice / data zero flags, LOG_VOICE ×3, LOG_DATA ×3) **plus PROBA_APP_VOICE** is the feature set of `customer_360`.
- **B-12** Scoring output with the original structure: `results_scoring_nba` (access_method_val, action, proba = **logged propensity**, tg_flag). **Code 2:** tg_flag `constraint` (Target 1) → `TG_40K_constraint_scored_nba`; tg_flag `challenger` (Target 2) → `TG_40k_challenger_scored_nba`.

### 7. ② Segmentation agent (full base)
**Week-0 history:** the segment from `targeting_dataiku_auto` + B-03 (used for the evidence).

**Current segment (S-xx)**, computed every week for the full base; first match wins, ⚙ in config:
- `REV_M1` = Sub_Amt_M1 + PG_Total_Amount_M1
- RGS = `REV_M1 > 0` and `Active_Days_Count_M1 > 0`
- voice_user = Local_Voice_min_M1 ≥ ⚙1; data_user = Data_Volume_MB_M1 ≥ ⚙10
- reconnect_p = max(PROBA_REACT_30, PROBA_REACT_15)
- fragility = max(PROBA_FRAG_DATA, PROBA_FRAG_VOICE, PROBA_REV_DECR)

| # | Condition | Segment | Rule action |
|---|---|---|---|
| S-1/2 | tenure < ⚙45 (at risk if PROBA_CHURN ≥ ⚙0.5) | Nursering_Early_Churn_(Not_)At_Risk | excluded |
| S-3/4 | tenure < ⚙90 (at risk if PROBA_CHURN ≥ ⚙0.5) | Onboarding_Early_Churn_(Not_)At_Risk | UPSELL |
| S-5/6/7 | not RGS; reconnect_p ≥ ⚙0.6 / ≥ ⚙0.3 / lower | Retention_High / Med / Low_Probability_To_Reconnect | RETAIN |
| S-8 | RGS; PROBA_CHURN ≥ ⚙0.5 or PROBA_CHURN_1D ≥ ⚙0.5 | Retention_RGS_At_Risk | RETAIN |
| S-9 | RGS; fragility ≥ ⚙0.5 or revenue M1 vs M2 ≤ ⚙−30 % | Retention_RGS_Fragile | UPSELL |
| S-10 | voice and data | RGS_Not_At_Risk_DualPlay_Upgrade | MAINTAIN |
| S-11/12 | data only; PROBA_APP_VOICE ≥ ⚙0.5 / lower | Data_Only_(No_)Voice_Appetency | UPSELL VOICE / CROSS SELL VOICE |
| S-13/14 | voice only or no usage ⚙; PROBA_APP_DATA ≥ ⚙0.5 / lower | Voice_Only_(No_)Data_Appetency | UPSELL DATA / CROSS SELL DATA |

- **Tenure source:** `customer_profile.tenure` when present, otherwise run date − activation date.
- **Output:** segment, rule action, rule id, trigger values, `segment_source`, plus nationality group, age band, tenure band and micro-segment.
- **The rule action is a hint for the LLM NBA agent** (and its fallback), not a forced choice; allowed actions come from the guardrails (section 9).
- **Tools:** `segment_customer`, `segment_population`, `segment_agreement_report` (S-rules vs the week-0 segment engine), `calibrate_thresholds` (proposes, never applies).

### 7.1 ②b Segment-action agent (new): two actions per customer
**Purpose:** keep my original segment → action logic **and** let an agent create its own, then measure which is better.

**Action 1, `action_original` (my code):**
- The original `dict_segm_actions` (12 segments → 7 actions, B-03) applied to the customer's segment.
- Segment source, in this order: the latest segment-engine segment in `targeting_dataiku_auto` (if provided that week) → otherwise the S-rule segment translated to the original segment name with a crosswalk in config (e.g. S-8 `Retention_RGS_At_Risk` → `Retention_RGS_At_Risk`; S-11 → `RGS_Not_At_Risk_Data_Only_Voice_Appetency`). Every crosswalk line is a CHECK item in the plan.
- `action_original_source` = segment_engine / crosswalk. Deterministic, no LLM.

**Action 2, `action_agent` (agent-created):**
- Every week (training, after the evidence tables), the LLM writes the **agent dictionary** `dict_segm_actions_agent_vN`: for each S-segment (14) × nationality group (5), ≈ 70 calls, the **ranked actions** (top 1–3) with a score, the evidence ids used and a one-line reason.
- Inputs per call: E-A1 / E-A2 for that segment × nationality (contacts, take rate, uplift vs control, n, intervals), the original action, the current agent action, the segment's trigger profile (average scores, ARPU, usage), and last week's change.
- Validated by ⑥ (allowed actions only; excluded segments stay excluded; H-06 at-risk segments cannot get aggressive upsell). Invalid → keep the previous version for that cell.
- Applied **in code** to all ~1M lines → `action_agent`, `action_agent_rank2`, `action_agent_version`.
- **Week 0:** the agent dictionary starts as a copy of the original dictionary; the LLM changes a cell only where the week-0 evidence shows another action with a clearly higher uplift (CI not overlapping). It moves further each week as our own campaign evidence comes in.
- Changes vs last week and vs the original are shown on the approval screen (no separate approval needed, because it is only a hint; ⑥ guarantees it is valid).

**How the two actions are used:**
- Both go into `customer_360` for the LLM NBA agent (③), which still makes the final choice among the allowed actions.
- **Target 1 (production rules):** `action_original` is the **rule action** (hint and fallback), as in production.
- **Target 2 (challenger rules):** `action_agent` is the **rule action** (hint and fallback).
- Each decision logs `action_original`, `action_agent`, `action_sent` and `action_agreement` = all same / sent = original / sent = agent / sent = other.

**Weekly report:** agreement rate per segment; uplift vs control for customers where the sent action = original vs = agent vs = other; the agent dictionary history (what changed and why).

**Tools:** `apply_original_dictionary`, `segment_crosswalk`, `write_agent_dictionary_llm`, `validate_agent_dictionary`, `apply_agent_dictionary`, `action_agreement_report`.

### 8. Nationality, age, tenure and micro-segments (both targets)
**Sources:** `customer_profile` (msisdn, age, tenure). If it is missing: AGE from `party_scoring`, tenure = run date − `access_method_subs_start_dt`.

**Nationality group:** Oman, Pakistan, India, Bangladesh, others.

**Age band:**
| Band | Ages |
|---|---|
| `young` | **under 40**, including under 20 |
| `middle` | **40–59** |
| `senior` | **60 and over** |
| `unknown` | missing, < 15 or > 100 |

Test splitting `young` into **under 25 / 25–39** per nationality; keep it only where it changes the evidence significantly.

**Tenure band** (days; boundaries in config):
| Band | Tenure | Note |
|---|---|---|
| `t0_lt3m` | < 90 days | Under 45 days = Nursering (excluded); 45–89 = Onboarding |
| `t1_3_6m` | 90 – 182 days | 3 to 6 months |
| `t2_6_12m` | 183 – 365 days | 6 to 12 months |
| `t3_1_2y` | 366 – 730 days | 1 to 2 years |
| `t4_gt2y` | > 730 days | More than 2 years |
| `unknown` | missing or negative | |

**Micro-segment** = **nationality × age band × tenure band**: 5 × 4 × 6 = 120 cells. Value band (ARPU tertile) and usage type (voice-only / data-only / dual / none) are **context in `customer_360`**, not micro-segment splits (all five would create more than 1,000 cells, most too small). ⑥B may test adding value or usage as a split for large cells only.

**Evidence level per customer** (fallback chain):

`nationality × age × tenure → nationality × tenure → nationality × age → nationality → global → B-10 priors`

Use the finest level that has enough data: action evidence ⚙300 contacts and ⚙30 takers per cell × action; offer evidence ⚙50 buyers per cell × offer (with 1M lines, most cells reach this for the main offers). Every evidence row carries its n and a 95 % interval, so the LLM can see how reliable it is. The level used is logged per customer.

**Reports:** take rate and incremental revenue per micro-segment and per target (weekly report); agreement between tenure and segmentation (e.g. Onboarding ⇔ `t0_lt3m`).

### 9. ③ LLM NBA agent (the action recommender)
It does the job of a contextual bandit:

| Bandit role | LLM NBA agent |
|---|---|
| Uses the customer context | Gets `customer_360`: segment, **`action_original`**, **`action_agent`** (and its rank-2 action), the 11 scores, revenue / voice / data trends, nationality, age band, tenure band, value band, usage type, region, device, contact and purchase history |
| Estimates how well each action works for this context | Gets the **evidence tables** (code) and its **cell policy** (level 1); returns a **score from 0 to 1 for each allowed action**, with a short reason |
| Explores (ε) | Code turns the scores into probabilities (normalised, then ε = ⚙0.1 spread over the allowed actions) and **samples** the action. The **propensity is logged** |
| Learns from rewards every week | The evidence tables are rebuilt every week from the campaign log, rewards and control (**in-context learning**); the level-1 update rewrites the cell policies |

**Action evidence** (built in code on the full base and the campaign log, per evidence level, with n and 95 % intervals):
- **E-A1:** per cell × action: contacts, take rate, **incremental take rate vs control**, average revenue change. Week 0: from `targeting_dataiku_auto` + `subs_btl_auto` (B-09); from week 1: our own campaign log + rewards.
- **E-A2:** per segment × action: the same at segment level.
- **E-A3:** the customer's own history: past actions received, responses, days since last purchase, contacts in the last 4 weeks.
- **E-A4:** the current policy for the customer's cell (level 1).
- **E-A0:** the B-10 priors (cold start only).
- **E-A5:** the ⑩ Action-bandit agent's probability and expected reward per allowed action for this customer, with its confidence (section 10.2).

**Allowed actions:** code passes only actions allowed by the rules. Examples: no aggressive upsell above the churn threshold (H-06); excluded segments get no action. The LLM cannot add an action.

**Output (JSON, validated):** scores for the allowed actions, the top action, a confidence, and a justification citing evidence ids. A missing or invalid reply → **deterministic fallback**: rank actions by the ⑩ expected reward (if its confidence is ⚙ medium or high), then E-A1 incremental take rate at the finest reliable level, then the target's rule action (`action_original` in Target 1, `action_agent` in Target 2), then the B-10 priors; the decision is marked `partial_fallback`.

**Tools:** `customer_360`, `action_evidence`, `score_actions_llm`, `sample_action`, `explain_action`.

### 10. ④ LLM NBO agent (the offer recommender)
It does the job of a collaborative-filtering ranker:

| Ranker role | LLM NBO agent |
|---|---|
| Learns from 270 days of purchases which offers similar customers take | **Evidence tables** built in code from `sub_btl_3months` for the **full ~1M base** and from the campaign log |
| Scores the catalogue offers for a customer | Code builds the **candidate list**; the LLM gives a **score from 0 to 1 per candidate** with a short reason |

**Candidates** (code; the LLM can only score listed offer ids):
- **Target 1 (production rules):** recommendable offers (B-20b) of the chosen action's families and generosity (B-21, B-22), per validity (7 and 30 days), price-feasible under passes B-28 … B-30
- **Target 2 (challenger rules):** recommendable offers (B-20b) allowed by the approved C-version (C-v1: any family and generosity that serves the action, inside the relative ARPU band C-02), per validity
- at most ⚙25 per validity, pre-sorted by evidence

**Offer evidence** (per evidence level, with n and intervals):
- **E-O1:** per cell × offer: buyer share in 270 days (full base), take rate when offered (campaign log), incremental take rate vs control
- **E-O2:** co-purchase: offers bought by customers who bought the customer's past offers ("also bought", counts, full base)
- **E-O3:** the customer's own purchases (offers, prices, validity, frequency, recency)
- **E-O4:** price vs anchor for each candidate (B-26 anchors), generosity, GB / minutes per OMR
- **E-O5:** market index per offer (⑤)
- **E-O6:** the ⑪ Offer-pattern agent's score and rank per candidate, with "similar to your past purchases" offers and a cold-start flag (section 10.2)

**Output (JSON, validated):** a score per candidate. The **selection** is then done by the target's rules:
- **Target 1:** exactly as in code 3: pass 1 → 2 → 3 (B-28 … B-30), 1 offer per type per validity (B-31), the **LLM score is the `score` column** in the sort (score desc, then price desc / asc)
- **Target 2:** the approved C-version selection (C-v1: C-04)

Both write the **original reco structure** (B-32), so code 5 (⑦) runs unchanged.

**Fallback:** if the LLM fails, score the candidates by evidence (E-O1 incremental take rate, then the ⑪ pattern score, then E-O3 own history, then E-O1 buyer share).

**Tools:** `offer_candidates`, `offer_evidence`, `rank_offers_llm`, `offer_economics`, `explain_offers`.

**10.1 Consistency:** the LLM NBO agent receives the action chosen by the LLM NBA agent. In either target, if no candidate exists for that action (e.g. all prices out of reach), ★ sends it back **once** to the NBA agent with that reason; the second choice is final.

### 10.2 ML helper agents: ⑩ Action-bandit and ⑪ Offer-pattern
**Why:** each classic model has one strength the LLM does not have. The helpers bring that strength in as **evidence**; the LLM keeps what the models lack (control-group uplift, price vs ARPU, rules, market, explanations).

| | ⑩ Action-bandit agent (strength of Vowpal Wabbit) | ⑪ Offer-pattern agent (strength of LightFM) |
|---|---|---|
| Strength brought in | Fine-grained effect of **continuous** features per action (churn 0.48 vs 0.52, trend −35 %), learned from many logged rows, with correct off-policy weighting | **Hidden co-purchase patterns** across the whole base ("customers like you, who bought these offers, also take that one") |
| Engine (⚙) | `vowpalwabbit` `--cb_adf -q UA` (pip, runs in WSL), **logged mode**; option: per-action uplift model (LightGBM, IPS weights) | **Implicit-feedback ALS / truncated SVD** (`scipy` / `implicit`, pip, CPU); option `lightfm` only if it installs on the WSL Python (real customer ids, per validity) |
| Trained on | Week 0: segment-engine history (B-03 labels, B-09 reward, B-10 propensities), flagged `weak`. Week n: **our campaign log**: feature snapshot at push + action sent + reward r1 / r3 + logged propensity, accumulated with recency weighting ⚙ | `sub_btl_3months` of the **full ~1M base** (weighted by nb_sub) + offers taken in our campaigns; side features (nationality, age band, tenure band, value band, usage type; offer type, validity, generosity, price band) where the engine supports them |
| Retrained | Every Thursday (training stage), CPU | Every Thursday (training stage), CPU |
| Output (per customer) | For each **allowed** action: probability, expected reward, confidence (low / medium / high from the number of similar logged rows) | For each **candidate** offer: pattern score (0–1, percentile for this customer), rank, the 3 past purchases it is most similar to, cold-start flag |
| Goes to | ③ LLM NBA as **E-A5** | ④ LLM NBO as **E-O6** |
| Quality gate (weekly) | OPE (IPS / SNIPS / DR) of the bandit policy vs the evidence-only policy; if not better, confidence forced to `low` | Hold-out of the last ⚙4 weeks of purchases: precision@4 and recall@10 vs a popularity baseline per micro-segment; if not better, scores flagged `weak` |

**Rules for both helpers:**
- They never decide and never write to the outputs; only ③ / ④ decide (H-09 still applies).
- They score only allowed actions / listed candidates.
- The LLM sees each helper score **with its confidence**, and the system prompt says: *follow a high-confidence helper unless the control-group evidence or the rules say otherwise; when you go against it, say why*.
- Logged per decision: helper scores, helper top choice, LLM choice, `helper_agreement`.
- Weekly report: agreement rate, and uplift vs control when the LLM **followed** vs **overrode** each helper (IPS-weighted), per target. This shows whether the helpers help.
- Switches `use_action_bandit`, `use_offer_pattern` (default on). Off → evidence-only, as before.

**Tools:** `train_action_bandit`, `bandit_scores`, `evaluate_bandit_ope`, `train_offer_patterns`, `pattern_scores`, `evaluate_patterns_holdout`.

### 11. ⑥A Production rules and ARPU policy: Target 1 (codes 3–4, B-20 … B-32)
- **B-20** History = purchases D-270…D-1, catalogue offers only.
- **B-20b** **Recommendable offers (both targets)** = offers that are **in the current `btl_catalog`** **and** were **bought by at least one customer of the base in the last 270 days** (`sub_btl_3months`), as in code 3 (the LightFM item list is built from the filtered purchase history, line "item_id = list(interaction1.BTL_NAME.unique())"). A catalogue offer nobody bought is never recommended; an offer no longer in the catalogue is never recommended. Slots 5–6 (favourites, F-06) are the customer's **own** past catalogue purchases; forced bundles (F-08) must also be in the current catalogue. Switch `only_own_past_offers` (default **off**, as in the original code): when on, slots 1–4 are limited to offers the customer himself bought (note: this conflicts with upsell, because the ⑬ guard then leaves few offers above his current bundle).
- **B-21** **Target 1** constraint mapping (active):
  - MAINTAIN, UPSELL, RETAIN → [COMBO, DATA]
  - UPSELL VOICE, CROSS SELL VOICE → [DATA, COMBO]
  - UPSELL DATA, CROSS SELL DATA → [VOICE, COMBO]
  - generosity **L2** for all
  - The normal mapping (MAINTAIN L0, UPSELL L1, RETAIN L2, UPSELL VOICE [L0, L1], others [L0, L2]) is a switch; for the 4 cross / upsell actions, service[0] takes gen[0] and service[1] takes gen[1].
- **B-22** CROSS SELL VOICE has its own filter (the original typo is documented as fixed).
- **B-23** The original free part (code 4: L2 only, no VOICE, action not used) is kept as a switch for Target 1 (`target1_free_share`, default 0 %; set 50 % to reproduce the production 10k constraint / 10k free mix).
- **B-24** Candidate scores come only from the **LLM NBO agent** (or its evidence fallback). No other scoring model.
- **B-25** Offer revenue = Total_Sub_Amt_P + PG_Total_Amount_P; missing → 0.
- **B-26** Anchor = mean + std (ddof = 1) of M1–M3 / W1–W4. Data = 0.9 ×; voice = 0.1 ×.
- **B-27** Anchor per offer:
  - MAINTAIN / UPSELL / RETAIN: COMBO → total, DATA → data
  - UPSELL VOICE / CROSS SELL VOICE: COMBO → total, DATA → data
  - UPSELL DATA / CROSS SELL DATA: COMBO → total, VOICE → voice
  - free part (B-23 switch): COMBO → total, DATA → data
- **B-28** **Pass 1:** anchor < price < anchor + k (strict), with k = 3 for UPSELL, UPSELL VOICE, CROSS SELL DATA, CROSS SELL VOICE, UPSELL DATA and the free part; k = 2 for MAINTAIN, RETAIN. Best per type: score desc, then price **desc**.
- **B-29** **Pass 2:** the missing type, price > anchor, no cap; score desc, then price **asc**. Customers still with one type keep it.
- **B-30** **Pass 3:** no offer → both types, price > anchor, no cap; same sort.
- **B-31** 1 offer per type per validity → up to 4 recommended offers.
- **B-32** Outputs with the original structure: Target 1 → `recos_monthly_40k_constraint`, `recos_weekly_40k_constraint` (and `recos_*_tg_free` if the B-23 switch is on); Target 2 → `recos_monthly_challenger`, `recos_weekly_challenger` (access_method_val, item_id, rank, score, btl_name, type, generosity, service, validity, price, action, plus the anchors).

### 12. ⑦ Final output agent: code 5 (F-01 … F-19) → `input_flytxt`
**Inputs:** `results_scoring_nba`, `TG_40K_constraint_scored_nba`, `TG_40k_challenger_scored_nba`, the reco tables of both targets, `subscriptions_tg`, `usage_data_voice_payg_tg`, `btl_catalog`, `sub_btl_3months`, `BTL_script_english_arabic_ok`, `conflicting_offers_h`.

**Rules (in this exact order):**
- **F-01** Population = Target 1 + Target 2. Monthly recos = Target 1 monthly + Target 2 monthly; weekly recos = Target 1 weekly + Target 2 weekly (in code 5, the challenger tables take the place of the free tables). The code-5 rules below are identical for both targets.
- **F-02** Inside monthly and inside weekly, rank each customer's offers by **price, highest first**, ties by first occurrence (`rank(method="first", ascending=False)`).
- **F-03** Slots: **1** = monthly rank 1, **2** = monthly rank 2, **3** = weekly rank 1, **4** = weekly rank 2, **5** = favourite, **6** = second favourite or a forced bundle.
- **F-04** Drop customers with nothing in slots 1–4. **At most 6 offers.**
- **F-05** Code-5 ARPU (separate from B-26):
  - revenue = Total_Sub_Amt + PG_Total_Amount, **zero periods treated as missing**
  - anchor = **min(mean + std, max)**; data 0.9 ×, voice 0.1 ×
  - no non-zero period → no anchor
- **F-06** Favourites: the customer's 270-day catalogue purchases (one row per purchase row), sorted by price highest first → row 1 → slot 5, row 2 → slot 6.
- **F-07** No ARPU filter on favourites (switch, default off).
- **F-08** Forced bundle in **slot 6** only if slot 6 is empty **and** no listed popular offer is in any slot. First match in this order:

  | Anchor | Band | Forced offer |
  |---|---|---|
  | monthly | [0, 3) | CM_3R_8GB_240MIN |
  | monthly | [3, 4) | CM_4R_10GB_300MIN |
  | monthly | [4, 5) | CM_5R_14GB_360MIN |
  | monthly data | [0, 3) | DM_3R_10GB |
  | monthly data | [3, 4) | DM_4R_14GB |
  | monthly data | [4, 5) | DM_5R_18GB |

  - Popular lists: the 3 CM offers for the monthly bands; the 3 CM + 3 DM offers for the data bands.
  - Earlier versions are kept as config history.
- **F-09** Same (catalogue `type`, price) in two slots → clear the later slot (priority 1 → 6).
- **F-10** English / Arabic scripts per slot from `BTL_script_english_arabic_ok` (matched on destination_offer_name).
- **F-11** Same destination offer in two slots → clear the later slot.
- **F-12** Conflicts:
  - `conflicting_offers_h` rows with offer_type_a = VOICE are ignored
  - for each of the 15 slot pairs, if (a, b) or (b, a) is listed → **remove that row's Offer_A**
- **F-13** One row per MSISDN.
- **F-14** SEGMENT = the action chosen by the LLM NBA agent.
- **F-15** SEGMENT_BTL_TAKER = "BTL_Taker" if any catalogue purchase in 270 days, else "No_BTL_Taker".
- **F-16** SEGMENT_BTL_TAKER_UPSELL = "BTL_upsell" if the action is UPSELL, else "".
- **F-17** DATE = run date; START_DATE = run date + 1; END_DATE = ""; CHANNEL_1…6 = ""; flag_splio = 1.
- **F-18** OPPORTUNITY_NAME_i = "nbo_LFM_" + DESTINATION_OFFER_NAME_i (technical prefix kept for the campaign platform; configurable).
- **F-19** `input_flytxt`, **exactly these 38 columns in this order, lower-case:** date, start_date, end_date, msisdn, channel_1…channel_6, opportunity_name_1…6, destination_offer_name_1…6, flag_splio, offer_script_en_1…6, offer_script_ar_1…6, segment, segment_btl_taker, segment_btl_taker_upsell.

**Code-5 issues** (fixed by default; reproduced only with the bug flag for the parity test):
- **(a)** A slot cleared by F-11 / F-12 keeps its script and opportunity name. Fixed: clear them.
- **(b)** Empty slot → "nbo_LFM_nan". Fixed: "".
- **(c)** Favourites can repeat the same offer and ignore purchase count. Fixed: one row per offer, sorted by price then count.
- **(d)** Two different anchors (B-26 vs F-05): keep both; never merge.

**Pushed file:** **one `input_flytxt` with both targets (≈40k rows)**, exactly the 38 columns above. The target of each MSISDN is **not** added to the file (format unchanged); it lives in `panel_<T>` and the campaign log. Per-target copies (`input_flytxt_target1`, `input_flytxt_target2`) are written for the audit.

**Final rules report:** customers affected per F-rule (favourites added, forced per band, cleared by F-09 / F-11 / F-12, dropped by F-04), per target.

### 13. ⑥B Challenger rules agent: Target 2 (Scoring B)
**Persona:** RL + telecom marketing expert.

**Role:** owns the rules of **Target 2**. Every week it (1) applies the **approved** challenger version to Target 2 and (2) proposes and tests the next version.

**Inputs:** production rules, campaign log and rewards per target, price elasticity (take rate vs price / anchor), nationality / age / tenure preferences, fatigue, uplift vs control, market index, OPE and simulation tools.

**C-v1: the first challenger version** (written in `challenger_rules_v1.yaml`, shown in the Phase 1 plan, **approved by me before the first push**; every value ⚙):
- **C-01 Candidates:** any recommendable offer (B-20b: in the catalogue and bought in the base in 270 days; COMBO, DATA, VOICE; any generosity) whose service fits the chosen action (e.g. DATA or COMBO for UPSELL DATA), not only the B-21 families. The LLM NBO agent chooses among them.
- **C-02 Relative ARPU band** instead of +2 / +3 OMR: price between ⚙1.0 × and ⚙1.5 × the customer's anchor (never below the anchor, so no downgrade; minimum step ⚙+0.5 OMR above the anchor for UPSELL actions; MAINTAIN and RETAIN may sit at the anchor). The ⑬ revenue guard (section 14.1) checks it again. Customers with no spend → the band of their micro-segment's median anchor.
- **C-03 Anchor from real usage:** same mean + std as B-26, but the data / voice split follows the customer's own data vs voice spend share (instead of 90 % / 10 %); no history → 90 / 10.
- **C-04 Selection:** per validity, the 2 best offers by **LLM score × price** (expected value), at most 1 per type; if fewer than 2 fit C-02, widen the band once **upwards only** to ⚙1.0 × – 2.0 ×; still none → the Target 1 passes B-28 … B-30 as fallback (logged).
- **C-05 Do-not-disturb:** customers who ignored ⚙3 offers in a row, or with fatigue flag, get **no contact** this week (logged as a "no contact" decision, measured against control like every Target 2 customer).
- **Unchanged in Target 2:** data rules (D), features (B-01 … B-12), segmentation (S), guardrails (H), code-5 rules (F-01 … F-19) and the `input_flytxt` format.

**Proposal ideas for the next versions** (each a versioned rule with hypothesis, expected effect and risk; evaluate, do not impose):
- relative price step instead of +2 / +3; step per nationality, age or micro-segment from elasticity; caps in passes 2 / 3
- median + λ × std or percentile anchors; data / voice split from real usage instead of 90 / 10
- generosity per customer or the normal L0 / L1 / L2 mapping
- ranking by expected incremental value
- weekly vs monthly from purchase cadence
- no contact for do-not-disturb customers; fatigue caps
- code-5 changes: favourites by frequency and recency, forced-bundle bands per segment, slot-6 policy, slot order by value

**Fixed (cannot change):** the output format (F-19), the guardrails (H), the target sizes.

**Process:**
1. Proposals as `challenger_rules_vN.yaml`.
2. Evaluate with OPE (95 % CI) on the logs **and** a 6-week closed-loop simulation.
3. Keep only proposals that beat the production rules on **both** incremental revenue and take-up (or revenue with take-up not lower by more than ⚙1 point), with the CI lower bound > 0.
4. Write a decision memo.
5. **Human approval.** An approved version replaces the current one **on Target 2 only**, so it is always measured against Target 1 (production rules) and control.

A Target 2 version that wins for ≥ ⚙4 weeks can become the **new production rules of Target 1** (section 19.4); ⑥B then continues with the next challenger.

### 14. Guardrails (H-xx; hard; both targets)
- **H-01** Eligibility and exclusions (D-07, B-02). The control group is never contacted and never in `input_flytxt`.
- **H-02** Catalogue offers only; price > 0.
- **H-03** ≤ 2 recommended offers per validity, ≤ 4 recommended, ≤ **6 in `input_flytxt`**.
- **H-04** No conflicting pair left (F-12).
- **H-05** Contact cap and cool-down (section 19.3); no customer in two campaigns the same week (checked against `targeting_auto` / other campaign lists).
- **H-06** No aggressive upsell above the churn threshold ⚙.
- **H-07** Exploration ≤ ⚙10 % of contacts per target.
- **H-08** **No push without ⑨ audit PASS and human approval.**
- **H-09** Every action and offer comes from the allowed lists; LLM text never reaches the file except through validated ids.

**Soft rules** (rule action, evidence level): the LLM may depart from them **only by its scores**, with a logged justification. The target's business rules (B for Target 1, C for Target 2, F for both) are never overridden by the LLM.

### 14.1 ⑬ Revenue guard agent: ARPU respect and no cannibalisation (both targets)
**Goal:** no recommendation, in Target 1 or Target 2, may lead a customer to pay less than he pays today for the same need. The guard applies **the same reference to both targets**, whatever rules selected the offer.

**Customer revenue reference** (code, per customer, written to the ledger):
- `anchor_ref(type, validity)`: the B-26 anchor for that offer type (total / data / voice) and validity (monthly M1–M3, weekly W1–W4), from `subscriptions_tg` + `usage_data_voice_payg_tg`. The same value for both targets (C-03 may change the split for selection, not for the guard).
- `current_bundle(service, validity)`: the catalogue offer the customer buys most (270 days, `sub_btl_3months`: highest nb_sub, then highest price) for each service (DATA, VOICE, COMBO) and validity, with its price and allowance (GB, minutes).
- `payg_spend(service)`: last month's pay-as-you-go spend for data and for voice.
- `guard_ref(offer)` = max(`anchor_ref` of the offer's type and validity, price of `current_bundle` of the same service and validity).

**Pre-push checks** (stage P6b after selection, and again P7b on the final 6 slots after code 5):
| Id | Check | Rule |
|---|---|---|
| CAN-01 | **Below ARPU / downgrade** | Every recommended offer (slots 1–4) price ≥ `guard_ref(offer)`; for UPSELL actions price > `guard_ref`. |
| CAN-02 | **Cheaper rate for the same need** | No offer with a lower price and an equal or bigger allowance than the customer's `current_bundle` of the same service and validity. |
| CAN-03 | **Weekly replaces monthly** | For a customer who buys a monthly bundle of a service, a weekly offer of that service × ⚙4.3 must be ≥ his monthly bundle price. |
| CAN-04 | **COMBO replaces separate bundles** | A COMBO offer's price ≥ the sum of the customer's current DATA and VOICE bundles of the same validity it would replace. |
| CAN-05 | **Bundle replaces pay-as-you-go** | Offer price ≥ ⚙1.0 × last month's `payg_spend` of the service it covers (configurable; the threshold is in the plan as an open item). |
| CAN-06 | **Forced bundles and favourites** | A forced bundle (F-08) is placed only if its price ≥ the customer's monthly `guard_ref`; favourites (F-06) are the customer's own purchases, so they pass by definition, but a favourite cheaper than the customer's current bundle of the same service is not used. |
| CAN-07 | **Expected revenue** | For each customer: Σ over slots of take-probability (E-O1 take rate at the evidence level) × (price − displaced current spend of the same service) ≥ ⚙0. |

**Repair** (code, in this order; every repair logged with the check id):
1. Replace the failing offer with the **next-ranked candidate** (LLM score order) that passes every check and the target's rules.
2. If none passes, **clear the slot**.
3. If slots 1–4 are all empty, the customer is dropped (F-04) and logged.

Repairs are **reported per target** on the approval screen. In Target 1 this can differ from the original production output; the mode is configurable (`guard_mode_target1` = `repair` (default) or `report`). **Any violation left after repair = hard FAIL**, blocking the push through ⑨ (A-27).

**After the campaign (training stage T1, at T+7), per target and per micro-segment, vs control:**
- each taker's revenue in the reward window vs his own previous weeks, and vs control customers of the same stratum
- `cannibalised_revenue` = revenue lost by takers who bought the recommended offer instead of a more expensive usual purchase
- **net incremental revenue** = (target revenue change − control revenue change), with a 95 % CI
- if a (offer × micro-segment) or (offer × segment) pair shows negative net incremental revenue (CI upper bound < 0) for ⚙2 weeks, it goes to the **block list** `cannibalising_pairs_vN`; the next week's candidates (both targets) exclude it, and ⑥B and ★ see why
- results go to the evidence tables, the weekly report and the ledger

**Also guarded:** every ⑥B proposal is simulated through ⑬ before evaluation; a proposal that creates CAN violations is rejected.

**Tools:** `revenue_reference`, `check_cannibalisation`, `repair_offers`, `measure_cannibalisation`, `update_block_list`, `revenue_guard_report`.

### 15. ★ Chief recommender (orchestrator) and ⑤ Market intelligence
**Shared system prompt** for ★, the LLM NBA agent and the LLM NBO agent (versioned in `nbx/fsdr/agents/prompts/`; each agent adds its own task section):

> You are part of FSDR's prepaid recommender. You are an expert in **reinforcement learning** (contextual bandits, exploration vs exploitation, off-policy learning, uncertainty, delayed rewards) and **telecom marketing** (lifecycle, retention vs growth, incremental uplift, price anchoring and elasticity, cannibalisation, fatigue, nationality, age and tenure preferences).
>
> **Goal:** for each customer, the action and offers that maximise **expected incremental value** (value created because of the contact), within every business rule.
>
> **Method:**
> 1. Read the customer profile and the evidence.
> 2. Compare the allowed options on incremental take rate, revenue, uncertainty (small n = less trust) and risk.
> 3. Score every allowed option from 0 to 1.
> 4. Return JSON only, with a justification citing evidence ids.
>
> **Principles:** never invent options, offers or numbers; only allowed actions and listed offer ids; retention before upsell when at risk; respect fatigue; the price step must be affordable vs the customer's spending; avoid offers clearly worse than comparable competitor plans; when a cell's evidence is weak, rely on the parent level and say so; when two options are close and evidence is weak, prefer the one that is less tested (it teaches us more).

**★ Orchestrator:**
- Runs LLM NBA → LLM NBO → ⑤ market → ⑥ rules → ⑦ final, with a step cap ⚙8 per customer and the consistency loop of 10.1.
- Validates every JSON (schema, allowed ids). Any failure → fallback, and the decision is marked `partial_fallback`.

**Two LLM levels:**
1. **Level 1, weekly policy (cell level), part of training:**
   - The LLM reads the evidence per cell (nationality × age × tenure, and segment) and writes a **policy per cell**: preferred actions with scores, offer families, price-step guidance, weekly vs monthly preference, message angle, and **what to test this week**.
   - About 120 cells × 12 segments: a few hundred calls.
   - Validated by ⑥ (no forbidden action or family) and logged with a version.
2. **Level 2, per customer, part of scoring:**
   - The LLM NBA and NBO agents score the options for **each of the 40k** (Target 1 and Target 2), using the cell policy plus the customer's own evidence.
   - If the weekly budget is exceeded (section 19.5), the remaining customers get their **cell policy applied deterministically** (still an LLM decision, at cell level). The share at each level is reported per target.

**Exploration and learning:**
- **ε = ⚙0.1** per target: the sampled action may be a non-top allowed action (section 9); the propensity is logged.
- Every decision logs the scores, the probabilities used, the propensity, and the evidence, policy and prompt versions.
- Next week's evidence tables include the results (rewards vs control). This is how the LLM agents learn.

**Determinism and cost:**
- temperature 0, fixed seed, JSON-schema output
- a cache keyed on (feature hash, evidence version, policy version, prompt version, model)
- batched generation with vLLM

**⑤ Market:** offline snapshot only; index = our value / best comparable competitor value (same validity and service); < ⚙0.9 → flag and suggest an eligible alternative; the LLM NBO agent sees the index (E-O5); market can never add an offer the rules reject.

### 16. ⑧ Strategy & monitor
- **Before scoring:** weekly playbook per cohort (segment × nationality × age × tenure), exploration budget, learning questions.
- **After feedback:** weekly report per target, nationality, age band, tenure band, micro-segment and **vs control**:
  - take rate, incremental take rate
  - incremental revenue, ARPU change, offer price paid
  - cannibalisation, 4-week retention
  - drift (PSI), evidence-level coverage, fallback share, the policy / prompt / rule versions in use

### 17. ⑨ Flow & rules auditor
**Runs:** after training (T-checks), after scoring and before approval (all A-checks), and after feedback (R-checks). Deterministic; the LLM only writes the narrative.

**Output:** `audit_report_<run_id>` (JSON + HTML in the UI). Each check has: id, status PASS / FAIL / WARN / NOT_APPLICABLE (with the reason), evidence (counts, examples, file and rule ids), severity. **Any hard FAIL blocks approval and push (H-08).**

**Flow checks**
- **A-01** Every step of the run happened in order (data → segmentation → features → evidence → policy → NBA → NBO → market → rules → final → audit), with its inputs present and outputs written. Run id and evidence, policy, prompt, model and rule versions are recorded.
- **A-02** **Row reconciliation:**
  - full base → eligible → panel → Target 1 (≈20k) / Target 2 (≈20k) / control → scored
  - → recos (per target, per validity) → `input_flytxt` rows per target
  - every drop is explained by a rule id (e.g. F-04: no slot 1–4)
- **A-03** Weekly data completeness per pipeline (section 17.9) and window checks (D-xx).

**Rule checks:** for every rule in the registries, decide if it is **applicable** to this run and target, then verify it on the data. Examples:
- **A-10** B-26 / F-05: recompute anchors for a ⚙1 % random sample and compare.
- **A-11** B-28 … B-30: every recommended offer satisfies its pass condition and sort order (using the logged LLM scores).
- **A-12b** Every recommended offer is in the current `btl_catalog` and in the 270-day purchased set (B-20b); every forced bundle is in the catalogue; favourites are the customer's own purchases.
- **A-12** Target 1: B-21 families and generosity per action, B-28 … B-30 passes. Target 2: C-01 service fit, C-02 band, C-04 selection of the approved version.
- **A-13** F-02 / F-03: slot 1 price ≥ slot 2 price; slots 1–2 are 30-day offers, slots 3–4 are 7-day offers.
- **A-14** F-08: forced bundles only where slot 6 was empty and the anchor is inside the band.
- **A-15** F-09 / F-11 / F-12: no duplicate (type, price), no repeated offer, no conflicting pair left.
- **A-16** H-01 … H-09 on every row.
- **A-17** Target 1 uses only the production rules; Target 2 uses only an **approved** challenger version (id and approver recorded).
- **A-26** **Helpers:** ⑩ and ⑪ trained this run (or switched off, logged); quality gates computed; no helper output written to a decision column; helper scores and `helper_agreement` logged for every decision.
- **A-19** **Two actions:** every line of the base has `action_original` (with its source) and `action_agent` (with its version); the agent dictionary passed ⑥; Target 1 uses `action_original` and Target 2 uses `action_agent` as the rule action; `action_agreement` logged for every decision.
- **A-18** **LLM decisions:** every action and offer id is in its allowed list; every decision has scores, probabilities, propensity and versions; fallback share ≤ ⚙5 % (WARN above, FAIL above ⚙20 %); exploration ≤ H-07.
- **Coverage:** list rules checked, passed, failed and not applicable, with the reasons.

**Output-format checks (`input_flytxt` must equal the code-5 format)**
- **A-20** Exactly the 38 columns, names and order of F-19, lower-case. No extra column.
- **A-28** **File format:** every input was a `.csv.gz` read with the contract types (IDs as text); every output is a `.csv.gz` in the FSDR CSV standard with its schema and manifest entry; `input_flytxt.csv.gz` read back equals the approved table value by value (no `nan`, no scientific notation, empty fields for empty slots, UTF-8 Arabic intact).
- **A-21** One row per msisdn; msisdn = 8-digit text; no control-group msisdn; no msisdn outside the panel.
- **A-22** date = run date, start_date = run date + 1, end_date = "", channel_1…6 = "", flag_splio = 1.
- **A-23** Every destination offer is in the catalogue. Each non-empty slot has its opportunity name ("nbo_LFM_" + offer) and EN and AR scripts. Each empty slot has empty script and opportunity fields (fixed mode).
- **A-24** segment ∈ the 7 actions; segment_btl_taker ∈ {BTL_Taker, No_BTL_Taker}; segment_btl_taker_upsell ∈ {"BTL_upsell", ""} and consistent with segment.
- **A-25** **Code-5 parity:** ⑦'s rows (before the ⑬ P7b repair) equal a direct replay of code 5 on the same reco tables and inputs (bug flag as configured), column by column; every difference after P7b is explained by a CAN repair id.

**Target integrity**
- **A-30** No msisdn in two targets or in a target and control. Sizes within ⚙±2 % of 20k / 20k.
- **A-31** Stratification balance: the distribution of the `Final` strata (nationality × revenue × voice × data × plan) **and of the age and tenure bands** differs between Target 1, Target 2 and control by at most ⚙2 points per stratum (chi-square reported).
- **A-32** Control group size and balance (section 19.2).

**Logging and learning**
- **A-27** **Revenue guard:** ⑬ ran at P6b and P7b for both targets; zero CAN violations left in `input_flytxt`; no offer from the block list; repairs counted per target and check id.
- **A-41** **Ledger:** one row per MSISDN for this run (no duplicates; count = eligible base + excluded lines); every Target 1 / Target 2 / control customer has groups A–H filled; each `input_flytxt` row equals the ledger's slot columns; at T+7, group I filled for 100 % of last week's 40k + control (R-01).
- **A-40** Every pushed row has a campaign-log record: target, send date, action, offers per slot, scores, propensity, versions, and a **feature snapshot**.

**Training checks**
- **T-1** The snapshot join covers ≥ ⚙98 % of the previous week's logged rows.
- **T-2** No feature or evidence uses data measured after START_DATE of the decisions it is evaluated on.
- **T-3** Evidence tables built on the full base: row count = eligible base; every cell has n and an interval; the level-1 policies pass ⑥.

**After feedback**
- **R-01** Rewards computed for 100 % of the logged decisions, with the window of D-09.
- **R-02** Control-group purchases measured.
- **R-03** No reward uses purchases from before START_DATE.

**Tools:** `audit_flow`, `audit_rules`, `audit_output`, `audit_targets`, `audit_llm`, `audit_logging`, `audit_training`, `audit_feedback`, `audit_report`.

### 17.9 Two separate pipelines: TRAINING and PRODUCTION SCORING
Build **two pipelines** with separate code, CLI commands, run ids and output folders. They are linked only by the **training bundle**.

| | TRAINING pipeline | PRODUCTION SCORING pipeline |
|---|---|---|
| Code | `nbx/fsdr/pipelines/training.py` | `nbx/fsdr/pipelines/scoring.py` |
| CLI | `fsdr-train --week0` / `fsdr-train --weekly` | `fsdr-score --bundle <bundle_id>` |
| Population | **Full base (~1M)** + campaign log + control | **40k** (Target 1 + Target 2) + control (measured, never contacted) |
| Purpose | Learn: features, segments, two actions, evidence, ⑩ / ⑪ models, LLM cell policies, rewards | Decide: action and offers per customer → `input_flytxt` → approval → push |
| LLM use | ②b agent dictionary (~70 calls), ★ cell policies (a few hundred calls) | ③ NBA + ④ NBO per customer (~80k calls) |
| Writes | `models/bundle_<run_date>/` (read-only after validation) | `data/out/<run_date>/<run_id>/`, `nba_campaign_log`, snapshots |
| Never does | Never writes `input_flytxt`, never contacts anyone | **Never trains, never rebuilds evidence or policies**; uses only the bundle's versions |
| Audit | T-1 … T-3 (+ R-checks in week n) → bundle status `validated` | A-checks → blocks the push |
| Time (1M, estimate) | ≈ 1–1.5 h | ≈ 3 h (mostly LLM) |

**Training bundle** `models/bundle_<run_date>/` (manifest with run id, input-file checksums and every version):
- `base_features_<run_date>` and `scored_base` (~1M: features, segment, micro-segment, `action_original`, `action_agent`, cell-policy top action, eligibility)
- `evidence_<version>` (E-A0 … E-A4, E-O1 … E-O5)
- `dict_segm_actions_agent_<version>`
- ⑩ action-bandit model and ⑪ offer-pattern model + their quality reports
- `policy_<version>` (cell policies)
- rewards of the previous push (`nba_rewards`, week n) and the evaluation report
- the T-check audit result

Scoring **refuses** a bundle that is not `validated`, or whose run date is not this week's (except `--allow-previous-bundle`, logged, used only to re-score after a failure with last week's approved bundle). The 40k features are **read from the bundle** (same rows as the full base), never recomputed, so training and scoring see identical numbers.

**Thursday order:** `fsdr-train` → (bundle validated) → `fsdr-score` → audit → approval → push. `fsdr-run-week` runs both in order and stops before approval.

**Which table is used where** (✓ = read directly; *bundle* = used through the training bundle; — = not used)
| Table | Training week 0 | Training week n | Production scoring | Used for |
|---|---|---|---|---|
| `base_prepaid` | ✓ | ✓ | *bundle* | Population, eligibility, tenure (D-07) |
| `consumer_aact_account` | ✓ | ✓ | *bundle* | Account → party |
| `access_method_id_auto` | ✓ | — | — | Targeting id → MSISDN for week-0 history |
| `party_auto` | ✓ | — | — | Nationality / age of the week-0 history |
| `party_scoring` | ✓ | ✓ | *bundle* | Nationality / age of this week's base |
| `customer_profile` (msisdn, age, tenure) | ✓ | ✓ | *bundle* | Age and tenure bands, micro-segments |
| `location_auto`, `DEVICEBEHAVIORSEGMENTATION.dv_segment` | ✓ | ✓ (when refreshed) | *bundle* | Context (region, device) |
| `subs_amt_auto`, `activities_*_auto` (7), `paygo_activity_auto` | ✓ | — | — | Week-0 features of the **history** (anchored on T) for the week-0 evidence and ⑩ |
| `subs_amt_score`, `activities_*_score` (7) | ✓ | ✓ | *bundle* | **This week's** features of the base (anchored on D) |
| 11 model-score tables | ✓ | ✓ | *bundle* | Features, segments |
| `subscriptions_tg` | ✓ | ✓ | ✓ | Training: value band, revenue reward r4. Scoring: **ARPU anchors** (B-26, F-05, C-03) |
| `usage_data_voice_payg_tg` | ✓ | ✓ | ✓ | Training: usage type, r4. Scoring: **ARPU anchors**, data / voice split |
| `targeting_dataiku_auto` | ✓ | optional | — | Week-0 action labels (segment → action) for evidence and ⑩; `action_original` source |
| `targeting_auto` | ✓ | ✓ | ✓ | Training: offered-vs-taken evidence, other campaigns. Scoring: **H-05 contact conflicts**, clean control |
| `subs_btl_auto` (dated purchases) | ✓ (D-7…D-1) | ✓ (reward window D-09) | — | **Rewards** (both targets and control) |
| `sub_btl_3months` (270 days) | ✓ | ✓ | ✓ | Training: E-O1 / E-O2, ⑪. Scoring: E-O3 own purchases, **favourites F-06**, **BTL taker F-15** |
| `btl_catalog` | ✓ | ✓ | ✓ | Training: evidence per offer, ⑪. Scoring: candidates, slots, F-09 |
| `BTL_script_english_arabic_ok` | — | — | ✓ | F-10 scripts |
| `conflicting_offers_h` | — | — | ✓ | F-12 conflicts |
| `sample_cg_tg_new_7oct2026` (optional) | — | — | ✓ (panel build) | Target 1 / Target 2 / control groups |
| `nba_campaign_log` (internal) | — | ✓ read | ✓ write (at push) | Rewards, evidence, ⑩ training / decisions sent |
| `feature_snapshot_<run_id>` (internal) | — | ✓ read (previous) | ✓ write | Features at push time for learning |
| `panel_<T>` (internal) | — | ✓ read | ✓ read / write | Target and control membership |
| Training bundle (internal) | write | write | ✓ read | Everything learned this week |

**Week 0 needs both `_auto` and `_score` tables:** `_auto` to learn from the history (the segment engine's past campaigns), `_score` for this week's features of the base that the 40k are scored on. From week 1, only `_score` tables (plus rewards and snapshots) are needed.

### 18. TRAINING pipeline (full base, ~1M lines) → training bundle
**18.0 Training stages**
| Stage | Week 0 | Week n | Who |
|---|---|---|---|
| T0 | Load + validate (`_auto` + `_score` + history) | Load + validate (`_score` + purchases) | ① |
| T1 | — | **Feedback:** rewards of the previous push vs control (18.2 step 1) | code |
| T2 | Features (history from `_auto`; this week from `_score`), segments, micro-segments for ~1M | Features, segments, micro-segments for ~1M (`_score`) | ② code |
| T3 | `action_original` + agent dictionary → `action_agent` for ~1M | same | ②b code + LLM |
| T4 | Evidence tables (week-0 history) | Evidence tables (our campaigns vs control) | code |
| T5 | Train ⑪ (purchases) and ⑩ (history, `weak`) | Retrain ⑩ (campaign log) and ⑪ (purchases) + quality gates | code (CPU) |
| T6 | ★ cell policies | ★ cell policies update | LLM |
| T7 | — | Evaluate (targets vs control, OPE of new policy), ⑥B proposals | code + LLM |
| T8 | `scored_base` overview, ⑫ ledger groups A–E for ~1M, ⑨ T-checks → bundle `validated` | same (+ ⑫ rewards into last week's rows) | code |

**18.1 Week 0 (first run): `_auto` tables**
1. ① Validate the `_auto` tables, `targeting_dataiku_auto`, `targeting_auto`, `access_method_id_auto`, `subs_btl_auto` (D-7…D-1), `sub_btl_3months`, the scores, the catalogue, `customer_profile`.
2. ② Segments (S-rules) and micro-segments for the **full base**; agreement report vs the segment engine (B-03).
3. Features for the full base (section 6) → `base_features_<run_date>`.
4. **Evidence tables (full base):**
   - E-A1 / E-A2 from `targeting_dataiku_auto` (segment → action via B-03) + `subs_btl_auto` purchases D-7…D-1 (B-09), compared with non-targeted lines of the same cell as a pseudo-control (flagged "week-0, not randomised")
   - E-A0 priors (B-10)
   - E-O1 / E-O2 / E-O3 from `sub_btl_3months` for all ~1M lines; E-O1 offered-vs-taken from `targeting_auto` + `subs_btl_auto`
5. **Helpers:** ⑪ Offer-pattern trained on the 270-day purchases of the full base; ⑩ Action-bandit trained on the segment-engine history (flagged `weak` until our own logs exist).
6. ★ **Level-1 LLM policy update** → the first cell policies.
7. Register the evidence, helper-model and policy versions; ⑨ runs T-3.

**18.2 Week n ≥ 1: `_score` tables + snapshots + rewards (no `_auto` needed)**
1. **Feedback:**
   - Join `subs_btl_auto` (dated purchases) to `nba_campaign_log` for the previous push.
   - The window is D-09 (START_DATE → day before this run).
   - Rewards:
     - (r1) a recommended offer bought (any slot)
     - (r2) an offer of the recommended service
     - (r3) any catalogue BTL
     - (r4) net revenue change (from the new `_score` / `_tg` tables)
   - The same measures for **control**.
   - ⑬ measures cannibalisation per target and micro-segment vs control and updates the block list (section 14.1).
   - Monthly offers are also tracked over 30 days (lagged).
2. **Learning data** = `feature_snapshot_<previous run>` (features **at push time**, before any purchase) + rewards + logged scores and propensity + the action and offers actually sent. Accumulate the weeks; recency weighting ⚙.
3. **Update (full base):**
   - recompute segments, micro-segments and features on the **full ~1M base** from the `_score` tables and `customer_profile`
   - rebuild the evidence tables: E-A1 / E-A2 now from **our randomised campaigns vs control** (week-0 pseudo-control evidence is down-weighted ⚙ and then dropped after ⚙4 weeks); E-O* from the refreshed `sub_btl_3months` and the campaign log
   - retrain the helpers: ⑩ on the accumulated campaign log (snapshots + action sent + rewards + logged propensities); ⑪ on the refreshed `sub_btl_3months` + offers taken; run their quality gates
   - ★ **level-1 policy update:** the LLM reads last week's results per cell and writes the new cell policies, with what changed and why; validated by ⑥
4. **Evaluate:** Target 1 vs Target 2 vs control (incremental take rate and revenue with CIs); off-policy estimates (IPS / SNIPS / DR from the logged propensities) of the new policy before it is used. A new policy or prompt version is used only if its estimate is not worse than the current one (CI); the previous evidence, policy and prompt versions are kept for rollback.
5. ⑥B: evaluate pending challenger proposals; write the memo.
6. ⑨ audits the training (T-1, T-2, T-3).

**18.3 Weekly data required:** see the table-usage matrix in section 17.9 (columns "Training week n" and "Production scoring"). ① `weekly_completeness` checks it before each pipeline starts.

### 19. PRODUCTION SCORING pipeline (40k) → `input_flytxt`
**19.1 Every Thursday, after the training bundle is `validated`**
1. ① Load and validate the scoring tables (section 17.9, column "Production scoring") and **load the training bundle** (refuse it if not validated or not this week's).
2. **Panel** (19.2): build (week 0) or update (week n), using the bundle's `scored_base` for eligibility and replacements.
3. Read the 40k + control features, segments, two actions and micro-segments **from the bundle**; compute the ARPU anchors (B-26, F-05, C-03) from `subscriptions_tg` and `usage_data_voice_payg_tg`.
4. Record the bundle id and all its versions in the scoring run manifest.
5. **Score the 40k** with ★ (level 2), using the bundle's evidence, ⑩ / ⑪ models and cell policies:
   - **Target 1 (20k):** ③ LLM NBA → ④ LLM NBO on production candidates (B-21) → ⑤ → ⑥A ARPU anchors and passes → `recos_*_40k_constraint`
   - **Target 2 (20k):** ③ LLM NBA → ④ LLM NBO on challenger candidates (C-01) → ⑤ → ⑥B approved version (C-02 … C-05) → `recos_*_challenger`
   - `results_scoring_nba` + code-2 split (B-12)
6. **Save `feature_snapshot_<run_id>`** for every panel customer (targets and control).
7. ⑦ Build `input_flytxt` (one file, 38 columns) + per-target copies + the final rules report.
8. ⑨ **Audit** (all A-checks). FAIL → stop, show the report, nothing pushed.
9. **Approval screen:**
   - counts per target
   - action and offer mix per target
   - price changes vs anchors
   - fallback share and evidence levels
   - final rules report
   - audit summary
   - sample rows with EN / AR scripts and the LLM justifications

   A named person approves.
10. **Push:** export the approved `input_flytxt` (CSV) through the approved flow (SFTP → data platform → campaign platform). Write every pushed row to `nba_campaign_log` (target, START_DATE, action, slots, scores, propensity, versions, snapshot id).
11. ⑧ Weekly report (week n).

**19.1a Run order: one weekly wave, executed in stages (not segment by segment over the week)**

All 40k are scored and pushed **in one wave** with **one START_DATE**. Reasons: the same reward window for everyone (D-09), a fair comparison with the control group and between Target 1 and Target 2, one `input_flytxt`, one audit, one approval. Sending segment after segment on different days would give each segment a different window and market week, and the results could not be compared. (If the campaign team ever needs staggered sends, `start_date_by_segment` exists in config, default off; each START_DATE then has its own reward window and its own control comparison.)

Inside the wave, the run goes **stage by stage for all customers** (not one customer through the whole chain, then the next). Each stage writes a checkpoint; a failed run restarts at the failed stage.

The training stages T0 … T8 are in section 18.0. The **scoring stages**:

| Stage | What | Who | Population | How |
|---|---|---|---|---|
| P0 | Load scoring tables + training bundle | ① code | scoring tables | refuse an unvalidated bundle |
| P1 | Panel build / update | ⑧ code | 40k + control | stratified random |
| P2 | Features from the bundle + ARPU anchors | code | 40k | vectorised |
| P3 | Action: ⑩ bandit scores, then LLM NBA | ⑩ code + ③ LLM | 40k | **batched by segment** |
| P4 | Candidates + ⑪ pattern scores | ④ code + ⑪ code | 40k | per target rules |
| P5 | Offer scores: LLM NBO | ④ LLM | 40k | **batched by segment** |
| P6 | Selection | ⑥A (Target 1) / ⑥B (Target 2) code | 40k | vectorised |
| P6b | Revenue guard on the recommended offers (CAN-01 … CAN-07) + repair | ⑬ code | 40k | vectorised |
| P7 | `input_flytxt` | ⑦ code | 40k | code-5 rules |
| P7b | Revenue guard on the final 6 slots (forced bundles, favourites) | ⑬ code | 40k | vectorised |
| P8 | Feature snapshots | code | 40k + control | written before push |
| P9 | Audit | ⑨ code | whole run | blocks on failure |
| P10 | Approval → push → campaign log → ⑫ ledger push columns | human + code | 40k | one file |

**Batched by segment** means: P3 and P5 send the customers to vLLM grouped by segment (and nationality), ⚙64 in parallel. Customers in the same group share the same system prompt, segment evidence and cell policy at the start of the prompt, so vLLM's prefix cache reuses them (faster), and the progress and LLM time are reported per segment.

**Priority order** for P3 and P5 (it only matters if the weekly LLM budget runs out; the rest then get their cell policy applied by code):
1. Retention_RGS_At_Risk, Retention_High / Med_Probability_To_Reconnect
2. Retention_RGS_Fragile, Onboarding
3. Data-only and voice-only segments
4. DualPlay and Retention_Low_Probability_To_Reconnect

Target 1 and Target 2 are interleaved inside each group, so both targets get the same coverage.

**19.2 Panel and targets (40k contacted + control)**
- **Contacted:** **40,000** = **Target 1: 20,000 (production rules)** + **Target 2: 20,000 (challenger rules)**. Sizes in config.
- **Control group:** **not** inside the 40k. A separate control of ⚙10,000 (or the sample's `cg` group), never contacted, used for the incremental results of both targets.
- **Assignment:**
  - stratified random on the `Final` strata (nationality group × revenue band × voice group × data group × plan), balanced on age and tenure bands
  - eligible base only, minus Nursering, excluded plans, customers in other campaigns this week and recent contacts
  - never selected by score
  - or use the provided `sample_cg_tg_new_7oct2026` groups: cstr → Target 1, free → Target 2, cg → control

**19.3 Week n panel update**
- Same customers in the same target for ⚙4–6 weeks; control stays control.
- Took an offer → cool-down ⚙7 days. Ignored ⚙3 offers in a row → fatigue pause.
- Churned or ineligible → replaced by a random customer from the **same stratum and target**.
- New panel at the end of the test period.

**19.4 Promotion, failure, operations**
- **Challenger rules → production rules:** ≥ ⚙4 weeks live on Target 2, incremental revenue and take-up both better than Target 1 (CI lower bound > 0), no guardrail breach, approval. Rollback always available.
- **Policy / prompt / LLM model changes:** only after OPE on the logs (18.2 step 4) and approval; previous versions kept.
- **Failure:** nothing pushed; the previous approved evidence, policy, prompt and rules stay; the audit report explains why.
- **Scheduler:** Windows Task Scheduler (calling WSL) / cron runs steps 1–8 on Thursday and **stops before approval**.
- **Run id:** links data, evidence, policies, snapshots, rule versions, outputs, audit.
- **Backups:** daily backup of the registry, campaign log, snapshots and outputs.
- **Logs:** forwarded to the SOC collectors.

### 19.5 Scale: ~1 million lines on one laptop
The base is about **1,000,000 lines** (design and test so that 2,000,000 also works).
- **Storage format:**
  - accept CSV / JSON, but convert each table on load to **Parquet** in `data/cache/<run_date>/`
  - read only the needed columns; explicit dtypes (float32, int32, categoricals)
  - **polars or duckdb (embedded, pip-only)** for joins and aggregations
- **No per-customer Python loops on the base:** the original code filters the whole table for each MSISDN (`df_[df_.access_method_val==var]`); that is impossible at 1M. Features, segments, evidence tables, candidate lists, the price passes and the code-5 slot logic are all vectorised. The bug-reproduction mode may run row by row **on the 40k only**.
- **Memory:** stay under ⚙70 % of RAM; process the 14 activity tables in chunks of ⚙250k lines; a memory and time report per step goes in the run log.
- **GPU use:** the GPU runs only the **local LLM**. **The LLM never runs per customer on the full base**: training uses level 1 (cells), scoring uses level 2 on the 40k.
- **LLM budget per week** (config, default ⚙8 hours of GPU time): level 1 always runs; level 2 covers the 40k in priority order (Target 1 and Target 2 interleaved, so both get the same coverage); the run reports LLM time, decisions per level and cache hits.
- **Reference laptop (the production machine):** ThinkPad P16 Gen 3 · Intel Core Ultra 9 275HX (24 cores) · **128 GB DDR5** · **NVIDIA RTX PRO 5000 Blackwell Laptop, 24 GB GDDR7** · 1 TB NVMe SSD · Windows 11 Pro.
  - **Runtime:** WSL2 Ubuntu (vLLM does not run natively on Windows). Project and data **inside the WSL filesystem** (`~/nbx`), not on `/mnt/c`. `.wslconfig`: `memory=100GB`, `processors=22`, `swap=16GB`. The NVIDIA Windows driver must support **CUDA 12.8+** (Blackwell); PyTorch and vLLM must be builds with Blackwell support. Phase 1 checks this with `nvidia-smi` inside WSL and a 1-prompt vLLM smoke test.
  - **LLM serving:** **vLLM** (OpenAI-compatible, 127.0.0.1) with `--enable-prefix-caching`, `--gpu-memory-utilization 0.85`, `--max-model-len 4096`, `--max-num-seqs 64` (⚙). Ollama remains for the Gradio demo and offline tests only.
  - **Model (⚙):** default an **8B instruct model in FP8** (~9 GB, leaves ~11 GB of KV cache). Phase 9 benchmarks a **14B instruct model (FP8 or 4-bit AWQ)** on the same 500 decision traces and keeps it only if it is better on the offline reward estimate **and** the 40k still fit the budget. One model per run.
  - **Prompt size:** ≤ 1,500 input tokens per level-2 call (system prompt first, for prefix caching; only the customer's own evidence rows), ≤ 150 output tokens (compact JSON).
  - **Expected weekly LLM load:** level 1 ≈ a few hundred calls (< 15 min); level 2 ≈ **80k calls** (NBA + NBO for 40k) ≈ 10–12M output tokens → **≈ 2–3 h with the 8B, ≈ 4–6 h with the 14B**. Default level-2 coverage is **100 %** of the 40k.
  - **Laptop operation:** mains power, Windows power mode "Best performance", sleep and Windows Update restarts disabled during the run window; the run log records GPU temperature and clock every minute and warns above ⚙87 °C.
  - **Disk (1 TB):** raw JSON / CSV deleted after a successful Parquet conversion (checksum logged); keep ⚙12 weekly input folders and archive older ones to the approved FSDR share; snapshots store the panel only (not the 1M base); logs and snapshots kept ⚙52 weeks; alert when free space < 150 GB.
- **Timing targets** (Phase 9 benchmarks them on a **1M-line dummy base**, and once on 2M, and reports actual figures):

  | Step | Target (1M) |
  |---|---|
  | Load + validate + Parquet | ≤ 20 min |
  | Features + segments + micro-segments (full base) | ≤ 10 min |
  | Evidence tables (full base) | ≤ 15 min |
  | Agent dictionary (②b, ~70 calls) + two actions for 1M | ≤ 10 min |
  | Train ⑩ bandit + ⑪ patterns (CPU) | ≤ 30 min |
  | LLM level 1 (cell policies) | ≤ 30 min |
  | Full-base overview (no LLM) | ≤ 10 min |
  | LLM level 2 (40k) | ≤ 3 h (8B) |
  | Candidates + price passes + final output + audit | ≤ 20 min |
  | **Whole Thursday run** | **≤ 5 h** |

### 20. Outputs (`data/out/<run_date>/<run_id>/`)
**Every output table is written as `<name>.csv.gz` in the FSDR CSV standard (section 2)**, the same format as the inputs, with a `<name>.schema.json` (column names and types) and an `_outputs_manifest.json` (file list, row counts, SHA-256, run id, versions). Reports (audit, weekly, rules) are also written as HTML / JSON for the UI.

**`input_flytxt` (the pushed file):**
- `input_flytxt.csv.gz`, exactly the **38 lower-case columns of F-19 in that order**, one row per MSISDN.
- `msisdn` as 8-digit text; `date` / `start_date` as `YYYY-MM-DD`; `end_date` and `channel_1…6` empty fields; `flag_splio` = `1` (integer, no decimals); empty slots = empty fields (never `nbo_LFM_nan`, `nan`, `None`).
- Config `push_compression` (default `gzip`): if the campaign platform needs an uncompressed file, set `none` and the same file is also written as `input_flytxt.csv` (identical content).
- After writing, the file is **read back** and compared with the in-memory table (row count, columns, every value); any difference blocks the push (A-28).

- **Original names and structure:**
  - `results_scoring_nba`
  - `TG_40K_constraint_scored_nba` (Target 1), `TG_40k_challenger_scored_nba` (Target 2)
  - `recos_monthly_40k_constraint`, `recos_weekly_40k_constraint` (Target 1), `recos_monthly_challenger`, `recos_weekly_challenger` (Target 2)
  - **`input_flytxt`** (pushed)
- **Also:**
  - `input_flytxt_target1`, `input_flytxt_target2`
  - `panel_<T>`, `feature_snapshot_<run_id>`, `segments`, `scored_base`, `base_features_<run_date>`
  - `evidence_<version>`, `policy_<version>`, `llm_decisions` (scores, probabilities, justifications)
  - `final_rules_report`, `rules_report`, `market_report`
  - `audit_report_<run_id>`, `weekly_report`
  - `nba_campaign_log`, `nba_rewards`

### 20.1 ⑫ Customer ledger agent: one backup table at MSISDN level
**Table:** `fsdr_customer_ledger`, Parquet in `data/ledger/run_date=<D>/` (+ a DuckDB view over all weeks for queries). **One row per MSISDN per weekly run**, for the whole base (~1M). Customers of Target 1, Target 2 and control get every column; the rest of the base gets the training columns only (scoring columns empty, `panel_group` = `not_selected`).

**What goes in:** every **calculated** value, step result and decision, with the version that produced it. **What does not:** the raw input tables (the activity windows, raw purchases, raw campaign rows). Those stay in the dated input folders.

**Who writes what, and when:**
| Moment | Pipeline / stage | Columns written |
|---|---|---|
| End of training | T8 | groups A–E for the ~1M lines |
| End of scoring, before approval | P9 | groups F–H for the 40k + control (audit result included) |
| After push | P10 | push status, START_DATE, approval id |
| T+7 feedback | next week's T1 | group I (rewards) for last week's 40k + control |

Each write creates a **new version of that week's partition** (the previous version is kept, with checksums), so nothing is ever lost or silently overwritten.

**Columns** (the full list, types and the stage that fills each one go in `docs/fsdr_ledger_dictionary.md`):

| Group | Columns |
|---|---|
| **A. Run and ids** | run_date, anchor_date T, train_run_id, score_run_id, bundle_id, msisdn, access_method_id, account_id; versions: evidence, agent dictionary, ⑩ model, ⑪ model, policy, prompt, LLM model, rules (production / C-vN) |
| **B. Profile and bands** | nationality_group, age, age_band, tenure_days, tenure_band, micro_segment, value_band, usage_type, region, device segment, `profile_source` |
| **C. New calculations** | REV_W1…W4, REV_M1…M3, VAR_REV ×5, VAR_VOICE ×4, VAR_DATA ×4, zero flags, LOG_* ×6, the 11 model scores as used (+ `*_missing` flags), REV_M1, RGS flag, reconnect_p, fragility |
| **D. Segment and actions** | eligible flag + exclusion reason (rule id), segment, segment rule id, trigger values (JSON), segment-engine segment, `action_original` + source, `action_agent` + rank 2 + version, cell-policy top action, evidence level for actions |
| **E. Panel** | panel_group (target1 / target2 / control / not_selected), stratum (`Final`), week in panel, replaced flag, cool-down / fatigue status, other-campaign conflict flag |
| **F. Action decision** | allowed actions, E-A1 key figures used (take rate, uplift, n), ⑩ bandit probabilities (JSON) + top + confidence, LLM action scores (JSON), probabilities after ε, `action_sent`, propensity, exploration flag, fallback flag, `action_agreement`, `helper_agreement`, LLM justification (text) |
| **G. Offers and ARPU** | B-26 anchors (total / data / voice, monthly / weekly), F-05 anchors, C-02 band (Target 2), candidates per validity, ⑪ top offers, LLM offer scores (top 10, JSON), the up-to-4 selected offers with the pass or C-rule that selected them, slots 1–6 offers, reason per slot (recommended / favourite / forced + band), offers removed with rule (F-09 / F-11 / F-12), segment_btl_taker, segment_btl_taker_upsell |
| **H. Output, audit, push** | in_input_flytxt flag, drop rule (e.g. F-04), customer-level audit status + failed check ids, approval id + approver, push timestamp, START_DATE |
| **I. Rewards (filled at T+7)** | reward window, r1 (recommended offer bought), r2 (same service), r3 (any BTL), r4 (revenue change), offers bought + dates, days to purchase |
| **K. Revenue guard** | anchor_ref per type / validity, current bundles (service, validity, price), payg_spend, guard_ref per slot, CAN checks failed and repaired (ids), offers replaced (old → new), expected revenue (CAN-07); at T+7: revenue change vs own past and vs control stratum, cannibalised revenue |
| **J. Summary** | one plain-language line built by code, e.g. "Target 2 · India 40–59 1–2y · Data_Only_Voice_Appetency · sent UPSELL VOICE (p 0.385; original UPSELL VOICE, agent UPSELL VOICE) · slots CM_4R… · bought CM_4R on day 2" |

**Rules:**
- Written **by code** from the stage outputs; the ledger never recomputes anything, so it always equals what the pipelines actually used.
- Every value of `input_flytxt` for a customer equals the ledger's slot columns for that customer and run.
- **Security:** the ledger contains MSISDNs, so it stays on the laptop's encrypted disk, inside the approved FSDR folders, access-controlled; exports hash the MSISDN unless the approver allows otherwise. Retention ⚙52 weeks, then archived to the approved FSDR share. Backup copied daily with the registry.
- **Size (estimate):** ~1M rows × ~200 columns ≈ 200–400 MB per week in Parquet; 52 weeks ≈ 10–20 GB.

**Tools:** `ledger_write_training`, `ledger_write_scoring`, `ledger_write_push`, `ledger_update_rewards`, `ledger_lookup(msisdn)` (all weeks for one customer), `ledger_export(run_date, filters)`, `ledger_reconcile`.

**UI / API:** a **Customer lookup** tab (enter an MSISDN → every week's row, as a timeline); `/fsdr/ledger/{msisdn}` and `/fsdr/ledger/run/{run_date}`. **CLI:** `fsdr-ledger --lookup <msisdn> | --export <run_date> | --reconcile <run_date>`.

### 21. Logging (one record per pushed customer)
- run_id, target (1 production rules / 2 challenger rules), rules version (production or C-vN), evidence / policy / prompt / model versions, playbook version
- segment and rule id, nationality, age band, tenure band, micro-segment
- `action_original` (+ source), `action_agent` (+ version, rank 2), `action_sent`, `action_agreement`
- NBA: evidence level, scores, probabilities, chosen action, propensity, justification, fallback flag
- NBO: evidence level, candidates, scores, offers per slot with the reason (recommended / favourite / forced / removed by F-09, F-11, F-12)
- anchors (B-26, F-05), market index
- rules checked / failed / repaired
- snapshot id, START_DATE
- then rewards r1–r4

### 22. UI, API, CLI (same theme)
- **Tabs:**
  - Data
  - Segmentation & micro-segments
  - Evidence & policies
  - Rules (B, F, H, S, P, C registries)
  - Targets 1 vs 2 vs control
  - Final file (`input_flytxt` preview + final rules report)
  - **Audit** (latest report, history, blocking checks)
  - Agentic decisions (per customer: evidence → NBA scores → NBO scores → selection → slots)
  - Production (panel, approval, push status, history, rollback)
  - Weekly report
- **API:** an endpoint per tool, plus `/fsdr/runs`, `/fsdr/evidence`, `/fsdr/policies`, `/fsdr/rules`, `/fsdr/panel`, `/fsdr/audit/{run_id}`, `/fsdr/approve`, `/fsdr/report/{week}`.
- **CLI:** `fsdr-validate --pipeline train|score`, `fsdr-train --week0|--weekly` (training pipeline → bundle), `fsdr-bundle --list|--show <id>`, `fsdr-score --bundle <id> [--target 1|2|all]` (scoring pipeline), `fsdr-run-week` (train then score, stops before approval), `fsdr-feedback`, `fsdr-panel --build|--update`, `fsdr-final-output`, `fsdr-audit`, `fsdr-approve`, `fsdr-push`, `fsdr-report`, `fsdr-challenger-propose`, `fsdr-challenger-evaluate`, `fsdr-simulate --weeks N`, `fsdr-benchmark --lines 1000000`.

### 23. Tests (golden cases; offline; no network asserted; the LLM is faked)
- **ML only as helpers:** importing an ML library under `nbx/fsdr/` outside `nbx/fsdr/agents/helpers/` fails the test suite; with ⑩ and ⑪ switched off, the run still completes (evidence-only).
- **⑩ Action-bandit agent:** IPS-correct training on logged rows (a row with propensity 0.2 weighs 5 × a row with 1.0); scores only for allowed actions; save / load gives identical scores; week-0 model flagged `weak`; OPE golden case (the 5-row IPS case → 1.1562861).
- **⑪ Offer-pattern agent:** real customer ids (not the loop index, original bug); a customer who bought A and B is scored higher on C when C is often bought with A and B (toy matrix); cold-start customer → micro-segment average, flagged; hold-out precision@4 is computed and compared with popularity.
- **Helper vs LLM:** when the LLM goes against the helper's top choice, a justification is present; agreement flag logged.
- **Contract:** every table of section 2 generated as `.csv.gz`, loaded and validated; a 12th model via config flows end to end.
- **File format (`.csv.gz`):**
  - an ID like `09123456` stays `09123456` (text) after read → Parquet → write
  - a 2,000-line round trip input `.csv.gz` → Parquet → `.csv.gz` gives byte-identical values (same header, separator, quoting, decimals, dates, empty fields)
  - missing values are written as empty fields, never `nan` / `None` / `NULL`; no scientific notation (e.g. price 12.5, amount 0.0001)
  - Arabic scripts survive the round trip unchanged (UTF-8)
  - a production run with a plain `.csv` or `.json` input fails validation unless `allow_test_formats` is on; a missing `.csv.gz` blocks the run
  - `input_flytxt.csv.gz`: 38 lower-case columns in order, `flag_splio` = `1`, `end_date` empty, read-back identical; with `push_compression = none` the `.csv` has the same content
  - every output has its `.schema.json` and appears in `_outputs_manifest.json` with the right row count and checksum
- **Calendar:** 2026-10-22 → T 2026-10-22; 2026-10-23 → 2026-10-22; 2026-10-21 → 2026-10-15; January location rollover; next week's W2 = this week's W1; reward window Friday → Wednesday (6 days) for a Thursday run.
- **Features:** REV W1 2.1 / W2 3.5, data 0 / 850, voice 40 / 0 → VAR_REV −40, LOG_REV −0.3726753, VAR_DATA −100, LOG_DATA −6.7464121, voice zero flag 1, LOG_VOICE 3.7135721 (older → recent).
- **Priors:** 10,000 rows (5,090 / 2,620 / 1,020 / 530 / 420 / 240 / 80) → 0.509 … 0.008, clipped to 0.01. **OPE:** the 5-row IPS case → 1.1562861.
- **Anchors:** B-26: 6, 4, 5 → 6.0 / 5.4 / 0.6; weekly 1, 2, 1, 2 → 2.0773503. **F-05:** 0, 4, 6 → **6.0**.
- **Price passes (with fixed LLM scores):** MAINTAIN, anchor 6.0 / 5.4, offers COMBO 7.0 (0.9), COMBO 8.0 (0.95), DATA 5.5 (0.7), DATA 7.4 (0.8), DATA 7.3 (0.8) → COMBO 7.0 + DATA 7.3; pass-2 and pass-3 cases; sort orders. **Target 2 (C-v1):** anchor 6.0, UPSELL DATA, offers 5.0 / 7.0 / 8.5 / 10.0 → band 6.5 – 9.0 keeps 7.0 and 8.5; selection by score × price; nothing in band → widened once upwards (6.5 – 12.0), then Target 1 passes as fallback; data / voice split 70 / 30 from a customer whose spend is 70 % data.
- **Code 5:**
  - slots 1 / 2 from monthly 6.8 / 3.4
  - forced: anchor 2.5 → CM_3R_8GB_240MIN; 3.5 → CM_4R_10GB_300MIN; 5.2 → none; already holding a popular offer → none
  - (type, price) duplicate cleared; conflict (A, B) → A removed; VOICE conflict row ignored
  - UPSELL → BTL_upsell; START_DATE = D + 1
  - 38 columns in order
  - **parity:** with the bug flag on, ⑦ equals a direct replay of code 5 on the same reco tables, column by column
- **Segmentation, age and tenure:**
  - the S-rule cases
  - age 19 → young, 40 → middle, 59 → middle, 60 → senior, null / 120 → unknown
  - tenure 40 → `t0_lt3m` (and Nursering); 89 → `t0_lt3m`; 90 → `t1_3_6m`; 182 → `t1_3_6m`; 183 → `t2_6_12m`; 365 → `t2_6_12m`; 366 → `t3_1_2y`; 730 → `t3_1_2y`; 731 → `t4_gt2y`; −5 / null → unknown
  - the profile overrides `party_scoring.AGE`
  - the micro-segment id = nationality × age × tenure
- **Segment-action agent:**
  - `Retention_RGS_At_Risk` → `action_original` RETAIN; a segment-engine segment overrides the crosswalk; source logged
  - week 0 with no clear evidence → `action_agent` = `action_original` for every cell
  - a fake LLM dictionary giving UPSELL to an at-risk segment, or an action to Nursering, is rejected and the previous version kept
  - Target 1 fallback = `action_original`, Target 2 fallback = `action_agent`; `action_agreement` values computed correctly
- **Recommendable offers (B-20b):** a catalogue offer with no purchase in 270 days is never a candidate; an offer bought but removed from the catalogue is never a candidate; with `only_own_past_offers` on, slots 1–4 contain only the customer's own past offers.
- **Revenue guard (both targets):**
  - anchor 6.0, current monthly DATA bundle 7.0 → guard_ref 7.0; a 6.5 DATA offer fails CAN-01 and is replaced by the next candidate ≥ 7.0 (or the slot is cleared)
  - current DM 5 OMR 18 GB; offer 4 OMR 20 GB → CAN-02 fail
  - monthly buyer of 6 OMR data; weekly data offer 1.2 OMR (× 4.3 = 5.16) → CAN-03 fail
  - current DATA 4 + VOICE 3; COMBO 6 → CAN-04 fail
  - PAYG voice 5 OMR last month; voice bundle 3 OMR → CAN-05 fail (threshold 1.0)
  - forced bundle CM_3R (3 OMR) for a customer with monthly guard_ref 3.5 → not placed (CAN-06)
  - Target 1 in `report` mode → flagged, not repaired; in `repair` mode → repaired
  - a violation left after repair → A-27 FAIL, push blocked
  - T+7: a pair with negative net incremental revenue 2 weeks in a row → block list → excluded next week in both targets
- **Ledger:** after a 2,000-line run, one row per MSISDN; a Target 2 customer's row has groups A–H filled and its slots equal its `input_flytxt` row; a not-selected line has scoring columns empty; after the next week's feedback, group I is filled and the previous partition version still exists; `ledger_lookup` returns every week for one MSISDN; no raw activity column is present.
- **Pipeline split:** `fsdr-score` refuses a missing, unvalidated or other-week bundle; scoring never writes to the bundle and never calls a training function (asserted); training never writes `input_flytxt` or the campaign log; the 40k features in scoring equal the bundle's rows exactly; week 0 fails validation if `_auto` or `_score` tables are missing, week n does not need `_auto`.
- **Run order:** stages run in order, each stage writes a checkpoint, a restart resumes at the failed stage without redoing finished stages; LLM batches are grouped by segment; one START_DATE for the whole 40k.
- **Evidence:** a cell under the minimum n → the fallback chain down to the B-10 priors; every row carries n and an interval; the level is logged; evidence on the full base has one row source per eligible line.
- **LLM NBA agent:**
  - scores only for allowed actions; an invented action is rejected
  - scores → probabilities: scores (0.6, 0.3, 0.1) on 3 allowed actions, ε = 0.1 → (0.6·0.9 + 0.1/3, 0.3·0.9 + 0.1/3, 0.1·0.9 + 0.1/3) = (0.5733333, 0.3033333, 0.1233333)
  - the sampled action's probability is logged as the propensity
  - an invalid JSON reply → evidence fallback, marked `partial_fallback`
- **LLM NBO agent:**
  - it can only score candidate ids from the list; an unknown id is rejected
  - Target 1 candidates respect B-21; Target 2 candidates respect C-01 / C-02
  - output columns equal the original `recos_*` structure, so code 5 runs unchanged
  - the 10.1 loop runs at most once
- **Level-1 policy:** a policy that proposes a forbidden action or family is rejected by ⑥.
- **Determinism:** the same inputs and versions give identical decisions (cache and temperature 0).
- **Auditor:** each check is broken on purpose and must FAIL:
  - a 39th column
  - a duplicate msisdn
  - a control msisdn in the file
  - a missing script
  - a conflict left
  - a slot-3 monthly offer
  - targets 22k / 18k
  - a Target 1 offer outside its B-21 family, or a Target 2 offer outside the C-02 band
  - an offer id the LLM invented
  - an unapproved challenger version
  - a missing snapshot

  The push is blocked in each case.
- **Production loop:** a 3-week simulation (2,000 lines; 40 % Target 1 / 40 % Target 2 / 20 % control) runs feedback → train → score → final → audit → push stub in < 2 minutes with the fake LLM; rewards use only purchases after START_DATE; nothing is pushed without audit PASS and approval; a failure keeps the previous versions.
- **All existing NBX tests pass.**

### 24. Docs
- Architecture, the input contract (every table and field, with an example record)
- How the LLM agents replace a bandit and a ranker (evidence → scores → sampling → selection → learning)
- The rule registries (id → code → test)
- The training and production runbook (week 0, week n, weekly data list, approval, rollback)
- The `input_flytxt` specification
- The auditor checks
- Known differences vs the original codes

### 25. Open items (list with your default in the plan)
1. The CHECK segment mappings (B-03).
2. Control-group size (default 10k, outside the 40k).
3. Approval of **C-v1** for Target 2 (values of C-02 band, C-05 do-not-disturb), and whether Target 1 keeps the production free part (B-23 switch, default off).
4. Margin costs for expected value (ASSUMPTION until finance provides them).
5. `young` age split; tenure unit in `customer_profile` (days by default).
6. Panel duration, cool-down and fatigue thresholds.
7. Reward window: 6 days or a lagged 7 days.
8. LLM model: 8B FP8 (default) or 14B, decided by the Phase 9 benchmark.
9. Base size (default ~1M; tested at 2M).
10. Revenue guard: CAN-05 pay-as-you-go threshold (default 1.0 ×), CAN-03 weekly factor (4.3), Target 1 guard mode (`repair` by default, or `report` to keep production output unchanged).

### 26. Phases (stop after each; show the evidence)
1. **Plan** (no code) + **environment check** (WSL2, `nvidia-smi`, vLLM smoke test) → approval.
2. **Input contract:** all tables, generator (1M and 2,000-line modes, consecutive weeks), ① data agent. *Show the validation report.*
3. **Full-base preparation:** features (B-01 … B-12), ② segments, age / tenure bands, micro-segments for 1M lines, and the deterministic part of ②b (`action_original`, crosswalk). *Show the segment and micro-segment counts, the crosswalk and the run time.*
4. **Evidence tables** E-A0 … E-A4, E-O1 … E-O5 with levels, n and intervals, then the **②b agent dictionary** (LLM) and `action_agent` for all lines. *Show the coverage per level and the agent dictionary vs the original (cells changed, with reasons).*
5. **⑩ Action-bandit + ⑪ Offer-pattern helpers** (training on the 1M base, quality gates), then the **LLM NBA agent** (scores, ε sampling, propensity, fallback). *Show the helper quality reports and 10 decision traces with E-A5.*
6. **LLM NBO agent** + candidates per target + ⑥A production rules (B-20 … B-32, Target 1) + ⑥B C-v1 (C-01 … C-05, Target 2) → the reco tables. *Show 10 traces per target.*
7. **⑦ Code 5** + parity + final rules report, **⑬ revenue guard** (P6b, P7b, repair, CAN tests), then **⑨ auditor** (all A / T / R checks) + the broken-on-purpose tests. *Show `input_flytxt` (first 5 rows), the parity diff and an audit report.*
8. **★ orchestrator** (levels 1 and 2), ⑤, ⑥B, ⑧. *Show the cell policies, 5 full decision traces and the top challenger rule proposals.*
9. **Production loop** with the two pipelines (`fsdr-train` → bundle → `fsdr-score`): panel 20k / 20k + control, snapshots, feedback, weekly evidence and policy update, scoring, audit, approval, push stub, rollback, scheduler. *Show a 3-week simulation: Target 1 vs Target 2 vs control, with CIs.* **Then run the 1M-line benchmark** (and once 2M) and the **LLM speed benchmark** (8B vs 14B, seconds per decision, coverage).
10. **⑫ Customer ledger** (writes from training, scoring, push and feedback; lookup; reconcile), then **UI, API, CLI, docs**. *Show the ledger rows of 3 customers (Target 1, Target 2, control) across 2 simulated weeks.*
