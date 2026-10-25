# FSDR agentic NBA / NBO upgrade: plan

> **Status:** Phases 1–4 done; Phase 5 done for the recommender (action, offer and competitor agents); FSDR API, 4 UI tabs and the batch flow (action → top 4 → roaming → international) are live.
> **Next:** Phase 5b (data / training / scoring / monitor agents), Phase 6 (campaign log, feedback, simulation),
> rest of Phase 7 (Simulation tab, real-JSON upload from the UI, rebrand of the 12 NBX tabs).
> This file is updated at the end of every phase (see [Progress log](#13-progress-log)).

Source of the requirements: [PROMPT_for_Claude_Code_FSDR_agentic_NBA.md](PROMPT_for_Claude_Code_FSDR_agentic_NBA.md)
(Part B), with the reference files in [fsdr_reference/](fsdr_reference/).

---

## 0. Starting point (what was checked)

| Item | Result |
|---|---|
| Folder | `nbx-python-upgrade/` = copy of `nbx-python/` without generated indexes or caches |
| Reference files | copied to `docs/fsdr_reference/` (7 files). `json_samples/` does not exist yet, so schemas come from `fsdr_original_sql_and_schemas.txt` |
| Tests | **104 passed** (`python run.py test`) |
| Fixed during setup | `run.py test -v` / `-k ...` failed (argparse dropped options that start with `-`). Fixed in both `nbx-python` and `nbx-python-upgrade` |
| Git | Not touched. This folder sits inside the parent repo, which is on a feature branch with unrelated uncommitted changes. See question Q0 |
| Libraries available | `pyarrow` ✓ (parquet), `duckdb` ✓, `jsonschema` ✓, `vowpalwabbit` ✓, `implicit` ✓, **`lightfm` ✗** (no Python 3.12 wheel) |
| Real operator names in NBX | Real operator and competitor names in 13 files (UI, theme, competitor module and data, docs, tests). Reference files have none. **Done (2026-10-24, audit A8):** operator = `OPERATOR_NAME` (FSDR), competitors = COMPETITOR_A / COMPETITOR_B, snapshot anonymised, scraper sources in a local git-ignored `data/competitors/sources.json` |
| Anchor date T | Checked: Thu 2026-10-22 → itself, Wed 2026-10-21 → Thu 2026-10-15, Fri 2027-01-01 → Thu 2026-12-31 |

### How NBX maps to the names used in the prompt

The prompt names a few modules that have a different name in NBX. The plan uses the real ones:

| Prompt says | In NBX |
|---|---|
| `core/settings` | `nbx/core/config.py` |
| `core/schemas` | `nbx/core/models.py` (pydantic API models) |
| "routers" in `api/main.py` | `main.py` has no routers today (one file, `@app` routes). FSDR routes go in **one `APIRouter`** in `nbx/api/fsdr.py`, mounted with one line in `main.py`. Same error format and same upload-job style |
| tool registry | `nbx/agents/agentic/tools.py::TOOLS` (one global dict) + `react.py::run_react` |

---

## 1. Architecture and data flow

Same layers as today: **UI → HTTP → FastAPI → agents → core models / vector stores → llm_client**.
FSDR is a new package next to the existing ones and plugs into each layer.

```
                         Gradio UI (5 new FSDR tabs, after the 12 existing ones)
                                              │ HTTP only
                         FastAPI  nbx/api/fsdr.py  (APIRouter /fsdr/*, SSE stream)
                                              │
            ┌─────────────────────────────────┼──────────────────────────────────┐
            ▼                                 ▼                                  ▼
   nbx/fsdr/agents/*                  nbx/fsdr/pipelines/*              nbx/fsdr/synth/*
   data · training · scoring ·        train · score · recommend ·       world · scenarios ·
   recommender (council ext.) ·       feedback · simulate               generator
   monitor                                    │                                  │
            │ tools = pipeline steps          │                                  │ writes
            ▼                                 ▼                                  ▼
   agents/agentic/react.py      nbx/fsdr/{features, labels, bandit,     data/fsdr/<YYYY-MM-DD>/*.json
   (same loop, same caps)        offer_ranker, offer_rules, registry,           │
                                 campaign_log, calendar, config}                │ read by
                                              │                                  │
                                              ▼                                  ▼
                       core/vw_format.py (new, shared)         nbx/fsdr/datasource.py
                       core/collab_filter.py (BPR, generalised)  JsonSource | SyntheticSource
                       core/knowledge_graph.py · rules_engine
                       llms/llm_client (offline | ollama | vllm)
```

### Weekly flow (real data or the simulation)

```
D (run date) ──► T = last Thursday ≤ D ──► data/fsdr/<D>/*.json (real export or generator)
   │
   ├─ fsdr-validate ─► validation report (schemas, keys, duplicates, coverage, cross-table)
   ├─ fsdr-train    ─► features(T-anchored *_auto) + labels (targeting D-8→D-2, rewards D-7→D-1
   │                   and/or campaign log) ─► VW logged training ─► evaluate (holdout IPS/SNIPS)
   │                   ─► registry: challenger ─► training agent: promote / keep / ask human
   ├─ fsdr-score    ─► champion + features(D-anchored *_score) ─► results_scoring_nba + per-arm files
   │                   + drift report (PSI, action-mix shift)
   ├─ fsdr-recommend ─► arm cstr: action → families → ranker → 3-pass price bands → recos_*
   │                    arm free: L2, DATA+COMBO only → ranker → 3-pass → recos_*
   │                    arm llm : recommender agent (LLM free, hard rules enforced) → recos_*
   │                    arm cg  : nothing sent (control)
   │                    ─► nba_campaign_log (append-only parquet)
   └─ fsdr-feedback (D+7) ─► fulfillments joined to the log ─► nba_rewards ─► next week's training
```

---

## 2. Package layout and file list

### 2.1 New files

```
nbx/fsdr/
├── __init__.py
├── config.py            every BR constant (BR-xx in a comment), MODEL_SCORE_REGISTRY, paths,
│                        arms, ε_llm, contact caps, churn threshold, generosity switch
├── schemas.py           raw schema per table (name, raw type, nullable, key), MODEL_SCORE_SCHEMA,
│                        output-table schemas; JSON-Schema export to docs/fsdr_json_schemas/
├── calendar.py          anchor_date(D); every window of BR §4 as (start, end) per table/period
├── datasource.py        DataSource protocol; JsonSource (array or .jsonl, auto-detect, _manifest);
│                        SyntheticSource (calls generator, same tables in memory)
├── validation.py        per-file + cross-table checks → ValidationReport (JSON + Markdown)
├── synth/
│   ├── world.py         hidden customer state + hidden response function + weekly evolution
│   ├── scenarios.py     baseline, churn_wave, data_growth, price_sensitive, new_offers,
│   │                    campaign_fatigue, data_quality_issues (combinable)
│   └── generator.py     daily dummy history → every table aggregated with the BR §4 windows
├── features.py          port of nba_lib.build_features (single function, train = score)
├── labels.py            segment → action, rewards (3 definitions), propensities
├── bandit.py            VW contextual bandit: train (logged | simulation), predict pmf, score,
│                        save/load + compatibility check
├── offer_ranker.py      OfferRanker protocol; ImplicitBPRRanker (default); LightFMRanker (optional)
├── offer_rules.py       ARPU anchors, action → families, 3-pass price bands, conflicts, eligibility,
│                        max-offer caps
├── registry.py          versioned model folders, metadata, champion / challenger, rollback
├── campaign_log.py      append-only parquet log + JSON export
├── report.py            weekly monitor report (take rate / revenue / uplift per arm, drift, agreement)
├── pipelines/
│   ├── train.py         7 steps, each a typed tool with a JSON report
│   ├── score.py         5 steps
│   ├── recommend.py     free / cstr / llm arms
│   ├── feedback.py      D+7 rewards
│   └── simulate.py      multi-week closed loop
└── agents/
    ├── tools.py         FSDR tool registry (same Tool dataclass as agents/agentic/tools.py)
    ├── data_agent.py
    ├── training_agent.py
    ├── scoring_agent.py
    ├── recommender_agent.py   extends the council for the llm arm
    └── monitor_agent.py

nbx/core/vw_format.py    corrected, shared VW text format (name:value, underscored action tokens,
                         fixed action order, labelled/unlabelled ADF examples)
nbx/api/fsdr.py          APIRouter /fsdr/* (+ SSE stream reusing the /agentic/stream pattern)
nbx/ui/fsdr_tabs.py      the 5 FSDR tabs, built with ui/theme.py components

docs/fsdr_plan.md (this file) · docs/FSDR_GUIDE.md · docs/FSDR_DATA_DICTIONARY.md ·
docs/FSDR_BUSINESS_RULES_MAP.md · docs/fsdr_json_schemas/*.json

tests/fsdr/  (see §9)
```

### 2.2 Existing files that change (all backward compatible)

| File | Change | Why |
|---|---|---|
| `nbx/agents/agentic/react.py` | `run_react(..., registry=TOOLS)` and `build_prompt(..., registry=TOOLS)` take an optional tool registry | FSDR agents reuse the exact loop with their own tools. Default argument keeps every current caller and test unchanged |
| `nbx/agents/agentic/council.py` | Small hooks: arm-aware entry `run_council(..., arm=None, tools=..., guardrails=...)`; `NO_CONTACT` accepted when the caller allows it | The llm-arm recommender is the council with FSDR tools, not a copy |
| `nbx/core/collab_filter.py` | Extract `fit_bpr(interactions, params)` + `score_items(...)`; current `_train` calls them | `ImplicitBPRRanker` reuses the same BPR code on FSDR interactions |
| `nbx/core/nba_bandit.py` | Uses `core/vw_format.py` helpers. **Default: keeps its current behaviour** (see Q11) | No duplicate VW code |
| `nbx/agents/rules_engine.py` | FSDR hard rules registered as `Rule` objects in a second list `FSDR_RULES` (same `Rule` dataclass, same `apply_rules`) | One rules mechanism; shows in the Business Rules tab |
| `nbx/api/main.py` | `app.include_router(fsdr.router)` (1 line) | Mount FSDR routes |
| `nbx/ui/gradio_app.py` | After the 12 tabs: `fsdr_tabs.build()` (a few lines) | FSDR tabs last, existing order unchanged |
| `run.py` | 8 new commands (`fsdr-generate` … `fsdr-report`) | CLI |
| `.env.example`, `nbx/core/config.py` | FSDR settings with defaults; `OPERATOR_NAME="FSDR"`, `COMPETITOR_LABELS` | Config + rebrand |
| Rebrand (Phase 7) | `ui/theme.py`, `ui/gradio_app.py`, `agents/agentic/*` texts, `core/competitors.py` labels, competitor snapshot, docs, tests | Rule 2: only names, logo text and wording change. Colours, layout and logic stay |
| `requirements.txt` | add `pyarrow` (already installed here), `jsonschema` | parquet log, schema checks |

---

## 3. Input data contract (JSON, one file per table)

- Folder: `data/fsdr/<YYYY-MM-DD>/` with `_manifest.json` (`run_date`, `anchor_date`, `source`:
  `generator|export`, `scenarios`, `seed`, `row_counts`, `schema_version`).
- File name = exact table name + `.json` (array) or `.jsonl`. The format is detected from the first
  non-blank character, so either extension works.
- Field names and case as in production, quirks kept: `UBSCRIPTION_ID`, `activities_auto_m2`,
  `Access_Method_id`, `"Price data"`, `"Sl No."`, string numbers in `btl_catalog` and in the sample table.
- Types: JSON string / number / null; dates `"YYYY-MM-DD"`, timestamps ISO 8601.
- `schemas.py` is the single source. `python run.py fsdr-schemas` writes `docs/fsdr_json_schemas/<table>.json`
  so real exports can be checked before loading.

### 3.1 Tables (45 files per run date)

| Group | Tables |
|---|---|
| Base and reference (10) | `base_prepaid`, `consumer_aact_account`, `access_method_id_auto`, `party_auto`, `party_scoring`, `location_auto`, `btl_catalog`, `BTL_script_english_arabic_ok`, `conflicting_offers_h`, `DEVICEBEHAVIORSEGMENTATION.dv_segment` |
| Revenue / usage, T-anchored (9) | `subs_amt_auto`, `activities_w1_auto` … `activities_w4_auto`, `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto`, `paygo_activity_auto` |
| Revenue / usage, D-anchored (8) | `subs_amt_score`, `activities_w1_score` … `activities_m3_score` |
| Offer pricing, D-anchored (2) | `subscriptions_tg`, `usage_data_voice_payg_tg` |
| Campaign (4) | `targeting_dataiku_auto`, `targeting_auto`, `subs_btl_auto`, `sub_btl_3months` |
| Model scores (11) | the 11 files of the registry, one shared `MODEL_SCORE_SCHEMA` |
| Test sample (1) | `sample_cg_tg_new_7oct2026` (name configurable: `FSDR_TARGET_SAMPLE_TABLE`) |

Production note: in the original code the scoring table is named `activities_w1_score_hfds`
(typo). In JSON the file is `activities_w1_score.json`, and the loader also accepts the typo name.

### 3.2 Model scores: generic

```python
MODEL_SCORE_REGISTRY = [  # file → feature → combine ; adding a model = adding a line
  ModelScore("DAIM_CHURN_GLOBAL_SCORING_TD.prediction", "PROBA_CHURN", combine="max", signal="churn"),
  ModelScore("DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction", "PROBA_CHURN", "max", "churn", tenure=(0, 30)),
  ModelScore("DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored", "PROBA_CHURN", "max", "churn", tenure=(30, 60)),
  ModelScore("DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored", "PROBA_REACT_30", signal="reactivation"),
  ... 7 more ...
]
```

`PROBA_FEATURES` (feature list), the VW shared line, the compatibility check, the generator, the
UI coverage table and the data dictionary are all **derived** from this list. A test adds a 12th
dummy model through config only and checks it reaches the features and the model metadata.

### 3.3 Validation report (blocking vs warning)

| Check | Level |
|---|---|
| Missing required file, missing required field, wrong raw type in a key field | **blocking** |
| Null keys, duplicate MSISDN in `base_prepaid`, keys not matching across tables (e.g. targeting `access_method_id` absent from `access_method_id_auto`) | **blocking** above a threshold (default 1 %), warning below |
| Unknown extra fields, wrong type in a non-key field, stacked duplicates in model scores, coverage < 70 % | warning |
| Window consistency on generator output (`Sub_Amt_m1` ≈ Σ weeks in the same days) | warning (info for real data) |

---

## 4. Dummy world and generator

### 4.1 Hidden state per customer (`synth/world.py`)

`churn_risk`, `data_appetite`, `voice_appetite`, `price_sensitivity`, `preferred_validity` (7/30),
`spend_level` (heavy-tailed), `usage_type` (voice-only / data-only / dual / inactive),
`nationality`, `tenure`, `plan`, `contacts_last_4w`, `last_accept_date`.

**Hidden response function** `P(accept | customer, action, offer, price, anchor)`:
logistic on: action fit (RETAIN ↔ churn risk; UPSELL DATA / CROSS SELL DATA ↔ data appetite on
voice-only; voice actions ↔ voice appetite), generosity (L2 > L1 > L0), price gap vs ARPU anchor
× price sensitivity, validity match, contact fatigue, plus an organic baseline purchase rate (what
the control group buys without contact).

Daily simulation: each customer has daily voice minutes, data MB, PAYG amounts and subscription
purchases drawn from the hidden state. Every table is aggregated from this daily history with
the **exact BR §4 windows** (subscriptions include the anchor day; activities exclude it; pricing
tables lag by 2 days), so W2 of date D equals W1 of date D-7 by construction.

**Memory at 200k lines:** the daily history is generated in chunks of 20k customers and
aggregated on the fly. A full 200k × 95-day matrix would need about 1.5 GB.

### 4.2 Realism

5,000 lines by default (configurable up to 200k), fixed seed, target under 60 s for 5k; MSISDN 8-digit
strings > `'69999999'`, the same value in `access_method_id_auto.MSISDN` (no prefix: production SQL `ACCESS_METHOD_VAL as MSISDN`);
nationality mix (Oman, India, Bangladesh, Pakistan + long tail); the 13 excluded plans present;
~60 BTL offers (DATA / VOICE / COMBO × L0/L1/L2 × 7/30 d, 0.5–15, generosity monotonic with price
within a family), dummy names such as `BTL_DATA_L2_30D_10GB`; English/Arabic dummy scripts;
conflict pairs; scores = sigmoid(hidden signal + noise), coverage 70–90 %, early-churn files limited
to tenure 0–30 / 30–60; segments with the real mixed case (`RGS_…` and `Rgs_…`).

### 4.3 Scenarios

| Scenario | Changes in the hidden world | Expected visible signal (tested) |
|---|---|---|
| `baseline` | none | — |
| `churn_wave` | one segment's churn risk rises weekly | PROBA_CHURN ↑ in that segment; RETAIN take rate highest there |
| `data_growth` | voice-only customers gain data appetite | data MB trend ↑, PROBA_APP_DATA ↑, UPSELL/CROSS SELL DATA ↑ |
| `price_sensitive` | strong penalty above the anchor | take rate falls with price gap |
| `new_offers` | new BTLs added mid-simulation | ranker cold start handled (popularity fallback), no crash |
| `campaign_fatigue` | acceptance ↓ with contacts in last 4 weeks | take rate ↓ with `NB_CONTACTS_LAST_4W` |
| `data_quality_issues` | duplicate MSISDNs, null keys, stacked churn duplicates, keys as strings | validation flags each issue |

---

## 5. Deterministic ML pipelines (each step = tool with typed input/output + JSON report)

### 5.1 Features (`features.py`, port of `nba_lib.build_features`)

- 9 probability features from the registry (3 churn files → `PROBA_CHURN` by max).
  **Missing → uniform(0, 0.02) per row**, seeded (fixes the "one value for all" bug).
- Revenue per period = `SUB_AMT_p + PG_DATA_AMOUNT_p` (BR-49). Missing usage/subscriptions → 0.
- Trends as (recent, older), **W1 = most recent** (fixes the inverted direction):
  REV 5 pairs, VOICE 4, DATA 4 → `VAR_*` (clipped to [−100, 500]), `LOG_*`, `*_ZERO` → 39 features.
- `NATIONALITY_GROUP` (BR-15). Total: 48 numeric + 1 categorical (+4 last-campaign features in Phase 6).
- Keys normalised to one integer type everywhere (fixes string vs integer joins).

### 5.2 Labels (`labels.py`)

- Targeting (`targeting_dataiku_auto`, `date1` in D-8→D-2), Nursering excluded, segment matched
  case-insensitively (`SEGMENT_TO_ACTION` from `nba_lib`), latest targeting per line.
- Reward: `any_btl` (BR-40, production) | `same_service` | `exact_offer`, from the reward window D-7→D-1
  **and** only for fulfillments after the customer's `date1` (see Q2).
- Propensity: real logged propensity when the campaign log has it; otherwise this week's action
  share; `ACTION_PRIOR` (BR-45) as the last fallback.

### 5.3 Bandit (`bandit.py` + `core/vw_format.py`)

- `--cb_explore_adf -q UA --epsilon 0.2` (configurable ε), **fixed action order** (`nba_lib.ACTIONS`),
  action tokens without spaces, numeric features as `name:value`.
- Modes: `logged` (default: all targeted lines, takers and non-takers, N passes) and `simulation`
  (original: takers only, match the logged action).
- **The model is saved before anything else and never `finish()`ed before scoring** (fixes that bug).
- Save → `vw.save()` + metadata; load refuses a model whose feature list or action list differs.

### 5.4 Training pipeline steps

`load_and_validate` → `build_features` → `build_labels` → `train_bandit` → `evaluate` (holdout 20 %:
logged take rate, IPS, SNIPS, action distribution, and on synthetic data the hidden-optimal policy
value) → `register_model` (challenger) → `score_full_base`.

### 5.5 Scoring pipeline steps

Load champion (refuse if incompatible or older than `FSDR_MAX_MODEL_AGE_DAYS`, default 14) → build
D-anchored features → score target sample or base, policy `sample|greedy`, seed = date by default →
write `results_scoring_nba` + `TG_free_scored_nba` / `TG_constraint_scored_nba` / `TG_llm_scored_nba`
→ drift report (PSI per feature vs training, action-mix shift).

### 5.6 Offers (`offer_ranker.py`, `offer_rules.py`, `pipelines/recommend.py`)

- Ranker trained per action × validity on the 270-day interactions (`sub_btl_3months`, BR-24/83),
  predicting **by user id** (fixes the LightFM loop-index bug). BPR by default, top 100 per customer
  (BR-85), popularity fallback for cold-start customers (BR-86).
- ARPU anchor: `mean + std` (BR-71, `ddof=1` as pandas), revenue = subscriptions + **total** PAYG
  (BR-68), data 0.9 / voice 0.1 (BR-74/75), weekly vs monthly anchors (BR-77).
- cstr arm: BR-66 families, with a generosity switch `normal | all_L2` (default `all_L2`, Q3).
  **CROSS SELL VOICE gets its own filter branch** (fixes the typo).
- free arm: L2 only, DATA + COMBO (BR-67).
- 3 passes (BR-78…82), k per action, sort orders as in production (pass 1: score ↓, price ↓; passes 2-3:
  score ↓, price ↑), at most 2 offers per validity, 4 in total.
- **New rule:** no two offers that are a pair in `conflicting_offers_h` go to the same customer
  (the lower-ranked one is dropped and the next candidate is tried).
- Outputs `recos_monthly_<arm>`, `recos_weekly_<arm>` with `pass` (1/2/3) and `anchor` on each row.

---

## 6. Agentic layer

All agents use `agents/agentic/react.py` unchanged in behaviour: one JSON reply per step, tool or
final, required tools before final, repeated calls refused, step cap, then a rule-based fallback
using the same tools. `mode` = `fallback` / `agentic` / `partial_fallback` as today.

| Agent | Tools (= pipeline steps) | Decides | Code-enforced limits |
|---|---|---|---|
| Data | `generate_data`, `load_folder`, `validate`, `table_preview` | data usable? blocking issues explained | blocking issues = cannot say "usable" |
| Training | `build_features`, `build_labels(reward_def)`, `train_bandit(mode, epsilon, passes)`, `evaluate`, `compare_with_champion`, `register_model` | mode, reward definition, hyper-parameters within ranges; **promote / keep champion / ask human** + justification | ranges in config; promote only if IPS/SNIPS ≥ champion − tolerance |
| Scoring | `load_champion`, `model_age`, `drift_report`, `score` | score now or refuse | refuses stale or incompatible model, PSI above threshold → "ask human" |
| Monitor | `arm_metrics`, `uplift_vs_control`, `bandit_llm_agreement`, `drift_history` | weekly report + next-week recommendations | — |
| Recommender (llm arm) | see 6.1 | action (7 or `NO_CONTACT`), offers, justification | hard rules |

### 6.1 Recommender agent: LLM as decision-maker

Built on the existing council (strategist → offer specialist → critic → messenger → explanation),
running with FSDR tools when `arm="llm"`.

- **Evidence tools:** `customer_360` (features, W/M trends, 11 scores, last campaigns and responses),
  `bandit_pmf` (7 actions), `ranker_top_offers(action, validity)`, `arpu_anchors`,
  `price_band_candidates`, `check_eligibility_and_conflicts`, `similar_customers` (cohort graph),
  `competitor_comparison`, `campaign_history`.
- **Freedom:** any of the 7 actions, any eligible catalogue offer, or `NO_CONTACT`. Required
  justification: `signals_used`, `override_reason` (when it differs from the bandit top action or the
  ranker), `expected_acceptance`, `expected_value`.
- **Hard rules (code):** offer exists · customer eligible (BR-01…12, excluded plans and segments) ·
  no conflicting pair · ≤ 2 per validity, ≤ 4 total · contact cap · price > 0 · no upsell-family action
  when `PROBA_CHURN ≥ FSDR_CHURN_BLOCK_THRESHOLD`. A violation goes to the critic; still violating → auto-fix.
- **Soft rules (shown, overridable with a reason):** ARPU price band, action → generosity, bandit top
  action. Every override is logged and counted.
- **Propensity (off-policy safe):** with probability ε_llm (default 0.1) the logged decision is a
  bandit-sampled action (propensity = pmf[a]); otherwise the LLM decision, logged with
  propensity = 1 − ε_llm + ε_llm × pmf[a]. `policy_source` ∈ bandit / ranker / llm / fallback.
- **Offline mode:** the fallback reproduces the cstr-arm logic exactly → llm arm = cstr arm (tested).
- **Batch:** async with a concurrency limit (vLLM), resume from a JSONL checkpoint (Ollama), cache keyed
  on (feature hash, catalogue version, model version).

---

## 7. Feedback loop and simulation

- `nba_campaign_log` (append-only parquet, `data/fsdr/state/campaign_log.parquet`): run_date, arm, msisdn,
  action, offers (codes, prices, validity), policy_source, propensity, bandit pmf, model version, LLM
  model, justification id. JSON export from the API/UI.
- `feedback.py` (D+7): joins fulfillments → reward under the 3 definitions → `nba_rewards` parquet.
- New features for the next scoring: `PREV_ACTION`, `PREV_OFFER_TAKEN`, `DAYS_SINCE_LAST_BTL`,
  `NB_CONTACTS_LAST_4W`. They are in the feature list, so the compatibility check forces retraining.
- Anchor escalation guard: `FSDR_ANCHOR_GROWTH_CAP` (default 1.25× week on week) or
  `FSDR_ANCHOR_EXCLUDE_CAMPAIGN` (default off).
- `simulate.py --weeks N --arms cg,free,cstr,llm --scenario ...`: generate D → train → score → recommend
  all arms → draw responses from the hidden function → write the next week's fulfillments → evolve the
  world → D += 7. Weekly table: take rate, revenue, uplift vs control, override rate, drift.

---

## 8. API, CLI, UI

### 8.1 CLI (`run.py`)

`fsdr-generate --date --n --scenario a,b --seed` · `fsdr-validate --data-dir` · `fsdr-schemas` ·
`fsdr-train --data-dir --mode --reward` · `fsdr-score --population target|base --policy sample|greedy` ·
`fsdr-recommend --arm free|cstr|llm` · `fsdr-feedback` · `fsdr-simulate --weeks --arms --scenario` ·
`fsdr-report --week`.

### 8.2 API (`/fsdr/*`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/fsdr/generate` | dummy data for a date / size / scenarios |
| POST | `/fsdr/upload`, GET `/fsdr/upload/status/{job}`, POST `/fsdr/upload/retry/{job}` | folder / zip upload, same retryable steps as `/upload` |
| POST | `/fsdr/validate` | validation report |
| GET | `/fsdr/tables/{run}/{table}?limit=` | table preview |
| POST | `/fsdr/train`, `/fsdr/score`, `/fsdr/recommend`, `/fsdr/feedback`, `/fsdr/simulate` | pipeline steps (deterministic or agent-driven via `agent=true`) |
| GET | `/fsdr/stream?job=` | SSE agent trace (same event types as `/agentic/stream`) |
| POST | `/fsdr/recommender/decide` | one customer, llm arm, full trace |
| GET | `/fsdr/runs`, `/fsdr/models`, `/fsdr/campaign-log`, `/fsdr/report/{week}` | history, registry, log, report |
| POST | `/fsdr/models/{version}/promote`, `/fsdr/models/rollback` | registry actions |

### 8.3 UI: 5 tabs after the existing 12, same theme and components

| Tab | Contents |
|---|---|
| FSDR Data | generate (size, date, scenarios) · upload folder/zip · per-table preview · validation report |
| FSDR Training | run with the agent trace (same trace renderer as the council tabs) · metrics · registry · promote / rollback |
| FSDR Scoring & Offers | per arm: action mix, offers with anchor and pass · CSV download |
| FSDR LLM Recommender | single customer: trace, overrides, hard-rule repairs · batch with progress |
| FSDR Simulation | weekly take rate / revenue per arm, uplift vs control, override rate, drift |

---

## 9. Test plan (all offline, fast; added under `tests/fsdr/`)

| Area | Tests |
|---|---|
| Schemas | every generated file matches `schemas.py` and the exported JSON Schema (names, case, raw types) |
| Round trip | generator → JSON (array and JSONL) → `JsonSource` → same dataframes as in memory |
| Calendar | T and every BR §4 window for a Thursday, a Wednesday, a January rollover |
| Generator consistency | `Sub_Amt_m1` ≈ Σ weeks over the same days; W2(D) = W1(D-7) |
| Scenarios | each scenario moves its expected signal; `data_quality_issues` caught by validation |
| Model registry | a 12th dummy model added through config reaches features and model metadata |
| Feature parity | train path = score path on the same raw data; equals `nba_lib.build_features` on a small fixture (with `dataiku` stubbed) |
| Business rules | eligibility, segment map (incl. mixed case), rewards (3 defs), ARPU anchor, each price pass, families (incl. CROSS SELL VOICE), conflicts, max offers |
| Bandit | stable action order, save/load gives identical pmf, incompatible model refused, `name:value` format |
| Agents (fake LLM) | justified override accepted and logged; hard-rule violation repaired by the critic; propensity formula; offline fallback = cstr arm; `NO_CONTACT` logged |
| Simulation smoke | 3 weeks × 2,000 customers < 60 s; llm/bandit arm take rate ≥ control on a world with a known signal |
| Regression | the existing 104 tests stay green; suite time target < 90 s |

---

## 10. Phases

| # | Scope | Done when |
|---|---|---|
| 1 | **Plan** (this file) | approved |
| 2 | `fsdr/config`, `schemas` (+ JSON Schemas), `calendar`, `datasource`, `validation`, `synth/*`; `fsdr-generate`, `fsdr-validate`, `fsdr-schemas` | schema, round-trip, calendar, consistency, scenario and registry tests green; 5k generation < 60 s |
| 3 | `core/vw_format`, `features`, `labels`, `bandit`, `registry`, `pipelines/train`, `pipelines/score`; `fsdr-train`, `fsdr-score` | parity, bandit, label tests green; training report + action distribution on dummy data |
| 4 | `offer_ranker`, `offer_rules`, `pipelines/recommend` (free, cstr) | business-rule tests green; recos with anchor + pass |
| 5 | FSDR tools, data / training / scoring / monitor agents, recommender (llm arm), `react.py` registry hook, council hooks | agent tests with fake LLM + offline fallback green |
| 6 | `campaign_log`, `feedback`, last-campaign features, `simulate`, `report` | simulation smoke test green; take rate per arm per week |
| 7 | API router, CLI completion, 5 UI tabs, FSDR rebrand, docs | full suite green; app starts; endpoint / tab / changed-file list |

At the end of each phase: full test suite, this file updated, short report, then **stop**.

---

## 11. Open questions, with the default I will use

| # | Question | Proposed default |
|---|---|---|
| Q0 | Git: Step 1 of the guide creates a branch and commits. This folder is inside the parent repo, which has unrelated uncommitted changes | **I don't commit.** You commit after each phase (or tell me to create `fsdr-agentic-nba` and commit only `nbx-python-upgrade/`) |
| Q1 | BR-39 CHECK: `Retention_High_Probability_To_Reconnect` → RETAIN or UPSELL? | **RETAIN** (last value Python kept, as in `nba_lib`). Configurable |
| Q1b | BR-39 CHECK: `RGS_Not_At_Risk_Data_Only_Voice_Appetency` → UPSELL VOICE or CROSS SELL VOICE? | **UPSELL VOICE** (as in `nba_lib`). Configurable |
| Q2 | Targeting (D-8→D-2) and reward (D-7→D-1) windows overlap | Keep both windows and the production rule: any fulfilment in D-7..D-1 counts (BR-40). The option `FSDR_REWARD_AFTER_TARGETING=true` also requires it on or after `date1` (changed to the production default after the 2026-10-24 audit, H2) |
| Q3 | Generosity forced to L2 for every action (temporary in production) | Switch `FSDR_GENEROSITY_MAPPING = all_L2` (default, = production today) \| `normal` |
| Q4 | Reward definition used for training | `any_btl` (BR-40, parity). All 3 are computed and logged; the training agent may choose among them |
| Q5 | Two revenue definitions (NBA: PAYG data; pricing: PAYG total) | Keep both exactly as production (BR-49, BR-68); documented in the BR map |
| Q6 | ARPU anchor = mean + std (pushes above spend) | Keep (BR-71), `ddof=1`; config switch for the old `min(mean, max)` method |
| Q7 | Fixed 90 % / 10 % data/voice split | Keep (BR-74/75) as constants; a `usage_mix` option is listed for later, not built |
| Q8 | Contact-frequency cap (not in the BRs) | max 1 contact per 7 days and 3 per 28 days per MSISDN |
| Q9 | Churn threshold that blocks upsell-family actions | `PROBA_CHURN ≥ 0.60` blocks UPSELL, UPSELL DATA, UPSELL VOICE |
| Q10 | `NO_CONTACT` is not one of the 7 VW actions (BR-38) | Logged with its propensity and counted in the monitor; **not** used as a VW training example (the 7-action model stays compatible) |
| Q11 | Existing `core/nba_bandit.py` (NBX demo) has the `name=value` and action-order issues | Leave its behaviour unchanged (rule "existing features keep working"), share the corrected format code. Say "fix it too" and I switch it to `name:value` in Phase 3 |
| Q12 | LightFM is not installable here (Python 3.12) | `ImplicitBPRRanker` default; `LightFMRanker` import-guarded, chosen with `FSDR_RANKER=lightfm` when installed. BR-84's LightFM parameters are kept in config for it |
| Q13 | Rebrand of the competitor module: the scraper URLs contain real operator domains | Display labels `COMPETITOR_A` / `COMPETITOR_B` everywhere; the shipped snapshot relabelled; scraper source URLs moved out of code into an optional, git-ignored `data/competitors/sources.local.json` (scraping off until it exists) |
| Q14 | Currency "OMR" and nationality "Oman" | Kept: they are data values in the business rules (BR-15, amounts), not operator names. Currency label is a config value |
| Q15 | Prior counts mismatch (inconsistency 6) | Priors only used as last-resort propensity; values kept as given |
| Q16 | Target sample table name changes per campaign | `FSDR_TARGET_SAMPLE_TABLE = "sample_cg_tg_new_7oct2026"` |
| Q17 | Simulation "control" take rate | Organic purchase rate of the `cg` arm (no contact) from the hidden world |
| Q18 | Model age limit for scoring | 14 days |

## 12. Risks

| Risk | Mitigation |
|---|---|
| ~~Segfault when pandas is loaded with other native libraries~~ **Found and fixed in Phase 2:** importing `pyarrow` (or pandas) **before** `vowpalwabbit` segfaults on this Windows setup; the other order works | `nbx/__init__.py` imports `vowpalwabbit` first, so every entry point is safe. Guarded by `test_pandas_and_pyarrow_load_safely_with_the_api` |
| Test-suite time grows | small fixtures (≤ 500 lines); the 3 × 2,000 simulation test marked `slow` (still run by default, < 60 s) |
| 200k-line generation memory | chunked daily simulation (§4.1) |
| Prompt size for small local LLMs | compact `customer_360` observation (top signals only), same 900-char cap as today |
| Real-data quirks unknown until a sample arrives | validation is warning-first for non-key issues; add `docs/fsdr_reference/json_samples/` when you can |

## 13. Progress log

| Date | Phase | Summary |
|---|---|---|
| 2026-10-22 | 1 | Folder created, references copied, baseline 104/104 tests, `run.py` test-argument fix, plan written. Waiting for approval |
| 2026-10-22 | 2 | Data layer done. **167/167 tests** (104 existing + 63 new in `tests/fsdr/`). Details below |

### Phase 2 report

**New files:** `nbx/fsdr/{config, calendar, schemas, datasource, validation, cli}.py`,
`nbx/fsdr/synth/{world, scenarios, generator}.py`, `docs/fsdr_json_schemas/` (45 schemas + `_index.json`),
`tests/fsdr/` (7 test files).
**Changed:** `nbx/__init__.py` (import-order fix), `run.py` (`fsdr-generate`, `fsdr-validate`, `fsdr-schemas`),
`.env.example`, `.gitignore` (**`data/fsdr/` ignored**, so real exports are never committed), `requirements.txt` (`jsonschema`, `pyarrow`).

**Results**
- 45 tables per run date, exact production names, case and raw types (string numbers in `btl_catalog` and the
  sample, bigint MSISDN in score files, double `ACCOUNT_ID` in `subs_amt_*`).
- 5,000 lines generated in **about 1 s** (target < 60 s); JSON and JSONL round trip is exact.
- Windows consistent by construction: activities M1 = W1+..+W4 (to rounding), `Sub_Amt_m1` ≥ its weeks,
  W2(D) = W1(D-7) (tested on two dates of the same world).
- Scores correlate with the hidden truth (churn 0.88, data appetency 0.89 on 5k lines), coverage 70–90 %,
  early-churn files limited to tenure 0–30 / 30–60.
- Last week's campaign: ~10–16 % take rate on targeted lines vs ~5 % organic; non-BTL noise products present.
- Validation: clean dummy data = `ok` with 0 issues; `data_quality_issues` = `blocked`, with all 4 injected
  issue types reported. A 12th model added through config flows into schemas, generator and validation.
- `fsdr-generate` refuses to overwrite a folder that holds non-generated data unless `--force` is given.

**Decisions taken while building (no change to the approved plan)**
- Daily history spans **D-96..D** (97 days), not D-95: `paygo_activity_auto` M3 starts at T-90 and T can be D-6.
- Test sample: production has no `strata` / `counts`; they are added as strings (like every sample column).
  `Final` has no documented meaning, so it is `CG` / `TG`.
- Regions are the 11 governorates (`North Batinah` drives `churn_wave`); wilayats and areas are generic labels (`WILAYAT_013`, `AREA_0042`); plans are dummy
  `FSDR ...` names plus the 13 excluded plans of BR-10.

**What to check:** `python run.py fsdr-generate --n 2000 --scenario baseline`, then
`python run.py fsdr-validate --data-dir data/fsdr/<date>`, and open `_validation.md` in that folder.

### Phases 3–5 report (2026-10-23)

Training, scoring, offers and the three recommender agents. **207/207 tests** (104 existing + 103 FSDR).

The user asked for the action and offer agents, a competitor agent, and training / scoring on the new data
in one step, so phases 3, 4 and the recommender part of 5 were done together.

**New files:** `nbx/core/vw_format.py`; `nbx/fsdr/{features, labels, bandit, registry, offer_rules, offer_ranker}.py`;
`nbx/fsdr/pipelines/{__init__, train, score, recommend}.py`; `nbx/fsdr/agents/{tools, recommender_agent}.py`;
`tests/fsdr/test_fsdr_{features, labels_bandit, pipelines, offers, agents}.py`.
**Changed (backward compatible):** `agents/agentic/react.py` and `tools.py` (optional tool `registry` argument),
`core/collab_filter.py` (`fit_bpr`, `bpr_scores` extracted; optional `num_threads`), `fsdr/config.py`, `fsdr/cli.py`, `run.py`.

**Commands:** `fsdr-train`, `fsdr-score`, `fsdr-recommend --arm cstr|free|llm`, `fsdr-decide --msisdn N --trace`.
Outputs go to `data/fsdr/<D>/outputs/` (training report and set, scored base, `results_scoring_nba`, per-arm
`TG_*_scored_nba`, `recos_monthly|weekly_<arm>`, `llm_decisions.jsonl`, reports). Model registry: `data/fsdr/state/models/`.

**Results on the 2,000-line baseline run (D = 2026-10-23)**
- Training: 787 labelled rows (Nursering excluded, mixed-case segments mapped); holdout IPS take rate of the
  model policy 0.133 vs 0.086 for the logged (segment) policy. Scoring: 3 arms scored, control not scored,
  max PSI 0.03.
- cstr: 178 / 202 customers with offers (639 offers); free: 155 / 197 (567). The rest have an anchor above every
  allowed price (BR-82).
- llm arm, offline: 196 decisions in 3.5 s, **identical to the constraint-arm logic for every customer**
  (tested); competitor verdicts 131 competitive / 25 mixed / 2 uncompetitive.

**Fixes from the reference bug list, all tested:** `name:value` VW features, action tokens without spaces,
fixed action order (same as `nba_lib`), model saved and loaded (never finished before scoring), per-row missing-score
fill, older → recent trends, one key type, ranker predicts by the user's own id, CROSS SELL VOICE family branch,
conflicts enforced. Feature values match `docs/fsdr_reference/nba_lib.py` exactly on a fixture.

**Decisions taken while building**
- Contact cap (Q8) adjusted: weekly campaigns put last week's contact at D-8..D-2, so "1 per 7 days" would block
  almost everyone. Hard rule = minimum 2-day gap since the last contact; the 3-per-28-days cap needs the campaign
  log (Phase 6).
- Hard rules on the *action* (churn block, contact gap) apply to LLM decisions. The offline fallback reproduces
  the constraint arm exactly, as the prompt requires. Offer hard rules (catalogue, price, ≤ 2 per validity, ≤ 4,
  conflicts) apply in every arm.
- BPR ranker pinned to one thread (`BPR_PARAMS.num_threads = 1`) so the same data always gives the same ranking.
- VW models are loaded with `arg_list` (model paths with spaces broke `-i` in the argument string).
- The critic is code-only for now (hard-rule findings, auto-repair, approve / revised). An LLM review of the
  justification quality can be added in Phase 5b.

**Not done yet:** data / training / scoring / monitor agents (Phase 5b), campaign log + feedback + simulation
(Phase 6), the Simulation tab, uploads of real JSON from the UI, and the full rebrand of the existing NBX screens (Phase 7).

### Phase 7, first part (2026-10-23): API and UI

Requested together with the commit, so the new logic runs in the app. **211/211 tests.**
- **API** `nbx/api/fsdr.py` (an `APIRouter`, mounted by one line in `main.py`): `GET /fsdr/runs`, `POST /fsdr/generate`,
  `POST /fsdr/validate/{run}`, `GET /fsdr/tables/{run}[/{table}]`, `POST /fsdr/train/{run}`, `POST /fsdr/score/{run}`,
  `GET /fsdr/models`, `POST /fsdr/models/{version}/promote`, `POST /fsdr/models/rollback`, `POST /fsdr/recommend/{run}`,
  `GET /fsdr/recos/{run}/{arm}`, `GET /fsdr/customers/{run}`, `POST /fsdr/decide/{run}`, `GET /fsdr/stream/{run}` (SSE).
- **UI** `nbx/ui/fsdr_tabs.py`, 4 tabs after the 12 NBX tabs, same theme and renderers: **FSDR Data**, **FSDR Training
  & Scoring**, **FSDR Offers** (CSV export), **FSDR Agentic Recommender** (live trace of the action, offer and
  competitor agents). `theme.py` gained the three agent labels and accepts string findings / offer names.
- Tests: `tests/fsdr/test_fsdr_api.py`; the UI build itself is covered by the existing Gradio test.

### Batch flow (2026-10-23): action → top 4 offers → top 2 roaming → international

Requested by the user. **218/218 tests.**
- `nbx/fsdr/addons.py` (roaming / international add-on engine), `nbx/fsdr/pipelines/batch.py`, `POST /fsdr/batch/{run}`,
  `python run.py fsdr-batch`, and a **Batch** section in the FSDR Agentic Recommender tab (processed batch by batch, CSV export).
- The dummy catalogue gained 6 `ROAMING` and 6 `INTERNATIONAL` offers (66 in total). The hidden world gained a
  `traveller` trait that drives roaming purchases, and international purchases follow the share of international calls.
- Rules (not in the BR catalogue, documented in FSDR_GUIDE §6): add-ons sit beside the 4 main offers (BR-82 applies
  to the main offers only), no duplicate or conflict with a main offer; international only with ≥ 3 international
  calls in M1 (`Call_Count_M1 − Local_Call_Count_M1`). The data contract has no roaming-usage column, so roaming is
  ranked on past roaming purchases, with popularity blended with the price fit to the customer's anchor for cold start.
- The 2026-10-23 run was regenerated with the new catalogue, retrained (champion `v0003`) and rescored.
- FSDR reads only `data/fsdr/<date>/` (plus the public competitor snapshot); the old NBX demo data is used only by the
  12 original tabs (FSDR_GUIDE §1).

### vLLM for the agents (2026-10-23)

- The agents run on **vLLM** on this laptop: vLLM 0.6.6.post1 in WSL2 Ubuntu, on the GTX 1650 (4 GB), serving
  `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` as `qwen2.5:3b`. `python run.py vllm-install` and `python run.py vllm-serve`
  (scripts in `vllm/`) set it up without sudo. `.env` (git-ignored) holds `LLM_BACKEND=vllm` and a random API key.
- Measured: AWQ ran at 4 tokens/s and GPTQ at 13–14 tokens/s. One decision took 553 s with AWQ and 310 s with GPTQ,
  then **25–150 s** with the evidence prefetch. A batch of 10 on the llm arm took 500–800 s; all 10 decisions were
  `mode: agentic`.
- `FSDR_AGENT_PREFETCH` (default on): the code runs the required evidence tools before the first LLM call and hands
  over the results. The trace is unchanged, and the LLM only reasons and decides.
- `contact_policy` now also returns a plain `verdict` sentence. The 3B model had read the churn-block rule as "high
  risk": NO_CONTACT went from 6 to 1 in the same 10 customers.
- `FSDR_AGENT_CONCURRENCY` (8 on vLLM, 4 otherwise) is used by the batch runs.
- Docs: FSDR_AGENTIC_RECOMMENDER §1, GPU_SETUP §14 (WSL2 on a 4 GB GPU), FSDR_GUIDE, RESUME_VLLM.

### Verification audit and fixes (2026-10-24)

An independent audit (`docs/fsdr_verification_report.md`, 102 checks) found 1 BLOCKER and 21 MAJOR items. All fixed on
branch `fix/fsdr-business-rules`, with a test each:

| Audit | Fix |
|---|---|
| B5 / E5 (BLOCKER) | `access_method_id_auto.MSISDN` is used as is: production SQL builds it as `ACCESS_METHOD_VAL as MSISDN` (BR-12 runs upstream). The loader stripped 3 characters, so contract data matched no targeted line |
| B6 | fulfilment lines unknown to the base are reported |
| C2 / C3 | `fsdr-generate --weeks N` from one world (W2 of D+7 = W1 of D); scores follow the run date (churn wave rises); North Batinah; 3 new offers; null `ACCESS_METHOD_VAL` injected |
| H2 | production reward by default (any BTL in D-7..D-1) |
| H5, M1–M4 | campaign log, D+7 feedback and `nba_rewards`, logged-propensity labels, last-campaign features, anchor growth guard |
| L1 | data / training / scoring / monitor agents with code guards |
| L4 | 28-day contact cap from the campaign log |
| L6 | logged propensity correct when exploration draws the LLM's own action |
| N1–N3 | `fsdr-simulate`: closed loop, about 45 s for 3 weeks × 2,000 lines, all arms above control |
| O1–O3 | `fsdr-feedback` / `fsdr-simulate` / `fsdr-report` / `fsdr-agent`, API routes, FSDR Simulation tab, real-JSON upload |
| A8 | operator `OPERATOR_NAME`, competitors COMPETITOR_A / B, snapshot anonymised, scraper sources local |
| A3 | one VW format layer for both bandits (the NBX demo bandit wrote numbers as `name=value`); cohort tool on `core/knowledge_graph` |
| offer filter | ranker proposes only offers bought in the filtered history (production `interaction1.BTL_NAME.unique()`) |
