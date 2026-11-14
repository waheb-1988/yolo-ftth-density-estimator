# FSDR – NBA: two separate pipelines

```
                ┌──────────────── PIPELINE 1 : TRAINING (weekly, Thursday T) ───────────────┐
SQL *_auto  ──► │ 1 read  → 2 features → 3 labels → 4 train VW → 5 evaluate → 6 save model  │──► nba_model (folder)
                │                                                       → 7 score full base │──► nba_scored_base
                └────────────────────────────────────────────────────────────────────────────┘
                ┌──────────────── PIPELINE 2 : SCORING ONLY (campaign start) ────────────────┐
nba_model   ──► │ 1 load model → 2 population → 3 fresh features → 4 predict → 5 write       │──► results_scoring_nba
SQL *_score ──► │                                                                            │──► TG_40k_free / TG_40K_constraint
                └────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼  (unchanged)
                     Code 3 LightFM constraint arm   /   Code 4 LightFM free arm
```

Files
- `nba_lib.py` → project library (`lib/python/`). Shared feature engineering, VW format, model save/load.
- `01_nba_train.py` → Python recipe "NBA training".
- `02_nba_score.py` → Python recipe "NBA scoring" (replaces old Code 1 scoring part + old Code 2).
- Create once a managed folder named **nba_model**.

## Pipeline 1 – Training: step by step

| Step | What happens | Inputs |
|---|---|---|
| 0. Extraction (SQL) | Anchored on last Thursday T | `base_prepaid`, `party_auto`, `access_method_id_auto`, `btl_catalog`, `targeting_dataiku_auto` (D-8→D-2), fulfillment last 7 days (`subs_btl_auto`), `activities_w1..w4_auto`, `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto`, `subs_amt_auto`, 11 model-score tables |
| 1. Read | Upper-case columns, same numeric type for all join keys | |
| 2. Features (whole base) | 9 probabilities (missing → random 0–0.02 per row), revenue = subscriptions + PAYG data, % / log / zero-flag trends for revenue, voice, data, nationality group | |
| 3. Labels | segment → action (Nursering excluded), REWARD = took a catalogue BTL, COST = −REWARD, PROPENSITY = this week's action share | |
| 4. Train | VW `--cb_explore_adf -q UA --epsilon 0.2`; mode `logged` (all targeted) or `simulation` (original, takers only) | |
| 5. Evaluate | 20 % holdout: current take rate vs IPS estimate of the model policy | |
| 6. Save | model + metadata (+ dated archive) in `nba_model` | |
| 7. Score full base | greedy action + probability for every prepaid line | → `nba_scored_base` |

## Pipeline 2 – Scoring only: step by step

| Step | What happens | Inputs |
|---|---|---|
| 0. Extraction (SQL) | Anchored on CURRENT_DATE | `party_scoring`, `subs_amt_score`, `activities_w1..w4_score`, `activities_m1..m3_score`, 11 model-score tables, target sample or `base_prepaid` |
| 1. Load model | Refuses to run if features/actions differ from training | `nba_model` |
| 2. Population | `target` (free + cstr arms) or `base` (full base) | |
| 3. Features | Same function as training | |
| 4. Predict | `sample` (exploration, original behaviour) or `greedy` | |
| 5. Write | all lines + split free / constraint | → Code 3 / Code 4 |

## SQL changes needed

1. **`activities_w1..w4_score`**: remove the hard-coded `ORIGINATING_ACCESS_METHOD_ID IN (...)` list, otherwise scoring the full base (or a new target) gets zero usage for everyone outside that list. If the list is kept to save time, generate it from the target table with a join instead.
2. **`activities_m1_score`** appears twice; keep only the CURRENT_DATE version (the other is anchored on Thursday T).
3. **`activities_m1_auto`** has no schema listed; make sure it outputs the same columns as the other activity tables.
4. Scheduling: Pipeline 1 only after `targeting_dataiku_auto` (D-8→D-2) and the fulfillments of D-7→D-1 are complete.

## Fixes versus the original code
- `vw.finish()` was called before scoring → model now saved, scoring uses a loaded model.
- Action list was shuffled during training while the model's probabilities are in fixed order → no shuffle.
- Numeric features written as `name=value` (read as categories by VW) → `name:value`.
- Action names with spaces broke VW tokens → `CROSS_SELL_DATA` etc.
- Trend direction inverted (W1 is the most recent week) → change now from older to recent; % change clipped to [−100, 500].
- Missing probabilities: one random value for all rows → one per row.
- `SUB_AMT_W1` unquoted, `.iloc[0][0]`, duplicate `proba_frag_voice`, string vs integer keys → fixed.
- Duplicate segment keys in the action dictionary → single lower-case dictionary; 2 mappings marked **CHECK**.
- Only the columns the model uses are loaded (less memory); location and device segment were merged but never used.
