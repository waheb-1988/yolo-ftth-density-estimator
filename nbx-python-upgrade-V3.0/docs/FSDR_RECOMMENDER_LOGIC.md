# FSDR Recommender — the complete logic, from the files to the offers

This document follows one weekly run from the JSON files to the offers sent and the rewards measured the
following week. Every join, filter, formula and threshold is listed, with the business rule (BR-xx) and the code
that implements it. The values are the V1.0 defaults; each `FSDR_*` switch is listed in `.env.example`.

```
 46 JSON tables (data/fsdr/<D>/)
   │  1 read          JsonSource: raw values for validation, typed columns for the pipelines
   │  2 validate      contract, types, keys, duplicates, joins, coverage, windows ──► blocked? stop
   ▼
 eligible base ── joins: line → account → party (nationality), line ↔ MSISDN, scores, usage, revenue
   │  3 features      61 per line: 9 scores (+9 missing flags), 39 trends, 3 last-campaign, tenure; 2 categorical
   │  4 labels        last week's campaign: action sent, reward (BTL bought D-7..D-1), propensity, uplift baseline
   │  5 train         Vowpal Wabbit contextual bandit; holdout evaluation; promotion rule; registry
   │  6 score         target sample (cg / free / cstr / llm) → probability of 7 actions → sampled action
   ▼
 offers ── 7 cstr / free arms (rules)            8 llm arm (agents, same rules as hard limits)
   │        families (BR-66/67), ARPU anchor,       action agent → offer agent → competitor agent
   │        value ranking or price bands,           → critic (code) → messenger (SMS)
   │        ≤ 2 per validity, ≤ 4, no conflict
   │  9 add-ons       roaming top 2, international (if the line calls abroad)
   │ 10 competitors   each offer vs the closest COMPETITOR_A / B plan
   ▼
 campaign log ──► next week: 11 feedback (rewards), 12 report (uplift vs control with CI, fairness)
```

---

## 0. The calendar

| Symbol | Meaning | Rule |
|---|---|---|
| **D** | run date: the day the files are exported and the cycle runs | folder name `data/fsdr/<D>/` |
| **T** | campaign anchor: the last Thursday on or before D | `T = D − (((D − 2026-03-12) mod 7) + 7) mod 7` |
| snapshot | base snapshot date | D − 2 (BR-01) |

Every aggregation window, inclusive (`nbx/fsdr/calendar.py`):

| Tables | Anchored on | W1 | W2 | W3 | W4 | M1 | M2 | M3 |
|---|---|---|---|---|---|---|---|---|
| `subs_amt_auto` / `_score` (BR 4.1, includes the anchor day) | T / D | −6…0 | −13…−7 | −20…−14 | −27…−21 | −29…0 | −59…−30 | −89…−60 |
| `activities_*_auto` / `_score` (BR 4.2, excludes the anchor day) | T / D | −7…−1 | −14…−8 | −21…−15 | −28…−22 | −28…−1 | −56…−29 | −85…−57 |
| `subscriptions_tg` (BR 4.3, 2-day lag) | D | −8…−2 | −15…−9 | −22…−16 | −29…−23 | −31…−2 | −61…−32 | −91…−62 |
| PAYG in `usage_data_voice_payg_tg` | D | −7…−1 | −14…−8 | −21…−15 | −28…−22 | −30…−1 | −60…−31 | −90…−61 |

Campaign windows, relative to D (BR-21…24):
- **targeting:** last week's campaign `date1` falls in D−8…D−2;
- **arbitration:** D−7…D;
- **reward:** fulfilments from D−7 to D−1;
- **interaction history:** D−270…D−1.

**Training** reads the T-anchored tables (`*_auto`, `party_auto`): the context as it was when last week's campaign
was decided. **Scoring** reads the D-anchored tables (`*_score`, `party_scoring`). The same feature function serves
both, so they cannot drift apart.

---

## 1. Reading the data (`nbx/fsdr/datasource.py`)

- **One file per table** (46 tables, `nbx/fsdr/schemas.py`), named exactly after the table: `<table>.json` (a JSON
  array) or `<table>.jsonl` (one record per line). The format is detected from the content.
- Production typos are kept on purpose (`UBSCRIPTION_ID`, `activities_auto_m2`, `Price data`), and one alias is
  accepted: `activities_w1_score_hfds` → `activities_w1_score`.
- `_manifest.json` (optional) gives the run date. Without it, the run date is the folder name.
- **Raw frame:** the records as JSON has them, with no type inference, so an int column with nulls stays ints.
  Schema columns come first.
- **Typed frame** (V1.0, `JsonSource(compact=True)`), used by the pipelines once the data is validated:
  - all-int columns → int64; ints with nulls, or ints mixed with floats → float64; text stays text;
  - about 4× less memory;
  - one typed copy of the run per process while its files are unchanged, so validate, train, score and offers parse
    the JSON once a week instead of four times.

## 2. Validation (`nbx/fsdr/validation.py`)

**Per table:**
- file present: a missing required table is **blocking**; a missing model-score file is a warning (its score falls
  back to random 0–0.02, BR-47); a missing optional table is a warning;
- rows > 0;
- every contract field present: missing fields are **blocking** for model tables;
- raw type of every value: string / int / bigint / tinyint / double / date `YYYY-MM-DD` / ISO timestamp. A wrong type
  is **blocking** on a key field;
- null keys: **blocking** above 1 % of rows;
- duplicates on the primary key: **blocking** above 1 %, unless duplicates are expected (targeting, model scores).

**Relations:** every key of the child table must exist in the parent.

| Child → parent | Key | Blocking above 1 % orphans |
|---|---|---|
| base_prepaid → access_method_id_auto | ACCESS_METHOD_ID | yes |
| base_prepaid → party_auto | PARTY_ID | yes |
| base_prepaid → consumer_aact_account | ACCOUNT_ID | no |
| targeting_dataiku_auto → access_method_id_auto | access_method_id | yes |
| test sample → base_prepaid / consumer_aact_account | access_method_id / account_id | yes |
| subs_amt_auto, activities_w1_auto, subs_btl_auto, sub_btl_3months, each model score file → base | ACCESS_METHOD_VAL / MSISDN | no |
| targeting offers 1–6, conflict pairs → btl_catalog | btl_name | no |

**Coverage and consistency:**
- **coverage:** share of the base with each model score (warning below 70 %);
- **activities:** `Local_Voice_min_M1` = W1 + W2 + W3 + W4 (tolerance 0.05 + 0.1 %);
- **subscriptions:** `Sub_Amt_m1` ≥ W1 + … + W4;
- **dates:** `date1` in D−8…D−2 and fulfilment dates in D−7…D−1;
- **MSISDN format:** 8 digits and > `69999999`.

The status is `blocked` if any issue is blocking (the run stops), else `warnings` or `ok`. Reports are written to
`_validation.json` / `.md`.

## 3. The eligible base and the joins (`features.eligible_base`, `labels.access_method_map`)

1. `base_prepaid` (snapshot D−2), with keys normalised:
   - an MSISDN is always an 8-digit string: `91234567`, `91234567.0` and `"91234567"` are the same;
   - ids are Int64.
2. Drop rows with a null `ACCESS_METHOD_VAL` / `ACCESS_METHOD_ID`.
3. **BR-02:** keep MSISDNs that are 8 digits and > `"69999999"` (string comparison).
4. **BR-10:** drop the 13 internal / employee / business plans (`config.EXCLUDED_PLANS`).
5. **BR-08:** one row per MSISDN: the latest `ACCESS_METHOD_SUBS_START_DT`.

Joins used everywhere:

| Join | Key | Gives |
|---|---|---|
| line → `consumer_aact_account` | ACCOUNT_ID | PARTY_ID (BR-11) |
| line → `party_auto` (T) / `party_scoring` (D) | PARTY_ID | NATIONALITY |
| `access_method_id_auto` | ACCESS_METHOD_ID | MSISDN; the latest activation kept |
| each model score file | access_method_val | proba_1 = the score, plus tenure |
| `activities_<p>_<anchor>`, `subs_amt_<anchor>` | ACCESS_METHOD_VAL | usage and subscription revenue per period |
| `subscriptions_tg`, `usage_data_voice_payg_tg` | ACCESS_METHOD_VAL | ARPU anchor inputs (offers) |
| `targeting_dataiku_auto` | access_method_id → MSISDN | last week's campaign: segment, offers sent |
| `subs_btl_auto` | access_method_val | fulfilments D−7…D−1 (rewards) |
| `sub_btl_3months` | access_method_val | 270-day BTL history (ranker) |
| `DEVICEBEHAVIORSEGMENTATION.dv_segment` | ACCESS_METHOD_ID | device segment (cohort graph) |
| `consent_contact` (V1.0, optional) | access_method_val | dnd, marketing_opt_in |

A line without a row in a usage or revenue table gets 0 (BR-48).

## 4. Features (`nbx/fsdr/features.py`) — 61 numeric + 2 categorical

**1. Model scores (BR-46):** the 11 score files map to 9 features; rows mapping to the same feature are merged by
**max**:
- PROBA_CHURN: max of the 3 churn files (global, early churn 0–30 d, 30–60 d);
- PROBA_REACT_30, PROBA_REACT_15, PROBA_APP_DATA, PROBA_APP_VOICE, PROBA_CHURN_1D, PROBA_REV_DECR, PROBA_FRAG_DATA,
  PROBA_FRAG_VOICE.

**2. Missing score (BR-47):** filled with uniform(0, 0.02), drawn **per row**. The seed is the generator seed when
training, and D's day number when scoring. V1.0 adds `PROBA_<x>_MISSING` = 1 when the score was missing (9 flags).

**3. Usage per period** (W1–W4, M1–M3), summed per line:
- from `activities_<p>`: LOCAL_VOICE_MIN, DATA_VOLUME_MB, PG_DATA_AMOUNT;
- from `subs_amt`: SUB_AMT;
- missing → 0.

**4. Revenue:** `REV_<p> = SUB_AMT_<p> + PG_DATA_AMOUNT_<p>` (BR-49).

**5. Trends (BR-50…53):** for each pair (recent, older):
- `VAR = (recent − older) / older × 100`, set to 0 when older = 0, and clipped to [−100, 500];
- `LOG = ln(1 + recent⁺) − ln(1 + older⁺)`;
- `<k>_<older>_ZERO = 1` when older = 0.

The pairs:
- REV: W1/W2, W2/W3, W3/W4, M1/M2, M2/M3;
- VOICE (local minutes) and DATA (MB): W1/W2, W2/W3, W3/W4, M1/M2.

That is 13 pairs × 3 = 39 trend features; 9 scores + 39 trends = **production's 48**.

**6. Last campaign** (M3, from the campaign log):
- PREV_OFFER_TAKEN: the exact offer was taken after the last contact;
- DAYS_SINCE_LAST_BTL: capped at 270;
- NB_CONTACTS_LAST_4W: contacts logged in D−28…D−1, control excluded;
- PREV_ACTION (categorical): `NONE` if never contacted.

**7. Tenure** (V1.0): `LOG_TENURE = ln(1 + max tenure in the score files)`.

**8. Nationality group (BR-15):** Oman / India / Bangladesh / Pakistan, else `others`. This is a categorical model
input (`FSDR_USE_NATIONALITY=false` drops it).

`FSDR_EXTENDED_FEATURES=false` returns to production's 48 + 3 features.

## 5. Training labels (`nbx/fsdr/labels.py`)

**Week 0: legacy targeting tables**
1. `targeting_dataiku_auto` rows with `date1` in D−8…D−2, mapped to an MSISDN.
2. Drop rows without a segment (BR-37).
3. Drop the two Nursering segments (BR-36).
4. Map segment → action (BR-39, case-insensitive; 12 segments → 7 actions, e.g. `retention_rgs_at_risk` → RETAIN,
   `rgs_not_at_risk_voice_only_data_appetency` → UPSELL DATA).
5. Keep the latest targeting per line (BR-42).
6. Propensity = **1.0** (V1.0): the legacy policy is deterministic by segment, so those weeks are flagged "not
   usable for off-policy estimates".

**From week 1: the app's campaign log**
1. Rows logged in D−8…D−2, excluding control and NO_CONTACT; the latest per line.
2. Action = the one actually sent; propensity = the **logged** probability of that action (else the BR-45 prior).
   Propensities are clipped to [0.01, 1].

**Reward:** fulfilments in `subs_btl_auto` from D−7 to D−1, catalogue BTLs only (BR-23/25/40/63). Three definitions:

| Definition | Reward = 1 when the line… |
|---|---|
| `any_btl` (production, default) | bought any catalogue BTL |
| `same_service` | bought a BTL of a service that was offered |
| `exact_offer` | bought one of the offers sent |

**Cost** = −reward (BR-41), because VW minimises cost.

**Uplift (V1.0)** subtracts the control group's organic take-up:
- **baseline model:** a logistic regression of the control group's reward on the 61 features (with ≥ 20 control
  takers; else the control take rate, smoothed);
- **cost** = −(reward − baseline).

The bandit then learns what a contact *adds*, not who buys anyway.

The labels are joined to the features of the line built on T, with the last-campaign context as of D−7.

## 6. The NBA model (`nbx/fsdr/bandit.py`, `nbx/core/vw_format.py`, `pipelines/train.py`, `registry.py`)

- **Vowpal Wabbit** `--cb_explore_adf -q UA --quiet --epsilon 0.2` (BR-54): a contextual bandit with action-dependent
  features, a User × Action interaction, and ε-greedy exploration.
- **One example per line:**
  ```
  shared |User nationality_group=India prev_action=NONE proba_churn:0.104 var_rev_w1_w2:-17.98 ...   (zeros omitted)
  |Action product=CROSS_SELL_DATA
  0:-1:0.828571 |Action product=UPSELL_DATA        ← label "0:cost:propensity" on the action that was sent
  |Action product=RETAIN  …  (7 actions, always in this order, BR-38)
  ```
- **Training:** 80 % of the labelled lines, 2 passes, rows shuffled each pass (seeded). The holdout is a random 20 %.
- **Evaluation on the holdout (V1.0):**
  - IPS = mean(π(a|x) / p × r), SNIPS = Σ π/p·r / Σ π/p, where π is the probability the **deployed** policy gives the
    logged action (sampling, BR-57);
  - SNIPS gets a bootstrap 95 % CI (500 resamples);
  - greedy IPS is kept for reference;
  - plus the take-up of the logged policy, the share of takers' actions reproduced, and the action mix.
- **Promotion** is code-enforced; no agent can bypass it:
  - the champion is scored on the challenger's holdout (out-of-time for the champion);
  - a paired bootstrap of the SNIPS gain decides:
    - upper bound < 0 → **refused** (significantly worse);
    - lower bound > 0 → promoted (significantly better);
    - otherwise → promoted (the fresher model).
  - Weeks with deterministic propensities fall back to V0's point rule.
  - The first model is always promoted.
- **Registry:** `state/models/v000N/` holds the model + metadata (features, actions, categoricals, dates, metrics,
  drift bins). Writes are locked and atomic.
  - Loading refuses a model trained with other features, actions or categoricals.
  - Scoring refuses a model **older than 14 days**.
- **Base scoring:** each training also gives the whole eligible base its most probable action (greedy) →
  `outputs/nba_scored_base.json`.

## 7. Scoring the target (`pipelines/score.py`)

1. **Population:** the test-sample table (`sample_cg_tg_…`, produced upstream, BR-59…62), with `group` ∈ cg / free /
   cstr / llm.
   - **cg is not scored**: it is logged as "not contacted" so that every arm can be compared with it.
   - `TG_FLAG`: free → `free`, cstr → `with_constraint`, llm → `llm`.
2. **Features** on the D tables (section 4).
3. **PMF** of the 7 actions from the champion.
4. **Action:** draw u ~ U(0, 1) (seed = D's day number); the action is the first whose cumulative probability
   reaches u (BR-57). PROBA = its probability, which is logged as the propensity. The greedy action is kept beside
   it. One row per MSISDN (BR-58).
5. **Drift:**
   - PSI per feature against the training deciles (alert > 0.25);
   - total-variation shift of the action mix.
6. **Outputs:** `results_scoring_nba.json`, `TG_free / TG_constraint / TG_llm _scored_nba.json`.

## 8. Offers for the cstr and free arms (`offer_rules.py`, `pipelines/recommend.py`, `acceptance.py`)

**Catalogue (BR-63…65):**
- `btl_catalog` (all strings in production), numbers parsed;
- service ∈ DATA / VOICE / COMBO (+ ROAMING / INTERNATIONAL add-ons);
- generosity L0 < L1 < L2; validity 7 or 30 days.

**Conflicts:** `conflicting_offers_h` pairs are never sent together.

**ARPU anchor (BR-68…77):**
- REV_p = TOTAL_SUB_AMT_p (`subscriptions_tg`) + PG_TOTAL_AMOUNT_p (`usage_data_voice_payg_tg`), missing = 0;
- **monthly** anchor = mean(REV_M1…M3) + std(ddof 1); **weekly** anchor = the same over W1…W4;
- data anchor = 90 % of it, voice anchor = 10 %;
- which anchor an offer uses (BR-77): COMBO → total, DATA → data, VOICE → voice; 7-day offers use the weekly
  anchors.
- **Growth guard (V1.0):** an anchor is capped at 1.10 × the anchor logged for the line last week.

**Families (BR-66), with the active mapping `all_L2`:**

| Action | Family A | Family B |
|---|---|---|
| MAINTAIN, UPSELL, RETAIN | COMBO L2 | DATA L2 |
| UPSELL VOICE, CROSS SELL VOICE | DATA L2 | COMBO L2 |
| CROSS SELL DATA, UPSELL DATA | VOICE L2 | COMBO L2 |
| free arm (BR-67, action not used, BR-60) | COMBO L2 | DATA L2 |

**Ranker (BR-83…86):**
- BPR (implicit; 32 factors, 60 iterations, lr 0.05, reg 0.01) on the 270-day history `sub_btl_3months`
  (weights = sub_flag);
- one model per (action, validity) over that family's offers; with `RANKER_ITEMS=history`, only offers bought in the
  history;
- known lines: scores rescaled to [1, 2]; lines without history: item popularity in [0, 1] (BR-86);
- top 100 per line (BR-85).

**Selection: one offer per family and validity, monthly first, then weekly; ≤ 2 per validity, ≤ 4 in total (BR-82).**

*Value policy (V1.0 default, `FSDR_OFFER_POLICY=value`):*
- **Acceptance model:** logistic regression (L2 = 5), trained on every offer sent last week (campaign log, or legacy
  targeting in week 0). The label is 1 if that exact offer was fulfilled in D−7…D−1. Needs ≥ 20 accepted offers,
  else the price bands are used.
- **Inputs:**
  - the offer: service one-hot, generosity level, weekly flag, ln(1 + price), ln(price / the line's mean revenue for
    that validity), gap above that revenue;
  - fit: data appetite × data offer, voice appetite × voice offer;
  - the line's history: share of its 270-day purchases in that service and that validity, number of purchases;
  - the line: churn, data and voice appetite, ln(1 + monthly revenue);
  - the action: RETAIN × churn, data actions × data appetite, voice actions × voice appetite, and the 7 action
    one-hots.
- **Value** = P(accept) × price × `OFFER_MARGIN`, ranked high to low (then cheaper first). The whole family competes,
  so a brand-new offer gets a score from its attributes.
- **Floor** (`FSDR_OFFER_PRICE_FLOOR=anchor`): only offers priced above the line's anchor, as in production; no
  offer if the anchor is above every price (BR-82).
- **Exploration:** 5 % of picks are a random offer of the family (pass 5); the others are pass 4. Seeded per (D,
  line).

*Production's price bands (`FSDR_OFFER_POLICY=price_band`):*
- **pass 1:** anchor < price < anchor + k, where k = 3 OMR (UPSELL family, CROSS SELL, free) or 2 OMR
  (MAINTAIN, RETAIN); ranker score desc, then price desc;
- **pass 2:** if exactly one family found an offer, the missing family with price > anchor (no cap); score desc,
  then price asc;
- **pass 3:** if neither did, both families with price > anchor; score desc, then price asc.

**Hard rules, always:**
- catalogue offers only, price > 0, no duplicate, no conflicting pair;
- the limits of BR-82;
- **consent** (V1.0): `dnd = 1` or `marketing_opt_in = 0` → no offer, in every arm.

Outputs: `recos_monthly_<arm>.json`, `recos_weekly_<arm>.json`; each line is also appended to the campaign log.

## 9. The llm arm: three agents (`nbx/fsdr/agents/`)

The bandit, the rules and the ranker are **evidence**: the LLM decides, and every disagreement needs a written
reason. The code enforces the hard rules. Offline (no LLM), each agent runs its rule-based path, which equals the
cstr arm.

| Stage | Tools (✓ = required before answering) | Code rules applied to the answer |
|---|---|---|
| **Action agent** | customer_360 ✓, bandit_pmf ✓, contact_policy ✓, segment_policy, campaign_history | action ∈ 7 actions or NO_CONTACT; a reason when it differs from the bandit's top action; **contact policy**; churn ≥ 0.60 turns an UPSELL-family action into RETAIN |
| exploration | — | with probability 0.10 the action is replaced by a bandit sample; logged propensity = (1 − ε)·1[a = LLM action] + ε·PMF[a] |
| **Offer agent** | price_band_candidates ✓, check_offers ✓, ranker_top_offers, cohort_offers (knowledge graph), arpu_anchors, search_catalog, expected_value | 1–4 catalogue offers; ≤ 2 per validity; no conflict; a reason when they differ from the rule-based list; invalid offers dropped, or replaced by the rule-based list |
| **Competitor agent** | competitor_compare ✓, competitor_market, search_catalog | verdict competitive / mixed / uncompetitive; a swap is applied only if the new list passes every hard rule |
| critic (code) | — | final check of every hard rule on the final list |
| messenger | — | SMS in English and Arabic from `BTL_script_english_arabic_ok` |

The **contact policy** blocks contact if any of these hold:
- the line is not eligible (BR-02 / BR-10 / BR-36);
- it has no marketing consent;
- it was contacted less than 2 days ago;
- it already had 3 contacts in 28 days.

**Prompts** carry `C-<salted HMAC>` instead of the MSISDN.

**Knowledge graph:** a networkx cohort graph, customer → device segment → offers bought. The offer agent may call it
(`cohort_offers`, optional).

## 10. Add-ons: roaming and international (`nbx/fsdr/addons.py`)

- **Roaming:** the top 2 roaming packs.
  - Ranked by BPR on the line's roaming history.
  - Without history: 0.5 × popularity + 1 / (1 + |price − anchor| / max(anchor, 1)), where the anchor is the line's
    total anchor for that validity.
- **International:** the top 2 packs, only if the line made **≥ 3 international calls in M1** (CALL_COUNT_M1 −
  LOCAL_CALL_COUNT_M1).
- An add-on never conflicts with a main offer. BR-82's limit of 4 applies to the main offers only.

## 11. Competitor comparison (`nbx/core/competitors.py`)

- **Snapshot:** `data/competitors/competitor_offers.json`, 68 plans, labelled COMPETITOR_A / COMPETITOR_B.
- **Monthly equivalent** = price × 30 / validity.
- **Candidates:** for each operator, plans of a compatible category, the same segment (mass) and the **same validity
  band** (daily ≤ 3 days, weekly ≤ 14, else monthly). The plan with the closest monthly equivalent is kept.
- **Verdict per plan:**
  - price: cheaper / more expensive / same, with a ±0.25 OMR tolerance;
  - data: more / less / similar data per month, with a ±5 % tolerance.
- **Position:**
  - strong: we beat every plan on price or data;
  - weak: a competitor gives more data for less money;
  - otherwise mixed.
- **At risk:** our monthly price is more than 1.0 OMR above a competitor's.
- **Decision verdict:**
  - competitive: no offer at risk;
  - uncompetitive: every offer at risk;
  - otherwise mixed.
- The free-text search (`/competitors/search`) is keyword matching over the scraped passages; there is no vector
  database.

## 12. The campaign log, feedback and report (`campaign_log.py`, `pipelines/feedback.py`)

- **Campaign log** (append-only JSONL, `state/campaign/nba_campaign_log.jsonl`): one row per (D, arm, line).
  - Fields: action, offers (code, price, validity), policy_source, propensity, bandit PMF, model version, LLM model,
    justification id, anchors.
  - Control lines are logged as NO_CONTACT with propensity 1.
- **Feedback at D+7:**
  1. The campaign logged in D−8…D−2 meets the fulfilments of D−7…D−1.
  2. Each line gets its three rewards and its revenue (sum of the prices of the catalogue BTLs bought) →
     `nba_rewards.jsonl`.
  3. These rewards, with the logged propensities, are next week's training labels.
- **Report per arm:**
  - lines, take-up (3 definitions), revenue per line;
  - **uplift vs control** with a 95 % CI and p-value (Welch);
  - the smallest uplift the arm's size can detect at 80 % power;
  - llm arm: override rate and agreement with the bandit;
  - **fairness by nationality group:** take-up, revenue, mean offer price, action mix, min / max ratios.

## 13. Running it every week

```
uv run python run.py fsdr-weekly --data-dir data/fsdr/<D> [--llm-limit N]
    validate → feedback (from week 1) → train (+ base) → score (target) → offers cstr, free, llm → report (last week)
    → outputs/weekly_run_report.json (status, seconds and summary per step); exit code 1 at the first failure
```

To schedule it on Windows, Wednesday 06:00 for example:

```
schtasks /Create /SC WEEKLY /D WED /ST 06:00 /TN FSDR-weekly /TR "cmd /c cd /d <V1.0 folder> && uv run python run.py fsdr-weekly >> weekly.log 2>&1"
```

**Expected time at 1.7 M lines** (this laptop, i7-10750H; measured at 100 k lines, scaled linearly; synthetic data):

| Step | 100 k lines (V1.0) | 1.7 M lines |
|---|---|---|
| validate | 41 s | ~12 min |
| train + score the whole base | 42 s | ~12 min |
| score the target | 18 s for 29 k lines | ~0.6 ms per target line |
| offers, value policy | 20 s for 9.8 k lines | ~2 ms per line (125 k lines per arm ≈ 4–5 min) |
| peak memory | 1.17 GB | **~20 GB: plan 32 GB of RAM** |
| llm arm | — | GPU-bound, ~75 s per line on a GTX 1650: keep that arm to a few thousand lines |

**The step-by-step journey** (web UI tab "FSDR Journey", `pipelines/journey.py`):
- upload: every table copied byte for byte, a green status per table, joins with match rates, key calculations
  recomputed from the raw tables;
- then action, offers, add-ons, competitors and output, for one MSISDN or a batch.

## 14. Tracing with LangSmith (`nbx/core/tracing.py`)

- **Off by default.** It starts when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set; the project is
  `LANGSMITH_PROJECT`, default `fsdr-recommender`.
- **Nesting:** `fsdr_weekly` → `fsdr_weekly:<step>` → `fsdr_train` / `fsdr_score` / `fsdr_recommend` /
  `fsdr_batch` → `fsdr_decision` → agent → `llm:<model>` and `tool:<name>`. Journey steps are `fsdr_journey:<step>`.
- **What is sent:** every prompt and LLM answer, every tool's arguments and result, and every pipeline's counts.
  - The MSISDN never appears; runs are tagged with the customer pseudonym.
  - **Tool results do contain the customer's features.** For real data, use a self-hosted LangSmith
    (`LANGSMITH_ENDPOINT`), or `LANGSMITH_HIDE_INPUTS=true` / `LANGSMITH_HIDE_OUTPUTS=true`.
- **Safety:** tracing never changes a decision, and a LangSmith error never stops a run.
