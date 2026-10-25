# New Strategy — Agentic NBA (Vowpal Wabbit) + NBO with Generosity & Aggressiveness

> Status: **proposal only**. Nothing in the code has been changed. Section 9
> lists what would have to change when we decide to build it.

**Scope**

- Data: 6 monthly snapshots **M1 → M6**, about **500 variables** per customer,
  **1M+ MSISDNs per month** (about 6M customer-month rows).
- Step 1 — **NBA**: choose the action per customer with a Vowpal Wabbit
  contextual bandit, inside the agentic council.
- New **NBA segmentation** built from the model inputs (scores and usage).
- Step 2 — **NBO**: choose the offer for that action, with two explicit
  dials: **generosity** and **aggressiveness**.

---

## 0. Summary

| # | Decision | Why |
|---|---|---|
| 1 | Build one **customer × month panel** (M1–M6) and always predict *next month* from *past months* | Prevents leakage, and we get 3 clean train/validate/score windows from 6 months |
| 2 | Reduce 500 variables to **~60–100 features** in families (level, trend, volatility, scores) | VW and the segments become stable, fast and explainable |
| 3 | Segments **give a set of allowed actions**, not one action | Today the segment fixes the action and the bandit only learns to copy it |
| 4 | Train VW on a **business reward** (take-up + revenue − cost − churn), with **logged propensities**, using `--cb_adf` + doubly-robust evaluation | Today's training cost is "did you guess the segment's action", so it learns imitation, not uplift |
| 5 | Add a **NO_ACTION / control** arm and a permanent 5–10% holdout | The only way to measure real uplift |
| 6 | NBO gets two dials, **Aggressiveness (A0–A4)** and **Generosity (G0–G4)**; the NBA action and the risk level set the allowed range | Today aggressiveness only exists implicitly, and generosity doesn't exist at all (see §6.1) |
| 7 | Batch scoring for 1M+ customers; the **LLM agents only handle exceptions** (high value, call centre, QA sample) | One agentic run takes about 5–8 minutes locally, so it can't run per customer at this scale |

---

## 1. What exists today (from the repo)

### 1.1 NBA (`nba-bandit-sim/`, `nbo-rag-local/core/nba_*`)

- **7 actions (arms)**: `CROSS SELL VOICE, UPSELL VOICE, CROSS SELL DATA, UPSELL DATA, MAINTAIN, UPSELL, RETAIN`.
- **18 segments map to 1 action each** (`02_BUSINESS_REQUIREMENTS.md` §4). The
  `Nursering_*` segments (tenure < 45 days) are excluded.
- **VW model**: `--cb_explore_adf -q UA --epsilon 0.2`, with shared context `|User`
  (nationality group, 8 probability scores, 15 variation features) and one
  `|Action product=<name>` line per arm.
- **Inputs**: weekly W1–W4 and monthly M1–M3 usage (voice minutes, data MB,
  call count, subscription amount, pay-go), plus 9 model scores (churn, churn 1-day,
  reactivation 15/30d, appetency data/voice, revenue decrease, fragility
  data/voice).

### 1.2 Weak points to fix in the new strategy

| Today | Problem | New strategy |
|---|---|---|
| Training cost = −1 when the bandit **predicts the same action as the segment map** (§11) | The bandit learns to **imitate the segment rule**; it can never beat it | Reward = real business outcome after contact (§4.3) |
| Trained on **takers only** (`reward == 1`) | No negative examples, so it can't learn what fails | Train on everyone contacted, plus the control group |
| Logged propensity = action's population share + `uniform(0, 0.02)` jitter | Not the real probability used at decision time, so off-policy correction is wrong | Log the exact probability of the executed action (§4.4) |
| Only 3 months of history | Short trends, noisy | Use all 6 months: slope, volatility, recency (§2.3) |
| Revenue variation direction differs between training and scoring (quirk §8) | Feature sign flips between train and score | One direction everywhere: `(later − earlier) / earlier` |
| `proba_app_voice` computed but not in the context | Lost signal for voice actions | Add it |
| Model not saved (`vw.save` commented out) | Policy is relearned every run, no continuity | `-f model.vw --save_resume`, versioned per month |
| ε = 0.2 (20% random) | Too expensive on 1M customers | ε 0.05–0.10, or `--squarecb` (§4.5) |

---

## 2. Data foundation: M1–M6, 500 variables, 1M+ MSISDNs

### 2.1 The panel

One table, partitioned by month (Parquet, Spark or DuckDB/Polars):

```
customer_month(msisdn, month, <500 variables>, <9 scores>, segment_v1, action_taken, offer_taken, revenue, ...)
```

About 6M rows × 500 columns. Spark handles this easily, and so does a single
machine with DuckDB/Polars if you read only the needed columns (columnar
Parquet).

### 2.2 Time windows (no leakage)

Features come **only** from months ≤ t; the outcome comes from month t+1.

| Use | Features from | Outcome from |
|---|---|---|
| Train | M1–M4 | M5 |
| Validate (out-of-time) | M2–M5 | M6 |
| Score (production) | M3–M6 | M7 = next campaign |

When more months arrive, the windows slide forward one month at a time.
Never pick validation customers at random from the same month: telecom
behaviour is seasonal (Ramadan, Eid, summer, salary days), and only
out-of-time validation is honest.

### 2.3 From 500 variables to ~60–100 features

**Step A — clean (automatic)**

1. Drop variables that are > 95% missing or near-constant (variance ≈ 0).
2. In each group of variables correlated |r| > 0.95, keep one (for example
   `data_volume_mb_m3` vs `data_volume_kb_m3`).
3. Fill amounts and usage with **0**. Fill a missing score with a
   *"missing" indicator* plus the median, **not** with random values (the
   current `uniform(0, 0.02)` fill draws one number per column, a known bug).

**Step B — engineer families over 6 months**, for each key metric (data MB,
local/on-net/off-net voice, revenue, subscription amount, pay-go, active
days, recharge count):

| Family | Formula | Example name |
|---|---|---|
| Level | mean of M4–M6 | `data_mb_lvl` |
| Trend | slope of a linear fit over M1–M6, divided by the mean | `data_mb_slope` |
| Short change | `(M6 − M5) / M5` (with the zero-baseline rules from §8 of the BR doc) | `data_mb_var_m5_m6` |
| Medium change | `mean(M4–M6) / mean(M1–M3) − 1` | `data_mb_q2q` |
| Volatility | std / mean over M1–M6 | `data_mb_cv` |
| Recency | months since the last month with 0 usage | `data_mb_last_zero` |
| Share | data revenue / total revenue | `rev_data_share` |

Plus: the 9 scores (latest month, and their change vs the previous month),
tenure, nationality group, device segment, network type, region, value
decile, and the last purchased product category and price.

**Step C — select (supervised)**

Train one LightGBM per outcome (take-up, churn, revenue increase) on the
training window and keep the union of the top 30–40 features by SHAP. That
gives about 60–100 features. Check that importances are stable between the
train and validate windows, and drop features whose importance jumps (a sign of leakage or drift).

> VW can technically take all 500 (feature hashing, `-b 24`), but the
> reduced set gives stabler policies, faster training and readable
> explanations for the agents and the business.

---

## 3. New NBA segmentation (built from the model inputs)

### 3.1 Principle

Segments **don't choose the action**. They define:

1. **Who is excluded** (no contact).
2. **The allowed actions** (the bandit's action mask).
3. **A default action** (cold start, and the fallback if the model is down).
4. **The NBO envelope** (aggressiveness / generosity range, §6).

The bandit then learns **which allowed action works best** for each
individual context. This replaces "18 segments → 1 action" with "segment →
set of actions, chosen by learning".

> The 18-segment v1 mapping is marked "authoritative" in the BR doc. Keep
> it running in parallel as the **benchmark policy** (champion), and present v2 as the challenger.

### 3.2 Five building blocks (each from existing inputs)

| Block | Inputs | Levels (starting thresholds, tune on data) |
|---|---|---|
| **Lifecycle** | tenure, days since last activity, active days | `NURSING` (< 45 d, excluded) · `ONBOARDING` (45–90 d) · `ESTABLISHED` · `DORMANT` (no activity ≥ 30 d) |
| **Risk** | proba_churn, proba_churn_1d, proba_rev_decr, proba_frag_data/voice | `HIGH` / `MEDIUM` / `LOW` (§3.3) |
| **Usage profile** | 6-month share of data vs voice | `DATA_ONLY` (voice < 10% of usage) · `VOICE_ONLY` (data < 10%) · `DUAL` |
| **Appetency** | proba_app_data, proba_app_voice | `APP_DATA`, `APP_VOICE`, `APP_BOTH`, `APP_NONE` (score > threshold) |
| **Value & trend** | revenue level decile, revenue slope | Value `LOW / MID / HIGH / PREMIUM` (deciles 1–4 / 5–7 / 8–9 / 10); Trend `GROWING` (slope > +5%/mo) · `STABLE` · `DECLINING` (< −5%/mo) |

### 3.3 Risk tier (one composite score)

```
risk_score = max( proba_churn,
                  0.8 × proba_churn_1d,
                  0.6 × proba_rev_decr,
                  0.5 × max(proba_frag_data, proba_frag_voice) )
```

`HIGH` = top 15% of risk_score, `MEDIUM` = the next 25%, `LOW` = the rest.
Use quantiles, not fixed numbers, so the share stays stable from month to month.
Calibrate the weights by checking which combination best predicts actual
churn in M5/M6.

### 3.4 Segment → allowed actions (v2 proposal)

| # | Segment v2 (rule) | Allowed actions (bandit mask) | Default | NBO envelope |
|---|---|---|---|---|
| S0 | `NURSING` | — (excluded) | NO_ACTION | — |
| S1 | `DORMANT` & reactivation proba high | RETAIN, UPSELL (cheap re-entry) | RETAIN | A0, G3–G4 |
| S2 | `DORMANT` & reactivation proba low | RETAIN, NO_ACTION | NO_ACTION | A0, G2–G3 |
| S3 | `ONBOARDING` & risk HIGH | RETAIN, MAINTAIN | RETAIN | A0, G2 |
| S4 | `ONBOARDING` & risk LOW/MED | UPSELL, UPSELL DATA, MAINTAIN | UPSELL | A1–A2, G1 |
| S5 | `ESTABLISHED` & risk HIGH | RETAIN, MAINTAIN | RETAIN | A0, G2–G3 |
| S6 | risk MED & `DECLINING` | RETAIN, MAINTAIN, UPSELL DATA | MAINTAIN | A0–A1, G1–G2 |
| S7 | risk LOW & `DATA_ONLY` & `APP_VOICE` | UPSELL VOICE, CROSS SELL VOICE, UPSELL | UPSELL VOICE | A1–A2, G0–G1 |
| S8 | risk LOW & `DATA_ONLY` & no voice appetency | UPSELL DATA, UPSELL, MAINTAIN | UPSELL DATA | A2–A3, G0 |
| S9 | risk LOW & `VOICE_ONLY` & `APP_DATA` | UPSELL DATA, CROSS SELL DATA, UPSELL | CROSS SELL DATA | A1–A2, G1 |
| S10 | risk LOW & `VOICE_ONLY` & no data appetency | UPSELL VOICE, MAINTAIN | MAINTAIN | A1, G0 |
| S11 | risk LOW & `DUAL` & `GROWING` | UPSELL, UPSELL DATA, UPSELL VOICE | UPSELL | A2–A4, G0 |
| S12 | risk LOW & `DUAL` & `STABLE` | MAINTAIN, UPSELL, UPSELL DATA | MAINTAIN | A1–A2, G0–G1 |

Rules are applied **in order**; the first match wins, and every customer
gets exactly one segment. All segments allow **NO_ACTION** as a control arm
(§4.2).

**Hard guardrails on top of the mask** (already in the agentic critic,
which keeps enforcing them):

- risk `HIGH`: never a push action (UPSELL*, CROSS SELL*).
- purchase in the last 3 days: no push (contact cooldown).
- excluded segment: NO_ACTION only.

### 3.5 Optional: data-driven micro-segments (context, not rules)

Run MiniBatch K-Means (k ≈ 10–15) on the standardized top-30 features (or
on 10 PCA components). Profile each cluster (mean usage, scores, value) and
give it a business name. Use `cluster_id` **as a VW feature**, not as a
rule. It helps the bandit generalize, and the agents can say "customers like
this one usually…".

---

## 4. NBA with Vowpal Wabbit (Step 1)

### 4.1 Example format

```
shared |U risk_score:0.62 proba_churn:0.55 proba_app_data:0.31 value_decile:7 tenure_m:14 ...
       |T data_mb_slope:-0.12 rev_q2q:-0.08 data_mb_cv:0.40 voice_last_zero:2 ...
       |S seg_v2=S6 lifecycle=ESTABLISHED usage=DUAL nat=Oman cluster=c07
|A act=RETAIN
|A act=MAINTAIN
|A act=UPSELL_DATA
|A act=NO_ACTION
```

- Only the **allowed actions** of the segment are listed (the mask is applied
  by building the example, not after the prediction).
- Namespaces: `U` = profile and scores, `T` = 6-month trends, `S` =
  segment/categorical, `A` = action.
- Interactions: `-q UA -q TA -q SA`, so each action gets its own weight on
  every context feature.

### 4.2 Arms

The current 7 actions **plus NO_ACTION**. NO_ACTION is both a legitimate
decision ("don't disturb this customer this month") and the control that
measures uplift. In addition, keep a **random 5% holdout** that is never
contacted, to measure the global effect of the programme.

### 4.3 Reward (the most important change)

Measured over a fixed window after the contact (for example 30 days):

```
reward =  1.0 × took_offer
        + 0.1 × Δrevenue_30d (OMR, capped at ±20)
        − 0.5 × churned_30d
        − c_contact          (≈ 0.02 per SMS, ≈ 0.3 per outbound call)
        − generosity_cost     (OMR of discount/bonus given, ×0.1)

VW cost = −reward
```

- Agree the weights with finance and CVM. Section 7 explains how to check them.
- Compute it for **every contacted customer**, not only takers. A customer
  who didn't take the offer gives cost ≈ 0 or positive, which is the
  negative signal the bandit needs.
- For NO_ACTION, compute the same reward (Δrevenue, churn, organic purchase).
  That is what makes uplift visible.

### 4.4 Logged propensity (needed for off-policy learning)

At decision time, store `(msisdn, month, segment, allowed_actions, chosen_action, p_chosen, model_version)`.
`p_chosen` is the PMF value of the action that was **actually executed**.

- If the agentic critic or a business rule **changes** the action, log the
  final action with the probability the whole decision process gave it.
  Without that, VW's IPS/DR corrections are biased.
- **Cold start (the M1–M6 history has no real propensities):**
  1. *Phase 0 (recommended):* one exploration month where 10–20% of
     eligible customers get a **uniformly random action within their
     mask**. The propensity is then known exactly: 1 / (number of allowed actions).
  2. *Meanwhile:* estimate historic propensities with a multinomial model
     `P(action_taken | features, segment_v1)` trained on M1–M6. Clip them to
     [0.02, 1] and use them with `--cb_type dr`. Don't trust it alone; it is
     only a warm start.

### 4.5 Training and exploration settings

| Setting | Proposal | Reason |
|---|---|---|
| Offline training on logs | `vw --cb_adf --cb_type dr -q UA -q TA -q SA -b 24 --l2 1e-7 -l 0.05 --passes 3 -c -f nba_M6.vw` | Doubly-robust: low bias and low variance |
| Online / production policy | `--cb_explore_adf --squarecb --gamma_scale 10` (or `--epsilon 0.05`) | Explores where the model is uncertain, instead of 20% random |
| Persist | `-f model.vw --save_resume`, one file per month, versioned | Today's model is not saved |
| Retrain | Monthly, on all logs from the last 6 months (rolling) | Matches the data rhythm |
| Scale | ~1M examples × 4–8 arms: minutes on one CPU | VW is streaming; no GPU needed |

### 4.6 Evaluation before going live

On the validation window (M6 outcomes), compare these policies **offline**:

1. v1 segment rule (today's champion).
2. Today's VW (imitation).
3. New VW on the v2 segments.
4. Random within the mask (sanity floor).

Use IPS, SNIPS and DR estimates of the average reward (`vw --cb_adf --eval`,
or compute them in Python from the logged PMFs). Go live only if the new VW
beats v1 on DR reward **and** does not raise churn among `HIGH` risk customers.

### 4.7 Where the agents fit (agentic NBA)

```
             ┌──────────────── batch (1M+ / month) ────────────────┐
 panel M3–M6 → features → segment v2 → mask → VW PMF → sample action → log
             └──────────────────────────────┬──────────────────────┘
                                            │ exceptions only
                                            ▼
                     Agentic council (LLM): Strategist → Offer → Critic → Messenger
```

- **Batch path (all customers):** the VW decision is final. It is fast, logged and auditable.
- **Agentic path (a few thousand):** high value (`PREMIUM`), call-centre
  sessions, conflicting signals (for example the bandit says UPSELL but risk
  is MEDIUM and declining), and a random QA sample.
  - The Strategist agent calls the `bandit_recommendation` tool and now
    receives the **whole PMF**, the mask, the segment and the top drivers.
  - It may choose another action **only inside the mask**, with a rationale.
    The Critic enforces the guardrails.
  - The executed action and its probability are logged (§4.4). An LLM
    override is logged as a separate policy (`policy=agent`), so its value
    can be measured against the pure bandit.
- The LLM explains; the bandit learns. The LLM must not become an
  unmeasured second policy.

---

## 5. NBO (Step 2): offer for the chosen action

Today's flow stays: action → allowed categories (`ACTION_CATEGORIES`) →
candidates (catalog + collaborative filtering + cohort graph) → eligibility →
ranking by expected value. **New:** each candidate is evaluated at an
**aggressiveness level** and a **generosity level**, and only levels inside
the envelope set by the NBA (§3.4 and §6.3) are allowed.

---

## 6. Generosity and aggressiveness

### 6.1 Do they exist today?

| Concept | Exists? | Where |
|---|---|---|
| **Aggressiveness** (how far the price is pushed above current spend) | **Partly, implicitly** | `_affordability_tier` in `nbo-rag-local/agents/agentic/tools.py`: "sweet-spot upsell" = price in (spend, spend + 3 OMR] (`AFFORDABILITY_BAND_OMR = 3.0`), "upsell" = above that, "at/below" = cheaper. Eligibility rejects prices > 2.5 × monthly spend. The LightFM NBO picks the bundle "just above current spend" (`price_band_width: 3.0`). These are fixed values, not a tunable level tied to the NBA action. |
| **Generosity** (how much value we give away: discount, bonus data/minutes) | **No** | The catalog has a `discount_pct` field, but every offer has only 0% or 5%. It isn't used in eligibility, expected value or ranking. |

So the strategy is to **turn the implicit aggressiveness into an explicit
dial** and to **create generosity**, both controlled by the NBA action and
the risk level.

### 6.2 Definitions

**Aggressiveness A** compares the offer's monthly-equivalent price with what
the customer actually spends:

```
uplift_ratio = (offer_monthly_price − avg_monthly_spend_M4_M6) / avg_monthly_spend_M4_M6
```

| Level | uplift_ratio | Meaning |
|---|---|---|
| **A0 Protect** | ≤ 0% | Same price or cheaper. Keep the customer and the habit |
| **A1 Soft** | 0 – 15% | Small step up |
| **A2 Standard** | 15 – 40% | Normal upsell (≈ today's "sweet spot" for mid spenders) |
| **A3 Push** | 40 – 100% | Strong upsell, for low risk and high appetency only |
| **A4 Stretch** | > 100% | Double the spend. Only for `GROWING` + `LOW` risk + `PREMIUM/HIGH` value |

A relative ratio works for every value tier, unlike the fixed 3 OMR band
(3 OMR is +100% for a 3 OMR customer but +10% for a 30 OMR customer).

**Generosity G** is the value we give away, as a share of the list value:

```
generosity_pct = (list_value − price_paid) / list_value
list_value = list price + OMR value of bonus data/minutes (at catalog price per GB/min)
```

| Level | generosity_pct | Typical mechanism |
|---|---|---|
| **G0 None** | 0% | Catalog price |
| **G1 Light** | ≤ 10% | +10% bonus data, or 5–10% discount |
| **G2 Medium** | 10 – 25% | Bonus GB, free week, or 15–20% discount |
| **G3 High** | 25 – 40% | Retention-level discount or double data |
| **G4 Win-back** | 40 – 60% | Reactivation of dormant or lost customers only, one per customer per 6 months |

**Catalog requirement:** to use G1–G4, each base offer needs variants (for
example `PS_30D_15R_SAVER_OT@G0`, `@G1`, `@G2`, …) or a runtime discount/bonus
parameter accepted by the provisioning system. **This is the main dependency
outside data science.**

### 6.3 Who gets which levels (the envelope)

| NBA action \ risk | LOW | MEDIUM | HIGH |
|---|---|---|---|
| **UPSELL / UPSELL DATA / UPSELL VOICE** | A2–A3 (A4 if growing & high value), G0–G1 | A1–A2, G1 | not allowed (guardrail) |
| **CROSS SELL DATA / VOICE** | A1–A2, G1 (first-time trial) | A1, G1–G2 | not allowed |
| **MAINTAIN** | A0–A1, G0–G1 | A0–A1, G1 | A0, G1–G2 |
| **RETAIN** | A0, G1 | A0, G2 | A0, G2–G3 |
| **Reactivation (DORMANT segments)** | A0, G2–G3 | A0, G3 | A0, G3–G4 |
| **NO_ACTION** | — | — | — |

This answers the earlier question about MAINTAIN: a MAINTAIN customer can
still get a better pack at **A1** (≤ +15%), but never a strong push.

### 6.4 Choosing the offer and the levels (optimization)

For each candidate `(offer, A, G)` inside the envelope:

```
p_accept   = model(customer features, offer features, A, G)
margin     = monthly_price × (1 − generosity_pct) − cost_of_service
incremental = p_accept(offer, A, G) − p_buy_anyway(customer)      ← uplift, not raw take-up
EV         = incremental × margin
           + p_accept × retention_value          (churn × avg_rev × churn_reduction)
           − p_accept × generosity_pct × list_value × cannibalization_rate
```

Pick the max-EV candidate. **Generosity must pay for itself**: a discount
only makes sense when it raises the *incremental* take-up enough. Giving 20%
to a customer who would have bought at full price is pure loss, which is why
uplift is needed.

**How to learn `p_accept` and the uplift from M1–M6:**

1. **Price elasticity per segment:** from historic purchases, take-up rate
   vs uplift_ratio (A level) and vs discount (where 5% offers existed).
   Fit a monotone curve (isotonic or logistic in A).
2. **Uplift model:** a T-learner or X-learner (LightGBM) with treatment =
   "received offer at level G≥1" vs G0. The history has almost no
   generosity, so **Phase 0 must also randomize G** within the envelope
   (for example, 20% of contacts get a random G level).
3. **Second bandit, recommended once data exists:** a VW `cb_adf` whose arms
   are `(offer, G level)` within the envelope, with reward = margin-based
   (§4.3, generosity cost included). It learns generosity directly from
   outcomes, with the same machinery as the NBA bandit.

**Budget control:** the monthly discount budget B is allocated greedily by
`EV gain per OMR of generosity`. Customers are sorted by that ratio and
generosity is assigned until B is used up; the others fall back to G0 (or
G1 for RETAIN). A Lagrangian penalty λ on generosity cost in the EV formula
does the same thing smoothly; tune λ until spend ≈ B.

**Guardrails for generosity:**

- No G ≥ 2 for anyone who bought the same offer at full price in the last 60 days (cannibalization).
- G4 at most once per customer per 6 months.
- Generosity never goes beyond what's needed to match the closest
  competitor plan (the agentic `competitor_benchmark` tool already provides
  Omantel and Vodafone Oman prices).
- Total generosity per segment is capped (finance rule), and A3–A4 are
  never combined with G ≥ 2. You don't push the price and discount at the same time.

### 6.5 Inside the agentic Offer agent

- The Offer agent receives the **envelope** (`A_allowed`, `G_allowed`) in
  its Step 1 → Step 2 handoff context, together with the action.
- The tools `check_eligibility` / `simulate_expected_value` return the A
  level, the G level and the uplift-based EV for each candidate.
- The Critic gets two new guardrails: `AGGRESSIVENESS_OUT_OF_ENVELOPE` and
  `GENEROSITY_OUT_OF_ENVELOPE` (hard), plus `GENEROSITY_NOT_PAID_BACK` (soft:
  the EV is lower than at G0).
- The Messenger must state the real discount and bonus. The compliance check
  verifies the numbers against the offer variant, as it already does for
  the price.

---

## 7. Measurement

| What | How |
|---|---|
| Global programme effect | 5% permanent holdout (never contacted) vs everyone else |
| v2 vs v1 policy | Customers randomly split 50/50 between champion (v1 segment rule) and challenger (VW v2) for 2 months |
| Bandit quality | DR/IPS estimates each month, plus the share of traffic that is exploration |
| Generosity ROI | Incremental margin per OMR given away, per G level and segment |
| Health KPIs | Take-up rate, ARPU uplift, churn 30/60 d by risk tier, opt-out/complaint rate, contacts per customer |

**Reward weights check:** if the bandit maximizes the reward but ARPU or
churn get worse, the weights in §4.3 are wrong. Review them with finance
after the first month.

---

## 8. Roll-out plan

| Phase | Duration | Content | Exit criterion |
|---|---|---|---|
| **P1 Data** | 2–3 weeks | Panel M1–M6, feature families, selection, v2 segments, segment profiling | Segments are stable month-to-month (≥ 80% of customers keep the same segment, except trend changes) |
| **P2 Offline model** | 2 weeks | Propensity estimates, VW `--cb_adf --cb_type dr`, offline evaluation vs v1 | DR reward > v1, with no churn degradation for HIGH risk |
| **P3 Phase 0 exploration** | 1 month | 10–20% randomized action within the mask + random G level, with exact logging | Clean logs with real propensities |
| **P4 Champion/challenger** | 2 months | v1 vs VW v2 50/50, generosity budget live, agentic path for exceptions | Significant uplift on the reward and ARPU, with controlled generosity spend |
| **P5 Production** | ongoing | Monthly retrain, monitoring dashboard, second bandit for (offer, G) | — |

---

## 9. What would change in the code later (not done now)

| Area | Files | Change |
|---|---|---|
| Features | `nba-bandit-sim/nba/features.py` | 6-month families, one variation direction, deterministic missing-score handling, `proba_app_voice` added |
| Segmentation v2 | new `nba/segments_v2.py`, `core/config.py` | Rule table §3.4: segment → mask, default action, envelope |
| Bandit | `nba-bandit-sim/nba/bandit.py`, `nbo-rag-local/core/nba_bandit.py` | Business reward, all contacted customers, logged propensities, mask in the examples, `--cb_type dr`, `--squarecb`, saved model |
| Agent tools | `nbo-rag-local/agents/agentic/tools.py` | `bandit_recommendation` returns PMF + mask + drivers; A/G levels in `eligibility` / `expected_value` |
| Council | `nbo-rag-local/agents/agentic/council.py` | Envelope in the Step 1 → Step 2 handoff; new guardrails; logging of the executed action and its probability |
| Catalog | `data/offers.json` + provisioning | Generosity variants per offer (G0–G4) |

---

## 10. Open questions for the business

1. Reward weights (§4.3): what is 1 accepted offer worth vs 1 OMR of revenue vs 1 churned customer?
2. The monthly generosity budget (OMR) and any per-segment caps.
3. Can provisioning apply a discount or bonus at runtime, or do we need catalog variants?
4. Is an exploration month (P3) acceptable to the campaign team, and what share (10% or 20%)?
5. Should v2 segments replace the 18 v1 segments in reporting, or run alongside?
