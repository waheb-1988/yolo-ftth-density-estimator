# FSDR – Prepaid NBA & BTL Offer Recommendation: Business Rules Catalogue

This catalogue lists every business rule found in the SQL extractions and in the four Python recipes (Code 1 NBA, Code 2 split, Code 3 constraint-arm offers, Code 4 free-arm offers): filters, calculations, ARPU / price limits, thresholds and constants.

Notation:
- **D** = CURRENT_DATE.
- **T** = campaign anchor = most recent Thursday on or before D. It is computed as `D - (((D - DATE '2026-03-12') MOD 7) + 7) MOD 7`; 12-Mar-2026 is a Thursday.
- Amounts are in local currency (OMR).

---

## 1. Customer eligibility (base population – `base_prepaid`)

| # | Rule | Value |
|---|---|---|
| BR-01 | Snapshot date | `ACTIVITY_DATE = D-2` |
| BR-02 | MSISDN range | `ACCESS_METHOD_VAL > '69999999'` (string comparison) |
| BR-03 | Customer type | `CUST_TYPE = 'CONSUMER'` |
| BR-04 | Subscription type | `SUBSCRIPTION_TYPE = 'MOBILE'` |
| BR-05 | Subscription sub-type | `SUBSCRIPTION_SUB_TYPE = 'PREPAID'` |
| BR-06 | Lifecycle status | `SUBS_LC_ACTIVITY_TYPE_ID = -1` (active) |
| BR-07 | Line still active | `ACCESS_METHOD_DEACTIVE_DT IS NULL OR > D-2` |
| BR-08 | One line per MSISDN | Keep the latest activation (`ROW_NUMBER() … ORDER BY ACCESS_METHOD_ACTIVE_DT DESC = 1`) |
| BR-09 | Plan name | If `OFFERING_ID >= 300000`, take the product commercial name; otherwise the offering name |
| BR-10 | Excluded plans (internal / employee / business) | Internal plan, Employee Data with Voice, Shahry Business, Shahry, Employee superfast 100 / 200 / 400 Mbps, 5G Employee Plan 50 / 100 Mbps, Employee Fast HBB 150 / 300 / 600 GB, Fast Home Broadband 150GB (Employee) – 13 plans |
| BR-11 | Active consumer accounts (account → party link) | `ACCOUNT_STATUS <> 'TERM'` and `ACCOUNT_CUST_TYPE = 'CONSUMER'` |
| BR-12 | MSISDN format in usage / fulfillment tables | First 3 characters (country code) removed: `SUBSTR(MSISDN, 4)` |

## 2. Location and demographics

| # | Rule | Value |
|---|---|---|
| BR-13 | Home location | Most frequent home cell site over the **previous calendar month** of D-2. January rolls back to December of the previous year. |
| BR-14 | Several locations for one line | `MIN(REGION)`, `MIN(WILAYAT)`, `MIN(AREA)` |
| BR-15 | Nationality group | Oman, India, Bangladesh, Pakistan kept as-is; everything else becomes `others` |

## 3. Revenue scope – product types counted as subscription revenue

| # | Rule | Value |
|---|---|---|
| BR-16 | Product types included in subscription revenue | CVM, DATA, Other Base Plans, DIGITAL SERVICE, EASY CREDIT, EXPIRED CREDIT, Flexi, Hala, INTERNATIONAL VOICE, New Shababiah, Roaming, SMS, VOICE, Hala Flex (14 types) |
| BR-17 | Data subscription revenue | `PRODUCT_CLASS IN ('Daily Data','Monthly Data','Weekly Data')` |
| BR-18 | Voice subscription revenue | `PRODUCT_CLASS = 'Local Voice'` |
| BR-19 | Amount field | NBA tables use `SUBSCRIPTION_TOT_AMT`; offer-pricing tables (`subscriptions_tg`) use `SUBSCRIPTION_AMT` |
| BR-20 | Missing amount | `COALESCE(amount, 0)` |

## 4. Time windows

### 4.1 Subscription amount (`subs_amt_auto` anchored on T, `subs_amt_score` anchored on D) – the window **includes** the anchor day

| Period | Window |
|---|---|
| W1 | anchor-6 → anchor |
| W2 | anchor-13 → anchor-7 |
| W3 | anchor-20 → anchor-14 |
| W4 | anchor-27 → anchor-21 |
| M1 | anchor-29 → anchor (30 days) |
| M2 | anchor-59 → anchor-30 |
| M3 | anchor-89 → anchor-60 |

### 4.2 Usage activity (`activities_*_auto` anchored on T, `activities_*_score` anchored on D) – the window **excludes** the anchor day

| Period | Window |
|---|---|
| W1 | anchor-7 → anchor-1 |
| W2 | anchor-14 → anchor-8 |
| W3 | anchor-21 → anchor-15 |
| W4 | anchor-28 → anchor-22 |
| M1 | anchor-28 → anchor-1 (28 days) |
| M2 | anchor-56 → anchor-29 (28 days) |
| M3 | anchor-85 → anchor-57 (29 days) |

### 4.3 Offer-pricing tables (anchored on D)

| Table | Period | Window |
|---|---|---|
| `subscriptions_tg` | W1 / W2 / W3 / W4 | D-8→D-2 / D-15→D-9 / D-22→D-16 / D-29→D-23 |
| `subscriptions_tg` | M1 / M2 / M3 | D-31→D-2 / D-61→D-32 / D-91→D-62 |
| `usage_data_voice_payg_tg` – voice & data | p1 / p2 / p3 | D-31→D-2 / D-61→D-32 / D-91→D-62 |
| `usage_data_voice_payg_tg` – PAYG | M1 / M2 / M3 | D-30→D-1 / D-60→D-31 / D-90→D-61 |
| `usage_data_voice_payg_tg` – PAYG | W1 / W2 / W3 / W4 | D-7→D-1 / D-14→D-8 / D-21→D-15 / D-28→D-22 |
| `paygo_activity_auto` (anchored on T, **not used**) | same as above | |

### 4.4 Campaign tables

| # | Rule | Value |
|---|---|---|
| BR-21 | Targeting used for learning | `targeting_dataiku_auto`: `date1` between D-8 and D-2 |
| BR-22 | Arbitration targeting | `targeting_auto`: `start_date` between D-7 and D (loaded, **not used**) |
| BR-23 | Takers (reward window) | BTL fulfilled with success date between D-7 and D-1 |
| BR-24 | Interaction history (LightFM) | Fulfillments over D-270 → D-1 (9 months, despite the name "3 months") |
| BR-25 | Fulfillment must be successful | `FULFILMENTSUCCESSDATE_TIME IS NOT NULL` |

## 5. Usage metric definitions

| # | Metric | Rule |
|---|---|---|
| BR-26 | Voice record | `CALL_TYPE_CD = 'V'` |
| BR-27 | Data record | `CALL_TYPE_CD = 'D'` |
| BR-28 | Voice minutes | rounded duration ÷ 60 |
| BR-29 | Data volume MB | rounded duration (bytes) ÷ 1024 ÷ 1024 |
| BR-30 | On-net voice | `ACTIVITY_PATH_TYPE = 'ONNET'` and not international |
| BR-31 | Off-net voice | `ACTIVITY_PATH_TYPE = 'OFFNET'` and not international |
| BR-32 | Local voice | Voice, `NATIONAL_INTERNATIONAL_IND <> 'INTERNATIONAL'` (actual duration ÷ 60) |
| BR-33 | PAYG amount | `TOTAL_CHARGED_AMT` (total / voice / data). In the `_tg` tables only **positive** charges are counted. |
| BR-34 | Data volume (`_tg`) | Only records with volume > 0 |
| BR-35 | Active days | Distinct activity dates (all / voice / data) |

## 6. NBA model rules (Code 1)

### 6.1 Segments and actions

| # | Rule | Value |
|---|---|---|
| BR-36 | Excluded segments | `Nursering_Early_Churn_At_Risk`, `Nursering_Early_Churn_Not_At_Risk` (tenure < 45 days) |
| BR-37 | Targeted rows without segment | Removed |
| BR-38 | Action list (7) | CROSS SELL DATA, UPSELL DATA, RETAIN, MAINTAIN, UPSELL, CROSS SELL VOICE, UPSELL VOICE |

**BR-39 – Segment → action mapping (effective values after Python duplicate-key resolution)**

| Segment | Action |
|---|---|
| Retention_Low_Probability_To_Reconnect | RETAIN |
| Retention_Med_Probability_To_Reconnect | RETAIN |
| Retention_High_Probability_To_Reconnect | RETAIN (⚠ also defined as UPSELL) |
| Retention_RGS_At_Risk | RETAIN |
| Retention_RGS_Fragile | UPSELL |
| RGS_Not_At_Risk_DualPlay_Upgrade | MAINTAIN |
| RGS_Not_At_Risk_Data_Only_Voice_Appetency | UPSELL VOICE (upper-case key) / CROSS SELL VOICE (mixed-case key) ⚠ |
| RGS_Not_At_Risk_Data_Only_No_Voice_Appetency | CROSS SELL VOICE |
| RGS_Not_At_Risk_Voice_Only_Data_Appetency | UPSELL DATA |
| RGS_Not_At_Risk_Voice_Only_No_Data_Appetency | CROSS SELL DATA |
| Onboarding_Early_Churn_At_Risk | UPSELL |
| Onboarding_Early_Churn_Not_At_Risk | UPSELL |

### 6.2 Reward, cost, propensity

| # | Rule | Value |
|---|---|---|
| BR-40 | Taker | Targeted MSISDN with **any** BTL in the catalogue fulfilled in the reward window (BR-23) |
| BR-41 | Reward / cost | REWARD = 1 for takers, 0 otherwise; COST = −1 for takers, 0 otherwise |
| BR-42 | One row per customer | Duplicates on MSISDN removed (first one kept) |
| BR-43 | Training population | **Takers only** (REWARD = 1) |
| BR-44 | Propensity | Action prior below + random uniform(0, 0.02) |

**BR-45 – Action priors (share of historical targeting volume)**

| Action | Prior |
|---|---|
| RETAIN | 0.5086577 |
| UPSELL | 0.2624428 |
| MAINTAIN | 0.1015183 |
| CROSS SELL VOICE | 0.0528180 |
| UPSELL DATA | 0.0422645 |
| UPSELL VOICE | 0.0239859 |
| CROSS SELL DATA | 0.0083128 |

### 6.3 Context features

| # | Rule | Value |
|---|---|---|
| BR-46 | Model scores used | Churn (max of global, early 0-30, early 30-60), reactivation 30d, reactivation 15d, data appetency, voice appetency, churn 1-day, revenue decrease, data fragility, voice fragility |
| BR-47 | Missing model score | Replaced by a random value in [0, 0.02] |
| BR-48 | Missing usage / subscription | 0 |
| BR-49 | NBA revenue per period | `SUB_AMT_p + PG_DATA_AMOUNT_p` (subscriptions + PAYG **data** only) |
| BR-50 | % variation | `(current − previous) / previous × 100`; 0 if previous = 0 |
| BR-51 | Zero flag | 1 if previous period = 0 |
| BR-52 | Log variation | `log(1+current) − log(1+previous)` |
| BR-53 | Variation pairs | Revenue: W1-W2, W2-W3, W3-W4, M1-M2, M2-M3. Voice minutes and data MB: W1-W2, W2-W3, W3-W4, M1-M2 |

### 6.4 Learning and scoring

| # | Rule | Value |
|---|---|---|
| BR-54 | Algorithm | Contextual bandit `--cb_explore_adf -q UA --epsilon 0.2` (20% exploration) |
| BR-55 | Training iterations | 20,000 simulated draws over takers |
| BR-56 | Simulated cost | −1 if the predicted action = the logged action, 0 otherwise |
| BR-57 | Scoring | Action **sampled** from the model probability distribution (not the max) |
| BR-58 | One action per customer | Duplicates on MSISDN removed after scoring |

## 7. Experiment design

| # | Rule | Value |
|---|---|---|
| BR-59 | Test arms | `group = 'free'` → free arm; `group = 'cstr'` → constraint arm (`tg_flag = with_constraint`) |
| BR-60 | Free arm | Offer chosen by LightFM only; the NBA action is **not** used |
| BR-61 | Constraint arm | The NBA action restricts allowed services and generosity |
| BR-62 | Sample stratification | Strata on nationality group, 3-month revenue, voice group, data group, plan (`strata`, `counts` columns) |

## 8. Offer eligibility rules

### 8.1 Common

| # | Rule | Value |
|---|---|---|
| BR-63 | Eligible offers | Only products present in the BTL catalogue |
| BR-64 | Validity | Recommendations computed separately for 30-day (monthly) and 7-day (weekly) offers |
| BR-65 | Generosity levels | L0 < L1 < L2 (L2 = most generous / aggressive) |

### 8.2 BR-66 – Constraint arm: action → offer families

| Action | Service A (`service[0]`) | Service B (`service[1]`) | Generosity – normal (commented out) | Generosity – **currently active** |
|---|---|---|---|---|
| MAINTAIN | COMBO | DATA | L0 | **L2** |
| UPSELL | COMBO | DATA | L1 | **L2** |
| RETAIN | COMBO | DATA | L2 | **L2** |
| UPSELL VOICE | DATA | COMBO | DATA L0 / COMBO L1 | **L2** |
| CROSS SELL VOICE | DATA | COMBO | DATA L0 / COMBO L2 | **L2** |
| CROSS SELL DATA | VOICE | COMBO | VOICE L0 / COMBO L2 | **L2** |
| UPSELL DATA | VOICE | COMBO | VOICE L0 / COMBO L2 | **L2** |

⚠ The filter branch for CROSS SELL VOICE is written as a second `CROSS SELL DATA`, so CROSS SELL VOICE is not filtered at the history stage.

### 8.3 BR-67 – Free arm

- History limited to **L2** offers.
- **VOICE offers excluded**: only DATA and COMBO.

## 9. ARPU / spending anchor calculation (Codes 3 & 4)

| # | Rule | Formula |
|---|---|---|
| BR-68 | Revenue per period | `total_sub_amt_p + pg_total_amount_p` (subscriptions + **total** PAYG) |
| BR-69 | Monthly periods | M1, M2, M3 (3 values) |
| BR-70 | Weekly periods | W1, W2, W3, W4 (4 values) |
| BR-71 | **ARPU anchor** | `avg_rev = mean(periods) + std(periods)` – deliberately above typical spend |
| BR-72 | Max revenue | `max(periods)` – computed but not used |
| BR-73 | Old methodology (commented out) | `avg_rev = min(mean, max)` or `max + 1` |
| BR-74 | Data ARPU | `avg_rev_data = 0.9 × avg_rev` |
| BR-75 | Voice ARPU | `avg_rev_voice = 0.1 × avg_rev` |
| BR-76 | Missing revenue | 0 |

**BR-77 – Which anchor applies to which offer**

| Action | Offer compared with total ARPU | Offer compared with data ARPU | Offer compared with voice ARPU |
|---|---|---|---|
| MAINTAIN / UPSELL / RETAIN | COMBO | DATA | – |
| UPSELL VOICE / CROSS SELL VOICE | COMBO | DATA | – |
| CROSS SELL DATA / UPSELL DATA | COMBO | – | VOICE |
| Free arm | COMBO | DATA | – |

Weekly offers use the weekly anchors, monthly offers the monthly anchors.

## 10. Price limits (offer selection – 3 passes)

### BR-78 – Pass 1: price band (customer gets one offer of each type if available)

`anchor < price < anchor + k`

| Arm / Action | k (upper margin) |
|---|---|
| UPSELL | +3 |
| MAINTAIN, RETAIN | +2 |
| UPSELL VOICE, CROSS SELL DATA, CROSS SELL VOICE, UPSELL DATA | +3 |
| Free arm | +3 |

Previous versions (commented out) also had a lower margin: `anchor + 0.5 < price < anchor + 4` (UPSELL), `anchor + 1 < price < anchor + 4`, `anchor + 0.5 < price < anchor + 3`.

### BR-79 – Pass 2: missing type only
A customer who got only one offer type in pass 1 gets the other type with `price > anchor` and **no upper cap**.

### BR-80 – Pass 3: customers with no offer
Both types, `price > anchor`, **no upper cap**.

### BR-81 – Ranking / tie-break

| Pass | Sort order | Comment in code |
|---|---|---|
| 1 | LightFM score ↓, then price ↓ | Since 9 October; before it was score ↓, price ↑ ("was pushing L2 mainly") |
| 2 | score ↓, price ↑ | Free arm: since 1 July 2026 (before: all ascending) |
| 3 | score ↓, price ↑ | |

### BR-82 – Number of offers
At most **2 offers per customer per validity** (one per service type), so at most **4** in total (monthly + weekly). A customer whose anchor exceeds every offer price gets **no offer**.

## 11. LightFM recommender parameters

| # | Rule | Value |
|---|---|---|
| BR-83 | Interaction | 1 per (customer, BTL) subscribed in the last 270 days |
| BR-84 | Model | WARP loss, 400 latent components, 30 epochs, k = 15, 40 threads |
| BR-85 | Candidate list | Top 100 offers per customer |
| BR-86 | Non-takers (cold start) | Scored in batches using existing users' embeddings |

## 12. Reference tables present but not applied in the code

| Table | Content |
|---|---|
| `conflicting_offers_h` | Pairs of conflicting offers (generosity, price, data, minutes, validity, price difference, data difference, same validity, same offer type) – **no conflict rule applied yet** |
| `BTL_script_english_arabic_ok` | English / Arabic script per destination offer |
| `paygo_activity_auto`, `targeting_auto`, `location`, `device segment` | Loaded or merged but not used by the model |

## 13. Inconsistencies to arbitrate with the business

1. **Revenue definition differs:** the NBA uses subscriptions + PAYG **data**; offer pricing uses subscriptions + PAYG **total**.
2. **Month length differs:** subscriptions use 30-day months, activity uses 28/29 days, and pricing tables lag by 2 days.
3. **Reward is broad:** any BTL counts, not the one actually offered. Targeting (D-8→D-2) and reward (D-7→D-1) windows overlap.
4. **Two ambiguous segment mappings:** see BR-39 (⚠).
5. **Generosity is forced to L2 for all actions:** this is temporary; the normal mapping is commented out.
6. **Prior counts don't match:** the CROSS SELL DATA comment shows 13,738 in the numerator but 7,473 in the denominator.
7. **ARPU anchor = mean + std:** this pushes offers above usual spend. Confirm that this is the intended upsell policy.
8. **Data / voice split is fixed:** 90 % / 10 % for every customer, regardless of real usage mix.
