# FSDR Recommender V1.0 — Improvements over V0 and their measured impact

V1.0 implements the findings of [FSDR_REVIEW_2026-10.md](FSDR_REVIEW_2026-10.md) (review of `nbx-python-upgrade`
at `ae800fc`, here called **V0**). This document answers three questions:

1. what changed and where in the code;
2. what each change does to recommendations, revenue and take-up;
3. how large the effect is, measured on the closed-loop simulator, V1.0 against V0 on the same customers.

Every change has a switch in `.env`. Its V0 value restores the previous behaviour, so each effect can be checked
one switch at a time.

**Scale, weekly run and tracing (no effect on results).** Key normalisation, type checks, JSON writing, offer
selection and the acceptance model are vectorised, and tables are held typed in memory: on 100 k lines, offers went
from 132 s to 20 s and peak memory from 1.85 GB to 1.17 GB, and all 124 output files are byte-identical to the
per-value code. `run.py fsdr-weekly` runs the whole week in one command, and LangSmith tracing is available, off by
default. The full engine logic, with timings at 1.7 M lines, is in [FSDR_RECOMMENDER_LOGIC.md](FSDR_RECOMMENDER_LOGIC.md).

## Summary of the measured impact

The test is a closed loop on the synthetic world, so customers react to what each version sends. V0 and V1.0 get
the same customers (same seed), 5 worlds per scenario, 5,000 lines, 3 weekly campaigns, with all arms on the
offline backend (50 runs, all completed). The differences are paired by world, with a t 95 % CI over the
5 worlds.

| What | Take-up vs V0 | Revenue per line vs V0 | Verdict |
|---|---|---|---|
| **V1.0 default** (value ranking, production price floor kept, every other change on) | −0.5 to −1.5 pts; significantly lower in 2 of 9 arm × scenario cells | no significant difference in any cell (−0.04 to +0.03 OMR) | **No gain on take-up or revenue.** The gains are in robustness, measurement, compliance and privacy (below). |
| **V1.0 with `FSDR_OFFER_PRICE_FLOOR=none`** (value ranking may also offer below the line's usual spend) | **+0.5 to +2.2 pts**; significant in 5 of 9 cells | **+0.08 to +0.22 OMR**; CI above 0 in 9 of 9 cells | **The revenue lever.** The revenue uplift vs control rises by +40 % to +281 %. It changes BR-78…82, so it **needs business sign-off**. |
| V1.0 with `FSDR_OFFER_POLICY=price_band` (only the non-offer changes) | ≈ 0 (no significant difference) | ≈ 0 | The evaluation, promotion, uplift, feature and consent changes are neutral on the metrics. |
| Robustness of the weekly cycle | — | — | **V0 refused 8–9 of 10 retrainings, and 3 of 5 V0 runs crashed at the 4th campaign** (stale champion > 14 days). V1.0 promoted 4–7 of 10 and never crashed. |

**Where the revenue comes from.** Production's price bands leave **13–19 % of each arm without any offer**: a
line whose ARPU anchor (mean + std of revenue) is above every price of its families gets nothing (BR-82). The
other lines get 3.6 offers on average. Without the floor, the value policy reaches 95 % of the lines (the other
5 % have no consent) with 4 offers each, choosing per line between a cheaper offer that is likely to be taken
and a dearer one. Inside the floor, the value ranking picks offers about 1.7 OMR dearer than production's, loses
a few takers, and revenue stays flat. Within production's price range, the acceptance model does not yet beat
production's ranking; ranking the 4 offers as a set and calibrating P per validity are the next steps.

**Caveats:**
- The world is synthetic: its response function rewards fit and penalises the price gap above the hidden
  spend (`nbx/fsdr/synth/world.py`). The numbers show the direction and the mechanism, not a forecast for real
  customers.
- An arm has about 500 lines a week, so a 1-point take-up difference is close to the detection limit. The
  feedback report now prints the detectable uplift (#13).
- The llm arm runs offline (rule-based agents); with vLLM it would be the same engine plus the agents'
  overrides.
- Reproduce with `scripts/compare_v0_v1.py` and `scripts/compare_analyze.py`. The full numbers are in
  [experiments/v0_vs_v1_summary.json](experiments/v0_vs_v1_summary.json).

### Paired differences vs V0, per arm (5 worlds, 95 % CI)

| Scenario | Configuration | Arm | Take-up Δ (pts) [95 % CI] | Revenue / line Δ (OMR) [95 % CI] | Revenue uplift vs control: V0 → V1.0 |
|---|---|---|---|---|---|
| baseline | **V1.0 default** | cstr | -0.66 [-1.86, +0.55] | -0.00 [-0.05, +0.05] | 0.235 → 0.234 (-0%) |
| baseline | **V1.0 default** | free | -0.86 [-2.36, +0.65] | -0.00 [-0.08, +0.08] | 0.194 → 0.191 (-1%) |
| baseline | **V1.0 default** | llm | -0.69 [-1.96, +0.59] | -0.01 [-0.07, +0.05] | 0.240 → 0.232 (-3%) |
| baseline | V1.0, floor removed | cstr | +1.13 [-0.12, +2.38] | +0.16 [+0.00, +0.32] | 0.235 → 0.373 (+59%) |
| baseline | V1.0, floor removed | free | +2.10 [+1.52, +2.68] | +0.22 [+0.14, +0.30] | 0.194 → 0.394 (+103%) |
| baseline | V1.0, floor removed | llm | +1.02 [-0.53, +2.57] | +0.12 [+0.04, +0.20] | 0.240 → 0.335 (+40%) |
| baseline | V1.0, price bands | cstr | -0.61 [-1.38, +0.15] | -0.04 [-0.11, +0.03] | 0.235 → 0.174 (-26%) |
| baseline | V1.0, price bands | free | +0.20 [-1.53, +1.93] | +0.03 [-0.05, +0.10] | 0.194 → 0.196 (+1%) |
| baseline | V1.0, price bands | llm | -0.14 [-1.39, +1.11] | -0.02 [-0.08, +0.04] | 0.240 → 0.191 (-20%) |
| price_sensitive | **V1.0 default** | cstr | -0.74 [-1.45, -0.03] | -0.01 [-0.03, +0.01] | 0.106 → 0.079 (-25%) |
| price_sensitive | **V1.0 default** | free | +0.35 [-0.66, +1.36] | +0.03 [-0.03, +0.09] | 0.065 → 0.076 (+18%) |
| price_sensitive | **V1.0 default** | llm | -0.56 [-1.81, +0.70] | -0.03 [-0.11, +0.04] | 0.139 → 0.090 (-35%) |
| price_sensitive | V1.0, floor removed | cstr | +0.52 [-0.29, +1.34] | +0.12 [+0.09, +0.14] | 0.106 → 0.235 (+122%) |
| price_sensitive | V1.0, floor removed | free | +2.17 [+1.31, +3.03] | +0.17 [+0.13, +0.21] | 0.065 → 0.247 (+281%) |
| price_sensitive | V1.0, floor removed | llm | +0.75 [-0.12, +1.63] | +0.08 [+0.01, +0.14] | 0.139 → 0.229 (+65%) |
| churn_wave | **V1.0 default** | cstr | -1.50 [-2.47, -0.54] | -0.04 [-0.12, +0.04] | 0.219 → 0.167 (-24%) |
| churn_wave | **V1.0 default** | free | -0.50 [-1.52, +0.52] | +0.03 [-0.07, +0.13] | 0.172 → 0.191 (+11%) |
| churn_wave | **V1.0 default** | llm | -0.85 [-2.74, +1.05] | +0.01 [-0.11, +0.12] | 0.234 → 0.230 (-2%) |
| churn_wave | V1.0, floor removed | cstr | +1.13 [+0.43, +1.83] | +0.16 [+0.09, +0.23] | 0.219 → 0.338 (+54%) |
| churn_wave | V1.0, floor removed | free | +1.37 [+0.07, +2.66] | +0.20 [+0.06, +0.34] | 0.172 → 0.333 (+94%) |
| churn_wave | V1.0, floor removed | llm | +0.98 [+0.06, +1.90] | +0.16 [+0.09, +0.23] | 0.234 → 0.355 (+52%) |

The relative uplift column divides by a small control difference, so it moves a lot. Read it together with the
two Δ columns.

### Levels, constraint arm (production's arm), mean over 5 worlds

| Scenario | Configuration | cstr take-up | cstr revenue / line | uplift vs cg (pts) | lines with an offer | offers per line | mean offer price | promotions (weeks 1–2, 5 worlds) |
|---|---|---|---|---|---|---|---|---|
| baseline | V0 | 7.45% | 0.362 | +3.46 | 87% | 3.60 | 6.28 | 1 / 10 |
| baseline | **V1.0 default** | 6.79% | 0.362 | +3.22 | 84% | 3.59 | 8.02 | 7 / 10 |
| baseline | V1.0, floor removed | 8.58% | 0.524 | +4.68 | 95% | 4.00 | 8.20 | 7 / 10 |
| baseline | V1.0, price bands | 6.83% | 0.326 | +2.79 | 83% | 3.60 | 6.31 | 6 / 10 |
| price_sensitive | V0 | 5.98% | 0.226 | +2.07 | 90% | 3.60 | 6.12 | 3 / 10 |
| price_sensitive | **V1.0 default** | 5.24% | 0.216 | +1.31 | 84% | 3.61 | 7.15 | 7 / 10 |
| price_sensitive | V1.0, floor removed | 6.50% | 0.342 | +3.14 | 95% | 4.00 | 8.01 | 7 / 10 |
| churn_wave | V0 | 7.33% | 0.351 | +3.55 | 88% | 3.60 | 6.27 | 2 / 10 |
| churn_wave | **V1.0 default** | 5.83% | 0.310 | +1.94 | 84% | 3.62 | 8.18 | 4 / 10 |
| churn_wave | V1.0, floor removed | 8.46% | 0.509 | +4.30 | 95% | 4.00 | 8.21 | 7 / 10 |

"Lines with an offer" in V1.0 includes the ~5 % of lines without consent, which are never contacted (#6). V0
contacts them.

**Recommendation.**
1. **Keep the default** (production's floor) in production: it respects BR-78…82 and fixes the robustness,
   measurement, compliance and privacy issues.
2. **Run `FSDR_OFFER_PRICE_FLOOR=none` as a business-approved pilot on one arm.** It is the lever that moved
   take-up and revenue in every scenario. The weekly report gives its uplift vs control with a CI, and the
   fairness view by nationality, so the pilot can be judged on real customers.

---

## 1. The changes, finding by finding

| # | Area | What V1.0 does | Files | Switch (V1.0 → V0) | How it acts on recommendations / revenue / take-up |
|---|---|---|---|---|---|
| 10, 14 | **Offer arbitration, cold start** | An acceptance model P(accept \| customer, action, offer) is trained each week on the offers sent last week and the fulfilments that followed. In each BR-66 family, the offer with the highest P × price wins. The inputs are offer attributes, so an offer never sold gets a score too. 5 % of picks explore a random offer of the family. | `nbx/fsdr/acceptance.py` (new), `pipelines/recommend.py` (`OfferEngine.select_value`), `offer_rules.py` (`Pick.p_accept`), `agents/tools.py` (`expected_value`) | `FSDR_OFFER_POLICY=value` → `price_band` | With production's price floor (default, `FSDR_OFFER_PRICE_FLOOR=anchor`): picks about 1.7 OMR dearer, take-up −0.5 to −1.5 pts, revenue unchanged. **Without the floor** (`none`, needs sign-off): it reaches the 13–19 % of lines production leaves without an offer and trades price against acceptance per line, giving **+0.08 to +0.22 OMR revenue per line and +0.5 to +2.2 pts take-up**. The agent's `expected_value` tool now uses the trained model instead of a hand-set sigmoid. |
| 2 | Evaluation | The holdout is evaluated for the policy that is deployed: actions sampled from the PMF (BR-57), weight π(a\|x)/p. Greedy IPS is kept for reference. | `pipelines/train.py` (`evaluate`), `stats.py` (new) | `holdout_ips_greedy_take_rate` = V0 metric | The promotion decision measures the policy that actually reaches customers. |
| 3 | Promotion | The champion is scored on the challenger's holdout (out-of-time for the champion); a paired bootstrap gives the 95 % CI of the SNIPS gain. The challenger is refused only if it is significantly worse; significantly better or inconclusive → promoted (the fresher model). | `registry.py` (`can_promote`), `train.py` (`_champion_for_eval`) | `FSDR_PROMOTION_RULE=paired_ci` → `point` | V0 compared point IPS from **different weeks**. It refused 9 of 10 retrainings in the 4-week test (week 0's IPS is inflated by the share propensities, #1), so the week-0 model stayed in charge, and **3 of 5 runs crashed** at week 3 on a stale champion (> 14 days). V1.0 keeps learning every week. |
| 1 | Propensity | Legacy targeting labels get propensity 1.0: the segment → action policy (BR-39) is deterministic. The week is flagged `ope_valid=false`, and promotion falls back to the point rule and says so. | `labels.py` (`build_labels`) | `FSDR_TARGETING_PROPENSITY=deterministic` → `share` | Week 0's model no longer up-weights rare actions by 1/share (×125 for CROSS SELL DATA). Its IPS is no longer an inflated benchmark later models can't beat. |
| 4 | Uplift reward | T-learner baseline: the control group's take rate, modelled from the features (logistic regression with ≥ 20 control takers, else the smoothed mean). The bandit's cost becomes −(reward − baseline). The labels report gives the lift per action. | `labels.py` (`uplift_adjust`), `train.py` | `FSDR_UPLIFT_REWARD=true` → `false` | The bandit learns what a contact *adds*, not who buys anyway. The effect is small in this simulator, where organic buying does not depend on the action; it matters in real data, where it does. |
| 6 | Consent | New contract table `consent_contact` (dnd, marketing_opt_in). The rule is hard in every arm: no offer in cstr / free, and NO_CONTACT forced by `contact_policy` in the llm arm. A missing table is a validation warning. | `schemas.py`, `features.py` (`consent_blocked`), `recommend.py`, `agents/tools.py`, `synth/generator.py` | remove the table | ~5 % of lines are no longer contacted: a compliance cost, measured below. |
| 7 | Fairness | The feedback report shows, per arm and nationality group: take-up, revenue, mean offer price, action mix, and the min / max ratios. `FSDR_USE_NATIONALITY=false` drops the feature from the model. | `pipelines/feedback.py` (`fairness`), `features.py` (`nationality_groups`), `config.py` | `FSDR_USE_NATIONALITY` | Makes disparate treatment visible. Whether nationality may be used is a legal decision (review Q4). |
| 8 | PII | Prompts and the `customer_360` tool carry `C-<salted HMAC>` instead of the MSISDN. | `nbx/core/privacy.py` (new), `agents/agentic/react.py`, `agents/tools.py` | — | None on decisions. The MSISDN never reaches the LLM or its logs. |
| 9 | Pricing | The anchor growth guard is on by default: the ARPU anchor grows at most +10 % a week vs the anchor logged last time. | `config.py` | `FSDR_ANCHOR_MAX_GROWTH=0.10` → `off` | Stops the escalation where campaign purchases raise next week's anchor, and therefore next week's prices. |
| 11, 12 | Features | `PROBA_<x>_MISSING` flags (9) and `LOG_TENURE`: 61 model features instead of 51. | `features.py` | `FSDR_EXTENDED_FEATURES=true` → `false` (production's list) | The bandit can tell "score 0.01" from "no score". Tenure feeds the early-churn and onboarding contexts. |
| 13 | Reporting | Every uplift has a 95 % CI and p-value. Each arm reports the smallest uplift its size can detect at 80 % power, and `stats.control_size_for` sizes the control group. The simulation report pools all weeks (`pooled`). | `stats.py`, `pipelines/feedback.py`, `pipelines/simulate.py` | — | Signal can be told from noise; the 25 % control share can be sized instead of fixed. |
| 15 | Performance | The campaign log and rewards are cached on (size, mtime); an append parses only the new bytes. | `campaign_log.py` | — | V0 re-read the whole JSONL on every append, which is quadratic over the weeks. |
| 16 | MLOps | Registry writes take a lock file and are atomic (temp + replace). Adds a Dockerfile (uv, non-root, healthcheck) and GitHub Actions CI (uv sync --locked, tests on Linux and Windows, Docker build). The uv lock only takes releases ≥ 2 weeks old. | `registry.py`, `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, `pyproject.toml` | — | Two trainings can no longer interleave registry updates. The Docker build and the CI were written but not run here (no Docker or GitHub runner on this machine). |
| 5 | Security | API key on `/fsdr/*` and `/competitors/*`; forced promotion needs an admin key; CORS restricted; the launcher warns when the API is exposed without a key. | `nbx/api/security.py` (new), `api/main.py`, `api/fsdr.py`, `run.py` | `FSDR_API_KEY` empty = open (local dev) | — |
| 19 | Secrets | The vLLM key goes through the environment, so it no longer appears in the startup log. | `vllm/serve_wsl.sh` | — | — |
| 18 | Code | FSDR-only package. The NBX demo, council, vector stores, embeddings and Ollama are removed; competitor search is keyword-based (no Chroma). uv project. | whole tree | — | 84 locked packages instead of the V0 stack (PyTorch, Chroma, sentence-transformers). |

Not done in V1.0, and still open:
- **#17**, a bigger model for the llm arm: hardware.
- **#20**, the minor leftovers from the verification report.
- Real-time decisioning, CLV objectives, offer-level OPE (MIPS) and budget-constrained allocation: the strategic
  roadmap of the review.
- Offer **margin and cost**: `FSDR_OFFER_MARGIN` is a single factor until the data contract carries them
  (review Q2).

---

## 2. Business rules: what is unchanged and what moved

**Unchanged hard rules, code-enforced and tested in `tests/fsdr/test_fsdr_offers.py` for both policies:**
- eligibility (BR-01 … BR-12);
- Nursering exclusion (BR-36);
- the action → family mapping (BR-66) and the free arm's L2 DATA + COMBO (BR-67);
- at most 2 offers per validity and 4 in total, and no offer when the anchor is above every price (BR-82, with
  the default price floor);
- the catalogue only (BR-63);
- conflicting pairs never sent together;
- the contact gap and the 28-day cap;
- the churn block on upsell actions;
- the IPS rule that the agents cannot bypass.

**What moved, under `FSDR_OFFER_POLICY=value`:**
- **The ranking inside the allowed price range.** The three price-band passes (BR-78 … BR-81) rank offers by
  ranker score, then price, inside the range above the ARPU anchor. The value policy ranks the same offers by
  P(accept) × price and records pass 4 (value) or pass 5 (exploration).
- **The price floor is kept by default** (`FSDR_OFFER_PRICE_FLOOR=anchor`): every offer is priced above the
  line's anchor, and a line whose anchor is above every family price gets no offer, as in BR-82.
- **`FSDR_OFFER_PRICE_FLOOR=none` removes the floor.** The value policy may then offer below the line's usual
  spend, and reach the 13–19 % of lines that production leaves without any offer. This changes BR-78 … BR-82. It
  is measured separately below and **needs business sign-off before use**.
- **`price_band` restores BR-78 … BR-81 exactly.** It is also the automatic fallback while the acceptance model
  has fewer than 20 accepted offers to learn from.

**New hard rule:** consent (#6).
