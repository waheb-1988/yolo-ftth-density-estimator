# FSDR NBA/NBO — Technical and Scientific Review

Read-only review of `nbx-python-upgrade` at commit `ae800fc`, done on 2026-10-26. During the review, no file was
changed and no git write, install or test command was run. The test suite rewrites a git-ignored index file
(`data/chroma/chroma.sqlite3`), so it was not run. The last results on the same commit, with an unchanged tree, are
261 passed, 5 runs in a row (2026-10-25).

**Skills.** None were loaded:
- **No relevant skill exists:** no project skills, and no user skill on recommenders, NBO/NBA or telecom.
- **`code-review` and `security-review`** review a diff or pending changes; there was none.
- **`simplify`** applies fixes, which breaks the read-only rule.
- **`deep-research`** spawns sub-agents; the literature step was done directly with web search.
- **Browser, Office, docs and artifact skills** are unrelated.

---

## A. Executive summary

**Health: 6.5 / 10.**

**Strengths:** 261 tests; a faithful port of production's business rules (BR-01 … BR-86); hard rules enforced in
code; logged propensities; an append-only campaign log; a closed-loop simulator.

**Weaknesses:** the causal and ML methodology, and production readiness (security, consent, CI).

**Top 3 risks**
1. **The off-policy numbers can't be trusted yet.**
   - Week-1 propensities are action *shares*, but the legacy policy was deterministic by segment.
   - IPS evaluates the *greedy* policy while production *samples* from the PMF.
   - Promotion compares point estimates with no confidence interval.
2. **The objective is take-up, not uplift.** The reward counts organic purchases, and high churn is routed to RETAIN
   by risk alone: the pattern Ascarza (2018) shows to be ineffective.
3. **Compliance gaps.** There is no consent / opt-out in the data contract; the API has no authentication (CORS
   `*`); the MSISDN is sent in LLM prompts and stored in plain text; nationality is a targeting feature.

**Top 3 quick wins**
1. IPS / SNIPS for the *sampling* policy, with bootstrap confidence intervals; promote on the lower bound.
2. An API key on `/fsdr/*`, with no public `force` promotion and restricted CORS.
3. A consent / do-not-contact flag in the data contract and a hard rule in `contact_policy`.

---

## B. Findings

| # | Area | File:line | Issue | Impact | Severity | Recommended improvement | Effort |
|---|---|---|---|---|---|---|---|
| 1 | Modeling / OPE | `nbx/fsdr/labels.py:104-109` | Week-1 propensity = this week's action share. The legacy policy assigns the action from the segment (`segment_action`), so P(action \| context) is about 1 or 0, not a share | IPS / SNIPS on targeting labels violate positivity: biased, unusable for promotion | High | Use logged propensities only (the campaign log has existed since 2026-10-24). Until then, randomise part of the legacy assignment, or treat week-1 metrics as non-causal | M |
| 2 | Evaluation | `nbx/fsdr/pipelines/train.py:79-88` vs `nbx/fsdr/config.py:261` | IPS matches the **greedy** argmax with the logged action, while production **samples** from the PMF (BR-57) | The metric does not measure the deployed policy | High | IPS / SNIPS / doubly-robust estimates for the sampling policy: Σ π(a\|x)·r / p | S |
| 3 | Evaluation / MLOps | `nbx/fsdr/registry.py:79`, `nbx/fsdr/pipelines/train.py:133` | Promotion on a point-estimate IPS from a single random 20 % holdout (about 220 rows) of the same week; no interval, no out-of-time check | Noisy promotions (e.g. 0.215 vs 0.133 on 221 rows) | High | Bootstrap or high-confidence lower bounds (Theocharous et al. 2015); out-of-time validation (train week t, evaluate on the t+1 logs) | M |
| 4 | Objective | `nbx/fsdr/labels.py:97`; `nbx/fsdr/config.py` `CHURN_BLOCK_THRESHOLD`; `agents/recommender_agent.py` `guard_action` | Reward `any_btl` includes organic purchases (control take rate about 2–4 %); high churn is forced to RETAIN | The model learns "who buys anyway"; retention targets by risk, not by lift | High | Uplift reward: (treated − control) per segment, using the cg arm; T / X-learners (Künzel et al. 2019); retention targeted on lift | L |
| 5 | Security | `nbx/api/main.py:57-58`; `nbx/api/fsdr.py:188-191` | No authentication; CORS `allow_origins=["*"]`; `/fsdr/upload`, `/fsdr/simulate` and `/models/{v}/promote?force=true` are open; `--host 0.0.0.0` exists | Anyone who reaches the port can upload data or force a worse model in, bypassing the IPS rule | High | API key / OAuth dependency on `/fsdr/*`; force-promotion behind an admin role; restricted CORS | S |
| 6 | Compliance | `nbx/fsdr/schemas.py` (no consent table); `nbx/fsdr/agents/tools.py` `contact_policy` | No consent / opt-out / do-not-contact input anywhere (code search found none) | Contacting opted-out lines: Oman PDPL / GDPR risk | High | A `consent` table in the contract; a hard rule in `contact_policy`; a filter in the cstr / free arms | M |
| 7 | Fairness | `nbx/fsdr/config.py:242` | `NATIONALITY_GROUP` is a model feature; the action drives the offer family and the price band | Possible disparate treatment by nationality | Med-High | Fairness audit per group (take rate, price, action mix); BEAT-style debiasing (Ascarza & Israeli 2022), or drop and test | M |
| 8 | PII | `nbx/agents/agentic/react.py:72`; `nbx/fsdr/campaign_log.py` | The MSISDN goes into every LLM prompt; the campaign log and rewards store MSISDNs in plain-text JSONL; no retention policy | PII exposure if the LLM is ever remote; no data minimisation | Med | Pseudonymise (salted hash) in prompts and logs; retention and encryption at rest | S-M |
| 9 | Pricing / cannibalisation | `nbx/fsdr/offer_rules.py:88` (anchor = mean + std), `:195` (pass 1 sorts price descending); `ANCHOR_MAX_WEEKLY_GROWTH = None` | Production logic pushes offers above usual spend, and campaign purchases raise next week's anchor; the growth guard is off by default | Price escalation; cannibalisation of paid subscriptions | Med | Turn the guard on; exclude campaign purchases from the anchor; track net ARPU, not take rate | S |
| 10 | Arbitration | `nbx/fsdr/agents/tools.py:402-416` | "Expected value" is a hand-set sigmoid (−2.2, 1.4, −0.8) × price; no margin, cost or cannibalisation anywhere in ranking | Ranking ≠ profit; the LLM is guided by a heuristic | Med | Rank offers on expected incremental margin (Olaya et al. 2023; Lemmens & Gupta 2020) | M |
| 11 | Features | `nbx/fsdr/features.py:183-189` | A missing score becomes uniform(0, 0.02), with no missing-indicator feature | Hides missingness and adds noise (kept for production parity, BR-47) | Med-Low | Add `PROBA_*_MISSING` flags; keep BR-47 for the parity mode | S |
| 12 | Features (telecom) | `nbx/fsdr/features.py` (`numeric_features`), `nbx/fsdr/schemas.py` | No recharge / top-up, tenure (present in the score files), device (`dv_segment` only feeds the cohort tool), network experience or plan features | Weaker context for upsell and retention decisions | Med | Extend the contract (recharge, network quality) and add the features behind a flag; the parity tests stay on the 48 | M-L |
| 13 | Experiment design | `nbx/fsdr/config.py:163`; `nbx/fsdr/pipelines/feedback.py` (`summarise`) | Control fixed at 25 %; uplift reported without a test or interval | 25 % of the base never contacted; no way to tell signal from noise | Med | Power analysis to size cg; intervals / tests on uplift in the report | S |
| 14 | Cold start (offers) | `nbx/fsdr/pipelines/recommend.py` (`items`, default `RANKER_ITEMS=history`) | New offers are never proposed under the production default | New products never reach customers | Med | Explore new offers on a small ε share, or a content-based prior | S-M |
| 15 | Performance | `nbx/fsdr/bandit.py:70-71`; `nbx/fsdr/pipelines/recommend.py:101`; `nbx/fsdr/campaign_log.py:55` | One VW predict per row in Python; a Python loop per customer; every log append re-reads the whole JSONL | O(N) per append, so quadratic over time; slow at 200k lines | Med | Batch / multi-line VW prediction; vectorised selection; parquet or SQLite log with a key index | M |
| 16 | MLOps | repo root (no `.github/`, no `Dockerfile`); `nbx/fsdr/pipelines/train.py:36`, `nbx/api/fsdr.py:23` | No CI, container or scheduler; file registry without locks; global caches not thread-safe | Fragile deployment; concurrent training would race | Med | CI running the tests; a Dockerfile; a scheduled weekly job; a file lock or MLflow | M |
| 17 | LLM arm | `nbx/fsdr/agents/recommender_agent.py`; `docs/FSDR_HANDBOOK.md` §4.4 | 3B model on a 4 GB GPU: 25–150 s per decision; reasoning errors observed (a mis-read churn score) | Cannot run a full arm; quality depends on the guards | Med | Bigger model on a server GPU; an LLM-quality eval set; latency targets | M |
| 18 | Code quality | `nbx/ui/gradio_app.py` (956 lines), `nbx/api/main.py` (744 lines); two `config.py` | Large modules; NBX demo and FSDR mixed in one package; config frozen at import time | Maintainability | Low | Split routers and UI modules; a typed settings object (pydantic-settings) | M |
| 19 | Secrets | `vllm/serve_wsl.sh:18,25` | vLLM prints its arguments, including the API key, in its startup log | The key leaks into the log | Low | Pass the key through env or a file and redact the log | S |
| 20 | Leftover MINORs | `docs/fsdr_verification_report.md` (re-check section) | E5 no warning, I4 rounding, L8 cache key uses only score features, Q11 `nb_sub` always 1, F7 float32 | Small correctness / traceability gaps | Low | As listed in the re-check | S |

---

## C. Telecom NBO/NBA gaps compared with a mature platform

| Capability | Status in FSDR |
|---|---|
| Real-time decisioning (event at recharge or low balance, sub-second response) | Weekly batch only |
| Uplift / incrementality models, Qini-based evaluation | Take rate only |
| Value-based arbitration (margin × incremental probability − cost), with budget and channel-capacity constraints | None |
| Consent, DND, quiet hours, channel preferences | Only a 2-day gap and a 3-per-28-days cap |
| Multi-channel orchestration (SMS, app push, USSD, call centre) | SMS scripts only |
| Cannibalisation and portfolio guardrails (net ARPU, existing subscriptions) | Partial: optional anchor growth guard |
| CLV-based / sequential (RL) optimisation | Single-step bandit |
| Monitoring dashboards (take and uplift by arm), alerting, drift beyond PSI, calibration monitoring | Reports + PSI only |
| Feature store, scheduled pipelines, CI/CD, containers, authentication, audit trail for LLM decisions | Not built yet |

---

## D. Literature comparison

| Paper | Key idea | Used in code? | Gap | Suggested improvement |
|---|---|---|---|---|
| Li, Chu, Langford & Schapire 2010 (LinUCB) | Recommendation as a contextual bandit with upper-confidence bounds on linear payoffs. Evaluated offline on randomly logged Yahoo! traffic (+12.5 % clicks) | **Partial**: VW `cb_explore_adf`, ε-greedy 0.2 (`config.py` `VW_ARGS_TEMPLATE`) | Fixed ε; no UCB or adaptive exploration; no random-bucket evaluation | Try VW `--squarecb`, `--cover` or bagging; keep a small uniformly random bucket for unbiased offline evaluation |
| Gutierrez & Gérardy 2017 | Uplift modelling is both causal inference and ML; reviews the two-model, class-transformation and direct approaches | **No** | The control arm exists but is used only for reporting | Two-model / class-transformation uplift from cg vs treated logs (`campaign_log.py` holds both) |
| Devriendt, Moldovan & Verbeke 2018 | Benchmarks uplift methods; uplift random forests are often best on Qini, but models are unstable across folds | **No** | No Qini or uplift curves | Add Qini / AUUC to `feedback.summarise`; test an uplift forest |
| Künzel, Sekhon, Bickel & Yu 2019 (PNAS) | Meta-learners (S / T / X) for heterogeneous treatment effects; the X-learner suits unbalanced groups | **No** | — | A T / X-learner per action with cg as control, feeding the action choice |
| Verbeke et al. 2012 (EJOR) | A profit-based measure (maximum profit) for selecting churn models; benchmark on 11 telecom datasets | **No** | Models and promotion are judged on take / IPS, not profit | A profit-based promotion metric |
| Ascarza 2018 (JMR) | Targeting retention by *lift* beats targeting by churn *risk* (field data, including a wireless operator) | **No; the opposite**: PROBA_CHURN ≥ 0.60 → RETAIN (`CHURN_BLOCK_THRESHOLD`, `guard_action`) | Risk-based retention routing | Route retention by estimated lift; keep the churn block only as a guard |
| Lemmens & Gupta 2020 (Marketing Science) | A profit loss function for retention targeting; field experiments show more profitable campaigns | **No** | No offer cost or margin | A profit-weighted reward / loss; offer cost in arbitration |
| Theocharous, Thomas & Ghavamzadeh 2015 (IJCAI) | Reinforcement learning for lifetime value in ad recommendation, with high-confidence off-policy evaluation before deployment | **Partial**: IPS / SNIPS (`train.py:75-90`), myopic | No confidence bounds; no lifetime value | High-confidence lower bound for promotion; later a CLV or multi-week reward |
| Ascarza & Israeli 2022 (PNAS), BEAT | Unintended bias comes from correlations with protected attributes; dropping them can hide or worsen it; BEAT balances the allocation | **No** | Nationality used as a feature; no fairness check | Fairness audit; BEAT-style targeting |
| Saito & Joachims 2022 (ICML), MIPS | Off-policy evaluation for large action spaces through action embeddings (marginalised importance weights) | **Partial**: plain IPS / SNIPS over 7 actions | No off-policy evaluation at the offer level (66 offers) | MIPS with offer embeddings (service, generosity, price) for offer-level evaluation |
| Olaya, Verbeke, Van Belle & Guerry 2023 (EJOR) | Cost-sensitive causal classification: an expected-causal-profit ranker built from the treatment effect, the outcome probability and costs | **No** | Offers ranked by BPR score and price | Rank offers and actions by expected causal profit |
| Sapru 2025, arXiv 2512.19805 (**preprint, not peer-reviewed**) | Uplift estimates followed by constrained allocation (budget, revenue protection), with an A/B test | **No** | No budget or capacity constraints | Constrained allocation over arms and offers. An industry idea, not evidence |

The search for a telecom NBA reinforcement-learning paper from 2022–2024 returned only vendor material (Grid Dynamics,
BCG, Subex). No such peer-reviewed paper could be verified, so none is cited.

---

## E. Prioritized roadmap

### Quick wins (≤ 1 week)
- **Evaluation:** IPS / SNIPS for the sampling policy, with bootstrap intervals; promotion on the lower bound (#2, #3).
- **API:** a key on `/fsdr/*`, no public `force`, restricted CORS (#5).
- **Reports:** intervals or tests on uplift; size the control group with a power analysis (#13).
- **Small switches:**
  - pseudonymise the MSISDN in LLM prompts (#8);
  - turn on `FSDR_ANCHOR_MAX_GROWTH` and add missing-score flags (#9, #11);
  - redact the vLLM key from the startup log (#19).

### Short term (1 month)
- Consent / DND table and hard rule (#6); fairness audit by nationality (#7).
- An uplift layer: T / X-learner over control vs treated from the campaign log, with Qini in the report (#4).
- Profit-based arbitration: margin, offer cost, cannibalisation (#10).
- CI pipeline, Dockerfile, scheduled weekly job, locked model registry (#16).
- Batch VW prediction; a parquet or SQLite campaign log (#15).
- Exploration for new offers (#14).

### Strategic (quarter)
- Real-time event decisioning and multi-channel orchestration, with channel caps.
- A CLV / multi-week objective (Theocharous et al. 2015); offer-level off-policy evaluation (Saito & Joachims 2022);
  budget-constrained allocation.
- Telecom features from a feature store (recharge, network experience, device).
- Monitoring dashboards and a model card.
- Governance for the LLM arm: a bigger model, an evaluation set, an audit trail.

---

## F. Questions and assumptions

1. **Is the legacy segment → action assignment fully deterministic?** If any randomisation existed, finding #1
   softens.
2. **Do offer margin and cost data exist?** The data contract has prices only.
3. **Is there a consent / DND source** (and are there channel preferences) that we can export?
4. **May nationality be used for targeting,** under your legal / PDPL review?
5. **Is real-time decisioning in scope,** or does the platform stay weekly batch?
6. **Will the LLM ever be hosted remotely?** If yes, findings #8 and #17 become High.
7. **Can the reference generator `fsdr_generate_dummy.py` be shared?** Generator parity (audit check C1) is still
   unverified.

**Assumption:** the simulator's results (uplift vs control) reflect the **synthetic** response function, not real
customer behaviour.

---

## G. References

1. Li, L., Chu, W., Langford, J., Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article
   Recommendation. *WWW 2010*, 661–670. https://doi.org/10.1145/1772690.1772758 ·
   [arXiv:1003.0146](https://arxiv.org/abs/1003.0146)
2. Gutierrez, P., Gérardy, J.-Y. (2017). Causal Inference and Uplift Modelling: A Review of the Literature. *PMLR*
   67:1–13. https://proceedings.mlr.press/v67/gutierrez17a.html
3. Devriendt, F., Moldovan, D., Verbeke, W. (2018). A literature survey and experimental evaluation of the
   state-of-the-art in uplift modeling: a stepping stone toward the development of prescriptive analytics.
   *Big Data* 6(1):13–41. https://doi.org/10.1089/big.2017.0104
4. Künzel, S. R., Sekhon, J. S., Bickel, P. J., Yu, B. (2019). Metalearners for estimating heterogeneous treatment
   effects using machine learning. *PNAS* 116(10):4156–4165. https://doi.org/10.1073/pnas.1804597116
5. Verbeke, W., Dejaeger, K., Martens, D., Hur, J., Baesens, B. (2012). New insights into churn prediction in the
   telecommunication sector: a profit driven data mining approach. *European Journal of Operational Research*
   218(1):211–229. https://doi.org/10.1016/j.ejor.2011.09.031
6. Ascarza, E. (2018). Retention Futility: Targeting High-Risk Customers Might Be Ineffective. *Journal of Marketing
   Research* 55(1). https://doi.org/10.1509/jmr.16.0163 (the DOI resolves to journals.sagepub.com)
7. Lemmens, A., Gupta, S. (2020). Managing Churn to Maximize Profits. *Marketing Science*.
   https://doi.org/10.1287/mksc.2020.1229
8. Theocharous, G., Thomas, P. S., Ghavamzadeh, M. (2015). Personalized Ad Recommendation Systems for Life-Time Value
   Optimization with Guarantees. *IJCAI 2015*, 1806–1812. https://www.ijcai.org/Abstract/15/257
9. Ascarza, E., Israeli, A. (2022). Eliminating unintended bias in personalized policies using bias-eliminating
   adapted trees (BEAT). *PNAS* 119(11): e2115293119.
   https://ideas.repec.org/a/nas/journl/v119y2022pe2115293119.html
10. Saito, Y., Joachims, T. (2022). Off-Policy Evaluation for Large Action Spaces via Embeddings. *ICML 2022*,
    PMLR 162:19089–19122. https://proceedings.mlr.press/v162/saito22a.html
11. Olaya, D., Verbeke, W., Van Belle, J., Guerry, M.-A. (2023). To do or not to do? Cost-sensitive causal
    classification with individual treatment effect estimates. *European Journal of Operational Research*
    305(2):838–852. https://ideas.repec.org/a/eee/ejores/v305y2023i2p838-852.html ·
    [arXiv:2101.01407](https://arxiv.org/abs/2101.01407)
12. Sapru, D. (2025). Guardrailed Uplift Targeting: A Causal Optimization Playbook for Marketing Strategy.
    *arXiv preprint* 2512.19805 (not peer-reviewed). https://arxiv.org/abs/2512.19805

**Sources used to verify the citations:**
- Li et al.: [Microsoft Research PDF](https://microsoft.com/en-us/research/wp-content/uploads/2016/02/p661.pdf)
- Künzel et al.: [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC6410831)
- Verbeke et al.: [Southampton repository](https://eprints.soton.ac.uk/336471)
- Ascarza: [AMA award page](https://www.ama.org/2019/04/24/escarzas-retention-futility-targeting-high-risk-customers-might-be-ineffective-wins-2018-award/)
- Lemmens & Gupta: [INFORMS supplement](https://pubsonline.informs.org/doi/suppl/10.1287/mksc.2020.1229)
- Saito & Joachims: [ICML 2022 poster](https://icml.cc/virtual/2022/poster/16729)
- Theocharous et al.: [MLAnthology](https://mlanthology.org/ijcai/2015/theocharous2015ijcai-personalized)
- Industry pages from the telecom RL search (not cited as evidence):
  [Grid Dynamics](https://www.griddynamics.com/blog/building-a-next-best-action-model-using-reinforcement-learning),
  [BCG](https://web-assets.bcg.com/pdf-src/prod-live/the-science-behind-next-best-action-programs.pdf)
