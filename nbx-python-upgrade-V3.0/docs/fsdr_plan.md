# FSDR NBX Recommender V3.0: build plan

Status: **V3.0**. Built: Phases 1–3 (contract and T0; features T2; segments and the two actions T3, §17) and the V3.0 changes: the `.csv.gz` contract and the four groups T1C / T1F / T2C / T2F + control (§18). Where §1–§16 say "Target 1 (20k) / Target 2 (20k) / 40k", read the four groups of §18.

Spec of record: `docs/fsdr_reference/PROMPT_FSDR_MASTER.md` (the master prompt). Section numbers below (§) refer to it.
Starting point: **V1.0** (`nbx-python-upgrade-V1.0`, 216 tests, 45 contract tables, vectorised for 1.7 M lines).

---

## 1. What changes from V1.0 to V2.0, in one picture

| Topic | V1.0 (today) | V2.0 (master prompt) |
|---|---|---|
| Who chooses the action | VW contextual bandit (`cb_explore_adf`); the LLM agent runs only on the `llm` arm | **③ LLM NBA agent** scores every allowed action. Code samples the action with ε and logs the propensity. The bandit becomes **⑩**, a helper that only provides evidence (E-A5) |
| Who ranks the offers | BPR ranker + acceptance model + value policy | **④ LLM NBO agent** scores the rule-filtered candidates. BPR is replaced by **⑪** (ALS or truncated SVD on the full base), a helper that only provides evidence (E-O6) |
| Business rules | Production rules + V1 value policy behind `.env` switches | Rules live in **registries** (`rules/*.yaml`, with ids D/S/B/F/H/P/C), each with a check function and a test. **Target 1** uses production rules B-20…B-32 exactly. **Target 2** uses the challenger C-v1 |
| Arms | cg / free / cstr / llm in one sample | **Target 1 (20k), Target 2 (20k), control (10k, outside the 40k)** |
| Final file | Offers per arm (JSON) | **`input_flytxt`**: exactly the 38 columns of code 5, plus a parity test against a replay of code 5 |
| Pipelines | One weekly runner | **Two pipelines**: `fsdr-train`, which builds a validated **bundle**, and `fsdr-score`, which only reads the bundle |
| Safety | Promotion rule, price floor | **⑨ auditor** (A/T/R checks, blocks the push), **⑬ revenue guard** (CAN-01…07), **H-08**: no push without audit PASS and human approval |
| Traceability | Campaign log, LangSmith (opt-in) | **⑫ customer ledger**: one row per MSISDN per week (groups A–K), plus a version registry |
| Data engine | pandas on JSON (typed) | **Parquet cache + polars** for the 1M-line base, pandas only on the 40k. No per-customer loops |

What V1.0 already gives us and V2.0 keeps:
- the 45-table contract, calendar D/T and validation;
- vectorised key normalisation and type checks;
- the synthetic world and closed-loop simulator;
- VW format and IPS/SNIPS/bootstrap statistics;
- the campaign log and feedback;
- pseudonymised prompts;
- the journey UI;
- the vLLM client and opt-in LangSmith;
- uv packaging and CI.

## 2. Environment check (Phase 1 evidence)

| Item | This laptop (dev) | Reference laptop (prod, §19.5) |
|---|---|---|
| CPU / RAM | i7-10750H 6C/12T, **15.8 GB** (WSL sees 7 GB) | Core Ultra 9 275HX, 24 cores (8 P up to 5.4 GHz + 16 E), **128 GB DDR5-5600** |
| GPU | GTX 1650 **4 GB**, driver 592.00, CUDA 13.1 in WSL | RTX PRO 5000 Blackwell Laptop **24 GB GDDR7** (≈ 900 GB/s, FP8/FP4 tensor cores) |
| Disk | **22 GB free on C:** (WSL's virtual disk lives on C:) | 1 TB NVMe PCIe Gen4 |
| OS | Windows 11 + WSL2 Ubuntu, Python 3.12.3, vLLM venv: OK | Windows 11 Pro + WSL2: to install, vLLM built for Blackwell (sm_120, CUDA ≥ 12.8) |
| vLLM 1-prompt smoke test | see the end of this file | to run in Phase 1 on the P16 |

Consequences:
1. **This laptop is a development machine, not the production machine.**
   - It can run the 2,000-line and 100k-line modes and the fake-LLM test suite.
   - It can run a 3B model for a small number of real LLM decisions.
   - It **cannot** run 80k LLM decisions a week (≈ 75 s per agentic line here), the 8B/14B models, or a 1M-line JSON base.
2. **Disk:** a 1M-line base as JSON is ≈ 30–40 GB (the 14 activity tables dominate), more than the free space.
   - The generator gets a **Parquet output mode** (≈ 2–3 GB at 1M).
   - The 1M benchmark runs here on Parquet if memory allows; the 2M benchmark runs on the P16.
   - Free some space on C: before Phase 9.
3. **Level-2 LLM cost:** 80k calls fit the 2–3 h budget on the P16 only if each call is **one batched JSON scoring prompt** (≤ 1,500 tokens in, ≤ 150 out, shared prefix for vLLM prefix caching).
   - The multi-step ReAct loop of V1 (several LLM turns per customer) stays for the journey screen and explanations, not for the weekly 40k.
   - I flag this because §15 says "step cap ⚙8 per customer". I propose: ★ runs NBA and NBO as **one call each**, and the loop only re-calls NBA in the 10.1 consistency case.

## 3. Repository layout (same structure as V1.0)

```
nbx-python-upgrade-V2.0/
├── pyproject.toml / uv.lock        uv (deps from PyPI; requirements.txt exported for pip-only installs)
├── run.py                          launcher (start, api, ui, test, every fsdr-* command)
├── nbx/
│   ├── fsdr/
│   │   ├── config.py               every constant with its rule id (⚙ values)
│   │   ├── rules/                  data_rules.yaml (D), segmentation_rules.yaml (S), production_rules_offers.yaml (B),
│   │   │   │                       final_output_rules.yaml (F), guardrails.yaml (H), production_flow_rules.yaml (P),
│   │   │   │                       challenger_rules_v1.yaml (C)
│   │   │   └── registry.py         load, version, applicability, check-function lookup
│   │   ├── contract/               schemas (46 tables incl. customer_profile), loader (json/jsonl/csv → Parquet cache), ① data agent
│   │   ├── calendar.py             D-01 … D-09
│   │   ├── features/               B-01 … B-12 in polars (full base), customer_360
│   │   ├── segmentation/           S-rules, bands, micro-segments, crosswalk, ②b dictionaries
│   │   ├── evidence/               E-A0 … E-A5, E-O1 … E-O6, levels, n, Wilson / bootstrap intervals
│   │   ├── offers/                 B-20b recommendable set, candidates per target, ⑥A passes, ⑥B C-v1, code-5 engine ⑦
│   │   ├── guard/                  ⑬ revenue guard (CAN-01 … 07, repair, block list)
│   │   ├── agents/
│   │   │   ├── prompts/            versioned system and task prompts (shared ★ prompt §15)
│   │   │   ├── nba.py nbo.py orchestrator.py segment_action.py challenger.py market.py strategy.py auditor.py ledger.py
│   │   │   └── helpers/            ⑩ action_bandit.py (VW --cb_adf, logged mode, OPE gate), ⑪ offer_pattern.py (ALS/SVD, hold-out gate)
│   │   ├── panel.py                stratified Target 1 / Target 2 / control, rotation, cool-down, fatigue
│   │   ├── bundle.py               training bundle (manifest, checksums, status validated)
│   │   ├── stats.py                IPS / SNIPS / DR, CIs, chi-square balance (from V1)
│   │   ├── pipelines/              training.py, scoring.py, feedback.py, run_week.py, simulate.py, benchmark.py
│   │   └── synth/                  generator (all 46 tables, consecutive weeks, 2k / 100k / 1M modes, Parquet mode)
│   ├── core/  llms/  api/  ui/     as in V1.0 (vLLM client, privacy, competitors, FastAPI, Gradio)
└── tests/                          golden cases of §23, fake LLM, no network
```

Guard against "ML deciding": a test fails if any ML library (`vowpalwabbit`, `implicit`, `sklearn`, …) is imported under `nbx/fsdr/` outside `agents/helpers/` (§23).

## 4. Input contract (§2)
- **The 45 V1.0 tables are kept as is**, with the same names, fields, raw types and aliases.
- **Added:** `customer_profile` (msisdn, age, tenure in days) and its checks:
  - coverage of the base;
  - age outside 15–100;
  - negative tenure;
  - age disagreeing with `party_*` by more than 2 years;
  - tenure disagreeing with activation-date tenure by more than 30 days.
- The `filterd_nationality` table is accepted and ignored.
- **Input folders:** the tables arrive in 5 folders (`usage/`, `models/`, `subscription/`, `catalog/`, `others/`); see §13.
- **Loader:** detects `.json`, `.jsonl`, `.csv` or `.parquet` and converts each table once to `data/cache/<run_date>/<table>/part-NN.parquet` (hash-partitioned on the line id, §15).
  - Keys are normalised to one type before any join.
  - The raw files are deleted only after the Parquet checksum is logged (switch, default **off** on the dev laptop).
- **Weekly completeness** per pipeline, from the table-usage matrix of §17.9:
  - week 0 needs `_auto` + `_score`;
  - week n needs `_score` + dated purchases + snapshots;
  - scoring needs the bundle + `_tg` + scripts + conflicts + `targeting_auto`.
- **Internal tables:** `base_features`, `evidence_<v>`, `policy_<v>`, `feature_snapshot_<run>`, `nba_campaign_log`, `nba_rewards`, `panel_<T>` and the registry. All are Parquet except the log, which is append-only Parquet partitions.

## 5. Agents → modules → main tools

| # | Module | Decides? | LLM | Fallback |
|---|---|---|---|---|
| ① | `contract/data_agent.py` | blocks the run | explains blocking issues | — |
| ② | `segmentation/rules.py` | no (hint) | one-line reason | — |
| ②b | `segmentation/segment_action.py` | no (hint) | ≈70 calls/week write `dict_segm_actions_agent_vN` | previous version per cell |
| ③ | `agents/nba.py` | **yes** (scores → code samples) | 1 call per customer | ⑩ (confidence ≥ medium) → E-A1 → rule action → B-10 priors |
| ④ | `agents/nbo.py` | **yes** (scores) | 1 call per customer | E-O1 uplift → ⑪ → E-O3 → buyer share |
| ⑤ | `agents/market.py` | no | narrative | — |
| ⑥A / ⑥B | `offers/production.py`, `offers/challenger.py` | apply rules | ⑥B writes proposals and memos | Target 1 passes |
| ⑦ | `offers/code5.py` | applies F-01 … F-19 | summary | — |
| ⑧ | `agents/strategy.py` | no | playbook, report | — |
| ⑨ | `agents/auditor.py` | **blocks the push** | narrative only | — |
| ⑩ / ⑪ | `agents/helpers/*` | **never** | none | off-switch → evidence only |
| ⑫ | `agents/ledger.py` | no | copies justifications | — |
| ⑬ | `guard/revenue_guard.py` | repairs or clears slots | narrative | — |
| ★ | `agents/orchestrator.py` | runs ③ → ④ → ⑤ → ⑥ → ⑦ | level 1 cell policies, level 2 per customer | cell policy applied by code |

**LLM contract** (shared by ②b, ③, ④, ★):
- local vLLM only, at temperature 0 with a fixed seed;
- **schema-constrained JSON** (vLLM guided decoding);
- the reply is validated against the allowed ids, and an invented id is rejected;
- a cache keyed on (feature hash, evidence v, policy v, prompt v, model);
- prompts carry a salted pseudonym, never the MSISDN.

## 6. The core algorithms (decisions I made; tell me if you disagree)

1. **Action sampling (§9):** p = (1−ε) · s/Σs + ε/k over the allowed actions, with ε = 0.1. The logged propensity is p of the sampled action. The golden case (0.6, 0.3, 0.1) → (0.5733, 0.3033, 0.1233) is in the tests.
2. **Evidence levels (§8):** the fallback chain goes cell → nationality×tenure → nationality×age → nationality → global → B-10 priors. Thresholds: 300 contacts and 30 takers for actions, 50 buyers for offers.
   - Rates get Wilson 95 % intervals.
   - Uplift vs control gets a Newcombe interval.
   - Revenue gets a bootstrap interval.
3. **⑩ Action bandit:** VW `--cb_adf -q UA` in **logged mode**, trained on the snapshot + action sent + reward + logged propensity, so IPS is correct by construction.
   - Week 0 uses the segment-engine history and is flagged `weak`.
   - Gate: SNIPS/DR of the bandit policy vs the evidence-only policy. If it is not better, confidence is forced to `low`.
4. **⑪ Offer pattern:** implicit-feedback **ALS** (`implicit`, CPU, pip) weighted by `nb_sub`, keyed on real customer ids (this fixes the code-3 index bug). It falls back to scipy truncated SVD if `implicit` does not install.
   - Cold start uses the micro-segment average and is flagged.
   - Gate: precision@4 / recall@10 on the last 4 weeks vs popularity per micro-segment.
5. **Target 1 = production, exactly.** B-21 families (all L2 active), B-26 anchor mean+std (ddof 1), 0.9/0.1 split, passes B-28 (k = 2/3, strict), B-29, B-30, sorts of B-81. The LLM score fills the `score` column.
   - V1's value policy is **not** used in Target 1.
6. **Target 2 = C-v1:** it builds on V1's value policy, which became C-04 (score × price).
   - C-01: any recommendable offer whose service fits the action.
   - C-02: band 1.0×–1.5× of the anchor (+0.5 minimum step for upsells), widened once to 2.0×, then the Target 1 passes as fallback.
   - C-03: usage-based data/voice split.
   - C-05: do-not-disturb.
7. **Code 5 (⑦):** a vectorised re-implementation, plus a **literal row-by-row replay** of code 5 used only for the parity test A-25 (bug flag on).
   - Issues (a)–(c) are fixed by default.
   - (d): both anchors are kept.
8. **Revenue guard:**
   - Runs at P6b (recommended offers) and P7b (final 6 slots).
   - Repair order: next candidate → clear slot → drop customer (F-04).
   - The block list is updated at T+7 from net incremental revenue vs control.
9. **Panel:** stratified on `Final` and balanced on age and tenure bands (chi-square check A-31). Customers stay 4–6 weeks; replacements come from the same stratum and target. Or the provided sample is used (cstr → Target 1, free → Target 2, cg → control).
10. **Learning loop:** week n uses only our randomised campaigns vs control. The week-0 pseudo-control evidence is down-weighted, then dropped after 4 weeks. New policy, prompt and model versions are adopted only if OPE is not worse (CI). Rollback is kept.

## 7. Two pipelines and the Thursday run (§17.9, §18, §19)
- `fsdr-train --week0|--weekly`:
  - stages T0 … T8: load, feedback, features, two actions, evidence, ⑩/⑪, level-1 policies, evaluation and ⑥B, overview + ledger A–E + T-checks;
  - result: `models/bundle_<D>/`, status `validated`, read-only afterwards.
- `fsdr-score --bundle <id>`:
  - stages P0 … P10, each one checkpointed;
  - P3 and P5 are batched by segment (64 in parallel, priority order of §19.1a, Target 1 and Target 2 interleaved);
  - it stops before approval.
- `fsdr-score` refuses a bundle that is not validated or not from this week (except `--allow-previous-bundle`, which is logged). It never trains and never writes to the bundle; tests assert both.
- `fsdr-run-week` = train, then score, stopping before approval.
- `fsdr-approve` then `fsdr-push`:
  - the push writes the CSV export plus the campaign log;
  - the SFTP push is a **stub** until you give me the approved flow.

## 8. Phases, with the evidence I show at the end of each

| Phase | Content | Evidence shown | Runs on |
|---|---|---|---|
| 1 | This plan + environment check | this file, smoke test | dev |
| 2 | Contract (46 tables), generator (2k / 100k / 1M Parquet, weeks), ① | validation report, completeness per pipeline | dev |
| 3 ✅ | Features B-01…B-12 (pandas per partition, see §17), S-rules, bands, micro-segments, `action_original` + crosswalk | segment and micro-segment counts, crosswalk, run time at 100k and 1M | dev |
| 4 | Evidence E-A*/E-O*, levels, intervals; ②b agent dictionary → `action_agent` | coverage per level, dictionary diff with reasons | dev (3B) |
| 5 | ⑩ + ⑪ with gates; ③ LLM NBA (scores, ε, propensity, fallback) | helper quality reports, 10 decision traces | dev |
| 6 | ④ LLM NBO, candidates per target, ⑥A, ⑥B C-v1 → reco tables | 10 traces per target | dev |
| 7 | ⑦ code 5 + parity, ⑬ guard, ⑨ auditor + broken-on-purpose tests | first 5 rows of `input_flytxt`, parity diff, audit report | dev |
| 8 | ★ levels 1/2, ⑤, ⑥B proposals, ⑧ | cell policies, 5 full traces, top proposals | dev |
| 9 | Two pipelines, panel, snapshots, feedback, approval, push stub, rollback, scheduler; benchmarks | 3-week simulation (T1 vs T2 vs control with CIs); 1M benchmark; **2M + 8B/14B LLM benchmark on the P16** | dev + P16 |
| 10 | ⑫ ledger, UI tabs, API, CLI, docs | ledger rows of 3 customers over 2 weeks | dev |

After each phase I run `run.py test`, show the evidence, and wait for `continue`. I commit only when you say so.

## 9. Golden cases of §23, checked by hand before coding

| Case | Result |
|---|---|
| VAR_REV −40, LOG_REV −0.3726753, VAR_DATA −100, LOG_DATA −6.7464121, LOG_VOICE 3.7135721 | consistent (recent − older) |
| B-26: (6, 4, 5) → 6.0 / 5.4 / 0.6; weekly (1, 2, 1, 2) → 2.0773503 | consistent |
| F-05: (0, 4, 6) → zeros dropped → min(5 + 1.414, 6) = **6.0** | consistent |
| MAINTAIN passes: COMBO 7.0 + DATA 7.3 (8.0 and 7.4 excluded by the strict upper bound) | consistent |
| C-v1: anchor 6.0, UPSELL DATA → band 6.5–9.0 keeps 7.0 / 8.5; widened to 6.5–12.0 | consistent |
| ε sampling (0.5733333, 0.3033333, 0.1233333) | consistent |
| Priors "0.509 … 0.008, **clipped to 0.01**" | ambiguous: 80/10,000 = 0.008. Default: floor 0.01 on the *propensity*, keep 0.008 as the prior → open item 12 |

## 10. Conflicts inside the spec, and what I propose

| # | Conflict | Proposal |
|---|---|---|
| 1 | "Local LLM only, no cloud" vs the LangSmith tracing added in V1.0 | Keep it, **off by default**. Only a self-hosted `LANGSMITH_ENDPOINT` is allowed when the data is real; the auditor gives a WARN if tracing points to the cloud |
| 2 | "pip-only" vs uv | uv installs the same PyPI wheels; I also export `requirements.txt` so `pip install -r` works |
| 3 | "Step cap 8 per customer" vs a 2–3 h budget for 80k decisions | One scoring call per agent per customer (see §2) |
| 4 | Target 1 must equal production, but the ⑬ guard (`repair`) changes some Target 1 offers | Keep the spec default `repair`, so both targets get the same guard and the comparison is fair. The report shows "production as-is" vs "after guard" for Target 1, so the guard's own effect is visible |
| 5 | C-05 "no contact" customers in Target 2 shrink the contacted 20k | Counted as Target 2 decisions (intention to treat) and measured vs control, as the spec says; A-30 checks the panel size, not the file size |
| 6 | Nationality drives the micro-segments, evidence and price-step guidance | Technically fine, but different prices by nationality can be a **legal / compliance risk**. Default: keep it as specified, add a fairness report (take-up and price paid per nationality); **needs your sign-off** (open item 13) |

## 11. Open items (§25), with my default

| # | Item | Default |
|---|---|---|
| 1 | B-03 CHECK mappings | Last value wins, as in Python: `Retention_High_Probability_To_Reconnect` → RETAIN; `RGS_Not…Voice_Appetency` → UPSELL VOICE, `Rgs_Not…` → CROSS SELL VOICE (case-sensitive) |
| 2 | Control size | 10k, outside the 40k |
| 3 | C-v1 values; Target 1 free part | Band 1.0–1.5×, step +0.5, widen to 2.0×, do-not-disturb after 3 ignored offers; B-23 free part **off** |
| 4 | Margin costs | ASSUMPTION: margin = price (revenue) until finance provides costs |
| 5 | `young` split; tenure unit | Test < 25 / 25–39 in Phase 4, keep it only if it is significant; tenure in days |
| 6 | Panel duration, cool-down, fatigue | 6 weeks, 7 days, 3 ignored |
| 7 | Reward window | 6 days (Fri–Wed) + a lagged 30-day report for monthly offers |
| 8 | LLM model | Dev: Qwen2.5-3B-Instruct GPTQ-Int4. Prod: 8B instruct FP8 (Qwen3-8B-FP8 candidate); 14B AWQ benchmarked in Phase 9 |
| 9 | Base size | 1M designed, 2M tested (on the P16) |
| 10 | Guard thresholds | CAN-05 1.0×, CAN-03 4.3, Target 1 mode `repair` |
| 11 | Push flow | CSV export + stub until you give the approved SFTP flow |
| 12 | Prior clipping wording | Floor 0.01 on propensities only |
| 13 | **Nationality in decisions** | Keep, with a fairness report; your compliance sign-off needed before the first real push |
| 14 | Dev vs prod machine | Phases 2–8 on this laptop (fake LLM + 3B); the 1M/2M + LLM benchmarks of Phase 9 on the ThinkPad P16. Please free ≥ 30 GB on C: before Phase 9 |

## 12. Security (carried over and extended)
- Never committed: `.env`, `data/`, raw reference extracts (the original SQL, Python codes and code 5), the ledger, snapshots.
- Names: FSDR, COMPETITOR_A / COMPETITOR_B; the technical identifiers of the original code are kept (`input_flytxt`, `nbo_LFM_`, …).
- The ledger holds MSISDNs: it lives only in `data/ledger/`, exports hash the MSISDN by default, and it is retained 52 weeks.
- MSISDNs never appear in prompts or traces (salted pseudonym); the API key is required as soon as the API leaves 127.0.0.1.

## 13. Input data: 5 folders

One folder per run date, 5 sub-folders. The 47 tables are filed as follows: the 46 tables V1.0 loads (its 45-table contract + `consent_contact`) plus the new `customer_profile`.

```
data/input/<run_date>/
├── usage/          18 tables: what the line consumes
│     subs_amt_auto, subs_amt_score,
│     activities_{w1,w2,w3,w4,m1,m3}_auto, activities_auto_m2,
│     activities_{w1,w2,w3,w4,m1,m2,m3}_score,
│     paygo_activity_auto, usage_data_voice_payg_tg
├── models/         11 tables: the model scores (one shared structure, proba_1)
│     DAIM_CHURN_GLOBAL…, DAIM_EARLY_CHURN_0_30…, DAIM_EARLY_CHURN_30_60…,
│     DAIM_PROPENSITY_REACTIVATION_30D_404…, PROPENSITY_REACTIVATION_15D_401…,
│     APPETENCY_DATA_204…, APPETENCY_VOICE_204…, CHURN_ONE_DAY_ACTIVE_107…,
│     REV_DECREASE_204…, DAIM_FRAGILITY_DATA…, DAIM_FRAGILITY_VOICE…
├── subscription/    6 tables: the lines and what they bought
│     base_prepaid, consumer_aact_account, access_method_id_auto,
│     subscriptions_tg, subs_btl_auto, sub_btl_3months
├── catalog/         3 tables: the offers
│     btl_catalog, BTL_script_english_arabic_ok, conflicting_offers_h
├── others/          9 tables: customer, location, consent, device, campaigns, sample
│     party_auto, party_scoring, customer_profile, location_auto, consent_contact,
│     DEVICEBEHAVIORSEGMENTATION.dv_segment, targeting_dataiku_auto, targeting_auto,
│     sample_cg_tg_new_<date>
└── _manifest.json   run date, source, row counts, checksums (written by the generator or by ①)
```

The rules:
- The folder of each table is in the contract (`schemas.py`), so ① checks it.
- A table in the wrong folder gives a **WARN** and is still loaded. A missing required table gives a **BLOCK**, as in V1.0.
- The V1.0 flat layout (all files in one folder) is still accepted, with one WARN, so old runs can be replayed.
- The generator writes this layout in every mode (2k / 100k / 1M, JSON or Parquet).
- The upload screen of the journey accepts a zip of the 5 folders.

## 14. Every step saves its output

Each step writes its result to its own folder before the next step starts. A run can therefore be inspected, resumed or replayed from any step.

```
data/runs/<run_date>/
├── train/                                     fsdr-train, full base (≈ 1M lines)
│   ├── T0_load/        validation_report.json, completeness.csv           (cache in data/cache/)
│   ├── T1_feedback/    rewards.parquet, feedback_summary.csv               last week's campaign vs control
│   ├── T2_features/    base_features.parquet                               B-01 … B-12
│   ├── T3_segments/    segments.parquet, crosswalk.csv,                    action_original / action_agent
│   │                   dict_segm_actions_agent_vN.json
│   ├── T4_evidence/    evidence_actions.parquet, evidence_offers.parquet, coverage.csv
│   ├── T5_helpers/     action_bandit.model, offer_pattern.npz, helper_quality.json   ⑩ / ⑪ + gates
│   ├── T6_policies/    cell_policies.json                                  ★ level 1
│   ├── T7_evaluation/  ope_report.json, challenger_proposals.md           ⑥B
│   └── T8_validate/    t_checks.json, overview.html
│                       → bundle copied to models/bundle_<D>/ (status validated)
├── score/                                     fsdr-score, 40k panel + 10k control
│   ├── P0_bundle/      bundle_check.json
│   ├── P1_panel/       panel.csv                                           Target 1 / Target 2 / control
│   ├── P2_context/     context.parquet                                     features + segments of the 50k
│   ├── P3_nba/         batch_0001.parquet … , nba_scores.csv               ③ LLM scores per action
│   ├── P4_actions/     actions.csv                                         ε sampling + propensity
│   ├── P5_nbo/         batch_0001.parquet … , nbo_scores.csv               ④ LLM scores per offer
│   ├── P6_offers/      reco_target1.csv, reco_target2.csv                  ⑥A production / ⑥B C-v1
│   ├── P6b_guard/      guard_offers.csv                                    ⑬ on recommended offers
│   ├── P7_code5/       input_flytxt_draft.csv, parity_diff.csv             ⑦ + parity A-25
│   ├── P7b_guard/      guard_slots.csv                                     ⑬ on the 6 final slots
│   ├── P8_audit/       audit_report.json, audit_report.html                ⑨ PASS / WARN / BLOCK
│   ├── P9_ledger/      ledger_rows.parquet                                 ⑫
│   └── P10_report/     run_report.html
└── final/
    ├── input_flytxt_<D>.csv        the file to push: exactly the 38 code-5 columns
    │                               (written only on audit PASS; otherwise BLOCKED.txt with the reasons)
    ├── control_<D>.csv             the 10k control lines (not contacted, kept for measurement)
    ├── summary.json                counts per target, LLM share vs fallback, run times, versions
    └── approval.json               written by fsdr-approve (who, when); fsdr-push reads it
```

In every step folder:
- **`_step.json`**: status, start, duration, rows in / out, input checksums and output files.
  - A step whose `_step.json` is `done` and whose inputs did not change is skipped on a re-run. This is the resume.
  - `--from P5` forces a re-run from that step.
- **Format:** scoring outputs (≤ 50k rows) are written as **CSV and Parquet**, so you can open them in Excel. Training outputs (1M rows) are Parquet only, plus a `preview.csv` with the first 1,000 rows.
- The journey screen and the API list these folders, and a click opens any step's file.
- `data/runs/` stays git-ignored (MSISDNs). Retention: 8 weeks of `runs/`, 2 weeks of raw input once the Parquet checksum is logged (settings).

## 15. Batches

**Training (1M lines), batched by partition:**
- At T0 every per-line table is split into **16 partitions** by a hash of the line id. All tables use the same hash, so the joins stay inside a partition.
- **T2 features, T3 segments:** run partition by partition, 8 partitions in parallel on the P16 (2 on the dev laptop). Memory stays bounded and a failed partition is re-run alone.
- **T4 evidence:**
  - each partition writes its counts (contacts, takers, revenue per cell);
  - the counts are summed, then the Wilson and Newcombe intervals are computed once;
  - bootstrap intervals use per-partition resampling.
- **T5 helpers:**
  - ⑩ VW reads the logged rows as one stream.
  - ⑪ ALS uses one sparse matrix of 1M lines × about 70 offers, which is small.
- **T6 / T7 LLM calls:** about 300 cell policies plus about 70 ②b cells. They go to vLLM in parallel batches of 32.

**Scoring (40k + 10k), batched by 1,000 customers:**
- **P0–P2:** run in one pass (50k rows, seconds).
- **P3 → P4 → P5:** run in **40 batches of 1,000 customers**.
  - Order: the priority of §19.1a, with Target 1 and Target 2 interleaved, so if the run is stopped, both targets have the same coverage.
  - Each batch sends **64 requests in parallel** to vLLM and writes its file (`P3_nba/batch_NNNN.parquet`) before the next batch starts.
  - A crash loses at most one batch.
- **Time box:**
  - if the LLM part goes over its budget (default 2 h), the remaining customers get the evidence fallback of §5;
  - they are flagged `decided_by = fallback_timebox`;
  - the auditor reports their share (WARN above 5 %).
- **P6 → P10:** need the whole file (conflicts, guard on 6 slots, audit), so they run once on the 40k.

**The LLM prompt has to be compact** for the 40k to fit the budget:
- **Shared prefix (≈ 900 tokens), cached by vLLM once per cell:** the rules, the allowed ids, the cell policy and the JSON schema.
- **Per customer (≈ 300 tokens):** a fixed-order feature line plus the evidence numbers.
- **Reply (≈ 40 tokens):** scores plus a reason code from a fixed list.
- The full written explanation is generated only for a 5 % sample (ledger, audit) and on the journey screen.

## 16. Run times on the ThinkPad P16 (estimates; Phase 9 measures them)

Assumptions:
- Production LLM: **Qwen3-8B FP8, thinking off**, vLLM in WSL2, 64 parallel requests, prefix caching on, FP8 KV cache.
- The pipelines run in **Windows-native Python** (uv), reading and writing on NTFS.
  - Only vLLM runs in WSL2, called over `127.0.0.1`.
  - This avoids slow `/mnt/c` file access from WSL.
- WSL limits: `.wslconfig` with memory = 64 GB and processors = 16, so Windows keeps room for polars.
- Input: about 12 GB of JSON at 1M lines (25 MB per 2,000 lines in the V1.0 dummy data, about 12.5 KB per line).
  - About 2–3 GB as Parquet.
  - Real activity tables may have more rows per line, which would mean up to about 40 GB of JSON.

**Training: `fsdr-train`, 1M lines**

| Step | What dominates | P16 estimate |
|---|---|---|
| T0 load: JSON → Parquet, plus ① checks | JSON parsing (CPU), 24 threads | 5–15 min |
| T1 feedback | joins on the campaign log | < 1 min |
| T2 features, T3 segments | polars group-bys, 16 partitions | 2–4 min |
| T4 evidence + intervals | aggregations + bootstrap | 1–3 min |
| T5 ⑩ VW + OPE gate, ⑪ ALS + hold-out gate | VW is single-threaded | 3–7 min |
| T6 / T7 LLM: level-1 policies, ②b, ⑥B | about 400 long LLM calls | 5–10 min |
| T8 T-checks + bundle | checksums | 1–2 min |
| **Total** | | **≈ 20–40 min** (≈ 10–25 min if the input already comes as Parquet) |

**Scoring: `fsdr-score`, 40k panel + 10k control**

| Step | What dominates | P16 estimate |
|---|---|---|
| P0–P2 bundle check, panel, context | reads | 1–2 min |
| P3 ③ NBA, 40k LLM calls | GPU | 20–35 min |
| P4 sampling | vectorised | seconds |
| P5 ④ NBO, 40k LLM calls (longer prompt: the candidates) | GPU | 25–40 min |
| P6 / P6b offers + guard | vectorised | 1 min |
| P7 code 5 + literal replay for parity | the replay runs row by row | 3–6 min |
| P7b – P10 guard, audit, ledger, report | | 2–4 min |
| **Total** | | **≈ 55 min – 1 h 30** |

**Whole Thursday run (train, then score, until the approval stop): ≈ 1 h 15 – 2 h 10, inside the 2–3 h budget.**

How the choice of model changes scoring:

| LLM | Scoring time (40k) | Note |
|---|---|---|
| Qwen2.5-3B (dev model) | ≈ 30–45 min | weaker reasoning |
| **Qwen3-8B FP8 (proposed)** | **≈ 55 min – 1 h 30** | good fit for 24 GB, with room for a 64-request batch |
| 14B AWQ int4 | ≈ 1 h 45 – 3 h | at the budget limit; the time box applies |
| Verbose prompts, no prefix cache (1,500 tokens in, 150 out), 8B | ≈ 3 – 3 h 30 | **over budget**; this is why §15 requires the compact prompt |

For comparison, this dev laptop:
- **Training:** 1M lines on Parquet in about 1–2 h (2 partitions in parallel, 16 GB RAM).
- **Scoring:** 80k real LLM calls are not possible: about 1 call per second on the GTX 1650, so about 22 h.
- **What runs here:** the fake LLM, or a sample of 500 customers.

Disk on the P16: about 12–40 GB of raw input plus 3 GB of Parquet plus about 1 GB of outputs per week. With the retention of §14, that is ≤ 100 GB, which fits in the 1 TB.

## 17. Phase 3 results (2026-11-02, dev laptop)

**Built:**
- `nbx/fsdr/base_features.py` (T2, B-01 … B-12);
- `nbx/fsdr/segmentation/` (`rules.py` S-rules, `bands.py` bands and micro-segments, `actions.py` action_original, crosswalk and agent dictionary);
- `nbx/fsdr/pipelines/training.py` (T2, T3, partition by partition, process pool);
- `fsdr-features`;
- notebook section 7;
- 39 tests in `tests/fsdr/test_v2_segments.py`.

**Deviations from §3 and §6:**

| Topic | Plan | Built | Why |
|---|---|---|---|
| Module name | `features/` package | `base_features.py` | `features.py` (V1.0) already exists and is still used |
| Data engine | polars | pandas per cache partition | 1M lines in under a minute, so polars is not needed (no new dependency); memory stays bounded at ≈ 62k lines per partition |
| B-04 churn | max of the 3 tables | **first row** of the stack (global, 0-30, 30-60), as the spec says | |
| B-05 missing score | V1.0 `uniform(0, 0.02)` | flag + median of the nationality group, on the whole base (two passes) | A missing score never triggers "at risk": the S-rules read a score only where its flag is 0 |
| H-06 in the agent dictionary | | an at-risk segment (Nursering / Onboarding / Retention_RGS `_At_Risk`) may not be moved **to** an UPSELL-family action; its original action stays allowed (Onboarding at risk = UPSELL in code 1) | **CHECK** |

**Run times (dev laptop, 2 partitions in parallel, Parquet input, synthetic world, week 0):**

| Lines | Generate | T0 load + cache | T2 features | T3 segments | Input / cache / T2 / T3 on disk |
|---|---|---|---|---|---|
| 2k | 2 s | 3 s | 3 s | 2 s | — |
| 100k | 25 s | 41 s | **9 s** | **3 s** | 133 MB / 153 MB / 41 MB / 5 MB |
| 1M | 303 s | 341 s | **44 s** | **14 s** | 1.2 GB / 1.4 GB / 360 MB / 41 MB |

T0 now dominates training-data preparation; on the P16 with 8 workers, T2 + T3 should take well under a minute.

**Segments at 1M:**
- 980,230 eligible lines out of 1,000,000 (excluded plans removed).
- 14/14 segments filled:

  | Segment | Lines |
  |---|---|
  | Retention_RGS_Fragile | 292k |
  | RGS_Not_At_Risk_DualPlay_Upgrade | 247k |
  | Retention_RGS_At_Risk | 95k |
  | Voice_Only_No_Data | 84k |
  | Voice_Only_Data | 65k |
  | Nursering (excluded) | 58.5k |
  | Data_Only_No_Voice | 47k |
  | Onboarding | 48k |
  | Data_Only_Voice | 28k |
  | Retention_* (not RGS) | 14.5k |

- **75 of 120 micro-segments are filled; 53 have ≥ 300 lines.** The 45 empty cells are mostly the `unknown` age and tenure bands, which the generator never produces.
- `action_original` comes from the segment engine for 533,920 lines and from the crosswalk for 446,310.

**Findings to check:**
1. **The segment engine and the S-rules agree on 50 % of the targeted lines.**
   - In the synthetic world, the engine segment is built from hidden scores with other thresholds (churn ≥ 0.6, fragility ≥ 0.55, appetency ≥ 0.4).
   - On real data this number is the first thing to read: it measures how far the S-rules are from production.
   - `calibrate_thresholds` (proposal only) comes with the evidence in Phase 4.
2. **Retention_RGS_Fragile is 30 % of the base.**
   - S-9 fires on fragility ≥ 0.5 **or** a revenue drop ≤ −30 % M1 vs M2. With PROBA_REV_DECR in the max, the rule is wide.
   - The threshold is a ⚙ value to confirm on real data.
3. **Recomputed week-0 priors differ from B-10** (MAINTAIN 0.38 vs 0.10): this is the synthetic targeting. `priors.json` keeps both; `used` = recomputed in week 0, as the spec says.
4. **Missing model scores are 14–29 % per score** (generator coverage 70–90 %). They are flagged and imputed; the share is printed by `fsdr-features`.

## 18. V3.0 (2026-11-08): the `.csv.gz` contract and 100k contacted in four groups

V3.0 = V2.0 (phases 1–3) + two spec changes, both in `docs/fsdr_reference/`:
- the updated master prompt: §2 / §20 *FSDR CSV standard*, A-28, the file-format tests;
- `PROMPT_FSDR_update_100k_targets.md`, which replaces the 2-target design (20k / 20k) everywhere.

### 18.1 The four groups

| Group | Size | Universe | Rules | Agent | Rule action (NBA hint / fallback) | Code-2 table | Reco tables |
|---|---|---|---|---|---|---|---|
| T1C | 25k | constraint (code 3) | production code 3, B-20 … B-32 | ⑥A-C | `action_original` | `TG_40K_constraint_scored_nba` | `recos_*_40k_constraint` (with action) |
| T1F | 25k | free (code 4) | production code 4, FR-01 … FR-14 | ⑥A-F | `action_original` | `TG_40k_free_scored_nba` | `recos_*_tg_free` (no action) |
| T2C | 25k | constraint | challenger CC-v1 | ⑥B-C | `action_agent` | `TG_challenger_constraint_scored_nba` | `recos_*_challenger_constraint` |
| T2F | 25k | free | challenger CF-v1 | ⑥B-F | `action_agent` | `TG_challenger_free_scored_nba` | `recos_*_challenger_free` |
| control | 20k | — | never contacted | — | — | — | — |

`FSDR_T2_SPLIT=false` gives one 50k T2 with the CC rules. The `TG_40k_challenger_*` / `recos_*_challenger` tables of the earlier update prompts are not written (`config.SUPERSEDED_TABLES`).

### 18.2 What changed, file by file

| Area | V3.0 |
|---|---|
| `nbx/fsdr/csvstd.py` (new) | FSDR CSV standard writer / reader: gzip 6 with mtime 0 (the same table gives the same bytes), UTF-8 without BOM, minimal quoting, `\n`, never scientific notation (`7.0`, `12.5`, `0.0001`), empty field for missing, IDs as text; `<name>.schema.json`; `_outputs_manifest.json` (rows, SHA-256, run id, versions). Vectorised per dtype: 100k lines in 78 s, 93 MB |
| loader, generator, validation | `.csv.gz` read in every folder (the `.schema.json` sidecars are skipped); `fsdr-generate` writes `.csv.gz` by default. A non-`.csv.gz` table **blocks T0** unless the data comes from the generator (manifest `source: generator`) or `FSDR_ALLOW_TEST_FORMATS=true` |
| `config.py` | the 5 sizes, `T2_SPLIT`, `GROUPS` (universe, rules, tables, rule action, NBO sees action, 10.1 loop, rules version), sample mapping, CC / CF values, LLM budget 10 h, level-2 output ≤ 120 tokens, `GUARD_MODE_TARGET1` |
| `panel.py` (new) | P1: eligible pool (not Nursering, not in another campaign this week) → stratified random 5-way assignment; sample mapping cstr → T1C, free → T1F, cg → control, topped up; balance report (A-31) and size check (A-30) |
| `offers/` (new) | `universe.py` (B-20b, `constraint_candidates`, `free_candidates`), `anchors.py` (B-26 / B-27, CC-03 split, micro-segment median for no spend), `selection.py` (`select_code3`, `select_code4`, `select_challenger`, all vectorised), `replay.py` (literal code 3 / code 4 selection, parity only), `recos.py` (the 13 tables) |
| `rules/*.yaml` (new) | `production_rules_constraint.yaml`, `production_rules_free.yaml`, `challenger_rules_constraint_v1.yaml`, `challenger_rules_free_v1.yaml` (id, description, source, stage, hard / soft, parameters, check function); CC-v1 / CF-v1 `approved: false` until open item 4 |
| `audit_groups.py` (new) | A-12 per group (FR-02 / FR-03, B-21 / CC-01, pass-1 band, C-02 band), A-12b, A-17, A-21, A-30, A-31; any FAIL blocks the push |
| `llm_budget.py` (new) | level-2 queue: priority tier first inside each group, the 4 groups round-robin, so a budget cut leaves them the same coverage; past the budget → cell policy (`cell_policy_timebox`) |
| `report_groups.py` (new) | each group vs control and the 2 × 2 effects (rules effect, universe effect, interaction) with 95 % CIs |
| `pipelines/scoring_groups.py` (new) | `fsdr-panel` (P1_panel), `fsdr-offers` (P6_offers): the 13 tables as `.csv.gz`, traces, group report, group audit |
| notebook | `.csv.gz` previews; section 8 "the panel and the four groups" |
| docs | `fsdr_known_differences.md` (new), terminal guide, README |

### 18.3 Decisions I made (tell me if you disagree)

1. **Panel method.** The pool is sorted by the `Final` stratum (nationality × value band × usage type × plan), then age band, tenure band, segment and a random key.
   - The 120k are a systematic sample of that order.
   - Labels are assigned in blocks of 24 (= 5 : 5 : 5 : 5 : 4), each block randomly permuted. So every stratum is split in the target proportions, up to one block.
   - A pool under 120k scales every group by the same factor (reported).
2. **A-31.** FAIL when a stratum share differs by more than 2 points between groups **and** the chi-square p < 0.01. A small panel shows random gaps above 2 points that are not imbalance.
3. **Challenger band (CC-02 / CF-02).**
   - Lower bound = max(1.0 × anchor, anchor + 0.5 for UPSELL / UPSELL DATA / UPSELL VOICE), inclusive; upper bound = 1.5 × anchor, inclusive.
   - MAINTAIN, RETAIN and the cross-sell actions may sit at the anchor.
   - T2F still uses the action for the +0.5 step (CF-02 "as CC-02"); the offers themselves do not depend on it.
4. **CC-04 / CF-04.** Best per type by score × price; only the missing types are searched in the widened band. The T1 passes are used only when the customer has nothing at all (logged `fallback B-28/29/30` or `fallback FR-09/10/11`).
5. **10.1 consistency** (T1C / T2C only). A customer with no offer for his action is sent back once. The second action is `action_agent` rank 2, else the other dictionary's action, else the next B-10 prior.
6. **Stable sorts.** Code 3 / code 4 used quicksort, so equal score and price had no defined order. V3.0 keeps the candidate order. The parity tests use continuous scores, where both agree (known differences #11).
7. **Until Phases 5–6, P6 uses the LLM agents' fallbacks**, marked on every row:
   - action = the group's rule action (`decided_by = fallback_rule_action`, propensity 1.0);
   - offer score = own past purchase, then buyer share (`score_source = fallback:E-O3+E-O1`).

### 18.4 LLM load at 100k (P16; Phase 9 measures it)

| | 40k design (§16) | 100k design |
|---|---|---|
| Level-2 calls (NBA + NBO) | 80k | **200k** |
| 8B FP8, compact prompts (§15: ≈900-token cached prefix, ≈300 per customer, ≤ 120 out) | 55 min – 1 h 30 | **≈ 2 h 20 – 3 h 45** |
| 8B FP8, verbose prompts without prefix cache (the update prompt's 5–7 h) | 3 – 3 h 30 | 7 – 9 h, at the budget limit |
| Weekly budget | — | 10 h; beyond it, the cell policy is applied by code, equally for the 4 groups |

For T1F / T2F the NBO call does not depend on the action, so it runs in the same wave as the NBA call.

**Thursday:** training ≈ 1 – 1.5 h, then scoring ≈ 3 – 5 h with compact prompts. The scheduler starts early in the morning; approval is in the afternoon. The 14B model does not fit at 100k.

### 18.5 Results at 1M (dev laptop, `.csv.gz` input, 2 workers)

| Step | Time | Result |
|---|---|---|
| Generate 1M lines, `.csv.gz` | 948 s | 961 MB (Parquet was 1.2 GB) |
| T0 load + validate + cache | 831 s | 0 blocking issues (V2 on Parquet: 341 s; CSV parsing costs the difference) |
| T2 + T3 | 64 s | as V2 |
| P1 panel | 26 s | pool 744,487 eligible (192,335 in other campaigns left out); **25,000 × 4 + 20,000**, 70,000 from the provided sample; largest stratum gap **1.21 points** (A-31 limit 2), min chi-square p 0.39 |
| P6 offers | 65 s | 13 tables, 4.1 MB `.csv.gz`; group audit **PASS** (A-12, A-12b, A-17, A-21, A-30, A-31) |

| Group | Contacted | With offers | Offers | 10.1 retries | Main rules |
|---|---|---|---|---|---|
| T1C | 25,000 | 21,073 | 76,775 | 3,951 | B-28 / B-29 / B-30 |
| T1F | 25,000 | 20,134 | 73,608 | — | FR-09 / FR-10 / FR-11 |
| T2C | 25,000 | 21,990 | 70,036 | 3,171 | CC-04, often `fallback B-28/30` |
| T2F | 25,000 | 21,462 | 68,699 | — | CF-04, often `fallback FR-09/11` |

**Finding:**
- The dummy catalogue has only about 6 recommendable L2 offers per validity and family, with big price steps (e.g. monthly DATA 8 → 14.4 GB).
- So the 1.0 – 1.5 × band (2.0 × widened) often holds no offer, and T2 falls back to the T1 passes. Pass 3 (no cap) is frequent in T1 for the same reason.
- On the real catalogue (≈ 70 offers) the band will catch more. Watch the `fallback` share in `group_report.csv`: it is a direct measure of how often CC-v1 / CF-v1 actually apply.

### 18.6 Open items of the update (my defaults)

| # | Item | Default |
|---|---|---|
| 1 | Control size | 20k |
| 2 | LLM weekly budget and model | 10 h, 8B FP8 |
| 3 | `guard_mode_target1` | `repair` |
| 4 | CC-v1 / CF-v1 values | band 1.0 – 1.5 ×, +0.5 OMR for upsells, widened to 2.0 ×, do-not-disturb after 3 ignored offers; **not approved yet** (`approved: false` in the YAML) |

### 18.7 Still to build (phases 4–10, now for four groups)

- Evidence and the ②b LLM dictionary.
- ⑩ / ⑪ helpers and the LLM NBA / NBO agents, which replace the P6 fallbacks.
- Code 5 with the F-01 table list of the 4 groups: one `input_flytxt.csv.gz` plus `input_flytxt_t1c/_t1f/_t2c/_t2f`.
- The ⑬ guard per group; the rest of the auditor (A-25 parity of `select_code3` / `select_code4`, already tested here).
- The weekly report with the 2 × 2 effects on real feedback.
- The ledger `panel_group` / `rules_version` values; `fsdr-score --group`; the UI group filter.

### 18.8 Target upload folders and the final files (2026-11-08)

- **Target lists.** `data/input/<D>/targets/{target1/t1_constraint, target1/t1_free, target2/t2_constraint, target2/t2_free, control/control}.csv.gz`.
  - When present, P1 uses them instead of drawing (`targets.py`).
  - Checks: a line in two groups → block; bad MSISDN, not in base, Nursering or another campaign → dropped, with the reason; size → warning.
  - `fsdr-targets --generate` writes dummy lists; the API takes them at `POST /fsdr/targets/<D>/<group>`.
- **⑦ code 5** (`final_output.py`, F-01 … F-19, vectorised, issues a–c fixed).
  - The population is the 4 groups (update §5).
  - Golden cases tested: F-05 (0, 4, 6) → 6.0, forced bundles 2.5 / 3.5 / 5.2, a popular offer held → no forcing, F-09, F-12 with the VOICE row ignored, F-04, the 38 columns.
  - The bug-reproduction replay for A-25 is still to come (Phase 7).
- **Final step** (`fsdr-final`, part of `fsdr-run-groups`), `data/runs/<D>/final/`:
  - `push/input_flytxt.csv.gz`, written only when A-20 … A-24 and the group audit pass; otherwise `push/BLOCKED.txt`, and an older push file is removed;
  - per target: `input_flytxt_<group>`, `input_flytxt_target1/2`, `details_<group>` (≈ 220 columns per line);
  - `details_all.parquet`, `final_rules_report.csv`, `_outputs_manifest.json`.
- **Dummy data limit.** The dummy catalogue has no `CM_*` / `DM_*` offers, so F-08 never forces a bundle there; the unit tests cover it. Customers whose anchor is above every offer price get nothing (BR-82) and are dropped by F-04, which is visible in the details.

## Appendix: vLLM smoke test (2026-11-01, dev laptop)
- Server: WSL2 vLLM, Qwen2.5-3B-Instruct-GPTQ-Int4 served as `qwen2.5:3b` on 127.0.0.1:8001, API key from `.env`.
- Prompt: 3 allowed actions, churn 0.62, revenue −35 %, JSON mode, temperature 0, seed 7.
- Reply (valid JSON): `{"scores": {"RETAIN": 0.4, "UPSELL": 0.3, "MAINTAIN": 0.3}, "top": "RETAIN"}`. That is the right direction (retention before upsell when at risk).
- 64 tokens in, 49 out, **10.8 s** (first call after start, includes warm-up). One customer per call on this GPU cannot cover 80k decisions a week, which confirms §2 of this plan: production LLM load belongs on the P16.
