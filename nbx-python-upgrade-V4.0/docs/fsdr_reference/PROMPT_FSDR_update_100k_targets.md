# FSDR NBX – UPDATE PROMPT: 100k contacted – Target 1 (50k = 25k constraint + 25k free) and Target 2 (50k challenger = 25k constraint + 25k free)

> A change request on top of `PROMPT_FSDR_MASTER.md`. **It replaces the two earlier update prompts** (`PROMPT_FSDR_update_target1_free.md` and `PROMPT_FSDR_update_target2_split.md`): do **not** apply those. If one of them was already applied, this prompt brings the project to the final state below.

## 1. The change in one sentence
The campaign contacts **100,000** customers in **four groups of 25k**: **Target 1 (50k)** reproduces the production design (25k **constraint** like code 3 + 25k **free** like code 4) with the LLM agents in place of VW / LightFM; **Target 2 (50k)** applies the **challenger rules (B)** to the same two offer universes (25k constraint + 25k free). A control group is never contacted.

## 2. The groups
| Group | Size ⚙ | Offer universe | ARPU and selection rules | Agent | Rule action for the LLM NBA |
|---|---|---|---|---|---|
| **T1C** – Target 1 constraint | 25,000 | **Constraint** (code 3: families and generosity by action) | **Production code 3** (B-20 … B-32) | **⑥A-C** | `action_original` |
| **T1F** – Target 1 free | 25,000 | **Free** (code 4: L2, DATA / COMBO, no VOICE, action not used) | **Production code 4** (FR-01 … FR-14) | **⑥A-F** | `action_original` |
| **T2C** – Target 2 challenger constraint | 25,000 | **Constraint** (code 3 universe) | **Challenger** CC-v1 | **⑥B-C** | `action_agent` |
| **T2F** – Target 2 challenger free | 25,000 | **Free** (code 4 universe) | **Challenger** CF-v1 | **⑥B-F** | `action_agent` |
| **Control** | ⚙20,000 (open item) | none – never contacted | none | — | — |

A **2 × 2 design**: rules (production vs challenger) × universe (constraint vs free), plus control. Config: `t1_constraint_size`, `t1_free_size`, `t2_constraint_size`, `t2_free_size`, `control_size`, `t2_split` (default true; false → one 50k T2 with the CC rules).

## 3. Panel and assignment
- Eligible base (master D-07, B-02, H-05, cool-down) → **stratified random** assignment to the 5 groups on the `Final` strata, balanced on age band, tenure band and segment. Never by score.
- If `sample_cg_tg_new_7oct2026` is provided: `cstr` → T1C, `free` → T1F, `cg` → control (topped up to `control_size` from the same strata); T2C and T2F are drawn from the remaining eligible base with the same strata proportions. Mapping in config.
- Same customers in the same group for ⚙4–6 weeks; replacements from the same stratum **and the same group**.
- `panel_group` values: `t1_constraint`, `t1_free`, `t2_constraint`, `t2_free`, `control`, `not_selected`.

## 4. Rules per group
**4.1 T1C – production constraint (⑥A-C), exactly code 3 `reco_btl3`, LLM score instead of LightFM score**
- Universe: recommendable offers (B-20b) filtered by the action's **service and generosity** (`map_action_offers`, active version = L2 for all): MAINTAIN, UPSELL, RETAIN → [COMBO, DATA]; UPSELL VOICE, CROSS SELL VOICE → [DATA, COMBO]; UPSELL DATA, CROSS SELL DATA → [VOICE, COMBO]. The CROSS SELL VOICE bug (it repeats `CROSS SELL DATA`, so no training filter) is **fixed**.
- Anchors per action (B-27): MAINTAIN / UPSELL / RETAIN → COMBO = total, DATA = data (0.9 ×); voice actions → COMBO = total, DATA = data; data actions → COMBO = total, VOICE = voice (0.1 ×). Anchor = mean + std (ddof 1) of (Total_Sub_Amt + PG_Total_Amount) over M1–M3 / W1–W4.
- Pass 1: anchor < price < anchor + **2** (MAINTAIN, RETAIN) or + **3** (UPSELL and the 4 voice / data actions); score desc, price desc. Pass 2: the missing type, price > anchor; score desc, price asc. Pass 3: nothing yet → both types, price > anchor; score desc, price asc. 1 offer per type per validity.
- Outputs: `TG_40K_constraint_scored_nba`, `recos_monthly_40k_constraint`, `recos_weekly_40k_constraint` (with the action column).
- NBO sees the action; 10.1 consistency loop used.

**4.2 T1F – production free (⑥A-F), exactly code 4 `reco_btl2`, LLM score instead of LightFM score**
| Id | Rule |
|---|---|
| FR-01 | 270-day purchases, catalogue offers only |
| FR-02 | Generosity **L2** only |
| FR-03 | **No VOICE** offers |
| FR-04 | Per validity (30, 7) |
| FR-05 | Recommendable = FR-01 … FR-04 and bought by someone in the base in 270 days (B-20b) |
| FR-06 | **Action not used** for offers |
| FR-07 | Anchor = mean + std (ddof 1) of (Total_Sub_Amt + PG_Total_Amount) M1–M3 / W1–W4; data = 0.9 × |
| FR-08 | COMBO → total anchor; DATA → data anchor |
| FR-09 | Pass 1: anchor < price < anchor **+ 3** (DATA and COMBO); score desc, price desc |
| FR-10 | Pass 2: missing type, price > anchor; score desc, price asc |
| FR-11 | Pass 3: no offer → both types, price > anchor; score desc, price asc |
| FR-12 | 1 offer per type per validity |
| FR-13 | Outputs `recos_monthly_tg_free`, `recos_weekly_tg_free` (no action column) |
| FR-14 | Code 2: `tg_flag = "free"` → `TG_40k_free_scored_nba` |
- `t1f_nbo_sees_action` default **false**; 10.1 loop not used.

**4.3 T2C – challenger constraint (⑥B-C), CC-v1**
| Id | Rule |
|---|---|
| CC-01 | Universe = the T1C universe (bug fixed) |
| CC-02 | Relative ARPU band: price between ⚙1.0 × and ⚙1.5 × the anchor of the offer type (never below; ⚙+0.5 OMR minimum for UPSELL actions) |
| CC-03 | Anchor split data / voice from the customer's own spend (no history → 90 / 10) |
| CC-04 | Per validity, best offer per type by **LLM score × price**; widen upwards once to 1.0 – 2.0 ×; still nothing → the T1C passes as fallback (logged) |
| CC-05 | No contact after ⚙3 ignored offers or fatigue flag (logged as a "no contact" decision) |
| CC-06 | NBO sees the action; 10.1 loop used |
- Outputs: `TG_challenger_constraint_scored_nba`, `recos_monthly_challenger_constraint`, `recos_weekly_challenger_constraint`.

**4.4 T2F – challenger free (⑥B-F), CF-v1**
| Id | Rule |
|---|---|
| CF-01 | Universe = the T1F universe (FR-01 … FR-06) |
| CF-02 | Relative ARPU band as CC-02, anchor per type as code 4 (COMBO total, DATA data) |
| CF-03 | Data anchor share from the customer's own spend (no history → 0.9) |
| CF-04 | Per validity, best offer per type (DATA, COMBO) by LLM score × price; widen upwards once; still nothing → the T1F passes as fallback |
| CF-05 | As CC-05 |
| CF-06 | `t2f_nbo_sees_action` default false; 10.1 loop not used |
- Outputs: `TG_challenger_free_scored_nba`, `recos_monthly_challenger_free`, `recos_weekly_challenger_free`.

**4.5 Common to all four groups:** up to 4 recommended offers (1 per type per validity, B-32 columns); the LLM NBA agent (⑩ helper, ε 10 %, logged propensity) for everyone; the LLM NBO agent (⑪ helper, E-O evidence) on each group's universe; the ⑬ revenue guard on all four groups (`guard_mode_target1` applies to T1C and T1F); CC-v1 and CF-v1 approved before the first push, then ⑥B proposes new versions per group, evaluated by OPE + simulation + ⑬, approved before use.

## 5. Tables, code 2 and code 5
| Group | `tg_flag` | Code-2 table | Monthly recos | Weekly recos |
|---|---|---|---|---|
| T1C | `constraint` | `TG_40K_constraint_scored_nba` | `recos_monthly_40k_constraint` | `recos_weekly_40k_constraint` |
| T1F | `free` | `TG_40k_free_scored_nba` | `recos_monthly_tg_free` | `recos_weekly_tg_free` |
| T2C | `challenger_constraint` | `TG_challenger_constraint_scored_nba` | `recos_monthly_challenger_constraint` | `recos_weekly_challenger_constraint` |
| T2F | `challenger_free` | `TG_challenger_free_scored_nba` | `recos_monthly_challenger_free` | `recos_weekly_challenger_free` |

- ⑦ Code 5, F-01: population = T1C + T1F + T2C + T2F; monthly recos = the 4 monthly tables; weekly = the 4 weekly tables. One `input_flytxt.csv.gz` (≈ 100k rows, 38 columns); the group is **not** in the file. Audit copies `input_flytxt_t1c`, `_t1f`, `_t2c`, `_t2f`.
- Tables of earlier updates (`TG_40k_challenger_scored_nba`, `recos_*_challenger`, …) are removed or kept as views.

## 6. Scale and LLM budget at 100k
- Level-2 LLM calls ≈ **200k per week**; 8B FP8 on vLLM ≈ **5–7 h**; the 14B would not fit.
- Level-2 output ≤ ⚙120 tokens; input ≤ 1,500 tokens with the shared prefix first. For T1F / T2F the NBO calls do not depend on the action, so they run in the same wave as the NBA calls. **Weekly LLM budget ⚙10 h**, priority order of master 19.1a, the 4 groups interleaved; beyond the budget the cell policy is applied by code (logged per group).
- Thursday run ≈ training 1–1.5 h + scoring 6–8 h: start early Thursday morning, approval in the afternoon / evening.
- Panel 100k + control 20k = 120k of a ~1M base: check each stratum has enough customers; if not, scale down the smallest strata.

## 7. Changes by area (summary)
config sizes and switches; 5-way panel; agents ⑥A-C, ⑥A-F, ⑥B-C, ⑥B-F; candidate builders `constraint_candidates(customer, action, validity)` and `free_candidates(customer, validity)`; selection `select_code3`, `select_code4`, `select_challenger`; registries `production_rules_constraint.yaml`, `production_rules_free.yaml`, `challenger_rules_constraint_v1.yaml`, `challenger_rules_free_v1.yaml`; auditor A-12 / A-17 / A-25 / A-30 / A-31 per group; guard repairs per group; ledger `panel_group` and `rules_version` values; weekly report per group plus the 2 × 2 effects (rules effect, universe effect, interaction) with CIs; group filter in UI / API / CLI (`fsdr-score --group t1c|t1f|t2c|t2f|all`); docs.

## 8. Tests (fake LLM, offline)
1. Assignment 100k + 20k → 25k × 4 / 20k (±2 %), stratified, no overlap; sample mapping cstr → T1C, free → T1F, cg → control.
2. T1C universe: UPSELL DATA → VOICE + COMBO L2; MAINTAIN → COMBO + DATA L2; CROSS SELL VOICE → DATA + COMBO (bug fixed).
3. T1C passes (golden): MAINTAIN, anchors total 6.0 / data 5.4, scores COMBO 7.0 (0.9), COMBO 8.0 (0.95), DATA 5.5 (0.7), DATA 7.4 (0.8), DATA 7.3 (0.8) → **COMBO 7.0 + DATA 7.3** (k = 2).
4. T1F universe: same candidates whatever the action; never VOICE, never L1, never an offer not bought in 270 days or not in the catalogue.
5. T1F passes (golden): same anchors and scores → **COMBO 8.0 + DATA 7.4** (k = 3); pass-2 tie → lower price; price = anchor or anchor + 3 not selected in pass 1.
6. T2C / T2F band: anchor 6.0, UPSELL, offers 5.0 / 6.4 / 7.0 / 8.5 / 10.0 → band 6.5 – 9.0 keeps 7.0 and 8.5; selection by score × price; widen once (6.5 – 12.0); then the matching T1 passes as fallback.
7. Anchor split: 70 % data spend → data anchor 0.7 × total.
8. Parity: with fixed scores, `select_code3` = replay of `reco_btl3` selection; `select_code4` = replay of `reco_btl2` selection.
9. Outputs: the 4 code-2 tables and 8 reco tables with the right columns (action column in constraint tables only); code 5 reads all 4 groups; `input_flytxt` 38 columns, one row per MSISDN, no group column.
10. Revenue guard: a violation in each group is repaired and reported per group.
11. Auditor: VOICE in T1F or T2F, a T1C offer outside its family, a T1C pass-1 offer at anchor + 2.5 for MAINTAIN, group sizes 30k / 20k, a T2F customer with a CC version → FAIL, push blocked.
12. Budget: with a fake LLM latency, coverage beyond the budget falls back to the cell policy, interleaved across the 4 groups (equal coverage ± 2 %).
13. Weekly report: the 2 × 2 effects with CIs on a 3-week simulation (2,000 lines, 5 groups).
14. Switch `t2_split = false` → one 50k T2 with CC rules.
15. All existing tests pass.

## 9. Do not change
Data, features, segmentation, micro-segments, ②b two actions, the LLM NBA agent, the ⑩ / ⑪ helpers, the evidence tables, the revenue guard rules, code 5 (except the F-01 table list), the 38-column `input_flytxt`, the training / scoring split, the ledger structure (new values only), approval and push.

## 10. Supersedes
Replaces `PROMPT_FSDR_update_target1_free.md` and `PROMPT_FSDR_update_target2_split.md`. Where the master prompt says "Target 1 (20k)" / "Target 2 (20k)" / "40k", read the groups and sizes above.

## 11. Open items
1. Control size (default 20k).
2. LLM weekly budget (default 10 h) and model (8B FP8).
3. `guard_mode_target1`: `repair` (default) or `report`.
4. Approval of CC-v1 and CF-v1 values (band 1.0 – 1.5 ×, +0.5 OMR, do-not-disturb after 3 ignored offers).

## 12. Evidence to show when done
The change plan and test results; the assignment report (sizes and balance of the 5 groups); 2 decision traces per group for the same type of customer; the 100k LLM benchmark; one audit report and 5 rows of `input_flytxt` from a dummy run.
