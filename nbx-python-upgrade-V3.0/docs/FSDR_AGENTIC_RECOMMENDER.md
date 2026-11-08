# FSDR Agentic Recommender

The **FSDR Agentic Recommender** decides, for one customer of the `llm` arm, **what to do this week** (the action),
**which offers to send** for it, and **how they compare with the competition**. Three agents work in a row: the
**action agent**, the **offer agent** and the **competitor agent**. A code-only **critic** and a **messenger** finish
the decision.

The models the team already uses are tools the agents call: the Vowpal Wabbit bandit, the segment policy, the BPR
offer ranker and the ARPU price bands. Each tool returns evidence; none dictates the answer. An LLM may disagree with
them, but only with a written reason, and it can never break a hard business rule.

Related documents: [FSDR_GUIDE.md](FSDR_GUIDE.md) (the whole weekly cycle and every business rule) ·
[fsdr_plan.md](fsdr_plan.md) (status) · [GPU_SETUP.md](GPU_SETUP.md) (vLLM server).

---

## Contents

1. [Is an LLM used? (offline, Ollama, vLLM)](#1-is-an-llm-used-offline-ollama-vllm)
2. [Where to find it](#2-where-to-find-it)
3. [One decision, end to end](#3-one-decision-end-to-end)
4. [Agent 1: action agent](#4-agent-1-action-agent)
5. [Agent 2: offer agent](#5-agent-2-offer-agent)
6. [Agent 3: competitor agent](#6-agent-3-competitor-agent)
7. [Critic and messenger](#7-critic-and-messenger)
8. [How the agent loop keeps an LLM on track](#8-how-the-agent-loop-keeps-an-llm-on-track)
9. [Logging for learning: policy source and propensity](#9-logging-for-learning-policy-source-and-propensity)
10. [What the screen shows](#10-what-the-screen-shows)
11. [The decision record, field by field (real example)](#11-the-decision-record-field-by-field-real-example)
12. [Batch mode](#12-batch-mode)
13. [Settings](#13-settings)
14. [Limits](#14-limits)

---

## 1. Is an LLM used? (offline, Ollama, vLLM)

**Yes: on this machine the agents run on vLLM.** Measured on 2026-10-23:

| Check | Result |
|---|---|
| `.env` (git-ignored) | `LLM_BACKEND=vllm`, `VLLM_URL=http://127.0.0.1:8001`, `DEFAULT_LLM=qwen2.5:3b`, `VLLM_API_KEY=…` |
| vLLM server | vLLM 0.6.6.post1 in WSL2 Ubuntu 24.04, started with `python run.py vllm-serve` |
| Model | `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`, served as `qwen2.5:3b` (about 2 GB of weights, fp16 activations, 8k context) |
| GPU | NVIDIA GTX 1650, 4 GB (compute capability 7.5) |
| `GET /health` | `"llm_backend": "vllm"`, `"loaded_models": ["qwen2.5:3b"]` |
| Decisions | `mode: "agentic"`, `policy_source: "llm"` (or `"bandit"` on an exploration draw) |

Without a server, set `LLM_BACKEND=offline` (or run `python run.py --backend offline start`). Every decision then
shows `mode: "fallback"` and `stats.llm_calls: 0`. Each agent takes its **rule-based path**, which uses the same tools
as the LLM path. The result is exactly the constraint (`cstr`) arm: the bandit's sampled action, then the 3-pass
price-band offers, then a computed competitor verdict. A test checks this equivalence customer by customer.

The app has three interchangeable backends (`nbx/llms/llm_client.py`). The agents never call a backend directly.

| `LLM_BACKEND` | What answers | When to use it | How to start it |
|---|---|---|---|
| `offline` | nothing; rule-based agents | demos, tests, any machine | nothing to start |
| `ollama` | **disabled** (`OLLAMA_ENABLED=0`) | not used: vLLM replaces it | set `OLLAMA_ENABLED=1` in `.env`, start Ollama, `ollama pull qwen3:4b`, then `python run.py --backend ollama start` |
| **`vllm`** | one model on a vLLM server, OpenAI-compatible API | **this laptop (WSL2) and production GPU servers** | `python run.py vllm-install` (once), `python run.py vllm-serve`, then `python run.py start`. Details below and in [GPU_SETUP.md](GPU_SETUP.md#14-small-gpu-on-windows-wsl2-gtx-1650-4-gb) |

If the LLM fails in the middle of a run, the remaining agents fall back and the decision shows
`mode: "partial_fallback"`.

### 1.1 vLLM on this laptop, step by step

```powershell
python run.py vllm-install     # once: uv + vLLM venv in WSL (~/vllm-env), no sudo needed
python run.py vllm-serve       # leave it open; ready after "Application startup complete" (~3 min)
python run.py start            # second window: API + UI on vLLM (reads .env)
```

`.env` holds `LLM_BACKEND=vllm`, `VLLM_URL`, `DEFAULT_LLM=qwen2.5:3b` and a random `VLLM_API_KEY`. The same key is
passed to the server, so other programs on the machine cannot use it. To choose the model, set `VLLM_HF_MODEL` and
`VLLM_SERVED_NAME` in `.env`; `vllm-serve` passes them into WSL.

### 1.2 What it costs on a 4 GB GPU (measured)

| Setup | Speed | One decision |
|---|---|---|
| Qwen2.5-3B **AWQ** | 4 tokens/s (the AWQ kernels are slow on Turing) | 553 s, and the model wasted most steps |
| Qwen2.5-3B **GPTQ-Int4** | 13–14 tokens/s | 310 s |
| GPTQ-Int4 + **evidence prefetch** (`FSDR_AGENT_PREFETCH=1`, the default) | 13–14 tokens/s | **25–150 s, 1–7 LLM calls** |
| Batch of 10 customers, concurrency 8 | the GPU is the limit | 500–800 s (about 80 s per customer) |

**Evidence prefetch.** Before the first LLM call, the code runs each agent's required evidence tools:
- action agent: `customer_360`, `bandit_pmf`, `contact_policy`;
- offer agent: `price_band_candidates`, then `check_offers` on the rule-based picks;
- competitor agent: `competitor_compare`.

The code hands the results to the agent as evidence it already has. The trace shows the same tool calls, and the
LLM can still call any tool. It no longer spends calls fetching what it must fetch anyway. Small models used to lose
most of their step budget on refused duplicates and blocked finals. Set `FSDR_AGENT_PREFETCH=0` to let the LLM make
every call itself.

**Plain-words evidence.** The 3B model read the rule text "upsell is blocked when PROBA_CHURN ≥ 0.60" as "this
customer is high risk", and chose NO_CONTACT for 6 customers out of 10. `contact_policy` now also returns a
`verdict` sentence, for example "Contact allowed and no action is blocked: churn_score 0.29 is below the 0.6
threshold". After this change the same 10 customers gave 1 NO_CONTACT, and the actions were spread across RETAIN ×3,
UPSELL VOICE ×2, UPSELL, UPSELL DATA, CROSS SELL VOICE and MAINTAIN.

**Quality.** A 3B model reasons correctly most of the time, but its thoughts can still misquote a number. The
hard rules do not depend on it: the code enforces eligibility, the churn block, the contact gap and every offer rule
(section 7). On a larger GPU, serve a 7B model (`Qwen/Qwen2.5-7B-Instruct-AWQ` on an A10 / L4 or bigger). There one
call takes about a second and batches run many customers in parallel.

## 2. Where to find it

| Interface | How |
|---|---|
| **Web UI** | tab **FSDR Agentic Recommender**: pick the run and a customer (the list is sorted by churn score, highest first), then click **Run the agents**. The **Batch** section below it runs many customers |
| **API** | `POST /fsdr/decide/{run}` with `{"msisdn": "70094659"}` returns the decision + trace. `GET /fsdr/stream/{run}?msisdn=…` streams every event live (Server-Sent Events). `POST /fsdr/batch/{run}` runs the batch flow |
| **Command line** | `python run.py fsdr-decide --msisdn 70094659 --trace` (one customer), `python run.py fsdr-recommend --arm llm` (the whole llm arm), `python run.py fsdr-batch --arm llm` (batch flow) |

**Prerequisite.** The run must be trained and scored (tabs FSDR Data → FSDR Training & Scoring, or `fsdr-train` and
`fsdr-score`). The recommender reads the scoring output: the bandit's probabilities per customer.

**Code:** `nbx/fsdr/agents/recommender_agent.py` (agents, rules, orchestration) and `nbx/fsdr/agents/tools.py`
(tools and the per-run workspace). Both run on the shared loop `nbx/agents/agentic/react.py`.

---

## 3. One decision, end to end

```
                    ┌─────────────────────── evidence (tools) ───────────────────────┐
                    │ customer_360 · bandit_pmf · segment_policy · campaign_history   │
                    │ contact_policy · price_band_candidates · ranker_top_offers      │
                    │ arpu_anchors · search_catalog · check_offers · expected_value   │
                    │ competitor_compare · competitor_market                          │
                    └──────────────┬──────────────────┬──────────────────┬───────────┘
                                   ▼                  ▼                  ▼
 customer ─► 1 ACTION AGENT ──action──► 2 OFFER AGENT ──offers──► 3 COMPETITOR AGENT ──► 4 CRITIC ──► 5 MESSENGER ──► decision
              hard rules on the          hard rules on the          swaps only if          (code: last    (BTL scripts     + trace
              action, exploration        offers, repair             they pass hard rules    rule gate)     EN / AR)         + log
```

Step by step, for one customer:

1. **Load the context.** The run's workspace is loaded once per run: the JSON tables, the scoring output, the features
   (D-anchored), the offer engine, the catalogue, the conflict pairs, the BTL scripts and the competitor snapshot.
   Each customer gets a context object that the three agents share.
2. **Action agent.** It calls its required tools and chooses one action or NO_CONTACT (section 4).
3. **Code check on the action.** Eligibility, contact gap and the churn block may change the action. Then, in agentic
   mode, the ε-exploration draw may replace it by a bandit sample. The final action gets its `policy_source` and
   `propensity`.
4. **Offer agent.** It chooses up to 4 offers for that action (section 5). Offers that break a rule are repaired.
   NO_CONTACT skips this step.
5. **Competitor agent.** It compares the offers with competitor plans, gives a verdict, and may swap an offer
   (section 6).
6. **Critic.** It checks every hard rule again on the final list and records every repair (section 7).
7. **Messenger.** It writes the SMS from each offer's English / Arabic script (section 7).
8. **Output.** The decision record (section 11) and every event of the trace. The UI renders the trace live.

---

## 4. Agent 1: action agent

**Goal:** pick the best action for this customer this week.

**Choices:** `CROSS SELL DATA`, `UPSELL DATA`, `RETAIN`, `MAINTAIN`, `UPSELL`, `CROSS SELL VOICE`, `UPSELL VOICE`, or
**`NO_CONTACT`**.

| Tool | Returns | Required? |
|---|---|---|
| `customer_360` | plan, line start date, nationality group, usage type (voice-only / data-only / dual / inactive), eligibility issues, the 9 model scores (PROBA_CHURN, reactivation, appetency, fragility, revenue decrease…), revenue W1…M3, trends (revenue / voice / data, W1 vs W2, M1 vs M2), M1 usage | yes |
| `bandit_pmf` | the bandit's probability for each of the 7 actions, its top action, and the action sampled by the scoring pipeline | yes |
| `contact_policy` | can we contact this week; eligibility issues (BR-02, BR-10, Nursering BR-36); days since the last contact; churn score; actions blocked by churn risk | yes |
| `segment_policy` | the customer's segment and its BR-39 action (what the old rule would do) | no |
| `campaign_history` | last week's targeting (date, segment, offers sent), BTLs taken in the last 7 days, whether an offered BTL was taken, top BTLs of the last 270 days | no |

**Final answer:** `{"action", "signals_used": [...], "override_reason", "confidence", "expected_acceptance"}`.

**Checks** (an answer that fails goes back to the agent with the reason):

- the action must be one of the 8 choices;
- if the action is **not the bandit's top action**, `override_reason` is required (at least a short sentence).

**Code-enforced hard rules after the answer:**

| Rule | Effect |
|---|---|
| customer not eligible, or contacted less than 2 days ago | action → **NO_CONTACT** |
| UPSELL / UPSELL DATA / UPSELL VOICE while **PROBA_CHURN ≥ 0.60** | action → **RETAIN** |

Each change is a critic finding.

**Without an LLM:** the agent calls the 3 required tools (they appear in the trace) and takes the bandit's sampled
action, exactly as the cstr arm does.

---

## 5. Agent 2: offer agent

**Goal:** choose 1 to 4 BTL offers that maximise acceptance and value for the customer and the chosen action.

| Tool | Returns | Required? |
|---|---|---|
| `price_band_candidates` | the rule-based answer: for the action's families (BR-66, `all_L2` or `normal`), the 3-pass selection around the ARPU anchor (BR-78…82), with price, generosity, validity, pass and ranker score | yes |
| `check_offers` | for a proposed list: **hard violations** (not in the catalogue, price ≤ 0, more than 2 per validity, more than 4 in total, conflicting pair, customer not eligible) and **soft notes** (outside the action's families / generosity, outside the price band) | yes |
| `ranker_top_offers` | the BPR ranker's top offers for the action, per validity | no |
| `arpu_anchors` | the customer's anchors (total / data / voice, monthly / weekly) and the soft price band per family | no |
| `search_catalog` | **any** catalogue offer, by service, validity, generosity or price | no |
| `expected_value` | heuristic acceptance probability and value per offer (not a trained model, labelled as such) | no |

**Final answer:** `{"offers": [codes], "rationale", "override_reason"}`.

**Freedom:** any eligible offer in the catalogue, even outside the action's families or the price band. Those are
**soft rules**, shown as guidance.

**Checks:**

- the list must pass every hard rule. If not, the agent is told which rule it breaks and must answer again;
- if the list differs from `price_band_candidates`, `override_reason` is required.

**Repair** (if the agent stops on an invalid list): offers are kept in order as long as they pass the hard rules
together. If none is left, the rule-based offers are used. Each repair is a critic finding.

**Without an LLM:** the 3-pass offers, identical to the cstr arm (each offer keeps its `pass` and `anchor`).

---

## 6. Agent 3: competitor agent

**Goal:** check whether the chosen offers are competitive, and propose a swap only when one is clearly beaten.

| Tool | Returns | Required? |
|---|---|---|
| `competitor_compare` | for each offer: our monthly-equivalent price; the closest plan of each competitor **at the same validity band** (plan, monthly price, data, monthly price difference, verdict such as "cheaper per month, less data"); `at_risk` when we cost more than 1 OMR/month over the closest plan | yes (always run first, so the comparison is in every decision) |
| `competitor_market` | competitor plans around the customer's monthly anchor | no |
| `search_catalog` | catalogue offers, to look for a better swap | no |

**Final answer:** `{"verdict": "competitive|mixed|uncompetitive", "summary", "swap": [{"out", "in", "reason"}]}`.

**Swaps:** each swap is applied **only if the new list passes every hard rule**. A refused swap is a critic finding,
and an applied swap is logged as an override.

**Without an LLM:** the verdict is computed: `competitive` if no offer is at risk, `uncompetitive` if all are,
`mixed` otherwise. No swap.

**Names:** competitor operators are always shown as **COMPETITOR_A** and **COMPETITOR_B** (`config.COMPETITOR_LABELS`),
also inside plan names. The prices come from the public snapshot `data/competitors/competitor_offers.json`.

---

## 7. Critic and messenger

**Critic (code).** It collects every finding of the run (action changed by a hard rule, offers dropped, swaps
refused) and re-checks the final offer list against all hard rules. Its verdict is `approved` (nothing changed) or
`revised` (something was repaired). The LLM never ships a decision that breaks a hard rule.

**Messenger.** It takes the English and Arabic scripts of the chosen offers from `BTL_script_english_arabic_ok`
(first two offers). Customers with NO_CONTACT or no offer get no message.

---

## 8. How the agent loop keeps an LLM on track

All three agents use the same loop as the NBX council (`react.py`). The code enforces the policy; the prompt does not
have to.

| Control | What it does |
|---|---|
| One JSON object per step | `{"thought", "tool", "args"}` or `{"thought", "final"}`; anything else is sent back as an error |
| Tool list | only the agent's own tools; an unknown tool is refused, with the list of valid ones |
| Required evidence | the agent cannot finish before calling its required tools |
| Answer validation | the final answer is checked (allowed action, reason for overrides, hard rules on offers). A failed check goes back to the agent with the reason |
| Repeated calls | an identical repeated call is refused, with the next step to take; a second repeat ends the agent |
| Step limit | 6 / 7 / 5 steps; then the code finishes from the evidence gathered |
| Fallback | if the LLM is unreachable or fails, the rule-based path takes over |

---

## 9. Logging for learning: policy source and propensity

To let the bandit learn from the llm arm without bias (off-policy), each decision records how its action was chosen.
With probability **ε_llm = 0.1**, the LLM's action is replaced by a **bandit sample**.

| `policy_source` | When | `propensity` |
|---|---|---|
| `llm` | the LLM's action is kept | 1 − ε + ε × pmf[action] |
| `bandit` | the exploration draw replaced it | ε × pmf[action] |
| `fallback` | no LLM | the bandit probability of the sampled action |

Each decision also logs:

- the full bandit pmf;
- the model and catalogue versions;
- the overrides, each with the model's choice, the agent's choice and the reason;
- the critic findings;
- a cache key.

In batch mode they go to `outputs/llm_decisions.jsonl`. The weekly campaign log and the D+7 rewards that turn these
logs into training data are Phase 6, not built yet.

---

## 10. What the screen shows

| Panel | Content |
|---|---|
| **Pipeline** | the three agents with their status (waiting, reasoning, done) and number of tool calls; the mode (LLM model name, or deterministic fallback) |
| **Agent reasoning, live** | every event as it happens: thoughts (LLM only), tool calls and their results, policy notes (refused answers, exploration, loop guard), the guardrail result |
| **Recommended action** | the action and its reason, with badges for mode, policy source, propensity, bandit top action, number of overrides and critic verdict |
| **Offers** | offer, service, generosity, validity, price, ARPU anchor and price pass |
| **Bandit evidence** | the probability of each action |
| **Overrides** | what the model or rules proposed, what the agent chose, and why |
| **Critic** | the hard-rule findings, or "All hard rules passed" |
| **Competitor agent** | the verdict, the summary, and for each offer the closest COMPETITOR_A / COMPETITOR_B plan, with prices, delta and verdict; any applied swap |
| **Customer message** | the English and Arabic SMS |

---

## 11. The decision record, field by field (real example)

A real decision from run `2026-10-23` (offline mode, customer 70094659; the competitor list is shortened):

```json
{
  "msisdn": "70094659", "arm": "llm", "run_date": "2026-10-23",
  "mode": "fallback", "model": null, "model_version": "v0003", "catalog_version": "c3dd4e33f5",
  "action": "UPSELL VOICE", "policy_source": "fallback", "propensity": 0.828571,
  "bandit_top_action": "UPSELL VOICE",
  "bandit_pmf": {"UPSELL VOICE": 0.8286, "CROSS SELL DATA": 0.0286, "UPSELL DATA": 0.0286, "RETAIN": 0.0286,
                 "MAINTAIN": 0.0286, "UPSELL": 0.0286, "CROSS SELL VOICE": 0.0286},
  "offers": [
    {"offer": "BTL_DATA_L2_30D_14.4GB",         "service": "DATA",  "validity": 30, "price": 8.0,  "pass": 1, "anchor": 6.83},
    {"offer": "BTL_COMBO_L2_30D_11.9GB_275MIN", "service": "COMBO", "validity": 30, "price": 11.0, "pass": 2, "anchor": 7.58},
    {"offer": "BTL_DATA_L2_7D_5.4GB",           "service": "DATA",  "validity": 7,  "price": 3.0,  "pass": 1, "anchor": 2.59},
    {"offer": "BTL_COMBO_L2_7D_3.8GB_87MIN",    "service": "COMBO", "validity": 7,  "price": 3.5,  "pass": 1, "anchor": 2.88}
  ],
  "overrides": [], "override_count": 0,
  "critic": {"verdict": "approved", "findings": []},
  "competitor": {
    "verdict": "mixed", "summary": "1 of 4 offer(s) priced above the closest competitor plan",
    "comparisons": [{"offer": "BTL_DATA_L2_30D_14.4GB", "our_monthly_equiv": 8.0,
                     "closest": [{"operator": "COMPETITOR_B", "plan": "RED 7", "monthly_equiv": 7.5, "data_gb": 10.0,
                                  "monthly_delta": 0.5, "verdict": "more expensive per month, more data per month"}],
                     "at_risk": false}],
    "swaps_applied": []
  },
  "message": {"sms_en": ["FSDR: enjoy 14.4GB data for 8 OMR, valid 30 days. Reply YES to activate."],
              "sms_ar": ["FSDR: استمتع بـ 14.4 جيجابايت مقابل 8 ر.ع لمدة 30 يوم. أرسل نعم للتفعيل."]},
  "justification": {"signals_used": ["bandit sampled action (constraint-arm logic)"], "override_reason": "",
                    "expected_acceptance": null, "offer_rationale": ""},
  "stats": {"llm_calls": 0, "tool_calls": 5, "elapsed_ms": 14},
  "cache_key": "e5f6b444446e4207"
}
```

How to read it:

- **`mode: fallback`, `llm_calls: 0`:** no LLM, so the rule-based path ran.
- **The bandit** gives UPSELL VOICE 82.9 % and each other action 2.9 %. That is ε-greedy exploration: the 0.2
  exploration share is spread over the 7 actions, 0.2 / 7 ≈ 0.029. The scoring pipeline sampled UPSELL VOICE, so the
  propensity is 0.8286.
- **The offers.** UPSELL VOICE uses families DATA + COMBO (BR-66, all L2). Three offers were found in **pass 1**
  (price between the anchor and anchor + 3). The monthly COMBO came from **pass 2**: no COMBO was inside the band, so
  the best-ranked COMBO above the anchor was taken (no upper limit; the cheaper one wins a tie). That makes 2 offers
  per validity, 4 in total.
- **The competitor check.** The 30-day DATA pack costs 0.5 OMR/month more than COMPETITOR_B's closest plan but gives
  more data. It is not "at risk" (the threshold is 1 OMR), so the verdict is `mixed` overall.
- **The trace** of this decision had 19 events: `run_start`; the action agent with 3 tool calls; the offer agent with
  `price_band_candidates`; the competitor agent with `competitor_compare`; `guardrails`; `run_end`.

---

## 12. Batch mode

In the same tab, the **Batch** section runs many customers in one click, processed batch by batch (default 25 per
batch). For each customer:

1. the action (this agent pipeline for the `llm` arm; the rules for `cstr` / `free`);
2. the top 4 offers (offer and competitor agents for `llm`);
3. the top 2 **roaming** packs (ranked on the customer's roaming history; cold start = popularity blended with the
   price fit to the customer's anchor);
4. **international** packs, only for customers with at least 3 international calls last month.

The table fills in as each batch finishes, and a CSV is ready at the end. Roaming and international packs never
duplicate or conflict with a main offer. Details and rules: [FSDR_GUIDE.md, step 6b](FSDR_GUIDE.md#step-6b-batch-flow-action--top-4-offers--top-2-roaming--international).

For the whole llm arm with resume support, use `python run.py fsdr-recommend --arm llm`. Each decision is appended to
`outputs/llm_decisions.jsonl` as soon as it is made, and an interrupted run continues where it stopped.

---

## 13. Settings

All in `nbx/fsdr/config.py`; some can be set in `.env`.

| Setting | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `offline` | `offline`, `ollama` or `vllm` |
| `DEFAULT_LLM` | `mistral:7b` (`.env` here: `qwen2.5:3b`) | model name the agents ask for (falls back to the only served model) |
| `VLLM_URL` | `http://127.0.0.1:8001` | vLLM server (when `LLM_BACKEND=vllm`) |
| `VLLM_API_KEY`, `VLLM_MAX_TOKENS` | none, `1024` (`.env` here: `512`) | server key; reply length limit per LLM call |
| `VLLM_HF_MODEL`, `VLLM_SERVED_NAME` | `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`, `qwen2.5:3b` | model served by `run.py vllm-serve` |
| `FSDR_AGENT_PREFETCH` | `1` | code runs the required evidence tools before the first LLM call (section 1.2) |
| `FSDR_AGENT_CONCURRENCY` | `8` on vLLM, else `4` | customers decided in parallel in batch runs |
| `FSDR_EPSILON_LLM` | `0.1` | share of llm-arm decisions replaced by a bandit sample |
| `CHURN_BLOCK_THRESHOLD` | `0.60` | PROBA_CHURN at or above this blocks the upsell actions |
| `CONTACT_MIN_GAP_DAYS` | `2` | minimum days since the last contact |
| `FSDR_GENEROSITY_MAPPING` | `all_L2` | offer families' generosity: `all_L2` or `normal` (BR-66) |
| `FSDR_COMPETITOR_LABELS` | `{}` (the shipped snapshot already uses COMPETITOR_A / COMPETITOR_B) | JSON map from the operator names of your own snapshot to the display labels |
| `COMPETITOR_UNCOMPETITIVE_DELTA` | `1.0` OMR | monthly price gap above the closest plan that marks an offer "at risk" |
| `ADDON_TOP_ROAMING`, `ADDON_TOP_INTERNATIONAL`, `INTL_MIN_CALLS_M1` | 2, 2, 3 | batch add-ons |
| step limits | 6 / 7 / 5 | action / offer / competitor agent |

---

## 14. Limits

- **Small GPU, small model.** On the 4 GB GTX 1650, vLLM serves a 3B model at about 80 s per customer: fine for a
  few customers or a batch of tens, too slow for a whole arm. In the UI batch, use a batch size of about 5 (each
  request has a 900 s timeout). A whole arm needs a server GPU and a 7B model.
- **Without a server** (`LLM_BACKEND=offline`) the agents reproduce the constraint arm. Overrides, swaps and written
  rationales only appear with an LLM.
- **The critic is code only.** It enforces the hard rules but does not yet ask an LLM to judge the quality of a
  justification (planned).
- **`expected_value` is a heuristic**, clearly labelled, not a trained uplift model.
- **The contact cap** uses the last targeting date only (minimum 2-day gap). The "3 contacts in 28 days" cap and the
  weekly learning loop need the campaign log (Phase 6).
- **The competitor snapshot is dated.** Refresh it with `python -m nbx.scripts.scrape_competitors` (needs internet).
  Plan names are the competitors' public product names; only the operator names are replaced by labels.
