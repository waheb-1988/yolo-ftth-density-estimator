# notebooks3 · FAST_MODE: agents 3 and 4 answer once per customer profile

The same weekly run as `notebooks2/` (train on `*_auto` at T, score on `*_score` at D, every agent on the laptop GPU),
with one change: **agent 3 (NBA) and agent 4 (NBO) ask the LLM once per customer profile, not once per customer.**
These two agents are almost all of the GPU time, so this is where a week gets shorter. Every other step is the
notebooks2 cell, and the engine (`nbx/`) is not modified.

| # | Notebook | Runs on | What it does |
|---|---|---|---|
| 02 | `02_prepare_week.ipynb` | CPU | `data_raw/` → `data/weekly/<D>/`: the new export names, D = t1 / T = t1 − 7, the latest campaign of `targeting_dataiku_auto`, the NBA dedup, and a test copy of `RUN_SIZE` = 1k / 5k / 10k lines |
| 03 | `03_agents_nba_fast_ollama_gpu.ipynb` | GPU (Ollama) | every agent; agents 3 / 4 per profile |
| 04 | `04_agents_nba_fast_vllm_gpu.ipynb` | GPU (vLLM, WSL2) | the same, with vLLM as the server |
| 04 real | `04_agents_nba_fast_ollama_real25k.ipynb` | GPU (Ollama) | **the real run**: dates from the data (D = last purchase, T = campaign day), training on the whole prepaid base, `RUN_SIZE` lines scored (1k / 10k / 25k / 50k / all), no LLM budget (retries, Ollama restart, resume), 6 offers per line (top-up + audit A-25), final report. Headless: `python notebooks3/run_notebook04.py` |

| **02 final** | `02_prepare_week_final.ipynb` | CPU | **the weekly run**: the raw exports + `nbx_llm_target.csv.gz` in `data_raw/` → `data/weekly/<D>/`; the target file is the only list (group a, constraint); stops if it is missing or if an export is older than the others by more than 5 days |
| **04 final** | `04_nbx_llm_final.ipynb` | GPU (Ollama) | **the weekly run**: every line of `nbx_llm_target`, training on the whole base, constraint logic, no LLM stop, 6 offers per line; outputs in `data/nbx_llm/`. Headless: `python notebooks3/run_notebook04.py --final` |
| 05 | `05_check_flytxt_results.ipynb` | CPU | every check of the final file and the 360 view, as tables, with the ARPU (no-downgrade) check and a scorecard |

**Every week (final run):** this week's exports + `nbx_llm_target.csv.gz` in `data/data_raw/` (last week's files removed)
→ `02_prepare_week_final` (Restart & Run All) → `python notebooks3/run_notebook04.py --final` → `05_check_flytxt_results`.

Optional helpers in `notebooks2/`: **01** duplicates, and **05** column check (only if step 2 of 03 is BLOCKED).
Every notebook prints the time of each step, its estimate and the time left (`[time]` lines).

**Run order:** `02_prepare_week` → `03` (or `04`). Use the same **`RUN_SIZE`** (`1_000`, `5_000`, `10_000` or `"all"`) in
both; with `RUN_DATE = None`, 02 reads D from the data and 03 takes the latest week 02 prepared.

- **How to run the real week, step by step** (from `data_raw/` to 50k scored): [RUN_REAL_DATA_STEP_BY_STEP.md](RUN_REAL_DATA_STEP_BY_STEP.md)
- **The full explanation** of every step, the agents, the data, training, scoring and time: [FULL_EXPLANATION.md](FULL_EXPLANATION.md)

## How FAST_MODE works

1. **Step 5b, before any LLM call:** each Target-2 customer's facts are rounded into bands. Customers with the same
   rounded facts make one **profile**. The cell counts the profiles at four levels and gives the GPU time of each,
   from the speed G2 has just measured:

   | Level | What the prompt keeps |
   |---|---|
   | `fine` | segment, value band, usage, revenue M1 (5 bands) and trend, data / voice (5 bands), active days, tenure, churn and appetencies (5 bands), last week with its action |
   | `medium` | the same with 3 bands |
   | `coarse` | 3 bands; no data, voice, active days or tenure |
   | `minimal` | as coarse, without the appetencies; last week = took an offer / no offer / not targeted |
   | `off` | the exact facts: one call per customer (= notebooks2) |

   `PROFILE_LEVEL = "auto"` picks the finest level whose estimate fits `TIME_BUDGET_MIN` (45 min by default).
2. **Agent 3:** one answer per T2C profile (the segment and cell policy are part of it, so the allowed actions are
   the same for all its customers). The action is still **drawn per customer** (ε draw, with the customer's own
   seed), so the logged propensity stays the customer's own and the uplift stays measurable.
3. **Agent 4:** one answer per (profile, action). The LLM scores the offers that its customers most often have on
   their short lists. Each customer gets those scores for the offers the rules allow them, and the passes still
   choose per customer (price band, one per family, EXT-6).
4. **Quality check** (`FAST_CHECK_N = 100`): that many customers are also asked with their **exact** facts. Step 6
   prints how often the profile and the exact facts give the same top action, and warns below 70 %.
5. **Shared cache** (`SHARED_LLM_CACHE = True`): answers are kept in `data/runs/_llm_cache.jsonl` (keyed by model +
   prompt, with no MSISDN). A profile already answered in an earlier week costs nothing the next time.

## Measured on dummy data (20k target lines, fake LLM)

These figures count LLM answers, not minutes: the fake server answers instantly. They show how much each level
groups; your real profile count and time are printed by step 5b on your data.

| Level | Customers per T2C profile | Agent-3 answers | Agent-4 answers | Calls saved |
|---|---|---|---|---|
| `fine` | 1.1 | 7,103 for 7,493 | about 15,000 | 4 % |
| `minimal` | 6.6 | 1,129 for 7,493 | 4,208 | 77 % |

- `fine` and `medium` hardly group anyone: with ten or more banded facts, almost every customer is unique. The gain
  comes from `coarse` / `minimal`, and it grows with the list size (more customers share each profile).
- On a real 50k + 50k week, expect a few times fewer calls at `minimal`, not ten times: check the `customers per
  profile` column of step 5b.

## Step by step

1. **02** (in `notebooks2/`): as usual, then **Restart & Run All**.
2. **03 / 04, first run with `TEST_10K = True`:** **Restart & Run All**, then read:
   - **G2:** the speed of one call on your GPU.
   - **Step 5b:** the table of the levels (profiles, customers per profile, estimated GPU minutes) and the level
     chosen.
   - **Step 6:** `quality check: ... same top action for X %`. Below 70 %, choose a finer level, or raise
     `TIME_BUDGET_MIN` so that `auto` picks one, then run from step 5b.
   - **Step 7:** customers per answer and the share of their own short-listed offers the answers scored.
   - **FAST_MODE in figures:** the calls saved and the GPU minutes against the per-customer estimate.
   - **G3:** `VERDICT: LLM + GPU run`.
3. **The real week:** `TEST_10K = False` in 02 and in 03 / 04. Step 5b prints the real profile counts before any
   call: if the level chosen is too coarse for you, set `PROFILE_LEVEL` yourself and raise `TIME_BUDGET_MIN`.
4. **Next week:** new exports in `data/data_raw/`, a new `RUN_DATE`. The shared cache answers the profiles already
   seen with the same model.

## Settings (in addition to those of notebooks2)

| Setting | Default | Meaning |
|---|---|---|
| `SCORE_GROUPS` | `["t2c", "t2f"]` | the Target 2 lists to score: `["t2c"]` = group a (~50k), `["t2f"]` = group b, both = ~100k |
| `PROFILE_LEVEL` | `"auto"` | `fine` / `medium` / `coarse` / `minimal` / `off`; `auto` = the finest that fits the budget |
| `TIME_BUDGET_MIN` | 45 | GPU minutes for agents 3 + 4; profiles past it get the cell policy |
| `FAST_CHECK_N` | 100 | customers also asked with their exact facts (0 = no check) |
| `SHARED_LLM_CACHE` | `True` | keep the answers across weeks in `data/runs/_llm_cache.jsonl` |

## Outputs

As in notebooks2 (`data/runs/<D>/`, `data/output/<D>/final/input_flytxt.csv.gz`, or under `data/test10k/` with
`TEST_10K`), plus:

- `agents/agent3_nba.csv`: a `PROFILE` column, the profile each customer was decided with;
- `data/output/<D>/target2/target2_scored_360.parquet` (+ `.csv.gz`): the scored MSISDNs with all the data and the check
  flags (step 12); `target2_360.parquet`: every target line, scored or not;
- `agents/fast_mode.json`: the level table, the level chosen, answers, calls saved and the quality check. The
  same figures are in `agents/llm_gpu_report.json` under `fast_mode`.

Everything under `data/` holds MSISDNs: keep it on this laptop, and never commit the notebooks with their outputs.

## Limits

- **Agents 3 and 4 only.** The CPU steps (02, T0, T2 / T3 on the full base, step 4b) take the same time as in
  notebooks2.
- **Customers in one profile get the same scores.** Agent 3 still draws each customer's action, and the passes
  still pick each customer's offers, but the LLM no longer sees a customer's exact figures. The quality check shows
  what that costs.
- **Agent 4 in profile mode** sees the offers that are most common in the profile, without the customer's price
  anchor. The rules have already kept only offers in the customer's price band.

## Rebuild

```powershell
uv run python notebooks3/build_notebooks3.py
```

The cells are in `nb3_fast.py`; every other cell comes from `notebooks2/` (and Part A from the engine). The notebooks
are written **without outputs**.
