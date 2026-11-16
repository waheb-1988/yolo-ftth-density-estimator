# The GPU notebook: every agent on the laptop GPU with Ollama

`notebooks/03_gpu_agents_step_by_step.ipynb` runs one week of FSDR V4.0 with **every agent**, one by one. The LLM agents run on the laptop GPU through **Ollama**.

- **Local only:** the only server it calls is `http://127.0.0.1:11434`.
- **Standalone:** like notebooks 01 and 02, it imports nothing from the project folder.
- **No API and no Gradio:** nothing else is started.

---

## 1. Before the first run

1. **Ollama is running.** Open the *Ollama* app, or run `ollama serve`.
2. **A model is installed.** This is the only download (5–9 GB):
   ```powershell
   ollama pull qwen3:8b        # fast; the default choice
   ollama pull qwen3:14b       # better answers, about 2x slower
   ```
   With 24 GB of GPU memory, both fit entirely on the GPU. With the setting `MODEL = "auto"`, the notebook picks the best one installed.
3. **Recommended: answer several customers at once.** In PowerShell, run this once:
   ```powershell
   setx OLLAMA_NUM_PARALLEL 8
   setx OLLAMA_FLASH_ATTENTION 1
   ```
   Then quit Ollama (right-click its icon, then **Quit**) and start it again. Set `PARALLEL = 8` in the notebook.
4. **Start JupyterLab:**
   ```powershell
   cd D:\production\nbx-python-upgrade-V4.0
   uv run jupyter lab
   ```
   Open `notebooks/03_gpu_agents_step_by_step.ipynb` and click **Kernel → Restart Kernel and Run All Cells**.

The defaults use dummy data (3,000 lines) and a 20-minute budget for the per-customer agents, so a first run takes a few minutes.

---

## 2. One weekly folder

Each week you export the tables **once**, into `data/weekly/<D>/`. `<D>` is the run date, written `YYYY-MM-DD`.

```
data/weekly/<D>/  usage/ models/ subscription/ catalog/ others/
                  target2/a_constraint/target2_a.csv.gz  target2/b_free/target2_b.csv.gz
```

| Copy every week | Why |
|---|---|
| `activities_w1…w4_auto`, `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto`, `subs_amt_auto`, `party_auto`, `access_method_id_auto`, `targeting_dataiku_auto`, `subs_btl_auto` | the fresh history: the features of every line come from these |
| `base_prepaid`, `consumer_aact_account`, the 11 model tables | the base and the model scores (churn, appetency, …) |
| `btl_catalog`, `sub_btl_3months` | the offers, and the 270-day purchases (the offer universe and the rule score) |
| `subscriptions_tg`, `usage_data_voice_payg_tg` | the price anchors (B-26, F-05) |
| `targeting_auto`, `BTL_script_english_arabic_ok`, `conflicting_offers_h`, `consent_contact` | the campaigns already running, the SMS scripts, the conflicting offers, and do-not-contact |
| the 5 target lists | who gets scored |

**Not needed:** the `*_score` tables and `party_scoring`.
- A run with fresh `*_auto` tables is a week-0 run, which reads the `*_auto` history only. T0 lists the missing `*_score` tables as warnings, not as blocking issues.
- No separate training or scoring folder is needed.
- **Scoring = the trained base joined with your target lists**, then the offers from the rules and the agents, then the output files.

With real data, set `GENERATE = False` and `GENERATE_TARGETS = False`. Real exports and real lists are never overwritten by dummy data, even if you forget.

---

## 3. The steps and the agents

| Step | Agent | Runs on | What it decides | LLM calls |
|---|---|---|---|---|
| G1 | — | GPU | checks the GPU and Ollama, picks the model, and shows that it sits 100% in GPU memory | 1 |
| G2 | — | GPU | speed test: seconds per customer, and the time estimate | 20 |
| 1 | — | CPU | the weekly folder (your exports, or dummy data) | — |
| 2 | **Agent 1** ① data | CPU + GPU | rules validate the data and **block** a bad run; the LLM explains the report | 1 |
| 3 | **Agent 2** ② segmentation | CPU + GPU | rules build the features and the S-segments; the LLM explains each segment | 14 |
| 4 | **Agent 2b** ②b dictionary | GPU | the action of each segment × nationality cell (Target 2), checked by H-06; saved as `dict_segm_actions_agent_v<N>` | ≤ 60 |
| 5 | — | CPU | target lists, then the panel | — |
| 6 | **Agent 3** ③ NBA | GPU | the action of each T2C customer: ε exploration, logged propensity | 1 per T2C customer |
| 7 | **Agent 4** ④ NBO + **Agent 6B** | GPU + CPU | the LLM scores the offers the rules allow; the challenger rules choose | 1 per Target-2 customer |
| 8 | **Agent 7** ⑦ code 5 | CPU | the 38-column file of Target 2 | — |
| 9 | **Agent 9** ⑨ auditor | CPU + GPU | rules **block** the final file; the LLM explains the checks | 1 |
| 10 | **Agent 8** ⑧ strategy | GPU | group a against group b, agents against the cell policy, and what to change next week | 1 |
| 11 | — | CPU | the final file of Target 2 | — |

**What runs on the GPU:** only the LLM. The data steps (pandas, Parquet) run on the CPU and take seconds to minutes. That is normal; they would not be faster on a GPU.

**The LLM proposes, the rules decide:**
- **Agent 3** can only choose among the allowed actions: no upsell for an at-risk segment (H-06). The code draws the action, p = (1 − ε)·score/Σscores + ε/k, and logs its propensity, so next week's uplift can be measured without bias.
- **Agent 4** only scores offers that the universe rules already allowed. The passes still choose: price band, one offer per family, and EXT-6 for slots 5 and 6.
- **Agent 2b:** an invalid dictionary cell keeps its current action.
- **Agent 9:** the rules give the verdict. If the LLM disagrees, the rules win.
- **Traceability:** every row says who decided. `DECIDED_BY` is `llm_nba`, `llm_nba_sampled` or `fallback_cell_policy`, and `score_source` is `llm:<model>` or `fallback:E-O3+E-O1`.
- **Privacy:** no MSISDN is ever sent to the LLM, only the customer's facts.

**Target 2 only (V4.0):** group a (T2C, 50,000) and group b (T2F, 50,000) are decided by the agents; Target 1 (the production rules) is not scored. The control group (20,000) is not contacted, so it measure what the agents add.

---

## 4. How long it takes with Ollama (no vLLM)

Agents 1, 2, 2b, 8 and 9 make about 80 calls in total: **1–3 minutes**. The time goes to **agents 3 and 4**, which make one call per Target-2 customer: T2C needs agent 3 (the action) and agent 4 (the offers), and T2F needs agent 4 only.

These estimates are for the RTX PRO 5000 Blackwell laptop GPU (24 GB) with `PARALLEL = 8`. **Step G2 measures the real numbers on your laptop**, and the last cell gives the measured time per customer.

| Model | Per Target-2 customer | Dummy run (930 Target-2 customers) | Production week (50k T2C + 50k T2F) |
|---|---|---|---|
| `qwen3:8b` | about 0.3–0.6 s | about 5–10 min | about **8–14 h** |
| `qwen3:14b` | about 0.6–1.2 s | about 10–20 min | about **16–28 h** |
| `qwen3:8b`, `PARALLEL = 1` | about 1.2–2.5 s | about 20–40 min | about 35–60 h |

**Measured on the development laptop** (GTX 1650, 4 GB, `qwen3:1.7b`, `PARALLEL = 2`):
- the whole notebook took 289 s for 3,000 lines, with the per-customer agents limited to 30 customers;
- agents 3 and 4 took 3.7 s per customer, and G2 had estimated 3.6 s;
- 0 LLM errors;
- agents 1, 2, 2b, 8 and 9 took 1.7 min of wall time.

Your RTX PRO 5000 has about 6x the memory and many times the compute, so expect the table above.

The time budget handles the rest:
- **`TIME_BUDGET_MIN`** (or `MAX_LLM_LINES`) caps the per-customer agents.
- Customers are taken in priority order: retention first, and the four groups interleaved, so they get equal coverage.
- Customers past the budget get the **cell policy of agent 2b**. That is still an agent decision, made once per cell instead of once per customer.
- For a full production week, set `TIME_BUDGET_MIN = 600` (10 h) and run it overnight. Or keep the budget, so the priority customers get the per-customer agents.

**Re-runs are free:** every answer is cached in `data/runs/<D>/agents/llm_cache.jsonl`, keyed by the model and the prompt.

vLLM (WSL2) batches many requests at once and is usually 3–5x faster at the same quality. It is optional: this notebook only needs Ollama. Notebook **04** runs the same steps with vLLM; see [VLLM_AGENTS_NOTEBOOK.md](VLLM_AGENTS_NOTEBOOK.md).

---

## 5. What it writes

| Where | What |
|---|---|
| `data/runs/<D>/train/T0_load/`, `T2_features/`, `T3_segments/` | the validation, the features, the segments and the actions; `dict_segm_actions_agent_v<N>.json` (agent 2b) and `agent_dictionary_check_v<N>.json` (the changes and what was refused) |
| `data/runs/<D>/score/P6_offers_target2/group_report.csv` | per group: `decided_by`, `llm_scored_offers` |
| `data/runs/<D>/agents/` | `agent3_nba.csv` (actions and propensities), `llm_cache.jsonl`, `agent_times.json` (calls, tokens and seconds per agent) |
| `data/output/<D>/target2/`, `final/` | the same output files as the other notebooks; `final/input_flytxt.csv.gz` is the file you hand over |

The engine only saves files; it sends nothing.

---

## 6. Problems

| What you see | Fix |
|---|---|
| `Ollama does not answer at http://127.0.0.1:11434` | Start the Ollama app (or `ollama serve`), then run the cell again |
| `model … is not installed` | `ollama pull qwen3:8b` |
| `in GPU memory: 70 %` (less than 100 %) | The model does not fit: use a smaller one (`qwen3:8b`), or close other programs that use the GPU |
| Very slow; `nvidia-smi` shows `GPU-Util` near 0 | Ollama is running on the CPU: update the NVIDIA driver, then restart Ollama |
| Agent answers `None`, or `failed` in G2 | `STATS` shows the last error. A timeout means the GPU is busy: lower `PARALLEL` |
| `T0 is BLOCKED by the rules` | Read `data/runs/<D>/train/T0_load/validation_report.md` and fix the data (see [DATA_UPDATE.md](DATA_UPDATE.md)) |
