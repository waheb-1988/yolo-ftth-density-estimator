# How to run `03_gpu_agents_step_by_step.ipynb` on the new laptop (Ollama)

**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB), driver 597.13 (CUDA 13.2), Windows 11.
**What it does:** one full week with **every agent** (Agent 1 to Agent 9), one by one. The LLM agents run on the GPU through **Ollama**, a native Windows app. No WSL is needed.
**Data:** one weekly folder `data/weekly/<D>/`, with fresh `*_auto` tables and no `*_score` tables.
**Target 2 only:** group a (T2C, 50,000) and group b (T2F, 50,000), decided by the agents. No control group, and Target 1 is not scored.

Full description: [docs/GPU_AGENTS_NOTEBOOK.md](../docs/GPU_AGENTS_NOTEBOOK.md).

---

## 1. Once per laptop

1. **The environment:** see [01_training_step_by_step.md](01_training_step_by_step.md) §1 (`uv sync --locked`, `.env`).
2. **The driver** is already right: 597.13 supports Blackwell. Check that `nvidia-smi` lists the RTX PRO 5000.
3. **Ollama** is already installed. Check it in PowerShell:
   ```powershell
   ollama --version
   ```
4. **A model.** This is the only download. With 24 GB, both fit entirely on the GPU:
   ```powershell
   ollama pull qwen3:8b       # about 5 GB, fast (recommended to start)
   ollama pull qwen3:14b      # about 9 GB, better answers, about 2x slower
   ```
5. **Answer several customers at once.** Run this once, then quit Ollama (tray icon → Quit) and start it again:
   ```powershell
   setx OLLAMA_NUM_PARALLEL 8
   setx OLLAMA_FLASH_ATTENTION 1
   ```

## 2. At each session

1. Start the **Ollama** app; its icon appears next to the clock.
2. Start JupyterLab:
   ```powershell
   cd D:\production\nbx-python-upgrade-V4.0
   uv run jupyter lab
   ```
3. Open `notebooks/03_gpu_agents_step_by_step.ipynb`.

## 3. Settings (first code cell)

| Setting | **Default = your real week** | A test | Why |
|---|---|---|---|
| `RUN_DATE` | **`"2026-11-19"`** | `"2026-10-29"` | the folder `data/weekly/<D>/` (create it with `run.py fsdr-folders --date <D>`) |
| `GENERATE` / `GENERATE_TARGETS` | **`False`** | `True` | real data is never overwritten anyway |
| `MODEL` | `"auto"` | `"auto"` or `"qwen3:14b"` | `"auto"` picks the best model installed |
| `PARALLEL` | **`8`** | **`8`** | the same as `OLLAMA_NUM_PARALLEL` |
| `TIME_BUDGET_MIN` | `20` | `20`, or `600` for every customer (overnight) | the time for agents 3 + 4; customers past it get the cell policy |
| `MAX_LLM_LINES` | `None` | `None` | or a number, to limit the customers |
| `AGENT_TARGET` | `"target2"` | `"target2"` | the agents decide groups a and b |

## 4. Run

Click **Kernel → Restart Kernel and Run All Cells**.

| Step | What you must see |
|---|---|
| G1 | `GPU : NVIDIA RTX PRO 5000 …`, `MODEL : qwen3:8b`, and **`in GPU memory: 100%`** |
| G2 | the seconds per call, and the estimate for a production week |
| 1 | the tables of `data/weekly/<D>/` |
| 2, Agent 1 | `T0 DONE -- pipeline train_week0`, then the LLM's summary |
| 3, Agent 2 | the segments, with one explanation each |
| 4, Agent 2b | `dict_segm_actions_agent_v<N>`: cells with a new top action, and what the check refused |
| 6, Agent 3 | the T2C decisions: `llm_nba` / `llm_nba_sampled`, and the propensity of each |
| 7, Agents 4 + 6B | `who decided, per group`, with `llm_scored_offers` > 0 |
| 8–11 | Target 2 READY, Agent 9's verdict, Agent 8's notes (group a vs b, agents vs cell policy), then `final file READY` |
| Time per agent | calls, tokens and seconds per agent, and the extrapolation to a production week |

Watch the GPU while it runs with `nvidia-smi -l 1`: `GPU-Util` goes up, and `ollama` is listed.

## 5. Time on this laptop (estimates; G2 measures them)

| Model, `PARALLEL = 8` | Dummy run (930 Target-2 customers) | Production week (50k a + 50k b = 100k customers) |
|---|---|---|
| `qwen3:8b` | about 5–10 min | about 8–14 h |
| `qwen3:14b` | about 10–20 min | about 16–28 h |

Agents 1, 2, 2b, 8 and 9 add about 1–3 minutes. A re-run of the same week is free: answers are cached in `data/runs/<D>/agents/llm_cache.jsonl`.

## 6. Output

- `data/output/<D>/target2|final/`
- `data/runs/<D>/agents/`: `agent3_nba.csv`, `agent_times.json`, the cache
- `data/runs/<D>/train/T3_segments/dict_segm_actions_agent_v<N>.json`: next week starts from this version

## 7. Problems

| What you see | Fix |
|---|---|
| `Ollama does not answer` | Start the Ollama app, then run G1 again |
| `model … is not installed` | `ollama pull qwen3:8b` |
| `in GPU memory: 70 %` | Close other programs that use the GPU, or use `qwen3:8b` |
| Very slow, `GPU-Util` 0 % | Ollama is running on the CPU: restart Ollama; check that `nvidia-smi` works |
| G2: `every test call failed` | Ollama stopped: restart it, then run G1 and G2 again |
