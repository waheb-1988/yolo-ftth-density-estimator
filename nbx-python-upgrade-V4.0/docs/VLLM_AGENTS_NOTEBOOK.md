# The vLLM notebook: every agent on the laptop GPU with vLLM

`notebooks/04_vllm_agents_step_by_step.ipynb` has the **same steps and the same agents** as notebook 03 ([GPU_AGENTS_NOTEBOOK.md](GPU_AGENTS_NOTEBOOK.md)). Only the LLM server differs: **vLLM**, which runs in WSL2 and batches many customers together on the GPU. For the per-customer agents 3 and 4, that is several times faster than Ollama.

- **Local only:** the notebook calls only `http://127.0.0.1:8001`.
- **Standalone:** it imports nothing from the project folder.
- **No API and no Gradio.**

WSL2 must be approved first: [VLLM_VS_OLLAMA_SECURITY.md](VLLM_VS_OLLAMA_SECURITY.md) compares Ollama and vLLM for the security team and contains the request email. The run guide of each notebook is in `notebooks/<notebook>.md`.

Both notebooks are built from the same builder (`notebooks/build_gpu_notebook.py`), so their agent steps cannot drift apart.

---

## 1. Set up the environment (once)

vLLM runs on Linux, so on Windows it runs inside **WSL2** (Ubuntu). The Windows NVIDIA driver gives WSL2 the GPU. Do **not** install an NVIDIA driver inside Ubuntu.

### 1.1 WSL2 and Ubuntu

In PowerShell **as administrator**:

```powershell
wsl --install -d Ubuntu
```

Restart the laptop. Ubuntu then opens and asks for a user name and password. Check that the GPU is visible inside WSL2:

```powershell
wsl -e nvidia-smi          # must list the NVIDIA RTX PRO 5000
```

### 1.2 Install vLLM

From the project folder, in a normal PowerShell:

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run python run.py vllm-install
```

`vllm/install_wsl.sh` runs inside WSL2. It installs uv, creates `~/vllm-env` (Python 3.12) and **chooses the vLLM version from the GPU**:

| GPU compute capability | Profile | vLLM |
|---|---|---|
| 8.0 or more: RTX 30xx / 40xx / 50xx, **RTX PRO Blackwell**, A100, H100 | `modern` | **0.32.0** (released 2026-10-18), with CUDA 12.8+ wheels, which Blackwell needs; set `VLLM_VERSION` in `.env` to change it |
| 7.x (e.g. GTX 1650) | `legacy` | 0.6.6.post1, the last release with the V0 engine |

It takes about 10 minutes and downloads about 5 GB, once. The profile is saved in `~/vllm-env/.fsdr_profile`, and the serve script reads it.

### 1.3 `.env`

`run.py vllm-serve` passes these settings into WSL2. They are all optional; the defaults of the `modern` profile are shown:

```ini
# VLLM_HF_MODEL=Qwen/Qwen3-8B-AWQ     # 4-bit, about 6 GB; or Qwen/Qwen3-14B-AWQ (better, about 2x slower)
# VLLM_SERVED_NAME=qwen3:8b           # or qwen3:14b
# VLLM_MAX_SEQS=32                    # customers batched together on the GPU
# VLLM_GPU_UTIL=0.85                  # share of the GPU memory vLLM takes (weights + KV cache)
# VLLM_MAX_MODEL_LEN=8192
VLLM_API_KEY=<a long random secret>   # recommended: the server then answers only requests that carry the key
```

The notebook reads `VLLM_API_KEY` from `.env` and never prints it. Never commit `.env`.

---

## 2. At each session

**Terminal 1 — the server** (keep it open):

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run python run.py vllm-serve
```

It prints `vLLM profile modern: serving Qwen/Qwen3-8B-AWQ as qwen3:8b on http://127.0.0.1:8001`.
- **The first time**, it downloads the model from Hugging Face (about 6 GB). This is the only download.
- Wait for `Application startup complete`; that takes 1–3 minutes.
- **After the first start**, set `VLLM_OFFLINE=1` in `.env`. From then on, nothing leaves the machine.

**Terminal 2 — the notebook:**

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab
```

Open `notebooks/04_vllm_agents_step_by_step.ipynb`, then click **Kernel → Restart Kernel and Run All Cells**.
- The first code cell after the engine (G1) must show the served model and `first answer in … s: {'ok': True}`.
- `nvidia-smi -l 1` shows the GPU at work. vLLM takes about 85% of the GPU memory as soon as it starts. That is normal: it is the KV cache.

Stop the server with **Ctrl+C** in terminal 1.

---

## 3. Settings of the notebook

They are the same as in notebook 03, except for these three:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_URL` | `http://127.0.0.1:8001` | the vLLM server; local only |
| `MODEL` | `"auto"` | the model the server serves (`VLLM_SERVED_NAME`) |
| `PARALLEL` | `32` | customers sent at the same time; keep it equal to `VLLM_MAX_SEQS` |

For real data, set `GENERATE = False` and `GENERATE_TARGETS = False`, and use the weekly folder `data/weekly/<D>/` ([GPU_AGENTS_NOTEBOOK.md](GPU_AGENTS_NOTEBOOK.md) §2). `TIME_BUDGET_MIN` and `MAX_LLM_LINES` limit the per-customer agents, as in notebook 03.

---

## 4. How long it takes

These are estimates for the RTX PRO 5000 Blackwell laptop GPU (24 GB) with `PARALLEL = 32`. **Step G2 measures your real numbers** before the long steps, and the last cell gives the measured time per customer.

| Model | Per Target-2 customer | Dummy run (930 Target-2 customers) | Production week (50k T2C + 50k T2F) |
|---|---|---|---|
| `Qwen3-8B-AWQ` | about 0.1–0.2 s | about 2–4 min | about **3–6 h** |
| `Qwen3-14B-AWQ` | about 0.2–0.4 s | about 4–8 min | about **6–12 h** |
| Ollama `qwen3:8b`, for comparison (notebook 03) | about 0.3–0.6 s | about 5–10 min | about 8–14 h |

Agents 1, 2, 2b, 8 and 9 add about 1 minute. Re-running the same week is free: every answer is cached in `data/runs/<D>/agents/llm_cache.jsonl`, separately from Ollama's answers.

---

## 5. Problems

| What you see | Fix |
|---|---|
| G1: `vLLM does not answer at http://127.0.0.1:8001` | Start `uv run python run.py vllm-serve` in another terminal and wait for `Application startup complete` |
| G1: `HTTP Error 401` | `VLLM_API_KEY` in `.env` is not the key the server started with: restart the server after changing `.env` |
| `wsl -e nvidia-smi` shows no GPU | Update the Windows NVIDIA driver, then run `wsl --shutdown` and try again. Never install a driver inside Ubuntu |
| `vllm-install`: `No GPU visible in WSL` | Same as above |
| Server log: `CUDA out of memory` at start | Lower `VLLM_GPU_UTIL` (0.75) or `VLLM_MAX_MODEL_LEN` (4096), or use the 8B model |
| G2: `every test call failed` | The server crashed: read its terminal, restart it, then run G1 and G2 again |
| Server log: `Engine process … died` | Not enough memory in WSL2 (it gets half of the laptop RAM by default). Close other programs, or lower `VLLM_MAX_SEQS` and `PARALLEL` |
| The first start is slow | It downloads the model once, then loads it and compiles the CUDA graphs (1–3 minutes). Later starts are quicker |

A failed LLM call never breaks the week. Those customers keep the cell policy of agent 2b and the rule score, and `DECIDED_BY` / `score_source` say so. The final file is still written, if the rule checks pass.

---

## 6. What was tested

**End to end against a fake vLLM server** (OpenAI API, JSON-schema answers, Bearer key) in the test suite, `tests/test_notebook.py`. The test checks that:
- the key is sent and never printed;
- thinking is switched off for Qwen3;
- the agents' decisions reach Target 2;
- the final file is written.

**For real** on the development laptop (GTX 1650, 4 GB): the `legacy` profile, vLLM 0.6.6 with `Qwen2.5-3B-Instruct-GPTQ-Int4`.
- Old vLLM versions crash on JSON schemas that contain number ranges. The notebook therefore sends schemas without ranges and enforces the ranges in code.
- The `modern` profile (vLLM 0.32.0) cannot run on a GTX 1650, so it is set up for your GPU but was not run here.

**Result of the real run:**
- the whole notebook took 985 s for 3,000 lines, with the per-customer agents limited to 30 customers;
- 140 LLM requests succeeded and 1 failed (that customer fell back to the rules), with no server crash;
- both targets passed the audit, and the final file has 1,580 rows;
- agents 3 and 4 took 14.3 s per customer.

On that old GPU, vLLM 0.6.6 was **slower** than Ollama with `qwen3:1.7b` (3.7 s per customer), for two reasons. Its old V0 engine has to run without CUDA graphs to fit in 4 GB, and the 3B model is larger than the 1.7B one. The speed-up of vLLM comes from the V1 engine of the `modern` profile on a modern GPU, which batches 32 customers at once. Step G2 measures it on your laptop.
