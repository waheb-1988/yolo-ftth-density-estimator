# How to run `04_vllm_agents_step_by_step.ipynb` on the new laptop (vLLM in WSL2)

**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB), driver 597.13 (CUDA 13.2), Windows 11.
**What it does:** the same steps and the same agents as notebook 03 ([03_gpu_agents_step_by_step.md](03_gpu_agents_step_by_step.md)), with **vLLM** as the LLM server.
- vLLM batches 32 customers at once on the GPU, which is faster for the per-customer agents.
- vLLM runs on Linux only, so on Windows it runs in **WSL2**. Ask the security team first: [docs/VLLM_VS_OLLAMA_SECURITY.md](../docs/VLLM_VS_OLLAMA_SECURITY.md).

Full description: [docs/VLLM_AGENTS_NOTEBOOK.md](../docs/VLLM_AGENTS_NOTEBOOK.md).

---

## 1. Once per laptop

1. **The environment:** see [01_training_step_by_step.md](01_training_step_by_step.md) §1.
2. **WSL2 + Ubuntu.** In PowerShell **as administrator**:
   ```powershell
   wsl --install -d Ubuntu
   ```
   Restart the laptop, then create the Ubuntu user when Ubuntu opens.
3. **The GPU in WSL2.** The Windows driver 597.13 provides it; do **not** install an NVIDIA driver inside Ubuntu.
   ```powershell
   wsl -e nvidia-smi          # must list the NVIDIA RTX PRO 5000 and CUDA 13.2
   ```
4. **Install vLLM**, from the project folder, in a normal PowerShell (about 10 min, about 5 GB from PyPI):
   ```powershell
   cd D:\production\nbx-python-upgrade-V4.0
   uv run python run.py vllm-install
   ```
   Check that it prints `GPU compute capability 12.0 -> vLLM profile: modern` and then `vllm 0.32.0 | … | cuda True`.
5. **`.env`:**
   ```ini
   VLLM_API_KEY=<a long random secret>   # the server only answers requests that carry it
   # optional, these are the defaults:
   # VLLM_HF_MODEL=Qwen/Qwen3-8B-AWQ     # or Qwen/Qwen3-14B-AWQ
   # VLLM_SERVED_NAME=qwen3:8b           # or qwen3:14b
   # VLLM_MAX_SEQS=32
   # VLLM_GPU_UTIL=0.85
   ```

## 2. At each session

**Terminal 1 — the server.** Keep it open:

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run python run.py vllm-serve
```

- It must print `vLLM profile modern: serving Qwen/Qwen3-8B-AWQ as qwen3:8b on http://127.0.0.1:8001`.
- **The first time** it downloads the model from Hugging Face (about 6 GB). Once it has, add `VLLM_OFFLINE=1` to `.env`: no download is possible after that.
- Wait for `Application startup complete` (1–3 min).

**Terminal 2 — the notebook:**

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab
```

Open `notebooks/04_vllm_agents_step_by_step.ipynb`.

## 3. Settings (first code cell)

These are the same as in notebook 03, except:

| Setting | Value | Why |
|---|---|---|
| `LLM_URL` | `"http://127.0.0.1:8001"` | the vLLM server, local only |
| `MODEL` | `"auto"` | the model the server serves |
| `PARALLEL` | **`32`** | the same as `VLLM_MAX_SEQS` |

Defaults = your real week: `RUN_DATE = "2026-11-19"`, `GENERATE = False`, `GENERATE_TARGETS = False`, and the weekly folder `data/weekly/2026-11-19/` (create it with `uv run python run.py fsdr-folders --date 2026-11-19`; checklist: [docs/REAL_DATA_2026-11-19.md](../docs/REAL_DATA_2026-11-19.md)).

## 4. Run

Click **Kernel → Restart Kernel and Run All Cells**.

| Step | What you must see |
|---|---|
| G1 | `vLLM : qwen3:8b (context 8192)`, `first answer in … s: {'ok': True}` |
| G2 | `0 failed`, the seconds per call, and the estimate for a production week |
| 1 → 11 | the same as notebook 03 ([03_gpu_agents_step_by_step.md](03_gpu_agents_step_by_step.md) §4) |

`nvidia-smi -l 1`: vLLM takes about 85% of the 24 GB as soon as it starts (its KV cache); that is normal. `GPU-Util` goes up while the agents run.

Stop the server with **Ctrl+C** in terminal 1 when you have finished.

## 5. Time on this laptop (estimates; G2 measures them)

| Model, `PARALLEL = 32` | Dummy run (930 Target-2 customers) | Production week (50k a + 50k b = 100k customers) |
|---|---|---|
| `Qwen3-8B-AWQ` | about 2–4 min | about 3–6 h |
| `Qwen3-14B-AWQ` | about 4–8 min | about 6–12 h |

The first start of the server takes 1–3 minutes: it loads the model and compiles the CUDA graphs.

## 6. Problems

| What you see | Fix |
|---|---|
| `vllm-install`: `No GPU visible in WSL` | Run `wsl --shutdown`, then try again; update the Windows NVIDIA driver if needed |
| `vllm-install` says `legacy` | `wsl -e nvidia-smi --query-gpu=compute_cap --format=csv` must give 12.0; if not, WSL does not see the right GPU |
| G1: `vLLM does not answer` | The server is not started, or not ready yet: wait for `Application startup complete` |
| G1: `HTTP Error 401` | `VLLM_API_KEY` changed: restart the server |
| Server: `CUDA out of memory` | Set `VLLM_GPU_UTIL=0.75` or `VLLM_MAX_MODEL_LEN=4096`, or use the 8B model |
| Server: `Engine process … died` | WSL2 ran out of RAM: close programs, or set `VLLM_MAX_SEQS=16` and `PARALLEL = 16` |
| WSL2 not allowed on this laptop | Use notebook 03 (Ollama, native Windows): the same agents, only slower |
