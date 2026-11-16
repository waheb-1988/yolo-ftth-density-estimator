# How to run `01_training_step_by_step.ipynb` on the new laptop

**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB), driver 597.13 (CUDA 13.2), Windows 11.
**What it does:** training of one run date D. It validates the data, builds the cache, computes the features of the whole base (T2) and assigns the segments and the two actions of every line (T3).
**GPU:** not used. Every step is pandas / Parquet on the CPU, and that is normal. No LLM, no Ollama, no vLLM.
**Next:** `02_scoring_step_by_step.ipynb`, with the same run date.

---

## 1. Once per laptop

1. Copy the folder `nbx-python-upgrade-V4.0` to `D:\production\`, **without** `.venv\`.
2. Create the environment ([docs/UV_ENV_SETUP.md](../docs/UV_ENV_SETUP.md)):
   ```powershell
   pip install uv
   cd D:\production\nbx-python-upgrade-V4.0
   uv python install 3.12
   uv sync --locked
   copy .env.example .env          # then set FSDR_PII_SALT to a long random secret
   uv run python run.py check      # every line [ok]
   ```

## 2. Open the notebook

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab
```

Open `notebooks/01_training_step_by_step.ipynb`. The kernel at the top right must be **Python 3 (ipykernel)**.

## 3. Settings (first code cell)

| Setting | **Default = your real run** | A test with dummy data |
|---|---|---|
| `RUN_DATE` | **`"2026-11-19"`** (next week: `"2026-11-26"`) | `"2026-10-22"` |
| `GENERATE` | **`False`** (your exports) | `True` (real exports are never overwritten anyway) |
| `N_LINES` | (not used) | `3000` |
| `WORKERS` | `1` (fixed: the notebook is standalone) | `1` |
| `FORCE` | `False` | `False` (`True` re-runs every step) |

For real data, create the folders with `uv run python run.py fsdr-folders --date 2026-11-19 --layout split`. It prints the exact file of every table. The exports go in `data/training/<D>/` and `data/scoring/<D>/`, each with the 5 sub-folders usage, models, subscription, catalog and others, as **`.csv.gz`** files. If the folder is empty, the notebook stops and says so. See [docs/REAL_DATA_2026-11-19.md](../docs/REAL_DATA_2026-11-19.md).

## 4. Run

Click **Kernel → Restart Kernel and Run All Cells**, or run the cells one by one with Shift+Enter.

| Cell or step | What you must see |
|---|---|
| Your machine | `kernel  : OK, the project's .venv`; `GPU 0 : NVIDIA RTX PRO 5000 …` (shown for information; this notebook does not use it) |
| Part A (the engine cells) | `engine loader ready`, then one cell per module, a few seconds in total |
| Step 0 | `47 tables generated for 3,000 lines` (dummy data only) |
| Step 1 | `misfiled: none`, `duplicates: none` |
| Step 2 (T0, Agent 1) | `T0_load DONE -- pipeline train_week0` (`train_weekly` from the second week on) |
| Step 4 (T2) | `T2_features DONE: … eligible lines` |
| Step 5 (T3, Agents 2 / 2b) | `T3_segments DONE` and the segment chart |
| Step 6 | T0, T2 and T3 all `done` |
| Proof | `all defined in this notebook: True`, `vowpalwabbit loaded: False` |

## 5. Time on this laptop

| Size | Time |
|---|---|
| Dummy, 3,000 lines | about 30 s (T0 about 10 s, T2 about 4 s, T3 about 1 s) |
| Production, 1M lines | T0 about 14 min, then T2 + T3 several minutes with one worker |

For the production size, the terminal is faster because it uses several workers: `uv run python run.py fsdr-load --date D`, then `fsdr-features --date D`. Then open the notebook to look at the results; steps that are already done are skipped.

## 6. Output

`data/runs/<D>/train/T0_load/`, `T2_features/` and `T3_segments/`, plus the cache `data/cache/<D>/`. Never send `data/` anywhere: it holds MSISDNs.

## 7. Problems

| What you see | Fix |
|---|---|
| `kernel : NOT the project's .venv` | Choose the `.venv` kernel, or start JupyterLab with `uv run jupyter lab` from the project folder |
| The kernel dies, "access denied", `msvcp140.dll` | Delete `.venv\Lib\site-packages\00_vw_preload.pth` if it exists, rebuild `.venv` ([docs/UV_ENV_SETUP.md](../docs/UV_ENV_SETUP.md) §9), and restart the kernel |
| `T0_load BLOCKED` | Read `data/runs/<D>/train/T0_load/validation_report.md`; a table is missing or its file name is wrong |
| `ModuleNotFoundError: nbx…` in a step | Part A was not run: run it, then the step again |
