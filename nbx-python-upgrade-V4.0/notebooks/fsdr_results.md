# How to run `fsdr_results.ipynb` on the new laptop

**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB), driver 597.13 (CUDA 13.2), Windows 11.
**What it does:** a **read-only** view of a run. It computes nothing: it reads what notebooks 01–04 (or the terminal commands) wrote and shows it in one place:
- the data and its validation;
- completeness and the cache;
- features and segments;
- the panel and the scoring of each target;
- the final files.

**GPU:** not used.

---

## 1. Before

- The environment is set up ([01_training_step_by_step.md](01_training_step_by_step.md) §1).
- At least one run exists in `data/runs/<D>/`.

## 2. Run

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab
```

Open `notebooks/fsdr_results.ipynb`, set the settings, then click **Kernel → Restart Kernel and Run All Cells**.

| Setting | Value |
|---|---|
| `RUN_DATE` | `None` = the latest run in `data/runs/`, or a date such as `"2026-10-22"` |
| `PREVIEW_ROWS` | rows shown per table (default 10) |

Or, without opening it, an HTML report of one run date:

```powershell
uv run python run.py notebook-html --date 2026-10-22
```

## 3. What you see

- **Runs of notebooks 01 + 02** (two folders, `data/training/<D>` and `data/scoring/<D>`): every section.
- **Runs of notebooks 03 / 04** (one weekly folder, `data/weekly/<D>`): the first cell prints `input : … MISSING`, because it looks for the two-folder layout. The run itself (`data/runs/<D>/`) and its output (`data/output/<D>/`) are found, so the sections on the steps, the targets and the final files still work.

## 4. Time

A few seconds. It only reads files.

## 5. Problems

| What you see | Fix |
|---|---|
| `run : … MISSING` | No run for that date: run notebook 01 (or 03 / 04) first, or set `RUN_DATE` |
| `ModuleNotFoundError: nbx` | This notebook reads the project code (unlike 01–04): start JupyterLab from the project folder with `uv run jupyter lab` |
| Outputs show MSISDNs | Normal, it is your data. Never commit or send a notebook with its outputs: use **Clear All Outputs** before saving |
