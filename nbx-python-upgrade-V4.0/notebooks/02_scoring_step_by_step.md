# How to run `02_scoring_step_by_step.ipynb` on the new laptop

**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB), driver 597.13 (CUDA 13.2), Windows 11.
**What it does:** scoring of one run date D, after `01_training_step_by_step.ipynb`:
- reads the target lists and builds the panel;
- scores **Target 2 only**: group a (T2C, 50,000 lines) and group b (T2F, 50,000 lines). Target 1 is not scored (`FSDR_TARGETS=target2`, the default);
- writes the final file `final/input_flytxt.csv.gz`. No control group by default (`FSDR_CONTROL_SIZE=0`).

**GPU:** not used. The decisions are the rule fallbacks (`DECIDED_BY = fallback_rule_action`). For the LLM agents on the GPU, use notebook 03 (Ollama) or 04 (vLLM).

---

## 1. Before

- The environment is set up, as for notebook 01 ([01_training_step_by_step.md](01_training_step_by_step.md) §1).
- **Notebook 01 has run for the same `RUN_DATE`**, and T0, T2 and T3 are all `done`.

## 2. Open the notebook

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab
```

Open `notebooks/02_scoring_step_by_step.ipynb`.

## 3. Settings (first code cell)

| Setting | **Default = your real run** | A test with dummy lists |
|---|---|---|
| `RUN_DATE` | **`"2026-11-19"`** (the same date as notebook 01) | `"2026-10-22"` |
| `GENERATE_TARGETS` | **`False`** (your lists) | `True` |
| `FORCE` | `True` (scoring is quick) | `True` |

Real lists go in `data/scoring/<D>/`. Each has one column `access_method_val` with the 8-digit MSISDN.

```
target2/a_constraint/target2_a.csv.gz   T2C, group a (CC-v1), 50,000 lines
target2/b_free/target2_b.csv.gz         T2F, group b (CF-v1), 50,000 lines
```

Your lists are recognised by their checksum and are never overwritten by dummy lists, even if `GENERATE_TARGETS` stays `True`.

## 4. Run

Click **Kernel → Restart Kernel and Run All Cells**.

| Step | What you must see |
|---|---|
| 1 | `training OK` |
| 2 (lists) | `lists are DUMMY` or `lists are YOURS`, then the lines kept and dropped (reasons in `targets_rejected.csv`) |
| 3 (P1) | `P1_panel DONE`, with the size of each group |
| 4 (Target 2, P6) | `group audit PASS`, for T2C (a) and T2F (b) |
| 5 (Target 2, final) | `target2 final READY` |
| 6 (final file) | `control: none (FSDR_CONTROL_SIZE=0)`, then `final file READY -- … rows` |
| 7 | every check `PASS`. If the file is blocked, Step 7 prints the reason from `BLOCKED.txt` |
| Proof | `all defined in this notebook: True` |

## 5. Time on this laptop

| Size | Time |
|---|---|
| Dummy (about 2,200 panel lines) | under 1 min |
| Production (50k a + 50k b) | about 2 min |

## 6. Output

`data/output/<D>/`:
- `target2/`: the file of Target 2, `a_constraint/` and `b_free/` per group, `details_target2.parquet`, and the checks;
- **`final/input_flytxt.csv.gz`**: the 38-column file of Target 2.

The engine only saves files; it sends nothing. You check the final file and hand it over yourself.

## 7. Problems

| What you see | Fix |
|---|---|
| Step 1: a step is not `done` | Run notebook 01 first, with the same `RUN_DATE` |
| Step 2: `MISSING` for a group | The list's file name or folder is wrong (§3) |
| Many lines "not in the eligible base" | The MSISDN format differs: a `968` prefix, a lost leading zero, or Excel damage. Export the IDs as 8-digit text |
| `NO FINAL FILE: it is blocked` | A target failed its audit or a check: the reason is printed. Fix it, then run Steps 2–8 again |
