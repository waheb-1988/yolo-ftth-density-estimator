# FSDR V4.0 — set up a new laptop and run everything

This guide starts from a laptop with nothing installed and ends with:

- the two targets scored;
- the file to push;
- the notebooks and the web interface running.

Follow the sections in order. Every command is run **from the project folder** `nbx-python-upgrade-V4.0/`.
Commands work in Windows PowerShell, macOS Terminal and Linux shells unless a line says otherwise.

| § | Step | Time |
|---|---|---|
| 1 | Prerequisites | 5 min |
| 2 | Install uv (`pip install uv`) | 2 min |
| 3 | Get the code | 2 min |
| 4 | Create the environment (uv, or pip + the 3 requirements files) | 3–5 min |
| 5 | The data folders | read |
| 6 | Configure `.env` | 2 min |
| 7 | First run on dummy data, step by step (terminal) | 2 min |
| 8 | The notebooks: training and scoring step by step | 5 min |
| 9 | Run on real data | — |
| 10 | The web interface | 2 min |
| 11 | Tests | 8 min |
| 12 | Optional: a local LLM (Ollama, llama.cpp, LM Studio, vLLM), LangSmith | — |
| 12a | Everything stays on this machine (local only) | read |
| 13 | Troubleshooting | — |
| 14 | Command card | — |

---

## 1. Prerequisites

| Item | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, macOS 13+, Ubuntu 22.04+ | Windows: PowerShell; WSL2 only for the optional GPU LLM |
| Python | **3.12** | `vowpalwabbit` and `implicit` ship wheels up to CPython 3.12: do not use 3.13. uv installs 3.12 for you (§4). |
| RAM | 16 GB (dummy / 100k) · 32 GB for the 1M base | |
| Disk | 2 GB for the code and environment, + about 3 GB per 1M-line run date | |
| Git | any recent version | or copy the folder from a zip |
| Editor | VS Code with the *Python* and *Jupyter* extensions (recommended) | JupyterLab works too (§8) |

You need a Python only to run `pip install uv`. If you have none, use the uv standalone installer in §2.

---

## 2. Install uv

[uv](https://docs.astral.sh/uv/) creates the virtual environment, installs the exact locked versions, and installs Python 3.12 itself.

**With pip** (any Python 3.8+ already on the machine):

```bash
pip install uv
uv --version            # e.g. uv 0.13.x
```

If `uv` is "not recognized" after `pip install uv`, pip installed it in a user folder that is not on your PATH. Either use `python -m uv ...` everywhere, or add the folder pip printed in its warning (Windows: `%APPDATA%\Python\Python3xx\Scripts`) to the PATH, then open a new terminal.

**Without Python**, use the standalone installer:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# or: winget install astral-sh.uv
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh        # or: brew install uv
```

Open a **new** terminal after installing, then check with `uv --version`.

---

## 3. Get the code

```bash
git clone <your repository URL> llm_recommendation_engin_telco
cd llm_recommendation_engin_telco
git checkout <the branch with V4.0>
cd nbx-python-upgrade-V4.0
```

Or unzip the folder `nbx-python-upgrade-V4.0/` anywhere and `cd` into it.

- **Windows:** avoid folders synced by OneDrive and very long paths; `C:\work\fsdr\` is a good place.
- Never copy `.venv/` from another machine: create it again (§4).

What is in the folder:

```
nbx-python-upgrade-V4.0/
├── pyproject.toml, uv.lock, .python-version   the uv project: dependencies + exact lock, Python 3.12
├── requirements.txt                runtime + tracing (pip alternative to uv sync)
├── requirements-dev.txt            + tests (pytest)
├── requirements-notebook.txt       + Jupyter, nbconvert, matplotlib
├── run.py                          the launcher: every command below
├── .env.example                    every setting, documented (copy to .env)
├── nbx/                            the engine (fsdr/), API (api/), web UI (ui/), LLM backends (llms/)
├── notebooks/                      01_training_step_by_step, 02_scoring_step_by_step, fsdr_results (+ their builders)
├── docs/                           this guide, interface guide, terminal guide, plan, rules, references
├── tests/                          the test suite
├── vllm/                           optional: scripts to serve a local LLM on a GPU (WSL2 / Linux)
└── Dockerfile                      optional: container of the API + UI
```

---

## 4. Create the environment

### Option A — uv (recommended: the exact locked versions)

```bash
uv python install 3.12        # once per machine; skipped if a 3.12 is already there
uv sync                       # creates .venv/ from uv.lock: runtime + dev + tracing + notebook groups
uv run python run.py check    # every package [ok], the data folders, the LLM backend
```

`uv sync` creates `.venv/` inside the project and installs exactly the versions in `uv.lock`. `uv run <command>` runs inside `.venv` without activating it, so every command in this guide starts with `uv run`.

Variants:

| Need | Command |
|---|---|
| Runtime only (server, no tests, no notebooks) | `uv sync --no-default-groups --group tracing` |
| Runtime + tests | `uv sync --no-default-groups --group dev --group tracing` |
| Exactly the lock, fail if `pyproject.toml` changed | `uv sync --locked` |
| After editing a dependency in `pyproject.toml` | `uv lock` then `uv sync` |

If you prefer an activated shell:

```bash
# Windows PowerShell                      # macOS / Linux
.venv\Scripts\Activate.ps1                source .venv/bin/activate
python run.py check                       python run.py check
```

(PowerShell may refuse the script once: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

### Option B — pip with the 3 requirements files

The three files are generated from `uv.lock`, so they hold the same pinned versions:

| File | Installs |
|---|---|
| `requirements.txt` | the runtime: FastAPI, Gradio, pandas, pyarrow, numpy, scipy, Vowpal Wabbit, implicit, … + LangSmith tracing |
| `requirements-dev.txt` | `-r requirements.txt` + pytest, pytest-asyncio, jsonschema |
| `requirements-notebook.txt` | `-r requirements.txt` + ipykernel, JupyterLab, nbconvert, matplotlib |

With uv creating the venv (it gets Python 3.12 for you):

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-notebook.txt
uv run python run.py check
```

Or with plain Python 3.12 and pip:

```bash
py -3.12 -m venv .venv            # Windows (python3.12 -m venv .venv on macOS / Linux)
.venv\Scripts\activate            # Windows (source .venv/bin/activate on macOS / Linux)
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt -r requirements-notebook.txt
python run.py check
```

With Option B, write `python run.py …` (venv activated) wherever this guide writes `uv run python run.py …`.

To regenerate the three files after a re-lock:

```bash
uv export --no-hashes --no-emit-project --no-default-groups --group tracing > requirements.txt
uv export --no-hashes --no-emit-project --only-group dev                     # -> requirements-dev.txt (+ "-r requirements.txt" first)
uv export --no-hashes --no-emit-project --only-group notebook                # -> requirements-notebook.txt (same)
```

`run.py check` ends with **All checks passed.** when the environment is complete. The data folders show `[todo]` until §7.

> V4.0 has no market-intelligence (competitor) module any more: no web scraping, no `beautifulsoup4` in the
> runtime. `requirements-notebook.txt` still lists `beautifulsoup4` because `nbconvert` needs it.

---

## 5. The data folders

Everything lives under `data/`, which is never committed (`.gitignore`). A run date *D* is a `YYYY-MM-DD` folder name. Production runs on Thursdays.

```
data/
├── training/<D>/                  INPUT for training (T0 - T3): the base, the history, the model scores
│   ├── usage/                     subs_amt_auto, activities_w1..w4_auto, activities_m1/auto_m2/m3_auto, paygo_activity_auto
│   ├── models/                    the 11 model-score tables (churn, appetency, fragility, reactivation, …)
│   ├── subscription/              base_prepaid, consumer_aact_account, access_method_id_auto, subs_btl_auto
│   ├── catalog/                   btl_catalog (catalogue of the history)
│   └── others/                    party_auto, customer_profile, location_auto, DEVICEBEHAVIORSEGMENTATION.dv_segment,
│                                  targeting_dataiku_auto
├── scoring/<D>/                   INPUT for scoring (P1 - P7): this week's tables + the target lists
│   ├── usage/                     usage_data_voice_payg_tg, *_score tables (weekly runs)
│   ├── models/                    (same 5 folders as training; empty when unused)
│   ├── subscription/              subscriptions_tg, sub_btl_3months
│   ├── catalog/                   btl_catalog, BTL_script_english_arabic_ok, conflicting_offers_h
│   ├── others/                    targeting_auto, consent_contact, party_scoring, sample_cg_tg_* (optional)
│   ├── target1/
│   │   ├── a_constraint/target1_a.csv.gz     group T1C: production code 3 (constraint universe)
│   │   └── b_free/target1_b.csv.gz           group T1F: production code 4 (free universe)
│   ├── target2/
│   │   ├── a_constraint/target2_a.csv.gz     group T2C: challenger CC-v1
│   │   └── b_free/target2_b.csv.gz           group T2F: challenger CF-v1
│   └── control/control.csv.gz                never contacted
├── output/<D>/                    RESULTS, one folder per target, each scored separately
│   ├── target1/
│   │   ├── input_flytxt_target1.csv.gz       Target 1's file (written only when its checks PASS)
│   │   ├── a_constraint/input_flytxt_t1c.csv.gz   details_t1c.csv.gz
│   │   ├── b_free/input_flytxt_t1f.csv.gz         details_t1f.csv.gz
│   │   ├── details_target1.parquet           every line, every column (≈ 240): the big file to validate
│   │   └── final_rules_report.csv, output_checks.json, audit_groups.json, summary.json
│   ├── target2/                   the same for T2C / T2F
│   ├── control/control_<D>.csv.gz the control lines and their strata
│   └── push/input_flytxt.csv.gz   THE FILE TO PUSH: both targets merged (only when both PASS; else BLOCKED.txt)
├── runs/<D>/train/ score/         every step's folder: _step.json (status, rows, time), reports, Parquet outputs
└── cache/<D>/                     Parquet cache of every table, 16 partitions (built by T0)
```

**Rules of the folders**

- **One file per table**, named after the table: `<table>.csv.gz`.
  - Format: the FSDR CSV standard — gzip, UTF-8 without BOM, `,`, IDs as text, empty field for missing.
  - `.csv`, `.json`, `.jsonl` and `.parquet` are accepted only for generated data, or with `FSDR_ALLOW_TEST_FORMATS=true`.
- T0 reads `training/<D>/` and `scoring/<D>/` **as one run**. A table filed on the wrong side still loads, with a warning listed in `misfiled_tables.json`.
  - A table found twice on the same side **blocks** the run: nobody can tell which file is right.
  - `btl_catalog` may sit on both sides; the scoring copy wins.
- **Target lists**: one row per line, with at least `access_method_val` (or `msisdn`). The other columns are kept and carried to the details.
  - Without lists, the panel is drawn at random (stratified).
- Training and scoring use the **same run date** *D*.
- **Each target is scored on its own**: it has its own P6 step, its own checks and its own folder. The push file waits for both targets to pass.

---

## 6. Configure `.env`

```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

Every value has a default, so `.env` is optional. The settings that matter on a new laptop:

| Setting | Value on a laptop | Why |
|---|---|---|
| `LLM_BACKEND` | `offline` | Pure Python, deterministic agents, no GPU or model download. `ollama`, `llamacpp`, `lmstudio`, `vllm`: a local LLM server (§12). |
| `DEFAULT_LLM` | `qwen2.5:3b` | The model name the server answers to; the model catalogue is in `.env.example`. |
| `FSDR_LOCAL_ONLY` | `true` | Refuses any LLM server or tracing endpoint that is not on this machine (§12a). |
| `FSDR_RECO_OFFERS` | `6` | Up to 6 offers per line: 3 monthly + 3 weekly. `4` = the V3.0 file. |
| `FSDR_T1_CONSTRAINT_SIZE` … `FSDR_CONTROL_SIZE` | 25000 ×4, 20000 | The group sizes the lists are checked against (±2 %). |
| `FSDR_TRAINING_DIR`, `FSDR_SCORING_DIR`, `FSDR_OUTPUT_DIR` | unset | Defaults `data/training`, `data/scoring`, `data/output`; set them to put the data on another disk. |
| `FSDR_FEATURE_WORKERS` | `2` (laptop) / `8` (big machine) | Partitions processed in parallel in T2 / T3. |
| `FSDR_API_KEY` | empty on a laptop | Required as soon as the API listens on the network (`--host 0.0.0.0`). |
| `FSDR_PII_SALT` | a long random string | Salt of the customer pseudonyms in LLM prompts; keep it private. |

**Never commit `.env`**: it can hold keys. It is in `.gitignore`.

---

## 7. First run on dummy data, step by step (terminal)

This runs the whole chain on 3,000 generated lines in about 1 minute. Pick a run date:

```bash
uv run python run.py fsdr-generate --date 2026-10-22 --n 3000
```
**What it does:** writes the 47 tables as `.csv.gz` into `data/training/2026-10-22/` and `data/scoring/2026-10-22/`, with the 5 folders on each side. `--mode full` writes the 1M production size (about 16 minutes).

```bash
uv run python run.py fsdr-load --date 2026-10-22
```
**Step T0 (training):**
- validates both folders against the data contract and checks completeness;
- builds the Parquet cache (16 partitions);
- prints `Read as one run: …training… + …scoring…`.

It must end with `T0_load DONE`. A blocking issue prints `BLOCKED` with the reason; see `data/runs/<D>/train/T0_load/validation_report.md`.

```bash
uv run python run.py fsdr-features --date 2026-10-22
```
**Steps T2 + T3 (training):**
- the features of the whole base (B-01 … B-12);
- the S-rule segments, bands and micro-segments;
- the week-0 history;
- the two actions of every line: `action_original` (production dictionary) and `action_agent` (agent dictionary).

```bash
uv run python run.py fsdr-targets --generate --date 2026-10-22
```
**Dummy target lists.** These come from a stratified draw of the eligible base and are written to `data/scoring/<D>/target1|target2/a_constraint|b_free/` and `control/`. Without `--generate`, the command checks the lists you put there and prints what is kept, dropped and blocking.

Then score. Pick one of the two ways below.

**All in one command:**

```bash
uv run python run.py fsdr-run-targets --date 2026-10-22
```

**Or one target at a time** (same result):

```bash
uv run python run.py fsdr-score-target --target target1 --panel --date 2026-10-22   # P1 panel, then Target 1
uv run python run.py fsdr-score-target --target target2 --date 2026-10-22           # Target 2 on the same panel
uv run python run.py fsdr-push --date 2026-10-22                                   # merge both -> push/
```

The output looks like this:

```
P1_panel DONE -- mode uploaded, sizes {'t1c': 465, 't1f': 465, 't2c': 465, 't2f': 465, 'control': 372}
target1 final DONE -- READY; input_flytxt_target1 762 rows; 6 offers per line max; details 930 lines x 242 columns
  checks: A-20 PASS, A-21 PASS, A-21 PASS, A-22 PASS, A-23 PASS, A-24 PASS | group audit PASS
  t1c  a_constraint production_constraint  lines 465  in file 391  offers/line 4.96  F-04 dropped 74 ...
  t1f  b_free       production_free        lines 465  in file 371  offers/line 4.79  F-04 dropped 94 ...
target2 final DONE -- READY; ...
push READY -- 1,562 rows
```

**What each scoring step does:**

| Step | Target 1 (T1C a / T1F b) | Target 2 (T2C a / T2F b) |
|---|---|---|
| P1 panel | the lists (or a stratified draw), joined to the T3 segments; sizes A-30, balance A-31 | (shared) |
| Action | `action_original` | `action_agent` |
| Candidates | a: the action's families (B-21); b: L2 DATA / COMBO, no VOICE | the same universes |
| Price rule | 3 passes around the ARPU anchor (B-28 … B-30 / FR-09 … FR-11) | relative band 1.0×–1.5× the challenger anchor, widened once to 2.0× (CC / CF-02 … 04) |
| 6 offers | EXT-6: a 3rd offer per validity, above the anchor, not the same service and price | the same |
| Final (code 5) | slots 1–2 monthly, 3–4 weekly, 5 = 3rd monthly, 6 = 3rd weekly; favourites / forced fill an empty 5 / 6; F-09 / F-11 / F-12 clearing; scripts; checks A-20 … A-24 | the same |
| Writes | `data/output/<D>/target1/` | `data/output/<D>/target2/` |

**The push.** `data/output/<D>/push/input_flytxt.csv.gz` is written only when both targets passed and no MSISDN is in two targets. Otherwise you get `push/BLOCKED.txt` with the reasons. A **human approval** (H-08) is still required before the file goes to the campaign platform.

`uv run python run.py check` now shows `[ok]` for `data/training/`, `data/scoring/` and `data/output/`.

---

## 8. The notebooks: training and scoring, step by step

There are three notebooks in `notebooks/`, stored without outputs:

| Notebook | Use it to |
|---|---|
| `01_training_step_by_step.ipynb` | run T0 → T2 → T3 one cell at a time. Each step has an explanation of its rules and shows what it wrote: the validation issues, completeness, the cache, feature statistics, segment counts, the action cross-table, the priors. |
| `02_scoring_step_by_step.ipynb` | run the target lists → P1 → Target 1 (P6, final) → Target 2 (P6, final) → control + push → the final validation of the file to push. Each step shows its audit, its rules (`selected_by`), the offers per line, what fills each of the 6 slots, and one line end to end. |
| `fsdr_results.ipynb` | read-only view of a run (computes nothing). |

**In VS Code:**

1. Open the folder `nbx-python-upgrade-V4.0/` in VS Code.
2. Open `notebooks/01_training_step_by_step.ipynb`.
3. Click **Select Kernel** → *Python Environments* → `.venv (Python 3.12)`. It is the project's `.venv`; with Option B, pick your venv.
4. In the first code cell, set `RUN_DATE`, `GENERATE` (True = dummy data), `N_LINES` and `WORKERS`.
5. Run the cells one by one with **Shift+Enter**, reading the explanation above each step. Then open `02_scoring_step_by_step.ipynb` with the same `RUN_DATE`.

Or from the terminal:

```bash
uv run python run.py notebook --name training        # opens it in VS Code
uv run python run.py notebook --name scoring --lab   # JupyterLab in the browser
uv run jupyter nbconvert --to html --execute notebooks/02_scoring_step_by_step.ipynb --output-dir data/runs  # headless
```

Every step is **resumable**: a step already done with the same inputs is skipped, and `FORCE = True` in the settings cell re-runs it. The notebooks call the same functions as the terminal commands of §7, so the files they write are the same.

**Do not commit a notebook with its outputs**: outputs hold MSISDNs. To change a notebook, edit its builder (`notebooks/build_step_notebooks.py` or `build_results_notebook.py`), then rebuild it:

```bash
uv run python notebooks/build_step_notebooks.py
```

If you ran a notebook in place, clear its outputs before committing: in VS Code, **Clear All Outputs**, then save. `tests/test_notebook.py` fails when a stored notebook has outputs or differs from its builder.

---

## 9. Run on real data

1. Create the two folders of the run date. For example, for Thursday 2026-11-12:
   ```
   data/training/2026-11-12/{usage,models,subscription,catalog,others}/
   data/scoring/2026-11-12/{usage,models,subscription,catalog,others}/
   ```
2. Copy each export, as `<table>.csv.gz`, into its folder: the lists in §5 (see `config.TABLE_FOLDERS` / `config.SCORING_TABLES`). The file name must be the exact table name.
3. Put the four target lists and the control in `data/scoring/<D>/target1|target2/a_constraint|b_free/` and `control/`. You can also upload them from the web interface (§10, step 2a).
4. Run §7 **without** `fsdr-generate` and without `--generate`. Or use the notebooks with `GENERATE = False` and `GENERATE_TARGETS = False`.
5. Read the checks before anything else:
   - `data/runs/<D>/train/T0_load/validation_report.md` (data);
   - `fsdr-targets --date <D>` (lists: blocking / dropped / warnings);
   - `data/output/<D>/<target>/output_checks.json` and `audit_groups.json`.
6. Validate the details with the business (`details_<target>.parquet`, or `details_<group>.csv.gz` per group). Then send `data/output/<D>/push/input_flytxt.csv.gz` after **human approval**.

The generator refuses to overwrite a folder that holds non-generated data unless you pass `--force`. Real exports cannot be replaced by dummy data by mistake.

At the production size (1M base, 100k contacted + 20k control), on a 32 GB laptop with `FSDR_FEATURE_WORKERS=2`:

| Step | Time |
|---|---|
| T0 | about 14 min |
| T2 + T3 | about 1–2 min |
| Each target | about 1 min |

---

## 10. The web interface

```bash
uv run python run.py start          # API (port 8000) + web UI (port 7860); Ctrl+C stops both
```

Open <http://localhost:7860> and go to the tab **FSDR Targets & Output**. Then:

1. **Data and training.** Set the run date, then:
   - **1a** generates dummy data, or skip it with real data;
   - **1b** runs the training (T0 + T2 + T3).
2. **Target lists.** Pick a group, choose its `.csv.gz` file, then **2a · Upload this list**. Do this once per group and once for the control, or click **2b** for dummy lists. **2c** checks them.
3. **Scoring, one target at a time:**
   - **3a** builds the panel;
   - **3b** scores Target 1;
   - **3c** scores Target 2;
   - **3d** builds the push file.

   **3a → 3d in one go** does all four steps.
4. **Download.** The file to push, each target's and group's files, the details and the rules reports appear in the download box.

API equivalent (local docs with a *Try it* form at <http://localhost:8000/docs>):

| Method and path | Does |
|---|---|
| `POST /fsdr/v4/<D>/generate` | dummy training + scoring data |
| `POST /fsdr/v4/<D>/prepare` | training T0 + T2 + T3 |
| `POST /fsdr/targets/<D>/<t1c\|t1f\|t2c\|t2f\|control>` | upload one list |
| `GET /fsdr/targets/<D>` | the lists and their check |
| `POST /fsdr/v4/<D>/targets/generate` | dummy lists |
| `POST /fsdr/v4/<D>/panel` | P1 |
| `POST /fsdr/v4/<D>/score/<target1\|target2>` | one target: P6 + final |
| `POST /fsdr/v4/<D>/push` | control file + push |
| `POST /fsdr/v4/<D>/run` | everything after training |
| `GET /fsdr/v4/<D>/output` | the summaries and the files of `data/output/<D>/` |

Every other tab and button is described in [USER_GUIDE_INTERFACE.md](USER_GUIDE_INTERFACE.md).

---

## 11. Tests

```bash
uv run python run.py test -q                       # the whole suite, about 8 minutes on a laptop
uv run python run.py test -q tests/fsdr/test_v4_split_targets.py    # V4.0 only, about 1 minute
```

All tests must pass. With Option B, install `requirements-dev.txt` first. The tests use temporary folders only: they never touch `data/`.

---

## 12. Optional

**A local LLM.** The agents run offline by default: deterministic rules, no model. To let a real model decide,
run one of these servers **on this laptop** and set two lines in `.env`. All four speak the same OpenAI-compatible
API, so the app works with any of them.

| Backend | Install / start | `.env` | Good for |
|---|---|---|---|
| **Ollama** (easiest) | install from ollama.com; `ollama pull qwen2.5:7b` | `LLM_BACKEND=ollama` `DEFAULT_LLM=qwen2.5:7b` | Windows / macOS / Linux, CPU or GPU |
| **llama.cpp** | download `llama-server` and a `.gguf` file; `llama-server -m model.gguf --host 127.0.0.1 --port 8080` | `LLM_BACKEND=llamacpp` | CPU-only laptops, smallest footprint |
| **LM Studio** | install, download a model, Developer tab → *Start server* (port 1234) | `LLM_BACKEND=lmstudio` `DEFAULT_LLM=<model id>` | a GUI to try models |
| **vLLM** | [GPU_SETUP.md](GPU_SETUP.md): `run.py vllm-install` once, then `run.py vllm-serve` (WSL2 / Linux) | `LLM_BACKEND=vllm` `VLLM_API_KEY=…` | NVIDIA GPU, fastest batch throughput |

Then check:
- `uv run python run.py check` shows the backend, its URL and `[ok] no prompt, trace or data leaves this machine`;
- <http://localhost:8000/health> says `ok` and lists the served models.

`.env.example` has a **model catalogue** with the name of each model per backend and the memory it needs:
- Qwen2.5 3B / 7B / 14B, Qwen3 8B;
- Llama 3.1 8B, Llama 3.2 3B;
- Mistral 7B, Mistral Nemo 12B;
- Gemma 2 9B, Phi-3.5 mini, DeepSeek-R1 distill 7B.

The agents need JSON output, so prefer the instruct models. If the server answers with another model than `DEFAULT_LLM`, the app uses the model it serves. If the server is down, every agent takes its rule-based path; nothing fails.

The model weights are downloaded **once**, when you install the model: `ollama pull`, the `.gguf` file, LM Studio's download, or vLLM's first serve. After that, the app and the server work without internet. For vLLM, set `VLLM_OFFLINE=1` to forbid any further download.

**LangSmith tracing.** Off by default. With `FSDR_LOCAL_ONLY=true` it only runs against a LangSmith **on this machine** (`LANGSMITH_ENDPOINT=http://127.0.0.1:…`). The LangSmith cloud is refused and tracing stays off.

**Docker.** `docker build -t fsdr .` then `docker run -p 8000:8000 -p 7860:7860 fsdr` (offline backend).

---

## 12a. Everything stays on this machine (local only)

V4.0 makes **no call to any cloud service**:
- no OpenAI, no Anthropic / Claude, no Hugging Face inference;
- no analytics;
- no CDN.

`FSDR_LOCAL_ONLY=true`, the default, enforces it:

| What | How it stays local |
|---|---|
| LLM calls | Only the URL of the chosen local server (`VLLM_URL`, `OLLAMA_URL`, `LLAMACPP_URL`, `LMSTUDIO_URL`), and only if it is on this machine: 127.0.0.1, localhost, ::1, or a host you list in `FSDR_LOCAL_HOSTS`, such as your own GPU box on the LAN. Any other URL is refused, and the agents use their rules. |
| Cloud LLM SDKs | None installed, none imported (a test checks the code and `pyproject.toml`). |
| Usage analytics | Gradio, Hugging Face and vLLM analytics are switched off at start (`GRADIO_ANALYTICS_ENABLED=False`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`, `VLLM_NO_USAGE_STATS=1`). |
| Web UI | Gradio's references to Google Fonts and to a cdnjs script are removed from the page; the fonts are local. |
| API docs | `/docs` is a local page with a *Try it* form. Swagger UI would load from a CDN; `FSDR_SWAGGER_CDN=true` brings it back. |
| Tracing | Only a self-hosted LangSmith on this machine; the cloud endpoint is refused. |
| Data | Everything under `data/` stays on disk. The push file is handed over by you, after approval. |

`uv run python run.py check` prints the local-only state, and every setting that would leave the machine if you changed it.

**What still needs internet, once:** installing (`pip install uv`, `uv sync` or `pip install -r …`, Python 3.12) and downloading the model weights. After that you can unplug the network.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `uv : The term 'uv' is not recognized` | Open a new terminal; or use `python -m uv`; or add pip's Scripts folder to the PATH (§2) |
| `No interpreter found for Python >=3.10, <3.13` | `uv python install 3.12`, then `uv sync` |
| pip tries to compile `vowpalwabbit` / `implicit` | You are on Python 3.13+: create the venv with 3.12 (`uv venv --python 3.12`) |
| `.venv` is locked / access denied (Windows) | Close VS Code terminals and Jupyter kernels that use it, then retry `uv sync` |
| `run.py check` shows `[FAIL]` for a package | `uv sync` (Option A) or `pip install -r requirements.txt` (Option B) in the right venv |
| `T0_load BLOCKED` | Read `data/runs/<D>/train/T0_load/validation_report.md`: missing table, wrong format (`.csv` instead of `.csv.gz`), a table in two files |
| `misfiled tables` warning | A table is on the wrong side (training / scoring); it still loads. Move it to silence the warning (§5) |
| `P1_panel BLOCKED` | The lists: a line in two groups, or a missing list. Run `fsdr-targets --date <D>` for the details |
| `BLOCKED: P6_offers_target2 ... is missing` | Score the target first: `fsdr-score-target --target target2` |
| `push BLOCKED` | One target failed a check or is not scored yet: read `data/output/<D>/push/BLOCKED.txt` and `<target>/BLOCKED.txt` |
| Port 8000 / 7860 in use | Stop the other app, or set `FASTAPI_PORT` / `GRADIO_PORT` in `.env` |
| Notebook: `ModuleNotFoundError: nbx` | The kernel is not the project's `.venv`: select it (§8). The first cell finds the project root itself |
| Out of memory at 1M lines | `FSDR_FEATURE_WORKERS=1`, close other apps, use 32 GB |
| Health `degraded` with a local LLM | The server is not running, or `DEFAULT_LLM` is not served: `ollama list` / the LM Studio model id / `VLLM_SERVED_NAME` |
| `FSDR_LOCAL_ONLY: … is not on this machine; refused` | The LLM URL points elsewhere: use 127.0.0.1, or add your own host to `FSDR_LOCAL_HOSTS` |

---

## 14. Command card

```bash
# once per laptop
pip install uv
uv python install 3.12
uv sync
copy .env.example .env               # cp on macOS / Linux
uv run python run.py check

# one run date D
uv run python run.py fsdr-generate --date D --n 3000          # dummy data only
uv run python run.py fsdr-load --date D                       # T0      (training)
uv run python run.py fsdr-features --date D                   # T2 + T3 (training)
uv run python run.py fsdr-targets --generate --date D         # dummy lists only (else: check yours)
uv run python run.py fsdr-score-target --target target1 --panel --date D
uv run python run.py fsdr-score-target --target target2 --date D
uv run python run.py fsdr-push --date D                       # -> data/output/D/push/input_flytxt.csv.gz
#   (or the last 3 in one: uv run python run.py fsdr-run-targets --date D)

# notebooks, interface, tests
uv run python run.py notebook --name training
uv run python run.py notebook --name scoring
uv run python run.py start                                    # http://localhost:7860
uv run python run.py test -q
```
