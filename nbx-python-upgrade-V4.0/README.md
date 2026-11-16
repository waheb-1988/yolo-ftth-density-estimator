# FSDR Recommender V4.0 — NBA / NBO with LLM agents on Target 2, 6 offers

For each prepaid line of the operator FSDR, the engine:

- decides **the next best action** (retain, upsell data, cross-sell voice, …);
- picks **the next best offers** for that action;
- writes the campaign file `input_flytxt` (38 columns).

The whole base (about 1.2M lines) is **trained** once per run date: validation, features, segments and actions, by rules on the CPU. Then **Target 2** is scored, in two groups. The LLM agents run on the laptop GPU (Ollama or vLLM, local only):

| Group | Folder | Lines | Offers allowed | Action | What the LLM decides |
|---|---|---|---|---|---|
| **a · T2C** (Target 2, Constraint) | `target2/a_constraint/` | 50,000 | the offer families of the customer's action | `action_agent` | the action (Agent 3), then the offers (Agent 4) |
| **b · T2F** (Target 2, Free) | `target2/b_free/` | 50,000 | the best L2 DATA / COMBO offers for the spend | not used | the offers (Agent 4) |

There is no Target 1 and no control group (`FSDR_TARGETS=target2`, `FSDR_CONTROL_SIZE=0`). `FSDR_TARGETS=target1,target2` and `FSDR_CONTROL_SIZE=20000` bring them back.

**Real data of 2026-11-19? Start here: [docs/REAL_DATA_2026-11-19.md](docs/REAL_DATA_2026-11-19.md).** **New laptop: [docs/SETUP_NEW_LAPTOP.md](docs/SETUP_NEW_LAPTOP.md)**. It goes from `pip install uv` to the environment, the data folders, every command, the notebooks, the interface and the tests.

## What V4.0 changes (from V3.0)

1. **Market intelligence removed.** The competitor agent, its tools, the scraper (`nbx/core/competitors.py`, `beautifulsoup4`), `data/competitors/`, the `/competitors/*` routes, the Competitors tab and the journey step are all gone. The agents are now action → offer → critic → messenger.
2. **Three requirements files**, generated from `uv.lock`:
   - `requirements.txt` (runtime + tracing);
   - `requirements-dev.txt` (+ tests);
   - `requirements-notebook.txt` (+ Jupyter).
3. **Training and scoring data in separate folders** with the same 5 sub-folders: `data/training/<D>/` and `data/scoring/<D>/`, each with `usage/ models/ subscription/ catalog/ others/`. The target lists sit in `data/scoring/<D>/target2/a_constraint|b_free/`. T0 reads both folders as one run. The GPU notebooks use **one weekly folder** `data/weekly/<D>/` instead (fresh `*_auto` tables, no `*_score`).
4. **Target 2 scored on its own** (its P6 step, checks and files) into `data/output/<D>/target2/`. `final/input_flytxt.csv.gz` is written only when it PASSES. The engine only saves files; it sends nothing.
5. **Up to 6 recommended offers per line** (3 monthly + 3 weekly):
   - slots 1–2 monthly, 3–4 weekly;
   - slot 5 = the 3rd monthly offer, slot 6 = the 3rd weekly one (rule EXT-6);
   - favourites and forced bundles only fill an empty slot 5 / 6.

   `FSDR_RECO_OFFERS=4` gives the V3.0 file.
6. **Step-by-step notebooks**, each with a run guide `notebooks/<name>.md`:
   - `01_training` and `02_scoring`: the rules only;
   - **`03_gpu_agents` (Ollama) and `04_vllm_agents` (vLLM)**: every agent, 1 → 9, with the LLM on the GPU.
7. Obsolete V1.0 review documents removed.

## Quick start

```bash
pip install uv
uv python install 3.12
uv sync
uv run python run.py check
uv run python run.py fsdr-folders --date 2026-11-19              # the folders of a run date + the file of every table
# real data: copy the exports and the 2 lists, then notebook 03 (Ollama) or 04 (vLLM): uv run jupyter lab
# a test on dummy data, from the terminal (rules only, no LLM):
uv run python run.py fsdr-generate --date 2026-10-22 --n 3000      # dummy data -> data/training + data/scoring
uv run python run.py fsdr-load --date 2026-10-22                   # training T0
uv run python run.py fsdr-features --date 2026-10-22               # training T2 + T3
uv run python run.py fsdr-targets --generate --date 2026-10-22     # dummy lists of group a and group b
uv run python run.py fsdr-run-targets --date 2026-10-22            # Target 2 (a + b), final file -> data/output
uv run python run.py start                                          # web UI http://localhost:7860, API :8000/docs
```

## Documentation

| Document | Read it for |
|---|---|
| **[notebooks/FINAL_GUIDE_ALL_STEPS.md](notebooks/FINAL_GUIDE_ALL_STEPS.md)** | **The complete guide: install, `.env`, LLM server, weekly data, every notebook step one by one, outputs, time, problems** |
| **[docs/REAL_DATA_2026-11-19.md](docs/REAL_DATA_2026-11-19.md)** | The checklist of the first real run: `.env`, folders, files, lists, settings, checks |
| **[docs/SETUP_NEW_LAPTOP.md](docs/SETUP_NEW_LAPTOP.md)** | Install, environment, folders, every step, notebooks, interface, tests, troubleshooting |
| [docs/UV_ENV_SETUP.md](docs/UV_ENV_SETUP.md) | The uv environment from scratch (folder copied without `.venv`) |
| [docs/GPU_AGENTS_NOTEBOOK.md](docs/GPU_AGENTS_NOTEBOOK.md), [docs/VLLM_AGENTS_NOTEBOOK.md](docs/VLLM_AGENTS_NOTEBOOK.md) | The agents on the GPU with Ollama, or with vLLM (WSL2) |
| [docs/VLLM_VS_OLLAMA_SECURITY.md](docs/VLLM_VS_OLLAMA_SECURITY.md) | Ollama vs vLLM for the security team, and the WSL2 request email |
| [docs/DATA_UPDATE.md](docs/DATA_UPDATE.md) | Real data in the two-folder layout, week after week |
| [docs/USER_GUIDE_INTERFACE.md](docs/USER_GUIDE_INTERFACE.md) | Every tab and button of the web interface |
| [docs/RUN_FROM_TERMINAL.md](docs/RUN_FROM_TERMINAL.md) | Every terminal command and what it writes |
| [docs/fsdr_plan.md](docs/fsdr_plan.md) | Plan and status (§19 = V4.0) |
| [docs/fsdr_reference/](docs/fsdr_reference/) | Specs of record: master prompt, 100k-targets update, business rules |
| [docs/fsdr_known_differences.md](docs/fsdr_known_differences.md) | Differences vs the original production codes |
| [docs/FSDR_RECOMMENDER_LOGIC.md](docs/FSDR_RECOMMENDER_LOGIC.md), [docs/FSDR_HANDBOOK.md](docs/FSDR_HANDBOOK.md), [docs/FSDR_AGENTIC_RECOMMENDER.md](docs/FSDR_AGENTIC_RECOMMENDER.md) | The V1.0 engine in depth (bandit, ranker, agents, weekly cycle); V1.0 text, the competitor parts no longer apply |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | Optional: a local LLM with vLLM on a GPU |

## What's inside

```
nbx-python-upgrade-V4.0/
├── pyproject.toml / uv.lock / .python-version     uv project, exact lock, Python 3.12
├── requirements.txt / -dev.txt / -notebook.txt     the same pins for pip
├── run.py                                          launcher (check, start, test, fsdr-*, notebook, vllm-*)
├── .env.example                                    every setting (copy to .env, never commit .env)
├── nbx/fsdr/                                       engine: config (rules), schemas, validation, contract (T0, cache),
│                                                   features, segmentation, offers, panel, targets, final output (code 5),
│                                                   pipelines (training, scoring_groups, journey, …), agents, synth
├── nbx/api/  nbx/ui/  nbx/llms/  nbx/core/         FastAPI, Gradio, LLM backends, shared config / tracing / privacy
├── notebooks/                                      01_training, 02_scoring, 03_gpu_agents (Ollama), 04_vllm_agents (step by step), fsdr_results + builders
├── docs/                                           guides, plan, references
├── tests/                                          the test suite (uv run python run.py test -q)
└── vllm/  Dockerfile  .github/                     optional: local LLM, container, CI
```

## Security and privacy

- Never commit `.env`, `data/` (training, scoring, output, runs, cache…) or the raw extracts `docs/fsdr_reference/fsdr_original_*.txt`; `.gitignore` covers them.
- Notebooks are stored **without outputs**, because outputs hold MSISDNs (`tests/test_notebook.py` checks it).
- **Local only.** No cloud API is called: no OpenAI, no Anthropic / Claude, no analytics, no CDN.
  - The LLM runs on this machine: offline rules, Ollama, llama.cpp, LM Studio or vLLM.
  - `FSDR_LOCAL_ONLY=true` (default) refuses any LLM server or tracing endpoint elsewhere.
  - Details: [docs/SETUP_NEW_LAPTOP.md](docs/SETUP_NEW_LAPTOP.md) §12a.
- LLM prompts carry a salted pseudonym of the customer, never the MSISDN (`FSDR_PII_SALT`).
- With `FSDR_API_KEY` set, every `/fsdr/*` route needs the `X-API-Key` header. Set it as soon as the API listens on the network.
- The final file is written only when every check passes (H-08). The engine only saves files; it sends nothing.
