# FSDR Recommender V1.0 — NBA / NBO engine with competitor intelligence

For each prepaid line of the operator FSDR, the engine picks **the next best action** (retain, upsell data,
cross-sell voice, …) with a Vowpal Wabbit contextual bandit, then **the next best offers** for that action, and
compares them with the competitors' public plans (COMPETITOR_A / COMPETITOR_B). Three arms are tested against a
control group every week: `cstr` (business rules), `free` (ranker only) and `llm` (three agents: action, offer,
competitor).

**V1.0** is the FSDR-only edition of `nbx-python-upgrade`, rebuilt from the review
[docs/FSDR_REVIEW_2026-10.md](docs/FSDR_REVIEW_2026-10.md):

- the NBX demo modules are removed (council, vector stores, embeddings, Ollama, demo data);
- the environment is managed by **uv** (`pyproject.toml` + `uv.lock`);
- 16 review findings are implemented. Each one is documented, with its measured effect on take-up, revenue and
  uplift vs V0, in **[docs/IMPROVEMENTS_V1.0.md](docs/IMPROVEMENTS_V1.0.md)**.

Every V1.0 change has a switch in `.env` that restores V0's behaviour (see `.env.example`).

---

## Quick start with uv

Install uv once: <https://docs.astral.sh/uv/getting-started/installation/>
(`winget install astral-sh.uv`, `brew install uv`, or `pip install uv`).

```bash
cd nbx-python-upgrade-V1.0
uv sync                                 # .venv with the exact locked versions (Python 3.12, + test tools)
uv run python run.py check              # packages, competitor snapshot, FSDR runs, LLM backend, API key
uv run python run.py fsdr-generate --date 2026-10-21 --n 2000      # a dummy run folder (or drop real exports there)
uv run python run.py start              # API + web UI
```

| What | URL |
|---|---|
| Web UI (Gradio) | http://localhost:7860 |
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

`uv run` uses the project's `.venv` without activating it. `uv sync --no-dev` installs the runtime only.
After you change a dependency in `pyproject.toml`, run `uv lock` and then `uv sync`. The lock only takes
releases older than the `exclude-newer` date in `pyproject.toml`; bump that date when you re-lock.

---

## The step-by-step journey (web UI tab "FSDR Journey")

Put the dummy data in `data/fsdr_dummy/<YYYY-MM-DD>/` (one JSON file per table, git-ignored). Then click:

| Button | What you get |
|---|---|
| **1 · Upload data** | Every contract table copied byte for byte (SHA-256 checked), with a green / yellow / red / grey status per table. Also: the joins the NBA agent relies on with their match rates, a training and scoring run on this data, and 7 key calculations recomputed from the raw tables and compared with the engine (REV, trends, PROBA max, ARPU anchor, nationality group). |
| **2 · Best action recommender** | The action, the bandit probabilities, the business rules applied, the **LLM's own explanation**, and a plain-words reason. |
| **3 · Best offer recommender** | The top 4 offers (2 monthly + 2 weekly): price, data / minutes, ARPU anchor, P(accept), value, how each was selected, SMS script, the LLM's rationale. |
| **4 · Roaming & international** | The top 2 roaming packs, plus international packs when the line calls abroad. |
| **5 · Competitor comparison** | Each offer against the closest COMPETITOR_A / COMPETITOR_B plans, the verdict, the LLM's summary, and any swap (checked against the hard rules). |
| **6 · Full output (save)** | Everything about the line (every raw-table row, every engine column) and every step's result, saved to `data/fsdr/<D>/outputs/journey/<msisdn>_output.json`, plus a CSV row per line. |

Switch **Mode** to *Batch* to run any step for a list of MSISDNs: upload a `.csv` / `.txt` with an `msisdn` column (or one per line), or paste a list. Each step continues from the previous one, and a missing earlier step is run first. API: `POST /fsdr/journey/ingest`, `POST /fsdr/journey/<D>/<action|offers|addons|competitors|output>`.

---

## The weekly cycle

Each command works on one run folder `data/fsdr/<D>/` (D = run date, T = the Thursday on or before D):

```bash
uv run python run.py fsdr-weekly --data-dir data/fsdr/2026-10-21   # the whole week in one command (below)
uv run python run.py fsdr-generate --date 2026-10-21 --n 2000 --scenario baseline   # 46 JSON tables
uv run python run.py fsdr-validate --data-dir data/fsdr/2026-10-21                 # contract, keys, windows
uv run python run.py fsdr-feedback                 # rewards of last week's campaign (campaign log, D+7)
uv run python run.py fsdr-train                    # bandit + paired comparison with the champion
uv run python run.py fsdr-score                    # arms free / cstr / llm of the test sample
uv run python run.py fsdr-recommend --arm cstr     # offers (value arbitration inside the BR-66 families)
uv run python run.py fsdr-batch --arm llm --limit 100   # agents: action -> top 4 -> roaming -> international
uv run python run.py fsdr-report                   # take-up, revenue, uplift vs control with 95 % CI, fairness
uv run python run.py fsdr-simulate --weeks 3 --n 5000   # closed loop on the hidden synthetic world
uv run python run.py competitors-refresh           # re-scrape data/competitors/sources.json (local, git-ignored)
```

`fsdr-weekly` chains validate -> feedback -> train -> score -> offers (cstr, free, llm) -> report, skips feedback and
report when there is no campaign log yet, stops at the first failure and writes `outputs/weekly_run_report.json`
(options `--arms`, `--llm`, `--llm-limit`, `--steps`). Schedule it once a week (Windows Task Scheduler or cron) after
the weekly extract lands. At 1.7 M lines expect about 25 min for validate + train, a few minutes per arm for
offers, and plan 32 GB of RAM; timings and memory are in [docs/FSDR_RECOMMENDER_LOGIC.md](docs/FSDR_RECOMMENDER_LOGIC.md).

The same steps are available in the web UI tabs (FSDR Data, Training & Scoring, Offers, Agentic Recommender,
Simulation, Competitors). Every business rule (BR-01 … BR-86) and every output is described in
[docs/FSDR_HANDBOOK.md](docs/FSDR_HANDBOOK.md).

---

## The launcher

| Command | What it does |
|---|---|
| `uv run python run.py check` | Python, packages, competitor snapshot, FSDR runs, LLM backend, API key |
| `uv run python run.py start [--host H]` | API, then the web UI once the API is healthy (Ctrl+C stops both) |
| `uv run python run.py api [--reload]` / `ui` | API only / web UI only |
| `uv run python run.py test [pytest args]` | Test suite (216 tests), e.g. `test -k journey -v` |
| `uv run python run.py fsdr-*` | The FSDR steps above (`--help` on each) |
| `uv run python run.py competitors-refresh` | Re-scrape the competitor pages, save the snapshot |
| `uv run python run.py vllm-install / vllm-serve` | vLLM for the agents (WSL2 on Windows), port 8001 |

`--backend offline|vllm` before the command overrides `LLM_BACKEND`. The API and UI bind to 127.0.0.1. With
`--host 0.0.0.0`, set `FSDR_API_KEY` too; the launcher warns if you don't.

---

## What's inside

```
nbx-python-upgrade-V1.0/
├── pyproject.toml / uv.lock / .python-version   uv project (dependencies + dev group), exact lock
├── run.py                      launcher
├── Dockerfile / .dockerignore  container: API + UI, offline backend, non-root user
├── .github/workflows/ci.yml    CI: uv sync --locked, check, tests on Linux + Windows, Docker build
├── .env.example                every setting; the V1.0 switches with their V0 values
├── data/competitors/           competitor snapshot (anonymised) + sources.example.json
├── docs/                       handbook, review, V1.0 improvements, agents, GPU / vLLM
├── vllm/                       install / serve scripts for WSL2
├── nbx/
│   ├── fsdr/                   the engine: config (BR-xx), schemas, validation, features, labels, bandit,
│   │                           registry, offer rules, ranker, acceptance model, stats, campaign log,
│   │                           pipelines (train, score, recommend, batch, feedback, simulate), agents, synth
│   ├── core/                   config, VW text format, BPR, cohort graph, competitors, privacy (pseudonyms)
│   ├── agents/agentic/         ReAct loop and tool registry used by the FSDR agents
│   ├── llms/                   offline and vLLM backends
│   ├── api/                    FastAPI: /health, /fsdr/*, /competitors/*, API-key security
│   └── ui/                     Gradio: Home, FSDR tabs, Competitors
└── tests/                      unit, pipeline, agent, API and V1.0 tests
```

## Security and privacy

- **API key.** With `FSDR_API_KEY` set, every `/fsdr/*` and `/competitors/*` route needs the `X-API-Key` header.
  Forcing a promotion that the rule refused needs `X-Admin-Key` (`FSDR_ADMIN_KEY`).
- **CORS** is limited to `FSDR_CORS_ORIGINS`.
- **Prompts** carry a salted pseudonym of the customer, never the MSISDN (set `FSDR_PII_SALT`).
- **The vLLM key** is passed through the environment, never on the command line.
- **LangSmith tracing** is off by default. With `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` set, every weekly step,
  pipeline, decision, agent, tool call and LLM call becomes a nested run in the project `LANGSMITH_PROJECT`. Traces
  leave the machine: the MSISDN is replaced by its pseudonym, but tool outputs carry customer features, so for real
  data use a self-hosted LangSmith (`LANGSMITH_ENDPOINT`) or `LANGSMITH_HIDE_INPUTS/OUTPUTS=true`.
- **Never commit** `.env`, `data/fsdr/`, `data/competitors/sources.json` or the raw extracts in
  `docs/fsdr_reference/`; `.gitignore` covers them.

## Documentation

| Document | Read it for |
|---|---|
| **[docs/IMPROVEMENTS_V1.0.md](docs/IMPROVEMENTS_V1.0.md)** | Every V1.0 change: review finding, files, switch, measured effect on take-up / revenue / uplift vs V0 |
| **[docs/FSDR_RECOMMENDER_LOGIC.md](docs/FSDR_RECOMMENDER_LOGIC.md)** | The whole engine end to end: reading, keys, joins, features, labels, bandit, OPE, promotion, scoring, offers, agents, report, timings at 1.7 M lines |
| [docs/FSDR_REVIEW_2026-10.md](docs/FSDR_REVIEW_2026-10.md) | The technical and scientific review V1.0 implements |
| [docs/FSDR_HANDBOOK.md](docs/FSDR_HANDBOOK.md) | Complete V0 handbook: data contract, weekly cycle, every BR, CLI, API, UI (V1.0 deltas are in IMPROVEMENTS_V1.0.md) |
| [docs/FSDR_AGENTIC_RECOMMENDER.md](docs/FSDR_AGENTIC_RECOMMENDER.md) | The action, offer and competitor agents in depth |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | vLLM on a GPU (WSL2 or Linux server) |
| [docs/fsdr_verification_report.md](docs/fsdr_verification_report.md) | The business-rule verification of V0 |
