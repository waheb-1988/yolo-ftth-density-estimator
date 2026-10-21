# NBX — Agentic Next-Best-Action / Next-Best-Offer engine (Python edition)

NBX picks, for each telco customer, **the best action to take** (retain, upsell
data, cross-sell voice, …) and then **the best offers for that action**, with a
customer message (English, Arabic and the customer's own language) and an
explanation of every decision.

This folder is the complete app in **pure Python**: a FastAPI backend and a
Gradio web UI. It installs with `pip` only and runs with **no external servers**:
no Ollama, no database, no Node.js. The LLM is optional; without one, every agent
runs its built-in deterministic policy (see [LLM backends](#llm-backends)).

---

## Quick start (5 commands)

**Windows (PowerShell)**

```powershell
cd nbx-python
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python run.py setup          # build the search indexes (about 1 min, once)
python run.py start          # API + web UI
```

**macOS / Linux**

```bash
cd nbx-python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python run.py setup
python run.py start
```

Then open:

| What | URL |
|---|---|
| Web UI (Gradio) | http://localhost:7860 |
| API | http://localhost:8000 |
| Interactive API docs (Swagger) | http://localhost:8000/docs |

Press **Ctrl+C** to stop both. To check everything is installed correctly:
`python run.py check`. To run the tests: `python run.py test`.

New to Python virtual environments, or something went wrong? Read
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**. It covers each step in detail.

---

## The `run.py` launcher

One entry point, the same on every OS:

| Command | What it does |
|---|---|
| `python run.py check` | Checks Python version, packages, data files, search index and LLM backend |
| `python run.py setup` | Builds the vector-store and competitor indexes from the data in `data/` (keeps the data) |
| `python run.py seed` | **Regenerates** the synthetic dataset, retrains the models, then builds the indexes |
| `python run.py start` | Starts the API, waits until it is healthy, then starts the web UI |
| `python run.py api [--reload]` | API only (`--reload` restarts it when code changes) |
| `python run.py ui` | Web UI only (the API must already be running) |
| `python run.py test [pytest args]` | Runs the test suite, e.g. `python run.py test -k agentic -v` |

Add `--backend offline|ollama|vllm` before the command to override the LLM
backend for that run, e.g. `python run.py --backend ollama start`.
`start` and `api` accept `--host 0.0.0.0` to expose the app on your network.

---

## What's inside

```
nbx-python/
├── run.py                       launcher (check / setup / seed / start / api / ui / test)
├── requirements.txt             runtime dependencies
├── requirements-dev.txt         + pytest
├── requirements-vectorstores.txt  optional: Qdrant, Milvus Lite, LanceDB
├── pyproject.toml               project metadata + pytest settings
├── .env.example                 every setting, with its default
├── data/                        the dataset (JSON); indexes are built here by `setup`
│   └── competitors/             Omantel / Vodafone Oman price snapshot
├── docs/                        full documentation (see below)
├── nbx/                         the application package
│   ├── api/main.py              FastAPI app: all HTTP endpoints
│   ├── ui/gradio_app.py         Gradio web UI (talks to the API over HTTP)
│   ├── ui/theme.py              UI look and feel (Ooredoo theme)
│   ├── agents/                  decision pipelines
│   │   ├── agentic/             agent council: NBA strategist, NBO specialist,
│   │   │                        critic, messenger, copilot, tools, explanations
│   │   ├── nba_agent.py         bandit + LLM action decision
│   │   ├── nbo_agent.py         offer pipeline: retrieve → reason → rerank
│   │   ├── rules_engine.py      business-rules hook
│   │   └── *ingestion_pipeline.py  step-by-step data upload
│   ├── core/                    config, schemas, models and data logic
│   │   ├── config.py            all settings (read from .env)
│   │   ├── nba_bandit.py        Vowpal Wabbit contextual bandit
│   │   ├── collab_filter.py     collaborative filtering (implicit BPR)
│   │   ├── knowledge_graph.py   networkx cohort graph
│   │   ├── competitors.py       competitor scraping, indexing and comparison
│   │   └── data_simulator.py    synthetic data generator
│   ├── llms/                    LLM + embedding clients
│   │   ├── llm_client.py        dispatches to the configured backend
│   │   ├── offline_client.py    pure-Python backend (default)
│   │   ├── ollama_client.py     Ollama backend
│   │   └── vllm_client.py       vLLM backend (GPU production)
│   ├── vectorstores/            Chroma, FAISS, Qdrant, Milvus Lite, LanceDB
│   └── scripts/                 seed_data, scrape_competitors, batch_score, benchmark_llms
└── tests/                       pytest suite (unit + end-to-end API)
```

---

## LLM backends

Set `LLM_BACKEND` in `.env` (copy it from `.env.example`) or pass `--backend`:

| Backend | Needs | What you get |
|---|---|---|
| **`offline`** (default) | nothing, pure Python | Every agent runs its deterministic policy: rule-based action choice + bandit, CF/graph offer ranking, eligibility and expected-value checks, template messages and explanations. Decisions show `mode: "fallback"`. |
| `ollama` | [Ollama](https://ollama.com) running + a model (`ollama pull mistral:7b`) | Real LLM agents (ReAct tool use, critic review, written messages). Decisions show `mode: "agentic"`. |
| `vllm` | Linux GPU server | Same as Ollama, built for many concurrent requests and nightly batch runs. See [docs/GPU_SETUP.md](docs/GPU_SETUP.md). |

In every mode the tools, guardrails and numbers are the same. The LLM only
changes who reasons over them and who writes the text.

---

## Documentation

| Document | Read it for |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Step-by-step install and first run, per OS |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | A tour of each web UI tab |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How a decision is made, module by module |
| [docs/API.md](docs/API.md) | Every endpoint, with `curl` examples |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every `.env` setting and the LLM backends |
| [docs/DATA.md](docs/DATA.md) | The data files, their fields, and how to use your own data |
| [docs/TESTING.md](docs/TESTING.md) | Running and writing tests |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems and fixes |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | Production on a Linux GPU server with vLLM |

---

## Requirements

- Python **3.10+** (verified on 3.12, Windows 11)
- About 2–3 GB of disk for the packages (PyTorch, through sentence-transformers, is
  the biggest one; see [GETTING_STARTED](docs/GETTING_STARTED.md#smaller-install-cpu-only-pytorch)
  for a smaller CPU-only install)
- Internet **once**: for `pip install` and for the first `setup`, which downloads
  the small MiniLM embedding model (~90 MB) from Hugging Face. After that the app
  runs offline.
