# Getting started

This guide takes you from a fresh machine to a running app, one step at a
time. Every command is shown for **Windows (PowerShell)** and **macOS / Linux**.

- [1. Install Python](#1-install-python)
- [2. Open a terminal in the project folder](#2-open-a-terminal-in-the-project-folder)
- [3. Create a virtual environment](#3-create-a-virtual-environment)
- [4. Install the dependencies](#4-install-the-dependencies)
- [5. (Optional) Create your `.env`](#5-optional-create-your-env)
- [6. Check the installation](#6-check-the-installation)
- [7. Build the search indexes](#7-build-the-search-indexes)
- [8. Start the app](#8-start-the-app)
- [9. Make your first decision](#9-make-your-first-decision)
- [10. Run the tests](#10-run-the-tests)
- [Next time: daily start](#next-time-daily-start)
- [Optional: real LLM agents with Ollama](#optional-real-llm-agents-with-ollama)

---

## 1. Install Python

You need Python **3.10 or newer** (3.12 recommended).

```powershell
python --version        # Windows
```
```bash
python3 --version       # macOS / Linux
```

If it is missing or too old, install it from https://www.python.org/downloads/.
On Windows, tick **"Add python.exe to PATH"** in the installer.

> On macOS/Linux, use `python3` wherever this guide says `python` until the
> virtual environment is active. Inside it, `python` works everywhere.

## 2. Open a terminal in the project folder

```powershell
cd C:\path\to\llm_recommendation_engin_telco\nbx-python
```
```bash
cd /path/to/llm_recommendation_engin_telco/nbx-python
```

Every command below is run from this folder (the one that contains `run.py`).

## 3. Create a virtual environment

A virtual environment keeps this app's packages separate from the rest of your
system.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. Activate it again each time you open a
new terminal.

> **Windows: "running scripts is disabled on this system"?** Run this once,
> then activate again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
> Or use `cmd.exe` and `.venv\Scripts\activate.bat`.

## 4. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

This installs the app and the test tools. It takes a few minutes the first time.

### Smaller install: CPU-only PyTorch

`sentence-transformers` brings in PyTorch, which by default can be a large
download (especially the CUDA build on Linux). If you have no GPU, install the
CPU build **before** step 4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

### Optional: the extra vector stores

The app uses Chroma and FAISS by default. To also compare Qdrant, Milvus Lite
and LanceDB in the **Vector store compare** tab:

```bash
pip install -r requirements-vectorstores.txt
```

Milvus Lite does not work on Windows (an upstream bug), so it shows
"unavailable" there. Everything else works.

## 5. (Optional) Create your `.env`

Every setting has a default, so you can skip this. To change ports, the LLM
backend, etc.:

```powershell
copy .env.example .env
```
```bash
cp .env.example .env
```

Then edit `.env`. Every setting is described in [CONFIGURATION.md](CONFIGURATION.md).

## 6. Check the installation

```bash
python run.py check
```

Expected output ends with **`All checks passed.`** Each package is listed as
`[ok]`. A `[todo] search index not built yet` line is normal at this point;
the next step builds it. If anything says `[FAIL]`, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## 7. Build the search indexes

```bash
python run.py setup
```

This embeds the offers and segments into the vector stores and indexes the
competitor price snapshot. The first run downloads the MiniLM embedding model
(~90 MB), so it takes about 1 minute. You only need it once, and again after
you change the data.

Expected output (Windows):

```
Built 64 documents for indexing (embedding_model=all-MiniLM-L6-v2).
  [chroma] indexed 64 documents
  [faiss] indexed 64 documents
  [qdrant] indexed 64 documents
  [milvus] FAILED: ...nullable...      <- Windows only, harmless
  [lancedb] indexed 64 documents
113 competitor documents indexed from ...competitor_offers.json
```

> `setup` keeps the data in `data/`. `python run.py seed` instead **generates a
> new random synthetic dataset** first. Use it only when you want fresh data.

## 8. Start the app

```bash
python run.py start
```

The launcher starts the API, waits until `/health` answers, then starts the
web UI:

```
  API      http://localhost:8000
  API docs http://localhost:8000/docs
  Web UI   http://localhost:7860

Press Ctrl+C to stop.
```

Open **http://localhost:7860** in your browser. Press **Ctrl+C** in the
terminal to stop both servers.

Prefer two terminals (for example, to see each log separately)? Then run:

```bash
python run.py api --reload     # terminal 1
python run.py ui               # terminal 2
```

## 9. Make your first decision

**In the web UI:** open the **Agentic Best Action Recommender** tab, pick a
customer from the list, then click the run button. You'll see the agents' steps
live, then the chosen action, channel, timing, message and explanation. The
**Agentic Offer Recommender** tab does the same and adds the offers for that
action. [USER_GUIDE.md](USER_GUIDE.md) describes every tab.

**From the API** (in a second terminal while the app is running):

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/agentic/decide -H "Content-Type: application/json" -d "{\"msisdn\": 1, \"scope\": \"full\"}"
```

On Windows PowerShell, `curl` is an alias, so use `curl.exe` or:

```powershell
Invoke-RestMethod http://localhost:8000/agentic/decide -Method Post -ContentType "application/json" -Body '{"msisdn": 1, "scope": "full"}'
```

Or try every endpoint from the browser at **http://localhost:8000/docs**.

## 10. Run the tests

```bash
python run.py test
```

You should see all tests pass in about 20–30 seconds. The tests need no
server and no LLM. More in [TESTING.md](TESTING.md).

---

## Next time: daily start

```powershell
cd C:\path\to\nbx-python
.venv\Scripts\Activate.ps1
python run.py start
```
```bash
cd /path/to/nbx-python
source .venv/bin/activate
python run.py start
```

---

## Optional: real LLM agents with Ollama

By default the agents run on deterministic policies (`LLM_BACKEND=offline`).
To have real LLMs reason, call the tools and write the messages:

1. Install Ollama from https://ollama.com and start it (`ollama serve`, or the
   desktop app).
2. Pull a model: `ollama pull mistral:7b`.
3. Start NBX on the Ollama backend:
   ```bash
   python run.py --backend ollama start
   ```
   or set `LLM_BACKEND=ollama` in `.env`.

Decisions now report `mode: "agentic"` instead of `"fallback"`. One agentic
run takes from a few seconds to several minutes, depending on your hardware.
The **Benchmark LLMs** tab compares every model you have pulled. See
[CONFIGURATION.md](CONFIGURATION.md#llm-backends) for the full model list.
