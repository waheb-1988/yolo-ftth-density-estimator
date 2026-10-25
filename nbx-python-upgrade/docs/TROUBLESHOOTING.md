# Troubleshooting

Start with `python run.py check`. It points at most problems directly.

## Installation

**`ModuleNotFoundError: No module named 'nbx'`**
Run commands from the project root (the folder that contains `run.py`), through
`python run.py ...` or `python -m nbx....`. Don't run files by path, as in
`python nbx/api/main.py`.

**`ModuleNotFoundError` for fastapi, gradio, …**
The virtual environment is not active, or the install did not finish.
Activate it (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`),
then run `pip install -r requirements-dev.txt`.

**Windows: "running scripts is disabled on this system" when activating**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
`cmd.exe` and `.venv\Scripts\activate.bat`.

**`implicit` or `vowpalwabbit` fails to install**
Both ship prebuilt wheels for Python 3.10–3.12 on Windows, macOS and Linux.
On a newer Python with no wheel yet, use Python 3.12 for the virtual
environment (`py -3.12 -m venv .venv` on Windows).

**The install is very large or slow (PyTorch)**
Install CPU-only PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Setup

**`[milvus] FAILED: ... nullable` during `setup`, or `milvus: unavailable`**
This is expected on Windows, because Milvus Lite has an upstream bug there.
The other 4 stores work. On Linux/macOS, `pip install -r requirements-vectorstores.txt`
installs a working version.

**`setup` cannot download `all-MiniLM-L6-v2`**
The first run needs internet access to Hugging Face. Behind a proxy, set
`HTTPS_PROXY`. Once downloaded, the model is cached (`~/.cache/huggingface`),
and later runs work offline. Without the model, the default `graph` retrieval
and the agents still work; only `vector` retrieval and competitor search need it.

**Competitor search returns `[]`**
The competitor index is not built yet: run `python run.py setup`.

## Running

**`ERROR: the API did not become healthy`**, or `[Errno 10048]` / `address already in use`
Something else is using port 8000 or 7860. Stop it, or change `FASTAPI_PORT` /
`GRADIO_PORT` in `.env`. To find the process on Windows:
`Get-NetTCPConnection -LocalPort 8000 | Select OwningProcess`. On Linux/macOS:
`lsof -i :8000`.

**The UI loads but every tab shows "backend offline"**
The API is not running, or it is on a different port. Use `python run.py start`,
which starts both. If you run them separately, start `python run.py api` first.

**Decisions say `mode: "fallback"`**
That is expected on the default `LLM_BACKEND=offline`. For LLM reasoning, see
[CONFIGURATION.md](CONFIGURATION.md#ollama-local-llms).

**On `ollama`: `/health` says `degraded`, or runs fall back**
Ollama is not running (`ollama serve`), or the model is not pulled
(`ollama pull mistral:7b`). Check with `curl http://localhost:11434/api/tags`.

**On `ollama`: runs are very slow or time out**
A 7B model on CPU or a small GPU takes a long time per agent step. Raise
`LLM_TIMEOUT_S`, use a smaller model (`phi3.5:3.8b`), or use the offline
backend for demos.

**The Benchmark LLMs tab finds no models**
It needs a real backend with at least one roster model pulled. The offline
backend has none.

**Warnings like `grpc ... RuntimeWarning` or `websockets.legacy is deprecated`**
These come from other packages installed in the same Python and are harmless.
A clean virtual environment avoids them.

## Resetting

```bash
# rebuild the indexes from the current data
python run.py setup
# fully fresh synthetic data + indexes (overwrites data/*.json)
python run.py seed
```

To start over with the original dataset, restore `data/*.json` from version
control (`git checkout -- data/`), delete the index folders listed in
[DATA.md](DATA.md#generated-not-in-version-control), and run `python run.py setup`.
