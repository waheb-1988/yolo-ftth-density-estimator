# Configuration

Settings are read from environment variables, or from a `.env` file in the
project root (`copy .env.example .env` on Windows, `cp .env.example .env`
elsewhere). Every setting has a default, so `.env` is optional. A real
environment variable overrides `.env`, and `run.py --backend X` overrides
both for `LLM_BACKEND`.

All settings live in `nbx/core/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `offline` | `offline`, `ollama` or `vllm` (see below) |
| `DEFAULT_LLM` | `mistral:7b` | model the agents use when the request names none |
| `LLM_TIMEOUT_S` | `300` | timeout per LLM call, in seconds |
| `OLLAMA_ENABLED` | `0` | Ollama is switched off: the app never calls it, `LLM_BACKEND=ollama` is refused and the Qwen3 embedders are unavailable. `1` re-enables it |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server, only when `OLLAMA_ENABLED=1` (also used for the Qwen3 embeddings) |
| `VLLM_URL` | `http://127.0.0.1:8001` | vLLM server |
| `VLLM_API_KEY` | empty | must match `vllm serve --api-key`. Never commit a real key |
| `VLLM_MAX_TOKENS` | `1024` | maximum tokens per vLLM completion |
| `DEFAULT_RETRIEVAL_MODE` | `graph` | `graph` (no embeddings) or `vector` |
| `DEFAULT_VECTORSTORE` | `chroma` | `chroma`, `faiss`, `qdrant`, `milvus`, `lancedb` |
| `DEFAULT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | embedding roster key (below) |
| `FASTAPI_PORT` | `8000` | API port (the UI calls the API on this port) |
| `GRADIO_PORT` | `7860` | web UI port |
| `DATA_DIR` | `<project>/data` | where data and indexes live. Use an absolute path |
| `LOG_LEVEL` | `INFO` | log level |

## LLM backends

### `offline` (default): Python only

No server and no model download. `generate()` never produces text, so every
agent runs its deterministic policy (see [ARCHITECTURE.md](ARCHITECTURE.md#deterministic-fallback-the-offline-backend)).
`/health` reports `status: ok`, `llm_backend: offline`. Decisions report
`mode: "fallback"`.

### `ollama`: local LLMs

```bash
ollama serve
ollama pull mistral:7b
python run.py --backend ollama start
```

The model roster (`OLLAMA_MODELS` in `config.py`), best first for this task:

| Model | Pull command |
|---|---|
| Mistral 7B (default) | `ollama pull mistral:7b` |
| LLaMA 3.1 8B | `ollama pull llama3.1:8b` |
| Gemma 2 9B | `ollama pull gemma2:9b` |
| Qwen 2.5 7B | `ollama pull qwen2.5:7b` |
| Phi 3.5 Mini | `ollama pull phi3.5:3.8b` |
| Neural Chat 7B | `ollama pull neural-chat:7b` |
| DeepSeek R1 7B | `ollama pull deepseek-r1:7b` |
| DeepSeek LLM 7B (base) | `ollama pull deepseek-llm:7b-base` |
| Ministral 3 8B Instruct | `ollama pull hf.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M` |

If Ollama stops while the app runs, the agents fall back automatically, and
`/health` reports `degraded`.

### `vllm`: Linux GPU production

One model per server process, with continuous batching for concurrent
requests and nightly batch scoring. Serve it under a roster name:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --served-model-name mistral:7b --port 8001
LLM_BACKEND=vllm python run.py start
LLM_BACKEND=vllm python -m nbx.scripts.batch_score --concurrency 64
```

Full setup: [GPU_SETUP.md](GPU_SETUP.md).

## Embedding models

| Key | Backend | Dimension | Needs |
|---|---|---|---|
| `all-MiniLM-L6-v2` (default) | sentence-transformers | 384 | nothing (downloaded once from Hugging Face) |
| `qwen3-embedding:0.6b` | Ollama | 1024 | `ollama pull qwen3-embedding:0.6b` |
| `qwen3-embedding:4b` | Ollama | 2560 | `ollama pull qwen3-embedding:4b` |
| `qwen3-embedding:8b` | Ollama | 4096 | `ollama pull qwen3-embedding:8b` |

Index another model without touching the MiniLM collections:

```bash
python -m nbx.scripts.seed_data --index-only --embedding-model qwen3-embedding:0.6b
```

## Scripts

Run them from the project root:

| Command | Does |
|---|---|
| `python -m nbx.scripts.seed_data [--index-only] [--embedding-model KEY]` | generate data and/or build vector indexes |
| `python -m nbx.scripts.scrape_competitors [--index-only]` | scrape the competitor pages, or re-index the snapshot |
| `python -m nbx.scripts.batch_score [--mode nba\|nbo] [--limit N] [--concurrency N] [--output FILE]` | score every customer to resumable JSONL |
| `python -m nbx.scripts.benchmark_llms` | compare the installed LLMs (needs Ollama/vLLM) |

## Using a different port

Set `FASTAPI_PORT` / `GRADIO_PORT` in `.env`. The UI reads `FASTAPI_PORT` to
find the API, so keep the two processes on the same `.env`.
