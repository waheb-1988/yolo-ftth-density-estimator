# GPU Setup Guide: NBO / NBA Recommendation Engine (vLLM production)

This guide explains how to run this repository on a company GPU server with
**vLLM** as the production LLM server. vLLM is built for two of our needs:

- **Many requests at the same time.** Several agents or systems call
  `/recommend` and `/nba/decide` concurrently, and vLLM batches them on the
  GPU continuously.
- **Nightly batch scoring.** `nbx/scripts/batch_score.py` runs the full decision
  for every subscriber.

No server yet? vLLM also runs on a Windows laptop with a small NVIDIA GPU, through
WSL2. See [section 14](#14-small-gpu-on-windows-wsl2-gtx-1650-4-gb). This is how the
FSDR agents are tested on a GTX 1650 with 4 GB.

---

## 0. TL;DR: how it fits together

The code supports two LLM backends, chosen with one setting in
`nbx-python/.env`:

| `LLM_BACKEND=` | Used for | Where |
|---|---|---|
| `ollama` | development, Windows laptops, comparing all 9 models (Benchmark tab) | your laptop |
| **`vllm`** | **production: concurrent API traffic and nightly batch scoring** | **company GPU server** |

On the GPU server:

1. Install the **NVIDIA driver**. You don't need a separate CUDA toolkit,
   because vLLM's pip wheel brings its own CUDA libraries.
2. Install **vLLM** in its own venv and serve **one model** on port **8001**,
   under the same name the app already uses (`--served-model-name
   mistral:7b`).
3. Set up the app (FastAPI and Gradio) in a second venv with
   `LLM_BACKEND=vllm`.
4. Open the UI from your laptop through an **SSH tunnel**, and schedule
   `batch_score.py` nightly.

```
 Your laptop                          Company GPU server (Linux)
 ───────────                          ────────────────────────────────────────────────
 browser ──SSH tunnel──▶ :7860 Gradio ─┐
 browser ──SSH tunnel──▶ :8000 FastAPI ┼──▶ :8001 vLLM (1 model, continuous batching) ──▶ GPU
 other systems ─(VPN/proxy)─▶ :8000 ───┘          ▲
 cron 01:00 ── scripts/batch_score.py ────────────┘   (64 requests in flight)
                                      CPU: graph, CF, bandit, vector stores, MiniLM
 (optional) :11434 Ollama, only for Qwen3 embedders or multi-model benchmarking
```

> **Ports:** vLLM's default port is 8000, but FastAPI already uses 8000
> here. Always start vLLM with `--port 8001`. That is the default
> `VLLM_URL` in `.env`.

---

## 1. What to ask IT for (checklist)

| Item | Minimum | Recommended | Why |
|---|---|---|---|
| OS | Ubuntu 22.04 LTS | **Ubuntu 24.04 LTS** | vLLM is Linux-only, and 24.04 ships Python 3.12 |
| GPU | 1× NVIDIA, **16 GB** (T4, A4000, RTX 4080) using a quantized (AWQ) model | 1× **24 GB+** (L4, A10, RTX 4090, A5000), or 40/80 GB (A100/H100) for heavy batch | see the sizing table below |
| NVIDIA driver | ≥ 550 | latest production branch (570+) | recent vLLM wheels are built on CUDA 12.8+ |
| CPU / RAM | 8 vCPU / 32 GB RAM | 16 vCPU / 64 GB RAM | the pipeline's CPU steps (graph, CF, bandit) run alongside the GPU |
| Disk | 60 GB SSD | 150 GB NVMe | 7B weights in FP16 ≈ 15 GB each, plus the HF cache and venvs |
| Access | SSH with `sudo` | SSH with `sudo` | driver, services, cron |
| Network | outbound HTTPS to `huggingface.co`, `pypi.org`, `github.com` (and `registry.npmjs.org` for the frontend) | the same | model weights and packages (see §10 if the server is air-gapped) |
| Hugging Face | a free HF account + access token | the same | `mistralai/Mistral-7B-Instruct-v0.3` is gated: accept its terms on HF once |

### GPU sizing for vLLM

vLLM loads the **full-precision (FP16/BF16) weights** by default, not the 4-bit
GGUF files Ollama uses. Whatever GPU memory is left over becomes **KV cache**,
which sets how many requests can run in parallel. More free memory means more
concurrency.

| Model | Weights | Fits on |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` (FP16) | ~14.5 GB | 24 GB+ (≈ 5–6 GB of KV cache on a 24 GB card) |
| `Qwen/Qwen2.5-7B-Instruct` (BF16) | ~15 GB | 24 GB+ |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit) | ~5.5 GB | **16 GB**, and on 24 GB it leaves a large KV cache (high concurrency) |
| Mistral-7B-Instruct-v0.3 AWQ (community build) | ~4.5 GB | 16 GB |

- **16 GB GPU**: use an AWQ model.
- **24 GB GPU**: FP16 works. AWQ gives roughly 3× more KV cache, which means
  more requests in parallel. That's a good trade for nightly batch scoring.
- **40/80 GB GPU**: FP16 with a very large batch. Use this if the subscriber
  base is in the millions.

### Which model to serve

vLLM serves **one** model. Pick the winner from the Ollama benchmark
(`python -m nbx.scripts.benchmark_llms` on your laptop):

- `mistral:7b` is the current `DEFAULT_LLM`. On HF it's
  `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0, gated: needs an HF token).
- `qwen2.5:7b` is `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0, not gated, and has
  an official AWQ build). It is better for mixed Arabic/English text.

> **License check:** only serve models whose license allows commercial use.
> Both models above are Apache 2.0.

---

## 2. Connect and check the GPU

```bash
ssh <user>@<gpu-server>
nvidia-smi          # must list the GPU and a driver version >= 550
```

If `nvidia-smi` fails or the driver is too old:

```bash
sudo apt update
sudo ubuntu-drivers devices          # shows the recommended driver
sudo ubuntu-drivers install
sudo reboot
nvidia-smi
```

---

## 3. Get the code onto the server

```bash
cd ~
git clone <your-git-remote-url> llm_recommendation_engin_telco
cd llm_recommendation_engin_telco
```

If the server can't reach your git remote, copy the repo from your laptop:

```bash
rsync -av --exclude node_modules --exclude __pycache__ --exclude .next --exclude .venv \
  ./llm_recommendation_engin_telco <user>@<gpu-server>:~/
```

Install the system packages for both venvs:

```bash
# Ubuntu 24.04
sudo apt install -y python3.12 python3.12-venv python3.12-dev build-essential tmux
# Ubuntu 22.04: add the deadsnakes PPA first
#   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
```

---

## 4. Install and run vLLM (the GPU part)

### 4.1 Use a separate venv for vLLM

vLLM pins an exact PyTorch/CUDA build. Keep it in **its own venv** so it
never conflicts with the app's `sentence-transformers` and `torch`.

```bash
python3.12 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm
pip freeze | grep -E "^(vllm|torch)==" > ~/vllm-versions.txt   # record what you validated
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True <GPU name>
```

After it works, pin that version (`pip install vllm==<version>`) so a
reinstall can't silently upgrade it.

### 4.2 Hugging Face access (only for gated models such as Mistral)

1. Log in to huggingface.co, open `mistralai/Mistral-7B-Instruct-v0.3`, and
   accept the terms.
2. Create a **read** token (Settings → Access Tokens).
3. On the server:

```bash
mkdir -p ~/.config/vllm
cat > ~/.config/vllm/env <<'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
VLLM_API_KEY=change-me-long-random-string
EOF
chmod 600 ~/.config/vllm/env
```

Generate the API key with `openssl rand -hex 32`. FastAPI sends it to vLLM
(set the same value in `.env`, see §5.2).

### 4.3 Start vLLM

**24 GB+ GPU, Mistral FP16** (matches today's `DEFAULT_LLM=mistral:7b`):

```bash
source ~/vllm-env/bin/activate
set -a; source ~/.config/vllm/env; set +a

vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --served-model-name mistral:7b \
  --host 127.0.0.1 --port 8001 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

**16 GB GPU (or maximum concurrency), Qwen 2.5 AWQ:**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5:7b \
  --host 127.0.0.1 --port 8001 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
# then set DEFAULT_LLM=qwen2.5:7b in nbx-python/.env
```

What each flag does:

| Flag | Why |
|---|---|
| `--served-model-name mistral:7b` | vLLM answers to the **same name** as the app's roster, so `DEFAULT_LLM`, `/models` and the UI dropdowns work unchanged |
| `--host 127.0.0.1 --port 8001` | local only (FastAPI is the public entry point); 8001 because FastAPI owns 8000 |
| `--api-key` | rejects calls without the key, even from the same machine |
| `--max-model-len 8192` | our prompts are short, so a smaller context window leaves more memory for parallel requests |
| `--gpu-memory-utilization 0.90` | share of GPU memory vLLM takes (weights + KV cache). Lower it to **0.75** if Ollama embedders share the GPU (§4.5) |
| *(T4 / older GPUs)* `--dtype half` | Turing GPUs have no bfloat16 |

The first start downloads the weights (≈ 15 GB for FP16) into
`~/.cache/huggingface`. Later starts take 1–2 minutes.

### 4.4 Check that vLLM works

```bash
# list served models (should print "mistral:7b")
curl -s http://127.0.0.1:8001/v1/models -H "Authorization: Bearer $VLLM_API_KEY"

# one chat call in JSON mode, the same way the app calls it
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"mistral:7b","messages":[{"role":"user","content":"Return {\"ok\": true} as JSON"}],
       "response_format":{"type":"json_object"},"max_tokens":50}'

nvidia-smi    # ~90% GPU memory in use is NORMAL: vLLM pre-allocates its KV cache
```

### 4.5 (Optional) Ollama next to vLLM

On the server you only need Ollama if:
- you use `retrieval_mode=vector` with a **Qwen3-Embedding** model (these
  always run on Ollama), or
- you want to run the multi-model Benchmark tab there.

The production defaults (`retrieval_mode=graph`, MiniLM embedder) don't need
it. If you do install it:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-embedding:0.6b
```

and start vLLM with `--gpu-memory-utilization 0.75` so both fit on the GPU.

---

## 5. Set up the app (`nbx-python`)

### 5.1 App venv

```bash
cd ~/llm_recommendation_engin_telco/nbx-python
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pytest -q                      # quick sanity check, needs no GPU or LLM server
```

On Linux, `requirements.txt` automatically uses the original
`milvus-lite==2.4.9` pin.

### 5.2 Configure `.env` for vLLM

Edit `nbx-python/.env` on the **server** only. Your laptop keeps
`LLM_BACKEND=ollama`.

```env
LLM_BACKEND=vllm
VLLM_URL=http://127.0.0.1:8001
VLLM_API_KEY=<same value as in ~/.config/vllm/env>
VLLM_MAX_TOKENS=1024
DEFAULT_LLM=mistral:7b          # = the --served-model-name you started vLLM with

OLLAMA_URL=http://localhost:11434   # only used by Qwen3 embedders, if any
DEFAULT_VECTORSTORE=chroma
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
FASTAPI_PORT=8000
GRADIO_PORT=7860
DATA_DIR=./data
LOG_LEVEL=INFO
```

> `.env` is committed to git. **Don't commit the real `VLLM_API_KEY`.**
> Either set it only in the server's copy, or export it as an environment
> variable in the systemd unit (§8), which overrides `.env`.

### 5.3 Data and indexes

```bash
python -m nbx.scripts.seed_data      # only if you want fresh synthetic data or new indexes
```

Load real data through the Gradio "Upload & Setup" tab or `POST /upload`.

### 5.4 Start the API and the UI

```bash
tmux new -s nbo
source .venv/bin/activate
uvicorn nbx.api.main:app --host 0.0.0.0 --port 8000      # ONE worker (see note) and no --reload
# Ctrl-b "  for a second pane:
source .venv/bin/activate && python -m nbx.ui.gradio_app
# detach with Ctrl-b d
```

> **One uvicorn worker is correct.** The app keeps job state in process
> (upload jobs, caches), so several workers would each have their own copy.
> The LLM call is async, so a single worker already keeps many requests in
> flight to vLLM, and vLLM does the batching. Scale through vLLM
> (`--gpu-memory-utilization`, a bigger GPU, or AWQ), not through uvicorn
> workers.

### 5.5 End-to-end check

```bash
curl -s localhost:8000/health
# expect: "status":"ok", "ollama_connected":true, "llm_backend":"vllm", "loaded_models":["mistral:7b"]
# (the field is called ollama_connected for backward compatibility; it means "LLM backend reachable")

# quality check of the served model on the 3 needle customers (need detection + valid JSON)
python -m nbx.scripts.benchmark_llms
```

Then send one request through `/docs` (`POST /nba/decide`) or the Gradio
**NBA Decision** tab.

> With `LLM_BACKEND=vllm`, the Benchmark tab and `/models` only show the one
> served model as available. That's expected. Compare all 9 models on your
> laptop with `LLM_BACKEND=ollama`.

---

## 6. Nightly batch scoring

`nbx/scripts/batch_score.py` runs **the same pipeline as `POST /nba/decide`**
(or `/recommend` with `--mode nbo`) for every customer in `customers.json`.
It keeps `--concurrency` requests in flight so vLLM can batch them, and
writes one JSON line per customer.

```bash
cd ~/llm_recommendation_engin_telco/nbx-python && source .venv/bin/activate

# smoke test on 200 customers first, then read the "/s" rate it prints
python -m nbx.scripts.batch_score --limit 200 --output data/batch_test.jsonl

# full run
python -m nbx.scripts.batch_score --concurrency 64
# -> data/batch_scores_nba_<YYYY-MM-DD>.jsonl
```

- **Resumable.** If the run stops, run the same command again. Customers
  already in the output file with `status: ok` are skipped.
- **Each output row** contains `msisdn`, `status` (`ok`/`error`),
  `llm_fallback`, `scored_at` and `result` (the full decision).
  `llm_fallback=true` means the LLM call failed and that customer was scored
  on the CF/graph/bandit signal only. The run prints a warning if any
  customer fell back.
- **Cost per customer:** `nba` mode makes **2 LLM calls** (NBA + NBO), plus
  one retry if the JSON doesn't match the schema. `nbo` mode makes 1.
- **Tuning `--concurrency`:** vLLM logs a line every few seconds with
  `Running: N reqs, Waiting: M reqs, GPU KV cache usage: X%`. Raise
  `--concurrency` (32 → 64 → 128) while throughput keeps rising and KV-cache
  usage stays below about 90%. If `Waiting` stays high, the GPU is saturated
  and more concurrency won't help.
- **Estimate the run time** from the smoke test: customers ÷ rate. For
  example, 500,000 customers at 20/s ≈ 7 hours. If that's too long, use AWQ
  (more KV cache), `--mode nbo`, or a bigger GPU.

### Schedule it (cron)

```bash
mkdir -p ~/llm_recommendation_engin_telco/nbx-python/logs
crontab -e
```

```cron
# every night at 01:00, server local time
0 1 * * * cd $HOME/llm_recommendation_engin_telco/nbx-python && .venv/bin/python -m nbx.scripts.batch_score --concurrency 64 >> logs/batch_$(date +\%F).log 2>&1
```

The batch job and live API traffic share the same vLLM server. If both have
to run at the same time, lower `--concurrency` so live requests aren't
queued behind the batch.

---

## 7. Open the UI from your laptop (SSH tunnel)

```bash
# run on your laptop (PowerShell, Git Bash, macOS or Linux)
ssh -N -L 7860:localhost:7860 -L 8000:localhost:8000 <user>@<gpu-server>
```

- Gradio: http://localhost:7860
- FastAPI docs: http://localhost:8000/docs
- Next.js frontend: run it **on your laptop** (`cd nbx-frontend && npm run
  dev`). It calls `http://localhost:8000`, which goes through the tunnel.

In VS Code **Remote-SSH**, the "Ports" tab forwards these automatically.

For **other systems** that call `/recommend` in production, ask IT to put
FastAPI behind the company reverse proxy or VPN, with authentication. Never
expose port 8001 (vLLM).

---

## 8. Keep everything running (systemd)

`/etc/systemd/system/vllm.service`:

```ini
[Unit]
Description=vLLM OpenAI-compatible server
After=network-online.target

[Service]
User=<your-user>
EnvironmentFile=/home/<your-user>/.config/vllm/env
ExecStart=/home/<your-user>/vllm-env/bin/vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --served-model-name mistral:7b --host 127.0.0.1 --port 8001 \
  --api-key ${VLLM_API_KEY} --max-model-len 8192 --gpu-memory-utilization 0.90
Restart=on-failure
RestartSec=10
# model load can take a few minutes on first start
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/nbo-api.service`:

```ini
[Unit]
Description=NBO FastAPI
After=vllm.service
Wants=vllm.service

[Service]
User=<your-user>
WorkingDirectory=/home/<your-user>/llm_recommendation_engin_telco/nbx-python
EnvironmentFile=/home/<your-user>/.config/vllm/env
Environment=LLM_BACKEND=vllm
ExecStart=/home/<your-user>/llm_recommendation_engin_telco/nbx-python/.venv/bin/uvicorn nbx.api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

`EnvironmentFile` gives the API the same `VLLM_API_KEY` as vLLM, so the key
never needs to be in the git-tracked `.env`. Real environment variables
override `.env`.

`/etc/systemd/system/nbo-gradio.service`:

```ini
[Unit]
Description=NBO Gradio UI
After=nbo-api.service

[Service]
User=<your-user>
WorkingDirectory=/home/<your-user>/llm_recommendation_engin_telco/nbx-python
ExecStart=/home/<your-user>/llm_recommendation_engin_telco/nbx-python/.venv/bin/python -m nbx.ui.gradio_app
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm nbo-api nbo-gradio
journalctl -u vllm -f            # watch the model load, then throughput / KV-cache lines
```

To switch models later, change the HF id and `--served-model-name` in
`vllm.service`, update `DEFAULT_LLM` in `.env`, then run
`sudo systemctl restart vllm nbo-api`.

---

## 9. Alternative: Docker Compose

Use this if IT prefers containers. The server needs Docker and the **NVIDIA
Container Toolkit**. Check with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that fails, IT installs the toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

`nbx-python/Dockerfile`:

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 7860
```

`docker-compose.gpu.yml` (repo root). Put `HF_TOKEN` and `VLLM_API_KEY` in
a `.env.secrets` file that is **not** committed:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest      # pin the exact tag you validated
    restart: unless-stopped
    ipc: host                           # vLLM needs large shared memory
    env_file: .env.secrets              # HF_TOKEN, VLLM_API_KEY
    volumes:
      - hf_cache:/root/.cache/huggingface
    command: >
      --model mistralai/Mistral-7B-Instruct-v0.3
      --served-model-name mistral:7b
      --api-key ${VLLM_API_KEY}
      --max-model-len 8192
      --gpu-memory-utilization 0.90
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all             # or device_ids: ["0"]
              capabilities: [gpu]

  api:
    build: ./nbx-python
    command: uvicorn nbx.api.main:app --host 0.0.0.0 --port 8000
    env_file: [./nbx-python/.env, .env.secrets]
    environment:
      - LLM_BACKEND=vllm
      - VLLM_URL=http://vllm:8000      # inside the compose network vLLM keeps its default port
    volumes:
      - ./nbx-python/data:/app/data
    # Gradio's port is published here too, because gradio shares this network
    ports: ["127.0.0.1:8000:8000", "127.0.0.1:7860:7860"]
    depends_on: [vllm]

  gradio:
    build: ./nbx-python
    command: python -m nbx.ui.gradio_app
    env_file: ./nbx-python/.env
    volumes:
      - ./nbx-python/data:/app/data
    # gradio_app.py hardcodes API_BASE = http://localhost:8000, so share the
    # api container's network namespace and "localhost" reaches the API
    network_mode: "service:api"
    depends_on: [api]

volumes:
  hf_cache:
```

Compose substitutes `${VLLM_API_KEY}` in `command:` from the shell
environment or from a `.env` file next to the compose file, **not** from
`env_file`. Run it like this:

```bash
set -a; source .env.secrets; set +a
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml logs -f vllm        # wait for "Application startup complete"
curl -s localhost:8000/health
# nightly batch inside the api container:
docker compose -f docker-compose.gpu.yml exec api python -m nbx.scripts.batch_score --concurrency 64
```

---

## 10. Air-gapped server (no internet)

1. **Model weights:** on a connected machine, run
   `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir mistral-7b-v0.3`.
   Copy the folder to the server and serve it by path:
   `vllm serve /models/mistral-7b-v0.3 --served-model-name mistral:7b ...`
   Set `HF_HUB_OFFLINE=1`.
2. **vLLM:** on a connected Linux x86_64 machine with Python 3.12, run
   `pip download vllm -d vllm-wheels/`, copy the folder, then
   `pip install --no-index --find-links vllm-wheels/ vllm`.
   Or `docker save` the `vllm/vllm-openai` image and `docker load` it on the
   server.
3. **App packages:** run `pip download -r requirements.txt -d app-wheels/`
   on a matching Linux machine, then `pip install --no-index --find-links
   app-wheels/ -r requirements.txt`.
4. **MiniLM:** copy `~/.cache/huggingface/` from a machine where the app has
   already run once.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` → `degraded`, `llm_backend: vllm` | vLLM isn't up yet (model still loading) or `VLLM_URL`/port is wrong. Check `journalctl -u vllm -f` and `curl 127.0.0.1:8001/v1/models -H "Authorization: Bearer ..."`. |
| `/health` OK but every result says "LLM call failed (... 401 ...)" | `VLLM_API_KEY` in the app ≠ `--api-key` given to vLLM. |
| "Model 'X' is not served by vLLM" | The app asked for a name vLLM doesn't serve. Make `DEFAULT_LLM` equal `--served-model-name`. (Unknown names normally fall back to the served model automatically.) |
| vLLM: `CUDA out of memory` at startup | The model doesn't fit. Lower `--max-model-len` (4096), use the AWQ model, or check that nothing else (for example Ollama) holds GPU memory (`nvidia-smi`). |
| vLLM: `No available memory for the cache blocks` | `--gpu-memory-utilization` is too low for FP16 weights. Raise it to 0.90 or switch to AWQ. |
| vLLM: `401 ... gated repo` while downloading | Accept the model terms on huggingface.co and set `HF_TOKEN`. |
| vLLM: `Bfloat16 is only supported on GPUs with compute capability >= 8.0` | Older GPU (T4/V100). Add `--dtype half`. |
| vLLM fails with a CUDA/driver version error | Driver too old for the vLLM wheel. Update the driver (§2) or pin an older vLLM release. |
| `uvicorn` fails to start: `address already in use :8000` | vLLM was started without `--port 8001`. |
| Batch run shows a high `llm_fallback` count | Check the vLLM logs for errors or timeouts. Lower `--concurrency` if requests time out (the client timeout is 120 s per call). |
| Batch throughput is flat when raising `--concurrency` | GPU saturated (`Waiting` > 0 in the vLLM logs). Use AWQ (more KV cache), a smaller `--max-model-len`, or a bigger GPU. |
| Many results say "did not return valid JSON" or "no usable offers array" | The model ignores the schema. Run `python -m nbx.scripts.benchmark_llms` against vLLM and compare against the Ollama benchmark; Qwen 2.5 usually holds the schema better. |

---

## 12. Security notes for a company server

- vLLM listens on **127.0.0.1:8001 only**, with `--api-key`. Never publish it.
- FastAPI currently allows CORS `*` and has **no authentication**. Reach it
  through the SSH tunnel, or put it behind the company reverse proxy or VPN
  with auth before other systems call it.
- Keep `HF_TOKEN` and `VLLM_API_KEY` in `~/.config/vllm/env` (mode 600) or
  `.env.secrets`, **never** in the git-tracked `.env`.
- Real customer data (MSISDNs, revenue) and the nightly
  `batch_scores_*.jsonl` files are sensitive. Restrict permissions on
  `data/`, follow the company retention rules, and don't copy them to
  personal laptops.

---

## 13. Final checklist

- [ ] `nvidia-smi` lists the GPU, driver ≥ 550
- [ ] HF terms accepted and `HF_TOKEN` set (gated models only)
- [ ] `vllm serve ... --served-model-name mistral:7b --port 8001` is up, and `/v1/models` lists the name
- [ ] app venv installed, `pytest -q` passes
- [ ] `.env` on the server has `LLM_BACKEND=vllm` and `DEFAULT_LLM` = the served name; API key provided by systemd/secrets
- [ ] `curl localhost:8000/health` → `"status":"ok"`, `"llm_backend":"vllm"`
- [ ] `python -m nbx.scripts.benchmark_llms` passes the needle cases on the served model
- [ ] `batch_score.py --limit 200` smoke test OK; rate noted and full-run time estimated
- [ ] systemd services enabled, nightly cron installed
- [ ] SSH tunnel works and Gradio opens at http://localhost:7860

---

## 14. Small GPU on Windows (WSL2, GTX 1650, 4 GB)

The FSDR agents run on vLLM on a Windows 11 laptop with an NVIDIA GTX 1650
(4 GB, compute capability 7.5). Two commands set it up; neither needs sudo.

```powershell
python run.py vllm-install     # once: uv + ~/vllm-env in WSL (vllm 0.6.6.post1 and pins)
python run.py vllm-serve       # serves Qwen2.5-3B-Instruct-GPTQ-Int4 as "qwen2.5:3b" on 127.0.0.1:8001
python run.py start            # in a second window; .env has LLM_BACKEND=vllm
```

The scripts are `vllm/install_wsl.sh` and `vllm/serve_wsl.sh`. `vllm-serve` passes
these settings from `.env` into WSL: `VLLM_API_KEY`, `VLLM_HF_MODEL`,
`VLLM_SERVED_NAME`, `VLLM_PORT`, `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_UTIL` and
`VLLM_MAX_SEQS`. Use `--distro` if your distribution is not called `Ubuntu`.

### 14.1 Requirements

- Windows 11 with WSL2 and an Ubuntu distribution (`wsl --install -d Ubuntu`).
- The NVIDIA Windows driver. It includes the WSL CUDA driver, so `nvidia-smi`
  works inside WSL without installing anything there.
- About 10 GB of disk in WSL (the packages and their cache) and 2 GB for the model.

### 14.2 Why these versions and options

| Choice | Reason |
|---|---|
| `vllm==0.6.6.post1` (V0 engine, `VLLM_USE_V1=0`) | still supports compute capability 7.x; newer releases target 8.0+ |
| `transformers==4.47.1` | 4.48+ needs torch 2.6; vLLM 0.6.6 ships torch 2.5.1 |
| `xgrammar==0.1.10` | JSON mode (`response_format`) uses xgrammar; later versions dropped an API vLLM 0.6.6 calls, which crashed the engine at the first JSON request |
| uv-managed Python (`--python-preference only-managed`) | ships `Python.h`; Triton builds a small CUDA helper at the first request, and Ubuntu's Python has no headers without `sudo apt install python3-dev` |
| `ziglang` as `CC` | a C compiler installed from pip; Triton needs one and WSL Ubuntu has no gcc by default |
| **GPTQ-Int4** model, not AWQ | AWQ ran at 4 tokens/s on Turing, GPTQ at 13–14 tokens/s (GPU at 97 % in both cases) |
| `--dtype half` | Turing has no bf16 |
| `--max-model-len 8192`, `--gpu-memory-utilization 0.85`, `--enforce-eager` | about 2 GB of weights plus 0.7 GB of KV cache fit in 4 GB; CUDA graphs would need memory the card does not have |

### 14.3 Checks

```powershell
curl.exe -s http://127.0.0.1:8001/v1/models -H "Authorization: Bearer <VLLM_API_KEY>"   # lists qwen2.5:3b
curl.exe -s http://127.0.0.1:8000/health          # "llm_backend":"vllm", "loaded_models":["qwen2.5:3b"]
python run.py fsdr-decide --msisdn 70094659 --trace   # "mode": "agentic"
```

### 14.4 What to expect

- First start: the model download, then about 2 minutes of memory profiling, before
  `Application startup complete`. Later starts take about 3 minutes.
- One FSDR decision takes 25–150 s (1–7 LLM calls). A batch of 10 customers takes
  8–13 minutes, because the GPU is the limit and concurrency barely helps. See
  [FSDR_AGENTIC_RECOMMENDER.md §1.2](FSDR_AGENTIC_RECOMMENDER.md#12-what-it-costs-on-a-4-gb-gpu-measured).
- In the UI batch, use a batch size of about 5, because each request times out after 900 s.
- The vLLM startup log prints its arguments, including the API key. Keep that log local.
- Stop the server with Ctrl+C, or `wsl.exe -d Ubuntu -- pkill -f "vllm serve"`.

### 14.5 Troubleshooting

| Symptom | Fix |
|---|---|
| `TokenizerInfo has no attribute from_huggingface` | `uv pip install xgrammar==0.1.10` in `~/vllm-env` |
| `Failed to find C compiler` | rerun `python run.py vllm-install` (creates `~/vllm-env/cc/cc`) or install gcc |
| `PyTorch >= 2.6 is required` (transformers) | `uv pip install transformers==4.47.1` |
| decisions show `partial_fallback` with `500` | read the server log; the engine has crashed, so restart `vllm-serve` |
| CUDA out of memory at start | `VLLM_GPU_UTIL=0.80` or `VLLM_MAX_MODEL_LEN=4096` in `.env` |
