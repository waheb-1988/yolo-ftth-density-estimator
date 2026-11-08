# FSDR recommender (API + web UI), offline LLM backend by default. Review #16.
#   docker build -t fsdr-recommender .
#   docker run -p 8000:8000 -p 7860:7860 -e FSDR_API_KEY=<long random key> -v fsdr-data:/data fsdr-recommender
# The vLLM server stays outside the container (GPU host): set LLM_BACKEND=vllm, VLLM_URL and VLLM_API_KEY.
FROM python:3.12-slim

# libgomp: OpenMP runtime of implicit (BPR) and Vowpal Wabbit
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# dependencies first (cached layer), exactly as locked
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY run.py ./
COPY nbx ./nbx
COPY data/competitors/competitor_offers.json data/competitors/competitor_passages.json data/competitors/sources.example.json ./data/competitors/

ENV LLM_BACKEND=offline \
    FSDR_DATA_DIR=/data/fsdr \
    FSDR_UI_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home fsdr && mkdir -p /data/fsdr && chown -R fsdr /data /app
USER fsdr
VOLUME /data
EXPOSE 8000 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["/app/.venv/bin/python", "run.py", "start", "--host", "0.0.0.0"]
