"""App-level settings (read from .env): operator, LLM backend, ports, paths, API security.

The FSDR engine's own business constants (BR-01 ... BR-86) are in nbx/fsdr/config.py.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: this file lives in nbx/core/.
BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

# Our operator's name in every prompt, SMS template and UI text.
OPERATOR_NAME = os.getenv("OPERATOR_NAME", "FSDR")

# ---------------------------------------------------------------------------
# LLM backend (llms/llm_client.py). Every backend runs ON THIS MACHINE (see LOCAL_ONLY below):
#   "offline"  -- pure Python, no server (default): every agent takes its rule-based path.
#   "vllm"     -- a vLLM server (GPU, WSL2 / Linux): `python run.py vllm-serve`            -> VLLM_URL
#   "ollama"   -- Ollama (Windows / macOS / Linux, CPU or GPU): `ollama serve`             -> OLLAMA_URL
#   "llamacpp" -- llama.cpp's llama-server with a GGUF file (CPU or GPU)                  -> LLAMACPP_URL
#   "lmstudio" -- LM Studio's local server (Developer tab -> Start server)                 -> LMSTUDIO_URL
# The four servers speak the same OpenAI-compatible API (/v1/models, /v1/chat/completions): one client
# (llms/vllm_client.py) serves them all. No cloud LLM API (OpenAI, Anthropic / Claude, …) is used anywhere.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "offline").lower()
LLM_SERVER_BACKENDS = ["vllm", "ollama", "llamacpp", "lmstudio"]
LLM_BACKENDS = ["offline", *LLM_SERVER_BACKENDS]
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "qwen2.5:3b")
# 127.0.0.1, not "localhost": vLLM binds IPv4 only, and "localhost" tries ::1 first (slow on Windows).
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://127.0.0.1:8080")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")            # ollama / llamacpp / lmstudio, only if you set one there
# Grammar-constrained JSON (response_format json_object) per server; LM Studio only takes json_schema, so the
# agents parse its free text instead (llm_client.extract_json).
LLM_JSON_MODE = {"vllm": True, "ollama": True, "llamacpp": True, "lmstudio": False}


def llm_url(backend: str | None = None) -> str:
    """The URL of the server of a backend (the current one by default); "" for offline."""
    b = (backend or LLM_BACKEND).lower()
    return {"vllm": VLLM_URL, "ollama": OLLAMA_URL, "llamacpp": LLAMACPP_URL, "lmstudio": LMSTUDIO_URL}.get(b, "")


def llm_api_key(backend: str | None = None) -> str:
    return VLLM_API_KEY if (backend or LLM_BACKEND).lower() == "vllm" else LLM_API_KEY


# ---------------------------------------------------------------------------
# V4.0 local only. True (default): the app talks only to servers on this machine (127.0.0.1 / localhost / ::1,
# plus the hosts you list in FSDR_LOCAL_HOSTS, e.g. your own GPU box on the LAN). An LLM URL or a tracing
# endpoint elsewhere is refused (the agents fall back to their rules, tracing stays off), and the usage
# analytics of Gradio / Hugging Face / vLLM are switched off (nbx/__init__.py).
# ---------------------------------------------------------------------------
LOCAL_ONLY = os.getenv("FSDR_LOCAL_ONLY", "true").lower() == "true"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"} | {h.strip().lower() for h in
                                                   os.getenv("FSDR_LOCAL_HOSTS", "").split(",") if h.strip()}


def is_local_url(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in LOCAL_HOSTS or host.startswith("127.")


def local_violations() -> list[str]:
    """What would leave the machine with the current settings (empty = everything local)."""
    out = []
    if LLM_BACKEND in LLM_SERVER_BACKENDS and not is_local_url(llm_url()):
        out.append(f"LLM_BACKEND={LLM_BACKEND} points to {llm_url()} (not this machine)")
    tracing = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() in ("true", "1")
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
    if tracing and not is_local_url(endpoint):
        out.append(f"LANGSMITH_TRACING=true sends traces to {endpoint} (not this machine)")
    return out
# Cap per completion; otherwise one runaway generation holds a batch slot for its whole length.
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "512"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "300"))

# ---------------------------------------------------------------------------
# Ports, paths, logging
# ---------------------------------------------------------------------------
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
UI_HOST = os.getenv("FSDR_UI_HOST", "127.0.0.1")            # 0.0.0.0 in the container (Dockerfile)
API_URL = os.getenv("FSDR_API_URL", "")                      # UI -> API base URL (default http://localhost:<port>)
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API security (review finding #5). With FSDR_API_KEY set, every /fsdr/* route needs the
# header `X-API-Key: <key>` (the UI sends it). Empty = open, for local development only: the API then binds
# 127.0.0.1 by default. Model promotion that bypasses the IPS rule needs FSDR_ADMIN_KEY as well.
# ---------------------------------------------------------------------------
API_KEY = os.getenv("FSDR_API_KEY", "")
ADMIN_KEY = os.getenv("FSDR_ADMIN_KEY", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("FSDR_CORS_ORIGINS", f"http://localhost:{GRADIO_PORT},"
                                             f"http://127.0.0.1:{GRADIO_PORT}").split(",") if o.strip()]

# ---------------------------------------------------------------------------
