"""App-level settings (read from .env): operator, LLM backend, ports, paths, API security, competitors.

The FSDR engine's own business constants (BR-01 ... BR-86) are in nbx/fsdr/config.py.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: this file lives in nbx/core/.
BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

# Our operator's name in every prompt, SMS template and UI text. Competitors are display labels only.
OPERATOR_NAME = os.getenv("OPERATOR_NAME", "FSDR")
COMPETITOR_LABELS = ["COMPETITOR_A", "COMPETITOR_B"]

# ---------------------------------------------------------------------------
# LLM backend (llms/llm_client.py):
#   "offline" -- pure Python, no server (default): every agent takes its rule-based path.
#   "vllm"    -- a vLLM server (OpenAI-compatible API), e.g. `python run.py vllm-serve` in WSL2.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "offline").lower()
LLM_BACKENDS = ["offline", "vllm"]
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "qwen2.5:3b")
# 127.0.0.1, not "localhost": vLLM binds IPv4 only, and "localhost" tries ::1 first (slow on Windows).
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
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
# API security (review finding #5). With FSDR_API_KEY set, every /fsdr/* and /competitors/* route needs the
# header `X-API-Key: <key>` (the UI sends it). Empty = open, for local development only: the API then binds
# 127.0.0.1 by default. Model promotion that bypasses the IPS rule needs FSDR_ADMIN_KEY as well.
# ---------------------------------------------------------------------------
API_KEY = os.getenv("FSDR_API_KEY", "")
ADMIN_KEY = os.getenv("FSDR_ADMIN_KEY", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("FSDR_CORS_ORIGINS", f"http://localhost:{GRADIO_PORT},"
                                             f"http://127.0.0.1:{GRADIO_PORT}").split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Competitor intelligence (core/competitors.py): the anonymised snapshot of the two competitors' public plans
# (COMPETITOR_A / COMPETITOR_B). The pages to scrape live OUTSIDE the code, in an optional local file
# (git-ignored), see data/competitors/sources.example.json:
#   {"api":   [{"operator": "COMPETITOR_A", "url": "..."}],          portal-plans JSON endpoints
#    "api_categories": {"<categoryCode prefix>": ["Bundle", "prepaid", "mass"], ...},
#    "pages": [{"operator": "COMPETITOR_B", "page_type": "prepaid", "url": "...",
#               "parser": "plan_cards" | "addon_tables"}]}
# Without the file the app uses the shipped snapshot and refresh has nothing to fetch.
# ---------------------------------------------------------------------------
COMPETITOR_DIR = DATA_DIR / "competitors"
COMPETITOR_OFFERS_FILE = COMPETITOR_DIR / "competitor_offers.json"
COMPETITOR_PASSAGES_FILE = COMPETITOR_DIR / "competitor_passages.json"
COMPETITOR_USER_AGENT = "FSDR-competitor-research/1.0 (+public price pages; contact: CVM analytics team)"
COMPETITOR_REQUEST_DELAY_S = 1.5
COMPETITOR_SOURCES_FILE = Path(os.getenv("COMPETITOR_SOURCES_FILE", str(COMPETITOR_DIR / "sources.json")))


def _competitor_sources() -> dict:
    try:
        return json.loads(COMPETITOR_SOURCES_FILE.read_text(encoding="utf-8")) if COMPETITOR_SOURCES_FILE.exists() else {}
    except (OSError, ValueError):
        return {}


_SOURCES = _competitor_sources()
COMPETITOR_API_SOURCES = [(s["operator"], s["url"]) for s in _SOURCES.get("api", [])]
PORTAL_API_CATEGORIES = {k: tuple(v) for k, v in _SOURCES.get("api_categories", {}).items()}   # code prefix -> (category, page_type, segment)
COMPETITOR_SOURCES = [(s["operator"], s["page_type"], s["url"], s.get("parser", "addon_tables")) for s in _SOURCES.get("pages", [])]
