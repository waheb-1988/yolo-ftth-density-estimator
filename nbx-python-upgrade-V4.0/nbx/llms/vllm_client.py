"""Thin async wrapper around a LOCAL OpenAI-compatible LLM server: vLLM, Ollama, llama.cpp (llama-server) or
LM Studio (config.LLM_BACKEND picks the URL, config.llm_url). With config.LOCAL_ONLY (default) a server that is
not on this machine is refused: no prompt ever leaves it. No cloud LLM API is called anywhere.

Same interface as llms/offline_client.py (is_available /
list_installed_models / model_is_installed / resolve_model / generate) so
llms/llm_client.py can swap the two without the agents noticing.

vLLM serves one model per process, under whatever names were passed to
`vllm serve ... --served-model-name`. Serving the HF weights under this
repo's Ollama-style name (e.g. `--served-model-name mistral:7b`) keeps
DEFAULT_LLM, the model roster and every UI dropdown working unchanged --
see docs/GPU_SETUP.md.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from nbx.core import config

DEFAULT_TIMEOUT = 120.0


# One pooled client per event loop instead of one per call: constructing an
# httpx.AsyncClient builds an SSL context synchronously (~0.3 s on Windows),
# which blocks the event loop -- with 2-3 LLM calls per customer that alone
# capped batch scoring at ~1.6 customers/s no matter the --concurrency.
# Keyed by loop because a client can't be reused across asyncio.run() calls.
_client_cache: tuple[asyncio.AbstractEventLoop, httpx.AsyncClient] | None = None


def _client() -> httpx.AsyncClient:
    global _client_cache
    loop = asyncio.get_running_loop()
    if _client_cache is None or _client_cache[0] is not loop or _client_cache[1].is_closed:
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=128)
        _client_cache = (loop, httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=limits))
    return _client_cache[1]


def _backend() -> str:
    b = config.LLM_BACKEND
    return b if b in config.LLM_SERVER_BACKENDS else "vllm"


NAMES = {"vllm": "vLLM", "ollama": "Ollama", "llamacpp": "llama.cpp", "lmstudio": "LM Studio"}


def _name() -> str:
    return NAMES[_backend()]


def _url() -> str:
    return config.llm_url(_backend())


def _refused() -> str | None:
    """V4.0 local only: the reason the current URL is refused, else None."""
    url = _url()
    if config.LOCAL_ONLY and not config.is_local_url(url):
        return (f"FSDR_LOCAL_ONLY: {url} is not on this machine; refused (set the server URL to 127.0.0.1, or add "
                f"your own host to FSDR_LOCAL_HOSTS)")
    return None


def _headers() -> dict:
    key = config.llm_api_key(_backend())
    return {"Authorization": f"Bearer {key}"} if key else {}


async def is_available() -> bool:
    if _refused():
        return False
    try:
        resp = await _client().get(f"{_url()}/v1/models", headers=_headers(), timeout=5.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def list_installed_models() -> list[str]:
    """Every name the server answers to: its --served-model-name aliases (vLLM), its pulled tags (Ollama), …"""
    if _refused():
        return []
    try:
        resp = await _client().get(f"{_url()}/v1/models", headers=_headers(), timeout=5.0)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except httpx.HTTPError:
        return []


def model_is_installed(installed: list[str], model: str) -> bool:
    # Exact match only: unlike Ollama tags, "mistral:7b" and "mistral:latest"
    # are not the same weights here, they're just whatever names were served.
    return model in installed


# resolve_model runs before every chat call; the served model list only
# changes when vLLM restarts, so don't spend a round trip on it per call.
_SERVED_CACHE_TTL_S = 30.0
_served_cache: tuple[float, list[str]] = (0.0, [])


async def _served_models_cached() -> list[str]:
    global _served_cache
    fetched_at, models = _served_cache
    if models and time.monotonic() - fetched_at < _SERVED_CACHE_TTL_S:
        return models
    models = await list_installed_models()
    if models:  # never cache an outage
        _served_cache = (time.monotonic(), models)
    return models


async def resolve_model(model: str) -> str:
    """Falls back to the only served model when the requested one isn't
    served -- vLLM has exactly one model loaded, so a request naming some
    other roster entry (e.g. a stale UI dropdown) still gets an answer
    instead of a 404 per customer in a nightly batch."""
    installed = await _served_models_cached()
    if model in installed or not installed:
        return model
    return installed[0]


async def generate(
    model: str,
    prompt: str,
    system: str | None = None,
    force_json: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Calls /v1/chat/completions (non-streaming) and returns the same
    {"content", "duration_ms", "model", "error"} shape as offline_client."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": config.VLLM_MAX_TOKENS,
    }
    if force_json and config.LLM_JSON_MODE.get(_backend(), True):
        # vLLM's equivalent of Ollama's format=json: grammar-constrained
        # decoding to syntactically valid JSON (not to our schema -- the
        # agents' own retry handles schema mismatches, same as with Ollama).
        payload["response_format"] = {"type": "json_object"}

    start = time.time()

    def _result(content: str = "", error: str | None = None) -> dict:
        return {
            "content": content,
            "duration_ms": int((time.time() - start) * 1000),
            "model": model,
            "error": error,
        }

    refused = _refused()
    if refused:
        return _result(error=refused)
    try:
        resp = await _client().post(
            f"{_url()}/v1/chat/completions", json=payload, headers=_headers(), timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or [{}]
        return _result(content=choices[0].get("message", {}).get("content") or "")
    except httpx.ConnectError:
        return _result(error=f"{_name()} is not reachable at {_url()}.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            error = (f"Model '{model}' is not served by {_name()} at {_url()}. Check the served name "
                     f"(vLLM --served-model-name, `ollama list`, ...) and DEFAULT_LLM.")
        else:
            error = f"{_name()} chat call failed for '{model}': {e.response.status_code} {e.response.text[:300]}"
        return _result(error=error)
    except httpx.HTTPError as e:
        return _result(error=str(e) or type(e).__name__)
