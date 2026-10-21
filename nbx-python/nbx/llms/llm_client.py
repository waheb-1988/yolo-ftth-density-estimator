"""Backend-agnostic LLM client: dispatches every chat call to the offline,
Ollama or vLLM backend according to config.LLM_BACKEND. Agents, readiness,
the benchmark and /models go through this module, never through a backend
module directly.

The backend is read at call time, not import time, so tests (and a running
process whose config was patched) switch backends without re-importing.
"""
from __future__ import annotations

from types import ModuleType

from nbx.core import config
from nbx.llms import offline_client, ollama_client, vllm_client

DEFAULT_TIMEOUT = config.LLM_TIMEOUT_S


def backend_name() -> str:
    if config.LLM_BACKEND not in config.LLM_BACKENDS:
        raise ValueError(f"LLM_BACKEND must be one of {config.LLM_BACKENDS}, got '{config.LLM_BACKEND}'.")
    return config.LLM_BACKEND


def is_offline() -> bool:
    return backend_name() == "offline"


def backend_url() -> str:
    name = backend_name()
    if name == "offline":
        return "in-process"
    return config.VLLM_URL if name == "vllm" else config.OLLAMA_URL


def _backend() -> ModuleType:
    return {"offline": offline_client, "vllm": vllm_client}.get(backend_name(), ollama_client)


async def is_available() -> bool:
    return await _backend().is_available()


async def list_installed_models() -> list[str]:
    return await _backend().list_installed_models()


def model_is_installed(installed: list[str], model: str) -> bool:
    name = backend_name()
    if name == "offline":
        return offline_client.model_is_installed(installed, model)
    if name == "vllm":
        return vllm_client.model_is_installed(installed, model)
    return ollama_client._model_is_installed(installed, model)


async def resolve_model(model: str) -> str:
    return await _backend().resolve_model(model)


async def generate(
    model: str,
    prompt: str,
    system: str | None = None,
    force_json: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    return await _backend().generate(model=model, prompt=prompt, system=system, force_json=force_json, timeout=timeout)


# Backend-independent: parses whatever text either server returned.
extract_json = ollama_client.extract_json
