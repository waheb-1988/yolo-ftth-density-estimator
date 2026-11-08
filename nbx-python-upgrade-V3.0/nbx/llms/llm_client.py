"""Backend-agnostic LLM client: dispatches every chat call to the offline or the vLLM backend according to
config.LLM_BACKEND. The agents go through this module, never through a backend module directly.

The backend is read at call time, not import time, so tests (and a running process whose config was patched)
switch backends without re-importing.
"""
from __future__ import annotations

import json
import re
from types import ModuleType

from nbx.core import config, tracing
from nbx.llms import offline_client, vllm_client

DEFAULT_TIMEOUT = config.LLM_TIMEOUT_S


def backend_name() -> str:
    if config.LLM_BACKEND not in config.LLM_BACKENDS:
        raise ValueError(f"LLM_BACKEND must be one of {config.LLM_BACKENDS}, got '{config.LLM_BACKEND}'.")
    return config.LLM_BACKEND


def is_offline() -> bool:
    return backend_name() == "offline"


def backend_url() -> str:
    return "in-process" if is_offline() else config.VLLM_URL


def _backend() -> ModuleType:
    return {"offline": offline_client, "vllm": vllm_client}[backend_name()]


async def is_available() -> bool:
    return await _backend().is_available()


async def list_installed_models() -> list[str]:
    return await _backend().list_installed_models()


def model_is_installed(installed: list[str], model: str) -> bool:
    return _backend().model_is_installed(installed, model)


async def resolve_model(model: str) -> str:
    return await _backend().resolve_model(model)


async def resolve_or_none(llm: str | None) -> str | None:
    """The model the agents should use, or None for the rule-based path (offline backend or server down)."""
    if is_offline() or not await is_available():
        return None
    return await resolve_model(llm or config.DEFAULT_LLM)


async def generate(model: str, prompt: str, system: str | None = None, force_json: bool = False,
                   timeout: float = DEFAULT_TIMEOUT) -> dict:
    with tracing.span(f"llm:{model}", "llm", inputs={"messages": [{"role": "system", "content": system or ""},
                                                                  {"role": "user", "content": prompt}]},
                      metadata={"ls_provider": backend_name(), "ls_model_name": model, "force_json": force_json}) as sp:
        out = await _backend().generate(model=model, prompt=prompt, system=system, force_json=force_json, timeout=timeout)
        sp.end(outputs={"content": out.get("content"), "duration_ms": out.get("duration_ms")}, error=out.get("error"))
        return out


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction: a direct parse, then the first {...} block in surrounding prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
