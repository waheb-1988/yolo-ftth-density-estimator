"""Thin async wrapper around the local Ollama HTTP API."""
from __future__ import annotations

import asyncio
import json
import re
import time

import httpx

from nbx.core import config

DEFAULT_TIMEOUT = 120.0


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config.OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def list_installed_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config.OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except httpx.HTTPError:
        return []


def _model_is_installed(installed: list[str], model: str) -> bool:
    # Ollama tags often include a digest suffix; match on the name prefix too.
    return any(m == model or m.startswith(f"{model}:") or model.startswith(m.split(":")[0]) for m in installed)


def match_installed_name(installed: list[str], model: str) -> str | None:
    """Returns the actual installed tag matching a requested model name
    (e.g. requested 'llama3.1:8b' matches installed 'llama3.1:latest')."""
    for m in installed:
        if m == model:
            return m
    for m in installed:
        if m.split(":")[0] == model.split(":")[0]:
            return m
    return None


async def resolve_model(model: str) -> str:
    """Resolves a configured model name to whatever tag is actually pulled
    in the local Ollama instance; falls back to the requested name unchanged
    if no match is found (Ollama will then surface a clear 'not found' error)."""
    installed = await list_installed_models()
    return match_installed_name(installed, model) or model


async def generate(
    model: str,
    prompt: str,
    system: str | None = None,
    force_json: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Calls Ollama's /api/chat endpoint (non-streaming) and returns
    {"content": str, "duration_ms": int, "model": str, "error": str | None}."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if force_json:
        payload["format"] = "json"

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{config.OLLAMA_URL}/api/chat", json=payload)
            if resp.status_code >= 500:
                # Ollama answers 500 when its runner crashes while (re)loading the
                # model (seen with CUDA init failures on small GPUs); the next
                # load usually succeeds, so retry once before giving up.
                await asyncio.sleep(2)
                resp = await client.post(f"{config.OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        return {
            "content": content,
            "duration_ms": int((time.time() - start) * 1000),
            "model": model,
            "error": None,
        }
    except httpx.ConnectError:
        return {
            "content": "",
            "duration_ms": int((time.time() - start) * 1000),
            "model": model,
            "error": f"Ollama is not reachable at {config.OLLAMA_URL}.",
        }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            error = f"Model '{model}' is not pulled in Ollama yet. Run: ollama pull {model}"
        else:
            error = f"Ollama chat call failed for '{model}': {e}"
        return {
            "content": "",
            "duration_ms": int((time.time() - start) * 1000),
            "model": model,
            "error": error,
        }
    except httpx.HTTPError as e:
        return {
            "content": "",
            "duration_ms": int((time.time() - start) * 1000),
            "model": model,
            "error": str(e),
        }


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction: try a direct parse, then fall back to
    pulling the first {...} block out of surrounding prose."""
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
