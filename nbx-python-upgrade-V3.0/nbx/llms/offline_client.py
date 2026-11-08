"""Offline LLM backend: pure Python, no server, no model download.

Selected with LLM_BACKEND=offline (the default in nbx-python). It reports
itself as available, so every UI tab and endpoint is enabled, but it never
generates text: generate() returns an error immediately, and every agent then
takes the deterministic path it already has for an unreachable LLM (the
council's rule policies, the bandit's own pick, CF/graph ranking, template
messages and explanations). The council and copilot skip the LLM entirely in
this mode, so their runs are reported as mode="fallback".

Use it to run and test the whole app with Python libraries only; switch to
"vllm" for real LLM reasoning.
"""
from __future__ import annotations

OFFLINE_NOTE = "offline mode (LLM_BACKEND=offline): deterministic agents, no LLM"


async def is_available() -> bool:
    return True


async def list_installed_models() -> list[str]:
    return []


def model_is_installed(installed: list[str], model: str) -> bool:
    return False


async def resolve_model(model: str) -> str:
    return model


async def generate(
    model: str,
    prompt: str,
    system: str | None = None,
    force_json: bool = False,
    timeout: float = 0.0,
) -> dict:
    return {"content": "", "duration_ms": 0, "model": model, "error": OFFLINE_NOTE}
