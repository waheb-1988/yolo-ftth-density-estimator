"""Readiness/health computation. Shared by GET /health and the ingestion
pipeline's readiness-check step (agents/ingestion_pipeline.py) so both report
exactly the same thing instead of two copies of the same check drifting apart.
"""
from __future__ import annotations

from nbx.core import config
from nbx.llms import llm_client
from nbx.vectorstores import try_get_store


async def compute_health() -> dict:
    llm_ok = await llm_client.is_available()
    installed = await llm_client.list_installed_models() if llm_ok else []

    vs_status = {}
    for cfg in config.VECTOR_STORES:
        store = try_get_store(cfg["key"])
        vs_status[cfg["key"]] = "ok" if store is not None else "unavailable"

    return {
        "status": "ok" if llm_ok else "degraded",
        # Field name predates LLM_BACKEND; it means "the configured LLM
        # backend is reachable" (Ollama or vLLM). Kept so the Gradio and
        # Next.js clients that read it don't break.
        "ollama_connected": llm_ok,
        "llm_backend": llm_client.backend_name(),
        "llm_backend_url": llm_client.backend_url(),
        "vectorstores": vs_status,
        "loaded_models": installed,
    }
