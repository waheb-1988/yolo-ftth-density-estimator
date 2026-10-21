"""Runs the full NBO agent pipeline across all available Ollama models on the
same customer and ranks them by average final recommendation score."""
from __future__ import annotations

from nbx.core import config
from nbx.llms import llm_client


async def get_models_with_availability() -> list[dict]:
    installed = await llm_client.list_installed_models()
    out = []
    for m in config.OLLAMA_MODELS:
        out.append({**m, "available": llm_client.model_is_installed(installed, m["name"])})
    return out


async def run_benchmark(msisdn: int, vectorstore: str | None = None) -> dict:
    from nbx.agents.nbo_agent import run_pipeline  # lazy import avoids import-order coupling

    models_info = await get_models_with_availability()
    candidate_models = [m["name"] for m in models_info if m["available"]] or [config.DEFAULT_LLM]

    results = []
    for model_name in candidate_models:
        try:
            resp = await run_pipeline(msisdn=msisdn, llm=model_name, vectorstore=vectorstore)
        except Exception:
            continue

        recs = resp["recommendations"]
        avg_score = round(sum(r["final_score"] for r in recs) / len(recs), 4) if recs else 0.0
        results.append(
            {
                "llm": model_name,
                "detected_need": resp["detected_need"],
                "recommendations": recs,
                "avg_final_score": avg_score,
                "pipeline_time_ms": resp["pipeline_time_ms"],
                "valid_json": bool(recs),
            }
        )

    ranking = sorted(results, key=lambda r: r["avg_final_score"], reverse=True)
    winner = ranking[0]["llm"] if ranking else ""

    return {
        "msisdn": msisdn,
        "results": results,
        "winner": winner,
        "ranking": [r["llm"] for r in ranking],
    }
