"""Needle-in-haystack LLM benchmark for the NBO agent pipeline.

Runs every installed Ollama model against three pinned customers whose
behavioural signature deterministically triggers a specific need signal
(see core/data_simulator.py):

    msisdn=1  -> REACTIVATION
    msisdn=15 -> INTERNATIONAL
    msisdn=25 -> RENEWAL

For each (model, msisdn) pair it checks that the pipeline: detects the
expected need, returns valid structured JSON, and (for INTERNATIONAL /
ROAMING) surfaces at least one recommendation from the matching offer
category. Results are ranked by average final score and printed as a
table; the full detail is written to data/benchmark_results.json.

Run directly:  python -m nbx.scripts.benchmark_llms
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nbx.core import config
from nbx.llms import llm_client
from nbx.agents.nbo_agent import run_pipeline

NEEDLE_CASES = [
    {"msisdn": 1, "expected_need": "REACTIVATION", "expected_category": None},
    {"msisdn": 15, "expected_need": "INTERNATIONAL", "expected_category": "International"},
    {"msisdn": 25, "expected_need": "RENEWAL", "expected_category": None},
]

RESULTS_FILE = config.DATA_DIR / "benchmark_results.json"


def _print_table(rows: list[dict], columns: list[str]) -> None:
    """Minimal dependency-free table printer (avoids pandas, which segfaults
    in this environment when loaded alongside torch/sentence-transformers)."""
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(col), max(len(str(r.get(col, ""))) for r in rows)) for col in columns]
    header = "  ".join(col.ljust(w) for col, w in zip(columns, widths))
    print(header)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r.get(col, "")).ljust(w) for col, w in zip(columns, widths)))


async def _run_case(model: str, case: dict, vectorstore: str) -> dict:
    start = time.time()
    try:
        resp = await run_pipeline(msisdn=case["msisdn"], llm=model, vectorstore=vectorstore)
    except Exception as e:
        return {
            "model": model,
            "msisdn": case["msisdn"],
            "expected_need": case["expected_need"],
            "detected_need": None,
            "need_correct": False,
            "valid_json": False,
            "category_hit": False,
            "avg_final_score": 0.0,
            "pipeline_time_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }

    recs = resp["recommendations"]
    avg_score = round(sum(r["final_score"] for r in recs) / len(recs), 4) if recs else 0.0
    category_hit = True
    if case["expected_category"]:
        category_hit = any(r["category"] == case["expected_category"] for r in recs)

    return {
        "model": model,
        "msisdn": case["msisdn"],
        "expected_need": case["expected_need"],
        "detected_need": resp["detected_need"],
        "need_correct": resp["detected_need"] == case["expected_need"],
        "valid_json": bool(recs),
        "category_hit": category_hit,
        "avg_final_score": avg_score,
        "pipeline_time_ms": resp["pipeline_time_ms"],
        "error": None,
    }


async def main(vectorstore: str | None = None) -> None:
    store_key = vectorstore or config.DEFAULT_VECTORSTORE

    if not await llm_client.is_available():
        print(f"ERROR: {llm_client.backend_name()} not reachable at {llm_client.backend_url()}. "
              "Start it with: ollama serve (or vllm serve when LLM_BACKEND=vllm)")
        sys.exit(1)

    installed = await llm_client.list_installed_models()
    candidate_models = [m["name"] for m in config.OLLAMA_MODELS if llm_client.model_is_installed(installed, m["name"])]
    if not candidate_models:
        print(f"ERROR: none of the configured models are available on {llm_client.backend_name()}. "
              "Run e.g. `ollama pull mistral:7b`, or serve one under its roster name with "
              "`vllm serve <hf-model> --served-model-name mistral:7b`.")
        sys.exit(1)

    print(f"Benchmarking {len(candidate_models)} installed model(s) against {len(NEEDLE_CASES)} needle case(s) "
          f"on vector store '{store_key}':\n  models: {candidate_models}\n")

    all_results: list[dict] = []
    for model in candidate_models:
        for case in NEEDLE_CASES:
            print(f"  running {model} @ msisdn={case['msisdn']} (expect {case['expected_need']})...")
            result = await _run_case(model, case, store_key)
            all_results.append(result)

    per_model: dict[str, dict] = {}
    for model in candidate_models:
        rows = [r for r in all_results if r["model"] == model]
        n = len(rows)
        per_model[model] = {
            "need_accuracy": round(sum(r["need_correct"] for r in rows) / n, 3),
            "valid_json_rate": round(sum(r["valid_json"] for r in rows) / n, 3),
            "category_hit_rate": round(sum(r["category_hit"] for r in rows) / n, 3),
            "avg_final_score": round(sum(r["avg_final_score"] for r in rows) / n, 4),
            "avg_pipeline_time_ms": round(sum(r["pipeline_time_ms"] for r in rows) / n, 1),
        }
    ranking = sorted(per_model, key=lambda m: per_model[m]["avg_final_score"], reverse=True)

    print("\n=== Per-case results ===")
    _print_table(all_results, ["model", "msisdn", "expected_need", "detected_need", "need_correct",
                                "valid_json", "category_hit", "avg_final_score", "pipeline_time_ms"])

    print("\n=== Ranking (best -> worst) ===")
    _print_table(
        [{"model": m, **per_model[m]} for m in ranking],
        ["model", "need_accuracy", "valid_json_rate", "category_hit_rate", "avg_final_score", "avg_pipeline_time_ms"],
    )

    winner = ranking[0] if ranking else None
    print(f"\nWinner: {winner}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "vectorstore": store_key,
                "cases": NEEDLE_CASES,
                "results": all_results,
                "per_model": per_model,
                "ranking": ranking,
                "winner": winner,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results written to {RESULTS_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Needle-in-haystack benchmark across installed Ollama models.")
    parser.add_argument("--vectorstore", default=None, choices=config.VECTOR_STORE_KEYS, help="Vector store to use (default: config.DEFAULT_VECTORSTORE)")
    args = parser.parse_args()

    asyncio.run(main(vectorstore=args.vectorstore))
