"""Nightly batch scoring: runs the full decision for every customer and
writes one JSON line per msisdn.

Built for LLM_BACKEND=vllm. vLLM batches concurrent requests on the GPU
(continuous batching), so throughput comes from keeping --concurrency
requests in flight, not from a separate offline code path. Each customer
goes through exactly the same pipeline as POST /nba/decide (or POST
/recommend with --mode nbo), so batch and online results never drift
apart. It also runs against Ollama, just far slower.

Resumable: msisdns already present in the output file are skipped, so a
crashed or interrupted run continues where it stopped.

Run directly:
    python -m nbx.scripts.batch_score                         # all customers, NBA+NBO
    python -m nbx.scripts.batch_score --mode nbo --concurrency 64
    python -m nbx.scripts.batch_score --limit 200 --output data/batch_test.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nbx.core import config
from nbx.llms import llm_client
from nbx.agents.final_agent import decide_next_best_action
from nbx.agents.nbo_agent import run_pipeline

LLM_FAILED_PREFIX = "LLM call failed"


def _already_scored(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a crash mid-write; that msisdn is re-scored
            if row.get("status") == "ok":
                done.add(row["msisdn"])
    return done


def _llm_fell_back(mode: str, result: dict) -> bool:
    """The pipelines don't raise on LLM failure -- they fall back to the
    CF/graph/bandit signal and say so in llm_reasoning. Count those so a
    nightly run that silently scored everything without the LLM is visible."""
    reasoning = result.get("nbo", {}).get("llm_reasoning", "") if mode == "nba" else result.get("llm_reasoning", "")
    return reasoning.startswith(LLM_FAILED_PREFIX)


async def _score_one(mode: str, msisdn: int, llm: str | None) -> dict:
    if mode == "nba":
        return await decide_next_best_action(msisdn, llm=llm)
    return await run_pipeline(msisdn=msisdn, llm=llm)


async def main(mode: str, concurrency: int, output: Path, limit: int | None, llm: str | None) -> None:
    if not await llm_client.is_available():
        print(f"ERROR: {llm_client.backend_name()} not reachable at {llm_client.backend_url()}.")
        sys.exit(1)

    customers = json.loads(config.CUSTOMERS_FILE.read_text(encoding="utf-8"))
    msisdns = [c["msisdn"] for c in customers]
    if limit:
        msisdns = msisdns[:limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    done = _already_scored(output)
    todo = [m for m in msisdns if m not in done]
    print(f"backend={llm_client.backend_name()} model={llm or config.DEFAULT_LLM} mode={mode} "
          f"concurrency={concurrency}\n{len(msisdns)} customers, {len(done)} already scored, {len(todo)} to go "
          f"-> {output}")
    if not todo:
        return

    counts = {"ok": 0, "error": 0, "llm_fallback": 0}
    start = time.time()

    with output.open("a", encoding="utf-8") as out:

        async def _run(msisdn: int) -> None:
            row: dict = {"msisdn": msisdn, "scored_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            try:
                result = await _score_one(mode, msisdn, llm)
                row.update(status="ok", llm_fallback=_llm_fell_back(mode, result), result=result)
                counts["ok"] += 1
                counts["llm_fallback"] += row["llm_fallback"]
            except Exception as e:  # one bad customer must not kill a 500k-row night
                row.update(status="error", error=f"{type(e).__name__}: {e}")
                counts["error"] += 1
            # Single event loop thread: whole-line writes can't interleave.
            out.write(json.dumps(row, default=str) + "\n")
            out.flush()

            n = counts["ok"] + counts["error"]
            if n % 100 == 0 or n == len(todo):
                rate = n / max(time.time() - start, 1e-9)
                eta_min = (len(todo) - n) / rate / 60 if rate else 0
                print(f"  {n}/{len(todo)}  ok={counts['ok']} error={counts['error']} "
                      f"llm_fallback={counts['llm_fallback']}  {rate:.1f}/s  eta {eta_min:.1f} min")

        # Warm-up: the first call builds the in-process caches (graph, CF
        # model, bandit). Run it alone so concurrent first calls don't each
        # build them in parallel.
        await _run(todo[0])

        sem = asyncio.Semaphore(concurrency)

        async def _bounded(msisdn: int) -> None:
            async with sem:
                await _run(msisdn)

        await asyncio.gather(*(_bounded(m) for m in todo[1:]))

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} min: ok={counts['ok']} error={counts['error']} "
          f"llm_fallback={counts['llm_fallback']} -> {output}")
    if counts["llm_fallback"]:
        print("WARNING: some customers were scored without the LLM step (see llm_fallback=true rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-score every customer (nightly job).")
    parser.add_argument("--mode", choices=["nba", "nbo"], default="nba",
                        help="nba = full NBA+NBO decision (POST /nba/decide); nbo = offers only (POST /recommend)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="requests kept in flight; raise it until vLLM's GPU KV cache usage nears 100%%")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL output (default: data/batch_scores_<mode>_<YYYY-MM-DD>.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N customers (smoke test)")
    parser.add_argument("--llm", default=None, help="model name (default: config.DEFAULT_LLM)")
    args = parser.parse_args()

    output = args.output or config.DATA_DIR / f"batch_scores_{args.mode}_{dt.date.today().isoformat()}.jsonl"
    asyncio.run(main(args.mode, args.concurrency, output, args.limit, args.llm))
