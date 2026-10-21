"""Generate synthetic Ooredoo data and index it into all 5 local vector stores.

Run directly:  python -m nbx.scripts.seed_data
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nbx.core import collab_filter, config, nba_bandit
from nbx.core.data_simulator import save_all


def _load_store(store_cfg: dict, embedding_model: str | None = None):
    module = importlib.import_module(store_cfg["module"])
    cls = getattr(module, store_cfg["class_name"])
    return cls(embedding_model=embedding_model)


def build_documents() -> list[dict]:
    """Builds the corpus of documents to embed: offers, segments, purchase patterns."""
    offers = json.loads(config.OFFERS_FILE.read_text())
    segments = json.loads(config.SEGMENTS_FILE.read_text())
    purchases = json.loads(config.PURCHASES_FILE.read_text())
    customers = json.loads(config.CUSTOMERS_FILE.read_text())

    docs: list[dict] = []

    # Offer documents
    for offer in offers:
        text = (
            f"{offer['offer_name']} ({offer['category']}, {offer['offer_type']}): "
            f"{offer['description']} Price {offer['price_omr']} OMR, data {offer['data_gb']}, "
            f"voice {offer['voice_mins']} minutes, validity {offer['validity_days']} days. "
            f"is_data={offer['is_data']} is_voice={offer['is_voice']} is_intl={offer['is_intl']} "
            f"is_roaming={offer['is_roaming']}."
        )
        docs.append(
            {
                "id": f"offer::{offer['program_rule']}",
                "text": text,
                "type": "offer",
                "metadata": offer,
            }
        )

    # Segment documents
    for seg in segments:
        text = f"Segment '{seg['segment']}': {seg['typical_behavior']}"
        docs.append(
            {
                "id": f"segment::{seg['segment']}",
                "text": text,
                "type": "segment",
                "metadata": seg,
            }
        )

    # Purchase pattern documents: aggregate "customers like X buy Y"
    customer_by_msisdn = {c["msisdn"]: c for c in customers}
    pattern_counts: dict[tuple[str, str], int] = {}
    for p in purchases:
        c = customer_by_msisdn.get(p["msisdn"])
        if not c:
            continue
        key = (c["dv_segment"], p["category"])
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    for (dv_segment, category), count in pattern_counts.items():
        text = (
            f"Customers in segment '{dv_segment}' frequently purchase '{category}' offers "
            f"({count} purchases observed). This indicates strong affinity between "
            f"'{dv_segment}' customers and '{category}' category offers."
        )
        docs.append(
            {
                "id": f"pattern::{dv_segment}::{category}",
                "text": text,
                "type": "pattern",
                "metadata": {"dv_segment": dv_segment, "category": category, "count": count},
            }
        )

    return docs


def seed_all(embedding_model: str | None = None) -> dict:
    """Regenerates the synthetic data and indexes it into all 5 vector stores
    under the given embedding model (default: config.DEFAULT_EMBEDDING_MODEL,
    i.e. MiniLM — CPU-only, no Ollama dependency, so this stays fast and
    always runnable). To also index one of the Qwen3-Embedding roster
    entries, call seed_embedding_model() once Ollama is running with that
    model pulled — it indexes the same documents into a parallel collection
    per store without touching the MiniLM one."""
    stats = save_all()
    print(f"Data generated: {stats}")

    cf_model = collab_filter.get_model(force_rebuild=True)
    print(f"Collaborative-filtering model trained: {collab_filter.get_stats(cf_model)}")

    nba_bandit.get_bandit(force_rebuild=True)
    print(f"NBA contextual bandit trained: {nba_bandit.get_stats()}")

    indexed = index_embedding_model(embedding_model)

    return {
        "customers": stats["customers"],
        "offers": stats["offers"],
        "purchases": stats["purchases"],
        "indexed": indexed,
    }


def index_embedding_model(embedding_model: str | None = None, progress_cb=None) -> dict:
    """Indexes the current data files into all 5 stores for one embedding
    model, without regenerating the data. Used both by seed_all() (default
    model) and directly for indexing additional embedding models on demand.

    progress_cb, if given, is called as progress_cb(store_key, count, error)
    after each store finishes (error is None on success) -- lets a caller
    (agents/ingestion_pipeline.py) report per-store progress live without
    duplicating this loop. Optional and additive: existing callers (seed_all,
    the CLI, POST /seed/embedding) are unaffected."""
    docs = build_documents()
    print(f"Built {len(docs)} documents for indexing (embedding_model={embedding_model or config.DEFAULT_EMBEDDING_MODEL}).")

    indexed = {}
    for store_cfg in config.VECTOR_STORES:
        key = store_cfg["key"]
        try:
            store = _load_store(store_cfg, embedding_model)
            count = store.index_documents(docs)
            indexed[key] = count
            print(f"  [{key}] indexed {count} documents")
            if progress_cb:
                progress_cb(key, count, None)
        except Exception as e:  # graceful degradation per store
            indexed[key] = 0
            print(f"  [{key}] FAILED: {e}")
            if progress_cb:
                progress_cb(key, 0, str(e))
    return indexed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding roster key to index (default: config.DEFAULT_EMBEDDING_MODEL). "
             "Use --index-only to add a model without regenerating data.",
    )
    parser.add_argument("--index-only", action="store_true", help="Skip data regeneration, just (re)index.")
    args = parser.parse_args()

    if args.index_only:
        result = {"indexed": index_embedding_model(args.embedding_model)}
    else:
        result = seed_all(args.embedding_model)
    print(json.dumps(result, indent=2))
