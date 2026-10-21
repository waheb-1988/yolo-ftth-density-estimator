"""Collaborative-filtering signal for NBO, standing in for the production
Dataiku/LightFM monthly-upsell job.

Same problem, same filters, different library: LightFM itself (PyPI, last
released 2020) fails to build on Python 3.12 / Windows in this environment --
its setup.py raises `AttributeError: 'dict' object has no attribute
'__LIGHTFM_SETUP__'` under modern setuptools build isolation, and no
prebuilt wheel exists for this platform either. `implicit`'s Bayesian
Personalized Ranking (BPR) is the maintained equivalent: implicit-feedback
matrix factorization trained with a pairwise ranking loss, the same problem
class as LightFM's WARP loss, and it ships a working cp312/win_amd64 wheel.

Mirrors the production job's own filters before training:
  - only offers with validity == 30 days (core.config.CF_MONTHLY_VALIDITY_DAYS)
  - never "Voice" category (core.config.CF_EXCLUDED_CATEGORY)
  - only purchases within the last 90 days (core.config.CF_INTERACTION_WINDOW_DAYS)

Deliberately NOT ported: the production script's cold-start handling scores
"no-taker" customers using `enumerate(user_ids)` as the trained user index --
i.e. recycled, unrelated customers' embeddings, not real personalization.
Customers with no purchase history in the training window get an honest
popularity fallback here instead (see _cold_start_scores).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
from scipy.sparse import csr_matrix

from nbx.core import config

try:
    from implicit.bpr import BayesianPersonalizedRanking
    _IMPLICIT_AVAILABLE = True
except ImportError:  # pragma: no cover - only if `implicit` isn't installed
    _IMPLICIT_AVAILABLE = False


def _load_json(path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def monthly_nonvoice_offers() -> list[dict]:
    """The exact item universe the production job trains and scores on:
    validity == 30 days, category != Voice."""
    offers = _load_json(config.OFFERS_FILE)
    return [
        o for o in offers
        if o.get("validity_days") == config.CF_MONTHLY_VALIDITY_DAYS
        and o.get("category") != config.CF_EXCLUDED_CATEGORY
    ]


class _CFModel:
    """Holds the trained model plus the index maps and popularity fallback
    needed to score a customer, cached like core.knowledge_graph's graph."""

    def __init__(self):
        self.model: BayesianPersonalizedRanking | None = None
        self.user_index: dict[int, int] = {}          # msisdn -> matrix row
        self.item_index: dict[str, int] = {}           # program_rule -> matrix col
        self.index_item: list[str] = []                # matrix col -> program_rule
        self.user_items: csr_matrix | None = None       # for implicit's .recommend()
        self.popularity: dict[str, float] = {}          # program_rule -> normalized count (cold-start fallback)
        self.trained_customers = 0
        self.n_items = 0
        self.n_interactions = 0


def _build_interaction_matrix(offers_universe: list[dict]) -> tuple[csr_matrix, dict[int, int], dict[str, int], list[str], dict[str, float], int]:
    item_rules = [o["program_rule"] for o in offers_universe]
    item_index = {rule: i for i, rule in enumerate(item_rules)}

    purchases = _load_json(config.PURCHASES_FILE)
    cutoff = datetime.now() - timedelta(days=config.CF_INTERACTION_WINDOW_DAYS)

    windowed = []
    for p in purchases:
        if p.get("program_rule") not in item_index:
            continue
        try:
            purchase_dt = datetime.fromisoformat(p["purchase_date"])
        except (KeyError, ValueError):
            continue
        if purchase_dt < cutoff:
            continue
        windowed.append(p)

    user_ids = sorted({p["msisdn"] for p in windowed})
    user_index = {msisdn: i for i, msisdn in enumerate(user_ids)}

    counts: dict[tuple[int, int], float] = {}
    item_popularity_raw: dict[str, float] = {rule: 0.0 for rule in item_rules}
    for p in windowed:
        key = (user_index[p["msisdn"]], item_index[p["program_rule"]])
        weight = p.get("purchase_count", 1)
        counts[key] = counts.get(key, 0.0) + weight
        item_popularity_raw[p["program_rule"]] += weight

    rows, cols, vals = [], [], []
    for (r, c), v in counts.items():
        rows.append(r)
        cols.append(c)
        vals.append(v)
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_rules)))

    max_pop = max(item_popularity_raw.values()) if item_popularity_raw else 0.0
    popularity = {
        rule: round(v / max_pop, 4) if max_pop > 0 else 0.0
        for rule, v in item_popularity_raw.items()
    }

    return matrix, user_index, item_index, item_rules, popularity, len(windowed)


def _train(offers_universe: list[dict]) -> _CFModel:
    cf = _CFModel()
    cf.n_items = len(offers_universe)
    if not offers_universe or not _IMPLICIT_AVAILABLE:
        return cf

    matrix, user_index, item_index, item_rules, popularity, n_interactions = _build_interaction_matrix(offers_universe)
    cf.user_index = user_index
    cf.item_index = item_index
    cf.index_item = item_rules
    cf.popularity = popularity
    cf.trained_customers = len(user_index)
    cf.n_interactions = n_interactions
    cf.user_items = matrix

    if matrix.shape[0] < 2 or matrix.nnz < 4:
        # Not enough training signal yet (e.g. right after a fresh seed with
        # an unlucky random draw) -- leave model=None, callers fall back to
        # popularity for every customer rather than fitting on near-empty data.
        return cf

    params = config.CF_MODEL_PARAMS
    model = BayesianPersonalizedRanking(
        factors=params["factors"],
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        regularization=params["regularization"],
        random_state=params["random_state"],
    )
    model.fit(matrix)
    cf.model = model
    return cf


_cf_cache: _CFModel | None = None


def get_model(force_rebuild: bool = False) -> _CFModel:
    global _cf_cache
    if force_rebuild or _cf_cache is None:
        _cf_cache = _train(monthly_nonvoice_offers())
    return _cf_cache


def get_stats(cf: _CFModel | None = None) -> dict:
    cf = cf or get_model()
    return {
        "available": _IMPLICIT_AVAILABLE,
        "model_fitted": cf.model is not None,
        "item_universe": cf.n_items,
        "trained_customers": cf.trained_customers,
        "interactions_in_window": cf.n_interactions,
        "window_days": config.CF_INTERACTION_WINDOW_DAYS,
    }


def score_customer(customer: dict, top_k: int = 10, cf: _CFModel | None = None) -> list[dict]:
    """Returns up to top_k [{program_rule, cf_score, peer_basis, cf_reason}],
    cf_score normalized 0-1 within this customer's own returned set.
    Personalized (model.recommend) when the customer has purchase history in
    the training window; an honest item-popularity fallback otherwise --
    never another customer's recycled embedding."""
    cf = cf or get_model()
    if not cf.n_items:
        return []

    msisdn = customer.get("msisdn")
    row = cf.user_index.get(msisdn)

    if cf.model is not None and row is not None:
        ids, scores = cf.model.recommend(
            row, cf.user_items[row], N=min(top_k, cf.n_items), filter_already_liked_items=False
        )
        pairs = list(zip(ids, scores))
        basis = "personalized"
    else:
        # Cold start: no trained embedding for this customer (either no
        # purchases in the 90-day window, or the model didn't fit at all).
        ranked = sorted(cf.popularity.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        pairs = [(cf.item_index[rule], score) for rule, score in ranked]
        basis = "cold_start_popularity"

    if not pairs:
        return []

    raw_scores = [float(s) for _, s in pairs]
    lo, hi = min(raw_scores), max(raw_scores)
    spread = hi - lo

    out = []
    for (idx, raw), score in zip(pairs, raw_scores):
        norm = (score - lo) / spread if spread > 1e-9 else 1.0
        program_rule = cf.index_item[idx]
        out.append(
            {
                "program_rule": program_rule,
                "cf_score": round(norm, 4),
                "basis": basis,
                "cf_reason": (
                    "Implicit BPR model trained on this customer's 90-day subscription history."
                    if basis == "personalized"
                    else "Cold start (no purchases in the 90-day window) — ranked by overall item popularity."
                ),
            }
        )
    return out
