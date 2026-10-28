"""Collaborative filtering: implicit-feedback BPR (`implicit`), used by the FSDR offer ranker
(nbx/fsdr/offer_ranker.py). LightFM (production's library) does not build on Python 3.12 / Windows;
`implicit`'s BayesianPersonalizedRanking solves the same problem (implicit-feedback top-N ranking with a
pairwise loss).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

try:
    from implicit.bpr import BayesianPersonalizedRanking
    _IMPLICIT_AVAILABLE = True
except ImportError:  # pragma: no cover - only if `implicit` isn't installed
    _IMPLICIT_AVAILABLE = False

DEFAULT_PARAMS = {"factors": 32, "iterations": 60, "learning_rate": 0.05, "regularization": 0.01, "random_state": 42}


def fit_bpr(matrix: csr_matrix, params: dict | None = None) -> "BayesianPersonalizedRanking":
    """Fits implicit's BPR on a users x items interaction matrix."""
    if not _IMPLICIT_AVAILABLE:
        raise RuntimeError("implicit is not installed")
    params = params or DEFAULT_PARAMS
    model = BayesianPersonalizedRanking(
        factors=params["factors"],
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        regularization=params["regularization"],
        random_state=params["random_state"],
        num_threads=params.get("num_threads", 0),      # 1 = reproducible (multi-threaded SGD is not)
    )
    model.fit(matrix, show_progress=False)
    return model


def bpr_scores(model: "BayesianPersonalizedRanking", user_rows) -> np.ndarray:
    """users x items scores (dot product of the learned factors) for the given matrix rows."""
    uf, itf = model.user_factors, model.item_factors
    if hasattr(uf, "to_numpy"):           # GPU builds return implicit Matrix objects
        uf, itf = uf.to_numpy(), itf.to_numpy()
    return np.asarray(uf)[user_rows] @ np.asarray(itf).T
