"""Offer rankers: score catalogue offers for customers from their 270-day BTL history (BR-83 .. BR-86).

    ImplicitBPRRanker   default; BPR from `implicit`, the same setup as core/collab_filter.py
    LightFMRanker       optional (production used LightFM, WARP); only if `lightfm` is installed

Fix vs the original: scores are looked up by the customer's own user id, never
by the loop index (the original predicted with the wrong user for every
customer after the first batch of non-takers).
Cold start (BR-86): customers without history get the item-popularity score.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nbx.core import collab_filter
from nbx.fsdr import config


class OfferRanker(Protocol):
    def fit(self, interactions: pd.DataFrame, items: list[str]) -> "OfferRanker": ...
    def score(self, users: list[str]) -> np.ndarray: ...


class ImplicitBPRRanker:
    """interactions: columns ACCESS_METHOD_VAL, BTL_NAME, WEIGHT (sub_flag summed, as production)."""

    def __init__(self, params: dict | None = None):
        self.params = params or config.BPR_PARAMS
        self.items: list[str] = []
        self.user_index: dict[str, int] = {}
        self.popularity = np.zeros(0)
        self.model = None

    def fit(self, interactions: pd.DataFrame, items: list[str]) -> "ImplicitBPRRanker":
        self.items = list(items)
        item_index = {it: i for i, it in enumerate(self.items)}
        df = interactions[interactions["BTL_NAME"].isin(item_index)]
        users = sorted(df["ACCESS_METHOD_VAL"].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        pop = np.zeros(len(self.items))
        if len(df):
            g = df.groupby(["ACCESS_METHOD_VAL", "BTL_NAME"])["WEIGHT"].sum().reset_index()
            rows = g["ACCESS_METHOD_VAL"].map(self.user_index).to_numpy()
            cols = g["BTL_NAME"].map(item_index).to_numpy()
            matrix = csr_matrix((g["WEIGHT"].astype(float).to_numpy(), (rows, cols)), shape=(len(users), len(self.items)))
            pop = np.asarray(matrix.sum(axis=0)).ravel()
            if matrix.shape[0] >= 2 and matrix.nnz >= 4 and len(self.items) >= 2:
                self.model = collab_filter.fit_bpr(matrix, self.params)
        self.popularity = pop / pop.max() if pop.size and pop.max() > 0 else pop   # empty family: no items
        return self

    def score(self, users: list[str]) -> np.ndarray:
        """len(users) x len(items). Known users: BPR scores (by THEIR row id); others: popularity."""
        out = np.tile(self.popularity, (len(users), 1)).astype(float)
        if self.model is None or not len(users):
            return out
        known = [(i, self.user_index[u]) for i, u in enumerate(users) if u in self.user_index]
        if known:
            pos, rows = zip(*known)
            s = collab_filter.bpr_scores(self.model, list(rows))
            out[list(pos)] = (s - s.min(axis=1, keepdims=True)) / np.maximum(np.ptp(s, axis=1, keepdims=True), 1e-9) + 1.0
        return out


class LightFMRanker:
    """Production's LightFM WARP model (BR-84). Not installable on Python 3.12 / Windows today."""

    def __init__(self, params: dict | None = None):
        try:
            from lightfm import LightFM  # noqa: F401
        except ImportError as e:
            raise RuntimeError("LightFMRanker needs the `lightfm` package; use FSDR_RANKER=bpr") from e
        self.params = params or config.LIGHTFM_PARAMS
        self.inner = ImplicitBPRRanker()      # same id bookkeeping; the factor model is swapped below

    def fit(self, interactions: pd.DataFrame, items: list[str]) -> "LightFMRanker":
        from lightfm import LightFM

        self.inner.fit(interactions, items)
        df = interactions[interactions["BTL_NAME"].isin(set(items))]
        if len(df) and len(self.inner.user_index) >= 2:
            item_index = {it: i for i, it in enumerate(self.inner.items)}
            g = df.groupby(["ACCESS_METHOD_VAL", "BTL_NAME"])["WEIGHT"].sum().reset_index()
            m = csr_matrix((g["WEIGHT"].astype(float), (g["ACCESS_METHOD_VAL"].map(self.inner.user_index), g["BTL_NAME"].map(item_index))),
                           shape=(len(self.inner.user_index), len(items)))
            self.model = LightFM(no_components=self.params["no_components"], loss=self.params["loss"], k=self.params["k"])
            self.model.fit(m, epochs=self.params["epochs"])
        else:
            self.model = None
        return self

    def score(self, users: list[str]) -> np.ndarray:
        out = self.inner.score(users) if self.model is None else np.tile(self.inner.popularity, (len(users), 1)).astype(float)
        if self.model is None:
            return out
        for i, u in enumerate(users):
            if u in self.inner.user_index:     # by the user's own id
                out[i] = self.model.predict(self.inner.user_index[u], np.arange(len(self.inner.items))) + 10.0
        return out


def make_ranker(name: str | None = None) -> OfferRanker:
    name = (name or config.RANKER).lower()
    if name == "lightfm":
        return LightFMRanker()
    return ImplicitBPRRanker()


def load_interactions(src, catalog: pd.DataFrame) -> pd.DataFrame:
    """sub_btl_3months (D-270..D-1) restricted to catalogue BTLs (BR-63/83)."""
    from nbx.fsdr.features import norm_val

    df = src.read("sub_btl_3months")
    out = pd.DataFrame({"ACCESS_METHOD_VAL": norm_val(df["access_method_val"]),
                        "BTL_NAME": df["PRODUCTNAME"].astype(str).str.strip(),
                        "WEIGHT": pd.to_numeric(df["sub_flag"], errors="coerce").fillna(1.0)})
    return out[out["BTL_NAME"].isin(set(catalog["BTL_NAME"]))].dropna(subset=["ACCESS_METHOD_VAL"])


def top_candidates(ranker: OfferRanker, users: list[str], items: pd.DataFrame, top_n: int = config.RANKER_TOP_N) -> pd.DataFrame:
    """Long frame (ACCESS_METHOD_VAL, BTL_NAME, SCORE, RANK + item attributes), top-N per user (BR-85)."""
    if not users or items.empty:   # e.g. no purchase history for the family: no candidate, so no offer (BR-82)
        return pd.DataFrame(columns=["ACCESS_METHOD_VAL", "BTL_NAME", "SCORE", "RANK", "SERVICE", "GENEROSITY", "VALIDITY", "PRICE"])
    scores = ranker.score(users)
    names = np.array(ranker.items)
    k = min(top_n, len(names))
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    rows = pd.DataFrame({
        "ACCESS_METHOD_VAL": np.repeat(users, k),
        "BTL_NAME": names[order].ravel(),
        "SCORE": np.take_along_axis(scores, order, axis=1).ravel(),
        "RANK": np.tile(np.arange(1, k + 1), len(users)),
    })
    return rows.merge(items[["BTL_NAME", "SERVICE", "GENEROSITY", "VALIDITY", "PRICE"]], on="BTL_NAME", how="left")
