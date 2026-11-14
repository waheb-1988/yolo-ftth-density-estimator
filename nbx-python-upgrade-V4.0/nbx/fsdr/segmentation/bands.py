"""Bands and micro-segments (master prompt §8). Micro-segment = nationality group x age band x tenure band
(5 x 4 x 6 = 120 cells); value band and usage type are context in customer_360, not splits."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from nbx.fsdr import config


def _bands(x: pd.Series, bands: list[tuple[str, float, float]], valid: pd.Series) -> np.ndarray:
    v = pd.to_numeric(x, errors="coerce")
    conds = [(valid & (v >= lo) & (v < hi)).fillna(False).to_numpy(dtype=bool) for _, lo, hi in bands]
    return np.select(conds, [b for b, _, _ in bands], "unknown").astype(object)


def age_band(age: pd.Series) -> np.ndarray:
    """young < 40 (incl. < 20), middle 40-59, senior 60+, unknown = missing or outside 15-100."""
    a = pd.to_numeric(age, errors="coerce")
    return _bands(a, config.AGE_BANDS, (a >= config.AGE_MIN) & (a <= config.AGE_MAX))


def young_split(age: pd.Series) -> np.ndarray:
    """Open item 5: young under 25 / 25-39 (tested in Phase 4, not a split yet)."""
    a = pd.to_numeric(age, errors="coerce")
    young = (a >= config.AGE_MIN) & (a < config.AGE_BANDS[0][2])
    return np.where(young & (a < config.YOUNG_SPLIT_AGE), "lt25", np.where(young, "25_39", None)).astype(object)


def tenure_band(days: pd.Series) -> np.ndarray:
    """t0_lt3m < 90 d, t1_3_6m, t2_6_12m, t3_1_2y, t4_gt2y > 730 d, unknown = missing or negative."""
    d = pd.to_numeric(days, errors="coerce")
    return _bands(d, config.TENURE_BANDS, d >= 0)


def micro_segment(nat: pd.Series, age_b, tenure_b) -> pd.Series:
    return (pd.Series(nat, dtype="string").fillna(config.NATIONALITY_OTHERS) + "|" + pd.Series(age_b, index=nat.index, dtype="string")
            + "|" + pd.Series(tenure_b, index=nat.index, dtype="string")).astype(object)


def all_micro_segments() -> list[tuple[str, str, str]]:
    return list(itertools.product(config.NATIONALITY_GROUPS, config.AGE_BAND_NAMES, config.TENURE_BAND_NAMES))


def value_band(arpu: pd.Series, cutoffs: list[float]) -> np.ndarray:
    """zero (ARPU_M1 = 0), then low / mid / high = tertiles of the paying lines over the whole base."""
    a = pd.to_numeric(arpu, errors="coerce").fillna(0.0)
    lo, hi = cutoffs
    return np.select([a <= 0, a <= lo, a <= hi], ["zero", "low", "mid"], "high").astype(object)


def usage_type(voice_user: pd.Series, data_user: pd.Series) -> np.ndarray:
    v, d = voice_user.astype(bool), data_user.astype(bool)
    return np.select([v & d, d, v], ["dual", "data_only", "voice_only"], "none").astype(object)
