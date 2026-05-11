"""
ML-powered demand forecasting, zone clustering, and market projection for FALCON.

Modules:
  - cluster_zones()          K-Means zone segmentation
  - forecast_revenue()       Per-zone 5-year revenue model with growth
  - predict_demand_score()   Feature-based demand scoring (no YOLO required)
  - market_share_projection() S-curve market share growth model
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.config import ARPU_MONTHLY_OMR, FTTH_UPTAKE_RATE, HOUSEHOLD_SIZE

logger = logging.getLogger(__name__)


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_zone_feature_matrix(scored_zones: list, zones_data: dict) -> pd.DataFrame:
    """Merge scored zone outputs with raw zone metadata into a flat DataFrame."""
    zone_meta = {z["id"]: z for z in zones_data["zones"]}
    _cd = {"high": 3, "medium": 2, "low": 1}
    rows = []
    for s in scored_zones:
        z = zone_meta.get(s["zone_id"], {})
        rows.append({
            "zone_id":              s["zone_id"],
            "zone_name":            s["zone_name"],
            "population":           s["population"],
            "area_km2":             s["area_km2"],
            "vehicle_density":      s["density_per_km2"],
            "ftth_coverage":        s["ftth_coverage_pct"],
            "coverage_gap":         s["coverage_gap_pct"],
            "priority_score":       s["priority_score"],
            "road_km":              z.get("road_network_km", 0),
            "commercial_density_n": _cd.get(s.get("commercial_density", "medium"), 2),
            "pop_density":          s["population"] / max(s["area_km2"], 0.1),
        })
    return pd.DataFrame(rows)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_zones(scored_zones: list, zones_data: dict,
                  n_clusters: int = 3) -> pd.DataFrame:
    """
    Segment zones using K-Means on normalised features.
    Returns DataFrame with columns: zone_id, zone_name, cluster, cluster_label,
    priority_score, pop_density, vehicle_density, coverage_gap.
    Returns empty DataFrame if scikit-learn is unavailable.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not installed — clustering unavailable")
        return pd.DataFrame()

    df = build_zone_feature_matrix(scored_zones, zones_data)
    feature_cols = [
        "pop_density", "vehicle_density", "coverage_gap",
        "commercial_density_n", "priority_score",
    ]

    X = df[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # Label clusters by descending average priority score
    cluster_means = (df.groupby("cluster")["priority_score"]
                       .mean()
                       .sort_values(ascending=False))
    _labels = ["High Opportunity", "Moderate Potential", "Low Priority",
               "Very Low Priority", "Emerging"]
    label_map = {cid: _labels[i] for i, cid in enumerate(cluster_means.index)}
    df["cluster_label"] = df["cluster"].map(label_map)

    logger.info("Zone clustering: %d zones → %d clusters", len(df), n_clusters)
    return df[[
        "zone_id", "zone_name", "cluster", "cluster_label",
        "priority_score", "pop_density", "vehicle_density", "coverage_gap",
    ]]


# ── Revenue forecast ──────────────────────────────────────────────────────────

def forecast_revenue(
    zone: dict,
    years: int = 5,
    arpu: float = ARPU_MONTHLY_OMR,
    uptake: float = FTTH_UPTAKE_RATE,
    growth_rate: float = 0.08,
) -> dict:
    """
    Project annual and cumulative revenue for a scored zone over N years.

    Model:
      - Year 1 subscribers = households * uptake * gap%
      - Each subsequent year grows by `growth_rate` (compound)
      - Revenue = subscribers * arpu * 12
    Returns dict with yearly breakdowns and breakeven year.
    """
    households  = zone["population"] / HOUSEHOLD_SIZE
    gap_pct     = zone.get("coverage_gap_pct", 50) / 100
    initial_subs = households * uptake * gap_pct

    annual_revs: list[float] = []
    for y in range(1, years + 1):
        subs = initial_subs * (1 + growth_rate) ** (y - 1)
        annual_revs.append(round(subs * arpu * 12, 0))

    cumulative = [round(sum(annual_revs[:i]), 0) for i in range(1, years + 1)]
    investment  = zone.get("recommended_investment_omr", 0)

    breakeven: Optional[int] = None
    running = 0.0
    for i, r in enumerate(annual_revs, 1):
        running += r
        if running >= investment and breakeven is None:
            breakeven = i

    return {
        "zone_id":            zone["zone_id"],
        "zone_name":          zone["zone_name"],
        "years":              list(range(1, years + 1)),
        "annual_revenue":     annual_revs,
        "cumulative_revenue": cumulative,
        "investment":         investment,
        "breakeven_year":     breakeven,
        "total_revenue":      cumulative[-1] if cumulative else 0,
        "initial_subscribers": round(initial_subs, 0),
        "growth_rate_pct":    growth_rate * 100,
    }


# ── Demand prediction (no YOLO) ───────────────────────────────────────────────

def predict_demand_score(zone: dict) -> float:
    """
    Feature-based demand score (0–100) as a cross-check against YOLO density.
    Uses population density, coverage gap, and commercial activity.
    """
    pop_density  = zone.get("population", 0) / max(zone.get("area_km2", 1), 0.1)
    gap          = 100 - zone.get("ftth_coverage_pct", 50)
    _cd          = {"high": 1.0, "medium": 0.65, "low": 0.30}
    commercial   = _cd.get(zone.get("commercial_density", "medium"), 0.65)

    density_norm = min(pop_density / 15_000, 1.0)
    gap_norm     = gap / 100

    score = (0.35 * density_norm + 0.40 * gap_norm + 0.25 * commercial) * 100
    return round(min(score, 100), 2)


# ── Market share S-curve ──────────────────────────────────────────────────────

def market_share_projection(
    current_share: float,
    target_share: float,
    years: int = 5,
    acceleration: float = 1.2,
) -> list[float]:
    """
    Project market share growth on a logistic (S-curve) model.
    Returns a list of share percentages for years 1…N.
    """
    gap    = target_share - current_share
    shares = []
    for y in range(1, years + 1):
        progress = 1.0 / (1.0 + np.exp(-acceleration * (y - years / 2.0)))
        shares.append(round(current_share + gap * progress, 2))
    return shares


# ── Traffic trend analysis ────────────────────────────────────────────────────

def traffic_trend(
    zone_profiles: list[dict],
    years: int = 5,
    annual_data_growth: float = 0.35,
) -> list[dict]:
    """
    Estimate network traffic growth trend for a list of zone telecom profiles.
    Assumes 35 % YoY data consumption growth (ITU benchmark).
    """
    rows = []
    for p in zone_profiles:
        peak_now = p.get("peak_hour_traffic_gbps", 0)
        for y in range(1, years + 1):
            rows.append({
                "zone_id":   p["zone_id"],
                "zone_name": p["zone_name"],
                "year":      2026 + y - 1,
                "peak_gbps": round(peak_now * (1 + annual_data_growth) ** (y - 1), 4),
            })
    return rows
