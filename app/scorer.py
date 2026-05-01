"""Zone scoring algorithm — converts vehicle counts + metadata into priority scores."""
from __future__ import annotations

from app.config import (ARPU_MONTHLY_OMR, COST_PER_COVERAGE_POINT_OMR,
                        FTTH_UPTAKE_RATE, HOUSEHOLD_SIZE, MAX_DENSITY_REF,
                        SCORE_WEIGHTS)


def _tier(score: float) -> str:
    if score >= 68:
        return "High Priority"
    if score >= 40:
        return "Medium Priority"
    return "Low Priority"


def _recommended_investment(zone: dict, coverage_gap_pct: float) -> int:
    return int(zone["area_km2"] * coverage_gap_pct / 100
               * COST_PER_COVERAGE_POINT_OMR)


def _roi_years(investment: int, zone: dict) -> float:
    if investment <= 0:
        return 0.0
    households = zone["population"] / HOUSEHOLD_SIZE
    new_subs = households * FTTH_UPTAKE_RATE * (zone.get("ftth_coverage_pct", 50) / 100)
    annual_rev = new_subs * ARPU_MONTHLY_OMR * 12
    if annual_rev <= 0:
        return 99.0
    return round(investment / annual_rev, 1)


class ZoneScorer:
    def __init__(self, zones_data: dict):
        self._zones = {z["id"]: z for z in zones_data["zones"]}

    def score_zone(self, zone: dict, vehicle_count: int) -> dict:
        area = max(zone["area_km2"], 0.1)
        density = vehicle_count / area
        density_score = min(density / MAX_DENSITY_REF, 1.0)

        coverage_gap = (100 - zone["ftth_coverage_pct"]) / 100
        pop_score = min(zone["population"] / 100_000, 1.0)

        raw = (SCORE_WEIGHTS["vehicle_density"] * density_score
               + SCORE_WEIGHTS["coverage_gap"] * coverage_gap
               + SCORE_WEIGHTS["population_density"] * pop_score)

        priority_score = round(min(raw * 100, 100), 2)
        gap_pct = 100 - zone["ftth_coverage_pct"]
        inv = _recommended_investment(zone, gap_pct)

        return {
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "lat": zone["lat"],
            "lon": zone["lon"],
            "vehicle_count": vehicle_count,
            "density_per_km2": round(density, 2),
            "ftth_coverage_pct": zone["ftth_coverage_pct"],
            "coverage_gap_pct": gap_pct,
            "population": zone["population"],
            "area_km2": zone["area_km2"],
            "commercial_density": zone.get("commercial_density", "medium"),
            "priority_score": priority_score,
            "tier": _tier(priority_score),
            "recommended_investment_omr": inv,
            "roi_years": _roi_years(inv, zone),
        }

    def score_all_zones(self, vehicle_counts: dict) -> list[dict]:
        scored = []
        for zid, zone in self._zones.items():
            count = vehicle_counts.get(zid, 0)
            scored.append(self.score_zone(zone, count))
        return sorted(scored, key=lambda x: x["priority_score"], reverse=True)
