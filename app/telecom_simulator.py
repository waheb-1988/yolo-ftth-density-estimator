"""
Telecom market simulation for Muscat, Oman.

Simulates realistic 5G / FTTH / mobile subscriber counts, ARPU,
traffic usage, and competitor market share for each zone.
All data is synthetic — no real operator information is used.
"""
from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

# ── Service ARPU benchmarks (OMR/month, Oman 2025 estimates) ──────────────────
ARPU_BY_SERVICE: dict[str, float] = {
    "ftth_home":       5.5,
    "ftth_business":  18.0,
    "mobile_4g":       3.2,
    "mobile_5g":       7.8,
    "fixed_broadband": 4.5,
}

# ── National penetration baselines ────────────────────────────────────────────
NATIONAL_PENETRATION: dict[str, float] = {
    "mobile":        0.82,
    "internet":      0.79,
    "ftth_eligible": 0.55,
    "5g_eligible":   0.28,
}

# ── Commercial-density multipliers ────────────────────────────────────────────
_CD_MULT: dict[str, float] = {"high": 1.35, "medium": 1.0, "low": 0.70}


@dataclass
class ZoneTelecomProfile:
    zone_id: str
    zone_name: str
    population: int
    households: int

    mobile_subscribers: int
    ftth_subscribers: int
    subscribers_5g: int
    fixed_broadband_subscribers: int

    mobile_penetration_pct: float
    ftth_penetration_pct: float
    penetration_5g_pct: float

    arpu_blended: float
    monthly_revenue_omr: float
    annual_revenue_omr: float

    avg_data_gb_per_user_month: float
    peak_hour_traffic_gbps: float

    residential_share_pct: float
    business_share_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompetitorProfile:
    company: str
    zone_id: str
    zone_name: str
    market_share_pct: float
    subscriber_count: int
    cabinet_count: int
    ftth_homes_passed: int
    avg_arpu: float
    monthly_revenue_omr: float

    def to_dict(self) -> dict:
        return asdict(self)


class TelecomSimulator:
    """Generates reproducible telecom market data for each zone."""

    def __init__(self, seed: int = 42):
        self._seed = seed

    def _rng(self, zone_id: str) -> random.Random:
        return random.Random(abs(hash(zone_id)) % 99_991 + self._seed)

    # ── Zone profile ──────────────────────────────────────────────────────────

    def simulate_zone(self, zone: dict) -> ZoneTelecomProfile:
        rng = self._rng(zone["id"])
        pop = zone["population"]
        households = int(pop / 4)
        cd_mult = _CD_MULT.get(zone.get("commercial_density", "medium"), 1.0)

        # Mobile
        mob_pen = min(0.95,
                      NATIONAL_PENETRATION["mobile"] * cd_mult * rng.uniform(0.90, 1.10))
        mobile_subs = int(pop * mob_pen)

        # FTTH
        ftth_pen = zone.get("ftth_coverage_pct", 30) / 100
        ftth_subs = int(
            households * ftth_pen * NATIONAL_PENETRATION["ftth_eligible"]
            * rng.uniform(0.85, 1.15)
        )

        # 5G — only strong where 4G coverage ≥ 80 %
        has_5g = zone.get("current_4g_coverage_pct", 50) >= 80
        pen_5g = (NATIONAL_PENETRATION["5g_eligible"] * rng.uniform(0.6, 1.4)
                  if has_5g else rng.uniform(0.02, 0.08))
        subs_5g = int(mobile_subs * pen_5g)

        # Fixed broadband (copper / cable — not FTTH)
        fb_subs = int(households * 0.12 * rng.uniform(0.8, 1.2))

        # Data traffic
        avg_gb = rng.uniform(18, 55) * cd_mult
        # Peak hour ≈ 15 % of daily traffic, in Gbps
        peak_gbps = (mobile_subs * avg_gb * 12) / (30 * 24 * 3600) * 1e9 * 0.15 / 1e9

        # Revenue
        monthly_rev = (
            ftth_subs * ARPU_BY_SERVICE["ftth_home"]
            + subs_5g * ARPU_BY_SERVICE["mobile_5g"]
            + (mobile_subs - subs_5g) * ARPU_BY_SERVICE["mobile_4g"]
            + fb_subs * ARPU_BY_SERVICE["fixed_broadband"]
        )
        total_subs = max(ftth_subs + mobile_subs + fb_subs, 1)
        arpu_blended = monthly_rev / total_subs

        res_share = rng.uniform(0.60, 0.85)

        return ZoneTelecomProfile(
            zone_id=zone["id"],
            zone_name=zone["name"],
            population=pop,
            households=households,
            mobile_subscribers=mobile_subs,
            ftth_subscribers=ftth_subs,
            subscribers_5g=subs_5g,
            fixed_broadband_subscribers=fb_subs,
            mobile_penetration_pct=round(mob_pen * 100, 1),
            ftth_penetration_pct=round(ftth_pen * 100, 1),
            penetration_5g_pct=round(pen_5g * 100, 1),
            arpu_blended=round(arpu_blended, 2),
            monthly_revenue_omr=round(monthly_rev, 0),
            annual_revenue_omr=round(monthly_rev * 12, 0),
            avg_data_gb_per_user_month=round(avg_gb, 1),
            peak_hour_traffic_gbps=round(peak_gbps, 4),
            residential_share_pct=round(res_share * 100, 1),
            business_share_pct=round((1 - res_share) * 100, 1),
        )

    # ── Competitor profiles ───────────────────────────────────────────────────

    def simulate_competitors(
        self, zone: dict, market_data: dict | None = None
    ) -> list[CompetitorProfile]:
        """Simulate Omantel, Ooredoo, and Vodafone presence in a zone."""
        from app.config import (ARPU_MONTHLY_OMR, DON_MARKET_SHARE,
                                 HOMES_PER_CABINET, RAMI_MARKET_SHARE,
                                 WAHEB_MARKET_SHARE)

        rng = self._rng(zone["id"] + "_comp")
        households = int(zone["population"] / 4)

        cab_data: dict[str, int] = {}
        if market_data:
            cab_counts = (market_data.get("market_summary", {})
                          .get("cabinet_counts", {}))
            zname = zone["name"]
            for company in ("Waheb", "RAMI", "Don"):
                cab_data[company] = cab_counts.get(company, {}).get(zname, 0)

        profiles: list[CompetitorProfile] = []
        for company_label, base_share, key in [
            ("Omantel", RAMI_MARKET_SHARE, "RAMI"),
            ("Ooredoo", DON_MARKET_SHARE,  "Don"),
            ("Vodafone", WAHEB_MARKET_SHARE, "Waheb"),
        ]:
            share = min(50.0, base_share * rng.uniform(0.7, 1.3))
            subs = int(households * share / 100 * 0.42)
            cabinets = cab_data.get(key, max(1, int(share / 3)))
            avg_arpu = round(ARPU_MONTHLY_OMR * rng.uniform(0.85, 1.15), 2)

            profiles.append(CompetitorProfile(
                company=company_label,
                zone_id=zone["id"],
                zone_name=zone["name"],
                market_share_pct=round(share, 1),
                subscriber_count=subs,
                cabinet_count=cabinets,
                ftth_homes_passed=cabinets * HOMES_PER_CABINET,
                avg_arpu=avg_arpu,
                monthly_revenue_omr=round(subs * avg_arpu, 0),
            ))

        return profiles

    # ── Full-city simulation ──────────────────────────────────────────────────

    def simulate_all(self, zones_data: dict,
                     market_data: dict | None = None) -> dict:
        """Run simulation for all zones. Returns nested dict keyed by zone_id."""
        results: dict = {
            "zone_profiles": {},
            "competitor_profiles": {},
            "market_totals": {},
        }
        totals: dict[str, float] = dict.fromkeys(
            ["mobile", "ftth", "5g", "revenue"], 0.0)

        for zone in zones_data["zones"]:
            zid = zone["id"]
            profile = self.simulate_zone(zone)
            results["zone_profiles"][zid] = profile.to_dict()
            results["competitor_profiles"][zid] = [
                c.to_dict()
                for c in self.simulate_competitors(zone, market_data)
            ]
            totals["mobile"]  += profile.mobile_subscribers
            totals["ftth"]    += profile.ftth_subscribers
            totals["5g"]      += profile.subscribers_5g
            totals["revenue"] += profile.monthly_revenue_omr

        results["market_totals"] = {
            "total_mobile_subscribers": int(totals["mobile"]),
            "total_ftth_subscribers":   int(totals["ftth"]),
            "total_5g_subscribers":     int(totals["5g"]),
            "total_monthly_revenue_omr": round(totals["revenue"], 0),
            "total_annual_revenue_omr":  round(totals["revenue"] * 12, 0),
            "zones_analyzed":            len(zones_data["zones"]),
        }

        logger.info(
            "Simulation complete — %d zones | %d mobile subs | %d FTTH subs",
            len(zones_data["zones"]), int(totals["mobile"]), int(totals["ftth"]),
        )
        return results
