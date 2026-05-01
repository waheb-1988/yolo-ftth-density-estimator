"""FALCON — Telecom network position generator for Waheb, RAMI, and Don."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Ensure project root on path when run standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (DATA_DIR, DON_COLOR, RAMI_COLOR, TOTAL_MUSCAT_HOMES,
                         WAHEB_COLOR, WAHEB_MARKET_SHARE, RAMI_MARKET_SHARE,
                         DON_MARKET_SHARE, UNSERVED_MARKET_SHARE)

# ── Muscat geographic bounds ──────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 23.52, 23.68
LNG_MIN, LNG_MAX = 58.32, 58.62

# Zones with low competition = Waheb opportunity areas
WAHEB_STRONG_ZONES = ["Ghubra", "Madinat Sultan Qaboos", "Bausher"]
RAMI_STRONG_ZONES  = ["Ruwi CBD", "Mutrah Port", "Al Wuttayyah", "Qurum Heights"]
DON_STRONG_ZONES   = ["Shatti Al Qurum", "Al Azaiba", "Al Khuwair"]
OPPORTUNITY_ZONES  = ["Bausher", "Wadi Kabir", "Sidab", "Al Amerat",
                       "Bowsher Hills", "Al Seeb City"]

# Competitive intensity per zone
COMPETITIVE_INTENSITY = {
    "Ruwi CBD":              "high",
    "Mutrah Port":           "high",
    "Qurum Heights":         "high",
    "Al Khuwair":            "high",
    "Shatti Al Qurum":       "medium",
    "Al Azaiba":             "medium",
    "Madinat Sultan Qaboos": "medium",
    "Al Wuttayyah":          "medium",
    "Al Seeb City":          "medium",
    "Ghubra":                "low",
    "Bausher":               "low",
    "Wadi Kabir":            "low",
    "Sidab":                 "low",
    "Al Amerat":             "low",
    "Bowsher Hills":         "low",
}


class NetworkGenerator:
    """Generates reproducible FTTH asset positions for Muscat telcos."""

    def __init__(self, zones_data: dict | None = None, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self._np_seed = seed
        self.zones = zones_data["zones"] if zones_data else []

    # ── Internals ─────────────────────────────────────────────────────────────

    def _clustered_positions(self, n: int, centers: list, weights: list,
                              spread: float = 0.012) -> list[tuple]:
        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        cluster_ids = np.random.choice(len(centers), size=n, p=weights_arr)
        positions = []
        for ci in cluster_ids:
            lat_c, lng_c = centers[ci]
            lat = float(np.clip(np.random.normal(lat_c, spread), LAT_MIN, LAT_MAX))
            lng = float(np.clip(np.random.normal(lng_c, spread), LNG_MIN, LNG_MAX))
            positions.append((lat, lng))
        return positions

    def _nearest_zone(self, lat: float, lng: float) -> str:
        if not self.zones:
            return "Muscat"
        return min(
            self.zones,
            key=lambda z: (z["lat"] - lat) ** 2 + (z["lon"] - lng) ** 2,
        )["name"]

    @staticmethod
    def _random_date(start_year: int = 2019, end_year: int = 2024) -> str:
        start = datetime(start_year, 1, 1)
        total_days = (datetime(end_year + 1, 1, 1) - start).days
        return (start + timedelta(days=random.randint(0, total_days))).strftime(
            "%Y-%m-%d")

    def _make_asset(self, prefix: str, idx: int, asset_type: str,
                    lat: float, lng: float, active_pct: float = 0.70) -> dict:
        status = "active" if random.random() < active_pct else "planned"
        connected = (
            random.randint(50, 200) if asset_type == "cabinet"
            else random.randint(500, 2000) if asset_type == "exchange"
            else 1
        )
        short = asset_type[:3].upper()
        return {
            "asset_id":          f"{prefix}-{short}-{idx:03d}",
            "company":           prefix,
            "type":              asset_type,
            "lat":               round(lat, 6),
            "lng":               round(lng, 6),
            "zone":              self._nearest_zone(lat, lng),
            "status":            status,
            "capacity_used_pct": round(random.uniform(20, 95), 1),
            "connected_homes":   connected,
            "install_date":      self._random_date(),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_coverage_radius(self, asset_type: str) -> float:
        return {"cabinet": 0.8, "exchange": 3.0, "home": 0.05}.get(asset_type, 0.5)

    def calculate_market_coverage(self, total_homes: int = TOTAL_MUSCAT_HOMES) -> dict:
        return {
            "waheb_covered_homes":    int(total_homes * WAHEB_MARKET_SHARE / 100),
            "waheb_market_share_pct": WAHEB_MARKET_SHARE,
            "rami_covered_homes":     int(total_homes * RAMI_MARKET_SHARE / 100),
            "rami_market_share_pct":  RAMI_MARKET_SHARE,
            "don_covered_homes":      int(total_homes * DON_MARKET_SHARE / 100),
            "don_market_share_pct":   DON_MARKET_SHARE,
            "unserved_homes":         int(total_homes * UNSERVED_MARKET_SHARE / 100),
            "total_homes":            total_homes,
        }

    # ── Waheb network ─────────────────────────────────────────────────────────

    def generate_waheb_network(self) -> list[dict]:
        """45 cabinets + 12 exchanges + 180 homes, clustered in Al Khuwair/Ghubra/Qurum/MQ."""
        np.random.seed(self._np_seed)
        random.seed(self._np_seed)

        centers = [
            (23.597, 58.517),  # Al Khuwair
            (23.598, 58.499),  # Ghubra
            (23.585, 58.533),  # Qurum
            (23.586, 58.505),  # Madinat Sultan Qaboos
        ]
        weights = [0.30, 0.25, 0.25, 0.20]

        assets: list[dict] = []
        for i, (lt, lg) in enumerate(
                self._clustered_positions(45, centers, weights), 1):
            assets.append(self._make_asset("WAH", i, "cabinet", lt, lg, 0.70))
        for i, (lt, lg) in enumerate(
                self._clustered_positions(12, centers, weights, spread=0.018), 1):
            assets.append(self._make_asset("WAH", i, "exchange", lt, lg, 0.70))
        for i, (lt, lg) in enumerate(
                self._clustered_positions(180, centers, weights, spread=0.008), 1):
            assets.append(self._make_asset("WAH", i, "home", lt, lg, 0.75))
        return assets

    # ── RAMI network ──────────────────────────────────────────────────────────

    def generate_rami_network(self) -> list[dict]:
        """95 cabinets + 28 exchanges + 420 homes, dominant in Ruwi/Wattayah/Darsait."""
        np.random.seed(self._np_seed + 1)
        random.seed(self._np_seed + 1)

        centers = [
            (23.606, 58.604),  # Ruwi
            (23.593, 58.560),  # Wattayah
            (23.610, 58.578),  # Darsait
            (23.577, 58.585),  # South Muscat
        ]
        weights = [0.35, 0.25, 0.20, 0.20]

        assets: list[dict] = []
        for i, (lt, lg) in enumerate(
                self._clustered_positions(95, centers, weights), 1):
            assets.append(self._make_asset("RAMI", i, "cabinet", lt, lg, 0.85))
        for i, (lt, lg) in enumerate(
                self._clustered_positions(28, centers, weights, spread=0.020), 1):
            assets.append(self._make_asset("RAMI", i, "exchange", lt, lg, 0.85))
        for i, (lt, lg) in enumerate(
                self._clustered_positions(420, centers, weights, spread=0.007), 1):
            assets.append(self._make_asset("RAMI", i, "home", lt, lg, 0.85))
        return assets

    # ── Don network ───────────────────────────────────────────────────────────

    def generate_don_network(self) -> list[dict]:
        """70 cabinets + 18 exchanges + 290 homes, strong in Al Mouj/Azaiba/Al Khuwair S."""
        np.random.seed(self._np_seed + 2)
        random.seed(self._np_seed + 2)

        centers = [
            (23.594, 58.519),  # Al Mouj / Shatti
            (23.594, 58.539),  # Azaiba
            (23.582, 58.510),  # Al Khuwair South
            (23.600, 58.460),  # West Bausher
        ]
        weights = [0.35, 0.25, 0.25, 0.15]

        assets: list[dict] = []
        for i, (lt, lg) in enumerate(
                self._clustered_positions(70, centers, weights), 1):
            assets.append(self._make_asset("DON", i, "cabinet", lt, lg, 0.80))
        for i, (lt, lg) in enumerate(
                self._clustered_positions(18, centers, weights, spread=0.018), 1):
            assets.append(self._make_asset("DON", i, "exchange", lt, lg, 0.80))
        for i, (lt, lg) in enumerate(
                self._clustered_positions(290, centers, weights, spread=0.008), 1):
            assets.append(self._make_asset("DON", i, "home", lt, lg, 0.80))
        return assets

    # ── Market summary ────────────────────────────────────────────────────────

    def generate_market_summary(self, waheb: list, rami: list,
                                  don: list) -> dict:
        def cab_counts(assets):
            return Counter(
                a["zone"] for a in assets if a["type"] == "cabinet")

        wc = cab_counts(waheb)
        rc = cab_counts(rami)
        dc = cab_counts(don)

        opp_zones = []
        for z in self.zones:
            zn = z["name"]
            total_comp = rc.get(zn, 0) + dc.get(zn, 0)
            waheb_n = wc.get(zn, 0)
            if total_comp < 8 and waheb_n < 5:
                opp_zones.append({
                    "zone": zn,
                    "waheb_cabinets": waheb_n,
                    "competitor_cabinets": total_comp,
                    "opportunity_score": max(0, 12 - total_comp - waheb_n),
                    "est_annual_revenue_omr": int(
                        z["population"] / 4 * 0.30 * 0.42 * 5.5 * 12),
                })
        opp_zones.sort(key=lambda x: x["opportunity_score"], reverse=True)

        intensity = {}
        for z in self.zones:
            zn = z["name"]
            total = wc.get(zn, 0) + rc.get(zn, 0) + dc.get(zn, 0)
            intensity[zn] = "high" if total >= 8 else "medium" if total >= 4 else "low"

        return {
            "total_market_homes": TOTAL_MUSCAT_HOMES,
            "market_shares": {
                "Waheb":    WAHEB_MARKET_SHARE,
                "RAMI":     RAMI_MARKET_SHARE,
                "Don":      DON_MARKET_SHARE,
                "Unserved": UNSERVED_MARKET_SHARE,
            },
            "waheb_opportunity_zones": [o["zone"] for o in opp_zones[:6]],
            "opportunity_zone_details": opp_zones[:6],
            "competitive_intensity":   intensity,
            "cabinet_counts": {
                "Waheb": dict(wc),
                "RAMI":  dict(rc),
                "Don":   dict(dc),
            },
        }

    # ── Save all ──────────────────────────────────────────────────────────────

    def save_all(self, output_dir: str | Path = DATA_DIR) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        waheb = self.generate_waheb_network()
        rami  = self.generate_rami_network()
        don   = self.generate_don_network()

        _write(out / "waheb_network.json", waheb)
        _write(out / "rami_network.json",  rami)
        _write(out / "don_network.json",   don)

        summary = self.generate_market_summary(waheb, rami, don)
        _write(out / "market_summary.json", summary)

        return {
            "waheb": waheb, "rami": rami, "don": don,
            "market_summary": summary,
        }

    # ── Print summary ─────────────────────────────────────────────────────────

    def print_competitive_summary(self, network: dict):
        waheb = network["waheb"]
        rami  = network["rami"]
        don   = network["don"]
        ms    = network["market_summary"]

        def counts(assets, typ):
            return sum(1 for a in assets if a["type"] == typ)

        header = (f"{'Company':<10} {'Cabinets':>9} {'Exchanges':>10} "
                  f"{'Homes':>7} {'Share':>8}  {'Strong Zones'}")
        print("\n" + "=" * 75)
        print("  FALCON Competitive Landscape — Muscat FTTH Market")
        print("=" * 75)
        print(header)
        print("-" * 75)
        rows = [
            ("Waheb",  waheb, WAHEB_MARKET_SHARE,
             ", ".join(WAHEB_STRONG_ZONES[:2])),
            ("RAMI",   rami,  RAMI_MARKET_SHARE,
             ", ".join(RAMI_STRONG_ZONES[:2])),
            ("Don",    don,   DON_MARKET_SHARE,
             ", ".join(DON_STRONG_ZONES[:2])),
        ]
        for name, assets, share, zones in rows:
            print(f"  {name:<8} {counts(assets,'cabinet'):>9,} "
                  f"{counts(assets,'exchange'):>10,} "
                  f"{counts(assets,'home'):>7,} "
                  f"{share:>7.0f}%  {zones}")
        print("-" * 75)
        print(f"\n  Opportunity zones for Waheb:")
        for oz in ms.get("opportunity_zone_details", [])[:4]:
            print(f"    * {oz['zone']:<28} "
                  f"opp={oz['opportunity_score']}  "
                  f"rev={oz['est_annual_revenue_omr']:,} OMR/yr")
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_network_files(data_dir: Path = DATA_DIR) -> dict | None:
    result = {}
    for company in ["waheb", "rami", "don"]:
        p = data_dir / f"{company}_network.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                result[company] = json.load(f)
    market_p = data_dir / "market_summary.json"
    if market_p.exists():
        with open(market_p, encoding="utf-8") as f:
            result["market_summary"] = json.load(f)
    return result if len(result) >= 4 else None


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from app.data_loader import load_zones
    zones_data = load_zones()
    gen = NetworkGenerator(zones_data, seed=42)
    network = gen.save_all()
    gen.print_competitive_summary(network)
    print(f"[FALCON] Saved waheb/rami/don_network.json + market_summary.json "
          f"to {DATA_DIR}")
