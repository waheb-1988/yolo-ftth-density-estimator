"""Zone Intelligence — the persisted output of the Computer Vision Engine.

The Computer Vision Engine analyzes aerial/satellite imagery and extracts
insights (buildings, market potential, FTTH opportunity, coverage gaps) that
feed the Marketing Intelligence Engine downstream.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.config import RESULTS_DIR

ZONE_INTELLIGENCE_FILE = RESULTS_DIR / "zone_intelligence.json"


@dataclass
class ZoneIntelligence:
    source: str
    zone_id: str | None
    zone_name: str | None
    buildings_detected: int
    cars_detected: int
    total_objects: int
    market_potential_households: int
    population_density_per_km2: float | None
    ftth_opportunity: str
    coverage_gap_pct: float | None
    detection_mode: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def save_zone_intelligence(zi: dict) -> None:
    """Append a Zone Intelligence record so downstream stages can consume it."""
    records = load_zone_intelligence()
    records.append(zi)
    ZONE_INTELLIGENCE_FILE.write_text(json.dumps(records, indent=2))


def load_zone_intelligence() -> list[dict]:
    if not ZONE_INTELLIGENCE_FILE.exists():
        return []
    try:
        return json.loads(ZONE_INTELLIGENCE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def latest_zone_intelligence() -> dict | None:
    records = load_zone_intelligence()
    return records[-1] if records else None
