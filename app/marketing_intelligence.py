"""Marketing Intelligence Engine output — persisted for the Executive Report
and Forecasting stages.

Consumes Zone Intelligence (Computer Vision Engine output) plus zone scoring
to produce market potential, customer opportunity, competitive positioning,
FTTH expansion recommendations, and target-area prioritization.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.config import RESULTS_DIR

MARKETING_INTELLIGENCE_FILE = RESULTS_DIR / "marketing_intelligence.json"


def save_marketing_intelligence(zones: list[dict]) -> None:
    """Persist the latest Marketing Intelligence snapshot (one record per zone)."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zones": zones,
    }
    MARKETING_INTELLIGENCE_FILE.write_text(json.dumps(payload, indent=2))


def load_marketing_intelligence() -> dict | None:
    if not MARKETING_INTELLIGENCE_FILE.exists():
        return None
    try:
        return json.loads(MARKETING_INTELLIGENCE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
