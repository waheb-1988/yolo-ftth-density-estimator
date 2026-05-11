"""
FALCON configuration.
All tuneable values are loaded from environment variables (or .env file).
Hard-coded defaults are business constants that rarely change.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env if present — must happen before any os.getenv() calls
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; fall back to OS environment

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = Path(os.getenv("FALCON_DATA_DIR",    str(PROJECT_ROOT / "data")))
OUTPUTS_DIR  = Path(os.getenv("FALCON_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")))
LOGS_DIR     = PROJECT_ROOT / "logs"

IMAGES_DIR   = DATA_DIR / "sample_images"
RESULTS_DIR  = DATA_DIR / "results"
ZONES_FILE   = DATA_DIR / "zones.json"

# Create required directories at startup (not at every import)
for _d in [IMAGES_DIR, RESULTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE  = os.getenv("LOG_TO_FILE", "true").lower() == "true"

# ── Application identity ───────────────────────────────────────────────────────
APP_TITLE    = os.getenv("FALCON_APP_TITLE", "FALCON")
APP_SUBTITLE = "Flavia AI Coverage & Location Opportunity Network"
APP_ICON     = "🦅"

# ── UI colours ─────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "High Priority":   "#dc3545",
    "Medium Priority": "#fd7e14",
    "Low Priority":    "#28a745",
}

TIER_FOLIUM_COLORS = {
    "High Priority":   "red",
    "Medium Priority": "orange",
    "Low Priority":    "green",
}

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
SCORE_WEIGHTS = {
    "vehicle_density":    float(os.getenv("WEIGHT_VEHICLE_DENSITY",    "0.40")),
    "coverage_gap":       float(os.getenv("WEIGHT_COVERAGE_GAP",       "0.35")),
    "population_density": float(os.getenv("WEIGHT_POPULATION_DENSITY", "0.25")),
}

MAX_DENSITY_REF = float(os.getenv("MAX_DENSITY_REF", "50.0"))

# ── Financial constants (OMR) ──────────────────────────────────────────────────
COST_PER_COVERAGE_POINT_OMR = float(os.getenv("COST_PER_COVERAGE_POINT_OMR", "80000"))
ARPU_MONTHLY_OMR            = float(os.getenv("ARPU_MONTHLY_OMR",            "5.5"))
HOUSEHOLD_SIZE              = float(os.getenv("HOUSEHOLD_SIZE",              "4"))
FTTH_UPTAKE_RATE            = float(os.getenv("FTTH_UPTAKE_RATE",            "0.42"))
COST_PER_CABINET_OMR        = float(os.getenv("COST_PER_CABINET_OMR",        "25000"))
HOMES_PER_CABINET           = int(os.getenv("HOMES_PER_CABINET",            "150"))

# ── Competitive landscape ──────────────────────────────────────────────────────
WAHEB_COLOR  = os.getenv("WAHEB_COLOR", "#02C39A")
RAMI_COLOR   = os.getenv("RAMI_COLOR",  "#065A82")
DON_COLOR    = os.getenv("DON_COLOR",   "#E84545")

WAHEB_MARKET_SHARE      = float(os.getenv("WAHEB_MARKET_SHARE",      "10.0"))
RAMI_MARKET_SHARE       = float(os.getenv("RAMI_MARKET_SHARE",       "35.0"))
DON_MARKET_SHARE        = float(os.getenv("DON_MARKET_SHARE",        "25.0"))
UNSERVED_MARKET_SHARE   = float(os.getenv("UNSERVED_MARKET_SHARE",   "30.0"))
TOTAL_MUSCAT_HOMES      = int(os.getenv("TOTAL_MUSCAT_HOMES",        "350000"))
WAHEB_TARGET_SHARE_2026 = float(os.getenv("WAHEB_TARGET_SHARE_2026", "25.0"))

# ── YOLO model ─────────────────────────────────────────────────────────────────
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "n")   # n / s / m / l / x
YOLO_CONF       = float(os.getenv("YOLO_CONF", "0.25"))

# ── Satellite tile provider ────────────────────────────────────────────────────
TILE_PROVIDER   = os.getenv("TILE_PROVIDER", "osm")
USE_REAL_TILES  = os.getenv("USE_REAL_TILES", "false").lower() == "true"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST        = os.getenv("API_HOST", "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT", "8000"))
DASHBOARD_PORT  = int(os.getenv("DASHBOARD_PORT", "8501"))
