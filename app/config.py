from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "sample_images"
RESULTS_DIR = DATA_DIR / "results"
ZONES_FILE = DATA_DIR / "zones.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

APP_TITLE = "FALCON"
APP_SUBTITLE = "Flavia AI Coverage & Location Opportunity Network"
APP_ICON = "🦅"

TIER_COLORS = {
    "High Priority": "#dc3545",
    "Medium Priority": "#fd7e14",
    "Low Priority": "#28a745",
}

TIER_FOLIUM_COLORS = {
    "High Priority": "red",
    "Medium Priority": "orange",
    "Low Priority": "green",
}

SCORE_WEIGHTS = {
    "vehicle_density": 0.40,
    "coverage_gap": 0.35,
    "population_density": 0.25,
}

MAX_DENSITY_REF = 50.0
COST_PER_COVERAGE_POINT_OMR = 80000
ARPU_MONTHLY_OMR = 5.5
HOUSEHOLD_SIZE = 4
FTTH_UPTAKE_RATE = 0.42

# ── Competitive landscape ─────────────────────────────────────────────────────
WAHEB_COLOR = "#02C39A"
RAMI_COLOR  = "#065A82"
DON_COLOR   = "#E84545"

WAHEB_MARKET_SHARE       = 10.0
RAMI_MARKET_SHARE        = 35.0
DON_MARKET_SHARE         = 25.0
UNSERVED_MARKET_SHARE    = 30.0
TOTAL_MUSCAT_HOMES       = 350_000
WAHEB_TARGET_SHARE_2026  = 25.0

COST_PER_CABINET_OMR     = 25_000
HOMES_PER_CABINET        = 150

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
