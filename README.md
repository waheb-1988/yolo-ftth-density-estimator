# FALCON — FTTH Coverage Optimizer

> AI-powered telecom market potential analysis for Muscat, Oman  
> YOLOv8 · Streamlit · FastAPI · Folium · scikit-learn

FALCON detects vehicles in satellite/aerial imagery, scores 15 Muscat zones by
demand density, simulates 5G/FTTH/mobile market data, and maps the full
competitive landscape (Waheb vs RAMI vs Don) to identify FTTH expansion
opportunities and revenue potential.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd yolo-ftth-density-estimator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full demo pipeline
python scripts/build_demo.py
```

Dashboard opens at **http://localhost:8501**

---

## Installation Options

### A — Local Python (recommended for development)

```bash
# Python 3.10 or 3.11 required
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Linux/macOS

pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env

# Run dashboard
python run.py --demo

# Run API (separate terminal)
uvicorn api.main:app --reload --port 8000
```

### B — Docker Compose (recommended for deployment)

```bash
# Build and start both services
docker compose up --build

# Dashboard → http://localhost:8501
# API       → http://localhost:8000
# API docs  → http://localhost:8000/docs

# Stop
docker compose down
```

### C — Dashboard only (minimal)

```bash
pip install streamlit streamlit-folium folium pandas numpy Pillow plotly
python run.py --demo
```

---

## Configuration

All settings are controlled via environment variables. Copy `.env.example`
to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `YOLO_MODEL_SIZE` | `n` | YOLOv8 size: n/s/m/l/x |
| `USE_REAL_TILES` | `false` | Download real OSM map tiles |
| `ARPU_MONTHLY_OMR` | `5.5` | Average revenue per user (OMR) |
| `FTTH_UPTAKE_RATE` | `0.42` | % of households that adopt FTTH |
| `HOMES_PER_CABINET` | `150` | Coverage per FTTH cabinet |
| `COST_PER_CABINET_OMR` | `25000` | Cabinet deployment cost (OMR) |
| `DASHBOARD_PORT` | `8501` | Streamlit port |
| `API_PORT` | `8000` | FastAPI port |

---

## Dashboard Workflow

The dashboard follows a 5-stage pipeline, each stage feeding the next:

| # | Tab | Content |
|---|---|---|
| 1 | 🗺️ Coverage Map | Waheb/RAMI/Don network layers, opportunity zones, priority markers |
| 2 | 🔬 Computer Vision Engine | Aerial image upload, YOLO building/car detection; extracts and persists **Zone Intelligence** (market potential, FTTH opportunity) for downstream stages |
| 3 | 📋 Executive Report | Business-facing report built from Marketing Intelligence outputs only — no raw detections or model details |
| 4 | 📊 Marketing Intelligence Engine | Market potential, customer opportunity, competitive positioning, FTTH expansion recommendations, target-area prioritization — persisted for the Executive Report and Forecasting stages |
| 5 | 📈 Forecasting | Telecom simulation, K-Means clustering, demand prediction, revenue/traffic forecasting — built on historical trends + Marketing Intelligence outputs |

---

## REST API

Once `uvicorn api.main:app --reload` is running, visit
**http://localhost:8000/docs** for the interactive Swagger UI.

### Key endpoints

```
GET  /api/v1/health                       System liveness
GET  /api/v1/zones                        All zones with priority scores
GET  /api/v1/zones/{zone_id}              Single zone detail
GET  /api/v1/zones/{zone_id}/forecast     5-year revenue forecast
GET  /api/v1/zones/{zone_id}/telecom      Simulated 5G/FTTH/mobile profile
GET  /api/v1/market/summary               City-wide competitive summary
GET  /api/v1/market/telecom               Full telecom market simulation
GET  /api/v1/market/clusters              K-Means zone segmentation
GET  /api/v1/market/share-projection      S-curve market share forecast
POST /api/v1/detect                       Upload image → YOLO vehicle count
GET  /api/v1/report/csv                   Download zone analysis as CSV
GET  /api/v1/report/summary               Executive summary text
```

### Example API calls

```bash
# Zone list
curl http://localhost:8000/api/v1/zones

# High-priority zones only
curl "http://localhost:8000/api/v1/zones?tier=High%20Priority"

# Revenue forecast for zone Z01
curl "http://localhost:8000/api/v1/zones/Z01/forecast?years=5&growth_rate=0.10"

# Upload an image for YOLO detection
curl -X POST http://localhost:8000/api/v1/detect \
     -F "file=@my_aerial_photo.jpg"

# Market clusters
curl "http://localhost:8000/api/v1/market/clusters?n_clusters=4"
```

---

## Architecture

```
yolo-ftth-density-estimator/
├── app/                        Core application package
│   ├── config.py               .env-driven configuration
│   ├── logger.py               Centralised logging setup
│   ├── detector.py             YOLOv8 vehicle detection engine
│   ├── image_generator.py      Synthetic 640×640 aerial images (PIL)
│   ├── satellite.py            Real map tile download (OSM)
│   ├── scorer.py               Zone priority scoring algorithm
│   ├── telecom_simulator.py    5G / FTTH / mobile market simulation
│   ├── forecaster.py           K-Means clustering + revenue forecasting
│   ├── network_generator.py    Waheb / RAMI / Don FTTH asset generator
│   ├── data_loader.py          Zone loading, result caching
│   ├── map_builder.py          Folium 6-layer competitive map
│   ├── report_exporter.py      CSV + executive summary export
│   ├── zone_intelligence.py    Computer Vision Engine output (persisted)
│   ├── marketing_intelligence.py  Marketing Intelligence Engine output (persisted)
│   └── dashboard.py            Streamlit 5-stage workflow dashboard
├── api/
│   ├── __init__.py
│   └── main.py                 FastAPI backend (12 endpoints)
├── scripts/
│   ├── build_demo.py           Full pipeline launcher
│   └── create_presentation.py  PowerPoint generator (17 slides)
├── data/
│   ├── zones.json              15 Muscat zones metadata
│   ├── waheb_network.json      237 Waheb FTTH assets
│   ├── rami_network.json       543 RAMI FTTH assets
│   ├── don_network.json        378 Don FTTH assets
│   ├── market_summary.json     Shares, intensity, opportunity zones
│   ├── sample_images/          640×640 synthetic aerial images
│   └── results/                Cached scoring results (.pkl)
├── outputs/
│   ├── falcon_competitive_map.html
│   ├── FALCON_Report.csv
│   └── FALCON_Executive_Summary.txt
├── logs/                       Rotating log files
├── .env.example                Environment template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Scoring Algorithm

**Priority Score (0–100)**  
`= 40% × (vehicle_density / 50) + 35% × coverage_gap% + 25% × (population / 100k)`

Weights are tunable via `.env`:
```
WEIGHT_VEHICLE_DENSITY=0.40
WEIGHT_COVERAGE_GAP=0.35
WEIGHT_POPULATION_DENSITY=0.25
```

| Tier | Score range |
|---|---|
| High Priority | ≥ 68 |
| Medium Priority | 40–68 |
| Low Priority | < 40 |

---

## Business Model

| Company | Market Share | Cabinets | Strong Zones |
|---|---|---|---|
| **Waheb** (us) | **10%** | 45 | Ghubra, MQ, Bausher |
| RAMI (Omantel) | 35% | 95 | Ruwi, Mutrah, Wattayah |
| Don (Ooredoo) | 25% | 70 | Shatti, Azaiba, Al Khuwair |
| Unserved | 30% | — | ~105,000 homes = our opportunity |

**Revenue rule of thumb:** each 1% market share gain ≈ 700,000 OMR/year  
**Growth scenario (10% → 18%):** ~30 cabinets · 750,000 OMR investment · 13-month payback

---

## Real Satellite Imagery (Optional)

To use real map tiles instead of synthetic images:

```bash
# .env
USE_REAL_TILES=true
TILE_PROVIDER=osm     # osm | stadia_terrain | stadia_toner
```

Images are downloaded from OpenStreetMap (free, attribution required).
For commercial use consider Mapbox or Google Maps Static API.

---

## Deployment Checklist

- [ ] `cp .env.example .env` and set all production values
- [ ] Set `LOG_LEVEL=WARNING` for production (reduces noise)
- [ ] Set `USE_REAL_TILES=false` unless you have a tile API key
- [ ] Pre-generate data: `python scripts/build_demo.py`
- [ ] Run `docker compose up --build -d` for containerised deployment
- [ ] Confirm health: `curl http://localhost:8000/api/v1/health`
- [ ] Check dashboard: `http://localhost:8501`
- [ ] Review `logs/falcon.log` for startup errors
- [ ] Add reverse proxy (nginx/Caddy) in front of both ports for HTTPS

---

## AI / ML Modules

| Module | Algorithm | Purpose |
|---|---|---|
| `detector.py` | YOLOv8-nano (COCO) | Vehicle detection in aerial images |
| `scorer.py` | Weighted multi-factor score | Zone demand ranking |
| `forecaster.py` `cluster_zones()` | K-Means + StandardScaler | Zone segmentation |
| `forecaster.py` `forecast_revenue()` | Compound growth model | 5-year revenue projection |
| `forecaster.py` `market_share_projection()` | Logistic S-curve | Market share growth |
| `forecaster.py` `predict_demand_score()` | Feature-weighted score | No-YOLO demand proxy |
| `telecom_simulator.py` | Statistical simulation | 5G/FTTH/mobile market data |

---

## Known Limitations

1. **Synthetic imagery only** — real satellite images require a commercial API key or manual download
2. **Simulated telecom data** — no real operator data is used
3. **No authentication** — the API has no auth layer (add OAuth2 / API keys for production)
4. **Single-worker pickle cache** — `demo_results.pkl` is not thread-safe under concurrent writes
5. **Static zone boundaries** — zone polygons are not implemented; only centre-point coordinates
6. **No real-time data** — all data is generated at demo-build time

---

## Next Production Enhancements

1. **Real satellite pipeline** — integrate Google Earth Engine or Maxar API
2. **Database backend** — replace pickle cache with PostgreSQL + SQLAlchemy
3. **Authentication** — add JWT/API-key middleware to FastAPI
4. **Async YOLO** — run detection in a background task queue (Celery / ARQ)
5. **PostGIS** — store zone polygons and run true geospatial queries
6. **Live competitor feed** — monitor public FTTH coverage maps via web scraping
7. **CI/CD pipeline** — GitHub Actions: test → build Docker → push to registry
8. **Monitoring** — add Prometheus metrics + Grafana dashboard

---

*FALCON v1.1 · Muscat FTTH Competitive Intelligence · Waheb Telecom · 2026*
