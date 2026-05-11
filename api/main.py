"""
FALCON FastAPI Backend — REST API for telecom market analysis.

Endpoints:
  GET  /api/v1/health                      System health
  GET  /api/v1/zones                       All zones with scores
  GET  /api/v1/zones/{zone_id}             Single zone detail
  GET  /api/v1/zones/{zone_id}/forecast    Revenue forecast
  GET  /api/v1/zones/{zone_id}/telecom     Simulated telecom profile
  GET  /api/v1/market/summary              City-wide market overview
  GET  /api/v1/market/clusters             Zone cluster analysis
  GET  /api/v1/market/share-projection     S-curve market share forecast
  POST /api/v1/detect                      Upload image → YOLO vehicle count
  GET  /api/v1/report/csv                  Download CSV report
  GET  /api/v1/report/summary              Executive summary text

Launch:
  uvicorn api.main:app --reload --port 8000
  or: python api/main.py
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Optional

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging (init before any FALCON imports) ──────────────────────────────────
from app.config import LOG_LEVEL, LOG_TO_FILE, LOGS_DIR  # noqa: E402
from app.logger import setup_logging

setup_logging(level=LOG_LEVEL, log_to_file=LOG_TO_FILE, log_dir=LOGS_DIR)
logger = logging.getLogger(__name__)

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import APP_SUBTITLE, APP_TITLE
from app.data_loader import (load_network_data, load_or_generate_results,
                              load_zones)
from app.forecaster import (cluster_zones, forecast_revenue,
                             market_share_projection)
from app.report_exporter import export_csv, generate_executive_summary
from app.telecom_simulator import TelecomSimulator

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"{APP_TITLE} API",
    description=(
        f"**{APP_TITLE}** — {APP_SUBTITLE}\n\n"
        "AI-powered FTTH coverage optimisation & competitive intelligence "
        "for Muscat, Oman."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (lazy-loaded on first request) ─────────────────────────────────
_zones_data:   Optional[dict] = None
_results:      Optional[dict] = None
_network_data: Optional[dict] = None
_simulator = TelecomSimulator()


def _get_zones() -> dict:
    global _zones_data
    if _zones_data is None:
        _zones_data = load_zones()
    return _zones_data


def _get_results() -> dict:
    global _results
    if _results is None:
        _results = load_or_generate_results(_get_zones(), demo_mode=True)
    return _results


def _get_network() -> Optional[dict]:
    global _network_data
    if _network_data is None:
        _network_data = load_network_data()
    return _network_data


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    system: str
    version: str


class ZoneListResponse(BaseModel):
    zones: list
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health() -> dict:
    """Service liveness check."""
    return {"status": "ok", "system": APP_TITLE, "version": "1.0.0"}


# ── Zone endpoints ────────────────────────────────────────────────────────────

@app.get("/api/v1/zones", tags=["Zones"])
def list_zones(
    tier: Optional[str] = Query(None, description="Filter: High Priority | Medium Priority | Low Priority"),
) -> dict:
    """Return all scored zones, optionally filtered by priority tier."""
    scored = _get_results()["scored_zones"]
    if tier:
        scored = [z for z in scored if z["tier"].lower() == tier.lower()]
    return {"zones": scored, "total": len(scored)}


@app.get("/api/v1/zones/{zone_id}", tags=["Zones"])
def get_zone(zone_id: str) -> dict:
    """Return detailed scored data for one zone."""
    for z in _get_results()["scored_zones"]:
        if z["zone_id"] == zone_id:
            return z
    raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")


@app.get("/api/v1/zones/{zone_id}/forecast", tags=["Zones"])
def zone_forecast(
    zone_id: str,
    years: int = Query(5, ge=1, le=10, description="Projection horizon"),
    growth_rate: float = Query(0.08, ge=0, le=0.5, description="Annual subscriber growth rate"),
) -> dict:
    """5-year (configurable) revenue forecast for a zone."""
    zone = next(
        (z for z in _get_results()["scored_zones"] if z["zone_id"] == zone_id),
        None,
    )
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    return forecast_revenue(zone, years=years, growth_rate=growth_rate)


@app.get("/api/v1/zones/{zone_id}/telecom", tags=["Zones"])
def zone_telecom(zone_id: str) -> dict:
    """Simulated 5G/FTTH/mobile telecom profile for a zone."""
    zones_data = _get_zones()
    zone = next((z for z in zones_data["zones"] if z["id"] == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")

    profile     = _simulator.simulate_zone(zone)
    competitors = _simulator.simulate_competitors(zone)
    return {
        "profile":     profile.to_dict(),
        "competitors": [c.to_dict() for c in competitors],
    }


# ── Market endpoints ──────────────────────────────────────────────────────────

@app.get("/api/v1/market/summary", tags=["Market"])
def market_summary() -> dict:
    """City-wide competitive market summary."""
    network = _get_network()
    if network:
        return network.get("market_summary", {})
    return {"message": "Run `python scripts/build_demo.py` to generate network data"}


@app.get("/api/v1/market/telecom", tags=["Market"])
def market_telecom() -> dict:
    """Full simulated telecom market across all zones."""
    return _simulator.simulate_all(_get_zones())


@app.get("/api/v1/market/clusters", tags=["Market"])
def market_clusters(
    n_clusters: int = Query(3, ge=2, le=5, description="Number of K-Means clusters"),
) -> dict:
    """Segment zones into clusters using K-Means."""
    results    = _get_results()
    zones_data = _get_zones()
    df = cluster_zones(results["scored_zones"], zones_data, n_clusters=n_clusters)
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="Clustering unavailable — install scikit-learn: pip install scikit-learn",
        )
    return {"clusters": df.to_dict(orient="records"), "n_clusters": n_clusters}


@app.get("/api/v1/market/share-projection", tags=["Market"])
def share_projection(
    current: float = Query(10.0, ge=0, le=100),
    target:  float = Query(25.0, ge=0, le=100),
    years:   int   = Query(5, ge=1, le=10),
) -> dict:
    """Logistic (S-curve) market share growth projection."""
    shares = market_share_projection(current, target, years=years)
    return {
        "current_share":     current,
        "target_share":      target,
        "years":             list(range(1, years + 1)),
        "projected_shares":  shares,
    }


# ── Detection endpoint ────────────────────────────────────────────────────────

@app.post("/api/v1/detect", tags=["Detection"])
async def detect_vehicles(file: UploadFile = File(...)) -> dict:
    """
    Upload a satellite/aerial image → run YOLOv8 → return vehicle detections.
    Accepts: JPEG, PNG, BMP, TIFF.
    """
    try:
        from PIL import Image

        from app.detector import get_detector

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        detector = get_detector("n")
        if not detector.available:
            raise HTTPException(
                status_code=503,
                detail=(
                    "YOLOv8 model not available. "
                    "Ensure `ultralytics` is installed and internet is accessible."
                ),
            )

        result = detector.detect_image_object(img)
        return {
            "vehicle_count": result.count,
            "detections": [
                {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                 "label": b[4], "confidence": b[5]}
                for b in result.boxes
            ],
            "source": result.source,
            "filename": file.filename,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Detection endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Report endpoints ──────────────────────────────────────────────────────────

@app.get("/api/v1/report/csv", tags=["Reports"])
def download_csv() -> StreamingResponse:
    """Download full zone analysis as CSV."""
    results    = _get_results()
    zones_data = _get_zones()
    csv_data   = export_csv(results["scored_zones"], zones_data)
    return StreamingResponse(
        io.StringIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=falcon_report.csv"},
    )


@app.get("/api/v1/report/summary", tags=["Reports"])
def get_summary() -> dict:
    """Executive summary text."""
    results    = _get_results()
    zones_data = _get_zones()
    return {"summary": generate_executive_summary(results["scored_zones"], zones_data)}


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    from app.config import API_HOST, API_PORT

    logger.info("Starting FALCON API on %s:%d", API_HOST, API_PORT)
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
