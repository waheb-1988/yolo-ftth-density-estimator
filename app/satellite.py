"""
Satellite / map tile image acquisition for FALCON.

Two modes:
  demo  — returns None; caller falls back to synthetic image generator.
  real  — downloads OpenStreetMap (or configurable) tiles and composites
          them into a 640×640 image suitable for YOLO inference.

Usage:
  img, meta = capture_area_image(lat, lng, use_real_tiles=True)
  if img is None:
      img, bboxes = generate_aerial_image(zone_id, vehicle_count)
"""
from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ── Tile providers ────────────────────────────────────────────────────────────
TILE_PROVIDERS: dict[str, str] = {
    "osm":            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "stadia_terrain": "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",
    "stadia_toner":   "https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png",
}

_TILE_SIZE = 256  # pixels per OSM tile


# ── Coordinate math ───────────────────────────────────────────────────────────

def _deg2tile(lat: float, lng: float, zoom: int) -> tuple[int, int]:
    """Convert WGS-84 lat/lng to OSM tile (x, y) indices."""
    import math
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r))
             / math.pi) / 2.0 * n)
    return x, y


# ── Tile download ─────────────────────────────────────────────────────────────

def _download_tile(x: int, y: int, zoom: int,
                   provider: str = "osm") -> Optional[Image.Image]:
    """Fetch a single 256×256 map tile. Returns None on any failure."""
    try:
        import requests  # lazy import — optional dep
    except ImportError:
        logger.warning("requests package not installed; cannot download tiles")
        return None

    url_tmpl = TILE_PROVIDERS.get(provider, TILE_PROVIDERS["osm"])
    url = url_tmpl.format(z=zoom, x=x, y=y)
    headers = {"User-Agent": "FALCON/1.0 (telecom network planning; non-commercial)"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        logger.warning("Tile %s returned HTTP %d", url, resp.status_code)
    except Exception as exc:
        logger.warning("Tile download error (%s): %s", url, exc)
    return None


# ── Composite image builder ────────────────────────────────────────────────────

def download_area_image(
    lat: float,
    lng: float,
    zoom: int = 16,
    tiles: int = 2,
    provider: str = "osm",
) -> Optional[Image.Image]:
    """
    Download an NxN grid of tiles centred on (lat, lng) and composite
    them into a single 640×640 image.

    Args:
        lat, lng  : Centre of the area of interest.
        zoom      : OSM zoom level. 15 = neighbourhood, 16 = street, 17 = building.
        tiles     : Grid dimension (2 → 2×2 = 4 tiles, 3 → 3×3 = 9 tiles).
        provider  : Key into TILE_PROVIDERS dict.

    Returns PIL Image or None if all tile downloads fail.
    """
    cx, cy = _deg2tile(lat, lng, zoom)
    offset  = tiles // 2

    canvas = Image.new("RGB", (_TILE_SIZE * tiles, _TILE_SIZE * tiles), color=(200, 200, 200))
    success = 0

    for dx in range(tiles):
        for dy in range(tiles):
            tile = _download_tile(cx + dx - offset, cy + dy - offset, zoom, provider)
            if tile:
                canvas.paste(tile, (dx * _TILE_SIZE, dy * _TILE_SIZE))
                success += 1
            time.sleep(0.08)  # polite rate limit

    if success == 0:
        logger.error("All tile downloads failed for (%.4f, %.4f) z=%d", lat, lng, zoom)
        return None

    logger.info("Downloaded %d/%d tiles for (%.4f, %.4f)", success, tiles * tiles, lat, lng)
    return canvas.resize((640, 640), Image.LANCZOS)


# ── Public API ────────────────────────────────────────────────────────────────

def capture_area_image(
    lat: float,
    lng: float,
    capture_time: Optional[datetime] = None,
    use_real_tiles: bool = False,
    provider: str = "osm",
    zoom: int = 16,
    tiles: int = 2,
) -> tuple[Optional[Image.Image], dict]:
    """
    Main entry point for satellite/map image acquisition.

    Returns (image_or_None, metadata_dict).
    When use_real_tiles=False (default / demo mode) the image is None and the
    caller should fall back to image_generator.generate_aerial_image().

    Metadata keys:
      lat, lng, capture_time (ISO), hour, is_peak_evening, source, provider.
    """
    if capture_time is None:
        capture_time = datetime.now()

    metadata = {
        "lat":             lat,
        "lng":             lng,
        "capture_time":    capture_time.isoformat(),
        "hour":            capture_time.hour,
        "is_peak_evening": 18 <= capture_time.hour <= 22,
        "source":          "tiles" if use_real_tiles else "synthetic",
        "provider":        provider,
        "zoom":            zoom,
    }

    if not use_real_tiles:
        return None, metadata

    img = download_area_image(lat, lng, zoom=zoom, tiles=tiles, provider=provider)
    if img:
        return img, metadata

    logger.warning("Tile download failed — caller should use synthetic fallback")
    metadata["source"] = "synthetic_fallback"
    return None, metadata


def get_after_hours_metadata(lat: float, lng: float,
                              hour: int = 20) -> dict:
    """
    Return metadata stamped for a simulated after-6-PM capture.
    Useful for annotating synthetic images with realistic timestamps.
    """
    from datetime import date
    dt = datetime.combine(date.today(), datetime.min.time().replace(hour=hour))
    _, meta = capture_area_image(lat, lng, capture_time=dt, use_real_tiles=False)
    return meta
