"""Data loading, caching, and demo data generation for FALCON."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from app.config import IMAGES_DIR, RESULTS_DIR, ZONES_FILE
from app.image_generator import (get_vehicle_count_for_zone,
                                  save_zone_images)
from app.scorer import ZoneScorer


def load_zones() -> dict:
    with open(ZONES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_demo_vehicle_counts(zones: list) -> dict:
    return {z["id"]: get_vehicle_count_for_zone(z) for z in zones}


def _images_complete(zones: list) -> bool:
    return all((IMAGES_DIR / f"{z['id']}.png").exists() for z in zones)


def initialize_demo_data(force: bool = False) -> dict:
    """Generate all demo images + scores; cache to disk. Returns results dict."""
    zones_data = load_zones()
    zones = zones_data["zones"]
    cache_path = RESULTS_DIR / "demo_results.pkl"

    if not force and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[FALCON] Generating synthetic aerial images ...")
    image_results = save_zone_images(zones, str(IMAGES_DIR))

    vehicle_counts = {zid: d["vehicle_count"] for zid, d in image_results.items()}

    scorer = ZoneScorer(zones_data)
    scored = scorer.score_all_zones(vehicle_counts)

    results = {
        "vehicle_counts": vehicle_counts,
        "scored_zones": scored,
        "image_results": image_results,
        "mode": "demo",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[FALCON] Demo data ready -- {len(zones)} zones, "
          f"{sum(vehicle_counts.values())} total vehicles.")
    return results


def load_network_data() -> dict | None:
    """Load Waheb/RAMI/Don network JSON files if they exist."""
    from app.network_generator import load_network_files
    from app.config import DATA_DIR
    return load_network_files(DATA_DIR)


def load_or_generate_results(zones_data: dict, demo_mode: bool = True) -> dict:
    zones = zones_data["zones"]
    cache_path = RESULTS_DIR / "demo_results.pkl"

    if demo_mode and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Verify image files still exist; regenerate if not
        if _images_complete(zones):
            return cached

    # Generate fresh
    return initialize_demo_data(force=True)
