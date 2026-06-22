"""
Fetch two real satellite patches from the Ghubrah zone (Muscat, Oman)
using ESRI World Imagery tiles and run YOLOv8 aerial detection on each.

Two sub-locations:
  G-A  23.6040, 58.5000  – northern Ghubrah (near Bareeq Al Shatti Rd)
  G-B  23.5910, 58.4960  – southern Ghubrah (dense residential grid)

Zoom 17 gives ~1.2 m/px → cars appear as 10-15 px rectangles, enough for
the 4× upscale + sliding-window pipeline in detector.py to find them.
"""
from __future__ import annotations

import io
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "ghubrah_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Satellite tile source (ESRI World Imagery – free, no key needed) ──────────
ESRI_TILE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_TILE_PX = 256


def _deg2tile(lat: float, lng: float, zoom: int) -> tuple[int, int]:
    lat_r = math.radians(lat)
    n = 2 ** zoom
    tx = int((lng + 180.0) / 360.0 * n)
    ty = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return tx, ty


def _fetch_tile(tx: int, ty: int, zoom: int) -> Image.Image | None:
    import requests
    url = ESRI_TILE.format(z=zoom, y=ty, x=tx)
    headers = {"User-Agent": "FALCON/1.0 (non-commercial research)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        print(f"    HTTP {r.status_code} for tile z={zoom} x={tx} y={ty}")
    except Exception as e:
        print(f"    Tile fetch error: {e}")
    return None


def download_satellite_patch(lat: float, lng: float, zoom: int = 17,
                              grid: int = 3) -> Image.Image | None:
    """Download grid×grid tiles centred on (lat,lng) → 640×640 composite."""
    cx, cy = _deg2tile(lat, lng, zoom)
    half = grid // 2
    canvas_size = _TILE_PX * grid
    canvas = Image.new("RGB", (canvas_size, canvas_size), (180, 180, 180))
    ok = 0
    for dx in range(grid):
        for dy in range(grid):
            tile = _fetch_tile(cx - half + dx, cy - half + dy, zoom)
            if tile:
                canvas.paste(tile, (dx * _TILE_PX, dy * _TILE_PX))
                ok += 1
            time.sleep(0.15)
    if ok == 0:
        return None
    print(f"    Downloaded {ok}/{grid*grid} tiles at z={zoom}")
    return canvas.resize((640, 640), Image.LANCZOS)


# ── YOLOv8 aerial detection ───────────────────────────────────────────────────

def run_yolo_aerial(img: Image.Image) -> list:
    """Run the aerial (upscale + sliding-window) detector. Returns bbox list."""
    from app.detector import get_detector
    detector = get_detector(model_size="n")
    if not detector.available:
        print("  [!] YOLOv8 not available – install: pip install ultralytics")
        return []
    result = detector.detect_aerial(img)
    return result.boxes


def draw_boxes(img: Image.Image, boxes: list) -> Image.Image:
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    COLORS = {"car": (0, 255, 0), "truck": (255, 128, 0),
               "bus": (255, 0, 0), "motorcycle": (0, 128, 255)}
    for b in boxes:
        x1, y1, x2, y2, label, conf = b
        color = COLORS.get(label, (200, 200, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"{label} {conf:.2f}"
        tw = len(tag) * 5 + 4
        draw.rectangle([x1, max(0, y1 - 11), x1 + tw, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 10)), tag, fill=(0, 0, 0))
    return annotated


# ── Ghubrah sub-patches ───────────────────────────────────────────────────────

PATCHES = [
    {
        "id":   "GHB-A",
        "name": "Ghubrah North (Bareeq Al Shatti corridor)",
        "lat":  23.6040,
        "lng":  58.5000,
    },
    {
        "id":   "GHB-B",
        "name": "Ghubrah South (residential grid)",
        "lat":  23.5910,
        "lng":  58.4960,
    },
]


def main():
    print("=" * 62)
    print("  FALCON – Real Satellite + YOLOv8  |  Ghubrah, Muscat")
    print("=" * 62)

    summary = []
    for p in PATCHES:
        pid   = p["id"]
        name  = p["name"]
        lat   = p["lat"]
        lng   = p["lng"]

        print(f"\n[{pid}] {name}")
        print(f"      coords  : {lat}, {lng}")

        img_path  = OUT_DIR / f"{pid}.png"
        ann_path  = OUT_DIR / f"{pid}_annotated.png"

        # ── Download ──────────────────────────────────────────────────────────
        print("  Downloading ESRI satellite tiles …")
        img = download_satellite_patch(lat, lng, zoom=17, grid=3)
        if img is None:
            print("  [!] All tile downloads failed – skipping.")
            continue
        img.save(img_path)
        print(f"  Saved base image -> {img_path.name}")

        # ── Detect ───────────────────────────────────────────────────────────
        print("  Running YOLOv8 aerial detection …")
        boxes = run_yolo_aerial(img)
        print(f"  Detected {len(boxes)} vehicles")

        by_class: dict[str, int] = {}
        for b in boxes:
            by_class[b[4]] = by_class.get(b[4], 0) + 1
        for cls, cnt in sorted(by_class.items()):
            print(f"    {cls:12s}: {cnt}")

        # ── Annotate & save ──────────────────────────────────────────────────
        annotated = draw_boxes(img, boxes)
        annotated.save(ann_path)
        print(f"  Saved annotated  -> {ann_path.name}")

        summary.append({
            "id":      pid,
            "name":    name,
            "vehicles": len(boxes),
            "by_class": by_class,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary:
        print("\n" + "=" * 62)
        print("  SUMMARY")
        print("=" * 62)
        total = 0
        for s in summary:
            print(f"  [{s['id']}] {s['name']}")
            print(f"         vehicles detected: {s['vehicles']}")
            total += s['vehicles']
        print(f"\n  Total vehicles across both patches: {total}")
        print(f"  Output folder: {str(OUT_DIR)}")
    print()


if __name__ == "__main__":
    main()
