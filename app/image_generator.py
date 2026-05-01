"""Synthetic aerial/top-down urban image generator for FALCON demo mode."""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _seed_rng(zone_id: str, seed: int = 42):
    h = abs(hash(zone_id)) % 99991
    return random.Random(seed + h), seed + h


def generate_aerial_image(zone_id: str, vehicle_count: int, seed: int = 42,
                           size: tuple = (640, 640)):
    """Return (PIL.Image base, list of bbox tuples (x1,y1,x2,y2,label,conf))."""
    rng, np_seed = _seed_rng(zone_id, seed)
    np.random.seed(np_seed % (2**31))
    W, H = size

    # ── Desert/urban base ─────────────────────────────────────────────────────
    rb = rng.randint(208, 228)
    gb = rng.randint(196, 214)
    bb = rng.randint(163, 183)
    base = np.full((H, W, 3), [rb, gb, bb], dtype=np.uint8)
    noise = (np.random.randn(H, W, 3) * 9).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # ── Major road grid ───────────────────────────────────────────────────────
    maj_w = rng.randint(24, 32)
    min_w = rng.randint(13, 17)
    road_col = (52, 52, 52)
    divider_col = (210, 210, 60)
    sidewalk_col = (170, 168, 158)

    n_h = rng.randint(2, 3)
    n_v = rng.randint(2, 3)
    h_roads = sorted([rng.randint(90, H - 90) for _ in range(n_h)])
    v_roads = sorted([rng.randint(90, W - 90) for _ in range(n_v)])

    def draw_road(draw, horiz, pos, width, major=True):
        hw = width // 2
        if horiz:
            draw.rectangle([0, pos - hw, W, pos + hw], fill=road_col)
            draw.rectangle([0, pos - hw, W, pos - hw + 3], fill=sidewalk_col)
            draw.rectangle([0, pos + hw - 3, W, pos + hw], fill=sidewalk_col)
            if major:
                draw.rectangle([0, pos - 1, W, pos + 1], fill=divider_col)
        else:
            draw.rectangle([pos - hw, 0, pos + hw, H], fill=road_col)
            draw.rectangle([pos - hw, 0, pos - hw + 3, H], fill=sidewalk_col)
            draw.rectangle([pos + hw - 3, 0, pos + hw, H], fill=sidewalk_col)
            if major:
                draw.rectangle([pos - 1, 0, pos + 1, H], fill=divider_col)

    for y in h_roads:
        draw_road(draw, True, y, maj_w, major=True)
    for x in v_roads:
        draw_road(draw, False, x, maj_w, major=True)

    # Minor roads between major blocks
    minor_h, minor_v = [], []
    for a, b in zip([0] + h_roads, h_roads + [H]):
        if b - a > 130:
            mid = (a + b) // 2 + rng.randint(-18, 18)
            minor_h.append(mid)
            draw_road(draw, True, mid, min_w, major=False)
    for a, b in zip([0] + v_roads, v_roads + [W]):
        if b - a > 130:
            mid = (a + b) // 2 + rng.randint(-18, 18)
            minor_v.append(mid)
            draw_road(draw, False, mid, min_w, major=False)

    all_h = sorted(h_roads + minor_h)
    all_v = sorted(v_roads + minor_v)

    # ── Buildings in blocks ───────────────────────────────────────────────────
    x_bounds = [0] + all_v + [W]
    y_bounds = [0] + all_h + [H]

    def fill_block(x1, y1, x2, y2):
        m = 6
        bx1, by1, bx2, by2 = x1 + m, y1 + m, x2 - m, y2 - m
        if bx2 - bx1 < 12 or by2 - by1 < 12:
            return
        if rng.random() < 0.45:
            s = rng.randint(148, 198)
            draw.rectangle([bx1, by1, bx2, by2], fill=(s, s - 7, s - 18))
            draw.rectangle([bx2, by1 + 2, bx2 + 3, by2 + 3], fill=(90, 90, 90))
            draw.rectangle([bx1 + 2, by2, bx2 + 3, by2 + 3], fill=(90, 90, 90))
        else:
            n_sub = rng.randint(2, 5)
            for _ in range(n_sub):
                sw = rng.randint(12, max(13, (bx2 - bx1) // 2))
                sh = rng.randint(12, max(13, (by2 - by1) // 2))
                sx = bx1 + rng.randint(0, max(1, bx2 - bx1 - sw))
                sy = by1 + rng.randint(0, max(1, by2 - by1 - sh))
                s = rng.randint(142, 192)
                draw.rectangle([sx, sy, sx + sw, sy + sh], fill=(s, s - 5, s - 14))

    for i in range(len(x_bounds) - 1):
        for j in range(len(y_bounds) - 1):
            x1 = x_bounds[i] + min_w // 2 + 3
            x2 = x_bounds[i + 1] - min_w // 2 - 3
            y1 = y_bounds[j] + min_w // 2 + 3
            y2 = y_bounds[j + 1] - min_w // 2 - 3
            if rng.random() < 0.82:
                fill_block(x1, y1, x2, y2)

    # ── Parking lots ──────────────────────────────────────────────────────────
    parking_zones = []
    for _ in range(rng.randint(1, 3)):
        pw = rng.randint(55, 110)
        ph = rng.randint(38, 68)
        px = rng.randint(10, W - pw - 10)
        py = rng.randint(10, H - ph - 10)
        draw.rectangle([px, py, px + pw, py + ph], fill=(68, 68, 68))
        for li in range(0, pw, 14):
            draw.line([px + li, py, px + li, py + ph], fill=(115, 115, 115), width=1)
        parking_zones.append((px, py, px + pw, py + ph))

    # ── Soft blur for realism ─────────────────────────────────────────────────
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    draw = ImageDraw.Draw(img)

    # ── Vehicle placement ─────────────────────────────────────────────────────
    slots = []
    for y in all_h:
        hw = maj_w // 2 if y in h_roads else min_w // 2
        for x in range(hw + 10, W - hw - 10, rng.randint(17, 26)):
            lane = rng.choice([-1, 1]) * max(3, hw // 2)
            slots.append(("h", x, y + lane))
    for x in all_v:
        hw = maj_w // 2 if x in v_roads else min_w // 2
        for y in range(hw + 10, H - hw - 10, rng.randint(17, 26)):
            lane = rng.choice([-1, 1]) * max(3, hw // 2)
            slots.append(("v", x + lane, y))
    for (px, py, px2, py2) in parking_zones:
        for col in range(px + 7, px2 - 7, 14):
            for row in range(py + 5, py2 - 5, 16):
                slots.append(("h", col, row))

    rng.shuffle(slots)

    CAR_COLORS = [
        (234, 234, 234), (38, 38, 38), (196, 48, 48),
        (48, 78, 196), (55, 128, 55), (188, 138, 45),
        (96, 96, 110), (198, 158, 45), (48, 148, 168),
        (175, 88, 45),
    ]

    bboxes = []
    actual = min(vehicle_count, len(slots))
    for i in range(actual):
        ori, cx, cy = slots[i]
        if ori == "h":
            cw, ch = rng.randint(12, 16), rng.randint(6, 8)
        else:
            cw, ch = rng.randint(6, 8), rng.randint(12, 16)

        x1 = max(1, cx - cw // 2)
        y1 = max(1, cy - ch // 2)
        x2 = min(W - 1, x1 + cw)
        y2 = min(H - 1, y1 + ch)
        if x2 <= x1 or y2 <= y1:
            continue

        col = rng.choice(CAR_COLORS)
        draw.rectangle([x1, y1, x2, y2], fill=col)
        if x2 - x1 > 4 and y2 - y1 > 4:
            wc = tuple(max(0, c - 52) for c in col)
            m2 = 2
            if ori == "h":
                draw.rectangle([x1 + m2, y1 + m2, x1 + cw // 2, y2 - m2], fill=wc)
            else:
                draw.rectangle([x1 + m2, y1 + m2, x2 - m2, y1 + ch // 2], fill=wc)

        conf = round(0.70 + rng.random() * 0.28, 3)
        bboxes.append((x1, y1, x2, y2, "car", conf))

    return img, bboxes


def draw_detections_on_image(img: Image.Image, bboxes: list,
                              box_color=(0, 255, 0), width=2) -> Image.Image:
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for b in bboxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        conf = b[5] if len(b) > 5 else 1.0
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=width)
        label = f"{conf:.2f}"
        lw = len(label) * 5 + 2
        draw.rectangle([x1, max(0, y1 - 10), x1 + lw, y1], fill=box_color)
        draw.text((x1 + 1, max(0, y1 - 9)), label, fill=(0, 0, 0))
    return annotated


def get_vehicle_count_for_zone(zone: dict, base_seed: int = 7) -> int:
    rng = random.Random(abs(hash(zone["id"])) % 99991 + base_seed)
    base = max(8, int(zone["population"] / 2100))
    mult = {"high": 1.95, "medium": 1.30, "low": 0.78}.get(
        zone.get("commercial_density", "medium"), 1.0)
    count = int(base * mult * (0.72 + rng.random() * 0.52))
    return max(5, min(count, 92))


def save_zone_images(zones: list, output_dir: str = "data/sample_images") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for zone in zones:
        zid = zone["id"]
        count = get_vehicle_count_for_zone(zone)
        img, bboxes = generate_aerial_image(
            zid, count, seed=abs(hash(zid)) % 49999)
        img_path = str(out / f"{zid}.png")
        ann_path = str(out / f"{zid}_annotated.png")
        img.save(img_path)
        draw_detections_on_image(img, bboxes).save(ann_path)
        results[zid] = {
            "image_path": img_path,
            "annotated_path": ann_path,
            "bboxes": bboxes,
            "vehicle_count": len(bboxes),
        }
    return results
