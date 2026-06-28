"""YOLOv8 vehicle detection engine.

Two detection modes:
  standard  — single-pass inference on the full image (good for synthetic 640×640)
  aerial    — 4× upscale + sliding-window tiling + IoU-NMS (needed for real
              satellite / Google Earth screenshots where vehicles are <10 px)
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# Classes used by the generic COCO pre-trained YOLOv8 weights
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Classes produced by scripts/train_yolo.py (custom FTTH dataset)
CUSTOM_CLASSES  = {0: "building", 1: "car"}

# Aerial detection parameters
_AERIAL_SCALE    = 4       # upscale factor before tiling
_AERIAL_TILE     = 640     # tile size (matches YOLO input)
_AERIAL_STEP     = 480     # step between tiles (75% overlap)
_AERIAL_CONF     = 0.08    # lower confidence for small aerial vehicles
_AERIAL_NMS_IOU  = 0.40    # IoU threshold for duplicate suppression
_AERIAL_CONTRAST = 1.3     # contrast enhancement multiplier
_AERIAL_SHARP    = 1.5     # sharpness enhancement multiplier


class DetectionResult:
    def __init__(self, image_path: str, boxes: list, count: int,
                 source: str = "yolo"):
        self.image_path = image_path
        self.boxes = boxes  # list of (x1, y1, x2, y2, label, conf)
        self.count = count
        self.source = source

    def to_dict(self) -> dict:
        return {
            "image_path":    self.image_path,
            "boxes":         self.boxes,
            "vehicle_count": self.count,
            "source":        self.source,
        }


def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(ua + ub - inter, 1e-6)


def _nms(boxes: list, iou_thresh: float = _AERIAL_NMS_IOU) -> list:
    boxes = sorted(boxes, key=lambda x: -x[5])
    kept: list = []
    for box in boxes:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


class VehicleDetector:
    def __init__(self, model_size: str = "n"):
        self._model        = None
        self._model_size   = model_size
        self._available    = False
        self._class_map    = VEHICLE_CLASSES  # updated when custom model is loaded
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
            from app.config import YOLO_CUSTOM_MODEL_PATH

            custom = Path(YOLO_CUSTOM_MODEL_PATH)
            if custom.exists():
                self._model     = YOLO(str(custom))
                self._class_map = CUSTOM_CLASSES
                logger.info("Custom FTTH model loaded from %s", custom)
            else:
                self._model     = YOLO(f"yolov8{self._model_size}.pt")
                self._class_map = VEHICLE_CLASSES
                logger.info("YOLOv8-%s (COCO pre-trained) loaded", self._model_size)

            self._available = True
        except Exception as exc:
            logger.warning("YOLO model unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    # ── Single-pass inference ─────────────────────────────────────────────────

    def detect(self, image_path: str, conf: float = 0.25,
               imgsz: int = 640) -> DetectionResult:
        if not self._available:
            return DetectionResult(image_path, [], 0, source="unavailable")
        try:
            results = self._model(image_path, conf=conf, imgsz=imgsz, verbose=False)
            boxes   = self._parse_results(results)
            return DetectionResult(image_path, boxes, len(boxes), source="yolo")
        except Exception as exc:
            logger.error("Detection error on %s: %s", image_path, exc)
            return DetectionResult(image_path, [], 0, source="error")

    def detect_image_object(self, img: Image.Image,
                             conf: float = 0.25) -> DetectionResult:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        try:
            img.convert("RGB").save(tmp)
            return self.detect(tmp, conf=conf)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ── Aerial / satellite inference (upscale + sliding window) ──────────────

    def detect_aerial(
        self,
        img: Image.Image,
        scale: int   = _AERIAL_SCALE,
        conf: float  = _AERIAL_CONF,
        tile: int    = _AERIAL_TILE,
        step: int    = _AERIAL_STEP,
    ) -> DetectionResult:
        """
        Detect vehicles in Google Earth / satellite imagery.

        Pipeline:
          1. Upscale image by `scale` (default 4×) so vehicles reach ~20+ px.
          2. Apply contrast + sharpness enhancement.
          3. Slide a 640×640 window across the upscaled image.
          4. Run YOLO on each tile, map bboxes back to original coordinates.
          5. Deduplicate overlapping boxes with IoU-NMS.

        Returns DetectionResult with source="yolo_aerial".
        """
        if not self._available:
            return DetectionResult("", [], 0, source="unavailable")

        img_rgb = img.convert("RGB")
        W0, H0  = img_rgb.size

        # Upscale
        img_up = img_rgb.resize((W0 * scale, H0 * scale), Image.LANCZOS)
        img_up = ImageEnhance.Contrast(img_up).enhance(_AERIAL_CONTRAST)
        img_up = ImageEnhance.Sharpness(img_up).enhance(_AERIAL_SHARP)
        W, H   = img_up.size

        img_arr = np.array(img_up)
        cols    = max(1, math.ceil((W - tile) / step) + 1)
        rows    = max(1, math.ceil((H - tile) / step) + 1)

        raw_boxes: list = []
        for row in range(rows):
            y = min(row * step, max(0, H - tile))
            for col in range(cols):
                x     = min(col * step, max(0, W - tile))
                patch = Image.fromarray(img_arr[y:y + tile, x:x + tile])
                try:
                    results = self._model(patch, conf=conf, imgsz=tile, verbose=False)
                except Exception as exc:
                    logger.warning("Tile (%d,%d) failed: %s", col, row, exc)
                    continue

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        if cls_id not in self._class_map:
                            continue
                        bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                        # Map back to original (non-upscaled) coordinates
                        ox1 = (bx1 + x) / scale
                        oy1 = (by1 + y) / scale
                        ox2 = (bx2 + x) / scale
                        oy2 = (by2 + y) / scale
                        raw_boxes.append((
                            ox1, oy1, ox2, oy2,
                            self._class_map[cls_id],
                            round(float(box.conf.item()), 3),
                        ))

        kept = _nms(raw_boxes)
        # Convert float coords to int for downstream use
        int_boxes = [
            (int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4], b[5])
            for b in kept
        ]
        logger.info(
            "Aerial detection: %d raw → %d after NMS (scale=%dx, tiles=%d)",
            len(raw_boxes), len(kept), scale, cols * rows,
        )
        return DetectionResult("", int_boxes, len(int_boxes), source="yolo_aerial")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_results(self, results) -> list:
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in self._class_map:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append((
                    x1, y1, x2, y2,
                    self._class_map[cls_id],
                    round(float(box.conf.item()), 3),
                ))
        return boxes


# ── Module-level singleton ─────────────────────────────────────────────────────

_detector: Optional[VehicleDetector] = None


def get_detector(model_size: str = "n") -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector(model_size)
    return _detector
