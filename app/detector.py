"""YOLOv8 vehicle detection engine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import Image


class DetectionResult:
    def __init__(self, image_path: str, boxes: list, count: int,
                 source: str = "yolo"):
        self.image_path = image_path
        self.boxes = boxes  # list of (x1, y1, x2, y2, label, conf)
        self.count = count
        self.source = source

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "boxes": self.boxes,
            "vehicle_count": self.count,
            "source": self.source,
        }


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_size: str = "n"):
        self._model = None
        self._model_size = model_size
        self._available = False
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(f"yolov8{self._model_size}.pt")
            self._available = True
        except Exception as e:
            print(f"[FALCON] YOLO model unavailable: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image_path: str, conf: float = 0.25,
               imgsz: int = 640) -> DetectionResult:
        if not self._available:
            return DetectionResult(image_path, [], 0, source="unavailable")

        try:
            results = self._model(
                image_path, conf=conf, imgsz=imgsz, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf_val = round(float(box.conf.item()), 3)
                    label = VEHICLE_CLASSES[cls_id]
                    boxes.append((x1, y1, x2, y2, label, conf_val))
            return DetectionResult(image_path, boxes, len(boxes), source="yolo")
        except Exception as e:
            print(f"[FALCON] Detection error on {image_path}: {e}")
            return DetectionResult(image_path, [], 0, source="error")

    def detect_image_object(self, img: Image.Image,
                             conf: float = 0.25) -> DetectionResult:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        try:
            img.save(tmp)
            return self.detect(tmp, conf=conf)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


_detector: Optional[VehicleDetector] = None


def get_detector(model_size: str = "n") -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector(model_size)
    return _detector
