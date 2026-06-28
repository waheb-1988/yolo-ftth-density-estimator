"""
Train YOLOv8 on the FTTH COCO dataset (Roboflow export).

Pipeline
--------
1. Extract data/yolo-ftth-density-estimator.coco.zip
2. Reorganise images into images/{train,valid,test}/
3. Convert COCO bbox annotations → YOLO .txt labels in labels/{train,valid,test}/
4. Write data.yaml for ultralytics
5. Fine-tune yolov8n.pt for 100 epochs
6. Copy weights/best.pt → models/best.pt

Classes produced
----------------
  0  building
  1  car
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ZIP_PATH     = PROJECT_ROOT / "data" / "yolo-ftth-density-estimator.coco.zip"
DATASET_DIR  = PROJECT_ROOT / "data" / "ftth_yolo_dataset"
MODELS_DIR   = PROJECT_ROOT / "models"

# COCO category_id → YOLO class index (skip root category 0)
_CATEGORY_NAMES = {1: "building", 2: "car"}
_CAT_TO_CLASS   = {1: 0, 2: 1}
NC              = 2
CLASS_NAMES     = ["building", "car"]

SPLITS = {
    "train": "train",
    "valid": "valid",
    "test":  "test",
}


# ── Step 1 + 2: extract and reorganise ────────────────────────────────────────

def extract_dataset() -> None:
    if DATASET_DIR.exists():
        print(f"Dataset already extracted at {DATASET_DIR}, skipping extraction.")
        return

    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATASET_DIR / "_raw")

    for split_src, split_dst in SPLITS.items():
        src_dir = DATASET_DIR / "_raw" / split_src
        img_dst = DATASET_DIR / "images" / split_dst
        img_dst.mkdir(parents=True, exist_ok=True)

        for img_file in src_dir.glob("*.png"):
            shutil.copy2(img_file, img_dst / img_file.name)
        for img_file in src_dir.glob("*.jpg"):
            shutil.copy2(img_file, img_dst / img_file.name)

    print("Extraction complete.")


# ── Step 3: convert COCO → YOLO labels ────────────────────────────────────────

def _convert_split(split_src: str, split_dst: str) -> int:
    ann_file  = DATASET_DIR / "_raw" / split_src / "_annotations.coco.json"
    labels_dir = DATASET_DIR / "labels" / split_dst
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    id_to_img: dict = {img["id"]: img for img in coco["images"]}

    img_anns: dict = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    count = 0
    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_w  = img_info["width"]
        img_h  = img_info["height"]
        stem   = Path(img_info["file_name"]).stem

        lines = []
        for ann in img_anns.get(img_id, []):
            cls = _CAT_TO_CLASS.get(ann["category_id"])
            if cls is None:
                continue
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h
            # Clamp to [0, 1] to handle any annotation rounding errors
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            count += 1

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))

    return count


def convert_labels() -> None:
    total = 0
    for split_src, split_dst in SPLITS.items():
        n = _convert_split(split_src, split_dst)
        print(f"  {split_dst}: {n} annotations converted")
        total += n
    print(f"Total annotations: {total}")


# ── Step 4: write data.yaml ────────────────────────────────────────────────────

def write_data_yaml() -> Path:
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/valid\n"
        f"test:  images/test\n"
        f"\n"
        f"nc: {NC}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"data.yaml written to {yaml_path}")
    return yaml_path


# ── Step 5: train ──────────────────────────────────────────────────────────────

def train(yaml_path: Path, epochs: int = 100, imgsz: int = 640,
          batch: int = 8, model_size: str = "n") -> Path:
    from ultralytics import YOLO

    model = YOLO(f"yolov8{model_size}.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(MODELS_DIR / "runs"),
        name="ftth_train",
        save=True,
        patience=20,
        augment=True,
        degrees=10,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


# ── Step 6: save best model ────────────────────────────────────────────────────

def save_best(best_src: Path) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_dst = MODELS_DIR / "best.pt"
    shutil.copy2(best_src, best_dst)
    print(f"\nBest model saved -> {best_dst}")
    return best_dst


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"COCO zip not found: {ZIP_PATH}")

    print("=" * 60)
    print("FTTH YOLOv8 Training Pipeline")
    print("=" * 60)

    print("\n[1/4] Extracting dataset …")
    extract_dataset()

    print("\n[2/4] Converting COCO -> YOLO labels ...")
    convert_labels()

    print("\n[3/4] Writing data.yaml ...")
    yaml_path = write_data_yaml()

    print("\n[4/4] Training YOLOv8n ...")
    best_src = train(yaml_path)

    save_best(best_src)

    print("\nDone. Update YOLO_CUSTOM_MODEL_PATH in .env (or config.py) to use this model.")


if __name__ == "__main__":
    main()
