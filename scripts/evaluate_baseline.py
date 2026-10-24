"""Zero-shot (no fine-tuning) baseline: class-agnostic detection quality of
a COCO-pretrained YOLOv8n on our camera-trap test set (unseen location 125).

Usage: PYTHONPATH=. .venv/bin/python scripts/evaluate_baseline.py
"""
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.eval.metrics import match_predictions

SPLIT = "test"
CONF_THRESHOLD = 0.25


def load_ground_truth(image_id: str, file_name: str, width: int, height: int) -> list[list[float]]:
    """Read a YOLO label .txt file and return ground-truth boxes as pixel xyxy."""
    label_path = Path("data/yolo/labels") / SPLIT / f"{file_name.rsplit('.', 1)[0]}.txt"
    boxes = []
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().splitlines():
            _, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height
            boxes.append([x1, y1, x2, y2])
    return boxes


def main() -> None:
    manifest = pd.read_csv(f"data/processed/{SPLIT}_manifest.csv")
    model = YOLO("yolov8n.pt")

    total_tp = total_fp = total_fn = 0

    for _, row in manifest.iterrows():
        image_path = Path("data/yolo/images") / SPLIT / row["file_name"]
        result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False, device="mps")[0]

        # Sort predictions by descending confidence, class-agnostic (ignore predicted label).
        boxes_confs = sorted(
            zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()),
            key=lambda bc: -bc[1],
        )
        pred_boxes = [b for b, _ in boxes_confs]

        gt_boxes = load_ground_truth(row["id"], row["file_name"], row["width"], row["height"])

        counts = match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += counts["tp"]
        total_fp += counts["fp"]
        total_fn += counts["fn"]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"class-agnostic precision={precision:.3f} recall={recall:.3f}")


if __name__ == "__main__":
    main()
