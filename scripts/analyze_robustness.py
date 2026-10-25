"""Site-robustness breakdown of the fine-tuned model on the test set
(location 125, unseen). Produces data/processed/robustness_breakdown.csv
with one row per ground-truth box (day/night, size bucket, found-or-not),
plus a separate false-positive count on genuinely empty frames.

Usage: PYTHONPATH=. .venv/bin/python scripts/analyze_robustness.py
"""
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.eval.conditions import bbox_size_category, classify_day_night
from src.eval.metrics import get_matched_gt_mask

SPLIT = "test"
CONF_THRESHOLD = 0.25
MODEL_PATH = "models/cct20_yolov8n_best.pt"


def load_ground_truth(file_name: str, width: int, height: int) -> list[list[float]]:
    label_path = Path("data/yolo/labels") / SPLIT / f"{file_name.rsplit('.', 1)[0]}.txt"
    boxes = []
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().splitlines():
            _, xc, yc, w, h = map(float, line.split())
            x1, y1 = (xc - w / 2) * width, (yc - h / 2) * height
            x2, y2 = (xc + w / 2) * width, (yc + h / 2) * height
            boxes.append([x1, y1, x2, y2])
    return boxes


def main() -> None:
    manifest = pd.read_csv(f"data/processed/{SPLIT}_manifest.csv")
    model = YOLO(MODEL_PATH)

    box_rows = []
    empty_frame_rows = []

    for _, row in manifest.iterrows():
        image_path = Path("data/yolo/images") / SPLIT / row["file_name"]
        day_night = classify_day_night(str(image_path))

        result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False, device="mps")[0]
        boxes_confs = sorted(
            zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()), key=lambda bc: -bc[1]
        )
        pred_boxes = [b for b, _ in boxes_confs]

        gt_boxes_xyxy = load_ground_truth(row["file_name"], row["width"], row["height"])

        if not gt_boxes_xyxy:
            # Genuinely empty frame (no real annotation) - any prediction here is a false positive.
            empty_frame_rows.append({"file_name": row["file_name"], "day_night": day_night, "num_fp": len(pred_boxes)})
            continue

        matched = get_matched_gt_mask(pred_boxes, gt_boxes_xyxy, iou_threshold=0.5)
        for (x1, y1, x2, y2), found in zip(gt_boxes_xyxy, matched):
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            size_bucket = bbox_size_category(bbox_xywh, row["width"], row["height"])
            box_rows.append(
                {"file_name": row["file_name"], "day_night": day_night, "size_bucket": size_bucket, "found": found}
            )

    box_df = pd.DataFrame(box_rows)
    empty_df = pd.DataFrame(empty_frame_rows)

    out_dir = Path("data/processed")
    box_df.to_csv(out_dir / "robustness_breakdown.csv", index=False)
    empty_df.to_csv(out_dir / "robustness_empty_frames.csv", index=False)

    print("=== Recall by day/night ===")
    print(box_df.groupby("day_night")["found"].agg(["mean", "count"]))
    print()
    print("=== Recall by size bucket ===")
    print(box_df.groupby("size_bucket")["found"].agg(["mean", "count"]))
    print()
    print("=== False positives on empty frames, by day/night ===")
    print(empty_df.groupby("day_night")["num_fp"].agg(["mean", "sum", "count"]))


if __name__ == "__main__":
    main()
