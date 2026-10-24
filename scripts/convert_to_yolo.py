"""Convert our CCT20 subset manifests into Ultralytics' YOLO dataset format.

Usage: PYTHONPATH=. .venv/bin/python scripts/convert_to_yolo.py
"""
import ast
from pathlib import Path

import pandas as pd
import yaml

from src.data.cct import load_pooled

LOCATIONS = {51, 108, 90, 125}
EMPTY_CATEGORY_ID = 30
PROCESSED_DIR = Path("data/processed")
IMAGES_SRC = Path("data/raw/images").resolve()
YOLO_DIR = Path("data/yolo")


def build_class_map() -> tuple[dict[int, int], list[str]]:
    """Map real (non-empty) category ids to contiguous YOLO class indices."""
    _, _, categories = load_pooled(LOCATIONS)
    real_category_ids = sorted(cid for cid in categories if cid != EMPTY_CATEGORY_ID)
    class_map = {cid: idx for idx, cid in enumerate(real_category_ids)}
    class_names = [categories[cid] for cid in real_category_ids]
    return class_map, class_names


def convert_split(split: str, class_map: dict[int, int], annotations_df: pd.DataFrame) -> None:
    manifest = pd.read_csv(PROCESSED_DIR / f"{split}_manifest.csv")

    images_out = YOLO_DIR / "images" / split
    labels_out = YOLO_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    anns_by_image = annotations_df.groupby("image_id")

    for _, row in manifest.iterrows():
        image_id, file_name = row["id"], row["file_name"]
        width, height = row["width"], row["height"]

        link_path = images_out / file_name
        if not link_path.exists():
            link_path.symlink_to(IMAGES_SRC / file_name)

        lines = []
        if image_id in anns_by_image.groups:
            for _, ann in anns_by_image.get_group(image_id).iterrows():
                if ann["category_id"] == EMPTY_CATEGORY_ID:
                    continue
                x, y, w, h = ast.literal_eval(ann["bbox"])
                x_center, y_center = (x + w / 2) / width, (y + h / 2) / height
                norm_w, norm_h = w / width, h / height
                class_id = class_map[ann["category_id"]]
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        (labels_out / f"{file_name.rsplit('.', 1)[0]}.txt").write_text("\n".join(lines))


def main() -> None:
    class_map, class_names = build_class_map()
    annotations_df = pd.read_csv(PROCESSED_DIR / "annotations.csv")

    for split in ["train", "val", "test"]:
        convert_split(split, class_map, annotations_df)

    dataset_yaml = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(YOLO_DIR / "dataset.yaml", "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print(f"{len(class_names)} classes: {class_names}")
    print(f"Wrote dataset config to {YOLO_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
