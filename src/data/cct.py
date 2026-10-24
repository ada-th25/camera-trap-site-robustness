"""Loading and pooling CCT20 COCO-style annotation files."""
from pathlib import Path
import json

import pandas as pd

ANNOTATIONS_DIR = Path("data/raw/annotations/extracted/eccv_18_annotation_files")

SPLIT_FILES = [
    "train_annotations.json",
    "cis_val_annotations.json",
    "cis_test_annotations.json",
    "trans_val_annotations.json",
    "trans_test_annotations.json",
]


def load_pooled(locations: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Pool all CCT20 split files and filter to the given camera locations.

    Returns (images_df, annotations_df, categories). images_df is indexed by
    image id; annotations_df has one row per bounding box, with an
    `image_id` column to join back to images_df.
    """
    all_images: dict[str, dict] = {}
    all_annotations: list[dict] = []
    categories: dict[int, str] = {}

    for split_file in SPLIT_FILES:
        with open(ANNOTATIONS_DIR / split_file) as f:
            d = json.load(f)

        categories.update({c["id"]: c["name"] for c in d["categories"]})

        kept_image_ids = set()
        for im in d["images"]:
            if im["location"] in locations:
                all_images[im["id"]] = im
                kept_image_ids.add(im["id"])

        for ann in d["annotations"]:
            if ann["image_id"] in kept_image_ids:
                all_annotations.append(ann)

    images_df = pd.DataFrame(all_images.values()).set_index("id")
    annotations_df = pd.DataFrame(all_annotations)
    return images_df, annotations_df, categories


def sample_sequences_per_location(
    images_df: pd.DataFrame, target_per_location: int = 300, seed: int = 0
) -> pd.DataFrame:
    """Subsample images_df to ~target_per_location images per location.

    Samples whole (location, seq_id) burst sequences rather than individual
    frames, so we never end up with two near-duplicate frames from the same
    trigger event while dropping the third.
    """
    kept_chunks = []

    for location, group in images_df.groupby("location"):
        seq_ids = group["seq_id"].drop_duplicates().sample(frac=1, random_state=seed)
        running_total = 0
        chosen_seq_ids = []
        seq_sizes = group.groupby("seq_id").size()
        for seq_id in seq_ids:
            if running_total >= target_per_location:
                break
            chosen_seq_ids.append(seq_id)
            running_total += seq_sizes[seq_id]
        kept_chunks.append(group[group["seq_id"].isin(chosen_seq_ids)])

    return pd.concat(kept_chunks)

def split_by_location(
    images_df: pd.DataFrame,
    train_locations: set[int],
    val_locations: set[int],
    test_locations: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split images_df into train/val/test by camera location.

    Raises if train/val/test location sets overlap, or if any image's
    location isn't in one of the three sets.
    """
    if train_locations & val_locations or train_locations & test_locations or val_locations & test_locations:
        raise ValueError("train/val/test location sets must be disjoint")

    all_locations = train_locations | val_locations | test_locations
    if not images_df["location"].isin(all_locations).all():
        raise ValueError("some images have locations outside of train/val/test sets")

    train_df = images_df[images_df["location"].isin(train_locations)]
    val_df = images_df[images_df["location"].isin(val_locations)]
    test_df = images_df[images_df["location"].isin(test_locations)]

    return train_df, val_df, test_df
