"""Build train/val/test manifest CSVs for the small CCT20 subset.

Usage: PYTHONPATH=. .venv/bin/python scripts/build_manifest.py
"""
from pathlib import Path

from src.data.cct import load_pooled, sample_sequences_per_location, split_by_location

LOCATIONS = {51, 108, 90, 125}
TRAIN_LOCATIONS = {51, 108}
VAL_LOCATIONS = {90}
TEST_LOCATIONS = {125}
OUT_DIR = Path("data/processed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images_df, annotations_df, categories = load_pooled(LOCATIONS)
    sampled = sample_sequences_per_location(images_df, target_per_location=300, seed=0)

    train_df, val_df, test_df = split_by_location(
        sampled,
        train_locations=TRAIN_LOCATIONS,
        val_locations=VAL_LOCATIONS,
        test_locations=TEST_LOCATIONS,
    )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(OUT_DIR / f"{name}_manifest.csv")
        print(f"{name}: {len(df)} images, locations={sorted(df['location'].unique())}")

    sampled_ids = set(sampled.index)
    annotations_df[annotations_df["image_id"].isin(sampled_ids)].to_csv(
        OUT_DIR / "annotations.csv", index=False
    )


if __name__ == "__main__":
    main()
