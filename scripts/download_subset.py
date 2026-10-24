"""Download a small, sequence-sampled subset of CCT20 images for local dev.

Usage: .venv/bin/python scripts/download_subset.py
"""
from pathlib import Path

import requests
from tqdm import tqdm

from src.data.cct import load_pooled, sample_sequences_per_location

IMAGE_BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"
OUT_DIR = Path("data/raw/images")
LOCATIONS = {51, 108, 90, 125}
TARGET_PER_LOCATION = 300


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images_df, _, _ = load_pooled(LOCATIONS)
    sampled = sample_sequences_per_location(images_df, TARGET_PER_LOCATION, seed=0)

    for file_name in tqdm(sampled["file_name"], desc="downloading"):
        dest = OUT_DIR / file_name
        if dest.exists():
            continue

        for attempt in range(3):
            try:
                resp = requests.get(f"{IMAGE_BASE_URL}/{file_name}", timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise


if __name__ == "__main__":
    main()
