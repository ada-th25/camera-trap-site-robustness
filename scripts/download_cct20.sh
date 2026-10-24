#!/usr/bin/env bash
# Downloads the CCT20 benchmark subset of Caltech Camera Traps from LILA BC.
# Source: https://lila.science/datasets/caltech-camera-traps/
# Licence: Community Data License Agreement (permissive variant)
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
ANN_DIR="$DATA_DIR/annotations"
IMG_DIR="$DATA_DIR/images"

mkdir -p "$ANN_DIR" "$IMG_DIR"

echo "Downloading annotations (~3MB)..."
curl -L -o "$ANN_DIR/eccv_18_annotations.tar.gz" \
  https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz
tar -xzf "$ANN_DIR/eccv_18_annotations.tar.gz" -C "$ANN_DIR"

echo "Downloading benchmark images (~6GB) — this will take a while..."
curl -L -o "$IMG_DIR/eccv_18_all_images_sm.tar.gz" \
  https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz
tar -xzf "$IMG_DIR/eccv_18_all_images_sm.tar.gz" -C "$IMG_DIR"

echo "Done. Annotations in $ANN_DIR/eccv_18_annotation_files, images in $IMG_DIR"
