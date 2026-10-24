# Data

This project uses the **CCT20 benchmark subset** of Caltech Camera Traps (CCT),
hosted by [LILA BC](https://lila.science/datasets/caltech-camera-traps/).

- Licence: Community Data License Agreement (permissive variant, CDLA-Permissive-1.0)
- Bounding box annotations: yes, included per-image in the split JSON files below
- Format: COCO-style (`images`, `annotations`, `categories`), bbox as `[x, y, width, height]` in pixels

## Splits (as provided by the dataset authors)

| split              | images | locations                          |
|--------------------|-------:|-------------------------------------|
| `train`            | 13,553 | 10 (33, 38, 43, 51, 61, 88, 90, ...) |
| `cis_val`          |  3,484 | same 10 as train ("seen" locations) |
| `cis_test`         | 15,827 | same 10 as train ("seen" locations) |
| `trans_val`        |  1,725 | 1 (125) — not in train               |
| `trans_test`       | 23,275 | 9 (0, 7, 28, 40, 46, 78, 100, 105) — not in train |

`cis` = same locations as training (in-distribution). `trans` = different, unseen
locations (out-of-distribution / transfer) — this is the dataset authors' own
cross-site split, which matches our research question directly.

**Known limitation:** none of the images in this benchmark subset are labelled
`empty` (no animal). All annotated images contain at least one real box. If we
want false-positive-on-empty-frame analysis (Step 3), we'll need to source
empty frames separately from the full (105GB) Caltech Camera Traps release.

## Getting the data

Annotations only (~3MB, already fetched during setup):
```
data/raw/annotations/eccv_18_annotation_files/*.json
```

Full benchmark images (~6GB) — do **not** run this yet. Step 1 will decide a
small subset of locations to work with first:
```
scripts/download_cct20.sh
```

Raw/processed data is gitignored and never committed; only this README and
the download script are checked in.
