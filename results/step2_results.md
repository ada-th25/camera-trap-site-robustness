# Step 2 results: baseline vs. fine-tuned YOLOv8n

## Setup

- Model: YOLOv8n (Ultralytics), fine-tuned on Colab (Tesla T4)
- Train: 602 images, locations 51 + 108
- Val: 300 images, location 90 (seen, different camera from train)
- Test: 302 images, location 125 (unseen - never used in training or model selection)
- Fine-tune run: 50 max epochs, early-stopped at epoch 46 (best weights from epoch 26, `patience=20`)
- Weights: `models/cct20_yolov8n_best.pt` (not committed to git; rerun `notebooks/colab_finetune.ipynb` to reproduce)

## Zero-shot baseline (COCO-pretrained, no fine-tuning)

Class-agnostic matching (ignores predicted class label - COCO has none of our species),
evaluated on test (location 125):

| metric | value |
|---|---|
| precision | 0.807 |
| recall | 0.234 |
| TP / FP / FN | 71 / 17 / 232 |

## Fine-tuned model: seen vs. unseen location

Standard per-class Ultralytics validation (IoU 0.5 for mAP50, averaged 0.5-0.95 for mAP50-95):

| metric | val (seen, loc 90) | test (unseen, loc 125) |
|---|---|---|
| precision | 0.480 | 0.188 |
| recall | 0.244 | 0.187 |
| mAP50 | 0.259 | 0.122 |
| mAP50-95 | 0.152 | 0.065 |

## Per-class, test set (location 125, unseen)

| class | instances | precision | recall | mAP50 |
|---|---|---|---|---|
| opossum | 77 | 0.447 | 0.377 | 0.355 |
| raccoon | 35 | 0.209 | 0.314 | 0.202 |
| bobcat | 135 | 0.445 | 0.230 | 0.230 |
| skunk | 8 | 0.356 | 0.250 | 0.206 |
| dog | 21 | 0.151 | 0.048 | 0.042 |
| coyote | 13 | 0.085 | 0.462 | 0.065 |
| rabbit | 7 | 0 | 0 | 0 |
| bird | 1 | 0 | 0 | 0 |
| cat | 6 | 0 | 0 | 0 |

## Key findings

1. **mAP50 roughly halves moving from a seen to an unseen camera location** (0.259 -> 0.122) -
   the core cross-site robustness gap this project measures.
2. **Fine-tuning did not clearly improve raw recall over the zero-shot baseline on the unseen
   site** (baseline class-agnostic recall 0.234 vs. fine-tuned per-class recall 0.187). What
   fine-tuning bought us is species-level identification, which the baseline had no capacity
   for at all - not better raw localization on new sites.
3. **`cat` scored 0 precision/recall/mAP on both val and test**, despite having real instances
   in each - a consistent failure, not a fluke of one location. Needs investigation in Step 3
   (confusion matrix check, manual inspection of `cat`-labelled frames).
4. **`coyote` recall held up reasonably across locations (0.58 -> 0.46) but precision collapsed
   (0.41 -> 0.08)** - likely confusion with a species common at location 125 (bobcat has 135
   test-set instances vs. 5 in training). To investigate in Step 3.
5. Species composition differs genuinely by location (location 125 is bobcat-dominant;
   training locations were squirrel/cat-dominant) - an ecological fact, not a bug.

## Known limitations of this result

- Very small dataset (602 train images, 2 locations) - the seen/unseen gap could shrink with
  more training location diversity; we can't yet tell whether the gap reflects a fundamental
  generalization limit or just insufficient training diversity.
- `rabbit`, `bird`, `cat` have very few test instances (6-7) - their 0 scores are high-variance
  and not statistically robust on their own.
