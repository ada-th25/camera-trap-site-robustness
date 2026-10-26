# Step 3 results: site-robustness breakdown

All numbers below are on the test set (location 125, unseen) using the fine-tuned
model from Step 2 (`models/cct20_yolov8n_best.pt`), at confidence threshold 0.25,
IoU 0.5 matching. Day/night classified from pixel color-channel difference
(`src/eval/conditions.py::classify_day_night`) - camera traps switch to IR
illumination at night, producing near-grayscale images. Size bucket is bbox area
as a fraction of image area (`bbox_size_category`), not COCO's fixed pixel
thresholds, since our images (2048x1494) are much larger than COCO's typical
~640x480.

## Recall by day vs. night/IR

| condition | recall | n |
|---|---|---|
| day | 0.127 | 110 |
| night | 0.399 | 193 |

![Recall by day/night](figures/recall_by_daynight.png)

**Counter to the naive expectation**, recall is substantially *higher* at night.
This holds even within each size bucket (see cross-tab below), so it isn't just
that night captures happen to have larger/closer animals.

| | small | medium | large |
|---|---|---|---|
| day | 0.077 | 0.143 | 0.182 |
| night | 0.122 | 0.458 | 0.667 |

**Working explanation (not yet independently verified beyond visual spot-checks):**
IR night images are high-contrast - a bright animal against a uniformly dark
background - while daytime images at these camera sites have cluttered, colorful
backgrounds (vegetation, dappled light, shadows) that a model fine-tuned on only
2 locations may not have learned to see past.

## Recall by animal size

| size bucket | recall | n |
|---|---|---|
| small (<1% of frame) | 0.102 | 88 |
| medium (1-5%) | 0.367 | 169 |
| large (>5%) | 0.435 | 46 |

![Recall by size](figures/recall_by_size.png)

Clear, expected pattern: smaller animals are much harder to detect. This is the
single strongest effect found in this breakdown.

## False positives on empty (no-animal) frames

| condition | mean FP/image | total FP | n images |
|---|---|---|---|
| day | 0.067 | 1 | 15 |
| night | 0.000 | 0 | 3 |

Sample size is too small (only 18 genuinely empty frames in this test set) to
draw a real conclusion here - noted as a limitation, not a finding.

## Manual error inspection

### `cat`: 0 precision/recall/mAP on both val and test (from Step 2)

Both `cat`-labeled frames at location 125 are the *same camera scene* (a bridge/
culvert), with the cat appearing as a small, dark silhouette against dark ground
in the same lower-left region of frame both times.

![Cat camouflage example](figures/cat_camouflage_example.jpg)

This reframes the Step 2 finding: it isn't "the model is bad at cats" in
general - it's that the only cat footage at this specific unseen camera happens
to be small, low-contrast, and camouflaged, consistent with the size-bucket
result above (small boxes: 0.10 recall). A different unseen camera might show
completely different cat performance.

### `coyote`: recall held up (0.58->0.46) but precision collapsed (0.41->0.08)

Initial manual check: sampled 9 of the model's 39 "coyote" predictions on the
test set against ground truth, and found 6 were actually **`bobcat`**.

![Bobcat predicted as coyote](figures/bobcat_coyote_confusion.jpg)

Localization is near-perfect in this example (predicted and ground-truth boxes
almost exactly overlap) - the failure is purely the species label, not
detection.

**Follow-up with the full confusion matrix** (pulled via `model.val()`,
`runs/detect/.../confusion_matrix.png`, copied to
`figures/confusion_matrix.png`) gave a fuller and partly different picture.
Cross-checking predicted-class counts against the known true-instance totals
from Step 2 as a sanity check:

- **`cat`** (6 true instances, fully reconciled): 0 correctly predicted - all 6
  were called opossum (1), squirrel (1), **rabbit (3)**, or missed (1). The
  manual 2-image camouflage inspection above is still a reasonable read of
  *why* these get missed, but the model's actual wrong guess is usually
  "rabbit," not something I'd have picked out by eye alone.
- **`dog`** (21 true instances, ~20/21 reconciled): mostly confused as
  **coyote (10)** and **skunk (6)**.
- **`bobcat`** (135 true instances, ~124/135 reconciled - treat as
  approximate): the *largest* confusion is actually **rabbit (34)**, with
  **coyote (24)** second. The manual sample above found a real pattern
  (bobcat->coyote confusion exists), but missed that bobcat->rabbit is even
  more common - a direct demonstration of why spot-checking 9 of 39
  predictions can find a real effect while still misjudging which effect is
  dominant.

**What I'm not reporting from this matrix:** the "background" row/column
values (in the hundreds, e.g. 590, 619) are far larger than the entire test
set (302 images) and are a known artifact of how Ultralytics computes that
matrix internally (pre-NMS candidate boxes, not final per-image counts) -
I can't verify what they actually represent, so I'm not stating them as
findings. The per-class precision/recall table from Step 2 remains the
reliable source for false-positive/miss rates.

Combined explanation: training locations (51, 108) had only 5 bobcat instances
combined, while location 125 is bobcat-dominant (135 instances) - the model
never learned strong bobcat-specific features and falls back to whichever
similarly-shaped class it does know, which the full matrix shows is split
between rabbit and coyote rather than being one clean confusion.

## Key takeaways

1. The **seen vs. unseen location gap from Step 2 is not uniform** - it concentrates
   in small/distant animals and in species underrepresented in training, not
   evenly across all conditions.
2. The day/night result is counterintuitive and worth highlighting in the
   write-up precisely because it contradicts the naive assumption - but the
   causal explanation (background clutter vs. contrast) is a working hypothesis
   from visual inspection, not independently proven.
3. Two of the weakest per-class results from Step 2 (`cat`, `coyote`) turned out
   to have specific, identifiable causes (camouflage + tiny sample; class
   confusion with an underrepresented species) rather than being generically
   "hard" - this came only from looking at actual images, not from the metrics
   alone.
4. Manual spot-checking (9 of 39 coyote predictions) found a real confusion
   pattern but the wrong relative ranking - the full confusion matrix showed
   bobcat->rabbit confusion is larger than bobcat->coyote. Spot-checks are a
   good way to generate hypotheses quickly, but the full matrix is needed to
   know which confusion is actually dominant.

## Limitations

- Small sample sizes in several breakdowns (3 night-empty frames, 6 cat
  instances, 1 bird instance) - several numbers here are illustrative, not
  statistically robust.
- Occlusion was not analyzed quantitatively - CCT provides no occlusion label,
  and no reliable automatic proxy was available; would require manual labeling
  of a sample, out of scope for this pass.
- Day/night causal explanation is a hypothesis from a handful of visual
  spot-checks, not a controlled comparison.
