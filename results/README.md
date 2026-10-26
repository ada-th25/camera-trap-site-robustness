# Results index

Results are numbered by the project step that produced them (see the project
plan in the main README once Step 6 is written; steps are: 0 setup, 1 data
pipeline, 2 baseline + fine-tuning, 3 site-robustness breakdown, 4 engineering,
5 efficiency, 6 write-up). Steps 0, 1, and 4 don't have a dedicated results
file - their key facts live in `data/README.md` (0, 1) or are directly
verifiable by running the tests/CI/Docker build (4), fully reproducible with
nothing experimental that would be lost.

- [`step2_results.md`](step2_results.md) - zero-shot COCO baseline vs.
  fine-tuned YOLOv8n, seen vs. unseen location comparison, per-class metrics.
- [`step3_results.md`](step3_results.md) - site-robustness breakdown (day/night,
  animal size, false positives on empty frames) and manual error inspection
  (`cat`, `coyote`/`bobcat` confusion, full confusion matrix).
- [`step5_results.md`](step5_results.md) - ONNX export, CPU (Mac) vs. GPU
  (Colab T4) latency benchmarking, and the accuracy/speed/size tradeoff.

`figures/` holds the charts and annotated example images referenced from both
files.
