# Step 5 results: ONNX export and CPU/GPU benchmarking

Model: fine-tuned YOLOv8n (`models/cct20_yolov8n_best.pt`). ONNX export via
Ultralytics (`model.export(format='onnx', imgsz=640)`), opset 18. All latency
numbers are single-image (`batch=1`), matching how `scripts/infer.py` is
actually used - one image at a time, not batched throughput. Accuracy is
mAP on the test set (location 125, unseen), same evaluation as Steps 2-3.

## Results

| format | hardware | inference latency | mAP50 | mAP50-95 | file size |
|---|---|---|---|---|---|
| PyTorch | CPU (Apple M4) | 22.7 ms | 0.122 | 0.065 | 6.2 MB |
| ONNX | CPU (Apple M4) | 51.5 ms | 0.118 | 0.064 | 12.2 MB |
| PyTorch | GPU (Colab T4) | 12.6 ms | 0.122 | 0.065 | 6.5 MB |
| ONNX | GPU (Colab T4) | 8.8 ms | 0.118 | 0.064 | 12.8 MB |

GPU ONNX run confirmed using `CUDAExecutionProvider` (not silently falling
back to CPU) - this required pre-installing `onnxruntime-gpu` before export,
since Ultralytics' export step otherwise auto-installs the CPU-only
`onnxruntime`, and the two packages silently conflict (whichever gets
imported first wins).

## Key findings

1. **The ONNX speed advantage is hardware-dependent, not universal.** On CPU,
   PyTorch is ~2.3x faster than ONNX (22.7ms vs 51.5ms). On GPU, ONNX is
   ~1.4x faster than PyTorch (8.8ms vs 12.6ms). "Export to ONNX for speed" is
   a common assumption that doesn't hold here on CPU - this was only
   discoverable by actually measuring both, not something to assume from
   general advice.
2. **ONNX export preserves accuracy well but not exactly**: mAP50 drops
   ~3% relative (0.122 -> 0.118), mAP50-95 similarly (0.065 -> 0.064),
   consistent across both CPU and GPU runs. Spot-checked on one image: the
   two formats agreed on the main detection (coyote, confidence 0.853 PT vs.
   0.843 ONNX) but a second, borderline detection (bobcat, 0.257 confidence
   in PyTorch, just above our 0.25 threshold) disappeared in ONNX - small
   floating-point differences between the two runtimes occasionally tip a
   borderline detection across a confidence threshold. This is expected
   behavior from format conversion, not a bug, but worth knowing about
   before trusting ONNX and PyTorch to give identical outputs.
3. **ONNX file is ~2x larger** (12.2-12.8MB vs 6.2-6.5MB), likely due to
   protobuf's explicit graph serialization (named tensors, node definitions)
   versus PyTorch's more compact state-dict checkpoint - a plausible
   explanation, not independently verified at the byte level.
4. **For this project's actual deployment target** (the CPU-only Docker
   container from Step 4, in the spirit of resource-constrained edge
   deployment), PyTorch is the better choice over ONNX for this specific
   model and hardware combination - the opposite of the usual "always
   export to ONNX for deployment" advice. `scripts/infer.py` / the
   Dockerfile currently use PyTorch directly, which this result supports
   rather than contradicts.
5. Both formats are comfortably fast enough in absolute terms for this
   project's realistic use case (offline/batch camera-trap analysis, not
   live video) - even the slower CPU/ONNX combination (51.5ms/image, ~19
   images/sec) is far from a bottleneck here.

## Limitations

- Benchmarked only the ONNX Runtime default CPU execution provider; did not
  try the CoreML execution provider, which could plausibly close or reverse
  the CPU gap on Apple Silicon specifically - noted as future work, not done
  here to keep scope limited to one model, two formats, two hardware
  targets (per the Step 5 scope decision).
- Did not benchmark a second model size (e.g. YOLOv8s) for the
  accuracy-vs-speed-vs-size tradeoff curve - deliberately out of scope (see
  Step 5 planning discussion) to avoid a second full fine-tuning run purely
  for comparison.
- Single-run timings, not averaged over multiple repeated runs - some
  measurement noise is expected, particularly for the CPU numbers.
