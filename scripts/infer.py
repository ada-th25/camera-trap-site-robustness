"""Run detection on a single image and print results as JSON.

Usage: python scripts/infer.py --weights models/cct20_yolov8n_best.pt --image path/to/image.jpg
"""
import argparse
import json

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a YOLO .pt weights file")
    parser.add_argument("--image", required=True, help="Path to an image to run detection on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.predict(args.image, conf=args.conf, device="cpu", verbose=False)[0]

    detections = [
        {
            "class": model.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]

    print(json.dumps({"image": args.image, "detections": detections}, indent=2))


if __name__ == "__main__":
    main()
