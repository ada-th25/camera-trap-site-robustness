import cv2
import numpy as np

from src.eval.conditions import bbox_size_category, classify_day_night


def test_classify_day_night_detects_grayscale(tmp_path):
    gray_image = np.full((50, 50, 3), 128, dtype=np.uint8)  # R=G=B everywhere
    path = tmp_path / "gray.jpg"
    cv2.imwrite(str(path), gray_image)

    assert classify_day_night(str(path)) == "night"

def test_classify_day_night_detects_colorful(tmp_path):
    rng = np.random.default_rng(0)
    colorful_image = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    path = tmp_path / "colorful.jpg"
    cv2.imwrite(str(path), colorful_image)

    assert classify_day_night(str(path)) == "day"

def test_bbox_size_category_boundary():
    bbox = [0, 0, 100, 100]  # Area = 10000, which is exactly 1% of 1000x1000
    image_width = 1000
    image_height = 1000
    assert bbox_size_category(bbox, image_width, image_height) == "medium"

    bbox = [0, 0, 223, 223]  # Area = 49729, which is just under 5% of 1000x1000
    image_width = 1000
    image_height = 1000
    assert bbox_size_category(bbox, image_width, image_height) == "medium"

    bbox = [0, 0, 224, 224]  # Area = 50176, which is about 5% of 1000x1000
    assert bbox_size_category(bbox, image_width, image_height) == "large"
