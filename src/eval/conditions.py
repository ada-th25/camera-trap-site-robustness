import cv2
import numpy as np


def classify_day_night(image_path: str, threshold: float = 5.0) -> str:
    """Classify an image as 'day' or 'night' based on color channel difference.

    Camera traps switch to IR illumination at night, producing near-grayscale
    images (R approx G approx B per pixel). Compute the same channel-difference
    proxy we just tested (mean |R-G| + mean |G-B|) and threshold it.

    Returns 'night' if the value is below `threshold`, else 'day'.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    # Convert to float for accurate calculations
    image = image.astype(np.float32)

    # Split the channels
    B, G, R = cv2.split(image)

    # Compute mean absolute differences
    mean_diff_RG = np.mean(np.abs(R - G))
    mean_diff_GB = np.mean(np.abs(G - B))

    # Calculate the combined difference metric
    combined_diff = mean_diff_RG + mean_diff_GB

    # Classify based on the threshold
    if combined_diff < threshold:
        return 'night'
    else:
        return 'day'


def bbox_size_category(bbox: list[float], image_width: int, image_height: int) -> str:
    """Bucket a [x, y, w, h] pixel bbox by its area as a fraction of the image.

    Camera-trap images vary a lot in resolution, so we use area *relative to
    the image* rather than COCO's fixed absolute-pixel thresholds (COCO's
    32x32/96x96 cutoffs assume ~640x480 photos; ours are 2048x1494).
    """
    _, _, w, h = bbox
    area_fraction = (w * h) / (image_width * image_height)

    if area_fraction < 0.01:
        return "small"
    elif area_fraction < 0.05:
        return "medium"
    else:
        return "large"