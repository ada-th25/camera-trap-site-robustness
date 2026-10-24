"""Class-agnostic detection matching metrics.

Box format throughout: [x1, y1, x2, y2] in pixel coordinates.
"""


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes.
    """
    # Compute the area of intersection rectangle
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Compute the area of both boxes
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the union area
    union_area = box_a_area + box_b_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0.
    return iou


def match_predictions(
    pred_boxes: list[list[float]], gt_boxes: list[list[float]], iou_threshold: float = 0.5
) -> dict[str, int]:
    """Greedily match predicted boxes to ground-truth boxes for one image.

    `pred_boxes` is assumed already sorted by descending confidence.
    Each predicted box may match at most one ground-truth box, and each
    ground-truth box may be matched at most once (highest-confidence
    prediction wins). A predicted box counts as a true positive (TP) if it
    matches an unmatched ground-truth box with IoU >= iou_threshold;
    otherwise it's a false positive (FP). Any ground-truth box left
    unmatched at the end is a false negative (FN).

    Returns {"tp": int, "fp": int, "fn": int} for this image.
    """
    tp = 0
    fp = 0
    fn = 0

    # Keep track of which ground-truth boxes have been matched
    gt_matched = [False] * len(gt_boxes)

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        # Find the best matching ground-truth box
        for i, gt_box in enumerate(gt_boxes):
            if not gt_matched[i]:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        # If a match is found with sufficient IoU, it's a true positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    # Count any unmatched ground-truth boxes as false negatives
    fn = sum(1 for unmatched in gt_matched if not unmatched)

    return {"tp": tp, "fp": fp, "fn": fn}
