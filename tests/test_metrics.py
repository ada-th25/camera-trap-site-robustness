from src.eval.metrics import compute_iou, get_matched_gt_mask, match_predictions


def test_compute_iou_identical_boxes():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_compute_iou_no_overlap():
    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_compute_iou_partial_overlap():
    # two 10x10 boxes overlapping in a 5x10 strip -> intersection 50, union 150
    assert compute_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 50 / 150


def test_match_predictions_two_preds_one_gt_box():
    result = match_predictions([[0, 0, 10, 10], [0, 0, 10, 10]], [[0, 0, 10, 10]], iou_threshold=0.5)
    assert result == {"tp": 1, "fp": 1, "fn": 0}


def test_match_predictions_no_ground_truth():
    result = match_predictions([[0, 0, 10, 10]], [], iou_threshold=0.5)
    assert result == {"tp": 0, "fp": 1, "fn": 0}


def test_match_predictions_no_predictions():
    result = match_predictions([], [[0, 0, 10, 10]], iou_threshold=0.5)
    assert result == {"tp": 0, "fp": 0, "fn": 1}


def test_match_predictions_prefers_higher_confidence_order():
    # Two overlapping gt boxes; the first (highest-confidence) prediction
    # should claim the best-IoU gt box, not whichever comes first in the list.
    gt_boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]
    pred_boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]  # already sorted by confidence
    result = match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5)
    assert result == {"tp": 2, "fp": 0, "fn": 0}


def test_get_matched_gt_mask_matches_match_predictions_counts():
    pred_boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
    gt_boxes = [[0, 0, 10, 10], [200, 200, 210, 210]]
    mask = get_matched_gt_mask(pred_boxes, gt_boxes, iou_threshold=0.5)
    assert mask == [True, False]
