"""Tests for detection.postprocess — pure logic, no model needed."""

from detection.postprocess import compute_iou, resolve_mutually_exclusive_conflicts
from detection.schemas import Detection


def _det(class_name: str, confidence: float, box: tuple[float, float, float, float]) -> Detection:
    x1, y1, x2, y2 = box
    return Detection(class_id=0, class_name=class_name, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)


def test_iou_of_identical_boxes_is_one() -> None:
    a = _det("helmet", 0.9, (0, 0, 10, 10))
    b = _det("no-helmet", 0.5, (0, 0, 10, 10))
    assert compute_iou(a, b) == 1.0


def test_iou_of_non_overlapping_boxes_is_zero() -> None:
    a = _det("helmet", 0.9, (0, 0, 10, 10))
    b = _det("no-helmet", 0.5, (100, 100, 110, 110))
    assert compute_iou(a, b) == 0.0


def test_drops_lower_confidence_overlapping_pair() -> None:
    detections = [
        _det("helmet", 0.9, (0, 0, 10, 10)),
        _det("no-helmet", 0.4, (1, 1, 11, 11)),  # heavily overlapping, lower confidence
    ]
    result = resolve_mutually_exclusive_conflicts(detections, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].class_name == "helmet"


def test_unrelated_classes_are_never_touched() -> None:
    # A "person" box heavily overlapping a "vest" box should NOT be
    # affected - only pairs explicitly listed as mutually exclusive are.
    detections = [
        _det("person", 0.95, (0, 0, 100, 200)),
        _det("vest", 0.8, (10, 50, 90, 150)),
    ]
    result = resolve_mutually_exclusive_conflicts(detections, iou_threshold=0.5)
    assert len(result) == 2


def test_non_overlapping_pair_members_both_survive() -> None:
    # Two different people: one wearing a helmet, one not - far apart,
    # should not be treated as a conflict.
    detections = [
        _det("helmet", 0.9, (0, 0, 10, 10)),
        _det("no-helmet", 0.9, (200, 200, 210, 210)),
    ]
    result = resolve_mutually_exclusive_conflicts(detections, iou_threshold=0.5)
    assert len(result) == 2
