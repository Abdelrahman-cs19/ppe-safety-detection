"""
Detection post-processing: resolve conflicts between mutually-exclusive
class pairs (helmet/no-helmet, vest/no-vest) without touching any other
classes.

Kept separate from detector.py so it's independently unit-testable with
plain Detection objects - no model or GPU required.
"""

from __future__ import annotations

from config.config import MUTUALLY_EXCLUSIVE_PAIRS
from detection.schemas import Detection


def compute_iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two detections' boxes, in [0, 1]."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def resolve_mutually_exclusive_conflicts(
    detections: list[Detection],
    iou_threshold: float = 0.5,
    pairs: list[tuple[str, str]] | None = None,
) -> list[Detection]:
    """
    Drop the lower-confidence detection whenever two boxes from an
    opposing pair (e.g. "helmet" vs "no-helmet") overlap heavily enough
    to almost certainly be describing the same physical region.

    Every other class (including "person") passes through untouched -
    this is deliberately narrower than blanket class-agnostic NMS.
    """
    pairs = pairs if pairs is not None else MUTUALLY_EXCLUSIVE_PAIRS
    to_drop: set[int] = set()

    for class_a, class_b in pairs:
        group_a = [(i, d) for i, d in enumerate(detections) if d.class_name == class_a]
        group_b = [(i, d) for i, d in enumerate(detections) if d.class_name == class_b]

        for idx_a, det_a in group_a:
            if idx_a in to_drop:
                continue
            for idx_b, det_b in group_b:
                if idx_b in to_drop:
                    continue
                if compute_iou(det_a, det_b) < iou_threshold:
                    continue
                # Same region, opposing classes - keep only the more confident one.
                if det_a.confidence >= det_b.confidence:
                    to_drop.add(idx_b)
                else:
                    to_drop.add(idx_a)
                    break  # det_a is gone; stop comparing it against group_b

    return [d for i, d in enumerate(detections) if i not in to_drop]
