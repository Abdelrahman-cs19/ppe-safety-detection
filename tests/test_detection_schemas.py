"""Unit tests for detection.schemas.Detection — pure math, no model needed."""

from detection.schemas import Detection


def _make_detection(x1=10.0, y1=10.0, x2=50.0, y2=90.0) -> Detection:
    return Detection(
        class_id=0,
        class_name="helmet",
        confidence=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def test_bbox_matches_constructor_values() -> None:
    det = _make_detection()
    assert det.bbox == (10.0, 10.0, 50.0, 90.0)


def test_center_is_midpoint() -> None:
    det = _make_detection(x1=0.0, y1=0.0, x2=100.0, y2=50.0)
    assert det.center == (50.0, 25.0)


def test_width_height_and_area() -> None:
    det = _make_detection(x1=0.0, y1=0.0, x2=40.0, y2=20.0)
    assert det.width == 40.0
    assert det.height == 20.0
    assert det.area == 800.0


def test_area_is_never_negative_for_degenerate_box() -> None:
    # A malformed box (x2 < x1) should not produce a negative area.
    det = _make_detection(x1=50.0, y1=50.0, x2=10.0, y2=10.0)
    assert det.area == 0.0
