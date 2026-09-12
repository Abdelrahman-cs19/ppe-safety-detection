"""
Shared detection data structure.

Kept separate from `detector.py` so tracking/association/rules can import
just the type without pulling in Ultralytics — useful once we write unit
tests for those modules using fake detections.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    """A single YOLO detection for one frame, in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2) in pixel coordinates."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        """(cx, cy) center point in pixel coordinates."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)
