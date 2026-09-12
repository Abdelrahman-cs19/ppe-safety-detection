"""
Milestone 2 — real-time detection demo.

Opens a webcam or video file, runs the trained PPEDetector on every
frame, draws bounding boxes color-coded by class, and overlays live FPS
and per-stage inference latency. Press 'q' to quit.

Usage
-----
    python -m scripts.webcam_demo                      # default webcam
    python -m scripts.webcam_demo --source 1            # a different camera index
    python -m scripts.webcam_demo --source path/to.mp4  # a video file
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np

from detection.detector import ModelLoadError, PPEDetector
from detection.schemas import Detection
from utils.logger import get_logger

logger = get_logger(__name__)

# BGR colors per class name — violation classes in red, PPE-present/person in green/blue.
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (255, 180, 0),
    "helmet": (0, 200, 0),
    "vest": (0, 200, 0),
    "no-helmet": (0, 0, 255),
    "no-vest": (0, 0, 255),
}
_DEFAULT_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time PPE detection demo.")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g. 0, 1) or a path to a video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to a .pt file to use instead of the configured model "
        "(e.g. models/reference_ppe.pt for benchmarking another model).",
    )
    return parser.parse_args()


def _resolve_source(source: str) -> int | str:
    """Webcam indices arrive as plain digit strings; anything else is a file path."""
    return int(source) if source.isdigit() else source


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    return frame


def draw_overlay(frame: np.ndarray, fps: float, inference_ms: float) -> np.ndarray:
    text = f"FPS: {fps:.1f}  |  Inference: {inference_ms:.1f} ms"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main() -> None:
    args = parse_args()
    source = _resolve_source(args.source)

    try:
        detector = PPEDetector(weights_path_override=args.weights)
    except ModelLoadError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Could not open video source: %s", source)
        raise SystemExit(1)

    logger.info("Source opened: %s. Press 'q' in the video window to quit.", source)

    # Rolling window of frame times for a smoothed FPS readout, since a
    # single frame's timing is noisy.
    frame_times: deque[float] = deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("Stream ended or frame could not be read.")
                break

            loop_start = time.perf_counter()
            detections, stats = detector.predict(frame)
            frame = draw_detections(frame, detections)

            frame_times.append(time.perf_counter() - loop_start)
            smoothed_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0.0
            frame = draw_overlay(frame, smoothed_fps, stats.inference_ms)

            cv2.imshow("PPE Safety Detection - Milestone 2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
