"""
Milestone 2 utility — run your trained model and a reference/pretrained
model on the exact same video source at the same time, side by side, so
you can visually compare them frame-for-frame.

This is a visual sanity check, not a formal benchmark: the two models
may have different class sets/orders, so this does NOT try to compute
comparative precision/recall - just draws both outputs next to each
other so you can eyeball the difference.

Usage
-----
    python -m scripts.compare_models --source 0
    python -m scripts.compare_models --source clip.mp4 --reference models/reference_ppe.pt
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np

from config.config import MODELS_DIR
from detection.detector import ModelLoadError, PPEDetector
from scripts.webcam_demo import draw_detections, draw_overlay
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two PPE models side by side.")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video path.")
    parser.add_argument(
        "--ours",
        type=str,
        default=None,
        help="Path to your model (defaults to the configured models/ppe_yolo.pt).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=str(MODELS_DIR / "reference_ppe.pt"),
        help="Path to the reference model (default: the downloaded reference_ppe.pt).",
    )
    return parser.parse_args()


def _resolve_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(frame, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main() -> None:
    args = parse_args()
    source = _resolve_source(args.source)

    try:
        ours = PPEDetector(weights_path_override=args.ours)
        reference = PPEDetector(weights_path_override=args.reference)
    except ModelLoadError as exc:
        logger.error(str(exc))
        logger.error(
            "If the reference model is missing, run: python -m scripts.download_reference_model"
        )
        raise SystemExit(1) from exc

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Could not open video source: %s", source)
        raise SystemExit(1)

    logger.info("Comparing models side by side. Press 'q' to quit.")
    frame_times: deque[float] = deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("Stream ended or frame could not be read.")
                break

            loop_start = time.perf_counter()

            # Run both models on identical copies of the same frame.
            ours_frame = frame.copy()
            ref_frame = frame.copy()

            ours_dets, ours_stats = ours.predict(ours_frame)
            ref_dets, ref_stats = reference.predict(ref_frame)

            ours_frame = draw_detections(ours_frame, ours_dets)
            ref_frame = draw_detections(ref_frame, ref_dets)

            ours_frame = draw_overlay(ours_frame, ours_stats.fps, ours_stats.inference_ms)
            ref_frame = draw_overlay(ref_frame, ref_stats.fps, ref_stats.inference_ms)

            _label_frame(ours_frame, "OURS (yolov8s, custom-trained)")
            _label_frame(ref_frame, "REFERENCE (harsh-77/ppe-detection)")

            combined = np.hstack([ours_frame, ref_frame])

            frame_times.append(time.perf_counter() - loop_start)

            cv2.imshow("PPE Model Comparison - ours (left) vs reference (right)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
