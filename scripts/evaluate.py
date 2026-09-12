"""
Milestone 1 — evaluate a trained YOLO checkpoint on the validation/test split.

Produces the metrics required for the portfolio write-up: precision,
recall, mAP50, mAP50-95, and a confusion matrix image. Ultralytics
computes all of these natively during `model.val()`; this script just
runs it and prints/saves a clean summary.

Usage
-----
    python -m scripts.evaluate --weights models/ppe_yolo.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PPE detection model.")
    parser.add_argument("--weights", type=Path, default=MODELS_DIR / "ppe_yolo.pt")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--run-name", type=str, default="ppe_yolo_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        logger.error("Weights not found at %s. Run scripts/train.py first.", args.weights)
        raise SystemExit(1)

    model = YOLO(str(args.weights))

    logger.info("Running validation on '%s' split...", args.split)
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        project=str(PROJECT_ROOT / "runs" / "val"),
        name=args.run_name,
        plots=True,  # writes confusion_matrix.png, PR curves, etc. to the run dir
    )

    # `metrics.box` holds the detection metrics (mean across all classes).
    logger.info("=== Evaluation summary ===")
    logger.info("Precision (mean):   %.4f", metrics.box.mp)
    logger.info("Recall (mean):      %.4f", metrics.box.mr)
    logger.info("mAP50:              %.4f", metrics.box.map50)
    logger.info("mAP50-95:           %.4f", metrics.box.map)

    logger.info("Per-class AP50:")
    for class_id, class_name in model.names.items():
        try:
            ap50 = metrics.box.ap50[class_id]
            logger.info("  %-16s AP50=%.4f", class_name, ap50)
        except (IndexError, KeyError):
            continue

    logger.info(
        "Confusion matrix and PR curves saved under: %s",
        Path(metrics.save_dir),
    )


if __name__ == "__main__":
    main()
