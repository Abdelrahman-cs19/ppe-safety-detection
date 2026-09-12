"""
Milestone 1 — train (or fine-tune) a YOLO model on the PPE dataset.

Usage
-----
    python -m scripts.train --epochs 100 --batch 16 --model yolov8s.pt

Run from the project root (ppe-safety/) so the `config` and `utils`
packages resolve correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPE detection YOLO model.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base checkpoint to fine-tune from (yolov8n/s/m.pt, or a .yaml to train from scratch).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "data.yaml",
        help="Path to the dataset YAML.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="'0' for first GPU, 'cpu' for CPU, '0,1' for multi-GPU.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (epochs with no val improvement).",
    )
    parser.add_argument("--run-name", type=str, default="ppe_yolo_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        logger.error(
            "Dataset config not found at %s. See README.md Milestone 1 for how "
            "to download and lay out the dataset before training.",
            args.data,
        )
        raise SystemExit(1)

    logger.info("Loading base model: %s", args.model)
    model = YOLO(args.model)

    logger.info(
        "Starting training | data=%s epochs=%d imgsz=%d batch=%d device=%s",
        args.data,
        args.epochs,
        args.imgsz,
        args.batch,
        args.device,
    )

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=args.run_name,
        # Light, safe-default augmentation for a PPE/construction-site domain.
        # hsv_h bumped up slightly (0.01 -> 0.03) so the model sees more hue
        # variation per training image - vests come in several real-world
        # colors (orange/yellow/lime/green) and if the dataset skews toward
        # one, this helps generalize to the others without needing more
        # data. Kept modest since too much hue jitter can also make helmet
        # color less distinguishable - revisit if evaluate.py shows helmet
        # AP50 regressing after this change.
        hsv_h=0.03,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_weights)

    if best_weights.exists():
        target = MODELS_DIR / "ppe_yolo.pt"
        target.write_bytes(best_weights.read_bytes())
        logger.info("Copied best weights to %s (used by the app config).", target)
    else:
        logger.warning("Could not find best.pt at expected path: %s", best_weights)


if __name__ == "__main__":
    main()
