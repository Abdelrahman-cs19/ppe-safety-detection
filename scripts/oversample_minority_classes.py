"""
Milestone 2 utility — oversample training images containing rare
classes (no-helmet, no-vest by default) by duplicating the image+label
pair on disk, so the model sees them more often per epoch.

This does NOT add new information (it's not a substitute for real new
data), but it's a legitimate, quick mitigation for class imbalance
while you're sourcing more real no-helmet examples.

Safe to re-run: duplicates are named with a "_dupN" suffix and are
skipped if they already exist, so running this twice won't endlessly
re-duplicate.

Usage
-----
    # Preview what would happen, without touching any files:
    python -m scripts.oversample_minority_classes --dry-run

    # Actually duplicate no-helmet/no-vest images ~3x in the train split:
    python -m scripts.oversample_minority_classes --classes no-helmet no-vest --multiplier 3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from config.config import CLASS_NAMES, CLASS_TO_ID, DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oversample rare-class training images.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["no-helmet"],
        help="Class name(s) to oversample (must match config.CLASS_NAMES).",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Each matching image will exist this many times total after oversampling.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to oversample (only ever do this to train, never val/test).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying files.")
    return parser.parse_args()


def _resolve_split_dirs(split: str) -> tuple[Path, Path]:
    data_yaml_path = DATA_DIR / "data.yaml"
    with open(data_yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    if not base_path.is_absolute():
        base_path = data_yaml_path.parent / base_path

    images_rel = data_cfg[split]
    labels_rel = images_rel.replace("images", "labels")
    return base_path / images_rel, base_path / labels_rel


def _image_contains_any_class(label_path: Path, target_ids: set[int]) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if int(line.split()[0]) in target_ids:
            return True
    return False


def main() -> None:
    args = parse_args()

    unknown = [c for c in args.classes if c not in CLASS_TO_ID]
    if unknown:
        logger.error("Unknown class name(s): %s. Valid classes: %s", unknown, CLASS_NAMES)
        raise SystemExit(1)

    target_ids = {CLASS_TO_ID[name] for name in args.classes}
    images_dir, labels_dir = _resolve_split_dirs(args.split)

    if not images_dir.exists() or not labels_dir.exists():
        logger.error("Split directories not found: %s / %s", images_dir, labels_dir)
        raise SystemExit(1)

    # Only consider "original" files - skip anything already produced by a
    # previous run of this script, so re-running doesn't compound growth.
    candidate_images = [
        p for p in images_dir.iterdir() if p.is_file() and "_dup" not in p.stem
    ]

    matches = [
        img for img in candidate_images
        if _image_contains_any_class(labels_dir / f"{img.stem}.txt", target_ids)
    ]

    logger.info(
        "Found %d/%d original images in '%s' containing class(es) %s",
        len(matches),
        len(candidate_images),
        args.split,
        args.classes,
    )

    if not matches:
        logger.warning("No matching images found - nothing to oversample.")
        return

    copies_per_image = args.multiplier - 1  # -1 because the original already counts as copy #1
    if copies_per_image <= 0:
        logger.info("Multiplier <= 1, nothing to do.")
        return

    created = 0
    for img_path in matches:
        label_path = labels_dir / f"{img_path.stem}.txt"
        for dup_index in range(1, copies_per_image + 1):
            new_img = images_dir / f"{img_path.stem}_dup{dup_index}{img_path.suffix}"
            new_label = labels_dir / f"{img_path.stem}_dup{dup_index}.txt"

            if new_img.exists() and new_label.exists():
                continue  # already created by a previous run

            if args.dry_run:
                logger.info("[dry-run] would create %s", new_img.name)
            else:
                shutil.copy2(img_path, new_img)
                shutil.copy2(label_path, new_label)
            created += 1

    action = "Would create" if args.dry_run else "Created"
    logger.info("%s %d duplicate image+label pairs.", action, created)
    if args.dry_run:
        logger.info("Re-run without --dry-run to actually create the files.")
    else:
        logger.info("Run `python -m scripts.dataset_stats` again to confirm the new balance.")


if __name__ == "__main__":
    main()
