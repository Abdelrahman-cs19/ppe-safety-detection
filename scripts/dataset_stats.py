"""
Milestone 1/2 utility — count label instances per class across the
dataset splits, to check for class imbalance (a common cause of one
class underperforming others).

Usage
-----
    python -m scripts.dataset_stats
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from config.config import CLASS_NAMES, DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def count_labels_in_dir(labels_dir: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not labels_dir.exists():
        logger.warning("Labels directory not found: %s", labels_dir)
        return counts

    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            counts[class_id] += 1
    return counts


def main() -> None:
    data_yaml_path = DATA_DIR / "data.yaml"
    if not data_yaml_path.exists():
        logger.error("data.yaml not found at %s", data_yaml_path)
        raise SystemExit(1)

    with open(data_yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    if not base_path.is_absolute():
        base_path = data_yaml_path.parent / base_path

    splits = {
        split_name: data_cfg[split_name]
        for split_name in ("train", "val", "test")
        if split_name in data_cfg
    }

    total_counts: Counter[int] = Counter()
    logger.info("=== Label counts per split ===")
    for split_name, images_rel_path in splits.items():
        # labels/ mirrors images/ in every YOLO dataset layout.
        labels_rel_path = images_rel_path.replace("images", "labels")
        labels_dir = base_path / labels_rel_path
        split_counts = count_labels_in_dir(labels_dir)
        total_counts.update(split_counts)

        logger.info("--- %s (%s) ---", split_name, labels_dir)
        for class_id, class_name in enumerate(CLASS_NAMES):
            logger.info("  %-16s %d", class_name, split_counts.get(class_id, 0))

    logger.info("=== Total across all splits ===")
    grand_total = sum(total_counts.values())
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = total_counts.get(class_id, 0)
        pct = (count / grand_total * 100) if grand_total else 0.0
        logger.info("  %-16s %6d  (%.1f%%)", class_name, count, pct)

    # Flag obvious imbalance so it's not easy to miss in the log output.
    if total_counts:
        max_count = max(total_counts.values())
        for class_id, class_name in enumerate(CLASS_NAMES):
            count = total_counts.get(class_id, 0)
            if count > 0 and max_count / count > 3:
                logger.warning(
                    "'%s' has %dx fewer instances than the largest class - "
                    "likely contributing to weaker performance on it.",
                    class_name,
                    round(max_count / count),
                )
            elif count == 0:
                logger.warning("'%s' has ZERO labeled instances in this dataset.", class_name)


if __name__ == "__main__":
    main()
