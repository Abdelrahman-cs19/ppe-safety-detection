"""
Download a third-party pretrained PPE model for side-by-side comparison
against our own trained model. Not used anywhere in the main pipeline -
purely a benchmarking reference.

Usage
-----
    python -m scripts.download_reference_model
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from config.config import MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# Community model whose classes (Human, Helmet, No-Helmet, Vest) closely
# match ours. Reported metrics are the author's own, not independently
# verified - this is for a rough visual comparison, not a benchmark claim.
REFERENCE_REPO_ID = "harsh-77/ppe-detection"
REFERENCE_FILENAME = "best.pt"
REFERENCE_LOCAL_NAME = "reference_ppe.pt"


def main() -> None:
    logger.info("Downloading reference model from Hugging Face: %s", REFERENCE_REPO_ID)
    downloaded_path = hf_hub_download(repo_id=REFERENCE_REPO_ID, filename=REFERENCE_FILENAME)

    target_path = Path(MODELS_DIR) / REFERENCE_LOCAL_NAME
    shutil.copy2(downloaded_path, target_path)
    logger.info("Saved reference model to %s", target_path)
    logger.info(
        "Try it with: python -m scripts.webcam_demo --weights %s --source 0",
        target_path,
    )
    logger.info(
        "Or compare both side-by-side with: python -m scripts.compare_models --source 0"
    )


if __name__ == "__main__":
    main()
