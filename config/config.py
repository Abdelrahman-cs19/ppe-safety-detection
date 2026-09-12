"""
Central configuration for the PPE Safety Detection System.

Every other module (detection, tracking, association, rules, database,
alerts, dashboard) imports its settings from here instead of hard-coding
values. This keeps the system tunable from one place and makes it easy
to override settings via environment variables in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load a local .env file if present (created from .env.example).
load_dotenv()

# --------------------------------------------------------------------------
# Path constants
# --------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
SCREENSHOTS_DIR: Final[Path] = DATA_DIR / "violation_screenshots"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SCREENSHOTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Detection classes
# --------------------------------------------------------------------------
# Index position matters: it must match the `names:` order in data/data.yaml
# and therefore the class order the trained YOLO model outputs.
#
# NOTE: this matches the actual Roboflow dataset in use (v1), which labels
# violations directly (no-helmet / no-vest) rather than only labeling PPE
# presence. It does not yet include gloves or safety_glasses — those were
# in the original spec but no public dataset had them alongside these
# classes, so they're deferred to a later dataset-merge + retrain pass.
# When that happens, CLASS_NAMES/CLASS_TO_ID grow and data.yaml must be
# updated together, then the model retrained from scratch (class indices
# in an already-trained checkpoint can't just be appended to).
CLASS_NAMES: Final[list[str]] = [
    "helmet",
    "no-helmet",
    "no-vest",
    "person",
    "vest",
]
CLASS_TO_ID: Final[dict[str, int]] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Direct "violation" classes the model predicts (v1 dataset design).
# The rule engine (Milestone 5) confirms a violation when one of these is
# detected on/near a tracked person for `violation_persistence_frames`
# frames in a row - see RuleEngineConfig below.
VIOLATION_CLASSES: Final[list[str]] = ["no-helmet", "no-vest"]

# PPE items this model can currently detect, kept for reference/logging.
# Extend this (and CLASS_NAMES/data.yaml) once gloves/safety_glasses data
# is merged in.
DETECTABLE_PPE: Final[list[str]] = ["helmet", "vest"]

# Class pairs describing the same physical region in opposite compliance
# states. Only these specific pairs get cross-class conflict resolution
# in detection/detector.py - unrelated classes (e.g. "person") are never
# touched by it.
MUTUALLY_EXCLUSIVE_PAIRS: Final[list[tuple[str, str]]] = [
    ("helmet", "no-helmet"),
    ("vest", "no-vest"),
]


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Settings for the YOLO detector (Milestone 2)."""

    model_path: Path = MODELS_DIR / "ppe_yolo.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    device: str = os.getenv("PPE_DEVICE", "cpu")  # "cpu", "cuda", "cuda:0", "mps"
    image_size: int = 640
    half_precision: bool = False  # set True on supported GPUs for faster inference
    # NOTE: NOT using YOLO's built-in class-agnostic NMS here - it suppresses
    # ANY overlapping boxes regardless of class (e.g. a "vest" box overlapping
    # something unrelated), which caused unrelated classes to interfere with
    # each other in testing. Instead, conflicting-pair resolution is handled
    # surgically in detector.py, only between mutually exclusive class pairs
    # (helmet/no-helmet, vest/no-vest) - see MUTUALLY_EXCLUSIVE_PAIRS below.
    agnostic_nms: bool = False
    # IoU threshold above which two boxes from a mutually-exclusive pair
    # (e.g. "helmet" and "no-helmet") are considered "the same physical
    # region" - the lower-confidence one is then dropped.
    conflict_resolution_iou: float = 0.5


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Settings for ByteTrack (Milestone 3)."""

    tracker_yaml: str = "bytetrack.yaml"  # Ultralytics ships this config
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30  # frames to keep a lost track alive
    match_thresh: float = 0.8


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    """Settings for PPE-to-worker association (Milestone 4)."""

    # An IoU/center-distance based association is used; see association/.
    min_overlap_ratio: float = 0.10  # PPE box must overlap this much w/ person box
    max_center_distance_ratio: float = 0.6  # normalized by person box diagonal


@dataclass(frozen=True, slots=True)
class RuleEngineConfig:
    """Settings for the violation confirmation logic (Milestones 5-6)."""

    violation_persistence_frames: int = 15  # ~0.5s at 30 FPS before confirming
    violation_cooldown_seconds: int = 60  # avoid re-alerting the same worker/type


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Settings for SQLite/SQLAlchemy storage (Milestone 7)."""

    db_url: str = f"sqlite:///{PROCESSED_DATA_DIR / 'ppe_safety.db'}"
    echo_sql: bool = False


@dataclass(frozen=True, slots=True)
class AlertConfig:
    """Settings for Telegram alerts (Milestone 9)."""

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN"))


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level container aggregating every sub-config."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    rules: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    log_level: str = os.getenv("PPE_LOG_LEVEL", "INFO")


# Single shared instance imported by every module: `from config.config import settings`
settings = AppConfig()
