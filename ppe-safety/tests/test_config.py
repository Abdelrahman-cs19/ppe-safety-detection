"""Sanity checks for config.config — cheap, no GPU/model required."""

from config.config import CLASS_NAMES, CLASS_TO_ID, DETECTABLE_PPE, VIOLATION_CLASSES, settings


def test_class_names_are_unique_and_ordered() -> None:
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))
    assert CLASS_NAMES[0] == "helmet"


def test_class_to_id_matches_class_names() -> None:
    for name, idx in CLASS_TO_ID.items():
        assert CLASS_NAMES[idx] == name


def test_violation_and_ppe_classes_are_valid() -> None:
    for item in VIOLATION_CLASSES + DETECTABLE_PPE:
        assert item in CLASS_NAMES


def test_settings_load_without_error() -> None:
    assert settings.detection.confidence_threshold > 0
    assert settings.rules.violation_persistence_frames > 0
