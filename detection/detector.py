"""
YOLO detector wrapper for the PPE safety system.

Wraps `ultralytics.YOLO` behind a small, typed interface so the rest of
the pipeline (tracking, association, rules) depends on `Detection`
objects and a `.predict()` method — not on Ultralytics' result objects
directly. Makes it easy to swap model versions or even detector
backends later without touching downstream code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config.config import DetectionConfig, settings
from detection.postprocess import resolve_mutually_exclusive_conflicts
from detection.schemas import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class InferenceStats:
    """Timing info for one predict() call, in milliseconds."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


class ModelLoadError(RuntimeError):
    """Raised when the configured YOLO weights can't be loaded."""


class PPEDetector:
    """Loads a trained YOLO model and runs PPE/violation detection on frames."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        weights_path_override: Path | str | None = None,
    ) -> None:
        self.config = config or settings.detection
        self._weights_path_override = Path(weights_path_override) if weights_path_override else None
        self._model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        weights_path = self._weights_path_override or Path(self.config.model_path)
        if not weights_path.exists():
            raise ModelLoadError(
                f"YOLO weights not found at {weights_path}. "
                "Run `python -m scripts.train` first (see README Milestone 1)."
            )
        try:
            logger.info("Loading YOLO model from %s", weights_path)
            self._model = YOLO(str(weights_path))
            # NOTE: deliberately not calling self._model.to(device) here -
            # torch.nn.Module.to() rejects bare device strings like "0"
            # (it needs "cuda:0"), whereas Ultralytics' own predict()/train()
            # calls already accept "0"/"cpu"/"cuda:0" and resolve it
            # correctly per-call. Device is passed via `device=` in predict().
        except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
            raise ModelLoadError(f"Failed to load YOLO weights from {weights_path}: {exc}") from exc

        loaded_names = list(self._model.names.values())
        logger.info("Model loaded. Classes: %s | device: %s", loaded_names, self.config.device)

    def predict(self, frame: np.ndarray) -> tuple[list[Detection], InferenceStats]:
        """
        Run detection on a single BGR frame (as read by OpenCV).

        Returns the list of detections above the configured confidence
        threshold, plus timing stats for the FPS/latency overlay.
        """
        if self._model is None:
            raise ModelLoadError("Model is not loaded.")

        start = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            half=self.config.half_precision,
            agnostic_nms=self.config.agnostic_nms,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        stats = self._extract_stats(result, fallback_total_ms=elapsed_ms)
        detections = self._parse_result(result)
        detections = resolve_mutually_exclusive_conflicts(
            detections, iou_threshold=self.config.conflict_resolution_iou
        )

        return detections, stats

    def _parse_result(self, result) -> list[Detection]:  # noqa: ANN001 - ultralytics Results type
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            class_name = self._model.names.get(class_id, str(class_id))

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    @staticmethod
    def _extract_stats(result, fallback_total_ms: float) -> InferenceStats:  # noqa: ANN001
        speed = getattr(result, "speed", None)
        if speed:
            return InferenceStats(
                preprocess_ms=speed.get("preprocess", 0.0),
                inference_ms=speed.get("inference", 0.0),
                postprocess_ms=speed.get("postprocess", 0.0),
            )
        # Fallback if Ultralytics ever stops exposing `.speed`.
        return InferenceStats(preprocess_ms=0.0, inference_ms=fallback_total_ms, postprocess_ms=0.0)

    @property
    def class_names(self) -> dict[int, str]:
        assert self._model is not None
        return dict(self._model.names)
