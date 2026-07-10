"""YOLO inference wrapper producing plain, serializable detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectConfig


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box_xyxy": [round(v, 2) for v in self.box_xyxy],
        }


def parse_result(result) -> list[Detection]:
    """Convert one ultralytics result object into Detections.

    Only relies on `result.names` and `result.boxes.{cls,conf,xyxy}`, so tests
    can exercise it with a lightweight stub.
    """
    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is None:
        return detections
    for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
        class_id = int(cls)
        coords = tuple(float(v) for v in np.asarray(xyxy).reshape(4))
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(result.names[class_id]),
                confidence=float(conf),
                box_xyxy=coords,
            )
        )
    return detections


class Detector:
    """Lazy-loading YOLO wrapper; model construction is deferred to first use."""

    def __init__(self, config: DetectConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO  # deferred: keeps import cheap for --help/tests

            self._model = YOLO(self.config.weights)
        return self._model

    def load(self) -> "Detector":
        """Force model construction now (e.g. at service startup)."""
        _ = self.model
        return self

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image_bgr,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )
        return parse_result(results[0])
