"""Runtime configuration for detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ENHANCE_MODES = ("auto", "on", "off")
DEVICES = ("cpu", "mps", "cuda")

DEFAULT_WEIGHTS = "yolov8n.pt"


@dataclass(frozen=True)
class DetectConfig:
    """Validated settings for a detection run."""

    weights: str = DEFAULT_WEIGHTS
    confidence: float = 0.25
    image_size: int = 640
    enhance: str = "off"
    device: str = "cpu"
    low_light_threshold: float = 80.0

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.image_size <= 0 or self.image_size % 32 != 0:
            raise ValueError(f"image_size must be a positive multiple of 32, got {self.image_size}")
        if self.enhance not in ENHANCE_MODES:
            raise ValueError(f"enhance must be one of {ENHANCE_MODES}, got {self.enhance!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if not 0.0 <= self.low_light_threshold <= 255.0:
            raise ValueError(
                f"low_light_threshold must be in [0, 255], got {self.low_light_threshold}"
            )
        # A bare name like "yolov8n.pt" is resolved (or downloaded) by ultralytics;
        # anything path-like must already exist.
        weights_path = Path(self.weights)
        if weights_path.parent != Path(".") and not weights_path.exists():
            raise FileNotFoundError(f"weights file not found: {self.weights}")
