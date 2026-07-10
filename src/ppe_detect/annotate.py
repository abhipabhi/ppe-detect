"""Drawing detections onto images."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

# Fixed BGR palette cycled by class id, so a given class always gets one color.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (80, 175, 76),   # green
    (219, 152, 52),  # blue
    (60, 76, 231),   # red
    (190, 146, 112), # steel
    (32, 165, 218),  # amber
    (153, 87, 189),  # purple
    (128, 128, 0),   # teal
)


def class_color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the image with labeled boxes drawn on it."""
    canvas = image_bgr.copy()
    thickness = max(1, round(min(canvas.shape[:2]) / 320))
    font_scale = max(0.4, min(canvas.shape[:2]) / 1280)
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box_xyxy)
        color = class_color(det.class_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        ty = y1 - baseline if y1 - th - baseline >= 0 else y1 + th + baseline
        cv2.rectangle(canvas, (x1, ty - th), (x1 + tw, ty + baseline), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return canvas
