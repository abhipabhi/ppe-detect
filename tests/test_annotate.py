import numpy as np

from ppe_detect.annotate import class_color, draw_detections
from ppe_detect.detector import Detection


def _image() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


def _detection(class_id: int = 0) -> Detection:
    return Detection(class_id, "person", 0.9, (16.0, 16.0, 96.0, 96.0))


def test_draw_returns_modified_copy():
    image = _image()
    out = draw_detections(image, [_detection()])
    assert out is not image
    assert not np.array_equal(out, image)
    assert np.count_nonzero(image) == 0  # input untouched


def test_draw_no_detections_is_unchanged_copy():
    image = _image()
    out = draw_detections(image, [])
    assert out is not image
    assert np.array_equal(out, image)


def test_class_color_is_stable_and_distinct():
    assert class_color(3) == class_color(3)
    assert class_color(0) != class_color(1)


def test_box_edge_pixels_are_class_color():
    out = draw_detections(_image(), [_detection(class_id=2)])
    b, g, r = out[56, 16]  # left edge of the box, below the label area
    assert (int(b), int(g), int(r)) == class_color(2)
