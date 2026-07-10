import numpy as np

from ppe_detect.detector import Detection, parse_result


class _StubBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy


class _StubResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


NAMES = {0: "person", 1: "helmet"}


def test_parse_result_converts_boxes():
    result = _StubResult(
        NAMES,
        _StubBoxes(
            cls=np.array([0, 1]),
            conf=np.array([0.91, 0.55]),
            xyxy=np.array([[10.0, 20.0, 110.0, 220.0], [30.0, 5.0, 60.0, 40.0]]),
        ),
    )
    detections = parse_result(result)
    assert len(detections) == 2
    first = detections[0]
    assert first.class_id == 0
    assert first.class_name == "person"
    assert first.confidence == 0.91
    assert first.box_xyxy == (10.0, 20.0, 110.0, 220.0)


def test_parse_result_empty_boxes():
    result = _StubResult(NAMES, _StubBoxes(np.array([]), np.array([]), np.zeros((0, 4))))
    assert parse_result(result) == []


def test_parse_result_no_boxes_attribute():
    assert parse_result(_StubResult(NAMES, None)) == []


def test_detection_to_dict_rounds_values():
    det = Detection(1, "helmet", 0.98765, (1.234, 2.345, 3.456, 4.567))
    payload = det.to_dict()
    assert payload["confidence"] == 0.9877
    assert payload["box_xyxy"] == [1.23, 2.35, 3.46, 4.57]
    assert payload["class_name"] == "helmet"
