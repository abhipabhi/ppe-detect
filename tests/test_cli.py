import json

import cv2
import numpy as np
import pytest

from ppe_detect import cli
from ppe_detect.detector import Detection


def test_default_output_path():
    from pathlib import Path

    assert cli.default_output_path(Path("a/b/photo.jpg")) == Path("a/b/photo_detections.jpg")
    assert cli.default_output_path(Path("noext")) == Path("noext_detections.jpg")


def test_missing_image_returns_error(tmp_path, capsys):
    rc = cli.main([str(tmp_path / "absent.jpg")])
    assert rc == 1
    assert "could not read image" in capsys.readouterr().err


def test_invalid_conf_returns_error(tmp_path, capsys):
    image_path = tmp_path / "img.jpg"
    cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    rc = cli.main([str(image_path), "--conf", "1.5"])
    assert rc == 2
    assert "confidence" in capsys.readouterr().err


def test_end_to_end_with_stubbed_detector(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "img.jpg"
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 200, dtype=np.uint8))
    json_path = tmp_path / "out.json"
    output_path = tmp_path / "out.jpg"

    class _StubDetector:
        def __init__(self, config):
            pass

        def detect(self, image):
            return [Detection(0, "person", 0.9, (4.0, 4.0, 40.0, 60.0))]

    monkeypatch.setattr(cli, "Detector", _StubDetector)

    rc = cli.main([str(image_path), "-o", str(output_path), "--json", str(json_path)])
    assert rc == 0
    assert output_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["enhanced"] is False
    assert payload["detections"][0]["class_name"] == "person"

    out = capsys.readouterr().out
    assert "person" in out
    assert str(output_path) in out


@pytest.mark.integration
def test_real_cpu_inference_smoke(tmp_path):
    """Full-stack smoke test; downloads yolov8n.pt on first run."""
    image_path = tmp_path / "img.jpg"
    cv2.imwrite(str(image_path), np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
    rc = cli.main([str(image_path), "-o", str(tmp_path / "out.jpg"), "--device", "cpu"])
    assert rc == 0
    assert (tmp_path / "out.jpg").exists()
