from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ppe_detect.api import create_app
from ppe_detect.config import DetectConfig
from ppe_detect.detector import Detection


class StubDetector:
    """Duck-typed Detector: records calls, returns one fixed detection."""

    def __init__(self, config):
        self.config = config
        self.loaded = False

    def load(self):
        self.loaded = True
        return self

    def detect(self, image_bgr):
        return [Detection(1, "helmet", 0.92, (10.0, 10.0, 50.0, 50.0))]


@pytest.fixture()
def stub_weights(tmp_path):
    weights = tmp_path / "stub.pt"
    weights.write_bytes(b"stub-weights")
    return weights


@pytest.fixture()
def client(stub_weights):
    config = DetectConfig(weights=str(stub_weights))
    app = create_app(config=config, detector_factory=StubDetector)
    with TestClient(app) as test_client:  # context manager runs the lifespan hook
        yield test_client


def _jpeg_bytes(value: int = 180, size: int = 64) -> bytes:
    image = np.full((size, size, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_healthz_reports_model_and_sha256(client, stub_weights):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "stub.pt"
    # sha256 of the stub weights file, so the endpoint proves what it serves
    import hashlib

    assert body["weights_sha256"] == hashlib.sha256(b"stub-weights").hexdigest()


def test_model_loaded_once_at_startup(stub_weights):
    config = DetectConfig(weights=str(stub_weights))
    holder = {}

    def factory(cfg):
        holder["detector"] = StubDetector(cfg)
        return holder["detector"]

    app = create_app(config=config, detector_factory=factory)
    with TestClient(app):
        assert holder["detector"].loaded is True


def test_detect_returns_json_payload(client):
    response = client.post("/detect", files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == ["helmet"]
    assert body["confidences"] == [0.92]
    assert body["boxes"] == [[10.0, 10.0, 50.0, 50.0]]
    assert body["enhanced"] is False
    assert body["timing"]["inference_ms"] >= 0
    assert body["image"] == {"width": 64, "height": 64}


def test_detect_enhance_auto_flags_dark_image(client):
    dark = np.full((64, 64, 3), 20, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", dark)
    assert ok
    response = client.post(
        "/detect?enhance=auto",
        files={"image": ("dark.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["enhanced"] is True


def test_detect_output_image_returns_jpeg(client):
    response = client.post(
        "/detect?output=image", files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (64, 64, 3)


def test_non_image_upload_is_400(client):
    response = client.post(
        "/detect", files={"image": ("notes.txt", b"hello, not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "decode" in response.json()["error"]


def test_corrupt_image_is_400(client):
    corrupt = _jpeg_bytes()[:40]  # truncated JPEG header
    response = client.post("/detect", files={"image": ("x.jpg", corrupt, "image/jpeg")})
    assert response.status_code == 400
    assert "error" in response.json()


def test_empty_upload_is_400(client):
    response = client.post("/detect", files={"image": ("x.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["error"] == "empty upload"


def test_oversized_payload_is_413(stub_weights, monkeypatch):
    monkeypatch.setenv("PPE_MAX_UPLOAD_MB", "0.001")  # 1 KB cap
    config = DetectConfig(weights=str(stub_weights))
    app = create_app(config=config, detector_factory=StubDetector)
    with TestClient(app) as small_client:
        response = small_client.post(
            "/detect", files={"image": ("x.jpg", _jpeg_bytes(size=256), "image/jpeg")}
        )
    assert response.status_code == 413
    assert "error" in response.json()


def test_missing_file_field_is_422_json(client):
    response = client.post("/detect")
    assert response.status_code == 422
    assert response.json()["detail"]


def test_invalid_enhance_value_is_422_json(client):
    response = client.post(
        "/detect?enhance=always", files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 422


REAL_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "ppe-detect-y8n-sfchd.pt"


@pytest.mark.integration
@pytest.mark.skipif(not REAL_WEIGHTS.exists(), reason="run scripts/get_weights.py first")
def test_real_weights_end_to_end():
    sample = Path(__file__).resolve().parents[1] / "assets" / "sample.jpg"
    config = DetectConfig(weights=str(REAL_WEIGHTS))
    app = create_app(config=config)
    with TestClient(app) as client:
        health = client.get("/healthz").json()
        assert health["model"] == "ppe-detect-y8n-sfchd.pt"
        response = client.post(
            "/detect", files={"image": ("sample.jpg", sample.read_bytes(), "image/jpeg")}
        )
    assert response.status_code == 200
    body = response.json()
    sfchd_classes = {
        "person", "helmet", "self_clothes", "safety_clothes", "head", "blur_head", "blur_clothes"
    }
    assert set(body["classes"]) <= sfchd_classes
    assert len(body["classes"]) > 0
