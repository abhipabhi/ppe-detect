import pytest

from ppe_detect.config import DetectConfig


def test_defaults_are_valid():
    config = DetectConfig()
    assert config.device == "cpu"
    assert config.enhance == "off"


@pytest.mark.parametrize("conf", [0.0, -0.1, 1.5])
def test_confidence_out_of_range_rejected(conf):
    with pytest.raises(ValueError, match="confidence"):
        DetectConfig(confidence=conf)


@pytest.mark.parametrize("size", [0, -640, 100, 641])
def test_image_size_must_be_positive_multiple_of_32(size):
    with pytest.raises(ValueError, match="image_size"):
        DetectConfig(image_size=size)


def test_unknown_enhance_mode_rejected():
    with pytest.raises(ValueError, match="enhance"):
        DetectConfig(enhance="always")


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="device"):
        DetectConfig(device="tpu")


def test_missing_weights_path_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        DetectConfig(weights=str(tmp_path / "nope" / "best.pt"))


def test_bare_weights_name_allowed_without_file():
    # ultralytics resolves/downloads bare names like yolov8n.pt itself
    assert DetectConfig(weights="yolov8n.pt").weights == "yolov8n.pt"


def test_existing_weights_path_allowed(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"stub")
    assert DetectConfig(weights=str(weights)).weights == str(weights)
