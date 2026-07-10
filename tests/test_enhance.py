import numpy as np
import pytest

from ppe_detect.enhance import apply_clahe, is_low_light, maybe_enhance, mean_brightness


def _flat_image(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _gradient_dark_image() -> np.ndarray:
    # dark image with some structure so CLAHE has contrast to work with
    row = np.linspace(0, 60, 64, dtype=np.uint8)
    return np.dstack([np.tile(row, (64, 1))] * 3)


def test_mean_brightness_flat_image():
    assert mean_brightness(_flat_image(50)) == pytest.approx(50, abs=1)


def test_is_low_light_thresholding():
    assert is_low_light(_flat_image(30), threshold=80)
    assert not is_low_light(_flat_image(200), threshold=80)


def test_clahe_preserves_shape_and_dtype():
    out = apply_clahe(_gradient_dark_image())
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_clahe_brightens_dark_structured_image():
    image = _gradient_dark_image()
    assert mean_brightness(apply_clahe(image)) > mean_brightness(image)


def test_clahe_does_not_mutate_input():
    image = _gradient_dark_image()
    before = image.copy()
    apply_clahe(image)
    assert np.array_equal(image, before)


def test_maybe_enhance_off_is_identity():
    image = _gradient_dark_image()
    out, enhanced = maybe_enhance(image, "off")
    assert not enhanced
    assert out is image


def test_maybe_enhance_on_always_applies():
    out, enhanced = maybe_enhance(_flat_image(200), "on")
    assert enhanced


def test_maybe_enhance_auto_gates_on_brightness():
    _, enhanced_dark = maybe_enhance(_gradient_dark_image(), "auto", threshold=80)
    _, enhanced_bright = maybe_enhance(_flat_image(200), "auto", threshold=80)
    assert enhanced_dark
    assert not enhanced_bright


def test_maybe_enhance_rejects_unknown_mode():
    with pytest.raises(ValueError):
        maybe_enhance(_flat_image(50), "sometimes")
