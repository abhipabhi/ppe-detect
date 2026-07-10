import numpy as np
import pytest

from ppe_detect.evaluation import (
    DARKEN_SEED,
    GAMMA_RANGE,
    ab_markdown,
    gamma_assignments,
    gamma_darken,
    markdown_table,
    metrics_markdown,
    rows_from_results,
    sha256_file,
)


def test_gamma_darken_darkens_and_preserves_extremes():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = gamma_darken(image, 2.5)
    assert out.mean() < image.mean()
    extremes = np.zeros((2, 1, 3), dtype=np.uint8)
    extremes[1] = 255
    out = gamma_darken(extremes, 2.5)
    assert out[0].max() == 0  # black stays black
    assert out[1].min() == 255  # white stays white


def test_gamma_one_is_identity():
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert np.array_equal(gamma_darken(image, 1.0), image)


def test_gamma_assignments_deterministic_and_order_independent():
    a = gamma_assignments(["b.jpg", "a.jpg", "c.jpg"])
    b = gamma_assignments(["c.jpg", "a.jpg", "b.jpg"])
    assert a == b
    assert all(GAMMA_RANGE[0] <= g <= GAMMA_RANGE[1] for g in a.values())


def test_gamma_assignments_seed_sensitivity():
    assert gamma_assignments(["a.jpg"], seed=DARKEN_SEED) != gamma_assignments(
        ["a.jpg"], seed=DARKEN_SEED + 1
    )


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"ppe")
    assert sha256_file(f) == (
        "79b296dde86122db801869195e25b6820ba9f3417ea83946f15fab114c0282b7"
    )


def test_markdown_table_shape():
    table = markdown_table(["A", "B"], [["1", "2"], ["3", "4"]])
    lines = table.splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"
    assert len(lines) == 4


class _StubBox:
    def __init__(self):
        self.ap_class_index = np.array([0, 2])
        self.mp, self.mr, self.map50, self.map = 0.8, 0.7, 0.75, 0.5

    def class_result(self, pos):
        return [(0.9, 0.8, 0.85, 0.6), (0.7, 0.6, 0.65, 0.4)][pos]


class _StubResults:
    names = {0: "person", 1: "helmet", 2: "self_clothes"}
    box = _StubBox()


def test_rows_from_results_skips_absent_classes():
    rows = rows_from_results(_StubResults())
    assert [r["name"] for r in rows] == ["person", "self_clothes", "all"]
    assert rows[0]["map50"] == 0.85
    assert rows[-1]["map50_95"] == 0.5


def test_metrics_markdown_formats_three_decimals():
    rows = rows_from_results(_StubResults())
    text = metrics_markdown(rows)
    assert "| person | 0.900 | 0.800 | 0.850 | 0.600 |" in text


def test_ab_markdown_signed_deltas():
    dark = [{"name": "all", "map50": 0.50, "map50_95": 0.30}]
    clahe = [{"name": "all", "map50": 0.55, "map50_95": 0.28}]
    text = ab_markdown(dark, clahe)
    assert "+0.050" in text
    assert "-0.020" in text


def test_ab_markdown_requires_matching_names():
    with pytest.raises(KeyError):
        ab_markdown(
            [{"name": "person", "map50": 0.5, "map50_95": 0.3}],
            [{"name": "helmet", "map50": 0.5, "map50_95": 0.3}],
        )
