from pathlib import Path

from ppe_detect.sfchd import (
    CLASS_NAMES,
    collect_image_stems,
    parse_split_file,
    reconcile,
    write_dataset_yaml,
    write_split_list,
)


def test_class_schema_is_published_sfchd_order():
    assert CLASS_NAMES == (
        "person",
        "helmet",
        "self_clothes",
        "safety_clothes",
        "head",
        "blur_head",
        "blur_clothes",
    )


def test_parse_split_file_takes_basenames(tmp_path):
    split = tmp_path / "train.txt"
    split.write_text(
        "/home/yfs/data/QY_final_dataset/images/cam1_001.jpg\n"
        "\n"
        "/other/root/images/cam2_007.jpg  \n"
    )
    assert parse_split_file(split) == ["cam1_001", "cam2_007"]


def test_reconcile_partitions_and_rates():
    report = reconcile({"a", "b", "c"}, {"b", "c", "d"})
    assert report.matched == ["b", "c"]
    assert report.images_without_labels == ["a"]
    assert report.labels_without_images == ["d"]
    assert report.total == 4
    assert report.match_rate == 0.5
    assert report.orphan_rate == 0.5
    assert "match rate: 50.0000%" in report.summary()


def test_reconcile_empty_sets():
    report = reconcile(set(), set())
    assert report.match_rate == 0.0
    assert report.total == 0


def test_collect_image_stems_filters_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    stems = collect_image_stems(tmp_path)
    assert stems == {"a": "a.jpg", "b": "b.PNG"}


def test_write_split_list_absolute_sorted(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    out = tmp_path / "train.txt"
    write_split_list(out, images_dir, ["b.jpg", "a.jpg"])
    lines = out.read_text().splitlines()
    assert lines == [
        str((images_dir / "a.jpg").resolve()),
        str((images_dir / "b.jpg").resolve()),
    ]


def test_write_dataset_yaml_contains_all_classes(tmp_path):
    yaml_path = tmp_path / "sfchd.yaml"
    write_dataset_yaml(yaml_path, tmp_path, {"train": "train.txt", "val": "val.txt"})
    text = yaml_path.read_text()
    assert f"path: {tmp_path.resolve()}" in text
    assert "train: train.txt" in text
    assert "6: blur_clothes" in text
    assert text.count(":") >= 2 + len(CLASS_NAMES)
