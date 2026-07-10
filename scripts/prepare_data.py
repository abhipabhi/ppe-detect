#!/usr/bin/env python
"""Assemble the SFCHD dataset for training and report image/label reconciliation.

Inputs (never modified):
  - downloaded images archive, already extracted to data/sfchd/images/
  - SFCHD YOLO label files + upstream split lists (reference folder)

Outputs (under data/sfchd/, gitignored):
  - labels/ copied next to images/
  - train.txt / val.txt / test.txt split lists with absolute local paths
  - sfchd.yaml ultralytics dataset config

Exits non-zero if the image/label orphan rate exceeds --max-orphan-rate
(default 2%) so a bad download cannot silently reach training.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_detect.sfchd import (
    collect_image_stems,
    parse_split_file,
    reconcile,
    redundant_splits,
    write_dataset_yaml,
    write_split_list,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path("/Users/abhi/dev/PPE-detection/SFCHD-SCALE-main/dataset_SFCHD")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="SFCHD reference folder holding labels/ and new_split_yolo/ (read-only)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / "data" / "sfchd"
    )
    parser.add_argument("--max-orphan-rate", type=float, default=0.02)
    args = parser.parse_args()

    images_dir = args.dataset_root / "images"
    labels_src = args.reference / "labels"
    labels_dst = args.dataset_root / "labels"
    splits_src = args.reference / "new_split_yolo"

    if not images_dir.is_dir():
        print(f"error: images directory not found: {images_dir}", file=sys.stderr)
        return 1
    if not labels_src.is_dir():
        print(f"error: labels directory not found: {labels_src}", file=sys.stderr)
        return 1

    # copy labels next to images (ultralytics resolves labels/ from images/)
    labels_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for label_file in labels_src.glob("*.txt"):
        target = labels_dst / label_file.name
        if not target.exists():
            shutil.copy2(label_file, target)
            copied += 1
    print(f"labels: {copied} copied, {len(list(labels_dst.glob('*.txt')))} total")

    image_stems = collect_image_stems(images_dir)
    label_stems = {p.stem for p in labels_dst.glob("*.txt")}
    report = reconcile(set(image_stems), label_stems)
    print("--- reconciliation (all files) ---")
    print(report.summary())

    # per-split reconciliation and local split lists
    print("--- per-split ---")
    usable: dict[str, list[str]] = {}
    members: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_file = splits_src / f"{split}.txt"
        if not split_file.exists():
            print(f"{split}: upstream split file missing, skipped")
            continue
        wanted = parse_split_file(split_file)
        present = [image_stems[s] for s in wanted if s in image_stems and s in label_stems]
        missing = len(wanted) - len(present)
        print(f"{split}: {len(wanted)} listed, {len(present)} usable, {missing} missing")
        usable[split] = present
        members[split] = {s for s in wanted if s in image_stems}

    splits: dict[str, str] = {}
    for split, present in usable.items():
        if split in redundant_splits(members):
            print(f"{split}: subset of another split — dropped as redundant")
            continue
        list_path = args.dataset_root / f"{split}.txt"
        write_split_list(list_path, images_dir, present)
        splits[split] = f"{split}.txt"

    write_dataset_yaml(args.dataset_root / "sfchd.yaml", args.dataset_root, splits)
    print(f"dataset config written to {args.dataset_root / 'sfchd.yaml'}")

    if report.orphan_rate > args.max_orphan_rate:
        print(
            f"FAIL: orphan rate {report.orphan_rate:.2%} exceeds "
            f"{args.max_orphan_rate:.2%} — stopping before training",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
