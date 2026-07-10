#!/usr/bin/env python
"""Evaluation harness: standard val metrics + pre-registered low-light CLAHE A/B.

Regenerates results/metrics.md end-to-end with one command:

  python scripts/evaluate.py --weights runs/sfchd-y8n/weights/best.pt --device mps

Steps:
  1. standard evaluation of the weights on the SFCHD val split (per-class table)
  2. low-light A/B (protocol pre-registered in ppe_detect.evaluation):
     a. darken every val image with a fixed-seed per-image gamma curve
     b. arm A: evaluate the darkened set as-is
     c. arm B: evaluate CLAHE-enhanced copies of the identical darkened set
  3. write results/metrics.md with full provenance

Derived image sets are cached under data/sfchd/lowlight_ab/ (gitignored);
--regenerate forces rebuilding them.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from datetime import date
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_detect.enhance import apply_clahe
from ppe_detect.evaluation import (
    DARKEN_SEED,
    GAMMA_RANGE,
    ab_markdown,
    gamma_assignments,
    gamma_darken,
    metrics_markdown,
    rows_from_results,
    sha256_file,
)
from ppe_detect.sfchd import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "sfchd"
AB_ROOT = DATASET_ROOT / "lowlight_ab"


def build_derived_set(name: str, source_images: list[Path], enhance: bool) -> Path:
    """Create <AB_ROOT>/<name>/{images,labels,val.txt,data.yaml}; return the yaml."""
    root = AB_ROOT / name
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    gammas = gamma_assignments([p.name for p in source_images])
    for src in source_images:
        dst = images_dir / src.name
        if not dst.exists():
            image = cv2.imread(str(src))
            if image is None:
                raise RuntimeError(f"unreadable image: {src}")
            image = gamma_darken(image, gammas[src.name])
            if enhance:
                image = apply_clahe(image)
            cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        label_src = DATASET_ROOT / "labels" / f"{src.stem}.txt"
        label_dst = labels_dir / f"{src.stem}.txt"
        if not label_dst.exists():
            shutil.copy2(label_src, label_dst)

    split_list = root / "val.txt"
    split_list.write_text("".join(f"{images_dir / p.name}\n" for p in source_images))
    yaml_path = root / "data.yaml"
    lines = [f"path: {root}", "train: val.txt", "val: val.txt", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def run_val(weights: str, data_yaml: Path, device: str, batch: int):
    from ultralytics import YOLO

    results = YOLO(weights).val(
        data=str(data_yaml), split="val", device=device, batch=batch, verbose=False, plots=False
    )
    return rows_from_results(results)


def provenance_block(args, val_list: Path) -> str:
    import numpy
    import torch
    import ultralytics

    import ppe_detect

    val_images = val_list.read_text().split()
    return "\n".join(
        [
            f"- date: {date.today().isoformat()}",
            f"- command: `python scripts/evaluate.py --weights {args.weights} "
            f"--device {args.device} --batch {args.batch}`",
            f"- weights: `{Path(args.weights).name}` sha256 `{sha256_file(Path(args.weights))}`",
            f"- split: SFCHD val, {len(val_images)} images, list sha256 `{sha256_file(val_list)}`",
            f"- low-light protocol: per-image gamma in {GAMMA_RANGE}, seed {DARKEN_SEED}, "
            "sorted-filename assignment; CLAHE clip 3.0, tiles 8x8 (pre-registered)",
            f"- versions: python {platform.python_version()}, ppe-detect {ppe_detect.__version__}, "
            f"ultralytics {ultralytics.__version__}, torch {torch.__version__}, "
            f"opencv {cv2.__version__}, numpy {numpy.__version__}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(REPO_ROOT / "runs/sfchd-y8n/weights/best.pt"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "metrics.md")
    parser.add_argument(
        "--regenerate", action="store_true", help="rebuild cached darkened image sets"
    )
    parser.add_argument(
        "--skip-ab", action="store_true", help="standard evaluation only (headline check)"
    )
    args = parser.parse_args()

    val_list = DATASET_ROOT / "val.txt"
    if not val_list.exists():
        print("error: run scripts/prepare_data.py first", file=sys.stderr)
        return 1
    if args.regenerate and AB_ROOT.exists():
        shutil.rmtree(AB_ROOT)

    print("== standard evaluation (val split) ==")
    standard_rows = run_val(args.weights, DATASET_ROOT / "sfchd.yaml", args.device, args.batch)
    print(metrics_markdown(standard_rows))

    sections = [
        "# Evaluation results",
        "",
        "## Provenance",
        "",
        provenance_block(args, val_list),
        "",
        "## Standard evaluation — SFCHD val split",
        "",
        metrics_markdown(standard_rows),
    ]

    if not args.skip_ab:
        source_images = [Path(line) for line in val_list.read_text().split()]
        print("== building darkened set (arm A) ==")
        dark_yaml = build_derived_set("dark", source_images, enhance=False)
        print("== building darkened+CLAHE set (arm B) ==")
        clahe_yaml = build_derived_set("dark_clahe", source_images, enhance=True)
        print("== evaluating arm A (dark) ==")
        dark_rows = run_val(args.weights, dark_yaml, args.device, args.batch)
        print("== evaluating arm B (dark + CLAHE) ==")
        clahe_rows = run_val(args.weights, clahe_yaml, args.device, args.batch)
        print(ab_markdown(dark_rows, clahe_rows))
        sections += [
            "",
            "## Low-light A/B — synthetically darkened val split (pre-registered)",
            "",
            "Same weights, same split, deterministic darkening (see Provenance). "
            "Arm A: darkened images. Arm B: CLAHE applied to the identical darkened images.",
            "",
            ab_markdown(dark_rows, clahe_rows),
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections) + "\n")
    print(f"metrics written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
