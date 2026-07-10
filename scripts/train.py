#!/usr/bin/env python
"""Fine-tune YOLOv8 on the SFCHD dataset.

Micro smoke run (~100 images, 1 epoch, MPS):
  python scripts/train.py --epochs 1 --fraction 0.0101 --name smoke

Full overnight run on Apple-Silicon MPS:
  python scripts/train.py --epochs 50 --name sfchd-y8n --device mps

Fallback (free Kaggle/Colab T4): upload data/sfchd + this script, run with
--device 0, download runs/<name>/weights/best.pt back into this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "sfchd" / "sfchd.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps, cpu, or CUDA index like 0")
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="fraction of train set (smoke runs)"
    )
    parser.add_argument("--name", default="sfchd-y8n", help="run name under runs/")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        name=args.name,
        project=str(REPO_ROOT / "runs"),
        workers=args.workers,
        seed=42,
        patience=15,
    )


if __name__ == "__main__":
    main()
