"""Command-line interface: detect PPE in a single image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .annotate import draw_detections
from .config import DEFAULT_WEIGHTS, DEVICES, ENHANCE_MODES, DetectConfig
from .detector import Detector
from .enhance import maybe_enhance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppe-detect",
        description="Detect PPE in an image and write an annotated copy.",
    )
    parser.add_argument("image", type=Path, help="path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="annotated image path (default: <input>_detections.<ext>)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        dest="json_path",
        help="also write detections as JSON to this path",
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument(
        "--enhance",
        choices=ENHANCE_MODES,
        default="off",
        help="CLAHE low-light preprocessing: on, off, or brightness-gated auto",
    )
    parser.add_argument("--device", choices=DEVICES, default="cpu", help="inference device")
    return parser


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".jpg"
    return input_path.with_name(f"{input_path.stem}_detections{suffix}")


def run(args: argparse.Namespace) -> int:
    config = DetectConfig(
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        enhance=args.enhance,
        device=args.device,
    )

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 1

    image, enhanced = maybe_enhance(image, config.enhance, config.low_light_threshold)
    detections = Detector(config).detect(image)

    output_path = args.output or default_output_path(args.image)
    annotated = draw_detections(image, detections)
    if not cv2.imwrite(str(output_path), annotated):
        print(f"error: could not write output image: {output_path}", file=sys.stderr)
        return 1

    if enhanced:
        print("low-light enhancement applied (CLAHE)")
    if detections:
        for det in detections:
            print(f"{det.class_name:<16} conf={det.confidence:.2f} box={det.to_dict()['box_xyxy']}")
    else:
        print("no detections above threshold")
    print(f"annotated image written to {output_path}")

    if args.json_path:
        payload = {
            "image": str(args.image),
            "enhanced": enhanced,
            "detections": [det.to_dict() for det in detections],
        }
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"detections written to {args.json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
