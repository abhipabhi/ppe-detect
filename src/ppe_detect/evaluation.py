"""Evaluation utilities: provenance, metric tables, pre-registered low-light transform.

The low-light A/B protocol is PRE-REGISTERED and must not be tuned after results
are seen:
  - darkening: per-image gamma curve I' = 255 * (I/255)^gamma, gamma drawn
    uniformly from GAMMA_RANGE with numpy default_rng(DARKEN_SEED), images
    processed in sorted filename order so the gamma assignment is deterministic
  - arm A evaluates the darkened images as-is
  - arm B applies CLAHE (clip 3.0, tiles 8x8 — the package defaults in
    ppe_detect.enhance) to the identical darkened images
  - both arms use the same split list and the same weights; per-class and
    overall deltas are reported whatever their sign
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

DARKEN_SEED = 42
GAMMA_RANGE = (2.0, 3.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_darken(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply I' = 255 * (I/255)^gamma via a lookup table (gamma > 1 darkens)."""
    lut = np.clip(np.round(255.0 * (np.arange(256) / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return cv2.LUT(image_bgr, lut)


def gamma_assignments(filenames: list[str], seed: int = DARKEN_SEED) -> dict[str, float]:
    """Deterministic per-file gamma: sorted order, fixed seed."""
    rng = np.random.default_rng(seed)
    return {name: float(rng.uniform(*GAMMA_RANGE)) for name in sorted(filenames)}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def rows_from_results(results) -> list[dict]:
    """Per-class + overall metric rows from an ultralytics DetMetrics-like object.

    Relies only on .names, .box.ap_class_index, .box.class_result(i) ->
    (p, r, ap50, ap) and .box.{mp,mr,map50,map}, so tests can pass a stub.
    """
    rows = []
    index_of = {int(c): pos for pos, c in enumerate(results.box.ap_class_index)}
    for class_id in sorted(results.names):
        if class_id not in index_of:
            continue
        p, r, ap50, ap = results.box.class_result(index_of[class_id])
        rows.append(
            {
                "name": results.names[class_id],
                "precision": float(p),
                "recall": float(r),
                "map50": float(ap50),
                "map50_95": float(ap),
            }
        )
    rows.append(
        {
            "name": "all",
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
        }
    )
    return rows


def metrics_markdown(rows: list[dict]) -> str:
    return markdown_table(
        ["Class", "Precision", "Recall", "mAP50", "mAP50-95"],
        [
            [
                row["name"],
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['map50']:.3f}",
                f"{row['map50_95']:.3f}",
            ]
            for row in rows
        ],
    )


def ab_markdown(dark_rows: list[dict], clahe_rows: list[dict]) -> str:
    """Side-by-side A/B table keyed by class name; deltas reported signed."""
    clahe_by_name = {row["name"]: row for row in clahe_rows}
    table_rows = []
    for dark in dark_rows:
        clahe = clahe_by_name[dark["name"]]
        table_rows.append(
            [
                dark["name"],
                f"{dark['map50']:.3f}",
                f"{clahe['map50']:.3f}",
                f"{clahe['map50'] - dark['map50']:+.3f}",
                f"{dark['map50_95']:.3f}",
                f"{clahe['map50_95']:.3f}",
                f"{clahe['map50_95'] - dark['map50_95']:+.3f}",
            ]
        )
    return markdown_table(
        [
            "Class",
            "mAP50 dark",
            "mAP50 dark+CLAHE",
            "Δ mAP50",
            "mAP50-95 dark",
            "mAP50-95 dark+CLAHE",
            "Δ mAP50-95",
        ],
        table_rows,
    )
