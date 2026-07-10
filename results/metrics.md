# Evaluation results

## Provenance

- date: 2026-07-10
- command: `python scripts/evaluate.py --weights /Users/abhi/dev/ppe-detect/runs/sfchd-y8n/weights/best.pt --device mps --batch 16`
- weights: `best.pt` sha256 `ed20fd001d5dfef61c738928c799201d3323f2c5148e3d0dfa0c521263d28870`
- split: SFCHD val, 2475 images, list sha256 `4994360eb4ee608c51bbe5f433731dc7e332dbf2d6ef5c0be3155c6b0162c17a`
- low-light protocol: per-image gamma in (2.0, 3.0), seed 42, sorted-filename assignment; CLAHE clip 3.0, tiles 8x8 (pre-registered)
- versions: python 3.11.1, ppe-detect 0.1.0, ultralytics 8.4.91, torch 2.13.0, opencv 5.0.0, numpy 2.4.6

## Standard evaluation — SFCHD val split

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| person | 0.924 | 0.940 | 0.961 | 0.701 |
| helmet | 0.898 | 0.898 | 0.921 | 0.596 |
| self_clothes | 0.681 | 0.703 | 0.768 | 0.540 |
| safety_clothes | 0.885 | 0.927 | 0.942 | 0.622 |
| head | 0.782 | 0.738 | 0.736 | 0.376 |
| blur_head | 0.720 | 0.388 | 0.426 | 0.200 |
| blur_clothes | 0.653 | 0.397 | 0.468 | 0.279 |
| all | 0.792 | 0.713 | 0.746 | 0.473 |

## Low-light A/B — synthetically darkened val split (pre-registered)

Same weights, same split, deterministic darkening (see Provenance). Arm A: darkened images. Arm B: CLAHE applied to the identical darkened images.

| Class | mAP50 dark | mAP50 dark+CLAHE | Δ mAP50 | mAP50-95 dark | mAP50-95 dark+CLAHE | Δ mAP50-95 |
|---|---|---|---|---|---|---|
| person | 0.911 | 0.923 | +0.013 | 0.633 | 0.646 | +0.013 |
| helmet | 0.844 | 0.853 | +0.009 | 0.509 | 0.505 | -0.004 |
| self_clothes | 0.626 | 0.637 | +0.011 | 0.443 | 0.437 | -0.006 |
| safety_clothes | 0.887 | 0.896 | +0.009 | 0.549 | 0.553 | +0.004 |
| head | 0.603 | 0.585 | -0.018 | 0.305 | 0.294 | -0.011 |
| blur_head | 0.318 | 0.291 | -0.027 | 0.153 | 0.129 | -0.024 |
| blur_clothes | 0.308 | 0.342 | +0.034 | 0.181 | 0.205 | +0.024 |
| all | 0.642 | 0.647 | +0.004 | 0.396 | 0.395 | -0.001 |
