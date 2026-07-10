# Data provenance — SFCHD

Record of the dataset acquisition and reconciliation backing the published metrics.

## Acquisition (2026-07-10)

- Source: the dataset authors' public Google Drive link (from the upstream
  [SFCHD-SCALE README](https://github.com/lijfrank-open/SFCHD-SCALE)),
  file id `1-2z7r3J4sZdLvVt5mllvSEwAFO49Y-zj`, downloaded via gdown 6.1.0.
- Archive: `SFCHD.zip`, 1.8 GB,
  sha256 `cbbb8aa20e556b9736b76a7260a3d86d292436c987b3c3b3dc61a3a5aa93824f`.
- Extracted: 12,372 JPEG images from `QY_final_dataset/images/`.
- Dataset images are **not** stored in this repository (see README weights
  provenance section for terms).

## Reconciliation (`scripts/prepare_data.py` output, 2026-07-10)

```
labels: 12372 copied, 12372 total
--- reconciliation (all files) ---
matched: 12372
images without labels: 0
labels without images: 0
match rate: 100.0000%
--- per-split ---
train: 9897 listed, 9897 usable, 0 missing
val: 2475 listed, 2475 usable, 0 missing
test: 6 listed, 6 usable, 0 missing
test: subset of another split — dropped as redundant
```

- The upstream `test.txt` (6 images) is a subset of val and is dropped;
  **val (2,475 images, disjoint from train) is the held-out evaluation split**,
  matching the upstream paper's protocol.
- Split list integrity: `val.txt` sha256 is stamped into
  [`metrics.md`](metrics.md) on every evaluation run.
- Class schema: the published 7 SFCHD classes, unchanged.
