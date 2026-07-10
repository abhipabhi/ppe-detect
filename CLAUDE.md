# CLAUDE.md — ppe-detect

PPE detection (person / helmet / safety clothing) with a YOLOv8-family model.
Clean rebuild of an earlier notebook prototype (stock COCO weights, no trained PPE
model); everything in this repo is written fresh. The prior project folder at
`/Users/abhi/dev/PPE-detection/` is reference-only — used solely for the SFCHD
label files and dataset layout, never modified, and no code is ported from it.

## Project spec

- Detect PPE classes in images with a fine-tuned YOLOv8 model (SFCHD label schema:
  `person, helmet, self_clothes, safety_clothes, head, blur_head, blur_clothes`).
- CPU-runnable inference is the reproducibility baseline; training uses Apple-Silicon
  MPS (fallback: free Kaggle/Colab T4, bring `best.pt` back).
- Optional low-light preprocessing (CLAHE) behind an explicit flag, evaluated via A/B
  metrics on darkened test images.
- Every number published in the README must be reproduced by a script in this repo.

## Rules

- No SFCHD dataset images are ever committed; eval visuals from SFCHD stay local.
  Committed images (sample.jpg, README examples) must be license-clean (own photos
  or CC0), with source/license recorded in `assets/ATTRIBUTION.md`.
- Trained weights ship as a GitHub release asset, not Git LFS.
- Pinned dependencies only (`requirements.txt` exact versions).

## Definition of Done

1. Fresh clone → documented setup → CPU inference on a sample image succeeds.
2. Clean structure: `src/`, `scripts/`, `tests/`, `assets/` (no notebooks-as-source).
3. Pinned environment (`requirements.txt`, exact versions).
4. Eval script → metrics table (mAP50, mAP50-95, per-class) saved to file.
5. Tests for non-ML glue: config, pre/post-processing, IO.
6. README: what/architecture/setup/usage/results table/sample detections/limitations.
7. Demo: FastAPI `/detect` endpoint.
8. 3–4 resume bullets grounded only in what reproduces.

## Phase plan & status

| Phase | Scope | Exit criteria (binary) | Status |
|-------|-------|------------------------|--------|
| 1 | Scaffold + pinned env + stock-weights CPU inference | `python -m ppe_detect.cli assets/sample.jpg` writes annotated image on CPU from fresh clone | DONE 2026-07-10 |
| 2 | Dataset acquisition (SFCHD primary) + fine-tune YOLOv8n | `best.pt` produces PPE-class detections on sample images | pending |
| 3 | Eval harness + CLAHE A/B on darkened split | `results/metrics.md` with mAP50/mAP50-95/per-class from one command | pending |
| 4 | FastAPI `/detect` demo | `curl -F image=@x.jpg :8000/detect` returns JSON + annotated image; endpoint test green | pending |
| 5 | README + polish + resume bullets | DoD items 1–8 all pass from fresh clone | pending |
| 6 (opt) | Dockerfile + CI | `docker run` inference OK; GH Actions green | pending |

## Key decisions

- **2026-07-10** Repo: `/Users/abhi/dev/ppe-detect`. Demo: FastAPI (over Streamlit).
- **2026-07-10** Dataset: SFCHD via its public Google Drive mirror (Phase 2 first
  verifies the link; if dead or blocked >1 day, fall back to a small public PPE
  dataset, e.g. Roboflow hard-hat, without further approval).
- **2026-07-10** Phase 2 training sized to an overnight MPS budget; documented
  fallback to Kaggle/Colab T4 if MPS throughput is poor.
- Inference default device is CPU; `--device` opt-in for MPS/CUDA.
- **2026-07-10** Env: Python 3.11, `pip freeze` pins in `requirements.txt`
  (torch 2.13.0, ultralytics 8.4.91, opencv-python 5.0.0.93, numpy 2.4.6).
- **2026-07-10** Sample image: CC0 construction-site photo from Wikimedia Commons
  (see `assets/ATTRIBUTION.md`). Stock COCO weights detect `person` on it —
  pipeline proof only; PPE classes arrive with Phase 2 weights.
