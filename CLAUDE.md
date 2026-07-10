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
| 2a | Dataset verified + train.py smoke-tested on MPS + full-run command documented | Micro-run (1 epoch, ~100 imgs) completes on MPS; reconciliation ≥98% | DONE 2026-07-10 |
| 2b | Full training executed by user outside session; then validate `best.pt`, sample detections, publish weights as GitHub release asset | `best.pt` produces PPE-class detections on sample images; release asset live | DONE 2026-07-10 |
| 3 | Eval harness + CLAHE A/B on darkened split | `results/metrics.md` with mAP50/mAP50-95/per-class from one command | DONE 2026-07-10 |
| 4 | FastAPI `/detect` demo | `curl -F image=@x.jpg :8000/detect` returns JSON + annotated image; endpoint test green | DONE 2026-07-10 |
| 5 | README + polish + resume bullets | DoD items 1–8 all pass from fresh clone | pending |
| 6 (opt) | Dockerfile + CI | `docker run` inference OK; GH Actions green | pending |

## Dataset record (SFCHD)

- Source: Google Drive `https://drive.google.com/file/d/1-2z7r3J4sZdLvVt5mllvSEwAFO49Y-zj/view` (link from the upstream SFCHD-SCALE README); downloaded 2026-07-10 via gdown 6.1.0.
- Archive: `SFCHD.zip`, 1.8 GB, sha256 `cbbb8aa20e556b9736b76a7260a3d86d292436c987b3c3b3dc61a3a5aa93824f` (kept locally at `data/downloads/`, gitignored).
- Contents: 12,372 images under `QY_final_dataset/images/`; reconciliation against the 12,372 reference label files: **100% match, zero orphans** (`scripts/prepare_data.py`).
- Splits (upstream `new_split_yolo`): train 9,897 / val 2,475, disjoint (80/20). The upstream `test.txt` is a 6-image subset of val — dropped as redundant by `prepare_data.py`; **val serves as the held-out eval split** (matches the upstream paper's protocol).
- Class schema: published 7 classes, unchanged — `person, helmet, self_clothes, safety_clothes, head, blur_head, blur_clothes`.

## Full training run (executed by user, outside session)

From the repo root (`caffeinate` keeps the Mac awake overnight):

```bash
caffeinate -dims .venv/bin/python scripts/train.py --epochs 25 --batch 16 --device mps --name sfchd-y8n
```

- Measured on the smoke run: ~1 it/s at batch 8, MPS memory ~2.2 GB → estimated
  15–25 min/epoch + ~1 min val pass, so 25 epochs ≈ 7–10 h. Checkpoints save every
  epoch; resume after interruption with:
  `.venv/bin/yolo train resume model=runs/sfchd-y8n/weights/last.pt`
- Output: `runs/sfchd-y8n/weights/best.pt` (+ `results.csv` per-epoch metrics).
- Expectation setting (upstream YOLOv8 curves on SFCHD): mAP50 ≈ 0.66 by epoch 15,
  0.78 took the authors 200 epochs. ~25 epochs should land ≈ 0.68–0.72 mAP50.

**T4 fallback** (if MPS throughput disappoints — free Kaggle/Colab GPU, ~3–4× faster):
1. Zip `data/sfchd/` (images, labels, train.txt, val.txt, sfchd.yaml) and upload as a
   private Kaggle dataset (or to Colab storage).
2. Rewrite the absolute paths for the cloud filesystem, e.g.
   `sed -i 's|/Users/abhi/dev/ppe-detect/data/sfchd|/kaggle/input/sfchd|g' train.txt val.txt sfchd.yaml`
3. `pip install ultralytics==8.4.91`, copy `scripts/train.py`, run with
   `--device 0 --epochs 50 --batch 16`, then download `best.pt` into `runs/sfchd-y8n/weights/`.

## Trained weights record

- Run: `runs/sfchd-y8n` — YOLOv8n, 25 epochs, imgsz 640, batch 16, MPS, seed 42,
  wall time ~5.8 h (user-executed 2026-07-10).
- Final-epoch validation (full 2,475-image val split): **mAP50 0.746, mAP50-95 0.473,
  precision 0.782, recall 0.716** (per-class table comes from the Phase 3 harness).
- Published: https://github.com/abhipabhi/ppe-detect/releases/tag/weights-v1
  asset `ppe-detect-y8n-sfchd.pt`, sha256 `ed20fd001d5dfef61c738928c799201d3323f2c5148e3d0dfa0c521263d28870`.
- Out-of-domain caveat: on non-CCTV photos (e.g. the CC0 sample) confidence drops and
  classes can shift toward `head`/`self_clothes` — SFCHD is chemical-plant CCTV footage;
  document under README limitations.

## Evaluation record (Phase 3)

- `scripts/evaluate.py` regenerates `results/metrics.md` end-to-end (standard eval +
  pre-registered low-light A/B); headline reproduced exactly (mAP50 0.746 / mAP50-95 0.473).
- Strong classes: person 0.961, safety_clothes 0.942, helmet 0.921 mAP50; weak: blur_head
  0.426, blur_clothes 0.468 (tiny, blurred instances — consistent with upstream findings).
- Low-light A/B (gamma 2.0–3.0 seed 42, pre-registered): darkening drops overall mAP50 to
  0.642; CLAHE recovers **+0.004 mAP50 overall (−0.001 mAP50-95)** — marginal; small gains
  on person/helmet/safety_clothes, losses on head/blur_head. Reported as-is; README must
  present CLAHE as a preprocessing *option* with near-neutral measured effect, not a win.

## License record

- This repo: MIT (`LICENSE`, © 2026 Abhi Patidar).
- SFCHD dataset / upstream SFCHD-SCALE repo (github.com/lijfrank-open/SFCHD-SCALE):
  **no explicit license anywhere** (repo has no LICENSE file/badge; the IEEE T-ASE paper
  says "dataset and code are publicly available", collection was de-identified; no
  research-only or non-commercial clause found). Consequences: we do not redistribute
  dataset images (already enforced); trained weights + aggregate metrics are shared with
  dataset credit in README and release notes. Flagged to user at Phase 3 STOP.

## API record (Phase 4)

- `ppe_detect.api:create_app` factory; run `uvicorn --factory ppe_detect.api:create_app`.
  Model loads once in the lifespan hook (`Detector.load()`), never per-request.
- Annotated image via `?output=image` query param on POST /detect (chosen over a second
  endpoint: one route, one upload path, content negotiated explicitly).
- Failure paths tested: non-image/corrupt/empty upload → 400 JSON, oversized → 413
  (cap `PPE_MAX_UPLOAD_MB`, default 10 MB), bad query values → 422. No 500s on bad input.
- `GET /healthz` returns weights filename + sha256 (verified live = release asset hash).
- `scripts/get_weights.py` downloads the release asset and verifies the recorded sha256
  (uses certifi CA bundle — macOS framework Pythons ship without root certs).
- Default test suite uses a stubbed detector; `pytest -m integration` covers real weights
  end-to-end (auto-skips if weights absent).
- License posture (user decision 2026-07-10): publish as-is; provenance note on release
  and README weights section — research/evaluation use, dataset uncredited-license caveat
  for commercial users. Code MIT.

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
