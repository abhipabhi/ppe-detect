# ppe-detect

Detect personal protective equipment (person / helmet / safety clothing) in images
with a YOLOv8 model, with optional CLAHE low-light preprocessing. CPU inference is
the supported baseline; no GPU required.

> Status: work in progress. Phase 1 ships stock-weights inference; a PPE fine-tuned
> model, evaluation results, and a FastAPI demo land in later phases.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/get_weights.py   # fetch released PPE weights (sha256-verified)
```

## Usage

```bash
# annotate an image (writes assets/sample_detections.jpg)
ppe-detect assets/sample.jpg

# equivalent module form
python -m ppe_detect.cli assets/sample.jpg

# options
ppe-detect image.jpg -o out.jpg --json out.json --conf 0.4 --enhance auto --device cpu
```

`--enhance auto` applies CLAHE contrast enhancement only when the image's mean
brightness falls below a threshold; `on`/`off` force the behavior.

## Trained PPE weights

YOLOv8n fine-tuned on the SFCHD dataset (7 PPE classes: person, helmet, self_clothes,
safety_clothes, head, blur_head, blur_clothes) is published as a release asset:

```bash
curl -L -o ppe-detect-y8n-sfchd.pt \
  https://github.com/abhipabhi/ppe-detect/releases/download/weights-v1/ppe-detect-y8n-sfchd.pt
ppe-detect image.jpg --weights ppe-detect-y8n-sfchd.pt
```

Validation on the 2,475-image held-out split: mAP50 0.746, mAP50-95 0.473
(25 epochs, imgsz 640; per-class table in [results/metrics.md](results/metrics.md)).

**Weights provenance:** trained on the SFCHD dataset (Yu, Li, et al., IEEE T-ASE;
see Dataset credit below). The dataset carries no explicit license; the weights are
shared for research and evaluation with credit to the dataset authors — commercial
users should verify dataset rights independently. Code in this repo is MIT.

Without `--weights`, the CLI defaults to stock COCO `yolov8n.pt` (auto-downloaded
by ultralytics on first run), which detects generic classes such as `person` —
useful to verify the pipeline end to end.

## API

Serve detections over HTTP (model loads once at startup):

```bash
uvicorn --factory ppe_detect.api:create_app --port 8000
```

```bash
# JSON detections
curl -F image=@assets/sample.jpg "http://127.0.0.1:8000/detect?enhance=auto"
# → {"boxes": [...], "classes": [...], "confidences": [...], "enhanced": false,
#    "timing": {"inference_ms": ...}, "image": {"width": ..., "height": ...}}

# annotated JPEG instead of JSON
curl -F image=@assets/sample.jpg "http://127.0.0.1:8000/detect?output=image" -o out.jpg

# liveness + which weights are being served
curl http://127.0.0.1:8000/healthz
```

Query parameters: `enhance=auto|on|off` (CLAHE preprocessing, same semantics as the
CLI), `output=json|image`. Bad inputs return JSON errors: non-image/corrupt upload
→ 400, payload over the cap (`PPE_MAX_UPLOAD_MB`, default 10 MB) → 413. Weights
path is configurable via `PPE_WEIGHTS`.

## Dataset credit

Trained and evaluated on the **SFCHD** dataset (Safety Clothing and Helmet Detection;
12,372 chemical-plant CCTV images, 7 classes) published by Yu, Li, et al. (HUST),
*"Large, Complex, and Realistic Safety Clothing and Helmet Detection: Dataset and
Method"* — [github.com/lijfrank-open/SFCHD-SCALE](https://github.com/lijfrank-open/SFCHD-SCALE).
The dataset's authors state it is publicly available; it carries no explicit license,
so no dataset images are redistributed in this repository — only trained weights and
aggregate metrics.

## Tests

```bash
pip install pytest
pytest                    # fast glue tests (no model load)
pytest -m integration     # full-stack CPU smoke test (downloads weights)
```
