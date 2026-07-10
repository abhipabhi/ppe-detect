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

Until the fine-tuned PPE weights ship (Phase 2), the CLI defaults to stock COCO
`yolov8n.pt` (auto-downloaded by ultralytics on first run), which detects generic
classes such as `person` — useful to verify the pipeline end to end.

## Tests

```bash
pip install pytest
pytest                    # fast glue tests (no model load)
pytest -m integration     # full-stack CPU smoke test (downloads weights)
```
