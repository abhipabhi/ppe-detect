"""FastAPI demo service.

Run with the app factory (loads the model once at startup via the lifespan hook):

  uvicorn --factory ppe_detect.api:create_app --port 8000

Configuration via environment variables:
  PPE_WEIGHTS        weights path (default: weights/ppe-detect-y8n-sfchd.pt;
                     fetch it first with scripts/get_weights.py)
  PPE_MAX_UPLOAD_MB  upload size cap in megabytes (default: 10)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from . import __version__
from .annotate import draw_detections
from .config import DetectConfig
from .detector import Detector
from .enhance import maybe_enhance
from .evaluation import sha256_file

DEFAULT_WEIGHTS = "weights/ppe-detect-y8n-sfchd.pt"


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def create_app(config: DetectConfig | None = None, detector_factory=Detector) -> FastAPI:
    """App factory. Tests inject a stub via `config` + `detector_factory`."""
    if config is None:
        weights = os.environ.get("PPE_WEIGHTS", DEFAULT_WEIGHTS)
        try:
            config = DetectConfig(weights=weights)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc} — fetch the released weights with: python scripts/get_weights.py"
            ) from exc
    max_upload_bytes = int(float(os.environ.get("PPE_MAX_UPLOAD_MB", "10")) * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        detector = detector_factory(config)
        load = getattr(detector, "load", None)
        if callable(load):
            load()  # model weights load once here, never per-request
        app.state.detector = detector
        weights_path = Path(config.weights)
        app.state.weights_name = weights_path.name
        app.state.weights_sha256 = sha256_file(weights_path) if weights_path.exists() else None
        yield

    app = FastAPI(title="ppe-detect", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "model": app.state.weights_name,
            "weights_sha256": app.state.weights_sha256,
            "version": __version__,
        }

    @app.post("/detect")
    def detect(
        image: UploadFile = File(...),
        enhance: Literal["auto", "on", "off"] = Query("off"),
        output: Literal["json", "image"] = Query(
            "json", description="'image' returns the annotated JPEG instead of JSON"
        ),
    ):
        data = image.file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            return _error(413, f"payload exceeds {max_upload_bytes // (1024 * 1024)} MB limit")
        if not data:
            return _error(400, "empty upload")

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return _error(400, "could not decode upload as an image")

        frame, enhanced = maybe_enhance(frame, enhance, config.low_light_threshold)
        started = time.perf_counter()
        detections = app.state.detector.detect(frame)
        inference_ms = (time.perf_counter() - started) * 1000

        if output == "image":
            ok, encoded = cv2.imencode(".jpg", draw_detections(frame, detections))
            if not ok:
                return _error(500, "failed to encode annotated image")
            return Response(content=encoded.tobytes(), media_type="image/jpeg")

        return {
            "boxes": [list(det.to_dict()["box_xyxy"]) for det in detections],
            "classes": [det.class_name for det in detections],
            "confidences": [round(det.confidence, 4) for det in detections],
            "enhanced": enhanced,
            "timing": {"inference_ms": round(inference_ms, 1)},
            "image": {"width": frame.shape[1], "height": frame.shape[0]},
        }

    return app
