# syntax=docker/dockerfile:1
# Multi-stage build: deps + verified weights in the builder, slim runtime image.
# python:3.11-slim pinned by multi-arch index digest (2026-07-11).

FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

# CPU torch wheels explicitly: the default index would pull CUDA-variant deps
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
       torch==2.13.0 torchvision==0.28.0 \
    && /opt/venv/bin/pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN /opt/venv/bin/pip install .

# weights fetched at build time — the sha256 check in get_weights.py gates the image
COPY scripts/get_weights.py scripts/get_weights.py
RUN /opt/venv/bin/python scripts/get_weights.py \
    && mv weights /opt/weights


FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

# opencv-python runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/weights /app/weights

ENV PATH="/opt/venv/bin:$PATH" \
    PPE_WEIGHTS=/app/weights/ppe-detect-y8n-sfchd.pt
WORKDIR /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -sf http://127.0.0.1:8000/healthz || exit 1
CMD ["uvicorn", "--factory", "ppe_detect.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
