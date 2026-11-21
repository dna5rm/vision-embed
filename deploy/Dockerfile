# syntax=docker/dockerfile:1
# vision-embed v1 — CUDA *runtime* base, pinned by digest.
# The GPU wheel (onnxruntime-gpu, CUDA 12 build per upstream compatibility
# table) keeps a working CPUExecutionProvider that loads and runs without a
# visible GPU or the nvidia container runtime, so the unchanged CPU default
# keeps this image valid under plain runc.
#
# cuDNN supply (open item 1, decided): the onnxruntime-gpu 1.20.x PyPI wheel
# requires cuDNN 9.x (upstream CUDA EP requirements table). Copied from the
# official NVIDIA pip wheel nvidia-cudnn-cu12 (9.x, verified on PyPI) onto
# LD_LIBRARY_PATH; CUDA runtime libs (libcudart, cuBLAS, etc.) already ship in
# the 12.6.3 runtime base. A missing cuDNN library surfaces as a startup
# fatal (config.invalid), never silent CPU mode.

# Pinned by amd64 manifest digest (resolved 2026-11-20 via registry manifest API).
FROM nvidia/cuda@sha256:2c8193530ecc423e0f123d0c85b68a15d1395adcddabfc943e2523dbfde172e1
# tag referenced: nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ubuntu 24.04 ships python3.12 as default python3 (3.12 exact minor per SPEC §2).
# No pip package here fetches model weights at build or runtime; weights are
# bind-mounted read-only by compose, never baked in.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu 1.20.0 pulls nvidia-cudnn-cu12's own cublas dep wheel; both
# lib dirs go on the loader path so the CUDA EP resolves cuDNN 9 unambiguously.
# cublas pinned to 12.9.3.13 — the exact version this
# image was verified against, so no rebuild needed on redeploy.
RUN python3.12 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir \
      "onnxruntime-gpu==1.20.0" \
      "nvidia-cudnn-cu12==9.26.0.51" \
      "nvidia-cublas-cu12==12.9.3.13" \
      "fastapi==0.115.6" \
      "uvicorn==0.34.0" \
      "python-multipart==0.0.20" \
      "numpy==2.1.3" \
      "pillow==11.0.0" \
      "pyyaml==6.0.2" \
      "jsonschema==4.23.0" \
      "prometheus_client==0.21.1"

# cuDNN 9 (+ its cublas wheel dep) from the nvidia pip wheels, on the loader
# path for the CUDA EP.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib

ENV PATH="/opt/venv/bin:${PATH}"

# Non-root uid 65532 (SPEC §3).
RUN groupadd -g 65532 app && useradd -m -u 65532 -g 65532 -s /usr/sbin/nologin app

# App code is baked from the real package (no `app` shim exists in this repo).
# Config, TLS certs, bearer token and model files are operator mounts, never baked.
COPY --chown=65532:65532 vision_embed/ /opt/app/vision_embed/

WORKDIR /opt/app
USER 65532:65532

# Read-only rootfs is enforced by compose `read_only: true`; /tmp is a tmpfs.
EXPOSE 8443 9090

# One process. Listen address/port and TLS paths come from the mounted config
# (SPEC §4); nothing is hardcoded here. Unknown keys / missing model files are
# fatal at startup (SPEC INV-4) before uvicorn binds.
ENTRYPOINT ["/opt/venv/bin/python", "-m", "vision_embed", "--config", "/opt/app/config.yaml"]
