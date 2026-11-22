# DEPLOY.md — deploying vision-embed with Docker

Deploy scaffold only. Nothing here starts a container, downloads weights, or
contains secrets.

## Files

- `deploy/Dockerfile` — embed service image build
- `deploy/docker-compose.yml` — compose definition for the embed service
- `deploy/.gitignore` — keeps certs/models/secrets out of the repo
- `deploy/WEIGHTS.md` — where the model files for the six roles come from
  (embed always; the Florence-2 directory and the WD14 tagger files only
  when the describe/tag switches are enabled)
- `deploy/config.example.yaml` — embed config; copy and fill in, never commit the deployed copy

## Image and dependency pins

The embed image is built from a CUDA **runtime** base, pinned by amd64
manifest digest:

```
nvidia/cuda:12.6.3-runtime-ubuntu24.04
sha256:2c8193530ecc423e0f123d0c85b68a15d1395adcddabfc943e2523dbfde172e1
```

It is a runtime base: it carries the CUDA/cuDNN shared libraries but does not
require a GPU, an NVIDIA driver, or the nvidia container runtime to start.
The CPU execution provider works on this image under plain runc, so the same
image serves CPU-only and GPU deployments.

Pip pins installed into the image (verify against the current `Dockerfile`
before relying on this list):

- `onnxruntime-gpu==1.20.0`
- `nvidia-cudnn-cu12==9.26.0.51`
- `nvidia-cublas-cu12==12.9.3.13`
- app dependencies: `fastapi==0.115.6`, `uvicorn==0.34.0`,
  `python-multipart==0.0.20`, `numpy==2.1.3`, `pillow==11.0.0`,
  `pyyaml==6.0.2`, `jsonschema==4.23.0`, `prometheus_client==0.21.1`

The cudnn and cublas wheel library dirs go on `LD_LIBRARY_PATH` so the CUDA
execution provider resolves its libraries unambiguously.

## Config example (illustrative only)

The deployed `config.yaml` is an operator file, never committed. Relevant
keys from `deploy/config.example.yaml`:

```yaml
listen:
  host: "0.0.0.0"
  port: 8443
  tls:
    enabled: true
    cert_path: "/etc/vision-embed/tls/fullchain.pem"    # bind mount
    key_path: "/etc/vision-embed/tls/privkey.pem"       # bind mount
auth:
  bearer_token_path: "/etc/vision-embed/secrets/authz.bearer"   # bind mount
models:
  det_path: "/models/det_10g.onnx"
  face_embed_path: "/models/w600k_r50.onnx"
  scene_embed_path: "/models/openai-ViT-L-14-visual.onnx"
  object_det_path: "/models/yolov8n.onnx"
  # describe/tag roles (needed only when the routes.* switch below is true):
  florence_dir_path: "/models/florence2/"             # operator-supplied directory of weights + processor files
  tagger_onnx_path: "/models/wd14.onnx"
  tag_names_path: "/models/tag_names.txt"             # ordered tag-name list, one per line
  tag_suppress_path: "/models/tag_suppress.txt"       # tag names never returned on the wire
  det_score_min: 0.5
  object_score_min: 0.35
  max_faces: 20
  execution_provider: "CPUExecutionProvider"   # CPUExecutionProvider | CUDAExecutionProvider
limits:
  max_body_bytes: 33554432
  allowed_mime: ["image/jpeg", "image/png", "image/webp"]
metrics:
  enabled: true
  port: 9090
adapter_mode: "production"
```

Unknown config keys are fatal at startup. Choose one of the two execution
providers; the app preflights the provider and exits fatal if the configured
provider is not actually available — there is no silent CPU fallback.

## Optional describe and tag routes

`POST /v1/describe` (Florence-2) and `POST /v1/tag` (WD14 tagger) are
**optional** routes of the same process, gated by `routes.describe_enabled`
and `routes.tag_enabled`, both default `false`; the switch is read at
startup only, and changing it requires a restart. With both `false` the
default deployment is embed-only: Florence-2 is not loaded, its weight and
processor files are not required, the WD14 files are not required,
`models_loaded` is 4, and a POST to either route returns 404
`route.disabled`. A deployment that wants embed only needs neither the
Florence-2 directory nor the WD14 files. When a switch is `true`, the
fail-closed rule stands — a missing weight or data file is a fatal startup
error — and, under `CUDAExecutionProvider`, the matching
`routes.describe_min_free_bytes` / `routes.tag_min_free_bytes` key
(integer bytes, no default) is required; a route that fails the startup
free-memory gate is not loaded and behaves exactly as disabled
(`unavailable_reason: "gpu.insufficient"` in `/v1/runtime`), and the
process still binds.

## GPU reservation

The GPU device reservation in `docker-compose.yml` is **optional** and
commented out in the example file. Use it only if the operator is running a
CUDA/tolerance-study host (a host with a GPU reserved for the embed service):

```yaml
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: ["gpu"]
```

Remove the comments (or uncomment `gpus: all` and drop the deploy block) only
when actually running `CUDAExecutionProvider`. The image still runs CPU-only
without any GPU reservation. Note: the code can enable
`CUDAExecutionProvider`, but numeric equality with CPU is not certified, and
there is no acceptance envelope yet.

## Runtime contract enforced by compose (SPEC §3)

- Non-root `65532:65532` (`user:` + Dockerfile `USER`).
- `read_only: true` rootfs; writable only via a 128 MiB tmpfs at `/tmp`.
- All mounts read-only (`:ro`); bearer file, TLS key, and the model files of
  the enabled roles (for the six-role deployment: the four ONNX files, the
  Florence-2 directory, and the WD14 tagger's data files) are bind mounts
  never baked into the image and never env vars.
- TLS-only listener on `8443` inside the container (uvicorn
  `--ssl-certfile/--ssl-keyfile`); metrics on 9090, same process, one worker.
- No runtime model download: no fetch step in the image or entrypoint.
- No device requests in a CPU-only deployment.

## Operator steps (generic)

1. Copy `deploy/` files wherever the operator's compose tree lives.
2. Create `config/`, `models/`, `tls/`, `secrets/` directories locally. Put
   the four embed ONNX weights there — but get them from verified sources first, per
   `deploy/WEIGHTS.md`. Add the Florence-2 directory (a directory of weights
   plus processor files, not a single ONNX file) and the WD14 tagger's
   `wd14.onnx`, `tag_names.txt`, and `tag_suppress.txt` only when the
   corresponding route switch is enabled. Also put the TLS cert/key pair and bearer token file
   there (permissions `0400`, owner `65532` for `privkey.pem` and the bearer
   file).
3. Fill in `config.yaml` from the example; set the model paths and provider.
4. `docker compose build` and `docker compose up -d`.
5. Firewall the host ports 8443 and 9090; the service enforces nothing at L3
   and the listener is TLS-only.
6. Verify `/healthz` and `/readyz` before pointing consumers at it.

## Notes

- Image is pinned by digest; no rebuild-pull drift.
- `.gitignore` excludes `tls/`, `models/`, `secrets/`, and deployed
  `config.yaml` so weights, certs, and tokens never enter the repo.
- The index is standalone and not deployed here: the CLI ships as
  `vision-index.py` with no container, and no index image or index config
  belongs in this deployment.
