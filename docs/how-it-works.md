# How vision-embed works

This document describes the code in this repository so a stranger to the
project can follow the architecture.

vision-embed is a stateless HTTP service. A caller sends one image, and the
service returns measurements: face boxes with 512-dim face embeddings, object
boxes with COCO-80 labels, and one 768-dim scene embedding. It returns
**vectors and boxes, nothing else** — no photos, no names, no cluster ids, no
state across requests or restarts, no image bytes held after the response.
Identity matching is the caller's job, done elsewhere.

That "elsewhere" is vision-index, implemented alongside it in
`vision_index/`. The index calls vision-embed over HTTPS for measurements,
keeps the returned vectors and cluster ids in a SQLite database, and records
a name only when an operator assigns one. It runs no models itself and stores
no image bytes. Two index operators do not share a people table.

## Request path

The HTTP surface is closed: `/healthz` (GET), `/readyz` (GET), and `/v1/embed`
(POST). Anything else is a 404, and other methods are a 405. Live docs pages
(`/docs`, `/redoc`, `/openapi.json`) are disabled and are not routes; no
published schema artifact exists.

`/healthz` always answers ok. `/readyz` answers 200 only after all four model
adapters have loaded, and 503 with a `not_ready` reason before that.

A `POST /v1/embed` runs a fixed order:

1. Size cap. A Content-Length above `limits.max_body_bytes` is rejected
   before the body is read, and streaming reads are capped again as they
   arrive.
2. Auth. The request must carry a bearer token that matches the token the
   operator supplies in a file, compared in constant time. Auth runs before
   the body is parsed.
3. Multipart parse and format sniff. The image part is found by field name,
   and its first bytes are sniffed against JPEG, PNG, and WebP magic. Unknown
   formats get a 415. Malformed multipart gets a 422.
4. Decode. The image part is decoded to RGB. Failures get a 422, not a 500.
5. Inference. Four model roles run against the image, described below.
6. Response validation. The body is checked against a closed JSON schema with
   `additionalProperties: false` on every object before it is sent.
7. Audit. Exactly one JSON line per HTTP request goes to stdout. It carries
   the request id, the image sha256, byte count, face and object counts, and
   the outcome. Image bytes, embeddings, and tokens are never logged.

Every response, success or error, carries an `X-Request-Id` header. The
service honors a caller's request id header if it is sane and short, otherwise
it generates a uuid. Errors use a closed set of reason tokens such as
`auth.invalid_token`, `format.unsupported`, and `decode.failed`.

## The four model roles

All four roles are defined as Protocols in `vision_embed/adapters/__init__.py`,
so production and fake implementations are interchangeable. All four models
are operator-supplied ONNX files at config paths. Nothing is downloaded at
runtime. In production mode a missing weight file is a fatal startup error,
never a 200 with empty arrays.

- Face detector: SCRFD-10G (`det_10g.onnx`, from the buffalo_l pack).
  Letterboxes to 640x640, decodes the per-stride SCRFD outputs, and returns
  boxes plus the 5 facial keypoints. The other three files in the buffalo_l
  pack are not used.
- Face embedder: ArcFace `w600k_r50.onnx`. Faces are aligned with the
  standard 5-point similarity warp to 112x112, normalized, and embedded to a
  512-dim L2-normalized vector.
- Scene embedder: CLIP ViT-L/14 visual tower (OpenAI pretrained). Bicubic
  resize, 224 center crop, CLIP normalization, 768-dim L2-normalized vector.
  One 768-dim vector is returned for the whole image, and optionally one per
  object when `include_object_vectors=true`.
- Object detector: YOLOv8n ONNX. Letterboxed 640x640, per-class max
  confidence, class-agnostic NMS at IoU 0.45, COCO-80 labels. Objects below
  `object_score_min` (default 0.35) are dropped, at most 30 per image.

Face NMS runs tighter than the object path: **class-agnostic NMS at IoU 0.40**
(`vision_embed/pipeline.py`), then faces above `det_score_min` (default 0.5)
are capped at `max_faces` (default 20), highest score first.

## Response shape

A success response has `request_id`, `image` (sha256, width, height), `scene`
(model, dim, vector), `faces`, and `objects`. Faces carry `bbox`, `det_score`,
`model`, `dim`, `vector`. Objects carry `label`, `bbox`, `score` and, when
asked for, the same `model`/`dim`/`vector` triplet. The face model string is
`buffalo_l/w600k_r50`, the scene model string is `ViT-L-14/openai`. No
identity fields exist anywhere in the payload: no cluster ids, no names, no
same-person judgments.

A repeated POST of the same bytes is just another inference. The logged
sha256 is for the caller's bookkeeping only.

## The index side

vision-index keeps state. Its contract (`docs/spec/vision-index-SPEC.md`)
covers clustering new embeddings against stored ones, cluster merge and
split operations that the operator triggers, name assignment, and export. A
repeated image is classified by cosine distance against existing centroids,
not re-registered as a new person.

The clustering threshold default in the spec is `face_match_max: 0.40`.
Because same-person cosine distances and cross-person distances overlap less
at tighter margins, a deployment may set a **tighter live threshold** in its
own config file; the spec number stays 0.40 and tighter operation is a
deployment configuration, not a contract change.

## Config and fail-closed behavior

Config is a YAML file with a closed key set. Unknown keys are fatal at
startup, as are invalid values. Config never holds token values: the bearer
token is read once at startup from a path, and the operator supplies a TLS
cert/key pair and the token file.

Two modes:

- `adapter_mode: production` requires TLS to be enabled with a readable cert
  and key pair that OpenSSL can load. It also requires all four weight files
  to exist. Production refuses anything less and exits with one JSON fatal
  line on stdout.
- `adapter_mode: test` with `models.adapter: fake` starts with zero ONNX
  files. The fake adapter set is deterministic, seeded from image content,
  and exercises the full request, response, and audit path without weights.

### Execution providers: CPU and CUDA

The execution provider is config-selected. `CPUExecutionProvider` requires
nothing special. `CUDAExecutionProvider` is optional and **fail-closed**:

- At startup, `provider_preflight` checks the provider against the
  allowlist and verifies ONNX Runtime actually exposes it. If CUDA is
  configured but the provider (or its cuDNN/CUDA libraries) is missing, the
  process exits with a single fatal line — it never silently falls back to
  CPU.
- CUDA does **not** equal CPU numerically. The vectors are not claimed
  bitwise equal in either direction; the provider choice is covered by the
  CUDA spec (`docs/spec/vision-embed-cuda-SPEC.md`) and its tolerance study.

A measured CPU-vs-CUDA tolerance study was run over the real models on the
same inputs (identical preprocessing either way, compared after
L2-normalization and on raw detector geometry):

| Vector family | max abs error (element-wise) | post-normalize cosine similarity |
| --- | --- | --- |
| Face, 512-d ArcFace | about 4.95e-04 | 0.9999945 |
| Scene, 768-d CLIP | about 7.72e-05 | — |

The study is a measurement only — the numbers above are recorded study
results, not an acceptance gate, and no numeric acceptance envelope exists
yet. The CUDA spec (lock 1) states that no numeric tolerance is pinned until
an amendment accepts one from a future study output. Neither provider is
claimed equal to the other beyond these measured numbers.

Session creation is isolated in one function (`create_session` in
`adapters/onnx.py`) so provider changes stay a config matter plus that single
call site.

## Observability

Metrics are plain Prometheus on a separate port (default 9090): request and
error counters by reason token, request latency, and a `models_loaded` gauge.
No labels carry image-derived values. The audit stream is the one JSON line
per request described above, including healthz, readyz, and closed-surface
404/405 responses.

## Tests

The pytest suite under `tests/` (and `tests/index/` for the index service)
covers the HTTP contracts for both services, config fail-closed rules, the
NMS unit, provider rules for both CPU and CUDA configuration, observability
(audit and metrics), and the container contracts. The embed tests run against
the fake adapter, so the ONNX adapters in `adapters/onnx.py` are exercised by
the production run itself rather than by this suite; container tests need
Docker tooling and are skipped where it is absent.

## Deployment

`deploy/` holds a generic Docker build (`DEPLOY.md`) for either service.
Weights, TLS material, and token files are operator-supplied bind mounts,
never baked into the images, never committed.
