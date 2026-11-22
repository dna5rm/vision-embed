# How vision-embed works

This document describes the code in this repository so a stranger to the
project can follow the architecture.

vision-embed is a stateless HTTP service. A caller sends one image, and the
service returns measurements: face boxes with 512-dim face embeddings, object
boxes with COCO-80 labels, and one 768-dim scene embedding; two optional
routes add a Florence-2 caption or region description and a WD14 tag list
with scores. It returns
**vectors and boxes, nothing else** — no photos, no names, no cluster ids, no
state across requests or restarts, no image bytes held after the response.
Identity matching is the caller's job, done elsewhere.

That "elsewhere" is vision-index, shipped as the standalone CLI
`vision-index.py` (installed at `/usr/local/bin/vision-index`). The CLI
calls vision-embed over HTTPS for measurements and keeps the returned
vectors in a local SQLite database, recording a name only when an operator
assigns one. It runs no models itself and stores no image bytes. The
default database path is per-user
(`~/.local/share/vision-index/index.sqlite3`), so two accounts on the same
host do not share a people table. Its contract is
`docs/spec/vision-index-cli-SPEC.md`.

## Request path

The HTTP surface is closed: `/healthz` (GET), `/readyz` (GET), `/v1/embed`
(POST), `/v1/describe` (POST), `/v1/tag` (POST), and `/v1/runtime` (GET).
Anything else is a 404, and other methods are a 405. Live docs pages
(`/docs`, `/redoc`, `/openapi.json`) are disabled and are not routes; no
published schema artifact exists.

`/healthz` always answers ok. `/readyz` answers 200 only after every model
role this process loads has loaded, and 503 with a `not_ready` reason before
that. `models_loaded` is the count of roles **actually loaded** — with both
route switches off (the default) that is the four embed roles, so it is 4,
5, or 6 by switch state, never a hardcoded 6.

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

### The /v1/describe and /v1/tag request paths

`POST /v1/describe` and `POST /v1/tag` are multipart, exactly like
`POST /v1/embed`: the image field is named `image`, auth runs before the
body is parsed, and the same `limits.max_body_bytes` cap applies. `describe`
takes one further field, `task`, whose value must be one of the eight closed
Florence-2 wire tasks (`<CAPTION>`, `<DETAILED_CAPTION>`,
`<MORE_DETAILED_CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`,
`<REGION_PROPOSAL>`, `<OCR>`, `<OCR_WITH_REGION>`); anything else is 422
`task.unsupported`. It returns the model (`florence2`), the echoed `task`,
and either generated `text` or a `regions` array. `tag` takes one further
field, `tag_threshold`; a bare `threshold` field is ignored and yields 422
`decode.failed` with message `missing tag_threshold field`. It returns the
model (`wd14`), `threshold_used`, and `tags` as a list of `tag` plus
`score`.

### /v1/runtime

`GET /v1/runtime` requires the same bearer token as the three POST routes.
It is an operator inventory read: `uptime_seconds`, the configured
`execution_provider`, a `routes` object with the three route flags, a
`models` array — the six roles in fixed order (face, face_detector, scene,
object, florence2, tagger), each with `role`, `model_id`, `loaded`, and
`unavailable_reason` — and a closed `gpu` object (`available`,
`device_count`, `name`, `memory_total_bytes`, `memory_used_bytes`, read via
the NVML library for device index 0). It loads no model and does not affect
readiness; a failed GPU read is reported through the `gpu` object, not as
an HTTP error.

### GPU admission lock

When `execution_provider` is `CUDAExecutionProvider`, `POST /v1/embed`,
`POST /v1/describe`, and `POST /v1/tag` share one in-process lock: a
request that arrives while one GPU inference is running is rejected 429
`gpu.busy` and does not wait — there is no queue; the caller retries. The
lock governs this process only; other processes on the same GPU are not
consulted. `/healthz`, `/readyz`, and `/v1/runtime` do not take the lock.
On `CPUExecutionProvider` there is no lock and no `gpu.busy`.

Every response, success or error, carries an `X-Request-Id` header. The
service honors a caller's request id header if it is sane and short, otherwise
it generates a uuid. Errors use a closed set of reason tokens; the
process-wide closed set is exactly twelve tokens — the nine embed-scope
tokens such as `auth.invalid_token`, `format.unsupported`, and
`decode.failed`, plus `task.unsupported` (422), `route.disabled` (404), and
`gpu.busy` (429).

## The six model roles

The four embed roles are defined as Protocols in `vision_embed/adapters/__init__.py`,
so production and fake implementations are interchangeable; the Florence-2
and tagger roles live behind their own adapters and are loaded only when
their startup switches are on. The four embed models are operator-supplied
ONNX files at config paths, Florence-2 is an operator-supplied local
directory of weights plus processor files, and the tagger is an
operator-supplied ONNX file plus two data files. Nothing is downloaded at
runtime. In production mode a missing file for an enabled role is a fatal
startup error, never a 200 with empty arrays.

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
- Florence-2 (describe, gated by `routes.describe_enabled`): loaded by
  transformers from an operator-supplied local directory of weights plus
  processor files (`models.florence_dir_path`), not from a single ONNX
  file. It serves the closed set of Florence-2 wire tasks via
  `POST /v1/describe`: caption tasks return generated `text`, and
  region tasks such as `<OD>` return a `regions` array. It is generative:
  on an object-free input, regions can be empty or low confidence.
- WD14 tagger (tag, gated by `routes.tag_enabled`): an operator-supplied
  ONNX tagger (`models.tagger_onnx_path`) served via `POST /v1/tag`. Tag
  scores map to names through an operator-supplied ordered tag-name list
  (`models.tag_names_path`, one tag name per line, in output-column order,
  length equal to the score vector length — a mismatch is a fatal startup
  error), and names listed in `models.tag_suppress_path` are never returned
  on the wire. Tags above `tag_threshold` are returned as an ordered list
  of tag plus score.

Both describe roles are loaded only when their startup switches are `true`;
with a switch `false` the role is not loaded, its files are not required,
it is not counted in `models_loaded`, and the route returns 404
`route.disabled`.

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

vision-index is not a service. It is `vision-index.py`, a single-file,
standard-library-only Python CLI at the repo root, and all durable identity
state lives in one local SQLite database it owns (default
`~/.local/share/vision-index/index.sqlite3`, per user). The vision-embed
service it calls stores nothing; the CLI's database is the only place
faces, vectors, clusters, and names exist. The full contract is
`docs/spec/vision-index-cli-SPEC.md`.

The CLI never talks to the service to keep state — it POSTs each image once
to the embed service (route `POST /v1/embed`) and stores the returned
vectors itself. Naming is explicit-only: `enroll` puts faces into unnamed
clusters, and a cluster becomes evidence of a person only when the operator
runs `link <cluster-id> <person>` (creating the person on first use) or the
bulk form `link --near <person> <dist>`. A repeated image is matched
by cosine distance against stored cluster members — identify is read-only
and never writes.

The face match threshold is a hardcoded module constant,
`FACE_MATCH_MAX = 0.28`; there is no environment, config, or command-line
override.

### Known limit: identification reliability depends on enrolled evidence volume (DEC-1)

`identify` matches a probe against the **nearest face across ALL of a
person's enrolled faces** (`d = min(cosine_distance(vec, pv) for pv in
vecs)`), not against a centroid or any single reference photo. Because the
minimum over N enrolled vectors can only decrease as N grows, the distance
that faces the threshold is partly a function of how much evidence is
enrolled for that person, not only of who the probe is.

This is a real, measured effect, observed live on a temp DB with an unseen
probe photo of a known person (`alice`), with the same probe photo and the
same corpus in both states:

| Enrolled faces of the person | Band       | `nearest_named_distance` |
|------------------------------|------------|--------------------------|
| 1                            | `probable` | 0.3871                   |
| 4                            | `matched`  | 0.1754                   |

Everything was identical in both measurements except the amount of enrolled
evidence. With `FACE_MATCH_MAX = 0.28`, the person enrolled from one photo
is below the matched band even though the probe is genuinely them.

Consequences, which apply on purpose and must be worked with, not
engineered around:

- Identification reliability depends on enrolled evidence volume. A person
  enrolled from only 1–2 photos is effectively unrecognisable by the
  tool even for a true match.
- Practical recommendation: enrol at least ~5 photos of a person before
  relying on `identify`. Below that, expect `probable` rather than
  `matched`.
- `status` surfaces this as an advisory text line for any named person with
  fewer than 5 enrolled faces, e.g.
  `warning: person 'alice' has 3 face(s) enrolled; identification needs ~5+ to reach 'matched' reliably`
  (status is read-only and adds nothing to the DB; the advisory is a
  text-mode line only — the 13.1 `status --json` shape is closed and gained
  no field).
- This behaviour is deliberate under **DEC-1** (false unknown is preferred
  over false match; see section 3 of the spec). It is NOT a bug. Do not
  lower `FACE_MATCH_MAX` to "catch" thin-evidence people: lower thresholds
  are exactly what starts matching strangers, the failure DEC-1 prevents.
  Threshold changes go through re-measurement and a new DEC (spec O-3),
  never an edit.

The earlier design — an HTTP service package with clustering
against centroids, an operator-triggered consolidation step, and export — was
abandoned in favour of this CLI.

## Config and fail-closed behavior

Config is a YAML file with a closed key set. Unknown keys are fatal at
startup, as are invalid values. Two startup switches under `routes:` —
`describe_enabled` and `tag_enabled`, both default `false` — decide whether
Florence-2 and the WD14 tagger are loaded at all; the switch is read at
startup only, and changing it requires a restart. When a switch is `true`
and the provider is `CUDAExecutionProvider`, its matching
`routes.describe_min_free_bytes` / `routes.tag_min_free_bytes` key
(integer bytes, no default) is required; a route that fails the startup
free-memory gate is not loaded and behaves exactly as disabled, with
`unavailable_reason: "gpu.insufficient"` reported on `/v1/runtime`, and the
process never exits for this gate. Config never holds token values: the bearer
token is read once at startup from a path, and the operator supplies a TLS
cert/key pair and the token file.

Two modes:

- `adapter_mode: production` requires TLS to be enabled with a readable cert
  and key pair that OpenSSL can load. It also requires the weight and data
  files of every enabled role to exist (the four embed ONNX files always;
  the Florence-2 directory and the WD14 tagger files only when the
  corresponding route switch is `true`). Production refuses anything less and
  exits with one JSON fatal line on stdout.
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

Metrics are plain Prometheus on a separate port (default 9090); the port is
unauthenticated. The core series are eight: four embed-scope
(`vision_embed_requests_total`, `vision_embed_latency_seconds`,
`vision_embed_errors_total`, and the `vision_embed_models_loaded` gauge)
plus, when at least one describe/tag route is enabled, four describe-scope
series (`vision_describe_requests_total`, `vision_describe_latency_seconds`,
`vision_describe_errors_total`, and the `vision_describe_models_loaded`
gauge, which counts the loaded describe/tag roles). Series for a disabled
route are absent, not stuck at zero. No labels carry image-derived values.
The audit stream is the one JSON line
per request described above, including healthz, readyz, and closed-surface
404/405 responses.

## Tests

The pytest suite under `tests/` covers the HTTP contracts of the
vision-embed service, config fail-closed rules, the NMS unit, provider
rules for both CPU and CUDA configuration, observability (audit and
metrics), and the container contracts. `tests/test_vision_index_cli.py`
is the black-box acceptance suite for the `vision-index.py` CLI (see its
docstring and `docs/spec/vision-index-cli-SPEC.md`). The embed tests run
against the fake adapter, so the ONNX adapters in `adapters/onnx.py` are
exercised by the production run itself rather than by this suite;
container tests need Docker tooling and are skipped where it is absent.

## Deployment

`deploy/` holds a generic Docker build (`DEPLOY.md`) for either service.
Weights, TLS material, and token files are operator-supplied bind mounts,
never baked into the images, never committed.
