# vision-embed — Build Contract SPEC

**Status: DRAFT v4**
Changelog: v2 closes critic failures F6 F7 F9 F11 F25 F26 F29 F30.
Changelog: v3 closes critic failures F7 and F25.
Changelog: v4 scopes the F2 and F23 register lines so they match fake+test mode.
**Product owner: the service owner (me).** This document is the complete build contract. An implementer builds exactly what is specified here. Every behavior not in this contract has an explicit "do not invent" instruction; if a needed detail is absent, the implementer MUST stop and ask, not fill the gap.

---

## 1. Status, owner, invariants

- Status: **DRAFT v4**. This is v4 scope. Nothing here is negotiable; reopen decisions only via section 15.
- Owner: the service owner (me) owns this contract, the config format, the reason-token set, and the audit schema.

Numbered invariants — each is an acceptance gate, not a suggestion:

1. **INV-1 Stateless.** The server stores nothing. No photos, no vectors, no names, no cluster ids, no database of people, no cache across requests, no state across restarts.
2. **INV-2 No identity judgments.** The response MUST NOT contain a cluster id, a person name, or any same-person/same-previous-call relationship. Matching against a caller's own index is the caller's job. The server has no index.
3. **INV-3 Determinism.** Same image bytes + same pinned model files ⇒ same vectors. v1 requires `CPUExecutionProvider`; CUDA is config-off until a documented numeric tolerance test exists (section 2). Do not enable a GPU provider in v1. The provider configuration key is governed by the additive CUDA spec (`vision-embed-cuda-SPEC.md`), which authorizes `CUDAExecutionProvider` as a fail-closed config — see that document for the current provider contract.
4. **INV-4 Fail closed.** Unknown config keys are a fatal startup error. A missing model file is a fatal startup error — never a 200 with empty arrays. Unknown routes are 404 with a reason token.
5. **INV-5 One process.** No background workers, no queue, no sidecars, no scheduler.
6. **INV-6 Auth minimal and owned here.** One static bearer token loaded from a file path in config, compared in constant time. No OAuth, no IdP, no mTLS in v1 (listener is still TLS-only). Callers never receive any model-host credentials; there are no vendor credentials anywhere in this service.
7. **INV-7 Audit-only observability of content.** Never log image bytes, embeddings, Authorization headers, or tokens. The audit record is exactly one JSON line per request on stdout (section 9). Metrics carry no content-derived labels (section 10).
8. **INV-8 Closed surfaces.** Closed endpoint set { `/healthz`, `/readyz`, `/v1/embed` }, closed reason-token set (section 6), closed request/response schemas with `additionalProperties: false` on every object.
9. **INV-9 Network posture.** TLS-only listener (cert + key file paths in config). The operator firewalls the listener off the public internet; the service itself offers no cloud deploy, ingress, or public exposure. v1 defines no mTLS.
10. **INV-10 No caching, no idempotent replay.** A repeated POST of the same bytes is simply another inference. The logged sha256 is the idempotency key for the *caller's* bookkeeping, not for this server.

Do not invent: a person store, an identity graph, a face-name mapping, a vector index, a Redis/SQLite sidecar, a warm-pool of workers, a retry queue, OAuth/jwt/session machinery, an admin UI, a model-download step at runtime, or any endpoint beyond the closed set.

---

## 2. Stack (pinned)

| Component | Pin |
|---|---|
| Language | Python **3.12** (exact minor) |
| ASGI framework | **FastAPI** (one ASGI app, served by `uvicorn` with `--workers 1`) |
| Face embedding | **InsightFace buffalo_l** — recognition model `w600k_r50.onnx` (512-d, L2-normalized output as emitted by the pack) |
| Detector | InsightFace buffalo_l pack's **`det_10g.onnx`** (SCRFD-10G). Pin by exact model file name; never by "latest". |
| Scene embedder | **CLIP ViT-L/14 (OpenAI pretrained) visual tower**, executed via onnxruntime as an operator-supplied ONNX file named **`openai-ViT-L-14-visual.onnx`**. I do not have a citable canonical ONNX export for this file, so the filename is **operator-supplied**; the model card is open_clip `ViT-L-14` / pretrained `openai` with a 224×224, mean/std CLIP preprocessing pipeline (bicubic resize, center crop, normalization). |
| Runtime | **onnxruntime**, `CPUExecutionProvider` only (v1). No GPU device is required or requested (section 3). |

### Weight acquisition
- Do **not** download weights at runtime. There is no fetch-on-start, no HTTP download, no Hugging Face pull in the container.
- `det_10g.onnx` and `w600k_r50.onnx` are distributed with buffalo_l and are mounted into the container (section 3); startup fails if absent.
- `openai-ViT-L-14-visual.onnx` is operator-supplied per the note above; startup fails if absent.

### Determinism
- Default provider is `CPUExecutionProvider` and this is a hard v1 default. CUDAExecutionProvider may exist in code as a config key but is **config-off by default and MUST remain off until a tolerance test exists**; as of this draft I cannot cite an onnxruntime cross-execution-provider bitwise-determinism guarantee, so v1 does not claim GPU/CPU equality. All determinism claims in this spec apply to CPU-only runs. Do not claim GPU/CPU equality on CPU-only determinism evidence; provider configuration is now governed by the additive CUDA spec (`vision-embed-cuda-SPEC.md`), which authorizes `CUDAExecutionProvider` fail-closed and certifies no numeric equality with CPU.
- Preprocessing (resize/crop/normalization) must be pure Python/numpy implemented identically for all paths; do not introduce library version drift between preprocessing runs.
- Document response field `vector` values as float32; compare with tolerance `|a-b| <= 2e-5` per element on the same image bytes across runs and restarts (CPU provider).

Do not invent: alternate model packs, a second detector, an ensemble, a YOLO object model beyond the one named object labeler (below), a "model zoo" abstraction, or runtime model switching.

## 2a. Object detection component

Object detection labels/boxes come from a single named detector: **YOLOv8n** exported to ONNX as operator-supplied **`yolov8n.onnx`** (COCO-80 label set, score threshold fixed in config default 0.35, top-30 objects max per image). Pin the file name, not a floating tag. Startup fails if the file is absent. Do not invent: class-name mapping beyond COCO-80, a tracking id, an "object type" hierarchy, or a second object detector.

**F11 — YOLO pipeline (normative).** The YOLO input is letterboxed to 640×640 RGB, pad value 114, stride 32. Decode the raw head with output shape `(1, 84, 8400)`: take the per-class max confidence per candidate, then apply class-agnostic NMS at IoU 0.45. Keep detections with `score >= object_score_min`, at most 30, highest score first. Map boxes back from letterbox space to original-image pixels before clipping. `objects[].bbox` uses the same integer pixel rules as `faces[].bbox` (top-left origin, `x`,`y` ≥ 0, `w`,`h` > 0, clipped to the decoded image).

## 2b. InsightFace pack note

buffalo_l ships five ONNX files (`det_10g`, `w600k_r50`, `1k3d68`, `2d106det`, `genderage`). This service mounts only `det_10g.onnx` and `w600k_r50.onnx`. The landmark and genderage files are **not** used and MUST NOT be loaded. Do not invent a use for them.

**F9 — detector keypoints and alignment (normative).** `det_10g.onnx` itself emits the 5 facial keypoints; they are not discarded. The detector protocol's `detect` returns a list of `(x, y, w, h, det_score, kps)` where `kps` is 5 `(x, y)` pairs in original-image pixels. `FaceEmbedder` aligns with the standard ArcFace 5-point similarity warp to 112×112 RGB, normalizes with `(pixel/255 - 0.5) / 0.5`, then runs `w600k_r50`. Do **not** load a separate landmark model, and do **not** crop-resize a face into the recognizer without using the keypoints. This requirement is restated in section 7 so the two sections match.

---

## 3. Container contract

- Base image: python:3.12-slim, pinned by digest. No GPU device required.

- Runs as non-root **uid 65532**, group 65532.

- **Read-only rootfs.** Writable tmpfs at `/tmp` sized 128 MiB. No writable volumes beyond tmpfs and the mounts below.

- Mounts (ro): model files (`/models/det_10g.onnx`, `/models/w600k_r50.onnx`, `/models/openai-ViT-L-14-visual.onnx`, `/models/yolov8n.onnx`) and secrets (`/secrets/authz.bearer`) as file mounts, never environment variables or CLI flags.

- TLS cert and key file paths are in config (section 4) and are mounted (key: ro mount as `0400:65532`; cert: ro).

- Listener: TLS-only (config `tls.cert_path`, `tls.key_path`). Service binds to the address the operator chooses (default `0.0.0.0:8443`); the operator firewalls the public internet out.

- No inbound connectivity is required. No outbound network is required by the code path; outbound egress is denied by container policy (nothing in v1 fetches at runtime).

- Metrics server is a second port (config `metrics.port`, default 9090) in the same process, no auth.

- Do not invent: a systemd unit, a Docker Compose stack with extra services, a sidecar proxy, a healthcheck that posts to /v1/embed, autoscaling config, GPU device maps, or a public ingress.

---

## 4. Config YAML — full example (doubles as `config.example.yaml`)

```yaml
# config.example.yaml — vision-embed v1
# Unknown keys are FATAL at startup. See section 6 reason token config.invalid.
# All secrets are file paths, read once at startup. Values are never logged.

listen:
  host: "0.0.0.0"
  port: 8443
  tls:
    enabled: true
    cert_path: "/etc/vision-embed/tls/fullchain.pem"
    key_path: "/etc/vision-embed/tls/privkey.pem"
  # operator firewalls public internet access; service enforces nothing at L3

auth:
  bearer_token_path: "/etc/vision-embed/secrets/authz.bearer"
  # file contains exactly one line; no trailing whitespace required; value is the token
  # read at startup, held in memory, compared in constant time

models:
  det_path: "/models/det_10g.onnx"              # pinned SCRFD-10G (buffalo_l pack)
  face_embed_path: "/models/w600k_r50.onnx"     # pinned 512-d ArcFace-R50 (buffalo_l pack)
  scene_embed_path: "/models/openai-ViT-L-14-visual.onnx"  # operator-supplied CLIP ViT-L/14 visual
  object_det_path: "/models/yolov8n.onnx"       # operator-supplied COCO-80
  det_score_min: 0.5            # face det_score floor; below it a face record is dropped
  object_score_min: 0.35        # per-object score floor
  max_faces: 20                 # hard cap; faces beyond cap are dropped (audit reason stays ok)
  execution_provider: "CPUExecutionProvider"   # only "CPUExecutionProvider" is valid in v1
  adapter: "production"         # F25: only "production" or "fake"; absent is fatal; unknown value is fatal

limits:
  max_body_bytes: 33554432        # 32 MiB
  allowed_mime: ["image/jpeg", "image/png", "image/webp"]  # sniffed, not trusted

metrics:
  enabled: true
  port: 9090

audit:
  enabled: true                   # section 9; stdout JSON-lines

# F25/F26: top-level "test" or "production"; default "production"; any other value is fatal
adapter_mode: "production"

request_id:
  header: "X-Request-Id"          # optional incoming id; else generated uuid4
```

Rules:
- **F26 — fatal startup output.** A fatal startup condition writes exactly one JSON line to stdout — `{"fatal": true, "reason": "config.invalid", "detail": "<why>"}` — then exits 1. Startup fatals are **not** HTTP responses.
- Unknown top-level or nested keys → fatal startup error (`config.unknown_key` on stdout as a fatal log, exit nonzero; not an HTTP error — the server must not come up). `config.unknown_key` is used **only** for YAML keys that are not in the schema; it is not in the HTTP reason-token set, and no HTTP token is added for startup conditions.
- Missing model file at a configured path → fatal startup error, exit nonzero.
- `execution_provider` anything other than `"CPUExecutionProvider"` → fatal startup error in v1.
- **F25:** top-level `adapter_mode` accepts only `"test"` or `"production"`; default is `"production"`; any other value is a fatal `config.invalid`.
- **F25:** `models.adapter` accepts only `"production"` or `"fake"`. Absent is fatal `config.invalid`. Unknown value is fatal `config.invalid`. `production` uses the real adapters; `fake` uses Fake adapters for **all four roles: face, face-detector, object-detector, and scene** — never a hybrid of fake and real.
- **F25/F26:** `models.adapter: "fake"` **and** `adapter_mode: "production"` together are fatal `config.invalid` — this is a config-validity error, not `config.unknown_key`. `models.adapter: "fake"` with `adapter_mode: "test"` is the only way tests run without weights.
- Secrets are file paths. Do not invent: env-var secrets, inline secrets in YAML, Vault/KMS integrations, config hot-reload, per-tenant config, a second auth file, or a "profile" system.

---

## 5. Authentication

- Exactly one static bearer token, read once at startup from `auth.bearer_token_path`, compared in **constant time** against the `Authorization: Bearer <token>` header on `/v1/embed` only.
- `/healthz` and `/readyz` are unauthenticated.
- No OAuth, no IdP, no mTLS, no JWT, no refresh, no rotation endpoint, no per-caller keys, no rate limiting in v1. Do not invent any of these.
- Callers never receive model-host credentials. There are no vendor credentials (no HF token, no S3 key, no API key to anything external) anywhere in this service — nothing external is called.
- An invalid or absent Authorization header is rejected before any model inference and before body decode (see section 6 order), with reason token `auth.invalid_token` (HTTP 401).

---

## 6. HTTP API

### Endpoints — closed set
- `GET /healthz` → 200 `{"status":"ok"}` — process alive. No model check.
- `GET /readyz` → 200 `{"status":"ready","models_loaded":4}` once all four adapter roles are loaded. In `adapter_mode: "production"` this means all four real model files are loaded; in `models.adapter: "fake"` + `adapter_mode: "test"` this means the four fake stubs are loaded and `/readyz` reports ready with **zero ONNX model files present** — the gauge is 4 because the four stubs are loaded, not because ONNX files exist. 503 `{"error":{...,"reason":"not_ready"}}` otherwise. No auth.
- `POST /v1/embed` → see below. Requires valid bearer token.
- **No other routes exist.** Unknown path/method → 404 with `{"error":{"reason":"route.not_found"}}`.

### POST /v1/embed request
- Header: `Content-Type: multipart/form-data`, form field name `image`, containing the raw image file bytes (jpeg, png, or webp).
- Header: `Authorization: Bearer <token>` (constant-time compare).
- Optional header: `X-Request-Id` (any printable ASCII ≤128 chars; else a uuid4 is generated server-side). **F6:** if `X-Request-Id` is present but empty, contains non-printable-ASCII, or is longer than 128 chars, the server ignores it and generates a uuid4 — this is not an error. The **effective request_id** (the echoed client id when valid, else a generated uuid4) is used identically in the success response, in the error envelope's `request_id`, and in the audit line's `request_id`. `X-Request-Id` is accepted on `GET /healthz`, `GET /readyz`, and `POST /v1/embed`.
- Optional form field `include_object_vectors=json-bool` (default `false`). **F7:** the field is parsed as the exact strings `"true"` or `"false"` (case-sensitive); it is absent → `false`. Any other present value → HTTP 422 with reason `decode.failed` and detail `invalid include_object_vectors`. Extra form fields are ignored. No new reason token is added for this. When true, each detected object's crop is embedded to a **768-d** vector by the **same scene embedder** that produces the whole-image scene vector (the pinned ViT-L/14 visual tower, `openai-ViT-L-14-visual.onnx`); no separate object-crop embedder model exists.
- Max body size: **32 MiB** (config `limits.max_body_bytes`). Larger → reject before reading the full body.
- Supported types: JPEG, PNG, WebP — determined by magic-byte sniffing, not by trusting the multipart `Content-Type` header. Any other format → rejected.

### Processing order (numbered, stops at first failure)
1. **Size limit** — enforce `Content-Length`/streamed size ≤ 32 MiB. Failure → `limits.body_too_large`.
2. **Auth** — constant-time bearer comparison. Failure → `auth.invalid_token`. On rejection the body is not read and not sniffed.
3. **Content type (magic-byte sniff)** — jpeg (`FF D8 FF`), png (`89 50 4E 47`), webp (`RIFF....WEBP`). Failure → `format.unsupported`.
4. **Decode** — decode full image to a numpy RGB uint8 array. Failure → `decode.failed`.
5. **Model inference** — detector → face records; InsightFace face embedding per face (cap `max_faces`); object detector; scene embedder (whole image); optional per-object crop embedder. Failure → `inference.failed`.
6. **Schema-validate** the response object against the closed schema (below). Failure → `response.validation_failed` (this is a server bug; must be surfaced, never swallowed).
7. **Audit line** — emit exactly one JSON line per section 9 to stdout.
8. **Body** — return the response body. If the stdout write in step 7 cannot be confirmed to have succeeded, we **do not** add a fallback gate: if audit-to-stdout write fails silently (e.g., stdout closed by the runtime), the body is still returned and there is nothing else this v1 can do; I am saying this plainly rather than inventing a syslog gate or a retry queue. Do not buffer audit for later, do not write to a fallback file, do not invent a stderr duplicate.

### Determinism guarantee
Same image bytes and same pinned model versions ⇒ identical vectors on CPUExecutionProvider, within per-element tolerance `|a-b| ≤ 2e-5`. Repeated POSTs of the same bytes are independent inferences; results are not cached and no idempotency-keyed dedup occurs server-side.

### Reason tokens — COMPLETE and CLOSED set

Errors share a single envelope:

```json
{
  "error": {
    "reason": "<one of the tokens below>",
    "detail": "<free-form single-line string, safe for logs, never contains image bytes, tokens, or vectors>",
    "request_id": "<the effective request_id: echoed client id when valid, else uuid4>"
  }
}
```

| Reason token | HTTP |
|---|---|
| `limits.body_too_large` | 413 |
| `format.unsupported` | 415 |
| `decode.failed` | 422 |
| `auth.invalid_token` | 401 |
| `inference.failed` | 500 |
| `response.validation_failed` | 500 |
| `route.not_found` | 404 |
| `method.not_allowed` | 405 |
| `not_ready` | 503 |

No other reason tokens exist. Do not invent: `rate_limited`, `insufficient_permissions`, `bad_payload`, `internal_error`, `timeout`, or any token not in this table.

### Success response schema (exact, `additionalProperties: false` on every object)

```json
{
  "request_id": "<uuid4 or echoed client id>",
  "image": {
    "sha256": "<hex of the raw bytes of the multipart field named image, not the multipart envelope>",
    "width": 1024,
    "height": 768
  },
  "scene": {
    "model": "ViT-L-14/openai",
    "dim": 768,
    "vector": [ /* float32, length 768 */ ]
  },
  "faces": [
    {
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
      "det_score": 0.0,
      "model": "buffalo_l/w600k_r50",
      "dim": 512,
      "vector": [ /* float32, length 512 */ ]
    }
  ],
  "objects": [
    {
      "label": "<COCO-80 class string>",
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
      "score": 0.0,
      "model": "ViT-L-14/openai",
      "dim": 768,
      "vector": [ /* float32, length 768; present only if include_object_vectors=true */ ]
    }
  ]
}
```

Schema rules:
- `faces` bbox coordinates are in pixels relative to the **decoded image** (top-left origin, `x`,`y` ≥ 0, `w`,`h` > 0, integer pixels).
- `objects[].vector` is present **only** when `include_object_vectors=true` and the corresponding crop embedder was run; it is absent (not null) otherwise.
- `faces[].vector` and `scene.vector` are always present.
- `model` and `dim` are present on every object that carries a `vector` (scene, face, object-crop). `dim` must match the actual vector length.
- `objects[].label` is one of the 80 COCO className strings (snake-cased); no free-form labels.
- Empty arrays are valid for `faces` and `objects`.
- `additionalProperties: false` on every object, including the top-level response, `image`, `scene`, `faces[]`, `objects[]`, and each `bbox`.
- Do not invent: a `duration_ms` on the response, a `det_model` field separate from face `model`, an `aliveness` field, a `face_quality` score, landmarks, gender/age, a pose field, thumbnails, a confidence interval, or pagination.

---

## 7. Adapters

Each model is behind a Protocol interface so tests run without weights:

```python
class FaceEmbedder(Protocol):
    def load(self, path: Path) -> None: ...
    def embed(self, bgr: np.ndarray, kps: tuple[tuple[float,float], ...]) -> tuple[str, int, np.ndarray]: ...  # kps = 5 (x,y) pairs; ArcFace 5-point similarity warp to 112x112

class Detector(Protocol):
    def load(self, path: Path) -> None: ...
    def detect(self, bgr: np.ndarray) -> list[tuple[int,int,int,int, float, tuple[tuple[float,float], ...]]]: ...  # (x,y,w,h,det_score,kps)

class SceneEmbedder(Protocol):
    def load(self, path: Path) -> None: ...
    def embed(self, rgb: np.ndarray) -> tuple[str, int, np.ndarray]: ...

class ObjectDetector(Protocol):
    def load(self, path: Path) -> None: ...
    def detect(self, rgb: np.ndarray) -> list[tuple[str, int,int,int,int, float]]: ...  # label, x,y,w,h, score
```

- **F9 — face alignment (normative, matches section 2b).** The Detector protocol returns the 5 facial keypoints emitted by `det_10g.onnx`: `detect` returns `(x, y, w, h, det_score, kps)` with `kps` being 5 `(x, y)` pairs in original-image pixels. `FaceEmbedder` aligns with the standard ArcFace 5-point similarity warp to 112×112 RGB, normalizes with `(pixel/255 - 0.5) / 0.5`, then runs `w600k_r50`. Do not load a separate landmark model; do not crop-resize without the keypoints.
- A **`Fake*`** adapter set exists for **all four roles: face, face-detector, object-detector, and scene**. Fakes are deterministic, seeded by the image sha256. With the fake set, the full request/response/audit path runs with **zero ONNX files present** via `models.adapter: "fake"` **and** `adapter_mode: "test"` in config (section 4). The fake object detector returns a deterministic fixture list seeded by the image sha256 — at least one object with a COCO-80 label, a score, an integer bbox, and (when `include_object_vectors=true`) a 768-d fixture vector.
- **F25/F26:** `models.adapter: "fake"` with `adapter_mode: "production"` → fatal startup error `config.invalid` (written as the one-line stdout JSON fatal, exit 1). This is a config-validity error, not `config.unknown_key`; `config.unknown_key` is only for YAML keys not in the schema.
- Do not invent: a plugin registry, an auto-discovery loader, an adapter factory keyed on environment variables, or more than four adapter roles.

---

## 8. Idempotency and statelessness

- Nothing is stored. The sha256 of the raw bytes of the multipart image part is logged (audit line) and that is the extent of the server's memory of any request.
- Restart loses nothing, because nothing was kept: no queue, no WAL, no cache, no session, no KV.
- A repeated POST is another inference and must produce the same vectors for the same bytes (same pinned model files, CPU provider). The caller uses the logged (or recomputed) sha256 of the multipart image part as their own idempotency key against their own store; this server does not dedupe and does not return "already seen."
- Do not invent: a hash-keyed response cache, a TTL, a bloom filter for seen-hash, an ETag, Idempotency-Key handling, or "last N requests" memory.

---

## 9. Audit

Exactly one **JSON line** per request on **stdout**:

```json
{"ts": "<RFC3339 UTC>", "request_id": "<effective request_id>", "sha256": "<hex", "bytes": 0, "faces": 0, "objects": 0, "outcome": "ok|error", "reason": "<reason token or \"ok\">"}
```

- **F30:** the audit `sha256` is SHA-256 over the exact bytes of the image part extracted from the multipart field named `image` — not the multipart envelope. It is the same hex digest as `image.sha256` in the success response.
- **F30/F6:** the audit `request_id` equals the effective request_id (echoed client id when valid, else uuid4) — the same value used in the success response and error envelope.
- **F29:** on success the audit `reason` is the literal string `ok`. Capping faces at `max_faces` does not change the reason; it stays `ok`. No prefixed or dotted variant of the success reason exists anywhere in this spec or the implementation — `ok` and nothing else.

- No syslog requirement in v1. Stdout JSON-lines is the record. **Do not ship vectors to syslog later without a new spec** — logging vectors is a spec violation, not an optimization, in any version.
- Never log image bytes, embeddings, Authorization headers, tokens, or the auth file contents. The `detail` field of error envelopes follows the same rule.
- Nothing persists across restart. No log file, no disk writes, no DB.

---

## 10. Metrics

Second port (same process), no auth, closed series **only**:

- `vision_embed_requests_total{outcome,reason}` — counter (`outcome` ∈ `ok|error`).
- `vision_embed_latency_seconds` — histogram, standard Prom buckets.
- `vision_embed_errors_total{reason}` — counter (reason token).
- `vision_embed_models_loaded` — gauge (0..4).

Forbidden in labels and in series names: any hash, any byte size, any vector, any value derived from image content beyond the `reason` token. **Face count as a metric label is forbidden** (it can fingerprint a person beyond a coarse reason token). Do not invent: extra series, custom buckets, an exporter, a push gateway, an alert rule, or a second metrics backend.

---

## 11. Tests the build must pass

Concrete assertions, not "further analysis":

1. **Config fail-closed**: injecting an unknown YAML key causes startup to exit nonzero; no HTTP listener opens.
2. **Missing model fail-closed**: deleting one of the four model files and starting up causes exit nonzero, 503 never becomes 200, and `/readyz` never reports ready. **This test applies only to `adapter_mode: "production"` (i.e. `models.adapter: "production"` with the four real model files configured); it does not apply to fake mode, which intentionally starts and reports ready with zero ONNX files.**
3. **Fake-adapter production refusal**: config with `models.adapter: "fake"` and `adapter_mode: "production"` → exit nonzero with the fatal JSON line `{"fatal": true, "reason": "config.invalid", ...}`. (Test keys must exist in the schema: `models.adapter` and top-level `adapter_mode` — nothing else.)
4. **Auth**: bad token → 401 with `auth.invalid_token`; constant-time compare verified (no timing-sensitive branch on token content that a test can exploit); valid token on all three endpoints behaves per spec (auth only required on `/v1/embed`).
5. **Body limits**: a 40,000,000-byte multipart upload → 413 `limits.body_too_large` before full-body read completes.
6. **Format sniff**: a `.txt` file uploaded → 415 `format.unsupported`. A JPEG file with the wrong multipart Content-Type header still decodes (magic bytes win).
7. **Closed route set**: `GET /v1/embed`, `POST /healthz`, `PATCH /v1/embed` → 404/405 with correct tokens.
8. **Schema strictness**: response JSON validated with `additionalProperties: false`; adding an invented key to the response object fails the validator.
9. **Face bbox relativity**: a fixture image with a known face produces a bbox inside `[0,width)×[0,height)` and `det_score ≥ det_score_min`, dim 512.
10. **Object vectors opt-in**: `include_object_vectors=false` → no `vector` key on any `objects[]` entry; `true` → present, dim 768.
11. **No-identity**: grep asserts the string "cluster" never appears in any response field name or in the codebase's response construction path; a fixture test asserts no field in the response is named or shaped like a person identity.
12. **Statelessness across restart**: run the same request twice across a process restart → identical vector values within `|a-b| ≤ 2e-5` (CPU provider).
13. **Reason token closed set**: a property-based test asserts the set of reason tokens in code equals exactly the 9 tokens in section 6, no more.
14. **Audit shape**: every POST yields exactly one stdout JSON line with all eight keys of section 9, and stdout never contains 40+ hex-char strings longer than the sha256, never contains "Bearer", and never contains the token file contents.
15. **Metrics closed set**: scraped metrics contain exactly the four series names in section 10 and no label value that parses as a 64-char hex string or a 512-length float array.
16. **No runtime fetch**: container in a network-isolated test starts, is ready, and serves a fixture image — proving no outbound fetch at startup or inference time.
17. **Read-only rootfs**: run the container with `--read-only` and a tmpfs `/tmp`; a fixture request and restart both succeed.
18. **Non-root**: container process UID is 65532 (assert in test).
19. **CUDA off by default**: config default `execution_provider` is `CPUExecutionProvider`; enabling CUDAExecutionProvider without a tolerance test flag causes startup to fail.

---

## 12. Done gates

Definition of done for v1 — named, checkable:

1. All 19 tests of section 11 pass.
2. `GET /healthz` returns 200 within 5 s of process start on a clean container run; `GET /readyz` returns 200 once four models are loaded.
3. `POST /v1/embed` with a fixture image containing 1 known face and 1 known object returns a response that validates against the schema in section 6 with `faces[0].dim == 512` and `scene.dim == 768`.
4. A restart test (kill -9 then restart) shows no state loss and no change in vector outputs on the same bytes.
5. A code search asserts no storage primitives (no `sqlite`, no `INSERT`, no `open(..., 'w')` outside tmpfs, no `redis`, no `boto3`, no S3 client) anywhere outside test fixtures.
6. Audit stdout lines conform exactly to section 9 for both success and every one of the nine reason tokens.
7. The metrics endpoint exposes exactly the four series in section 10.
8. The two docs artifacts exist and match: `config.example.yaml` (section 4) and this SPEC.

---

## 13. Out of scope (v1)

Storage of any kind, clustering, naming, GUI, MCP server, phone backup ingestion, public internet exposure, path-based ingest (a filesystem path in the public request contract), CUDA in v1, OpenViking, agent-runtime-specific code, mTLS, multitenancy, rate limiting, model download at runtime, syslog, retry queues, batch endpoints, video, audio. The caller owns any index, matching, clustering, naming, and persistence.

---

## 14. Feature register

| ID | Feature (one sentence, for critic scoring) |
|---|---|
| F1 | exposes `GET /healthz` liveness with no auth and no model dependency |
| F2 | exposes `GET /readyz` whose readiness after the four model files load applies to production mode, while fake+test reports ready with zero ONNX model files present (section 6) |
| F3 | exposes `POST /v1/embed` accepting multipart form field `image` as the only ingest path |
| F4 | enforces the 32 MiB max body size before full-body read with reason `limits.body_too_large` |
| F5 | sniffs magic bytes for jpeg/png/webp magic bytes and rejects any other format with reason `format.unsupported` |
| F6 | accepts an optional X-Request-Id header or generates a uuid4 when absent |
| F7 | includes the optional `include_object_vectors` multipart flag (default false) to opt into object-crop embeddings |
| F8 | detects faces using the pinned SCRFD-10G detector `det_10g.onnx` |
| F9 | embeds each face with the pinned 512-d `w600k_r50.onnx` buffalo_l recognition model |
| F10 | embeds the whole image with the pinned operator-supplied `openai-ViT-L-14-visual.onnx` scene model |
| F11 | detects objects with the pinned operator-supplied `yolov8n.onnx` COCO-80 model |
| F12 | reports every face bbox in pixels relative to the decoded image |
| F13 | reports every face `det_score` and drops faces below `det_score_min` |
| F14 | caps faces per image at `max_faces` |
| F15 | reports every object with a COCO-80 label, bbox, and score above `object_score_min` |
| F16 | includes an object `vector` field only when `include_object_vectors=true` was requested |
| F17 | runs all models on `CPUExecutionProvider` only and rejects any other provider at startup |
| F18 | returns identical vectors for identical image bytes plus the same pinned model files within `|a-b| ≤ 2e-5` on the CPU provider |
| F19 | rejects any non-`CPUExecutionProvider` config value as a fatal startup error until a GPU/CPU tolerance test exists |
| F20 | compares the bearer token from a file path in config in constant time |
| F21 | rejects any request with an invalid or missing Authorization header with reason `auth.invalid_token` and HTTP 401 |
| F22 | treats unknown config keys as a fatal startup error (fail closed) |
| F23 | treats a missing model file as a fatal startup error — never a 200 with empty faces — only in production mode, with fake+test exempt (section 11 test 2) |
| F24 | exposes four adapter Protocols (FaceEmbedder, Detector, SceneEmbedder, ObjectDetector) |
| F25 | provides a fake-adapter set so tests run without model weights present |
| F26 | rejects a production config that selects any fake adapter as a fatal startup error |
| F27 | never returns a cluster id, person name, or any same-person judgment in the response |
| F28 | stores nothing — no photos, vectors, names, cluster ids, cache, or database, across requests or restarts |
| F29 | emits exactly one audit JSON line per request on stdout with the eight keys of section 9 |
| F30 | uses the raw image sha256 as the logged idempotency key and never caches responses server-side |
| F31 | never logs image bytes, embeddings, Authorization headers, or tokens |
| F32 | exposes four metrics series on a second unauthenticated port with no content-derived labels and no face-count labels |
| F33 | runs as uid 65532 with read-only rootfs, tmpfs /tmp, and ro-mounted secrets and model files |
| F34 | runs a TLS-only listener on configured cert/key file paths, with the operator responsible for firewalling the public internet |
| F35 | requires no outbound network access at runtime and performs no runtime model downloads |
| F36 | enforces a closed endpoint set of exactly { /healthz, /readyz, /v1/embed } with reason tokens for 404 and 405 |
| F37 | enforces the closed nine-token reason set with a uniform error envelope and no reason tokens outside the list |
| F38 | validates every response against the closed success schema with `additionalProperties: false` on every object |
| F39 | stops processing at the first failure in the fixed order (size limit → auth → content type → decode → inference → schema-validate → audit → body) with each step's own reason token |
| F40 | still returns the response body if the stdout audit write fails silently, without inventing a fallback file, syslog, or retry queue |

## 14a. Feature register — reason-token coverage (map)

- `limits.body_too_large` → F4
- `format.unsupported` → F5
- `decode.failed` → F39 (step 4)
- `auth.invalid_token` → F21
- `inference.failed` → F39 (step 5)
- `response.validation_failed` → F38, F39 (step 6)
- `route.not_found` → F36
- `method.not_allowed` → F36
- `not_ready` → F2

## 15. What the service owner still has to decide — open items only

| Item | My recommended default |
|---|---|
| Exact export recipe for `openai-ViT-L-14-visual.onnx` (scene dim is closed at 768) | open_clip export script, opset 14, mean/std CLIP normalization. |
| `det_score_min` exact value | 0.5 (config default listed above). |
| `object_score_min` exact value | 0.35. |
| `max_faces` exact value | 20. |
| Metrics port default | 9090. |
| Whether to publish an OpenAPI schema file alongside this SPEC | No. The FastAPI apps run with `docs_url`, `redoc_url`, and `openapi_url` all `None`; `/openapi.json`, `/docs`, and `/redoc` are not routes and no schema artifact is committed. |

Everything else above, including all other section 4 defaults, is decided and closed. Do not reopen decided items.

---

*End of SPEC — DRAFT v4.*

