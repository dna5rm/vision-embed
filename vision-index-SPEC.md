# vision-index — Build Contract SPEC

**Status: DRAFT v4**
**Changelog v4:** pins auth before body parse, with Content-Length over the limit rejected before auth.
**Changelog v3:** adds `GET /v1/search/objects` so object labels are searchable.
**Changelog v2:** closes critic failures FI5, FI7, FI10, FI19, FI20, FI23, FI37, FI38, FI40.
**Product owner: the service owner (me).** This document is the complete build contract for the stateful identity index. An implementer builds exactly what is specified here. Every behavior not in this contract has an explicit "do not invent" instruction; if a needed detail is absent, the implementer MUST stop and ask, not fill the gap.

Companion document: `vision-embed-SPEC.md` (DRAFT v4). This service **calls** vision-embed for measurements only and consumes **exactly the fields vision-embed's success schema returns**: `image.sha256`, `image.width`, `image.height`, `scene.model/dim/vector` (768), `faces[].bbox/det_score/model/dim/vector` (512), `objects[].label/bbox/score` and `objects[].vector` only when `include_object_vectors=true`. vision-embed returns **no keypoints, no cluster ids, no names**; this spec must not reference fields vision-embed does not return. Do not contradict vision-embed-SPEC.md. Do not rewrite that file.

---

## 1. Status, owner, invariants

Numbered invariants — each is an acceptance gate, not a suggestion:

1. **INV-1 One deployment, one owner, one SQLite file.** Not multi-tenant, no shared people table across callers. Another person runs their own deployment and their own database. Do not invent: tenancy ids, per-caller keyspaces, shared schema, or remote DB backends.
2. **INV-2 Stateful here, stateless there.** This service stores vectors, cluster ids, and names. It never asks vision-embed for a cluster id or a name. It computes all clustering itself. vision-embed stores nothing; this service is the source of truth for names. Do not invert these roles.
3. **INV-3 Face index is array position.** A `face_index` on the wire is the zero-based position in the `faces` array returned by vision-embed for that asset's stored embedding run. This service stores the face rows in that array order. Do not invent a separate face ordering or keypoint alignment.
4. **INV-4 Fail closed.** Unknown config keys are fatal at startup. Missing database file is fatal only if the file cannot be created at the configured path; missing embed token file is always fatal. Missing embed URL fatal. Unknown routes are 404 with a reason token.
5. **INV-5 One process.** One ASGI app, `uvicorn --workers 1`, one SQLite file opened by that process (WAL mode). No background workers, no queues, no schedulers. Do not invent: a compactor, a background reclustering job, a sync worker.
6. **INV-6 Auth minimal and owned here.** One static bearer token loaded from a file path in config, compared in constant time. TLS-only listener. `/healthz` unauthenticated; all `/v1/*` and `/readyz`... — `/readyz` is unauthenticated; all `/v1/*` routes require the token. Do not invent: OAuth, mTLS, JWT, rotation, per-caller tokens, rate limiting.
7. **INV-7 Audit-only observability of content.** Exactly one JSON audit line per request on stdout. Never log image bytes, embeddings, Authorization headers, tokens, or name values. Metrics carry no content-derived labels.
8. **INV-8 Closed surfaces.** Closed endpoint set (section 6a), closed reason-token set (section 6c), closed schemas with `additionalProperties: false` on every object.
9. **INV-9 Outbound call only to vision-embed.** The process calls exactly one external service: vision-embed over HTTPS. No OpenViking reads or writes, no other HTTP egress. OpenViking is out of scope entirely; a caller may mirror a name elsewhere, this service does not and does not learn about it.
10. **INV-10 No InsightFace in this process.** This process runs no ONNX models at all. All inference is delegated. Do not install or load any model file here.
11. **INV-11 Secrets are file paths.** Config never contains an inline token value; only paths. Values are never logged.

Do not invent: a second embed provider, a vector database other than SQLite+sqlite-vec, an ANN index library, a person graph beyond the tables in section 7, a split endpoint, merge of object clusters, auto-naming from scene vectors, face-keypoint storage, image byte storage, or any endpoint beyond the closed set.

---

## 2. Stack (pinned)

| Component | Pin |
|---|---|
| Language | Python **3.12** (exact minor) |
| ASGI framework | **FastAPI**, one ASGI app, served by `uvicorn --workers 1` |
| Vector storage | **sqlite-vec** (SQLite extension, float32 vectors; exact-match nearest scan at this scale) |
| Database | **SQLite** single file (config `db.path`), PRAGMA `journal_mode=WAL`, `foreign_keys=ON` |
| SQL access | stdlib `sqlite3` with explicit migration at startup; no ORM, no Alembic |
| Metric type | cosine distance = `1 - dot(a, b)` on L2-normalized float32 vectors; lower is closer |

Vectors stored via sqlite-vec `embedding float[512]` / `float[768]` columns. All vectors written to storage are re-normalized to unit length before insert (defensive; vision-embed's face model emits L2-normalized vectors, this service does not trust that silently). Do not invent: faiss, hnswlib, pgvector, quantization, a second index format, or batching beyond a single request.

---

## 3. Container contract

- Base image: python:3.12-slim, pinned by digest. No GPU device, no model mounts.
- Runs as non-root **uid 65532**, group 65532.
- **Read-only rootfs.** Writable volumes: tmpfs `/tmp` (128 MiB) and the database directory (rw mount, exactly the one volume). No other writable volumes.
- Mounts: TLS cert/key (key `0400:65532`), bearer token file `/secrets/index-authz.bearer` (ro, `0400:65532`), the SQLite db file's directory (rw), and config file (ro).
- Listener: TLS-only, cert/key paths in config; default bind `0.0.0.0:9443`.
- Metrics server is a second port (default 9190) in the same process, no auth, no TLS requirement stated for it — pin: metrics port is plain HTTP on loopback-bind `127.0.0.1` by default; the operator may firewall it.
- Outbound network: allowed **only** to the configured `embed.base_url`. Do not invent: egress to example.com, a proxy knob, service discovery, retries to other hosts.

---

## 4. Config YAML — full example (doubles as `config.example.yaml`)

```yaml
# config.example.yaml — vision-index v1
# Unknown keys are FATAL at startup (exit 1, one JSON fatal line on stdout).
# All secrets are file paths. Values are never logged.

listen:
  host: "0.0.0.0"
  port: 9443
  tls:
    enabled: true
    cert_path: "/etc/vision-index/tls/fullchain.pem"
    key_path: "/etc/vision-index/tls/privkey.pem"

auth:
  bearer_token_path: "/etc/vision-index/secrets/index-authz.bearer"
  # file contains exactly one line; that line is the token; read once at startup

embed:
  base_url: "https://192.0.2.10:8443"        # vision-embed base URL; no default; absent is fatal
  bearer_token_path: "/etc/vision-index/secrets/embed-authz.bearer"
  timeout_seconds: 60                         # per embed call; no retry loop beyond a single retry on connect failure
  # this service sends include_object_vectors=true on every embed call so object
  # crop vectors are available to store (section 7, objects/object_vec)

db:
  path: "/var/lib/vision-index/index.sqlite3"   # exactly one file; parent dir must exist

clustering:
  face_match_max: 0.40        # cosine-distance join threshold; see section 8

name:
  min_length: 1
  max_length: 64              # name strings validated to this range; no other constraints

metrics:
  enabled: true
  port: 9190
  host: "127.0.0.1"

audit:
  enabled: true

adapter_mode: "production"    # only "production" or "test"; other values fatal
# adapter_mode "test" selects the fake embed client + in-memory-ish fake db path
# helpers for tests; production must never reach the fake embed client (section 10).
```

Rules:
- Unknown top-level or nested keys → fatal startup (`{"fatal": true, "reason": "config.invalid", "detail": "unknown key ..."}`, exit 1). Unknown-key token is `config.invalid`; it is not in the HTTP reason-token set.
- Missing or unreadable `auth.bearer_token_path` or `embed.bearer_token_path` → fatal startup.
- Absent `embed.base_url` → fatal startup. Do not hardcode a host; do not default to `http://localhost`.
- `adapter_mode` only `production` or `test`; anything else fatal.
- `adapter_mode: test` **and** `listen.tls.enabled: true` together are fatal — tests run plain HTTP localhost; production TLS-only. `adapter_mode: test` with TLS enabled → fatal `config.invalid`.
- Do not invent: env-var secrets, config hot-reload, includes/overlays, per-caller config, profile merging.

---

## 5. Authentication

- Exactly one static bearer token protecting this service, read once at startup from `auth.bearer_token_path`, compared in **constant time** (`hmac.compare_digest`) on every `/v1/*` route.
- `/healthz` and `/readyz` unauthenticated.
- Second token file `embed.bearer_token_path` is the credential presented to vision-embed; never returned to callers, never logged.
- An invalid or absent `Authorization: Bearer <token>` header → 401 `auth.invalid_token`, rejected **before the body is read or parsed** — section 6b is normative on the order: the only pre-auth step is the Content-Length pre-check (step 1), and a request within the limit, or with no Content-Length, is rejected here with `auth.invalid_token` and the body is never parsed. Do not invent: OAuth, mTLS, JWT, refresh, rate limiting, per-caller keys.

---

## 6. HTTP API

### 6a. Endpoints — closed set

- `GET /healthz` → 200 `{"status":"ok"}`. No auth, no embed or DB check.
- `GET /readyz` → 200 `{"status":"ready","sqlite_ok":true,"embed_ready":true}` only when (a) the SQLite file opens and migrations pass and (b) a proxied `embed.base_url/readyz` returns ready. If either fails → 503 `not_ready`. `/readyz` never returns ready while embed is down. No auth.
- `POST /v1/ingest` — multipart, form field `image` (raw jpeg/png/webp bytes), optional form field `path` (string, the caller's label; stored verbatim, never read from disk, never resolved server-side). Requires auth. Calls vision-embed `/v1/embed` with `include_object_vectors=true`, then per section 6b processing order. Returns the ingest success schema below.
- `GET /v1/assets/{id}` — integer id. Returns the full asset record per schema (asset + scene vector + faces + clusters + names + objects). Requires auth.
- `GET /v1/search/person?name=<string>` — resolves the name to the face clusters carrying that name and to every face row in those clusters (face_id, asset_id, cluster_id, bbox), with asset ids. Exact-name match only; no cosine distance is computed or returned for this route. Requires auth.
- `GET /v1/search/objects?label=<string>` — exact match on the stored COCO-80 label, over the `objects` rows. Returns matching object rows (object_id, asset_id, score, bbox, cluster_id — `cluster_id` may be null) and the deduplicated asset ids. Auth required. No cosine distance. Label not among the 80 COCO names → 422 `decode.failed`, detail `invalid label`; a valid COCO label with no matching row (or an absent label) → 404 `not_found`. Do not invent a fuzzy search.
- `POST /v1/identify` — multipart image. Runs vision-embed, computes nearest face cluster per face **and** nearest scene asset, returns matches. **Persists nothing** — no asset row, no face rows, no vectors, no db writes of any kind. Requires auth.
- `POST /v1/name` — JSON body, exactly one of three forms (additionalProperties: false on each; providing two forms or none → `decode.failed`):
  1. `{ "cluster_id": int, "name": string }` — names an existing cluster.
  2. `{ "asset_id": int, "face_index": int, "name": string }` — `face_index` is the zero-based index in the stored faces for that asset, ordered by face id ascending; resolves via that stored face row and names its `cluster_id`.
  3. `{ "asset_id": int, "object_index": int, "name": string }` — `object_index` is the zero-based index in the stored objects for that asset, ordered by object id ascending; if that object row's `cluster_id` is null, an object-kind cluster is created for it first, then the name is set. This is how an object instance gets named. Requires auth.
- `POST /v1/merge` — JSON body `{ "from_id": int, "into_id": int }`, face clusters only. Requires auth.
- `GET /v1/clusters?kind=face&unnamed=true` — filter by `kind` (`face`|`object`), optional `unnamed` (`true`|`false`). Requires auth.
- **No other routes exist.** Unknown path/method → 404/405 with reason tokens. No DELETE of assets in v1. No split endpoint. Do not invent: `/v1/clusters/{id}`, `/v1/people`, a bulk ingest, a DELETE route, a stats route, OpenViking mirroring.

### 6b. Processing order — numbered, stops at the first failure

The order is **fixed and singular**. Steps 1 (Content-Length pre-check) and 2 (auth) happen before the body is read; the only order in this spec is:

1. **Content-Length pre-check** — if the `Content-Length` header is present and greater than 32 MiB, respond 413 `limits.body_too_large`; do not read the body; do not check the token. 32 MiB is fixed, not config (the same limit vision-embed enforces). An unauthenticated request with Content-Length over 32 MiB returns `limits.body_too_large`, **not** `auth.invalid_token`.
2. **Auth** — constant-time bearer compare. Missing or wrong token → 401 `auth.invalid_token`; do not read the body; do not sniff. A request within the limit, or with no Content-Length, is rejected here and the body is never parsed. This is what section 5 means by rejected before body decode.
3. **Read the body**, enforcing the 32 MiB limit while reading. If the stream exceeds 32 MiB (chunked encoding, or a Content-Length that lied low) → 413 `limits.body_too_large`.
4. **Magic-byte sniff** — jpeg/png/webp, same rules as vision-embed. Fail → `format.unsupported`. Do not trust the multipart Content-Type.
5. **Field validation** — `path` length ≤ 64 (fail → `path.too_long`); `/v1/name` and `/v1/merge` bodies schema-validated before any db access (fail → `decode.failed`).
6. **sha256 dedupe lookup (ingest only)** — compute sha256 of the raw image bytes and look up `assets.sha256` **before** calling vision-embed. On a hit: skip the embed call and the write entirely, return the existing asset with `created: false` (no new rows; steps 7–8 skipped). On a miss: continue.
7. **Embed call (skipped on dedupe hit)** — POST to `embed.base_url + /v1/embed`, bearer token from `embed.bearer_token_path`, `include_object_vectors=true`. `/v1/identify` always performs this step. Connection failure / timeout / non-200 / non-schema response → `embed.unavailable`. Do not fall back to any other embed source, do not re-decode locally, do not compute partial vectors.
8. **Transaction (skipped on dedupe hit)** — open a single SQLite transaction, then do all ingest writes inside it, and only there; commit or roll back atomically on failure. Fail → `db.error`. For `/v1/identify` this step is skipped entirely (nothing is written, ever).
9. **Schema-validate** the outbound response. Fail → `response.validation_failed` (server bug; surface, never swallow).
10. **Audit line** — one JSON line to stdout.
11. **Body** — return response body.

### 6c. Reason tokens — COMPLETE and CLOSED set

Error envelope (identical shape to vision-embed):

```json
{
  "error": {
    "reason": "<one of the tokens below>",
    "detail": "<free-form single-line string, safe for logs, never contains image bytes, tokens, vectors, or names>",
    "request_id": "<echoed X-Request-Id when valid, else uuid4>"
  }
}
```

`X-Request-Id` handling matches vision-embed exactly: optional header, printable ASCII ≤128 chars, else server-generated uuid4; the effective id is echoed in every error and success response and in the audit line.

| Reason token | HTTP |
|---|---|
| `limits.body_too_large` | 413 |
| `format.unsupported` | 415 |
| `decode.failed` | 422 |
| `path.too_long` | 422 |
| `not_found` | 404 |
| `route.not_found` | 404 |
| `method.not_allowed` | 405 |
| `auth.invalid_token` | 401 |
| `name.conflict` | 409 |
| `name.too_long` | 422 |
| `cluster.not_face` | 409 |
| `embed.unavailable` | 502 |
| `no_face` | 422 |
| `db.error` | 500 |
| `not_ready` | 503 |
| `response.validation_failed` | 500 |

No other tokens exist. `not_found` is used when an id or name on a valid route resolves to nothing; `route.not_found` for unknown paths; they are distinct tokens and both are valid. Do not invent further tokens; do not remove these.

### 6d. Success schemas — exact, `additionalProperties: false` on every object

`POST /v1/ingest` →

```json
{
  "request_id": "...",
  "asset_id": 1,
  "created": true,
  "image": { "sha256": "...", "width": 0, "height": 0 },
  "scene_vector": ["768 floats"],
  "face_count": 0,
  "face_cluster_ids": [0],
  "object_count": 0
}
```

- `created: true` when the sha256 is new; `false` when the asset already existed (idempotent re-ingest, all fields still returned, no rows added).
- Faces with zero faces detected: `face_count: 0`, empty arrays — this is normal, not an error. Do not return `no_face` on ingest with zero faces; store the asset and scene vector anyway.

`POST /v1/identify` →

```json
{
  "request_id": "...",
  "sha256": "...",
  "faces": [
    {
      "face_index": 0,
      "nearest_cluster_id": 1,
      "nearest_cluster_name": null,
      "distance": 0.0,
      "matched": true
    }
  ],
  "scene": [
    { "asset_id": 1, "distance": 0.0, "path": null }
  ]
}
```

- `matched` is true iff distance ≤ `face_match_max` **and** the nearest cluster is the nearest (per section 8 step 5 — matching semantics identical to ingest, evaluated without persisting).
- `faces` is empty when no face is detected; `no_face` is **only** returned by `/v1/name` when the referenced asset has no face at that index. `scene` list returns up to 10 nearest assets, ascending distance.
- identify never writes. Repeating identify of the same bytes yields the same computation; no dedupe of the request occurs.

`GET /v1/assets/{id}` →

```json
{
  "request_id": "...",
  "id": 1,
  "path": null,
  "sha256": "...",
  "width": 0,
  "height": 0,
  "bytes": 0,
  "indexed_at": "2026-11-20T00:00:00Z",
  "scene_vector": ["768 floats"],
  "faces": [
    { "id": 1, "asset_id": 1, "bbox": {"x":0,"y":0,"w":0,"h":0}, "det_score": 0.0,
      "cluster_id": 1, "cluster_name": "alice" }
  ],
  "objects": [
    { "id": 2, "label": "car", "score": 0.0, "bbox": {"x":0,"y":0,"w":0,"h":0},
      "cluster_id": null }
  ]
}
```

- `path` is null when the caller omitted it on ingest. No points stored on faces — do not add keypoints to this schema.
- If a referenced cluster has been merged, `cluster_id` reports the surviving cluster id (queries follow the pointer, section 8 step 7); merged-away ids never appear.

`POST /v1/name` →

```json
{
  "request_id": "...",
  "cluster_id": 1,
  "kind": "face",
  "name": "alice",
  "previous_name": null,
  "named_at": "2026-11-20T00:00:00Z"
}
```

`POST /v1/merge` →

```json
{
  "request_id": "...",
  "surviving_cluster_id": 2,
  "merged_cluster_id": 1,
  "face_count_moved": 0,
  "name": "bob"
}
```

`GET /v1/clusters` →

```json
{
  "request_id": "...",
  "clusters": [
    { "id": 1, "kind": "face", "name": null, "face_count": 0, "asset_count": 0 }
  ]
}
```

`GET /v1/search/person?name=` →

```json
{
  "request_id": "...",
  "name": "alice",
  "clusters": [1],
  "faces": [
    { "face_id": 1, "asset_id": 1, "cluster_id": 1, "bbox": {"x":0,"y":0,"w":0,"h":0} }
  ],
  "asset_ids": [1]
}
```

`GET /v1/search/objects?label=` →

```json
{
  "request_id": "...",
  "label": "car",
  "objects": [
    { "object_id": 1, "asset_id": 1, "score": 0.0, "bbox": {"x":0,"y":0,"w":0,"h":0}, "cluster_id": null }
  ],
  "asset_ids": [1]
}
```

- `objects[].cluster_id` may be null (an instance only carries a cluster after `/v1/name` names it). `asset_ids` is the deduplicated list of asset ids of the matching object rows. `additionalProperties: false` on every object; no cosine distance in the response.

Rules:
- Every success schema above includes `request_id` — including `GET /v1/assets/{id}` and `GET /v1/clusters` — and the effective request id is echoed in every success body, every error envelope, and the audit line.
- Additional properties forbidden on every object in every response above. No `duration_ms`, no thumbnails, no embeddings returned in `/v1/name` or `/v1/merge` responses, no pagination anywhere, no confidence intervals, no aliveness, no gender/age, no pose. Do not invent any field beyond these schemas.
- `faces[].bbox` and `objects[].bbox` are the integer pixel values returned by vision-embed, stored verbatim. Do not recompute or rescale.

---

## 7. SQLite schema (normative)

```sql
-- migrations applied in order at startup, in one transaction, into db.path
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
-- load sqlite-vec extension here

CREATE TABLE assets (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,                -- nullable; the caller's label; never read from disk
  sha256 TEXT UNIQUE NOT NULL,     -- sha256 of the raw image bytes (matches vision-embed image.sha256)
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  bytes INTEGER NOT NULL,          -- original image byte count from the multipart body
  indexed_at TEXT NOT NULL         -- RFC3339 UTC
);

CREATE TABLE faces (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL REFERENCES assets(id),
  x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
  det_score REAL NOT NULL,
  cluster_id INTEGER REFERENCES clusters(id)
);

CREATE VIRTUAL TABLE face_vec USING vec0(
  face_id INTEGER PRIMARY KEY,
  embedding float[512]
);

CREATE TABLE clusters (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('face','object')),
  name TEXT,                       -- null = unnamed
  named_at TEXT,                   -- null until named
  exemplar_face_id INTEGER,        -- face id; set at creation, per section 8 step 3
  merged_into INTEGER REFERENCES clusters(id)   -- null unless merged
);
CREATE UNIQUE INDEX clusters_kind_name ON clusters(kind, name) WHERE name IS NOT NULL;

CREATE TABLE objects (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL REFERENCES assets(id),
  label TEXT NOT NULL,             -- COCO-80 string exactly as returned by vision-embed
  score REAL NOT NULL,
  x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
  cluster_id INTEGER               -- stays null unless the caller names an instance
);

CREATE VIRTUAL TABLE object_vec USING vec0(
  object_id INTEGER PRIMARY KEY,
  embedding float[768]
);

CREATE VIRTUAL TABLE scene_vec USING vec0(
  asset_id INTEGER PRIMARY KEY,
  embedding float[768]
);
```

Rules:
- Vectors are the exact float32 values returned by vision-embed (re-normalized to unit length as in section 2). No other numeric transforms are applied.
- **Do not store keypoints.** vision-embed does not return them and this schema has no place for them. Do not add a keypoints table or columns.
- **Do not store image bytes.** The `bytes` column is the raw byte count only. Do not invent a blob table, a thumbnail table, a local file cache, or an OpenViking mirror record.
- `clusters.kind=object` exists so object instances can carry a unique name; object instances are **never** auto-clustered. `cluster_id` on `objects` rows stays null unless `/v1/name` targets an object cluster. Do not invent object clustering.
- One SQLite file, WAL mode. Do not invent: sharding, replication, a second database, a migrations framework, an ORM, or stored procedures beyond the tables above.

---

## 8. Clustering and naming — numbered, normative

Distance is cosine distance `1 - dot(a, b)` on L2-normalized float32 vectors. Lower is closer.

1. On ingest, one face row per element of vision-embed's `faces` array, in array order; each face's 512-d vector goes to `face_vec`, the `scene` 768-d vector to `scene_vec` keyed by the asset (once per asset, never per face), and each object's 768-d crop vector to `object_vec` (only present because this service sends `include_object_vectors=true` on every embed call).
2. For each new face: compute cosine distance from the face vector to the **exemplar vector of every face cluster that is not merged and not object-kind**.
3. Join rule — a new face joins a face cluster iff **both**: cosine distance to that cluster's exemplar ≤ `face_match_max` (default 0.40), **and** that exemplar is the cluster exemplar with the minimum distance among all clusters within `face_match_max` (i.e. the nearest exemplar). If no cluster qualifies, create a new face cluster.
4. **Exemplar pin.** `exemplar_face_id` is set at cluster creation to the first face that formed the cluster and **never changes** — not on subsequent joins and not on merge. On merge, the surviving cluster keeps its own exemplar untouched. Do not invent: recomputing an exemplar on insert or merge, a k-centroid model, a moving-mean anchor, or exemplar aging.
5. `matched` semantics (used identically on ingest and identify): a face matches the cluster that satisfies step 3; distance reported is distance to that exemplar. Distance > `face_match_max` or a nearer exemplar absent → not matched, new (or, in identify, unmatched) cluster.
6. **Object instances are not auto-clustered.** Object rows store label, score, bbox, and optional 768-d crop vector. `objects.cluster_id` stays null until `/v1/name` names the instance explicitly (creating an object-kind cluster). Do not auto-cluster object instances by crop-vector similarity, do not merge two object clusters, and do not apply `face_match_max` to objects.
7. **Merge (face clusters only).** `POST /v1/merge {from_id, into_id}`: both must resolve to existing, non-merged, `kind='face'` clusters (else `cluster.not_face` or `not_found`). Reassign every `faces.cluster_id` pointing at `from_id` to `into_id`. Set `clusters.merged_into` on `from_id` to `into_id`. Leave `into_id`'s name and exemplar untouched; the merged cluster id now resolves through `into_id`. The two names must not conflict (if both are named and different → `name.conflict`; if only `from_id` is named, delete `from_id`'s name and the surviving cluster id stays `into_id`). Do not invent: auto-merge, a split endpoint, an unmerge endpoint, or merge of object clusters.
8. **Naming.** `POST /v1/name` accepts exactly three body forms: `{cluster_id, name}` for an existing cluster; `{asset_id, face_index, name}` where `face_index` is the zero-based index in the stored faces for that asset, ordered by face id ascending (resolve via that stored face row; its `cluster_id` is named); `{asset_id, object_index, name}` where `object_index` is the zero-based index in the stored objects for that asset, ordered by object id ascending — in the object form, if that object row's `cluster_id` is null, an object-kind cluster is created for it first (this is how an object instance gets named), then the name is set. Providing two forms or none → `decode.failed`. Name is unique per kind: attempting to name a face cluster with a name already used by a face cluster → `name.conflict` (409); a name already used by an object cluster is allowed. Re-naming is permitted (the previous name is overwritten; `previous_name` is returned). `named_at` set only on first naming, not overwritten on rename — pin: `named_at` never changes after first set. A cluster with `name IS NULL` is unnamed. Do not invent: prefix matching, fuzzy name search, aliases across kinds, a rename history table.
9. **Transactions.** Ingest writes only inside the single transaction of section 6b step 8, opened only after a dedupe miss, rolled back atomically on failure. `/v1/name` and `/v1/merge` each use their own single SQLite transaction. Identify writes nothing. No SQLite write occurs outside these transactions.

**Scene search.** `scene_vec` is queried by sqlite-vec nearest-K (K=10 fixed) on cosine distance; results are asset rows, not people. Do not invent scene-based clustering.

---

## 9. Idempotency

- `sha256` of the raw image bytes (matching vision-embed's `image.sha256`) is the idempotency key. On ingest, the sha256 lookup happens **before** the embed call (section 6b step 6). If a row with that `sha256` already exists, the embed call is skipped entirely and the write is skipped entirely (no wasted inference, no new rows) and the existing asset is returned with `created: false`. Do not double-insert face rows on re-ingest of the same sha256. Do not create a second asset row for the same bytes.
- If the caller changed the `path` label on re-ingest, the existing asset's `path` is **not** updated — the first label wins. Do not invent an update path.
- `/v1/identify` is never idempotent-tracked and never writes; repeated calls repeat the computation.
- Do not invent: an `Idempotency-Key` header, a TTL, an ETag, or any cache.

---

## 10. Fake embed adapter

- A **fake embed client** exists behind the same Protocol interface as the real vision-embed HTTP client, selected in `adapter_mode: "test"` only. It returns deterministic, image-sha256-seeded fixture responses in the exact vision-embed success schema shape (768-d scene vector, 512-d face vectors with bbox+det_score, COCO-80 objects with 768-d crop vectors). Keyed by sha256 so a fixture image always yields the same vectors, which is what makes clustering tests reproducible.
- In `adapter_mode: "test"` the fake client may be pointed at an in-process stub; in `adapter_mode: "production"` the real HTTPS client is used and **must never** fall back to the fake client for any reason. Production config cannot select fake — there is no config key that selects the fake client at all; the mode is the only switch. Tests run without the real embed service present.
- Do not invent: an adapter registry, additional adapter modes, a replay-from-disk mode, or a "fake" prefix token.

---

## 11. Audit

Exactly one **JSON line** per request on **stdout**:

```json
{"ts":"<RFC3339 UTC>","request_id":"<effective id>","route":"/v1/ingest","outcome":"ok|error","reason":"<token or \"ok\">","sha256":"<hex or null>","asset_id":null,"faces":0,"clusters_created":0}
```

- `sha256` and `asset_id` are filled only on ingest/identify; `null` on routes without an image body. No vectors, no image bytes, no Authorization header, no token values, no name strings, no path labels in the audit line.
- On success `reason` is the literal string `ok`. Do not invent prefixed variants. Emit even on failure, per section 6b step ordering; if stdout write fails silently, the response body still returns (no fallback file, no retry queue).

---

## 12. Metrics

Second port (default 9190, bound to `127.0.0.1`), no auth. Closed series **only**:

- `vision_index_requests_total{outcome,reason}` — counter.
- `vision_index_latency_seconds` — histogram, standard Prom buckets.
- `vision_index_errors_total{reason}` — counter.
- `vision_index_assets_total` — gauge, count of `assets` rows.
- `vision_index_clusters_total{kind}` — gauge (`kind` ∈ `face|object`).

No identity labels — no names, no paths, no hashes, no labels derived from image content beyond the `reason` token. No per-name or per-cluster-id labels. Do not invent extra series, custom buckets, an exporter, or a second backend.

---

## 13. Tests the build must pass

1. **Config fail-closed**: unknown YAML key → exit nonzero, no listener opens, one stdout fatal JSON line.
2. **Missing secrets fail-closed**: absent `bearer_token_path` or `embed.bearer_token_path` or `embed.base_url` → exit nonzero.
3. **Fake-client production refusal**: no config path can reach the fake client in `adapter_mode: "production"`. Property test asserts the fake client is only constructed under `test` mode.
4. **Auth**: bad token → 401 `auth.invalid_token`; constant-time compare verified; `/healthz` and `/readyz` unauthenticated; every `/v1/*` route requires the token.
5. **TLS-only in production**: `adapter_mode: production` with TLS disabled → exit nonzero.
6. **Body limits, auth, and sniffing order**: an unauthenticated request with Content-Length over 32 MiB → 413 `limits.body_too_large` (token not checked); an unauthenticated request within the limit, or with no Content-Length → 401 `auth.invalid_token` with the body never read or sniffed; a stream exceeding 32 MiB while reading (chunked, or Content-Length lying low) → 413; `.txt` under image field → 415 `format.unsupported`; wrong multipart Content-Type on a real JPEG still succeeds.
7. **Closed route set**: `GET /v1/ingest`, `POST /healthz`, `PATCH /v1/name` → correct 404/405 tokens.
8. **Idempotent re-ingest**: same image twice → same `asset_id`, `created:false` on second call, face row count unchanged, embed client called once.
9. **Clustering deterministic**: fixture images A (one face), B (same person), C (different person) under the fake client → A and B join one cluster, C forms a second; re-run in a fresh database yields identical cluster boundaries and ids (1 and 2).
10. **Threshold respect**: a synthetic face vector at cosine distance 0.41 from an exemplar creates a new cluster; at exactly 0.40 it joins; nearest-exemplar rule enforced by a fixture where two clusters are within threshold.
11. **Naming conflict**: name a face cluster "alice", then attempt to name another face cluster "alice" → 409 `name.conflict`; naming an object cluster "alice" → 200; naming an object instance via `{asset_id, object_index, name}` creates an object-kind cluster when `objects.cluster_id` is null and sets the name on it; naming via `{asset_id, face_index, name}` resolves by zero-based index in the stored faces ordered by face id ascending.
12. **Merge semantics**: merge cluster 1 into 2 → faces moved, `merged_into` pointer set, `GET /v1/assets` shows `cluster_id: 2`, `GET /v1/clusters` never lists 1, the merged id resolves through the pointer; merging an object cluster → `cluster.not_face`; merging a merged cluster → `not_found`.
13. **identify writes nothing**: run identify, then assert the SQLite file's row counts and WAL are byte-identical to before the call; `GET /v1/clusters` unchanged.
14. **Object instance rule**: after ingest with objects, `objects.cluster_id` are all null; only `/v1/name` on an object cluster sets one; no object clustering exists in the code path (code-search assertion).
15. **Scene search**: ingest two distinct images, nearest-neighbor query returns them ordered by distance ascending, K=10 cap respected, results are asset ids not people.
16. **Audit shape**: every request yields exactly one stdout JSON line with the nine keys of section 11; success reason is literally `ok`; stdout never contains "Bearer", any name string on a `/v1/name` call, any 512/768-length float array, or image bytes.
17. **Metrics closed set**: exactly the five series of section 12, no label value that parses as 64-hex or a float array, no name or path label ever.
18. **Schema strictness**: every response validated with `additionalProperties: false`; any invented extra key fails the validator.
19. **Reason-token closed set**: property test asserts exactly the 16 tokens of section 6c appear in code and no others.
20. **Read-only rootfs + non-root + uid**: container runs `--read-only` with tmpfs `/tmp`, db dir rw, process UID 65532, fixture ingest round-trip succeeds.
21. **Embed down**: stop the embed service, `POST /v1/ingest` → 502 `embed.unavailable`; `/readyz` → 503 `not_ready`; no partial rows written.
22. **No OpenViking**: code search asserts zero references to OpenViking anywhere in the codebase.
23. **Object-label search**: ingest a fixture containing a known COCO label, then `GET /v1/search/objects?label=<that label>` returns the fixture's asset id in `asset_ids`; a different valid COCO label that matches no stored row → 404 `not_found`. The response validates against the search/objects schema with `additionalProperties: false`.

### 13a. Done gates

1. All 23 tests of section 13 pass.
2. `GET /healthz` 200 within 5 s of start; `/readyz` 200 once SQLite opens and embed reports ready.
3. A fixture ingest with one face and one object produces `face_count == 1`, a face cluster of id 1, and an object row with null `cluster_id`.
4. Restart test (kill -9 then restart): no data loss; re-ingest of the same sha256 returns `created:false` and unchanged cluster assignments.
5. A code search asserts no InsightFace/onnxruntime imports, no image-byte writes, no OpenViking client, no second database engine anywhere outside tests.
6. `config.example.yaml` exists matching section 4 exactly.

---

## 14. Out of scope (v1)

GUI, MCP, OpenViking (read or write, in any direction), CUDA or any GPU, multi-tenancy, public internet exposure, storing images or image bytes, running InsightFace or any ONNX model in this process, face-keypoint storage, cluster split or unmerge, object auto-clustering, DELETE of assets, pagination, batch endpoints, video, audio, name mirroring to any other system, rate limiting, mTLS, background jobs, replication. Do not invent any of these; if a needed behavior is absent, stop and ask.

---

## 15. Feature register

| ID | Feature (one sentence, for critic scoring) |
|---|---|
| FI1 | Exposes `GET /healthz` liveness with no auth and no embed or database dependency |
| FI2 | Exposes `GET /readyz` that reports ready only when the SQLite file opens and migrations pass AND the configured embed service reports ready, else 503 `not_ready` |
| FI3 | Exposes `POST /v1/ingest` accepting multipart field `image` with optional caller-supplied `path` label stored but never read from disk |
| FI4 | Exposes `GET /v1/assets/{id}` returning the full asset record with scene vector, faces, cluster ids, names, and objects |
| FI5 | Exposes `GET /v1/search/person?name=` resolving exact names to the face clusters carrying that name and returning every face row in those clusters (face_id, asset_id, cluster_id, bbox) with asset ids, and no cosine distance |
| FI6 | Exposes `POST /v1/identify` that runs embed plus nearest-cluster and nearest-scene matching and persists nothing |
| FI7 | Exposes `POST /v1/name` accepting exactly three body forms — `{cluster_id, name}`, `{asset_id, face_index, name}` (zero-based index in stored faces ordered by face id ascending), and `{asset_id, object_index, name}` (zero-based index in stored objects ordered by object id ascending, creating an object-kind cluster on a null cluster_id) — unique per kind, with `name.conflict` on clashes |
| FI8 | Exposes `POST /v1/merge` merging two face clusters into one via a `merged_into` pointer that queries follow |
| FI9 | Exposes `GET /v1/clusters?kind=&unnamed=` filtered cluster listing |
| FI10 | Enforces a closed endpoint set of exactly the ten routes of section 6a (`/healthz`, `/readyz`, ingest, assets, search/person, search/objects, identify, name, merge, clusters) with 404/405 reason tokens for anything else |
| FI11 | Enforces the closed 16-token reason set in one uniform error envelope with `additionalProperties: false` schemas |
| FI12 | Enforces bearer-token auth on every `/v1/*` route only, comparing in constant time, with `/healthz` and `/readyz` unauthenticated |
| FI13 | Calls vision-embed over HTTPS at a config-supplied base URL with a config-supplied token file path, never hardcoded, sending `include_object_vectors=true` on every call |
| FI14 | Consumes only the fields vision-embed's success schema returns (sha256, scene 768-d, faces 512-d with bbox and det_score, object label/bbox/score and optional crop vector) and no others |
| FI15 | Stores no face keypoints, no image bytes, and no data vision-embed does not return |
| FI16 | Stores vectors via sqlite-vec in SQLite float[512]/float[768] columns in a single WAL-mode database file |
| FI17 | Implements the assets, faces, face_vec, clusters, objects, object_vec, scene_vec tables exactly as section 7 defines, including the unique (kind,name) partial index |
| FI18 | Clusters faces with cosine distance to exemplar ≤ `face_match_max` (default 0.40) AND nearest-exemplar rule, creating a new unnamed cluster otherwise |
| FI19 | Pins the exemplar to the first face of a cluster and never changes it, including across merges; the surviving cluster keeps its exemplar |
| FI20 | Never auto-clusters object instances; `objects.cluster_id` stays null until `/v1/name` names the instance via the `{asset_id, object_index, name}` form, which creates an object-kind cluster on a null cluster_id and then sets the name |
| FI21 | Searches scene vectors by nearest-K cosine distance returning assets, not people, capped at K=10 |
| FI22 | Stores one scene vector per asset and never per face or per cluster |
| FI23 | Computes sha256 of the image bytes and looks up assets.sha256 before calling vision-embed, so re-ingest of the same sha256 skips the embed call and the write and returns the existing asset with `created:false` |
| FI24 | Never updates the `path` label on re-ingest (first label wins) |
| FI25 | Rejects any name clash per kind with 409 `name.conflict` while allowing the same name across kinds |
| FI26 | Sets `cluster_id` on merge by reassigning faces, setting `merged_into` on the source cluster, and resolving merged ids through the pointer on all subsequent reads |
| FI27 | Rejects merge of object clusters or already-merged clusters with `cluster.not_face` / `not_found` |
| FI28 | Persists nothing on `/v1/identify` — no asset rows, face rows, vectors, or cluster mutations of any kind |
| FI29 | Uses the image sha256 as the idempotency key with no cache, TTL, ETag, or Idempotency-Key handling |
| FI30 | Emits exactly one audit JSON line per request on stdout with the nine keys of section 11 and never a vector, byte, token, name, path, or Authorization value |
| FI31 | Exposes five metrics series on a second unauthenticated port with no content-derived or identity labels |
| FI32 | Stores tokens and the embed token as file paths read once at startup, never env vars or inline config values |
| FI33 | Treats unknown config keys, disabled TLS in production, or missing secret file paths as fatal startup errors writing one JSON fatal line and exiting 1 |
| FI34 | Provides a fake embed adapter for `adapter_mode: "test"` only, unreachable in production by any config path |
| FI35 | Runs as uid 65532 with read-only rootfs, tmpfs /tmp, a single rw mount for the database directory, and ro-mounted secrets |
| FI36 | Runs a TLS-only listener with cert/key file paths in config and no public-internet exposure responsibility beyond the operator's firewall |
| FI37 | Rejects before body parse in the fixed order Content-Length pre-check (over 32 MiB → 413 `limits.body_too_large` even when unauthenticated) → auth (missing or wrong token → 401 `auth.invalid_token`, body never read) → body read within 32 MiB → magic-byte sniff → field validation → sha256 dedupe lookup → embed (skipped on dedupe hit) → transaction (skipped on dedupe hit) → schema-validate → audit → body, each step's own reason token; identify always calls embed and never writes |
| FI38 | Uses an effective request_id (echoed X-Request-Id when valid, else uuid4) identically in success, error, and audit records |
| FI39 | Stores the exact integer bbox and det_score from vision-embed verbatim without recomputation or rescaling |
| FI40 | Performs ingest writes only inside the single transaction opened after a dedupe miss (rolled back atomically on failure), uses one single transaction each for `/v1/name` and `/v1/merge`, and writes nothing on identify |
| FI41 | Exposes `GET /v1/search/objects?label=` matching the stored COCO-80 label exactly (no fuzzy search, no cosine distance), returning matching object rows (object_id, asset_id, score, bbox, nullable cluster_id) and deduplicated asset ids, with 422 `decode.failed` `invalid label` for a label outside the 80 COCO names and 404 `not_found` when no row matches |

## 15a. Reason-token coverage (map)

- `limits.body_too_large` → FI37
- `format.unsupported` → FI37
- `decode.failed` → FI37
- `path.too_long` → FI3, FI37
- `not_found` → FI4, FI8, FI27
- `route.not_found` → FI10
- `method.not_allowed` → FI10
- `auth.invalid_token` → FI12
- `name.conflict` → FI7, FI25
- `name.too_long` → FI7
- `cluster.not_face` → FI8, FI27
- `embed.unavailable` → FI13, FI37
- `no_face` → FI7
- `db.error` → FI40
- `not_ready` → FI2
- `response.validation_failed` → FI11, FI37

## 16. Open decisions

None blocking. All thresholds, defaults, and schemas above are pinned so a oneshot build does not stop. Defaults that may be revisited only by a new spec revision: `face_match_max` (0.40), K for scene search (10), name max length (64), metrics port (9190), listen port (9443), embed timeout (60 s).

---

*End of SPEC — DRAFT v4.*
