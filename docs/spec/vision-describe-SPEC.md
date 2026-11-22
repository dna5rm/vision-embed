# vision-describe — Build Contract SPEC

**Status: DRAFT v12**
Changelog: v10 executive decision (closed): a new route `GET /v1/runtime` is added for operator inventory — enabled endpoints, loaded models, and runtime info — placed in this document because it inventories all six roles of the one process. `/readyz` and `/healthz` stay unauthenticated probes and are unchanged: `/readyz` remains `{status, models_loaded}` where `models_loaded` is the count of roles actually loaded, and **no model names, switch flags, versions, GPU memory, driver version, nvidia-smi output, or CUDA-vs-CPU field is added there**. `GET /v1/runtime` runs on the same listener, same bearer as the three POST routes — unauthenticated calls get the existing 401 `auth.invalid_token`; **it cannot be disabled** (no switch exists for it, like `/healthz`, `/readyz`, `/v1/embed`); it loads no model and does not change readiness. The closed endpoint set becomes exactly { `/healthz`, `/readyz`, `/v1/embed`, `/v1/describe`, `/v1/tag`, `/v1/runtime` } in both specs, and `X-Request-Id` is accepted on that same six-route set. The response is JSON with `additionalProperties: false` and exactly these fields, and no others: `gpu` (object, added by the v10 amendment changelog below and specified in section 6d), `uptime_seconds` (integer, seconds since process start), `execution_provider` (the configured provider string, one of the two allowlist values of section 2d), `routes` (object with exactly `embed` — always `true`, `describe` — the `routes.describe_enabled` startup switch value, `tag` — the `routes.tag_enabled` startup switch value), and `models` (array, one object per role this process knows, in the fixed order face, face_detector, scene, object, florence2, tagger; each object exactly `{role, model_id, loaded}` — amended per v12 below to carry one further field, `unavailable_reason`; `loaded` is `false` when the role is not loaded, including a role behind a `false` startup switch; `model_id` is the short id that role already uses in its responses, or the empty string when the role is not loaded). No filesystem paths, weight filenames, tokens, or image bytes are returned. No driver version, no nvidia-smi, and no CUDA-equals-CPU or CUDA-vs-CPU field appears on this route; the one closed `gpu` object of section 6d (v10 amendment) is the route's only device-derived data, `nvidia-smi` remains banned as a startup probe, and this route is not a CUDA fail-closed gate — startup still uses the existing onnxruntime and transformers placement checks of section 2d. (The original v10 sentence forbidding GPU memory on this route is superseded by the v10 amendment changelog below.) Audit: one JSON line, same privacy rules as section 10, `"route": "/v1/runtime"`, no request or response body content. No second port, no new reason token — auth failures reuse `auth.invalid_token`, and unknown methods on the route reuse the existing 404/405 tokens. No person is named. Do not commit.
Changelog: v1 initial contract. Florence-2 describe endpoint plus a separate WD14 tagger endpoint.
Changelog: v2 closes the three critic fails — (1) Florence-2 runtime contradiction removed (loaded via transformers processor/model from an operator-supplied local directory, not onnxruntime, no pure-numpy preprocessing pin), (2) tagger mapping closed via an operator-supplied ordered tag-name list file plus a suppress file, no rating-block exposure question left open, (3) all personal handle references replaced by the non-personal class name "WD14 Danbooru ONNX tagger family".
Changelog: v3 closes the v2 critic pass's parent rejection: (1) the section 2d startup order is split by runtime — the onnxruntime preflight, session provider list, and `get_providers()[0]` activation check apply only to the tagger; Florence-2 has its own gate (transformers model placed on CUDA before bind, or startup is fatal) and onnxruntime is not the ground truth for it; (2) section 3's base image pin is the CUDA base named in `vision-embed-cuda-SPEC.md` so the fail-closed CUDA path can actually run — one image, no second Dockerfile, Python stays 3.12.
Changelog: v4 records the base-image digest the parent fetched from the registry: the section 3 pin is now the full `nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924` reference — the registry Docker-Content-Digest of the multi-arch manifest list for that tag, fetched by the parent on 2026-11-21.
Changelog: v5 removes the either-or on task errors.
Changelog: v6 owner decision: one container. This document is an **additive contract for vision-embed** (same pattern as `vision-embed-cuda-SPEC.md`): `/v1/describe` and `/v1/tag` are routes on the existing vision-embed process, container, image, listener, bearer file, and TLS config. Three v1 routes on the one service: `/v1/embed`, `/v1/describe`, `/v1/tag`. No second process, no second container, no second image, no second Dockerfile, no second port, no second auth file, no second metrics server.
Changelog: v7 closes the three v6 critic fails across both specs. (1) The embed spec's stale CPU-only provider lines (INV-3, section 2 runtime pin, section 4 rule, test 19, F-lines, out-of-scope) are amended there to the same two-value `execution_provider` allowlist this spec already fixes in section 2d: `CPUExecutionProvider` default, `CUDAExecutionProvider` opt-in and fail-closed; no tolerance is invented and `vision-embed-cuda-SPEC.md` is not deleted. (2) There is **one** `config.example.yaml` — the embed spec's section 4 example, now carrying the additive `models` keys below; section 4 of this document is an additive fragment, not a second `config.example.yaml`. (3) `/readyz` means the same thing in both specs: ready only when every model role this process loads is loaded — the four embed roles plus Florence-2 plus the tagger. No personal names. Do not commit.
Changelog: v7 amendment (status unchanged at DRAFT v7) — the embed spec's `/readyz` example and `vision_embed_models_loaded` gauge (0..4) were amended there to 6: `{"status":"ready","models_loaded":6}` and gauge (0..6) in `adapter_mode: "production"`, covering the four embed roles plus this spec's two roles (Florence-2 and the WD14 tagger); fake+test still reports ready with zero ONNX files. The embed spec's "more than four adapter roles" ban was amended there to a fifth-embed-Protocol ban: this spec owns the FlorenceDescribe and Tagger adapter roles, and they are no longer phrased as forbidden. The `vision_describe_models_loaded` gauge of section 11 (0..2) is unchanged — it counts this spec's two roles on the same readyz condition. No commit.
Changelog: v8 closes the six cross-spec contradictions against `vision-embed-SPEC.md` DRAFT v8 for the one-container contract: (1) the embed spec now pins the same base image and the same digest this contract already uses; (2) one metrics server whose closed series set is the four embed series plus the four describe series; (3) the process-wide closed reason-token set is the ten tokens (nine embed-scope tokens plus `task.unsupported`, governable only by this spec's routes); (4) audit scope is split: embed spec section 9 governs `POST /v1/embed` only, this spec's section 10 line governs `POST /v1/describe` and `POST /v1/tag` — no single line shape is required for every POST; (5) the one bearer token is required on all three POST routes (`/v1/embed`, `/v1/describe`, `/v1/tag`), with `/healthz` and `/readyz` unauthenticated; (6) `X-Request-Id` is accepted on exactly { `/healthz`, `/readyz`, `/v1/embed`, `/v1/describe`, `/v1/tag` } — the same sentence in both specs. No personal names. Do not commit.
Changelog: v9 owner decision (closed): startup switches for the two additive routes — two boolean config keys `routes.describe_enabled` and `routes.tag_enabled`, both **default `false`**. `/healthz`, `/readyz`, and `POST /v1/embed` cannot be disabled; no switch exists for them. When a key is `false`: that model is not loaded, its weight/mapping/suppress files are not required and not mount-checked, its role is not counted in `models_loaded` and not required by the ready gate, and a request to that route returns 404 with reason token `route.disabled` — added to the process-wide closed reason-token set (now eleven tokens); it is not `task.unsupported` and not a generic `route.not_found`. When a key is `true`: the current fail-closed rule stands unchanged (missing weight file fatal before bind; the role counts toward ready). `models_loaded` is the count of model roles this process actually loads — never a hardcoded 6 (both switches `false` → 4, both `true` → 6, one `true` → 5); the fake mode exception stays as already written. Metrics series for a disabled route are absent (not stuck at zero); the one metrics server's process-wide closed series set is the four embed series plus the describe series of section 11 **only for each enabled route**. Unknown config keys are still fatal; missing keys are not fatal because the defaults apply. The switch is read at startup only — changing it requires a restart; no second process, no plugin system, no runtime enable is invented. The section 4 example shows both keys set to `false`, so the live embed behavior is the default. Do not commit.

Changelog: v9 amendment (status unchanged at DRAFT v9) — closes the three defects of the parent's rejected v9 critic pass: (1) done gate 3 now states the fixture POSTs run with that route's startup switch set to `true` — the default `false` config is not the config under which that gate runs; (2) the feature register gains F34 for the `routes.describe_enabled` / `routes.tag_enabled` startup switches (disabled = not loaded, not counted in `models_loaded` or the ready gate, 404 `route.disabled`, metrics series absent; enabled = missing weight/mapping/suppress file fatal before bind), `route.disabled` is added to the section 14a coverage map, and F2, F23, and F29 are scoped so the readyz coverage of the two describe roles and the existence of the four describe metrics series hold only when the route is enabled. No new routes, no personal names. Do not commit.

Changelog: v10 amendment (status unchanged at DRAFT v10) — closes the parent's v10 critic fail (DRAFT v10 rejected at 60): the embed spec's **bodies** — INV-8, section 5 (bearer on `GET /v1/runtime`), section 6 endpoint list, its no-other-routes line, the X-Request-Id sentence, test 4, test 7, and feature F36 — now list the same six-route closed set { `/healthz`, `/readyz`, `/v1/embed`, `/v1/describe`, `/v1/tag`, `/v1/runtime` } and the same bearer-on-runtime rule, instead of carrying `/v1/runtime` in a changelog note only; F3 of section 14 here now lists the six routes. Status stays DRAFT v10 in both specs. No new fields, no `/readyz` change, no personal names. Do not commit.

Changelog: v10 amendment (executive decision; status unchanged at DRAFT v10) — a closed object `gpu` is added to the `GET /v1/runtime` response (section 6d, `additionalProperties: false`, and no other fields on `gpu`): `available` (boolean — true **only** when the NVML library call opens and reports at least one device; false if NVML is missing, permission is denied, or the device count is zero — and that false is **not** an HTTP error: the route stays 200), `device_count` (integer; 0 when `available` is false), `name` (string — device 0's name when `available`, else the empty string; no other device's name is ever returned), `memory_total_bytes` (integer, or **null** when `available` is false; 0 is never written to mean unknown), and `memory_used_bytes` (integer or null, same rule). The read is an NVML library call, never a shell-out to `nvidia-smi`; only device index 0 is described even when `device_count` is greater than 1 — no device picker and no per-device array is invented. A failed GPU read does not change `/readyz`, does not unload a model, and adds no reason token. The prior v10 sentence that forbade GPU memory on this route is deleted; the bans on filesystem paths, tokens, and on `nvidia-smi` as a startup probe stand, and this GPU read is **not** the CUDA fail-closed gate — startup provider placement is still decided by the onnxruntime and transformers checks of section 2d. No `/readyz` change, no personal names. Do not commit.

Changelog: v11 owner decision (closed): **this process runs at most one GPU inference at a time when `execution_provider` is `CUDAExecutionProvider`.** One in-process lock; `POST /v1/embed`, `POST /v1/describe`, and `POST /v1/tag` share it. A request that needs the GPU while one inference is already running **does not start and does not wait** — no queue, no retry, no wait, the caller retries — and returns **HTTP 429 with reason token `gpu.busy`** — the twelfth token of the process-wide closed set, which now counts exactly **twelve** tokens (the eleven prior tokens plus `gpu.busy`; the prior count of eleven is verified in this file's v9 and v10 text before this line). `gpu.busy` is not `route.disabled` and not a 500, and it is never mapped from anything else: a request that **holds** the lock and then runs out of memory is still 500 `inference.failed` — no second OOM token is invented. `GET /healthz`, `GET /readyz`, and `GET /v1/runtime` do not take the lock and are never rejected `gpu.busy`. When `execution_provider` is `CPUExecutionProvider` there is no lock and no `gpu.busy`. **The lock is in-process only and this process's own utilization is the only thing it governs**: the owner confirmed other processes on the GPU are ignored — do not read NVML, no `nvidia-smi`, do not look at other processes, do not invent a memory threshold, do not use the `gpu` object of section 6d as an admission check (it stays a read-only inventory echo). The `/v1/runtime` `gpu` object is unchanged. Startup fail-closed CUDA checks of section 2d are unchanged. No personal names, no URL, no tolerance value anywhere in this amendment. Do not commit. This spec is amended in its bodies (processing order, token table, tests, feature register) and is the authoritative document; `vision-embed-SPEC.md` carries an additive changelog note only (its status stays DRAFT v10).

Changelog: v12 owner decision (closed): a **startup free-memory gate** protects `/v1/embed` from the describe/tag CUDA loads — a route without enough free GPU memory is **not loaded and behaves exactly as disabled**, and **the process never exits for this gate**. Two new config keys `routes.describe_min_free_bytes` and `routes.tag_min_free_bytes` — integers measured in **bytes**, **no default** — are required **only when that route's switch (`routes.describe_enabled` / `routes.tag_enabled`) is `true` AND `models.execution_provider` is `CUDAExecutionProvider`**; missing in exactly that case is a fatal `config.invalid` before bind, and no byte value is invented anywhere here. The check applies only in that state: under `CPUExecutionProvider` it never runs, and a `false` switch never runs it and never required the key. At startup, **after the fail-closed CUDA provider gate of section 2d has passed** (this gate is not that gate; a non-activating provider still exits the process exactly as already specified) and **before either model is loaded**, the service reads free GPU memory **once** via the **same NVML library call the `/v1/runtime` `gpu` object uses** — free = `memory_total_bytes` − `memory_used_bytes` on device index 0, never a shell-out to `nvidia-smi`. The comparison runs in fixed order **describe, then tag**: a route passes when its operator minimum is ≤ the free bytes still remaining after subtracting the minima of routes already accepted — **only operator minima are subtracted, never a guessed model size** — and a route that fails is not loaded: no weight required, not in `models_loaded`, not required by the ready gate, its metrics series absent, and requests to it return **404 `route.disabled`** (no new HTTP reason token; the process-wide closed set stays at twelve). If **NVML cannot be read**, no value is guessed and no default applies: every CUDA-enabled describe/tag route that required the check is disabled exactly as above, and the process still binds. `/v1/embed` and the four embed roles are unaffected in every case. `/v1/runtime` (section 6d) gains exactly one field on each `models[]` object: **`unavailable_reason`** (string, empty when the role is loaded) with the closed values `""`, `"disabled"` (switch false), and `"gpu.insufficient"` (this gate failed the route, including the NVML-unreadable case) — inventory only, no HTTP meaning. The section 6e GPU `gpu.busy` lock is unchanged and still reads no memory at request time. The v11 sentence that forbade inventing a memory threshold governs only the v11 admission lock (nothing device-derived is consulted at request time); this gate's threshold is **operator-set, startup-only, and owner-decided**, not invented. Amended in the bodies (sections 4, 6, 6d, 6e, 12, 12a, 14, 14a). No personal names, no invented byte default. Do not commit.

Changelog: v12 amendment (status unchanged at DRAFT v12) — section 2e is amended to the **load-once design the implementation now uses**: the order at startup is config validation, then the section 2d activation gate, then **each enabled describe/tag role is loaded exactly once — that single load IS the section 2d activation proof** (a failed load is fatal `config.invalid` before bind); **no model is loaded twice.** The 2e free-memory read then happens **with those role models resident**: free bytes are reported by NVML total minus used, and **the residency of each already-loaded role is added back before the compare, using the measured delta between the free reading taken before that role's load and the free reading taken after it** — only that measured delta is added back, and **no model-size estimate is invented**. The compare then runs in fixed order **describe, then tag** against the operator minima, and a route that fails is **unloaded and released** and behaves exactly as disabled (not in `models_loaded` or the ready gate, metrics series absent, requests 404 `route.disabled`, `unavailable_reason` `gpu.insufficient`) — **the process still binds.** A failed or unavailable NVML read remains the soft-disable rule, not a guess. The section 2d activation failure remains fatal regardless of free memory, and no sentence of the spec now says the gate runs before either model is loaded. Amended bodies: sections 2e, 6d, 6e, and the `vision-embed-SPEC.md` v12 changelog note (status unchanged there). No personal names, no invented byte value. Do not commit.

**Product owner: the service owner (me).** This document is the complete build contract. An implementer builds exactly what is specified here. Every behavior not in this contract has an explicit "do not invent" instruction; if a needed detail is absent, the implementer MUST stop and ask, not fill the gap.

Companion documents (read, never edit):
- `docs/spec/vision-embed-SPEC.md` — the service these routes live on. `/v1/describe` and `/v1/tag` are **routes on vision-embed**, an additive route contract approved by the owner decision recorded in the v6 changelog; they are NOT a sibling service. This document extends vision-embed's closed endpoint set with exactly these two routes and edits nothing else in that spec — except the v7 amendments, which fix lines of that spec that contradicted this document or the CUDA spec: the two-value `execution_provider` allowlist, the `/readyz` readiness condition, and the section 4 `config.example.yaml`, which is the one config example for the one container and now carries the additive `models` keys of section 4 below.
- `docs/spec/vision-embed-cuda-SPEC.md` — the fail-closed CUDA pattern (executive locks, startup order, no numeric envelope) that this service's provider handling copies.
- **The index is standalone.** **vision-index MUST NOT call these routes.** No route call between vision-index and `/v1/describe` or `/v1/tag` exists in any direction. These routes make no outbound calls at all.

---

## 1. Status, owner, invariants

- Status: **DRAFT v12**. The executive has decided this contract; every normative pin below is decided and closed. Reopen decisions only via section 16; the section 16 open items are the only unknowns and must not contradict any normative pin.
- Owner: the service owner (me) owns this contract, the config format, the reason-token set, and the audit schema.

Numbered invariants — each is an acceptance gate, not a suggestion:

1. **INV-1 Additive routes, one process.** `/v1/describe` and `/v1/tag` are routes on the existing vision-embed process and container (owner decision, closed). One process, one container, one image, one listener, one bearer file, one TLS config, one bind address. Do not invent a second process, a second container, a second Dockerfile, a second image variant, a second port for these routes, a second bearer file, or a second bind address, and do not specify process isolation or a sidecar between the routes. The v3/v4/v5 "separate process" wording of this document is superseded by this invariant.
2. **INV-2 Stateless.** The server stores nothing. No images, no captions, no tags, no names, no database, no cache across requests, no state across restarts. The service **writes no files for the caller** — no caption files, no `.txt` sidecars, no training datasets, no scratch artifacts. The response body is the only deliverable.
3. **INV-3 No identity, no person knowledge.** The response and the audit line contain no cluster id, no person name, no person reference of any kind. The service knows nothing about any person store or index. vision-index MUST NOT call these routes; there is no caller-relationship concept in these routes.
4. **INV-4 Fail closed.** Unknown config keys are a fatal startup error. A missing weight file is a fatal startup error — never a 200 with empty text and never an inference on a placeholder. Unknown routes are 404 with a reason token.
5. **INV-5 One process.** No background workers, no queue, no sidecars, no scheduler.
6. **INV-6 Auth minimal and owned here.** One static bearer token loaded from a file path in config, compared in constant time — the **same single bearer file and the same single auth config already used by the embed routes** (`vision-embed-SPEC.md` section 4 / section 5); no second auth file and no second bearer token are created for these routes. No OAuth, no IdP, no mTLS in v1 (listener is still TLS-only). There are no vendor credentials anywhere in this service; no model download, so no Hugging Face token.
7. **INV-7 Audit-only observability of content.** Never log image bytes, captions, tags, OCR text, Authorization headers, or tokens. The audit record is exactly one JSON line per request on stdout (section 10). Metrics carry no content-derived labels (section 11).
8. **INV-8 Closed surfaces.** The service's closed endpoint set is exactly { `/healthz`, `/readyz`, `/v1/embed`, `/v1/describe`, `/v1/tag`, `/v1/runtime` } — the existing embed trio unchanged, the two routes this document specifies for Describe/Tag, and the one runtime-inventory route the v10 executive decision adds (section 6d). Do not add a `/status` route or any other endpoint. Closed reason-token set (section 7), closed request/response schemas with `additionalProperties: false` on every object.
9. **INV-9 Network posture.** The single existing TLS-only listener (`vision-embed-SPEC.md` section 3; the same TLS cert/key config, not a second one). The operator firewalls the listener off the public internet. No egress: the service performs no runtime model download, no fetch-on-start, and no external call of any kind in the code path.
10. **INV-10 Two model roles, nothing more.** Exactly two model roles exist: the Florence-2 describe model and the WD14 tagger. Do not invent a second tagger, a caption post-processor, an NLI checker, a translation step, or a chaining of tasks inside one request.

Do not invent: a database, a photo store, an identity graph, a person name, a cluster id, a warm-pool of workers, a retry queue, OAuth/jwt/session machinery, an admin UI, a model-download step at runtime, a second tagger, a Florence-2 task outside the closed set, or any endpoint beyond the closed set.

---

## 2. Stack (pinned)

| Component | Pin |
|---|---|
| Language | Python **3.12** (exact minor) |
| ASGI framework | **FastAPI** (one ASGI app, served by `uvicorn` with `--workers 1`) |
| Describe model | **Florence-2**, loaded through the **transformers processor and model** from an **operator-supplied local directory** (weights + processor files). Pin the exact configured directory; never a floating tag, never "latest". **Not executed via onnxruntime; no ONNX export is part of this role.** |
| Tagger model | **WD14 Danbooru ONNX tagger family**, executed via onnxruntime as an **operator-supplied ONNX file**. Pin the exact configured file name; never a floating tag. |
| Tag vocabulary | WD14 / Danbooru-style tags as emitted by the WD14 Danbooru ONNX tagger family. |
| Runtime | The **two roles do not share a runtime**: the tagger runs on **onnxruntime** (`CPUExecutionProvider` default; `CUDAExecutionProvider` opt-in and fail-closed, section 2d); Florence-2 runs through **transformers** (section 2a). Do not merge them into one runtime. |

### The WD14 vocabulary, stated plainly

The WD14 / Danbooru vocabulary is built for **anime/style** images and the caption-file workflows around them. It is the right vocabulary for producing anime/style LoRA caption files. It is the **wrong vocabulary for ordinary photographs** — tags will be skewed toward anime-styled classes for many real-world photos, and this service does not correct, translate, filter, or re-weight them. The response carries tags with their scores, cut at the caller-supplied threshold, and nothing else. The caller owns any post-processing.

### Determinism

- Default provider is `CPUExecutionProvider` and this is a hard v1 default. `CUDAExecutionProvider` may be configured but is **opt-in and fail-closed** (section 2a/2d), following `vision-embed-cuda-SPEC.md` in shape: for the tagger, startup is fatal if the provider is not registered or does not become the active provider; for Florence-2, startup is fatal if the model cannot be placed on the CUDA device — **no silent CPU fallback, ever, for either role.**
- **No numeric CPU-vs-CUDA equality claim is made anywhere in either direction.** This service copies the CUDA spec's lock: no tolerance envelope is invented, no claimed tolerance value exists, and per-element numeric equality between providers is not certified. Do not claim CUDA equals CPU. Do not invent a numeric tolerance.
- Same image bytes + same pinned model files ⇒ same outputs on the CPU provider. Determinism claims in this spec apply to CPU-only runs.
- Florence-2 preprocessing and tokenization are whatever the operator-supplied processor files bundled in the configured directory do (section 2a). **Do not reimplement them; do not require a pure-numpy preprocessing pipeline for Florence; do not invent an input resolution or a mean/std normalization pair.**
- Florence-2 decoding is deterministic: `do_sample` is `False` and `num_beams` is `1`, always, for every task.
- Decode length is bounded by the config key `models.max_new_tokens`, a positive integer with an **operator-set default of 1024 — an operator default chosen by this service's owner, not a model-card fact** (no token-count claim is cited here because no source page was opened in this task).

## 2a. /v1/describe — Florence-2 tasks (closed set)

`POST /v1/describe` runs exactly one Florence-2 task per request. The caller names exactly one task. **No chaining inside the request** — the request carries exactly one task value, and the server never runs a second task to feed the first.

The closed task set (caller-facing name → wire value, using the Florence-2 prompt tokens with angle brackets **as the literal wire values**):

| Task name | Wire value (exact string on the wire) |
|---|---|
| `CAPTION` | `<CAPTION>` |
| `DETAILED_CAPTION` | `<DETAILED_CAPTION>` |
| `MORE_DETAILED_CAPTION` | `<MORE_DETAILED_CAPTION>` |
| `OD` | `<OD>` |
| `DENSE_REGION_CAPTION` | `<DENSE_REGION_CAPTION>` |
| `REGION_PROPOSAL` | `<REGION_PROPOSAL>` |
| `OCR` | `<OCR>` |
| `OCR_WITH_REGION` | `<OCR_WITH_REGION>` |

Rules:
- **Runtime mechanics (decision, closed):** `/v1/describe` loads Florence-2 through the **transformers processor and model** from the operator-supplied local directory configured by `models.florence_dir_path` (containing the weights and the processor files). **No runtime download happens and no Hugging Face token is used or accepted.** The task wire values of the table above are passed **as the processor text prompt** — that is the published Florence-2 interface this service relies on. Preprocessing and tokenization are whatever those bundled processor files do; do not reimplement, replace, or pin them numerically. Decode is deterministic: `do_sample: False`, `num_beams: 1`, bounded by `models.max_new_tokens` (section 2).
- All eight tasks are **image-only** tasks: no caller text input, no caller box input. The ONLY valid caller input to the model is the image itself.
- Every other Florence-2 task is **rejected at request validation, before any inference**, with reason token `task.unsupported` (HTTP 422) — because those tasks require caller text or a caller box, which this service does not accept:
  - CAPTION_TO_PHRASE_GROUNDING (phrase grounding — needs caller text)
  - OPEN_VOCAB_DETECTION / FLORENCE_DETECTION-style open-vocabulary detection with caller text classes (needs caller text)
  - REFERRING_EXPRESSION_SEGMENTATION (needs caller text)
  - REGION_TO_SEGMENTATION (needs a caller box)
  - REGION_TO_DESCRIPTION (needs a caller box)
- Do not use custom/extended prompt tokens (e.g. `<CAPTION_TO_PHRASE_GROUNDING>` with extra arguments, `<REFERRING_EXPRESSION_SEGMENTATION>` with an expression, region args, or any task key that starts a region or phrase parameter). A task value that parses as one of the rejected families, or any string other than the eight exact wire values, → `task.unsupported`. Task matching is **exact string match** on the wire value, case-sensitive.
- A caller wanting grounding, segmentation, region-to-description, or open-vocab detection must run that elsewhere; this service does not grow those tasks in v1.

## 2b. /v1/tag — the WD14 tagger

`POST /v1/tag` runs the pinned WD14 tagger. It is **a tagger, not Florence**: a completely separate model role and a separate route from `/v1/describe`. It does NOT share the Florence task set, and Florence is not run on this route.

- Input: the image only, plus a caller-supplied threshold.
- Output: a list of `{tag, score}` pairs where `score >= tag_threshold`, sorted by score descending. Tags are the WD14/Danbooru vocabulary strings resolved through the operator-supplied ordered tag-name list file (section 2c).
- The tagger does **not** write caption files, `.txt` sidecars, or training datasets, and it produces no per-tag prompt-weight strings (no `(tag:1.2)` synthesis). The caller constructs any caption file themselves from the response. Do not invent a sidecar-writing mode, an "export as booru tag string" field, or a dataset writer.

## 2c. Tagger weight note (no URL invented)

The WD14 Danbooru ONNX tagger family weights are **operator-supplied files**, mounted or placed at the configured path. The operator records their source and provenance themselves. **This spec does not cite a canonical weight URL** — no model card was opened in this task to verify one — and it does not invent a Hugging Face URL. Do not add a URL, a repo id, a fetch step, or a verify-by-download anywhere in this service.

The same rule applies to Florence-2 weights: an operator-supplied **local directory** (weights + processor files), no runtime download, no fetch-on-start. Do not invent a Hugging Face URL.

Licenses: **the repo code license is MIT; the model weights remain under their own terms as published by their authors.** This spec does not invent, classify, or restate any weight license. No person is named in this document, its config, or its fixtures.

### Tagger output mapping — decided (closed in v2)

The mapping is **closed in v2 by operator-supplied data files; no output-column index is hardcoded or invented here**:

- The operator supplies an **ordered tag-name list file** (`models.tag_names_path`): one tag name per line, in output-column order, and **its length must exactly equal the length of the tagger's score vector** — a mismatch is a fatal startup error (`config.invalid`), because an index without a name cannot appear on the wire.
- The operator supplies a **suppress file** (`models.tag_suppress_path`): tag names, one per line, that are **never returned on the wire** (for example ratings, character names, or anything else the operator does not want exposed). Do not hardcode any rating names or any suppress list in code; the suppress list is entirely operator data.
- The response carries only `{tag, score}` pairs for names that are **not suppressed** and whose score is **>= the caller threshold**, sorted score-descending. There is no rating field, no character field, and no unsuppressed-name escape hatch: suppression is absolute.
- Both files are mounted read-only like the weights (section 3); a missing mapping or suppress file at a configured path is a fatal startup error, same as a missing weight file.

## 2d. Execution provider (fail-closed, copies the CUDA spec)

- Key: `models.execution_provider`, string, under the closed models key set, default `"CPUExecutionProvider"`.
- Explicit two-value allowlist, exactly as in `vision-embed-cuda-SPEC.md` section 3:
  - `"CPUExecutionProvider"` (default)
  - `"CUDAExecutionProvider"` (opt-in)
- Any other string → fatal `config.invalid` at startup.
- **Startup order for a configured `CUDAExecutionProvider`** (all before any listener or metrics port binds):
  1. Config validation: value must be in the two-value allowlist, else fatal.
  2. Tagger provider-registered pre-flight: before loading any weights, `"CUDAExecutionProvider"` must be present in `ort.get_available_providers()`, else fatal `config.invalid` with detail naming the provider. One startup log line on success, and no other provider-related logging. This pre-flight is onnxruntime and therefore **tagger-only**; it says nothing about Florence-2.
  3. **Tagger session gate.** The tagger's onnxruntime session is built with `providers=["CUDAExecutionProvider"]` (single-element list, **no fallback chaining**); if construction raises or `session.get_providers()[0] != "CUDAExecutionProvider"`, startup is fatal. A session that constructs while CPU leads the active list is exactly as fatal as a missing device. **No silent CPU fallback** at config validation, session build, or inference time. onnxruntime's activation check is the ground truth **for the tagger only**.
  4. **Florence-2 gate (separate from onnxruntime).** Florence-2 runs through transformers, not onnxruntime, so onnxruntime's pre-flight and activation check do not govern it. Its gate is placement: if `execution_provider` is `CUDAExecutionProvider`, the transformers model must be placed on CUDA **before any listener or port bind**, and startup is fatal if that placement does not succeed (the model is not on CUDA). No silent CPU fallback, ever. No app-code probing for either role: no `/dev/nvidia*`, no `nvidia-smi`, no driver checks — for the tagger, onnxruntime's activation check decides; onnxruntime is **not** the ground truth for Florence-2 placement.
- Fatal shape: exactly one JSON line on stdout `{"fatal": true, "reason": "config.invalid", "detail": "<why>"}`, exit 1, before any socket opens. No HTTP 503 for a missing GPU. The reason token set does not grow for provider failures.
- The one provider key governs both roles; no per-model provider override — but the roles do **not** share a runtime (section 2). For the tagger it selects the onnxruntime session providers exactly as above. For Florence-2 (transformers) it means **the model must be placed on the CUDA device and all model inputs used at inference must be CUDA tensors**; if that placement or any subsequent inference cannot execute on CUDA, startup is fatal — **no silent CPU fallback**. Do not invent device-id maps, warmup calls, or `TensorrtExecutionProvider`.

## 2e. Startup free-memory gate — describe/tag only (v12 owner decision, closed)

Owner decision (closed; amended within v12 to the load-once design): under a CUDA configuration, each enabled describe/tag role is loaded exactly once — that single load **is** its section 2d activation gate (a failed load is fatal `config.invalid` before bind) — and this service then checks, with those role models resident, that the operator's configured minimum of free GPU memory is available per route. The 2e gate therefore runs **after** the role models are resident, not before either model is loaded. A route whose minimum is not available is **unloaded and released** after the compare — it behaves in every way as a route behind a `false` startup switch (section 4, v9) — and **the process continues, binds, and serves `/v1/embed`: the process does not exit for this gate.** The section 2d activation failure remains fatal **regardless of free memory**. `/v1/embed` and the four embed roles are never gated by it and are unaffected in every case below.

- **Applies only when `models.execution_provider` is `CUDAExecutionProvider` AND that route's startup switch is `true`.** Under `CPUExecutionProvider` the check does not run at all — no memory read, no comparison. A route whose switch is `false` does not run the check, never required its minimum key, and stays disabled for the v9 reason (switch `false`), not for GPU memory.
- **Config keys (v12, closed).** `routes.describe_min_free_bytes` and `routes.tag_min_free_bytes` — integers, measured in **bytes**, **no default**. Each is required **only** when its route's switch is `true` and the provider is `CUDAExecutionProvider`; missing in exactly that case is a fatal `config.invalid` before bind (the section 4 fatal shape: one stdout JSON line, exit 1, no socket). In every other state the key may legally be absent and nothing validates it. **No byte value is invented here** — the value is the operator's, and this spec picks no number.
- **Startup order and when the read happens (v12 amendment).** The order at startup is exactly: config validation, then the fail-closed activation gate of section 2d, then each enabled describe/tag role is **loaded exactly once** — that single load **is** the section 2d activation proof, and **a failed load is fatal `config.invalid` before bind**; no model is loaded twice. The 2e read then happens **with those role models resident** (no model is loaded to satisfy the compare, because every enabled role already has its one load behind it; the residency itself, however, is accounted for per the next rules). This gate is still **not** the CUDA fail-closed provider gate: if `CUDAExecutionProvider` is configured and does not activate, or the single load of an enabled role fails, the process still exits exactly as section 2d already specifies — regardless of free memory — and this memory check runs only after that gate has passed.
- **The read (v12 amendment).** At startup the service reads free GPU memory via the **same NVML library call the `/v1/runtime` `gpu` object uses** (section 6d), on device index 0 only. **Free = `memory_total_bytes` − `memory_used_bytes` on device 0.** Never a shell-out to `nvidia-smi` — it stays banned everywhere in this service, including here. If NVML cannot be read — NVML missing, permission denied, or zero devices — **no value is guessed and no default applies: every describe/tag route that required the check (switch `true` under the CUDA provider) is disabled in exactly the same way as a route that fails the comparison below, and the process still binds.** A failed read is not an exit and not a fatal.
- **The comparison, in fixed order describe → then tag (v12 amendment — residency added back).** A baseline free reading is taken after the section 2d gate, before the loads; in the load-once order each enabled role is then loaded exactly once, and for every role already loaded a free reading is taken again after its load — **the residency of that role is added back before the compare, using the measured delta between the free reading taken before that role's load and the free reading taken after it.** Only that measured delta is added back; **no model-size estimate, no heuristic figure, no byte default of any kind is invented or subtracted — the model-size footprint enters the numbers only through its measured delta.** The compare then runs in fixed order **describe first, then tag**, against the operator minima, using effective free = the latest free reading plus the added-back residency of each already-loaded role (as accumulated by the minima subtracted below): if `routes.describe_min_free_bytes` ≤ effective free, the describe role stays loaded — its earlier single load was its section 2d activation proof — and the operator minimum is subtracted; otherwise the describe role is **unloaded and released** and behaves as disabled below. Then **tag**: if `routes.tag_min_free_bytes` ≤ the remaining effective free, the tag route stays loaded; otherwise it is unloaded and released the same way. The order is fixed: describe first, then tag, always — never tag first, never best-fit, never a combined decision.
- **A route that fails — below its operator minimum in the compare order, or disabled by the NVML-unreadable rule — behaves exactly as disabled (section 4, v9):** its model is **unloaded and released** after the compare and behaves as not loaded, its weight/mapping/suppress files are **not required and not mount-checked as gating failures** (a missing file is then not a fatal, the same as a switch `false`), its role is **not counted** in `models_loaded` and **not required by the ready gate**, and its metrics series are absent (section 11). A request to it returns **404 `route.disabled`** — the existing token, never a new one, never `gpu.insufficient` on the wire, never `task.unsupported` and never a generic `route.not_found`. The process-wide closed reason-token set therefore stays at **twelve** tokens.
- **`/v1/runtime` shows why (v12).** Each `models[]` object of section 6d gains exactly one field: **`unavailable_reason`**, a string, empty when the role is loaded, with the closed values `""`, `"disabled"` (the role's startup switch is `false`), and `"gpu.insufficient"` (the role failed this gate — below its operator minimum, or the NVML read could not be taken so no value was guessed). Nothing else on the route is added or moved, and `unavailable_reason` is inventory only — it has no HTTP meaning.
- **The section 6e GPU lock is unchanged.** No memory value is read at request time; the operator minima of this section are the only memory thresholds in the service and are read once at startup. The `/readyz` shape is unchanged (`{status, models_loaded}`); only the count follows the v9 rule extended by this gate.
- **No exception is invented for fake mode.** The gate runs under exactly the two conditions of the first rule — provider and switch — and nothing else gates or exempts it. No fake-mode carve-out is specified.
- Do not invent: a per-request memory check, a re-read at runtime, a watermark rule in the request path, a second device index, a model-size table, a byte default of any kind, a second load of any role, or any add-back into the compare other than the measured before/after-load delta of a role already loaded.

---

## 3. Container contract

These routes run **inside the existing vision-embed container** (owner decision, closed). There is no second container, no second Dockerfile, and no second image variant.

- **Image: the one image vision-embed already runs.** Base image `nvidia/cuda:12.6.3-runtime-ubuntu24.04`, pinned by digest in `deploy/Dockerfile` under the owner decision recorded there; this document does not name its own digest and does not replace the live embed image pin with a second pin. The same tag is already named in `vision-embed-cuda-SPEC.md` (it pins base images; no different tag is invented here). One image serves all provider paths: the fail-closed `CUDAExecutionProvider` gate must be able to run, and `CPUExecutionProvider` — the default — also runs on this same image. A slim variant cannot satisfy the CUDA gate, so none is used.
- **transformers is an additional dependency of this same image.** The Florence-2 role adds `transformers` (plus its required tokenizer/processing support packages) to the existing pip set in `deploy/Dockerfile`; no other image change is needed for the describe/tag routes. **No `transformers` version pin is stated here** — no version source was opened in this task, so an exact constraint must be resolved from an opened source before the pin is written; absent that, stop and ask (section 16). Do not invent a pin, a floating tag, or a second dependency set.
- Python remains 3.12 (exact minor), as already in the image.
- Runs as non-root uid 65532, group 65532, read-only rootfs with the same tmpfs `/tmp` sizing — unchanged from the embed contract.
- Mounts (ro): the existing embed mounts (embed model files, bearer file, TLS cert/key) are unchanged; these routes **add** the ro model-artifact mounts `models.florence_dir_path`, `models.tagger_onnx_path`, `models.tag_names_path`, and `models.tag_suppress_path` (exact operator paths are config; example section 4). The bearer file and TLS cert/key are the **same single files** the embed routes already mount — `/secrets/authz.bearer` and the existing TLS paths — never environment variables or CLI flags, and never a second bearer or TLS file pair.
- Listener: the **single existing TLS-only listener** (`tls.cert_path`, `tls.key_path` from the existing vision-embed config; bind default `0.0.0.0:8443` per `vision-embed-SPEC.md` section 3, overridden by the operator). **No second listener and no second bind address are added for these routes.** The operator firewalls the public internet out.
- No inbound connectivity is required. No outbound network is required by the code path; outbound egress is denied by container policy.
- Metrics server: the **existing embed metrics server** (config `metrics.port`, default 9090, per `vision-embed-SPEC.md` section 3/10) in the same process, no auth. **No second metrics server or second metrics port is added**; the routes' series on it are the closed set of section 11.
- Do not invent: a systemd unit, a Compose stack with extra services, a sidecar proxy, a second container or Dockerfile, a healthcheck that POSTs to a v1 route, autoscaling config, GPU device maps beyond the section 2d provider mechanics, a public ingress, or a second container image variant.

---

## 4. Config YAML — additive fragment (NOT a second `config.example.yaml`)

**There is exactly one `config.example.yaml` in this one container, and it is the embed spec's section 4 example (`vision-embed-SPEC.md` section 4), which carries the additive `models` keys shown below.** Section 4 here is an **additive fragment** for the describe/tag keys only — a second full config example is not created and does not double as `config.example.yaml`. Everything else in the config (listen, tls, auth, limits, metrics, audit, adapter_mode, request_id) is the embed spec's config, unchanged; the `models.*` keys shown here were added there in the v7 amendments.

```yaml
# Additive fragment — the models.* keys the /v1/describe and /v1/tag routes
# add to the ONE config.example.yaml (vision-embed-SPEC.md section 4).
# Unknown keys are FATAL at startup.
# listen, tls, auth, limits, metrics, audit, adapter_mode, request_id, and the
# embed models.* keys are the embed spec's config, unchanged here.

models:
  # (the four embed models.* keys are unchanged in the embed spec's example)
  florence_dir_path: "/models/florence2/"   # operator-supplied LOCAL DIRECTORY (weights + processor files); exact directory pinned here
  tagger_onnx_path: "/models/wd14.onnx"     # operator-supplied; exact filename pinned here
  tag_names_path: "/models/tag_names.txt"   # ordered tag-name list, one per line; length must equal the score vector length
  tag_suppress_path: "/models/tag_suppress.txt"  # tag names never returned on the wire (ratings, character names, etc.)
  max_new_tokens: 1024                      # OPERATOR DEFAULT for Florence-2 decode length, not a model-card fact

routes:
  describe_enabled: false                   # STARTUP-ONLY switch; changing it requires a restart. false: Florence-2 is NOT loaded,
                                            # its weight/processor files are not required or mount-checked, the role is not counted
                                            # in models_loaded and not required by the ready gate, and POST /v1/describe returns
                                            # 404 route.disabled. true: current fail-closed rule stands (missing weight file fatal before bind).
  tag_enabled: false                        # same semantics for POST /v1/tag and the WD14 tagger. Both keys default to false, so the
                                            # live embed behavior (four embed roles, models_loaded = 4) is the default. There is NO
                                            # switch for /healthz, /readyz, or POST /v1/embed — they cannot be disabled. Unknown keys
                                            # under routes: are FATAL; a missing key is not fatal because its default (false) applies.
  # v12 startup free-memory minima (section 2e). Integers in BYTES, NO DEFAULT.
  # Each key is REQUIRED only when its switch above is true AND
  # models.execution_provider is "CUDAExecutionProvider"; missing in exactly
  # that case is fatal config.invalid before bind. Otherwise it may be absent.
  # describe_min_free_bytes: <operator-set integer>   # shown commented: not required while the switch is false
  # tag_min_free_bytes: <operator-set integer>        # same rule; the value is the operator's — no byte default exists
```

`limits.max_regions` (100; hard cap on `regions[]` returned — beyond cap dropped, outcome stays `ok`) is likewise part of the additive route config and is carried in the embed spec's section 4 example.

Rules:
- **Fatal startup output.** A fatal startup condition writes exactly one JSON line to stdout — `{"fatal": true, "reason": "config.invalid", "detail": "<why>"}` — then exits 1. Startup fatals are **not** HTTP responses.
- Unknown top-level or nested keys → fatal startup error (`config.unknown_key` as a stdout fatal log, exit 1; the server must not come up). `config.unknown_key` is used **only** for YAML keys not in the schema; it is not an HTTP reason token.
- **`routes.describe_min_free_bytes` / `routes.tag_min_free_bytes` (v12 owner decision, closed):** integers measured in **bytes**, **no default**. Each is required **only** when its route's startup switch is `true` **and** `models.execution_provider` is `CUDAExecutionProvider`; missing in exactly that case is a fatal `config.invalid` before bind (the fatal shape above, before any socket opens). In every other state — the provider is `CPUExecutionProvider`, or that route's switch is `false` — the key is not required, nothing validates it, and the section 2e check does not run. The stored value is the **operator's** free-memory minimum for that route; **no byte value, tolerance, or model-size estimate is invented in this spec or in the config example.** The minima are known keys under `routes:` (an unknown `routes:` key is still fatal `config.unknown_key`); a non-integer, negative, or otherwise malformed minimum value is fatal `config.invalid`.
- Missing model file at a configured path → fatal startup error, exit nonzero. Never a 200 with empty text.
- `execution_provider` outside the two-value allowlist → fatal startup error.
- `adapter_mode` accepts only `"test"` or `"production"`; default `"production"`; anything else fatal `config.invalid`.
- `models.adapter` accepts only `"production"` or `"fake"`. Absent is fatal. Unknown value is fatal.
- `models.adapter: "fake"` **and** `adapter_mode: "production"` together are fatal `config.invalid`. Fakes select **both** roles at once (describe AND tagger) — never a hybrid of fake and real.
- Secrets are file paths. Do not invent: env-var secrets, inline secrets in YAML, Vault/KMS integrations, config hot-reload, per-tenant config, a second auth file, or a "profile" system.
- Do not invent a tag threshold config default: the tag threshold is caller-supplied per request (section 7), never a server default.
- **`routes.describe_enabled` / `routes.tag_enabled` (v9 owner decision, closed):** booleans, both default `false`, read at **startup only** — changing either requires a restart. No runtime enable, no plugin system, no second process is invented for this. When a key is `false`: that route's model is **not loaded**, its weight/mapping/suppress files are **not required and not mount-checked**, and the role is **not counted** in `models_loaded` and **not required by the ready gate** — a request to that route returns 404 with reason token `route.disabled` (it is not `task.unsupported` and not a generic `route.not_found`). When a key is `true`: the current fail-closed rule stands — a missing weight file is a fatal startup error before bind, and the role counts toward ready unless **this same route was soft-disabled by the section 2e startup free-memory gate (v12)**, in which case it behaves in every way as disabled: no weight required, not counted, 404 `route.disabled`, and **the process binds — it does not exit.** `route.disabled` is added to the process-wide closed reason-token set, which at v9 was **eleven** tokens (the v11 owner decision later adds `gpu.busy` as the twelfth — section 6e). Unknown keys under `routes:` are fatal (`config.unknown_key`); a **missing** switch key is not fatal because its default `false` applies — but the one `config.example.yaml` must show both keys set to `false` (see the embed spec's section 4 example, amended per the v9 changelog of `vision-embed-SPEC.md`).

---

## 5. Authentication

- Exactly one static bearer token, read once at startup from `auth.bearer_token_path`, compared in **constant time** against the `Authorization: Bearer *** header on `/v1/describe` and `/v1/tag`. This is the same single token and the same single bearer file the embed routes compare on `/v1/embed` (`vision-embed-SPEC.md` section 5); no second token exists.
- `/healthz` and `/readyz` are unauthenticated.
- No OAuth, no IdP, no mTLS, no JWT, no refresh, no rotation endpoint, no per-caller keys, no rate limiting in v1. Do not invent any of these.
- Callers never receive model-host credentials. There are no vendor credentials (no HF token, no S3 key, no API key to anything external) anywhere in this service — nothing external is called.
- An invalid or absent Authorization header is rejected before any model inference and before body decode (section 7 order), with reason token `auth.invalid_token` (HTTP 401).

---

## 6. HTTP API

### Endpoints — closed set

The service's closed endpoint set is exactly { `/healthz`, `/readyz`, `/v1/embed`, `/v1/describe`, `/v1/tag`, `/v1/runtime` } — the existing embed trio as specified in `vision-embed-SPEC.md` section 6, the two routes below, and the runtime-inventory route of section 6d. No second port serves any of these routes.

- `GET /healthz` — the **existing** vision-embed liveness route: 200 `{"status":"ok"}` — process alive. No model check. No auth.
- `GET /readyz` — the **existing** vision-embed readiness route, unchanged in shape and auth, at the same listener. Its readiness condition now also requires the two roles added by this contract: `/readyz` must not report ready until Florence-2 and the WD14 tagger are loaded alongside the embed roles — ready only when **every** model role this process loads is loaded: the four embed roles plus Florence-2 plus the tagger **for each route the operator has enabled** — a role behind a `false` startup switch (section 4, v9) is not loaded, not required here, and not counted. A role soft-disabled by the section 2e startup free-memory gate (v12) is likewise not loaded, not required here, and not counted — but a weight failure while the switch is `true` remains fatal (the process exits); only the memory gate disables without exiting. **`models_loaded` — in `/readyz` and in both gauge series — is the count of model roles this process actually loads, never a hardcoded 6:** production with both switches `false` is **4** (the embed roles), both `true` is **6**, one `true` is **5**; the JSON example `{"status":"ready","models_loaded":6}` shows one legal body, not the only legal one. In `adapter_mode: "production"` an enabled role's real model file must load or startup is fatal; in `models.adapter: "fake"` + `adapter_mode: "test"` the fake stubs are loaded as already written — that fake mode exception stays as written. 503 `{"error":{"reason":"not_ready",...}}` until then. Startup is **fatal** if the Florence-2 directory or any tagger file is missing **while its route's switch is `true` and the route was not soft-disabled by section 2e (v12)**, exactly as a missing embed weight file is — a disabled role's missing files are not checked at all, under either disabling reason. `/readyz` is not a way to come up without a model an enabled switch requires — except by the section 2e memory gate (v12), which is the one specified way an enabled route comes up un-loaded, always with the process bound. A describe/tag failure at startup (while enabled and not memory-soft-disabled) is a failure of this process.
- `POST /v1/embed` — unchanged; specified in `vision-embed-SPEC.md` section 6. Valid bearer token. **Never gated by section 2e in any way — including when both describe/tag routes are soft-disabled for GPU memory; the process and the embed route bind regardless.**
- `POST /v1/describe` → section 6a. Requires valid bearer token.
- `POST /v1/tag` → section 6b. Requires valid bearer token.
- `GET /v1/runtime` → section 6d. Requires valid bearer token. **It cannot be disabled** — like `/healthz`, `/readyz`, and `/v1/embed`, no startup switch exists for it.
- **No other routes exist.** Unknown path/method → 404 with `{"error":{"reason":"route.not_found"}}` or 405 `method.not_allowed`. Do not add a `/status` route.
- No `/openapi.json`, `/docs`, `/redoc`: FastAPI runs with `docs_url`, `redoc_url`, and `openapi_url` all `None`, and no OpenAPI schema artifact is committed.

### 6a. POST /v1/describe request

- Header: `Content-Type: multipart/form-data`.
- Form field `image`: the raw image file bytes (jpeg, png, or webp).
- Form field `task`: the exact wire value from the section 2a table (angle brackets included), case-sensitive. Exactly one value; a repeated field or extra task-like field → `decode.failed`.
- Header: `Authorization: Bearer *** (constant-time compare).
- Optional header: `X-Request-Id` (any printable ASCII ≤128 chars; else a uuid4 is generated server-side). If present but empty, non-printable-ASCII, or longer than 128 chars, the server ignores it and generates a uuid4 — not an error. The **effective request_id** (echoed client id when valid, else a generated uuid4) is used identically in the success response, in the error envelope, and in the audit line. X-Request-Id is accepted on exactly { `/healthz`, `/readyz`, `/v1/embed`, `/v1/describe`, `/v1/tag`, `/v1/runtime` } — the same sentence in `vision-embed-SPEC.md` section 6.
- Max body size: **32 MiB** (config `limits.max_body_bytes`). Larger → reject before reading the full body.
- Supported types: JPEG, PNG, WebP — magic-byte sniffing, not the multipart `Content-Type`. Any other format → rejected.
- **No chaining, no extra task knob.** The request schema is exactly `image` + `task`; no text prompt field, no grounding phrase, no box input, no task-specific parameters. Extra form fields are ignored.
- **Disabled route handling (v12).** The two disabling reasons — startup switch `false` (v9) and the section 2e soft-disable (v12) — are indistinguishable on the wire: both return **404 `route.disabled`**. No request to a disabled route loads a model.

### 6b. POST /v1/tag request

- Header: `Content-Type: multipart/form-data`.
- Form field `image`: the raw image file bytes (jpeg, png, or webp).
- Form field `tag_threshold`: the caller-supplied cutoff, a decimal number in `[0.0, 1.0]`. Parsed as an exact decimal string; absent → `decode.failed` (the threshold is required); outside `[0.0, 1.0]` or unparseable → `decode.failed`. The response carries tags with `score >= tag_threshold`.
- Header/`X-Request-Id`/body-limit/type rules: identical to section 6a. Extra form fields are ignored.
- Disabled route handling: identical to section 6a's v12 rule — both disabling reasons return 404 `route.disabled`.

### Processing order (numbered, stops at first failure — both POST routes; the GPU lock of section 6e is shared with `POST /v1/embed`)

0. **Disabled route** (v12) — if the route's startup switch is `false` (v9) or the route was soft-disabled by the section 2e startup free-memory gate (v12), return **404 `route.disabled`** immediately; no body read, no model load, no lock taken.
1. **Size limit** — enforce `Content-Length`/streamed size ≤ 32 MiB. Failure → `limits.body_too_large`. (A request with a present `Content-Length` over the limit is rejected here even when unauthenticated, following the vision-index order.)
2. **Auth** — constant-time bearer comparison. Failure → `auth.invalid_token`. On rejection the body is not read and not sniffed.
3. **Content type (magic-byte sniff)** — jpeg (`FF D8 FF`), png (`89 50 4E 47`), webp (`RIFF....WEBP`). Failure → `format.unsupported`.
4. **Decode** — decode the full image to a numpy RGB uint8 array. Failure → `decode.failed`.
5. **Field validation** — `/v1/describe`: exact-string task match against the eight wire values (failure → `task.unsupported`). `/v1/tag`: threshold present, parseable, in `[0.0, 1.0]` (failure → `decode.failed`).
6. **Model inference** — under `CUDAExecutionProvider` the GPU admission lock of section 6e is taken here, after field validation and before any model run: `/v1/describe` runs the one named task's model (Florence-2 or the tagger), never both; `/v1/tag` runs the WD14 tagger. A request that needs the GPU while the lock is held does not start and returns 429 `gpu.busy`. A run that **holds** the lock and then fails — including an out-of-memory failure — is 500 `inference.failed`, never `gpu.busy`. Failure → `inference.failed`.
7. **Schema-validate** the response object against the closed schema. Failure → `response.validation_failed` (server bug; surfaced, never swallowed).
8. **Audit line** — exactly one JSON line per section 10 to stdout.
9. **Body** — return the response body. If the stdout write in step 8 fails silently, the body is still returned; no fallback file, no retry queue, no stderr duplicate. Do not buffer audit for later.

### 6e. GPU inference concurrency — one at a time, process-local (v11 owner decision, closed)

**When `execution_provider` is `CUDAExecutionProvider`, this process runs at most one GPU inference at a time.** A single in-process lock governs admission to the GPU; `POST /v1/embed`, `POST /v1/describe`, and `POST /v1/tag` share that one lock — no per-route locks, no priority, no fairness ordering is specified.

- A request that needs the GPU while one inference is already running **does not start and does not wait**: it returns **HTTP 429 with reason token `gpu.busy`**. `gpu.busy` is not `route.disabled` and not a 500, and it is **never** mapped from a failure that occurs while the lock is held — a request that holds the lock and then runs out of memory or otherwise fails is a 500 `inference.failed` (the existing step-6 token); no second token for OOM is invented.
- **No queue, no retry, no wait.** The caller retries. Do not invent a wait queue, a timeout-then-proceed, a busy-wait, a per-caller quota, or a backpressure mechanism.
- `GET /healthz`, `GET /readyz`, and `GET /v1/runtime` do not take the lock and are never rejected `gpu.busy` — they perform no inference.
- When `execution_provider` is `CPUExecutionProvider` there is **no lock and no `gpu.busy`**: the token is never returned on the CPU path, and concurrent CPU inferences are governed by nothing here (no admission control is added for the CPU provider).
- **The lock is in-process only and senses nothing external.** With `--workers 1` (section 2) one process is the whole service. Do not read NVML to decide busy, do not read `nvidia-smi`, do not look at other processes on the GPU — the **owner confirmed: other processes on the GPU are ignored**, and this contract governs only this process's own inference utilization. Do not invent a VRAM-watermark rule or any device-derived admission signal **at request time**; the only memory threshold in this service is the section 2e startup minimum (v12), read at startup under the load-once order of section 2e, and it is not part of the admission lock — no memory value is read while serving a request. The `gpu` object on `/v1/runtime` (section 6d) is unchanged and is **not** an admission check — it is a read-only inventory echo and never feeds the lock.
- The audit line and `vision_describe_errors_total{route,reason}` carry `gpu.busy` as the reason token when it is returned; nothing else on the route changes.
- Startup fail-closed CUDA checks (section 2d) are unchanged by this section, and the section 2e free-memory gate (v12) is startup-only: it can change which routes exist before bind, never how a running request is admitted.

Schema-validation failures (`response.validation_failed`) are unchanged. All other sections are unaffected. No new field, no endpoint, no config key is invented for the lock: the lock has no configuration — it exists iff `execution_provider` is `CUDAExecutionProvider`, and nothing else. (The section 2e minima keys configure the startup gate, not the lock.)

### Reason tokens — COMPLETE and CLOSED set

Errors share a single envelope (identical shape to vision-embed):

```json
{
  "error": {
    "reason": "<one of the tokens below>",
    "detail": "<free-form single-line string, safe for logs, never contains image bytes, tokens, captions, tags, or OCR text>",
    "request_id": "<the effective request_id: echoed client id when valid, else uuid4>"
  }
}
```

| Reason token | HTTP | Where |
|---|---|---|
| `limits.body_too_large` | 413 | step 1 |
| `format.unsupported` | 415 | step 3 |
| `decode.failed` | 422 | step 4 / step 5 (tag threshold unparseable, absent, or out of range; duplicate `task` field) |
| `task.unsupported` | 422 | step 5: task not one of the eight exact wire values, including rejected Florence families needing caller text or a caller box |
| `auth.invalid_token` | 401 | step 2 |
| `inference.failed` | 500 | step 6 |
| `response.validation_failed` | 500 | step 7 |
| `route.not_found` | 404 | unknown path |
| `route.disabled` | 404 | path known but its route is disabled — its startup switch is `false` (section 4, v9) **or its section 2e startup free-memory check disabled it (v12)** — emitted only by `/v1/describe` and `/v1/tag`; never by `/healthz`, `/readyz`, `/v1/embed`, or `/v1/runtime` (the runtime route cannot be disabled). The two disabling reasons are not distinguished on the wire; `/v1/runtime` distinguishes them only in `unavailable_reason` |
| `gpu.busy` | 429 | section 6e: under `CUDAExecutionProvider`, a request that needs the GPU while the one shared admission lock is held by another inference — it does not start and does not wait; emitted only by `/v1/embed`, `/v1/describe`, and `/v1/tag`; the lock is process-local, no external device state is read |
| `method.not_allowed` | 405 | wrong method on a known path |
| `not_ready` | 503 | models not loaded |

No other reason tokens exist. Do not invent: `rate_limited`, `insufficient_permissions`, `bad_payload`, `internal_error`, `timeout`, `no_tags`, `empty_result`, **`gpu.insufficient` (that value exists only as a `/v1/runtime` inventory string of section 6d — never an HTTP token)**, or any token not in this table. An empty tag list or empty caption is a normal 200, not an error. This twelve-token table is the process-wide closed reason-token set for the one container: the nine embed-scope tokens, plus `task.unsupported` and `route.disabled` (both emitted only by this spec's routes), plus `gpu.busy` (section 6e, v11 — shared with `/v1/embed`). `route.disabled` (v9) is not `task.unsupported` and not a generic `route.not_found`; it is returned when `/v1/describe` or `/v1/tag` is addressed while its startup switch is `false` **or while the section 2e gate (v12) soft-disabled it**; v12 adds **no** thirteenth token. `gpu.busy` (v11) is not `route.disabled` and not a 500: a failure that occurs **while the lock is held** (including an out-of-memory failure) is 500 `inference.failed`, never `gpu.busy`.

### 6c. Success response schemas — exact, `additionalProperties: false` on every object

`POST /v1/describe` →

```json
{
  "request_id": "<uuid4 or echoed client id>",
  "image": {
    "sha256": "<hex of the raw bytes of the multipart field named image, not the multipart envelope>",
    "width": 1024,
    "height": 768
  },
  "task": "<CAPTION>",
  "model": "florence2",
  "text": "<model text output>",
  "regions": [
    { "label": "<string>", "bbox": {"x": 0, "y": 0, "w": 0, "h": 0}, "score": 0.0, "text": "<string>" }
  ]
}
```

Field presence rules (part of the schema; absence means the key is not present at all, not null):
- `text` is present **iff** task ∈ {`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OCR>`}.
- `regions` is present **iff** task ∈ {`<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR_WITH_REGION>`}.
- Within a region:
  - `score` is present **only** for `<OD>`. For `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, and `<OCR_WITH_REGION>`, `score` is absent.
  - `text` is present **only** for `<OCR_WITH_REGION>` (the region's recognized text). For `<OD>`, `text` is absent.
  - `label`: for `<OD>`, the model's object string; for `<DENSE_REGION_CAPTION>` and `<OCR_WITH_REGION>`, the model's per-string output (the recognized/region text string in the model's own form); for `<REGION_PROPOSAL>`, the empty string.
- `model` is the literal config-independent string `"florence2"` for `/v1/describe` and `"wd14"` for `/v1/tag`.
- `regions` is capped at `limits.max_regions` (config, 100), items ordered by descending `score` where scores exist and by model output order otherwise. Capping does not change the outcome; the audit reason stays `ok`.
- Bbox coordinates are pixels relative to the decoded image, top-left origin, `x`,`y` ≥ 0, `w`,`h` > 0, integer, clipped to the decoded image.
- Empty `text` (empty string) and empty `regions` list are valid responses (normal 200, not an error).

`POST /v1/tag` →

```json
{
  "request_id": "<uuid4 or echoed client id>",
  "image": {
    "sha256": "<hex of the raw bytes of the multipart field named image>",
    "width": 1024,
    "height": 768
  },
  "model": "wd14",
  "threshold_used": 0.0,
  "tags": [
    { "tag": "<danbooru tag string>", "score": 0.0 }
  ]
}
```

- `threshold_used` echoes the parsed caller threshold verbatim.
- `tags[]` are the score-cut tag list, sorted by score descending; each item has exactly `tag` and `score` and nothing else. Tags above the cut but beyond any cap are dropped — v1 defines **no max_tags knob** (stop-and-ask if one is wanted, section 16).
- Empty `tags` is a normal 200.

Schema rules common to both:
- `additionalProperties: false` on every object, including the top-level response, `image`, each region, and each tag.
- `image.sha256` is SHA-256 over the exact bytes of the multipart field named `image` — the same hex as the audit line.
- Do not invent: a `duration_ms` field, a per-tag `(tag:weight)` prompt-formatted string, a rating field, a character block field, a translated caption, a confidence interval, thumbnails, pagination, a second `model` naming the task head, or an `adapter` echo.

### 6d. GET /v1/runtime — runtime inventory (v10 executive decision, closed)

An operator-facing inventory route: enabled endpoints, loaded models, and runtime info, on the **same single listener** as every other route — **no second port**. It is a GET, takes no body, and requires the valid bearer token exactly like the three POST routes (same `Authorization: Bearer` header, constant-time compare against the same single bearer file; an invalid or absent header → 401 `auth.invalid_token`, the existing token). Any bearer token failure is a 401, and any request body is ignored and is not read — nothing loads, nothing runs.

- **It cannot be disabled.** No startup switch exists for it (there is no `routes.runtime_enabled` key; unknown keys under `routes:` remain fatal per section 4). Like `/healthz`, `/readyz`, and `/v1/embed`, it has no switch — the section 2e gate (v12) never disables this route either.
- **It loads no model and does not change readiness.** A call to `/v1/runtime` triggers no inference, no lazy load, no warm-up, and does not alter `/readyz` output — it is a read-only report of the current process state.
- **`/readyz` and `/healthz` stay unauthenticated probes and are unchanged by this route.** Do not move this inventory onto `/readyz` or `/healthz`; no model names, switch flags, versions, or provider strings are added to `/readyz` — it remains `{"status":"ready","models_loaded":N}` with `models_loaded` equal to the count of roles actually loaded (section 6).
- Success response — a single JSON object, `additionalProperties: false` on every object, exactly these fields and no others — `gpu` (the object of the v10 amendment changelog) added:

```json
{
  "uptime_seconds": 0,
  "execution_provider": "<CPUExecutionProvider|CUDAExecutionProvider: the configured provider string>",
  "routes": {
    "embed": true,
    "describe": false,
    "tag": false
  },
  "models": [
    { "role": "face", "model_id": "buffalo_l/w600k_r50", "loaded": true, "unavailable_reason": "" },
    { "role": "face_detector", "model_id": "buffalo_l/det_10g", "loaded": true, "unavailable_reason": "" },
    { "role": "scene", "model_id": "ViT-L-14/openai", "loaded": true, "unavailable_reason": "" },
    { "role": "object", "model_id": "yolov8n", "loaded": true, "unavailable_reason": "" },
    { "role": "florence2", "model_id": "", "loaded": false, "unavailable_reason": "disabled" },
    { "role": "tagger", "model_id": "wd14", "loaded": false, "unavailable_reason": "disabled" }
  ],
  "gpu": { "available": true, "device_count": 1, "name": "Example GPU", "memory_total_bytes": 12582912, "memory_used_bytes": 2097152 }
}
```

Schema rules (closed):
- `uptime_seconds` — integer, seconds since process start (floor of elapsed wall-clock).
- `execution_provider` — the configured provider string, one of the two allowlist values of section 2d (`"CPUExecutionProvider"` or `"CUDAExecutionProvider"`). This echoes config; the route does not probe the device.
- `routes` — object with exactly `embed` (always `true`; `/v1/embed` cannot be disabled), `describe` (the current `routes.describe_enabled` startup switch value), and `tag` (the current `routes.tag_enabled` startup switch value). No other keys.
- `models` — array with exactly **six** elements in this fixed order: `face`, `face_detector`, `scene`, `object`, `florence2`, `tagger`. Each element has exactly `role` (one of those six strings), `model_id`, `loaded`, and `unavailable_reason` (v12), nothing else.
  - `loaded` is `true` iff the role's model is loaded in this process right now; `false` when not loaded, **including a describe or tag role whose startup switch is `false` (section 4, v9) or that the section 2e startup free-memory gate soft-disabled (v12)**. Fake-adapter stubs count as loaded (consistent with the ready rule of section 6, where fake stubs are the loaded roles).
  - `model_id` is the **short id that role already uses in its responses** — `face` → `"buffalo_l/w600k_r50"`, `face_detector` → `"buffalo_l/det_10g"`, `scene` and the object-crop embedding → `"ViT-L-14/openai"`, `object` → `"yolov8n"`, `florence2` → `"florence2"`, `tagger` → `"wd14"` (the same literal `model` strings of section 6c and `vision-embed-SPEC.md` section 6). When the role is not loaded, `model_id` is the **empty string** and never a path or filename.
  - `unavailable_reason` (v12, closed) — a string with exactly three legal values and no others: `""` when the role is loaded (including a switch-`true` describe/tag role under `CPUExecutionProvider`, where the section 2e check never runs, and the always-loaded four embed roles); `"disabled"` when the describe/tag role's startup switch is `false`; `"gpu.insufficient"` when a switch-`true` describe/tag role was **not loaded because the section 2e startup free-memory gate (v12) did not pass it** — its operator minimum exceeded the free bytes available in the compare order, **or the NVML read could not be taken so no value was guessed** (every check-requiring route is then `gpu.insufficient`). **`unavailable_reason` is inventory only and has no HTTP meaning: `route.disabled` stays the HTTP token, no reason token is added, and the process-wide closed set stays at twelve tokens.** The embed roles' `unavailable_reason` is always `""`.
- `gpu` — object with exactly `available`, `device_count`, `name`, `memory_total_bytes`, and `memory_used_bytes`, and no other keys (`additionalProperties: false`). `available` is true **only** when the NVML library call opens and reports at least one device; it is false when NVML is missing, permission is denied, or the device count is zero — and a false `available` is **not** an HTTP error: the route still returns 200. `device_count` is an integer, `0` when `available` is false. `name` is the device-0 name string when `available`, else the empty string; names for devices other than index 0 are never returned. `memory_total_bytes` and `memory_used_bytes` are integers when `available`, else `null`; `0` is never written to mean unknown. The read is NVML via a library call, never a shell-out to `nvidia-smi`. Only device index 0 is described; when `device_count` > 1 the other devices are not described — do not invent a device picker or a per-device array. A failed GPU read does not change `/readyz`, does not unload a model, and does not add a reason token — the reason-token set stays closed. (v12, amended to the load-once order of section 2e: this NVML call is the same one the section 2e startup gate reads with — a baseline reading before the loads and a reading before/after each enabled role's single load for the residency add-back — and it happens with the role models resident; the route itself reads per request as already written and never re-runs the section 2e comparison.)
- **Do not return: filesystem paths, weight file names, mapping/suppress file names, tokens, the bearer value, image bytes, config file contents, driver version, or any environment detail beyond the fields above.** This route is not the CUDA fail-closed gate: startup provider placement still uses the existing onnxruntime and transformers checks of section 2d, and this GPU read never feeds that gate or the section 6e lock.
- **Do not add:** on `models[]`, any field other than the four above (`role`, `model_id`, `loaded`, `unavailable_reason` — v12 adds exactly the one field and closes it at three values); on `gpu`, any field other than the five above — no VRAM beyond the two memory fields, no free-bytes echo, no driver version, no CUDA-equals-CPU or CUDA-vs-CPU field, no device id beyond index 0, no per-device list, no device picker — and on the top level no onnxruntime/transformers version strings, no request counter, no PID, hostname, or container id. `nvidia-smi` stays banned everywhere in this service, including as a startup probe; the `gpu` read is not a startup probe and is not the CUDA fail-closed gate. `execution_provider` remains config echo only.
- The response is schema-validated with `additionalProperties: false` like every other response (step 7 order applies); a schema failure is `response.validation_failed`.
- Audit: exactly one JSON line per request on stdout per section 10, with `"route": "/v1/runtime"`, `"sha256"` null and `"bytes"` 0 (there is no image), and no tags/threshold/regions values. Same privacy rules; no request or response body content is logged.

---

## 7. Adapters

Each model role sits behind a Protocol so tests run without weights:

```python
class FlorenceDescribe(Protocol):
    def load(self, path: Path) -> None: ...
    def run(self, rgb: np.ndarray, task: str) -> tuple[str, list[tuple[str, int, int, int, int, float | None, str | None]]]:
        # returns (text, regions); regions tuple: (label, x, y, w, h, score|None, text|None)
        # exactly one of text/regions is meaningful per the task presence rules of section 6c

class Tagger(Protocol):
    def load(self, path: Path) -> None: ...
    def tag(self, rgb: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        # returns [(tag, score)] already cut at threshold, score-descending
```

- **Fake adapters** exist for **both roles** (florence and tagger), selected only by `models.adapter: "fake"` **and** `adapter_mode: "test"`. Fakes are deterministic, seeded by the image sha256 and by the requested task/threshold, so fixture tests are reproducible. With the fake set, the full request/response/audit path runs with **zero ONNX files present**. The fake tagger returns at least three tags spanning values above and below a 0.5 threshold, so threshold cut behavior is testable.
- In `adapter_mode: "production"` the real adapters are used and **must never** fall back to or touch the fake adapters for any reason. There is no config key that reaches the fakes in production mode.
- Do not invent: a plugin registry, an auto-discovery loader, a "model zoo" abstraction, adapter hot-swap at runtime, or a third adapter role.
- The section 2e startup free-memory gate (v12) is adapter-independent: it runs or does not run per its two closed conditions (provider and switch), and no fake-mode exception or production-mode special case is invented for it.

---

## 8. Statelessness and no file output

- Nothing is stored. The sha256 of the raw image bytes is logged in the audit line, and that is the entire server's memory of a request. A repeated POST is another inference; outputs are not cached and nothing is deduped server-side.
- The service **writes no files for the caller**: no caption file, no `.txt` sidecar, no dataset directory, no training-data packager, no manifest. The response body is the only output channel, and it is JSON only. Do not invent an output-path field in config or request, a job that materializes files, or a "batch mode" that walks a directory.
- Restart loses nothing because nothing was kept: no queue, no WAL, no cache, no session, no KV.
- Do not invent: a hash-keyed response cache, a TTL, an ETag, `Idempotency-Key` handling, or request memory.

---

## 9. Cross-service boundary

- vision-index **MUST NOT call** `/v1/describe` or `/v1/tag`. Those routes are not part of the vision-index contract, and this spec adds nothing to the vision-index route set. Any index-style matching, caption-file storage, or dataset construction is the **caller's** job, done on the caller's side of the boundary.
- The service calls nothing. Exactly zero outbound HTTP calls exist in v1. Do not invent an embed call from `/v1/describe` or `/v1/tag` to `/v1/embed` or vice versa (they are routes of one process calling the same in-process models), an index call, an OpenViking read or write, or a webhook.

---

## 10. Audit

Exactly one **JSON line** per request on **stdout**:

```json
{"ts": "<RFC3339 UTC>", "request_id": "<effective request_id>", "route": "/v1/describe", "sha256": "<hex or null>", "bytes": 0, "task": "<wire value or null>", "regions": 0, "tags": 0, "threshold": null, "outcome": "ok|error", "reason": "<reason token or \"ok\">"}
```

- `/v1/describe` fills `task` (the exact wire value) and `regions` (count after the `max_regions` cap); `tags` and `threshold` are `null`.
- `/v1/tag` fills `tags` (count after the cut) and `threshold` (the caller threshold verbatim); `task` and `regions` are `null`.
- `/healthz` and `/readyz` emit **no** audit line (they carry no image body). The audit line exists only for the two POST routes this spec owns **plus, since v10, `GET /v1/runtime`:** that request yields exactly one JSON line with `"route": "/v1/runtime"`, `"sha256": null`, `"bytes": 0`, and `task`/`regions`/`tags`/`threshold` all `null` — the same privacy rules, no request or response body content. **Audit scope for the one process is split:** `POST /v1/embed` emits the embed-spec audit line (`vision-embed-SPEC.md` section 9, faces/objects keys); `POST /v1/describe`, `POST /v1/tag`, and `GET /v1/runtime` emit this section 10 line. One line shape for every request is not required, and no spec imposes one.
- On success `reason` is the literal string `ok`. Capping regions or an empty caption/tag list does not change it. No prefixed or dotted variant exists anywhere in this spec or implementation.
- Never log image bytes, captions, tag strings, OCR text, embeddings of any kind, Authorization headers, tokens, or the auth file contents. The `detail` field of error envelopes follows the same rule. `unavailable_reason` (v12) is never written into an audit line — it appears only in the `/v1/runtime` response.
- No syslog requirement in v1. Stdout JSON-lines is the record. Do not ship captions, tags, or images to syslog later without a new spec — logging any of them is a spec violation, not an optimization, in any version. Nothing persists across restart: no log file, no disk writes, no DB.

---

## 11. Metrics

The **existing embed metrics server** — same process, same port (config `metrics.port`, default 9090), no auth — carries the closed series **only**; no second metrics server is added. The one metrics server's process-wide closed series set is the **four embed series of `vision-embed-SPEC.md` section 10 plus the four series below only for each enabled route** (v9) — where "enabled" retains its v12 meaning: a route behind a `false` switch **or a switch-`true` route soft-disabled by the section 2e free-memory gate** has **no series at all — absent, not stuck at zero** — so the closed set is **not** always eight series: both switches `false` → the four embed series only (4 series); one switch `true` and loaded → 6; both `true` and both loaded → 8; a switch-`true` route soft-disabled by section 2e contributes no series exactly as if its switch were `false`. The old "exactly eight" gate is superseded by this rule: a scrape asserting exactly eight series regardless of switch or gate state contradicts section 2e and must not be built.

- `vision_describe_requests_total{route,outcome,reason}` — counter (`route` ∈ `describe|tag`, `outcome` ∈ `ok|error`).
- `vision_describe_latency_seconds` — histogram, standard Prom buckets.
- `vision_describe_errors_total{route,reason}` — counter (reason token).
- `vision_describe_models_loaded` — gauge (0..2).

Forbidden in labels and in series names: any hash, any byte size, any caption, tag, or OCR text, any value derived from image content beyond the `reason` token. Do not invent: extra series, custom buckets, an exporter, a push gateway, an alert rule, or a second metrics backend.

---

## 12. Tests the build must pass

Concrete assertions, not "further analysis":

1. **Config fail-closed**: injecting an unknown YAML key causes startup to exit nonzero with one stdout fatal JSON line; no HTTP listener opens.
2. **Missing weight fail-closed**: renaming one of the two model files, the Florence-2 directory, or one of the tagger mapping/suppress files and starting up (`adapter_mode: "production"`, **with the file's route's switch `true`**) causes exit nonzero; `/readyz` never reports ready. A missing describe directory or tagger file is fatal exactly as a missing embed weight is — **but only while its route is enabled and not soft-disabled by section 2e (v12)**; with the switch `false` the file is not required, not mount-checked, and this startup succeeds without that role. **This test does not apply to fake mode, which intentionally starts and reports ready with zero ONNX files.**
3. **Fake-adapter production refusal**: config with `models.adapter: "fake"` and `adapter_mode: "production"` → exit nonzero with the fatal JSON line `{"fatal": true, "reason": "config.invalid", ...}`.
4. **Auth**: bad token → 401 `auth.invalid_token` on both POST routes; constant-time compare verified; `/healthz` and `/readyz` unauthenticated.
5. **Body limits, order**: a request with `Content-Length` over 32 MiB and no token → 413 `limits.body_too_large` (token not checked, body unread); a 33,000,000-byte authenticated upload → 413 before full-body read.
6. **Format sniff**: a `.txt` file uploaded → 415 `format.unsupported`. A JPEG with the wrong multipart Content-Type header still decodes (magic bytes win).
7. **Closed task set**: each of the eight exact wire values → 200. Each of the five rejected Florence families (phrase grounding, open-vocab detection, referring-expression segmentation, region-to-segmentation, region-to-description — by their wire token forms, with or without accompanying text/box arguments) → 422 `task.unsupported`. An arbitrary string (e.g. `<NONSENSE>`), an unbracketed task name, or a wrong case → `task.unsupported`; a missing `task` field, a duplicate `task` field, or a body that is not valid JSON → `decode.failed`, per the closed rule of section 7 step 5.
8. **One task, no chaining**: a request carrying two task values → reject; a code-search assertion asserts no code path runs a second Florence task per request.
9. **Tag threshold**: caller threshold 0.9 on a fixture → only tags with score ≥ 0.9; threshold 0.1 → includes low-score tags; absent threshold → 422 `decode.failed`; threshold `1.1`, `-0.01`, and `abc` → 422 `decode.failed`. `threshold_used` echoes the caller value exactly.
10. **No file writes**: after fixtures on both routes, no file anywhere outside tmpfs is created or modified (test asserts directory trees before/after are unchanged), including no `.txt` sidecar and no dataset directory, for every task and tagger fixture run.
11. **Closed route set**: `GET /v1/describe`, `GET /v1/tag`, `POST /healthz`, `PATCH /v1/describe`, `DELETE /v1/tag` → correct 404/405 tokens. `/openapi.json`, `/docs`, `/redoc` → 404.
12. **Schema strictness**: response JSON validated with `additionalProperties: false` on every object; adding an invented key (e.g. a rating field, a prompt-formatted tag string) fails the validator.
13. **Field presence rules**: `<CAPTION>` families return `text` and no `regions`; `<OD>` returns `regions` each carrying a label and `score` and no `text`; `<OCR_WITH_REGION>` returns regions carrying `text` and no `score`; `<REGION_PROPOSAL>` returns regions with empty-string labels, no `score`, no `text`; `<OCR>` returns `text` and no `regions`.
14. **Region cap**: a fake returning `max_regions + 5` regions → exactly `max_regions` returned, outcome `ok`.
15. **Statelessness across restart**: run the same fixture twice across a process restart → identical text/regions/tags on the CPU provider.
16. **Reason token closed set**: property test asserts the set of reason tokens in code equals exactly the 12 tokens of section 7 (the eleven prior tokens plus `gpu.busy`), no more — and that `gpu.busy` is returned only as the 429 of section 6e, never as a mapping for a lock-held inference failure (that stays 500 `inference.failed`).
17. **Audit shape**: each of `/v1/describe` and `/v1/tag` yields exactly one stdout JSON line with the eleven keys of section 10; success reason literally `ok`; stdout never contains "Bearer", any caption text, any tag string, any 64-plus-hex string other than the sha256, or image bytes. (`POST /v1/embed` follows the embed-spec audit line, `vision-embed-SPEC.md` section 9 — no single line shape for every POST.)
18. **Metrics closed set**: scraped metrics contain the series names of section 11 for **each enabled route only**, alongside the four embed series of `vision-embed-SPEC.md` section 10 — the process-wide closed set sized per the v9 rule as extended by the section 2e gate (4/6/8 series by switch and gate state; a disabled or soft-disabled route's series are absent); any scrape gate requiring exactly eight regardless of the switches must not be built. No label value parses as a 64-char hex string or contains any caption/tag text.
19. **No runtime fetch**: container in a network-isolated test starts, reports ready, and serves fixtures — proving no outbound fetch at startup or inference time.
20. **Read-only rootfs**: container run with `--read-only` and tmpfs `/tmp`; a fixture request and restart both succeed.
21. **Non-root**: container process UID is 65532 (asserted in test).
22. **CUDA fail-closed**: default `execution_provider` is `CPUExecutionProvider`. Configuring `CUDAExecutionProvider` on a CPU-only image → startup fatal `config.invalid` before bind, one JSON line, no CPU fallback, session never constructed on CPU. Any third provider string → fatal at config parse.
23. **No identity**: code-search asserts the strings "cluster", "person", and "identify" never appear in this service's response construction path; no field in any response is named or shaped like a person identity.
24. **Index boundary**: code-search asserts this codebase contains no HTTP client construction and no URL pointing at vision-index or vision-embed.
25. **Route startup switches (v9)**: with `routes.describe_enabled: false`, `POST /v1/describe` → 404 `route.disabled` (not `task.unsupported`, not `route.not_found`); Florence-2 is not loaded, its weight/processor files are not required and not mount-checked (missing-file startup succeeds), the role is absent from `models_loaded` and the ready gate, and its metrics series are absent from the scrape — the process is ready with `models_loaded` 4. Same for `routes.tag_enabled: false` on `/v1/tag` and the WD14 tagger. One switch `true` → ready with `models_loaded` 5 and only that role's four series present. Unknown keys under `routes:` → fatal `config.unknown_key`; missing switch keys → defaults `false` apply (not fatal). With a switch `true`, the fail-closed weight rules of test 2 apply. A restart is required for a switch change; no runtime enable path exists in the code (code-search asserts the config is read once at startup).

26. **Runtime inventory route (v10, amended v12)**: `GET /v1/runtime` with no token → 401 `auth.invalid_token` and no model load occurs; with the valid token → 200 with a JSON object whose top-level keys are exactly {`uptime_seconds`, `execution_provider`, `routes`, `models`, `gpu`}, `additionalProperties: false` enforced by the schema validator; `gpu` is an object whose keys are exactly {`available`, `device_count`, `name`, `memory_total_bytes`, `memory_used_bytes`} — with no GPU present or NVML unavailable, `available` is `false`, `device_count` is `0`, `name` is `""`, and both memory fields are `null`, and the route still returns 200 (a null memory field is schema-valid only when `available` is `false`); with a device present, `device_count` >= 1, `name` is the device-0 name, and a simulated `device_count` > 1 still describes only device 0; `uptime_seconds` is an integer ≥ 0 and increases across calls; `execution_provider` equals the configured provider string (the default run: `"CPUExecutionProvider"`); `routes` has exactly `embed` (true), `describe`, and `tag` matching the startup switches; `models` has exactly six elements in the order face, face_detector, scene, object, florence2, tagger, each with exactly `role`, `model_id`, `loaded`, and `unavailable_reason` (v12); every entry carries a `unavailable_reason` of exactly one of the three closed values — `""` for every loaded role and for the four embed roles always; `"disabled"` for a role behind a `false` switch; `"gpu.insufficient"` for a switch-`true` role soft-disabled by the section 2e gate (test 28) — and no fourth value passes the validator; a role behind a `false` switch has `loaded: false`, `model_id: ""`, and `unavailable_reason: "disabled"`; no filesystem path, weight filename, version string, `nvidia-smi` output, driver version string, or CUDA-equals-CPU field appears anywhere in the response (test asserts absence; the closed `gpu` object of the v10 amendment is the only device-derived data and is exempt from this absence assertion); with both switches `false` the response still returns 200 with the six role entries and the disabled roles' `loaded: false` — `GET /v1/runtime` itself is never 404 `route.disabled`; `POST /v1/runtime` → 405 `method.not_allowed`; the audit line for a `/v1/runtime` request has `"route": "/v1/runtime"`, `"sha256": null`, `"bytes": 0`, and `"reason": "ok"` on success.
27. **GPU single-flight (v11)**: with `execution_provider: "CUDAExecutionProvider"` and a slow fake/real inference holding the lock, an authenticated POST to `/v1/embed` returns **429 `gpu.busy`** without starting its own inference, the same for `/v1/describe` and `/v1/tag`; `GET /healthz`, `GET /readyz`, and `GET /v1/runtime` still return 200 while the lock is held; with `execution_provider: "CPUExecutionProvider"` concurrent POSTs are never 429 (no lock exists); a fixture whose inference fails while the lock is held → 500 `inference.failed`, never `gpu.busy`; a code-search assertion asserts the busy decision reads no NVML call, no `nvidia-smi` invocation, and no other-process state, and that the `/v1/runtime` `gpu` object's data is not consulted by the admission path.
28. **Startup free-memory gate (v12)**: with `execution_provider: "CUDAExecutionProvider"`, `routes.describe_enabled: true`, and `routes.tag_enabled: true` (`adapter_mode: "production"`), a startup config **missing** `routes.describe_min_free_bytes` or `routes.tag_min_free_bytes` (in either combination) → exit nonzero before bind with exactly one stdout fatal line `{"fatal": true, "reason": "config.invalid", ...}`; with `execution_provider: "CPUExecutionProvider"` or either switch `false`, the same keys may be absent and startup proceeds (the check and its keys do not apply). With a simulated device-0 NVML free value (the same library call of section 6d) **below `describe_min_free_bytes`**: the process **binds and serves** — `/healthz` 200, `/v1/embed` works — Florence-2 is not loaded, POST `/v1/describe` → 404 `route.disabled` (never `task.unsupported`, never 500), the florence2 role is absent from `models_loaded` and the ready gate, its metrics series are absent, and `/v1/runtime` shows that role `loaded: false`, `model_id: ""`, `unavailable_reason: "gpu.insufficient"` — and **the process did not exit** (no fatal line is written). **Order**: with the free value such that describe passes but the tag minimum does not fit the remaining free (`free ≥ describe_min_free_bytes` but `free − describe_min_free_bytes < tag_min_free_bytes`), describe loads and serves a fixture 200 while the tagger is soft-disabled — `models_loaded` 5, `tagger` `unavailable_reason: "gpu.insufficient"`, `florence2` `""` — and never tag first. With both minima exceeding the free value, both roles are `gpu.insufficient`, `models_loaded` is 4, and the process still binds; with both minima within the free value, both roles load (`""`) and `models_loaded` is 6. **NVML unreadable** (simulated missing/denied/zero-device NVML): no byte value is guessed, **every** switch-`true` describe/tag route under the CUDA provider is soft-disabled with `unavailable_reason: "gpu.insufficient"`, the process binds and serves `/v1/embed`, no fatal line — it must not exit. Under `CPUExecutionProvider` the gate never runs: a `true` switch and no minimum keys is a normal start with the roles loaded. A code-search assertion asserts the gate uses only the NVML library call (no `nvidia-smi` subprocess anywhere), subtracts only operator minima and no model-size estimate, and that no memory value is read in any request-time path (the section 6e lock stays untouched).

### 12a. Done gates

1. All 28 tests of section 12 pass.
2. `GET /healthz` returns 200 within 5 s of process start on a clean container run; `GET /readyz` returns 200 once the two models are loaded.
3. A fixture image POSTed to `/v1/describe` with **`routes.describe_enabled: true`** and POSTed to `/v1/tag` with **`routes.tag_enabled: true`** — the gate does not run on the default `false` config (section 4, v9), where both routes return 404 `route.disabled` — with `<MORE_DETAILED_CAPTION>` returns a schema-valid `text` response without `regions`, and at threshold 0.35 on `/v1/tag` returns a schema-valid tag list sorted by score descending, all scores ≥ 0.35.
4. A restart test (kill -9 then restart) shows no state loss and unchanged outputs on the same bytes.
5. A code search asserts no storage primitives (no `sqlite`, no `INSERT`, no `open(..., 'w')` outside tmpfs, no `redis`, no `boto3`, no S3 client, no requests/httpx client) anywhere outside test fixtures.
6. Audit stdout lines conform exactly to section 10 for both POST routes and for every one of the twelve reason tokens.
7. The metrics endpoint exposes the series of section 11 for each loaded (enabled and not soft-disabled by section 2e) route alongside the four embed series of `vision-embed-SPEC.md` section 10 — the shared server's process-wide closed set per the v9 rule as extended by section 2e (4/6/8 series by switch and gate state; a disabled or soft-disabled route's series are absent, not zero), and nothing else.
8. The one `config.example.yaml` exists and matches this SPEC: it is the embed spec's section 4 example (`vision-embed-SPEC.md` section 4) carrying the additive `models` keys, `limits.max_regions`, and the `routes:` block of this document's section 4 — with **both `routes.describe_enabled` and `routes.tag_enabled` shown set to `false`**, so the live embed behavior is the default, and the v12 minima keys shown commented (they are required only under their closed two-condition rule) — there is no second config example.

---

## 13. Out of scope (v1)

Storage of any kind, file writes of any kind (captions, sidecars, datasets, manifests), clustering, naming, identity, an index, vision-index as a caller, GUI, MCP server, public internet exposure, path-based ingest (a filesystem path in the public request contract), any Florence-2 task requiring caller text or a caller box, phrase grounding, segmentation, region-to-description, open-vocabulary detection, a second tagger, a second tagger vocabulary, tag translation or filtering or prompt weight formatting, batch endpoints, directory walking, video, audio, mTLS, multitenancy, rate limiting, model download at runtime, syslog, retry queues. The caller owns persistence, matching, caption-file writing, and any training-data pipeline.

---

## 14. Feature register

| ID | Feature (one sentence, for critic scoring) |
|---|---|
| F1 | Exposes `GET /healthz` liveness with no auth and no model dependency |
| F2 | Exposes the existing `GET /readyz` readiness route whose condition covers the two describe roles **only when each route's startup switch is `true` and the role was not soft-disabled by the section 2e free-memory gate (v12)** (section 4, v9) — a role behind a `false` switch or below its operator minimum is not loaded, not required, and not counted — with the `models_loaded` gauge reflecting the roles this process actually loads; fake mode reporting ready with zero ONNX files, else 503 `not_ready` |
| F3 | Exposes `POST /v1/describe` as an additive route on the existing vision-embed process (v6 owner decision), within the closed endpoint set exactly { /healthz, /readyz, /v1/embed, /v1/describe, /v1/tag, /v1/runtime } — the last one the v10 executive decision of this spec adds (section 6d) |
| F4 | Accepts on `/v1/describe` exactly one task per request with no chaining, as the exact wire value from the eight-entry closed set { `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR>`, `<OCR_WITH_REGION>` } |
| F5 | Rejects every other Florence task — phrase grounding, open-vocab detection, referring-expression segmentation, region-to-segmentation, region-to-description — with 422 `task.unsupported` before any inference, because they need caller text or a caller box |
| F6 | Exposes `POST /v1/tag` as a tagger route on the same vision-embed process, backed by the WD14 Danbooru ONNX tagger family and never by Florence |
| F7 | Cuts the tag list at the caller-supplied `tag_threshold` (required, `[0.0, 1.0]`), returning only `{tag, score}` pairs sorted score-descending |
| F8 | Writes no files for the caller — no caption files, no `.txt` sidecars, no training datasets — returning JSON only |
| F9 | States plainly that the WD14/Danbooru vocabulary targets anime/style caption files and is the wrong vocabulary for ordinary photographs, without translation, filtering, or reweighting |
| F10 | Loads only operator-supplied weight files (Florence-2 and WD14) with no runtime download, no fetch-on-start, and no weight URL anywhere in the spec or code |
| F11 | Leaves weight licenses under their own published terms while the repo code stays MIT, naming no person |
| F12 | Defaults to `CPUExecutionProvider` and treats `CUDAExecutionProvider` as opt-in fail-closed: not registered or not active at session build → fatal `config.invalid` before bind, with no silent CPU fallback and no fallback chaining |
| F13 | Makes no numeric CPU-vs-CUDA equality claim in either direction and invents no tolerance value |
| F14 | Enforces the 32 MiB max body size before full-body read with reason `limits.body_too_large`, even when unauthenticated |
| F15 | Sniffs magic bytes for jpeg/png/webp and rejects any other format with reason `format.unsupported` |
| F16 | Accepts an optional X-Request-Id header or generates a uuid4, using the effective request_id identically in success, error, and audit |
| F17 | Compares the bearer token from a file path in config in constant time on both POST routes, with `/healthz` and `/readyz` unauthenticated |
| F18 | Enforces the closed twelve-token reason set in one uniform error envelope with `additionalProperties: false` on every object of every schema |
| F19 | Stops processing at the first failure in the fixed order (disabled-route 404 → size limit → auth → magic-byte sniff → decode → field validation/task match → inference → schema-validate → audit → body), each step its own reason token |
| F20 | Returns a schema-valid per-task response shape: `text` iff caption/OCR task, `regions` iff region/OCR-with-region task, region `score` only for `<OD>` and region `text` only for `<OCR_WITH_REGION>` |
| F21 | Caps `regions[]` at `limits.max_regions` (100) without changing outcome or the audit reason `ok` |
| F22 | Returns bbox coordinates as integer pixels relative to the decoded image, clipped to the decoded image |
| F23 | Treats a missing weight file as a fatal startup error — never a 200 with empty text — **while that role's route switch is `true` and the role was not soft-disabled by section 2e (v12)** (a `false` switch or a gate-disabled route does not require the file at all, section 4), except under fake+test mode |
| F24 | Treats unknown config keys as a fatal startup error writing one JSON fatal line and exiting 1 |
| F25 | Provides fake adapters for both roles, selected only by `models.adapter: "fake"` plus `adapter_mode: "test"`, unreachable in production by any config path |
| F26 | Stores nothing — no images, captions, tags, names, cluster ids, or cache, across requests or restarts |
| F27 | Never returns or logs a cluster id, person name, or person identity in any response or audit line |
| F28 | Emits exactly one audit JSON line per POST on stdout with the eleven keys of section 10 and never image bytes, captions, tags, tokens, or Authorization values |
| F29 | Exposes the four metrics series **for each enabled route only** (section 11, v9, as extended by the section 2e gate in v12 — a disabled or soft-disabled route's series are absent, not zero) on the existing unauthenticated metrics server of the same process (no second metrics server) with no content-derived labels |
| F30 | Runs as uid 65532 with read-only rootfs, tmpfs /tmp, and ro-mounted secrets and model files |
| F31 | Runs a TLS-only listener on configured cert/key file paths, the operator firewalling the public internet |
| F32 | Performs zero outbound network calls and never calls vision-index |
| F33 | Accepts no text prompt, grounding phrase, box input, or task parameter on any route — image plus fixed form fields only |
| F34 | Reads two startup-only switches `routes.describe_enabled` / `routes.tag_enabled` (both default `false`; no switch for `/healthz`, `/readyz`, `/v1/embed`): `false` means the route's model is not loaded, its role is not counted in `models_loaded` or the ready gate, a request returns 404 `route.disabled`, and its metrics series are absent; `true` means a missing weight/mapping/suppress file is fatal before bind and the role counts toward ready unless section 2e soft-disabled it; changing a switch requires a restart (v9, section 4) |
| F35 | Exposes `GET /v1/runtime` (v10 executive decision, as amended v10 and v12) on the same listener and bearer as the POST routes — undisableable, loads no model, does not change readiness — returning exactly `uptime_seconds`, `execution_provider`, `routes` (`embed`/`describe`/`tag`), `models` (six entries in the fixed order face, face_detector, scene, object, florence2, tagger with `role`/`model_id`/`loaded`/`unavailable_reason`; empty `model_id` when not loaded), and the closed `gpu` object (`available`/`device_count`/`name`/`memory_total_bytes`/`memory_used_bytes`; NVML library call only, device 0 only, false-`available` is still a 200, not the CUDA fail-closed gate), with no filesystem paths, weight filenames, versions, tokens, driver version, `nvidia-smi` output or use as a startup probe, or CUDA-equals-CPU field, and one privacy-preserving audit line for the route |
| F36 | Enforces one GPU inference at a time when `execution_provider` is `CUDAExecutionProvider` (v11 owner decision, section 6e): a single in-process lock shared by `POST /v1/embed`, `POST /v1/describe`, and `POST /v1/tag`; a request needing the GPU while the lock is held returns 429 `gpu.busy` without starting or waiting (no queue, retry, or wait — the caller retries); `gpu.busy` is not `route.disabled` and not a 500, and a lock-held failure — OOM included — is 500 `inference.failed`; `/healthz`, `/readyz`, and `/v1/runtime` never take the lock; on `CPUExecutionProvider` there is no lock and no `gpu.busy`; the lock is in-process only — no NVML, no `nvidia-smi`, no other-process state, no memory threshold at request time, and the `/v1/runtime` `gpu` object is not an admission check |
| F37 | Runs the section 2e startup free-memory gate (v12, amended to the load-once design) at startup, only when a describe/tag route's switch is `true` **and** `execution_provider` is `CUDAExecutionProvider`: after the fail-closed provider gate, each enabled role is loaded exactly once (that single load IS its section 2d activation proof, a failed load fatal `config.invalid` before bind), then free GPU memory is read via the same NVML call as the section 6d `gpu` object (total − used on device 0, never `nvidia-smi`) with the role models resident, the residency of each already-loaded role added back before the compare using its measured before/after-load delta and no model-size estimate invented, compared in fixed order describe then tag against the operator-set integer minima `routes.describe_min_free_bytes` / `routes.tag_min_free_bytes` (bytes, **no default** — required fatal `config.invalid` only in the CUDA+switch-`true` state); a failing route is unloaded and released — or every check-requiring route soft-disabled when NVML cannot be read — (not counted in `models_loaded`, 404 `route.disabled`, series absent, `/v1/runtime` `unavailable_reason: "gpu.insufficient"`) and **the process binds without exiting, with `/v1/embed` unaffected** |

## 14a. Reason-token coverage (map)

- `limits.body_too_large` → F14, F19
- `format.unsupported` → F15, F19
- `decode.failed` → F19 (decode/field validation), F7 (threshold parsing)
- `task.unsupported` → F4, F5
- `auth.invalid_token` → F17
- `inference.failed` → F19 (step 6)
- `response.validation_failed` → F18, F19 (step 7)
- `route.not_found` → F3
- `method.not_allowed` → F3
- `route.disabled` → F34, F37
- `gpu.busy` → F36
- `not_ready` → F2

## 15. What the service owner still has to decide — open items only

| # | Item | Held why |
|---|------|----------|
| 1 | The operator-supplied exact directory for `models.florence_dir_path` and exact filenames for `models.tagger_onnx_path`, `models.tag_names_path`, and `models.tag_suppress_path` in the deployed config (the example paths are placeholders and are not weight URLs) | Paths are the operator's mount layout; only the config key shape and file formats are fixed here. |
| 2 | The operator's content decisions for the two data files: which tag names go in the suppress file and which ordered tag-name list matches the chosen weight variant's score-vector order | Both are operator data by design (section 2c); the files and format are closed, the contents are the operator's. The service validates length and starts fatally if they mismatch. |
| 3 | The operator's deployed integer values for `routes.describe_min_free_bytes` and `routes.tag_min_free_bytes` (section 2e, v12 — the two closed conditions govern when each key is required, and no byte number is written anywhere in this spec) | The minima are the operator's risk decisions about their GPU; the spec fixes key shape, units (bytes), the read method, the compare order, and the soft-disable outcome, and no default value is invented. |

Everything else above, including all other section 4 defaults (the 1024 `max_new_tokens` default is an operator default, not a model-card fact), is decided and closed. Do not reopen decided items. Absent detail means stop and ask, not invent.

---

*End of SPEC — DRAFT v12.*
