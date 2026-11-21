---
name: vision-embed
description: "Follow the HTTP contracts of vision-embed and vision-index."
---

# vision-embed / vision-index caller guide

Two separate services, one contract each. Know which one you are calling.

## What each service is

- **vision-embed** is a stateless measurement service. It measures one image
  and returns the measurement. It stores nothing — not image bytes, not
  vectors, not names, no state across requests or restarts.
- **vision-index** is the only place identity state lives: vectors, clusters,
  and assigned names. It calls vision-embed over HTTPS for measurements and
  stores the results. It runs no models itself.

## Authentication

- Both services authenticate with a bearer token supplied via the
  `Authorization` header; the token value is read from an
  operator-supplied file.
- Never place the token in a prompt, a log line, or a committed file.

## vision-embed routes (closed set)

- `GET /healthz`
- `GET /readyz`
- `POST /v1/embed` — multipart form; the field name for the image is
  `image`. An `include_object_vectors` flag is optional.
- `POST /v1/describe` — multipart form. Image field name is `image`; the
  task field is `task`, whose value must be one of the closed set of
  Florence-2 wire tasks (including `<CAPTION>`, `<OD>`, and
  `<MORE_DETAILED_CAPTION>`; the full closed set is in the spec). Returns
  the model (`florence2`), the echoed `task`, and either `text` or a
  `regions` array.
- `POST /v1/tag` — multipart form. Image field name is `image`; the
  threshold field is **`tag_threshold`**. Returns the model (`wd14`),
  `threshold_used`, and `tags` as a list of `tag` plus `score`.
- `GET /v1/runtime` — authenticated operator inventory: `uptime_seconds`,
  the execution provider, the three route flags, the six models in fixed
  order each with `loaded` and `unavailable_reason`, and a `gpu` object.
  It takes no image and loads no model.

The request/response schema and the exact error tokens are defined in
`docs/spec/vision-embed-SPEC.md`, section 6. Do not guess them; consult the
spec rather than pasting or reconstructing it here.

`POST /v1/describe` and `POST /v1/tag` are optional: each may be switched
off at startup (`routes.describe_enabled` / `routes.tag_enabled`, default
`false`) and then returns 404 `route.disabled` — see "Retrying on the POST
routes" below for the retry rule.

## Field-name rules for the multipart routes

- The image part is always named `image` on all three POST routes.
- `describe` takes the task in the field named `task`. Closed values only;
  anything else is 422 `task.unsupported`.
- `tag` takes the threshold in the field named `tag_threshold`. A bare
  `threshold` field is silently ignored and yields 422 `decode.failed` with
  message `missing tag_threshold field`.
- curl pitfall: `curl -F task=<OD>` makes curl treat `<OD>` as a file path
  and it exits 26 before any request is sent. Send the task field with
  `--form-string` (for example `--form-string 'task=<OD>'`).

## Retrying on the POST routes

- 429 `gpu.busy`: another GPU inference is running — the three POST routes
  share one in-process lock under `CUDAExecutionProvider`, there is no
  queue, and the caller retries. Do not retry in a tight loop. On CPU there
  is no lock and no `gpu.busy`.
- 404 `route.disabled`: the route is switched off at startup (the
  `routes.describe_enabled` / `routes.tag_enabled` startup switches, both
  default `false`; read only at startup). Do not retry it as if transient —
  it will not come back without an operator restarting the service.

## vision-index routes (names only; details in spec)

Full shapes are in `docs/spec/vision-index-SPEC.md`, section 6. The routes
are:

- `POST /v1/ingest`
- `POST /v1/identify`
- `POST /v1/name`
- `POST /v1/merge`
- `GET /v1/assets/{id}`
- `GET /v1/clusters`
- `GET /v1/search/person`
- `GET /v1/search/objects`
- plus `GET /healthz` and `GET /readyz`

## State rules

- `identify` persists nothing. It is a pure lookup against the index.
- `ingest` stores vectors and boxes — never image bytes.

## Consent boundaries

- Do not call `name`, `merge`, or `ingest` for a person unless the operator
  asked for that specific action in the current turn. Seeing the call being
  demonstrated or referenced is not permission to enroll someone.

## Hard rules

- Do not invent routes beyond the closed sets above.
- Do not send a filesystem path as the image. The image bytes themselves go
  in the multipart field (field name `image`).
- A filename is not an identity. Never treat a source file's name as a
  person's name or identity.

## Thresholds and determinism

- The face detector's NMS IoU is **0.40**.
- The spec default for `face_match_max` is **0.40**. The live deployment's
  operator config wins — do not hardcode a tighter lab override.
- On `CPUExecutionProvider`, the same image bytes and the same pinned model
  versions produce matching vectors within per-element |a-b| <= 2e-5.
  That envelope covers **embedding vectors only** — it does not extend to
  the describe and tag routes. A caption is generated text and must never
  be byte-compared. And on `CUDAExecutionProvider`, tag scores drift
  slightly between identical runs: measured 2026-11-21, two identical tag
  calls on the same image returned the same 53 tags in the same order with
  a maximum absolute score delta of 2.38e-07. Practical rule: tag
  membership and order are stable, treat tag scores as agreeing to within
  about 1e-5, and never byte-compare captions.
- CUDA can be enabled and is fail-closed, but numeric equality with CPU is
  **not** certified, and no acceptance envelope exists for CUDA output.

## Florence-2 `<OD>` on an object-free image

`<OD>` is a generative task. On an image that contains no objects it can
legitimately return one whole-frame region with score `0.0` rather than an
empty list. Measured 2026-11-21: a solid 256x256 grey image returned
`[{"label": "picture frame", "bbox": <covering the full image>, "score": 0.0}]`.

A caller or test must therefore **not** assert an empty `regions` array.
Assert instead that no region exceeds a confidence floor (i.e. no region
has a score above a small epsilon).

## Installing this skill for an agent

Copy or symlink this `skills/vision-embed/` directory into the target
agent's skills tree. The file is plain markdown and is not
agent-framework-specific.
