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

The request/response schema and the exact error tokens are defined in
`docs/spec/vision-embed-SPEC.md`, section 6. Do not guess them; consult the
spec rather than pasting or reconstructing it here.

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
- CUDA can be enabled and is fail-closed, but numeric equality with CPU is
  **not** certified, and no acceptance envelope exists for CUDA output.

## Installing this skill for an agent

Copy or symlink this `skills/vision-embed/` directory into the target
agent's skills tree. The file is plain markdown and is not
agent-framework-specific.
