---
name: vision-embed
description: "Follow the HTTP contract of vision-embed and the CLI contract of vision-index."
---

# vision-embed / vision-index caller guide

A measurement service and a CLI, one contract each. Know which one you are
calling.

## What each is

- **vision-embed** is a stateless measurement service. It measures one image
  and returns the measurement. It stores nothing — not image bytes, not
  vectors, not names, no state across requests or restarts.
- **vision-index** is a standalone CLI (`vision-index.py`, installed as
  `/usr/local/bin/vision-index`) and the only place identity state lives:
  a per-user local SQLite database of vectors, clusters, and assigned
  names. It calls vision-embed over HTTPS for measurements and stores the
  results locally. It runs no models itself.

## Authentication

- The vision-embed service authenticates with a bearer token supplied via
  the `Authorization` header; the token value is read from an
  operator-supplied file (for the CLI: `VISION_EMBED_TOKEN`).
- The CLI's `status` and `list` verbs are local-DB-only and need no token
  or network.
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

## vision-index CLI verbs (closed set)

Run `vision-index --help` for the exact syntax. The verbs:

| verb | what it does |
|---|---|
| `enroll <photo> [<photo> ...]` | embeds the photos, stores and clusters their faces into UNNAMED clusters; never names anyone or creates a person. Prints a JSON summary |
| `identify <photo>` | read-only: matches each detected face against named people, prints a JSON object with one entry per detected face |
| `<file> [<file> ...]` (bare / glob, no verb) | like the unix `file` command: one text line per file, `<file>: <names>`; exit 0 only if at least one file matched |
| `link <cluster-id> <person>` | records the cluster as evidence of `<person>` (creates the person on first use) |
| `link --near <person> <dist>` | links every unlinked cluster within `<dist>` of that person |
| `unlink <cluster-id>` | the cluster is evidence of nobody |
| `delete cluster <cluster-id> [--yes]` | deletes the cluster and its faces; prompts unless `--yes` |
| `delete person <name> [--yes]` | deletes the person, their clusters and faces (cascade); prompts unless `--yes` |
| `list people \| list clusters \| list faces [--json]` | read-only tabular listing, `--json` for JSON |
| `status` | read-only: unlinked work queue plus the suggested next command; local DB only |
| `--version` | prints the script's run-time sha256 — use it to check installed-vs-repo drift |

Key contract points for an agent:

- The CLI is a plain CLI, not a JSON RPC: `--help` prints human-readable
  text. `enroll`, `identify`, and other mutating verbs print JSON;
  the bare-file form prints one text line per file.
- Exit codes: `0` success; `1` operation failure, with exactly one JSON
  line on stderr, `{"error":{"reason":"<token>","detail":...}}` where
  `<token>` is from a closed set (`transport.connect`,
  `transport.http_status`, `input.unreadable`, `link.no_such_cluster`,
  `link.already_linked`, `delete.no_such_person`,
  `delete.no_such_cluster`, `aborted`, `db.*`, ...); `2` usage error or
  missing token (`error: VISION_EMBED_TOKEN is not set`). Never a
  traceback.
- A rejected face is reported `unknown`, never a near-miss name. Treat
  `unknown` as unknown.
- `identify` and the bare-file form write nothing to the database (their
  only side effect is the TLS host-fingerprint pin file used for
  trust-on-first-use).
- `link` takes a CLUSTER ID from `list clusters`, not a name.
- Configuration lives in `VISION_EMBED_URL`, `VISION_EMBED_TOKEN`,
  `VISION_INDEX_DB` (env wins over the optional
  `~/.config/vision-index/env` file, which wins over defaults); the default
  DB is per-user (`~/.local/share/vision-index/index.sqlite3`).

Do not guess the JSON shapes; they are defined field-by-field in
`docs/spec/vision-index-cli-SPEC.md` (sections 9, 10, 10.4 and table 16.1).

## Consent boundaries

- Do not call `enroll`, `link`, `unlink`, or any `delete` for a
  person unless the operator asked for that specific action in the current
  turn. Seeing the command being demonstrated or referenced is not
  permission to enroll someone.
- A filename is not a person's name: only the operator types names, via
  `link <cluster-id> <person>`.

## Hard rules

- Do not invent routes beyond the closed set above, and do not invent
  verbs beyond the CLI's closed set above.
- For the service: do not send a filesystem path as the image; the image
  bytes themselves go in the multipart field (field name `image`).
- For the CLI: the argument IS a filesystem path to the photo (the CLI
  reads and hashes the file locally), never image bytes on the command
  line.
- A filename is not an identity. Never treat a source file's name as a
  person's name or identity.

## Thresholds and determinism

- The face detector's NMS IoU is **0.40**.
- The vision-index CLI's face match threshold is a hardcoded module
  constant, `FACE_MATCH_MAX = 0.28` — no environment, config, or
  command-line override exists. The old 0.40 configurable default belongs
  to the superseded HTTP-service spec.
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
