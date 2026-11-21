# vision-embed

The code in this repository is MIT-licensed — see [LICENSE](LICENSE); model
weights are not included and are not covered by that license.

Two programs, one contract each.

**vision-embed** is a stateless measurement service. It accepts one image and
returns face boxes with 512-dim embeddings, COCO-80 object labels, and one
768-dim scene vector; two more routes add a generated image caption or
region description (Florence-2) and an ordered tag list with scores (a WD14
ONNX tagger). It stores nothing: no photos, no vectors, no names, no
cluster ids, no state across requests or restarts. It holds no image bytes at
any point after a request is answered.

**vision-index** is a separate stateful identity index. The primary
interface is the `vision-index.py` script at the repo root: a standalone CLI
and library that calls vision-embed over HTTPS for embeddings and keeps
vectors and cluster ids in a per-user SQLite file, recording a name only
when an operator assigns one. It runs no models itself. Indexes do not
share identity data. The HTTP service in `vision_index/` remains for
anyone who needs a shared multi-agent index, but it is no longer the
default. There is no GUI; operations go through the CLI or the HTTP
surfaces.

The split is deliberate: **measurement** (embed) is pure and repeatable —
on `CPUExecutionProvider`, the same image bytes and the same pinned model
versions produce identical vectors within per-element |a-b| <= 2e-5
(CUDA is not covered by that claim) — while **identity** (index) is
where state, clustering, and naming live. Keeping them apart means nothing
derived from identity ever leaks back into the measurement service.

The describe and tag routes do not carry that determinism statement: a
caption is generated text and must not be byte-compared across runs or
providers, and tag scores on CUDA vary slightly between identical runs
(measured: two identical tag calls on the same image returned the same 53
tags in the same order with a maximum absolute score delta of 2.38e-07).
The practical rule is that tag membership and order are stable and scores
agree to about 1e-5, while captions are generated text and must not be
byte-compared across providers.

Documentation:

- `docs/how-it-works.md` — walk-through of the request path, the six model
  roles, config, and observability.
- `docs/spec/` — the contracts. `vision-embed-SPEC.md`,
  `vision-index-SPEC.md`, `vision-embed-cuda-SPEC.md`, and
  `vision-describe-SPEC.md`; the first two are DRAFT v4, the CUDA spec is
  DRAFT v1, and the describe spec (the `/v1/describe`, `/v1/tag`, and
  `/v1/runtime` contract) is DRAFT v12.
- `docs/REPO.md` — what belongs in git and what must stay out.
- `skills/vision-embed/SKILL.md` — a public caller skill (plain markdown) for
  an external agent that drives vision-embed or vision-index.
- `deploy/WEIGHTS.md` — where the weights for the six model roles come from
  (only needed for production embed; the Florence-2 directory and the WD14
  tagger files are required only when the corresponding route switch is
  enabled; tests and vision-index need no weights).
- `deploy/` — generic Docker deployment (`DEPLOY.md`, Dockerfiles, compose,
  config examples).

## What's in the repo

- `vision_embed/` — the measurement service (FastAPI + ONNX Runtime).
- `vision-index.py` — the standalone identity-index CLI (single script at
  the repo root; SQLite-backed, calls the embed service over HTTPS).
- `vision_index/` — the identity index as an HTTP service (HTTPS client of
  the embed service, SQLite-backed). Its HTTP surface is optional; the CLI
  above is the primary interface.
- `tests/` — pytest suite covering the HTTP contracts for both services,
  config fail-closed rules, the NMS unit, the CUDA provider rules,
  observability, and container contracts.

## Running the tests

From the repo root:

```sh
python3.12 -m venv .venv
.venv/bin/pip install fastapi==0.115.6 uvicorn==0.34.0 python-multipart==0.0.20 numpy==2.1.3 pillow==11.0.0 pyyaml==6.0.2 jsonschema==4.23.0 prometheus_client==0.21.1 httpx==0.28.1 pytest
.venv/bin/python -m pytest tests/ -v
```

Those are the app pins from deploy/DEPLOY.md, plus httpx and pytest; GPU
wheels are not required to run the tests. (The image additionally pins
torch==2.6.0, torchvision==0.21.0, transformers==4.48.3, timm==1.0.15,
einops==0.8.1, and nvidia-ml-py==13.615.71 for the Florence-2 and tagger
roles; verify against deploy/Dockerfile before relying on that list.)

The embed tests run against a deterministic fake adapter (seeded from image
content), so no ONNX weights are needed. Skips apply where tests need
container tooling or a real CUDA device. Index tests need no network; the
index client is exercised against an in-process embed stub.

Both services use `adapter_mode: production` only with operator-supplied
weights, TLS cert/key, and a bearer token file; `adapter_mode: test` with
`models.adapter: fake` boots the embed service with zero model files.

Provider choice is config-driven: `CPUExecutionProvider` (default) or
`CUDAExecutionProvider`. CUDA is optional and fail-closed — if it is
configured but the provider is missing, startup is a fatal error, never a
silent CPU fallback. Enabling CUDA requires the GPU device reservation shown
in `deploy/docker-compose.yml`. Note: the code can enable
`CUDAExecutionProvider`, but numeric equality with CPU is not certified, and
there is no acceptance envelope yet.

Note on thresholds: the clustering `face_match_max` default in the index spec
is **0.40**. A deployment may set a tighter live value in its own config; the
spec number is not changed by doing so.

## Using the CLI

`vision-index.py` is a standalone face identity index over the vision-embed
service. It talks HTTP to the embed server for embeddings and keeps a local
SQLite index; it runs no models and no server of its own.

```sh
export VISION_EMBED_TOKEN=<token from an operator-supplied file>
python3 vision-index.py identify photo.jpg
```

Alternatively, write the settings to `~/.config/vision-index/env` (plain
`KEY=VALUE` lines, chmod 600) and export nothing:

```text
# ~/.config/vision-index/env — chmod 600
VISION_EMBED_URL=https://<host>:8443
VISION_EMBED_TOKEN=<bearer token>
```

Precedence for every setting: environment variables win (even when set to
the empty string), then `~/.config/vision-index/env`, then compiled-in
defaults (`VISION_EMBED_URL` defaults to `https://127.0.0.1:8443`, a
placeholder kept generic because the repo is public — point it at your
deployment). The config file **may** hold the bearer token for a fast local
script; if you keep the token there, the file must stay `chmod 600`, owned
by the user, and never be committed or shared. If the token resolves to
empty, the CLI exits with code 2 and names the variable.

Subcommands:

```text
vision-index.py identify <image>            who is in this image; writes nothing
vision-index.py enroll <image> [--name N]   add to index; unnamed cluster unless --name
vision-index.py name <cluster_id> <name>    assign/rename a cluster
vision-index.py clusters [--unnamed]        list clusters
vision-index.py people                      list named people only
```

The index lives by default at `~/.local/share/vision-index/index.sqlite3`
(override with `VISION_INDEX_DB`). `identify` is read-only and writes
nothing to the index; `enroll` does write, adding the image's face clusters.
`python3 vision-index.py --help` lists all options and environment
variables.

## Out of scope until a spec change

GUI, MCP server, phone backup, multi-tenant people tables, public
unauthenticated access, storing image bytes, automatic naming.
