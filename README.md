# vision-embed

The code in this repository is MIT-licensed — see [LICENSE](LICENSE); model
weights are not included and are not covered by that license.

Two programs, one contract each.

**vision-embed** is a stateless measurement service. It accepts one image and
returns face boxes with 512-dim embeddings, COCO-80 object labels, and one
768-dim scene vector. It stores nothing: no photos, no vectors, no names, no
cluster ids, no state across requests or restarts. It holds no image bytes at
any point after a request is answered.

**vision-index** is a separate stateful identity index. It calls vision-embed
over HTTPS, keeps vectors and cluster ids in SQLite, and records a name only
when an operator assigns one. It runs no models itself. One index per
operator: indexes do not share identity data. There is no GUI; the operator
drives both services over their HTTP surfaces.

The split is deliberate: **measurement** (embed) is pure and repeatable —
on `CPUExecutionProvider`, the same image bytes and the same pinned model
versions produce identical vectors within per-element |a-b| <= 2e-5
(CUDA is not covered by that claim) — while **identity** (index) is
where state, clustering, and naming live. Keeping them apart means nothing
derived from identity ever leaks back into the measurement service.

Documentation:

- `docs/how-it-works.md` — walk-through of the request path, the four model
  roles, config, and observability.
- `docs/spec/` — the contracts. `vision-embed-SPEC.md`,
  `vision-index-SPEC.md`, and `vision-embed-cuda-SPEC.md`; the first two
  are DRAFT v4, the CUDA spec is DRAFT v1.
- `docs/REPO.md` — what belongs in git and what must stay out.
- `skills/vision-embed/SKILL.md` — a public caller skill (plain markdown) for
  an external agent that drives vision-embed or vision-index.
- `deploy/WEIGHTS.md` — where the four ONNX weights come from (only needed
  for production embed; tests and vision-index need no weights).
- `deploy/` — generic Docker deployment (`DEPLOY.md`, Dockerfiles, compose,
  config examples).

## What's in the repo

- `vision_embed/` — the measurement service (FastAPI + ONNX Runtime).
- `vision_index/` — the identity index (HTTPS client of the embed service,
  SQLite-backed).
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
wheels are not required to run the tests.

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

## Out of scope until a spec change

GUI, MCP server, phone backup, multi-tenant people tables, public
unauthenticated access, storing image bytes, automatic naming.
