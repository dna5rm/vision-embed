# Repo hygiene: what belongs in git and what must stay out

## Belongs in git

- `vision_embed/` and `vision_index/` source packages
- `vision-index.py` — the standalone identity-index CLI (source at the repo
  root)
- `tests/` and `tests/index/`
- `deploy/` — Dockerfiles, compose example, `.gitignore`, `config.example.yaml`,
  `config.index.example.yaml`, and `WEIGHTS.md`
- `docs/` — specs, how-it-works, this file
- `skills/vision-embed/` — the public caller skill (plain markdown)
- `README.md`

## Must stay out of git

- **Secrets** — TLS private keys/certs, bearer token files, anything in
  `tls/` or `secrets/`. Token values never appear in config files or logs.
- **Model weights** — any `.onnx` files (det_10g, w600k_r50,
  openai-ViT-L-14-visual, yolov8n), or model files fetched at build/run.
- **SQLite databases** — any `*.db`/`*.sqlite*` index store or people tables.
- **Bearer files** — `authz.bearer` or any token material, in any format.
- **Deployed config** — real `config.yaml` / `config.index.yaml` (only the
  committed `*.example.yaml` copies are tracked).
- **Python build/runtime artifacts** — `.venv/`, `__pycache__/`,
  `.pytest_cache/`.
- **Anything operator-specific** — host names, IP addresses, firewall rules
  tied to a specific site, live topology detail. Example files use
  placeholders.

`deploy/.gitignore` covers `tls/`, `models/`, `secrets/`, and deployed
`config.yaml`. Do not weaken it.

## Internal notes

`docs/internal/` holds internal architect/engineer notes for the CUDA work.
These documents still reference internal process and hardware-study detail
and should **not** be published if they still contain lab detail; they are
kept out of the public-facing docs intentionally. Prefer scrubbing over
deleting design decisions.
