# Repo hygiene: what belongs in git and what must stay out

## Belongs in git

- `vision-index.py` — the shipped `vision-index` CLI, at the repo root
  (single file, stdlib only; the installed copy lives at
  `/usr/local/bin/vision-index`)
- `vision_embed/` — the measurement-service source package
- `tests/` — pytest suites, including `tests/test_vision_index_cli.py`
  (the CLI acceptance suite)
- `deploy/` — Dockerfiles, compose example, `.gitignore`, `config.example.yaml`,
  and `WEIGHTS.md`
- `docs/` — specs (including `vision-index-cli-SPEC.md`,
  the current vision-index contract), how-it-works, this file
- `skills/vision-embed/` — the public caller skill (plain markdown)
- `README.md`

## Must stay out of git

- **Secrets** — TLS private keys/certs, bearer token files, anything in
  `tls/` or `secrets/`. Token values never appear in config files or logs.
- **Model weights** — any `.onnx` files (det_10g, w600k_r50,
  openai-ViT-L-14-visual, yolov8n), or model files fetched at build/run.
- **SQLite databases** — any `*.db`/`*.sqlite*` index store or people tables.
- **Bearer files** — `authz.bearer` or any token material, in any format.
- **Deployed config** — real `config.yaml` (only the
  committed `*.example.yaml` copies are tracked).
- **Python build/runtime artifacts** — `.venv/`, `__pycache__/`,
  `.pytest_cache/`.
- **Anything operator-specific** — host names, IP addresses, firewall rules
  tied to a specific site, live topology detail. Example files use
  placeholders.

`deploy/.gitignore` covers `tls/`, `models/`, `secrets/`, and deployed
`config.yaml`. Do not weaken it.
