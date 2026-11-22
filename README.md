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

**vision-index** is the identity side: a per-user, standalone CLI
(`vision-index.py`, installed as `/usr/local/bin/vision-index`). It is a
single-file, standard-library-only Python program that calls vision-embed
over HTTPS for embeddings and holds the identity state itself — a local
SQLite database of vectors, clusters, and names. It runs no models. Each
user's index is separate: the default database path is per-user, so two
accounts on the same host do not share identity data. The contract is
[`docs/spec/vision-index-cli-SPEC.md`](docs/spec/vision-index-cli-SPEC.md).

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
  `vision-embed-cuda-SPEC.md`, and
  `vision-describe-SPEC.md` cover the service; `vision-index-cli-SPEC.md`
  is the contract for the `vision-index.py` CLI.
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

- `vision-index.py` — the shipped `vision-index` CLI (single file, stdlib only).
- `vision_embed/` — the measurement service (FastAPI + ONNX Runtime).
- `tests/` — pytest suites: `tests/test_vision_index_cli.py` is the
  acceptance suite for the CLI; the remaining suites cover the HTTP
  contracts for the service, config fail-closed rules, the NMS unit, the
  CUDA provider rules, observability, and container contracts.

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
container tooling or a real CUDA device. The CLI acceptance suite
(`tests/test_vision_index_cli.py`) is described below.

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

Note on thresholds: the CLI's face match threshold is a hardcoded module
constant `FACE_MATCH_MAX = 0.28` in `vision-index.py`. There is no
environment variable, config key, or command-line override for it.

## vision-index: the CLI

`vision-index` is a per-user identity index over photos, standing on
vision-embed for the measurements. It is one Python file (`vision-index.py`)
that uses only the standard library, and it keeps all its durable state in
one local SQLite database. The service it talks to (vision-embed) stores
nothing; the CLI's database is the only place faces, vectors, clusters, and
names exist.

The full contract — every command, output shape, and error — is
[`docs/spec/vision-index-cli-SPEC.md`](docs/spec/vision-index-cli-SPEC.md).

### Install

Copy the single file to the installed location:

```sh
cp vision-index.py /usr/local/bin/vision-index
chmod 755 /usr/local/bin/vision-index
```

The spec requires the installed copy (owner `root:root`, mode `0755`) to be
byte-identical to the repo copy; compare `vision-index --version` run from
`PATH` versus the repo file — `--version` prints the script's own sha256 at
run time, so differing values mean the two copies have drifted.

### Quick start

Enroll a photo, look at the unnamed cluster its faces landed in, record that
cluster as evidence of a person with `link`, then identify:

```sh
vision-index enroll a-photo.jpg
vision-index list clusters --json      # find the unnamed cluster's id
vision-index link <cluster-id> alice
vision-index enroll another-photo-of-alice.jpg
vision-index identify new-photo.jpg
```

`enroll` never creates or names a person; it embeds photos and clusters their
faces mechanically. A cluster becomes evidence of a person only through
`link <cluster_id> <person_name>` (or, in bulk, `link --near <name> <dist>`),
which creates the person on first use. Names are typed only by the operator —
there is no automatic naming, ever.

### The bare-file form (primary UX)

With no verb at all, `vision-index <file> [<file> ...]` — including a
glob — identifies every argument, like the unix `file` command, and prints
one text line per file:

```sh
vision-index ~/*.png
# 20221218_125451.png: alice, probable bob
```

Each line is `<file>: <names>`:

- a face at or above the detection floor (`FACE_DET_MIN`), at or below the
  0.28 match threshold, contributes its person's name; a *probable* match
  prints as `probable <name>` rather than a bare name;
- a face below the detection floor does NOT contribute a name to the line;
- if every detected face in a file is below the detection floor, the line
  reads `<file>: unknown`; a file with no detected faces reads
  `<file>: no faces`; a failed file reads `<file>: <reason>` and the run
  continues with the remaining files.

Exit behaviour of the run: `0` if **at least one** file yielded a
trustworthy matched or probable face; `1` otherwise.

Full verb list from `vision-index --help`:

```text
usage: vision-index <verb> [args]

Commands:
  enroll <photo> [<photo> ...]
      Embed photos, store and cluster their faces (no naming, no person
      is created). Prints a JSON summary.
  identify <photo>
      Identify faces in one photo against named people (read-only).
      Prints a JSON object with one entry per detected face.
  <file> [<file> ...]          (bare files or a glob, no verb)
      Identify every argument after glob expansion, like the unix `file`
      command: one text line per file, e.g.
        vision-index ~/*.png
        20221218_125451.png: alice, probable bob
  link <cluster_id> <person_name>
      Record a cluster as evidence of a person.
  link --near <person_name> <dist>
      Link every unlinked cluster within <dist> of the person.
  unlink <cluster_id>
      The cluster is evidence of nobody.
  delete cluster <id> [--yes]
      Delete a cluster and its faces.
  delete person <name> [--yes]
      Delete a person, their clusters and faces (cascade).
  list people | list clusters | list faces [--json]
      Tabular listing (or JSON with --json).
  status
      Unlinked work queue plus the suggested next command (read-only).

  --help                        this text on stderr, exit 0
  --version                     print the run-time sha256 of this script
```

Deletions prompt on stderr (`--yes` skips the prompt). `status` and the
three `list` subcommands never touch the network — they read
only the local database.

### Configuration

Exactly three environment variables:

| variable | meaning | default |
|---|---|---|
| `VISION_EMBED_URL` | base URL of the vision-embed service | `https://vision-embed.invalid:8443` (placeholder; set it to your service) |
| `VISION_EMBED_TOKEN` | bearer token for the service | none; required for anything that POSTs an image |
| `VISION_INDEX_DB` | path to the SQLite index | `~/.local/share/vision-index/index.sqlite3` (absolute, honours `$XDG_DATA_HOME`) |

The default database path is per-user, so each account on a host gets its
own index unless it points `VISION_INDEX_DB` elsewhere. There is no
multi-user mode.

An optional per-user config file, `~/.config/vision-index/env`, can carry
the same three variables as `KEY=VALUE` lines (no quoting; blank lines and
`#` comments ignored). It must be mode `0600` — the CLI refuses a looser
file with a `config.env_file_permissions` error. Precedence, first wins:
environment variable, then the config file, then the compiled default.

### Exit codes and errors

- `0` — success.
- `1` — operation failure. stderr carries exactly one JSON line:
  `{"error": {"reason": "<token>", "detail": "<optional>"}}`, where
  `<token>` is from a closed set such as `transport.connect`,
  `transport.http_status`, `input.unreadable`, `link.no_such_cluster`,
  `delete.no_such_person`, `delete.no_such_cluster`,
  `config.env_file_permissions`, `db.absent`, or `aborted`. The detail
  never contains a full path or another person's name.
- `2` — usage error or missing token (for example,
  `error: VISION_EMBED_TOKEN is not set`).

A traceback never reaches the user; unexpected errors become
`internal.error` with the exception class name only.

### Running the acceptance suite

The suite needs three things from its environment or it skips rather than
inventing test data: `VISION_INDEX_ACCEPTANCE=1` to opt in to live-service
use, a real `VISION_EMBED_TOKEN`, and `VISION_INDEX_FIXTURES` pointing at a
directory of fixture photos with a `manifest.json` of measured distances
(spec section 13a).

```sh
VISION_INDEX_ACCEPTANCE=1 \
VISION_EMBED_TOKEN=<real token> \
VISION_INDEX_FIXTURES=/path/to/fixtures \
python -m pytest tests/test_vision_index_cli.py -v
```

## The abandoned HTTP index service

The previous design was an HTTP service (and library) that exposed index
operations over authenticated routes with TLS certificates, bearer files,
and Prometheus metrics. That service was abandoned and replaced by the
standalone `vision-index` CLI described above. The CLI is the supported
interface, and the current contract is
[`docs/spec/vision-index-cli-SPEC.md`](docs/spec/vision-index-cli-SPEC.md).

## Out of scope until a spec change

GUI, MCP server, phone backup, multi-tenant people tables, public
unauthenticated access, storing image bytes, automatic naming.
