# SPEC — vision-index.py v2: people, clusters, and faces (evidence-link model)

## 1. STATUS AND INVARIANTS

**Status:** BUILD CONTRACT, v2. This document fully replaces the v1 spec of the same
path, which described the v1 asset/object/scene data model. That model is withdrawn in
its entirety; no v1 result carries forward except where restated here. Companion
document (illustrative of shape; where the companion mockup document
`docs/spec/vision-index-output-mockups.md` and this spec disagree, THIS SPEC governs).
Precedence is therefore general, not one-off (the same rule already replaces the mockup
rendering of `list clusters` id order with 8.7's sort order, and specifically supersedes
the mockup's former "`delete person` — their clusters become unlinked" row: deleting a
person CASCADES to their clusters and faces per section 8.6).

**Invariants** — each is falsifiable and must never be violated:

1. **I-1 (cluster has no name).** The `clusters` table has no name column. No output,
   verb, argument, or code path may name a cluster. A cluster is addressed only by its
   `cluster_id`. Any string a human typed is a person's name, never a cluster's.
2. **I-2 (a person is a name).** `people.name` is the only name in the database. A
   person row comes into existence only through an explicit human act: `link` with a
   name supplied by the operator. No command derives, suggests, or invents a person name.
3. **I-3 (link is the only person-evidence edge).** The association between a cluster
   and a person exists only as `clusters.person_id`, written only by `link`, cleared
   only by `unlink`; `delete cluster` and `delete person` remove rows through the
   section 4 cascades rather than clearing the column. A
   face's `cluster_id` is never changed by a link or unlink.
4. **I-4 (enroll is mechanical).** `enroll` never creates a person, never writes
   `clusters.person_id`, never converts evidence of nobody into evidence of anyone. It
   only embeds, stores faces, and clusters. There is no code path by which an
   enrollment links anything to anyone.
5. **I-5 (identify and status never write).** `identify` and `status` are read-only
   with respect to the DB: they open the DB read-only in intent, write nothing,
   create nothing. The sole permitted side effect of any POST verb is the section 7
   TOFU pin file, and that verb is `enroll` ONLY: `identify` uses an already-pinned
   host and NEVER creates the pin file (section 7, pre-flight order): with an absent
   pin file the host was never pinned, so `identify` fails `tls.pin_absent` (7a, 9)
   rather than pinning or proceeding unpinned.
6. **I-6 (conservative naming, unknown-preferred).** `identify` NEVER emits a name as
   an assertion unless the band is `matched` (section 5). A wrong name is a defect; a
   missed name is acceptable (DEC-1). In bands `probable` and `unknown` a name may
   appear only in the diagnostic `nearest_person` field, never in an assertion position.
7. **I-7 (closed verbs).** The command set in section 8 is closed. Any verb outside it
   is a defect. The withdrawn v1 verbs (all name-a-cluster and person-photo-join forms,
   the v1 search verbs, and any `enroll --person` form) must not be reintroduced;
   section 2's removed list is normative.
8. **I-8 (face needs exactly one cluster).** `faces.cluster_id` is NOT NULL and
   references `clusters`; a face cannot exist without a cluster. A below-det-floor face
   is stored in a single-face cluster per DEC-5; every face sits in exactly one cluster.
9. **I-9 (vec_hash is identity).** `faces.vec_hash` is the sha256 of the face vector's
   512 float32 little-endian bytes and is the PRIMARY KEY. Re-embedding the same
   vector can never create a duplicate face: the insert is `INSERT OR IGNORE` and the
   per-face result reports the face as already present.
10. **I-10 (no surprise writes on failure).** A failed operation (exit != 0) leaves the
    DB exactly as it was before the command started, EXCEPT for per-file transactions
    of a batch that committed before the failure (section 7.1 — that is stated batch
    semantics, not a violation).
11. **I-11 (no empty clusters).** After any successful command, no cluster row with
    zero faces exists: `delete cluster` deletes a cluster together with its faces
    (cascade), and no verb detaches a face from its cluster. `status` and
    `list clusters` never report a zero-face cluster.
12. **I-12 (constants are constants).** The three threshold constants of section 5
    (`FACE_MATCH_MAX`, `CLUSTER_JOIN_MAX`, `FACE_DET_MIN`) are never overridable by
    environment, config file, or flag. The same no-override rule applies to all four
    named numeric constants of section 5, including `BAND_PROBABLE_MAX`; it fixes
    values, not the count of module-level constants.
13. **I-13 (service stores nothing).** The script is a pure client of the embed
    service; the script's DB is the only durable state besides the TOFU pin file.
14. **I-14 (no bulk payloads).** No output ever contains vector bytes (float or base64
    or hex of a vector), image bytes, or any bulk payload of a vector. The
    64-character `vec_hash` is the only face identity in output; the DB is the only
    place vectors live.
15. **I-15 (no persisted suggestions).** `status` computes its table at query time and
    stores nothing. The suggested `link --near` command is advisory text: the tool
    never executes it and never records it.

## 2. PURPOSE AND SCOPE

`vision-index.py` is a single-file Python CLI (stdlib only) maintaining a local SQLite
index of face vectors from the remote `vision-embed` service under the
**evidence-link model**:

- A **cluster** is a mechanical grouping of similar faces. It has no name and no
  identity beyond its numeric id. It is EVIDENCE of a person (when linked) or evidence
  of nobody (when unlinked).
- A **person** is a name. Naming is a human act, performed by typing a name into
  `link`.
- A **face** carries its 512-d vector and its `det_score` and belongs to exactly one
  cluster. It carries NO position (bbox), NO filename, and NO per-photo provenance
  (section 2 Out-of-scope, DEC-6).

The operator's loop: enroll photos (mechanical clustering), read `status` to see the
unlinked work queue, `link --near` to record clusters as evidence of a person, then
`identify` photos. Fragmentation is harmless by design: many small clusters linked to
one person is the healthy state, not a defect. There is **no consolidation pass**.

**Removed from the v1 model, normatively (living on as anything other than this
withdrawal statement is a defect):** the v1 `rename`, `merge`, and `attach` verb
families and the v1 `find object`/`find like` search verbs; `enroll --add-to` and any
`enroll --person` form; the v1 asset rows (the `assets` table); object vectors
(`object_vec`) and scene vectors (`scene_vec`); `clusters.name`; any concept of
name-a-cluster or naming from a photo. Section 8's closed verb set and section 4's
schema replace them completely.

**Out-of-scope (normative):** no GUI, no daemon, no HTTP server of its own, no
multi-user, no image storage (only vectors and metadata), no text search, no undo
ledger, no sync, no numpy / sqlite-vec (DEC-4). **Per-face position and provenance
are OUT OF SCOPE by binding operator ruling (DEC-6):** per-face positional
information (bbox), source filename, face_index, and any per-photo provenance are
deliberately NOT stored and no column will ever be added for them. This index is a
face-and-cluster identity index, not a photo organiser: a face is identified by its
vector (`vec_hash`) and its cluster (`cluster_id`) and by nothing else; nothing in
the schema or output answers "where is this face in an image" or "which file did it
come from". **Directory walking and recursive enumeration in `enroll` are explicitly
OUT OF SCOPE** (carried over from the v1
decision): batch selection belongs to the shell via multiple positional arguments and
globs (`vision-index enroll a.jpg photos/*.png`); the tool never walks a directory.

### 2.1 DISTRIBUTION AND INSTALL

Unchanged in intent from v1: the single file `vision-index.py` at the repo root; the
installed copy at `/usr/local/bin/vision-index`, mode `0755`, `root:root`; update is a
plain file copy. **Drift detection is content-hash:** `vision-index --version` prints,
as its ONLY token, the sha256 of the script's own source bytes computed AT RUN TIME. A
git short sha is not printed and is not a drift signal (it cannot detect post-commit
edits). Comparing `--version` output from the installed copy and the repo copy detects
any byte difference between them.

## 3. DECISIONS (binding)

**DEC-1 — Conservative, unknown-preferred.** When the tool is unsure, it answers
without asserting a name. `FACE_MATCH_MAX` sits at the LOW end of the acceptable band.
A wrong name is a defect, not a tuning problem; a missed name is the accepted cost.

**DEC-2 — Naming is a human act; never derived from a photo.** Every person name is
typed by the operator as an argument to `link`. `enroll` never creates a person and
never links (I-4). There is no automatic naming on any path, ever.

**DEC-3 — The evidence-link model itself.** The v1 architecture made naming a cluster
the act of creating a person; drift then required heavyweight verbs that moved faces
between clusters to reconcile people over time. The v2 model removes the problem by
construction: clustering stays cheap, mechanical, and never reshaped; linking is the
operator's human act and is reversible at the cluster level (`unlink`). Fragmentation
is resolved by linking MORE clusters as evidence, never by moving faces.

**DEC-4 — Zero dependencies; pure-Python brute-force cosine.** stdlib only, no numpy,
no sqlite-vec, no `enable_load_extension`. Cosine distance over stored 512-d float32
tuples is pure Python, brute-force, over all faces in the DB. Revisit point: ~20,000
faces on the measured v1 cost curve (55 us per 512-d cosine); unchanged here. DEC-4 is
about simplicity and zero install steps — sqlite-vec IS installable on this platform,
which is exactly why this decision names it and forbids it.

**DEC-5 — Below-det-floor faces are stored in a single-face cluster of their own.** A
face with `det_score` strictly below `FACE_DET_MIN` is stored exactly like any other
face, but in a single-face cluster it forms and shares with nothing: it never joins an
existing cluster and no face is ever clustered TOGETHER with it by any automatic rule,
and `link --near` never auto-links that cluster. This keeps `faces.cluster_id` NOT NULL
exactly as the binding schema requires — the alternative (a nullable cluster_id that is
NULL only for below-floor faces) was rejected because it would put the NOT-NULL
guarantee under threshold control — while satisfying the mockup contract "never
clustered": such a face shares no cluster, joins no person's evidence, and stays
evidence of nobody until a human links its cluster.

**DEC-6 — No position, no filename, no per-photo provenance on a face (binding
operator ruling; D6 is WITHDRAWN, not deferred).** Storing a per-face bbox, a source
filename, a face_index, or any per-photo provenance was proposed and REJECTED by the
operator: position and provenance would never be something queried against — this is
a face-and-cluster identity index, not a photo organiser. A face is identified by its
vector and its cluster only. This is permanent out-of-model, not a future item (see
O-6); re-raising it is itself a defect.

## 4. DATA MODEL AND DDL

Three tables, nothing else. The DDL is created in one transaction on first write;
`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA user_version = 3;`.

```sql
CREATE TABLE people (
  person_id  INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE clusters (
  cluster_id INTEGER PRIMARY KEY,
  person_id  INTEGER NULL REFERENCES people(person_id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  inert      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE faces (
  vec_hash   TEXT PRIMARY KEY,     -- sha256 of the 512 float32 LE bytes of `vec`
  vec        BLOB NOT NULL,        -- packed little-endian float32, 512 x 4 bytes
  dim        INTEGER NOT NULL CHECK(dim = 512),
  det_score  REAL NOT NULL,
  cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  added_at   TEXT NOT NULL
);
```

Normative notes:
- `clusters` has NO name column (I-1). `person_id` is NULL for unlinked evidence.
  Deleting a person CASCADES (`ON DELETE CASCADE`): their cluster rows are deleted,
  and each deleted cluster's faces are deleted by `faces.cluster_id ...
  ON DELETE CASCADE`. Person → clusters → faces is removed in ONE transaction.
  Nothing is orphaned and there is NO post-delete residue of unlinked clusters that
  were once linked — such clusters no longer exist at all.
- `faces.vec_hash` = `sha256(struct.pack('<512f', *vector)).hexdigest()`. Because it
  is the PRIMARY KEY, an enroll of the same vector is a conflict, not a duplicate: the
  face insert is `INSERT OR IGNORE` and the per-face result reports `already_present`.
- Deletion semantics: `delete cluster` removes the cluster row; its faces are removed
  by the cascade. `delete person` removes the person row; their linked clusters AND
  all their faces are removed with them by the cascades, in one transaction.
- There is NO migration from a v1 database — deliberately: v1 has `clusters.name`, an
  incompatible data model the tool never reads; a DB whose `PRAGMA user_version` is 1
  is a hard `db.version_unsupported` failure (section 9), unchanged. A **v2 database
  DOES migrate to v3 on first write** (the one-transaction `ALTER TABLE clusters ADD
  COLUMN inert INTEGER NOT NULL DEFAULT 0` + inert backfill + `PRAGMA user_version =
  3` of the design), and a v3 database is read and written directly with no
  migration. A **v0 (fresh) DB is created at v3 directly** — fresh-create DDL +
  `user_version = 3` in one transaction; the fresh-create and migrate paths are
  schema-identical (`pragma table_info(clusters)` includes `inert` either way).
- Timestamps are ISO 8601 UTC with millisecond precision (stdlib `datetime`).
- Vectors are stored as packed little-endian float32 BLOBs via `struct.pack`; never
  vec0, never float16, never base64.

## 5. THRESHOLDS AND BANDS

Four module-level constants, each bound to one use and never shared (I-12). The
three threshold constants govern ingest clustering and the assertive match band;
`BAND_PROBABLE_MAX` is the identification band upper bound (`probable` strictly
below it, `unknown` at or above it). It is deliberately a separate constant from
`CLUSTER_JOIN_MAX` — the two answer different questions and are free to diverge,
even though their values are currently equal.

| constant | value | used by |
|---|---|---|
| `FACE_MATCH_MAX` | `0.28` | `identify` only (assertive match) |
| `CLUSTER_JOIN_MAX` | `0.40` | ingest clustering only |
| `FACE_DET_MIN` | `0.65` | ingest clustering only |
| `BAND_PROBABLE_MAX` | `0.40` | `identify` only (probable/unknown band bound) |

All comparisons are STRICT: band and join membership is strictly below the bound; a
distance of exactly 0.28 is `probable`, never `matched`; exactly 0.40 at ingest does
not join; exactly `<distance>` in `link --near` does not link.

**Three identification bands (normative):**

| band | condition | identify behaviour |
|---|---|---|
| `matched` | d < 0.28 | identify may ASSERT the name; `nearest_person` is an assertion |
| `probable` | 0.28 <= d < BAND_PROBABLE_MAX | identify MUST NOT assert a name; emits `status: "probable"` with `nearest_person` and `nearest_named_distance` as diagnostics only |
| `unknown` | d >= BAND_PROBABLE_MAX | no name in any assertion position; `reason: "no_match"` |

- **d** is the minimum cosine distance from the queried face vector to any face of any
  cluster linked to a person: a person's evidence set is the union of all face vectors
  in its linked clusters, and the minimum is taken over that union. Never a centroid,
  never an average, never a pinned single exemplar.
- **Ambiguous tie (normative):** if two or more PERSONS independently have a minimum
  distance strictly below `FACE_MATCH_MAX` for the queried face, the face is `unknown`
  with `reason: "ambiguous_tie"`; no name is emitted in any assertion position;
  `nearest_named_distance` (the real measured minimum) is still emitted. Do not add a
  margin constant, do not pick the nearest, do not add a third status value (DEC-1/I-6).
- **Ingest joining (normative):** each face at or above `FACE_DET_MIN` joins the
  existing cluster (linked or unlinked alike — clusters carry no identity, so joining
  one is not touching a person) whose member faces it is strictly below
  `CLUSTER_JOIN_MAX` of: the nearest such cluster by minimum face distance. At or above,
  a new cluster is created. Ingest never creates, writes, or clears a `person_id` (I-3,
  I-4).
- **Below-floor rule (normative, DEC-5):** a face strictly below `FACE_DET_MIN` is
  stored in a single-face cluster of its own; it never joins an existing cluster and is
  never a join target; `link --near` never auto-links such clusters. It remains stored
  and queryable (`list faces` shows it; only its cluster stays out of every automatic
  link path).

Normative reference bands from the v1 measurements (same live session): same-person
0.1753–0.2780; different-person 0.3060–0.5317. 0.28 covers every observed same-person
pair with 0.002 margin and sits 0.026 below the nearest observed impostor. These bands
bound `FACE_MATCH_MAX` only; the ingest constants are not derived from them.

## 6. CONFIGURATION

Exactly three environment variables; nothing else is read.

| var | meaning | default |
|---|---|---|
| `VISION_EMBED_URL` | base URL of the vision-embed service (https) | `https://vision-embed.invalid:8443` (placeholder; set it in the env file) |
| `VISION_EMBED_TOKEN` | bearer token sent as `Authorization: Bearer …` | — (no default; required for POST verbs) |
| `VISION_INDEX_DB` | path to the SQLite index file | `~/.local/share/vision-index/index.sqlite3` (absolute; honours `$XDG_DATA_HOME`; parent directory created on first write) |

**DB default (normative).** The default is an ABSOLUTE per-user path, never relative to
the current working directory: `$XDG_DATA_HOME/vision-index/index.sqlite3` when
`XDG_DATA_HOME` is set, else `~/.local/share/vision-index/index.sqlite3` (parent
directory created on first write). An agent runs from arbitrary directories; a
cwd-relative default would silently scatter indexes and produce false negatives.

**Config file layer (normative).** Per-user env file `~/.config/vision-index/env` with
simple `KEY=VALUE` lines for the three names above; blank lines and `#` comments
ignored; no quoting. Precedence, first wins per key: (1) environment variable, (2) the
env file, (3) compiled default; the file is read only for keys not already set. The
file is optional; its absence is not an error. Because the file may hold the bearer
token, it MUST be mode `0600`: a POST verb refuses to use a looser file with
`config.env_file_permissions` (exit 1). The read split (normative):
- `status`, `list *` are LOCAL-ONLY reads: no token, no network, no pin file, and they
  fail `db.absent` if the DB is missing.
- `identify` is a NETWORK read: it DOES require the token and DOES POST (it must embed
  the query photo), but it writes nothing to the index and creates no pin file.

**Token handling (normative).** The token is never written by the script to any file
it creates, never printed in any output, and never echoed in a URL or error detail.
On a POST verb with no token: exactly one line on stderr,
`error: VISION_EMBED_TOKEN is not set`, exit 2, nothing else, no call attempted.

## 7. TRANSPORT

The script is a client; the service stores nothing (I-13). Route used: `POST /v1/embed`
only, multipart/form-data with field name **`image`** (proven required by a live 422
`missing multipart field: image`). The response's `faces` array is the ONLY content
consumed by v2; `scene` and `objects` are neither stored nor deeply validated — their
presence is not required. Response schema (closed for consumed keys):

```json
{
  "request_id": "string",
  "image": {"sha256": "string", "width": 0, "height": 0},
  "faces": [{"bbox": [0, 0, 0, 0], "det_score": 0.0, "model": "string",
             "dim": 512, "vector": [0.0]}]
}
```

A missing `request_id`/`image`/`faces`, a face vector whose length differs from its
`dim`, `dim` != 512, or a non-integral bbox value is `transport.schema_invalid`.

**HTTPS and trust-on-first-use (normative, carried from the implementation's
`tofu_check_and_pin`).** For an https URL: connect, receive the peer certificate, and
record its SHA-256 hex fingerprint in the pin file (`VISION_INDEX_DB` + `.hosts`)
BEFORE any body is sent. If an already-pinned host's fingerprint differs, refuse with
`transport.cert_pin_changed` before any body is sent. No CA bundle, no verify=False;
http URLs get no pin. The default URL is https, so a normal installation pins before
its first upload.

**One POST per image; never retry after an upload.** In a batch, each file is its own
POST; a file is never re-sent after a success, parse error, or schema error. Failures
before the body is sent (connect, TLS) abort that file with the consequences of 7.1.

**Pre-flight check order (normative, EVERY verb, stated once here):** the checks run
in this fixed total order, and each short-circuits the ones after it:

1. config/token resolution (env vars, env file, `config.env_file_permissions` for a
   POST verb, `error: VISION_EMBED_TOKEN is not set` at exit 2 for a POST verb);
2. DB presence check — for read verbs (`identify`, `status`, `list *`) an absent DB
   file fails with `db.absent` BEFORE any network activity; the file is NEVER created
   by a read;
3. TLS pin check (`connect`, peer certificate, pin file compare/copy per the TOFU rule
   above; for a network read (`identify`) with an ABSENT pin file, this step FAILS with
   `tls.pin_absent` — the consequences below);
4. network POST.

Consequences (normative): a read verb must NOT upload anything before `db.absent` is
raised, and MUST NOT create the pin file on ANY path — the pin file is only ever
created by `enroll`, a write verb, which may pin before its first POST. A read with an
absent DB never touches the network and never pins. If a verb requires network access
(`enroll`, `identify`) and the pin file is ABSENT: `enroll` MAY create it (TOFU), but
`identify` MUST NOT — `identify` with an absent pin file FAILS with the token
`tls.pin_absent` (exit 1), stderr exactly one line — the error envelope with detail
telling the operator to run `enroll` once to establish trust. The read path thus never
creates the pin file and never writes trust state.

### 7.1 BATCH TRANSACTION MODEL (normative)

`enroll` with multiple positional arguments is a batch of independent per-file
transactions:

- Each file's work is ONE transaction (`BEGIN IMMEDIATE ... COMMIT`): embed response →
  face inserts (`INSERT OR IGNORE` on `vec_hash`) → cluster inserts. A crash mid-batch
  leaves earlier files' transactions committed. Per-file atomicity, NOT batch atomicity.
- A file that fails is reported and the batch continues to the next file.
- **Exit rule:** exit 0 if at least one file enrolled successfully or was already
  present; exit 1 only if ALL files failed (`batch.all_failed`).
- Progress goes to stderr (one line per file as processed); the summary goes to stdout.
  With `--json`, stdout is a single machine-readable summary object with per-file
  results (8.1); stderr still carries progress only, never the summary.

### 7a. ERROR TAXONOMY AND EXIT CODES

One JSON line on stderr for operational failures, exit `1`:
`{"error": {"reason": "<token>", "detail": "<optional string>"}}`.

**Closed reason-token set (normative):**

| token | meaning |
|---|---|
| `transport.connect` | TCP/DNS failure before body sent |
| `transport.tls` | TLS handshake/cert failure |
| `tls.pin_absent` | `identify` with the DB present but the pin file absent: no fingerprint to compare and a read never creates it; detail tells the operator to run `enroll` once to establish trust |
| `transport.cert_pin_changed` | TOFU fingerprint mismatch (refused before body) |
| `transport.http_status` | non-200; detail carries the code and service reason |
| `transport.schema_invalid` | response outside section 7's schema |
| `transport.timeout` | request timeout; no retry; caller decides to re-run |
| `input.unreadable` | file missing/empty/unreadable; detail = basename ONLY |
| `link.no_such_cluster` | `link <cluster_id>` / `unlink` with absent cluster id |
| `link.no_faces_to_measure` | `link --near` on a person with zero linked faces |
| `link.already_linked` | `link <cluster_id>` on an already-linked cluster |
| `delete.no_such_cluster` | `delete cluster` with absent id |
| `delete.no_such_person` | `delete person`, or `link --near`, with absent name |
| `batch.all_failed` | every file in an enroll batch failed |
| `config.env_file_permissions` | env file looser than 0600, on a POST verb |
| `db.absent` | read verb with no DB file (a read never creates it) |
| `delete.person_required` | `delete person` with no `<name>` argument or an empty string `""`: usage error, exit 2, NOTHING deleted, no prompt shown, no DB row read or written; stderr is one line naming the missing argument and stating that nothing was deleted |
| `no_evidence` | `identify` REASON token inside a success (exit 0), not a failure: zero faces across all linked clusters, so no band is measurable (8.2) |
| `db.version_unsupported` | `PRAGMA user_version` not in {0, 2, 3} |
| `db.corrupt` | sqlite DatabaseError on open/read; no repair |
| `aborted` | declined or EOF-without-y delete confirmation |
| `internal.error` | unexpected exception; detail = exception class name only |

`2` = usage error or missing token (usage text on stderr; no traceback ever). `1` =
operational failure with the envelope above. `0` = success. A top-level
`except Exception` converts any unexpected exception to `internal.error` with detail =
exception class name only (never the traceback, never message text that could carry a
file or network path).

`batch.partial_failure` is NOT an exit token: partial failure is a per-file `failed`
row in the success summary (8.1) with exit 0, and only the all-failed case exits 1.
(Reason tokens never change meaning between versions of this script.)

## 8. COMMANDS

Closed set (I-7): `enroll`, `identify`, `link`, `unlink`, `delete cluster`,
`delete person`, `list people`, `list clusters`, `list faces`, `status`,
`--version`, `usage`/`--help`, and the bare-file/glob form of 8.10
(`vision-index <file> [<file> ...]`, arguments must resolve to regular files).
Nothing else exists. `--json` is accepted by every verb except
`--version`/`usage`. Output shapes below are literal and complete: no optional fields,
no extra fields.

### 8.1 `enroll <photo> [<photo> ...]`

Embeds each file, stores faces (I-9 dedup), clusters each face mechanically (the
ingest rule of section 5): joins the nearest cluster strictly below `CLUSTER_JOIN_MAX`,
else a new cluster; below `FACE_DET_MIN`, a single-face cluster (DEC-5). NEVER creates
a person, NEVER links (I-4). No directory walking, no recursion (section 2).

**stdout success shape (this verb is JSON-native with or without `--json`):**

```json
{"enrolled": true, "files": [
  {"file": "<basename>", "face_status_summary": [{"vec_hash": "<64 hex>",
     "face_status": "created" | "already_present" | "below_det_min"}],
   "image_sha256": "<hex>", "faces_stored": 0, "faces_already_present": 0,
   "faces_below_det_min": 0, "clusters_created": 0, "clusters_joined": 0}
]}
```

- `file` — basename ONLY (never a full path, normative).
- `face_status_summary` has exactly one entry per detected face, per face:
  - `created` — new face stored and clustered;
  - `already_present` — vec_hash conflict with the existing row (I-9): nothing written
    for this face, counts as a successful file;
  - `below_det_min` — new face stored in its own single-face cluster (DEC-5).
- `faces_stored` = count of `created` + `below_det_min`.
- **Counting identity (normative, stated once here and binding everywhere else):**
  `clusters_created` = the number of NEW cluster rows created by this enroll,
  INCLUDING the single-face clusters created for below-floor faces (those clusters
  exist: DEC-5). `clusters_joined` = the number of above-floor faces placed into a
  PRE-EXISTING cluster. `faces_below_det_min` counts the below-floor faces. Below-floor
  faces contribute to `clusters_created` but NEVER to `clusters_joined`.
- **Reconciliation identity (normative):** for every enroll,
  (faces created this call) = (faces that joined existing clusters) + (faces that
  created new clusters); and the total cluster count shown by `list clusters` = the
  sum of `clusters_created` over all enrolls minus clusters removed by any subsequent
  `delete`. Every other counting sentence in this spec — the `list clusters` trailing
  line (8.7), the `status` trailing line (8.8), test V-d(b) — must reconcile with this
  identity; no face and no cluster is counted nowhere or twice.
- A zero-face file is a file with NO detected faces: its entry has empty
  `face_status_summary`, all zero counts, and counts as a successful no-op. Zero
  detected faces is a SUCCESS: the file stored nothing, exit 0, no error token —
  no command requires faces, and no zero-face token exists.
- A failed file's entry carries `"failed": true` and `"error": {"reason", "detail"}`;
  the batch CONTINUES (7.1). Every file gets exactly one entry.

**Failures:** a whole-batch inability to even produce per-file rows (e.g. zero
arguments) is usage (exit 2). A batch with all files failed exits 1 with stderr
`batch.all_failed` and stdout summary still carried when per-file rows exist. The DB is
created only AFTER the first successful embed of the batch (I-10): an entirely failed
batch leaves NO DB file.

**Progress (stderr, one line per file as processed):**

```
enroll: 1/3 photo_a.jpg -> 2 faces (1 created, 1 already_present)
enroll: 2/3 photo_b.jpg FAILED input.unreadable
```

### 8.2 `identify <photo>`

Read-only (I-5). POST the image, store nothing. For each detected face, compute the
minimum distance to each person's evidence set (section 5), classify into the three
bands, and emit:

```json
{"image_sha256": "<hex>", "faces": [
  {"face_index": 0, "det_score": 0.87,
   "band": "matched", "confidence": "matched",
   "nearest_person": "alice", "nearest_named_distance": 0.21}
]}
```

- `band` — one of the three bands, the authoritative field; `confidence` states the
  band name as well (redundant by design, the first-200-bytes rule of section 13).
- Band `matched` → `nearest_person` is an ASSERTION of identity (I-6).
- Band `probable` → `nearest_person` and `nearest_named_distance` are diagnostics;
  identify MUST NOT assert the name in any other field (I-6).
- Band `unknown` → `reason` is `"no_match"`, `"ambiguous_tie"` (section 5), or
  `"no_evidence"` (the empty-evidence rule below); when a person IS measurable,
  `nearest_person` still carries the nearest person's name as a pure measurement
  label. `nearest_named_distance` is `null` only when the DB holds zero faces across
  all linked clusters (the empty-set minimum).
- **Empty evidence set (normative; see the `no_evidence` taxonomy row in section 7):** when the DB holds ZERO faces across all
  linked clusters — no people, or people with zero linked faces — every queried face
  is emitted with band `unknown`, `reason: "no_evidence"`, `nearest_named_distance:
  null`, and `nearest_person` absent (`null`) — and identify MUST NOT assert any name
  (I-6). The empty set is never fabricated into a distance or a name.
- Faces are emitted one per element of the embed response's `faces` array, in that
  array order, `face_index` counting from 0 — deterministic, never geometry-sorted.
- Zero faces detected is a SUCCESS: `faces: []`, exit 0, no error token (a zero-face
  file is a success, never an error; when faces ARE detected and the linked evidence is
  empty, each face carries band `unknown`, reason `no_evidence`, per the rule above).

### 8.3 `link <cluster_id> <person_name>`

Records that this cluster is EVIDENCE of that person: sets `clusters.person_id` to the
person row with that name, creating the person row if the name is new (DEC-2 — this is
the ONLY sanctioned creation of a person, and the operator typed the name). The verb is
never distance-gated and never derived from a photo.

- Absent cluster id → `link.no_such_cluster` (exit 1, DB unchanged).
- Already-linked cluster → `link.already_linked` (exit 1, DB unchanged); the operator
  must `unlink` first. No silent re-pointing: destroying a link needs an explicit step.
- Success stdout:

```json
{"linked": true, "cluster_id": 0, "person": "<name>", "created_person": false}
```

### 8.4 `link --near <person_name> <distance>`

Links EVERY unlinked cluster whose minimum distance to any face of any cluster already
linked to that person is STRICTLY below `<distance>`. One transaction writes all links
(I-3).

- **Preconditions (normative):** the person must exist (`delete.no_such_person` if
  not), and the person must have at least one face across linked clusters: if the
  person has ZERO faces, the tool has nothing to measure against and REFUSES with
  `link.no_faces_to_measure` (exit 1, detail explaining zero faces to measure against).
  A distance is never computed over an empty set and never fabricated.
- `<distance>` parses as a non-negative number; anything else is usage (exit 2).
- **Skipped clusters are reported with a reason each (normative):** every unlinked
  cluster NOT linked by this command appears in the `skipped` array with either
  `distance_above` (measured min distance at or above `<distance>`) or
  `above_floor_only` (every face of the cluster is strictly below `FACE_DET_MIN`, so
  DEC-5 keeps it unlinked regardless of its measured distance — such a cluster may
  still be linked by hand with 8.3).
- Strictly-below semantics: d == `<distance>` is NOT linked.

**stdout success shape** (in the shared illustrative DB of 8.7, before this command
clusters 2, 4, 5 are unlinked; a measured `0.50` links 4, skips 2 by distance and 5
because it is entirely below-floor):

```json
{"linked_near": true, "person": "alice", "distance": 0.50,
 "linked_clusters": [4], "skipped": [
   {"cluster_id": 2, "reason": "distance_above",
    "nearest_person_distance": 0.5313},
   {"cluster_id": 5, "reason": "above_floor_only",
    "nearest_person_distance": 0.5867}]}
```

### 8.5 `unlink <cluster_id>`

The cluster is evidence of NOBODY: sets `clusters.person_id` to NULL. Faces are never
touched (I-3). Absent id → `link.no_such_cluster` (exit 1). Already-unlinked is an
idempotent success (exit 0, `already_unlinked: true`).

```json
{"unlinked": true, "cluster_id": 0}
```

### 8.6 `delete cluster <id> [--yes]` / `delete person <name> [--yes]`

- **A person must be named (normative, binding).** `delete person` with NO `<name>`
  argument, or with an empty string `""` as the name, deletes NOTHING: exit 2, stderr
  is exactly one line — `usage: delete person <name> (nothing deleted; a person must
  be named — no default, no wildcard, no selection, no prompt)` — error token
  `delete.person_required` (section 7a). NO prompt is shown, NO DB connection is
  opened for writing, NO row is read or deleted. There is no "delete all", no
  wildcard, no default target, no interactive selection; an empty name is not a
  person.
- `delete cluster` with no `<id>` argument deletes NOTHING: exit 2, stderr exactly one
  line — `usage: delete cluster <id> (nothing deleted)`.
- `delete` alone, with no subcommand (neither `person` nor `cluster`), deletes
  NOTHING: exit 2 usage error, stderr exactly one line — `usage: delete person
  <name> | delete cluster <id> (nothing deleted)`.
- No other delete target exists; there is no all-people or all-clusters form (I-7).
- **Cascade semantics (normative).** `delete cluster <id>` removes the cluster row;
  its faces go with it (`ON DELETE CASCADE`). `delete person <name>` removes the
  person row AND their linked clusters AND every face of those clusters, in ONE
  transaction (person → clusters → faces, both cascades). Nothing is orphaned and
  there is no post-delete residue of unlinked clusters that were once linked: such
  clusters no longer exist at all. Absent id → `delete.no_such_cluster` (exit 1);
  absent name → `delete.no_such_person` (exit 1); nothing deleted in either case.
- **Confirmation (normative).** A valid target WITHOUT `--yes` shows ONE line on
  STDERR naming the target and its blast radius (real counts read from the DB before
  any deletion):
  - person: `delete person <name>? <P> linked cluster(s) and <F> face(s) will be deleted. proceed? [y/N]`
  - cluster: `delete cluster <id>? <F> face(s) will be deleted. proceed? [y/N]`
  where P is the person's linked-cluster count and F is the number of faces the
  cascade will remove. stdin `y` (a line consisting of exactly `y`) proceeds;
  ANYTHING else, or EOF, aborts with exit 1 `aborted`, DB unchanged, no stdout JSON.
  `--yes` skips the prompt entirely and behaves exactly as a typed `y`, making the
  verbs safe for non-interactive callers that acknowledge destruction explicitly at
  the call site.

```json
{"deleted": "cluster", "cluster_id": 5, "faces_deleted": 1}
{"deleted": "person", "person": "alice", "clusters_deleted": 2, "faces_deleted": 42}
```

### 8.7 `list people` / `list clusters` / `list faces [--json]`

Text tables per the mockup document; `--json` emits the same content as one JSON
object. DB-only: no token, no file, no network (section 7 pre-flight order).

**Shared illustrative DB.** All examples in this section and in 8.4 and 8.8 describe
the SAME database state, so every number reconciles exactly. That state:

- people: `alice` (linked to clusters 1 and 3), `bob` (linked to nothing, zero faces
  — a valid state: created, no evidence yet).
- clusters (unlinked: 2, 4, 5): cluster 1 = 30 faces, cluster 3 = 12 faces, cluster 2
  = 8 faces, cluster 4 = 5 faces, cluster 5 = 1 face (a below-floor single-face
  cluster of DEC-5). All added 2026-11-21.
- Totals: 5 clusters, 2 of them linked to alice, 3 unlinked; 56 faces DB-wide, of
  which 14 live in unlinked clusters.

Trailing-line variables (normative, used by EVERY list trailing line in this spec):
**N** = the number of people in the DB; **M** = the number of UNLINKED clusters in the
DB; **F** = the TOTAL number of faces in the DB (not only unlinked ones); **F′** where
used = the number of faces living in unlinked clusters.

**Sort order (normative, stated once here, overriding any mockup rendering of id
order):** `list people` sorts by name ASCENDING; `list clusters` sorts by `faces`
DESCENDING then `cluster_id` ASCENDING (most evidence first is the useful order — this
agrees with the mockup's most-faces-first display and deliberately replaces the older
id-ascending rule); `list faces` sorts by `cluster_id` ASCENDING then `vec_hash`
ASCENDING.

- `list people`:

```
$ vision-index list people

PEOPLE
  name        faces  linked clusters  last added
  alice         42                2  2026-11-21
  bob             0                0  -

  2 people. 3 clusters unlinked (56 faces).
```

  `faces` counts faces across all linked clusters of the person; `linked clusters`
  counts those clusters; `last added` is the latest `added_at` among those faces (`-`
  when the person has zero faces). Trailing line, with the variables above:
  `<N> person(s). <M> clusters unlinked (<F> faces).` — here N=2, M=3, F=56.
- **`list people` with ZERO people (normative empty state):**

```
$ vision-index list people

PEOPLE
  name        faces  linked clusters  last added

  0 people. 5 clusters waiting for a first link (56 faces).
```

  one line after the (always-printed) header: `0 people. M clusters waiting for a
  first link (F′ faces).` where M and F′ are the unlinked-cluster and unlinked-face
  counts. `--json` with zero people — the trailing keys are ALWAYS present in every
  `list people --json` output (not only the empty case):

```json
{"people": [], "clusters_unlinked": 5, "faces_in_unlinked": 56}
```

  (the non-empty `--json` shape below carries the same two keys after `"people"`.)
- `list clusters`:

```
$ vision-index list clusters

CLUSTERS
  id  person    faces  added
   1  alice       30  2026-11-21
   3  alice       12  2026-11-21
   2  -             8  2026-11-21
   4  -             5  2026-11-21
   5  -             1  2026-11-21

  5 clusters: 2 linked, 3 unlinked
```

  **No name column** (I-1); `-` means unlinked evidence. Trailing line:
  `<total> clusters: <L> linked, <M> unlinked` with L + M = total (consistent with the
  reconciliation identity of 8.1).
- `list faces`:

```
$ vision-index list faces

FACES
  vec_hash    cluster  person   det    added
  3f9a2c8b1d  1        alice   0.80   2026-11-21
  9c1e44aa20  2        -        0.91   2026-11-21
  c28d5b01e7  3        alice   0.86   2026-11-21

  56 faces, 2 clusters linked, 3 unlinked
```

  `vec_hash` truncated to 10 chars in the table view; `--json` carries the full 64.
  `det` is `det_score` at 2dp. Trailing line: `<F> faces, <L> clusters linked, <M>
  unlinked` — same variables as `list clusters`.
- `--json` shapes:

```json
{"people": [{"name": "<name>", "faces": 0, "linked_clusters": 0,
             "last_added": "<ISO|null>"}],
 "clusters_unlinked": 0, "faces_in_unlinked": 0}
{"clusters": [{"cluster_id": 0, "person": "<name|null>", "faces": 0,
               "added_at": "<ISO>"}]}
{"faces": [{"vec_hash": "<64 hex>", "cluster_id": 0, "person": "<name|null>",
            "det_score": 0.0, "added_at": "<ISO>"}]}
```

### 8.8 `status`

READ-ONLY (I-5, I-15). Shows the unlinked work queue and ends with a suggested
`link --near` command. Rows are sorted by `faces` DESCENDING then `cluster_id`
ASCENDING (the 8.7 order applied to the unlinked set).

**Shared illustrative DB:** the example below is the same state as 8.7's.

```
$ vision-index status

UNLINKED CLUSTERS
  id  faces  nearest person  distance  nearest cluster  distance
   2      8  alice             0.5313  cluster 4          0.4210
   4      5  alice             0.4535  cluster 2          0.4110
   5      1  alice             0.5867  cluster 4          0.4310

  3 unlinked clusters, 14 faces
  suggested: vision-index link --near alice 0.60
```

- **Two distances (normative):** `nearest person/distance` — the person with the
  minimum distance from this unlinked cluster's faces to that person's evidence set,
  and that distance; `nearest cluster/distance` — the nearest OTHER unlinked cluster
  by minimum cross-cluster face distance. The second column makes fragmentation
  visible on the first enroll, not after many photos. `-` when not measurable.
- **Suggestion formula (normative):** let
  `D_max` = the LARGEST measured nearest-person distance among unlinked clusters that
  have a measurable nearest person; if none does, the suggestion line is OMITTED.
  Otherwise the suggested distance is
  `suggested = (ceil(D_max × 100) / 100) + 0.01`
  computed in integer hundredths (never float-rounded: scale to integer cents first,
  divide by 100 only for display). Equivalently: round `D_max` UP to the next
  multiple of 0.01 (already a multiple stays itself), then ADD 0.01. This is
  non-circular and provably inclusive: for every considered cluster d ≤ D_max <
  suggested, so the strictly-below semantics of 8.4 link EVERY cluster whose
  nearest-person distance was measured for the suggestion (below-floor-only clusters
  remain skipped per DEC-5 as `above_floor_only`, a separate 8.4 rule, not an
  exclusion by this formula). Example: D_max = 0.5867 → 0.59 + 0.01 = 0.60; if D_max
  is exactly 0.5400 → 0.54 + 0.01 = 0.55, strictly greater. The suggestion takes the
  exact form `suggested: vision-index link --near <person> <d>`, names a real
  measured person, and is advisory text only (I-15): the tool never executes it and
  never records it.
- **`status` with ZERO unlinked clusters (normative empty state):** the table has no
  data rows and there is no suggestion line; output is:

```
$ vision-index status

UNLINKED CLUSTERS
  id  faces  nearest person  distance  nearest cluster  distance

  0 unlinked clusters.
```

  (the header row is always printed; the trailing line is exactly `0 unlinked
  clusters.`; the suggestion line is never present in this state).
- **Thin-evidence advisory (non-normative addition, documented):** because
  `identify` takes the minimum distance across ALL of a person's enrolled
  faces (DEC-1 consequence, measured live: 1 enrolled face -> `probable` at
  0.3871; 4 faces -> `matched` at 0.1754; see `docs/how-it-works.md`,
  "Known limit"), the TEXT output may carry, after the count line and
  before any suggestion line, one per named person with fewer than 5
  enrolled faces:
  `warning: person '<name>' has <n> face(s) enrolled; identification needs ~5+ to reach 'matched' reliably`
  The line is advisory (I-15), printed in text mode only; the `--json`
  shape above is UNCHANGED — no field was added, no key was widened. Read
  the table as: a text `status` line list ends with the suggested command
  WHEN ONE EXISTS, but thin-evidence warning lines sit between the count
  line and the suggestion. This is documentation of shipped behaviour, not
  a threshold or contract change.
- `--json`:

```json
{"unlinked_clusters": [{"cluster_id": 0, "faces": 1, "created_at": "<ISO>",
   "nearest_person": "<name|null>", "nearest_person_distance": 0.0,
   "nearest_cluster_id": 0, "nearest_cluster_distance": 0.0}],
 "suggested_command": "<string|null>"}
```

  Normative nullability rule: a field whose text-table form is `-` has JSON value
  `null`; no `-` ever appears in JSON, and no `0`/`0.0` is ever emitted as a
  stand-in for "not measurable". Every key in the shape above is ALWAYS present
  (never omitted). Concretely:

  - `nearest_person`: JSON `null` when no nearest person is measurable.
  - `nearest_person_distance`: JSON `null` when no nearest person is measurable.
  - `nearest_cluster_id`: JSON `null` when no nearest OTHER unlinked cluster exists
    to compare against (a lone unlinked cluster), or when the cross-cluster
    distance is not measurable.
  - `nearest_cluster_distance`: JSON `null` under the same conditions as
    `nearest_cluster_id`.
  - `suggested_command`: JSON `null` when no person's distance is measurable.

  With ZERO unlinked clusters, `--json` is exactly
  `{"unlinked_clusters": [], "suggested_command": null}` — `suggested_command` is
  explicitly `null` (the key is NEVER omitted, never an empty string).

  With an unmeasurable row (e.g. one lonely unlinked cluster in a zero-people DB,
  or a lone unlinked cluster with no other cluster to compare against), `--json`
  is exactly:

```json
{"unlinked_clusters": [{"cluster_id": 0, "faces": 1, "created_at": "<ISO>",
   "nearest_person": null, "nearest_person_distance": null,
   "nearest_cluster_id": null, "nearest_cluster_distance": null}],
 "suggested_command": null}
```

### 8.9 `--version`, `usage`/`--help`

- `--version` prints one line on stdout: `<sha256>` — the run-time sha256 of the
  script's own source bytes (section 2.1). No DB, no token, no network, exit 0, no
  arguments accepted.
- `usage` / `--help` print the verb list (the headings of section 8) on stderr;
  `--help` exits 0; `usage` and any unsupported verb exit 2.
- Invocation with NO arguments prints the same help text on stderr and exits 0
  (operator ruling: no input -> help). The help text carries a short
  description for every verb plus one worked example, plain text, no ANSI,
  wrapped for an 80-column terminal. It remains recognisably the verb list of
  section 8. It is always written to STDERR and never to stdout.

### 8.10 bare-file / glob identify (`vision-index <file> [<file> ...]`)

Operator ruling (settles the I-7 conflict): `vision-index <file>` with no verb
identifies faces in that file, and globs must work — `vision-index *.png`
behaves like the unix `file` command: one line per FILE.

**Disambiguation rule (normative, stat-based):** expand each argument as a glob
(bash-style: a pattern with no matches stays a literal argument). If EVERY
expanded argument is an existing regular file
(`stat` says regular file), the invocation is identify. Otherwise — no argument
resolving to a file, or any argument that is not a file — fall through to the
existing behaviour: unsupported verb -> usage text on stderr, exit 2, and NO
network call. A typo like `vision-index enrol x.png` must therefore exit 2 and
make no request to the embed service.

**Output (STDOUT, one line per FILE):**

```
<file>: alice                      matched face (assertion, band `matched`)
<file>: probable alice             probable face (diagnostic only, never a bare name)
<file>: unknown                     faces detected, none reaches matched or probable
<file>: alice, probable bob        multiple faces, joined with ", " in face order
<file>: no faces                    zero detected faces
<file>: <reason>                    read or transport failure, e.g. input.unreadable;
                                    the run continues to the next file
```

- A probable face prints the word `probable` — MANDATORY (DEC-1): a probable
  face must never print a bare name in an assertion position.
- Ambiguous-tie and below-evidence faces count as unknown on their line.
- **AMPLIFIED 2026-11-21 (weak detection):** a face whose `det_score` is
  STRICTLY below `FACE_DET_MIN` (0.65) NEVER contributes its name to the
  summary line. The face still counts as a detected
  face for the `unknown` / `no faces` decision: a file whose every face is
  weak prints `<file>: unknown` (the detections are not trustworthy enough to
  say `no faces`, and no name is asserted). Exit-code rule: `0` requires a
  matched/probable face FROM A FACE THAT IS NOT WEAK; an all-weak file alone
  yields `1`.
- Exit codes:
  - `0` if at least one face across ALL files is matched or probable (any file
    identified someone);
  - `1` if EVERY file is entirely unknown or face-less (this includes the
    all-failed and DB-absent cases: nothing was identified);
  - `2` for usage errors (the disambiguation rule above, or a missing token —
    the same one-line stderr token error as `enroll`/`identify`, exit 2, no
    call).
- `identify <photo>` (8.2) is UNCHANGED: single file, its exact current JSON
  output. Only the bare-file/glob form emits the text lines. `--json` is
  accepted on the bare form and is currently a no-op there (the line format
  emits text; if a JSON array is ever added it is a DEC), so the black-box
  stdout contract does not silently change.

## 9. FAILURE MODES

| failure | behaviour | exit |
|---|---|---|
| no faces in the image (`enroll`) | per-file entry, empty summary, zero counts; nothing stored; batch continues | 0 |
| no faces in the image (`identify`) | `faces: []` | 0 |
| transport failure / non-200 / schema invalid / timeout on one file | that file's entry `"failed": true` with the envelope's reason; batch continues | 0 (unless ALL failed) |
| token missing on `enroll` or `identify` | stderr `error: VISION_EMBED_TOKEN is not set`, nothing else, no call | 2 |
| file unreadable in a batch | `input.unreadable`, detail = basename only; batch continues | 0 unless all failed |
| every file in an enroll batch failed | stdout summary (rows, if any); stderr `batch.all_failed` | 1 |
| `link` / `unlink` absent cluster id | `link.no_such_cluster`, id echoed, DB unchanged | 1 |
| `link` on an already-linked cluster | `link.already_linked`, DB unchanged | 1 |
| `link --near` on a person with zero linked faces | `link.no_faces_to_measure`, DB unchanged | 1 |
| `link --near` with nonexistent person | `delete.no_such_person`, DB unchanged | 1 |
| `delete` with absent id/name | `delete.no_such_cluster` / `delete.no_such_person`, nothing deleted | 1 |
| `delete person` with no name argument, or `""` | `delete.person_required`, one stderr line stating nothing was deleted; NO prompt, NO DB row read or written | 2 |
| `delete cluster` with no id argument | usage line `usage: delete cluster <id> (nothing deleted)` | 2 |
| `delete` with no subcommand | usage line `usage: delete person <name> \| delete cluster <id> (nothing deleted)` | 2 |
| confirmation declined or EOF | `aborted`, DB unchanged, no stdout JSON | 1 |
| DB absent on a read verb (`identify`, `status`, `list *`) | `db.absent`; the file is NEVER created by a read | 1 |
| DB absent on id-addressed writes (`link`, `unlink`, both deletes) | their own not-found token (a non-existent DB cannot contain the id); NO DB file created | 1 |
| DB `user_version` not in {0, 2, 3} | `db.version_unsupported`, detail `user_version=<n>`; never migrated, never read (v1 and unknown versions only — v2 migrates on first write, v3 reads and writes directly) | 1 |
| corrupted DB | `db.corrupt`; no repair attempt | 1 |
| env file looser than 0600 on a POST verb | `config.env_file_permissions` | 1 |
| `identify` with the DB present but the pin file absent | `tls.pin_absent`, one stderr line telling the operator to run `enroll` once to establish trust; the pin file is never created by a read | 1 |
| crash mid-batch | WAL rolls back the in-flight per-file transaction; earlier per-file commits stand (7.1) | n/a |

Additional normative rows:

| `identify` with the empty evidence set (zero faces across all linked clusters) | every face: band `unknown`, `reason: no_evidence`, `nearest_named_distance: null`, no name asserted (8.2) | 0 |
| any verb, check ordering | the section 7 pre-flight order is TOTAL and short-circuits: config/token → DB presence → TLS pin → POST; a read never uploads before `db.absent` and never creates the pin file | — |

## 10. IDENTITY RULES (operational consequences)

- **A face never joins a person on enroll** (I-4). `list clusters` before and after any
  enroll shows the `person` column UNCHANGED for every cluster, touched or not.
- **Creating a person is always visible as the human act it is** (I-2): `link` emits
  `created_person` truthfully; `link --near` NEVER creates a person (it operates on an
  existing one; a new name there is an error token, not a row).
- **Near-miss names are never volunteered as assertions** (I-6). The diagnostic
  `nearest_person` in `probable`/`unknown` rows is a measurement label; assertion
  fields only ever carry a name under band `matched`.
- **Cascade purity, and no residue from a deleted person (normative, ruling of
  section 4):** `unlink` followed by `link <id> <name>` with the original name
  restores the exact prior visible DB state. `delete person` removes the person
  together with EVERY cluster they were linked to and every face of those clusters,
  in one transaction: nothing remains of the person's links, and NO formerly-linked
  cluster is left behind unlinked — there is no unlinked residue of a deleted
  person's links and no fresh human link is needed to clean anything up, because
  there is nothing left to clean. (The earlier SET-NULL behaviour — clusters left
  unlinked awaiting a fresh link — no longer exists.)
- **Batch observability:** after a batch where file k failed, `list faces` contains
  exactly the faces of files 1..k-1 (plus any already present), and nothing of file k.

## 11. ACCEPTANCE TESTS (normative)

Fixture distances come from a manifest (operator-supplied test data outside the repo;
same shape as other manifests: `fixtures` name → `path` + `faces`, and a symmetric
top-level `pairwise` array of `{a, b, distance}`; the CLI never reads it; a run with
skipped normative tests is not a pass, even if everything else ran green).

1. **V-a THRESHOLD BOUNDARY.** Two images of the same person at a manifest-recorded
   distance strictly below 0.28 (within the same-person band). `enroll` both (separate
   calls), `link` each resulting cluster to a person, then `identify` a never-enrolled
   third image of the same person: `band: "matched"`, `nearest_named_distance` strictly
   below 0.28 and within the manifest band, that ONE name in the assertion position and
   no other. A build with an inflated effective threshold FAILS (the probe reads
   `probable`); the distance is read from the manifest, never hardcoded.
2. **V-b DIFFERENT-PERSON NEGATIVE CONTROL.** Enroll person A and link the clusters;
   enroll a genuinely different person B whose manifest distance to A is at or above
   0.306. `identify` a never-enrolled A image: `band: "matched"` with A's name only; B
   must NOT appear. `identify` a B image against A's evidence: if the manifest records
   that pair at or above 0.40, the output must be `unknown` — proving the bands keep
   the two names apart.
3. **V-c PROBABLE-BAND NON-ASSERTION.** With a linked person whose evidence faces sit
   at a manifest-recorded distance d in [0.28, 0.40) from a probe face, `identify` the
   probe: `band: "probable"`, `nearest_person` present, `nearest_named_distance`
   within [0.28, 0.40), and NO name in any assertion position (I-6) — parse the output
   and assert that the only name-bearing fields are the diagnostic ones. A build whose
   boundary is off by one side (asserting at exactly 0.28) FAILS.
4. **V-d BELOW-DET-FLOOR EXCLUSION.** Fixture whose manifest records a detected face
   with `det_score` strictly below 0.65. Enroll it, then enroll same-person companions:
   assert (a) the face appears in `list faces` (stored and queryable), (b) it sits in a
   single-face cluster (`face_status: "below_det_min"`, and for that enroll
   `clusters_created` counts a full +1 — the below-floor single-face cluster is a
   real cluster row per DEC-5 and the 8.1 counting identity — with
   `clusters_joined: 0`; the 8.1 reconciliation identity holds: 1 face created =
   0 joined + 1 new cluster), (c) NO later enroll moves it into a shared
   cluster, and (d) `link --near` at ANY distance — even far above its measured
   distance to the person — never links its cluster (the `skipped` row reads
   `above_floor_only`), while a manual `link <its cluster> <person>` (8.3) is permitted
   (a human may link anything).
5. **V-e BATCH PARTIAL FAILURE.** Batch of three: one unreadable, one with zero
   detected faces, one valid. Assert exit 0; stdout summary has 3 `files` entries with
   statuses `failed`, a zero-face success, and `created` faces respectively; stderr
   carries progress lines only (no JSON, no summary); the DB gained exactly the valid
   file's faces and nothing of the others. Re-run the same batch: the valid file now
   reports `already_present` for its faces (I-9) and exit remains 0. Variant: a batch
   of three unreadable files — exit 1, `batch.all_failed` on stderr, and NO DB file
   when none existed.
6. **V-f LINK/UNLINK ROUND-TRIP.** Enroll one file; snapshot `list clusters`;
   `link <cluster> <name>`: the person appears in `list people` with the cluster
   counted; `unlink <cluster>`: `-` in `list clusters`, the cluster stays in the
   unlinked queue of `status`; `link` again with the same name: the visible state is
   identical to the state after the first link. Assert `link.already_linked` fires on a
   linked cluster and `unlink` of an unlinked cluster exits 0 idempotently.
7. **V-g IDENTIFY ACROSS MULTIPLE CLUSTERS OF ONE PERSON.** Enroll two same-person
   fixtures whose manifest pair distance is at or above 0.40, so they form TWO clusters;
   `link` BOTH ids to one person. Assert (a) `list people` shows one person with both
   clusters; (b) `identify` of a never-enrolled third image near either cluster returns
   `band: "matched"` with that person and a distance within the manifest band — the
   minimum is taken over the UNION of both clusters' faces; (c) `status` before the
   second link shows both clusters unlinked with a measurable nearest person, and the
   suggested `link --near` command's distance is EXACTLY the 8.8 formula
   `(ceil(D_max × 100) / 100) + 0.01` applied to the measured nearest-person distances
   of the shown table (a test recomputes it and compares against the 8.8 formula); the command executed
   verbatim links BOTH clusters (strictly-below holds at the suggested distance
   because the formula adds 0.01 after rounding up, so `suggested > D_max ≥ d` for
   every considered cluster).
8. **V-h NO-BYTES / NO-ABSOLUTE-PATH / NO-TRACEBACK.** For every verb and every failure
   mode above: stdout and stderr contain no vector bytes (float, base64, or hex), no
   image bytes, and no absolute path (basenames only); every failure is exactly one
   stderr line (JSON envelope or one-line token error) with the section 7a exit code;
   every success emits exactly the section 8 shape for the verb — exactly one JSON
   value where the table of section 13.1 says single JSON object, exactly that text
   form where it says otherwise.

Coverage rule (normative): every verb of section 8 has at least one positive and one
failure-path test; every I-n cited in section 8 and every DEC-n of section 3 is
exercised by at least one named test above; no failure mode of section 9 is left
untested.

## 12. DONE GATES (falsifiable)

1. `vision-index.py` exists as one stdlib-only file; `python3 vision-index.py --help`
   exits 0; an AST walk of its imports asserts stdlib-only.
2. The three constants appear once each with the exact values of section 5; no code
   path reads them from the environment, the config file, or a flag (I-12).
3. First write creates the DB with the exact section 4 DDL (including
   `clusters.inert`); `sqlite3 <db> "pragma user_version;"` prints 3 for a fresh
   DB; `pragma journal_mode` prints `wal`; `pragma foreign_keys` reports enabled.
   A v2 DB (user_version 2) migrates to 3 on first write and REMAINS READABLE
   afterwards — `status` and `list people` on the migrated DB succeed.
   [AMEND v2→v3→v4]
4. A v1 DB (user_version 1) is refused with `db.version_unsupported` and never upgraded.
5. Tests V-a through V-h all pass; the run log prints each measured distance next to
   its manifest band.
6. The token never appears in the DB, stdout, or the stderr of a successful call.
7. No file the script writes (DB, TOFU pin file) contains an absolute photo path or the
   token; only basenames, ids, and hashes.
8. `--version` output changes if and only if the script's bytes change (content-hash
   drift detection, section 2.1).
9. Sending a request with the token unset prints exactly
   `error: VISION_EMBED_TOKEN is not set` and exits 2.
10. Output-contract gate: a full run of every verb of section 8, including error paths,
    matches its row of the section 13.1 table exactly.

## 13. AGENT CALLABILITY (normative)

The primary caller is an agent; the human operator is the second. Properties that make
the tool cheap and safe to call, each falsifiable:

1. **One JSON object on stdout** for every verb whose 13.1 row says `single JSON
   object`; no banners or progress on stdout (enroll's progress is stderr-only).
   Compact JSON when stdout is not a TTY; pretty when it is. Never JSON-lines.
2. **No bulk payloads** (I-14): largest legitimate output is the per-face arrays of
   8.1/8.2 plus the `list *` tables, bounded by the service's per-image face cap.
3. **Exit codes are the complete signal:** 0 / 1 / 2 per section 7a; never 0 with an
   error sitting inside the stdout JSON.
4. **Discriminating field first:** `enrolled`, `linked`, `linked_near`, `unlinked`,
   `deleted` lead their shapes; in 8.2 the per-face `band`/`confidence` precede the
   diagnostic fields. A caller can classify from the first 200 bytes.
5. **Stable failure reasons:** the section 7a token set is closed and never changes
   meaning between versions.
6. **No stdin prompts except the two deletes** (8.6). A delete WITHOUT `--yes` prints
   its one-line stderr prompt naming the target and its blast radius (person:
   linked-cluster and face counts; cluster: face count), and `--yes` skips the prompt
   entirely. EOF on stdin during any prompt is the `aborted` exit — never a hang,
   never an implicit yes. `delete person` with no name (or `""`) never prompts at all:
   it exits 2 with `delete.person_required` and nothing is deleted.
7. **Invocation shape:** the DB default is absolute and the token lives in the config
   file, so a bare `vision-index enroll ~/Pictures/x.jpg` works for agent and operator
   alike with no per-invocation environment.

### 13.1 OUTPUT CONTRACT TABLE (single source of truth)

| Command | stdout form | Exit codes |
|---|---|---|
| `enroll <photo>...` | single JSON summary object (8.1) | 0, 1 (`batch.all_failed`), 2 |
| `identify <photo>` | single JSON object (8.2) | 0, 1, 2 |
| `<file> [<file> ...]` / glob (8.10) | one text line per FILE: `<file>: <name|probable <name>|unknown|no faces|reason>` | 0, 1, 2 |
| `link <cluster_id> <person_name>` | single JSON object (8.3) | 0, 1, 2 |
| `link --near <person_name> <distance>` | single JSON object (8.4) | 0, 1, 2 |
| `unlink <cluster_id>` | single JSON object (8.5) | 0, 1, 2 |
| `delete cluster <id> [--yes]` | single JSON object (8.6); no stdout line when declined | 0, 1, 2 |
| `delete person <name> [--yes]` | single JSON object (8.6); no stdout line when declined | 0, 1, 2 |
| `list people` / `list clusters` / `list faces` | tabular text (8.7) | 0, 1 (`db.absent`), 2 |
| `list ... --json` | single JSON object (8.7) | 0, 1 (`db.absent`), 2 |
| `status` | tabular text ending with the suggested command WHEN ONE EXISTS (8.8; the zero-unlinked empty state has no suggestion line) | 0, 1, 2 |
| `status --json` | single JSON object (8.8) | 0, 1, 2 |
| `--version` | bare line: `<sha256>` | 0 |
| `usage` / `--help` | usage text on stderr (no stdout) | 2 (0 for `--help`) |

Every command not in this table does not exist (I-7). Where prose above and this table
disagree, the table wins.

## 14. OPEN ITEMS

- **O-1:** the `status` suggested-distance formula is now fixed normatively in 8.8
  (`(ceil(D_max × 100) / 100) + 0.01`, integer-hundredths arithmetic) — it is
  intentionally simple. If usage shows it overshooting, the FORMULA is re-specified
  here through a new DEC — never retuned silently in code.
- **O-2:** whether `link --near` gains a `--dry-run`. Deferred; the `skipped` report
  plus read-only `status` already previews the queue without writing.
- **O-3 (assumption):** the measured bands generalize beyond this session's fixtures
  (as in v1). Threshold changes go through re-measurement and a new DEC, not edits.
- **O-4:** `identify`'s `confidence` field is the band name repeated by design
  (O-4 of this spec, not v1's); if a numeric confidence is wanted later it is a DEC,
  and it must never become a per-face float derived from the distance in place of the
  band.
- **O-6 (CLOSED — binding operator ruling, never re-raise).** Storing per-face bbox,
  source filename, face_index, or any per-photo provenance was REJECTED by the
  operator, with the reasoning on the record: "why would we store bbox or file name?
  zero benefit… bbox and filename would never be something queried against. This is a
  face+cluster identifier not a file or bbox identifier." The proposal is WITHDRAWN,
  not deferred: the schema will never carry per-face position or provenance. A face
  is identified by its vector and its cluster only (DEC-6, section 2 Out-of-scope).
