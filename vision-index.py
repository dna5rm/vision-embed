#!/usr/bin/env python3
"""vision-index.py — v2 face/people/cluster index CLI (evidence-link model).

Spec: docs/spec/vision-index-cli-SPEC.md (v2, BUILD CONTRACT). A cluster is
mechanical evidence; a person is a name typed by the operator via `link`; a
face carries its 512-d vector and belongs to exactly one cluster.

Python 3 stdlib only. No numpy, no sqlite-vec. Vectors are packed
little-endian float32 BLOBs; cosine distance is pure Python, brute force.
"""

import datetime
import glob
import hashlib
import json
import math
import os
import socket
import sqlite3
import ssl
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import NoReturn

# ---------------------------------------------------------------------------
# Constants (section 5 / I-12): no environment override, no config key, no flag
# ---------------------------------------------------------------------------

FACE_MATCH_MAX = 0.28    # identify only (assertive match band)
CLUSTER_JOIN_MAX = 0.40  # ingest clustering only
FACE_DET_MIN = 0.65      # ingest clustering only (detection floor)
# Identification band upper bound: `probable` when d < BAND_PROBABLE_MAX, else
# `unknown`. Deliberately a SEPARATE constant from CLUSTER_JOIN_MAX even though
# the values are currently equal: they answer different questions (identify
# naming bands vs. ingest clustering) and must be free to diverge.
BAND_PROBABLE_MAX = 0.40

# DEC-7 / DEC-8 (design DEDUP-AND-INCIDENTAL v4): per-response dedup and the
# incidental-face rule. Module-level constants, same I-12 regime (never env,
# config, or flag).
FACE_DEDUP_TINY_MAX = 80      # dedup applies only when BOTH boxes have max side below this (px)
FACE_DEDUP_CENTER_RATIO = 1.25  # ... and centre distance <= this * max box dim
FACE_INCIDENTAL_RATIO = 0.20  # defect 2: relative-size gate, strict inequality

FACE_DIM = 512

# ---------------------------------------------------------------------------
# DDL (section 4): three tables, nothing else. user_version = 3 (fresh-create
# at v3 directly; see DEC-8 and design v4 section M).
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS people (
  person_id  INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id INTEGER PRIMARY KEY,
  person_id  INTEGER NULL REFERENCES people(person_id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  inert      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS faces (
  vec_hash   TEXT PRIMARY KEY,
  vec        BLOB NOT NULL,
  dim        INTEGER NOT NULL CHECK(dim = 512),
  det_score  REAL NOT NULL,
  cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  added_at   TEXT NOT NULL
);
"""

TABLES = ("people", "clusters", "faces")

MIME_BOUNDARY = "visionindex7d1a2cboundary"

RESPONSE_TOP = ("request_id", "image", "faces")
IMAGE_KEYS = {"sha256", "width", "height"}
FACE_KEYS = {"bbox", "det_score", "dim", "vector"}

DEFAULT_EMBED_URL = "https://vision-embed.invalid:8443"  # set VISION_EMBED_URL in the env file

ENV_FILE_VARS = ("VISION_EMBED_URL", "VISION_EMBED_TOKEN", "VISION_INDEX_DB")

POST_VERBS = ("enroll", "identify")


class OpError(Exception):
    """Operational failure: one JSON error line on stderr, exit 1."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def now_iso():
    dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# Output (section 13)
# ---------------------------------------------------------------------------

def emit_json(obj):
    if sys.stdout.isatty():
        text = json.dumps(obj, indent=2)
    else:
        text = json.dumps(obj, separators=(",", ":"))
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def emit_error_and_exit(err):
    obj = {"error": {"reason": err.reason}}
    if err.detail is not None and err.detail != "":
        obj["error"]["detail"] = err.detail
    sys.stderr.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stderr.flush()
    sys.exit(1)


def err_detail(oe):
    return None if oe.detail in (None, "") else oe.detail


USAGE_TEXT = """usage: vision-index <verb> [args]

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
"""


def usage_exit(code=2) -> NoReturn:
    sys.stderr.write(USAGE_TEXT)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Distance arithmetic (section 5, DEC-4)
# ---------------------------------------------------------------------------

def unpack_vec(blob):
    return struct.unpack("<%df" % (len(blob) // 4), blob)


def pack_vec(values):
    return struct.pack("<%df" % len(values), *values)


def vec_hash_of(values):
    return hashlib.sha256(pack_vec(values)).hexdigest()


def cosine_distance(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    return 1.0 - (dot / ((na ** 0.5) * (nb ** 0.5)))


def min_pair_distance(vecs_a, vecs_b):
    """Minimum cross distance over two vec lists; None when either is empty."""
    if not vecs_a or not vecs_b:
        return None
    return min(min(cosine_distance(a, b) for a in vecs_a) for b in vecs_b)


def fmt4(value):
    return "%.4f" % value


# ---------------------------------------------------------------------------
# Per-response face pipeline (DEC-7 dedup, then DEC-8 incidental): applied in
# memory, per embed response, immediately after the response is validated and
# BEFORE any consumer (enroll, identify, bare-file form) sees the faces. Two
# strictly ordered stages (design v3 §3a): dedup first, incidental second.
# ---------------------------------------------------------------------------

def _bbox_xywh(bbox):
    """(x, y, w, h) from a validated response bbox (any accepted shape)."""
    if isinstance(bbox, dict):
        if "x" in bbox:
            return (float(bbox["x"]), float(bbox["y"]),
                    float(bbox["w"]), float(bbox["h"]))
        return (float(bbox["x1"]), float(bbox["y1"]),
                float(bbox["x2"]) - float(bbox["x1"]),
                float(bbox["y2"]) - float(bbox["y1"]))
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def bbox_area(face):
    x, y, w, h = _bbox_xywh(face["bbox"])
    return max(w, 0.0) * max(h, 0.0)


def dedup_faces(faces):
    """DEC-7: greedy in-request duplicate suppression. Faces are processed in
    det_score-DESCENDING order (stable tiebreak: original index ASCENDING);
    a face is accepted unless it is a DUPLICATE of an already-accepted face:
    BOTH max(wA,hA,wB,hB) < FACE_DEDUP_TINY_MAX AND centre_distance <=
    FACE_DEDUP_CENTER_RATIO * max(wA,hA,wB,hB). NO vector test, NO IoU. The
    survivor keeps its own box and vector; the loser is dropped entirely, no
    union. Returns (survivors in ORIGINAL array order, suppressed_count)."""
    order = sorted(range(len(faces)),
                   key=lambda i: (-float(faces[i]["det_score"]), i))
    accepted = []  # (x, y, w, h) of each accepted face
    survivors = []  # indices, original order
    for i in order:
        x, y, w, h = _bbox_xywh(faces[i]["bbox"])
        cand_max = max(w, h)
        dup = False
        for ax, ay, aw, ah in accepted:
            pair_max = max(aw, ah, w, h)
            if pair_max >= FACE_DEDUP_TINY_MAX:
                continue  # tiny-face scope fails: never merge at >= 80 px
            d = math.hypot((x + w / 2.0) - (ax + aw / 2.0),
                           (y + h / 2.0) - (ay + ah / 2.0))
            if d <= FACE_DEDUP_CENTER_RATIO * pair_max:
                dup = True
                break
        if dup:
            continue
        accepted.append((x, y, w, h))
        survivors.append(i)
    return [faces[i] for i in survivors], len(faces) - len(survivors)


def incidental_mask(faces):
    """DEC-8, stage 2 (ONLY over the DEDUPED set — A_max is the largest
    deduped face area): True where bbox_area < FACE_INCIDENTAL_RATIO * A_max
    AND det_score >= FACE_DET_MIN (strict inequality: exactly-ratio is NOT
    incidental). Same order as `faces`."""
    if not faces:
        return []
    a_max = max(bbox_area(f) for f in faces)
    return [bbox_area(f) < FACE_INCIDENTAL_RATIO * a_max
            and float(f["det_score"]) >= FACE_DET_MIN
            for f in faces]



# ---------------------------------------------------------------------------
# Configuration (section 6): env var > env file > compiled default
# ---------------------------------------------------------------------------

def env_file_path():
    return os.path.join(os.path.expanduser("~"), ".config", "vision-index", "env")


def read_env_file_vars():
    path = env_file_path()
    if not os.path.isfile(path):
        return {}
    mode = os.stat(path).st_mode & 0o777
    if (mode & 0o077) != 0:
        raise OpError("config.env_file_permissions")
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in ENV_FILE_VARS:
                out[key] = value.strip()
    return out


def default_db_path():
    xdg = os.environ.get("XDG_DATA_HOME")
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "vision-index", "index.sqlite3")


def resolve_config(is_post):
    """Precedence per key: (1) environment variable, (2) env file, (3) default.
    A looser-than-0600 env file is refused on a POST verb (section 6)."""
    vals = {
        "VISION_EMBED_URL": os.environ.get("VISION_EMBED_URL"),
        "VISION_EMBED_TOKEN": os.environ.get("VISION_EMBED_TOKEN"),
        "VISION_INDEX_DB": os.environ.get("VISION_INDEX_DB"),
    }
    if any(v is None for v in vals.values()) and os.path.isfile(env_file_path()):
        try:
            file_vars = read_env_file_vars()
        except OpError:
            if is_post:
                raise
            # Only a POST verb reaches the token; non-POST verbs proceed
            # without the file's contributions (they need none).
            file_vars = {}
        for key in vals:
            if vals[key] is None:
                vals[key] = file_vars.get(key)
    url = vals["VISION_EMBED_URL"] or DEFAULT_EMBED_URL
    return url, vals["VISION_EMBED_TOKEN"], vals["VISION_INDEX_DB"] or default_db_path()


# ---------------------------------------------------------------------------
# Database (section 4)
# ---------------------------------------------------------------------------

def check_db_version(db_path):
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
    except sqlite3.Error:
        raise OpError("db.corrupt")
    if v not in (0, 2, 3):
        raise OpError("db.version_unsupported", detail="user_version=%d" % v)
    return v


def open_read_only(db_path):
    """Read verbs only: never creates the file. Absent -> db.absent."""
    if not os.path.exists(db_path):
        raise OpError("db.absent")
    check_db_version(db_path)
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
        conn.execute("SELECT 1").fetchone()
        return conn
    except sqlite3.Error:
        raise OpError("db.corrupt")


def open_write_no_create(db_path):
    """Id-addressed writes: no file creation ever. Returns conn or None.
    A v2-format DB migrates to v3 on the first write (design v4 M.1)."""
    if not os.path.exists(db_path):
        return None
    check_db_version(db_path)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        if v == 2:
            migrate_v2_to_v3(conn)
        return conn
    except sqlite3.Error:
        raise OpError("db.corrupt")


def migrate_v2_to_v3(conn):
    """v2 -> v3 migration (design v2 §4.1 / v4 M.1): one transaction —
    ALTER TABLE clusters ADD COLUMN inert, the inert backfill (every
    all-below-FACE_DET_MIN cluster becomes inert, exactly as a newly created
    one would be), then PRAGMA user_version = 3 in the SAME transaction."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(clusters)").fetchall()]
        if "inert" not in cols:
            conn.execute(
                "ALTER TABLE clusters ADD COLUMN inert "
                "INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "UPDATE clusters SET inert = 1 WHERE cluster_id IN "
            "(SELECT c.cluster_id FROM clusters c WHERE NOT EXISTS "
            "(SELECT 1 FROM faces f WHERE f.cluster_id = c.cluster_id "
            "AND f.det_score >= ?))", (FACE_DET_MIN,))
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
    except sqlite3.Error:
        raise OpError("db.corrupt")


def ensure_schema(conn):
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if not set(TABLES).issubset(names):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(DDL)
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    except sqlite3.Error:
        raise OpError("db.corrupt")


def open_index_write(db_path):
    """For enroll: create the DB only when absent (first write, I-10)."""
    conn = open_write_no_create(db_path)
    if conn is not None:
        return ensure_schema(conn)
    parent = os.path.dirname(db_path)
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(DDL)
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
        return conn
    except sqlite3.Error:
        raise OpError("db.corrupt")


def has_schema(conn):
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return set(TABLES).issubset(names)


def all_person_evidence(conn):
    """[(pid, name, [vec, ...]), ...] — the union of face vectors of every
    cluster linked to each person (section 5: never a centroid)."""
    ev = {}
    for blob, pid in conn.execute(
            "SELECT f.vec, c.person_id FROM faces f JOIN clusters c "
            "ON f.cluster_id = c.cluster_id WHERE c.person_id IS NOT NULL"):
        ev.setdefault(pid, []).append(unpack_vec(blob))
    out = []
    for pid, name in sorted(conn.execute(
            "SELECT person_id, name FROM people").fetchall()):
        out.append((pid, name, ev.get(pid, [])))
    return out


def all_cluster_faces(conn):
    """{cluster_id: [(vec, det_score), ...]} for every cluster."""
    out = {}
    for blob, ds, cid in conn.execute(
            "SELECT vec, det_score, cluster_id FROM faces"):
        out.setdefault(cid, []).append((unpack_vec(blob), ds))
    return out


# ---------------------------------------------------------------------------
# Transport (section 7): one POST per image, TOFU pinning
# ---------------------------------------------------------------------------

def pin_path_for(db_path):
    return db_path + ".hosts"


def _tls_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def tofu_check_and_pin(db_path, base_url, allow_pin):
    """https: connect, get the peer certificate, compare/create the pin file
    BEFORE any body is sent. http: no pin, plain context. The pin file is
    created ONLY by enroll (allow_pin=True, I-5)."""
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https":
        return _tls_context()
    host = parsed.netloc
    port = parsed.port or 443
    if db_path is None:
        raise OpError("transport.connect")
    pin_path = pin_path_for(db_path)
    try:
        probe = _tls_context()
        with socket.create_connection((parsed.hostname, port), timeout=30) as sock:
            with probe.wrap_socket(sock, server_hostname=parsed.hostname) as tsock:
                der = tsock.getpeercert(binary_form=True)
    except ssl.SSLError:
        raise OpError("transport.tls")
    except OSError:
        raise OpError("transport.connect")
    if der is None:
        raise OpError("transport.tls")
    fingerprint = hashlib.sha256(der).hexdigest()
    pins = {}
    if os.path.exists(pin_path):
        with open(pin_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and " " in line:
                    h, fp = line.split(" ", 1)
                    pins[h] = fp
    prev = pins.get(host)
    if prev is not None and prev != fingerprint:
        raise OpError("transport.cert_pin_changed")
    if prev is None:
        if not allow_pin:
            raise OpError(
                "tls.pin_absent",
                detail="host was never pinned; run enroll once to establish trust")
        pin_parent = os.path.dirname(pin_path)
        if pin_parent:
            try:
                os.makedirs(pin_parent, exist_ok=True)
            except OSError:
                pass
        pins[host] = fingerprint
        body = "".join("%s %s\n" % (h, fp) for h, fp in sorted(pins.items()))
        with open(pin_path, "w", encoding="utf-8") as f:
            f.write(body)
    return _tls_context()


def multipart_body(image_bytes):
    buf = BytesIO()
    buf.write(("--%s\r\n" % MIME_BOUNDARY).encode())
    buf.write(b"Content-Disposition: form-data; name=\"image\"; filename=\"blob\"\r\n")
    buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
    buf.write(image_bytes)
    buf.write(("\r\n--%s\r\n" % MIME_BOUNDARY).encode())
    buf.write(b"Content-Disposition: form-data; name=\"include_object_vectors\"\r\n\r\n")
    buf.write(b"true")
    buf.write(("\r\n--%s--\r\n" % MIME_BOUNDARY).encode())
    return buf.getvalue()


def schema_error(detail):
    return OpError("transport.schema_invalid", detail=detail)


def validate_bbox(b):
    vals = None
    if isinstance(b, dict) and {"x", "y", "w", "h"} <= set(b.keys()):
        vals = [b["x"], b["y"], b["w"], b["h"]]
    elif isinstance(b, dict) and {"x1", "y1", "x2", "y2"} <= set(b.keys()):
        vals = [b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"]]
    elif isinstance(b, list) and len(b) == 4:
        vals = list(b)
    if vals is None:
        raise schema_error("bbox not x/y/w/h")
    out = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise schema_error("bbox not integral")
        iv = int(v)
        if iv != v:
            raise schema_error("bbox not integral")
        out.append(iv)
    return out


def validate_response(obj):
    if not isinstance(obj, dict):
        raise schema_error("response not an object")
    for key in RESPONSE_TOP:
        if key not in obj:
            raise schema_error("response missing %s" % key)
    if not isinstance(obj["request_id"], str):
        raise schema_error("request_id invalid")
    img = obj["image"]
    if not isinstance(img, dict) or not IMAGE_KEYS.issubset(img.keys()):
        raise schema_error("image keys invalid")
    if not isinstance(img["sha256"], str) or not img["sha256"]:
        raise schema_error("image.sha256 invalid")
    if not isinstance(img["width"], int) or not isinstance(img["height"], int):
        raise schema_error("image dims invalid")
    if not isinstance(obj["faces"], list):
        raise schema_error("faces not a list")
    for f in obj["faces"]:
        if not isinstance(f, dict) or not FACE_KEYS.issubset(f.keys()):
            raise schema_error("face keys invalid")
        if f["dim"] != FACE_DIM:
            raise schema_error("face dim != 512")
        if not isinstance(f["vector"], list) or len(f["vector"]) != f["dim"]:
            raise schema_error("face vector length != dim")
        if isinstance(f["det_score"], bool) or not isinstance(f["det_score"],
                                                             (int, float)):
            raise schema_error("det_score invalid")
        validate_bbox(f["bbox"])  # non-integral bbox -> schema_invalid
    return obj


def load_photo(photo_path):
    base = os.path.basename(photo_path)
    if not os.path.isfile(photo_path) or os.path.getsize(photo_path) == 0:
        raise OpError("input.unreadable", detail=base)
    try:
        with open(photo_path, "rb") as f:
            data = f.read()
    except OSError:
        raise OpError("input.unreadable", detail=base)
    if not data:
        raise OpError("input.unreadable", detail=base)
    return data


def perform_embed(base_url, token, image_bytes, db_path, allow_pin):
    """Pre-flight order (section 7): DB presence is checked by the caller
    BEFORE this; here: TLS pin check, then the network POST. Exactly ONE POST
    per call; the body is never re-sent."""
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme == "https" and not allow_pin and \
            not os.path.exists(pin_path_for(db_path)):
        # a read never creates the pin file (I-5)
        raise OpError("tls.pin_absent",
                      detail="host was never pinned; run enroll once to "
                             "establish trust")
    ctx = tofu_check_and_pin(db_path, base_url, allow_pin)
    base = base_url if base_url.endswith("/") else base_url + "/"
    endpoint = urllib.parse.urljoin(base, "/v1/embed")
    headers = {
        "Authorization": "Bearer %s" % token,
        "Content-Type": "multipart/form-data; boundary=%s" % MIME_BOUNDARY,
    }
    req = urllib.request.Request(
        endpoint, data=multipart_body(image_bytes), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as he:
        detail = str(he.code)
        try:
            svc = json.loads(he.read().decode("utf-8", "replace"))
            if isinstance(svc, dict) and svc.get("detail"):
                detail += " %s" % svc["detail"]
            else:
                detail += " %s" % (he.reason or "")
        except Exception:
            detail += " %s" % (he.reason or "")
        raise OpError("transport.http_status", detail=detail.strip())
    except urllib.error.URLError as ue:
        reason = getattr(ue, "reason", ue)
        if isinstance(reason, ssl.SSLError):
            raise OpError("transport.tls")
        raise OpError("transport.connect", detail=type(reason).__name__)
    except socket.timeout:
        raise OpError("transport.timeout")
    except Exception as exc:
        raise OpError("transport.connect", detail=type(exc).__name__)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        raise schema_error("response not JSON")
    return validate_response(obj)


# ---------------------------------------------------------------------------
# enroll (section 8.1) — per-file transaction, mechanical clustering only (I-4)
# ---------------------------------------------------------------------------

def enroll_one_file(conn, faces, incidental, en_time):
    """Insert faces and cluster inside one BEGIN IMMEDIATE transaction.
    `faces` is the DEDUPED face list (original order), `incidental` the DEC-8
    mask over it. Returns the 8.1 file entry fields."""
    faces_out = []
    stored = 0
    already = 0
    below_floor = 0
    incidental_count = 0
    clusters_created = 0
    clusters_joined = 0
    conn.execute("BEGIN IMMEDIATE")
    existing = all_cluster_faces(conn)  # {cid: [(vec, det), ...]}
    inert_ids = {cid for (cid,) in conn.execute(
        "SELECT cluster_id FROM clusters WHERE inert = 1")}
    for fi, f in enumerate(faces):
        vec = tuple(f["vector"])
        h = vec_hash_of(vec)
        ds = float(f["det_score"])
        row = conn.execute("SELECT vec_hash FROM faces WHERE vec_hash=?",
                           (h,)).fetchone()
        if row is not None:
            # I-9: vec_hash is the PK, INSERT is a conflict: never a duplicate
            faces_out.append({"vec_hash": h, "face_status": "already_present"})
            already += 1
            continue
        if ds < FACE_DET_MIN:
            # DEC-5: single-face cluster of its own; never a join target
            # (created inert: all-below-floor clusters are inert clusters)
            cur = conn.execute(
                "INSERT INTO clusters (person_id, created_at, inert) "
                "VALUES (NULL, ?, 1)", (en_time,))
            conn.execute(
                "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
                "added_at) VALUES (?,?,?,?,?,?)",
                (h, pack_vec(vec), FACE_DIM, ds, cur.lastrowid, en_time))
            faces_out.append({"vec_hash": h, "face_status": "below_det_min"})
            stored += 1
            below_floor += 1
            clusters_created += 1
            existing[cur.lastrowid] = [(vec, ds)]
            inert_ids.add(cur.lastrowid)
            continue
        if incidental[fi]:
            # DEC-8: an above-floor incidental face is stored in its OWN
            # inert=1 single-face cluster; never a join target, never
            # auto-linked (hand-link permitted with the 8.3 advisory).
            cur = conn.execute(
                "INSERT INTO clusters (person_id, created_at, inert) "
                "VALUES (NULL, ?, 1)", (en_time,))
            conn.execute(
                "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
                "added_at) VALUES (?,?,?,?,?,?)",
                (h, pack_vec(vec), FACE_DIM, ds, cur.lastrowid, en_time))
            faces_out.append({"vec_hash": h, "face_status": "incidental"})
            stored += 1
            incidental_count += 1
            clusters_created += 1
            existing[cur.lastrowid] = [(vec, ds)]
            inert_ids.add(cur.lastrowid)
            continue
        # join the nearest cluster it is strictly below CLUSTER_JOIN_MAX of;
        # inert clusters are never join targets (DEC-5/DEC-8, marker-keyed)
        best = None
        for cid, members in existing.items():
            if cid in inert_ids:
                continue
            m = min(cosine_distance(vec, tv) for tv, _d in members)
            if m < CLUSTER_JOIN_MAX and (best is None or m < best[1]):
                best = (cid, m)
        if best is not None:
            cid = best[0]
            conn.execute(
                "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
                "added_at) VALUES (?,?,?,?,?,?)",
                (h, pack_vec(vec), FACE_DIM, ds, cid, en_time))
            status = "created"
            stored += 1
            clusters_joined += 1
            existing[cid].append((vec, ds))
        else:
            cur = conn.execute(
                "INSERT INTO clusters (person_id, created_at) VALUES (NULL, ?)",
                (en_time,))
            conn.execute(
                "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
                "added_at) VALUES (?,?,?,?,?,?)",
                (h, pack_vec(vec), FACE_DIM, ds, cur.lastrowid, en_time))
            status = "created"
            stored += 1
            clusters_created += 1
            existing[cur.lastrowid] = [(vec, ds)]
        faces_out.append({"vec_hash": h, "face_status": status})
    conn.commit()
    created = stored - below_floor
    return {
        "face_status_summary": faces_out,
        "faces_stored": stored,
        "faces_already_present": already,
        "faces_below_det_min": below_floor,
        "_faces_incidental": incidental_count,
        "clusters_created": clusters_created,
        "clusters_joined": clusters_joined,
        "_created": created,
    }


def cmd_enroll(url, token, db_path, photo_paths):
    check_db_version(db_path)
    total = len(photo_paths)
    files_out = []
    successes = 0
    for idx, path in enumerate(photo_paths):
        base = os.path.basename(path)
        prefix = "enroll: %d/%d %s" % (idx + 1, total, base)
        try:
            image = load_photo(path)
        except OpError as oe:
            files_out.append({"file": base, "failed": True, "error": {
                "reason": oe.reason, "detail": err_detail(oe)}})
            sys.stderr.write("%s FAILED %s\n" % (prefix, oe.reason))
            continue
        try:
            resp = perform_embed(url, token, image, db_path, allow_pin=True)
        except OpError as oe:
            files_out.append({"file": base, "failed": True, "error": {
                "reason": oe.reason, "detail": err_detail(oe)}})
            sys.stderr.write("%s FAILED %s\n" % (prefix, oe.reason))
            continue
        conn = open_index_write(db_path)
        try:
            en_time = now_iso()
            faces, suppressed = dedup_faces(resp["faces"])
            entry = enroll_one_file(conn, faces, incidental_mask(faces), en_time)
        finally:
            conn.close()
        entry["image_sha256"] = resp["image"]["sha256"]
        entry["file"] = base
        files_out.append(entry)
        successes += 1
        internal_created = entry.pop("_created")
        # DEC-8 reporting: the 8.1 JSON per-file key set is a PUBLISHED, closed
        # shape (spec section 8 / black-box suite); the incidental count stays
        # internal and surfaces via the face_status_summary enum value
        # "incidental" instead. (Widening requires a spec amendment round.)
        entry.pop("_faces_incidental")
        counts = []
        if internal_created > 0:
            counts.append("%d created" % internal_created)
        if entry["faces_already_present"] > 0:
            counts.append("%d already_present" % entry["faces_already_present"])
        if entry["faces_below_det_min"] > 0:
            counts.append("%d below_det_min" % entry["faces_below_det_min"])
        if suppressed > 0:
            counts.append("%d dedup-suppressed" % suppressed)
        line = "%s -> %d faces" % (prefix, len(entry["face_status_summary"]))
        if counts:
            line += " (%s)" % ", ".join(counts)
        sys.stderr.write(line + "\n")
    emit_json({"enrolled": successes > 0, "files": files_out})
    if successes == 0:
        emit_error_and_exit(OpError("batch.all_failed"))
    return 0


# ---------------------------------------------------------------------------
# identify (section 8.2) — read-only, three bands, conservative (I-5/I-6)
# ---------------------------------------------------------------------------

def band_for_face(vec, persons):
    """persons = [(pid, name, [vec, ...]), ...]. Persons with zero faces are
    not scored. Returns (band, reason, nearest_name, nearest_distance)."""
    scored = []
    for _pid, name, vecs in persons:
        if not vecs:
            continue
        d = min(cosine_distance(vec, pv) for pv in vecs)
        scored.append((d, name))
    if not scored:
        return ("unknown", "no_evidence", None, None)
    nd, nname = min(scored)
    matched = [s for s in scored if s[0] < FACE_MATCH_MAX]
    if not matched:
        # probable band (0.28 <= d < BAND_PROBABLE_MAX, spec section 5 band
        # table): NOT an assertion. Emit status "probable" with the nearest
        # name+distance as diagnostics only; the name stays out of any
        # assertion position (I-5/I-6). A distance >= BAND_PROBABLE_MAX is
        # returned below as unknown/no_match. The band bound is its own
        # constant, not a share of CLUSTER_JOIN_MAX (I-12).
        if nd < BAND_PROBABLE_MAX:
            return ("probable", None, nname, nd)
        return ("unknown", "no_match", nname, nd)
    if len(matched) > 1:
        return ("unknown", "ambiguous_tie", nname, nd)
    bd, bname = matched[0]
    return ("matched", None, bname, bd)


def cmd_identify(url, token, db_path, photo_path):
    check_db_version(db_path)
    image = load_photo(photo_path)
    conn = open_read_only(db_path)
    try:
        persons = all_person_evidence(conn) if has_schema(conn) else []
    finally:
        conn.close()
    resp = perform_embed(url, token, image, db_path, allow_pin=False)
    faces, _suppressed = dedup_faces(resp["faces"])
    # DEC-7: one entry per SURVIVING face; face_index is renumbered over the
    # deduped set, det_score-DESCENDING (stable tiebreak: original index ASC).
    faces = [faces[i] for i in sorted(
        range(len(faces)), key=lambda i: (-float(faces[i]["det_score"]), i))]
    faces_out = []
    for idx, f in enumerate(faces):
        band, reason, name, dist = band_for_face(f["vector"], persons)
        entry = {
            "face_index": idx,
            "det_score": f["det_score"],
            "band": band,
            "confidence": band,
        }
        if reason is not None:
            entry["reason"] = reason
        entry["nearest_person"] = name
        entry["nearest_named_distance"] = dist
        faces_out.append(entry)
    emit_json({"image_sha256": resp["image"]["sha256"], "faces": faces_out})
    return 0


# ---------------------------------------------------------------------------
# bare-file / glob identify (operator ruling, DEC-1, spec new section 8.10)
# ---------------------------------------------------------------------------

def identify_candidate_files(args):
    """Expand each argument as a glob (bash-style: a pattern with no matches
    stays literal). Return the file list when EVERY expanded argument is an
    existing regular file, else None (fall through to usage, exit 2)."""
    expanded = []
    for a in args:
        hits = sorted(glob.glob(a))
        expanded.extend(hits if hits else [a])
    if not expanded:
        return None
    if all(os.path.isfile(p) for p in expanded):
        return expanded
    return None


def cmd_identify_bare(url, token, db_path, paths):
    """`vision-index <file> [<file>...]` / glob form: one text line per FILE,
    like the unix `file` command (spec 8.10). A failed file prints
    `<file>: <reason>` and the run continues."""
    if token is None:
        sys.stderr.write("error: VISION_EMBED_TOKEN is not set\n")
        return 2
    any_ident = False
    for path in paths:
        try:
            image = load_photo(path)
            conn = open_read_only(db_path)
            try:
                persons = all_person_evidence(conn) if has_schema(conn) else []
            finally:
                conn.close()
            resp = perform_embed(url, token, image, db_path, allow_pin=False)
            faces, _suppressed = dedup_faces(resp["faces"])
            parts = []
            for f in faces:
                # Below-floor (weak-detection) faces keep their honest
                # identify measurement but NEVER contribute a name to the
                # bare summary line: a sliver at the image edge must not
                # repeat a person the trustworthy face already named.
                if float(f["det_score"]) < FACE_DET_MIN:
                    continue
                # DEC-1: a probable face NEVER prints a bare name in an
                # assertion position; matched/probable wording is normative.
                band, _reason, name, _dist = band_for_face(f["vector"], persons)
                if band == "probable":
                    parts.append("probable %s" % name)
                elif band == "matched":
                    parts.append(name)
            if parts:
                line = "%s: %s" % (path, ", ".join(parts))
                any_ident = True
            elif faces:
                line = "%s: unknown" % path
            else:
                line = "%s: no faces" % path
        except OpError as oe:
            line = "%s: %s" % (path, oe.reason)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    return 0 if any_ident else 1



# ---------------------------------------------------------------------------
# link (sections 8.3 / 8.4) / unlink (8.5)
# ---------------------------------------------------------------------------

def parse_cluster_id(text):
    try:
        return int(text)
    except ValueError:
        usage_exit(2)


def cmd_link(url, token, db_path, argv):
    if argv and argv[0] == "--near":
        return cmd_link_near(db_path, argv[1:])
    if len(argv) != 2:
        usage_exit(2)
    cluster_id = parse_cluster_id(argv[0])  # unparseable id: usage (exit 2)
    person_name = argv[1]
    conn = open_write_no_create(db_path)
    if conn is None:
        # a non-existent DB cannot contain the id; never create one here
        raise OpError("link.no_such_cluster", detail=str(cluster_id))
    try:
        conn.execute("BEGIN IMMEDIATE")
        crow = conn.execute(
            "SELECT person_id, inert FROM clusters WHERE cluster_id=?",
            (cluster_id,)).fetchone()
        if crow is None:
            conn.rollback()
            raise OpError("link.no_such_cluster", detail=str(cluster_id))
        if crow[0] is not None:
            conn.rollback()
            raise OpError("link.already_linked", detail=str(cluster_id))
        cluster_inert = bool(crow[1])
        prow = conn.execute(
            "SELECT person_id FROM people WHERE name=?",
            (person_name,)).fetchone()
        created_person = prow is None
        if created_person:
            # DEC-2: the ONLY sanctioned creation of a person
            cur = conn.execute(
                "INSERT INTO people (name, created_at) VALUES (?, ?)",
                (person_name, now_iso()))
            pid = cur.lastrowid
        else:
            pid = prow[0]
        conn.execute("UPDATE clusters SET person_id=? WHERE cluster_id=?",
                     (pid, cluster_id))
        conn.commit()
    except sqlite3.Error:
        raise OpError("db.corrupt")
    finally:
        conn.close()
    if cluster_inert:
        # 8.3 amendment (DEC-2/DEC-5/DEC-8 carve-out): the human may link
        # anything; stdout JSON shape unchanged, exactly one stderr advisory.
        sys.stderr.write(
            "warning: cluster %d is inert (below det floor or incidental); "
            "its faces will join this person's evidence\n" % cluster_id)
        sys.stderr.flush()
    emit_json({"linked": True, "cluster_id": cluster_id, "person": person_name,
               "created_person": created_person})
    return 0


def cmd_link_near(db_path, argv):
    if len(argv) != 2:
        usage_exit(2)
    person_name = argv[0]
    try:
        dist = float(argv[1])
    except ValueError:
        usage_exit(2)
    if math.isnan(dist) or dist < 0:
        usage_exit(2)
    conn = open_write_no_create(db_path)
    if conn is None:
        raise OpError("delete.no_such_person", detail=person_name)
    try:
        conn.execute("BEGIN IMMEDIATE")
        prow = conn.execute(
            "SELECT person_id FROM people WHERE name=?", (person_name,)).fetchone()
        if prow is None:
            conn.rollback()
            raise OpError("delete.no_such_person", detail=person_name)
        pid = prow[0]
        evidence = [unpack_vec(blob) for (blob,) in conn.execute(
            "SELECT f.vec FROM faces f JOIN clusters c "
            "ON f.cluster_id = c.cluster_id WHERE c.person_id = ?",
            (pid,)).fetchall()]
        if not evidence:
            conn.rollback()
            raise OpError(
                "link.no_faces_to_measure",
                detail="person has zero linked faces to measure against")
        inert_ids = {cid for (cid,) in conn.execute(
            "SELECT cluster_id FROM clusters WHERE inert = 1")}
        linked_clusters = []
        skipped = []
        for (cid,) in sorted(conn.execute(
                "SELECT cluster_id FROM clusters WHERE person_id IS NULL").fetchall()):
            members = all_cluster_faces(conn).get(cid, [])
            if not members:
                continue  # I-11: no zero-face cluster exists, but stay safe
            d = min(min(cosine_distance(uv, pv) for pv in evidence)
                    for uv, _ds in members)
            if all(ds < FACE_DET_MIN for _v, ds in members):
                skipped.append((cid, "above_floor_only", d))
            elif cid in inert_ids:
                # DEC-8: inert clusters are never auto-linked; the skip REASON
                # stays derived (report only). `above_floor_only` keeps its
                # stable meaning (all-below-floor, necessarily inert);
                # `inert_cluster` fires only for inert clusters with at least
                # one at/above-floor face (the incidental case).
                skipped.append((cid, "inert_cluster", d))
            elif d < dist:
                conn.execute("UPDATE clusters SET person_id=? WHERE cluster_id=?",
                             (pid, cid))
                linked_clusters.append(cid)
            else:
                skipped.append((cid, "distance_above", d))
        conn.commit()
    except sqlite3.Error:
        raise OpError("db.corrupt")
    finally:
        conn.close()
    emit_json({
        "linked_near": True,
        "person": person_name,
        "distance": dist,
        "linked_clusters": linked_clusters,
        "skipped": [{"cluster_id": cid, "reason": reason,
                     "nearest_person_distance": d} for cid, reason, d in skipped],
    })
    return 0


def cmd_unlink(db_path, argv):
    if len(argv) != 1:
        usage_exit(2)
    cluster_id = parse_cluster_id(argv[0])
    conn = open_write_no_create(db_path)
    if conn is None:
        raise OpError("link.no_such_cluster", detail=str(cluster_id))
    try:
        conn.execute("BEGIN IMMEDIATE")
        crow = conn.execute(
            "SELECT person_id FROM clusters WHERE cluster_id=?",
            (cluster_id,)).fetchone()
        if crow is None:
            conn.rollback()
            raise OpError("link.no_such_cluster", detail=str(cluster_id))
        already_unlinked = crow[0] is None
        if not already_unlinked:
            # I-3: the only clearing of clusters.person_id; faces untouched
            conn.execute("UPDATE clusters SET person_id=NULL WHERE cluster_id=?",
                         (cluster_id,))
        conn.commit()
    except sqlite3.Error:
        raise OpError("db.corrupt")
    finally:
        conn.close()
    if already_unlinked:
        emit_json({"unlinked": True, "already_unlinked": True,
                   "cluster_id": cluster_id})
    else:
        emit_json({"unlinked": True, "cluster_id": cluster_id})
    return 0


# ---------------------------------------------------------------------------
# delete (section 8.6)
# ---------------------------------------------------------------------------

def read_confirmation(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
    try:
        line = sys.stdin.readline()
    except Exception:
        line = ""
    if line.rstrip("\r\n") != "y":  # a line consisting of exactly `y`
        raise OpError("aborted")


def cmd_delete(url, token, db_path, argv):
    if not argv:
        sys.stderr.write(
            "usage: delete person <name> | delete cluster <id> (nothing deleted)\n")
        sys.exit(2)
    sub = argv[0]
    rest = argv[1:]
    if sub not in ("person", "cluster"):
        sys.stderr.write(
            "usage: delete person <name> | delete cluster <id> (nothing deleted)\n")
        sys.exit(2)
    yes = False
    if rest and rest[-1] == "--yes":
        yes = True
        rest = rest[:-1]
    if any(a == "--yes" for a in rest):
        usage_exit(2)
    if sub == "person":
        # binding: no name argument or "" deletes NOTHING, exit 2, no prompt
        if len(rest) != 1 or rest[0] == "":
            sys.stderr.write(
                "usage: delete person <name> (nothing deleted; a person must be "
                "named — no default, no wildcard, no selection, no prompt)\n")
            sys.exit(2)
        return _delete_person(db_path, rest[0], yes)
    if not rest:
        sys.stderr.write("usage: delete cluster <id> (nothing deleted)\n")
        sys.exit(2)
    if len(rest) != 1:
        usage_exit(2)
    cluster_id = parse_cluster_id(rest[0])
    return _delete_cluster(db_path, cluster_id, yes)


def _delete_person(db_path, name, yes):
    conn = open_write_no_create(db_path)
    if conn is None:
        raise OpError("delete.no_such_person", detail=name)
    try:
        prow = conn.execute(
            "SELECT person_id FROM people WHERE name=?", (name,)).fetchone()
        if prow is None:
            raise OpError("delete.no_such_person", detail=name)
        pid = prow[0]
        pclusters = conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE person_id=?", (pid,)).fetchone()[0]
        pfaces = conn.execute(
            "SELECT COUNT(*) FROM faces f JOIN clusters c "
            "ON f.cluster_id = c.cluster_id WHERE c.person_id=?",
            (pid,)).fetchone()[0]
        if not yes:
            read_confirmation(
                "delete person %s? %d linked cluster(s) and %d face(s) will be "
                "deleted. proceed? [y/N]" % (name, pclusters, pfaces))
        # one transaction: person -> clusters -> faces via both cascades
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM people WHERE person_id=?", (pid,))
        conn.commit()
    except sqlite3.Error:
        raise OpError("db.corrupt")
    finally:
        conn.close()
    emit_json({"deleted": "person", "person": name,
               "clusters_deleted": pclusters, "faces_deleted": pfaces})
    return 0


def _delete_cluster(db_path, cluster_id, yes):
    conn = open_write_no_create(db_path)
    if conn is None:
        raise OpError("delete.no_such_cluster", detail=str(cluster_id))
    try:
        crow = conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_id=?",
            (cluster_id,)).fetchone()
        if crow is None:
            raise OpError("delete.no_such_cluster", detail=str(cluster_id))
        cfaces = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE cluster_id=?",
            (cluster_id,)).fetchone()[0]
        if not yes:
            read_confirmation(
                "delete cluster %d? %d face(s) will be deleted. proceed? [y/N]"
                % (cluster_id, cfaces))
        conn.execute("BEGIN IMMEDIATE")
        # faces go with the cluster via the ON DELETE CASCADE
        conn.execute("DELETE FROM clusters WHERE cluster_id=?", (cluster_id,))
        conn.commit()
    except sqlite3.Error:
        raise OpError("db.corrupt")
    finally:
        conn.close()
    emit_json({"deleted": "cluster", "cluster_id": cluster_id,
               "faces_deleted": cfaces})
    return 0


# ---------------------------------------------------------------------------
# list (section 8.7) — DB-only reads
# ---------------------------------------------------------------------------

def db_stats(conn):
    m = conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE person_id IS NULL").fetchone()[0]
    f = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    fu = conn.execute(
        "SELECT COUNT(*) FROM faces f JOIN clusters c "
        "ON f.cluster_id = c.cluster_id WHERE c.person_id IS NULL").fetchone()[0]
    return m, f, fu


def load_index(conn):
    if not has_schema(conn):
        return [], [], [], (0, 0, 0)
    people = conn.execute(
        "SELECT person_id, name, created_at FROM people").fetchall()
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(clusters)").fetchall()]
    if "inert" in cols:
        clusters = conn.execute(
            "SELECT cluster_id, person_id, created_at, inert FROM clusters"
        ).fetchall()
    else:
        # a pre-migration (v2) DB stays readable; inert is 0 everywhere
        clusters = [(cid, pid, created, 0) for cid, pid, created in
                    conn.execute(
                        "SELECT cluster_id, person_id, created_at "
                        "FROM clusters").fetchall()]
    faces = conn.execute(
        "SELECT vec_hash, cluster_id, det_score, added_at FROM faces").fetchall()
    return people, clusters, faces, db_stats(conn)


def cmd_list(db_path, sub, json_flag):
    conn = open_read_only(db_path)
    try:
        people, clusters, faces, (m, f, fu) = load_index(conn)
    finally:
        conn.close()
    name_by_pid = {pid: name for pid, name, _ in people}
    pid_by_cid = {cid: pid for cid, pid, _c, _i in clusters}
    l_linked = sum(1 for _cid, pid, _c, _i in clusters if pid is not None)

    if sub == "people":
        agg = {}  # pid -> [face_count, latest added_at]
        lcount = {}
        for cid, pid, _c, _i in clusters:
            if pid is not None:
                lcount[pid] = lcount.get(pid, 0) + 1
                agg.setdefault(pid, [0, None])
        for _h, cid, _d, added in faces:
            pid = pid_by_cid.get(cid)
            if pid is not None and pid in agg:
                agg[pid][0] += 1
                if added is not None and (agg[pid][1] is None or added > agg[pid][1]):
                    agg[pid][1] = added
        rows = []
        for pid, name, _created in sorted(people, key=lambda r: r[1]):
            cnt, latest = agg.get(pid, [0, None])
            rows.append({"name": name, "faces": cnt,
                         "linked_clusters": lcount.get(pid, 0),
                         "last_added": latest})
        if json_flag:
            emit_json({"people": rows, "clusters_unlinked": m,
                       "faces_in_unlinked": fu})
            return 0
        n = len(rows)
        lines = ["", "PEOPLE",
                 "  %-10s %6s  %16s  %s" % ("name", "faces", "linked clusters",
                                            "last added")]
        for r in rows:
            lines.append("  %-10s %6d  %16d  %s" % (
                r["name"], r["faces"], r["linked_clusters"],
                "-" if r["last_added"] is None else r["last_added"][:10]))
        lines.append("")
        if n == 0:
            lines.append("%d people. %d clusters waiting for a first link (%d faces)."
                         % (n, m, fu))
        else:
            lines.append("%d %s. %d clusters unlinked (%d faces)."
                         % (n, "person" if n == 1 else "people", m, f))
        sys.stdout.write("\n".join(lines) + "\n")
        return 0

    if sub == "clusters":
        fcount_by_cid = {}
        n_inert = sum(1 for _c, _p, _cr, i in clusters if i)
        for _h, cid, _d, _a in faces:
            fcount_by_cid[cid] = fcount_by_cid.get(cid, 0) + 1
        rows = [{"cluster_id": cid,
                 "person": name_by_pid.get(pid) if pid is not None else None,
                 "faces": fcount_by_cid.get(cid, 0), "added_at": created}
                for cid, pid, created, _i in clusters]
        rows.sort(key=lambda r: (-r["faces"], r["cluster_id"]))
        if json_flag:
            emit_json({"clusters": rows})
            return 0
        lines = ["", "CLUSTERS",
                 "  %4s  %-8s  %5s  %s" % ("id", "person", "faces", "added")]
        for r in rows:
            lines.append("  %4d  %-8s  %5d  %s" % (
                r["cluster_id"], r["person"] if r["person"] else "-",
                r["faces"], r["added_at"][:10]))
        lines.append("")
        lines.append("%d clusters: %d linked, %d unlinked%s"
                     % (len(rows), l_linked, m,
                        (", %d inert" % n_inert) if n_inert else ""))
        sys.stdout.write("\n".join(lines) + "\n")
        return 0

    # faces
    rows = [{"vec_hash": h, "cluster_id": cid,
             "person": name_by_pid.get(pid_by_cid.get(cid))
             if pid_by_cid.get(cid) is not None else None,
             "det_score": det, "added_at": added}
            for h, cid, det, added in faces]
    rows.sort(key=lambda r: (r["cluster_id"], r["vec_hash"]))
    if json_flag:
        emit_json({"faces": rows})
        return 0
    lines = ["", "FACES",
             "  %-10s  %-8s  %-8s  %-6s %s" % ("vec_hash", "cluster", "person",
                                                "det", "added")]
    for r in rows:
        lines.append("  %-10s  %-8d  %-8s  %-6s %s" % (
            r["vec_hash"][:10], r["cluster_id"],
            r["person"] if r["person"] else "-",
            "%.2f" % r["det_score"], r["added_at"][:10]))
    lines.append("")
    lines.append("%d faces, %d clusters linked, %d unlinked"
                 % (len(rows), l_linked, m))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


# ---------------------------------------------------------------------------
# status (section 8.8) — read-only work queue + suggestion (I-15)
# ---------------------------------------------------------------------------

def suggestion_cents(d_max):
    """(ceil(D_max * 100) / 100) + 0.01 in integer hundredths (8.8)."""
    return int(math.ceil(round(d_max * 100, 9))) + 1


# DEC-1 documentation aid: `identify` matches against the NEAREST face across
# ALL of a person's enrolled faces, so the band the probe lands in depends on
# how many photos are enrolled. Measured (temp DB, unseen probe of the same
# person): 1 enrolled face -> probable at 0.3871; 4 enrolled faces -> matched
# at 0.1754 (FACE_MATCH_MAX = 0.28). Fewer than ~5 enrolled faces therefore
# means `probable`, not `matched`, even for a genuine match. NOT tunable away
# by lowering FACE_MATCH_MAX (that would match strangers — the exact failure
# DEC-1 rejects); surfaced as a status warning line instead (advisory, I-15).
THIN_EVIDENCE_MIN_FACES = 5


def thin_evidence_warnings(people, person_vecs):
    """Text-only status advisory lines for named people with < 5 faces.

    Deliberately NOT emitted in JSON output: the 13.1 `status --json` shape is
    closed ({unlinked_clusters, suggested_command}, no optional fields per
    section 8's no-optional rule), and widening it is a DEC, not an edit.
    """
    warnings = []
    for pid, name, _created in people:
        if not name:
            continue
        n = len(person_vecs.get(pid, []))
        if n < THIN_EVIDENCE_MIN_FACES:
            warnings.append(
                "person '%s' has %d face(s) enrolled; identification needs "
                "~%d+ to reach 'matched' reliably"
                % (name, n, THIN_EVIDENCE_MIN_FACES))
    return warnings


def cmd_status(db_path, json_flag):
    conn = open_read_only(db_path)
    try:
        people, clusters, faces, (m, f, fu) = load_index(conn)
        rows = []
        suggested = None
        if not has_schema(conn):
            people, clusters, faces = [], [], []
        name_by_pid = {pid: name for pid, name, _ in people}
        person_vecs = {}
        created_by_cid = {cid: created for cid, _p, created, _i in clusters}
        inert_ids = {cid for cid, _p, _c, i in clusters if i}
        inert_count = len(inert_ids)
        for blob, pid in conn.execute(
                "SELECT f.vec, c.person_id FROM faces f JOIN clusters c "
                "ON f.cluster_id = c.cluster_id WHERE c.person_id IS NOT NULL"):
            person_vecs.setdefault(pid, []).append(unpack_vec(blob))
        unlinked = {}
        for blob, cid in conn.execute(
                "SELECT f.vec, c.cluster_id FROM faces f JOIN clusters c "
                "ON f.cluster_id = c.cluster_id WHERE c.person_id IS NULL"):
            if cid in inert_ids:
                # DEC-8: the work queue EXCLUDES inert clusters (they are not
                # auto-link work; the hand path is their only route)
                continue
            unlinked.setdefault(cid, []).append(unpack_vec(blob))
        m_queue = len([c for c, pid, _cr, i in clusters
                       if pid is None and not i])
        order = sorted(unlinked, key=lambda c: (-len(unlinked[c]), c))
        for cid in order:
            members = unlinked[cid]
            pn = pd = cn = cd = None
            for pid in sorted(person_vecs):
                if not person_vecs[pid]:
                    continue
                d = min_pair_distance(members, person_vecs[pid])
                if d is not None and (pd is None or d < pd):
                    pd, pn = d, name_by_pid.get(pid)
            for cid2 in sorted(unlinked):
                if cid2 == cid:
                    continue
                d = min_pair_distance(members, unlinked[cid2])
                if d is not None and (cd is None or d < cd):
                    cd, cn = d, cid2
            rows.append({"cluster_id": cid, "faces": len(members),
                         "created_at": created_by_cid.get(cid),
                         "nearest_person": pn, "nearest_person_distance": pd,
                         "nearest_cluster_id": cn, "nearest_cluster_distance": cd})
        measurable = [r for r in rows if r["nearest_person_distance"] is not None]
        if measurable:
            dmax_row = max(measurable, key=lambda r: r["nearest_person_distance"])
            cents = suggestion_cents(dmax_row["nearest_person_distance"])
            suggested = "vision-index link --near %s %.2f" % (
                dmax_row["nearest_person"], cents / 100.0)
        warnings = thin_evidence_warnings(people, person_vecs)
    finally:
        conn.close()
    if json_flag:
        emit_json({"unlinked_clusters": rows, "suggested_command": suggested})
        return 0
    lines = ["", "UNLINKED CLUSTERS",
             "  %4s  %5s  %-14s  %8s  %-15s  %8s" % (
                 "id", "faces", "nearest person", "distance",
                 "nearest cluster", "distance")]
    for r in rows:
        lines.append("  %4d  %5d  %-14s  %8s  %-15s  %8s" % (
            r["cluster_id"], r["faces"],
            r["nearest_person"] if r["nearest_person"] is not None else "-",
            fmt4(r["nearest_person_distance"])
            if r["nearest_person_distance"] is not None else "-",
            ("cluster %d" % r["nearest_cluster_id"])
            if r["nearest_cluster_id"] is not None else "-",
            fmt4(r["nearest_cluster_distance"])
            if r["nearest_cluster_distance"] is not None else "-"))
    lines.append("")
    if not rows:
        lines.append("%d unlinked clusters." % m_queue)
    else:
        lines.append("%d unlinked clusters, %d faces" % (m_queue, fu))
    # thin-evidence advisories (DEC-1): printed AFTER the count line and
    # BEFORE the suggestion line, so the output still ends with
    # `suggested: ...` when one exists (8.8); in the zero-unlinked empty
    # state no suggestion exists and these are the final lines.
    for w in warnings:
        lines.append("warning: %s" % w)
    if suggested is not None:
        lines.append("suggested: %s" % suggested)
    if inert_count > 0:
        # DEC-8: inert visibility is TEXT-ONLY (the --json shape is closed).
        lines.append(
            "%d inert cluster(s) excluded from this queue (below det floor "
            "or incidental); hand-link with: link <id> <name>" % inert_count)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


# ---------------------------------------------------------------------------
# --version (section 2.1): run-time sha256 of the script's own source bytes
# ---------------------------------------------------------------------------

def cmd_version():
    with open(os.path.abspath(__file__), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    sys.stdout.write("%s\n" % digest)
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# main dispatch (closed verb set, I-7)
# ---------------------------------------------------------------------------

def strip_json_flag(argv):
    if "--json" in argv:
        return [a for a in argv if a != "--json"], True
    return argv, False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] in ("--help", "-h"):
            usage_exit(0)
        if argv and argv[0] == "usage":
            usage_exit(2)
        if argv and argv[0] == "--version":
            if len(argv) > 1:
                usage_exit(2)
            return cmd_version()
        if not argv:
            # operator ruling: no input -> help on stderr (stderr keeps the
            # black-box "no stdout for help" invariant), exit 0 (8.9).
            sys.stderr.write(USAGE_TEXT)
            return 0
        verb = argv[0]
        rest, json_flag = strip_json_flag(argv[1:])
        # the bare-file form treats ALL argv (verb included) as file args
        bare_args, _jf2 = strip_json_flag(list(argv))

        is_post = verb in POST_VERBS
        url, token, db_path = resolve_config(is_post)

        if verb == "enroll":
            if not rest:
                usage_exit(2)  # zero arguments: usage (8.1)
            if token is None:
                # one line on stderr, exit 2, nothing else, no call (section 6)
                sys.stderr.write("error: VISION_EMBED_TOKEN is not set\n")
                return 2
            return cmd_enroll(url, token, db_path, rest)
        if verb == "identify":
            if len(rest) != 1:
                usage_exit(2)
            if token is None:
                sys.stderr.write("error: VISION_EMBED_TOKEN is not set\n")
                return 2
            return cmd_identify(url, token, db_path, rest[0])
        if verb == "link":
            return cmd_link(url, token, db_path, rest)
        if verb == "unlink":
            return cmd_unlink(db_path, rest)
        if verb == "delete":
            return cmd_delete(url, token, db_path, rest)
        if verb == "list":
            if not rest or rest[0] not in ("people", "clusters", "faces"):
                usage_exit(2)
            return cmd_list(db_path, rest[0], json_flag)
        if verb == "status":
            if rest:
                usage_exit(2)
            return cmd_status(db_path, json_flag)
        # Bare-file / glob identify (spec 8.10): every argument must glob-
        # expand to an existing regular file; anything else falls through
        # to usage (exit 2) with NO network call, so a typo'd verb such as
        # `enrol x.png` never reaches the service.
        paths = identify_candidate_files(bare_args)
        if paths is not None:
            return cmd_identify_bare(url, token, db_path, paths)
        usage_exit(2)
    except OpError as oe:
        emit_error_and_exit(oe)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        emit_error_and_exit(OpError("aborted"))
    except sqlite3.Error:
        emit_error_and_exit(OpError("db.corrupt"))
    except Exception as exc:
        # detail = exception class name ONLY (section 7a)
        emit_error_and_exit(OpError("internal.error", detail=type(exc).__name__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
