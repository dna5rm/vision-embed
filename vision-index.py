#!/usr/bin/env python3
"""vision-index.py — standalone face identity index over the vision-embed service.

Talks HTTP to a vision-embed server for embeddings and keeps a local SQLite
(sqlite-vec) index of assets, faces and face clusters. No server, no container,
no HTTP surface of its own.

Usage:
  vision-index.py identify <image>            who is in this image; writes nothing
  vision-index.py enroll <image> [--name N]   add to index; unnamed cluster unless --name
  vision-index.py name <cluster_id> <name>    assign/rename a cluster
  vision-index.py clusters [--unnamed]        list clusters
  vision-index.py people                      list named people only

Environment:
  VISION_EMBED_URL      service base URL (default https://127.0.0.1:8443)
  VISION_EMBED_TOKEN    bearer token (default "" — must be set)
  VISION_INDEX_DB       sqlite path (default ~/.local/share/vision-index/index.sqlite3)
  VISION_FACE_MATCH_MAX cosine distance floor for a same-person match (default 0.15)

Output: one JSON object on stdout, indent=2 on a TTY, compact when piped.
Errors: one JSON line on stderr, exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# -- retargetable constants ---------------------------------------------------

DEFAULT_EMBED_URL = "https://127.0.0.1:8443"
DEFAULT_INDEX_DB = "~/.local/share/vision-index/index.sqlite3"
DEFAULT_FACE_MATCH_MAX = 0.15
GPU_BUSY_RETRY_SLEEP = 2.0
NAME_MIN_LEN = 1
NAME_MAX_LEN = 64


def _die(reason: str, detail: str = "") -> None:
    print(json.dumps({"error": reason, "detail": detail}, separators=(",", ":")), file=sys.stderr)
    sys.exit(1)


def _bootstrap_sqlite_vec() -> None:
    """sqlite-vec (plus numpy) is the only non-stdlib dep, needed only when the
    DB opens. Install into a python that has pip and re-exec; if this
    interpreter has no pip at all, fall back to an existing venv one."""
    import glob
    import importlib.util
    import subprocess

    def _importable(python) -> bool:
        return (
            subprocess.run(
                [python, "-c", "import sqlite_vec, numpy"], capture_output=True
            ).returncode
            == 0
        )

    import sys  # noqa: F811 (local rebind reads clearer)

    if importlib.util.find_spec("sqlite_vec") is not None:
        return

    candidates = []
    venv_dir = os.environ.get("VIRTUAL_ENV")
    if venv_dir:
        candidates.append(os.path.join(venv_dir, "bin", "python"))
    candidates += sorted(
        glob.glob(os.path.expanduser("~/.local/venv*/bin/python")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
        reverse=True,
    )

    # A venv interpreter that already has the dep: just move to it.
    for py in candidates:
        if os.path.exists(py) and _importable(py):
            os.execve(py, [py] + sys.argv, os.environ)

    # Otherwise install into the first candidate that carries pip.
    targets = [sys.executable] + [py for py in candidates if os.path.exists(py)]
    seen = set()
    for py in targets:
        if py in seen:
            continue
        seen.add(py)
        code = subprocess.run(
            [py, "-m", "pip", "install", "--quiet", "sqlite-vec"],
            capture_output=True,
        )
        if code.returncode == 0 and _importable(py):
            os.execve(py, [py] + sys.argv, os.environ)
    _die(
        "bootstrap.failed",
        "sqlite-vec could not be installed: no interpreter with pip available",
    )


class _Spinner:
    """Braille spinner for blocking steps on a TTY; dependency-free (rich-free)."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str) -> None:
        self.label = label
        self.on = sys.stderr.isatty()

    def __enter__(self):
        if self.on:
            import threading

            self._stop = threading.Event()
            self._t = threading.Thread(target=self._spin, daemon=True)
            self._t.start()
        else:
            print(self.label + " ...", file=sys.stderr)
        return self

    def _spin(self):
        import itertools

        for f in itertools.cycle(self.FRAMES):
            print(f"\r{f} {self.label}", end="", file=sys.stderr, flush=True)
            self._stop.wait(0.08)

    def __exit__(self, *exc):
        if self.on:
            self._stop.set()
            self._t.join()
            print("\r" + " " * (len(self.label) + 4) + "\r", end="", file=sys.stderr)
        return False


def _open_db(path: str, read_only: bool, token_check: bool = False):
    """Open the index and load the sqlite-vec extension.

    identify opens read-only and never runs migrations — it writes nothing.
    """
    import sqlite3

    _bootstrap_sqlite_vec()
    import sqlite_vec

    real = os.path.expanduser(path)
    if read_only:
        if not os.path.exists(real):
            return None
        uri = "file:" + real.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        Path(real).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(real)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


DDL = """
CREATE TABLE assets (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  sha256 TEXT UNIQUE NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  indexed_at TEXT NOT NULL
);
CREATE TABLE faces (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL REFERENCES assets(id),
  x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
  det_score REAL NOT NULL,
  cluster_id INTEGER REFERENCES clusters(id)
);
CREATE VIRTUAL TABLE face_vec USING vec0(
  face_id INTEGER PRIMARY KEY,
  embedding float[512]
);
CREATE TABLE clusters (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('face','object')),
  name TEXT,
  named_at TEXT,
  exemplar_face_id INTEGER,
  merged_into INTEGER REFERENCES clusters(id)
);
CREATE UNIQUE INDEX clusters_kind_name ON clusters(kind, name) WHERE name IS NOT NULL;
CREATE TABLE objects (
  id INTEGER PRIMARY KEY,
  asset_id INTEGER NOT NULL REFERENCES assets(id),
  label TEXT NOT NULL,
  score REAL NOT NULL,
  x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
  cluster_id INTEGER
);
CREATE VIRTUAL TABLE object_vec USING vec0(
  object_id INTEGER PRIMARY KEY,
  embedding float[768]
);
CREATE VIRTUAL TABLE scene_vec USING vec0(
  asset_id INTEGER PRIMARY KEY,
  embedding float[768]
);
"""


def _migrate(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _index_migrations ("
        "version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    conn.commit()
    applied = {r[0] for r in conn.execute("SELECT version FROM _index_migrations")}
    if "001_create_tables" not in applied:
        conn.executescript(DDL)
        conn.execute("INSERT INTO _index_migrations (version) VALUES ('001_create_tables')")
        conn.commit()


def _vec_blob(values) -> bytes:
    """float32 unit-norm blob, the exact on-disk format of vec0 float[] columns."""
    import numpy as np

    v = np.asarray(values, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.tobytes()


def _vec_from_blob(b: bytes):
    import numpy as np

    return np.frombuffer(bytes(b), dtype=np.float32)


def _tls_context(ca_path: str = None, pinned_pem: str = None) -> ssl.SSLContext:
    """Default verification, minus OpenSSL strict-X509 which some CAs trip.

    Chain verification stays on. `pinned_pem` (trust-on-first-use) trusts only
    the exact pinned server cert as its own CA; hostname checking is replaced
    by that pin, so a MITM presenting any other cert still fails the handshake.
    """
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.options &= ~ssl.VERIFY_X509_STRICT
    if ca_path:
        ctx.load_verify_locations(cafile=os.path.expanduser(ca_path))
    if pinned_pem:
        ctx.load_verify_locations(cadata=pinned_pem)
        ctx.check_hostname = False
    return ctx


TLS_PIN_FILE = "~/.config/vision-index/tls-fingerprints.json"


def _server_fingerprint(host: str, port: int) -> str:
    """Read the peer cert WITHOUT sending any request data, to pin it (TOFU)."""
    from urllib.parse import urlparse

    probe = ssl.create_default_context()
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=15) as sock:
        with probe.wrap_socket(sock, server_hostname=host) as tls:
            der = tls.getpeercert(binary_form=True)
    return ssl.DER_cert_to_PEM_cert(der), hashlib.sha256(der).hexdigest()


def _pin_tls(host: str):
    """Trust-on-first-use for self-signed servers: record the cert fingerprint
    on first contact, require a match afterwards. Returns the pinned PEM."""
    from urllib.parse import urlparse

    port = urlparse(embed_url).port or 443
    pem, fp = _server_fingerprint(host, port)
    pin_path = Path(os.path.expanduser(TLS_PIN_FILE))
    pins = {}
    if pin_path.exists():
        try:
            pins = json.loads(pin_path.read_text())
        except (OSError, ValueError):
            pins = {}
    known = pins.get(f"{host}:{port}")
    if known is not None and known.get("sha256") != fp:
        _die(
            "tls.pin_mismatch",
            f"TLS certificate changed for {host}:{port} — investigate before retrying",
        )
    if known is None:
        pins[f"{host}:{port}"] = {"sha256": fp, "seen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(pin_path) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(pins, fh, indent=2)
            os.chmod(tmp, 0o600)
        os.replace(tmp, pin_path)
        print(
            f"warning: self-signed TLS cert fingerprint recorded for {host}:{port} "
            f"(sha256 {fp}); future visits must match",
            file=sys.stderr,
        )
    return pem


def _embed(image_path: str, base_url: str, token: str) -> dict:
    """POST /v1/embed with the raw file bytes; verify the server's sha256."""
    try:
        raw = Path(image_path).read_bytes()
    except OSError as exc:
        _die("file.unreadable", str(exc.__class__.__name__))
    sha = hashlib.sha256(raw).hexdigest()

    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="image"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + raw
        + f"\r\n--{boundary}--\r\n".encode()
    )

    url = base_url.rstrip("/") + "/v1/embed"
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "multipart/form-data; boundary=" + boundary,
    }
    from urllib.parse import urlparse

    host = urlparse(base_url).hostname
    port = urlparse(base_url).port or 443
    ca = os.environ.get("VISION_EMBED_CA", "")
    ctx = _tls_context(ca_path=(ca or None))

    resp, status, pinned, retried_busy = None, None, False, False
    for attempt in range(3):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with _Spinner(f"POST /v1/embed (attempt {attempt + 1})"):
                resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            status = resp.status
            break
        except ssl.SSLError as exc:
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                # Self-signed server: pin its cert once (fingerprint file), then
                # trust that exact cert as its own CA here and on every later
                # connection — chain verification on a chain of length one.
                if pinned:
                    _die("tls.verify_failed", str(exc))
                ctx = _tls_context(pinned_pem=_pin_tls(host))
                pinned = True
                continue
            _die("embed.unreachable", str(exc))
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                err_body = json.loads(exc.read().decode())
                reason = err_body.get("error", {}).get("reason", "")
            except Exception:
                reason = ""
            if status == 429 and reason == "gpu.busy" and not retried_busy:
                retried_busy = True
                time.sleep(GPU_BUSY_RETRY_SLEEP)
                continue
            if status == 401:
                _die("auth.rejected", "server returned 401 auth.invalid_token — check VISION_EMBED_TOKEN")
            if status == 404:
                _die("route.not_found", "server returned 404 — is VISION_EMBED_URL right?")
            _die("embed.http_error", f"HTTP {status} {reason}")
        except urllib.error.URLError as exc:
            err = getattr(exc, "reason", None)
            if isinstance(err, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(err):
                # Same self-signed path: URLError wraps the handshake failure.
                if pinned:
                    _die("tls.verify_failed", str(err))
                ctx = _tls_context(pinned_pem=_pin_tls(host))
                pinned = True
                continue
            _die("embed.unreachable", str(err or exc.__class__.__name__))
        except OSError as exc:
            _die("embed.unreachable", str(exc.__class__.__name__))

    try:
        data = json.loads(resp.read().decode())
    except Exception as exc:
        _die("embed.bad_response", str(exc.__class__.__name__))

    if data.get("image", {}).get("sha256") != sha:
        _die(
            "sha.mismatch",
            "server sha256 does not match the bytes sent — refusing to index",
        )
    return data


def _nearest_exemplar(conn, vector):
    """Mirror of the index join rule: nearest non-merged face-cluster exemplar
    by cosine distance (1 - dot over stored unit vectors)."""
    import numpy as np

    rows = conn.execute(
        "SELECT c.id, fv.embedding FROM clusters c"
        " JOIN faces f ON f.id = c.exemplar_face_id"
        " JOIN face_vec fv ON fv.face_id = f.id"
        " WHERE c.kind = 'face' AND c.merged_into IS NULL"
    ).fetchall()
    if not rows:
        return None, None
    cands = np.vstack([_vec_from_blob(r[1]) for r in rows])
    q = np.asarray(vector, dtype=np.float32).ravel()
    dists = 1.0 - (cands @ q)
    idx = int(np.argmin(dists))
    return int(rows[idx][0]), float(dists[idx])


# -- CLI ----------------------------------------------------------------------

ap = argparse.ArgumentParser(
    prog="vision-index", description="Local face identity index over vision-embed."
)
sub = ap.add_subparsers(dest="cmd", required=True)

p = sub.add_parser("identify", help="who is in this image; writes nothing")
p.add_argument("image")
p = sub.add_parser("enroll", help="add image to index")
p.add_argument("image")
p.add_argument("--name", default=None)
p = sub.add_parser("name", help="assign/rename a cluster")
p.add_argument("cluster_id", type=int)
p.add_argument("name")
p = sub.add_parser("clusters", help="list clusters")
p.add_argument("--unnamed", action="store_true")
sub.add_parser("people", help="list named people only")

args = ap.parse_args()

embed_url = os.environ.get("VISION_EMBED_URL", DEFAULT_EMBED_URL)
token = os.environ.get("VISION_EMBED_TOKEN", "")
index_db = os.environ.get("VISION_INDEX_DB", DEFAULT_INDEX_DB)
face_match_max = float(os.environ.get("VISION_FACE_MATCH_MAX", str(DEFAULT_FACE_MATCH_MAX)))

if token == "":
    print(
        "error: set VISION_EMBED_TOKEN (bearer token for the vision-embed service)",
        file=sys.stderr,
    )
    sys.exit(2)

is_tty = sys.stdout.isatty()


def _out(obj) -> None:
    print(json.dumps(obj, indent=(2 if is_tty else None), default=float))


def _resolve(conn, cluster_id: int):
    """Follow merged_into pointers to the surviving cluster."""
    seen = set()
    current = int(cluster_id)
    while True:
        if current in seen:
            return None
        seen.add(current)
        row = conn.execute(
            "SELECT merged_into FROM clusters WHERE id = ?", (current,)
        ).fetchone()
        if row is None:
            return None
        if row[0] is None:
            return current
        current = int(row[0])


def _cluster_name(conn, cluster_id: int):
    row = conn.execute(
        "SELECT name FROM clusters WHERE id = ?", (cluster_id,)
    ).fetchone()
    return row[0] if row is not None else None


def _face_matches(conn, response) -> list:
    """identify path — read-only: nearest exemplar per detected face, no writes."""
    out = []
    for face in response.get("faces", []):
        cid, dist = _nearest_exemplar(conn, face["vector"])
        name = _cluster_name(conn, cid) if cid is not None else None
        matched = cid is not None and dist is not None and dist <= face_match_max
        out.append(
            {
                "cluster_id": cid if matched else None,
                "name": name if matched else None,
                "distance": dist,
                "matched": matched,
            }
        )
    return out


if args.cmd == "identify":
    conn = _open_db(index_db, read_only=True)
    try:
        response = _embed(args.image, embed_url, token)
        if conn is None:
            faces = [
                {"cluster_id": None, "name": None, "distance": None, "matched": False}
                for _ in response.get("faces", [])
            ]
        else:
            import sqlite3

            try:
                faces = _face_matches(conn, response)
            except sqlite3.Error as exc:
                # An index file that predates the schema simply has no people yet.
                _die("db.error", f"index unreadable: {exc.__class__.__name__}")
    finally:
        if conn is not None:
            conn.close()
    _out({"faces": faces, "count": len(faces)})

elif args.cmd == "enroll":
    import datetime as dt

    conn = _open_db(index_db, read_only=False)
    try:
        _migrate(conn)
        response = _embed(args.image, embed_url, token)
        sha = response["image"]["sha256"]
        if response["scene"]["dim"] != 768:
            _die("embed.bad_response", "scene dim is not 768")
        existing = conn.execute(
            "SELECT id FROM assets WHERE sha256 = ?", (sha,)
        ).fetchone()
        if existing is not None:
            asset_id = int(existing[0])
            cids = [
                int(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT cluster_id FROM faces WHERE asset_id = ? AND cluster_id IS NOT NULL",
                    (asset_id,),
                )
            ]
            _out(
                {
                    "asset_id": asset_id,
                    "created": False,
                    "face_cluster_ids": cids,
                }
            )
        else:
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                conn.execute("BEGIN")
                cur = conn.execute(
                    "INSERT INTO assets (path, sha256, width, height, bytes, indexed_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        os.path.abspath(args.image),
                        sha,
                        int(response["image"]["width"]),
                        int(response["image"]["height"]),
                        len(Path(args.image).read_bytes()),
                        now,
                    ),
                )
                asset_id = int(cur.lastrowid)
                conn.execute(
                    "INSERT INTO scene_vec(asset_id, embedding) VALUES (?, ?)",
                    (asset_id, _vec_blob(response["scene"]["vector"])),
                )
                cluster_ids = []
                first_new_cluster = None
                for face in response.get("faces", []):
                    f = conn.execute(
                        "INSERT INTO faces (asset_id, x, y, w, h, det_score, cluster_id)"
                        " VALUES (?, ?, ?, ?, ?, ?, NULL)",
                        (
                            asset_id,
                            face["bbox"]["x"], face["bbox"]["y"],
                            face["bbox"]["w"], face["bbox"]["h"],
                            face["det_score"],
                        ),
                    )
                    face_id = int(f.lastrowid)
                    conn.execute(
                        "INSERT INTO face_vec(face_id, embedding) VALUES (?, ?)",
                        (face_id, _vec_blob(face["vector"])),
                    )
                    vec = _vec_from_blob(
                        conn.execute(
                            "SELECT embedding FROM face_vec WHERE face_id = ?", (face_id,)
                        ).fetchone()[0]
                    )
                    best_id, best_dist = _nearest_exemplar(conn, vec)
                    if best_id is not None and best_dist is not None and best_dist <= face_match_max:
                        conn.execute(
                            "UPDATE faces SET cluster_id = ? WHERE id = ?", (best_id, face_id)
                        )
                        cluster_ids.append(best_id)
                    else:
                        ccur = conn.execute(
                            "INSERT INTO clusters (kind, name, named_at, exemplar_face_id, merged_into)"
                            " VALUES ('face', NULL, NULL, ?, NULL)",
                            (face_id,),
                        )
                        cid = int(ccur.lastrowid)
                        conn.execute(
                            "UPDATE faces SET cluster_id = ? WHERE id = ?", (cid, face_id)
                        )
                        cluster_ids.append(cid)
                        if first_new_cluster is None:
                            first_new_cluster = cid
                for obj in response.get("objects", []):
                    o = conn.execute(
                        "INSERT INTO objects (asset_id, label, score, x, y, w, h, cluster_id)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                        (
                            asset_id,
                            obj["label"], obj["score"],
                            obj["bbox"]["x"], obj["bbox"]["y"],
                            obj["bbox"]["w"], obj["bbox"]["h"],
                        ),
                    )
                    if obj.get("vector") is not None:
                        conn.execute(
                            "INSERT INTO object_vec(object_id, embedding) VALUES (?, ?)",
                            (int(o.lastrowid), _vec_blob(obj["vector"])),
                        )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                _die("db.error", f"enroll transaction rolled back: {exc.__class__.__name__}")
            if args.name and first_new_cluster is not None:
                try:
                    conn.execute(
                        "UPDATE clusters SET name = ?, named_at = COALESCE(named_at, ?)"
                        " WHERE id = ?",
                        (args.name, now, first_new_cluster),
                    )
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    _die("name.conflict", str(exc.__class__.__name__))
            _out(
                {
                    "asset_id": asset_id,
                    "created": True,
                    "face_cluster_ids": cluster_ids,
                    "name": args.name if first_new_cluster is not None else None,
                }
            )
    finally:
        conn.close()

elif args.cmd == "name":
    import datetime as dt

    conn = _open_db(index_db, read_only=False)
    try:
        _migrate(conn)
        if not (NAME_MIN_LEN <= len(args.name) <= NAME_MAX_LEN):
            _die("name.too_long", f"name must be {NAME_MIN_LEN}-{NAME_MAX_LEN} chars")
        row = conn.execute(
            "SELECT merged_into, name FROM clusters WHERE id = ? AND kind = 'face'",
            (args.cluster_id,),
        ).fetchone()
        if row is None or row[0] is not None:
            _die("not_found", "cluster_id resolves to nothing")
        clash = conn.execute(
            "SELECT id FROM clusters WHERE name = ? AND id != ? AND merged_into IS NULL",
            (args.name, args.cluster_id),
        ).fetchone()
        if clash is not None:
            _die("name.conflict", "name already used by another cluster")
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            "UPDATE clusters SET name = ?, named_at = COALESCE(named_at, ?) WHERE id = ?",
            (args.name, now, args.cluster_id),
        )
        conn.commit()
        _out({"cluster_id": args.cluster_id, "name": args.name, "previous_name": row[1]})
    finally:
        conn.close()

elif args.cmd in ("clusters", "people"):
    conn = _open_db(index_db, read_only=True)
    try:
        if conn is None:
            _out({"clusters": [] if args.cmd == "clusters" else []})
        else:
            if args.cmd == "clusters":
                sql = (
                    "SELECT id, name FROM clusters WHERE kind = 'face'"
                    " AND merged_into IS NULL"
                )
                if args.unnamed:
                    sql += " AND name IS NULL"
                else:
                    sql += " AND name IS NOT NULL"
            else:
                sql = (
                    "SELECT id, name FROM clusters WHERE kind = 'face'"
                    " AND merged_into IS NULL AND name IS NOT NULL"
                )
            rows = conn.execute(sql + " ORDER BY id").fetchall()
            out = []
            for cid, name in rows:
                out.append(
                    {
                        "cluster_id": int(cid),
                        "name": name,
                        "face_count": int(
                            conn.execute(
                                "SELECT COUNT(*) FROM faces WHERE cluster_id = ?", (cid,)
                            ).fetchone()[0]
                        ),
                    }
                )
            _out({"clusters": out})
    finally:
        if conn is not None:
            conn.close()
