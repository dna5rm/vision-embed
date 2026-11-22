"""DEDUP-AND-INCIDENTIAL v4 tests (design docs/internal/design/
DEDUP-AND-INCIDENTAL-v4.md, carried v3 §2/§3/§6 and v4 §M.7).

NEW file — the independent black-box suite tests/test_vision_index_cli.py is
NOT modified. Covers:

- the critical post-migration regression: test_migrate_v2_then_subsequent_read_
  succeeds (v4 §M.7) — a v2-schema DB migrates on first write to user_version 3
  with the inert column, and a SUBSEQUENT status / list people read exits 0;
- dedup unit tests incl. the real 47x54 triple geometry (v3 §1.1);
- the two boundary pairs pinning FACE_DEDUP_CENTER_RATIO = 1.25 (v3 §6a) and
  FACE_DEDUP_TINY_MAX = 80 (v3 §6b) from both sides;
- the incidental strict-inequality boundary (v2 §5.2 V-k, area ratio exactly
  0.20 NOT incidental, 0.19 incidental);
- the real-people protection: dedup never merges two large faces (v3 §1.2);
- an end-to-end incidental/dead-cluster run through the fake embed service
  (inert cluster created, excluded from the status queue, skipped by
  link --near with reason inert_cluster).

Every CLI invocation redirects HOME / XDG_* / VISION_INDEX_DB (the imported
bare-glob fixtures do this) and sha256-guards the operator's real DB.
"""

import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import struct
import subprocess
import sys

import pytest

from test_vision_index_bare_glob import (  # noqa: F401  (fixtures imported)
    E1, ENV, UNKNOWN, clean_env, env,  # noqa: F401
    photo_file, run_cli, service)  # noqa: F401

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vision-index.py")

_spec = importlib.util.spec_from_file_location("vision_index_impl_v4", SCRIPT)
vi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vi)

DIM = 512
V0 = [0.001] * DIM  # arbitrary constant vector; dedup/incidental ignore vectors
V1 = [0.002] * DIM


def face(det, x, y, w, h, vec=None):
    return {"bbox": {"x": x, "y": y, "w": w, "h": h}, "det_score": det,
            "dim": DIM, "vector": vec if vec is not None else list(V0)}


# --------------------------------------------------------------------------
# DEC-7 dedup unit tests
# --------------------------------------------------------------------------

def test_dedup_real_triple_geometry_merges_to_one():
    """The measured 20230902_162623 triple (47x54 / 47x54 / 47x55 tiny boxes,
    pairwise centre ratios 0.52-1.02, all under 1.25) merges to ONE survivor;
    the 669x990 subject (det 0.7677) survives separately: 4 in, 2 out,
    2 dedup-suppressed (v3 §1.1, §2.5)."""
    faces = [
        face(0.7677, 100, 100, 669, 990),   # the real person (large face)
        face(0.7350, 1180, 568, 47, 54),    # triple peak det
        face(0.7340, 1215, 575, 47, 54),
        face(0.7330, 1240, 590, 47, 55),
    ]
    survivors, suppressed = vi.dedup_faces(faces)
    assert suppressed == 2
    assert len(survivors) == 2
    # survivors keep their OWN boxes and vectors, original order
    assert survivors[0]["bbox"] == {"x": 100, "y": 100, "w": 669, "h": 990}
    assert survivors[1]["bbox"] == {"x": 1180, "y": 568, "w": 47, "h": 54}
    # the survivor is the highest-det face of the merged group
    assert survivors[1]["det_score"] == 0.7350


def test_dedup_center_ratio_boundary_12_merges_13_does_not():
    """Pins FACE_DEDUP_CENTER_RATIO = 1.25 from both sides (v3 §6a): 10x10
    boxes; centre distance 12 <= 1.25 * 10 = 12.5 merges; 13 does not."""
    merged, sup = vi.dedup_faces(
        [face(0.9, 0, 0, 10, 10), face(0.9, 12, 0, 10, 10)])
    assert len(merged) == 1 and sup == 1
    kept, sup = vi.dedup_faces(
        [face(0.9, 0, 0, 10, 10), face(0.9, 13, 0, 10, 10)])
    assert len(kept) == 2 and sup == 0


def test_dedup_tiny_max_boundary_79_merges_81_does_not():
    """Pins FACE_DEDUP_TINY_MAX = 80 from both sides (v3 §6b): two 79x79 boxes
    20 px apart merge (in scope); two 81x81 boxes at the same 20 px offset do
    NOT (max side 81 >= 80: the centre arm would hold, only the scope floor
    separates the arms)."""
    merged, sup = vi.dedup_faces(
        [face(0.9, 0, 0, 79, 79), face(0.9, 20, 0, 79, 79)])
    assert len(merged) == 1 and sup == 1
    kept, sup = vi.dedup_faces(
        [face(0.9, 0, 0, 81, 81), face(0.9, 20, 0, 81, 81)])
    assert len(kept) == 2 and sup == 0


def test_dedup_never_merges_two_large_faces():
    """Real-people protection (v3 §1.2): 207x265 vs 209x267 at a ~0.37 centre
    ratio — the scope floor excludes the pair entirely, so the centre arm is
    never even evaluated (here the centre WOULD pass: ~111 <= 1.25 * 267)."""
    a = face(0.9, 0, 0, 207, 265)
    b = face(0.9, 110, 0, 209, 267)
    ca = (a["bbox"]["x"] + 207 / 2, a["bbox"]["y"] + 265 / 2)
    cb = (b["bbox"]["x"] + 209 / 2, b["bbox"]["y"] + 267 / 2)
    assert math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= 1.25 * 267
    kept, sup = vi.dedup_faces([a, b])
    assert len(kept) == 2 and sup == 0


# --------------------------------------------------------------------------
# DEC-8 incidental strict-inequality boundary (V-k, v2 §5.2 / v3 §3a)
# --------------------------------------------------------------------------

def test_incidental_ratio_boundary_exactly_020_not_below_is():
    """Strict inequality: A_max = 100x100 = 10000 px². A 50x40 face is exactly
    0.20 -> NOT incidental; a 50x38 face is 1900 px² = 0.19 -> incidental.
    Both faces are at/above FACE_DET_MIN; the dedup never merges them
    (second face at some distance, both tiny are NOT merged since 50 px boxes
    60 px apart: centre 60 > 1.25 * 50)."""
    largest = face(0.9, 0, 0, 100, 100)
    at_ratio = face(0.9, 400, 0, 50, 40)    # 2000 px², ratio exactly 0.20
    below = face(0.8, 400, 200, 50, 38)     # 1900 px², ratio 0.19
    survivors, suppressed = vi.dedup_faces([largest, at_ratio, below])
    assert (len(survivors), suppressed) == (3, 0)
    mask = vi.incidental_mask(survivors)
    assert mask == [False, False, True], mask


# --------------------------------------------------------------------------
# v4 §M.7 — the critical post-migration regression
# --------------------------------------------------------------------------

PRE_V4_DDL = """
CREATE TABLE people (
  person_id  INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE TABLE clusters (
  cluster_id INTEGER PRIMARY KEY,
  person_id  INTEGER NULL REFERENCES people(person_id) ON DELETE CASCADE,
  created_at TEXT NOT NULL
);
CREATE TABLE faces (
  vec_hash   TEXT PRIMARY KEY,
  vec        BLOB NOT NULL,
  dim        INTEGER NOT NULL CHECK(dim = 512),
  det_score  REAL NOT NULL,
  cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
  added_at   TEXT NOT NULL
);
"""


def _vec(seed):
    return [((seed * 9301 + i * 49297) % 233280) / 233280.0 for i in range(512)]


def build_v2_db(path):
    """A real sqlite file at the PRE-v4 (v2) schema: no inert column,
    user_version = 2, one linked cluster + one face, one below-floor
    unlinked cluster + one face."""
    with sqlite3.connect(path) as conn:
        conn.executescript(PRE_V4_DDL)
        conn.execute("PRAGMA user_version = 2")
        t = "2026-11-21T00:00:00.000Z"
        conn.execute("INSERT INTO people (name, created_at) VALUES (?, ?)",
                     ("alice", t))
        conn.execute(
            "INSERT INTO clusters (person_id, created_at) VALUES (?, ?)",
            (1, t))
        v = _vec(1)
        conn.execute(
            "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
            "added_at) VALUES (?, ?, 512, ?, 1, ?)",
            (hashlib.sha256(struct.pack("<%df" % len(v), *v)).hexdigest(),
             struct.pack("<%df" % len(v), *v), 0.9, t))
        conn.execute("INSERT INTO clusters (person_id, created_at) "
                     "VALUES (NULL, ?)", (t,))
        v2 = _vec(2)
        conn.execute(
            "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
            "added_at) VALUES (?, ?, 512, ?, 2, ?)",
            (hashlib.sha256(struct.pack("<%df" % len(v2), *v2)).hexdigest(),
             struct.pack("<%df" % len(v2), *v2), 0.50, t))
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    return conn.execute("SELECT 'ok'").fetchone()


def test_migrate_v2_then_subsequent_read_succeeds(env, tmp_path):
    """v4 §M.7: one write verb migrates the v2 DB (ALTER + backfill +
    user_version = 3, one transaction); a SUBSEQUENT status and list people
    on the SAME file must exit 0 (NOT db.version_unsupported) with the data
    intact; user_version is 3 and the inert column is present with
    NOT NULL DEFAULT 0. Neither read alters the file beyond the migration."""
    db_path = ENV["VISION_INDEX_DB"]
    build_v2_db(db_path)
    pre_links = os.listdir(os.path.dirname(db_path))

    def sha():
        with open(db_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    # --- one write verb opens the DB read-write -> migration runs
    p = run_cli(env, ["link", "2", "second-person"])
    assert p.returncode == 0, (p.stdout, p.stderr)

    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    # 1. user_version is 3 after the write
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    # 2. table_info(clusters) contains inert, NOT NULL, DEFAULT 0
    cols = conn.execute("PRAGMA table_info(clusters)").fetchall()
    by_name = {r[1]: r for r in cols}
    assert "inert" in by_name
    assert by_name["inert"][3] == 1 and by_name["inert"][4] == "0"
    # the backfill: the all-below-floor cluster became inert, the other not
    inert_by_id = dict(conn.execute(
        "SELECT cluster_id, inert FROM clusters").fetchall())
    assert inert_by_id == {1: 0, 2: 1}
    conn.close()

    # 3. a SUBSEQUENT status read exits 0 (the exact regression of v4 §M.7)
    sha_after_write = sha()
    p = run_cli(env, ["status"])
    assert p.returncode == 0, p.stderr
    assert "db.version_unsupported" not in p.stderr
    # 4. list people exits 0 and the migrated row data is intact
    p = run_cli(env, ["list", "people"])
    assert p.returncode == 0, p.stderr
    assert "alice" in p.stdout and "second-person" in p.stdout
    p = run_cli(env, ["list", "clusters"])
    assert p.returncode == 0, p.stderr
    assert "2 clusters: 2 linked, 0 unlinked, 1 inert" in p.stdout
    # 5. read-only invariant: neither read altered the file further
    assert sha() == sha_after_write
    assert sorted(os.listdir(os.path.dirname(db_path))) == sorted(pre_links)


# --------------------------------------------------------------------------
# DEC-8 end-to-end through the fake embed service
# --------------------------------------------------------------------------

def test_incidental_face_is_stored_in_own_inert_cluster(env, service, tmp_path):
    """A group enroll: one 100x100 face + one 10x10 above-floor face at ratio
    0.01 -> the small face is stored with face_status 'incidental' in its OWN
    inert=1 cluster; status gets the trailing inert line, list clusters the
    ', 1 inert' count, and link --near never auto-links it (reason
    inert_cluster, its distance still reported). A manual link emits the one
    stderr advisory and still succeeds."""
    group = photo_file(env, service, str(tmp_path), "group.png", [
        (E1, {"x": 0, "y": 0, "w": 100, "h": 100}),
        (UNKNOWN, {"x": 200, "y": 0, "w": 10, "h": 10}),
    ])
    p = run_cli(env, ["enroll", group])
    assert p.returncode == 0, p.stderr
    entry = json.loads(p.stdout)["files"][0]
    det_entry = {e["face_status"] for e in entry["face_status_summary"]}
    assert det_entry == {"created", "incidental"}, entry["face_status_summary"]
    # counting identity reconciles: 2 faces stored, 2 clusters created
    assert entry["faces_stored"] == 2 and entry["clusters_created"] == 2
    assert entry["clusters_joined"] == 0
    assert entry["face_status_summary"][1]["face_status"] == "incidental"

    conn = sqlite3.connect("file:%s?mode=ro" % ENV["VISION_INDEX_DB"], uri=True)
    inert = dict(conn.execute(
        "SELECT cluster_id, inert FROM clusters").fetchall())
    conn.close()
    assert sorted(inert.values()) == [0, 1]
    inert_cid = [cid for cid, i in inert.items() if i][0]

    p = run_cli(env, ["list", "clusters"])
    assert p.returncode == 0, p.stderr
    # the reconciliation identity L + M = total is unchanged (I = inert is a
    # NOT-disjoint third property, design v2 §4.3)
    assert "2 clusters: 0 linked, 2 unlinked, 1 inert" in p.stdout
    p = run_cli(env, ["status"])
    assert p.returncode == 0, p.stderr
    assert ("1 inert cluster(s) excluded from this queue (below det floor or "
            "incidental); hand-link with: link <id> <name>" in p.stdout)

    # link --near at a distance ABOVE the inert cluster's real distance (1.0):
    # it is never auto-linked, the skip reason is inert_cluster with the
    # measured distance still reported; the big cluster is out of range.
    p = run_cli(env, ["link", "1", "alice"])
    assert p.returncode == 0, p.stderr
    p = run_cli(env, ["link", "--near", "alice", "0.5"])
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["linked_clusters"] == []
    skipped = {s["cluster_id"]: s for s in out["skipped"]}
    assert skipped[inert_cid]["reason"] == "inert_cluster"
    assert skipped[inert_cid]["nearest_person_distance"] == pytest.approx(1.0)

    # manual link of the inert cluster is permitted, with EXACTLY ONE stderr
    # advisory and an unchanged stdout shape
    p = run_cli(env, ["link", str(inert_cid), "other"])
    assert p.returncode == 0, p.stderr
    assert p.stderr.count("\n") == 1, p.stderr
    assert p.stderr.startswith(
        "warning: cluster %d is inert (below det floor or incidental); "
        "its faces will join this person's evidence" % inert_cid)
    assert json.loads(p.stdout)["linked"] is True
    # and it no longer counts as unlinked work
    p = run_cli(env, ["list", "clusters"])
    assert "2 clusters: 2 linked, 0 unlinked, 1 inert" in p.stdout
