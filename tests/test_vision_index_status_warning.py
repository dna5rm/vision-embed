"""Thin-evidence status warning (DEC-1 documentation aid).

Companion suite to tests/test_vision_index_cli.py (which is written from the
spec alone as a black-box acceptance suite and is NOT to be modified). This
file covers the DEC-1 documentation aid added to `status`: named people with
fewer than ~5 enrolled faces get an advisory warning line in text output,
because the identify threshold is compared against the nearest face across
ALL of a person's enrolled faces, so evidence volume shifts the measured
distance (measured live: 1 enrolled face -> probable 0.3871; 4 faces ->
matched 0.1754; FACE_MATCH_MAX = 0.28).

Isolation, same rules as the acceptance suite: every invocation runs with
VISION_INDEX_DB pointed at a fresh temp DB and HOME / XDG_DATA_HOME /
XDG_CONFIG_HOME redirected; the operator's real DB is untouched. The DBs are
built directly with the section-4 DDL, so this suite needs no embed service
and no model weights. `status` is verified READ-ONLY by sha256 before/after.
"""

import hashlib
import os
import re
import sqlite3
import struct
import subprocess
import sys

import pytest

from pathlib import Path

import pwd

_REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
SCRIPT = str(Path(__file__).resolve().parent.parent / "vision-index.py")
OPERATOR_REAL_DB = os.path.join(_REAL_HOME, ".local", "share", "vision-index",
                                "index.sqlite3")

DDL = """
CREATE TABLE IF NOT EXISTS people (
  person_id  INTEGER PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id INTEGER PRIMARY KEY,
  person_id  INTEGER NULL REFERENCES people(person_id) ON DELETE CASCADE,
  created_at TEXT NOT NULL
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

WARNING_RE = re.compile(
    r"^warning: person '([^']*)' has (\d+) face\(s\) enrolled; "
    r"identification needs ~5\+ to reach 'matched' reliably$")


def _sha_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _vec(seed):
    # deterministic 512-dim vector, distinct per seed
    return [((seed * 9301 + i * 49297) % 233280) / 233280.0
            for i in range(512)]


def build_db(path, people_faces):
    """Section-4 DDL DB. people_faces: list of (name, n_faces)."""
    with sqlite3.connect(path) as conn:
        conn.executescript(DDL)
        conn.execute("PRAGMA user_version = 2")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        # (inserting rows happens below)
        for pid, (name, n) in enumerate(people_faces, start=1):
            t = "2026-11-21T00:00:00.000Z"
            conn.execute(
                "INSERT INTO people (name, created_at) VALUES (?, ?)",
                (name, t))
            conn.execute(
                "INSERT INTO clusters (person_id, created_at) VALUES (?, ?)",
                (pid, t))
            cid = conn.execute(
                "SELECT cluster_id FROM clusters WHERE person_id = ?",
                (pid,)).fetchone()[0]
            for i in range(n):
                v = _vec(pid * 100 + i)
                conn.execute(
                    "INSERT INTO faces (vec_hash, vec, dim, det_score, "
                    "cluster_id, added_at) VALUES (?, ?, 512, ?, ?, ?)",
                    (hashlib.sha256(struct.pack("<%df" % len(v), *v)).hexdigest(),
                     struct.pack("<%df" % len(v), *v), 0.9, cid, t))
        # flush the WAL into the main DB file before closing, so a later
        # read-only connection cannot move storage pages around and make a
        # byte-comparison look like a write
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()


class CliEnv:
    """Fresh HOME + fresh temp DB; env redirect identical to the suite."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        (self.home / ".config" / "vision-index").mkdir(parents=True)
        (self.home / ".config" / "vision-index" / "env").write_text("")
        (self.home / ".config" / "vision-index" / "env").chmod(0o600)
        self.xdg_data = tmp_path / "xdgdata"
        self.xdg_data.mkdir()
        self.db = tmp_path / "dbdir" / "index.sqlite3"
        self.db.parent.mkdir()
        self.env = dict(os.environ)
        for k in ("VISION_EMBED_URL", "VISION_EMBED_TOKEN", "VISION_INDEX_DB"):
            self.env.pop(k, None)
        self.env.update({
            "HOME": str(self.home),
            "XDG_DATA_HOME": str(self.xdg_data),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "VISION_INDEX_DB": str(self.db),
        })


@pytest.fixture()
def cli_env(tmp_path):
    return CliEnv(tmp_path)


@pytest.fixture(autouse=True)
def _operator_db_guard():
    """Fail loudly if any test touches the operator's real DB."""
    if not os.path.exists(OPERATOR_REAL_DB):
        yield
        return
    st_before = os.stat(OPERATOR_REAL_DB)
    yield
    st_after = os.stat(OPERATOR_REAL_DB)
    assert (st_before.st_size, st_before.st_mtime_ns) == \
           (st_after.st_size, st_after.st_mtime_ns), \
        "the operator's real index was touched: %s" % OPERATOR_REAL_DB


def run_status(env, *args):
    return subprocess.run(
        [sys.executable, SCRIPT, "status"] + [a for a in args],
        env=env.env, capture_output=True, text=True, timeout=60,
        cwd=str(env.tmp))


def test_thin_evidence_warning_printed(cli_env):
    """A person with 1 enrolled face gets the exact DEC-1 advisory line."""
    build_db(str(cli_env.db), [("alice", 1)])
    r = run_status(cli_env)
    assert r.returncode == 0, r.stderr
    warns = [ln for ln in r.stdout.splitlines() if ln.startswith("warning:")]
    assert len(warns) == 1, r.stdout
    m = WARNING_RE.fullmatch(warns[0])
    assert m, warns[0]
    assert m.group(1) == "alice" and m.group(2) == "1", warns[0]


def test_no_warning_when_every_person_has_5plus_faces(cli_env):
    build_db(str(cli_env.db), [("bob", 5), ("erin", 7)])
    r = run_status(cli_env)
    assert r.returncode == 0, r.stderr
    assert not [ln for ln in r.stdout.splitlines()
                if ln.startswith("warning:")], r.stdout


def test_no_warning_for_unnamed_or_absent_people(cli_env):
    # a person with zero faces would be a data anomaly; only NAMED people
    # with 1..4 faces trigger the line; nobody named -> nothing
    build_db(str(cli_env.db), [])
    r = run_status(cli_env)
    assert r.returncode == 0, r.stderr
    assert "warning:" not in r.stdout, r.stdout


def test_warning_is_text_mode_only_json_shape_unchanged(cli_env):
    """13.1 is closed: `status --json` gains NO field, warning never leaks
    into JSON output."""
    build_db(str(cli_env.db), [("dave", 2)])
    r = run_status(cli_env, "--json")
    assert r.returncode == 0, r.stderr
    import json as _json
    body = _json.loads(r.stdout)
    assert set(body.keys()) == {"unlinked_clusters", "suggested_command"}, body
    assert not any("warning" in k.lower() for k in body), body
    # while the text mode of the same DB shows the warning
    rt = run_status(cli_env)
    assert any(WARNING_RE.fullmatch(ln) for ln in
               rt.stdout.splitlines()
               if ln.startswith("warning:")), rt.stdout


def test_status_is_read_only_thin(cli_env):
    build_db(str(cli_env.db), [("carol", 3)])
    sha = _sha_of(str(cli_env.db))
    r = run_status(cli_env)
    assert r.returncode == 0, r.stderr
    assert _sha_of(str(cli_env.db)) == sha, "status wrote to the DB"


def test_status_is_read_only_fat(cli_env):
    build_db(str(cli_env.db), [("alice", 5), ("bob", 6)])
    sha = _sha_of(str(cli_env.db))
    r = run_status(cli_env)
    assert r.returncode == 0, r.stderr
    assert _sha_of(str(cli_env.db)) == sha, "status wrote to the DB"


def test_warning_line_placement_before_suggestion(cli_env):
    """When a suggestion exists, the text still ENDS with the suggested
    line (8.8: output ends with the suggested command when one exists)."""
    build_db(str(cli_env.db), [("alice", 2), ("bob", 5)])
    # add an UNLINKED cluster with one face so a suggestion exists
    with sqlite3.connect(str(cli_env.db)) as conn:
        t = "2026-11-21T00:00:00.000Z"
        conn.execute(
            "INSERT INTO clusters (person_id, created_at) VALUES (NULL, ?)",
            (t,))
        cid = conn.execute(
            "SELECT MAX(cluster_id) FROM clusters").fetchone()[0]
        v = _vec(999)
        conn.execute(
            "INSERT INTO faces (vec_hash, vec, dim, det_score, cluster_id, "
            "added_at) VALUES (?, ?, 512, ?, ?, ?)",
            (hashlib.sha256(struct.pack("<%df" % len(v), *v)).hexdigest(),
             struct.pack("<%df" % len(v), *v), 0.9, cid, t))
    r = run_status(cli_env)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines[-1].startswith("suggested: "), r.stdout
    warns = [i for i, ln in enumerate(lines) if ln.startswith("warning:")]
    assert warns, r.stdout
    count_idx = next(i for i, ln in enumerate(lines)
                     if "unlinked clusters," in ln)
    for i in warns:
        assert count_idx < i < len(lines) - 1, (i, lines)
