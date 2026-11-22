"""Local tests for the no-arg help and bare-file/glob identify changes
(spec 8.9 amendment + new 8.10). Written WITHOUT touching the independent
black-box suite tests/test_vision_index_cli.py.

Every invocation redirects HOME / XDG_DATA_HOME / XDG_CONFIG_HOME /
VISION_INDEX_DB to a temp dir. A tiny local fake embed service replaces the
real vision-embed service, and the operator's real DB is sha256-guarded
before and after each test.

Vector geometry (all unit vectors, cosine distance 1 - dot):
  E1 = e0   -> evidence for person "alice"
  E2 = e1   -> evidence for person "bob"
  PROB  = (e0+e1)/sqrt(2) -> d = 0.293 to BOTH persons -> probable band
  UNKNOWN = e2 -> d = 1.0 to both -> unknown band
"""

import hashlib
import http.server
import json
import math
import os
import subprocess
import sys
import threading
import types

import pytest

from pathlib import Path

import pwd

_REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
SCRIPT = str(Path(__file__).resolve().parent.parent / "vision-index.py")
OPERATOR_REAL_DB = os.path.join(_REAL_HOME, ".local", "share", "vision-index",
                                "index.sqlite3")
DIM = 512
MARKER = b"Content-Type: application/octet-stream\r\n\r\n"


def unit(i):
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


E1 = unit(0)                     # "alice" evidence
E2 = unit(1)                     # "bob" evidence
PROB = norm([1.0, 1.0] + [0.0] * (DIM - 2))      # d = 0.293 both: probable
UNKNOWN = unit(2)                # d = 1.0 both: unknown


class Service(types.SimpleNamespace):
    requests = None     # list of 12-hex image keys, in arrival order
    faces_by_key = None

    def faces_for(self, key):
        return self.faces_by_key.get(key, [])


class Handler(http.server.BaseHTTPRequestHandler):
    service = None  # class attribute, set by the fixture

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        image = body.split(MARKER, 1)[1] if MARKER in body else b""
        end = image.find(b"\r\n--visionindex7d1a2cboundary")
        if end != -1:
            image = image[:end]
        key = hashlib.sha256(image).hexdigest()[:12]
        svc = type(self).service
        svc.requests.append(key)
        faces = [{
            # DEC-7 fixture fix (design v3 §7.2): per-face (vector, bbox) pairs;
            # a bare vector keeps the old default 10x10 box at 0,0 so
            # single-face fixtures stay byte-identical in behavior.
            "bbox": (bbox if bbox is not None else
                     {"x": 0, "y": 0, "w": 10, "h": 10}),
            "det_score": 0.9,
            "dim": DIM,
            "vector": vec,
        } for (vec, bbox) in svc.faces_for(key)]
        resp = json.dumps({
            "request_id": "fake-1",
            "image": {"sha256": key, "width": 64, "height": 64},
            "faces": faces,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


@pytest.fixture()
def service():
    svc = Service(requests=[], faces_by_key={})
    Handler.service = svc
    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    svc.url = "http://127.0.0.1:%d" % srv.server_address[1]
    SERVICE_OVERRIDES["VISION_EMBED_URL"] = svc.url
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield svc
    srv.shutdown()
    srv.server_close()
    SERVICE_OVERRIDES.pop("VISION_EMBED_URL", None)


@pytest.fixture(autouse=True)
def clean_env(tmp_path):
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    (home / "conf").mkdir(parents=True)
    (home / "data" / "vision-index").mkdir(parents=True, exist_ok=True)
    ENV.clear()
    ENV.update({
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_CONFIG_HOME": str(home / "conf"),
        "VISION_INDEX_DB": str(home / "data" / "vision-index" / "index.sqlite3"),
    })
    for k, v in ENV.items():
        os.environ[k] = v
    saved = {k: os.environ.get(k)
             for k in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME")}
    db_before = None
    if os.path.exists(OPERATOR_REAL_DB):
        db_before = operator_db_sha()
    yield ENV
    for k in saved:
        if saved[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]
    db_after = os.path.exists(OPERATOR_REAL_DB)
    if not db_after:
        if db_before is not None:
            pytest.fail("operator real DB vanished during the run")
        # Operator DB absent (deleted for re-enrollment): no snapshot to guard.
        return
    assert operator_db_sha() == db_before, "operator real DB was modified"


ENV = {}  # filled in by clean_env


@pytest.fixture()
def env():
    yield ENV


def operator_db_sha():
    with open(OPERATOR_REAL_DB, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


SERVICE_OVERRIDES = {}  # filled in by the service fixture


def run_cli(env, args):
    full = dict(env)
    full.update(SERVICE_OVERRIDES)
    full.setdefault("VISION_EMBED_TOKEN", "test-token")
    return subprocess.run([sys.executable, SCRIPT] + args,
                          env=full, capture_output=True, text=True,
                          timeout=120)


def photo_file(env, service, dirpath, name, vecs):
    """Write a fake image file and register its bytes with the fake service.
    `vecs` items are a vector, or a (vector, bbox) pair (DEC-7 fixture fix,
    design v3 §7.2); a bare vector gets the default 10x10 box at 0,0."""
    data = json.dumps({"salt": name, "vecs": vecs}).encode()
    path = os.path.join(dirpath, name)
    with open(path, "wb") as f:
        f.write(data)
    service.faces_by_key[hashlib.sha256(data).hexdigest()[:12]] = [
        (f, None) if not (isinstance(f, tuple) and len(f) == 2) else f
        for f in vecs]
    return path


def enroll_persons(env, service, dirpath):
    """Enroll one photo per person via the CLI, then link the two clusters
    as people 'alice' and 'bob'."""
    p1 = photo_file(env, service, dirpath, "ev_alice.png", [E1])
    p2 = photo_file(env, service, dirpath, "ev_bob.png", [E2])
    p = run_cli(env, ["enroll", p1, p2])
    assert p.returncode == 0, p.stderr
    p = run_cli(env, ["list", "clusters", "--json"])
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    rows = out["clusters"] if isinstance(out, dict) and "clusters" in out \
        else out
    ids = sorted(int(c["cluster_id"]) for c in rows)
    assert len(ids) >= 2, rows
    for cid, name in ((ids[0], "alice"), (ids[1], "bob")):
        p = run_cli(env, ["link", str(cid), name])
        assert p.returncode == 0, p.stderr


# --------------------------- help changes (8.9) ---------------------------

def test_no_args_prints_help_exit0(env, tmp_path):
    p = run_cli(env, [])
    assert p.returncode == 0
    assert "usage: vision-index" in p.stderr
    assert "enroll" in p.stderr and "identify" in p.stderr and "status" \
        in p.stderr
    assert p.stdout == "", p.stdout


def test_help_flags_exit0_stderr_no_stdout(env, tmp_path):
    for flag in ("--help", "-h"):
        p = run_cli(env, [flag])
        assert p.returncode == 0, (flag, p.stderr)
        assert "usage: vision-index" in p.stderr
        assert p.stdout == "", (flag, p.stdout)
    assert 80 >= max(len(l) for l in run_cli(env, ["--help"]).stderr.splitlines())


def test_usage_verb_still_exit2(env, tmp_path):
    p = run_cli(env, ["usage"])
    assert p.returncode == 2
    assert "usage: vision-index" in p.stderr


# --------------------- bare-file / glob identify (8.10) --------------------

def test_bare_single_file_matched(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    f = photo_file(env, service, str(tmp_path), "m1.png", [E1])
    p = run_cli(env, [f])
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert lines == ["%s: alice" % f], p.stdout


def test_bare_single_file_probable_band(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    f = photo_file(env, service, str(tmp_path), "pr1.png", [PROB])
    p = run_cli(env, [f])
    assert p.returncode == 0, p.stderr
    # DEC-1 core rule: the word 'probable' is mandatory, never a bare name.
    line = p.stdout.strip()
    assert line.split(": ", 1)[1].startswith("probable "), p.stdout


def test_bare_file_unknown_and_no_faces_exit1(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    fu = photo_file(env, service, str(tmp_path), "u1.png", [UNKNOWN])
    fn = photo_file(env, service, str(tmp_path), "n1.png", [])
    p = run_cli(env, [fu])
    assert p.returncode == 1, p.stdout
    assert p.stdout.strip().endswith(": unknown")
    p = run_cli(env, [fn])
    assert p.returncode == 1, p.stdout
    assert p.stdout.strip().endswith(": no faces")


def test_glob_one_line_per_file_and_multi_face_join(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    # DEC-7 fixture fix (§7.2): two genuinely separated faces (centre
    # distance 20 px > 1.25 * 10), so the dedup rule never merges them.
    a = photo_file(env, service, str(tmp_path), "g_a.png",
                   [(E1, {"x": 0, "y": 0, "w": 10, "h": 10}),
                    (UNKNOWN, {"x": 20, "y": 0, "w": 10, "h": 10})])
    b = photo_file(env, service, str(tmp_path), "g_b.png", [UNKNOWN])
    pattern = os.path.join(str(tmp_path), "g_*.png")
    p = run_cli(env, [pattern])
    assert p.returncode == 0, p.stdout
    lines = p.stdout.splitlines()
    assert len(lines) == 2, p.stdout
    # face order: matched face first, unknown face contributes nothing
    assert lines[0].endswith(": alice"), p.stdout
    assert lines[0] == "%s: alice" % a, p.stdout
    assert lines[1] == "%s: unknown" % b, p.stdout


def test_multi_face_join_matched_and_probable(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    f = photo_file(env, service, str(tmp_path), "mix.png",
                   [(E1, {"x": 0, "y": 0, "w": 10, "h": 10}),
                    (PROB, {"x": 20, "y": 0, "w": 10, "h": 10})])
    p = run_cli(env, [f])
    assert p.returncode == 0, p.stdout
    line = p.stdout.strip()
    assert line.endswith(": alice, probable alice") or \
        line.endswith(": alice, probable bob"), line


def test_glob_failure_line_then_continue(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    ok = photo_file(env, service, str(tmp_path), "f_ok.png", [])
    bad = os.path.join(str(tmp_path), "f_bad.png")
    with open(bad, "wb") as f:
        f.write(b"")  # zero bytes -> input.unreadable
    pattern = os.path.join(str(tmp_path), "f_*.png")
    p = run_cli(env, [pattern])
    assert p.returncode == 1, p.stdout  # nothing identified, but run continued
    got = dict(l.split(": ", 1) for l in p.stdout.splitlines())
    assert got[bad] == "input.unreadable", p.stdout
    assert got[ok] == "no faces", p.stdout


def test_typo_verb_exit2_no_network(env, service, tmp_path):
    x = os.path.join(str(tmp_path), "x.png")
    with open(x, "wb") as f:
        f.write(b"x")
    p = run_cli(env, ["enrol", x])
    assert p.returncode == 2, p.stdout
    assert "usage:" in p.stderr
    assert service.requests == [], "no network call allowed on a typo'd verb"


def test_exit0_when_any_file_identifies(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    m = photo_file(env, service, str(tmp_path), "mix_m.png", [E1])
    u = photo_file(env, service, str(tmp_path), "mix_u.png", [UNKNOWN])
    p = run_cli(env, [m, u])
    assert p.returncode == 0, p.stdout
    p = run_cli(env, [u])
    assert p.returncode == 1, p.stdout


def test_identify_verb_json_unchanged(env, service, tmp_path):
    enroll_persons(env, service, str(tmp_path))
    f = photo_file(env, service, str(tmp_path), "j1.png", [E1])
    p = run_cli(env, ["identify", f, "--json"])
    assert p.returncode == 0, p.stderr
    obj = json.loads(p.stdout)
    assert isinstance(obj, dict)
    assert obj["faces"][0]["band"] == "matched"
    assert obj["faces"][0]["nearest_person"] == "alice"
    # the bare form of the same photo is TEXT, not JSON
    p = run_cli(env, [f])
    assert p.stdout.strip().endswith(": alice")
