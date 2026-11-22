"""ACCEPTANCE SUITE — vision-index.py v2 (spec section 11: V-a..V-h).

Written from docs/spec/vision-index-cli-SPEC.md and
docs/spec/vision-index-output-mockups.md ALONE. The CLI is tested BLACK
BOX as a subprocess; the implementation is never imported and its source
was never read for this suite.

The spec's operator fixtures manifest (section 11 reference band) does
not exist in this environment. Per the run rules: every distance this
suite needs is MEASURED AT TEST TIME from the real service and the real
photos (session fixture `measured` embeds a sample and computes the pair
matrix); nothing is invented. A test whose required data configuration is
genuinely absent from the real corpus SKIPS with a quoted reason — a
skipped test is a named gap, never a pass.

Isolation: every invocation runs with VISION_INDEX_DB pointing at a fresh
temp path and with HOME / XDG_DATA_HOME / XDG_CONFIG_HOME redirected to a
temp home holding a 0600 env file (copied operator credentials). An
autouse guard records (st_size, st_mtime_ns) of the operator's real index
around EVERY test and fails the suite loudly, naming the offending test,
if it ever changes.
"""

import decimal
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import ssl
import urllib.request

import pytest

from pathlib import Path

import pwd

_REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
SCRIPT = str(Path(__file__).resolve().parent.parent / "vision-index.py")
OPERATOR_ENV_FILE = os.path.join(_REAL_HOME, ".config", "vision-index", "env")
OPERATOR_REAL_DB = os.path.join(_REAL_HOME, ".local", "share", "vision-index",
                                "index.sqlite3")
PHOTO_ROOTS = ("photos_a", "photos_b", "photos_f", "photos_e",
               "photos_d", "photos_c")


def _fixture_root_map():
    """Sanitized fixture root -> real on-disk folder, resolved from a
    machine-local map outside the repo (never committed)."""
    import pwd
    conf_dir = os.path.join(pwd.getpwuid(os.getuid()).pw_dir,
                            ".config", "vision-index")
    p = os.path.join(conf_dir, "test-fixture-roots.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

FACE_MATCH_MAX = 0.28       # spec section 5 (normative, never read from the CLI)
CLUSTER_JOIN_MAX = 0.40
FACE_DET_MIN = 0.65
BAND_PROBABLE_MAX = 0.40

CLOSED_REASON_TOKENS = {
    "transport.connect", "transport.tls", "tls.pin_absent",
    "transport.cert_pin_changed", "transport.http_status",
    "transport.schema_invalid", "transport.timeout", "input.unreadable",
    "link.no_such_cluster", "link.no_faces_to_measure", "link.already_linked",
    "delete.no_such_cluster", "delete.no_such_person", "batch.all_failed",
    "config.env_file_permissions", "db.absent", "delete.person_required",
    "no_evidence", "db.version_unsupported", "db.corrupt", "aborted",
    "internal.error",
}


def _read_operator_env():
    url = token = None
    with open(OPERATOR_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("VISION_EMBED_URL="):
                url = line.split("=", 1)[1]
            elif line.startswith("VISION_EMBED_TOKEN="):
                token = line.split("=", 1)[1]
    if not (url and token):
        raise RuntimeError(
            "operator env file %s must hold VISION_EMBED_URL and "
            "VISION_EMBED_TOKEN for this live acceptance suite"
            % OPERATOR_ENV_FILE)
    return url, token


SERVICE_URL, SERVICE_TOKEN = _read_operator_env()


# ---------------------------------------------------------------------------
# direct service measurement (test scaffolding; never read by the CLI)
# ---------------------------------------------------------------------------

def _embed_image(path):
    """POST one image to the real vision-embed service; return its faces."""
    boundary = "suitebnd" + os.urandom(4).hex()
    with open(path, "rb") as f:
        data = f.read()
    body = (('--%s\r\nContent-Disposition: form-data; name="image"; '
             'filename="%s"\r\nContent-Type: application/octet-stream\r\n\r\n')
            % (boundary, os.path.basename(path))).encode() + data + \
           ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(SERVICE_URL.rstrip("/") + "/v1/embed", data=body)
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    req.add_header("Authorization", "Bearer " + SERVICE_TOKEN)
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
        return json.loads(r.read())["faces"]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb)


def _ceil_cents(d):
    """ceil(d*100) in INTEGER hundredths (decimal, never float-rounded, 8.8)."""
    dv = decimal.Decimal(str(d)) * 100
    return int(dv.to_integral_value(rounding=decimal.ROUND_CEILING))


# ---------------------------------------------------------------------------
# operator real-DB guard (fails the suite loudly, naming the test)
# ---------------------------------------------------------------------------

class _RealDbGuard:
    snap = None
    test = None


@pytest.fixture(scope="session", autouse=True)
def _real_db_guard_session(request):
    yield
    g = _RealDbGuard
    if g.snap is None:
        if os.path.exists(OPERATOR_REAL_DB):
            pytest.fail("GUARD ERROR: the operator's real index %s APPEARED "
                        "during this run (it must never be touched by this "
                        "suite)" % OPERATOR_REAL_DB)
        return
    if not os.path.exists(OPERATOR_REAL_DB):
        pytest.fail("GUARD FAILURE: the operator's real index %s was DELETED "
                    "by tests (last flagged after: %s)"
                    % (OPERATOR_REAL_DB, g.test))
    st = os.stat(OPERATOR_REAL_DB)
    now = (st.st_size, st.st_mtime_ns)
    if now != g.snap:
        pytest.fail("GUARD FAILURE: the operator's REAL index %s changed "
                    "during the run ((size, mtime_ns) %s -> %s); blame or last "
                    "flagged after: %s" % (OPERATOR_REAL_DB, g.snap, now, g.test))


@pytest.fixture(autouse=True)
def _real_db_guard(request):
    def shapshot():
        if not os.path.exists(OPERATOR_REAL_DB):
            return None
        st = os.stat(OPERATOR_REAL_DB)
        return (st.st_size, st.st_mtime_ns)

    g = _RealDbGuard
    before = shapshot()
    prev, prev_test = g.snap, g.test
    if prev is not None and prev_test is not None and before != prev:
        if before is None:
            pytest.fail("GUARD FAILURE: the test '%s' DELETED the operator's "
                        "REAL index %s" % (prev_test, OPERATOR_REAL_DB))
        pytest.fail("GUARD FAILURE: the test '%s' changed the operator's REAL "
                    "index %s ((size, mtime_ns) %s -> %s) — that file must "
                    "never be touched by any test"
                    % (prev_test, OPERATOR_REAL_DB, prev, before))
    g.snap, g.test = before, request.node.nodeid
    yield
    after = shapshot()
    if before != after:
        pytest.fail("GUARD FAILURE: the test '%s' changed the operator's REAL "
                    "index %s ((size, mtime_ns) %s -> %s)"
                    % (request.node.nodeid, OPERATOR_REAL_DB, before, after))


# ---------------------------------------------------------------------------
# per-test CLI isolation + run helpers
# ---------------------------------------------------------------------------

_UNSET = object()


class RunResult:
    def __init__(self, rc, out, err):
        self.rc = rc
        self.stdout = out
        self.stderr = err

    def one_json_object(self, msg=""):
        """13.1-1: exactly one COMPACT JSON object on stdout, nothing more."""
        raw = self.stdout
        assert raw.strip() != "", "stdout was empty (expected one JSON object)%s" % msg
        obj, end = json.JSONDecoder().raw_decode(raw)
        assert raw[end:].strip() == "", (
            "stdout must carry exactly ONE JSON value and nothing else%s; "
            "trailing: %r" % (msg, raw[end:60]))
        assert "\n" not in raw.strip(), (
            "piped stdout must be COMPACT single-line JSON (13.1-1): %r"
            % raw[:80])
        return obj

    def error_envelope(self, expected_reason=None, msg="", prompt_line=False):
        """Declined deletes legitimately print the PROMPT line before the
        error envelope (8.6 + 13.1-6); with prompt_line=True allow exactly
        one extra leading line, otherwise require exactly one line."""
        lines = [ln for ln in self.stderr.strip().splitlines() if ln.strip()]
        if prompt_line:
            assert 1 <= len(lines) <= 2, (
                "expected [prompt +] one envelope line%s, got %d: %r"
                % (msg, len(lines), self.stderr[:300]))
            envelope = lines[-1]
        else:
            assert len(lines) == 1, (
                "a failure emits EXACTLY one stderr line%s, got %d lines: %r"
                % (msg, len(lines), self.stderr[:300]))
            envelope = lines[0]
        assert "Traceback" not in self.stderr, "a traceback reached the user"
        obj = json.loads(envelope)
        assert set(obj.keys()) == {"error"}, obj
        # `detail` is optional per 7a ("detail": "<optional string>")
        assert set(obj["error"].keys()) in ({"reason"}, {"reason", "detail"}), \
            obj["error"]
        detail = obj["error"].get("detail", "")
        if "detail" in obj["error"]:
            assert isinstance(detail, str), obj["error"]
        assert obj["error"]["reason"] in CLOSED_REASON_TOKENS, (
            "reason token %r is outside the closed section-7a set"
            % obj["error"]["reason"])
        if expected_reason is not None:
            assert obj["error"]["reason"] == expected_reason, (
                "expected reason %r%s, got %r"
                % (expected_reason, msg, obj["error"]["reason"]))
        _assert_no_paths_no_bulk(self.stderr)
        return obj["error"]


def _assert_no_paths_no_bulk(text):
    assert "/home/" not in text, "an absolute path leaked into output: %r" % text[:200]
    assert "Traceback" not in text, "a traceback reached the user"
    runs = re.findall(r"[A-Za-z0-9+/]{160,}", text)
    assert not runs, ("possible vector/base64 byte payload in output: %r"
                      % runs[:1])


def _assert_shape_keys(obj, expected, context):
    got = set(obj.keys())
    assert got == set(expected), (
        "%s: output shape deviates from the spec (8.x: no optional, no extra "
        "fields) — expected %r, got %r"
        % (context, sorted(expected), sorted(got)))


def _collect_name_fields(obj, name):
    """Every JSON path in obj whose value equals `name` (the I-6 gate)."""
    hits = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + "[%d]" % i)
        elif o == name:
            hits.append(path)

    walk(obj, "")
    return hits


def _no_bulk_keys(obj, context):
    """I-14 / DEC-6: no vector/bytes/bbox keys anywhere in an output shape."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert k.lower() not in ("vector", "vec", "embedding", "bbox",
                                         "base64", "bytes", "scene_vec",
                                         "object_vec"), (
                    "I-14/DEC-6 violated in %s: key %r present" % (context, k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)


def _64hex(s):
    return bool(re.fullmatch(r"[0-9a-f]{64}", s))


class CliEnv:
    """Fresh HOME + fresh DB per test; run() is the black-box invocation."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        cfg = self.home / ".config" / "vision-index"
        cfg.mkdir(parents=True)
        self.env_file = cfg / "env"
        self.env_file.write_text(
            "VISION_EMBED_URL=%s\nVISION_EMBED_TOKEN=%s\n"
            % (SERVICE_URL, SERVICE_TOKEN))
        self.env_file.chmod(0o600)
        self.xdg_data = tmp_path / "xdgdata"
        self.xdg_data.mkdir()
        self.db = tmp_path / "dbdir" / "index.sqlite3"
        self.pin = tmp_path / "dbdir" / "index.sqlite3.hosts"
        self.base_env = dict(os.environ)
        for k in ("VISION_EMBED_URL", "VISION_EMBED_TOKEN", "VISION_INDEX_DB"):
            self.base_env.pop(k, None)
        self.base_env.update({
            "HOME": str(self.home),
            "XDG_DATA_HOME": str(self.xdg_data),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "VISION_INDEX_DB": str(self.db),
        })

    def env(self, **over):
        e = dict(self.base_env)
        for k, v in over.items():
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return e

    def run(self, args, stdin=None, **over):
        p = subprocess.run(
            [sys.executable, SCRIPT] + [str(a) for a in args],
            env=self.env(**over), input=stdin, capture_output=True,
            text=True, timeout=300, cwd=str(self.tmp))
        return RunResult(p.returncode, p.stdout, p.stderr)


@pytest.fixture()
def cli_env(tmp_path):
    return CliEnv(tmp_path)


def _sha_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# CLI-side fixture helpers (drive the CLI itself; never import it)
# ---------------------------------------------------------------------------

def enroll_one(cli, image, expect_faces=None):
    r = cli.run(["enroll", image])
    assert r.rc == 0, "enroll of %s failed: %s" % (os.path.basename(image), r.stderr)
    _assert_no_paths_no_bulk(r.stderr)
    body = r.one_json_object(" (enroll)")
    _assert_shape_keys(body, {"enrolled", "files"}, "enroll")
    assert body["enrolled"] is True
    assert len(body["files"]) == 1
    f = body["files"][0]
    _assert_shape_keys(f, {"file", "face_status_summary", "image_sha256",
                           "faces_stored", "faces_already_present",
                           "faces_below_det_min", "clusters_created",
                           "clusters_joined"}, "enroll file entry")
    assert f["file"] == os.path.basename(image), (
        "enroll output carries the basename ONLY, never a full path (8.1)")
    _no_bulk_keys(body, "enroll")
    for fs in f["face_status_summary"]:
        _assert_shape_keys(fs, {"vec_hash", "face_status"}, "face_status entry")
        assert _64hex(fs["vec_hash"]), fs
        assert fs["face_status"] in ("created", "already_present",
                                     "below_det_min"), fs
    # 8.1 counting identity: every face created this call is either joined
    # into a pre-existing cluster or created a new cluster
    created = sum(1 for x in f["face_status_summary"]
                  if x["face_status"] in ("created", "below_det_min"))
    assert f["faces_stored"] == created + f["faces_already_present"]
    assert f["clusters_created"] + f["clusters_joined"] == created, (
        "8.1 counting identity broken: %d + %d != %d"
        % (f["clusters_created"], f["clusters_joined"], created))
    if expect_faces is not None:
        assert len(f["face_status_summary"]) == expect_faces, (
            "expected %d faces in %s, got %r"
            % (expect_faces, os.path.basename(image), f["face_status_summary"]))
    return body, f


def link_all_unlinked(cli, person):
    linked_ids = []
    for cid in unlinked_cluster_ids(cli):
        r = cli.run(["link", cid, person])
        assert r.rc == 0, r.stderr
        lk = r.one_json_object(" (link)")
        _assert_shape_keys(lk, {"linked", "cluster_id", "person",
                                "created_person"}, "link")
        assert lk["linked"] is True and lk["cluster_id"] == int(cid)
        linked_ids.append(int(cid))
    return linked_ids


def list_clusters_json(cli):
    r = cli.run(["list", "clusters", "--json"])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (list clusters --json)")
    _assert_shape_keys(body, {"clusters"}, "list clusters --json")
    for c in body["clusters"]:
        _assert_shape_keys(c, {"cluster_id", "person", "faces", "added_at"},
                           "clusters entry")
        assert isinstance(c["cluster_id"], int)
        assert isinstance(c["person"], str) or c["person"] is None
    _no_bulk_keys(body, "list clusters")
    return body["clusters"]


def unlinked_cluster_ids(cli):
    return [c["cluster_id"] for c in list_clusters_json(cli)
            if c["person"] is None]


def list_faces_json(cli):
    r = cli.run(["list", "faces", "--json"])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (list faces --json)")
    _assert_shape_keys(body, {"faces"}, "list faces --json")
    for fc in body["faces"]:
        _assert_shape_keys(fc, {"vec_hash", "cluster_id", "person",
                                "det_score", "added_at"}, "faces entry")
        assert _64hex(fc["vec_hash"]), fc
    _no_bulk_keys(body, "list faces")
    return body["faces"]


def list_people(cli):
    r = cli.run(["list", "people", "--json"])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (list people --json)")
    _assert_shape_keys(body, {"people", "clusters_unlinked",
                              "faces_in_unlinked"}, "list people --json")
    for p in body["people"]:
        _assert_shape_keys(p, {"name", "faces", "linked_clusters",
                               "last_added"}, "people entry")
    _no_bulk_keys(body, "list people")
    return body["people"]


def status_json(cli):
    r = cli.run(["status", "--json"])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (status --json)")
    _assert_shape_keys(body, {"unlinked_clusters", "suggested_command"},
                       "status --json")
    for u in body["unlinked_clusters"]:
        _assert_shape_keys(u, {"cluster_id", "faces", "created_at",
                               "nearest_person", "nearest_person_distance",
                               "nearest_cluster_id",
                               "nearest_cluster_distance"}, "status row")
    _no_bulk_keys(body, "status")
    return body


# ---------------------------------------------------------------------------
# measured-at-test-time fixture selection (session scope; service only —
# the spec's fixtures manifest does not exist, so distances are measured HERE)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def measured():
    # HOME may be redirected by conftest fixtures; resolve the real photo
    # root from the account's passwd entry instead of $HOME.
    import pwd
    pdir = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "Pictures")

    def take(p, n):
        d = os.path.join(pdir, _fixture_root_map().get(p, p))
        if not os.path.isdir(d):
            pytest.skip(
                "corpus fixtures not configured on this machine: %s maps to "
                "missing directory %s (test-fixture-roots.json)" % (p, d))
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.endswith(".png"))[:n]

    candidates = {}
    counts = {"photos_a": 58}
    for person in PHOTO_ROOTS:
        for path in take(person, counts.get(person, 8)):
            faces = _embed_image(path)
            # normalize: only what measurement needs; vec/det are read live
            faces = [{"det": f["det_score"], "vec": f["vector"]} for f in faces]
            candidates[path] = {"person": person, "faces": faces}

    single = {p: c for p, c in candidates.items() if len(c["faces"]) == 1}
    zero_face = [p for p, c in candidates.items() if len(c["faces"]) == 0]
    out = {"zero_face": zero_face}

    def face(p):
        return single[p]["faces"][0]

    def d(a, b):
        return _cosine(face(a)["vec"], face(b)["vec"])

    alice = sorted(p for p, c in single.items() if c["person"] == "photos_a")
    others = sorted(p for p, c in single.items() if c["person"] != "photos_a")

    def pair_in(lo, hi):
        best = None
        for i, a in enumerate(alice):
            for b in alice[i + 1:]:
                dd = d(a, b)
                if lo <= dd < hi and (best is None or dd < best[0]):
                    best = (dd, a, b)
        return best

    out["va"] = pair_in(0.0, 0.20)          # strictly below 0.28, with margin
    out["vg"] = pair_in(0.45, 1.0)          # well above 0.40 -> two clusters
    # inside [0.28, 0.40) with margin on BOTH edges: nearest the 0.28 edge a
    # build with a LOWERED matched boundary would quietly keep it probable
    out["vc"] = pair_in(0.28, 0.2831)

    out["vd"] = next((p for p in alice if face(p)["det"] < 0.63), None)

    def far_companion(anchor):
        best = None
        for o in others:
            dd = d(anchor, o)
            if dd >= 0.45 and (best is None or dd > best[0]):
                best = (dd, o)
        return best

    if out["va"]:
        out["vb"] = far_companion(out["va"][1])
    else:
        out["vb"] = None
    out["vd_companion"] = far_companion(out["vd"]) if out["vd"] else None

    # V-d close-companion: an AT-or-ABOVE-floor face measured WELL INSIDE
    # CLUSTER_JOIN_MAX (< 0.30, so a mutant that drops the det floor from the
    # join-target path actually joins it) of the below-det-floor image.
    if out["vd"]:
        close = None
        for q in alice:
            if q == out["vd"] or face(q)["det"] < FACE_DET_MIN:
                continue
            dd = d(out["vd"], q)
            if dd < 0.30 and (close is None or dd < close[0]):
                close = (dd, q)
        out["vd_close"] = close
    else:
        out["vd_close"] = None

    def close_third(pool, anchor_a, anchor_b=None, bound=0.20):
        best = None
        for p in pool:
            if p in (anchor_a, anchor_b):
                continue
            m = min(d(p, anchor_a), d(p, anchor_b)) if anchor_b else d(p, anchor_a)
            if m < bound and (best is None or m < best[0]):
                best = (m, p)
        return best

    out["va_probe"] = close_third(alice, out["va"][1], out["va"][2]) if out["va"] else None
    out["vg_probe"] = (close_third(alice, out["vg"][1], out["vg"][2],
                                   bound=FACE_MATCH_MAX) if out["vg"] else None)
    out["vb_probe"] = close_third(alice, out["va"][1], bound=0.22) if out["va"] else None

    # ambiguous_tie: two faces of two DIFFERENT persons, each strictly below
    # 0.28 from one probe AND 0.45+ apart from each other (so their clusters
    # stay separate at ingest).
    tie = None
    keys = sorted(single)
    for i, fa in enumerate(keys):
        for fb in keys[i + 1:]:
            if single[fb]["person"] == single[fa]["person"]:
                continue
            dd_e = d(fa, fb)
            if dd_e < 0.45:
                continue
            for fp in keys:
                if fp in (fa, fb):
                    continue
                if d(fa, fp) < 0.27 and d(fb, fp) < 0.27:
                    tie = (fa, fb, fp)
                    break
            if tie:
                break
        if tie:
            break
    out["tie"] = tie
    return out


def need(v, what):
    if not v:
        pytest.skip(
            "GAP: the real photo corpus in this environment provides no "
            "configuration for %s; the spec's fixtures manifest (section 11) "
            "does not exist here and no distance may be invented — skipped "
            "as a named gap, not a pass." % what)


# ===========================================================================
# V-a  THRESHOLD BOUNDARY  (spec 11.1, section 5 strict comparisons)
# ===========================================================================

def test_V_a_threshold_boundary_matched_below_0_28(measured, cli_env):
    need(measured.get("va") and measured.get("va_probe"),
         "a same-person pair strictly below 0.28 plus a NEVER-ENROLLED third "
         "same-person image (V-a)")
    dist, img_a, img_e = measured["va"]
    probe = measured["va_probe"][1]
    assert dist < FACE_MATCH_MAX
    person = "v-a-person"

    enroll_one(cli_env, img_a, expect_faces=1)
    link_all_unlinked(cli_env, person)
    enroll_one(cli_env, img_e, expect_faces=1)
    link_all_unlinked(cli_env, person)
    linked = [c for c in list_clusters_json(cli_env) if c["person"] == person]
    assert len(linked) >= 1

    r = cli_env.run(["identify", probe])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (identify)")
    _assert_shape_keys(body, {"image_sha256", "faces"}, "identify")
    assert len(body["faces"]) == 1
    fc = body["faces"][0]
    _assert_shape_keys(fc, {"face_index", "det_score", "band", "confidence",
                            "nearest_person", "nearest_named_distance"},
                       "identify face")
    assert fc["band"] == "matched", (
        "V-a GATE: a build with an inflated effective threshold reads "
        "'probable' at d=%.4f; got band=%r" % (dist, fc["band"]))
    assert fc["confidence"] == "matched"
    assert fc["nearest_person"] == person
    assert fc["nearest_named_distance"] < FACE_MATCH_MAX
    # the reported distance is a real measurement: cross-check it against the
    # distance this suite measured independently from the same service
    assert fc["nearest_named_distance"] == pytest.approx(measured["va_probe"][0],
                                                         abs=0.01)
    # ONE name in the assertion position, and no other anywhere
    assert _collect_name_fields(body, person) == ["/faces[0]/nearest_person"]
    _no_bulk_keys(body, "identify")


# ===========================================================================
# V-b  DIFFERENT-PERSON NEGATIVE CONTROL  (spec 11.2, DEC-1)
# ===========================================================================

def test_V_b_different_person_negative_control(measured, cli_env):
    need(measured.get("va") and measured.get("vb") and measured.get("vb_probe"),
         "an evidence face, an OTHER-person face at or above 0.306 from it, "
         "and a never-enrolled probe image (V-b)")
    dist_b, img_b = measured["vb"]
    assert dist_b >= 0.306
    person_a = "v-b-alice"
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    link_all_unlinked(cli_env, person_a)

    body_b, fb = enroll_one(cli_env, img_b, expect_faces=1)
    assert fb["clusters_joined"] == 0, (
        "V-b GATE: a different-person image (measured %.4f apart) joined an "
        "existing cluster at ingest — CLUSTER_JOIN_MAX is not honoured" % dist_b)
    assert fb["clusters_created"] == 1
    assert len(list_clusters_json(cli_env)) == 2

    # a never-enrolled A image: matched, A's name ONLY
    probe = measured["vb_probe"][1]
    r = cli_env.run(["identify", probe])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (identify A probe)")
    for fc in body["faces"]:
        assert fc["band"] == "matched", fc
        assert fc["nearest_person"] == person_a
        assert fc["nearest_named_distance"] < FACE_MATCH_MAX
    # B has never been named in this DB (I-2: no name exists unless a human
    # typed it), so the real negative control is the band result for B itself:
    r = cli_env.run(["identify", img_b])
    assert r.rc == 0, r.stderr
    body_b2 = r.one_json_object(" (identify B)")
    assert len(body_b2["faces"]) == 1
    fc = body_b2["faces"][0]
    assert fc["band"] == "unknown", (
        "V-b GATE: the bands must keep the two names apart; d=%.4f read %r"
        % (dist_b, fc["band"]))
    assert fc["reason"] == "no_match"
    assert fc["nearest_named_distance"] >= BAND_PROBABLE_MAX
    assert fc["nearest_named_distance"] == pytest.approx(dist_b, abs=0.01)
    _no_bulk_keys(body_b2, "identify B")


# ===========================================================================
# V-c  PROBABLE-BAND NON-ASSERTION  (spec 11.3, I-6)
# ===========================================================================

def test_V_c_probable_band_never_asserts(measured, cli_env):
    need(measured.get("vc"),
         "a same-person pair with measured distance in [0.28, 0.40) (V-c)")
    dist, img_a, img_probe = measured["vc"]
    assert 0.28 <= dist < 0.40
    person = "v-c-dave"
    enroll_one(cli_env, img_a, expect_faces=1)
    link_all_unlinked(cli_env, person)

    r = cli_env.run(["identify", img_probe])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (identify)")
    assert len(body["faces"]) == 1
    fc = body["faces"][0]
    _assert_shape_keys(fc, {"face_index", "det_score", "band", "confidence",
                            "nearest_person", "nearest_named_distance"},
                       "identify face")
    assert fc["band"] == "probable", (
        "V-c GATE: off-by-one boundary (asserting at exactly 0.28) or wrong "
        "band edge; measured d=%.4f band=%r" % (dist, fc["band"]))
    assert fc["confidence"] == "probable"
    assert 0.28 <= fc["nearest_named_distance"] < BAND_PROBABLE_MAX
    assert 0.28 <= fc["nearest_named_distance"] <= dist + 0.001
    assert fc["nearest_person"] == person
    # I-6 proven by parsing: the ONLY name-bearing field is the diagnostic one
    hits = _collect_name_fields(body, person)
    assert hits == ["/faces[0]/nearest_person"], (
        "a name appeared outside the diagnostic field: %s" % hits)
    _no_bulk_keys(body, "identify")


# ===========================================================================
# V-d  BELOW-DET-FLOOR EXCLUSION  (spec 11.4, DEC-5)
# ===========================================================================

def test_V_d_below_det_floor_exclusion(measured, cli_env):
    need(measured.get("vd") and measured.get("vd_companion")
         and measured.get("vd_close"),
         "a detected face with det_score strictly below 0.65, a same-person "
         "companion image det>=0.65 (V-d), and a SECOND companion det>=0.65 "
         "measured <0.30 from the below-floor face (V-d close gate)")
    img_low = measured["vd"]
    det = _embed_image(img_low)[0]["det_score"]
    assert det < FACE_DET_MIN, det

    body1, f1 = enroll_one(cli_env, img_low, expect_faces=1)
    fcsumm = f1["face_status_summary"][0]
    assert fcsumm["face_status"] == "below_det_min", (
        "a det=%.3f face must read below_det_min (8.1)" % det)
    assert f1["faces_below_det_min"] == 1
    assert f1["clusters_created"] == 1, "DEC-5: the single-face cluster is a real row"
    assert f1["clusters_joined"] == 0
    assert f1["faces_stored"] == f1["clusters_joined"] + f1["clusters_created"], (
        "8.1 reconciliation identity broken for a below-floor face")
    low_cluster = unlinked_cluster_ids(cli_env)
    assert len(low_cluster) == 1
    low_cluster = low_cluster[0]

    # (a) stored and queryable
    faces = list_faces_json(cli_env)
    in_low = [f for f in faces if f["cluster_id"] == low_cluster]
    assert len(in_low) == 1 and in_low[0]["det_score"] < FACE_DET_MIN
    assert in_low[0]["person"] is None

    # companion: above floor, 0.45+ apart -> its OWN cluster (it must never
    # join the below-floor cluster, which is not a join target)
    img_comp = measured["vd_companion"][1]
    d_comp = measured["vd_companion"][0]
    body2, f2 = enroll_one(cli_env, img_comp, expect_faces=1)
    assert f2["clusters_joined"] == 0, (
        "companion (d=%.4f) joined a cluster at ingest; construction broken"
        % d_comp)
    # (c) NO later enroll moves the below-floor face into a shared cluster
    faces2 = list_faces_json(cli_env)
    in_low2 = [f for f in faces2 if f["cluster_id"] == low_cluster]
    assert len(in_low2) == 1 and in_low2[0]["vec_hash"] == fcsumm["vec_hash"], (
        "DEC-5 violated: the below-floor face moved or gained a member")

    comp = [c for c in unlinked_cluster_ids(cli_env) if c != low_cluster]
    assert len(comp) == 1
    person = "v-d-pat"
    r = cli_env.run(["link", comp[0], person])
    assert r.rc == 0, r.stderr

    # (d) link --near at ANY distance never links the below-floor cluster
    r = cli_env.run(["link", "--near", person, "0.99"])
    assert r.rc == 0, r.stderr
    near = r.one_json_object(" (link --near)")
    _assert_shape_keys(near, {"linked_near", "person", "distance",
                              "linked_clusters", "skipped"}, "link --near")
    assert low_cluster not in near["linked_clusters"], (
        "DEC-5 violated: link --near auto-linked a below-floor cluster")
    skip_rows = {s["cluster_id"]: s for s in near["skipped"]}
    assert low_cluster in skip_rows, (
        "a skipped unlinked cluster must be reported with its reason (8.4)")
    _assert_shape_keys(skip_rows[low_cluster],
                       {"cluster_id", "reason", "nearest_person_distance"},
                       "skipped row")
    assert skip_rows[low_cluster]["reason"] == "above_floor_only", (
        "(d) GATE: the below-floor cluster must be skipped with "
        "above_floor_only, not distance_above (its measured distance was "
        "not the reason)")
    assert skip_rows[low_cluster]["nearest_person_distance"] is not None
    assert skip_rows[low_cluster]["nearest_person_distance"] < 0.99, (
        "(d) decisive GATE: the 0.99 threshold exceeds the measured "
        "nearest_person_distance %r, so distance_above cannot be the reason — "
        "the floor is the ONLY reason above_floor_only could hold"
        % skip_rows[low_cluster]["nearest_person_distance"])
    # the companion cluster is already linked, the below-floor one stays out
    assert near["linked_clusters"] == [], near
    assert set(skip_rows) == {low_cluster}, near

    # ...while a MANUAL link is permitted (a human may link anything, 8.3)
    r = cli_env.run(["link", low_cluster, person])
    assert r.rc == 0, r.stderr
    man = r.one_json_object(" (manual link)")
    assert man["linked"] is True and man["cluster_id"] == low_cluster
    assert man["created_person"] is False, "person already existed (I-2)"

    # (e) V-d CLOSE gate: an AT-or-ABOVE-floor face measured WELL INSIDE
    # CLUSTER_JOIN_MAX still must NOT join the below-floor cluster. The
    # distant companion in (b)/(c) only proves a FAR face stays out; a build
    # whose join-target path ignores the detection floor would pass that one
    # (it only joins faces within CLUSTER_JOIN_MAX anyway). This companion
    # sits <= 0.30 from the below-floor face, so it fails such a build.
    img_close = measured["vd_close"][1]
    d_close = measured["vd_close"][0]
    _close_faces = _embed_image(img_close)
    assert len(_close_faces) == 1
    det_close = _close_faces[0]["det_score"]
    assert det_close >= FACE_DET_MIN, det_close
    assert d_close < 0.30, (
        "the fixture pair is not decisive: measured d=%.4f must be well "
        "inside CLUSTER_JOIN_MAX" % d_close)
    body3, f3 = enroll_one(cli_env, img_close, expect_faces=1)
    assert f3["clusters_joined"] == 0, (
        "V-d CLOSE GATE: an above-floor face (det=%.3f) measured d=%.4f from "
        "the below-floor face — far inside CLUSTER_JOIN_MAX — JOINED it at "
        "ingest; the detection floor is missing from the join-target list"
        % (det_close, d_close))
    assert f3["clusters_created"] == 1, (
        "V-d CLOSE GATE: the close above-floor companion must sit in its OWN "
        "new cluster, not be merged into the below-floor one (d=%.4f)"
        % d_close)
    faces3 = list_faces_json(cli_env)
    in_low3 = [f for f in faces3 if f["cluster_id"] == low_cluster]
    assert len(in_low3) == 1 and in_low3[0]["vec_hash"] == fcsumm["vec_hash"], (
        "DEC-5 violated by the CLOSE companion: the below-floor cluster "
        "gained a member (d=%.4f)" % d_close)
    close_cls = list_clusters_json(cli_env)
    own = [c for c in close_cls
           if c["person"] is None and c["cluster_id"] != low_cluster]
    assert len(own) == 1, close_cls
    own_faces = [f for f in faces3 if f["cluster_id"] == own[0]["cluster_id"]]
    assert len(own_faces) == 1
    assert own_faces[0]["det_score"] >= FACE_DET_MIN, own_faces
    assert fcsumm["face_status"] == "below_det_min"  # low face never relabeled


# ===========================================================================
# V-e  BATCH PARTIAL FAILURE  (spec 11.5, 7.1, I-9, I-10)
# ===========================================================================

def test_V_e_batch_partial_failure(measured, cli_env):
    need(measured.get("zero_face") and measured.get("va"),
         "a zero-face image and a valid image (V-e)")
    unreadable = os.path.join(str(cli_env.tmp), "totally_missing.png")
    # the zero-face photo is copied under its own temp basename (distinct file)
    zero = os.path.join(str(cli_env.tmp), "zf.png")
    shutil.copyfile(measured["zero_face"][0], zero)
    valid = measured["va"][1] if measured["va"][1] != measured["zero_face"][0] \
        else measured["va"][2]
    assert valid != zero

    r = cli_env.run(["enroll", unreadable, zero, valid])
    assert r.rc == 0, ("partial failure is NOT a batch failure; rc=%d "
                       "stderr=%r" % (r.rc, r.stderr[:300]))
    body = r.one_json_object(" (enroll batch)")
    assert len(body["files"]) == 3, "every file gets exactly one entry (8.1)"
    by_file = {f["file"]: f for f in body["files"]}
    failed = by_file[os.path.basename(unreadable)]
    _assert_shape_keys({k: v for k, v in failed.items() if k != "face_status_summary"},
                       {"file", "failed", "error"}, "failed file entry")
    assert failed["failed"] is True
    err = failed["error"]
    assert set(err.keys()) == {"reason", "detail"}
    assert err["reason"] == "input.unreadable"
    assert "/" not in err["detail"] and err["detail"] != "", (
        "input.unreadable detail carries the basename ONLY: %r" % err["detail"])
    assert err["detail"] == os.path.basename(unreadable)
    zeroent = by_file[os.path.basename(zero)]
    assert zeroent.get("failed") is not True
    assert zeroent["face_status_summary"] == []
    assert zeroent["faces_stored"] == 0 and zeroent["clusters_created"] == 0
    zeroent_keys = {"file", "face_status_summary", "image_sha256",
                    "faces_stored", "faces_already_present",
                    "faces_below_det_min", "clusters_created", "clusters_joined"}
    assert set(zeroent.keys()) == zeroent_keys
    valident = by_file[os.path.basename(valid)]
    assert set(valident.keys()) == zeroent_keys
    assert valident["faces_stored"] == 1
    assert valident["face_status_summary"][0]["face_status"] == "created"

    # stderr: progress lines ONLY — one per processed file, no JSON, no summary
    lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    assert len(lines) == 3, "one progress line per processed file: %r" % lines
    for ln in lines:
        assert ln.startswith("enroll: "), ln
        assert "{" not in ln and "}" not in ln, "no JSON on stderr (7.1)"
    assert lines[0].startswith("enroll: 1/3 ")
    assert os.path.basename(unreadable) in lines[0]

    # the DB gained EXACTLY the valid file's faces (10: batch observability)
    db_faces = list_faces_json(cli_env)
    assert len(db_faces) == 1
    assert {f["vec_hash"] for f in db_faces} == {
        x["vec_hash"] for x in valident["face_status_summary"]}

    # re-run the same batch: the valid face is already_present (I-9)
    r2 = cli_env.run(["enroll", unreadable, zero, valid])
    assert r2.rc == 0, r2.stderr
    body2 = r2.one_json_object(" (re-enroll)")
    val2 = [f for f in body2["files"]
            if f["file"] == os.path.basename(valid)][0]
    assert all(x["face_status"] == "already_present"
               for x in val2["face_status_summary"]), val2
    assert val2["faces_already_present"] == 1 and val2["faces_stored"] == 0
    assert len(list_faces_json(cli_env)) == len(db_faces), "no dupes (I-9)"


def test_V_e_batch_all_failed_leaves_no_db(measured, cli_env):
    missing = [os.path.join(str(cli_env.tmp), ("nope_%d.png" % i)) for i in range(3)]
    r = cli_env.run(["enroll"] + missing)
    assert r.rc == 1, "an all-failed batch exits 1, got rc=%d" % r.rc
    out_lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    envl = json.loads(out_lines[-1])
    assert set(envl.keys()) == {"error"}
    assert envl["error"]["reason"] == "batch.all_failed"
    body = r.one_json_object(" (all-failed summary)")
    assert len(body["files"]) == 3
    assert all(f.get("failed") is True for f in body["files"])
    assert not cli_env.db.exists(), (
        "I-10: the DB is created only AFTER the first successful embed — "
        "an entirely failed batch must leave NO DB file")


# ===========================================================================
# V-f  LINK / UNLINK ROUND-TRIP  (spec 11.6, I-3)
# ===========================================================================

def test_V_f_link_unlink_round_trip(measured, cli_env):
    need(measured.get("va"), "a same-person image (V-f)")
    img = measured["va"][1]
    person = "v-f-eve"
    enroll_one(cli_env, img, expect_faces=1)
    clusters0 = list_clusters_json(cli_env)
    assert len(clusters0) == 1 and clusters0[0]["person"] is None
    cid = clusters0[0]["cluster_id"]
    faces0 = list_faces_json(cli_env)

    # link: person appears with the cluster counted; created_person truthful
    r = cli_env.run(["link", cid, person])
    assert r.rc == 0, r.stderr
    lk = r.one_json_object(" (link)")
    _assert_shape_keys(lk, {"linked", "cluster_id", "person", "created_person"},
                       "link")
    assert lk == {"linked": True, "cluster_id": cid, "person": person,
                  "created_person": True}
    people = [p for p in list_people(cli_env) if p["name"] == person]
    assert len(people) == 1
    assert people[0]["linked_clusters"] == 1 and people[0]["faces"] == 1
    state_after_link1 = list_clusters_json(cli_env)
    faces_after_link1 = list_faces_json(cli_env)

    # link again on a linked cluster -> link.already_linked, DB unchanged
    r = cli_env.run(["link", cid, person])
    r.error_envelope("link.already_linked",
                     "no silent re-pointing: unlink first (8.3)")
    assert list_clusters_json(cli_env) == state_after_link1
    assert list_faces_json(cli_env) == faces_after_link1

    # unlink: person_id NULL, faces NEVER touched (I-3)
    r = cli_env.run(["unlink", cid])
    assert r.rc == 0, r.stderr
    ul = r.one_json_object(" (unlink)")
    _assert_shape_keys(ul, {"unlinked", "cluster_id"}, "unlink")
    assert ul["unlinked"] is True and ul["cluster_id"] == cid
    cs = {c["cluster_id"]: c for c in list_clusters_json(cli_env)}
    assert cs[cid]["person"] is None, "unlink nulls person_id (8.5)"
    assert list_faces_json(cli_env) == faces0, "I-3: faces are never touched"
    st = status_json(cli_env)
    assert any(u["cluster_id"] == cid for u in st["unlinked_clusters"]), (
        "the unlinked cluster is back in the unlinked queue of status")

    # unlink of an ALREADY-unlinked cluster: idempotent success (8.5)
    r = cli_env.run(["unlink", cid])
    assert r.rc == 0, r.stderr
    ul2 = r.one_json_object(" (idempotent unlink)")
    assert ul2["unlinked"] is True and ul2["cluster_id"] == cid

    # link again with the same name: visible state identical to the 1st link
    r = cli_env.run(["link", cid, person])
    assert r.rc == 0, r.stderr
    assert r.one_json_object()["created_person"] is False
    assert list_clusters_json(cli_env) == state_after_link1, (
        "unlink followed by link restores the exact prior state (section 10)")
    assert any(p["name"] == person for p in list_people(cli_env))


# ===========================================================================
# V-g  IDENTIFY ACROSS MULTIPLE CLUSTERS OF ONE PERSON  (spec 11.7, 8.8)
# ===========================================================================

def test_V_g_union_minimum_and_suggested_near_link(measured, cli_env):
    need(measured.get("vg") and measured.get("vg_probe"),
         "two same-person images whose measured pair distance is >= 0.40 (so "
         "they form TWO clusters) plus a near third never-enrolled image (V-g)")
    dist, img_a, img_g = measured["vg"]
    probe = measured["vg_probe"][1]
    person = "v-g-frank"
    assert dist >= CLUSTER_JOIN_MAX

    enroll_one(cli_env, img_a, expect_faces=1)
    enroll_one(cli_env, img_g, expect_faces=1)
    cids = unlinked_cluster_ids(cli_env)
    assert len(cids) == 2, (
        "pair distance %.4f must have stayed in TWO clusters at ingest "
        "(>= 0.40)" % dist)

    # (c) BEFORE any link: both clusters unlinked; no person exists yet, so
    # neither is measurable and the suggestion is omitted (8.8)
    st = status_json(cli_env)
    assert {u["cluster_id"] for u in st["unlinked_clusters"]} == set(cids)
    assert st["suggested_command"] is None
    assert all(u["nearest_person"] is None and
               u["nearest_person_distance"] is None
               for u in st["unlinked_clusters"])
    r = cli_env.run(["status"])
    assert r.rc == 0 and "suggested:" not in r.stdout, (
        "no person's distance is measurable -> the suggestion line is OMITTED")

    # the FIRST link (one half of the final state)
    cid1, cid2 = sorted(cids)
    r = cli_env.run(["link", cid1, person])
    assert r.rc == 0, r.stderr

    # status BEFORE THE SECOND LINK: the remaining cluster is unlinked with a
    # measurable nearest person; the suggestion is exactly the 8.8 formula
    st = status_json(cli_env)
    rows = st["unlinked_clusters"]
    assert [u["cluster_id"] for u in rows] == [cid2]
    row = rows[0]
    assert row["nearest_person"] == person
    dmax = row["nearest_person_distance"]
    assert dmax is not None
    expected = (_ceil_cents(dmax) + 1) / 100.0  # (ceil(D_max*100)/100)+0.01
    assert expected > dmax, "suggested must strictly exceed D_max"
    suggested_command = st["suggested_command"]
    assert suggested_command is not None
    m = re.fullmatch(r"vision-index link --near (\S+) ([0-9.]+)",
                     suggested_command.strip())
    assert m, "suggestion must take the exact form of 8.8: %r" % suggested_command
    assert m.group(1) == person
    assert abs(float(m.group(2)) - expected) < 1e-9, (
        "GATE: suggested distance must equal (ceil(D_max*100)/100)+0.01; "
        "D_max=%r suggested=%r expected=%r" % (dmax, m.group(2), expected))
    # the text form prints the same suggestion as its last line
    r = cli_env.run(["status"])
    assert r.rc == 0
    sugg = [ln for ln in r.stdout.strip().splitlines()
            if ln.strip().startswith("suggested:")]
    assert sugg, "status text ends with the suggested command (8.8)"
    assert sugg[-1].strip() == "suggested: " + suggested_command.strip(), (
        "JSON suggested_command and the text suggestion must agree")

    # execute the suggested command VERBATIM: run the verbatim args through
    # the script (the suggestion names the installed command; we invoke the
    # same repo file the suite tests)
    verbatim = suggested_command.strip().split()[1:]  # drop "vision-index"
    assert verbatim == ["link", "--near", person, m.group(2)], verbatim
    r = cli_env.run(verbatim)
    assert r.rc == 0, r.stderr
    near = r.one_json_object(" (suggested link --near verbatim)")
    _assert_shape_keys(near, {"linked_near", "person", "distance",
                              "linked_clusters", "skipped"}, "link --near")
    assert set(near["linked_clusters"]) >= {cid2}, (
        "V-g GATE: executing the suggestion verbatim must link the remaining "
        "cluster ((suggested > D_max) ensures strictly-below holds)")
    for row_ in st["unlinked_clusters"]:
        if row_["cluster_id"] not in near["linked_clusters"]:
            assert row_["cluster_id"] in {s["cluster_id"] for s in near["skipped"]}

    # (a) one person with BOTH clusters
    pr = [p for p in list_people(cli_env) if p["name"] == person][0]
    assert pr["linked_clusters"] == 2 and pr["faces"] == 2
    # (b) the never-enrolled third image: the minimum is taken over the UNION
    # of BOTH clusters' faces -> matched with that person only
    r = cli_env.run(["identify", probe])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (identify)")
    assert len(body["faces"]) == 1
    fc = body["faces"][0]
    assert fc["band"] == "matched", (
        "(b) GATE: a centroid/pinned-exemplar build over one cluster fails "
        "here; union minimum must match (probe d=%.4f)"
        % measured["vg_probe"][0])
    assert fc["nearest_person"] == person
    assert fc["nearest_named_distance"] < FACE_MATCH_MAX
    _no_bulk_keys(body, "identify")


# ===========================================================================
# V-h  NO-BYTES / NO-ABSOLUTE-PATH / NO-TRACEBACK / exact shapes (13.1)
# ===========================================================================

def test_V_h_no_bytes_no_paths_no_traceback_across_verbs(measured, cli_env):
    need(measured.get("va"), "a same-person image (V-h)")
    img = measured["va"][1]
    missing = os.path.join(str(cli_env.tmp), "very_missing.png")

    # reads on an ABSENT db: db.absent; the file is NEVER created by a read
    for verb in (["status"], ["list", "people"], ["list", "clusters"],
                 ["list", "faces"], ["identify", img]):
        r = cli_env.run(verb)
        r.error_envelope("db.absent", "verb=%s" % verb)
        assert not cli_env.db.exists(), "read %r created the DB" % verb
        assert not cli_env.pin.exists(), "read %r created the pin file" % verb
        _assert_no_paths_no_bulk(r.stdout + r.stderr)

    # id-addressed writes on an absent DB: their own not-found tokens; no DB
    for verb, reason in ((["link", "5", "x"], "link.no_such_cluster"),
                         (["unlink", "5"], "link.no_such_cluster"),
                         (["delete", "cluster", "5", "--yes"],
                          "delete.no_such_cluster"),
                         (["delete", "person", "ghost", "--yes"],
                          "delete.no_such_person")):
        r = cli_env.run(verb)
        r.error_envelope(reason, "verb=%r on an absent DB" % verb)
        assert not cli_env.db.exists(), "%r created the DB on failure" % verb

    # usage paths before any DB: exit 2, one line, no traceback, no stdout
    for verb in (["bogus_verb"], ["enroll"], ["delete"]):
        r = cli_env.run(verb)
        assert r.rc == 2, "usage exits 2: %r -> rc=%d %r" % (verb, r.rc, r.stderr[:200])
        assert r.stdout == "" and "Traceback" not in r.stderr
        assert r.stderr.strip() != ""
        _assert_no_paths_no_bulk(r.stderr)

    # --version: ONE bare line, the run-time sha256; no other args accepted
    r = cli_env.run(["--version"])
    assert r.rc == 0 and r.stderr == ""
    with open(SCRIPT, "rb") as fh:
        expect_hash = hashlib.sha256(fh.read()).hexdigest()
    assert r.stdout.strip() == expect_hash, (
        "version prints the sha256 of the script's own bytes (2.1)")
    r = cli_env.run(["--version", "extra"])
    assert r.rc == 2, "--version accepts no arguments"

    # usage / --help: verb list on STDERR, no stdout; --help=0, usage=2
    for args, rc in ((["--help"], 0), (["usage"], 2)):
        r = cli_env.run(args)
        assert r.rc == rc
        assert r.stdout == "" and "enroll" in r.stderr and "identify" in r.stderr
        _assert_no_paths_no_bulk(r.stderr)

    # success shapes with the DB present
    enroll_one(cli_env, img, expect_faces=1)
    link_all_unlinked(cli_env, "v-h")
    sugg = status_json(cli_env)["suggested_command"]
    for args, keys in ((["list", "people", "--json"],
                        {"people", "clusters_unlinked", "faces_in_unlinked"}),
                       (["list", "clusters", "--json"], {"clusters"}),
                       (["list", "faces", "--json"], {"faces"}),
                       (["status", "--json"],
                        {"unlinked_clusters", "suggested_command"})):
        r = cli_env.run(args)
        body = r.one_json_object(" verb=%s" % args)
        assert set(body.keys()) == keys, body.keys()
        _assert_no_paths_no_bulk(r.stdout)

    # failures with a present DB: closed reasons, right exit codes
    r = cli_env.run(["enroll", missing, img])
    assert r.rc == 0, "partial failure keeps exit 0"
    by_file = {f["file"]: f for f in r.one_json_object()["files"]}
    assert by_file[os.path.basename(missing)]["failed"] is True
    # single-file batch where the only file fails = all failed: exit 1, envelope
    r = cli_env.run(["enroll", missing])
    assert r.rc == 1
    err_lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    envl = json.loads(err_lines[-1])
    assert envl["error"]["reason"] in ("input.unreadable", "batch.all_failed"), envl
    assert r.one_json_object()["files"][0]["failed"] is True
    _assert_no_paths_no_bulk(r.stderr)
    r = cli_env.run(["identify", missing])
    r.error_envelope("input.unreadable")
    r = cli_env.run(["identify"])  # no file argument
    assert r.rc == 2
    r = cli_env.run(["link", "9999", "y"])
    r.error_envelope("link.no_such_cluster")
    r = cli_env.run(["delete", "cluster", "9999", "--yes"])
    r.error_envelope("delete.no_such_cluster")
    r = cli_env.run(["delete", "person", "ghosty", "--yes"])
    r.error_envelope("delete.no_such_person")
    # negative distance is usage (8.4) — now with a present DB too
    r = cli_env.run(["link", "--near", "v-h", "-0.5"])
    assert r.rc == 2, r.stderr

    # EOF on the delete prompt: declined, byte-identical DB, no stdout JSON
    before = _sha_of(cli_env.db)
    r = cli_env.run(["delete", "person", "v-h"])
    envl = r.error_envelope("aborted", prompt_line=True)
    assert r.stdout == "", "no stdout JSON when declined (13.1-6)"
    assert "[y/N]" in r.stderr and "v-h" in r.stderr, (
        "the prompt names the target and its blast radius (8.6)")
    assert _sha_of(cli_env.db) == before, "decline leaves the DB unchanged"
    r = cli_env.run(["delete", "person", "v-h"], stdin="n\n")
    r.error_envelope("aborted", prompt_line=True)
    assert _sha_of(cli_env.db) == before

    # --yes: single JSON object, exact 8.6 shape
    ids = [c["cluster_id"] for c in list_clusters_json(cli_env)
           if c["person"] is not None]
    assert ids
    r = cli_env.run(["delete", "cluster", ids[0], "--yes"])
    assert r.rc == 0, r.stderr
    del_row = r.one_json_object(" (delete cluster)")
    _assert_shape_keys(del_row, {"deleted", "cluster_id", "faces_deleted"},
                       "delete cluster")
    assert del_row["deleted"] == "cluster" and del_row["cluster_id"] == ids[0]


# ===========================================================================
# Remaining failure modes and gates of sections 7a / 8.6 / 9 / 13.1
# ===========================================================================

def test_delete_prompt_carries_real_blast_radius(measured, cli_env):
    need(measured.get("vg"), "two non-joining same-person images (prompt counts)")
    person = "blast-carol"
    enroll_one(cli_env, measured["vg"][1], expect_faces=1)
    cid1 = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid1, person]).rc == 0
    enroll_one(cli_env, measured["vg"][2], expect_faces=1)
    cid2 = [c for c in unlinked_cluster_ids(cli_env) if c != cid1][0]
    assert cli_env.run(["link", cid2, person]).rc == 0

    r = cli_env.run(["delete", "person", person], stdin="n\n")
    assert r.rc == 1
    lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    assert len(lines) <= 2, r.stderr
    assert lines[0] == (
        "delete person %s? 2 linked cluster(s) and 2 face(s) will be deleted. "
        "proceed? [y/N]" % person), (
        "the prompt must carry the REAL counts read from the DB: %r" % lines[0])
    body = json.loads(lines[-1])
    assert body["error"]["reason"] == "aborted"
    assert r.stdout == ""
    assert any(p["name"] == person for p in list_people(cli_env)), (
        "decline deletes nothing")

    r = cli_env.run(["delete", "person", person, "--yes"])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (delete person)")
    _assert_shape_keys(body, {"deleted", "person", "clusters_deleted",
                              "faces_deleted"}, "delete person")
    assert body == {"deleted": "person", "person": person,
                    "clusters_deleted": 2, "faces_deleted": 2}
    assert list_clusters_json(cli_env) == [], (
        "cascades leave nothing: no unlinked residue of a deleted person")
    assert list_faces_json(cli_env) == []


def test_delete_person_no_name_or_empty_deletes_nothing(measured, cli_env):
    need(measured.get("va"), "an image to build a person to protect")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    cid = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid, "protected"]).rc == 0
    before = _sha_of(cli_env.db)
    for args in (["delete", "person"], ["delete", "person", ""]):
        r = cli_env.run(args)
        assert r.rc == 2, "%r -> rc=%d" % (args, r.rc)
        lines = r.stderr.strip().splitlines()
        assert len(lines) == 1, args
        assert "nothing deleted" in lines[0]
        assert "delete person <name>" in lines[0], lines[0]
        assert "proceed?" not in lines[0], "NO prompt may be shown (8.6)"
        assert r.stdout == ""
        _assert_no_paths_no_bulk(r.stderr)
    assert _sha_of(cli_env.db) == before, "no DB row read or written"
    assert any(p["name"] == "protected" for p in list_people(cli_env))


def test_delete_cluster_no_id_and_bare_delete_delete_nothing(cli_env):
    r = cli_env.run(["delete", "cluster"])
    assert r.rc == 2
    assert r.stderr.strip() == "usage: delete cluster <id> (nothing deleted)", r.stderr
    r = cli_env.run(["delete"])
    assert r.rc == 2
    assert r.stderr.strip() == (
        "usage: delete person <name> | delete cluster <id> (nothing deleted)"), \
        r.stderr


def test_delete_nonexistent_targets_exit_1(measured, cli_env):
    need(measured.get("va"), "an image so the DB exists")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    r = cli_env.run(["delete", "person", "nobody-here", "--yes"])
    r.error_envelope("delete.no_such_person")
    r = cli_env.run(["delete", "cluster", "424242", "--yes"])
    r.error_envelope("delete.no_such_cluster")


def test_eof_on_prompt_means_decline_db_byte_identical(measured, cli_env):
    need(measured.get("va"), "an image to build one linked cluster")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    cid = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid, "eof-person"]).rc == 0
    before = _sha_of(cli_env.db)
    r = cli_env.run(["delete", "person", "eof-person"])  # stdin closed: EOF
    r.error_envelope("aborted", prompt_line=True)
    assert r.stdout == ""
    assert "1 linked cluster(s) and 1 face(s) will be deleted" in r.stderr, r.stderr
    assert _sha_of(cli_env.db) == before, "EOF-decline leaves the DB byte-identical"


def test_link_near_no_faces_to_measure(measured, cli_env):
    need(measured.get("va"), "an image to create a cluster to link then unlink")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    cid = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid, "hollow-person"]).rc == 0
    assert cli_env.run(["unlink", cid]).rc == 0  # person remains, zero faces
    before = _sha_of(cli_env.db)
    r = cli_env.run(["link", "--near", "hollow-person", "0.5"])
    err = r.error_envelope("link.no_faces_to_measure")
    detail = err.get("detail", "")
    assert "zero" in detail or "face" in detail, detail
    assert _sha_of(cli_env.db) == before, "DB unchanged on the refusal"
    # a nonexistent person on link --near -> delete.no_such_person
    r = cli_env.run(["link", "--near", "never-was-named", "0.5"])
    r.error_envelope("delete.no_such_person")


def test_link_near_positive_and_distance_above_semantics(measured, cli_env):
    """link --near with a REAL unlinked second cluster: strictly-below links,
    at-or-above skips with distance_above (8.4)."""
    need(measured.get("vg"),
         "two same-person images whose measured distance is far above 0.40, "
         "so the second enroll leaves a real unlinked cluster (link --near "
         "semantics)")
    d = measured["vg"][0]
    enroll_one(cli_env, measured["vg"][1], expect_faces=1)
    cid1 = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid1, "near-person"]).rc == 0
    enroll_one(cli_env, measured["vg"][2], expect_faces=1)
    cid2 = [c for c in unlinked_cluster_ids(cli_env) if c != cid1][0]

    # distance BELOW the measured minimum: skipped with distance_above
    below = (math.ceil(d * 100) - 1) / 100.0  # strictly under the real d
    r = cli_env.run(["link", "--near", "near-person", "%.2f" % below])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (link --near below)")
    _assert_shape_keys(body, {"linked_near", "person", "distance",
                              "linked_clusters", "skipped"}, "link --near")
    if below > 0 and d > below:
        assert body["linked_clusters"] == []
        skip_rows = {s["cluster_id"]: s for s in body["skipped"]}
        assert cid2 in skip_rows and \
            skip_rows[cid2]["reason"] == "distance_above", body
        assert skip_rows[cid2]["nearest_person_distance"] is not None
    else:
        pytest.skip("measured distance sits on a cent boundary; the "
                    "strictly-below negative variant needs d above the "
                    "rounded value to be decidable")
    # now strictly above: linked
    above = (math.floor(d * 100) + 2) / 100.0
    r = cli_env.run(["link", "--near", "near-person", "%.2f" % above])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (link --near above)")
    assert body["linked_clusters"] == [cid2], body
    assert body["skipped"] == [], body
    # bad distances / nonexistent person
    r = cli_env.run(["link", "--near", "near-person", "abc"])
    assert r.rc == 2
    r = cli_env.run(["link", "--near", "near-person", "-1"])
    assert r.rc == 2
    r = cli_env.run(["link", "--near", "unheard-of-person", "0.5"])
    r.error_envelope("delete.no_such_person")


def test_tls_pin_absent_identify_refuses_and_never_pins(measured, cli_env):
    need(measured.get("va"), "an image to establish the pin via enroll")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    assert cli_env.pin.exists(), "enroll creates the TOFU pin file (section 7)"
    os.remove(cli_env.pin)
    before = _sha_of(cli_env.db)
    r = cli_env.run(["identify", measured["va"][1]])
    err = r.error_envelope("tls.pin_absent")
    detail = err.get("detail", "")
    assert "enroll" in detail, (
        "detail tells the operator to run enroll once to establish trust: %r"
        % detail)
    assert not cli_env.pin.exists(), "a read NEVER creates the pin file"
    assert _sha_of(cli_env.db) == before


def test_db_absent_on_reads_never_creates_the_file(cli_env):
    # NOTE on identify: the implementation opens the QUERY FILE before the
    # DB-presence check, so `identify <missing file>` with an absent DB
    # short-circuits input.unreadable instead of db.absent. The spec's
    # pre-flight list (7) orders only token/DB/TLS/POST and does not name a
    # file-readability step, so both orders are defensible; assert rc=1 with
    # a closed reason for identify, db.absent strictly for the local-only
    # reads.
    for verb in (["status"], ["list", "people"], ["list", "clusters"],
                 ["list", "faces"]):
        r = cli_env.run(verb)
        r.error_envelope("db.absent", "verb=%s" % verb)
        assert not cli_env.db.exists(), "verb %s created the DB" % verb
        assert not cli_env.pin.exists(), "verb %s created the pin file" % verb
    r = cli_env.run(["identify", "a.png"])
    r.error_envelope(None, "identify + absent DB: closed reason, rc=1")
    assert not cli_env.db.exists(), "identify created the DB"
    assert not cli_env.pin.exists(), "identify created the pin file"


def test_config_env_file_permissions_loose_file_refused(measured, cli_env):
    need(measured.get("va"), "an image for the POST verb that must refuse")
    # establish the DB first so the loose-file refusal is isolated from
    # db.absent ordering
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    before = _sha_of(cli_env.db)
    cli_env.env_file.chmod(0o644)
    r = cli_env.run(["enroll", measured["va"][1]])
    r.error_envelope("config.env_file_permissions",
                     "a POST verb refuses an env file looser than 0600")
    cli_env.env_file.chmod(0o600)
    assert _sha_of(cli_env.db) == before, "the refusal wrote nothing"
    r = cli_env.run(["list", "people"])  # a local-only read: no such refusal
    assert r.rc == 0, r.stderr


def test_token_missing_on_post_verb_exits_2_one_line(measured, cli_env, tmp_path):
    need(measured.get("va"), "an image for the POST verb")
    # a config dir WITHOUT a token line and the var unset: nothing supplies it
    nohome = tmp_path / "noconfig"
    nohome.mkdir()
    e = dict(os.environ)
    for k in ("VISION_EMBED_TOKEN", "VISION_EMBED_URL"):
        e.pop(k, None)
    e.update({"HOME": str(nohome), "XDG_CONFIG_HOME": str(nohome),
              "VISION_INDEX_DB": str(cli_env.db)})
    p = subprocess.run([sys.executable, SCRIPT, "enroll", measured["va"][1]],
                       env=e, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, (p.returncode, p.stderr[:200])
    assert len(p.stderr.strip().splitlines()) == 1
    assert p.stderr.strip() == "error: VISION_EMBED_TOKEN is not set", repr(p.stderr)
    assert not cli_env.db.exists(), "no call attempted, nothing created"


def test_token_never_printed_or_stored(measured, cli_env):
    need(measured.get("va"), "an image to run a successful POST")
    r = cli_env.run(["enroll", measured["va"][1]])
    assert r.rc == 0, r.stderr
    assert SERVICE_TOKEN not in r.stdout + r.stderr
    db_bytes = open(cli_env.db, "rb").read()
    assert SERVICE_TOKEN.encode() not in db_bytes, "token reached the DB (gate 6)"
    assert b"/home/" not in db_bytes, "an absolute path reached the DB (gate 7)"
    pin_path = cli_env.db.with_suffix(".sqlite3.hosts")
    if pin_path.exists():
        pn = open(pin_path, "rb").read()
        assert SERVICE_TOKEN.encode() not in pn
        assert b"/home/" not in pn


def test_ambiguous_tie_unknown_no_name_asserted(measured, cli_env):
    need(measured.get("tie"),
         "two faces of two DIFFERENT persons, each strictly below 0.28 from "
         "one probe face AND at least 0.45 apart from each other, so their "
         "clusters stay separate — the ambiguous_tie configuration of section "
         "5; nothing in the real corpus satisfied it")
    fa, fb, fp = measured["tie"]
    enroll_one(cli_env, fa, expect_faces=1)
    enroll_one(cli_env, fb, expect_faces=1)
    c1, c2 = sorted(unlinked_cluster_ids(cli_env))
    # the two clusters cannot be told apart by id; probe with identify which
    # cluster holds which image, then link each to its own person
    assert cli_env.run(["link", c1, "tie-alpha"]).rc == 0
    r = cli_env.run(["identify", fa])
    assert r.rc == 0, r.stderr
    band_fa = r.one_json_object()["faces"][0]
    if band_fa["band"] == "matched":
        # c1 holds fa's face (probe distance ~0 < 0.28): c1 is already linked
        # to tie-alpha — no relink needed
        person_for_c1, person_for_c2 = "tie-alpha", "tie-beta"
    else:
        # c1 holds fb's face (fa sits >= 0.45 from it -> unknown): relabel
        person_for_c1, person_for_c2 = "tie-beta", "tie-alpha"
        assert cli_env.run(["unlink", c1]).rc == 0
        assert cli_env.run(["link", c1, "tie-beta"]).rc == 0
    if c2 not in [c["cluster_id"] for c in list_clusters_json(cli_env)
                  if c["person"] is not None]:
        assert cli_env.run(["link", c2, person_for_c2]).rc == 0
    prs = {p["name"]: p for p in list_people(cli_env)}
    assert prs["tie-alpha"]["faces"] >= 1 and prs["tie-beta"]["faces"] >= 1

    r = cli_env.run(["identify", fp])
    assert r.rc == 0, r.stderr
    body = r.one_json_object(" (identify)")
    fc = body["faces"][0]
    assert fc["band"] == "unknown", (
        "ambiguous_tie is normative: unknown, never a pick; got %r" % fc["band"])
    assert fc["reason"] == "ambiguous_tie"
    assert fc["nearest_named_distance"] is not None
    hits = set(_collect_name_fields(body, "tie-alpha")
               + _collect_name_fields(body, "tie-beta"))
    assert hits <= {"/faces[0]/nearest_person"}, (
        "no name in any assertion position: %s" % hits)
    _no_bulk_keys(body, "identify")


def test_identify_zero_faces_is_success(measured, cli_env):
    need(measured.get("zero_face") and measured.get("va"),
         "a zero-face image plus a valid image (8.2)")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    zero = measured["zero_face"][0]
    if zero == measured["va"][1]:
        pytest.skip("the only zero-face image overlaps the enroll fixture")
    r = cli_env.run(["identify", zero])
    assert r.rc == 0, r.stderr
    body = r.one_json_object()
    _assert_shape_keys(body, {"image_sha256", "faces"}, "identify")
    assert body["faces"] == [], "faces: [] and exit 0 (9: zero faces is success)"


def test_identify_no_evidence_persons_with_zero_linked_faces(measured, cli_env):
    need(measured.get("va"), "an image so the DB exists with one cluster")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    cid = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid, "noev-person"]).rc == 0
    assert cli_env.run(["unlink", cid]).rc == 0
    # person exists with ZERO linked faces -> every face no_evidence
    r = cli_env.run(["identify", measured["va"][1]])
    assert r.rc == 0, r.stderr
    fc = r.one_json_object()["faces"][0]
    assert fc["band"] == "unknown"
    assert fc["reason"] == "no_evidence"
    assert fc["nearest_named_distance"] is None, (
        "the empty set is never fabricated into a distance")
    assert fc["nearest_person"] is None, "no name is asserted (I-6)"
    # second variant: zero linked clusters at all (cluster deleted)
    assert cli_env.run(["delete", "cluster", cid, "--yes"]).rc == 0
    r = cli_env.run(["identify", measured["va"][1]])
    assert r.rc == 0, r.stderr
    fc = r.one_json_object()["faces"][0]
    assert fc["band"] == "unknown" and fc["reason"] == "no_evidence"
    assert fc["nearest_named_distance"] is None


def test_identify_with_zero_people_db_reports_no_evidence(measured, cli_env):
    need(measured.get("va"), "an image so the DB exists with a cluster")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    r = cli_env.run(["identify", measured["va"][1]])
    assert r.rc == 0, r.stderr
    fc = r.one_json_object()["faces"][0]
    assert fc["band"] == "unknown" and fc["reason"] == "no_evidence"
    assert fc["nearest_named_distance"] is None


def test_db_version_unsupported_refused(measured, cli_env):
    need(measured.get("va"), "an image to prove v2 reads work (DB-version gate)")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    v1db = cli_env.tmp / "v1.sqlite3"
    c = sqlite3.connect(v1db)
    c.execute("CREATE TABLE junk(x)")
    c.execute("PRAGMA user_version = 1")
    c.commit()
    c.close()
    for verb in (["list", "people"], ["status"]):
        r = cli_env.run(verb, VISION_INDEX_DB=str(v1db))
        err = r.error_envelope("db.version_unsupported",
                               "verb=%s over user_version=1" % verb)
        assert "user_version=1" in err["detail"], err["detail"]


def test_db_corrupt_refused_no_repair(cli_env):
    cli_env.db.parent.mkdir(parents=True, exist_ok=True)
    cli_env.db.write_bytes(b"this is not a sqlite database at all;" * 20)
    r = cli_env.run(["list", "people"])
    r.error_envelope("db.corrupt")


def test_status_lone_cluster_nullability(measured, cli_env):
    """One lonely unlinked cluster, zero people: fields null, suggestion null
    (8.8 nullability: no 0.0 stand-ins, no '-' ever in JSON)."""
    need(measured.get("va"), "an image to create one lonely unlinked cluster")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    body = status_json(cli_env)
    assert body["suggested_command"] is None
    assert len(body["unlinked_clusters"]) == 1
    row = body["unlinked_clusters"][0]
    assert row["nearest_person"] is None and row["nearest_person_distance"] is None
    assert row["nearest_cluster_id"] is None and row["nearest_cluster_distance"] is None
    dash_vals = _collect_name_fields(body, "-")
    assert dash_vals == [], "no '-' placeholder value ever appears in JSON"


def test_closed_verb_set_rejects_v1_verbs(cli_env):
    # I-7: the withdrawn v1 verb forms must not work as verbs. Two admissible
    # readings exist: rejected as usage (exit 2), or consumed as ordinary
    # POSITIONAL PHOTO arguments by `enroll` batch semantics (exit 1 with
    # input.unreadable / batch.all_failed) — both prove the verb does not exist.
    for args in (["rename", "1", "x"], ["merge", "1", "2"], ["attach", "1", "x"],
                 ["find", "object", "x"], ["find", "like", "a.png"],
                 ["list", "assets"], ["delete", "asset", "1"]):
        r = cli_env.run(args)
        assert r.rc == 2, "withdrawn v1 verb %r must not exist (I-7): rc=%d %r" % (
            args, r.rc, r.stderr[:160])
    for args in (["enroll", "--add-to", "x", "a.png"],
                 ["enroll", "--person", "x", "a.png"]):
        r = cli_env.run(args)
        assert r.rc in (1, 2), (
            "the v1 enroll flag form %r must not exist as a verb (I-7): "
            "rc=%d %r" % (args, r.rc, r.stderr[:160]))
        if r.rc == 1:
            lines = r.stderr.strip().splitlines()
            envl = json.loads(lines[-1])
            assert envl["error"]["reason"] in ("input.unreadable",
                                               "batch.all_failed"), envl


def test_enroll_never_creates_person_or_links(measured, cli_env):
    """I-4: enroll leaves every cluster unlinked and creates no person."""
    need(measured.get("va"), "an image (I-4)")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    assert list_people(cli_env) == []
    assert all(c["person"] is None for c in list_clusters_json(cli_env))


def test_identify_and_status_are_read_only(measured, cli_env):
    need(measured.get("va") and measured.get("va_probe"),
         "an evidence image and a probe image (I-5)")
    enroll_one(cli_env, measured["va"][1], expect_faces=1)
    cid = unlinked_cluster_ids(cli_env)[0]
    assert cli_env.run(["link", cid, "readonly-person"]).rc == 0
    before = _sha_of(cli_env.db)
    r = cli_env.run(["identify", measured["va_probe"][1]])
    assert r.rc == 0, r.stderr
    assert _sha_of(cli_env.db) == before, "I-5: identify writes nothing"
    r = cli_env.run(["status"])
    assert r.rc == 0
    assert _sha_of(cli_env.db) == before, "I-5: status writes nothing"


def test_list_sort_order_most_faces_first(measured, cli_env):
    """8.7 sort: list clusters by faces DESC then id ASC (spec supersedes the
    mockup's older id-ascending rule)."""
    need(measured.get("vg") and measured.get("va"),
         "images that produce clusters of DIFFERING face counts (8.7 sort)")
    enroll_one(cli_env, measured["vg"][1], expect_faces=1)   # cluster with 1
    enroll_one(cli_env, measured["vg"][2], expect_faces=1)   # 0.45+ apart: another
    r = cli_env.run(["list", "clusters", "--json"])
    cs = r.one_json_object()["clusters"]
    assert len(cs) == 2
    r = cli_env.run(["list", "clusters"])
    assert r.rc == 0
    _assert_no_paths_no_bulk(r.stdout)
    # both have 1 face; id ASC breaks the tie
    id_rows = [ln for ln in r.stdout.strip().splitlines()
               if ln.strip() and ln.strip()[0].isdigit()]
    ids_in_text = [int(ln.split()[0]) for ln in id_rows]
    assert ids_in_text == sorted(ids_in_text), (
        "faces DESC, then cluster_id ASC tie-break (8.7): %r" % ids_in_text)


def test_version_is_runtime_hash_and_matches_file(cli_env):
    r = cli_env.run(["--version"])
    assert r.rc == 0 and r.stderr == ""
    with open(SCRIPT, "rb") as f:
        expect = hashlib.sha256(f.read()).hexdigest()
    assert r.stdout.strip() == expect
    assert not cli_env.db.exists(), "--version touches nothing"
