"""Bare-form summary (spec 8.10, weak-detection) tests, 2026-11-21.

NEW file — the independent black-box suite tests/test_vision_index_cli.py is
NOT modified. Covered, for the bare-file summary line:

- a weak (below FACE_DET_MIN) face NEVER contributes its name, so a photo
  with one trusted face plus one weak edge-sliver of the same person prints
  the name ONCE (the reproduced operator bug);
- a file whose every face is weak prints `unknown` and, alone, exits 1;
- one trusted matched face plus one weak face still exits 0.

The declined 8.2 JSON-boolean mark is not tested here: identify JSON
keeps its frozen six-key face shape.

Fixtures (run_cli, env, service, photo_file, enroll_persons) are imported from
tests/test_vision_index_bare_glob.py exactly as
tests/test_vision_index_v4_dedup_incidental.py does. The shared fake service
hardcodes det_score = 0.9, so this file installs a flexible do_POST on the SAME
Handler class (used by the SAME service fixture) that passes full face dicts
through verbatim — giving per-face det_score control without touching the
shared module.
"""

import hashlib
import json
import os

import pytest

import test_vision_index_bare_glob as bg
from test_vision_index_bare_glob import (  # noqa: F401  (fixtures imported)
    E1, E2, ENV, clean_env, env, enroll_persons, photo_file, run_cli,  # noqa
    service)  # noqa: F401

WEAK_DET = 0.59      # the measured sliver det (20230927_133717 face 1)
GOOD_DET = 0.9       # above FACE_DET_MIN (0.65): trusted

@pytest.fixture()
def flexible_service(service):
    """Same fake service, but faces_for() may return FULL face dicts
    (arbitrary det_score) alongside the shared (vec, bbox) tuple form."""
    _orig = bg.Handler.do_POST

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        image = body.split(bg.MARKER, 1)[1] if bg.MARKER in body else b""
        end = image.find(b"\r\n--visionindex7d1a2cboundary")
        if end != -1:
            image = image[:end]
        key = hashlib.sha256(image).hexdigest()[:12]
        svc = type(self).service
        svc.requests.append(key)
        faces = []
        for entry in svc.faces_for(key):
            if isinstance(entry, tuple) and isinstance(entry[0], dict):
                entry = entry[0]   # photo_file wraps bare entries as (e, None)
            if isinstance(entry, dict):
                faces.append(entry)
            else:
                vec, bbox = entry
                faces.append({
                    "bbox": (bbox if bbox is not None
                             else {"x": 0, "y": 0, "w": 10, "h": 10}),
                    "det_score": GOOD_DET,
                    "dim": bg.DIM,
                    "vector": vec,
                })
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

    bg.Handler.do_POST = do_POST
    yield service
    bg.Handler.do_POST = _orig     # restore the shared module's handler


# --------------------------------------------------------------------------
# bare-form summary (spec 8.10) — LIVE
# --------------------------------------------------------------------------

def test_bare_summary_name_printed_once_amid_weak_dup(env, flexible_service,
                                                      tmp_path):
    """The reproduced bug: trusted face + weak edge-sliver of the SAME person
    printed the name twice. Now: the name prints ONCE."""
    enroll_persons(env, flexible_service, str(tmp_path))
    f = photo_file(env, flexible_service, str(tmp_path), "dup.png", [
        {"bbox": {"x": 100, "y": 100, "w": 591, "h": 775},
         "det_score": GOOD_DET, "dim": bg.DIM, "vector": list(E1)},
        {"bbox": {"x": 1000, "y": 0, "w": 15, "h": 115},
         "det_score": WEAK_DET, "dim": bg.DIM, "vector": list(E1)},
    ])
    p = run_cli(env, [f])
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == ["%s: alice" % f], p.stdout


def test_bare_all_weak_file_prints_unknown_exit1(env, flexible_service, tmp_path):
    """Every face below the floor: the line is `<file>: unknown` (names
    suppressed, and the detections are not trustworthy enough for
    `no faces`); alone in the run it exits 1."""
    enroll_persons(env, flexible_service, str(tmp_path))
    f = photo_file(env, flexible_service, str(tmp_path), "allweak.png", [
        {"bbox": {"x": 1000, "y": 0, "w": 15, "h": 115},
         "det_score": WEAK_DET, "dim": bg.DIM, "vector": list(E1)},
    ])
    p = run_cli(env, [f])
    assert p.returncode == 1, (p.returncode, p.stdout, p.stderr)
    assert p.stdout.splitlines() == ["%s: unknown" % f], p.stdout


def test_bare_trusted_plus_weak_still_exit0(env, flexible_service, tmp_path):
    """One trusted matched face plus one weak face still exits 0."""
    enroll_persons(env, flexible_service, str(tmp_path))
    f = photo_file(env, flexible_service, str(tmp_path), "okplus.png", [
        {"bbox": {"x": 100, "y": 100, "w": 591, "h": 775},
         "det_score": GOOD_DET, "dim": bg.DIM, "vector": list(E1)},
        {"bbox": {"x": 1000, "y": 0, "w": 15, "h": 115},
         "det_score": WEAK_DET, "dim": bg.DIM, "vector": list(E1)},
    ])
    p = run_cli(env, [f])
    assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
    assert p.stdout.splitlines() == ["%s: alice" % f], p.stdout
