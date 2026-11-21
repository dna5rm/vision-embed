"""Audit, metrics, reason closed-set, no-identity, and statelessness tests (SPEC section 11, tests 11-15)."""

import contextlib
import io
import json
import re
import subprocess
import time
import urllib.request

import pytest

from vision_embed.app import build_from_config_path
from conftest_helpers import (
    METRICS_PORT,
    REPO,
    PYTHON,
    make_image_bytes,
    write_config,
    write_token,
)

TOKEN = "test-token-abc123"
EXPECTED_AUDIT_KEYS = {"ts", "request_id", "sha256", "bytes", "faces", "objects", "outcome", "reason"}
EXPECTED_TOKENS = {
    "limits.body_too_large",
    "format.unsupported",
    "decode.failed",
    "task.unsupported",
    "route.disabled",
    "auth.invalid_token",
    "inference.failed",
    "response.validation_failed",
    "route.not_found",
    "method.not_allowed",
    "not_ready",
    "gpu.busy",
}


@pytest.fixture()
def running(tmp_path, capsys):
    write_token(tmp_path, TOKEN)
    cfg = write_config(tmp_path)
    app, _cfg = build_from_config_path(cfg)
    with boot_client(app) as c:
        yield c, (lambda: capsys.readouterr().out), tmp_path


def boot_client(app):
    from starlette.testclient import TestClient

    return TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})


def post_image(c, image=None, token=TOKEN):
    image = image if image is not None else make_image_bytes("PNG")
    return c.post(
        "/v1/embed",
        files={"image": ("img.png", image, "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )


def audit_lines(get_out) -> list[dict]:
    return [json.loads(ln) for ln in get_out().splitlines() if ln.startswith("{") and '"ts"' in ln]


def test_audit_line_shape_per_post(running):
    c, get_out, _ = running
    for _ in range(3):
        assert post_image(c).status_code == 200
    lines = audit_lines(get_out)
    assert len(lines) == 3
    for line in lines:
        assert line["outcome"] == "ok"
        assert line["reason"] == "ok"
        assert set(line.keys()) == EXPECTED_AUDIT_KEYS
        assert re.fullmatch(r"[0-9a-f]{64}", line["sha256"])


def test_audit_per_http_request_including_healthz_readyz_and_405(running):
    """Ruling 2: one JSON line per HTTP request — healthz/readyz/404/405 included."""
    c, get_out, _ = running
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").status_code == 200
    assert c.get("/nope").status_code == 404
    assert c.get("/v1/embed").status_code == 405
    img = make_image_bytes("PNG")
    assert post_image(c, image=img).status_code == 200
    lines = audit_lines(get_out)
    assert len(lines) == 5
    for line in lines:
        assert set(line.keys()) == EXPECTED_AUDIT_KEYS
    # image-less lines carry sha256 "" and bytes 0 (rulings 2 & 3)
    for line in lines[:4]:
        assert line["sha256"] == "" and line["bytes"] == 0
        assert line["faces"] == 0 and line["objects"] == 0
    assert lines[0]["outcome"] == "ok" and lines[0]["reason"] == "ok"
    assert lines[1]["outcome"] == "ok" and lines[1]["reason"] == "ok"
    assert lines[2]["outcome"] == "error" and lines[2]["reason"] == "route.not_found"
    assert lines[3]["outcome"] == "error" and lines[3]["reason"] == "method.not_allowed"
    assert lines[4]["outcome"] == "ok"
    import hashlib

    assert lines[4]["sha256"] == hashlib.sha256(img).hexdigest()
    assert lines[4]["bytes"] == len(img)

    # early-failure lines keep the eight-key shape with sha256 "" / bytes 0 (ruling 3)
    get_out()  # clear buffer
    post_image(c, image=make_image_bytes("PNG"), token="bad")  # 401
    lines = audit_lines(get_out)
    assert len(lines) == 1
    assert set(lines[0].keys()) == EXPECTED_AUDIT_KEYS
    assert lines[0]["sha256"] == "" and lines[0]["bytes"] == 0
    assert lines[0]["outcome"] == "error" and lines[0]["reason"] == "auth.invalid_token"


def test_readyz_not_audits_when_not_ready(tmp_path, capsys):
    from vision_embed import metrics as metrics_mod

    write_token(tmp_path, TOKEN)
    cfg = write_config(tmp_path)
    app, _cfg = build_from_config_path(cfg)
    # force the not-ready state (the gauge is process-global and set by
    # other tests' lifespans), then restore it
    models_loaded_was = metrics_mod.models_loaded._value.get()
    metrics_mod.models_loaded._value.set(0)
    try:
        c = boot_client(app)
        r = c.get("/readyz")
        assert r.status_code == 503
    finally:
        metrics_mod.models_loaded._value.set(models_loaded_was)
    out = capsys.readouterr().out
    lines = [json.loads(ln) for ln in out.splitlines() if ln.startswith("{") and '"ts"' in ln]
    assert len(lines) == 1
    assert lines[0]["reason"] == "not_ready"
    assert lines[0]["sha256"] == "" and lines[0]["bytes"] == 0


def test_audit_error_outcomes(running):
    c, get_out, _ = running
    post_image(c, image=b"garbage-not-an-image", token=TOKEN)  # 415
    post_image(c, image=make_image_bytes("PNG"), token="bad")  # 401
    post_image(c)  # 200
    lines = audit_lines(get_out)
    assert len(lines) == 3
    assert lines[0]["outcome"] == "error" and lines[0]["reason"] == "format.unsupported"
    assert lines[1]["outcome"] == "error" and lines[1]["reason"] == "auth.invalid_token"
    assert lines[2]["outcome"] == "ok" and lines[2]["reason"] == "ok"


def test_stdout_never_leaks_token_or_hex_hashes_beyond_sha256(running):
    c, get_out, _ = running
    post_image(c)
    text = get_out()
    assert "Bearer" not in text
    assert TOKEN not in text
    # every 40+-char hex run must be exactly the sha256 of a logged audit line
    for run in re.findall(r"[0-9a-f]{40,}", text):
        assert len(run) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", run)


def test_metrics_closed_series_set(running):
    c, _, _ = running
    post_image(c)
    post_image(c, image=b"nope", token=TOKEN)
    with urllib.request.urlopen(f"http://127.0.0.1:{METRICS_PORT}/metrics", timeout=5) as resp:
        body = resp.read().decode()
    def base_name(ln: str) -> str:
        name = ln.split("{")[0].split(" ")[0]
        for suffix in ("_created", "_bucket", "_sum", "_count", "_total"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        canonical = {
            "vision_embed_requests",
            "vision_embed_latency_seconds",
            "vision_embed_errors",
            "vision_embed_models_loaded",
        }
        if name not in canonical:
            return ""
        return {"vision_embed_requests": "vision_embed_requests_total", "vision_embed_errors": "vision_embed_errors_total"}.get(name, name)
    names = {n for n in (base_name(ln) for ln in body.splitlines() if ln.startswith("vision_embed_")) if n}
    assert names == {
        "vision_embed_requests_total",
        "vision_embed_latency_seconds",
        "vision_embed_errors_total",
        "vision_embed_models_loaded",
    }
    # no 64-char hex or vector material in labels
    hexish = re.findall(r"[0-9a-f]{64,}", body)
    assert not hexish
    for line in body.splitlines():
        if line.startswith("vision_embed_"):
            assert "Bearer" not in line and "vector" not in line


def test_reason_token_set_is_closed_in_code(running):
    from vision_embed.errors import REASON_TOKENS

    assert REASON_TOKENS == EXPECTED_TOKENS
    # every reason= literal in the codebase must be one of the twelve (plus startup tokens)
    tokens_in_code = set()
    for py in (REPO / "vision_embed").rglob("*.py"):
        for m in re.finditer(r'reason\s*=\s*["\']([a-z_.]+)["\']', py.read_text()):
            tokens_in_code.add(m.group(1))
    allowed = EXPECTED_TOKENS | {"config.invalid", "config.unknown_key", "ok"}
    assert tokens_in_code <= allowed, tokens_in_code - allowed


def test_no_identity_fields_in_response_or_construction_path(running):
    c, _, _ = running
    r = post_image(c)
    assert r.status_code == 200
    body_text = json.dumps(r.json())
    assert "cluster" not in body_text

    # grep the response-construction path for identity/cluster vocabulary
    for py in (REPO / "vision_embed").glob("*.py"):
        src = py.read_text()
        if "cluster" in src.lower():
            pytest.fail(f"identity vocabulary leaked into {py.name}")


def test_statelessness_same_request_same_vectors(running):
    c, _, _ = running
    img = make_image_bytes("PNG")
    r1 = post_image(c, image=img)
    r2 = post_image(c, image=img)
    b1, b2 = r1.json(), r2.json()
    assert b1["scene"]["vector"] == b2["scene"]["vector"]
    for f1, f2 in zip(b1["faces"], b2["faces"]):
        for a, b in zip(f1["vector"], f2["vector"]):
            assert abs(a - b) <= 2e-5


def test_statelessness_across_process_restart(tmp_path):
    import socket
    import urllib.request

    http_port = 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        http_port = s.getsockname()[1]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        metrics_port = s.getsockname()[1]
    write_token(tmp_path, TOKEN)
    cfg = write_config(tmp_path, port=http_port, metrics_port=metrics_port)
    img = make_image_bytes("PNG")
    vectors = []
    base = f"http://127.0.0.1:{http_port}"
    for _ in range(2):
        proc = subprocess.Popen(
            [PYTHON, "-m", "vision_embed", "--config", str(cfg)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO),
        )
        try:
            deadline = time.time() + 15
            ready = False
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"{base}/readyz", timeout=1) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(0.2)
            assert ready
            form = (
                ("--boundary123\r\n"
                 "Content-Disposition: form-data; name=\"image\"; filename=\"img.png\"\r\n"
                 "Content-Type: image/png\r\n\r\n").encode()
                + img
                + b"\r\n--boundary123\r\n"
                + b"Content-Disposition: form-data; name=\"include_object_vectors\"\r\n\r\nfalse\r\n"
                + b"--boundary123--\r\n"
            )
            req = urllib.request.Request(
                f"{base}/v1/embed",
                data=form,
                headers={
                    "Content-Type": "multipart/form-data; boundary=boundary123",
                    "Authorization": f"Bearer {TOKEN}",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            vectors.append(data["scene"]["vector"])
        finally:
            proc.kill()
            proc.wait(timeout=10)
    assert len(vectors) == 2
    for a, b in zip(vectors[0], vectors[1]):
        assert abs(a - b) <= 2e-5
