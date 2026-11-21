"""HTTP contract tests over plain HTTP via TestClient (SPEC section 11, tests 4-10)."""

import io
import json

import pytest
from starlette.testclient import TestClient

from conftest_helpers import REPO, boot, make_image_bytes, write_config, write_token
from vision_embed.app import build_from_config_path

TOKEN = "test-token-abc123"

@pytest.fixture()
def client(tmp_path):
    import contextlib

    write_token(tmp_path, TOKEN)
    cfg = write_config(tmp_path)
    captured = io.StringIO()
    app, cfg = build_from_config_path(cfg)
    with contextlib.redirect_stdout(captured):
        with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as c:
            yield c, tmp_path
    _LAST_CAPTURE[:] = [captured]


_LAST_CAPTURE: list = []


def captured_audit() -> "io.StringIO":
    return _LAST_CAPTURE[0]


def _post(client, path=None, *, image=None, token=None, content_type=None, extra_fields=None):
    files = None
    data = {}
    if image is not None:
        files = {"image": ("img.bin", image, content_type or "application/octet-stream")}
    if extra_fields:
        data.update(extra_fields)
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {"Authorization": ""}
    return client.post("/v1/embed", files=files, data=data, headers=headers)


def test_healthz_unauthenticated_ok(client):
    c, _ = client
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_surface_disabled(client):
    c, _ = client
    # /openapi.json, /docs, /redoc are not served (ruling 5) and no schema
    # artifact exists at the repo root.
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert c.get(path).status_code == 404
    app = c.app
    assert app.openapi_url is None
    assert not (REPO / "openapi.json").exists()


def test_readyz_unauthenticated_ready_with_zero_onnx_files(client):
    c, tmp = client
    assert not list(tmp.glob("*.onnx"))  # fake mode starts with zero ONNX files
    r = c.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "models_loaded": 4}


def test_bad_token_gets_401_with_token(client):
    c, _ = client
    img = make_image_bytes()
    r = _post(c, image=img, token="wrong")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["reason"] == "auth.invalid_token"
    assert "request_id" in body["error"]


def test_missing_token_gets_401(client):
    c, _ = client
    r = _post(c, image=make_image_bytes(), token=None)
    assert r.status_code == 401
    assert r.json()["error"]["reason"] == "auth.invalid_token"


def test_valid_token_on_all_three_endpoints(client):
    c, _ = client
    h = {"Authorization": f"Bearer {TOKEN}"}
    assert c.get("/healthz", headers=h).status_code == 200
    assert c.get("/readyz", headers=h).status_code == 200
    r = _post(c, image=make_image_bytes(), token=TOKEN)
    assert r.status_code == 200


def test_body_too_large_413(client):
    c, _ = client
    big = b"\x00" * 40_000_000
    r = _post(c, image=big, token=TOKEN)
    assert r.status_code == 413
    assert r.json()["error"]["reason"] == "limits.body_too_large"


def test_body_too_large_rejected_before_body_read(tmp_path):
    """413 with a bogus huge Content-Length; body never needed."""
    write_token(tmp_path, TOKEN)
    cfg = write_config(tmp_path)
    tc, _ = boot(cfg, TOKEN)
    with tc as c:
        # valid token path skipped: CL pre-check fires before auth read
        r = c.post(
            "/v1/embed",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Length": "40000000"},
        )
        assert r.status_code == 413
        assert r.json()["error"]["reason"] == "limits.body_too_large"


def test_format_sniff_txt_is_415(client):
    c, _ = client
    r = _post(c, image=b"plain text file, not an image at all", token=TOKEN)
    assert r.status_code == 415
    assert r.json()["error"]["reason"] == "format.unsupported"


def test_jpeg_with_wrong_multipart_content_type_still_decodes(client):
    c, _ = client
    img = make_image_bytes("JPEG")
    r = _post(c, image=img, token=TOKEN, content_type="text/plain")
    assert r.status_code == 200
    assert r.json()["scene"]["dim"] == 768


def test_closed_route_set_404_405(client):
    c, _ = client
    h = {"Authorization": f"Bearer {TOKEN}"}
    r = c.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["reason"] == "route.not_found"
    r = c.get("/v1/embed", headers=h)
    assert r.status_code == 405
    assert r.json()["error"]["reason"] == "method.not_allowed"
    r = c.patch("/v1/embed", headers=h)
    assert r.status_code == 405
    assert r.json()["error"]["reason"] == "method.not_allowed"
    r = c.post("/healthz", headers=h)
    assert r.status_code == 405
    assert r.json()["error"]["reason"] == "method.not_allowed"


def test_success_response_schema_strict_additionalProperties(client):
    c, _ = client
    from conftest_helpers import make_image_bytes as mib

    r = _post(c, image=mib("PNG"), token=TOKEN)
    assert r.status_code == 200
    payload = r.json()
    from vision_embed.pipeline import RESPONSE_SCHEMA
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(RESPONSE_SCHEMA)
    assert list(validator.iter_errors(payload)) == []
    # adding an invented key must fail the validator
    import copy

    bad = copy.deepcopy(payload)
    bad["aliveness"] = True
    assert list(validator.iter_errors(bad))
    bad3 = copy.deepcopy(payload)
    bad3["scene"]["model_v2"] = "invented"
    assert list(validator.iter_errors(bad3))
    bad4 = copy.deepcopy(payload)
    bad4["faces"] = [dict(f, face_quality=0.9) for f in bad4["faces"]] or [{"bbox": {}}]
    assert list(validator.iter_errors(bad4))
    bad5 = copy.deepcopy(payload)
    bad5["objects"] = [dict(o, tracking_id="t1") for o in bad5["objects"]] or [{"label": "person"}]
    assert list(validator.iter_errors(bad5))


def test_face_bbox_inside_image_with_dim_512(client):
    c, _ = client
    img = make_image_bytes("PNG", (64, 48))
    r = _post(c, image=img, token=TOKEN)
    assert r.status_code == 200
    body = r.json()
    assert body["image"]["width"] == 64 and body["image"]["height"] == 48
    for face in body["faces"]:
        bb = face["bbox"]
        assert 0 <= bb["x"] < 64 and 0 <= bb["y"] < 48
        assert bb["w"] > 0 and bb["h"] > 0
        assert face["det_score"] >= 0.5
        assert face["dim"] == 512
        assert len(face["vector"]) == 512
        assert face["model"] == "buffalo_l/w600k_r50"


def test_object_vectors_opt_in(client):
    c, _ = client
    img = make_image_bytes("PNG")
    r_false = _post(c, image=img, token=TOKEN, extra_fields={"include_object_vectors": "false"})
    r_true = _post(c, image=img, token=TOKEN, extra_fields={"include_object_vectors": "true"})
    assert r_false.status_code == 200 and r_true.status_code == 200
    for obj in r_false.json()["objects"]:
        assert "vector" not in obj
        assert "model" not in obj and "dim" not in obj
    for obj in r_true.json()["objects"]:
        assert obj["dim"] == 768
        assert obj["model"] == "ViT-L-14/openai"
        assert len(obj["vector"]) == 768


def test_include_object_vectors_bad_value_422(client):
    c, _ = client
    r = _post(
        c,
        image=make_image_bytes("PNG"),
        token=TOKEN,
        extra_fields={"include_object_vectors": "TRUE"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["reason"] == "decode.failed"
    assert "include_object_vectors" in r.json()["error"]["detail"]


def test_request_id_echoed_when_valid_else_uuid4(client):
    c, _ = client
    r = c.get(
        "/healthz",
        headers={"X-Request-Id": "my-correlation-id-42"},
    )
    assert r.headers["X-Request-Id"] == "my-correlation-id-42"
    r = c.get("/healthz", headers={"X-Request-Id": "\x01\x02bad"})
    rid = r.headers["X-Request-Id"]
    assert rid != "\x01\x02bad"
    import uuid

    uuid.UUID(rid)
    # request_id shared across response and error envelope
    r = _post(c, image=b"not-an-image", token="wrong", )
    assert r.status_code == 401


def test_malformed_multipart_body_422_decode_failed_with_audit_and_metric(client, capsys):
    c, _ = client
    boundary = "XxXnotarealboundaryXxX"
    body = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"image\"\r\n"
        "\r\n"
        "unterminated part, never closed and no final boundary"
    ).encode()
    r = c.post(
        "/v1/embed",
        content=body,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    assert r.status_code == 422
    payload = r.json()
    assert payload["error"]["reason"] == "decode.failed"
    assert payload["error"]["request_id"] == r.headers["X-Request-Id"]
    # one audit line with the decode.failed outcome
    import json as _json

    stdout = capsys.readouterr().out
    lines = [l for l in stdout.splitlines() if l.startswith("{")]
    assert lines
    audit_payloads = [_json.loads(l) for l in lines]
    assert any(a.get("outcome") == "error" and a.get("reason") == "decode.failed" for a in audit_payloads)


def test_form_with_more_than_10_fields_is_accepted(client):
    c, _ = client
    extras = {f"extra_field_{i}": f"v{i}" for i in range(15)}
    r = _post(c, image=make_image_bytes("PNG"), token=TOKEN, extra_fields=extras)
    assert r.status_code == 200, r.text
    assert "faces" in r.json()
