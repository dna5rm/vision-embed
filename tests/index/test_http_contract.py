"""Tests 4, 6, 7 — auth, body-limit order, closed routes (SPEC section 13)."""

from __future__ import annotations

import io

import pytest
from starlette.testclient import TestClient

from tests.index import helpers_index as H
from tests.index.helpers_index import build_client, ingest


@pytest.fixture()
def client(tmp_path, capsys):
    import hashlib

    a = H.make_image_bytes()
    fake = H.make_fake(None, persons={H.sha_of(a): "alice"},
                       faces_for={H.sha_of(a): 1}, labels_for={H.sha_of(a): ["car"]},
                       face_distances={H.sha_of(a): [0.0]})
    with build_client(tmp_path, fake, capsys=capsys) as c:
        yield c, fake, a


def test_healthz_and_readyz_unauthenticated(client, tmp_path):
    c, fake, _ = client
    r = TestClient(c.app).get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    r = TestClient(c.app).get("/readyz")
    assert r.status_code == 200 and r.json() == {
        "status": "ready", "sqlite_ok": True, "embed_ready": True,
    }


def test_bad_token_401_on_every_v1_route(client):
    c, _, a = client
    anon = TestClient(c.app)
    assert anon.post("/v1/ingest", files={"image": ("x.png", a, "image/png")}).status_code == 401
    assert anon.post("/v1/identify", files={"image": ("x.png", a, "image/png")}).status_code == 401
    assert anon.get("/v1/assets/1").status_code == 401
    assert anon.get("/v1/search/person?name=alice").status_code == 401
    assert anon.get("/v1/search/objects?label=car").status_code == 401
    assert anon.post("/v1/name", json={"cluster_id": 1, "name": "x"}).status_code == 401
    assert anon.post("/v1/merge", json={"from_id": 1, "into_id": 2}).status_code == 401
    assert anon.get("/v1/clusters?kind=face").status_code == 401
    for r in [anon.post("/v1/ingest", files={"image": ("x.png", a, "image/png")})]:
        body = r.json()
        assert body["error"]["reason"] == "auth.invalid_token"
        assert set(body["error"]) == {"reason", "detail", "request_id"}


def test_constant_time_compare_used():
    import inspect

    from vision_index.config import constant_time_token_eq

    assert constant_time_token_eq("a", "a") is True
    assert constant_time_token_eq("a", "b") is False
    # enforcement site: the primitive is hmac.compare_digest
    assert "compare_digest" in inspect.getsource(constant_time_token_eq)


UNLIMITED = 32 * 1024 * 1024


def test_content_length_over_limit_wins_over_auth(client):
    c, _, _ = client
    anon = TestClient(c.app)
    huge = b"0" * (UNLIMITED + 1)
    r = anon.post(
        "/v1/ingest",
        content=huge,
        headers={"Authorization": "Bearer wrong",
                 "Content-Type": "multipart/form-data; boundary=x",
                 "Content-Length": str(len(huge))},
    )
    assert r.status_code == 413
    assert r.json()["error"]["reason"] == "limits.body_too_large"


def test_no_content_length_unauthenticated_401_body_not_read(client):
    c, fake, a = client
    anon = TestClient(c.app)
    calls_before = list(fake.calls)
    r = anon.post("/v1/ingest", files={"image": ("x.png", a, "image/png")})
    assert r.status_code == 401
    assert r.json()["error"]["reason"] == "auth.invalid_token"
    assert list(fake.calls) == calls_before  # body never reached the embed path


def test_stream_exceeds_limit_while_reading_413(client, use_no_content_length_headers=None):
    c, _, _ = client

    # chunked/streamed body larger than the cap with Content-Length unset
    from starlette.testclient import TestClient as TC

    def stream_bytes():
        for _ in range(33):
            yield b"x" * (1024 * 1024)

    r = c.post(
        "/v1/ingest",
        content=stream_bytes(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["reason"] == "limits.body_too_large"


def test_txt_sniffs_415(client, capsys):
    c, _, _ = client
    r = H.ingest(c, H.make_text_bytes())
    assert r.status_code == 415
    assert r.json()["error"]["reason"] == "format.unsupported"


def test_wrong_multipart_content_type_on_real_jpeg_ok(client, tmp_path):
    c, fake, a = client
    jpeg = H.make_image_bytes(fmt="JPEG")
    r = c.post(
        "/v1/ingest",
        files={"image": ("img.bin", jpeg, "application/octet-stream")},
        data={"path": "x"},
    )
    assert r.status_code == 200, r.json()
    assert r.json()["image"]["sha256"] == H.sha_of(jpeg)


def test_path_too_long_422(client):
    c, _, a = client
    r = H.ingest(c, a, path="x" * 65)
    assert r.status_code == 422
    assert r.json()["error"]["reason"] == "path.too_long"


def test_non_image_with_overlong_path_is_415_not_path_too_long(client, capsys):
    # §6b step 4 (magic-byte sniff) precedes step 5 (field validation):
    # a non-image with an over-long path fails as format.unsupported.
    c, _, _ = client
    r = H.ingest(c, H.make_text_bytes(), path="x" * 65)
    assert r.status_code == 415
    assert r.json()["error"]["reason"] == "format.unsupported"


class _FakeResponse:
    status_code = 200

    def __init__(self, content: bytes = b"{}") -> None:
        self.content = content


class _FakeHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return _FakeResponse()

    def get(self, *args, **kwargs):
        return _FakeResponse()


def _install_fake_httpx(monkeypatch):
    from vision_index.adapters import http_client as hc

    monkeypatch.setattr(hc.httpx, "Client", _FakeHttpxClient)
    return hc


def test_production_client_reraises_schema_api_error(monkeypatch):
    # A schema violation must stay response.validation_failed, never be
    # rewrapped as EmbedUnavailableError (-> 502 embed.unavailable).
    from vision_index.errors import ApiError

    hc = _install_fake_httpx(monkeypatch)

    def _raise_schema(raw):
        raise ApiError("response.validation_failed", "missing faces")

    monkeypatch.setattr(hc, "validate_embed_payload", _raise_schema)
    prod = hc.ProductionEmbedClient("https://embed.test", "tok", 5)
    with pytest.raises(ApiError) as exc_info:
        prod.embed(b"jpeg-bytes")
    assert exc_info.value.reason == "response.validation_failed"


def test_production_client_wraps_non_schema_failures(monkeypatch):
    # Non-schema failures from the validator still surface as
    # EmbedUnavailableError (-> embed.unavailable), unchanged.
    from vision_index.adapters.embed import EmbedUnavailableError

    hc = _install_fake_httpx(monkeypatch)

    def _raise_typeerror(raw):
        raise TypeError("bad content")

    monkeypatch.setattr(hc, "validate_embed_payload", _raise_typeerror)
    prod = hc.ProductionEmbedClient("https://embed.test", "tok", 5)
    with pytest.raises(EmbedUnavailableError):
        prod.embed(b"jpeg-bytes")


def test_closed_route_set_404_405(client):
    c, _, _ = client
    r = c.request("GET", "/v1/ingest")
    assert r.status_code == 405 and r.json()["error"]["reason"] == "method.not_allowed"
    r = c.request("PATCH", "/v1/name")
    assert r.status_code == 405 and r.json()["error"]["reason"] == "method.not_allowed"
    r = c.request("POST", "/v1/clusters")
    assert r.status_code == 405 and r.json()["error"]["reason"] == "method.not_allowed"
    r = c.get("/v1/people")
    assert r.status_code == 404 and r.json()["error"]["reason"] == "route.not_found"
    r = c.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["reason"] == "route.not_found"
    r = c.get("/v1/clusters/1")
    assert r.status_code == 404 and r.json()["error"]["reason"] == "route.not_found"
    r = c.request("DELETE", "/v1/assets/1")
    assert r.status_code == 405
    r = c.get("/v1/merge")
    assert r.status_code == 405


def test_request_id_echoed(client):
    c, _, a = client
    r = c.post("/v1/ingest", files={"image": ("x.png", a, "image/png")},
               headers={"X-Request-Id": "req-42"})
    assert r.headers.get("x-request-id") == "req-42"
    assert r.json()["request_id"] == "req-42"
    # invalid header → server-generated uuid4
    r2 = c.post("/v1/ingest", files={"image": ("x.png", a, "image/png")},
                headers={"X-Request-Id": "bad\nid"})
    req = r2.headers.get("x-request-id")
    assert req and req != "bad\nid" and len(req) == 36
