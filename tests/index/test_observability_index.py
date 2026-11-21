"""Tests 16-19, 21, 22 — audit shape, metrics series, reason-token closed set,
embed-down behavior, no-OpenViking / no-InsightFace code assertions."""

from __future__ import annotations

import json
import re
import pathlib

import pytest
from starlette.testclient import TestClient

from tests.index import helpers_index as H
from tests.index.helpers_index import build_client, ingest


def test_audit_line_shape(tmp_path, capsys):
    a = H.make_image_bytes(size=(64, 48))
    fake = H.make_fake(None, persons={H.sha_of(a): "p1"},
                       faces_for={H.sha_of(a): 1}, labels_for={H.sha_of(a): ["car"]})
    with build_client(tmp_path, fake) as c:
        assert c.get("/healthz").status_code == 200
        H.ingest(c, a)
        c.post("/v1/name", json={"cluster_id": 1, "name": "alice"})
        c.get("/v1/nope")
    from vision_index.audit import AUDIT_KEYS

    lines = []
    for line in capsys.readouterr().out.splitlines():
        if line.strip().startswith("{"):
            rec = json.loads(line)
            assert set(rec) == AUDIT_KEYS, set(rec)
            lines.append(rec)
    assert len(lines) >= 4  # one per served request
    ingest_line = next(l for l in lines if l["route"] == "/v1/ingest"
                       and l["outcome"] == "ok")
    assert ingest_line["reason"] == "ok" and ingest_line["sha256"] == H.sha_of(a)
    assert isinstance(ingest_line["asset_id"], int)
    assert ingest_line["faces"] == 1 and ingest_line["clusters_created"] == 1
    name_line = next(l for l in lines if l["route"] == "/v1/name")
    assert "alice" not in json.dumps(name_line)  # no name strings in audit


def test_stdout_never_carries_secrets_or_vectors(tmp_path, capsys):
    a = H.make_image_bytes(size=(64, 48))
    sha = H.sha_of(a)
    fake = H.make_fake(None, persons={sha: "p1"}, faces_for={sha: 1})
    with build_client(tmp_path, fake, token="hush-token-xyz") as c:
        H.ingest(c, a, path="quiet-label")
        c.post("/v1/name", json={"cluster_id": 1, "name": "whisper"})
    out = capsys.readouterr().out
    assert "Bearer" not in out and "hush-token-xyz" not in out
    assert "whisper" not in out and "quiet-label" not in out
    assert "Bearer " not in out
    # no 768/512-length float arrays: audit sha256 and asset_id keys only
    for line in out.splitlines():
        if line.strip().startswith("{"):
            rec = json.loads(line)
            for k, v in rec.items():
                assert not (isinstance(v, list) and len(v) >= 128), (k, v)
    # image bytes never appear
    assert a[:4].hex() not in out and a.decode("utf-8", "ignore") not in out


def test_metrics_closed_set(tmp_path, capsys):
    # use a real (ephemeral) metrics port
    from vision_index import metrics as metrics_mod

    a = H.make_image_bytes(size=(64, 48))
    fake = H.make_fake(None, persons={H.sha_of(a): "p1"}, faces_for={H.sha_of(a): 1})
    with build_client(tmp_path, fake) as c:
        H.ingest(c, a)
        c.get("/v1/nope")
    body = metrics_mod.REGISTRY.get_prometheus_exposition_output() if hasattr(
        metrics_mod.REGISTRY, "get_prometheus_exposition_output"
    ) else None
    if body is None:
        from prometheus_client import generate_latest

        body = generate_latest(metrics_mod.REGISTRY).decode()
    # collapse histogram/counter suffixes to the five base series
    base_names = set()
    for m in re.finditer(r"^(vision_index_\w+)", body, re.M):
        n = m.group(1)
        if n.endswith("_created"):
            continue  # counter/histogram sample suffix, not an inventoried series
        for suf in ("_bucket", "_sum", "_count"):
            if n.endswith(suf):
                n = n[: -len(suf)]
        base_names.add(n)
    assert base_names == {
        "vision_index_requests_total",
        "vision_index_latency_seconds",
        "vision_index_errors_total",
        "vision_index_assets_total",
        "vision_index_clusters_total",
    }
    # no label value parses as 64-hex or a float array; no name/path labels
    hexish = re.findall(r'"([0-9a-f]{64})"', body)
    assert not hexish
    for forbidden in ("alice", "quiet", "path:", H.sha_of(a)):
        assert forbidden not in body


def test_reason_token_closed_set():
    from vision_index.errors import REASON_TOKENS, _TOKEN_HTTP, fatal_line

    expected = {
        "limits.body_too_large", "format.unsupported", "decode.failed",
        "path.too_long", "not_found", "route.not_found", "method.not_allowed",
        "auth.invalid_token", "name.conflict", "name.too_long",
        "cluster.not_face", "embed.unavailable", "no_face", "db.error",
        "not_ready", "response.validation_failed",
    }
    assert REASON_TOKENS == expected
    assert set(_TOKEN_HTTP) == expected
    # reason outside the closed set can never be constructed
    from vision_index.errors import ApiError

    with pytest.raises(ValueError):
        ApiError("fake.prefix", "nope")


def test_embed_down_ingest_502_and_readyz_503(tmp_path, capsys):
    a = H.make_image_bytes(size=(64, 48))
    fake = H.make_fake(None, fail=True, ready=False)
    with build_client(tmp_path, fake) as c:
        r = H.ingest(c, a)
        assert r.status_code == 502 and r.json()["error"]["reason"] == "embed.unavailable"
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["error"]["reason"] == "not_ready"
        # no partial rows written
        rec = c.get("/v1/assets/1")
        assert rec.status_code == 404
        listing = c.get("/v1/clusters?kind=face").json()["clusters"]
        assert listing == []
        assert fake.calls == []  # failed before the fake returned anything? calls recorded pre-fail
        # audit shows the failure, exactly one line per request
        lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()
                 if l.strip().startswith("{")]
        ingest_lines = [l for l in lines if l["route"] == "/v1/ingest"]
        assert len(ingest_lines) == 1 and ingest_lines[0]["reason"] == "embed.unavailable"


def test_no_openviking_anywhere():
    # implementation code must have zero OpenViking references (INV-9)
    root = H.REPO / "vision_index"
    text = "\n".join(p.read_text() for p in root.rglob("*.py"))
    assert "openviking" not in text.lower()
    assert "OpenViking" not in text
    example = H.REPO / "deploy" / "config.index.example.yaml"
    assert "openviking" not in example.read_text().lower()


def test_no_model_runtime_in_index_process():
    from pathlib import Path

    root = H.REPO / "vision_index"
    text = "\n".join(p.read_text() for p in root.rglob("*.py"))
    low = text.lower()
    for forbidden in ("insightface", "onnxruntime", "faiss", "hnswlib", "pgvector"):
        assert forbidden not in low, forbidden


def test_config_example_exists():
    candidates = [H.REPO / "deploy" / "config.index.example.yaml"]
    found = [p for p in candidates if p.is_file()]
    assert found, "index example config missing (see EXEC-QUESTIONS-INDEX.md Q1)"
    text = found[0].read_text()
    for key in ("adapter_mode", "face_match_max: 0.40", "embed", "bearer_token_path"):
        assert key in text
    assert re.search(r'base_url: "https://', text)
