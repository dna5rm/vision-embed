"""Acceptance tests for the v11 GPU admission lock (SPEC v12, section 6e).

One in-process lock shared by /v1/embed, /v1/describe, /v1/tag under
CUDAExecutionProvider only: no queue, no wait — a second GPU inference is
HTTP 429 gpu.busy and does not start. healthz/readyz/runtime are never
blocked. CPU mode has no lock and never returns gpu.busy. A failure while
holding the lock is 500 inference.failed, never gpu.busy.
"""

import contextlib
import io
import threading
import time

import pytest
from starlette.testclient import TestClient

import describe_helpers as dh
from vision_embed import app as app_mod
from vision_embed.adapters import fake as fake_mod
from vision_embed.app import build_from_config_path

TOKEN = dh.TOKEN
CPU = dh.CPU
CUDA = dh.CUDA
IMG = {"image": ("img.png", dh.make_image_bytes(), "image/png")}


def _build(tmp_path, exec_provider):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dh.write_token(tmp_path, TOKEN)
    cfg_path = dh.write_describe_config(
        tmp_path,
        describe_enabled=True,
        tag_enabled=True,
        tag_files_present=True,
        florence_dir_present=True,
        exec_provider=exec_provider,
    )
    from vision_embed.errors import FatalStartupError

    try:
        app, _cfg = app_mod.build_from_config_path(cfg_path)
    except FatalStartupError as exc:
        pytest.fail(f"startup fatal (routes/lock not implemented yet): {exc}")
    return app


def _client(app):
    return TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})


def test_cuda_second_inference_429_gpu_busy_does_not_start(tmp_path):
    """While one inference holds the shared lock, concurrent POSTs to the
    other GPU routes return 429 gpu.busy and do not start (no queue/wait)."""
    app = _build(tmp_path / "cuda", CUDA)
    entered = threading.Event()
    release = threading.Event()
    original = fake_mod.FakeDetector.detect

    def blocking_detect(self, bgr):
        entered.set()
        release.wait(timeout=15)
        return original(self, bgr)

    fake_mod.FakeDetector.detect = blocking_detect
    try:
        results = {}

        def holder():
            c = _client(app)
            results["holder"] = c.post("/v1/embed", files=dict(IMG))

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=15), "holder request never reached the fake detector"
        c = _client(app)
        results["describe"] = c.post(
            "/v1/describe", files=dict(IMG),
            data={"task": "<CAPTION>"},
        )
        results["tag"] = c.post(
            "/v1/tag", files=dict(IMG), data={"tag_threshold": "0.5"},
        )
        release.set()
        t.join(timeout=30)
        assert results["holder"].status_code == 200, results["holder"].text
        for key in ("describe", "tag"):
            r = results[key]
            assert r.status_code == 429, f"{key}: {r.status_code} {r.text}"
            assert r.json()["error"]["reason"] == "gpu.busy"
    finally:
        fake_mod.FakeDetector.detect = original
        release.set()


def test_probes_not_blocked_while_lock_held(tmp_path):
    """healthz, readyz and /v1/runtime take no lock: 200 while the lock is
    held by /v1/embed."""
    app = _build(tmp_path / "probes", CUDA)
    entered = threading.Event()
    release = threading.Event()
    original = fake_mod.FakeDetector.detect

    def blocking_detect(self, bgr):
        entered.set()
        release.wait(timeout=15)
        return original(self, bgr)

    fake_mod.FakeDetector.detect = blocking_detect
    try:
        holder_done = {}

        def holder():
            c = _client(app)
            holder_done["r"] = c.post("/v1/embed", files=dict(IMG))

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=15)
        c = _client(app)
        for path in ("/healthz", "/readyz", "/v1/runtime"):
            r = c.get(path)
            assert r.status_code == 200, f"{path} was blocked: {r.text}"
            assert "gpu.busy" not in r.text
        release.set()
        t.join(timeout=30)
        assert holder_done["r"].status_code == 200
    finally:
        fake_mod.FakeDetector.detect = original
        release.set()


def test_cpu_provider_never_gpu_busy(tmp_path):
    """CPUExecutionProvider: no lock exists — a concurrent POST is never 429
    gpu.busy."""
    app = _build(tmp_path / "cpu", CPU)
    entered = threading.Event()
    release = threading.Event()
    original = fake_mod.FakeDetector.detect

    def blocking_detect(self, bgr):
        entered.set()
        release.wait(timeout=15)
        return original(self, bgr)

    fake_mod.FakeDetector.detect = blocking_detect
    try:
        def holder():
            c = _client(app)
            return c.post("/v1/embed", files=dict(IMG))

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=15)
        c = _client(app)
        r = c.post("/v1/describe", files=dict(IMG), data={"task": "<CAPTION>"})
        release.set()
        t.join(timeout=30)
        assert r.status_code != 429 or r.json()["error"]["reason"] != "gpu.busy", r.text
        assert r.status_code == 200
    finally:
        fake_mod.FakeDetector.detect = original
        release.set()


def test_failure_while_holding_lock_is_inference_failed_not_gpu_busy(tmp_path):
    """A run that fails inside the lock is 500 inference.failed — gpu.busy is
    never mapped from it (v11 v12 rule)."""
    app = _build(tmp_path / "fail", CUDA)
    original = fake_mod.FakeDetector.detect
    fake_mod.FakeDetector.detect = lambda self, bgr: (_ for _ in ()).throw(RuntimeError("simulated inference failure"))
    try:
        c = _client(app)
        r = c.post("/v1/embed", files=dict(IMG))
        assert r.status_code == 500, r.text
        assert r.json()["error"]["reason"] == "inference.failed", r.text
    finally:
        fake_mod.FakeDetector.detect = original
