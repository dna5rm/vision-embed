"""Acceptance tests for the v12 startup free-memory gate (SPEC 2e).

- Missing routes.describe_min_free_bytes / routes.tag_min_free_bytes when the
  switch is true AND provider is CUDA is fatal config.invalid before bind.
- Under CPU, or with the switch false, the keys are not required and the boot
  succeeds.
- A switch-true route whose operator minimum exceeds the NVML device-0 free
  bytes is soft-disabled (behaves exactly as disabled) and the process still
  binds and serves /v1/embed; /v1/runtime shows unavailable_reason
  'gpu.insufficient'. The compare order is describe, then tag.
- NVML unreadable → every check-requiring route is soft-disabled and the
  process binds; no byte value is guessed.

No byte threshold is hardcoded: every minimum is supplied by the test config.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

import describe_helpers as dh

TOKEN = dh.TOKEN
CPU = dh.CPU
CUDA = dh.CUDA


def _run_fatal(tmp_path, **kwargs):
    dh.write_token(tmp_path, TOKEN)
    cfg = dh.write_describe_config(tmp_path, **kwargs)
    proc = dh.run_entry(cfg)
    assert proc.returncode != 0, f"expected fatal, rc=0 stdout={proc.stdout!r}"
    payload = dh.fatal_payload(proc)
    assert payload["fatal"] is True
    return payload


def _boot_nvml(monkeypatch, tmp_path, *, free_bytes, fail=False, **kwargs):
    """Inject a pynvml stub (the same NVML library call of SPEC 6d/2e) and
    build the app in-process — proves the process binds (no fatal)."""
    stub = dh.make_pynvml_stub(free_bytes, fail=fail)
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    dh.write_token(tmp_path, TOKEN)
    from vision_embed.app import build_from_config_path
    from vision_embed.errors import FatalStartupError

    cfg = dh.write_describe_config(tmp_path, **kwargs)
    app: Any = None  # type: ignore[annotation]
    try:
        app, _cfg = build_from_config_path(cfg)
    except FatalStartupError as exc:
        pytest.fail(f"startup fatal (memory gate/config not implemented yet): {exc}")
    return TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})


def test_missing_describe_min_fatal_when_cuda_and_enabled(tmp_path):
    """CUDA + describe_enabled true + missing describe_min_free_bytes →
    fatal config.invalid before bind, detail names the key."""
    payload = _run_fatal(
        tmp_path,
        describe_enabled=True,
        tag_enabled=False,
        exec_provider=CUDA,
        florence_dir_present=True,
    )
    assert payload["reason"] == "config.invalid"
    assert "describe_min_free_bytes" in payload["detail"]


def test_missing_tag_min_fatal_when_cuda_and_enabled(tmp_path):
    payload = _run_fatal(
        tmp_path,
        describe_enabled=False,
        tag_enabled=True,
        exec_provider=CUDA,
        tag_files_present=True,
    )
    assert payload["reason"] == "config.invalid"
    assert "tag_min_free_bytes" in payload["detail"]


def test_minima_not_required_under_cpu_provider(monkeypatch, tmp_path):
    """CPUE ExecutionProvider: the check never runs and the keys may be
    absent — an enabled switch loads normally."""
    client = _boot_nvml(
        monkeypatch, tmp_path,
        free_bytes=0, describe_enabled=True, tag_enabled=True,
        tag_files_present=True, florence_dir_present=True,
        exec_provider=CPU,
    )
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"status": "ready", "models_loaded": 6}
    files = {"image": ("img.png", dh.make_image_bytes(), "image/png")}
    r = client.post("/v1/describe", files=files, data={"task": "<CAPTION>"})
    assert r.status_code == 200, r.text
    r = client.post("/v1/tag", files=files, data={"tag_threshold": "0.5"})
    assert r.status_code == 200, r.text


def test_minima_not_required_when_switch_false_under_cuda(tmp_path):
    """A false switch never runs the check and never required the key."""
    dh.write_token(tmp_path, TOKEN)
    from vision_embed.app import build_from_config_path
    from vision_embed.errors import FatalStartupError

    cfg = dh.write_describe_config(
        tmp_path, describe_enabled=False, tag_enabled=False, exec_provider=CUDA
    )
    try:
        app, _cfg = build_from_config_path(cfg)
    except FatalStartupError as exc:
        pytest.fail(f"startup fatal (config not implemented yet): {exc}")
    client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"status": "ready", "models_loaded": 4}
    for path, data in (
        ("/v1/describe", {"task": "<CAPTION>"}),
        ("/v1/tag", {"tag_threshold": "0.5"}),
    ):
        r = client.post(path, files={"image": ("img.png", dh.make_image_bytes(), "image/png")}, data=data)
        assert r.status_code == 404, r.text
        assert r.json()["error"]["reason"] == "route.disabled"


def test_below_operator_minimum_soft_disables_route_process_binds(monkeypatch, tmp_path):
    """device-0 free below describe_min_free_bytes: describe soft-disabled,
    the process binds and serves /v1/embed (no exit, no fatal), florence2 is
    not counted in models_loaded and shows gpu.insufficient on /v1/runtime."""
    free = 1 * 1024 * 1024 * 1024  # 1 GiB free
    client = _boot_nvml(
        monkeypatch, tmp_path,
        free_bytes=free,
        describe_enabled=True,
        tag_enabled=True,
        tag_files_present=True,  # a gated route needs no files: present anyway
        florence_dir_present=True,
        exec_provider=CUDA,
        describe_min_free_bytes=2 * 1024 * 1024 * 1024,  # 2 GiB > free
        tag_min_free_bytes=512 * 1024 * 1024,  # 512 MiB ≤ free
    )
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    files = {"image": ("img.png", dh.make_image_bytes(), "image/png")}
    r = client.post("/v1/describe", files=files, data={"task": "<CAPTION>"})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["reason"] == "route.disabled"
    assert r.json()["error"]["reason"] != "task.unsupported"
    body = client.get("/v1/runtime").json()
    by_role = {m["role"]: m for m in body["models"]}
    assert by_role["florence2"]["loaded"] is False
    assert by_role["florence2"]["model_id"] == ""
    assert by_role["florence2"]["unavailable_reason"] == "gpu.insufficient"
    assert by_role["tagger"]["unavailable_reason"] == ""
    r = client.post("/v1/embed", files=files)
    assert r.status_code == 200, r.text


def test_compare_order_describe_then_tag(monkeypatch, tmp_path):
    """Fixed order: describe passes (its minimum is subtracted), tag does not
    fit the remaining free → tagger soft-disabled, describe still 200."""
    free = 1 * 1024 * 1024 * 1024  # 1 GiB
    client = _boot_nvml(
        monkeypatch, tmp_path,
        free_bytes=free,
        describe_enabled=True,
        tag_enabled=True,
        tag_files_present=True,
        florence_dir_present=True,
        exec_provider=CUDA,
        describe_min_free_bytes=256 * 1024 * 1024,  # ≤ free: passes
        tag_min_free_bytes=1024 * 1024 * 1024,  # > free − 256 MiB: fails
    )
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"status": "ready", "models_loaded": 5}
    files = {"image": ("img.png", dh.make_image_bytes(), "image/png")}
    r = client.post("/v1/describe", files=files, data={"task": "<CAPTION>"})
    assert r.status_code == 200, f"describe must load and serve: {r.text}"
    r = client.post("/v1/tag", files=files, data={"tag_threshold": "0.5"})
    assert r.status_code == 404
    assert r.json()["error"]["reason"] == "route.disabled"
    body = client.get("/v1/runtime").json()
    by_role = {m["role"]: m for m in body["models"]}
    assert by_role["florence2"]["unavailable_reason"] == ""
    assert by_role["florence2"]["loaded"] is True
    assert by_role["tagger"]["unavailable_reason"] == "gpu.insufficient"


def test_nvml_unreadable_soft_disables_all_check_routes_binds(monkeypatch, tmp_path):
    """NVML cannot be read: no value is guessed, every check-requiring route
    is soft-disabled with gpu.insufficient and the process still binds."""
    client = _boot_nvml(
        monkeypatch, tmp_path,
        free_bytes=0, fail=True,
        describe_enabled=True,
        tag_enabled=True,
        tag_files_present=True,
        florence_dir_present=True,
        exec_provider=CUDA,
        describe_min_free_bytes=1,
        tag_min_free_bytes=1,
    )
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"status": "ready", "models_loaded": 4}
    files = {"image": ("img.png", dh.make_image_bytes(), "image/png")}
    for path, data in (
        ("/v1/describe", {"task": "<CAPTION>"}),
        ("/v1/tag", {"tag_threshold": "0.5"}),
    ):
        r = client.post(path, files=files, data=data)
        assert r.status_code == 404, f"{path}: {r.text}"
        assert r.json()["error"]["reason"] == "route.disabled"
    r = client.post("/v1/embed", files=files)
    assert r.status_code == 200, r.text
    body = client.get("/v1/runtime").json()
    by_role = {m["role"]: m for m in body["models"]}
    for role in ("florence2", "tagger"):
        assert by_role[role]["unavailable_reason"] == "gpu.insufficient"
        assert by_role[role]["loaded"] is False


# ---------------------------------------------------------------------------
# Ordering test: the section 2d CUDA activation gate must run BEFORE the
# section 2e free-memory read can soft-disable a route and silently mask a
# CUDA session that constructs while CPU leads. Per describe spec §2e ("When
# the read happens"), the free-memory gate runs only AFTER the fail-closed
# CUDA gate of §2d has passed (provider registered, tagger session gate,
# Florence-2 placement) — so a switch-true describe route whose CUDA placement
# fails must be fatal config.invalid before bind, even when the operator's
# describe_min_free_bytes would (later, legitimately) soft-disable it.
# ---------------------------------------------------------------------------

_STUB_PORT = 18795

# Tiny valid ONNX (Identity x→y) — same blob test_config_failclosed.py uses.
_TINY_ONNX_B64 = (
    "CAo6UgoQCgF4EgF5IghJZGVudGl0eRIEdGlueVobCgF4EhYKFAgBEhAKAggBCgIIAQoCCAEKAggBYhsK"
    "AXkSFgoUCAESEAoCCAEKAggBCgIIAQoCCAFCBAoAEBA="
)
import base64 as _base64


def _write_stub_packages(tmp_path: Path, free_bytes: int) -> str:
    """Subprocess equivalent of this repo's existing in-process mechanism
    (monkeypatch.setitem(sys.modules, "pynvml", dh.make_pynvml_stub(...))):
    the same stubs injected as PYTHONPATH shadow modules — no production hook.
    - pynvml: the identical device-0 read shape of make_pynvml_stub.
    - onnxruntime: CUDA is registered and every session ACTIVATES CUDA, so
      the §2d provider preflight and the four embed roles pass, isolating the
      non-activation to the Florence-2 placement (§2d step 4).
    - transformers: an AutoModelForCausalLM whose parameters are NOT on the
      CUDA device — exactly the "session constructs while CPU leads" state of
      §2d: construction succeeds, activation does not.
    """
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    total = free_bytes * 4
    (stubs / "pynvml.py").write_text(
        "class NVMLError(Exception):\n"
        "    pass\n"
        "\n"
        "def nvmlInit():\n"
        "    return None\n"
        "\n"
        "def nvmlShutdown():\n"
        "    return None\n"
        "\n"
        "def nvmlDeviceGetCount():\n"
        "    return 1\n"
        "\n"
        "def nvmlDeviceGetHandleByIndex(index):\n"
        "    return object()\n"
        "\n"
        "def nvmlDeviceGetName(handle):\n"
        "    return b'Stub GPU'\n"
        "\n"
        "class _MemInfo:\n"
        f"    total = {total}\n"
        f"    used = {total - free_bytes}\n"
        f"    free = {free_bytes}\n"
        "\n"
        "def nvmlDeviceGetMemoryInfo(handle):\n"
        "    return _MemInfo()\n",
        encoding="utf-8",
    )
    (stubs / "onnxruntime.py").write_text(
        "def get_available_providers():\n"
        "    return ['CUDAExecutionProvider', 'CPUExecutionProvider']\n"
        "\n"
        "class _IO:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "class InferenceSession:\n"
        "    def __init__(self, path, providers=None, **kw):\n"
        "        # the session CONSTRUCTS and the activation check reads CUDA:\n"
        "        # get_providers()[0] == 'CUDAExecutionProvider'\n"
        "        self._providers = ['CUDAExecutionProvider']\n"
        "\n"
        "    def get_providers(self):\n"
        "        return list(self._providers)\n"
        "\n"
        "    def get_inputs(self):\n"
        "        return [_IO('input')]\n"
        "\n"
        "    def get_outputs(self):\n"
        "        return [_IO('output')]\n",
        encoding="utf-8",
    )
    (stubs / "transformers.py").write_text(
        "class _Param:\n"
        "    # NOT on the CUDA device: the §2d Florence placement does not hold\n"
        "    is_cuda = False\n"
        "\n"
        "class _FakeModel:\n"
        "    def to(self, device):\n"
        "        return self\n"
        "\n"
        "    def parameters(self):\n"
        "        return iter((_Param(),))\n"
        "\n"
        "class AutoModelForCausalLM:\n"
        "    @staticmethod\n"
        "    def from_pretrained(path, **kw):\n"
        "        return _FakeModel()\n"
        "\n"
        "class _FakeProcessor:\n"
        "    def batch_decode(self, *a, **kw):\n"
        "        raise RuntimeError('stub processor never decodes')\n"
        "\n"
        "class AutoProcessor:\n"
        "    @staticmethod\n"
        "    def from_pretrained(path, **kw):\n"
        "        return _FakeProcessor()\n",
        encoding="utf-8",
    )
    return str(stubs)


def _cuda_activation_gate_before_memory_gate(tmp_path: Path):
    """Build the config and the env; run the entry with a Popen loop and
    return the fatal payload, failing with a precise message if the process
    instead binds (the current soft-disable-and-bind behavior)."""
    free = 1 * 1024 * 1024 * 1024  # 1 GiB free on the stubbed device 0
    stubs = _write_stub_packages(tmp_path, free)
    dh.write_token(tmp_path, dh.TOKEN)
    # production mode requires the four embed model files to exist; the stub
    # onnxruntime never parses them, so the same tiny valid ONNX that
    # test_config_failclosed.py uses is good enough.
    tiny = _base64.b64decode(_TINY_ONNX_B64)
    for name in ("det_10g.onnx", "w600k_r50.onnx", "openai-ViT-L-14-visual.onnx", "yolov8n.onnx"):
        (tmp_path / name).write_bytes(tiny)
    cfg = dh.write_describe_config(
        tmp_path,
        adapter="production",
        adapter_mode="production",
        exec_provider=CUDA,
        describe_enabled=True,
        tag_enabled=False,
        describe_min_free_bytes=3 * 1024 * 1024 * 1024,  # 3 GiB > 1 GiB free
        tag_files_present=False,
        florence_dir_present=True,
        port=_STUB_PORT,
    )
    # production mode requires TLS with a loadable PEM pair (SPEC §4)
    text = cfg.read_text(encoding="utf-8").replace('enabled: "false"', 'enabled: "true"')
    cfg.write_text(text, encoding="utf-8")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(tmp_path / "key.pem"), "-out", str(tmp_path / "cert.pem"),
            "-days", "2", "-nodes", "-subj", "/CN=localhost",
        ],
        check=True, capture_output=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = stubs
    proc = subprocess.Popen(
        [dh.PYTHON, "-m", "vision_embed", "--config", str(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(dh.REPO),
        env=env,
    )
    try:
        deadline = time.monotonic() + 60
        bound = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                sock = socket.create_connection(("127.0.0.1", _STUB_PORT), timeout=1)
                sock.close()
                bound = True
                break
            except OSError:
                time.sleep(0.25)
        if bound or (proc.poll() is None):
            if proc.poll() is None:
                proc.terminate()
            try:
                out, err = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate(timeout=10)
            raise AssertionError(
                "startup memory gate ordering violated: with describe below its "
                "operator minimum AND a CUDA Florence placement that does not "
                "activate, the process soft-disabled describe and bound its "
                "listener — the §2d CUDA activation gate must be fatal "
                f"config.invalid before bind (rc={proc.returncode} stdout={out!r} stderr={err!r})"
            )
        out, err = proc.communicate(timeout=15)
        assert proc.returncode != 0, f"expected fatal, rc=0 stdout={out!r}"
        fatal_lines = [
            ln for ln in out.splitlines()
            if ln.startswith("{") and json.loads(ln).get("fatal") is True
        ]
        assert len(fatal_lines) == 1, (
            f"expected exactly one fatal line, got {len(fatal_lines)}: stdout={out!r} "
            f"stderr={err!r}"
        )
        payload = json.loads(fatal_lines[0])
        assert payload["fatal"] is True
        assert payload["reason"] == "config.invalid"
        assert "cuda" in payload["detail"].lower() or "florence" in payload["detail"].lower()
        try:
            sock = socket.create_connection(("127.0.0.1", _STUB_PORT), timeout=2)
            sock.close()
            raise AssertionError(f"listener is still up on {_STUB_PORT} after fatal startup")
        except (ConnectionRefusedError, OSError):
            pass
        return payload
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def test_cuda_activation_gate_is_fatal_before_memory_gate_soft_disables(tmp_path):
    """§2e order: the describe route's CUDA Florence placement (§2d step 4)
    is checked BEFORE the free-memory read. Config: CUDA provider, describe
    enabled with a minimum above the stubbed free memory (the §2e gate would
    soft-disable it), tag switch false, and a CUDA placement that constructs
    while CPU leads. The process must exit nonzero before bind with exactly
    one stdout fatal line {fatal: true, reason: config.invalid, ...} and must
    never bind /v1/embed. If the current code soft-disables describe and
    binds instead, this test fails — that is the bug this test pins."""
    _cuda_activation_gate_before_memory_gate(tmp_path)
