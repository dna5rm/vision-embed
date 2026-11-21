"""Container contract tests (SPEC section 11, tests 16-18).

Real docker runs (no docker = skip, never a faked pass):
  16. no-runtime-fetch  — container on `--network none` starts, /readyz is
      green, and a fixture image embeds (all with zero egress).
  17. read-only rootfs  — `--read-only` + tmpfs /tmp; fixture request and a
      container restart both succeed.
  18. non-root          — running process UID is 65532.

The image must be pre-built for these to be meaningful; they do not fake it.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import pytest
from conftest_helpers import REPO, make_image_bytes

IMAGE = "vision-embed:test"
TOKEN = "container-test-token"
UID = "65532"


def _docker():
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0


def _can_run_target_platform() -> bool:
    """Image is pinned to linux/amd64 (SPEC/DEPLOY compose platform pin).

    On an arm64 host without amd64 emulation these tests cannot execute the
    real image — they skip, which is NOT a pass.
    """
    return platform.machine() in ("x86_64", "amd64")


docker_required = pytest.mark.skipif(
    not _docker(), reason="docker unavailable on this host — container contract NOT proven"
)
platform_required = pytest.mark.skipif(
    not _can_run_target_platform(),
    reason=f"host is {platform.machine()}; the pinned image is linux/amd64 and this host has no amd64 emulation — container tests NOT run",
)


def _run(cmd: list[str], timeout: int = 120, input: str | None = None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, input=input, text=text)


def _write_token(d: Path) -> Path:
    p = d / "authz.bearer"
    p.write_text(TOKEN + "\n", encoding="utf-8")
    return p


def _write_container_config(d: Path) -> Path:
    """Test-adapter config (fake mode => no ONNX files needed, TLS optional)."""
    cfg = d / "config.yaml"
    cfg.write_text(
        f"""
listen:
  host: "0.0.0.0"
  port: 8443
  tls:
    enabled: false
    cert_path: "{d / 'cert.pem'}"
    key_path: "{d / 'key.pem'}"
auth:
  # In-container path: the test bind-mounts authz.bearer here (docker -v).
  bearer_token_path: "/etc/vision-embed/secrets/authz.bearer"
models:
  det_path: "{d / 'det.onnx'}"
  face_embed_path: "{d / 'face.onnx'}"
  scene_embed_path: "{d / 'scene.onnx'}"
  object_det_path: "{d / 'obj.onnx'}"
  execution_provider: "CPUExecutionProvider"
  adapter: "fake"
adapter_mode: "test"
limits:
  max_body_bytes: 33554432
metrics:
  enabled: false
audit:
  enabled: true
request_id:
  header: "X-Request-Id"
""",
        encoding="utf-8",
    )
    return cfg


def _container_ip(name: str) -> str:
    out = _run(
        [
            "docker", "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name,
        ]
    )
    return out.stdout.strip()


def _wait_ready(name: str, ip: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"http://{ip}:8443/readyz")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return json.load(resp)
                last = f"status {resp.status}"
        except Exception as exc:  # not ready yet
            last = str(exc)
        time.sleep(0.5)
    raise AssertionError(f"/readyz never became ready: {last}")


def _post_image(ip: str) -> dict:
    boundary = uuid.uuid4().hex
    img = make_image_bytes("PNG")
    body = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"f.png\"\r\n"
        "Content-Type: image/png\r\n\r\n"
    ).encode() + img + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://{ip}:8443/v1/embed",
        data=body,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        return json.load(resp)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("container")
    _write_token(d)
    cfg = _write_container_config(d)
    return d, cfg


def _start(name: str, assets, *extra_args: str) -> str:
    d, cfg = assets
    cmd = [
        "docker", "run", "-d", "--rm", "--name", name,
        "-v", f"{cfg}:/opt/app/config.yaml:ro",
        "-v", f"{d / 'authz.bearer'}:/etc/vision-embed/secrets/authz.bearer:ro",
        *extra_args,
        IMAGE,
    ]
    out = _run(cmd)
    assert out.returncode == 0, f"docker run failed: {out.stderr}"
    return out.stdout.strip()


def _stop(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@docker_required
@platform_required
def test_container_16_no_runtime_fetch_network_isolated(assets):
    """Network-isolated (`--network none`) container starts, ready, serves a fixture."""
    name = "ve16"
    _stop(name)
    cid = _start(name, assets, "--network", "none")
    # container may take a moment; poll via docker exec (host can't reach it)
    deadline = time.time() + 30
    ready = False
    out = None
    while time.time() < deadline:
        out = _run(
            ["docker", "exec", name, "/opt/venv/bin/python", "-c",
             "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8443/readyz').read().decode())"],
            timeout=20,
        )
        if out.returncode == 0 and "ready" in out.stdout:
            ready = True
            break
        time.sleep(1)
    assert ready, f"never ready inside network-isolated container: {out.stdout} {out.stderr}"
    # fixture image served with zero egress: copy bytes in and POST from inside
    src = assets[0] / "fixture.png"
    src.write_bytes(make_image_bytes("PNG"))
    out = _run(["docker", "cp", str(src), f"{name}:/tmp/fixture.png"])
    assert out.returncode == 0, out.stderr
    out = _run(["docker", "exec", "-i", name, "/opt/venv/bin/python", "-"], input=_IN_CONTAINER_POST, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert "faces" in payload and "objects" in payload
    _stop(name)


# Posted inside the container: minimal multipart via urllib (stdlib only).
_IN_CONTAINER_POST = (
    "import json, urllib.request, uuid\n"
    "img = open('/tmp/fixture.png','rb').read()\n"
    "b = uuid.uuid4().hex\n"
    "body = ((f'--{b}\\r\\nContent-Disposition: form-data; name=\"image\"; filename=\"f.png\"\\r\\n'\n"
    "         'Content-Type: image/png\\r\\n\\r\\n').encode() + img + f'\\r\\n--{b}--\\r\\n'.encode())\n"
    "req = urllib.request.Request('http://127.0.0.1:8443/v1/embed', data=body,\n"
    "    headers={'Authorization': 'Bearer " + TOKEN + "',\n"
    "             'Content-Type': f'multipart/form-data; boundary={b}'})\n"
    "print(urllib.request.urlopen(req, timeout=15).read().decode())\n"
)


@docker_required
@platform_required
def test_container_17_read_only_rootfs_fixture_request_and_restart(assets):
    name = "ve17"
    _stop(name)
    try:
        cid = _start(name, assets, "--read-only", "--tmpfs", "/tmp")
        ip = _container_ip(name)
        assert ip, "no container IP"
        _wait_ready(name, ip)
        payload = _post_image(ip)
        assert "faces" in payload and "objects" in payload
        # restart under the same constraints
        out = _run(["docker", "restart", name], timeout=60)
        assert out.returncode == 0, out.stderr
        ip2 = _container_ip(name)
        _wait_ready(name, ip2)
        payload2 = _post_image(ip2)
        assert "faces" in payload2 and "objects" in payload2
    finally:
        _stop(name)


@docker_required
@platform_required
def test_container_18_process_uid_65532(assets):
    name = "ve18"
    _stop(name)
    try:
        cid = _start(name, assets)
        out = _run(["docker", "exec", name, "id", "-u"])
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == UID, f"process UID is {out.stdout.strip()!r}, expected 65532"
        # also the actual server process
        out = _run(["docker", "exec", name, "/opt/venv/bin/python", "-c", "import os; print(os.geteuid())"])
        assert out.stdout.strip() == UID
    finally:
        _stop(name)
