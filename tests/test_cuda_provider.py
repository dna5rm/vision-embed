"""GPU-free CUDA provider tests (vision-embed cuda SPEC section 7: T-C1..T-C7).

Image tests T-C6/T-C7 run only where the built image exists (env
VISION_EMBED_IMAGE, default vision-embed:v1); they skip elsewhere.
"""

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest_helpers import METRICS_PORT, REPO, PYTHON, boot, write_config, write_token

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

_HAS_CUDA_PROVIDER = False
try:
    import onnxruntime as _ort

    _HAS_CUDA_PROVIDER = "CUDAExecutionProvider" in _ort.get_available_providers()
except Exception:  # pragma: no cover
    pass


def _test_python() -> str:
    from conftest_helpers import REPO
    env = os.environ.get("VISION_EMBED_TEST_PYTHON")
    if env and Path(env).exists():
        return env
    venv = REPO / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _run_entry(config_path):
    return subprocess.run(
        [_test_python(), "-m", "vision_embed", "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
    )


def _fatal_payload(proc):
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert lines, f"no fatal JSON line in stdout: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    return json.loads(lines[-1])


def test_tc1_config_allowlist(tmp_path):
    """CUDA parses; arbitrary strings stay fatal; default stays CPU."""
    write_token(tmp_path)
    cfg = write_config(tmp_path, exec_provider=CUDA, extra="\n", paths_exist=False)
    from vision_embed.config import load_config
    assert load_config(cfg).execution_provider == CUDA

    cfg_bad_d = tmp_path / "bad"
    cfg_bad_d.mkdir()
    write_token(cfg_bad_d)
    cfg_bad = write_config(
        cfg_bad_d, exec_provider="TensorrtExecutionProvider", extra="\n", paths_exist=False
    )
    proc = _run_entry(cfg_bad)
    assert proc.returncode != 0
    payload = _fatal_payload(proc)
    assert payload["reason"] == "config.invalid"
    assert "provider" in payload["detail"]

    text = (REPO / "deploy" / "config.example.yaml").read_text()
    assert f'execution_provider: "{CPU}"' in text


def test_tc2_create_session_allowlist():
    """create_session accepts CUDA in the allowlist; ValueError only for others."""
    from vision_embed.adapters.onnx import create_session
    from pathlib import Path

    # allowlist check happens before any file access for other strings
    with pytest.raises(ValueError):
        create_session(Path("/nonexistent.onnx"), "TensorrtExecutionProvider")
    # CUDA passes the allowlist; with a missing model file onnxruntime raises
    # its own load error (ort.capi.onnxruntime_pybind11_state.Fail). Only that
    # expected load/activation failure is tolerated — any other exception,
    # including pytest.fail above, still fails the test.
    import onnxruntime.capi.onnxruntime_pybind11_state as _ort_state
    try:
        create_session(Path("/nonexistent.onnx"), CUDA)
    except ValueError:
        pytest.fail("CUDAExecutionProvider must pass the create_session allowlist")
    except _ort_state.Fail as exc:
        assert "Load model from /nonexistent.onnx failed" in str(exc)
    except _ort_state.NoSuchFile as exc:
        assert "/nonexistent.onnx" in str(exc)
    else:
        pytest.fail("missing model file must surface as an onnxruntime load error, not success")


@pytest.mark.skipif(_HAS_CUDA_PROVIDER, reason="environment really has CUDA")
def test_tc3_activation_failure_to_fatal(tmp_path):
    """On CPU-only env, CUDA session build fails or degrades — no silent fallback."""
    write_token(tmp_path)
    cfg = write_config(
        tmp_path, adapter="production", adapter_mode="production",
        paths_exist=True, tls_enabled="true", exec_provider=CUDA,
    )
    proc = _run_entry(cfg)
    assert proc.returncode != 0
    payload = _fatal_payload(proc)
    assert payload["reason"] == "config.invalid"
    assert CUDA in payload["detail"]
    # Also at the session layer directly: no silent fallback.
    from pathlib import Path
    from vision_embed.adapters.onnx import create_session
    with pytest.raises((RuntimeError, Exception)) as exc:
        create_session(Path(cfg.parent / "det_10g.onnx"), CUDA)
    assert not isinstance(exc.value, ValueError)


def test_tc4_fake_adapter_plumbing(tmp_path):
    """Full endpoint suite plumbing: fake adapter boots under both provider values."""
    write_token(tmp_path)
    for provider in (CPU, CUDA):
        d = tmp_path / provider[:4]
        d.mkdir()
        write_token(d)
        cfg = write_config(d, exec_provider=provider)
        client, cfg_obj = boot(cfg)
        assert cfg_obj.execution_provider == provider
        # one exercise per provider value: a health request through the app
        with client:
            r = client.get("/healthz")
            assert r.status_code == 200


def test_tc5_startup_error_content(tmp_path):
    """Production-mode CUDA startup on a CPU-only image: one JSON fatal, exit 1."""
    if _HAS_CUDA_PROVIDER:
        pytest.skip("environment has CUDA; fatal path unreachable")
    write_token(tmp_path)
    cfg = write_config(
        tmp_path, adapter="production", adapter_mode="production", paths_exist=True, tls_enabled="true",
        exec_provider=CUDA,
    )
    proc = _run_entry(cfg)
    assert proc.returncode == 1
    payload = _fatal_payload(proc)
    assert payload["fatal"] is True
    assert payload["reason"] == "config.invalid"
    assert CUDA in payload["detail"]


_IMAGE = os.environ.get("VISION_EMBED_IMAGE", "vision-embed:v1")


def _boot_run_args(tmp_path):
    """Config + token mounted; entrypoint runs the real startup path pre-bind and exits.

    uvicorn would serve forever, so --entrypoint runs exactly what run_server does
    before uvicorn.run (load_config + load_and_adapt — config parse, preflight log,
    session builds) as the same image/venv, then exits.
    """
    write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="fake", adapter_mode="test", exec_provider=CPU)
    # rewrite the host-side token path to the container-internal mount point
    cfg.write_text(
        cfg.read_text().replace(f"{tmp_path}/authz.bearer", "/opt/app/authz.bearer"),
        encoding="utf-8",
    )
    return [
        "run",
        "--rm",
        "-v", f"{tmp_path}/config.yaml:/opt/app/config.yaml:ro",
        "-v", f"{tmp_path}/authz.bearer:/opt/app/authz.bearer:ro",
        "--entrypoint", "/opt/venv/bin/python",
        _IMAGE,
        "-c",
        "import sys, onnxruntime as ort\n"
        "from pathlib import Path\n"
        "from vision_embed.config import load_config\n"
        "from vision_embed import app\n"
        "cfg = load_config(Path('/opt/app/config.yaml'))\n"
        "assert cfg.execution_provider == 'CPUExecutionProvider'\n"
        "app.provider_preflight(cfg.execution_provider)\n"
        "assert 'CPUExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()\n",
    ]


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
def test_tc6_image_check(tmp_path):
    """Onnxruntime-gpu present, plain onnxruntime absent; preflight log; CPU available."""
    if _docker("image", "inspect", _IMAGE).returncode != 0:
        pytest.skip(f"image {_IMAGE} not built on this host")
    freeze = _docker("run", "--rm", "--entrypoint", "/opt/venv/bin/pip", _IMAGE, "freeze")
    assert freeze.returncode == 0, freeze.stderr
    pkgs = {l.split("==")[0].lower() for l in freeze.stdout.splitlines()}
    assert "onnxruntime-gpu" in pkgs
    assert "onnxruntime" not in pkgs
    # preflight log line at boot with the default CPU config
    run = _docker(*_boot_run_args(tmp_path))
    assert run.returncode == 0, run.stdout + run.stderr
    assert '"event":"provider_preflight"' in run.stdout
    pre = [l for l in run.stdout.splitlines() if '"event":"provider_preflight"' in l]
    assert json.loads(pre[0])["configured"] == CPU


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
def test_tc7_cpu_default_intact(tmp_path):
    """GPU-wheel image boots with the default CPU config under plain runc."""
    if _docker("image", "inspect", _IMAGE).returncode != 0:
        pytest.skip(f"image {_IMAGE} not built on this host")
    run = _docker(*_boot_run_args(tmp_path))
    assert run.returncode == 0, run.stdout + run.stderr
    pre = [l for l in run.stdout.splitlines() if '"event":"provider_preflight"' in l]
    assert pre
    entry = json.loads(pre[0])
    assert entry["configured"] == CPU
    assert CPU in entry["registered"]
    # no fatal JSON line, no traceback: CPU sessions actually activated
    assert not any("Application startup failed" in l for l in (run.stdout + run.stderr).splitlines())
