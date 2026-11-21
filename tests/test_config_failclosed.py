"""Config fail-closed tests (SPEC section 11, tests 1-3 and 19)."""

import json
import subprocess
import sys
from pathlib import Path

from conftest_helpers import REPO, PYTHON, write_config, write_token


def run_entry(config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "vision_embed", "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO),
    )


def test_unknown_yaml_key_fails_startup(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, extra="\n\nevil_extra: 1\n")
    proc = run_entry(cfg)
    assert proc.returncode != 0, proc.stderr
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines
    payload = json.loads(fatal_lines[-1])
    assert payload["fatal"] is True
    assert payload["reason"] == "config.unknown_key"


def test_missing_model_file_fails_startup_in_production(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="production", adapter_mode="production", paths_exist=False, tls_enabled="true")
    proc = run_entry(cfg)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines
    payload = json.loads(fatal_lines[-1])
    assert payload["reason"] == "config.invalid"
    assert "missing" in payload["detail"] or "model file" in payload["detail"]


def test_missing_single_model_file_after_creation_fails_startup(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="production", adapter_mode="production", paths_exist=True, tls_enabled="true")
    # remove one of the four files
    Path(tmp_path / "yolov8n.onnx").unlink()
    proc = run_entry(cfg)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines


def test_fake_adapter_with_production_mode_is_fatal(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="fake", adapter_mode="production")
    proc = run_entry(cfg)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines
    payload = json.loads(fatal_lines[-1])
    assert payload == {
        "fatal": True,
        "reason": "config.invalid",
        "detail": "models.adapter=fake requires adapter_mode=test",
    }


def test_execution_provider_cuda_is_accepted(tmp_path):
    """CUDA spec T-C1: CUDAExecutionProvider now parses; arbitrary strings stay fatal."""
    token = write_token(tmp_path)
    from vision_embed.config import load_config
    cfg = write_config(tmp_path, exec_provider="CUDAExecutionProvider")
    assert load_config(cfg).execution_provider == "CUDAExecutionProvider"
    cfg2 = write_config(tmp_path, exec_provider="AzureExecutionProvider")
    proc = run_entry(cfg2)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert json.loads(fatal_lines[-1])["reason"] == "config.invalid"


def test_default_provider_is_cpu_in_example_config():
    text = (REPO / "deploy" / "config.example.yaml").read_text()
    assert 'execution_provider: "CPUExecutionProvider"' in text


def test_production_mode_refuses_tls_disabled(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="production", adapter_mode="production", tls_enabled="false")
    proc = run_entry(cfg)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines
    payload = json.loads(fatal_lines[-1])
    assert payload["reason"] == "config.invalid"
    assert "tls" in payload["detail"].lower()


def test_production_mode_missing_cert_file_is_fatal(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="production", adapter_mode="production", paths_exist=True, tls_enabled="true")
    (Path(tmp_path) / "cert.pem").unlink()
    proc = run_entry(cfg)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines
    payload = json.loads(fatal_lines[-1])
    assert payload["reason"] == "config.invalid"
    assert "cert" in payload["detail"]


def test_production_mode_unparseable_cert_pair_is_fatal(tmp_path):
    token = write_token(tmp_path)
    cfg = write_config(tmp_path, adapter="production", adapter_mode="production", paths_exist=True, tls_enabled="true")
    (Path(tmp_path) / "cert.pem").write_text("not a pem\n", encoding="utf-8")
    proc = run_entry(cfg)
    assert proc.returncode != 0
    fatal_lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert fatal_lines
    payload = json.loads(fatal_lines[-1])
    assert payload["reason"] == "config.invalid"
    assert "tls" in payload["detail"].lower() or "cert" in payload["detail"].lower()


def test_bearer_token_trailing_cr_is_stripped(tmp_path):
    token = write_token(tmp_path)
    (Path(tmp_path) / "authz.bearer").write_text("test-token-abc123\r\n", encoding="utf-8")
    from vision_embed.config import load_config

    cfg = load_config(write_config(tmp_path))
    assert cfg.bearer_token == "test-token-abc123"
    assert "\r" not in cfg.bearer_token


# ---------------------------------------------------------------------------
# v12 production fail-closed tests for the two startup switches (SPEC §4):
# - routes.*_enabled=true + missing route model files -> one stdout fatal line,
#   exit nonzero, no listener;
# - the same config with the switches false starts successfully.
# Production startup really builds ONNX Runtime sessions, so the embed model
# paths here carry a tiny valid ONNX model (base64 below), not garbage bytes.
# ---------------------------------------------------------------------------

import base64
import socket
import time

_TINY_ONNX_B64 = (
    "CAo6UgoQCgF4EgF5IghJZGVudGl0eRIEdGlueVobCgF4EhYKFAgBEhAKAggBCgIIAQoCCAEKAggBYhsK"
    "AXkSFgoUCAESEAoCCAEKAggBCgIIAQoCCAFCBAoAEBA="
)

_TEST_PORT = 18777
_TEST_METRICS_PORT = 18778


def _production_config_with_routes(tmp_path: Path, *, describe_enabled: bool, tag_enabled: bool, tag_files: bool) -> Path:
    """Production config with the v12 additive keys; embed model files exist."""
    token = write_token(tmp_path)
    cfg = write_config(
        tmp_path,
        adapter="production",
        adapter_mode="production",
        paths_exist=True,
        tls_enabled="true",
        port=_TEST_PORT,
        metrics_port=_TEST_METRICS_PORT,
    )
    tiny = base64.b64decode(_TINY_ONNX_B64)
    for name in ("det_10g.onnx", "w600k_r50.onnx", "openai-ViT-L-14-visual.onnx", "yolov8n.onnx"):
        (Path(tmp_path) / name).write_bytes(tiny)
    florence_dir = Path(tmp_path) / "florence2"
    florence_dir.mkdir(exist_ok=True)
    text = cfg.read_text()
    tagger_lines = (
        f'  tagger_onnx_path: "{tmp_path / "wd14.onnx"}"\n'
        f'  tag_names_path: "{tmp_path / "tag_names.txt"}"\n'
        f'  tag_suppress_path: "{tmp_path / "tag_suppress.txt"}"\n'
    )
    if tag_files:
        (Path(tmp_path) / "wd14.onnx").write_bytes(tiny)
        (Path(tmp_path) / "tag_names.txt").write_text("tag0\n")
        (Path(tmp_path) / "tag_suppress.txt").write_text("")
    text = text.replace(
        '  execution_provider: "CPUExecutionProvider"\n',
        f'  execution_provider: "CPUExecutionProvider"\n'
        f'  florence_dir_path: "{florence_dir}"\n'
        + tagger_lines
        + "  max_new_tokens: 256\n",
    )
    text += (
        "routes:\n"
        f"  describe_enabled: {str(describe_enabled).lower()}\n"
        f"  tag_enabled: {str(tag_enabled).lower()}\n"
    )
    cfg.write_text(text, encoding="utf-8")
    assert token  # token file exists for auth file validation
    return cfg


def _fatal_assertions(proc: subprocess.CompletedProcess, port: int) -> dict:
    assert proc.returncode != 0, f"expected nonzero exit, got {proc.returncode}: stdout={proc.stdout!r}"
    fatal_lines = [
        ln for ln in proc.stdout.splitlines()
        if ln.startswith("{") and json.loads(ln).get("fatal") is True
    ]
    assert len(fatal_lines) == 1, f"expected exactly one fatal line, got {len(fatal_lines)}: {proc.stdout!r}"
    payload = json.loads(fatal_lines[0])
    assert payload["fatal"] is True
    assert payload["reason"] == "config.invalid"
    # no listener: the process exited and the port is not accepting
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        sock.close()
        raise AssertionError(f"listener is still up on {port} after fatal startup")
    except (ConnectionRefusedError, OSError):
        pass
    return payload


def _starts_successfully(config_path: Path, port: int) -> None:
    """The server must come up: process alive and port accepting, then kill it."""
    proc = subprocess.Popen(
        [PYTHON, "-m", "vision_embed", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO),
    )
    try:
        deadline = time.monotonic() + 30
        listening = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"startup exited {proc.returncode} instead of listening: stdout={_pipe_text(proc.stdout)!r} stderr={_pipe_text(proc.stderr)!r}"
                )
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                sock.close()
                listening = True
                break
            except OSError:
                time.sleep(0.25)
        assert listening, f"server never opened a listener on {port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _pipe_text(stream) -> str:
    if stream is None:
        return ""
    return stream.read() if not stream.closed else ""


def test_production_describe_enabled_missing_florence_dir_is_fatal(tmp_path):
    cfg = _production_config_with_routes(tmp_path, describe_enabled=True, tag_enabled=False, tag_files=False)
    (Path(tmp_path) / "florence2").rmdir()
    proc = run_entry(cfg)
    payload = _fatal_assertions(proc, _TEST_PORT)
    assert "florence_dir_path" in payload["detail"]
    assert "listener" not in json.dumps(payload)


def test_production_tag_enabled_missing_tagger_files_is_fatal(tmp_path):
    cfg = _production_config_with_routes(tmp_path, describe_enabled=False, tag_enabled=False, tag_files=False)
    # rebuild with only the tag switch on; tagger files stay absent
    text = cfg.read_text().replace("  tag_enabled: false", "  tag_enabled: true")
    cfg.write_text(text, encoding="utf-8")
    proc = run_entry(cfg)
    payload = _fatal_assertions(proc, _TEST_PORT)
    assert "tagger_onnx_path" in payload["detail"] or "tag_names_path" in payload["detail"] or "tag_suppress_path" in payload["detail"]


def test_production_switches_false_starts_successfully(tmp_path):
    cfg = _production_config_with_routes(tmp_path, describe_enabled=False, tag_enabled=False, tag_files=False)
    _starts_successfully(cfg, _TEST_PORT)
