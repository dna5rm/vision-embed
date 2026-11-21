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
