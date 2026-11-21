"""Tests 1-3, 5 — config fail-closed (SPEC section 13)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

from tests.index import helpers_index as H

PYTHON = str(H.REPO / ".venv" / "bin" / "python") if (H.REPO / ".venv" / "bin" / "python").exists() else sys.executable


def run_main(config_path, tmp_path):
    return subprocess.run(
        [PYTHON, "-m", "vision_index", "--config", str(config_path)],
        capture_output=True, text=True, timeout=30, cwd=str(H.REPO),
        env={"PYTHONPATH": str(H.REPO), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


def test_unknown_key_is_fatal(tmp_path):
    config = H.write_config(tmp_path, extra="clustering:\n  bogus_metric: 1\n")
    result = run_main(config, tmp_path)
    assert result.returncode != 0, result.stdout
    assert '"reason":"config.invalid"' in result.stdout
    assert "bogus_metric" in result.stdout


def test_unknown_top_level_key_is_fatal(tmp_path):
    config = H.write_config(tmp_path, extra="tenancy:\n  keyspace: x\n")
    result = run_main(config, tmp_path)
    assert result.returncode != 0
    assert '"reason":"config.invalid"' in result.stdout


def test_missing_auth_bearer_is_fatal(tmp_path):
    # auth bearer file path points at a nonexistent file → fatal
    config = H.write_config(tmp_path, token_path=str(tmp_path / "does-not-exist.bearer"))
    result = run_main(config, tmp_path)
    assert result.returncode != 0, result.stdout


def test_missing_embed_bearer_is_fatal(tmp_path):
    config = H.write_config(tmp_path, embed_token_path=str(tmp_path / "no-embed.bearer"))
    result = run_main(config, tmp_path)
    assert result.returncode != 0, result.stdout


def test_missing_embed_base_url_is_fatal(tmp_path):
    config = H.write_config(tmp_path)
    text = "\n".join(
        ln for ln in config.read_text().splitlines() if "base_url" not in ln
    )
    config.write_text(text + "\n")
    result = run_main(config, tmp_path)
    assert result.returncode != 0, result.stdout


def test_adapter_mode_test_with_tls_is_fatal(tmp_path):
    import shutil
    import subprocess as sp

    d = tmp_path / "tlsmode"
    d.mkdir()
    config = H.write_config(d, adapter_mode="test", tls_enabled=True)
    if shutil.which("openssl"):
        sp.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(d / "key.pem"),
             "-out", str(d / "cert.pem"), "-days", "1", "-nodes", "-subj", "/CN=localhost"],
            check=True, capture_output=True,
        )
    result = run_main(config, tmp_path)
    assert result.returncode != 0
    assert '"reason":"config.invalid"' in result.stdout


def test_production_without_tls_is_fatal(tmp_path):
    import shutil
    import subprocess as sp

    d = pathlib.Path(str(tmp_path)) / "prod"
    d.mkdir()
    config = H.write_config(d, adapter_mode="production", tls_enabled=False)
    result = run_main(config, tmp_path)
    assert result.returncode != 0
    assert '"reason":"config.invalid"' in result.stdout
    assert "TLS" in result.stdout


def test_fake_only_constructible_in_test_mode():
    from vision_index.adapters.fake import FakeEmbedClient

    try:
        FakeEmbedClient(mode="production")
        raise AssertionError("fake client must refuse production mode")
    except RuntimeError:
        pass
    FakeEmbedClient(mode="test")  # fine


def test_production_build_path_never_constructs_fake(monkeypatch, tmp_path):
    import dataclasses
    import pathlib as pl

    import vision_index.app as app_mod
    from vision_index.adapters.http_client import ProductionEmbedClient
    from vision_index.config import load_config

    d = pl.Path(str(tmp_path)) / "prod_cfg"
    d.mkdir(exist_ok=True)
    cfg = load_config(H.write_config(d, adapter_mode="test", tls_enabled=False))
    # the production branch requires a production-mode config object
    cfg = cfg.__class__(
        **{**cfg.__dict__, "adapter_mode": "production", "tls_enabled": True}
    )
    called = {"fake": False}

    def boom(*a, **kw):
        called["fake"] = True
        raise AssertionError("fake client constructed in production path")

    monkeypatch.setattr(app_mod, "FakeEmbedClient", boom)
    client_obj = app_mod._build_embed_client(cfg)
    assert called["fake"] is False
    assert isinstance(client_obj, ProductionEmbedClient)
