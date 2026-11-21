import io
import json
import pathlib
import subprocess
import sys
import typing
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
PYTHON = str(REPO / ".venv" / "bin" / "python")
METRICS_PORT = 19171


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (180, 30, 30))
    img.save(buf, format=fmt)
    return buf.getvalue()


def write_token(dir_path: Path, token: str = "test-token-abc123") -> str:
    p = dir_path / "authz.bearer"
    p.write_text(token + "\n", encoding="utf-8")
    return str(p)


def write_config(
    dir_path: Path,
    *,
    adapter: str = "fake",
    adapter_mode: str = "test",
    paths_exist: bool = False,
    extra: str = "",
    drop: str = "",
    tls_enabled: str = "false",
    exec_provider: str = "CPUExecutionProvider",
    port: int = 8443,
    metrics_port: int = METRICS_PORT,
) -> Path:
    d = {
        "det_path": str(dir_path / "det_10g.onnx"),
        "face_embed_path": str(dir_path / "w600k_r50.onnx"),
        "scene_embed_path": str(dir_path / "openai-ViT-L-14-visual.onnx"),
        "object_det_path": str(dir_path / "yolov8n.onnx"),
    }
    if paths_exist:
        for name, data in [
            ("det_10g.onnx", b"x"),
            ("w600k_r50.onnx", b"x"),
            ("openai-ViT-L-14-visual.onnx", b"x"),
            ("yolov8n.onnx", b"x"),
        ]:
            (dir_path / name).write_bytes(data)
    lines = [
        "listen:",
        "  host: \"127.0.0.1\"",
        f"  port: {port}",
        "  tls:",
        f"    enabled: {tls_enabled}",
        f"    cert_path: \"{dir_path / 'cert.pem'}\"",
        f"    key_path: \"{dir_path / 'key.pem'}\"",
        "auth:",
        f"  bearer_token_path: \"{dir_path / 'authz.bearer'}\"",
        "models:",
        f"  det_path: \"{d['det_path']}\"",
        f"  face_embed_path: \"{d['face_embed_path']}\"",
        f"  scene_embed_path: \"{d['scene_embed_path']}\"",
        f"  object_det_path: \"{d['object_det_path']}\"",
        "  det_score_min: 0.5",
        "  object_score_min: 0.35",
        "  max_faces: 20",
        f"  execution_provider: \"{exec_provider}\"",
        f"  adapter: \"{adapter}\"",
        f"adapter_mode: \"{adapter_mode}\"",
        "limits:",
        "  max_body_bytes: 33554432",
        "metrics:",
        "  enabled: true",
        f"  port: {metrics_port}",
        "audit:",
        "  enabled: true",
        "request_id:",
        "  header: \"X-Request-Id\"",
    ]
    text = "\n".join(lines) + "\n"
    if drop:
        text = "\n".join(
            ln for ln in text.split("\n") if drop not in ln
        ) + "\n"
    text += extra
    if tls_enabled == "true":
        # Production mode fail-closed (SPEC §4/§6): TLS must be enabled with a
        # loadable cert/key pair, so generate a real self-signed pair.
        cert = dir_path / "cert.pem"
        key = dir_path / "key.pem"
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(cert), "-days", "2",
                "-nodes", "-subj", "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        text = text.replace(str(dir_path / "cert.pem"), str(cert))
        text = text.replace(str(dir_path / "key.pem"), str(key))
    cfg = dir_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    return cfg


from starlette.testclient import TestClient

from contextlib import redirect_stdout

from vision_embed.app import build_from_config_path

TOKEN = "test-token-abc123"


def boot(config_path: Path, token: str = "test-token-abc123") -> tuple[TestClient, typing.Any]:
    """Build the app from a config path; caller supplies its own stdout redirect."""
    app, cfg = build_from_config_path(config_path)
    client = TestClient(app, headers={"Authorization": f"Bearer {token}"})
    return client, cfg
