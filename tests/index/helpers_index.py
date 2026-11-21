import io
import json
import pathlib
import sys
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

# tests live at tests/index/; the package is at repo root
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

TOKEN = "index-test-token-abc123"


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (180, 30, 30))
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_text_bytes() -> bytes:
    return b"plain text, not an image"


def write_token(dir_path, token: str = TOKEN) -> str:
    p = dir_path / "index_authz.bearer"
    p.write_text(token + "\n", encoding="utf-8")
    return str(p)


def write_config(
    dir_path,
    *,
    adapter_mode: str = "test",
    tls_enabled: bool = False,
    token_path: str | None = None,
    embed_token_path: str | None = None,
    drop: str = "",
    extra: str = "",
    metrics_port: int = 19177,
) -> Path:
    token_path = token_path or write_token(dir_path)
    embed_token_path = embed_token_path or write_token(dir_path)
    db_path = dir_path / "index.sqlite3"
    lines = [
        'listen:',
        '  host: "127.0.0.1"',
        '  port: 9443',
        '  tls:',
        f'    enabled: {"true" if tls_enabled else "false"}',
        f'    cert_path: "{dir_path / "cert.pem"}"',
        f'    key_path: "{dir_path / "key.pem"}"',
        'auth:',
        f'  bearer_token_path: "{token_path}"',
        'embed:',
        '  base_url: "https://192.0.2.10:8443"',
        f'  bearer_token_path: "{embed_token_path}"',
        '  timeout_seconds: 1',
        'db:',
        f'  path: "{db_path}"',
        'clustering:',
        '  face_match_max: 0.40',
        'name:',
        '  min_length: 1',
        '  max_length: 64',
        'metrics:',
        '  enabled: false',
        f'  port: {metrics_port}',
        '  host: "127.0.0.1"',
        'audit:',
        '  enabled: true',
        f'adapter_mode: "{adapter_mode}"',
    ]
    text = "\n".join(lines) + "\n"
    if drop:
        text = "\n".join(ln for ln in text.split("\n") if drop not in ln) + "\n"
    text += extra
    cfg = dir_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    return cfg


def make_fake(client_cls, **kwargs):
    from vision_index.adapters.fake import FakeEmbedClient

    kwargs.setdefault("mode", "test")
    return FakeEmbedClient(**kwargs)


def build_client(tmp_path, fake, token: str = TOKEN, **_ignored):
    from vision_index.app import build_app
    from vision_index.config import load_config

    cfg = load_config(write_config(tmp_path, adapter_mode="test"))
    app = build_app(cfg, embed_client=fake)
    return TestClient(app, headers={"Authorization": f"Bearer {token}"})


def sha_of(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def ingest(client, data: bytes, path: str | None = None):
    files = {"image": ("img.png", data, "image/png")}
    form = {} if path is None else {"path": path}
    return client.post("/v1/ingest", files=files, data=form)


def audit_lines(capsys) -> list[dict]:
    from vision_index.audit import AUDIT_KEYS

    out = capsys.readouterr().out
    lines = []
    for line in out.splitlines():
        if line.strip().startswith("{"):
            rec = json.loads(line)
            assert set(rec) == AUDIT_KEYS, f"audit keys: {set(rec)}"
            lines.append(rec)
    return lines
