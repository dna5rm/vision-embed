"""Config loading — strictly fail closed (SPEC section 4).

Unknown top-level or nested keys are fatal (config.invalid). Missing or
unreadable bearer-token files are fatal. Absent embed.base_url is fatal.
adapter_mode is only "production" or "test"; "test" with TLS enabled is fatal.
No secrets are ever held or logged — only token file paths, read at startup.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import FatalStartupError

# Closed config schema (section 4). Anything not listed is an unknown key.
_TOP_KEYS = {
    "listen",
    "auth",
    "embed",
    "db",
    "clustering",
    "name",
    "metrics",
    "audit",
    "adapter_mode",
}
_LISTEN_KEYS = {"host", "port", "tls"}
_TLS_KEYS = {"enabled", "cert_path", "key_path"}
_AUTH_KEYS = {"bearer_token_path"}
_EMBED_KEYS = {"base_url", "bearer_token_path", "timeout_seconds"}
_DB_KEYS = {"path"}
_CLUSTERING_KEYS = {"face_match_max"}
_NAME_KEYS = {"min_length", "max_length"}
_METRICS_KEYS = {"enabled", "port", "host"}
_AUDIT_KEYS = {"enabled"}

BODY_LIMIT_BYTES = 32 * 1024 * 1024  # 32 MiB, fixed, not config (section 6b step 1)


def _check_unknown_keys(cfg: dict[str, Any]) -> None:
    # (display path, allowed keys at that node); the node itself is walked.
    checks: list[tuple[list[str], set]] = [
        ([], _TOP_KEYS),
        (["listen"], _LISTEN_KEYS),
        (["listen", "tls"], _TLS_KEYS),
        (["auth"], _AUTH_KEYS),
        (["embed"], _EMBED_KEYS),
        (["db"], _DB_KEYS),
        (["clustering"], _CLUSTERING_KEYS),
        (["name"], _NAME_KEYS),
        (["metrics"], _METRICS_KEYS),
        (["audit"], _AUDIT_KEYS),
    ]
    for path, allowed in checks:
        node: Any = cfg
        for part in path:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is None:
            continue  # optional section absent; requiredness checked later
        if not isinstance(node, dict):
            raise FatalStartupError(
                f"invalid yaml node under {'.'.join(path)}"
            )
        unknown = set(node) - allowed
        if unknown:
            raise FatalStartupError(f"unknown key {sorted(unknown)[0]!r}") from None


@dataclass(frozen=True)
class IndexConfig:
    host: str
    port: int
    tls_enabled: bool
    tls_cert_path: str | None
    tls_key_path: str | None
    bearer_token_path: str
    bearer_token: str  # read once at startup; never logged
    embed_base_url: str
    embed_bearer_token: str  # read once at startup; never logged
    embed_timeout_seconds: int
    db_path: str
    face_match_max: float
    name_min_length: int
    name_max_length: int
    metrics_enabled: bool
    metrics_port: int
    metrics_host: str
    audit_enabled: bool
    adapter_mode: str


def _read_token_file(path_str: str) -> str:
    """One line in the file is the token; read once at startup (section 4)."""
    path = Path(path_str)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FatalStartupError(
            f"cannot read bearer token file: {exc.__class__.__name__}"
        ) from None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise FatalStartupError("bearer token file must contain exactly one line")
    return lines[0]


def load_config(config_path: Path) -> IndexConfig:
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FatalStartupError(
            f"cannot read config file: {exc.__class__.__name__}"
        ) from None
    try:
        cfg = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise FatalStartupError(f"invalid yaml: {exc.__class__.__name__}") from None
    if not isinstance(cfg, dict):
        raise FatalStartupError("config must be a yaml mapping")

    _check_unknown_keys(cfg)

    listen = cfg.get("listen", {})
    tls = listen.get("tls", {})
    auth = cfg.get("auth", {})
    embed = cfg.get("embed", {})
    db = cfg.get("db", {})
    clustering = cfg.get("clustering", {})
    name = cfg.get("name", {})
    metrics = cfg.get("metrics", {})
    audit = cfg.get("audit", {})
    adapter_mode = cfg.get("adapter_mode")

    if adapter_mode not in ("production", "test"):
        raise FatalStartupError("adapter_mode must be 'production' or 'test'")
    tls_enabled = bool(tls.get("enabled", False))
    if adapter_mode == "test" and tls_enabled:
        raise FatalStartupError("adapter_mode 'test' with TLS enabled is fatal")
    if adapter_mode == "production" and not tls_enabled:
        raise FatalStartupError("adapter_mode 'production' requires TLS enabled")

    # Bearer files: read once at startup; missing/unreadable is fatal (section 5).
    bearer_path = auth.get("bearer_token_path")
    if not isinstance(bearer_path, str) or not bearer_path:
        raise FatalStartupError("auth.bearer_token_path is required")
    bearer_token = _read_token_file(bearer_path)

    embed_base_url = embed.get("base_url")
    if not isinstance(embed_base_url, str) or not embed_base_url:
        raise FatalStartupError("embed.base_url is required and has no default")
    embed_bearer_path = embed.get("bearer_token_path")
    if not isinstance(embed_bearer_path, str) or not embed_bearer_path:
        raise FatalStartupError("embed.bearer_token_path is required")
    embed_bearer_token = _read_token_file(embed_bearer_path)

    db_path = db.get("path")
    if not isinstance(db_path, str) or not db_path:
        raise FatalStartupError("db.path is required")

    try:
        face_match_max = float(clustering.get("face_match_max", 0.40))
    except (TypeError, ValueError):
        raise FatalStartupError("clustering.face_match_max must be a number") from None
    if not (0.0 <= face_match_max < 1.0):
        raise FatalStartupError("clustering.face_match_max out of range")

    name_min_length = int(name.get("min_length", 1))
    name_max_length = int(name.get("max_length", 64))
    if name_min_length < 1 or name_max_length < name_min_length:
        raise FatalStartupError("name.max_length must be >= name.min_length >= 1")

    if tls_enabled:
        cert = Path(str(tls.get("cert_path", "")))
        key = Path(str(tls.get("key_path", "")))
        if not cert.is_file() or not key.is_file():
            raise FatalStartupError("TLS enabled but cert/key file missing")

    return IndexConfig(
        host=str(listen.get("host", "0.0.0.0")),
        port=int(listen.get("port", 9443)),
        tls_enabled=tls_enabled,
        tls_cert_path=str(tls["cert_path"]) if tlsEnabled_cert(tls) else None,
        tls_key_path=str(tls["key_path"]) if tlsEnabled_cert(tls) else None,
        bearer_token_path=bearer_path,
        bearer_token=bearer_token,
        embed_base_url=str(embed_base_url),
        embed_bearer_token=embed_bearer_token,
        embed_timeout_seconds=int(embed.get("timeout_seconds", 60)),
        db_path=db_path,
        face_match_max=face_match_max,
        name_min_length=name_min_length,
        name_max_length=name_max_length,
        metrics_enabled=bool(metrics.get("enabled", True)),
        metrics_port=int(metrics.get("port", 9190)),
        metrics_host=str(metrics.get("host", "127.0.0.1")),
        audit_enabled=bool(audit.get("enabled", True)),
        adapter_mode=adapter_mode,
    )


def tlsEnabled_cert(tls: dict[str, Any]) -> bool:
    return bool(tls.get("enabled")) and "cert_path" in tls and "key_path" in tls


def constant_time_token_eq(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
