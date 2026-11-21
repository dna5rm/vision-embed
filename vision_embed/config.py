"""Config loading — strictly fail closed (SPEC section 4).

Unknown YAML keys are fatal (config.unknown_key). Invalid values are fatal
(config.invalid). Missing model files or the bearer-token file are fatal in
production mode; fake+test mode intentionally starts with zero ONNX files.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import FatalStartupError

# Closed config schema. Anything not listed is an unknown key.
_TOP_KEYS = {
    "listen",
    "auth",
    "models",
    "limits",
    "metrics",
    "audit",
    "adapter_mode",
    "request_id",
    "routes",
}
_LISTEN_KEYS = {"host", "port", "tls"}
_TLS_KEYS = {"enabled", "cert_path", "key_path"}
_AUTH_KEYS = {"bearer_token_path"}
_MODELS_KEYS = {
    "det_path",
    "face_embed_path",
    "scene_embed_path",
    "object_det_path",
    "det_score_min",
    "object_score_min",
    "max_faces",
    "execution_provider",
    "adapter",
    "florence_dir_path",
    "tagger_onnx_path",
    "tag_names_path",
    "tag_suppress_path",
    "max_new_tokens",
}
_LIMITS_KEYS = {"max_body_bytes", "allowed_mime", "max_regions"}
_METRICS_KEYS = {"enabled", "port"}
_AUDIT_KEYS = {"enabled"}
_REQUEST_ID_KEYS = {"header"}
_ROUTES_KEYS = {
    "describe_enabled",
    "tag_enabled",
    "describe_min_free_bytes",
    "tag_min_free_bytes",
}

_MODEL_FILES = ("det_path", "face_embed_path", "scene_embed_path", "object_det_path")


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    tls_enabled: bool
    tls_cert_path: str
    tls_key_path: str
    bearer_token: str  # read once at startup; never logged
    det_path: str
    face_embed_path: str
    scene_embed_path: str
    object_det_path: str
    det_score_min: float
    object_score_min: float
    max_faces: int
    execution_provider: str
    adapter: str  # "production" | "fake"
    adapter_mode: str  # "production" | "test"
    max_body_bytes: int
    allowed_mime: list[str] = field(default_factory=list)
    metrics_enabled: bool = True
    metrics_port: int = 9090
    audit_enabled: bool = True
    request_id_header: str = "X-Request-Id"
    # v12 additive keys (describe/tag routes). Switches default false.
    florence_dir_path: str = ""
    tagger_onnx_path: str = ""
    tag_names_path: str = ""
    tag_suppress_path: str = ""
    max_new_tokens: int = 1024
    describe_enabled: bool = False
    tag_enabled: bool = False
    describe_min_free_bytes: int | None = None  # no default is ever invented
    tag_min_free_bytes: int | None = None
    max_regions: int = 100


def _unknown_keys(d: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise FatalStartupError("config.unknown_key", f"unknown keys under {where}: {sorted(extra)}")


def load_config(path: Path) -> Config:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise FatalStartupError("config.invalid", f"cannot read config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise FatalStartupError("config.invalid", f"cannot parse config: {exc}") from exc
    if not isinstance(raw, dict):
        raise FatalStartupError("config.invalid", "config root must be a YAML mapping")

    _unknown_keys(raw, _TOP_KEYS, "top-level")

    listen = raw.get("listen") or {}
    _unknown_keys(listen, _LISTEN_KEYS, "listen")
    tls = listen.get("tls") or {}
    _unknown_keys(tls, _TLS_KEYS, "listen.tls")

    auth = raw.get("auth") or {}
    _unknown_keys(auth, _AUTH_KEYS, "auth")

    models = raw.get("models") or {}
    _unknown_keys(models, _MODELS_KEYS, "models")

    limits = raw.get("limits") or {}
    _unknown_keys(limits, _LIMITS_KEYS, "limits")

    metrics = raw.get("metrics") or {}
    _unknown_keys(metrics, _METRICS_KEYS, "metrics")

    audit = raw.get("audit") or {}
    _unknown_keys(audit, _AUDIT_KEYS, "audit")

    request_id = raw.get("request_id") or {}
    _unknown_keys(request_id, _REQUEST_ID_KEYS, "request_id")

    # F25: top-level adapter_mode, default production.
    adapter_mode = raw.get("adapter_mode", "production")
    if adapter_mode not in ("production", "test"):
        raise FatalStartupError("config.invalid", f"adapter_mode must be production or test, got {adapter_mode!r}")

    # F25: models.adapter is required, production or fake.
    if "adapter" not in models:
        raise FatalStartupError("config.invalid", "models.adapter is required")
    adapter = models["adapter"]
    if adapter not in ("production", "fake"):
        raise FatalStartupError("config.invalid", f"models.adapter must be production or fake, got {adapter!r}")

    # F25/F26: fake adapter may only run in test mode.
    if adapter == "fake" and adapter_mode == "production":
        raise FatalStartupError("config.invalid", "models.adapter=fake requires adapter_mode=test")

    # Closed two-value allowlist (SPEC section 3).
    execution_provider = models.get("execution_provider", "CPUExecutionProvider")
    if execution_provider not in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        raise FatalStartupError(
            "config.invalid",
            f"execution_provider must be CPUExecutionProvider or CUDAExecutionProvider, got {execution_provider!r}",
        )

    host = str(listen.get("host", "0.0.0.0"))
    port = int(listen.get("port", 8443))
    tls_enabled = bool(tls.get("enabled", False))

    # Secrets are file paths, read once at startup (SPEC section 4/5).
    token_path = auth.get("bearer_token_path")
    if not token_path:
        raise FatalStartupError("config.invalid", "auth.bearer_token_path is required")
    try:
        content = Path(token_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise FatalStartupError("config.invalid", f"cannot read bearer token file: {exc}") from exc
    bearer_token = content.split("\n", 1)[0].rstrip("\r")
    if not bearer_token:
        raise FatalStartupError("config.invalid", "bearer token file is empty")

    # F26/TLS fail-closed: production mode refuses to start without TLS, and the
    # cert/key must exist, be readable, and parse as a loadable PEM pair.
    if adapter_mode == "production":
        if not tls_enabled:
            raise FatalStartupError(
                "config.invalid", "listen.tls.enabled=false is not allowed in production mode"
            )
        for label, key in (("cert_path", "listen.tls.cert_path"), ("key_path", "listen.tls.key_path")):
            p = tls.get(label)
            if not p:
                raise FatalStartupError("config.invalid", f"{key} is required in production mode")
            pp = Path(str(p))
            if not pp.is_file() or not os.access(pp, os.R_OK):
                raise FatalStartupError(
                    "config.invalid", f"{key} is not a readable file: {p}"
                )
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(tls["cert_path"]), str(tls["key_path"]))
        except (ssl.SSLError, OSError) as exc:
            raise FatalStartupError(
                "config.invalid", f"cannot load TLS cert/key pair: {exc}"
            ) from exc

    required_model_keys = [k for k in _MODEL_FILES]
    if adapter == "production":
        # F23: a missing model file is fatal — never a 200 with empty arrays.
        for key in required_model_keys:
            if key not in models:
                raise FatalStartupError("config.invalid", f"models.{key} is required")
            if not Path(models[key]).is_file():
                raise FatalStartupError("config.invalid", f"model file missing at models.{key}: {models[key]}")
    else:
        for key in required_model_keys:
            if key not in models:
                raise FatalStartupError("config.invalid", f"models.{key} is required")

    det_score_min = float(models.get("det_score_min", 0.5))
    object_score_min = float(models.get("object_score_min", 0.35))
    max_faces = int(models.get("max_faces", 20))
    max_body_bytes = int(limits.get("max_body_bytes", 33554432))
    allowed_mime = list(limits.get("allowed_mime", ["image/jpeg", "image/png", "image/webp"]))
    max_regions = int(limits.get("max_regions", 100))

    # v12 routes block: unknown keys fatal; switch defaults false; minima have
    # NO default — presence is only *required* under the exact two-condition
    # rule, which the server bootstrap (run_server) enforces.
    routes = raw.get("routes") or {}
    _unknown_keys(routes, _ROUTES_KEYS, "routes")

    def _switch(name: str) -> bool:
        val = routes.get(name, False)
        if val not in (True, False):
            raise FatalStartupError("config.invalid", f"routes.{name} must be a boolean, got {val!r}")
        return val

    describe_enabled = _switch("describe_enabled")
    tag_enabled = _switch("tag_enabled")

    def _minimum(name: str) -> int | None:
        if name not in routes:
            return None
        val = routes[name]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise FatalStartupError(
                "config.invalid", f"routes.{name} must be a non-negative integer byte count"
            )
        return val

    florence_dir_path = str(models.get("florence_dir_path", ""))
    tagger_onnx_path = str(models.get("tagger_onnx_path", ""))
    tag_names_path = str(models.get("tag_names_path", ""))
    tag_suppress_path = str(models.get("tag_suppress_path", ""))
    max_new_tokens = int(models.get("max_new_tokens", 1024))

    return Config(
        host=host,
        port=port,
        tls_enabled=tls_enabled,
        tls_cert_path=str(tls.get("cert_path", "")),
        tls_key_path=str(tls.get("key_path", "")),
        bearer_token=bearer_token,
        det_path=str(models["det_path"]),
        face_embed_path=str(models["face_embed_path"]),
        scene_embed_path=str(models["scene_embed_path"]),
        object_det_path=str(models["object_det_path"]),
        det_score_min=det_score_min,
        object_score_min=object_score_min,
        max_faces=max_faces,
        execution_provider=execution_provider,
        adapter=adapter,
        adapter_mode=adapter_mode,
        max_body_bytes=max_body_bytes,
        allowed_mime=allowed_mime,
        metrics_enabled=bool(metrics.get("enabled", True)),
        metrics_port=int(metrics.get("port", 9090)),
        audit_enabled=bool(audit.get("enabled", True)),
        request_id_header=str(request_id.get("header", "X-Request-Id")),
        florence_dir_path=florence_dir_path,
        tagger_onnx_path=tagger_onnx_path,
        tag_names_path=tag_names_path,
        tag_suppress_path=tag_suppress_path,
        max_new_tokens=max_new_tokens,
        describe_enabled=describe_enabled,
        tag_enabled=tag_enabled,
        describe_min_free_bytes=_minimum("describe_min_free_bytes"),
        tag_min_free_bytes=_minimum("tag_min_free_bytes"),
        max_regions=max_regions,
    )


def validate_route_minima(cfg: Config) -> None:
    """v12 section 2e: the minima keys are REQUIRED exactly when their route's
    switch is true AND models.execution_provider is CUDAExecutionProvider.
    Enforced in the server bootstrap, before any listener binds; nothing is
    validated or required in any other state, and no byte default exists.
    """
    if cfg.execution_provider != "CUDAExecutionProvider":
        return
    for switch_key, enabled, min_key in (
        ("routes.describe_enabled", cfg.describe_enabled, "describe_min_free_bytes"),
        ("routes.tag_enabled", cfg.tag_enabled, "tag_min_free_bytes"),
    ):
        if enabled and getattr(cfg, min_key) is None:
            raise FatalStartupError(
                "config.invalid",
                f"routes.{min_key} is required when {switch_key} is true and "
                "models.execution_provider is CUDAExecutionProvider",
            )


def fatal_line(reason: str, detail: str) -> str:
    """F26: the single stdout fatal JSON line."""
    import json

    return json.dumps({"fatal": True, "reason": reason, "detail": detail}, separators=(",", ":"))
