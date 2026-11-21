"""App construction and boot (SPEC sections 3, 5, 6, 9, 10)."""

from __future__ import annotations

import hmac
import inspect
import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.responses import JSONResponse  # resolvable for OpenAPI ForwardRefs
from PIL import Image
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request
from starlette.datastructures import Headers

from .adapters import Detector, FaceEmbedder, ObjectDetector, SceneEmbedder
from .adapters import fake as fake_mod
from .adapters import onnx as onnx_mod
from .audit import emit_audit_line
from .config import Config, load_config
from .errors import ApiError, FatalStartupError
from . import metrics as metrics_mod
from .pipeline import embed_image, sniff_format

_PROVIDER_ALLOWLIST = ("CPUExecutionProvider", "CUDAExecutionProvider")


def provider_preflight(provider: str) -> None:
    """Provider-registered pre-flight (SPEC section 4 step 2).

    Runs before any weights load and before any listener or metrics bind. The
    activation check later in create_session is the ground truth for a device;
    this check only catches a wheel without the provider compiled in, giving
    one clean fatal instead of four identical session errors. Emits exactly one
    startup log line on both the success and the CPU path; no other
    provider-related logging exists.
    """
    import json

    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in available:
        raise FatalStartupError(
            "config.invalid",
            "execution_provider CUDAExecutionProvider requested but the provider is "
            "not registered in this onnxruntime build (available: "
            + ", ".join(available)
            + "); likely cause: onnxruntime-gpu is not installed",
        )
    print(
        json.dumps(
            {
                "event": "provider_preflight",
                "configured": provider,
                "registered": list(available),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_adapters(cfg: Config) -> dict[str, Any]:
    """F25: fake mode uses fakes for all four roles, never a hybrid."""
    if cfg.adapter == "fake":
        return {
            "detector": fake_mod.FakeDetector(),
            "face_embedder": fake_mod.FakeFaceEmbedder(),
            "scene_embedder": fake_mod.FakeSceneEmbedder(),
            "object_detector": fake_mod.FakeObjectDetector(),
        }
    provider_preflight(cfg.execution_provider)
    return {
        "detector": onnx_mod.OnnxDetector(cfg.execution_provider),
        "face_embedder": onnx_mod.OnnxFaceEmbedder(cfg.execution_provider),
        "scene_embedder": onnx_mod.OnnxSceneEmbedder(cfg.execution_provider),
        "object_detector": onnx_mod.OnnxObjectDetector(
            cfg.execution_provider, score_min=cfg.object_score_min
        ),
    }


def load_and_adapt(cfg: Config) -> dict[str, Any]:
    """Construct adapters and load their model files (production paths)."""
    adapters = build_adapters(cfg)
    loaders = {
        "detector": (adapters["detector"], Path(cfg.det_path)),
        "face_embedder": (adapters["face_embedder"], Path(cfg.face_embed_path)),
        "scene_embedder": (adapters["scene_embedder"], Path(cfg.scene_embed_path)),
        "object_detector": (adapters["object_detector"], Path(cfg.object_det_path)),
    }
    for role, (adapter, path) in loaders.items():
        try:
            adapter.load(path)
        except Exception as exc:
            raise FatalStartupError("config.invalid", f"failed to load {role} from {path}: {exc}") from exc
    metrics_mod.models_loaded.set(4)
    return adapters


def _effective_request_id(request: Request, header: str) -> str:
    raw = request.headers.get(header)
    if raw is not None and raw.isascii() and raw.isprintable() and 0 < len(raw) <= 128:
        return raw
    return str(uuid.uuid4())


def _error_body(reason: str, detail: str, request_id: str) -> dict:
    return {"error": {"reason": reason, "detail": detail, "request_id": request_id}}


def create_app(cfg: Config, preloaded_adapters: dict[str, Any] | None = None) -> Any:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    adapters: dict[str, Any] = {**preloaded_adapters} if preloaded_adapters else {}  # loaded in lifespan (or eager pre-bind)
    state = {"ready": False}
    already_loaded = bool(preloaded_adapters)

    @asynccontextmanager
    async def lifespan(app: Any):
        if not already_loaded:
            adapters.update(load_and_adapt(cfg))
        state["ready"] = True
        if cfg.metrics_enabled:
            metrics_mod.start_metrics_server(cfg.metrics_port)
        yield
        state["ready"] = False

    app = FastAPI(
        docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.cfg = cfg

    def _audit_http_line(request: Request, outcome: str, reason: str) -> None:
        emit_audit_line(
            request_id=_effective_request_id(request, cfg.request_id_header),
            sha256="",
            byte_count=0,
            faces=0,
            objects=0,
            outcome=outcome,
            reason=reason,
            enabled=cfg.audit_enabled,
        )

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)
        _audit_http_line(request, "ok", "ok")
        metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
        return JSONResponse({"status": "ok"}, headers={"X-Request-Id": request_id})

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)
        loaded = metrics_mod.models_loaded._value.get()
        if loaded != 4:
            metrics_mod.requests_total.labels(outcome="error", reason="not_ready").inc()
            metrics_mod.errors_total.labels(reason="not_ready").inc()
            _audit_http_line(request, "error", "not_ready")
            return JSONResponse(
                _error_body("not_ready", "models not loaded yet", request_id),
                status_code=503,
                headers={"X-Request-Id": request_id},
            )
        metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
        _audit_http_line(request, "ok", "ok")
        return JSONResponse(
            {"status": "ready", "models_loaded": 4},
            headers={"X-Request-Id": request_id},
        )

    @app.post("/v1/embed")
    async def v1_embed(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)
        sha = ""
        byte_count = 0
        faces = 0
        objects = 0

        def audit(outcome: str, reason: str) -> None:
            emit_audit_line(
                request_id=request_id,
                sha256=sha,
                byte_count=byte_count,
                faces=faces,
                objects=objects,
                outcome=outcome,
                reason=reason,
                enabled=cfg.audit_enabled,
            )

        try:
            with metrics_mod.latency_seconds.time():
                # Step 1: size limit — Content-Length pre-check before reading anything.
                content_length = request.headers.get("content-length")
                if content_length is not None and content_length.isdigit():
                    if int(content_length) > cfg.max_body_bytes:
                        raise ApiError(413, "limits.body_too_large", "body exceeds max_body_bytes")

                # Step 2: auth — rejected before the body is read or sniffed.
                header = request.headers.get("authorization", "")
                scheme, _, token = header.partition(" ")
                if scheme.lower() != "bearer" or not token or not hmac.compare_digest(
                    token, cfg.bearer_token
                ):
                    raise ApiError(401, "auth.invalid_token", "missing or invalid bearer token")

                # Step 3: magic-byte sniff — read the body with the size cap enforced.
                buf = bytearray()
                async for chunk in request.stream():
                    buf += chunk
                    if len(buf) > cfg.max_body_bytes:
                        raise ApiError(413, "limits.body_too_large", "body exceeds max_body_bytes")
                raw = bytes(buf)
                # Step 3: magic-byte sniff — on the image part bytes, after
                # extracting them from the multipart envelope.
                try:
                    form = await parse_multipart(request, raw)
                except MultiPartException as exc:
                    # Malformed multipart envelope (bad boundary, truncated part,
                    # cap exceeded) is a client payload problem, not a 500.
                    raise ApiError(422, "decode.failed", f"malformed multipart body: {exc}") from exc
                if "image" not in form:
                    raise ApiError(422, "decode.failed", "missing multipart field: image")
                image_part = form["image"]
                if hasattr(image_part, "read"):
                    raw = await image_part.read()
                else:
                    raw = str(image_part).encode()
                byte_count = len(raw)
                sha = _sha256_hex(raw)
                fmt = sniff_format(raw)
                if fmt is None:
                    raise ApiError(415, "format.unsupported", "unsupported image format (sniffed)")

                include_raw = form.get("include_object_vectors")
                if include_raw is None:
                    include_object_vectors = False
                elif include_raw == "true":
                    include_object_vectors = True
                elif include_raw == "false":
                    include_object_vectors = False
                else:
                    raise ApiError(422, "decode.failed", "invalid include_object_vectors")

                try:
                    rgb = np.asarray(
                        Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8
                    )
                except Exception as exc:
                    raise ApiError(422, "decode.failed", "image decode failed") from exc

                # Step 5: model inference.
                payload = embed_image(
                    raw=raw,
                    rgb=rgb,
                    detector=adapters["detector"],
                    face_embedder=adapters["face_embedder"],
                    scene_embedder=adapters["scene_embedder"],
                    object_detector=adapters["object_detector"],
                    det_score_min=cfg.det_score_min,
                    object_score_min=cfg.object_score_min,
                    max_faces=cfg.max_faces,
                    include_object_vectors=include_object_vectors,
                    request_id=request_id,
                )
                payload["request_id"] = request_id
                faces = len(payload["faces"])
                objects = len(payload["objects"])

                # Step 7: audit (one line, stdout), then Step 8: body.
                audit("ok", "ok")
                metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
                return JSONResponse(payload, status_code=200, headers={"X-Request-Id": request_id})
        except ApiError as err:
            audit("error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return JSONResponse(
                _error_body(err.reason, err.detail, request_id),
                status_code=err.status,
                headers={"X-Request-Id": request_id},
            )

    async def parse_multipart(request: Request, raw: bytes) -> dict:
        content_type = request.headers.get("content-type", "") or "multipart/form-data"

        async def stream() -> Any:  # pragma: no cover - shape only
            yield raw

        # starlette >=0.41 dropped/added kwargs across versions: pass
        # max_part_size only when this starlette supports it (body cap F7 is
        # also enforced earlier via limits.max_body_bytes).
        parser_kwargs: dict[str, Any] = {"max_files": 100, "max_fields": 100}
        if "max_part_size" in inspect.signature(MultiPartParser.__init__).parameters:
            parser_kwargs["max_part_size"] = cfg.max_body_bytes
        parser = MultiPartParser(
            headers=Headers({"content-type": content_type}),
            stream=stream(),
            # F7: extra form fields are ignored, not rejected. Caps are set well
            # above legitimate use so a form with >10 harmless fields parses;
            # exceeding these is treated as malformed multipart (422 decode.failed).
            **parser_kwargs,
        )
        form = await parser.parse()
        return dict(form)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)
        if exc.status_code == 405:
            reason = "method.not_allowed"
        else:
            reason = "route.not_found"
        metrics_mod.requests_total.labels(outcome="error", reason=reason).inc()
        metrics_mod.errors_total.labels(reason=reason).inc()
        # Ruling 2: one audit line per HTTP request — the 404/405 closed-surface
        # responses are part of that stream (sha256 "" / bytes 0).
        _audit_http_line(request, "error", reason)
        return JSONResponse(
            _error_body(reason, "closed endpoint set", request_id),
            status_code=exc.status_code,
            headers={"X-Request-Id": request_id},
        )

    return app


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def build_from_config_path(config_path: Path) -> tuple[Any, Config]:
    cfg = load_config(config_path)
    return create_app(cfg), cfg


def run_server(config_path: Path) -> None:
    import uvicorn

    cfg = load_config(config_path)
    # Fail fatal BEFORE any listener or metrics port binds (SPEC cuda section 4
    # step 3): eager adapter load here, uvicorn only serves afterwards. The
    # lifespan reuses the preloaded adapters instead of loading twice.
    adapters = load_and_adapt(cfg)
    app = create_app(cfg, preloaded_adapters=adapters)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        workers=1,
        ssl_certfile=cfg.tls_cert_path if cfg.tls_enabled else None,
        ssl_keyfile=cfg.tls_key_path if cfg.tls_enabled else None,
        access_log=False,
        log_level="warning",
    )
