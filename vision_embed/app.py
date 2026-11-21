"""App construction and boot (SPEC sections 3, 5, 6, 9, 10; the v12 additive
describe/tag/runtime contract of docs/spec/vision-describe-SPEC.md)."""

from __future__ import annotations

import contextlib
import hmac
import inspect
import io
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.responses import JSONResponse  # resolvable for OpenAPI ForwardRefs
from PIL import Image
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request
from starlette.datastructures import FormData, Headers

from .adapters import Detector, FaceEmbedder, ObjectDetector, SceneEmbedder
from .adapters import fake as fake_mod
from .adapters import onnx as onnx_mod
from .adapters import real_describe as real_describe_mod
from .audit import emit_audit_line, emit_describe_audit_line
from . import errors as errors_mod
from .config import Config, load_config
from .errors import ApiError, FatalStartupError
from . import metrics as metrics_mod
from . import memory_gate
from .pipeline import embed_image, sniff_format

_PROVIDER_ALLOWLIST = ("CPUExecutionProvider", "CUDAExecutionProvider")

WIRE_TASKS = frozenset(
    {
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
        "<OCR>",
        "<OCR_WITH_REGION>",
    }
)
_DESCRIBE_TEXT_TASKS = frozenset(
    {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>"}
)
# Region tasks are the complement: score only on <OD>, text only on OCR_WITH_REGION.

ROLE_ORDER = ("face", "face_detector", "scene", "object", "florence2", "tagger")
ROLE_MODEL_IDS = {
    "face": "buffalo_l/w600k_r50",
    "face_detector": "buffalo_l/det_10g",
    "scene": "ViT-L-14/openai",
    "object": "yolov8n",
    "florence2": "florence2",
    "tagger": "wd14",
}


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


def _build_florence(cfg: Config) -> Any:
    """Section 7: the Florence role behind its Protocol; fakes only when both
    fake flags select them (config validation already enforces fake+test)."""
    if cfg.adapter == "fake":
        return fake_mod.FakeFlorence()
    return real_describe_mod.RealFlorenceDescribe(cfg.execution_provider, cfg.max_new_tokens)


def _build_tagger(cfg: Config) -> Any:
    if cfg.adapter == "fake":
        return fake_mod.FakeTagger()
    return real_describe_mod.RealTagger(cfg.execution_provider, Path(cfg.tag_names_path))


def _read_suppressed(path: str) -> frozenset[str]:
    """Operator suppress data (section 2c): never hardcoded in code."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return frozenset()
    return frozenset(ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


def _describe_tag_minima(cfg: Config) -> dict[str, int]:
    """Routes whose 2e check applies: switch true AND the provider is CUDA.
    A route whose key is absent has no check to run and loads normally."""
    minima: dict[str, int] = {}
    if cfg.execution_provider != "CUDAExecutionProvider":
        return minima
    if cfg.describe_enabled and cfg.describe_min_free_bytes is not None:
        minima["describe"] = cfg.describe_min_free_bytes
    if cfg.tag_enabled and cfg.tag_min_free_bytes is not None:
        minima["tag"] = cfg.tag_min_free_bytes
    return minima


def _require_route_files(cfg: Config, route: str, gate_passed: bool) -> None:
    """Production fail-closed weight rule: a missing file is fatal ONLY while
    its route is enabled and was not soft-disabled (sections 2, 2e)."""
    if cfg.adapter != "production" or not gate_passed:
        return
    if route == "describe":
        if not cfg.florence_dir_path or not Path(cfg.florence_dir_path).is_dir():
            raise FatalStartupError(
                "config.invalid", f"models.florence_dir_path missing or not a directory: {cfg.florence_dir_path!r}"
            )
    else:
        for key, path in (
            ("tagger_onnx_path", cfg.tagger_onnx_path),
            ("tag_names_path", cfg.tag_names_path),
            ("tag_suppress_path", cfg.tag_suppress_path),
        ):
            if not path or not Path(path).is_file():
                raise FatalStartupError("config.invalid", f"models.{key} missing: {path!r}")


def build_role_state(cfg: Config, gate: dict[str, bool]) -> dict[str, tuple[bool, str]]:
    """Six fixed roles -> (loaded, unavailable_reason) for /v1/runtime (6d)."""
    insufficient = errors_mod.GPU_INSUFFICIENT_TOKEN
    state: dict[str, tuple[bool, str]] = {}
    for role in ("face", "face_detector", "scene", "object"):
        state[role] = (True, "")
    # describe/tag: switch false -> disabled; switch true but the 2e gate did
    # not pass the role -> gpu.insufficient; loaded -> "".
    if cfg.describe_enabled:
        passed = bool(gate.get("describe", True))
        state["florence2"] = (passed, "" if passed else insufficient)
    else:
        state["florence2"] = (False, "disabled")
    if cfg.tag_enabled:
        passed = bool(gate.get("tag", True))
        state["tagger"] = (passed, "" if passed else insufficient)
    else:
        state["tagger"] = (False, "disabled")
    return state


def _release_role_model(model: Any) -> None:
    """Best-effort release of a role model the 2e gate soft-disabled: drop the
    transformer/ONNX references and clear the CUDA cache so the process is not
    holding memory it will not serve. Never fatal."""
    for attr in ("_model", "_processor", "_session"):
        with contextlib.suppress(Exception):
            delattr(model, attr)
    with contextlib.suppress(Exception):
        import gc as _gc

        _gc.collect()
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()


def _load_role_model(
    route: str,
    cfg: Config,
    spec: tuple[Any, Any, str],
    loaded: dict[str, Any],
    deltas: dict[str, int],
) -> None:
    """Build and load one enabled describe/tag role model exactly once.

    Under production+CUDA this single load doubles as the 2d activation probe
    (a load failure is the fatal CUDA activation gate error before bind); the
    measured residency delta around the load becomes the resident_bytes input
    of the 2e gate. Under CPU or the fake adapter there is no 2e check, so the
    same single load is just the production (or fake) load."""
    build, path, name = spec
    production_cuda = (
        cfg.adapter == "production" and cfg.execution_provider == "CUDAExecutionProvider"
    )
    before = (
        memory_gate.available_bytes(cfg.execution_provider) if production_cuda else None
    )
    model = build()
    try:
        model.load(path)
    except Exception as exc:
        if production_cuda:
            # 2d: fatal before bind, regardless of what the 2e gate would later
            # decide — a CUDA role that cannot construct/load never binds.
            raise FatalStartupError(
                "config.invalid", f"{name} CUDA activation gate failed: {exc}"
            ) from exc
        raise FatalStartupError("config.invalid", f"failed to load {name}: {exc}") from exc
    if production_cuda and before is not None:
        after = memory_gate.available_bytes(cfg.execution_provider)
        if after is not None:
            # Measured delta only: bytes this load took from device-0 free
            # memory. Never a model-size estimate.
            deltas[route] = before - after
    loaded[route] = model


def load_and_adapt(cfg: Config) -> dict[str, Any]:
    """Construct adapters and load their model files (production paths).

    One model per role: each enabled describe/tag role is built and loaded
    exactly ONCE, before the 2e startup free-memory gate. Under production+CUDA
    that single load is the 2d CUDA activation probe (kept fatal before bind);
    the loaded instance is kept and then either promoted to the production
    adapter when its gate check passes, or dropped and released when the gate
    soft-disables the role. The gate's NVML device-0 read is taken with the
    role models resident; their measured residency deltas are added back so
    the operator-minimum arithmetic compares against pristine free bytes.
    """
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

    # Build + load each enabled role model once, BEFORE the memory gate.
    role_specs = {
        "describe": (
            cfg.describe_enabled,
            (
                lambda: _build_florence(cfg),
                Path(cfg.florence_dir_path) if cfg.florence_dir_path else Path(),
                "florence2",
            ),
        ),
        "tag": (
            cfg.tag_enabled,
            (
                lambda: _build_tagger(cfg),
                Path(cfg.tagger_onnx_path) if cfg.tagger_onnx_path else Path(),
                "tagger",
            ),
        ),
    }
    loaded_roles: dict[str, Any] = {}
    deltas: dict[str, int] = {}
    for route, (enabled, spec) in role_specs.items():
        if not enabled:
            continue
        if cfg.adapter == "production" and cfg.execution_provider != "CUDAExecutionProvider":
            # CPU production: the fail-closed file rule fires before the load
            # (the gate never soft-disables under CPU — no minima apply).
            _require_route_files(cfg, route, True)
        _load_role_model(route, cfg, spec, loaded_roles, deltas)

    # 2e gate on the post-load NVML read; resident_bytes adds the measured
    # residency of the already-loaded role models back so the compare uses
    # free bytes as the spec's operator-minimum arithmetic assumes.
    gate = memory_gate.evaluate_memory_gate(
        cfg.execution_provider,
        _describe_tag_minima(cfg),
        resident_bytes=sum(deltas.values()),
    )

    # Promote or drop-and-release: the same instance that proved CUDA
    # activation becomes the production adapter; no model is loaded twice and
    # a soft-disabled route releases its memory before serving.
    florence_adapter: Any = None
    tagger_adapter: Any = None
    suppressed = frozenset[str]()
    for route, model in loaded_roles.items():
        if gate.get(route, True):
            if route == "describe":
                florence_adapter = model
            else:
                tagger_adapter = model
                suppressed = _read_suppressed(cfg.tag_suppress_path)
        else:
            _release_role_model(model)

    role_state = build_role_state(cfg, gate)
    count = 4 + int(florence_adapter is not None) + int(tagger_adapter is not None)
    metrics_mod.models_loaded.set(count)
    metrics_mod.ensure_describe_series(
        [r for r, loaded in (("describe", florence_adapter is not None), ("tag", tagger_adapter is not None)) if loaded]
    )
    metrics_mod.set_describe_models_loaded(int(florence_adapter is not None) + int(tagger_adapter is not None))
    adapters.update(
        {
            "describe_adapter": florence_adapter,
            "tagger_adapter": tagger_adapter,
            "suppressed_tags": suppressed,
            "role_state": role_state,
            "models_loaded_count": count,
        }
    )
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

    # Eager pre-bind load: a fatal is raised before any socket opens. The
    # lifespan only flips readiness and starts the metrics server.
    if preloaded_adapters:
        adapters: dict[str, Any] = {**preloaded_adapters}
        already_loaded = True
    else:
        adapters = load_and_adapt(cfg)
        already_loaded = False
    started_at = time.monotonic()
    gpu_lock: threading.Lock | None = (
        threading.Lock() if cfg.execution_provider == "CUDAExecutionProvider" else None
    )

    @contextmanager
    def gpu_admission():
        """One shared in-process GPU lock for embed, describe, tag (6e). No
        queue, no wait: a second GPU inference is 429 gpu.busy and does not
        start. No lock exists under CPUExecutionProvider."""
        if gpu_lock is None:
            yield
        else:
            if not gpu_lock.acquire(blocking=False):
                raise ApiError(429, "gpu.busy", "another inference holds the GPU admission lock")
            try:
                yield
            finally:
                gpu_lock.release()

    def _require_bearer(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(
            token, cfg.bearer_token
        ):
            raise ApiError(401, "auth.invalid_token", "missing or invalid bearer token")

    async def _read_body_capped(request: Request, request_id: str, decode_errors: bool = False) -> tuple[bytes, FormData]:
        buf = bytearray()
        try:
            async for chunk in request.stream():
                buf += chunk
                if len(buf) > cfg.max_body_bytes:
                    raise ApiError(413, "limits.body_too_large", "body exceeds max_body_bytes")
            raw = bytes(buf)
            form = await parse_multipart(request, raw)
        except MultiPartException as exc:
            raise ApiError(422, "decode.failed", f"malformed multipart body: {exc}") from exc
        except ApiError:
            raise
        except Exception as exc:
            if decode_errors:
                # A body that cannot be read at the transport level is an
                # undecodable payload (step 4), not an inference failure.
                raise ApiError(422, "decode.failed", "request body could not be read") from exc
            raise
        return raw, form

    async def _extract_image(form: FormData, decode_errors: bool = False) -> bytes:
        try:
            if "image" not in form:
                raise ApiError(422, "decode.failed", "missing multipart field: image")
            image_part = form["image"]
            if hasattr(image_part, "read"):
                raw = await image_part.read()
            else:
                raw = str(image_part).encode()
        except ApiError:
            raise
        except Exception as exc:
            if decode_errors:
                raise ApiError(422, "decode.failed", "request body could not be read") from exc
            raise
        return raw

    async def _decode_rgb(raw: bytes, request_id: str) -> np.ndarray:
        fmt = sniff_format(raw)
        if fmt is None:
            raise ApiError(415, "format.unsupported", "unsupported image format (sniffed)")
        try:
            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ApiError(422, "decode.failed", "image decode failed") from exc
        return rgb

    state = {"ready": False}

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

    def _count_describe_request(route: str, outcome: str, reason_token: str) -> None:
        counter = metrics_mod.describe_metric("requests_total")
        if counter is not None:
            counter.labels(route=route, outcome=outcome, reason=reason_token).inc()

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)
        _audit_http_line(request, "ok", "ok")
        metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
        return JSONResponse({"status": "ok"}, headers={"X-Request-Id": request_id})

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)
        expected = int(adapters.get("models_loaded_count", 4))
        loaded = metrics_mod.models_loaded._value.get()
        if loaded != expected:
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
            {"status": "ready", "models_loaded": expected},
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
                _require_bearer(request)

                # Step 3: magic-byte sniff — read the body with the size cap enforced.
                _, form = await _read_body_capped(request, request_id, decode_errors=True)
                raw = await _extract_image(form, decode_errors=True)
                byte_count = len(raw)
                sha = _sha256_hex(raw)
                rgb = await _decode_rgb(raw, request_id)

                include_raw = form.get("include_object_vectors")
                if include_raw is None:
                    include_object_vectors = False
                elif include_raw == "true":
                    include_object_vectors = True
                elif include_raw == "false":
                    include_object_vectors = False
                else:
                    raise ApiError(422, "decode.failed", "invalid include_object_vectors")

                # Step 5: model inference under the shared GPU admission lock (6e).
                with gpu_admission():
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
        except Exception as exc:
            # A run that holds the lock and then fails is inference.failed —
            # never gpu.busy (section 6e). The lock has already been released.
            audit("error", "inference.failed")
            metrics_mod.requests_total.labels(outcome="error", reason="inference.failed").inc()
            metrics_mod.errors_total.labels(reason="inference.failed").inc()
            return JSONResponse(
                _error_body("inference.failed", f"inference failed: {type(exc).__name__}", request_id),
                status_code=500,
                headers={"X-Request-Id": request_id},
            )

    async def _describe_handler(request: Request) -> JSONResponse:
        route_name = "/v1/describe"
        role = "describe"
        request_id = _effective_request_id(request, cfg.request_id_header)
        florence = adapters.get("describe_adapter")
        disabled = florence is None  # switch false or soft-disabled by 2e

        def audit(outcome: str, reason: str, *, sha: str | None = "", byte_count: int = 0,
                  task: str | None = None, regions: int | None = None) -> None:
            emit_describe_audit_line(
                request_id=request_id,
                route=route_name,
                sha256=sha,
                byte_count=byte_count,
                task=task,
                regions=regions,
                tags=None,
                threshold=None,
                outcome=outcome,
                reason=reason,
                enabled=cfg.audit_enabled,
            )

        if disabled:
            # Step 0 (v12): 404 route.disabled immediately; no body read, no
            # lock, no inference, and the soft-disabled route emits no series.
            audit("error", "route.disabled")
            return JSONResponse(
                _error_body("route.disabled", "route is disabled", request_id),
                status_code=404,
                headers={"X-Request-Id": request_id},
            )
        sha = ""
        byte_count = 0
        regions_count = 0
        task: str | None = None
        try:
            with metrics_mod.describe_metric("latency_seconds").time():
                content_length = request.headers.get("content-length")
                if content_length is not None and content_length.isdigit():
                    if int(content_length) > cfg.max_body_bytes:
                        raise ApiError(413, "limits.body_too_large", "body exceeds max_body_bytes")
                _require_bearer(request)
                _, form = await _read_body_capped(request, request_id, decode_errors=True)
                raw = await _extract_image(form, decode_errors=True)
                byte_count = len(raw)
                sha = _sha256_hex(raw)
                rgb = await _decode_rgb(raw, request_id)

                # Step 5: field validation — exact-string task match.
                tasks = [v for k, v in form.multi_items() if k == "task"]
                if len(tasks) > 1:
                    raise ApiError(422, "decode.failed", "duplicate task field")
                if not tasks:
                    raise ApiError(422, "decode.failed", "missing task field")
                task = tasks[0]
                if task not in WIRE_TASKS:
                    raise ApiError(422, "task.unsupported", f"task is not one of the eight closed wire values")

                # Step 6: inference under the shared GPU admission lock.
                with gpu_admission():
                    text, regions = florence.run(rgb, task)
                payload: dict[str, Any] = {
                    "request_id": request_id,
                    "image": {"sha256": sha, "width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
                    "task": task,
                    "model": "florence2",
                }
                if task in _DESCRIBE_TEXT_TASKS:
                    payload["text"] = str(text)
                else:
                    h, w = int(rgb.shape[0]), int(rgb.shape[1])
                    capped = regions[: cfg.max_regions]
                    regions_count = len(capped)
                    out_regions = []
                    for label, x, y, bw, bh, score, region_text in capped:
                        x0, y0 = max(0, int(x)), max(0, int(y))
                        x1, y1 = min(w, x0 + max(0, int(bw))), min(h, y0 + max(0, int(bh)))
                        item: dict[str, Any] = {
                            "label": str(label),
                            "bbox": {"x": x0, "y": y0, "w": max(0, x1 - x0), "h": max(0, y1 - y0)},
                        }
                        if task == "<OD>":
                            item["score"] = float(score) if score is not None else 0.0
                        if task == "<OCR_WITH_REGION>" and region_text is not None:
                            item["text"] = str(region_text)
                        out_regions.append(item)
                    payload["regions"] = out_regions
                audit("ok", "ok", sha=sha, byte_count=byte_count, task=task, regions=regions_count)
                _count_describe_request(role, "ok", "ok")
                return JSONResponse(payload, status_code=200, headers={"X-Request-Id": request_id})
        except ApiError as err:
            audit(
                "error", err.reason, sha=sha, byte_count=byte_count,
                task=task,
                regions=regions_count,
            )
            _count_describe_request(role, "error", err.reason)
            errs = metrics_mod.describe_metric("errors_total")
            if errs is not None:
                errs.labels(route=role, reason=err.reason).inc()
            return JSONResponse(
                _error_body(err.reason, err.detail, request_id),
                status_code=err.status,
                headers={"X-Request-Id": request_id},
            )
        except Exception as exc:
            audit("error", "inference.failed", sha=sha, byte_count=byte_count,
                  task=None, regions=regions_count)
            _count_describe_request(role, "error", "inference.failed")
            errs = metrics_mod.describe_metric("errors_total")
            if errs is not None:
                errs.labels(route=role, reason="inference.failed").inc()
            return JSONResponse(
                _error_body("inference.failed", f"inference failed: {type(exc).__name__}", request_id),
                status_code=500,
                headers={"X-Request-Id": request_id},
            )

    async def _tag_handler(request: Request) -> JSONResponse:
        route_name = "/v1/tag"
        role = "tag"
        request_id = _effective_request_id(request, cfg.request_id_header)
        tagger = adapters.get("tagger_adapter")
        suppressed: frozenset[str] = adapters.get("suppressed_tags", frozenset())
        disabled = tagger is None

        def audit(outcome: str, reason: str, *, sha: str | None = "", byte_count: int = 0,
                  tags_count: int | None = None, threshold: str | None = None) -> None:
            emit_describe_audit_line(
                request_id=request_id,
                route=route_name,
                sha256=sha,
                byte_count=byte_count,
                task=None,
                regions=None,
                tags=tags_count,
                threshold=threshold,
                outcome=outcome,
                reason=reason,
                enabled=cfg.audit_enabled,
            )

        if disabled:
            audit("error", "route.disabled")
            return JSONResponse(
                _error_body("route.disabled", "route is disabled", request_id),
                status_code=404,
                headers={"X-Request-Id": request_id},
            )
        sha = ""
        byte_count = 0
        tags_count = 0
        threshold_raw: str | None = None
        try:
            with metrics_mod.describe_metric("latency_seconds").time():
                content_length = request.headers.get("content-length")
                if content_length is not None and content_length.isdigit():
                    if int(content_length) > cfg.max_body_bytes:
                        raise ApiError(413, "limits.body_too_large", "body exceeds max_body_bytes")
                _require_bearer(request)
                _, form = await _read_body_capped(request, request_id, decode_errors=True)
                raw = await _extract_image(form, decode_errors=True)
                byte_count = len(raw)
                sha = _sha256_hex(raw)
                rgb = await _decode_rgb(raw, request_id)

                # Step 5: threshold required, parseable decimal in [0, 1].
                thresholds = [v for k, v in form.multi_items() if k == "tag_threshold"]
                if len(thresholds) > 1:
                    raise ApiError(422, "decode.failed", "duplicate tag_threshold field")
                if not thresholds:
                    raise ApiError(422, "decode.failed", "missing tag_threshold field")
                threshold_raw = thresholds[0]
                try:
                    dec = Decimal(threshold_raw)
                    if not dec.is_finite():
                        raise InvalidOperation()
                except (InvalidOperation, ValueError):
                    raise ApiError(422, "decode.failed", "tag_threshold is not a parseable decimal")
                if dec < Decimal(0) or dec > Decimal(1):
                    raise ApiError(422, "decode.failed", "tag_threshold must be within [0.0, 1.0]")
                threshold_value = float(dec)

                # Step 6: inference under the shared GPU admission lock.
                with gpu_admission():
                    pairs = tagger.tag(rgb, threshold_value)
                pairs = [(t, s) for (t, s) in pairs if t not in suppressed]
                pairs.sort(key=lambda p: p[1], reverse=True)
                tags_count = len(pairs)
                payload = {
                    "request_id": request_id,
                    "image": {"sha256": sha, "width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
                    "model": "wd14",
                    "threshold_used": threshold_value,
                    "tags": [{"tag": str(t), "score": float(s)} for t, s in pairs],
                }
                audit("ok", "ok", sha=sha, byte_count=byte_count,
                      tags_count=tags_count, threshold=threshold_raw)
                _count_describe_request(role, "ok", "ok")
                return JSONResponse(payload, status_code=200, headers={"X-Request-Id": request_id})
        except ApiError as err:
            audit("error", err.reason, sha=sha, byte_count=byte_count,
                  tags_count=tags_count, threshold=threshold_raw)
            _count_describe_request(role, "error", err.reason)
            errs = metrics_mod.describe_metric("errors_total")
            if errs is not None:
                errs.labels(route=role, reason=err.reason).inc()
            return JSONResponse(
                _error_body(err.reason, err.detail, request_id),
                status_code=err.status,
                headers={"X-Request-Id": request_id},
            )
        except Exception as exc:
            audit("error", "inference.failed", sha=sha, byte_count=byte_count,
                  tags_count=tags_count, threshold=threshold_raw)
            _count_describe_request(role, "error", "inference.failed")
            errs = metrics_mod.describe_metric("errors_total")
            if errs is not None:
                errs.labels(route=role, reason="inference.failed").inc()
            return JSONResponse(
                _error_body("inference.failed", f"inference failed: {type(exc).__name__}", request_id),
                status_code=500,
                headers={"X-Request-Id": request_id},
            )

    @app.post("/v1/describe")
    async def v1_describe(request: Request) -> JSONResponse:
        return await _describe_handler(request)

    @app.post("/v1/tag")
    async def v1_tag(request: Request) -> JSONResponse:
        return await _tag_handler(request)

    @app.get("/v1/runtime")
    async def v1_runtime(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request, cfg.request_id_header)

        def audit(outcome: str, reason: str) -> None:
            emit_describe_audit_line(
                request_id=request_id,
                route="/v1/runtime",
                sha256=None,
                byte_count=0,
                task=None,
                regions=None,
                tags=None,
                threshold=None,
                outcome=outcome,
                reason=reason,
                enabled=cfg.audit_enabled,
            )

        try:
            _require_bearer(request)
        except ApiError as err:
            audit("error", err.reason)
            return JSONResponse(
                _error_body(err.reason, err.detail, request_id),
                status_code=err.status,
                headers={"X-Request-Id": request_id},
            )
        role_state: dict[str, tuple[bool, str]] = adapters.get("role_state", {})
        models = []
        for role in ROLE_ORDER:
            loaded, reason_value = role_state.get(role, (False, "disabled"))
            models.append(
                {
                    "role": role,
                    "model_id": ROLE_MODEL_IDS[role] if loaded else "",
                    "loaded": bool(loaded),
                    "unavailable_reason": str(reason_value),
                }
            )
        body = {
            "uptime_seconds": int(max(0, time.monotonic() - started_at)),
            "execution_provider": cfg.execution_provider,
            "routes": {
                "embed": True,
                "describe": bool(cfg.describe_enabled),
                "tag": bool(cfg.tag_enabled),
            },
            "models": models,
            "gpu": memory_gate.gpu_inventory(),
        }
        audit("ok", "ok")
        return JSONResponse(body, status_code=200, headers={"X-Request-Id": request_id})

    async def parse_multipart(request: Request, raw: bytes) -> FormData:
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
        return form

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
    # v12 section 2e: the minima keys are required, before bind and before any
    # model loads, exactly when their switch is true and the provider is CUDA.
    from .config import validate_route_minima

    validate_route_minima(cfg)
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
