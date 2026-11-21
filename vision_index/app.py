"""App — one ASGI app (SPEC sections 1, 5, 6). Fixed processing order (§6b):
Content-Length pre-check → auth → body read (32 MiB cap) → magic-byte sniff →
field validation → sha256 dedupe → embed → transaction → schema validation →
audit → body. Identify always embeds and never writes.

Production uses the HTTPS embed client; the fake client exists behind the same
interface and is constructed only under adapter_mode "test" — no config key
reaches it (§10; the construction-site check is the enforcement, test 3).
"""

from __future__ import annotations

import hashlib
import io
import inspect
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser
from starlette.requests import Request
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from . import metrics as metrics_mod
from . import store as store_mod
from .adapters.embed import EmbedClient, EmbedUnavailableError
from .adapters.fake import FakeEmbedClient
from .adapters.http_client import ProductionEmbedClient
from .audit import emit_audit_line
from .config import BODY_LIMIT_BYTES, IndexConfig, constant_time_token_eq, load_config
from .coco import is_coco80
from .db import IndexStore
from .errors import ApiError, REASON_TOKENS

REQUEST_ID_HEADER = "X-Request-Id"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _effective_request_id(request: Any) -> str:
    raw = request.headers.get(REQUEST_ID_HEADER)
    if raw is not None and raw.isascii() and raw.isprintable() and 0 < len(raw) <= 128:
        return raw
    return str(uuid.uuid4())


def _error_body(reason: str, detail: str, request_id: str) -> dict:
    return {"error": {"reason": reason, "detail": detail, "request_id": request_id}}


def _json_error(request_id: str, reason: str, detail: str, status: int) -> JSONResponse:
    return JSONResponse(
        _error_body(reason, detail, request_id),
        status_code=status,
        headers={REQUEST_ID_HEADER: request_id},
    )


def _status_for_reason(reason: str) -> int:
    from .errors import _TOKEN_HTTP

    return _TOKEN_HTTP[reason]


class BodyAuthGuard:
    """Pure-ASGI middleware enforcing §6b steps 1-2 on /v1/* routes only, in
    order: Content-Length pre-check (413, token never checked), then auth
    (401, body never read or sniffed). Health/readiness are unauthenticated."""

    def __init__(self, app: Any, token: str, cfg_params: dict) -> None:
        self.app = app
        self.token = token
        self.cfg_params = cfg_params

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request = Request(scope)
        request_id = _effective_request_id(request)
        route = scope["path"]

        # Step 1: Content-Length pre-check before auth (§6b — fixed order).
        content_length = headers.get("content-length")
        if content_length is not None and content_length.isdigit():
            if int(content_length) > BODY_LIMIT_BYTES:
                self._record(scope, request_id, route, "limits.body_too_large")
                await self._emit(
                    scope, send, request_id, 413, "limits.body_too_large",
                    "Content-Length exceeds 32 MiB",
                )
                return

        # Step 2: auth — constant-time compare; body is never read on failure.
        auth = headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token or not constant_time_token_eq(token, self.token):
            self._record(scope, request_id, route, "auth.invalid_token")
            await self._emit(
                scope, send, request_id, 401, "auth.invalid_token",
                "missing or invalid bearer token",
            )
            return
        await self.app(scope, receive, send)

    def _record(self, scope: dict, request_id: str, route: str, reason: str) -> None:
        emit_audit_line(
            request_id=request_id, route=route, outcome="error", reason=reason,
            enabled=self.cfg_params["audit_enabled"],
        )
        metrics_mod.requests_total.labels(outcome="error", reason=reason).inc()
        metrics_mod.errors_total.labels(reason=reason).inc()

    async def _emit(self, scope: dict, send: Any, request_id: str, status: int,
                    reason: str, detail: str) -> None:
        body = json.dumps(_error_body(reason, detail, request_id)).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (REQUEST_ID_HEADER.encode(), request_id.encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


async def read_body_capped(request: Request) -> bytes:
    """§6b step 3: read the body with the 32 MiB cap enforced while reading
    (chunked encodings and low-lying Content-Lengths included)."""
    buf = bytearray()
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            raise ApiError("decode.failed", "client disconnected mid-body")
        chunk = message.get("body", b"")
        buf += chunk
        if len(buf) > BODY_LIMIT_BYTES:
            raise ApiError("limits.body_too_large", "body exceeds 32 MiB while reading")
        if not message.get("more_body"):
            break
    return bytes(buf)


def sniff_format(data: bytes) -> bool:
    """Magic-byte sniff — jpeg/png/webp only; multipart Content-Type never
    trusted (§6b step 4, same rules as vision-embed)."""
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


async def parse_multipart(request: Request, raw: bytes) -> dict:
    content_type = request.headers.get("content-type", "") or "multipart/form-data"

    async def stream():  # one buffered chunk; cap already enforced by reader
        yield raw

    parser_kwargs: dict[str, Any] = {"max_files": 100, "max_fields": 100}
    if "max_part_size" in inspect.signature(MultiPartParser.__init__).parameters:
        parser_kwargs["max_part_size"] = BODY_LIMIT_BYTES
    parser = MultiPartParser(
        headers=Headers({"content-type": content_type}),
        stream=stream(),
        **parser_kwargs,
    )
    form = await parser.parse()
    return dict(form)


async def _image_parts(request: Request) -> tuple[bytes, str | None, bytes]:
    raw = await read_body_capped(request)
    try:
        form = await parse_multipart(request, raw)
    except Exception as exc:
        raise ApiError("decode.failed", f"malformed multipart body: {exc}") from None
    if "image" not in form:
        raise ApiError("decode.failed", "missing multipart field: image")
    part = form["image"]
    image = await part.read() if hasattr(part, "read") else str(part).encode()
    # §6b step 4 (magic-byte sniff) precedes step 5 (field validation): a
    # non-image with an over-long path fails as format.unsupported, not
    # path.too_long.
    if not sniff_format(image):
        raise ApiError("format.unsupported", "unsupported image format (sniffed)")
    path_field = form.get("path")
    if path_field is not None:
        path_value = await path_field.read() if hasattr(path_field, "read") else str(path_field)
        path = path_value.decode("utf-8", "replace") if isinstance(path_value, bytes) else path_value
        if len(path) > 64:
            raise ApiError("path.too_long", "path label exceeds 64 characters")
    else:
        path = None
    return image, path, raw


def _build_embed_client(cfg: IndexConfig) -> EmbedClient:
    """Production refusal (§10): the fake client is ONLY constructible under
    adapter_mode 'test', and no config value other than the mode reaches here."""
    if cfg.adapter_mode == "test":
        return FakeEmbedClient(mode="test")
    if cfg.adapter_mode == "production":
        return ProductionEmbedClient(
            cfg.embed_base_url, cfg.embed_bearer_token, cfg.embed_timeout_seconds
        )
    raise RuntimeError("unreachable")


def _embed_or_502(client: EmbedClient, image: bytes):
    try:
        return client.embed(image)
    except EmbedUnavailableError:
        raise ApiError("embed.unavailable", "embed service unavailable") from None
    except ApiError:
        raise  # response.validation_failed travels as-is
    except Exception:
        raise ApiError("embed.unavailable", "embed client failure") from None


def _refresh_row_gauges(store: IndexStore) -> None:
    metrics_mod.assets_total.set(store.table_counts()["assets"])
    metrics_mod.clusters_total.labels(kind="face").set(
        int(store.conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE kind= 'face' AND merged_into IS NULL"
        ).fetchone()[0])
    )
    metrics_mod.clusters_total.labels(kind="object").set(
        int(store.conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE kind= 'object' AND merged_into IS NULL"
        ).fetchone()[0])
    )


def build_app(cfg: IndexConfig, embed_client: EmbedClient | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: Any):
        store = IndexStore(cfg.db_path)  # missing/uncreatable file is fatal here
        state["store"] = store
        state["client"] = embed_client if embed_client is not None else _build_embed_client(cfg)
        if cfg.metrics_enabled:
            metrics_mod.start_metrics_server(cfg.metrics_port, cfg.metrics_host)
        try:
            yield
        finally:
            state["store"].close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(BodyAuthGuard, token=cfg.bearer_token,
                       cfg_params={"audit_enabled": cfg.audit_enabled})

    def store_get() -> IndexStore:
        return state["store"]

    def client_get() -> EmbedClient:
        return state["client"]

    def audit(request_id: str, route: str, outcome: str, reason: str, *, sha256=None,
              asset_id=None, faces=0, clusters_created=0) -> None:
        emit_audit_line(
            request_id=request_id, route=route, outcome=outcome, reason=reason,
            sha256=sha256, asset_id=asset_id, faces=faces,
            clusters_created=clusters_created, enabled=cfg.audit_enabled,
        )

    def audit_line_from_request(request: Request, outcome: str, reason: str) -> str:
        request_id = _effective_request_id(request)
        audit(request_id, request.url.path, outcome, reason)
        return request_id

    # -- unauthenticated routes --------------------------------------------

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        audit(request_id, "/healthz", "ok", "ok")
        metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
        return JSONResponse(
            {"status": "ok"}, headers={REQUEST_ID_HEADER: request_id}
        )

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        ready = False
        if "store" in state:
            try:
                sqlite_ok = state["store"].run_migrations() is None
            except ApiError:
                sqlite_ok = False
            embed_ready = client_get().ready()
            ready = bool(sqlite_ok and embed_ready)
        if not ready:
            audit(request_id, "/readyz", "error", "not_ready")
            metrics_mod.requests_total.labels(outcome="error", reason="not_ready").inc()
            metrics_mod.errors_total.labels(reason="not_ready").inc()
            return _json_error(request_id, "not_ready", "sqlite or embed not ready", 503)
        audit(request_id, "/readyz", "ok", "ok")
        metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
        return JSONResponse(
            {"status": "ready", "sqlite_ok": True, "embed_ready": True},
            headers={REQUEST_ID_HEADER: request_id},
        )

    # -- v1 routes (middleware already enforced steps 1-2) ------------------

    @app.post("/v1/ingest")
    async def v1_ingest(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        audit_fields: dict[str, Any] = {
            "sha256": None, "asset_id": None, "faces": 0, "clusters_created": 0,
        }

        def audit_now(outcome: str, reason: str) -> None:
            audit(request_id, "/v1/ingest", outcome, reason, **audit_fields)

        try:
            with metrics_mod.latency_seconds.time():
                image, path, _ = await _image_parts(request)
                sha = _sha256_hex(image)
                audit_fields["sha256"] = sha

                # §6b step 6: dedupe lookup BEFORE the embed call.
                existing = store_mod.dedupe_sha(store_get(), sha)
                if existing is not None:
                    scene = store_get().get_vectors(
                        "scene_vec", "asset_id", [existing["asset_id"]]
                    ).get(existing["asset_id"])
                    payload = {
                        "request_id": request_id,
                        "asset_id": existing["asset_id"],
                        "created": False,
                        "image": {"sha256": existing["sha256"], "width": existing["width"],
                                  "height": existing["height"]},
                        "scene_vector": [float(v) for v in scene] if scene is not None else [],
                        "face_count": existing["face_count"],
                        "face_cluster_ids": existing["face_cluster_ids"],
                        "object_count": existing["object_count"],
                    }
                    audit_now("ok", "ok")
                    metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
                    return JSONResponse(
                        payload, headers={REQUEST_ID_HEADER: request_id}
                    )

                # §6b step 7: embed call with include_object_vectors=true.
                embed = _embed_or_502(client_get(), image)
                if embed.sha256 != sha:
                    raise ApiError("response.validation_failed", "embed sha256 mismatch")
                result = store_mod.ingest_asset(
                    store_get(), raw_bytes=len(image), sha=sha,
                    width=embed.width, height=embed.height, path=path,
                    scene_vector=embed.scene_vector,
                    face_records=[
                        {"bbox": f.bbox, "det_score": f.det_score, "vector": f.vector}
                        for f in embed.faces
                    ],
                    object_records=[
                        {"label": o.label, "bbox": o.bbox, "score": o.score, "vector": o.vector}
                        for o in embed.objects
                    ],
                    face_match_max=cfg.face_match_max,
                )
                audit_fields.update(
                    asset_id=result["asset_id"], faces=result["face_count"],
                    clusters_created=result["clusters_created"],
                )
                _refresh_row_gauges(store_get())
                payload = {
                    "request_id": request_id,
                    "asset_id": result["asset_id"],
                    "created": True,
                    "image": {"sha256": sha, "width": embed.width, "height": embed.height},
                    "scene_vector": [float(v) for v in to_list(embed.scene_vector)],
                    "face_count": result["face_count"],
                    "face_cluster_ids": result["face_cluster_ids"],
                    "object_count": result["object_count"],
                }
                audit_now("ok", "ok")
                metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
                return JSONResponse(payload, headers={REQUEST_ID_HEADER: request_id})
        except ApiError as err:
            audit_now("error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.post("/v1/identify")
    async def v1_identify(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        audit_fields: dict[str, Any] = {"sha256": None, "asset_id": None, "faces": 0,
                                        "clusters_created": 0}

        def audit_now(outcome: str, reason: str) -> None:
            audit(request_id, "/v1/identify", outcome, reason, **audit_fields)

        try:
            with metrics_mod.latency_seconds.time():
                image, _path, _ = await _image_parts(request)
                sha = _sha256_hex(image)
                audit_fields["sha256"] = sha
                embed = _embed_or_502(client_get(), image)  # identify always embeds, never writes
                if embed.sha256 != sha:
                    raise ApiError("response.validation_failed", "embed sha256 mismatch")

                face_matches = []
                for i, face in enumerate(embed.faces):
                    q = np.ascontiguousarray(face.vector, dtype=np.float32)
                    cluster_id, dist, matched = store_mod.match_face_vector(
                        store_get(), q, cfg.face_match_max
                    )
                    name = None
                    if cluster_id is not None:
                        crow = store_get().conn.execute(
                            "SELECT name FROM clusters WHERE id = ?", (cluster_id,)
                        ).fetchone()
                        name = crow[0] if crow is not None else None
                    face_matches.append({
                        "face_index": i,
                        "nearest_cluster_id": cluster_id,
                        "nearest_cluster_name": name,
                        "distance": dist,
                        "matched": matched,
                    })
                scene_hits = store_mod.scene_nearest(
                    store_get(), np.ascontiguousarray(embed.scene_vector, dtype=np.float32), k=10
                )
                audit_now("ok", "ok")
                metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
                return JSONResponse(
                    {
                        "request_id": request_id,
                        "sha256": sha,
                        "faces": face_matches,
                        "scene": scene_hits,
                    },
                    headers={REQUEST_ID_HEADER: request_id},
                )
        except ApiError as err:
            audit_now("error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.get("/v1/assets/{asset_id}")
    async def v1_asset(request: Request, asset_id: str) -> JSONResponse:
        request_id = _effective_request_id(request)
        route = "/v1/assets/" + asset_id
        try:
            if not asset_id.isdigit():
                raise ApiError("route.not_found", "asset id must be an integer")
            record = store_mod.get_asset_record(store_get(), int(asset_id))
            scene = store_get().get_vectors("scene_vec", "asset_id", [int(asset_id)]).get(
                int(asset_id)
            )
            payload = {
                "request_id": request_id,
                **{k: record[k] for k in
                   ("id", "path", "sha256", "width", "height", "bytes", "indexed_at")},
                "scene_vector": [float(v) for v in scene] if scene is not None else [],
                "faces": record["faces"],
                "objects": record["objects"],
            }
            audit(request_id, route, "ok", "ok")
            metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
            return JSONResponse(payload, headers={REQUEST_ID_HEADER: request_id})
        except ApiError as err:
            audit(request_id, route, "error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.get("/v1/search/person")
    async def v1_search_person(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        name = request.query_params.get("name", "")
        try:
            result = store_mod.search_person(store_get().conn, name) if name else None
            if result is None:
                raise ApiError("not_found", f"name not found")
            audit(request_id, "/v1/search/person", "ok", "ok")
            metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
            return JSONResponse(
                {"request_id": request_id, "name": name, **result},
                headers={REQUEST_ID_HEADER: request_id},
            )
        except ApiError as err:
            audit(request_id, "/v1/search/person", "error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.get("/v1/search/objects")
    async def v1_search_objects(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        label = request.query_params.get("label")
        try:
            if label is not None and not is_coco80(label):
                raise ApiError("decode.failed", "invalid label")
            rows = store_mod.search_objects(store_get().conn, label) if label else []
            asset_ids: list[int] = []
            for row in rows:
                if row["asset_id"] not in asset_ids:
                    asset_ids.append(row["asset_id"])
            if label is None or not rows:
                raise ApiError("not_found", "no object rows match the label")
            audit(request_id, "/v1/search/objects", "ok", "ok")
            metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
            return JSONResponse(
                {"request_id": request_id, "label": label,
                 "objects": rows, "asset_ids": asset_ids},
                headers={REQUEST_ID_HEADER: request_id},
            )
        except ApiError as err:
            audit(request_id, "/v1/search/objects", "error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.post("/v1/name")
    async def v1_name(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        st = store_get()
        try:
            raw = await read_body_capped(request)
            body = _strict_json_body(raw)
            payload = _resolve_name_body(st, body, cfg, request_id)
            audit(request_id, "/v1/name", "ok", "ok")
            metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
            return JSONResponse(payload, headers={REQUEST_ID_HEADER: request_id})
        except ApiError as err:
            audit(request_id, "/v1/name", "error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.post("/v1/merge")
    async def v1_merge(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        st = store_get()
        conn = st.conn
        try:
            raw = await read_body_capped(request)
            body = _strict_json_body(raw)
            if set(body) != {"from_id", "into_id"} or not isinstance(
                body.get("from_id"), int
            ) or not isinstance(body.get("into_id"), int):
                raise ApiError("decode.failed", "merge body must be {from_id, into_id}")
            from_id, into_id = int(body["from_id"]), int(body["into_id"])
            try:
                conn.execute("BEGIN")
                result = store_mod.merge_clusters(conn, from_id, into_id)
            except ApiError:
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                raise ApiError("db.error", f"merge rolled back: {exc.__class__.__name__}") from None
            _refresh_row_gauges(st)
            audit(request_id, "/v1/merge", "ok", "ok")
            metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
            return JSONResponse(
                {"request_id": request_id, **result},
                headers={REQUEST_ID_HEADER: request_id},
            )
        except ApiError as err:
            audit(request_id, "/v1/merge", "error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    @app.get("/v1/clusters")
    async def v1_clusters(request: Request) -> JSONResponse:
        request_id = _effective_request_id(request)
        try:
            kind = request.query_params.get("kind")
            if kind is None:
                raise ApiError("decode.failed", "kind is required (face|object)")
            unnamed_raw = request.query_params.get("unnamed")
            unnamed: bool | None
            if unnamed_raw is None:
                unnamed = None
            elif unnamed_raw == "true":
                unnamed = True
            elif unnamed_raw == "false":
                unnamed = False
            else:
                raise ApiError("decode.failed", "unnamed must be true or false")
            results = store_mod.clusters_listing(store_get().conn, kind, unnamed)
            audit(request_id, "/v1/clusters", "ok", "ok")
            metrics_mod.requests_total.labels(outcome="ok", reason="ok").inc()
            return JSONResponse(
                {"request_id": request_id, "clusters": results},
                headers={REQUEST_ID_HEADER: request_id},
            )
        except ApiError as err:
            audit(request_id, "/v1/clusters", "error", err.reason)
            metrics_mod.requests_total.labels(outcome="error", reason=err.reason).inc()
            metrics_mod.errors_total.labels(reason=err.reason).inc()
            return _json_error(request_id, err.reason, err.detail, err.status)

    # -- closed surface 404/405 ---------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _effective_request_id(request)
        reason = "method.not_allowed" if exc.status_code == 405 else "route.not_found"
        metrics_mod.requests_total.labels(outcome="error", reason=reason).inc()
        metrics_mod.errors_total.labels(reason=reason).inc()
        # One audit line per request — closed-surface 404/405 responses included.
        audit(request_id, request.url.path, "error", reason)
        return _json_error(request_id, reason, "closed endpoint set", exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _effective_request_id(request)
        metrics_mod.requests_total.labels(outcome="error", reason="db.error").inc()
        metrics_mod.errors_total.labels(reason="db.error").inc()
        audit(request_id, request.url.path, "error", "db.error")
        return _json_error(request_id, "db.error", "internal error", HTTP_500_INTERNAL_SERVER_ERROR)

    return app


def to_list(vec: np.ndarray) -> list:
    return [float(v) for v in np.asarray(vec, dtype=np.float32).ravel()]


def _strict_json_body(raw: bytes) -> dict:
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ApiError("decode.failed", "body is not JSON") from None
    if not isinstance(body, dict):
        raise ApiError("decode.failed", "body must be a JSON object")
    return body


def _resolve_name_body(st: IndexStore, body: dict, cfg: IndexConfig, request_id: str) -> dict:
    """§8 step 8: exactly one of the three forms (FI7), one transaction."""
    conn = st.conn
    forms = []
    if set(body) == {"cluster_id", "name"}:
        forms.append(("cluster", body))
    elif set(body) == {"asset_id", "face_index", "name"}:
        forms.append(("face", body))
    elif set(body) == {"asset_id", "object_index", "name"}:
        forms.append(("object", body))
    if len(forms) != 1:
        raise ApiError("decode.failed", "exactly one naming form is required")
    kind, payload = forms[0]
    name = payload["name"]
    if not isinstance(name, str):
        raise ApiError("decode.failed", "name must be a string")

    try:
        conn.execute("BEGIN")
        if kind == "cluster":
            if not isinstance(payload["cluster_id"], int):
                raise ApiError("decode.failed", "cluster_id must be an integer")
            cluster_id = store_mod.resolve_cluster(conn, int(payload["cluster_id"]))
            if cluster_id is None:
                raise ApiError("not_found", "cluster_id resolves to nothing")
            result = store_mod.set_cluster_name(
                conn, cluster_id, name, min_len=cfg.name_min_length, max_len=cfg.name_max_length
            )
        elif kind == "face":
            for key in ("asset_id", "face_index"):
                if not isinstance(payload[key], int):
                    raise ApiError("decode.failed", f"{key} must be an integer")
            asset_id = int(payload["asset_id"])
            faces = store_mod.faces_for_asset(st, asset_id)  # ordered by face id ASC
            if not faces:
                raise ApiError("no_face", f"asset {asset_id} has no faces")
            idx = int(payload["face_index"])
            if idx < 0 or idx >= len(faces):
                raise ApiError("no_face", f"face_index {idx} out of range")
            cluster_id = faces[idx]["cluster_id"]
            if cluster_id is None:
                raise ApiError("no_face", "that face has no cluster")
            result = store_mod.set_cluster_name(
                conn, cluster_id, name, min_len=cfg.name_min_length, max_len=cfg.name_max_length
            )
        else:
            for key in ("asset_id", "object_index"):
                if not isinstance(payload[key], int):
                    raise ApiError("decode.failed", f"{key} must be an integer")
            asset_id = int(payload["asset_id"])
            objects = store_mod.objects_for_asset(st, asset_id)  # ordered by object id ASC
            idx = int(payload["object_index"])
            if idx < 0 or idx >= len(objects):
                raise ApiError("not_found", f"object_index {idx} out of range")
            object_id = objects[idx]["object_id"]
            if objects[idx]["cluster_id"] is None:
                store_mod.create_object_cluster_for(conn, object_id)
                row = conn.execute(
                    "SELECT cluster_id FROM objects WHERE id = ?", (object_id,)
                ).fetchone()
                cluster_id = int(row[0])
            else:
                cluster_id = int(objects[idx]["cluster_id"])
            result = store_mod.set_cluster_name(
                conn, cluster_id, name, min_len=cfg.name_min_length, max_len=cfg.name_max_length
            )
        conn.commit()
    except ApiError:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise ApiError("db.error", f"naming transaction rolled back: {exc.__class__.__name__}") from None
    return {"request_id": request_id, **result}


def build_from_config_path(config_path, embed_client: EmbedClient | None = None):
    cfg = load_config(config_path)
    return build_app(cfg, embed_client=embed_client), cfg


def run_server(config_path) -> int:
    import uvicorn

    app, cfg = build_from_config_path(config_path)
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
    return 0
