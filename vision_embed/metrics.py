"""Metrics — closed four-series set on a second unauthenticated port (SPEC section 10)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from typing import Any

REGISTRY = CollectorRegistry(auto_describe=True)

# Closed series only. Labels carry no hash, byte size, vector, or face count.
requests_total = Counter(
    "vision_embed_requests_total",
    "Requests by outcome and reason token.",
    ["outcome", "reason"],
    registry=REGISTRY,
)
latency_seconds = Histogram(
    "vision_embed_latency_seconds",
    "POST /v1/embed latency in seconds (standard Prometheus buckets).",
    registry=REGISTRY,
)
errors_total = Counter(
    "vision_embed_errors_total",
    "Errors by reason token.",
    ["reason"],
    registry=REGISTRY,
)
models_loaded = Gauge(
    "vision_embed_models_loaded",
    "Number of loaded adapter roles (0..4).",
    registry=REGISTRY,
)

_metrics_server_started = False


def start_metrics_server(port: int, addr: str = "0.0.0.0") -> None:
    global _metrics_server_started
    if not _metrics_server_started:
        start_http_server(port, addr=addr, registry=REGISTRY)
        _metrics_server_started = True


# --- v12 describe/tag series (SPEC v12 section 11) — created at most once,
# only when at least one describe/tag route is enabled, so a disabled route's
# series are absent rather than stuck at zero. ---

_describe_series: dict[str, Any] = {}


def ensure_describe_series(enabled_routes: list[str]) -> None:
    if not enabled_routes or _describe_series:
        return
    _describe_series["requests_total"] = Counter(
        "vision_describe_requests_total",
        "Describe/tag requests by outcome and reason token.",
        ["route", "outcome", "reason"],
        registry=REGISTRY,
    )
    _describe_series["latency_seconds"] = Histogram(
        "vision_describe_latency_seconds",
        "Describe/tag request latency in seconds (standard Prometheus buckets).",
        registry=REGISTRY,
    )
    _describe_series["errors_total"] = Counter(
        "vision_describe_errors_total",
        "Describe/tag errors by reason token.",
        ["route", "reason"],
        registry=REGISTRY,
    )
    _describe_series["models_loaded"] = Gauge(
        "vision_describe_models_loaded",
        "Number of loaded describe/tag roles (0..2).",
        registry=REGISTRY,
    )


def describe_metric(name: str):
    return _describe_series.get(name)


def set_describe_models_loaded(count: int) -> None:
    gauge = _describe_series.get("models_loaded")
    if gauge is not None:
        gauge.set(count)
