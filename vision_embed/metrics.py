"""Metrics — closed four-series set on a second unauthenticated port (SPEC section 10)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, start_http_server

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
