"""Metrics — closed five-series set on a second unauthed port (SPEC section 12).

No identity labels: no names, no paths, no 64-hex hashes, no label values
derived from image content beyond the closed reason token.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

REGISTRY = CollectorRegistry(auto_describe=True)

requests_total = Counter(
    "vision_index_requests_total",
    "Requests by outcome and reason token.",
    ["outcome", "reason"],
    registry=REGISTRY,
)
latency_seconds = Histogram(
    "vision_index_latency_seconds",
    "Request latency in seconds (standard Prometheus buckets).",
    registry=REGISTRY,
)
errors_total = Counter(
    "vision_index_errors_total",
    "Error requests by reason token.",
    ["reason"],
    registry=REGISTRY,
)
assets_total = Gauge(
    "vision_index_assets_total",
    "Count of assets rows.",
    registry=REGISTRY,
)
clusters_total = Gauge(
    "vision_index_clusters_total",
    "Count of clusters rows by kind.",
    ["kind"],
    registry=REGISTRY,
)

_started = False


def start_metrics_server(port: int, host: str = "127.0.0.1") -> None:
    global _started
    if not _started:
        start_http_server(port, addr=host, registry=REGISTRY)
        _started = True
