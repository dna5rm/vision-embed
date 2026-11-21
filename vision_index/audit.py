"""Audit — exactly one JSON line per HTTP request on stdout (SPEC section 11, INV-7).

The nine closed keys, in order. Never logs image bytes, embeddings,
Authorization headers, tokens, name strings, or path labels. Success reason is
the literal string "ok". If the stdout write fails the response body still
returns (no fallback file, no retry queue).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys

AUDIT_KEYS = frozenset({
    "ts", "request_id", "route", "outcome", "reason",
    "sha256", "asset_id", "faces", "clusters_created",
})


def rfc3339_utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_audit_line(
    *,
    request_id: str,
    route: str,
    outcome: str,
    reason: str,
    sha256: str | None = None,
    asset_id: int | None = None,
    faces: int = 0,
    clusters_created: int = 0,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    record = {
        "ts": rfc3339_utc_now(),
        "request_id": request_id,
        "route": route,
        "outcome": outcome,
        "reason": reason,
        # sha256 and asset_id are filled only on ingest/identify; null otherwise.
        "sha256": sha256,
        "asset_id": asset_id,
        "faces": faces,
        "clusters_created": clusters_created,
    }
    line = json.dumps(record, separators=(",", ":")) + "\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except (ValueError, OSError):
        pass
