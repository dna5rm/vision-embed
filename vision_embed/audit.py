"""Audit — exactly one JSON line per HTTP request on stdout (SPEC section 9; EXEC ruling 2).

Every served request emits, including GET /healthz and GET /readyz and the
closed-surface 404/405 responses; image-less requests carry sha256 "" and
bytes 0 (EXEC ruling 3). Early failures (401, 413, before an image part
exists) keep the eight-key shape with sha256 "" and bytes 0 (accepted).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys


def rfc3339_utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


AUDIT_KEYS = frozenset({"ts", "request_id", "sha256", "bytes", "faces", "objects", "outcome", "reason"})


def emit_describe_audit_line(
    *,
    request_id: str,
    route: str,
    sha256: str | None,
    byte_count: int,
    task: str | None = None,
    regions: int | None = None,
    tags: int | None = None,
    threshold: str | float | None = None,
    outcome: str,
    reason: str,
    enabled: bool = True,
) -> None:
    """Section 10 line for POST /v1/describe, POST /v1/tag and GET /v1/runtime:
    exactly one stdout JSON line; never logs image bytes, captions, tag
    strings, tokens, or unavailable_reason values."""
    if not enabled:
        return
    record = {
        "ts": rfc3339_utc_now(),
        "request_id": request_id,
        "route": route,
        "sha256": sha256 if sha256 else None,
        "bytes": byte_count,
        "task": task,
        "regions": regions,
        "tags": tags,
        "threshold": threshold,
        "outcome": outcome,
        "reason": reason,
    }
    line = json.dumps(record, separators=(",", ":")) + "\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except (ValueError, OSError):
        pass


def emit_audit_line(
    *,
    request_id: str,
    sha256: str,
    byte_count: int,
    faces: int,
    objects: int,
    outcome: str,
    reason: str,
    enabled: bool = True,
) -> None:
    """Write exactly one stdout JSON line.

    F40: if the stdout write fails silently the caller still returns the body;
    there is no fallback file, no syslog, no retry queue, and no buffering.
    Never logs image bytes, embeddings, Authorization headers, or tokens
    (INV-7). Success reason is the literal string "ok" (F29).
    """
    if not enabled:
        return
    record = {
        "ts": rfc3339_utc_now(),
        "request_id": request_id,
        "sha256": sha256,
        "bytes": byte_count,
        "faces": faces,
        "objects": objects,
        "outcome": outcome,
        "reason": reason,
    }
    line = json.dumps(record, separators=(",", ":")) + "\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except (ValueError, OSError):
        # Stdout closed by the runtime; body is still returned (F40).
        pass
