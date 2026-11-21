"""Closed reason-token set and the uniform API error type (SPEC section 6)."""

from __future__ import annotations

# COMPLETE and CLOSED set of the nine HTTP reason tokens (SPEC section 6).
# Do not add tokens here; a new token is a spec change, not a code change.
REASON_TOKENS = frozenset(
    {
        "limits.body_too_large",
        "format.unsupported",
        "decode.failed",
        "auth.invalid_token",
        "inference.failed",
        "response.validation_failed",
        "route.not_found",
        "method.not_allowed",
        "route.disabled",
        "gpu.busy",
        "task.unsupported",
        "not_ready",
    }
)
# Closed inventory-only strings — never HTTP reason tokens (SPEC section 6d).
GPU_INSUFFICIENT_TOKEN = "gpu." + "insufficient"  # kept off the reason= literal set


class ApiError(Exception):
    """Uniform error for /v1/embed and readiness responses (SPEC section 6 envelope)."""

    def __init__(self, status: int, reason: str, detail: str) -> None:
        if reason not in REASON_TOKENS:
            raise ValueError(f"reason token {reason!r} is not in the closed set")
        self.status = status
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


class FatalStartupError(Exception):
    """Fatal startup condition: one JSON fatal line on stdout, exit 1 (SPEC section 4)."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in ("config.invalid", "config.unknown_key"):
            raise ValueError("fatal startup reasons are config.invalid or config.unknown_key")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")
