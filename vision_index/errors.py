"""Closed reason-token set, uniform API error, fatal startup (SPEC section 6c/4).

REASON_TOKENS is COMPLETE and CLOSED — 16 tokens, exactly the set of section
6c. Do not add or remove tokens; a new token is a spec change, not a code
change. Test 19 asserts this set appears in code and no others.
"""

from __future__ import annotations

# COMPLETE and CLOSED set of the 16 HTTP reason tokens (SPEC section 6c).
REASON_TOKENS = frozenset(
    {
        "limits.body_too_large",
        "format.unsupported",
        "decode.failed",
        "path.too_long",
        "not_found",
        "route.not_found",
        "method.not_allowed",
        "auth.invalid_token",
        "name.conflict",
        "name.too_long",
        "cluster.not_face",
        "embed.unavailable",
        "no_face",
        "db.error",
        "not_ready",
        "response.validation_failed",
    }
)

_TOKEN_HTTP = {
    "limits.body_too_large": 413,
    "format.unsupported": 415,
    "decode.failed": 422,
    "path.too_long": 422,
    "not_found": 404,
    "route.not_found": 404,
    "method.not_allowed": 405,
    "auth.invalid_token": 401,
    "name.conflict": 409,
    "name.too_long": 422,
    "cluster.not_face": 409,
    "embed.unavailable": 502,
    "no_face": 422,
    "db.error": 500,
    "not_ready": 503,
    "response.validation_failed": 500,
}

# Reason tokens that a fatal startup line may use. Section 4 pins
# `config.invalid` (including unknown keys); no other fatal token exists.
FATAL_REASONS = frozenset({"config.invalid"})


class ApiError(Exception):
    """Uniform error for the closed surface (SPEC section 6c envelope shape)."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASON_TOKENS:
            raise ValueError(f"reason token {reason!r} is not in the closed set")
        self.reason = reason
        self.status = _TOKEN_HTTP[reason]
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


class FatalStartupError(Exception):
    """Fatal startup condition: one JSON fatal line on stdout, exit 1."""

    def __init__(self, detail: str, reason: str = "config.invalid") -> None:
        if reason not in FATAL_REASONS:
            raise ValueError("fatal startup uses only config.invalid")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def fatal_line(reason: str, detail: str) -> str:
    """The one JSON fatal line (SPEC section 4 rules)."""
    import json

    return json.dumps(
        {"fatal": True, "reason": reason, "detail": detail}, separators=(",", ":")
    )
