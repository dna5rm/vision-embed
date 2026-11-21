"""Production embed client — HTTPS to the configured vision-embed (FI13).

Sends `include_object_vectors=true` on every call (section 4). One retry on
connect failure, no others. Never falls back to the fake client for any
reason (section 10). TLS verification is never disabled: a self-signed embed
endpoint must be trusted by the host trust store (fail-closed reading; see
EXEC-QUESTIONS-INDEX.md Q3).
"""

from __future__ import annotations

import hashlib
import logging

import httpx

from .embed import (
    EmbedResponse,
    EmbedUnavailableError,
    validate_embed_payload,
)
from ..errors import ApiError

log = logging.getLogger("vision_index.embed_client")


class ProductionEmbedClient:
    def __init__(self, base_url: str, bearer_token: str, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = bearer_token
        self._timeout = timeout_seconds

    def embed(self, image_bytes: bytes) -> EmbedResponse:
        if self._base_url.startswith("http://"):
            raise EmbedUnavailableError("embed.base_url must be https")

        def _call() -> httpx.Response:
            headers = {"Authorization": f"Bearer {self._token}"}
            with httpx.Client(timeout=self._timeout, verify=True) as client:
                return client.post(
                    f"{self._base_url}/v1/embed",
                    files={"image": ("image", image_bytes, "application/octet-stream")},
                    data={"include_object_vectors": "true"},
                    headers=headers,
                )

        try:
            try:
                response = _call()
            except httpx.ConnectError:
                # Single retry on connect failure, nothing else (section 4).
                response = _call()
        except (httpx.HTTPError, OSError):
            raise EmbedUnavailableError("embed service unreachable") from None
        if response.status_code != 200:
            raise EmbedUnavailableError(f"embed status {response.status_code}")
        try:
            return validate_embed_payload(response.content)
        except ApiError:
            # A schema violation is the embed service's bug; it stays
            # response.validation_failed and must NOT be rewrapped as
            # embed.unavailable (502).
            raise
        except Exception as exc:
            # A non-schema failure (e.g. a validator crash) surfaces as
            # embed.unavailable with the failure detail (paths/messages only —
            # never vectors or auth material). True schema violations raise
            # ApiError above and stay response.validation_failed.
            log.error("embed response rejected by validator: %s", exc)
            raise EmbedUnavailableError("embed response invalid") from None

    def ready(self) -> bool:
        try:
            with httpx.Client(timeout=self._timeout, verify=True) as client:
                response = client.get(f"{self._base_url}/readyz")
        except (httpx.HTTPError, OSError):
            return False
        return response.status_code == 200


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
