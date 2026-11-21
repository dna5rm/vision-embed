"""Embed adapter — the one outbound surface, vision-embed (INV-9, section 10).

Exactly one Protocol exists here: the vision-embed embed client. It consumes
exactly the fields vision-embed's success schema returns and validates the
outbound payload against that closed schema (section 6b step 9). A mismatch is
`response.validation_failed` — surfaced, never swallowed.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np
from jsonschema import Draft202012Validator

from ..errors import ApiError

SCENE_MODEL_CONST = "ViT-L-14/openai"
FACE_MODEL_CONST = "buffalo_l/w600k_r50"
SCENE_DIM = 768
FACE_DIM = 512

_BBOX = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x", "y", "w", "h"],
    "properties": {"x": {"type": "integer", "minimum": 0}, "y": {"type": "integer", "minimum": 0},
                   "w": {"type": "integer", "minimum": 1}, "h": {"type": "integer", "minimum": 1}},
}


class EmbedUnavailableError(Exception):
    """Connection failure / timeout / non-200 / non-schema of the transport."""


def _embed_response_schema() -> dict:
    """Mirror of vision-embed's closed success schema (spec section 6 preamble)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["request_id", "image", "scene", "faces", "objects"],
        "properties": {
            "request_id": {"type": "string", "minLength": 1},
            "image": {"type": "object", "additionalProperties": False,
                      "required": ["sha256", "width", "height"],
                      "properties": {"sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                                     "width": {"type": "integer", "minimum": 1},
                                     "height": {"type": "integer", "minimum": 1}}},
            "scene": {"type": "object", "additionalProperties": False,
                      "required": ["model", "dim", "vector"],
                      "properties": {"model": {"type": "string"}, "dim": {"const": SCENE_DIM},
                                     "vector": _vec_items(SCENE_DIM)}},
            "faces": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["bbox", "det_score", "model", "dim", "vector"],
                "properties": {"bbox": _BBOX,
                               "det_score": {"type": "number"},
                               "model": {"type": "string"}, "dim": {"const": FACE_DIM},
                               "vector": _vec_items(FACE_DIM)}}},
            "objects": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["label", "bbox", "score"],
                "properties": {"label": {"type": "string", "minLength": 1},
                               "bbox": _BBOX,
                               "score": {"type": "number"},
                               # vision-embed includes model/dim/vector on object
                               # records when include_object_vectors=true
                               # (pipeline.embed_image); schema mirrors that.
                               "model": {"type": "string"}, "dim": {"type": "integer", "minimum": 1},
                               "vector": _vec_items(SCENE_DIM)}}},
        },
    }


def _vec_items(dim: int) -> dict:
    return {"type": "array", "items": {"type": "number"}, "minItems": dim, "maxItems": dim}


_validator = Draft202012Validator(_embed_response_schema())


class EmbedFaces:
    __slots__ = ("bbox", "det_score", "vector")

    def __init__(self, bbox: dict[str, int], det_score: float, vector: np.ndarray) -> None:
        self.bbox = bbox
        self.det_score = det_score
        self.vector = vector


class EmbedObjects:
    __slots__ = ("label", "bbox", "score", "vector")

    def __init__(self, label: str, bbox: dict[str, int], score: float,
                 vector: np.ndarray | None) -> None:
        self.label = label
        self.bbox = bbox
        self.score = score
        self.vector = vector


class EmbedResponse:
    __slots__ = ("sha256", "width", "height", "scene_vector", "faces", "objects")

    def __init__(self, sha256: str, width: int, height: int, scene_vector: np.ndarray,
                 faces: list[EmbedFaces], objects: list[EmbedObjects]) -> None:
        self.sha256 = sha256
        self.width = width
        self.height = height
        self.scene_vector = scene_vector
        self.faces = faces
        self.objects = objects


def validate_embed_payload(raw: bytes | str) -> EmbedResponse:
    """Section 6b step 9: schema-validate the outbound response. Fail →
    APIError response.validation_failed (server bug; surfaced)."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ApiError("response.validation_failed", "embed response is not JSON") from None
    errors = sorted(_validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.path) or "root"
        raise ApiError(
            "response.validation_failed", f"embed response schema violation at {where}"
        )
    scene = payload["scene"]
    if scene["model"] != SCENE_MODEL_CONST:
        raise ApiError("response.validation_failed", "unknown scene model")
    faces = [
        EmbedFaces(
            {k: int(f["bbox"][k]) for k in ("x", "y", "w", "h")},
            float(f["det_score"]),
            np.asarray(f["vector"], dtype=np.float32),
        )
        for f in payload["faces"]
    ]
    objects = [
        EmbedObjects(
            str(o["label"]),
            {k: int(o["bbox"][k]) for k in ("x", "y", "w", "h")},
            float(o["score"]),
            np.asarray(o["vector"], dtype=np.float32) if "vector" in o else None,
        )
        for o in payload["objects"]
    ]
    return EmbedResponse(
        sha256=str(payload["image"]["sha256"]),
        width=int(payload["image"]["width"]),
        height=int(payload["image"]["height"]),
        scene_vector=np.asarray(scene["vector"], dtype=np.float32),
        faces=faces,
        objects=objects,
    )


@runtime_checkable
class EmbedClient(Protocol):
    """The single outbound client surface (production: HTTPS to vision-embed)."""

    def embed(self, image_bytes: bytes) -> EmbedResponse: ...

    def ready(self) -> bool: ...
