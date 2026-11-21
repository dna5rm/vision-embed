"""Embedding pipeline — steps 4-6 of the processing order, plus the closed response schema."""

from __future__ import annotations

import hashlib

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

from .nms import nms_records

from .adapters import Detector, FaceEmbedder, ObjectDetector, SceneEmbedder
from .errors import ApiError

FACE_MODEL = "buffalo_l/w600k_r50"
SCENE_MODEL = "ViT-L-14/openai"

_BBOX = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x", "y", "w", "h"],
    "properties": {
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "w": {"type": "integer", "minimum": 1},
        "h": {"type": "integer", "minimum": 1},
    },
}

# Closed success schema (SPEC section 6), additionalProperties: false everywhere.
RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["request_id", "image", "scene", "faces", "objects"],
    "properties": {
        "request_id": {"type": "string", "minLength": 1},
        "image": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sha256", "width", "height"],
            "properties": {
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
            },
        },
        "scene": {
            "type": "object",
            "additionalProperties": False,
            "required": ["model", "dim", "vector"],
            "properties": {
                "model": {"const": SCENE_MODEL},
                "dim": {"const": 768},
                "vector": {"type": "array", "items": {"type": "number"}, "minItems": 768, "maxItems": 768},
            },
        },
        "faces": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["bbox", "det_score", "model", "dim", "vector"],
                "properties": {
                    "bbox": _BBOX,
                    "det_score": {"type": "number"},
                    "model": {"const": FACE_MODEL},
                    "dim": {"const": 512},
                    "vector": {"type": "array", "items": {"type": "number"}, "minItems": 512, "maxItems": 512},
                },
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "bbox", "score"],
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "bbox": _BBOX,
                    "score": {"type": "number"},
                    "model": {"const": SCENE_MODEL},
                    "dim": {"const": 768},
                    "vector": {"type": "array", "items": {"type": "number"}, "minItems": 768, "maxItems": 768},
                },
            },
        },
    },
}

_validator = Draft202012Validator(RESPONSE_SCHEMA)


def sniff_format(data: bytes) -> str | None:
    """Magic-byte sniff; returns the mime type or None (SPEC section 6 step 3)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate_response(payload: dict) -> None:
    errors = sorted(_validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.path) or "<root>"
        raise ApiError(500, "response.validation_failed", f"schema violation at {path}: {first.message}")  # type: ignore[arg-type]


def embed_image(
    *,
    raw: bytes,
    rgb: np.ndarray,
    detector: Detector,
    face_embedder: FaceEmbedder,
    scene_embedder: SceneEmbedder,
    object_detector: ObjectDetector,
    det_score_min: float,
    object_score_min: float,
    max_faces: int,
    include_object_vectors: bool,
    request_id: str,
) -> dict:
    """Decode, run all model roles, and build the schema-validated response body."""
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    sha = hashlib.sha256(raw).hexdigest()
    height, width = rgb.shape[0], rgb.shape[1]

    try:
        detected = detector.detect(bgr)
        faces = [f for f in detected if f[4] >= det_score_min]
        # Face NMS runs tighter than the YOLO path (0.40 vs 0.45): residual
        # duplicate-scans of the same face from the detector sit at IoU 0.42-0.44
        # and must collapse to a single face.
        faces = nms_records(faces, iou_threshold=0.40)[:max_faces]
        face_records = []
        for x, y, w, h, score, kps in faces:
            model, dim, vec = face_embedder.embed(bgr, kps)
            face_records.append(
                {
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "det_score": float(score),
                    "model": model,
                    "dim": dim,
                    "vector": [float(v) for v in vec],
                }
            )
        detected_objects = object_detector.detect(rgb)
        object_records = []
        for label, x, y, w, h, score in detected_objects:
            record = {
                "label": label,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "score": float(score),
            }
            if include_object_vectors:
                x1, y1 = min(width, x + w), min(height, y + h)
                crop = rgb[y:y1, x:x1]
                model, dim, vec = scene_embedder.embed(np.ascontiguousarray(crop))
                record["model"] = model
                record["dim"] = dim
                record["vector"] = [float(v) for v in vec]
            object_records.append(record)
        scene_model, scene_dim, scene_vec = scene_embedder.embed(rgb)
    except ApiError:
        raise
    except Exception as exc:  # any model failure → inference.failed
        raise ApiError(500, "inference.failed", f"model inference failed: {type(exc).__name__}") from exc

    payload = {
        "request_id": request_id,
        "image": {"sha256": sha, "width": width, "height": height},
        "scene": {
            "model": scene_model,
            "dim": scene_dim,
            "vector": [float(v) for v in scene_vec],
        },
        "faces": face_records,
        "objects": object_records,
    }
    _validate_response(payload)
    # No identity fields exist anywhere in this payload (INV-2, F27).
    return payload
