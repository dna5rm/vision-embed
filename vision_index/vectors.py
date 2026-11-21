"""Vector math — cosine distance = 1 - dot(a, b) on L2-normalized float32 vectors.

Lower is closer (SPEC section 2). Every vector written to storage is
re-normalized to unit length here, defensively; no other transform ever
applies.
"""

from __future__ import annotations

import numpy as np


def to_float32_unit(values) -> np.ndarray:
    """Cast to float32 and L2-normalize to unit length (defensive, section 2)."""
    vec = np.asarray(values, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return np.ascontiguousarray(vec, dtype=np.float32)


def vec_to_blob(vec: np.ndarray) -> bytes:
    """Raw little-endian float32 bytes for a sqlite-vec embedding column."""
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """1 - dot for one query against a (n, dim) candidate matrix."""
    return 1.0 - (np.asarray(candidates, dtype=np.float32) @ query.astype(np.float32))
