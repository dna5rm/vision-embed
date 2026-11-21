"""Fake embed client — adapter_mode "test" ONLY (SPEC section 10, FI34).

Deterministic, image-sha256-seeded fixtures in the exact vision-embed success
schema shape. Constructing one under any other mode raises; there is no config
key that selects it and production code never touches it. Sameers can be
grouped for reproducible clustering tests via `persons` (sha256 -> tag) and
`faces_for` (sha256 -> face count).
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from .embed import EmbedResponse, EmbedUnavailableError, validate_embed_payload

SCENE_MODEL = "ViT-L-14/openai"
FACE_MODEL = "buffalo_l/w600k_r50"
SCENE_DIM = 768
FACE_DIM = 512

# The label set real vision-embed reports (COCO-80). Fake defaults keep this
# list so a "fake-only label" can never occur.
COCO80 = [
    "person", "bicycle", "car", "dog", "chair", "cup", "book", "clock",
    "laptop", "cell phone", "bottle", "cat", "bird", "truck", "train", "tv",
]


def _rng_for(*parts: bytes) -> np.random.Generator:
    digest = hashlib.sha256(b"|".join(parts)).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0.0 else vec.astype(np.float32)


class FakeEmbedClient:
    """Raises unless mode == "test" — the construction site is the only switch."""

    def __init__(
        self,
        *,
        mode: str,
        persons: Mapping[str, str] | None = None,
        faces_for: Mapping[str, int] | None = None,
        labels_for: Mapping[str, list[str]] | None = None,
        sizes: Mapping[str, tuple[int, int]] | None = None,
        face_distances: Mapping[str, list[float]] | None = None,
        person_rotations: Mapping[str, tuple[str, float]] | None = None,
        fail: bool = False,
        ready: bool = True,
    ) -> None:
        if mode != "test":
            raise RuntimeError(
                "fake embed client may only be constructed in adapter_mode 'test'"
            )
        self.persons = dict(persons or {})
        self.faces_for = dict(faces_for or {})
        self.labels_for = dict(labels_for or {})
        self.sizes = dict(sizes or {})
        self.face_distances = dict(face_distances or {})
        self.person_rotations = dict(person_rotations or {})
        self.fail = fail
        self._ready = ready
        self.calls: list[str] = []  # sha256 per embed call, assertion surface only

    # -- public API --------------------------------------------------------

    def ready(self) -> bool:
        return self._ready

    def embed(self, image_bytes: bytes) -> EmbedResponse:
        if self.fail:
            raise EmbedUnavailableError("fake embed configured to fail")
        sha = hashlib.sha256(image_bytes).hexdigest()
        self.calls.append(sha)
        person = self.persons.get(sha, "anon:" + sha[:12])
        width, height = self.sizes.get(sha, (640, 480))
        payload = {
            "request_id": "fake-" + sha[:8],
            "image": {"sha256": sha, "width": width, "height": height},
            "scene": {
                "model": SCENE_MODEL,
                "dim": SCENE_DIM,
                "vector": [float(v) for v in _l2(_rng_for(b"scene", sha.encode()).normal(size=SCENE_DIM))],
            },
            "faces": self._faces(sha, person),
            "objects": self._objects(sha, width, height),
        }
        return validate_embed_payload(json.dumps(payload))

    # -- fixture shaping ----------------------------------------------------

    def _tag_vector(self, person: str) -> np.ndarray:
        rot = self.person_rotations.get(person)
        if rot is None:
            return _l2(_rng_for(b"person", person.encode()).normal(size=FACE_DIM))
        base_tag, dist = rot
        base = self._tag_vector(base_tag)
        return _l2(self._rotate(base, person, float(dist)))

    def _rotate(self, base: np.ndarray, person: str, dist: float) -> np.ndarray:
        import math

        theta = math.acos(max(-1.0, min(1.0, 1.0 - dist)))
        h = hashlib.sha256(person.encode()).digest()
        i, j = h[0] % FACE_DIM, h[1] % FACE_DIM
        if i == j:
            j = (i + 1) % FACE_DIM
        v = base.copy()
        c, s = math.cos(theta), math.sin(theta)
        vi, vj = float(v[i]), float(v[j])
        v[i], v[j] = c * vi - s * vj, s * vi + c * vj
        return v

    def _faces(self, sha: str, person: str) -> list[dict]:
        count = int(self.faces_for.get(sha, 1))
        tag = self._tag_vector(person)
        per_shim = self.face_distances.get(sha, [])[:count]
        faces = []
        for i in range(count):
            base = per_shim[i] if i < len(per_shim) else 0.0
            if base > 0.0:
                # Synthetic geometry: ortho is first orthogonalized to the tag
                # so the requested cosine distance is exact in the rotation
                # plane.
                raw_o = _rng_for(b"ortho", sha.encode(), str(i).encode()).normal(size=FACE_DIM)
                raw_o = raw_o - float(np.dot(raw_o, tag)) * tag
                ortho = _l2(raw_o)
                theta = float(np.arccos(np.clip(1.0 - base, -1.0, 1.0)))
                raw = np.cos(theta) * tag + np.sin(theta) * ortho
            else:
                raw = tag + 0.002 * _rng_for(b"jitter", sha.encode(), str(i).encode()).normal(size=FACE_DIM)
            vec = _l2(raw)
            rng = _rng_for(b"det", sha.encode(), str(i).encode())
            bw, bh = 96, 112
            x = int(rng.integers(0, 64))
            y = int(rng.integers(0, 64))
            faces.append(
                {
                    "bbox": {"x": x, "y": y, "w": bw, "h": bh},
                    "det_score": 0.82 + 0.1 * float(rng.random()),
                    "model": FACE_MODEL,
                    "dim": FACE_DIM,
                    "vector": [float(v) for v in vec],
                }
            )
        return faces

    def _objects(self, sha: str, width: int, height: int) -> list[dict]:
        labels = self.labels_for.get(sha)
        if labels is None:
            rng = _rng_for(b"obj", sha.encode())
            labels = [COCO80[int(rng.integers(0, 4))]]
        objects = []
        for i, label in enumerate(labels):
            rng = _rng_for(b"objdet", sha.encode(), str(i).encode())
            bw, bh = max(8, width // 4), max(8, height // 4)
            objects.append(
                {
                    "label": label,
                    "bbox": {
                        "x": int(rng.integers(0, max(1, width - bw))),
                        "y": int(rng.integers(0, max(1, height - bh))),
                        "w": bw,
                        "h": bh,
                    },
                    "score": 0.55 + 0.3 * float(rng.random()),
                    "vector": [
                        float(v)
                        for v in _l2(_rng_for(b"crop", sha.encode(), label.encode()).normal(size=SCENE_DIM))
                    ],
                }
            )
        return objects
