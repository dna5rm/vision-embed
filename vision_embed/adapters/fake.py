"""Fake adapter set for all four roles (SPEC section 7, F25).

Deterministic, seeded by the image sha256. The full request/response/audit
path runs with zero ONNX files present. Used only via models.adapter=fake +
adapter_mode=test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import DetFaceRecord, DetObjectRecord

SCENE_MODEL = "ViT-L-14/openai"
SCENE_DIM = 768
FACE_MODEL = "buffalo_l/w600k_r50"
FACE_DIM = 512
COCO_LABELS = [
    "person", "chair", "car", "dog",
]


def _rng_for(*parts: bytes) -> np.random.Generator:
    digest = hashlib.sha256(b"|".join(parts)).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class FakeFaceEmbedder:
    """Fake face embedder: deterministic 512-d vector from kps geometry."""

    def load(self, path: Path) -> None:  # loads nothing, by design
        return None

    def embed(
        self, bgr: np.ndarray, kps: tuple[tuple[float, float], ...]
    ) -> tuple[str, int, np.ndarray]:
        seed = np.ascontiguousarray(np.asarray(kps, dtype=np.float64)).tobytes()
        rng = _rng_for(b"face", seed)
        vec = _l2(rng.normal(size=FACE_DIM))
        return FACE_MODEL, FACE_DIM, vec


class FakeDetector:
    """Fake face detector: one deterministic fixture face inside the image."""

    def load(self, path: Path) -> None:
        return None

    def detect(self, bgr: np.ndarray) -> list[DetFaceRecord]:
        h, w = bgr.shape[0], bgr.shape[1]
        rng = _rng_for(b"det", str(w).encode(), b"x", str(h).encode())
        bw = max(8, int(w * (0.2 + 0.05 * float(rng.random()))))
        bh = max(8, int(h * (0.2 + 0.05 * float(rng.random()))))
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        kps = [
            (float(x + 0.30 * bw), float(y + 0.35 * bh)),
            (float(x + 0.70 * bw), float(y + 0.35 * bh)),
            (float(x + 0.50 * bw), float(y + 0.55 * bh)),
            (float(x + 0.35 * bw), float(y + 0.75 * bh)),
            (float(x + 0.65 * bw), float(y + 0.75 * bh)),
        ]
        score = 0.85 + 0.1 * float(rng.random())
        return [(x, y, bw, bh, float(score), tuple(kps))]


class FakeSceneEmbedder:
    """Fake scene embedder: deterministic 768-d vector from pixel content."""

    def load(self, path: Path) -> None:
        return None

    def embed(self, rgb: np.ndarray) -> tuple[str, int, np.ndarray]:
        rng = _rng_for(b"scene", np.ascontiguousarray(rgb).tobytes())
        vec = _l2(rng.normal(size=SCENE_DIM))
        return SCENE_MODEL, SCENE_DIM, vec


class FakeObjectDetector:
    """Fake object detector: deterministic COCO-80-label fixture list."""

    def load(self, path: Path) -> None:
        return None

    def detect(self, rgb: np.ndarray) -> list[DetObjectRecord]:
        h, w = rgb.shape[0], rgb.shape[1]
        rng = _rng_for(b"objdet", str(w).encode(), b"x", str(h).encode())
        count = 1 + int(rng.integers(0, 2))  # one or two objects
        out: list[DetObjectRecord] = []
        for i in range(count):
            label = COCO_LABELS[int(rng.integers(0, len(COCO_LABELS)))]
            bw = max(8, int(w * (0.15 + 0.2 * float(rng.random()))))
            bh = max(8, int(h * (0.15 + 0.2 * float(rng.random()))))
            x = int(rng.integers(0, max(1, w - bw)))
            y = int(rng.integers(0, max(1, h - bh)))
            score = 0.55 + 0.3 * float(rng.random())
            out.append((label, x, y, bw, bh, float(score)))
        return out


# --- v12 fake describe/tagger roles (SPEC section 7, section 2a/2b) ---

DESCRIBE_TEXT_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>")
DESCRIBE_REGION_TASKS = ("<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<OCR_WITH_REGION>")

TAG_VOCAB = [
    "1girl", "solo", "long_hair", "brown_hair", "smile", "standing",
    "upper_body", "outdoors", "sky", "tree", "day", "blue_sky",
]
# Fixed score slots spanning well above and below a 0.5 threshold so threshold
# cutting is always testable; the rng adds a small deterministic jitter only.
TAG_SCORE_SLOTS = (0.95, 0.78, 0.62, 0.51, 0.44, 0.35, 0.27, 0.19, 0.13, 0.08, 0.05, 0.03)


class FakeFlorence:
    """Fake Florence-2 role: deterministic text/regions seeded by the image
    content and the requested task. Zero files loaded, by design."""

    def load(self, path: Path) -> None:
        return None

    def run(
        self, rgb: np.ndarray, task: str
    ) -> tuple[str, list[tuple[str, int, int, int, int, float | None, str | None]]]:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        rng = _rng_for(b"florence", task.encode(), np.ascontiguousarray(rgb).tobytes())
        digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()[:8]
        if task in DESCRIBE_TEXT_TASKS:
            if task == "<OCR>":
                text = f" recognized text sample {digest} "
            else:
                text = f"a small image with {w} by {h} pixels, sample {digest}"
            return text, []
        regions: list[tuple[str, int, int, int, int, float | None, str | None]] = []
        bw = max(8, int(w * 0.5))
        bh = max(8, int(h * 0.5))
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if task == "<OD>":
            labels = ["chair", "car"]
            for i in range(2):
                ox = (x + i) % max(1, w - 8)
                oy = (y + i) % max(1, h - 8)
                score = 0.55 + 0.4 * float(rng.random())
                regions.append(
                    (labels[i], int(ox), int(oy), max(1, min(bw, w - int(ox))), max(1, min(bh, h - int(oy))), float(score), None)
                )
        elif task == "<DENSE_REGION_CAPTION>":
            regions.append((f"sample region {digest}", x, y, x2 - x, y2 - y, None, None))
        elif task == "<REGION_PROPOSAL>":
            regions.append(("", x, y, max(1, x2 - x), max(1, y2 - y), None, None))
        elif task == "<OCR_WITH_REGION>":
            regions.append((f"text {digest}", x, y, max(1, x2 - x), max(1, y2 - y), None, f"text {digest}"))
        return "", regions


class FakeTagger:
    """Fake WD14 tagger: deterministic [(tag, score)] cut at the caller
    threshold, score-descending, seeded by the image content only."""

    def load(self, path: Path) -> None:
        return None

    def tag(self, rgb: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        rng = _rng_for(b"tagger", np.ascontiguousarray(rgb).tobytes())
        pairs = [
            (TAG_VOCAB[i], TAG_SCORE_SLOTS[i] + 0.02 * float(rng.random()) - 0.01)
            for i in range(len(TAG_VOCAB))
        ]
        pairs = [(t, s) for (t, s) in pairs if s >= threshold]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs
