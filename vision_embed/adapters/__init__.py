"""Adapter Protocols — exactly four roles (SPEC section 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Tuple, runtime_checkable

import numpy as np

# (x, y, w, h, det_score, kps) — kps = 5 (x, y) pairs in original-image pixels
DetFaceRecord = Tuple[int, int, int, int, float, Tuple[Tuple[float, float], ...]]
# (label, x, y, w, h, score)
DetObjectRecord = Tuple[str, int, int, int, int, float]


@runtime_checkable
class FaceEmbedder(Protocol):
    def load(self, path: Path) -> None: ...

    def embed(
        self, bgr: np.ndarray, kps: tuple[tuple[float, float], ...]
    ) -> tuple[str, int, np.ndarray]: ...


@runtime_checkable
class Detector(Protocol):
    def load(self, path: Path) -> None: ...

    def detect(self, bgr: np.ndarray) -> list[DetFaceRecord]: ...


@runtime_checkable
class SceneEmbedder(Protocol):
    def load(self, path: Path) -> None: ...

    def embed(self, rgb: np.ndarray) -> tuple[str, int, np.ndarray]: ...


@runtime_checkable
class ObjectDetector(Protocol):
    def load(self, path: Path) -> None: ...

    def detect(self, rgb: np.ndarray) -> list[DetObjectRecord]: ...
