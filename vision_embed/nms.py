"""Class-agnostic hard NMS shared by face and object detection paths.

Kept outside adapters/onnx.py so it is unit-testable without onnxruntime.
Detection records are (x, y, w, h, score, ...) tuples in image coordinates.
"""

from __future__ import annotations

from typing import Any, Sequence


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union) if union > 0 else 0.0


def nms_records(records: Sequence[tuple], iou_threshold: float = 0.45) -> list[tuple]:
    """Class-agnostic NMS, highest score first, same rule as the YOLO path.

    Records are (x, y, w, h, score, ...) tuples. Suppresses any record whose
    IoU with an already-kept (higher-score) record exceeds iou_threshold.
    """
    ordered = sorted(records, key=lambda r: float(r[4]), reverse=True)
    kept: list[tuple] = []
    for record in ordered:
        if all(_iou_xywh(record, kept_rec) <= iou_threshold for kept_rec in kept):
            kept.append(record)
    return kept
