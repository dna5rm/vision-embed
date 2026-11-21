"""Tests for class-agnostic face NMS (nms.nms_records)."""

from __future__ import annotations

from vision_embed.nms import nms_records

# Face path threshold (pipeline.py passes 0.40 explicitly; YOLO path stays 0.45).
FACE_IOU = 0.40


def _face(x, y, w, h, score):
    # DetFaceRecord shape: (x, y, w, h, score, kps) with dummy kps
    return (x, y, w, h, score, ((0.0, 0.0),) * 5)


def test_overlapping_boxes_collapse_to_one():
    # Two 200x200 boxes offset by 20px in both axes: IoU = 180*180/(2*40000 - 32400) ≈ 0.68 > 0.40
    boxes = [
        _face(100, 100, 200, 200, 0.80),
        _face(120, 120, 200, 200, 0.89),
    ]
    kept = nms_records(boxes, iou_threshold=FACE_IOU)
    assert len(kept) == 1
    assert kept[0][4] == 0.89  # highest score survives


def test_residual_duplicate_scan_at_iou_043_collapses():
    # duplicate detector scans of the same face land at IoU ~0.43: must collapse.
    d = 22.450106292877393  # 100x100 boxes offset 22.45px -> IoU == 0.43
    boxes = [
        _face(0, 0, 100, 100, 0.80),
        _face(d, d, 100, 100, 0.89),
    ]
    kept = nms_records(boxes, iou_threshold=FACE_IOU)
    assert len(kept) == 1
    assert kept[0][4] == 0.89


def test_distinct_face_at_iou_020_survives():
    # a genuinely distinct face must survive: offset 42.26px -> IoU == 0.20
    d = 42.264973081037425
    boxes = [
        _face(0, 0, 100, 100, 0.80),
        _face(d, d, 100, 100, 0.89),
    ]
    kept = nms_records(boxes, iou_threshold=FACE_IOU)
    assert len(kept) == 2
    assert {kept[0][4], kept[1][4]} == {0.80, 0.89}


def test_iou_040_boundary_is_kept():
    # Exact IoU == 0.40 must be kept (rule is suppression only when IoU > 0.40).
    d = 24.407105398154556  # 100x100 boxes offset -> IoU == 0.40 exactly
    boxes = [_face(0.0, 0.0, 100.0, 100.0, 0.9), _face(d, d, 100.0, 100.0, 0.8)]
    kept = nms_records(boxes, iou_threshold=FACE_IOU)
    assert len(kept) == 2


def test_distant_third_box_survives():
    boxes = [
        _face(100, 100, 200, 200, 0.80),
        _face(120, 120, 200, 200, 0.89),
        _face(1000, 1000, 200, 200, 0.60),  # zero overlap with the others
    ]
    kept = nms_records(boxes, iou_threshold=FACE_IOU)
    assert len(kept) == 2
    assert {kept[0][4], kept[1][4]} == {0.89, 0.60}


def test_results_sorted_by_score_desc():
    boxes = [
        _face(0, 0, 50, 50, 0.55),
        _face(600, 600, 50, 50, 0.95),
        _face(300, 300, 50, 50, 0.75),
    ]
    kept = nms_records(boxes, iou_threshold=FACE_IOU)
    assert [f[4] for f in kept] == [0.95, 0.75, 0.55]


def test_non_overlapping_pair_both_kept():
    # Offset by 300px on a 200x200 box: no overlap at all -> both survive.
    boxes = [
        _face(0, 0, 200, 200, 0.85),
        _face(300, 300, 200, 200, 0.70),
    ]
    assert len(nms_records(boxes, iou_threshold=FACE_IOU)) == 2


def test_empty_input():
    assert nms_records([], iou_threshold=FACE_IOU) == []


def test_yolo_default_threshold_unchanged():
    # The YOLO path relies on the 0.45 default; do not tighten it.
    d = 21.21614028416647  # 100x100 boxes offset 21.22px -> IoU == 0.45 exactly
    boxes = [
        _face(0, 0, 100, 100, 0.80),
        _face(d, d, 100, 100, 0.89),
    ]
    assert len(nms_records(boxes)) == 2  # default threshold call, no explicit iou
