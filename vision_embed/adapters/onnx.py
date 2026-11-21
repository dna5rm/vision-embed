"""Production ONNX adapters (SPEC sections 2, 2a, 2b, 7).

Loads operator-supplied ONNX files from config paths. No model download at
runtime (F35). Session creation is isolated in create_session() so a later
CUDAExecutionProvider flip is a config change plus this function only.

These paths are exercised only when real weight files are supplied by the
operator; the fake adapter is the tested path in this build.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import DetFaceRecord, DetObjectRecord

FACE_MODEL = "buffalo_l/w600k_r50"
SCENE_MODEL = "ViT-L-14/openai"

# ArcFace reference 5-point template for 112x112 (standard ArcFace similarity warp src).
_ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# CLIP ViT-L/14 preprocessing constants (openai pretrained per model card).
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_CLIP_SIZE = 224

# COCO-80 class names, snake-cased (SPEC section 2a). Pinned set, no dynamic mapping.
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def create_session(path: Path, provider: str) -> ort.InferenceSession:
    """Single place where an ONNX Runtime session is built.

    Two-value allowlist; config validation already rejected anything else and
    this is the final backstop. Single-element provider list so failure is
    explicit — the activation gate below never falls back.
    """
    if provider not in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        raise ValueError(f"execution_provider {provider!r} is not enabled")
    session = ort.InferenceSession(str(path), providers=[provider])
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"onnxruntime did not activate {provider}")
    return session


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec.astype(np.float32) if norm == 0.0 else (vec / norm).astype(np.float32)


def _resize_to_640_bgr(image: np.ndarray, pad_value: int = 114) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(640 / h, 640 / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = np.array(_pil_like_resize(image, (nw, nh)), dtype=np.uint8)
    padded = np.full((640, 640, 3), pad_value, dtype=np.uint8)
    padded[:nh, :nw] = resized
    return padded, scale, 0, 0


def _pil_like_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # Bilinear resize implemented in numpy for a pinned preprocessing path.
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


class OnnxDetector:
    """SCRFD-10G face detector (det_10g.onnx, buffalo_l pack)."""

    def __init__(self, provider: str) -> None:
        self._provider = provider
        self._session: ort.InferenceSession | None = None
        self._input_name = "input.1"

    def load(self, path: Path) -> None:
        self._session = create_session(path, self._provider)
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, bgr: np.ndarray) -> list[DetFaceRecord]:
        if self._session is None:
            raise RuntimeError("detector not loaded")
        h, w = bgr.shape[:2]
        blob, scale, ox, oy = _resize_to_640_bgr(bgr)
        x = ((blob[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]  # BGR->RGB, normalized per scrfd (input_mean 127.5, std 128)
        outputs = self._session.run(None, {self._input_name: x})
        # Real det_10g.onnx output order matches InsightFace scrfd.py
        # forward(): per-stride scores[0..fmc), then bboxes[0..fmc), then
        # kpss[0..fmc) (verified shapes 12800/3200/800 branch-grouped).
        fmc = len(outputs) // 3  # 3 strides
        scores, bboxes, kpss = outputs[:fmc], outputs[fmc : 2 * fmc], outputs[2 * fmc :]
        blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for idx in range(fmc):
            blocks.append((scores[idx], bboxes[idx], kpss[idx]))
        # Anchor centers per stride.
        results: list[tuple[float, np.ndarray, np.ndarray]] = []
        for idx, (score, bbox, kps) in enumerate(blocks):
            stride = 8 * (2 ** idx)
            height_anchors = 640 // stride
            width_anchors = 640 // stride
            # Real det_10g.onnx outputs (verified on the weights):
            # scores/bbox/kps are (h*w*num_anchors, *) with the num_anchors
            # copies of each cell center CONSECUTIVE (InsightFace scrfd.py
            # np.stack([anchor_centers]*num_anchors, axis=1).reshape(-1,2)).
            s = score.reshape(-1)
            b = bbox.reshape(-1, 4)
            k = kps.reshape(-1, 10)
            keep = np.nonzero(s >= 0.02)[0]
            if len(keep) == 0:
                continue
            num_anchors = max(1, b.shape[0] // (height_anchors * width_anchors))
            ax = (np.arange(width_anchors, dtype=np.float32) * stride).repeat(height_anchors)
            ay = np.tile(np.arange(height_anchors, dtype=np.float32) * stride, width_anchors)
            centers_cell = np.stack([ax, ay], axis=1).reshape(height_anchors * width_anchors, 2)
            centers_cell = np.repeat(centers_cell, num_anchors, axis=0)
            centers = centers_cell[keep]
            if num_anchors * height_anchors * width_anchors != b.shape[0]:
                raise ValueError(
                    f"det output anchor count {b.shape[0]} does not match grid "
                    f"{num_anchors}x{height_anchors}x{width_anchors}"
                )
            boxes = np.concatenate(
                [
                    centers[:, :1] - b[keep, 0:1] * stride,
                    centers[:, 1:2] - b[keep, 1:2] * stride,
                    centers[:, :1] + b[keep, 2:3] * stride,
                    centers[:, 1:2] + b[keep, 3:4] * stride,
                ],
                axis=1,
            )
            # kps: (n, 5, 2) -> (n, 10) interleaved x0,y0,x1,y1,...
            pts_rel = k[keep].reshape(-1, 5, 2) * stride
            points = (centers_cell[keep][:, None, :] + pts_rel).reshape(keep.shape[0], 10)
            results.extend(zip(s[keep].tolist(), boxes, points))
        results.sort(key=lambda t: t[0], reverse=True)
        faces: list[DetFaceRecord] = []
        for score, box, points in results:
            x0 = max(0, int((box[0] - ox) / scale))
            y0 = max(0, int((box[1] - oy) / scale))
            x1 = min(w, int((box[2] - ox) / scale))
            y1 = min(h, int((box[3] - oy) / scale))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            kps = (
                (float(points[0] / scale), float(points[1] / scale)),
                (float(points[2] / scale), float(points[3] / scale)),
                (float(points[4] / scale), float(points[5] / scale)),
                (float(points[6] / scale), float(points[7] / scale)),
                (float(points[8] / scale), float(points[9] / scale)),
            )
            faces.append((x0, y0, x1 - x0, y1 - y0, float(score), kps))
        return faces[:30]


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate the 2x3 affine similarity transform mapping src -> dst (Umeyama)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_d = src - src_mean
    dst_d = dst - dst_mean
    norm_src = np.linalg.norm(src_d) / math.sqrt(num)
    norm_dst = np.linalg.norm(dst_d) / math.sqrt(num)
    norm_dst_src = norm_dst / norm_src
    src_dn = src_d / norm_src
    dst_dn = dst_d / norm_dst
    u, d, vt = np.linalg.svd(np.dot(dst_dn.T, src_dn))
    t = np.eye(dim + 1, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -d[-1]
    m = np.dot(u, np.dot(np.diag(d), vt)).T
    # affine part
    R = m
    scale = norm_dst_src
    t[:dim, :dim] = scale * R
    t[:dim, dim] = (
        dst_mean - scale * np.dot(R, src_mean)
    ).reshape(-1)
    return t.astype(np.float32)


class OnnxFaceEmbedder:
    """w600k_r50 recognizer: ArcFace 5-point similarity warp to 112x112, then embed."""

    def __init__(self, provider: str) -> None:
        self._provider = provider
        self._session: ort.InferenceSession | None = None

    def load(self, path: Path) -> None:
        self._session = create_session(path, self._provider)

    def embed(
        self, bgr: np.ndarray, kps: tuple[tuple[float, float], ...]
    ) -> tuple[str, int, np.ndarray]:
        if self._session is None:
            raise RuntimeError("face embedder not loaded")
        from PIL import Image

        rgb = bgr[:, :, ::-1]
        src = np.asarray(kps, dtype=np.float32)[:5]
        m = _similarity_transform(src, _ARCFACE_SRC)
        full = np.asarray(Image.fromarray(rgb), dtype=np.uint8)
        h, w = full.shape[:2]
        # Apply affine warp via inverse mapping with PIL's transform.
        m_inv = np.linalg.inv(
            np.vstack([m[:2], [0, 0, 1]])
        )[:2].astype(np.float64)
        warped = np.asarray(
            Image.fromarray(full).transform(
                (112, 112),
                Image.AFFINE,
                data=tuple(m_inv.flatten()),
                resample=Image.BILINEAR,
            ),
            dtype=np.float32,
        )
        # Normalize (pixel/255 - 0.5) / 0.5 over RGB channels.
        blob = ((warped[:, :, ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self._session.run(None, {self._session.get_inputs()[0].name: blob.astype(np.float32)})[0]
        vec = np.asarray(out).squeeze()
        assert vec.shape == (512,), f"unexpected face vector dim {vec.shape}"
        return FACE_MODEL, 512, _l2(vec)


class OnnxSceneEmbedder:
    """CLIP ViT-L/14 visual tower via operator-supplied ONNX, 768-d output."""

    def __init__(self, provider: str) -> None:
        self._provider = provider
        self._session: ort.InferenceSession | None = None

    def load(self, path: Path) -> None:
        self._session = create_session(path, self._provider)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(rgb)
        # Bicubic resize shortest edge to 224, then center crop 224.
        w, h = img.size
        scale = _CLIP_SIZE / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        w, h = img.size
        left = (w - _CLIP_SIZE) // 2
        top = (h - _CLIP_SIZE) // 2
        img = img.crop((left, top, left + _CLIP_SIZE, top + _CLIP_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = ((arr - _CLIP_MEAN) / _CLIP_STD).transpose(2, 0, 1)[None]
        return arr.astype(np.float32)

    def embed(self, rgb: np.ndarray) -> tuple[str, int, np.ndarray]:
        if self._session is None:
            raise RuntimeError("scene embedder not loaded")
        out = self._session.run(None, {self._session.get_inputs()[0].name: self._preprocess(rgb)})[0]
        vec = np.asarray(out).squeeze().astype(np.float32)
        assert vec.shape == (768,), f"unexpected scene vector dim {vec.shape}"
        return SCENE_MODEL, 768, _l2(vec)


class OnnxObjectDetector:
    """YOLOv8n COCO-80 detector via operator-supplied ONNX (SPEC section 2a, F11)."""

    def __init__(self, provider: str, score_min: float, max_objects: int = 30) -> None:
        self._provider = provider
        self._score_min = score_min
        self._max_objects = max_objects
        self._session: ort.InferenceSession | None = None

    def load(self, path: Path) -> None:
        self._session = create_session(path, self._provider)

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = rgb.shape[:2]
        scale = min(640 / h, 640 / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        from PIL import Image

        resized = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        ox, oy = (640 - nw) // 2, (640 - nh) // 2
        padded[oy : oy + nh, ox : ox + nw] = resized
        return padded, scale, ox, oy

    def detect(self, rgb: np.ndarray) -> list[DetObjectRecord]:
        if self._session is None:
            raise RuntimeError("object detector not loaded")
        h, w = rgb.shape[:2]
        padded, scale, ox, oy = self._letterbox(rgb)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = np.asarray(
            self._session.run(None, {self._session.get_inputs()[0].name: blob})[0]
        )  # (1, 84, 8400)
        pred = out.reshape(out.shape[1], out.shape[2]).T  # (8400, 84)
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]  # per-class confidence
        best_cls = class_scores.argmax(axis=1)
        best_score = class_scores.max(axis=1)
        keep = best_score >= 0.001
        cands: list[tuple[float, int, np.ndarray]] = []
        for i in np.nonzero(keep)[0]:
            cands.append((float(best_score[i]), int(best_cls[i]), boxes_xywh[i]))
        cands.sort(key=lambda t: t[0], reverse=True)
        results: list[tuple[float, int, np.ndarray]] = []
        for score, cls, box in cands:
            # Class-agnostic NMS at IoU 0.45.
            ok = True
            for _, _, kept in results:
                if _iou_xywh(box, kept) > 0.45:
                    ok = False
                    break
            if ok:
                results.append((score, cls, box))
            if len(results) >= self._max_objects:
                break
        out_records: list[DetObjectRecord] = []
        for score, cls, box in results:
            cx, cy, bw, bh = box
            x0 = int(round((cx - bw / 2 - ox) / scale))
            y0 = int(round((cy - bh / 2 - oy) / scale))
            x1 = int(round((cx + bw / 2 - ox) / scale))
            y1 = int(round((cy + bh / 2 - oy) / scale))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            if score < self._score_min:
                continue
            out_records.append((COCO_80[cls], x0, y0, x1 - x0, y1 - y0, float(score)))
        return out_records


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union) if union > 0 else 0.0
