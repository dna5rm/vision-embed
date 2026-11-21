"""Production describe/tagger adapters (SPEC section 2: Florence-2 via
transformers from an operator-supplied local directory; WD14 tagger via
onnxruntime from an operator-supplied ONNX file). No runtime downloads, no
HF tokens, no silent CPU fallback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fake import DESCRIBE_TEXT_TASKS


class RealFlorenceDescribe:
    """Florence-2 through transformers processor+model from a local directory."""

    def __init__(self, execution_provider: str, max_new_tokens: int) -> None:
        self._provider = execution_provider
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self, path: Path) -> None:
        from PIL import Image as _Image  # noqa: F401  (processor consumes PIL input)
        from transformers import AutoModelForCausalLM, AutoProcessor

        p = Path(path)
        if not p.is_dir():
            raise RuntimeError(f"florence directory missing: {p}")
        # v12 QA: this is an operator-supplied local directory with pinned
        # custom Florence-2 code (auto_map to modeling_florence2); loading it
        # is by definition executing that local code.
        self._processor = AutoProcessor.from_pretrained(
            str(p), local_files_only=True, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(p), local_files_only=True, trust_remote_code=True
        )
        if self._provider == "CUDAExecutionProvider":
            model.to("cuda")
            import torch

            if not next(model.parameters()).is_cuda:
                raise RuntimeError("florence model was not placed on the CUDA device")
            self._device = "cuda"
        self._model = model
        self._torch = torch

    def run(
        self, rgb: np.ndarray, task: str
    ) -> tuple[str, list[tuple[str, int, int, int, int, float | None, str | None]]]:
        import json as _json
        from PIL import Image

        pil = Image.fromarray(np.ascontiguousarray(rgb))
        inputs = self._processor(text=task, images=pil, return_tensors="pt")
        if self._device == "cuda":
            inputs = inputs.to("cuda")
        generated = self._model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, do_sample=False, num_beams=1
        )
        decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        # This pinned Florence-2 checkpoint emits its special tokens BEFORE the
        # answer ("</s><s>answer</s>"), and its region answers in loc_-token
        # form; the shipped processor owns the conversion to real boxes, so
        # route every answer through post_process_generation instead of
        # hand-parsing the raw string.
        try:
            parsed = self._processor.post_process_generation(
                decoded, task=task, image_size=(pil.height, pil.width)
            )
        except Exception:
            parsed = {}
        if task in DESCRIBE_TEXT_TASKS:
            text = str(parsed.get(task, "") or "").strip()
            if not text:
                body = decoded[4:] if decoded.startswith("</s>") else decoded
                text = body.split("</s>")[0].replace("<s>", "").strip()
            return text, []
        entry = parsed.get(task) or {}
        bboxes = entry.get("bboxes") or []
        labels = entry.get("bboxes_labels") or entry.get("labels") or []
        quad_boxes = entry.get("quad_boxes") or []
        quad_labels = entry.get("labels") or []
        regions: list[tuple[str, int, int, int, int, float | None, str | None]] = []
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            regions.append(
                (str(label), int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)), None, None)
            )
        for quad, label in zip(quad_boxes, quad_labels):
            xs = [float(v) for v in quad[0::2]]
            ys = [float(v) for v in quad[1::2]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            regions.append(
                (str(label), int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)), None, str(label))
            )
        return "", regions


class RealTagger:
    """WD14 Danbooru ONNX tagger family via onnxruntime, mapping output
    columns through the operator-supplied ordered tag-name list."""

    def __init__(self, execution_provider: str, names_path: Path, input_size: tuple[int, int] | None = None) -> None:
        self._provider = execution_provider
        self._names_path = Path(names_path)
        self._input_size = input_size
        self._session = None
        self._names: list[str] = []

    def load(self, path: Path) -> None:
        import onnxruntime

        if not Path(path).is_file():
            raise RuntimeError(f"tagger ONNX file missing: {path}")
        names = [
            ln.strip()
            for ln in self._names_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not names:
            raise RuntimeError(f"tag-name list is empty: {self._names_path}")
        providers = (
            ["CUDAExecutionProvider"]
            if self._provider == "CUDAExecutionProvider"
            else ["CPUExecutionProvider"]
        )
        session = onnxruntime.InferenceSession(str(path), providers=providers)
        if self._provider == "CUDAExecutionProvider" and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("tagger session did not activate CUDAExecutionProvider")
        out_shape = session.get_outputs()[0].shape
        width = out_shape[-1] if isinstance(out_shape[-1], int) else None
        if width is not None and width != len(names):
            raise RuntimeError(
                f"tag-name list length {len(names)} does not equal the tagger output width {width}"
            )
        self._session = session
        self._names = names
        try:
            h, w = session.get_inputs()[0].shape[1:3]
            if isinstance(h, int) and isinstance(w, int):
                self._input_size = (w, h)
        except Exception:
            pass

    def tag(self, rgb: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        from PIL import Image

        # Correct WD14 preprocessing (from model's reference implementation,
        # SmilingWolf/wd-tagger Predictor.prepare_image):
        #   1. alpha compositing — NOT done here: the caller's decode path
        #      (vision_embed/app.py _decode_rgb) always produces an HxWx3
        #      RGB uint8 array (Image.convert("RGB")), so no alpha channel
        #      can reach this layer. Alpha handling is upstream at decode.
        #   2. pad to square with WHITE fill, centered, aspect preserved
        #   3. resize to the model's input size with Image.BICUBIC
        #   4. np.asarray float32 of RAW 0-255 values — NO /255 scaling
        #   5. reverse channels RGB -> BGR
        #   6. expand_dims -> NHWC (this ONNX graph is NHWC
        #      [batch,448,448,3], verified live against the deployed graph).
        # No mean/std normalisation anywhere.
        w, h = self._input_size or (448, 448)
        pil = Image.fromarray(np.ascontiguousarray(rgb))
        side = max(pil.width, pil.height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(pil, ((side - pil.width) // 2, (side - pil.height) // 2))
        arr = np.asarray(canvas.resize((w, h), Image.BICUBIC), dtype=np.float32)
        blob = np.expand_dims(arr[:, :, ::-1], axis=0)
        scores = self._session.run(None, {self._session.get_inputs()[0].name: blob})[0][0]
        pairs = [
            (name, float(score))
            for name, score in zip(self._names, scores)
            if float(score) >= threshold
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs
