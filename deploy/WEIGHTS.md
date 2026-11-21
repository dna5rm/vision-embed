# WEIGHTS.md — obtaining the four ONNX files

Read this before the embed deployment, not after it hits startup failure.

**First: only the production embed service needs model weights.** The test
suite runs entirely on a deterministic fake adapter and needs no ONNX files at
all, and **vision-index never runs a model** — it calls embed over HTTPS. Only
the embed service started with `adapter_mode: "production"` requires the four
model files to exist at the configured paths; without them startup fails. If
you are only running tests or only the index, you can skip this document.

Filenames are fixed by the deploy config and spec. Do not substitute a
"latest" variant, a renamed file, or a different pack — verify each file name
against the config paths in `deploy/config.example.yaml`:

- `det_10g.onnx`
- `w600k_r50.onnx`
- `yolov8n.onnx`
- `openai-ViT-L-14-visual.onnx`

## 1. `det_10g.onnx` and `w600k_r50.onnx` — InsightFace buffalo_l

Both face models ship inside the **`buffalo_l`** pack (SCRFD-10GF detector +
ResNet50@WebFace600K recognition) distributed by the InsightFace model zoo.

Verified source (asset URL live when checked; release page lists the same
asset with its sha256):

- Release page: <https://github.com/deepinsight/insightface/releases/tag/model-zoo>
  — asset `buffalo_l.zip`
- Direct asset: <https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip>

Scope note: the pack contains five ONNX files. **Only `det_10g.onnx` and
`w600k_r50.onnx` are used.** The `1k3d68`, `2d106det`, and `genderage` files
in the same zip are unused by this service and must not be mounted. Unpack
the zip into the operator's `models/` directory and keep the two needed
filenames unchanged inside `models/`.

The model Zoo page and the pack README both state the models are provided for
**non-commercial research purposes only**; the Python-package model docs use
the same wording for library-provided pretrained models:

- <https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md>
- <https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md>

License note: the operator must comply with the InsightFace model license
terms (non-commercial research restrictions on the pack) before using it.

## 2. `yolov8n.onnx` — Ultralytics YOLOv8n export

Spec (docs/spec/vision-embed-SPEC.md): COCO-80 label set, single static-shape
detect model. The matching export shape is `output0` of `(1, 84, 8400)` at
input 640 (4 box coordinates + 80 class scores × 8400 anchors, raw output,
NMS applied outside the model, which is what this service's NMS unit does).

The file is **not** shipped in this repo. The operator exports it from the
official `yolov8n.pt` checkpoint with the Ultralytics package.

Verified against the current Ultralytics ONNX export documentation
(<https://docs.ultralytics.com/integrations/onnx/>) — `format: onnx` export,
default `nms=None` raw-output layout with `output0` `(1, 84, 8400)` for an
80-class detection model at `imgsz=640`, `simplify` on by default:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")            # official COCO 80-class checkpoint
model.export(format="onnx", imgsz=640)  # -> yolov8n.onnx
```

After exporting, verify the ONNX graph before deployment: input `images`
shape `(1, 3, 640, 640)`, output `output0` shape `(1, 84, 8400)`, and no
embedded NMS output (do not pass `nms=True` — that changes the output shape
and the raw tensor layout this service expects).

License note: the operator must comply with Ultralytics license terms
(AGPL-3.0 or Ultralytics Enterprise License; see
<https://github.com/ultralytics/ultralytics/blob/main/LICENSE>).

## 3. `openai-ViT-L-14-visual.onnx` — CLIP ViT-L/14 visual tower

**No citable public canonical ONNX exists for this file.** The official OpenAI
CLIP checkout distributes a single PyTorch JIT checkpoint (`ViT-L-14.pt` from
<https://github.com/openai/CLIP>), not an ONNX export; the open_clip project
distributes PyTorch checkpoints and no documented ONNX export step. No
third-party ONNX upload is treated as canonical, and none is cited here.

The operator therefore exports the file themselves and must make it match
these requirements (from the spec):

- model `ViT-L-14` pretrained on **OpenAI WIT**, **visual tower only**
  (the OpenAI-weights model definition needs the QuickGELU/`-quickgelu` model
  variant or `force_quick_gelu=True`; open_clip's own README states this)
- 224×224 input, the standard CLIP bicubic-resize/center-crop/normalize
  preprocessing in the calling service (this repo does not preprocess inside
  the ONNX file)
- output: the pooler-projected visual embedding, 768-d — this is the value
  returned as the scene `model` string `ViT-L-14/openai`

The OpenAI checkpoint and the open_clip pretrained-key (`ViT-L-14`,
`pretrained='openai'`) are both verified against their current repos:

- <https://github.com/openai/CLIP>
- <https://github.com/mlfoundations/open_clip>
  (`open_clip.list_pretrained()` includes `('ViT-L-14', 'openai')` and
  `('ViT-L-14-quickgelu', 'openai')`; see its pretrained-model docs)

Export method: not specified here. open_clip's own documentation does not
provide a supported ONNX export command, so no command is documented in this
file. Implement an export with the PyTorch checkpoints above and satisfy the
requirements list, or treat the exact export recipe as an implementation task
open item tracked in the spec.

License note: the operator must comply with the OpenAI CLIP model license
terms (see the CLIP README) when using these weights.

## 4. Summary and placement

| File | Source of weights | Shipped here? |
| --- | --- | --- |
| `det_10g.onnx` | InsightFace `buffalo_l` zip (model-zoo release) | No — operator-provided |
| `w600k_r50.onnx` | same as above, from same zip | No — operator-supplied |
| `yolov8n.onnx` | official `yolov8n.pt` → export per Ultralytics docs | No — operator-supplied |
| `openai-ViT-L-14-visual.onnx` | operator's export of OpenAI CLIP ViT-L/14 visual tower; no citeable canonical ONNX | No — operator-supplied |

All four files land in the operator's `models/` directory and are bind-mounted
read-only; they are never baked into the image and never downloaded at
runtime (no fetch step in the image or entrypoint). Directory layout and
mount rules are in `deploy/DEPLOY.md` step 2 and the spec.
