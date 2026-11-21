"""Shared helpers for the v12 describe/tag/runtime acceptance tests.

Mirrors conftest_helpers.py's write_config style but adds the v12 additive
config surface (models.florence/tagger keys, routes: block, min-free keys).
"""

import io
import json
import subprocess
import sys
import types
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
PYTHON = str(REPO / ".venv" / "bin" / "python")
TOKEN = "test-token-abc123"

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"

WIRE_TASKS = [
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR>",
    "<OCR_WITH_REGION>",
]

FLORENCE_ROLES = ["face", "face_detector", "scene", "object", "florence2", "tagger"]


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (180, 30, 30))
    img.save(buf, format=fmt)
    return buf.getvalue()


def write_token(dir_path: Path, token: str = TOKEN) -> str:
    p = dir_path / "authz.bearer"
    p.write_text(token + "\n", encoding="utf-8")
    return str(p)


def write_describe_config(
    dir_path: Path,
    *,
    adapter: str = "fake",
    adapter_mode: str = "test",
    exec_provider: str = CPU,
    describe_enabled: bool = False,
    tag_enabled: bool = False,
    describe_min_free_bytes: int | None = None,
    tag_min_free_bytes: int | None = None,
    suppress_names: list[str] | None = None,
    tag_files_present: bool = False,
    florence_dir_present: bool = False,
    suppress_file_present: bool | None = None,
    port: int = 8443,
) -> Path:
    """Write a config carrying the v12 additive keys.

    Model path files are only created when the *_present flags say so, so the
    "disabled route does not require its files" rule is directly testable.
    """
    suppress_file_present = (suppress_names is not None) if suppress_file_present is None else suppress_file_present
    florence_dir = dir_path / "florence2"
    if florence_dir_present:
        florence_dir.mkdir(exist_ok=True)
        (florence_dir / "model.safetensors").write_bytes(b"w")
    if tag_files_present:
        (dir_path / "wd14.onnx").write_bytes(b"x")
        (dir_path / "tag_names.txt").write_text("\n".join(f"tag{i}" for i in range(10)) + "\n")
        if suppress_file_present:
            (dir_path / "tag_suppress.txt").write_text(
                ("\n".join(suppress_names) + "\n") if suppress_names else ""
            )
    lines = [
        "listen:",
        '  host: "127.0.0.1"',
        f"  port: {port}",
        "  tls:",
        '    enabled: "false"',
        f'    cert_path: "{dir_path / "cert.pem"}"',
        f'    key_path: "{dir_path / "key.pem"}"',
        "auth:",
        f'  bearer_token_path: "{dir_path / "authz.bearer"}"',
        "models:",
        f'  det_path: "{dir_path / "det_10g.onnx"}"',
        f'  face_embed_path: "{dir_path / "w600k_r50.onnx"}"',
        f'  scene_embed_path: "{dir_path / "openai-ViT-L-14-visual.onnx"}"',
        f'  object_det_path: "{dir_path / "yolov8n.onnx"}"',
        "  det_score_min: 0.5",
        "  object_score_min: 0.35",
        "  max_faces: 20",
        f'  execution_provider: "{exec_provider}"',
        f'  adapter: "{adapter}"',
        f'  florence_dir_path: "{florence_dir}"',
        f'  tagger_onnx_path: "{dir_path / "wd14.onnx"}"',
        f'  tag_names_path: "{dir_path / "tag_names.txt"}"',
        f'  tag_suppress_path: "{dir_path / "tag_suppress.txt"}"',
        "  max_new_tokens: 256",
        f'adapter_mode: "{adapter_mode}"',
        "limits:",
        "  max_body_bytes: 33554432",
        "  max_regions: 100",
        "metrics:",
        "  enabled: false",
        "  port: 19199",
        "audit:",
        "  enabled: true",
        "request_id:",
        '  header: "X-Request-Id"',
        "routes:",
        f"  describe_enabled: {str(describe_enabled).lower()}",
        f"  tag_enabled: {str(tag_enabled).lower()}",
    ]
    if describe_min_free_bytes is not None:
        lines.append(f"  describe_min_free_bytes: {describe_min_free_bytes}")
    if tag_min_free_bytes is not None:
        lines.append(f"  tag_min_free_bytes: {tag_min_free_bytes}")
    cfg = dir_path / "config.yaml"
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


def run_entry(config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "vision_embed", "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
    )


def fatal_payload(proc: subprocess.CompletedProcess) -> dict:
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"no fatal JSON line in stdout: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    return json.loads(lines[-1])


def make_pynvml_stub(free_bytes: int, total_bytes: int | None = None, *, fail: bool = False) -> types.ModuleType:
    """Sys.modules-injected pynvml stub exposing the device-0 read of SPEC 6d/2e."""
    mod = types.ModuleType("pynvml")
    total = total_bytes if total_bytes is not None else max(free_bytes * 4, free_bytes)
    mem = types.SimpleNamespace(total=int(total), used=int(total - free_bytes), free=int(free_bytes))

    class _exc(Exception):
        pass

    def _raise(cmd):
        err = mod.NVML_ERROR_UNINITIALIZED if not fail else None
        raise err or _exc(cmd)

    def nvmlInit():
        if fail:
            raise mod.NVMLError(mod.NVML_ERROR_UNINITIALIZED, "stub: NVML unavailable")

    def nvmlShutdown():
        return None

    def nvmlDeviceGetCount():
        if fail:
            raise mod.NVMLError(mod.NVML_ERROR_UNINITIALIZED, "stub: NVML unavailable")
        return 1

    def nvmlDeviceGetHandleByIndex(index):
        if fail:
            raise mod.NVMLError(mod.NVML_ERROR_UNINITIALIZED, "stub: NVML unavailable")
        return object()

    def nvmlDeviceGetName(handle):
        return b"Stub GPU"

    def nvmlDeviceGetMemoryInfo(handle):
        if fail:
            raise mod.NVMLError(mod.NVML_ERROR_UNINITIALIZED, "stub: NVML unavailable")
        return mem

    class NVMLError(Exception):
        def __init__(self, code=None, msg=""):
            super().__init__(msg)
            self.code = code

    for name, fn in [
        ("nvmlInit", nvmlInit),
        ("nvmlShutdown", nvmlShutdown),
        ("nvmlDeviceGetCount", nvmlDeviceGetCount),
        ("nvmlDeviceGetHandleByIndex", nvmlDeviceGetHandleByIndex),
        ("nvmlDeviceGetName", nvmlDeviceGetName),
        ("nvmlDeviceGetMemoryInfo", nvmlDeviceGetMemoryInfo),
    ]:
        setattr(mod, name, fn)
    mod.NVMLError = NVMLError
    mod.NVML_ERROR_UNINITIALIZED = 1
    mod.NVML_ERROR_NO_PERMISSION = 2
    mod.LIB_ERROR = NVMLError
    return mod


def tree_snapshot(root: Path) -> list[tuple[str, int]]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        try:
            st = p.stat()
            out.append((rel, st.st_size))
        except OSError:
            out.append((rel, -1))
    return out
