"""Acceptance tests for GET /v1/runtime (SPEC v12, section 6d) and unchanged
/readyz shape (v7/v9/v12: {status, models_loaded} only, no model names, no
gpu field)."""

import contextlib
import io
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

import describe_helpers as dh
from vision_embed.app import build_from_config_path
from vision_embed.errors import FatalStartupError

TOKEN = dh.TOKEN
CPU = dh.CPU
CUDA = dh.CUDA
ROLES = ["face", "face_detector", "scene", "object", "florence2", "tagger"]


@pytest.fixture()
def boot(tmp_path):
    def _boot(subdir=None, **kwargs):
        base = tmp_path / subdir if subdir else tmp_path
        base.mkdir(exist_ok=True)
        dh.write_token(base, TOKEN)
        cfg_path = dh.write_describe_config(base, **kwargs)
        captured = io.StringIO()
        app: Any = None
        with contextlib.redirect_stdout(captured):
            try:
                app, _cfg = build_from_config_path(cfg_path)
            except FatalStartupError as exc:
                pytest.fail(f"startup fatal (runtime route/config not implemented yet): {exc}")
            client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        return client
    return _boot


def test_runtime_requires_bearer(boot):
    client = boot(describe_enabled=False, tag_enabled=False)
    # client carries a valid default bearer; empty dict would be merged back in
    r = client.get("/v1/runtime", headers={"Authorization": ""})
    assert r.status_code == 401
    assert r.json()["error"]["reason"] == "auth.invalid_token"
    r = client.get("/v1/runtime", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    r = client.get("/v1/runtime", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_runtime_body_exact_top_level_keys(boot):
    """additionalProperties: false: exactly uptime_seconds, execution_provider,
    routes, models, gpu — no others."""
    client = boot(describe_enabled=False, tag_enabled=False)
    r = client.get("/v1/runtime")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"uptime_seconds", "execution_provider", "routes", "models", "gpu"}
    assert isinstance(body["uptime_seconds"], int)
    assert body["uptime_seconds"] >= 0
    assert body["execution_provider"] == CPU
    assert set(body["routes"]) == {"embed", "describe", "tag"}
    assert body["routes"]["embed"] is True
    assert body["routes"]["describe"] is False
    assert body["routes"]["tag"] is False
    # no filesystem path, weight filename, driver version or nvidia-smi output
    for bad in (".onnx", ".safetensors", "driver", "CUDA Version", "nvidia-smi", "bear"):
        assert bad not in str(body)


def test_runtime_uptime_is_monotonic_non_negative_integer(boot):
    client = boot(describe_enabled=False, tag_enabled=False)
    first = client.get("/v1/runtime").json()["uptime_seconds"]
    assert isinstance(first, int) and first >= 0
    client.get("/v1/runtime")
    second = client.get("/v1/runtime").json()["uptime_seconds"]
    assert second >= first


def test_runtime_six_roles_fixed_order_disabled_reasons(boot):
    """Default switches: six roles, fixed order, florence2/tagger
    loaded=false, model_id='', unavailable_reason='disabled'."""
    client = boot(describe_enabled=False, tag_enabled=False)
    body = client.get("/v1/runtime").json()
    models = body["models"]
    assert isinstance(models, list) and len(models) == 6
    assert [m["role"] for m in models] == ROLES
    for m in models:
        assert set(m) == {"role", "model_id", "loaded", "unavailable_reason"}
    for m in models[:4]:
        assert m["loaded"] is True
        assert m["unavailable_reason"] == ""
        assert m["model_id"] != ""
    for m in models[4:]:
        assert m["loaded"] is False
        assert m["model_id"] == ""
        assert m["unavailable_reason"] == "disabled"
    assert models[0]["model_id"] == "buffalo_l/w600k_r50"
    assert models[1]["model_id"] == "buffalo_l/det_10g"
    assert models[2]["model_id"] == "ViT-L-14/openai"
    assert models[3]["model_id"] == "yolov8n"


def test_runtime_enabled_switches_under_cpu_show_loaded(boot):
    """Under CPU the 2e gate never runs: switches true → both roles loaded,
    model_id florence2/wd14, unavailable_reason ''."""
    client = boot(
        "enabled", describe_enabled=True, tag_enabled=True,
        tag_files_present=True, florence_dir_present=True, exec_provider=CPU,
    )
    body = client.get("/v1/runtime").json()
    assert body["routes"]["describe"] is True
    assert body["routes"]["tag"] is True
    by_role = {m["role"]: m for m in body["models"]}
    for role, mid in (("florence2", "florence2"), ("tagger", "wd14")):
        assert by_role[role]["loaded"] is True
        assert by_role[role]["model_id"] == mid
        assert by_role[role]["unavailable_reason"] == ""
    r = client.get("/readyz")
    assert r.json() == {"status": "ready", "models_loaded": 6}


def test_gpu_object_shape_no_device(boot):
    """No NVML device here: available false, device_count 0, name '',
    both memory fields null — and the route still returns 200."""
    client = boot(describe_enabled=False, tag_enabled=False)
    body = client.get("/v1/runtime").json()
    gpu = body["gpu"]
    assert set(gpu) == {"available", "device_count", "name", "memory_total_bytes", "memory_used_bytes"}
    assert gpu["available"] is False
    assert gpu["device_count"] == 0
    assert gpu["name"] == ""
    assert gpu["memory_total_bytes"] is None
    assert gpu["memory_used_bytes"] is None


def test_readyz_shape_unchanged_no_model_names_no_gpu(boot):
    """readyz stays {status, models_loaded} with the count rule: default
    config → 4; nothing else on it, ever."""
    client = boot(describe_enabled=False, tag_enabled=False)
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "models_loaded"}
    assert body == {"status": "ready", "models_loaded": 4}


def test_readyz_count_role_rule_one_switch(boot):
    """models_loaded is the count of roles actually loaded: 4/5 never 6."""
    client = boot(
        "one", describe_enabled=True, tag_enabled=False,
        florence_dir_present=True, tag_files_present=False, exec_provider=CPU,
    )
    body = client.get("/readyz").json()
    assert body == {"status": "ready", "models_loaded": 5}
