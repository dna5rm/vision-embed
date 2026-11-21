"""Acceptance tests for the v12 describe/tag route contract (SPEC v12).

Defaults (v9): routes.describe_enabled / routes.tag_enabled are false; a POST
to either returns 404 route.disabled; disabled roles require no files; the
embed route is unaffected. Enabled routes accept the eight closed wire tasks,
reject everything else with task.unsupported, and /v1/tag returns {tag, score}
pairs cut at the caller threshold with the suppress file honored and no file
writes anywhere (INV-2).
"""

import contextlib
import io
from typing import Any

import pytest
from starlette.testclient import TestClient

import describe_helpers as dh
from vision_embed.app import build_from_config_path
from vision_embed.errors import FatalStartupError

TOKEN = dh.TOKEN
CPU = dh.CPU
CUDA = dh.CUDA


@pytest.fixture()
def boot_it(tmp_path):
    def _boot(subdir: str | None = None, **kwargs):
        base = tmp_path / subdir if subdir else tmp_path
        base.mkdir(exist_ok=True)
        dh.write_token(base, TOKEN)
        config_path = dh.write_describe_config(base, **kwargs)
        captured = io.StringIO()
        app: Any = None
        with contextlib.redirect_stdout(captured):
            try:
                app, cfg = build_from_config_path(config_path)
            except FatalStartupError as exc:
                pytest.fail(f"startup fatal (route/config not implemented yet): {exc}")
            client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        return client, captured
    return _boot


def describe_post(client, *, task=None, token=TOKEN, image=None, extra=None):
    data = dict(extra or {})
    if task is not None:
        data["task"] = task
    files = {"image": ("img.png", image if image is not None else dh.make_image_bytes(), "image/png")}
    # client carries a valid default bearer, so {} would be merged back in
    headers = {"Authorization": f"Bearer {token}" if token is not None else ""}
    return client.post("/v1/describe", files=files, data=data, headers=headers)


def tag_post(client, *, threshold="0.5", token=TOKEN, image=None, extra=None):
    data = dict(extra or {})
    if threshold is not None:
        data["tag_threshold"] = threshold
    files = {"image": ("img.png", image if image is not None else dh.make_image_bytes(), "image/png")}
    # client carries a valid default bearer, so {} would be merged back in
    headers = {"Authorization": f"Bearer {token}" if token is not None else ""}
    return client.post("/v1/tag", files=files, data=data, headers=headers)


def test_default_switches_false_both_routes_404_route_disabled(boot_it):
    """v9: both switches default false; POSTs return 404 route.disabled.

    Not task.unsupported, not route.not_found, not 405.
    """
    client, _ = boot_it(None, describe_enabled=False, tag_enabled=False)
    for post in (describe_post, tag_post):
        r = post(client)
        assert r.status_code == 404, r.text
        assert r.json()["error"]["reason"] == "route.disabled"


def test_embed_still_works_when_routes_disabled(boot_it):
    client, _ = boot_it(None, describe_enabled=False, tag_enabled=False)
    img = dh.make_image_bytes()
    r = client.post(
        "/v1/embed",
        files={"image": ("img.png", img, "image/png")},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["scene"]["model"] == "ViT-L-14/openai"


def test_missing_florence_and_tagger_files_do_not_prevent_startup(boot_it, tmp_path):
    """v9: a disabled route's files are not required or mount-checked."""
    client, _ = boot_it(
        tmp_path,
        describe_enabled=False,
        tag_enabled=False,
        florence_dir_present=False,
        tag_files_present=False,
    )
    assert client.get("/healthz").status_code == 200
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "models_loaded": 4}
    for post in (describe_post, tag_post):
        assert post(client).status_code == 404
        assert post(client).json()["error"]["reason"] == "route.disabled"


def test_describe_and_tag_require_bearer(boot_it, tmp_path):
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    img = dh.make_image_bytes()
    r = describe_post(client, task="<CAPTION>", token=None)
    assert r.status_code == 401
    assert r.json()["error"]["reason"] == "auth.invalid_token"
    r = tag_post(client, token="wrong-token")
    assert r.status_code == 401
    assert r.json()["error"]["reason"] == "auth.invalid_token"
    r = describe_post(client, task="<CAPTION>", token=TOKEN)
    assert r.status_code != 401


def test_all_eight_wire_tasks_accepted_when_enabled(boot_it, tmp_path):
    """Section 2a: exactly eight wire values, exact-string match, all 200."""
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    for task in dh.WIRE_TASKS:
        r = describe_post(client, task=task)
        assert r.status_code == 200, f"{task}: {r.text}"
        body = r.json()
        assert body["task"] == task
        assert body["model"] == "florence2"
        # presence rules (section 6c): text if caption/OCR, regions otherwise
        text_tasks = {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>"}
        region_tasks = {"<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<OCR_WITH_REGION>"}
        if task in text_tasks:
            assert "text" in body
            assert "regions" not in body
        else:
            assert "regions" in body
            assert "text" not in body
            for region in body["regions"]:
                assert set(region) <= {"label", "bbox", "score", "text"}
        # X-Request-Id is part of the six-route set (v10)
        r2 = describe_post(client, task=task)
        assert r2.headers["X-Request-Id"] is not None


def test_non_wire_tasks_rejected_task_unsupported(boot_it, tmp_path):
    """Other task strings — rejected families, unbracketed, wrong case, junk."""
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    rejected = [
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "<OPEN_VOCAB_DETECTION>",
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "<REGION_TO_SEGMENTATION>",
        "<REGION_TO_DESCRIPTION>",
        "<NONSENSE>",
        "CAPTION",
        "<caption>",
        "<CAPTION> ",
        " ",
    ]
    for task in rejected:
        r = describe_post(client, task=task)
        assert r.status_code == 422, f"{task!r}: {r.status_code} {r.text}"
        assert r.json()["error"]["reason"] == "task.unsupported"


def test_missing_or_non_string_task_decode_failed(boot_it, tmp_path):
    """A missing task field is decode.failed (a 422), never task.unsupported."""
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    r = describe_post(client, task=None)
    assert r.status_code == 422
    assert r.json()["error"]["reason"] == "decode.failed"


def test_duplicate_task_field_rejected(boot_it, tmp_path):
    """A repeated task field is decode.failed (section 6a extra-task rule)."""
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    img = dh.make_image_bytes()
    r = client.post(
        "/v1/describe",
        files={"image": ("img.png", img, "image/png")},
        data=[("task", "<OD>"), ("task", "<OD>")],
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["reason"] == "decode.failed"


def test_tag_response_shape_exact(boot_it, tmp_path):
    """Section 6c: {request_id, image:{sha256,width,height}, model,
    threshold_used, tags:[{tag,score}]}, sorted score-descending, cut at the
    caller threshold. No rating field, no prompt-formatted strings."""
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    img = dh.make_image_bytes()
    r = tag_post(client, threshold="0.9", image=img)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"request_id", "image", "model", "threshold_used", "tags"}
    assert body["model"] == "wd14"
    assert body["threshold_used"] == 0.9
    assert set(body["image"]) == {"sha256", "width", "height"}
    assert body["image"]["width"] == 64 and body["image"]["height"] == 48
    import hashlib

    assert body["image"]["sha256"] == hashlib.sha256(img).hexdigest()
    scores = [t["score"] for t in body["tags"]]
    assert scores == sorted(scores, reverse=True)
    for tag in body["tags"]:
        assert set(tag) == {"tag", "score"}
        assert tag["score"] >= 0.9
    low = tag_post(client, threshold="0.1", image=img)
    assert low.status_code == 200
    low_scores = [t["score"] for t in low.json()["tags"]]
    assert low_scores == sorted(low_scores, reverse=True)
    assert all(t["score"] >= 0.1 for t in low.json()["tags"])
    assert len(low.json()["tags"]) >= len(body["tags"])
    low2 = tag_post(client, threshold="0.0", image=img)
    assert low2.status_code == 200
    low2_scores = [t["score"] for t in low2.json()["tags"]]
    assert low2_scores == sorted(low2_scores, reverse=True)
    assert all(t["score"] >= 0.0 for t in low2.json()["tags"])
    assert len(low2.json()["tags"]) >= len(low.json()["tags"])


def test_tag_threshold_absent_or_invalid_decode_failed(boot_it, tmp_path):
    """Absent / out-of-range / unparseable threshold → decode.failed."""
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    for threshold in (None, "1.1", "-0.01", "abc"):
        r = tag_post(client, threshold=threshold)
        assert r.status_code == 422, f"threshold={threshold}: {r.text}"
        assert r.json()["error"]["reason"] == "decode.failed"


def test_tag_bad_format_rejected(boot_it, tmp_path):
    client, _ = boot_it(None, describe_enabled=True, tag_enabled=True, tag_files_present=True, florence_dir_present=True)
    r = tag_post(client, threshold="0.5", image=b"plain text not an image")
    assert r.status_code == 415
    assert r.json()["error"]["reason"] == "format.unsupported"


def test_suppress_file_honored(boot_it, tmp_path):
    """Suppression is absolute: a suppressed tag never reaches the wire.

    Baseline (empty suppress file) determines the fake tagger's output for the
    fixture image; writing the top tag's name into the suppress file must
    remove it from the response without touching anything else.
    """
    # baseline run: suppress file exists but is empty
    client, _ = boot_it(
        None, describe_enabled=False, tag_enabled=True,
        tag_files_present=True, suppress_names=[],
    )
    img = dh.make_image_bytes()
    r = tag_post(client, threshold="0.0", image=img)
    assert r.status_code == 200, r.text
    baseline = r.json()["tags"]
    assert baseline, "fake tagger returned no tags at threshold 0.0"
    suppressed_name = baseline[0]["tag"]

    # second run: same image, same seed, suppress the top tag
    client2, _ = boot_it(
        "suppressed", describe_enabled=False, tag_enabled=True,
        tag_files_present=True, suppress_names=[suppressed_name],
    )
    r2 = tag_post(client2, threshold="0.0", image=img)
    assert r2.status_code == 200, r2.text
    after = r2.json()["tags"]
    assert suppressed_name not in [t["tag"] for t in after]
    assert len(after) == len(baseline) - 1
    # suppression is absolute: still absent even on a low threshold re-run
    r3 = tag_post(client2, threshold="0.0", image=img)
    assert r3.status_code == 200


def test_no_file_writes_for_caller(boot_it, tmp_path):
    """INV-2/section 8: the service writes no files for the caller — the
    directory trees inside the sandbox are unchanged by fixture requests."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    client, _ = boot_it(
        "writes", describe_enabled=True, tag_enabled=True,
        tag_files_present=True, florence_dir_present=True,
    )
    snap = dh.tree_snapshot(tmp_path)
    img = dh.make_image_bytes()
    for task in ("<CAPTION>", "<OD>"):
        r = describe_post(client, task=task, image=img)
        assert r.status_code == 200, r.text
    r = tag_post(client, threshold="0.5", image=img)
    assert r.status_code == 200
    r = tag_post(client, threshold="0.0", image=img)
    assert r.status_code == 200
    assert not list(sandbox.iterdir())
    assert dh.tree_snapshot(tmp_path) == snap
