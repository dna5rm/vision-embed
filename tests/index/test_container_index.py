"""Container contract test (SPEC section 13, test 20) — real docker only.

Skipped (never faked as a pass) unless the vision-index image is built: like
the vision-embed container tests, they run against `vision-index:test`.
Not built by default because deployment is out of scope for this change
(`Do not deploy yet`), but the check is real when the image exists.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.index import helpers_index as H

IMAGE = "vision-index:test"


def _docker_available() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available() or
    subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0,
    reason=f"docker or {IMAGE} not available; build deploy/Dockerfile.index first",
)


def test_read_only_rootfs_and_uid(tmp_path):
    subprocess.run(
        [
            "docker", "run", "--rm", "--read-only",
            "--tmpfs", "/tmp:rw,size=134217728",
            "--user", "65532:65532",
            IMAGE, "--", "id",
        ],
        check=True, capture_output=True, timeout=60,
    )
