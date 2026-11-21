"""Entrypoint — fatal startup line + uvicorn with workers=1 (SPEC sections 3, 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import run_server
from .config import fatal_line
from .errors import FatalStartupError


def main() -> int:
    parser = argparse.ArgumentParser(prog="vision-embed")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    try:
        run_server(Path(args.config))
    except FatalStartupError as err:
        print(fatal_line(err.reason, err.detail), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
