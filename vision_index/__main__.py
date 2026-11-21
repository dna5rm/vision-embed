"""Entrypoint — fatal startup line + uvicorn --workers 1 (SPEC sections 3, 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import run_server
from .config import load_config
from .db import IndexStore
from .errors import FatalStartupError, fatal_line


def main() -> int:
    parser = argparse.ArgumentParser(prog="vision-index")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    try:
        # Fail fast on config (exit 1, one JSON fatal line) before uvicorn binds.
        load_config(Path(args.config))
        run_server(Path(args.config))
    except FatalStartupError as err:
        print(fatal_line(err.reason, err.detail), flush=True)
        return 1
    except Exception as err:  # e.g. db open failure at startup (INV-4)
        print(fatal_line("config.invalid", f"startup failed: {err.__class__.__name__}"), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
