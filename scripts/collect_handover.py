"""Canonical entry point for isolated bimanual handover collection."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("collect_handover_demos.py")),
        run_name="__main__",
    )
