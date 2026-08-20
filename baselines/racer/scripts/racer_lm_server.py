#!/usr/bin/env python3
"""Bind RACER's unchanged language-service app to a configurable TCP port."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path


def tcp_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "upstream" / "Open-LLaVA-NeXT"
    parser = argparse.ArgumentParser(
        description="Run the official RACER language-service app on a chosen port."
    )
    parser.add_argument("--open-llava-root", type=Path, default=default_root)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=tcp_port, default=18000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.open_llava_root.expanduser().resolve()
    official_entry = root / "deploy" / "lm_server.py"
    if not official_entry.is_file():
        raise SystemExit(
            f"Official language-service entry point not found: {official_entry}"
        )

    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("deploy.lm_server")
    spec_origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if spec_origin != official_entry.resolve():
        raise SystemExit(
            "Refusing unexpected deploy.lm_server resolution: "
            f"expected {official_entry}, got {spec_origin}"
        )
    os.chdir(root)
    official_module = importlib.import_module("deploy.lm_server")
    imported_entry = Path(official_module.__file__).resolve()
    if imported_entry != official_entry.resolve():
        raise SystemExit(
            "Refusing unexpected deploy.lm_server import: "
            f"expected {official_entry}, got {imported_entry}"
        )

    from uvicorn import run

    run(official_module.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
