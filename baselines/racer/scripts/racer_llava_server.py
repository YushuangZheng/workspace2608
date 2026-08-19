#!/usr/bin/env python3
"""Run RACER's unchanged LLaVA server with a bounded two-GPU memory map."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "upstream" / "Open-LLaVA-NeXT"
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-llava-root", type=Path, default=default_root)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--model-name", default="llava_llama3_lora")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--max-memory-gib", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.open_llava_root.resolve()
    deploy = root / "deploy" / "llava_server.py"
    if not deploy.is_file():
        raise SystemExit(f"Open-LLaVA deploy entry point not found: {deploy}")

    sys.path.insert(0, str(root))
    import torch
    import llava.model.builder as builder

    visible_count = torch.cuda.device_count()
    if visible_count != 2:
        raise SystemExit(
            f"Expected exactly two visible GPUs for RACER LLaVA, got {visible_count}."
        )

    official_loader = builder.load_pretrained_model
    max_memory = {
        index: f"{args.max_memory_gib}GiB" for index in range(visible_count)
    }

    def bounded_loader(*loader_args, **loader_kwargs):
        loader_kwargs.setdefault("device_map", "auto")
        loader_kwargs.setdefault("max_memory", max_memory)
        loader_kwargs.setdefault("use_flash_attn", False)
        return official_loader(*loader_args, **loader_kwargs)

    builder.load_pretrained_model = bounded_loader
    sys.argv = [
        str(deploy),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-path",
        args.model_path,
        "--model-base",
        args.model_base,
        "--model-name",
        args.model_name,
        "--device",
        "cuda",
        "--limit-model-concurrency",
        str(args.limit_model_concurrency),
    ]
    runpy.run_path(str(deploy), run_name="__main__")


if __name__ == "__main__":
    main()
