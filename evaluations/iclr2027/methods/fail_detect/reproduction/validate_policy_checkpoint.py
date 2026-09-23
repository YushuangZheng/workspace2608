"""Validate that a distributed policy checkpoint follows the upstream resume path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import dill
import torch

EXPECTED_COMMIT = "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/_external/FAIL-Detect"),
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-epoch", type=int, default=50)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repository: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _tensor_summary(module: Any) -> dict[str, Any]:
    state = module.state_dict()
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in tensors
        if value.is_floating_point() or value.is_complex()
    )
    return {
        "state_entries": len(state),
        "tensor_entries": len(tensors),
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "all_floating_tensors_finite": finite,
    }


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    official_root = args.official_root.resolve()
    if args.minimum_epoch <= 0 or args.poll_seconds <= 0:
        raise ValueError("minimum-epoch and poll-seconds must be positive")

    while not checkpoint.is_file():
        time.sleep(args.poll_seconds)

    commit = _git_head(official_root)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"FAIL-Detect commit mismatch: {commit}")
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
        TrainDiffusionUnetHybridWorkspace,
    )

    started = time.monotonic()
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    required = {"cfg", "state_dicts", "pickles"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing {sorted(missing)}")
    workspace = TrainDiffusionUnetHybridWorkspace(
        payload["cfg"], output_dir=str(checkpoint.parent.parent)
    )
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    epoch = int(workspace.epoch)
    global_step = int(workspace.global_step)
    if epoch < args.minimum_epoch:
        raise ValueError(
            f"checkpoint epoch {epoch} is below required {args.minimum_epoch}"
        )
    if global_step <= 0:
        raise ValueError("checkpoint global_step must be positive")
    model = _tensor_summary(workspace.model)
    ema_model = _tensor_summary(workspace.ema_model)
    if not model["all_floating_tensors_finite"]:
        raise ValueError("model checkpoint contains non-finite tensors")
    if not ema_model["all_floating_tensors_finite"]:
        raise ValueError("EMA checkpoint contains non-finite tensors")

    result = {
        "status": "pass",
        "scope": "upstream_workspace_cpu_checkpoint_resume_validation",
        "official_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint),
        "epoch": epoch,
        "global_step": global_step,
        "state_dict_keys": sorted(payload["state_dicts"]),
        "pickle_keys": sorted(payload["pickles"]),
        "model": model,
        "ema_model": ema_model,
        "optimizer_state_entries": len(workspace.optimizer.state),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
