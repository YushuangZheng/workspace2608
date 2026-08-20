#!/usr/bin/env python3
"""Classify generated FAIL-Detect artifacts as missing, complete, resumable, or damaged."""

import argparse
import json
import math
import sys
from pathlib import Path


EXIT_MISSING = 10
EXIT_RESUMABLE = 11
EXIT_DAMAGED = 12


def emit(state, path, detail, exit_code):
    print(json.dumps({
        "schema": "dynamac-fail-detect-generated-artifact-state-v1",
        "state": state,
        "path": str(path),
        "detail": detail,
    }, sort_keys=True))
    raise SystemExit(exit_code)


def classify_detector_metadata(epoch, loss_count, optimizer_present):
    if epoch == 200 and loss_count == 199:
        return "complete"
    if isinstance(epoch, int) and 1 <= epoch < 200 and loss_count == epoch - 1 and optimizer_present:
        return "resumable"
    return "damaged"


def inspect_features(path, validator):
    if not path.exists():
        emit("missing", path, "feature file does not exist", EXIT_MISSING)
    if not path.is_file() or path.stat().st_size == 0:
        emit("damaged", path, "feature path is not a non-empty regular file", EXIT_DAMAGED)
    try:
        detail = validator.validate_features(path)
    except Exception as exc:
        emit("damaged", path, "{}: {}".format(type(exc).__name__, exc), EXIT_DAMAGED)
    emit("complete", path, detail, 0)


def inspect_detector(path, upstream):
    if not path.exists():
        emit("missing", path, "detector checkpoint does not exist", EXIT_MISSING)
    if not path.is_file() or path.stat().st_size == 0:
        emit("damaged", path, "detector path is not a non-empty regular file", EXIT_DAMAGED)
    try:
        import torch

        sys.path.insert(0, str(upstream / "UQ_baselines/logpZO"))
        import net_CFM as Net

        payload = torch.load(str(path), map_location="cpu")
        epoch = payload.get("epoch")
        losses = payload.get("losses")
        if not isinstance(epoch, int) or not isinstance(losses, list):
            raise RuntimeError("checkpoint epoch/losses schema is invalid")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in losses):
            raise RuntimeError("checkpoint losses contain NaN or Inf")
        network = Net.get_unet(20)
        result = network.load_state_dict(payload["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("strict model load reported incompatible keys")
        optimizer_valid = False
        if "optimizer" in payload:
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
            optimizer.load_state_dict(payload["optimizer"])
            optimizer_valid = True
        detail = {
            "reported_epoch": epoch,
            "completed_loss_entries": len(losses),
            "strict_model_load": True,
        }
        state = classify_detector_metadata(epoch, len(losses), optimizer_valid)
        if state == "complete":
            emit("complete", path, detail, 0)
        if state == "resumable":
            detail["resume"] = "official train.py checkpoint resume path"
            emit("resumable", path, detail, EXIT_RESUMABLE)
        raise RuntimeError(
            "checkpoint is neither released-complete (epoch 200, 199 losses) nor safely resumable"
        )
    except SystemExit:
        raise
    except Exception as exc:
        emit("damaged", path, "{}: {}".format(type(exc).__name__, exc), EXIT_DAMAGED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("features", "detector"), required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_logpzo_inputs as validator

    if args.kind == "features":
        inspect_features(args.path, validator)
    inspect_detector(args.path, repo_root / "baselines/fail_detect/upstream")


if __name__ == "__main__":
    main()
