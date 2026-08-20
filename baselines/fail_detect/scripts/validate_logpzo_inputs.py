#!/usr/bin/env python3
"""Validate released feature and logpZO checkpoint schemas for Transport."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def validate_features(path):
    import torch

    payload = torch.load(str(path), map_location="cpu")
    if set(payload) != {"X", "Y"}:
        raise RuntimeError("feature payload must contain exactly X and Y")
    x, y = payload["X"], payload["Y"]
    if x.ndim != 2 or x.shape[1] != 548:
        raise RuntimeError("Transport global_cond must have shape (N, 548): {}".format(tuple(x.shape)))
    if y.ndim != 2 or y.shape[1] != 320:
        raise RuntimeError("Transport action trajectory must have shape (N, 320): {}".format(tuple(y.shape)))
    if x.shape[0] != y.shape[0] or x.shape[0] <= 0:
        raise RuntimeError("feature tensors must have the same positive row count")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError("feature payload contains NaN or Inf")
    return {"rows": int(x.shape[0]), "x_shape": list(x.shape), "y_shape": list(y.shape)}


def validate_detector(path, upstream):
    import torch

    sys.path.insert(0, str(upstream / "UQ_baselines/logpZO"))
    import net_CFM as Net

    payload = torch.load(str(path), map_location="cpu")
    if payload.get("epoch") != 200:
        raise RuntimeError("released logpZO checkpoint must report epoch 200")
    losses = payload.get("losses")
    # Upstream saves immediately before every epoch, so its final epoch=200
    # checkpoint intentionally contains the first 199 completed losses.
    if not isinstance(losses, list) or len(losses) != 199:
        raise RuntimeError("released pre-epoch save semantics require 199 stored losses")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in losses):
        raise RuntimeError("logpZO losses contain NaN or Inf")
    network = Net.get_unet(20)
    result = network.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("logpZO strict load reported incompatible keys")
    return {
        "reported_epoch": 200,
        "completed_loss_entries": 199,
        "strict_model_load": True,
        "parameters": int(sum(parameter.numel() for parameter in network.parameters())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    result = {
        "schema": "dynamac-fail-detect-logpzo-input-validation-v1",
        "features": validate_features(args.features),
        "feature_sha256": sha256_file(args.features),
    }
    if args.checkpoint:
        result["detector"] = validate_detector(
            args.checkpoint, repo_root / "baselines/fail_detect/upstream"
        )
        result["detector_sha256"] = sha256_file(args.checkpoint)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
