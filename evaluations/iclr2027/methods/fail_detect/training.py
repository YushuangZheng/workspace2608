"""Train one Main-10 FAIL-Detect velocity scorer from successful demos only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA
from evaluations.iclr2027.methods.fail_detect.demo_features import (
    DEFAULT_OUTPUT as DEFAULT_FEATURE_ROOT,
    FEATURE_DATA_SCHEMA,
)
from evaluations.iclr2027.methods.fail_detect.model import build_official_velocity_model
from evaluations.iclr2027.methods.fail_detect.preprocessing import ENCODER_SCHEMA, FeatureLayout
from integrations.rlbench.iclr2027.task_registry import experiment_task_set

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/configs/methods/m3_fail_detect_main10.json"
)
DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "evaluations/iclr2027/artifacts/checkpoints/m3"
DEFAULT_TRAINING_ROOT = REPOSITORY_ROOT / "evaluations/iclr2027/artifacts/training/m3/runs"
CHECKPOINT_SCHEMA = "essay2608.iclr2027.m3-logpzo.v1"
CHECKPOINT_MANIFEST_SCHEMA = "essay2608.iclr2027.monitor-checkpoint.v1"
OFFICIAL_COMMIT = "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _prepared_features(features: np.ndarray, input_dim: int) -> np.ndarray:
    rows, feature_dim = features.shape
    padded_dim = int(np.ceil(feature_dim / input_dim)) * input_dim
    time = padded_dim // input_dim
    padded_time = int(np.ceil(time / 4)) * 4
    result = np.zeros((rows, padded_time, input_dim), dtype=np.float32)
    result.reshape(rows, -1)[:, :feature_dim] = features
    return result


def _load_features(
    task: str,
    *,
    feature_root: Path,
    std_floor: float,
    clip: float,
) -> tuple[np.ndarray, dict[str, Any], FeatureLayout, dict[str, list[float]]]:
    index_path = feature_root / task / "index.json"
    index = _json(index_path)
    if (
        index.get("schema") != FEATURE_DATA_SCHEMA
        or index.get("task") != task
        or index.get("split") != "normal_demonstrations"
        or index.get("demonstration_count") != 5
        or any(
            index.get(field) is not False
            for field in (
                "contains_failure_train",
                "contains_normal_calibration",
                "contains_sealed_test",
                "contains_fault_or_audit_labels",
            )
        )
    ):
        raise ValueError("M3 feature index violates the success-demo-only contract")
    feature_path = REPOSITORY_ROOT / index["features_relative_path"]
    if _sha256(feature_path) != index["features_sha256"]:
        raise ValueError("M3 demonstration feature hash mismatch")
    features = np.load(feature_path, allow_pickle=False)["features"].astype(
        np.float32, copy=False
    )
    layout = FeatureLayout.from_dict(index["layout"])
    if (
        features.shape != (int(index["rows"]), layout.input_dim)
        or not np.isfinite(features).all()
    ):
        raise ValueError("M3 demonstration feature tensor/layout mismatch")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(std_floor))
    normalized = np.clip((features - mean) / std, -clip, clip).astype(np.float32)
    normalizer = {"mean": mean.tolist(), "std": std.tolist()}
    return normalized, index, layout, normalizer


def train_task(
    task: str,
    *,
    device: str,
    config_path: Path = DEFAULT_CONFIG,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    training_root: Path = DEFAULT_TRAINING_ROOT,
    epochs: int | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    config_path = Path(config_path).resolve()
    config = _json(config_path)
    if config.get("schema") != "essay2608.iclr2027.m3-method-config.v1":
        raise ValueError("unsupported M3 backend config")
    frozen_epochs = int(config["training"]["epochs"])
    requested_epochs = frozen_epochs if epochs is None else int(epochs)
    if requested_epochs < 1 or requested_epochs > frozen_epochs:
        raise ValueError("epochs must lie between one and the frozen schedule")
    formal = requested_epochs == frozen_epochs
    seed = int(config["training"]["seed"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    std_floor = float(config["normalizer"]["std_floor"])
    clip = float(config["normalizer"]["clip"])
    input_dim = int(config["unet_input_channels"])
    features, feature_index, layout, normalizer = _load_features(
        task,
        feature_root=Path(feature_root),
        std_floor=std_floor,
        clip=clip,
    )
    prepared = _prepared_features(features, input_dim)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.manual_seed_all(seed)
    model = build_official_velocity_model(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(prepared)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    metrics_path = Path(training_root) / task / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    losses = []
    started = time.monotonic()
    model.train()
    for epoch in range(requested_epochs):
        loss_sum = 0.0
        examples = 0
        epoch_started = time.monotonic()
        for (normal_observation,) in loader:
            x0 = normal_observation.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            x1 = torch.randn_like(x0)
            true_velocity = x1 - x0
            continuous_time = torch.rand(
                len(x0),
                *([1] * (x0.ndim - 1)),
                device=x0.device,
                dtype=x0.dtype,
            )
            current = x0 + continuous_time * true_velocity
            discrete_time = (continuous_time.reshape(-1) * 100).long()
            predicted_velocity = model(current, discrete_time)
            loss = (predicted_velocity - true_velocity).square().mean()
            loss.backward()
            optimizer.step()
            count = len(x0)
            loss_sum += float(loss.detach()) * count
            examples += count
        epoch_loss = loss_sum / examples
        losses.append(epoch_loss)
        metric = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "epoch_seconds": time.monotonic() - epoch_started,
            "elapsed_seconds": time.monotonic() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, sort_keys=True) + "\n")
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == requested_epochs:
            print(json.dumps({"task": task, **metric}, sort_keys=True), flush=True)

    checkpoint_dir = Path(checkpoint_root) / task / "seed_1103"
    checkpoint_path = checkpoint_dir / ("model.pt" if formal else "development_model.pt")
    metadata = {
        "task": task,
        "feature_schema": FEATURE_SCHEMA,
        "encoder_schema": ENCODER_SCHEMA,
        "layout": layout.to_dict(),
        "normalizer": normalizer,
        "fit_split": "normal_demonstrations",
        "demonstration_count": 5,
        "unet_input_channels": input_dim,
        "score_definition": config["score_definition"],
        "config_sha256": _sha256(config_path),
        "official_commit": OFFICIAL_COMMIT,
        "model": model.checkpoint_metadata(),
        "training": {
            "objective": "conditional_flow_matching_velocity_mse",
            "x0": "normalized_normal_demonstration_feature",
            "x1": "standard_normal_noise",
            "time": "uniform_0_1_discretized_by_floor_100t",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs": requested_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "precision": "float32",
        },
        "feature_index": {
            "path": str(
                (Path(feature_root) / task / "index.json").resolve().relative_to(
                    REPOSITORY_ROOT
                )
            ),
            "sha256": _sha256(Path(feature_root) / task / "index.json"),
            "features_sha256": feature_index["features_sha256"],
            "rows": feature_index["rows"],
        },
        "development_only": not formal,
        "reads_failure_train": False,
        "reads_normal_calibration": False,
        "reads_sealed_test": False,
    }
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    _atomic_torch_save(
        checkpoint_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_state_dict": cpu_state,
            "metadata": metadata,
        },
    )
    checkpoint_hash = _sha256(checkpoint_path)
    summary = {
        "schema": "essay2608.iclr2027.m3-training-run.v1",
        "task": task,
        "formal": formal,
        "device": str(device),
        "epochs": requested_epochs,
        "examples": int(len(prepared)),
        "prepared_shape": list(prepared.shape),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path.resolve().relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": _sha256(config_path),
        "metrics": str(metrics_path.resolve().relative_to(REPOSITORY_ROOT)),
        "metrics_sha256": _sha256(metrics_path),
    }
    _write_json(Path(training_root) / task / "summary.json", summary)
    if formal:
        manifest = {
            "schema": CHECKPOINT_MANIFEST_SCHEMA,
            "method_id": "M3",
            "checkpoint_relative_path": summary["checkpoint"],
            "checkpoint_sha256": checkpoint_hash,
            "config_relative_path": str(config_path.relative_to(REPOSITORY_ROOT)),
            "config_sha256": _sha256(config_path),
            "training_budget": 0,
            "training_seed": seed,
            "held_out_family": None,
            "feature_schema": FEATURE_SCHEMA,
        }
        _write_json(checkpoint_dir / "checkpoint_manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    main10 = {task.task_id for task in experiment_task_set("main10")}
    if args.task not in main10:
        raise ValueError(f"M3 training is restricted to Main-10: {args.task}")
    result = train_task(
        args.task,
        device=args.device,
        config_path=args.config,
        feature_root=args.feature_root,
        checkpoint_root=args.checkpoint_root,
        training_root=args.training_root,
        epochs=args.epochs,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_CHECKPOINT_ROOT", "train_task"]
