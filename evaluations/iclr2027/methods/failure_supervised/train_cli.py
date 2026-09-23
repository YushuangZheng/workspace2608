"""Manifest-bound M4 training. Run as a module; never loads calibration/test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from evaluations.iclr2027.interfaces.failure_train import (
    load_failure_train_manifest,
    load_failure_train_sequence,
    select_failure_train_rows,
)
from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.fail_detect.preprocessing import FeatureLayout

from .model import CausalGRUClassifier
from .runtime import CanonicalFailureSupervisedMonitor, sha256
from .training import save_training_checkpoint, train_step

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "evaluations/iclr2027"
CONFIG = BASE / "configs/methods/m4_failure_supervised.json"
MANIFEST = BASE / "manifests/main10_failure_train.jsonl"
DATASET = BASE / "datasets/failure_train"
TRAINING = BASE / "artifacts/training/m4"
CHECKPOINTS = BASE / "artifacts/checkpoints/m4"
GOLDEN = BASE / "artifacts/development_golden/m4"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def config() -> dict:
    value = json.loads(CONFIG.read_text())
    if value["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("config feature schema differs from frozen interface")
    return value


def code_identity() -> dict[str, str]:
    paths = [
        *Path(__file__).parent.glob("*.py"),
        BASE / "methods/fail_detect/preprocessing.py",
        BASE / "interfaces/feature_schema.py",
        BASE / "interfaces/failure_train.py",
        BASE / "interfaces/runtime_monitor.py",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def cache_dir(task: str) -> Path:
    if task not in config()["main10"]:
        raise ValueError("unknown task")
    return TRAINING / "encoded" / task


def prepare_task(task: str) -> dict:
    destination = cache_dir(task)
    meta_path = destination / "cache.json"
    encoder_hash = sha256(BASE / "methods/fail_detect/preprocessing.py")
    identities = {
        "manifest_sha256": sha256(MANIFEST),
        "encoder_sha256": encoder_hash,
        "reader_sha256": sha256(BASE / "interfaces/failure_train.py"),
        "feature_schema_sha256": sha256(BASE / "interfaces/feature_schema.py"),
        "train_handoff_sha256": sha256(BASE / "results/a2_acceptance/B_FAILURE_TRAIN_HANDOFF.json"),
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if any(meta[k] != v for k, v in identities.items()) or meta["npz_sha256"] != sha256(
            destination / "raw.npz"
        ):
            raise ValueError("frozen encoded cache identity mismatch")
        return meta
    rows = select_failure_train_rows(load_failure_train_manifest(MANIFEST), task=task, budget=200)
    features, labels, offsets, ids = [], [], [0], []
    layout = None
    for row in rows:
        seq = load_failure_train_sequence(DATASET, row, verify_hash=True)
        if not seq.features:
            raise ValueError("empty training episode")
        if layout is None:
            layout = FeatureLayout.from_record(seq.features[0])
        x = np.stack([layout.encode_validated(f) for f in seq.features])
        y = np.asarray(seq.labels, dtype=np.float32)
        if len(x) != len(y) or not np.isin(y, (0, 1)).all():
            raise ValueError("bad labels or label alignment")
        features.append(x)
        labels.append(y)
        offsets.append(offsets[-1] + len(x))
        ids.append(seq.episode_id)
    x, y = np.concatenate(features), np.concatenate(labels)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "raw.tmp.npz"
    np.savez_compressed(
        temporary,
        x=x,
        y=y,
        offsets=np.asarray(offsets, dtype=np.int64),
        episode_ids=np.asarray(ids),
    )
    temporary.replace(destination / "raw.npz")
    meta = {
        "schema": "essay2608.iclr2027.m4-encoded-train.v1",
        "task": task,
        **identities,
        "npz_sha256": sha256(destination / "raw.npz"),
        "layout": layout.to_dict(),
        "episodes": len(ids),
        "cycles": len(y),
        "positive_cycles": int(y.sum()),
        "input_dim": x.shape[1],
        "label_alignment": config()["label_alignment"],
        "source": "A_canonical_reader_with_cycle_sha256_verification",
        "normalizer_fit": False,
        "formal_evaluation": False,
    }
    write_json(meta_path, meta)
    print(
        json.dumps(
            {
                "prepared": task,
                "episodes": len(ids),
                "cycles": len(y),
                "positive_cycles": int(y.sum()),
                "input_dim": x.shape[1],
            }
        ),
        flush=True,
    )
    return meta


def run_id(task: str, seed: int, budget: int | None, family: str | None) -> str:
    if task not in config()["main10"] or seed not in config()["training_seeds"]:
        raise ValueError("task/seed must be frozen in method config")
    if family not in (
        None,
        "actuation_delay",
        "missed_interaction",
        "relation_loss",
        "environment_change",
        "coordination_delay",
    ):
        raise ValueError("unsupported family")
    view = "lofo_" + family if family else "budget_" + str(budget or 200)
    return f"{task}/{view}/seed_{seed}"


def train(
    task: str,
    seed: int,
    budget: int | None,
    family: str | None,
    *,
    device: str = "cuda:0",
    smoke_epochs: int | None = None,
) -> dict:
    cfg = config()
    identity = run_id(task, seed, budget, family)
    run = TRAINING / ("smoke" if smoke_epochs else "runs") / identity
    checkpoint_dir = run if smoke_epochs else CHECKPOINTS / identity
    done_path = run / "complete.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if (
            done["config_sha256"] != sha256(CONFIG)
            or done["checkpoint_sha256"] != sha256(ROOT / done["checkpoint_relative_path"])
            or done["code_sha256"] != code_identity()
        ):
            raise ValueError("completed run changed; refusing to overwrite")
        return done
    if run.exists() and any(run.iterdir()):
        raise ValueError(f"incomplete run exists; inspect before retry: {run}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise ValueError("checkpoint directory exists without verified completion")
    meta = prepare_task(task)
    selected = select_failure_train_rows(
        load_failure_train_manifest(MANIFEST), task=task, budget=budget, held_out_family=family
    )
    with np.load(cache_dir(task) / "raw.npz", allow_pickle=False) as saved:
        ids = saved["episode_ids"].tolist()
        offsets = saved["offsets"]
        x, y = saved["x"], saved["y"]
        selection = [ids.index(row["episode_id"]) for row in selected]
        xs = [x[offsets[i] : offsets[i + 1]].copy() for i in selection]
        ys = [y[offsets[i] : offsets[i + 1]].copy() for i in selection]
    all_x, all_y = np.concatenate(xs), np.concatenate(ys)
    # Fit ONLY selected episodes. No small-budget / LOFO leakage via normalization.
    mean = all_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(all_x.std(axis=0, dtype=np.float64), cfg["normalizer"]["std_floor"]).astype(
        np.float32
    )
    positives = int(all_y.sum())
    if positives == 0 or positives == len(all_y):
        raise ValueError("selected training data must contain both causal target classes")
    pos_weight = min((len(all_y) - positives) / positives, cfg["positive_weight_cap"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = CausalGRUClassifier(meta["input_dim"], **cfg["architecture"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    tensors = [
        torch.as_tensor(
            np.clip((v - mean) / std, -cfg["normalizer"]["clip"], cfg["normalizer"]["clip"]),
            device=device,
        )
        for v in xs
    ]
    targets = [torch.as_tensor(v, device=device) for v in ys]
    selection_ids = [row["episode_id"] for row in selected]
    selection_hash = hashlib.sha256(("\n".join(selection_ids) + "\n").encode()).hexdigest()
    epochs = cfg["epochs"] if smoke_epochs is None else smoke_epochs
    metadata = {
        "method_id": "M4",
        "task": task,
        "training_budget": 200 if family else budget or 200,
        "actual_training_episodes": len(selected),
        "training_seed": seed,
        "held_out_family": family,
        "feature_schema": FEATURE_SCHEMA,
        "layout": meta["layout"],
        "normalizer": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit_episode_ids_sha256": selection_hash,
        },
        "config_relative_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "code_sha256": code_identity(),
        "source_manifest_sha256": sha256(MANIFEST),
        "source_train_handoff_sha256": meta["train_handoff_sha256"],
        "encoded_cache_sha256": meta["npz_sha256"],
        "selection_episode_ids_sha256": selection_hash,
        "positive_weight": pos_weight,
        "training_cycles": len(all_y),
        "positive_cycles": positives,
        "label_alignment": cfg["label_alignment"],
        "epochs": epochs,
        "checkpoint_selection": cfg["checkpoint_selection"],
        "development_smoke_only": smoke_epochs is not None,
    }
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "training_identity.json", {**metadata, "selected_episode_ids": selection_ids})
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "deterministic_algorithms": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    write_json(run / "environment.json", environment)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    step = 0
    with (run / "epochs.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(tensors))
            total_loss = total_cycles = 0
            max_grad = 0.0
            for offset in range(0, len(order), cfg["batch_size"]):
                batch = order[offset : offset + cfg["batch_size"]]
                lengths = [len(tensors[i]) for i in batch]
                bx = torch.nn.utils.rnn.pad_sequence([tensors[i] for i in batch], batch_first=True)
                by = torch.nn.utils.rnn.pad_sequence([targets[i] for i in batch], batch_first=True)
                mask = (
                    torch.arange(bx.shape[1], device=device)[None, :]
                    < torch.tensor(lengths, device=device)[:, None]
                )
                metrics = train_step(
                    model,
                    optimizer,
                    bx,
                    by,
                    mask,
                    positive_weight=pos_weight,
                    max_gradient_norm=cfg["max_gradient_norm"],
                )
                total_loss += metrics.loss * metrics.valid_cycles
                total_cycles += metrics.valid_cycles
                max_grad = max(max_grad, metrics.gradient_norm)
                step += 1
            report = {
                "epoch": epoch,
                "weighted_training_bce": total_loss / total_cycles,
                "valid_cycles": total_cycles,
                "max_unclipped_gradient_norm": max_grad,
                "elapsed_seconds": time.monotonic() - start,
            }
            log.write(json.dumps(report, allow_nan=False) + "\n")
            log.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(json.dumps({"run": identity, **report}), flush=True)
    model.eval()
    with torch.no_grad():
        probe = tensors[0][: min(40, len(tensors[0]))][None]
        expected, _ = model(probe)
        expected = expected.detach().cpu()
    model.cpu()
    with torch.no_grad():
        cpu_reference, _ = model(probe.cpu())
    checkpoint = checkpoint_dir / "model.pt"
    save_training_checkpoint(checkpoint, model, training_step=step, metadata=metadata)
    monitor = CanonicalFailureSupervisedMonitor(checkpoint, CONFIG, device="cpu")
    with torch.no_grad():
        actual, _ = monitor.scorer.model(probe.cpu())
    delta = float((actual - expected).abs().max())
    # Serialization is checked on the SAME backend, independently of cuDNN/CPU
    # roundoff. Formal inference is frozen to CPU for calibration and evaluation.
    if not torch.equal(actual, cpu_reference):
        raise ValueError("CPU serialization round trip changed predictions")
    for key, value in model.state_dict().items():
        if not torch.equal(value, monitor.scorer.model.state_dict()[key]):
            raise ValueError(f"checkpoint changed parameter: {key}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite inference after training")
    probability_delta = float((actual.sigmoid() - expected.sigmoid()).abs().max())
    ckmanifest = {
        "schema": "essay2608.iclr2027.monitor-checkpoint.v1",
        "method_id": "M4",
        "checkpoint_relative_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "config_relative_path": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "training_budget": metadata["training_budget"],
        "training_seed": seed,
        "held_out_family": family,
        "feature_schema": FEATURE_SCHEMA,
        "task": task,
        "actual_training_episodes": len(selected),
        "selection_episode_ids_sha256": selection_hash,
    }
    write_json(checkpoint_dir / "checkpoint_manifest.json", ckmanifest)
    done = {
        **ckmanifest,
        "status": "pass",
        "code_sha256": metadata["code_sha256"],
        "epochs": epochs,
        "training_steps": step,
        "elapsed_seconds": time.monotonic() - start,
        "cpu_serialization_max_abs_logit_error": float((actual - cpu_reference).abs().max()),
        "gpu_cpu_max_abs_logit_difference_diagnostic_only": delta,
        "gpu_cpu_max_abs_probability_difference_diagnostic_only": probability_delta,
        "formal_inference_backend": "cpu_float32",
        "formal_evaluation": False,
        "formal_calibration": "pending_A_only",
        "development_smoke_only": smoke_epochs is not None,
        "final_weighted_training_bce": report["weighted_training_bce"],
    }
    write_json(done_path, done)
    return done


def golden(task: str, seed: int, budget: int | None, family: str | None) -> dict:
    identity = run_id(task, seed, budget, family)
    checkpoint = CHECKPOINTS / identity / "model.pt"
    monitor = CanonicalFailureSupervisedMonitor(checkpoint, CONFIG)
    fixture = BASE / "tests/fixtures/development_examples/causal_records.jsonl"
    records = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    outputs = []
    previous_id = None
    previous_cycle = None
    for record in records:
        if record["episode_id"].split("/")[1] != task:
            continue
        reset = record["episode_id"] != previous_id or record["cycle"] != previous_cycle + 1
        if reset:
            monitor.reset(
                EpisodeContext(
                    record["episode_id"],
                    task,
                    "M4",
                    len(monitor.layout.arms) == 2,
                    1000,
                    FEATURE_SCHEMA,
                    monitor.config_hash,
                    monitor.checkpoint_hash,
                )
            )
        monitor.observe_record(record)
        outputs.append(
            {"episode_id": record["episode_id"], "reset_before": reset, **monitor.cycle_output()}
        )
        previous_id, previous_cycle = record["episode_id"], record["cycle"]
    report = {
        "schema": "essay2608.iclr2027.development-golden.v1",
        "method_id": "M4",
        "task": task,
        "training_seed": seed,
        "checkpoint_sha256": monitor.checkpoint_hash,
        "config_sha256": monitor.config_hash,
        "fixture_sha256": sha256(fixture),
        "scope": "sparse_development_segments_with_explicit_resets_not_full_rollouts",
        "complete_development_rollout_acceptance": False,
        "status": "pass" if outputs else "no_development_example_for_this_task",
        "outputs": outputs,
        "comparison_atol": 2e-5,
        "comparison_rtol": 2e-5,
    }
    write_json(GOLDEN / identity / "golden.json", report)
    return {"task": task, "seed": seed, "golden_records": len(outputs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "train", "golden"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=1103)
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--budget", type=int, choices=[20, 50, 100, 200])
    views.add_argument("--held-out-family")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-epochs", type=int)
    args = parser.parse_args()
    if args.smoke_epochs is not None and args.smoke_epochs <= 0:
        parser.error("smoke epochs must be positive")
    if args.operation == "prepare":
        result = prepare_task(args.task)
    elif args.operation == "golden":
        result = golden(args.task, args.seed, args.budget, args.held_out_family)
    else:
        result = train(
            args.task,
            args.seed,
            args.budget,
            args.held_out_family,
            device=args.device,
            smoke_epochs=args.smoke_epochs,
        )
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
