"""Build and freeze the missing Horizon-3 assets for E4 FAIL-Detect.

The Main-10 M3 artifact covers only ``place_cups_3`` from Horizon-3.  E4 uses
eight task levels, so every level needs its own success-demo scorer and a
threshold fitted on an independent A-only normal-calibration split.  No
failure data, development fault label, or sealed E4 result is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.calibration.m3 import calibrate_m3
from evaluations.iclr2027.manifests.materialize_calibration import materialize
from evaluations.iclr2027.methods.fail_detect.demo_features import export_task
from integrations.rlbench.iclr2027.task_registry import experiment_task_set


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
E4_ROOT = EVAL_ROOT / "results" / "controlled" / "e4"
ASSET_ROOT = E4_ROOT / "m3_horizon_assets"
CANDIDATE_MANIFEST = ASSET_ROOT / "horizon3_normal_calibration_candidates.jsonl"
CALIBRATION_MANIFEST = ASSET_ROOT / "horizon3_normal_calibration.jsonl"
CANDIDATE_RESULTS = (
    EVAL_ROOT / "datasets" / "horizon3_m3_normal_calibration_candidates"
)
CALIBRATION_ROOT = (
    EVAL_ROOT / "artifacts" / "calibration" / "monitors" / "m3" / "horizon_v1"
)
CALIBRATION_ARTIFACT = CALIBRATION_ROOT / "calibration.json"
CHECKPOINT_ROOT = EVAL_ROOT / "artifacts" / "checkpoints" / "m3"
FEATURE_ROOT = EVAL_ROOT / "artifacts" / "training" / "m3" / "demo_features"
TRAINING_ROOT = EVAL_ROOT / "artifacts" / "training" / "m3" / "runs"
METHOD = EVAL_ROOT / "configs" / "methods" / "m0_dynamac.json"
BACKEND_CONFIG = EVAL_ROOT / "configs" / "methods" / "m3_fail_detect_main10.json"
RECORD = ASSET_ROOT / "M3_HORIZON_ASSETS.json"
TASKS = tuple(task.task_id for task in experiment_task_set("horizon3"))
TASK_LEVELS = {task.task_id: task.task_level for task in experiment_task_set("horizon3")}
BASE_SEED = 2_712_000_000
CANDIDATES_PER_TASK = 600
SUCCESSES_PER_TASK = 50
WORKERS = 32
TIMEOUT_SECONDS = 900.0


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


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare() -> dict[str, Any]:
    import torch

    rows = []
    for task_offset, task in enumerate(TASKS):
        for index in range(CANDIDATES_PER_TASK):
            episode_id = f"horizon3_m3_normal_calibration_candidates/{task}/{index:04d}"
            rows.append(
                {
                    "schema": "essay2608.iclr2027.episode-manifest.v1",
                    "episode_id": episode_id,
                    "split": "horizon3_m3_normal_calibration_candidates",
                    "task": task,
                    "task_level": TASK_LEVELS[task],
                    "variation": 0,
                    "seed": BASE_SEED + task_offset * 100_000 + index,
                    "condition": "nominal",
                    "fault_family": None,
                    "fault_severity": None,
                    "trigger_stage": None,
                    "pair_id": episode_id,
                    "horizon": 1000,
                    "recovery_budget": 400,
                }
            )
    formal_sources = (
        EVAL_ROOT / "manifests" / "horizon3_single_event.jsonl",
        EVAL_ROOT / "manifests" / "horizon3_per_stage.jsonl",
        E4_ROOT / "manifest_views" / "nominal_100_per_level.jsonl",
    )
    calibration_keys = {(row["task"], int(row["seed"])) for row in rows}
    for source in formal_sources:
        if not source.is_file():
            continue
        overlap = calibration_keys & {
            (row["task"], int(row["seed"])) for row in _rows(source)
        }
        if overlap:
            raise RuntimeError(f"Horizon M3 calibration seeds overlap {source}: {len(overlap)}")
    candidate_payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if CANDIDATE_MANIFEST.is_file():
        if CANDIDATE_MANIFEST.read_text(encoding="utf-8") != candidate_payload:
            raise RuntimeError("existing Horizon M3 candidate manifest changed")
    else:
        CANDIDATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATE_MANIFEST.write_text(candidate_payload, encoding="utf-8")
    plan = {
        "schema": "essay2608.iclr2027.e4-m3-horizon-asset-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": list(TASKS),
        "training_source": "same_five_successful_demonstrations_as_each_frozen_horizon3_policy",
        "failure_trajectories_read": False,
        "sealed_test_read": False,
        "normal_calibration": {
            "candidate_manifest": str(CANDIDATE_MANIFEST.relative_to(ROOT)),
            "candidate_manifest_sha256": _sha256(CANDIDATE_MANIFEST),
            "base_seed": BASE_SEED,
            "candidates_per_task": CANDIDATES_PER_TASK,
            "successes_per_task": SUCCESSES_PER_TASK,
            "selection": "first_successes_in_preregistered_manifest_order",
        },
        "backend_config": {
            "path": str(BACKEND_CONFIG.relative_to(ROOT)),
            "sha256": _sha256(BACKEND_CONFIG),
        },
        "execution_environment": {
            "python": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_count": int(torch.cuda.device_count()),
        },
        "offline_pickle_compatibility": {
            "scope": "feature_export_process_only",
            "rule": "cache_numpy_core_multiarray_before_restricted_unpickle",
            "changes_source_bytes_or_decoded_values": False,
            "changes_frozen_endpoint_code": False,
        },
    }
    plan_path = ASSET_ROOT / "M3_HORIZON_ASSET_PLAN.json"
    if plan_path.is_file():
        previous = _json(plan_path)
        old = dict(previous)
        new = dict(plan)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError("existing Horizon M3 asset plan changed")
        return previous
    _atomic_json(plan_path, plan)
    return plan


def export_and_train() -> dict[str, Any]:
    prepare()
    missing = []
    for task in TASKS:
        checkpoint = CHECKPOINT_ROOT / task / "seed_1103" / "model.pt"
        if checkpoint.is_file():
            continue
        feature_index = FEATURE_ROOT / task / "index.json"
        if not feature_index.is_file():
            # NumPy 1.26 exposes ``numpy.core`` lazily.  Constructing the
            # restricted allowlist repeatedly can make ``numpy._core`` appear
            # halfway through one dict expression, although the fixed
            # multiarray functions are unchanged.  Cache that known numeric
            # namespace inside this offline export process only.  Do not edit
            # the A5-frozen adapter or the stored demonstration bytes.
            from integrations.rlbench.rlbench_dynamac.data import demo_adapter

            cached_multiarray = demo_adapter.np.core.multiarray
            demo_adapter._numpy_multiarray = lambda: cached_multiarray
            export_task(task, FEATURE_ROOT)
        missing.append(task)
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for index, task in enumerate(missing):
        code = (
            "import json; "
            "from evaluations.iclr2027.methods.fail_detect.training import train_task; "
            f"print(json.dumps(train_task({task!r}, device={('cuda:' + str(index % 8))!r}), sort_keys=True))"
        )
        processes.append(
            (
                task,
                subprocess.Popen(
                    [sys.executable, "-c", code],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    failures = []
    for task, process in processes:
        output, _ = process.communicate()
        print(output, end="", flush=True)
        if process.returncode:
            failures.append((task, process.returncode))
    if failures:
        raise RuntimeError(f"Horizon M3 training failed: {failures}")
    checkpoints = {}
    for task in TASKS:
        checkpoint = CHECKPOINT_ROOT / task / "seed_1103" / "model.pt"
        manifest = checkpoint.parent / "checkpoint_manifest.json"
        if not checkpoint.is_file() or not manifest.is_file():
            raise RuntimeError(f"missing Horizon M3 checkpoint: {task}")
        manifest_value = _json(manifest)
        if (
            manifest_value.get("checkpoint_sha256") != _sha256(checkpoint)
            or manifest_value.get("config_sha256") != _sha256(BACKEND_CONFIG)
        ):
            raise RuntimeError(f"invalid Horizon M3 checkpoint identity: {task}")
        checkpoints[task] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": _sha256(checkpoint),
            "manifest_path": str(manifest.relative_to(ROOT)),
            "manifest_sha256": _sha256(manifest),
        }
    return checkpoints


def collect() -> dict[str, Any]:
    prepare()
    command = [
        sys.executable,
        "-m",
        "evaluations.iclr2027.runners.launch",
        "--manifest",
        str(CANDIDATE_MANIFEST),
        "--output-root",
        str(CANDIDATE_RESULTS),
        "--workers",
        str(WORKERS),
        "--episode-timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--retry-infrastructure",
        "1",
        "--until-successes-per-task",
        str(SUCCESSES_PER_TASK),
        "--method",
        str(METHOD),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    status_path = CANDIDATE_RESULTS / "QUEUE_STATUS.json"
    status = _json(status_path) if status_path.is_file() else {}
    successes = status.get("successes", {})
    complete = all(int(successes.get(task, 0)) >= SUCCESSES_PER_TASK for task in TASKS)
    if completed.returncode not in (0, 2) or not complete:
        raise RuntimeError("Horizon M3 normal calibration candidates are incomplete")
    # ``materialize`` is also used in the canonical manifest directory, where
    # a global index already exists.  This experiment-scoped asset directory
    # deliberately has no global index, so seed it with the frozen candidate
    # identity before the common helper appends the read-only 400-row view.
    local_index = ASSET_ROOT / "MANIFEST_INDEX.json"
    if not local_index.is_file():
        _atomic_json(
            local_index,
            {
                "schema": "essay2608.iclr2027.manifest-index.v1",
                "manifests": {
                    CANDIDATE_MANIFEST.name: {
                        "rows": len(_rows(CANDIDATE_MANIFEST)),
                        "sha256": _sha256(CANDIDATE_MANIFEST),
                    }
                },
                "calibration_candidate_extensions": {},
                "sealed_executed": False,
                "result_based_task_selection": False,
                "scope": "E4 M3 Horizon-3 normal-calibration assets only",
            },
        )
    materialize(
        CANDIDATE_MANIFEST,
        CANDIDATE_RESULTS,
        CALIBRATION_MANIFEST,
        successes_per_task=SUCCESSES_PER_TASK,
    )
    return status


def calibrate() -> dict[str, Any]:
    checkpoints = export_and_train()
    if not CALIBRATION_MANIFEST.is_file():
        collect()
    artifact = calibrate_m3(
        manifest_path=CALIBRATION_MANIFEST,
        result_root=CANDIDATE_RESULTS,
        output_root=CALIBRATION_ROOT,
        device="cuda:0",
    )
    if set(artifact.get("tasks", {})) != set(TASKS):
        raise RuntimeError("Horizon M3 calibration task coverage is incomplete")
    record = {
        "schema": "essay2608.iclr2027.e4-m3-horizon-assets.v1",
        "status": "PASS",
        "tasks": list(TASKS),
        "checkpoints": checkpoints,
        "calibration": str(CALIBRATION_ARTIFACT.relative_to(ROOT)),
        "calibration_sha256": _sha256(CALIBRATION_ARTIFACT),
        "calibration_manifest": str(CALIBRATION_MANIFEST.relative_to(ROOT)),
        "calibration_manifest_sha256": _sha256(CALIBRATION_MANIFEST),
        "calibration_episodes": len(_rows(CALIBRATION_MANIFEST)),
        "failure_trajectories_read": False,
        "sealed_test_read": False,
        "existing_main10_results_read": False,
    }
    _atomic_json(RECORD, record)
    preflight()
    return record


def preflight() -> dict[str, Any]:
    """Validate every task asset and execute one real CPU model forward."""

    import numpy as np
    import torch

    from evaluations.iclr2027.methods.fail_detect.model import (
        build_official_velocity_model,
    )
    from evaluations.iclr2027.methods.fail_detect.preprocessing import FeatureLayout
    from evaluations.iclr2027.methods.fail_detect.training import _prepared_features

    if not RECORD.is_file() or not CALIBRATION_ARTIFACT.is_file():
        raise RuntimeError("Horizon M3 assets are not frozen")
    record = _json(RECORD)
    calibration = _json(CALIBRATION_ARTIFACT)
    if record.get("status") != "PASS" or tuple(record.get("tasks", ())) != TASKS:
        raise RuntimeError("Horizon M3 asset record is incomplete")
    if set(calibration.get("tasks", {})) != set(TASKS):
        raise RuntimeError("Horizon M3 calibration task coverage is incomplete")
    if _sha256(CALIBRATION_ARTIFACT) != record.get("calibration_sha256"):
        raise RuntimeError("Horizon M3 calibration hash mismatch")
    backend_hash = _sha256(BACKEND_CONFIG)
    for task in TASKS:
        entry = record["checkpoints"][task]
        checkpoint = ROOT / entry["path"]
        manifest_path = ROOT / entry["manifest_path"]
        if (
            _sha256(checkpoint) != entry["sha256"]
            or _sha256(manifest_path) != entry["manifest_sha256"]
        ):
            raise RuntimeError(f"Horizon M3 checkpoint changed: {task}")
        manifest = _json(manifest_path)
        if (
            manifest.get("checkpoint_sha256") != entry["sha256"]
            or manifest.get("config_sha256") != backend_hash
        ):
            raise RuntimeError(f"Horizon M3 checkpoint identity mismatch: {task}")
        calibrated = calibration["tasks"][task]
        upper = calibrated.get("threshold_schedule", {}).get("upper", [])
        horizon = int(calibrated.get("policy_horizon", 0))
        if (
            calibrated.get("checkpoint_sha256") != entry["sha256"]
            or horizon < 1
            or len(upper) != horizon
            or not np.isfinite(upper).all()
        ):
            raise RuntimeError(f"Horizon M3 threshold identity is invalid: {task}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        if (
            payload.get("schema") != "essay2608.iclr2027.m3-logpzo.v1"
            or metadata.get("task") != task
            or metadata.get("config_sha256") != backend_hash
            or metadata.get("fit_split") != "normal_demonstrations"
        ):
            raise RuntimeError(f"Horizon M3 checkpoint metadata mismatch: {task}")
        layout = FeatureLayout.from_dict(metadata["layout"])
        prepared = _prepared_features(
            np.zeros((1, layout.input_dim), dtype=np.float32),
            int(metadata["unet_input_channels"]),
        )
        model = build_official_velocity_model(int(metadata["unet_input_channels"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        tensor = torch.from_numpy(prepared)
        with torch.no_grad():
            output = model(tensor, torch.zeros(1, dtype=torch.long))
        if output.shape != tensor.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"Horizon M3 forward check failed: {task}")
    return {
        "status": "PASS",
        "tasks": list(TASKS),
        "python": str(Path(sys.executable).resolve()),
        "torch": str(torch.__version__),
        "device": "cpu",
    }


def status() -> dict[str, Any]:
    return {
        "prepared": CANDIDATE_MANIFEST.is_file(),
        "trained_tasks": [
            task for task in TASKS
            if (CHECKPOINT_ROOT / task / "seed_1103" / "model.pt").is_file()
        ],
        "candidate_results": len(list((CANDIDATE_RESULTS / "episodes").glob("*.json"))),
        "materialized_calibration": (
            len(_rows(CALIBRATION_MANIFEST)) if CALIBRATION_MANIFEST.is_file() else 0
        ),
        "frozen": RECORD.is_file() and _json(RECORD).get("status") == "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "train", "collect", "calibrate", "preflight", "all", "status"),
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare()
    elif args.command == "train":
        value = export_and_train()
    elif args.command == "collect":
        value = collect()
    elif args.command == "calibrate":
        value = calibrate()
    elif args.command == "preflight":
        value = preflight()
    elif args.command == "all":
        export_and_train()
        collect()
        value = calibrate()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
