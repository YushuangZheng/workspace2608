"""Single acceptance audit for the server-A A2 evaluation infrastructure."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluations.iclr2027.interfaces.feature_schema import validate_feature_record
from evaluations.iclr2027.manifests.build import ROOT as MANIFEST_ROOT
from evaluations.iclr2027.manifests.build import validate_all_manifests
from evaluations.iclr2027.runners.episode_io import load_episode, resolve_cycle_file

SCHEMA = "essay2608.iclr2027.a2-acceptance.v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result(root: Path, episode_id: str) -> dict[str, Any]:
    safe = episode_id.replace("/", "__")
    return load_episode(root / "episodes" / (safe + ".json"))


def _validate_artifact(result_path: Path, result: dict[str, Any]) -> None:
    cycle_path = resolve_cycle_file(result_path, result)
    if not cycle_path.is_file() or _sha256(cycle_path) != result["cycle_file_sha256"]:
        raise ValueError(f"cycle artifact hash mismatch: {result['episode_id']}")
    count = 0
    with gzip.open(cycle_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["cycle"] != count:
                raise ValueError(f"non-contiguous cycles: {result['episode_id']}")
            validate_feature_record(row["feature"])
            if row["feature"]["episode_id"] != result["episode_id"]:
                raise ValueError(f"feature/result identity mismatch: {result['episode_id']}")
            if row["audit"]["cycle"] != count:
                raise ValueError(f"audit/feature alignment mismatch: {result['episode_id']}")
            count += 1
    if count != result["cycle_records"] or count != result["cycles"]:
        raise ValueError(f"cycle count mismatch: {result['episode_id']}")


def _run_metadata(root: Path) -> dict[str, Any]:
    value = json.loads((root / "RUN_METADATA.json").read_text())
    status = json.loads((root / "QUEUE_STATUS.json").read_text())
    if value["workers"] != 48 or status["peak_active"] != 48:
        raise ValueError(f"48-worker admission was not reached in {root}")
    if not value["one_episode_per_job"] or not value["dynamic_global_queue"]:
        raise ValueError(f"invalid queue semantics in {root}")
    if status["infrastructure_errors"]:
        raise ValueError(f"infrastructure failures remain in {root}")
    return {
        "workers": value["workers"],
        "peak_active": status["peak_active"],
        "renderer": value["renderer"],
        "manifest_sha256": value["manifest_sha256"],
        "finished_by_task": status["finished"],
        "successes_by_task": status["successes"],
        "infrastructure_errors": status["infrastructure_errors"],
    }


def accept(
    development_root: Path,
    calibration_root: Path,
    failure_train_root: Path,
) -> dict[str, Any]:
    manifest_counts = validate_all_manifests(MANIFEST_ROOT)
    development_manifest = _rows(MANIFEST_ROOT / "main10_development.jsonl")
    if len(development_manifest) != 200:
        raise ValueError("development manifest is not 10+10 per Main-10 task")
    development = [_result(development_root, row["episode_id"]) for row in development_manifest]
    if any(row["reason"] == "infrastructure_error" for row in development):
        raise ValueError("development contains infrastructure errors")
    eligible = Counter()
    triggered = Counter()
    for row in development:
        if row["condition"] != "perturbed":
            continue
        family = row["fault_family"]
        eligible[family] += int(bool(row["audit"]["eligible"]))
        triggered[family] += int(bool(row["audit"]["physically_triggered"]))
    rates = {
        family: triggered[family] / eligible[family]
        for family in sorted(eligible)
        if eligible[family]
    }
    if not rates or min(rates.values()) < 0.8:
        raise ValueError(f"development trigger gate failed: {rates}")

    calibration_manifest = _rows(MANIFEST_ROOT / "main10_normal_calibration.jsonl")
    calibration_counts = Counter(row["task"] for row in calibration_manifest)
    if len(calibration_manifest) != 500 or set(calibration_counts.values()) != {50}:
        raise ValueError(f"calibration view is not 50 successes/task: {calibration_counts}")
    calibration_results = []
    for row in calibration_manifest:
        source = _result(calibration_root, row["source_episode_id"])
        if not source["success"] or source["condition"] != "nominal":
            raise ValueError(f"invalid retained calibration source: {row['source_episode_id']}")
        calibration_results.append(source)

    train_manifest = _rows(MANIFEST_ROOT / "main10_failure_train.jsonl")
    if len(train_manifest) != 2000:
        raise ValueError("failure-train manifest is not 200/task")
    train_results = [_result(failure_train_root, row["episode_id"]) for row in train_manifest]
    if any(row["reason"] == "infrastructure_error" for row in train_results):
        raise ValueError("failure train contains infrastructure errors")
    train_counts = Counter(row["task"] for row in train_results)
    if len(train_counts) != 10 or set(train_counts.values()) != {200}:
        raise ValueError("failure train is not balanced at 200/task")

    result_paths = {
        result["episode_id"]: root / "episodes" / (result["episode_id"].replace("/", "__") + ".json")
        for root, results in (
            (development_root, development),
            (calibration_root, calibration_results),
            (failure_train_root, train_results),
        )
        for result in results
    }
    all_results = development + calibration_results + train_results
    for result in all_results:
        _validate_artifact(result_paths[result["episode_id"]], result)
    incomplete = [
        str(path)
        for root in (development_root, calibration_root, failure_train_root)
        for path in root.rglob("*.tmp")
    ]
    if incomplete:
        raise ValueError(f"incomplete artifacts remain: {incomplete[:5]}")
    sealed_result_paths = [
        str(path)
        for path in (MANIFEST_ROOT.parent / "results").rglob("*")
        if path.is_file() and "sealed" in str(path)
    ]
    if sealed_result_paths:
        raise ValueError("sealed results were touched during A2")

    train_eligible = sum(bool(row["audit"]["eligible"]) for row in train_results)
    train_triggered = sum(bool(row["audit"]["physically_triggered"]) for row in train_results)
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "purpose": "A2_INFRASTRUCTURE_AND_TRAINING_DATA_NOT_PAPER_RESULTS",
        "manifests": manifest_counts,
        "development": {
            "episodes": len(development),
            "infrastructure_errors": 0,
            "eligible": sum(eligible.values()),
            "physically_triggered": sum(triggered.values()),
            "trigger_rate_by_family": rates,
        },
        "normal_calibration": {
            "retained_successful_nominal_rollouts": len(calibration_results),
            "per_task": dict(sorted(calibration_counts.items())),
            "server_visibility": "A_only",
        },
        "failure_train": {
            "episodes": len(train_results),
            "per_task": dict(sorted(train_counts.items())),
            "eligible": train_eligible,
            "physically_triggered": train_triggered,
            "trigger_rate_given_eligible": train_triggered / train_eligible if train_eligible else None,
        },
        "concurrency": {
            "calibration_candidates": _run_metadata(calibration_root),
            "failure_train": _run_metadata(failure_train_root),
        },
        "validated_cycle_artifacts": len(all_results),
        "sealed_test_executed": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--failure-train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = accept(args.development_root, args.calibration_root, args.failure_train_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
