"""Freeze and run the A6 E4 Horizon-3 long-horizon evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.runners.a6_endpoint_compatibility import (
    assert_frozen_a5_with_ablation_extension,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULT_ROOT = EVAL_ROOT / "results" / "controlled" / "e4"
VIEW_ROOT = RESULT_ROOT / "manifest_views"
PLAN_PATH = RESULT_ROOT / "A6_E4_RUN_PLAN.json"
A5_PLAN = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_RUN_PLAN.json"
A5_ACCEPTANCE = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_ACCEPTANCE.json"
SINGLE_SOURCE = EVAL_ROOT / "manifests" / "horizon3_single_event.jsonl"
PER_STAGE_SOURCE = EVAL_ROOT / "manifests" / "horizon3_per_stage.jsonl"
M3_HORIZON_ASSET_RECORD = (
    RESULT_ROOT / "m3_horizon_assets" / "M3_HORIZON_ASSETS.json"
)
M3_HORIZON_CALIBRATION = (
    EVAL_ROOT
    / "artifacts"
    / "calibration"
    / "monitors"
    / "m3"
    / "horizon_v1"
    / "calibration.json"
)
M5_HORIZON_ASSET_RECORD = (
    RESULT_ROOT / "m5_horizon_assets" / "M5_HORIZON_ASSETS.json"
)
M5_HORIZON_WORKER_PREFLIGHT = (
    RESULT_ROOT / "m5_horizon_assets" / "M5_HORIZON_WORKER_PREFLIGHT.json"
)
M5_HORIZON_SIMULATOR_PREFLIGHT = (
    RESULT_ROOT / "m5_horizon_assets" / "M5_HORIZON_SIMULATOR_PREFLIGHT.json"
)
M3_POLICY_PYTHON = Path(
    "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python"
)
TASK_ADAPTER = ROOT / "integrations" / "rlbench" / "iclr2027" / "tasks.py"
HORIZON_EVENTS = EVAL_ROOT / "audit" / "horizon_events.py"
HORIZON_EPISODE = EVAL_ROOT / "runners" / "a6_horizon_episode.py"
HORIZON_LAUNCH = EVAL_ROOT / "runners" / "a6_horizon_launch.py"
TASKS = (
    "place_cups_1",
    "place_cups_2",
    "place_cups_3",
    "remove_cups_1",
    "remove_cups_2",
    "push_buttons_1",
    "push_buttons_2",
    "push_buttons_3",
)
CONDITIONS = ("nominal", "single_event", "per_stage")
METHODS = (
    ("m0", EVAL_ROOT / "configs" / "methods" / "m0_dynamac.json", None),
    ("m1", EVAL_ROOT / "configs" / "methods" / "m1_restart.json", None),
    (
        "m3",
        EVAL_ROOT / "configs" / "methods" / "m3_fail_detect_runtime.json",
        M3_HORIZON_CALIBRATION,
    ),
    ("m5", EVAL_ROOT / "configs" / "methods" / "m5_full.json", None),
)
WORKERS = 32
TIMEOUT_SECONDS = 900.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
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
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _assert_frozen_a5() -> dict[str, Any]:
    acceptance = _read_json(A5_ACCEPTANCE)
    if not str(acceptance.get("status", "")).startswith("PASS"):
        raise RuntimeError("A5 acceptance is not a frozen PASS")
    plan, _extension = assert_frozen_a5_with_ablation_extension(A5_PLAN)
    return plan


def _assert_m3_horizon_assets() -> dict[str, Any]:
    """Reject E4 before launch unless every Horizon-3 M3 asset is frozen.

    Main-10 contains only ``place_cups_3`` from the Horizon-3 suite.  Reusing
    its calibration artifact for the other seven task levels is invalid: both
    the normal-data scorer and the conformal threshold are task-specific.
    This explicit gate prevents a missing threshold from being misclassified
    as an intention-to-treat infrastructure failure.
    """

    if not M3_HORIZON_ASSET_RECORD.is_file():
        raise RuntimeError("E4 M3 Horizon-3 asset record is missing")
    record = _read_json(M3_HORIZON_ASSET_RECORD)
    if record.get("status") != "PASS" or tuple(record.get("tasks", ())) != TASKS:
        raise RuntimeError("E4 M3 Horizon-3 asset record is not a complete PASS")
    calibration = _read_json(M3_HORIZON_CALIBRATION)
    if set(calibration.get("tasks", {})) != set(TASKS):
        raise RuntimeError("E4 M3 calibration does not cover every Horizon-3 task")
    if _sha256(M3_HORIZON_CALIBRATION) != record.get("calibration_sha256"):
        raise RuntimeError("E4 M3 Horizon-3 calibration changed after freezing")
    checkpoints = record.get("checkpoints", {})
    for task in TASKS:
        entry = checkpoints.get(task, {})
        path = ROOT / str(entry.get("path", ""))
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"E4 M3 checkpoint is missing or changed: {task}")
        manifest_path = ROOT / str(entry.get("manifest_path", ""))
        if (
            not manifest_path.is_file()
            or _sha256(manifest_path) != entry.get("manifest_sha256")
        ):
            raise RuntimeError(f"E4 M3 checkpoint manifest changed: {task}")
        task_calibration = calibration["tasks"][task]
        if task_calibration.get("checkpoint_sha256") != entry.get("sha256"):
            raise RuntimeError(f"E4 M3 threshold/checkpoint mismatch: {task}")
    # Run the deserialization/forward check in the exact policy environment.
    # The orchestration environment intentionally does not carry PyTorch.
    completed = subprocess.run(
        [
            str(M3_POLICY_PYTHON),
            "-m",
            "evaluations.iclr2027.runners.a6_e4_m3_assets",
            "preflight",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(f"E4 M3 runtime preflight failed:\n{completed.stdout}")
    return record


def _assert_m5_horizon_assets() -> dict[str, Any]:
    """Fail before launch unless all Horizon M5 task assets actually run."""

    if not M5_HORIZON_ASSET_RECORD.is_file():
        raise RuntimeError("E4 M5 Horizon-3 asset record is missing")
    record = _read_json(M5_HORIZON_ASSET_RECORD)
    if record.get("status") != "PASS" or tuple(record.get("tasks", ())) != TASKS:
        raise RuntimeError("E4 M5 Horizon-3 asset record is not a complete PASS")
    method = record.get("method_config", {})
    method_path = ROOT / str(method.get("path", ""))
    if (
        not method_path.is_file()
        or _sha256(method_path) != method.get("sha256")
        or bool(method.get("changed"))
    ):
        raise RuntimeError("E4 M5 method identity changed while adding Horizon assets")
    runtime_configs = record.get("runtime_configs", {})
    model_files = record.get("model_files", {})
    for task in TASKS:
        runtime = runtime_configs.get(task, {})
        path = ROOT / str(runtime.get("installed_path", ""))
        if not path.is_file() or _sha256(path) != runtime.get("installed_sha256"):
            raise RuntimeError(f"E4 M5 boundary config is missing or changed: {task}")
        files = model_files.get(task, {})
        if not files:
            raise RuntimeError(f"E4 M5 model identity is missing: {task}")
        for relative, expected in files.items():
            model_path = ROOT / str(relative)
            if not model_path.is_file() or _sha256(model_path) != expected:
                raise RuntimeError(f"E4 M5 model file is missing or changed: {relative}")
    conditional = record.get("conditional_calibration", {})
    if bool(conditional.get("runtime_explanation_gate_changed")):
        raise RuntimeError("E4 M5 conditional calibration opened the runtime support gate")
    for path in (M5_HORIZON_WORKER_PREFLIGHT, M5_HORIZON_SIMULATOR_PREFLIGHT):
        if not path.is_file() or _read_json(path).get("status") != "PASS":
            raise RuntimeError(f"E4 M5 hard preflight is missing or failed: {path.name}")
    for key, expected_path in (
        ("worker_preflight", M5_HORIZON_WORKER_PREFLIGHT),
        ("simulator_preflight", M5_HORIZON_SIMULATOR_PREFLIGHT),
    ):
        frozen = record.get(key, {})
        if (
            str(expected_path.relative_to(ROOT)) != frozen.get("path")
            or _sha256(expected_path) != frozen.get("sha256")
        ):
            raise RuntimeError(f"E4 M5 {key} identity changed after PASS")
    completed = subprocess.run(
        [
            str(M3_POLICY_PYTHON),
            "-m",
            "evaluations.iclr2027.runners.a6_e4_m5_assets",
            "worker-preflight",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(f"E4 M5 runtime preflight failed:\n{completed.stdout}")
    return record


def _view(source: Path, output: Path, *, per_stage: bool) -> list[dict[str, Any]]:
    selected = {task: [] for task in TASKS}
    for row in _rows(source):
        task = str(row["task"])
        if task in selected and len(selected[task]) < 100:
            selected[task].append(row)
    rows = [row for task in TASKS for row in selected[task]]
    if Counter(str(row["task"]) for row in rows) != Counter({task: 100 for task in TASKS}):
        raise RuntimeError("E4 view must contain exactly 100 rows per Horizon-3 level")
    if per_stage != all(
        row.get("event_schedule") == "one_eligible_event_per_interaction_stage"
        for row in rows
    ):
        raise RuntimeError("E4 event-schedule identity mismatch")
    _atomic_jsonl(output, rows)
    return rows


def _nominal_view(
    paired_single_event_rows: Iterable[Mapping[str, Any]], output: Path
) -> list[dict[str, Any]]:
    """Create the prespecified undisturbed side of each Horizon-3 pair.

    E4 compares horizon length under the same initialization both with and
    without one physical event.  The nominal rows therefore retain the seed,
    variation, horizon, task level, and pair identity of the exactly-one-event
    rows while removing only the fault assignment.
    """

    rows = []
    for source in paired_single_event_rows:
        row = dict(source)
        suffix = str(source["episode_id"]).split("/", 1)[1]
        row.update(
            {
                "episode_id": f"horizon3_nominal/{suffix}",
                "split": "horizon3_nominal",
                "condition": "nominal",
                "fault_family": None,
                "fault_severity": None,
                "trigger_stage": None,
                "source_episode_id": source["episode_id"],
            }
        )
        row.pop("event_schedule", None)
        rows.append(row)
    _atomic_jsonl(output, rows)
    return rows


def _assert_nominal_pairing(
    nominal_rows: Iterable[Mapping[str, Any]],
    single_event_rows: Iterable[Mapping[str, Any]],
) -> None:
    nominal = list(nominal_rows)
    single = list(single_event_rows)
    if len(nominal) != len(single):
        raise RuntimeError("E4 nominal and single-event views differ in length")
    paired_fields = ("task", "task_level", "variation", "seed", "pair_id", "horizon")
    for left, right in zip(nominal, single):
        if any(left.get(field) != right.get(field) for field in paired_fields):
            raise RuntimeError("E4 nominal/single-event initialization mismatch")
        if left.get("condition") != "nominal" or any(
            left.get(field) is not None
            for field in ("fault_family", "fault_severity", "trigger_stage")
        ):
            raise RuntimeError("E4 nominal view carries a fault assignment")


def prepare() -> dict[str, Any]:
    a5 = _assert_frozen_a5()
    m3_horizon_assets = _assert_m3_horizon_assets()
    m5_horizon_assets = _assert_m5_horizon_assets()
    nominal_path = VIEW_ROOT / "nominal_100_per_level.jsonl"
    single_path = VIEW_ROOT / "exactly_one_event_100_per_level.jsonl"
    per_stage_path = VIEW_ROOT / "per_stage_100_per_level.jsonl"
    single_rows = _view(SINGLE_SOURCE, single_path, per_stage=False)
    nominal_rows = _nominal_view(single_rows, nominal_path)
    _assert_nominal_pairing(nominal_rows, single_rows)
    per_stage_rows = _view(PER_STAGE_SOURCE, per_stage_path, per_stage=True)
    methods = []
    frozen_methods = {item["key"]: item for item in a5["methods"]}
    for key, config, calibration in METHODS:
        if not config.is_file() or (calibration is not None and not calibration.is_file()):
            raise RuntimeError(f"missing frozen E4 method input: {key}")
        if _sha256(config) != frozen_methods[key]["config_sha256"]:
            raise RuntimeError(f"E4 method differs from frozen A5 identity: {key}")
        methods.append(
            {
                "key": key,
                "config": str(config.relative_to(ROOT)),
                "config_sha256": _sha256(config),
                "calibration": None
                if calibration is None
                else {
                    "path": str(calibration.relative_to(ROOT)),
                    "sha256": _sha256(calibration),
                },
            }
        )
    infrastructure = {}
    for path in (TASK_ADAPTER, HORIZON_EVENTS, HORIZON_EPISODE, HORIZON_LAUNCH):
        infrastructure[str(path.relative_to(ROOT))] = _sha256(path)
    plan = {
        "schema": "essay2608.iclr2027.a6-e4-run-plan.v3",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "experiment": "E4_Horizon-3",
        "sealed_test": True,
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "one_episode_per_job": True,
        "dynamic_global_queue": True,
        "selection_rule": "first_100_rows_per_task_level_in_A3_frozen_source_order",
        "nominal_pairing_rule": (
            "same_task_level_variation_seed_pair_id_and_horizon_as_single_event;"
            "fault_assignment_removed"
        ),
        "a5_endpoint_identity": a5["endpoint_code_identity"]["aggregate_sha256"],
        "m3_horizon_asset_record": {
            "path": str(M3_HORIZON_ASSET_RECORD.relative_to(ROOT)),
            "sha256": _sha256(M3_HORIZON_ASSET_RECORD),
            "calibration_sha256": m3_horizon_assets["calibration_sha256"],
            "sealed_test_read": False,
        },
        "m5_horizon_asset_record": {
            "path": str(M5_HORIZON_ASSET_RECORD.relative_to(ROOT)),
            "sha256": _sha256(M5_HORIZON_ASSET_RECORD),
            "method_config_sha256": m5_horizon_assets["method_config"]["sha256"],
            "existing_main10_files_overwritten": False,
            "failure_trajectories_read": False,
            "formal_results_read": False,
        },
        "infrastructure": infrastructure,
        "manifests": {
            "nominal": {
                "path": str(nominal_path.relative_to(ROOT)),
                "sha256": _sha256(nominal_path),
                "episodes": len(nominal_rows),
                "paired_source": str(single_path.relative_to(ROOT)),
                "paired_source_sha256": _sha256(single_path),
            },
            "single_event": {
                "path": str(single_path.relative_to(ROOT)),
                "sha256": _sha256(single_path),
                "episodes": len(single_rows),
                "source": str(SINGLE_SOURCE.relative_to(ROOT)),
                "source_sha256": _sha256(SINGLE_SOURCE),
            },
            "per_stage": {
                "path": str(per_stage_path.relative_to(ROOT)),
                "sha256": _sha256(per_stage_path),
                "episodes": len(per_stage_rows),
                "source": str(PER_STAGE_SOURCE.relative_to(ROOT)),
                "source_sha256": _sha256(PER_STAGE_SOURCE),
            },
        },
        "methods": methods,
        "expected_new_episodes": len(methods) * (
            len(nominal_rows) + len(single_rows) + len(per_stage_rows)
        ),
    }
    if PLAN_PATH.is_file():
        previous = _read_json(PLAN_PATH)
        old = dict(previous)
        new = dict(plan)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError("existing E4 run plan disagrees with frozen inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def _run_cell(method: Mapping[str, Any], condition: str, plan: Mapping[str, Any]) -> None:
    manifest = ROOT / str(plan["manifests"][condition]["path"])
    output = RESULT_ROOT / "end_to_end" / str(method["key"]) / condition
    module = (
        "evaluations.iclr2027.runners.a6_horizon_launch"
        if condition == "per_stage"
        else "evaluations.iclr2027.runners.launch"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--manifest",
        str(manifest),
        "--output-root",
        str(output),
        "--workers",
        str(WORKERS),
        "--episode-timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--retry-infrastructure",
        "1",
        "--method",
        str(ROOT / str(method["config"])),
    ]
    if method["calibration"] is not None:
        command.extend(
            ["--calibration-artifact", str(ROOT / method["calibration"]["path"])]
        )
    completed = subprocess.run(command, cwd=ROOT)
    status_path = output / "QUEUE_STATUS.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    expected = int(plan["manifests"][condition]["episodes"])
    complete_itt = (
        int(status.get("completed_episode_count", -1)) == expected
        and int(status.get("missing_selected_count", -1)) == 0
    )
    if completed.returncode not in (0, 2) or not complete_itt:
        raise SystemExit(completed.returncode or 4)


def run() -> None:
    plan = prepare()
    for method in plan["methods"]:
        for condition in CONDITIONS:
            _run_cell(method, condition, plan)


def status() -> dict[str, Any]:
    plan = prepare()
    cells = {}
    total = 0
    for method in plan["methods"]:
        for condition in CONDITIONS:
            count = len(
                list(
                    (
                        RESULT_ROOT
                        / "end_to_end"
                        / str(method["key"])
                        / condition
                        / "episodes"
                    ).glob("*.json")
                )
            )
            expected = int(plan["manifests"][condition]["episodes"])
            cells[f"{method['key']}/{condition}"] = {
                "completed": count,
                "expected": expected,
            }
            total += count
    return {
        "stage": "A6",
        "experiment": "E4_Horizon-3",
        "prepared": True,
        "completed_episodes": total,
        "expected_episodes": int(plan["expected_new_episodes"]),
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    args = parser.parse_args(argv)
    if args.command == "run":
        run()
        value = status()
    elif args.command == "prepare":
        value = prepare()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
