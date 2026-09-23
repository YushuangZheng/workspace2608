"""Freeze and run the three paper-level E5 ablations on Stress-4.

The runner is deliberately orchestration-only.  It constructs a deterministic
read-only nominal view, verifies that the retained E1/A5 endpoint is still the
frozen one, and delegates every physical episode to the common 32-worker
queue.  No policy mechanism or task-specific behavior is implemented here.
"""

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
    assert_frozen_m5_post_e4,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULT_ROOT = EVAL_ROOT / "results" / "controlled" / "e5"
VIEW_ROOT = RESULT_ROOT / "manifest_views"
PLAN_PATH = RESULT_ROOT / "A6_E5_CORE_RUN_PLAN.json"
A5_PLAN = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_RUN_PLAN.json"
A5_ACCEPTANCE = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_ACCEPTANCE.json"
M5_POST_E4_FREEZE = EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"

NOMINAL = EVAL_ROOT / "manifests" / "main10_nominal.jsonl"
PERTURBED = EVAL_ROOT / "manifests" / "ablation4.jsonl"
STRESS4 = (
    "open_drawer",
    "place_cups_3",
    "bimanual_handover_item",
    "bimanual_lift_tray",
)
METHODS = {
    "motion_only": EVAL_ROOT / "configs" / "methods" / "ablation_motion_only.json",
    "open_loop_progress": EVAL_ROOT / "configs" / "methods" / "ablation_open_loop_progress.json",
    "generic_retry": EVAL_ROOT / "configs" / "methods" / "ablation_generic_retry.json",
}
WORKERS = 32
TIMEOUT_SECONDS = 900.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def _assert_frozen_m5() -> dict[str, Any]:
    acceptance = _read_json(A5_ACCEPTANCE)
    if not str(acceptance.get("status", "")).startswith("PASS"):
        raise RuntimeError("A5 acceptance is not a frozen PASS")
    return assert_frozen_m5_post_e4()


def prepare() -> dict[str, Any]:
    freeze = _assert_frozen_m5()
    # E5 preregisters 100 nominal episodes per task whereas the retained E1
    # manifest contains 200.  Select the first 100 rows in the immutable
    # source-manifest order, independently of all observed outcomes.  The
    # materialized view is shared by every variant and its hash is frozen
    # below, so this cannot become a result-dependent subset.
    nominal_by_task = {task: [] for task in STRESS4}
    for row in _rows(NOMINAL):
        task = str(row["task"])
        if task in nominal_by_task and len(nominal_by_task[task]) < 100:
            nominal_by_task[task].append(row)
    nominal_rows = [row for task in STRESS4 for row in nominal_by_task[task]]
    nominal_counts = Counter(str(row["task"]) for row in nominal_rows)
    if len(nominal_rows) != 400 or nominal_counts != Counter({task: 100 for task in STRESS4}):
        raise RuntimeError("E5 nominal view must be Stress-4 x 100")
    perturbed_rows = _rows(PERTURBED)
    perturbed_counts = Counter(str(row["task"]) for row in perturbed_rows)
    if len(perturbed_rows) != 800 or perturbed_counts != Counter({task: 200 for task in STRESS4}):
        raise RuntimeError("E5 perturbed view must be Stress-4 x 200")
    if any(row.get("condition") != "nominal" for row in nominal_rows):
        raise RuntimeError("E5 nominal view contains a perturbed row")
    if any(row.get("condition") != "perturbed" for row in perturbed_rows):
        raise RuntimeError("E5 perturbed view contains a nominal row")

    nominal_view = VIEW_ROOT / "stress4_nominal.jsonl"
    _atomic_jsonl(nominal_view, nominal_rows)
    methods = []
    for key, path in METHODS.items():
        config = _read_json(path)
        if config.get("method_id") != "ablation_" + key:
            raise RuntimeError(f"E5 method identity mismatch: {path}")
        methods.append(
            {
                "key": key,
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "feature_profile": config.get("feature_profile"),
            }
        )
    plan = {
        "schema": "essay2608.iclr2027.a6-e5-core-run-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "experiment": "E5_core_ablation",
        "sealed_test": True,
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "one_episode_per_job": True,
        "dynamic_global_queue": True,
        "a5_run_plan": {
            "path": str(A5_PLAN.relative_to(ROOT)),
            "sha256": _sha256(A5_PLAN),
        },
        "m5_post_e4_freeze": {
            "path": str(M5_POST_E4_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(M5_POST_E4_FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        },
        "manifests": {
            "nominal": {
                "path": str(nominal_view.relative_to(ROOT)),
                "sha256": _sha256(nominal_view),
                "episodes": 400,
                "source_path": str(NOMINAL.relative_to(ROOT)),
                "source_sha256": _sha256(NOMINAL),
                "selection_rule": "first_100_rows_per_task_in_source_manifest_order",
            },
            "perturbed": {
                "path": str(PERTURBED.relative_to(ROOT)),
                "sha256": _sha256(PERTURBED),
                "episodes": 800,
            },
        },
        "methods": methods,
        "expected_new_episodes": 3 * (400 + 800),
    }
    if PLAN_PATH.exists():
        previous = _read_json(PLAN_PATH)
        comparable = dict(plan)
        old_comparable = dict(previous)
        comparable.pop("created_utc", None)
        old_comparable.pop("created_utc", None)
        if comparable != old_comparable:
            raise RuntimeError("existing E5 core run plan disagrees with frozen inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def _run_cell(method: Mapping[str, Any], condition: str, plan: Mapping[str, Any]) -> None:
    expected = int(plan["manifests"][condition]["episodes"])
    manifest = ROOT / str(plan["manifests"][condition]["path"])
    config = ROOT / str(method["path"])
    output = RESULT_ROOT / "core" / str(method["key"]) / condition
    command = [
        sys.executable,
        "-m",
        "evaluations.iclr2027.runners.launch",
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
        str(config),
    ]
    print(
        json.dumps(
            {"event": "cell_start", "method": method["key"], "condition": condition, "episodes": expected}
        ),
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT)
    status_path = output / "QUEUE_STATUS.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    complete_itt = (
        int(status.get("completed_episode_count", -1)) == expected
        and int(status.get("missing_selected_count", -1)) == 0
    )
    if completed.returncode not in (0, 2) or not complete_itt:
        raise SystemExit(completed.returncode or 4)
    print(
        json.dumps(
            {
                "event": "cell_complete",
                "method": method["key"],
                "condition": condition,
                "episodes": expected,
                "infrastructure_errors": int(status.get("infrastructure_errors", 0)),
            }
        ),
        flush=True,
    )


def run() -> None:
    plan = prepare()
    for method in plan["methods"]:
        for condition in ("nominal", "perturbed"):
            _run_cell(method, condition, plan)


def status() -> dict[str, Any]:
    if not PLAN_PATH.is_file():
        return {"stage": "A6", "experiment": "E5", "prepared": False}
    plan = _read_json(PLAN_PATH)
    cells = {}
    total = 0
    expected_total = 0
    for method in plan["methods"]:
        for condition in ("nominal", "perturbed"):
            expected = int(plan["manifests"][condition]["episodes"])
            output = RESULT_ROOT / "core" / str(method["key"]) / condition / "episodes"
            completed = len(list(output.glob("*.json")))
            cells[f"{method['key']}/{condition}"] = {
                "completed": completed,
                "expected": expected,
            }
            total += completed
            expected_total += expected
    return {
        "stage": "A6",
        "experiment": "E5_core_ablation",
        "prepared": True,
        "completed_episodes": total,
        "expected_episodes": expected_total,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run":
        run()
        result = status()
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
