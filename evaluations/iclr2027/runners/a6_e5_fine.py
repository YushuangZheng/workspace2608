"""Freeze and run the seven appendix-level E5 ablations on Stress-4.

Fine ablations run only after E5 core and E4 are complete.  This keeps every
experiment that reuses the byte-frozen A5 endpoint ahead of the additive
fine-ablation switches, which are installed once and exercised in one batch.
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
    assert_frozen_m5_current,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULT_ROOT = EVAL_ROOT / "results" / "controlled" / "e5"
VIEW_ROOT = RESULT_ROOT / "manifest_views"
PLAN_PATH = RESULT_ROOT / "A6_E5_FINE_RUN_PLAN.json"
A5_PLAN = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_RUN_PLAN.json"
CURRENT_M5_FREEZE = (
    EVAL_ROOT
    / "results"
    / "a6_execution"
    / "M5_PENDING_DIRECT_IDENTITY_FREEZE.json"
)
CURRENT_M5_IMPACT = (
    EVAL_ROOT / "results" / "a6_execution" / "M5_POST_E4_IMPACT_SCOPE.json"
)
E5_CORE = RESULT_ROOT / "derived" / "E5_CORE_ANALYSIS.json"
E4_ANALYSIS = EVAL_ROOT / "results" / "controlled" / "e4" / "derived" / "E4_ANALYSIS.json"
NOMINAL = EVAL_ROOT / "manifests" / "main10_nominal.jsonl"
PERTURBED = EVAL_ROOT / "manifests" / "ablation4.jsonl"
DEVELOPMENT = EVAL_ROOT / "manifests" / "main10_development.jsonl"
PREFLIGHT_ROOT = (
    EVAL_ROOT
    / "results"
    / "development"
    / "e5_fine_preflight_pending_direct_identity"
)
PREFLIGHT_MANIFEST = PREFLIGHT_ROOT / "stress4_one_per_task.jsonl"
PREFLIGHT_RECORD = PREFLIGHT_ROOT / "E5_FINE_PREFLIGHT.json"
STRESS4 = (
    "open_drawer",
    "place_cups_3",
    "bimanual_handover_item",
    "bimanual_lift_tray",
)
METHODS = {
    "no_relation_evidence": EVAL_ROOT / "configs" / "methods" / "ablation_no_relation_evidence.json",
    "no_scene_evidence": EVAL_ROOT / "configs" / "methods" / "ablation_no_scene_evidence.json",
    "static_stream_roles": EVAL_ROOT / "configs" / "methods" / "ablation_static_stream_roles.json",
    "no_boundary_guards": EVAL_ROOT / "configs" / "methods" / "ablation_no_boundary_guards.json",
    "no_active_verification": EVAL_ROOT / "configs" / "methods" / "ablation_no_active_verification.json",
    "retry_same_state": EVAL_ROOT / "configs" / "methods" / "ablation_retry_same_state.json",
    "no_control_equivalence": EVAL_ROOT / "configs" / "methods" / "ablation_no_control_equivalence.json",
}
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


def _require_pass(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not str(value.get("status", "")).startswith("PASS"):
        raise RuntimeError(f"required predecessor is not PASS: {path}")
    return value


def _implementation_identity() -> dict[str, Any]:
    paths = sorted((ROOT / "source" / "policy" / "tsf").rglob("*.py"))
    entries = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in paths
    ]
    aggregate = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in entries).encode()
    ).hexdigest()
    return {"entries": entries, "aggregate_sha256": aggregate}


def _ensure_development_preflight(methods: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise every fine profile before a sealed run plan can exist.

    Configuration parsing and unit tests are necessary but do not prove that a
    real simulator episode can load the task assets, start the policy worker,
    and emit actions.  Use one immutable development row for each Stress-4
    task and require every profile to produce at least one non-infrastructure,
    non-zero-cycle result.  A fixed-wall-clock ``EpisodeProcessTimeout`` is an
    admissible development outcome after the same profile has emitted actions
    on another task: removing a mechanism can itself create a long tail, and
    the formal ITT run must measure rather than pre-screen that behavior.  No
    sealed manifest or outcome is read here.
    """

    first_by_task: dict[str, dict[str, Any]] = {}
    for row in _rows(DEVELOPMENT):
        task = str(row["task"])
        if task in STRESS4 and task not in first_by_task:
            first_by_task[task] = row
    if set(first_by_task) != set(STRESS4):
        raise RuntimeError("E5 fine preflight lacks a development row for Stress-4")
    development_rows = [first_by_task[task] for task in STRESS4]
    _atomic_jsonl(PREFLIGHT_MANIFEST, development_rows)

    expected_identity = {
        "development_manifest_sha256": _sha256(PREFLIGHT_MANIFEST),
        "implementation_aggregate_sha256": _implementation_identity()[
            "aggregate_sha256"
        ],
        "methods": {str(row["key"]): str(row["sha256"]) for row in methods},
        "current_m5_freeze_sha256": _sha256(CURRENT_M5_FREEZE),
        "current_m5_impact_sha256": _sha256(CURRENT_M5_IMPACT),
    }
    if PREFLIGHT_RECORD.is_file():
        record = _require_pass(PREFLIGHT_RECORD)
        if record.get("identity") != expected_identity:
            raise RuntimeError("E5 fine preflight identity changed after validation")
        return record

    checked: list[dict[str, Any]] = []
    for method in methods:
        key = str(method["key"])
        output = PREFLIGHT_ROOT / key
        command = [
            sys.executable,
            "-m",
            "evaluations.iclr2027.runners.launch",
            "--manifest",
            str(PREFLIGHT_MANIFEST),
            "--output-root",
            str(output),
            "--workers",
            "4",
            "--episode-timeout-seconds",
            str(TIMEOUT_SECONDS),
            "--retry-infrastructure",
            "0",
            "--method",
            str(ROOT / str(method["path"])),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        queue_path = output / "QUEUE_STATUS.json"
        queue = _read_json(queue_path) if queue_path.is_file() else {}
        if (
            completed.returncode not in (0, 2)
            or int(queue.get("completed_episode_count", -1)) != len(development_rows)
            or int(queue.get("missing_selected_count", -1)) != 0
        ):
            raise RuntimeError(
                f"E5 fine development preflight failed for {key}:\n"
                + completed.stdout[-4000:]
            )
        profile_nonzero_results = 0
        profile_wall_clock_timeouts = 0
        for row in development_rows:
            episode_path = (
                output
                / "episodes"
                / (str(row["episode_id"]).replace("/", "__") + ".json")
            )
            result = _read_json(episode_path)
            observed = result.get("method_config_identity", {})
            if observed.get("sha256") != method["sha256"]:
                raise RuntimeError(
                    f"E5 fine development preflight method mismatch: {key}"
                )
            is_timeout = (
                result.get("termination_reason") == "infrastructure_error"
                and (result.get("error") or {}).get("type")
                == "EpisodeProcessTimeout"
                and float(result.get("wall_seconds", 0.0)) >= TIMEOUT_SECONDS
            )
            if is_timeout:
                profile_wall_clock_timeouts += 1
            elif result.get("termination_reason") == "infrastructure_error":
                raise RuntimeError(
                    f"E5 fine development preflight infrastructure failure: "
                    f"{key} {row['task']} {result.get('error')}"
                )
            elif int(result.get("cycles", 0)) <= 0:
                raise RuntimeError(
                    f"E5 fine development preflight produced zero cycles: {key} "
                    f"{row['task']}"
                )
            else:
                profile_nonzero_results += 1
            checked.append(
                {
                    "method": key,
                    "task": row["task"],
                    "episode_id": row["episode_id"],
                    "cycles": int(result["cycles"]),
                    "termination_reason": result.get("termination_reason"),
                    "admissible_wall_clock_timeout": is_timeout,
                }
            )
        if profile_nonzero_results == 0:
            raise RuntimeError(
                f"E5 fine development preflight emitted no completed simulator "
                f"episode for {key}"
            )
    record = {
        "schema": "essay2608.iclr2027.a6-e5-fine-preflight.v1",
        "status": "PASS",
        "sealed_test_read": False,
        "identity": expected_identity,
        "checks": checked,
        "requirements": {
            "profiles": len(methods),
            "tasks_per_profile": len(development_rows),
            "real_simulator_episode": True,
            "nonzero_cycles": True,
            "non_timeout_infrastructure_errors": 0,
            "fixed_wall_clock_timeouts_are_formal_itt_outcomes": True,
        },
    }
    _atomic_json(PREFLIGHT_RECORD, record)
    return record


def prepare() -> dict[str, Any]:
    freeze = assert_frozen_m5_current()
    core = _require_pass(E5_CORE)
    horizon = _require_pass(E4_ANALYSIS)
    from source.policy.tsf.ablation import TSFFeatureProfile

    available_profiles = set(TSFFeatureProfile.names())
    nominal_by_task = {task: [] for task in STRESS4}
    for row in _rows(NOMINAL):
        task = str(row["task"])
        if task in nominal_by_task and len(nominal_by_task[task]) < 100:
            nominal_by_task[task].append(row)
    nominal_rows = [row for task in STRESS4 for row in nominal_by_task[task]]
    nominal_counts = Counter(str(row["task"]) for row in nominal_rows)
    if len(nominal_rows) != 400 or nominal_counts != Counter({task: 100 for task in STRESS4}):
        raise RuntimeError("E5 fine nominal view must be Stress-4 x 100")
    perturbed_rows = _rows(PERTURBED)
    perturbed_counts = Counter(str(row["task"]) for row in perturbed_rows)
    if len(perturbed_rows) != 800 or perturbed_counts != Counter({task: 200 for task in STRESS4}):
        raise RuntimeError("E5 fine perturbed view must be Stress-4 x 200")
    nominal_view = VIEW_ROOT / "stress4_nominal.jsonl"
    _atomic_jsonl(nominal_view, nominal_rows)

    methods = []
    for key, path in METHODS.items():
        config = _read_json(path)
        expected_id = "ablation_" + key
        if config.get("method_id") != expected_id:
            raise RuntimeError(f"fine method identity mismatch: {path}")
        profile = str(config.get("feature_profile"))
        if profile not in available_profiles:
            raise RuntimeError(f"fine profile is not implemented: {profile}")
        methods.append(
            {
                "key": key,
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "feature_profile": profile,
                "paper_name": config["paper_name"],
            }
        )
    preflight = _ensure_development_preflight(methods)

    plan = {
        "schema": "essay2608.iclr2027.a6-e5-fine-run-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "A6",
        "experiment": "E5_fine_ablation",
        "sealed_test": True,
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
        "one_episode_per_job": True,
        "dynamic_global_queue": True,
        "predecessors": {
            "e5_core": {"path": str(E5_CORE.relative_to(ROOT)), "sha256": _sha256(E5_CORE), "status": core["status"]},
            "e4": {"path": str(E4_ANALYSIS.relative_to(ROOT)), "sha256": _sha256(E4_ANALYSIS), "status": horizon["status"]},
        },
        "a5_run_plan_reference": {
            "path": str(A5_PLAN.relative_to(ROOT)),
            "sha256": _sha256(A5_PLAN),
        },
        "current_m5_freeze": {
            "path": str(CURRENT_M5_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(CURRENT_M5_FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
        },
        "current_m5_impact": {
            "path": str(CURRENT_M5_IMPACT.relative_to(ROOT)),
            "sha256": _sha256(CURRENT_M5_IMPACT),
            "selection_rule": _read_json(CURRENT_M5_IMPACT)["selection_rule"],
        },
        "fine_implementation_identity": _implementation_identity(),
        "development_preflight": {
            "path": str(PREFLIGHT_RECORD.relative_to(ROOT)),
            "sha256": _sha256(PREFLIGHT_RECORD),
            "status": preflight["status"],
            "sealed_test_read": preflight["sealed_test_read"],
        },
        "manifests": {
            "nominal": {"path": str(nominal_view.relative_to(ROOT)), "sha256": _sha256(nominal_view), "episodes": 400},
            "perturbed": {"path": str(PERTURBED.relative_to(ROOT)), "sha256": _sha256(PERTURBED), "episodes": 800},
        },
        "methods": methods,
        # ``retry_same_state`` is executed once in E2 for Figure 3(c) and is
        # reused here byte-for-byte.  The seven-variant appendix contains
        # 8,400 evaluated rows, but only the other six variants add new
        # physical episodes at this point in the mainline.
        "expected_new_episodes": 6 * (400 + 800),
        "expected_evaluated_episodes": 7 * (400 + 800),
        "reused_from_e2": {
            "method_key": "retry_same_state",
            "episodes": 400 + 800,
            "result_root": str(
                (RESULT_ROOT / "fine" / "retry_same_state").relative_to(ROOT)
            ),
        },
    }
    if PLAN_PATH.exists():
        previous = _read_json(PLAN_PATH)
        comparable = dict(plan)
        old_comparable = dict(previous)
        comparable.pop("created_utc", None)
        old_comparable.pop("created_utc", None)
        if comparable != old_comparable:
            raise RuntimeError("existing E5 fine run plan disagrees with current inputs")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def _assert_plan_identity(plan: Mapping[str, Any]) -> None:
    assert_frozen_m5_current()
    if _implementation_identity() != plan["fine_implementation_identity"]:
        raise RuntimeError("fine-ablation implementation changed after plan freeze")
    for method in plan["methods"]:
        if _sha256(ROOT / str(method["path"])) != method["sha256"]:
            raise RuntimeError(f"fine-ablation config changed: {method['path']}")


def _run_cell(method: Mapping[str, Any], condition: str, plan: Mapping[str, Any]) -> None:
    expected = int(plan["manifests"][condition]["episodes"])
    command = [
        sys.executable,
        "-m",
        "evaluations.iclr2027.runners.launch",
        "--manifest",
        str(ROOT / str(plan["manifests"][condition]["path"])),
        "--output-root",
        str(RESULT_ROOT / "fine" / str(method["key"]) / condition),
        "--workers",
        str(WORKERS),
        "--episode-timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--retry-infrastructure",
        "1",
        "--method",
        str(ROOT / str(method["path"])),
    ]
    print(json.dumps({"event": "cell_start", "method": method["key"], "condition": condition, "episodes": expected}), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    status_path = RESULT_ROOT / "fine" / str(method["key"]) / condition / "QUEUE_STATUS.json"
    queue = _read_json(status_path) if status_path.is_file() else {}
    complete_itt = int(queue.get("completed_episode_count", -1)) == expected and int(queue.get("missing_selected_count", -1)) == 0
    if completed.returncode not in (0, 2) or not complete_itt:
        raise SystemExit(completed.returncode or 4)
    print(json.dumps({"event": "cell_complete", "method": method["key"], "condition": condition, "episodes": expected, "infrastructure_errors": int(queue.get("infrastructure_errors", 0))}), flush=True)


def run() -> None:
    plan = prepare()
    _assert_plan_identity(plan)
    for method in plan["methods"]:
        # This exact cell is the E2 middle system.  Its output root, method
        # identity and manifests are shared deliberately; never launch it a
        # second time during E5.
        if str(method["key"]) == "retry_same_state":
            for condition in ("nominal", "perturbed"):
                expected = int(plan["manifests"][condition]["episodes"])
                completed = len(
                    list(
                        (
                            RESULT_ROOT
                            / "fine"
                            / "retry_same_state"
                            / condition
                            / "episodes"
                        ).glob("*.json")
                    )
                )
                if completed != expected:
                    raise RuntimeError(
                        "E2 retry_same_state reuse is incomplete: "
                        f"{condition} {completed}/{expected}"
                    )
            continue
        for condition in ("nominal", "perturbed"):
            _run_cell(method, condition, plan)


def status() -> dict[str, Any]:
    if not PLAN_PATH.is_file():
        return {"stage": "A6", "experiment": "E5_fine_ablation", "prepared": False}
    plan = _read_json(PLAN_PATH)
    cells: dict[str, Any] = {}
    total = 0
    expected_total = 0
    for method in plan["methods"]:
        for condition in ("nominal", "perturbed"):
            expected = int(plan["manifests"][condition]["episodes"])
            completed = len(list((RESULT_ROOT / "fine" / str(method["key"]) / condition / "episodes").glob("*.json")))
            cells[f"{method['key']}/{condition}"] = {"completed": completed, "expected": expected}
            total += completed
            expected_total += expected
    return {
        "stage": "A6",
        "experiment": "E5_fine_ablation",
        "prepared": True,
        "completed_evaluated_episodes": total,
        "expected_evaluated_episodes": expected_total,
        "expected_new_episodes": int(plan["expected_new_episodes"]),
        "reused_from_e2_episodes": int(plan["reused_from_e2"]["episodes"]),
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
