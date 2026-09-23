"""Build and hard-preflight the Horizon-3 boundary assets used by E4 M5.

The frozen M5 method points at the shared normal-task boundary directory.  Its
Main-10 calibration originally contained only ``place_cups_3`` from the eight
Horizon-3 task levels.  This module adds the seven missing task-level runtime
configs without changing any existing Main-10 file or the frozen M5 method
configuration.  Inputs are limited to the same five successful demonstrations
and the already-built frozen action/task models; no fault or formal outcome is
read.

The button-press boundaries need a documented conditional calibration: all
five normal terminal demonstrations discriminate each outgoing boundary by
posterior end mass, but their recorded terminal poses do not always satisfy
the separate fused-action normal-explanation gate.  The runtime gate remains
unchanged and fail-closed; the conditional fit supplies only theta_local/H and
must pass a real worker cycle plus development-only simulator execution before
E4 may launch.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from essay2608.policy.tsf import (
    BeliefUpdaterConfig,
    BoundaryCalibration,
    BoundaryRuntimeConfig,
)
from evaluations.development.boundary_calibration.run import (
    BELIEF_CONFIG_PATH,
    _acceptance_rows,
    _base_runtime_config,
    _calibrate,
    _longest_true_run,
    _read_config,
    _replay_task_demo,
)
from evaluations.iclr2027.calibration.boundary import load_cases
from integrations.rlbench.iclr2027.build_assets import (
    TSF_MODEL_ROOT,
    DATA_ROOT,
    DYNAMAC_ROOT,
)
from integrations.rlbench.iclr2027.task_registry import (
    TASK_SPECS_PATH,
    experiment_task_set,
)
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    demonstration_paths,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
E4_ROOT = EVAL_ROOT / "results" / "controlled" / "e4"
ASSET_ROOT = E4_ROOT / "m5_horizon_assets"
CONFIG = EVAL_ROOT / "configs" / "shared" / "horizon3_normal_task_boundary_calibration.json"
METHOD = EVAL_ROOT / "configs" / "methods" / "m5_full.json"
GENERATED_ROOT = EVAL_ROOT / "artifacts" / "calibration" / "normal_task_boundaries" / "horizon3_v1"
SHARED_RUNTIME_ROOT = EVAL_ROOT / "artifacts" / "calibration" / "normal_task_boundaries" / "main10" / "runtime_configs"
RECORD = ASSET_ROOT / "M5_HORIZON_ASSETS.json"
PREFLIGHT = ASSET_ROOT / "M5_HORIZON_WORKER_PREFLIGHT.json"
SIM_MANIFEST = ASSET_ROOT / "development_preflight_manifest.jsonl"
SIM_RESULTS = EVAL_ROOT / "results" / "development" / "e4_m5_horizon_preflight"
A5_ACCEPTANCE = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "A5_ACCEPTANCE.json"
POLICY_PYTHON = Path("/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python")
TASKS = tuple(task.task_id for task in experiment_task_set("horizon3"))
TASK_LEVELS = {task.task_id: task.task_level for task in experiment_task_set("horizon3")}
FROZEN_MAIN10_OVERLAP = "place_cups_3"
NEW_TASKS = tuple(task for task in TASKS if task != FROZEN_MAIN10_OVERLAP)
CONDITIONAL_TASK = "push_buttons_1"
BASE_SEED = 2_714_000_000


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


def _all_runtime(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit every normal boundary, conditionally when only global support blocks it."""

    belief_config = BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH)
    hold_cycles = math.ceil(
        float(config["terminal_hold_seconds"]) / float(config["control_period_seconds"])
    )
    settling_cycles = math.ceil(
        float(config["terminal_settling_seconds"]) / float(config["control_period_seconds"])
    )
    all_rows: list[dict[str, Any]] = []
    model_tokens: dict[str, set[str]] = {}
    for task in NEW_TASKS:
        cases = list(load_cases(task, 5))
        if len(cases) != 1:
            raise RuntimeError(f"{task} conditional calibration expects one arm")
        model_tokens[task] = {
            boundary.boundary_id.token for boundary in cases[0].model.boundaries.values()
        }
        base_runtime = _base_runtime_config(cases, config)
        for demonstration in config["demonstration_indices"]:
            all_rows.extend(
                _replay_task_demo(
                    task,
                    cases,
                    int(demonstration),
                    belief_config,
                    base_runtime,
                    hold_cycles,
                    include_skill_entry_cycles=True,
                )
            )
    summaries, _standard_trials, standard_runtime = _calibrate(all_rows, config)
    demonstration_indices = tuple(int(v) for v in config["demonstration_indices"])
    conditional_boundaries = []
    calibrations = {
        task: dict(standard_runtime.get(task, {}).get("calibrations", {}))
        for task in NEW_TASKS
    }
    for summary in summaries:
        task = str(summary["task"])
        token = str(summary["boundary"])
        if summary["status"] == "calibrated":
            continue
        if summary["status"] != "blocked_normal_terminal_support":
            raise RuntimeError(f"unsupported normal-boundary failure: {task}/{token}")
        members = [
            row
            for row in all_rows
            if row["task"] == task and row["boundary"] == token
        ]
        stable_hold = [
            row
            for row in members
            if row["phase"] == "terminal_hold"
            and int(row["hold_cycle"]) >= settling_cycles
        ]
        preterminal = [
            row
            for row in members
            if row["phase"] == "recorded"
            and not int(row["truth_in_terminal_window"])
        ]
        positive_trials = {int(row["demonstration"]) for row in stable_hold}
        if not stable_hold or positive_trials != set(demonstration_indices):
            raise RuntimeError(f"{task}/{token} lacks a terminal hold for every demo")
        positive_floor = min(float(row["local_score"]) for row in stable_hold)
        preterminal_ceiling = max(
            (float(row["local_score"]) for row in preterminal), default=0.0
        )
        separated = preterminal_ceiling < positive_floor
        threshold = (
            0.5 * (preterminal_ceiling + positive_floor)
            if separated
            else float(config["positive_floor_fraction"]) * positive_floor
        )
        false_runs = []
        hold_runs = []
        trials = []
        for demonstration in demonstration_indices:
            recorded = sorted(
                (
                    row
                    for row in preterminal
                    if int(row["demonstration"]) == demonstration
                ),
                key=lambda row: int(row["tick"]),
            )
            held = sorted(
                (
                    row
                    for row in members
                    if row["phase"] == "terminal_hold"
                    and int(row["demonstration"]) == demonstration
                ),
                key=lambda row: int(row["hold_cycle"]),
            )
            false_run = _longest_true_run(
                float(row["local_score"]) > threshold for row in recorded
            )
            hold_run = _longest_true_run(
                float(row["local_score"]) > threshold for row in held
            )
            false_runs.append(false_run)
            hold_runs.append(hold_run)
            trials.append(
                {
                    "demonstration": demonstration,
                    "maximum_preterminal_score_run": false_run,
                    "terminal_hold_score_run": hold_run,
                    "terminal_hold_cycles": len(held),
                    "terminal_evidence_available_cycles": sum(
                        int(row["local_evidence_available"]) for row in held
                    ),
                    "terminal_progress_statuses": sorted(
                        {str(row["progress_status"]) for row in held}
                    ),
                }
            )
        confirmation_cycles = max(false_runs) + 1
        if positive_floor <= 0.0 or min(hold_runs) < confirmation_cycles:
            raise RuntimeError(f"{task}/{token} conditional score is not discriminative")
        calibration = BoundaryCalibration(
            local_score_threshold=float(threshold),
            confirmation_cycles=int(confirmation_cycles),
        )
        calibrations[task][token] = {
            "local_score_threshold": calibration.local_score_threshold,
            "confirmation_cycles": calibration.confirmation_cycles,
            "relation_thresholds": {},
        }
        conditional_boundaries.append(
            {
                "task": task,
                "boundary": token,
                "normal_terminal_score_floor": positive_floor,
                "normal_preterminal_score_ceiling": preterminal_ceiling,
                "positive_support_separated": separated,
                "local_score_threshold": threshold,
                "confirmation_cycles": confirmation_cycles,
                "trials": trials,
            }
        )
    runtime_by_task = {}
    for task in NEW_TASKS:
        if set(calibrations[task]) != model_tokens[task]:
            raise RuntimeError(f"{task} runtime config does not cover every model boundary")
        runtime_by_task[task] = BoundaryRuntimeConfig.from_mapping(
            {
                "calibrations": calibrations[task],
                "default_relation_probability": float(config["default_relation_probability"]),
                "minimum_tracking_reliability": float(config["minimum_tracking_reliability"]),
                "minimum_scene_reliability": float(config["minimum_scene_reliability"]),
                "minimum_information_weight": float(config["minimum_information_weight"]),
            }
        ).to_dict()
    # Re-run EntryGuard with the final runtime objects.  Standard boundaries
    # must retain the full phase-four acceptance criterion.  A conditional
    # boundary must remain fail-closed at a recorded pose that lacks the
    # independent normal-explanation evidence; only the local score threshold
    # and persistence were fitted above.
    validation_rows: list[dict[str, Any]] = []
    for task in NEW_TASKS:
        cases = list(load_cases(task, 5))
        final_runtime = BoundaryRuntimeConfig.from_mapping(runtime_by_task[task])
        for demonstration in demonstration_indices:
            validation_rows.extend(
                _replay_task_demo(
                    task,
                    cases,
                    demonstration,
                    belief_config,
                    final_runtime,
                    hold_cycles,
                    include_skill_entry_cycles=True,
                )
            )
    acceptance = _acceptance_rows(validation_rows)
    conditional_keys = {
        (str(row["task"]), str(row["boundary"])) for row in conditional_boundaries
    }
    standard_failures = [
        row
        for row in acceptance
        if (str(row["task"]), str(row["boundary"])) not in conditional_keys
        and not bool(row["accepted"])
    ]
    if standard_failures:
        raise RuntimeError(f"standard Horizon boundary acceptance failed: {standard_failures[:5]}")
    conditional_fail_open = [
        row
        for row in acceptance
        if (str(row["task"]), str(row["boundary"])) in conditional_keys
        and int(row["premature_preterminal_permits"]) > 0
    ]
    if conditional_fail_open:
        raise RuntimeError(f"conditional Horizon boundary permitted prematurely: {conditional_fail_open[:5]}")
    audit = {
        "schema": "essay2608.iclr2027.e4-m5-conditional-boundary-calibration.v1",
        "tasks": list(NEW_TASKS),
        "source": "same_five_successful_demonstrations",
        "runtime_explanation_gate_changed": False,
        "formal_or_fault_results_read": False,
        "reason": (
            "The normal-demo posterior end mass is discriminative, while the separately "
            "frozen fused-action normal-explanation gate is not satisfied at the recorded "
            "terminal pose. Only theta_local/H are fitted here; runtime still requires a "
            "plausible state and therefore remains fail-closed."
        ),
        "standard_calibrated_boundaries": [
            {"task": row["task"], "boundary": row["boundary"]}
            for row in summaries
            if row["status"] == "calibrated"
        ],
        "conditional_boundaries": conditional_boundaries,
        "runtime_acceptance": {
            "standard_rows": sum(
                (str(row["task"]), str(row["boundary"])) not in conditional_keys
                for row in acceptance
            ),
            "standard_rows_accepted": sum(
                (str(row["task"]), str(row["boundary"])) not in conditional_keys
                and bool(row["accepted"])
                for row in acceptance
            ),
            "conditional_rows": sum(
                (str(row["task"]), str(row["boundary"])) in conditional_keys
                for row in acceptance
            ),
            "conditional_premature_permits": sum(
                int(row["premature_preterminal_permits"])
                for row in acceptance
                if (str(row["task"]), str(row["boundary"])) in conditional_keys
            ),
        },
    }
    return runtime_by_task, audit


def calibrate() -> dict[str, Any]:
    config = _read_config(CONFIG)
    if tuple(config["tasks"]) != TASKS:
        raise RuntimeError("Horizon boundary config does not cover the frozen task order")
    if GENERATED_ROOT.exists():
        raise FileExistsError(f"generated Horizon boundary root already exists: {GENERATED_ROOT}")
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_ROOT.mkdir(parents=True)
    (GENERATED_ROOT / "runtime_configs").mkdir()
    shutil.copy2(CONFIG, GENERATED_ROOT / "config.json")
    runtime_by_task, conditional_audit = _all_runtime(config)
    for task, runtime in runtime_by_task.items():
        _atomic_json(GENERATED_ROOT / "runtime_configs" / f"{task}.json", runtime)
    conditional_audit_path = GENERATED_ROOT / "conditional_boundary_calibration.json"
    _atomic_json(conditional_audit_path, conditional_audit)

    installed = {}
    for task in TASKS:
        target = SHARED_RUNTIME_ROOT / f"{task}.json"
        if task == FROZEN_MAIN10_OVERLAP:
            if not target.is_file():
                raise RuntimeError("frozen Main-10 place_cups_3 boundary config is missing")
            installed[task] = {
                "source": "frozen_main10",
                "generated_path": None,
                "generated_sha256": None,
                "installed_path": str(target.relative_to(ROOT)),
                "installed_sha256": _sha256(target),
            }
            continue
        generated = GENERATED_ROOT / "runtime_configs" / f"{task}.json"
        if target.is_file():
            if _json(target) != _json(generated):
                raise RuntimeError(f"refusing to overwrite an existing boundary config: {task}")
        else:
            shutil.copy2(generated, target)
        installed[task] = {
            "source": "horizon_success_demo_calibration",
            "generated_path": str(generated.relative_to(ROOT)),
            "generated_sha256": _sha256(generated),
            "installed_path": str(target.relative_to(ROOT)),
            "installed_sha256": _sha256(target),
        }
    model_files = {}
    for task in TASKS:
        files = {}
        for root in (TSF_MODEL_ROOT / task, DYNAMAC_ROOT / task):
            for path in sorted(root.glob("*")):
                if path.is_file():
                    files[str(path.relative_to(ROOT))] = _sha256(path)
        model_files[task] = files
    record = {
        "schema": "essay2608.iclr2027.e4-m5-horizon-assets.v1",
        "status": "CALIBRATED_AWAITING_PREFLIGHT",
        "tasks": list(TASKS),
        "method_config": {
            "path": str(METHOD.relative_to(ROOT)),
            "sha256": _sha256(METHOD),
            "changed": False,
        },
        "calibration_config": {
            "path": str(CONFIG.relative_to(ROOT)),
            "sha256": _sha256(CONFIG),
            "demonstrations_per_task": 5,
            "global_rules_match_main10": True,
        },
        "runtime_configs": installed,
        "model_files": model_files,
        "conditional_calibration": {
            "tasks": sorted({row["task"] for row in conditional_audit["conditional_boundaries"]}),
            "audit_path": str(conditional_audit_path.relative_to(ROOT)),
            "audit_sha256": _sha256(conditional_audit_path),
            "runtime_explanation_gate_changed": False,
        },
        "existing_main10_files_overwritten": False,
        "frozen_main10_overlap_reused": FROZEN_MAIN10_OVERLAP,
        "failure_trajectories_read": False,
        "formal_results_read": False,
    }
    files = sorted(
        path
        for path in GENERATED_ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (GENERATED_ROOT / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(GENERATED_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    _atomic_json(RECORD, record)
    return record


def worker_preflight() -> dict[str, Any]:
    """Start each real policy worker and execute reset/act/abort once."""

    record = _json(RECORD)
    rows = []
    from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import PolicyProcess

    for task in TASKS:
        expected = record["runtime_configs"][task]
        installed = ROOT / expected["installed_path"]
        if not installed.is_file() or _sha256(installed) != expected["installed_sha256"]:
            raise RuntimeError(f"installed Horizon M5 boundary identity changed: {task}")
        episode = load_low_dim_obs_pickles(demonstration_paths(DATA_ROOT, task, 1))[0]
        observation = list(episode)[0]
        worker = PolicyProcess(
            POLICY_PYTHON,
            task,
            DYNAMAC_ROOT,
            policy_type="task_state_feedback",
            tsf_models_dir=TSF_MODEL_ROOT,
            tsf_feature_profile="full",
            task_specs_path=TASK_SPECS_PATH,
            tsf_boundaries_config_root=SHARED_RUNTIME_ROOT,
        )
        try:
            reset = worker.request("reset", observation=observation)
            act = worker.request("act", observation=observation)
            action = act.get("action")
            if not isinstance(action, list) or not action:
                raise RuntimeError(f"M5 preflight produced no action: {task}")
            worker.request("abort", transaction_id=act["transaction_id"])
            identity = worker.model_identity.get("boundary_runtime_config", {})
            if identity.get("sha256") != expected["installed_sha256"]:
                raise RuntimeError(f"worker loaded a different boundary config: {task}")
            rows.append(
                {
                    "task": task,
                    "ready": True,
                    "policy_steps": worker.policy_steps,
                    "reset_complete": bool(reset.get("complete", False)),
                    "action_dimensions": len(action),
                    "boundary_config_sha256": identity.get("sha256"),
                }
            )
        finally:
            worker.close()
    value = {
        "schema": "essay2608.iclr2027.e4-m5-horizon-worker-preflight.v1",
        "status": "PASS",
        "checks": [
            "all_task_assets_exist_and_match_frozen_hashes",
            "real_policy_worker_ping",
            "demonstration_observation_reset",
            "one_real_policy_act_transaction",
            "transaction_abort",
        ],
        "tasks": rows,
    }
    _atomic_json(PREFLIGHT, value)
    return value


def simulator_preflight() -> dict[str, Any]:
    """Run one non-formal nominal simulator episode per task."""

    worker_preflight()
    rows = []
    for index, task in enumerate(TASKS):
        episode_id = f"e4_m5_horizon_preflight/{task}/0000"
        rows.append(
            {
                "schema": "essay2608.iclr2027.episode-manifest.v1",
                "episode_id": episode_id,
                "split": "e4_m5_horizon_preflight",
                "task": task,
                "task_level": TASK_LEVELS[task],
                "variation": 0,
                "seed": BASE_SEED + index * 100_000,
                "condition": "nominal",
                "fault_family": None,
                "fault_severity": None,
                "trigger_stage": None,
                "pair_id": episode_id,
                "horizon": 1000,
                "recovery_budget": 400,
            }
        )
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if SIM_MANIFEST.is_file() and SIM_MANIFEST.read_text(encoding="utf-8") != payload:
        raise RuntimeError("existing M5 simulator-preflight manifest changed")
    if not SIM_MANIFEST.is_file():
        _atomic_jsonl(SIM_MANIFEST, rows)
    command = [
        sys.executable,
        "-m",
        "evaluations.iclr2027.runners.launch",
        "--manifest",
        str(SIM_MANIFEST),
        "--output-root",
        str(SIM_RESULTS),
        "--workers",
        "8",
        "--episode-timeout-seconds",
        "900",
        "--retry-infrastructure",
        "0",
        "--method",
        str(METHOD),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    episode_paths = sorted((SIM_RESULTS / "episodes").glob("*.json"))
    episodes = [_json(path) for path in episode_paths]
    by_task = {str(row["task"]): row for row in episodes}
    if completed.returncode not in (0, 2) or set(by_task) != set(TASKS):
        raise RuntimeError("M5 Horizon simulator preflight is incomplete")
    failures = {
        task: row.get("reason")
        for task, row in by_task.items()
        if task != FROZEN_MAIN10_OVERLAP
        and (int(row.get("cycles", 0)) <= 0 or row.get("reason") == "infrastructure_error")
    }
    if failures:
        raise RuntimeError(f"M5 Horizon simulator preflight failed: {failures}")
    overlap = by_task[FROZEN_MAIN10_OVERLAP]
    overlap_reuse = False
    if int(overlap.get("cycles", 0)) <= 0 or overlap.get("reason") == "infrastructure_error":
        acceptance = _json(A5_ACCEPTANCE)
        if not str(acceptance.get("status", "")).startswith("PASS"):
            raise RuntimeError("frozen Main-10 overlap lacks an accepted A5 endpoint")
        overlap_reuse = True
    push = by_task[CONDITIONAL_TASK]
    if not bool(push.get("success", False)):
        raise RuntimeError("push_buttons_1 conditional boundary failed its development episode")
    simulator = {
        "schema": "essay2608.iclr2027.e4-m5-horizon-simulator-preflight.v1",
        "status": "PASS",
        "manifest": str(SIM_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(SIM_MANIFEST),
        "formal_results_read": False,
        "frozen_a5_acceptance_read_for_overlap_only": overlap_reuse,
        "frozen_main10_overlap": {
            "task": FROZEN_MAIN10_OVERLAP,
            "development_result": overlap.get("reason"),
            "development_error": overlap.get("error"),
            "disposition": (
                "reused_frozen_A5_endpoint_acceptance_after_development_X11_timeout"
                if overlap_reuse
                else "development_simulator_preflight_completed"
            ),
        },
        "episodes": {
            task: {
                "success": bool(row.get("success", False)),
                "cycles": int(row.get("cycles", 0)),
                "reason": row.get("reason"),
            }
            for task, row in sorted(by_task.items())
        },
    }
    simulator_path = ASSET_ROOT / "M5_HORIZON_SIMULATOR_PREFLIGHT.json"
    _atomic_json(simulator_path, simulator)
    record = _json(RECORD)
    record["status"] = "PASS"
    record["worker_preflight"] = {
        "path": str(PREFLIGHT.relative_to(ROOT)),
        "sha256": _sha256(PREFLIGHT),
    }
    record["simulator_preflight"] = {
        "path": str(simulator_path.relative_to(ROOT)),
        "sha256": _sha256(simulator_path),
    }
    _atomic_json(RECORD, record)
    return simulator


def status() -> dict[str, Any]:
    return {
        "record": None if not RECORD.is_file() else _json(RECORD),
        "worker_preflight": None if not PREFLIGHT.is_file() else _json(PREFLIGHT),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "worker-preflight", "sim-preflight", "all", "status"))
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        value = calibrate()
    elif args.command == "worker-preflight":
        value = worker_preflight()
    elif args.command == "sim-preflight":
        value = simulator_preflight()
    elif args.command == "all":
        calibrate()
        value = simulator_preflight()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
