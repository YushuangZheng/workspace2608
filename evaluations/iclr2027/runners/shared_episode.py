"""Execute exactly one manifest episode in RLBench.

This module is launched inside the pinned Python 3.8 simulator environment.
The frozen policy runs in its existing Python 3.10 subprocess.  Fault
injection, monitor-visible features, and evaluator-only audit labels are kept
as three separate data paths and joined only in the persisted cycle record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from evaluations.iclr2027.audit.faults import (
    build_fault_environment,
    default_fault_arm,
)
from evaluations.iclr2027.audit.physical_events import PhysicalEventAuditor
from evaluations.iclr2027.interfaces.feature_schema import (
    EPISODE_SCHEMA,
    FeatureRecord,
)
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.registry import (
    MethodSpec,
    build_monitor,
    load_method_spec,
)
from evaluations.iclr2027.recovery.skill_retry import SkillRetry
from evaluations.iclr2027.runners.episode_io import EpisodeWriter
from evaluations.iclr2027.runners.shadow import shadow_observe
from integrations.rlbench.iclr2027.task_registry import (
    TASK_SPECS_PATH,
    experiment_task,
)
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    GLOBAL_IK_CONTROLLER_PROFILE,
    commit_joint_hold_after_primary_failure,
    initialize_fresh_task_generation,
    policy_action_execution_status,
    policy_action_execution_statuses,
    run_final_settling,
    set_policy_gripper_authorization,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_POLICY_PYTHON",
        "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python",
    )
)
SINGLE_MODELS = INTEGRATION_ROOT / "models" / "iclr2027" / "dynamac"
BIMANUAL_MODELS = INTEGRATION_ROOT / "models" / "v4"
CLOSED_LOOP_MODELS = INTEGRATION_ROOT / "models" / "iclr2027" / "closed_loop"
FAULT_CONFIG = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "configs" / "shared" / "faults.json"


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_manifest_row(path: Path, episode_id: str) -> dict:
    matches = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id") == episode_id:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one requested episode")
    return matches[0]


def _task_state(observation: Any) -> np.ndarray:
    value = observation.task_low_dim_state
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _arms(observation: Any, bimanual: bool) -> dict:
    if not bimanual:
        return {
            "single": {
                "ee_pose_xyzw": np.asarray(observation.gripper_pose).tolist(),
                "gripper_open": float(observation.gripper_open),
            }
        }
    return {
        arm: {
            "ee_pose_xyzw": np.asarray(getattr(observation, arm).gripper_pose).tolist(),
            "gripper_open": float(getattr(observation, arm).gripper_open),
        }
        for arm in ("left", "right")
    }


def _compact_policy_state(response: Mapping[str, Any], policy_step: int) -> dict:
    raw = response.get("policy_state")
    result = {
        "policy_step": int(policy_step),
        "reference_state_if_available": None,
        "active_streams_if_available": None,
    }
    if not isinstance(raw, Mapping):
        return result
    monitor = raw.get("monitor")
    if isinstance(monitor, Mapping):
        result["monitor"] = {
            "alarm": bool(monitor.get("alarm", False)),
            "reasons": list(monitor.get("reasons", ())),
            "intents": list(monitor.get("intents", ())),
        }
    if "left" in raw and "right" in raw:
        per_arm = {}
        for arm in ("left", "right"):
            fields = raw.get(arm, {})
            per_arm[arm] = {
                "reference_state": {
                    "skill": fields.get("skill_index"),
                    "progress": fields.get("time_index"),
                    "mode": fields.get("mode"),
                },
                "active_streams": list(fields.get("active_frames", ())),
                "selected_streams": list(fields.get("selected_frames", ())),
                "poe_weights": dict(fields.get("poe_weights", {})),
                "marginal_means": dict(fields.get("marginal_means", {})),
                "marginal_covariances": dict(
                    fields.get("marginal_covariances", {})
                ),
            }
        result["reference_state_if_available"] = {
            arm: per_arm[arm]["reference_state"] for arm in per_arm
        }
        result["active_streams_if_available"] = {
            arm: per_arm[arm]["active_streams"] for arm in per_arm
        }
        result["stream_metadata"] = per_arm
    else:
        result["reference_state_if_available"] = {
            "skill": raw.get("skill_index"),
            "progress": raw.get("time_index"),
            "mode": raw.get("mode"),
        }
        result["active_streams_if_available"] = list(raw.get("active_frames", ()))
        result["stream_metadata"] = {
            "active_streams": list(raw.get("active_frames", ())),
            "selected_streams": list(raw.get("selected_frames", ())),
            "poe_weights": dict(raw.get("poe_weights", {})),
            "marginal_means": dict(raw.get("marginal_means", {})),
            "marginal_covariances": dict(raw.get("marginal_covariances", {})),
        }
    return result


def _policy_process(
    task,
    policy_python: Path,
    method: MethodSpec,
    *,
    diagnostics_dir: Path | None = None,
):
    boundary_root = None
    if method.runtime is not None:
        configured = method.runtime.get("boundary_config_root")
        if configured is not None:
            boundary_root = REPOSITORY_ROOT / str(configured)
    if task.spec.bimanual:
        from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import PolicyProcess

        return PolicyProcess(
            policy_python,
            task.task_id,
            BIMANUAL_MODELS,
            policy_type=method.policy_type,
            closed_loop_models_dir=CLOSED_LOOP_MODELS,
            closed_loop_feature_profile=method.feature_profile or "full",
            closed_loop_boundary_config_root=boundary_root,
            diagnostics_dir=diagnostics_dir,
        )
    from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import PolicyProcess

    return PolicyProcess(
        policy_python,
        task.task_id,
        SINGLE_MODELS,
        policy_type=method.policy_type,
        closed_loop_models_dir=CLOSED_LOOP_MODELS,
        closed_loop_feature_profile=method.feature_profile or "full",
        task_specs_path=TASK_SPECS_PATH,
        closed_loop_boundary_config_root=boundary_root,
        diagnostics_dir=diagnostics_dir,
    )


def _reference_entries(policy_state: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw = policy_state.get("reference_state_if_available")
    if not isinstance(raw, Mapping):
        return {}
    if "skill" in raw and "progress" in raw:
        return {
            "single": {
                "skill": int(raw["skill"]),
                "progress": int(raw["progress"]),
            }
        }
    result = {}
    for arm, state in raw.items():
        if not isinstance(state, Mapping):
            continue
        if state.get("skill") is None or state.get("progress") is None:
            continue
        result[str(arm)] = {
            "skill": int(state["skill"]),
            "progress": int(state["progress"]),
        }
    return result


def _environment(task):
    import rlbench.environment as environment_module
    from rlbench.observation_config import ObservationConfig

    configuration = ObservationConfig()
    configuration.set_all(False)
    configuration.gripper_open = True
    configuration.gripper_pose = True
    configuration.task_low_dim_state = True
    if task.spec.bimanual:
        from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
            _controller_config,
            _make_action_mode,
        )

        environment = environment_module.Environment(
            action_mode=_make_action_mode(
                GLOBAL_IK_CONTROLLER_PROFILE,
                _controller_config(GLOBAL_IK_CONTROLLER_PROFILE),
            ),
            obs_config=configuration,
            headless=True,
            robot_setup="dual_panda",
        )
        return environment, (lambda: None)
    from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
        _controller_config,
        _make_action_mode,
        _prepare_low_dim_headless_scene,
    )

    environment = environment_module.Environment(
        action_mode=_make_action_mode(
            GLOBAL_IK_CONTROLLER_PROFILE,
            _controller_config(GLOBAL_IK_CONTROLLER_PROFILE),
        ),
        obs_config=configuration,
        headless=True,
    )
    restore, _metadata = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=True,
        camera_observations_requested=False,
    )
    return environment, restore


def _injector_metadata(environment: Any) -> Optional[dict]:
    getter = getattr(environment, "protocol_metadata", None)
    return dict(getter()) if callable(getter) else None


def run_episode(
    row: Mapping[str, Any],
    output_root: Path,
    *,
    policy_python: Path = DEFAULT_POLICY_PYTHON,
    method: str | Path = "m0_dynamac",
    calibration_artifact: Path | None = None,
    policy_diagnostics_dir: Path | None = None,
) -> dict:
    from rlbench.backend.exceptions import InvalidActionError

    writer = EpisodeWriter(Path(output_root), str(row["episode_id"]))
    method_spec = load_method_spec(method)
    calibration = (
        None
        if calibration_artifact is None
        else _load_json(Path(calibration_artifact))
    )
    monitor = build_monitor(
        method_spec,
        calibration=calibration,
        task_id=str(row["task"]),
    )
    recovery_config = method_spec.recovery or {}
    retry = (
        SkillRetry(
            recovery_budget=int(recovery_config["budget_cycles"]),
            maximum_retries=int(recovery_config["maximum_retries"]),
        )
        if recovery_config.get("kind") == "skill_retry"
        else None
    )
    task = experiment_task(str(row["task"]))
    fault_config = _load_json(FAULT_CONFIG)
    environment = None
    restore_scene = lambda: None
    worker = None
    launched = False
    started = time.monotonic()
    cycles = 0
    invalid_actions = 0
    recovery_cycles = 0
    first_alarm_cycle = None
    false_interventions = 0
    retry_start_skills: dict[str, int] | None = None
    previous_violation_active = False
    success = False
    reason = "infrastructure_error"
    generation = None
    policy_identity = None
    final_injector = None
    injector_event_cursor = 0
    audit_summary = {
        "eligible": False,
        "physically_triggered": False,
    }
    try:
        task_class = getattr(
            importlib.import_module(task.spec.module), task.spec.class_name
        )
        environment, restore_scene = _environment(task)
        environment.launch()
        launched = True
        task_environment, _descriptions, observation, generation = (
            initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=int(row["seed"]),
                variation=int(row["variation"]),
                verify_instance=False,
            )
        )
        worker = _policy_process(
            task,
            policy_python,
            method_spec,
            diagnostics_dir=policy_diagnostics_dir,
        )
        policy_identity = dict(worker.model_identity)
        worker.request("reset", observation)
        if monitor is not None:
            monitor.reset(
                EpisodeContext(
                    episode_id=str(row["episode_id"]),
                    task_id=task.task_id,
                    method_id=method_spec.method_id,
                    bimanual=task.spec.bimanual,
                    horizon=int(row["horizon"]),
                    feature_schema="essay2608.iclr2027.causal-features.v1",
                    method_config_hash=method_spec.config_sha256,
                )
            )
        wrapped = build_fault_environment(
            task_environment,
            task,
            family=row.get("fault_family"),
            severity=row.get("fault_severity"),
            trigger_stage=row.get("trigger_stage"),
            policy_steps=int(worker.policy_steps),
            config=fault_config,
        )
        earliest_cycle = (
            0
            if row.get("fault_family") in {None, "composed_event"}
            else max(
                0,
                int(
                    round(
                        float(fault_config["trigger_stages"][row["trigger_stage"]])
                        * int(worker.policy_steps)
                    )
                ),
            )
        )
        auditor = PhysicalEventAuditor(
            wrapped,
            task,
            family=row.get("fault_family"),
            target_arm=default_fault_arm(task.task_id, row.get("fault_family") or "none"),
            earliest_cycle=earliest_cycle,
            motion_threshold=float(fault_config["eligibility"]["motion_trigger_distance_m"]),
            effect_tolerance=float(
                fault_config["eligibility"]["physical_displacement_tolerance_m"]
            ),
        )
        previous_resolution = {
            "aggregate": "initial",
            "per_arm": {},
            "primary_action_applied": False,
        }
        for cycle in range(int(row["horizon"])):
            response = worker.request("act", observation)
            action = response.get("action")
            if action is None:
                settling = run_final_settling(
                    wrapped,
                    physics_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
                )
                success = bool(settling["success"])
                reason = (
                    "success_after_final_settling"
                    if success
                    else "policy_complete_without_task_success"
                )
                break
            policy_command = np.asarray(action, dtype=np.float64)
            policy_state = _compact_policy_state(response, cycle)
            entries = _reference_entries(policy_state)
            if retry is not None and entries:
                representative = entries[sorted(entries)[0]]
                if representative["progress"] == 0 and not retry.in_recovery:
                    retry.confirm_skill_entry(representative)
            feature = FeatureRecord(
                episode_id=str(row["episode_id"]),
                cycle=cycle,
                observation_timestamp=cycle,
                action_timestamp=cycle,
                arms=_arms(observation, task.spec.bimanual),
                task_state=tuple(_task_state(observation).tolist()),
                action=tuple(policy_command.tolist()),
                policy_state=policy_state,
                action_resolution=previous_resolution,
            ).to_dict()
            monitor_diagnostic = None
            if monitor is not None:
                monitor_diagnostic = shadow_observe(monitor, feature)
                if monitor_diagnostic["alarm"] and first_alarm_cycle is None:
                    first_alarm_cycle = cycle
            command = policy_command
            retry_decision = None
            if (
                retry is not None
                and monitor_diagnostic is not None
                and monitor_diagnostic["alarm"]
                and not retry.in_recovery
            ):
                retry_decision = retry.request(True)
                if retry_decision.requested:
                    if not previous_violation_active:
                        false_interventions += 1
                    original_transaction = int(response["transaction_id"])
                    worker.request("abort", transaction_id=original_transaction)
                    retried = worker.request("retry_current_skill")
                    retry_start_skills = {
                        arm: int(state["skill"])
                        for arm, state in retried["reference_entries"].items()
                    }
                    response = worker.request("act", observation)
                    if response.get("action") is None:
                        raise RuntimeError("Skill-Retry produced no entry action")
                    command = np.asarray(response["action"], dtype=np.float64)
            if retry is not None and retry.in_recovery:
                recovery_cycles += 1
                retry.consume_cycle()
            auditor.before_step(cycle, observation, command)
            transaction_id = response.get("transaction_id")
            set_policy_gripper_authorization(
                wrapped, response.get("gripper_authorization")
            )
            primary_applied = True
            try:
                observation, reward, terminate = wrapped.step(command)
            except InvalidActionError:
                invalid_actions += 1
                primary_applied = False
                observation, reward, terminate, policy_complete = (
                    commit_joint_hold_after_primary_failure(
                        wrapped,
                        worker,
                        transaction_id=int(transaction_id),
                    )
                )
                notify_fallback = getattr(wrapped, "record_committed_fallback", None)
                if callable(notify_fallback):
                    notify_fallback()
                aggregate = "stopped"
                per_arm = {
                    arm: "stopped" for arm in _arms(observation, task.spec.bimanual)
                }
            else:
                aggregate = policy_action_execution_status(wrapped)
                per_arm = policy_action_execution_statuses(wrapped)
                committed = worker.request(
                    "commit",
                    transaction_id=int(transaction_id),
                    primary_action_status=aggregate,
                    primary_action_statuses=per_arm,
                )
                policy_complete = bool(committed.get("complete"))
            previous_resolution = {
                "aggregate": aggregate,
                "per_arm": per_arm,
                "primary_action_applied": primary_applied,
            }
            injector = _injector_metadata(wrapped)
            if injector is None:
                cycle_injector = None
            else:
                events = list(injector.get("events", ()))
                cycle_injector = {
                    **injector,
                    "events": events[injector_event_cursor:],
                }
                injector_event_cursor = len(events)
                final_injector = injector
            audit = auditor.after_step(cycle, observation, cycle_injector)
            previous_violation_active = bool(audit.get("violation_active", False))
            if retry is not None and retry.in_recovery and retry_start_skills:
                current_entries = _reference_entries(
                    _compact_policy_state(response, cycle)
                )
                if current_entries and all(
                    current_entries.get(arm, {}).get("skill", start) > start
                    for arm, start in retry_start_skills.items()
                ):
                    retry.finish()
                    retry_start_skills = None
            writer.write_cycle(
                feature,
                audit,
                execution={
                    "action_resolution": previous_resolution,
                    "reward": float(reward),
                    "terminate": bool(terminate),
                    "policy_complete": bool(policy_complete),
                    "injector": cycle_injector,
                    "monitor": monitor_diagnostic,
                    "retry": (
                        None
                        if retry_decision is None
                        else {
                            "requested": retry_decision.requested,
                            "reference_state": retry_decision.reference_state,
                            "reason": retry_decision.reason,
                            "remaining_budget": retry_decision.remaining_budget,
                        }
                    ),
                    "applied_action": command.tolist(),
                },
            )
            cycles = cycle + 1
            if float(reward) > 0.0:
                success = True
                reason = "success"
                break
            if bool(terminate):
                reason = "task_terminated"
                break
            if policy_complete:
                settling = run_final_settling(
                    wrapped,
                    physics_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
                )
                success = bool(settling["success"])
                reason = (
                    "success_after_final_settling"
                    if success
                    else "policy_complete_without_task_success"
                )
                break
        else:
            reason = "episode_horizon"
        audit_summary = auditor.summary()
    except Exception as exc:
        reason = "infrastructure_error"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    else:
        error = None
    finally:
        try:
            if worker is not None:
                worker.close()
            if launched and environment is not None:
                environment.shutdown()
        finally:
            restore_scene()
    summary = {
        "schema": EPISODE_SCHEMA,
        "episode_id": str(row["episode_id"]),
        "split": str(row["split"]),
        "task": task.task_id,
        "task_level": row.get("task_level"),
        "variation": int(row["variation"]),
        "seed": int(row["seed"]),
        "condition": str(row["condition"]),
        "fault_family": row.get("fault_family"),
        "fault_severity": row.get("fault_severity"),
        "trigger_stage": row.get("trigger_stage"),
        "method_id": method_spec.method_id,
        "method_config_identity": {
            "path": str(method_spec.config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": method_spec.config_sha256,
            "policy_model": policy_identity,
            "fault_config_sha256": _sha256(FAULT_CONFIG),
            "monitor_calibration": (
                None
                if calibration_artifact is None
                else {
                    "path": str(
                        Path(calibration_artifact)
                        .resolve()
                        .relative_to(REPOSITORY_ROOT)
                    ),
                    "sha256": _sha256(Path(calibration_artifact)),
                }
            ),
        },
        "success": bool(success),
        "final_success": bool(success),
        "reason": reason,
        "termination_reason": reason,
        "cycles": int(cycles),
        "recovery_cycles": int(recovery_cycles),
        "first_alarm_cycle": first_alarm_cycle,
        "false_interventions": int(false_interventions),
        "relation_restored_cycle": audit_summary.get("relation_restored_cycle"),
        "legal_reentry_cycle": audit_summary.get("legal_reentry_cycle"),
        "post_reentry_completion": None,
        "invalid_actions": int(invalid_actions),
        "wall_seconds": time.monotonic() - started,
        "peak_memory_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "audit": audit_summary,
        "fault_protocol": final_injector,
        "fresh_task_generation": generation,
        "error": error,
    }
    writer.finalize(summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--method", default="m0_dynamac")
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--policy-diagnostics-dir", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    row = _load_manifest_row(args.manifest, args.episode_id)
    result = run_episode(
        row,
        args.output_root,
        policy_python=args.policy_python,
        method=args.method,
        calibration_artifact=args.calibration_artifact,
        policy_diagnostics_dir=args.policy_diagnostics_dir,
    )
    return 0 if result["reason"] != "infrastructure_error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
