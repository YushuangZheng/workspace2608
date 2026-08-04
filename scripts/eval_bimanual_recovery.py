"""Evaluate relation-gated receiver recovery in physical bimanual handover."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Essay2608-Bimanual-Physical-Handover-v0"
METHODS = (
    "clocked_expert",
    "relation_gate",
    "relation_recovery",
    "oracle_relation_recovery",
)
CONDITIONS = (
    "normal",
    "receiver_miss_once",
    "receiver_brief_loss",
    "receiver_loss_once",
    "receiver_loss_after_release",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
parser.add_argument("--seeds", nargs="+", type=int, default=[8600])
parser.add_argument("--max_steps", type=int, default=2200)
parser.add_argument("--run_kind", choices=("development", "formal"), default="development")
parser.add_argument(
    "--relation_config",
    type=Path,
    default=Path("configs/experiments/bimanual_relation_offline_v1.json"),
)
parser.add_argument("--recovery_config", type=Path)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("outputs/bimanual_recovery/dev_v1"),
)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker_method", choices=METHODS, help=argparse.SUPPRESS)
parser.add_argument("--worker_condition", choices=CONDITIONS, help=argparse.SUPPRESS)
parser.add_argument("--worker_seed", type=int, help=argparse.SUPPRESS)
parser.add_argument("--result_path", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--trace_path", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_fingerprint() -> str:
    root = repository_root()
    sources = [
        Path(__file__).resolve(),
        root / "source/essay2608/essay2608/bimanual.py",
        root / "source/essay2608/essay2608/bimanual_physical.py",
        root / "source/essay2608/essay2608/eval/bimanual_relation.py",
        root / "source/essay2608/essay2608/eval/bimanual_relation_study.py",
        root / "source/essay2608/essay2608/eval/bimanual_recovery_study.py",
        root / "source/essay2608/essay2608/policy/relation.py",
        root / "source/essay2608/essay2608/policy/bimanual_relation.py",
        root / "source/essay2608/essay2608/policy/bimanual_recovery.py",
        root
        / "source/essay2608/essay2608/tasks/manager_based/bimanual_physical_handover"
        / "physical_handover_env_cfg.py",
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def relation_config_path() -> Path:
    path = args.relation_config.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"双臂关系配置不存在：{path}")
    return path


def recovery_config_values() -> dict:
    from essay2608.policy.bimanual_recovery import BimanualRecoveryConfig

    if args.recovery_config is None:
        return BimanualRecoveryConfig().as_dict()
    path = args.recovery_config.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"双臂恢复配置不存在：{path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    return BimanualRecoveryConfig(**values).as_dict()


def recovery_config_sha256() -> str:
    payload = json.dumps(
        recovery_config_values(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def experiment_fingerprint(method: str, condition: str, seed: int) -> str:
    payload = {
        "task_id": TASK_ID,
        "method": method,
        "condition": condition,
        "seed": int(seed),
        "max_steps": int(args.max_steps),
        "source_sha256": source_fingerprint(),
        "relation_config_sha256": sha256(relation_config_path()),
        "recovery_config_sha256": recovery_config_sha256(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def trial_stem(method: str, condition: str, seed: int) -> str:
    return f"bimanual_recovery__{method}__{condition}__seed_{seed}"


def worker_command(method: str, condition: str, seed: int) -> list[str]:
    output_dir = args.output_dir.resolve()
    stem = trial_stem(method, condition, seed)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--headless",
        "--worker_method",
        method,
        "--worker_condition",
        condition,
        "--worker_seed",
        str(seed),
        "--max_steps",
        str(args.max_steps),
        "--relation_config",
        str(relation_config_path()),
        "--output_dir",
        str(output_dir),
        "--result_path",
        str(output_dir / "trials" / f"{stem}.json"),
        "--trace_path",
        str(output_dir / "trials" / f"{stem}.npz"),
    ]
    if args.recovery_config is not None:
        command.extend(("--recovery_config", str(args.recovery_config.resolve())))
    if args.device:
        command.extend(("--device", args.device))
    return command


def aggregate_rows(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "num_trials": count,
        "num_faults_realized": sum(row["fault_realization"]["realized"] for row in rows),
        "num_task_successes": sum(row["task_success"] for row in rows),
        "num_safe_releases": sum(row["metrics"]["safe_release"] for row in rows),
        "num_unsafe_releases": sum(row["metrics"]["unsafe_release"] for row in rows),
        "num_recovery_triggers": sum(row["metrics"]["recovery_triggered"] for row in rows),
        "num_geometry_recoveries": sum(
            row["metrics"]["geometry_recovery_executed"] for row in rows
        ),
        "num_transient_loss_cancellations": sum(
            row["metrics"]["transient_loss_cancelled"] for row in rows
        ),
        "num_recovery_successes": sum(
            row["metrics"]["recovery_success"] is True for row in rows
        ),
        "num_false_recovery_triggers": sum(
            row["metrics"]["false_recovery_trigger"] for row in rows
        ),
        "mean_final_xy_error_m": (
            sum(row["final_xy_error_m"] for row in rows) / count if count else None
        ),
        "mean_four_value_accuracy": (
            sum(row["relation_metrics"]["four_value_accuracy"] for row in rows) / count
            if count
            else None
        ),
    }


def run_controller() -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"拒绝覆盖已有双臂恢复评测目录：{output_dir}")
    (output_dir / "trials").mkdir(parents=True)
    rows = []
    total = len(args.methods) * len(args.conditions) * len(args.seeds)
    index = 0
    for method in args.methods:
        for condition in args.conditions:
            for seed in args.seeds:
                index += 1
                print(
                    f"[bimanual-recovery] trial={index}/{total} method={method} "
                    f"condition={condition} seed={seed}",
                    flush=True,
                )
                completed = subprocess.run(worker_command(method, condition, seed), check=False)
                result_path = output_dir / "trials" / f"{trial_stem(method, condition, seed)}.json"
                if not result_path.is_file():
                    rows.append(
                        {
                            "method": method,
                            "condition": condition,
                            "seed": seed,
                            "worker_result_missing": True,
                            "worker_returncode": completed.returncode,
                        }
                    )
                    continue
                row = json.loads(result_path.read_text(encoding="utf-8"))
                row["worker_returncode"] = completed.returncode
                rows.append(row)

    valid = [row for row in rows if not row.get("worker_result_missing")]
    by_method_condition = {
        method: {
            condition: aggregate_rows(
                [
                    row
                    for row in valid
                    if row["method"] == method and row["condition"] == condition
                ]
            )
            for condition in args.conditions
        }
        for method in args.methods
    }
    all_faults_realized = len(valid) == total and all(
        row["fault_realization"]["realized"] for row in valid
    )
    summary = {
        "artifact_type": f"bimanual_relation_recovery_{args.run_kind}",
        "task_id": TASK_ID,
        "source_sha256": source_fingerprint(),
        "relation_config_sha256": sha256(relation_config_path()),
        "recovery_config": recovery_config_values(),
        "recovery_config_sha256": recovery_config_sha256(),
        "dataset_sha256": (
            valid[0]["dataset_sha256"] if valid else None
        ),
        "maximum_steps": int(args.max_steps),
        "control_dt_s": (
            valid[0]["control_dt_s"] if valid else None
        ),
        "methods": list(args.methods),
        "conditions": list(args.conditions),
        "seeds": list(args.seeds),
        "num_expected_trials": total,
        "num_valid_trials": len(valid),
        "all_faults_physically_realized": all_faults_realized,
        "by_method_condition": by_method_condition,
        "trials": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "trials"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all_faults_realized else 1


if not args.worker:
    raise SystemExit(run_controller())


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import essay2608  # noqa: F401, E402
from essay2608.bimanual import actions_to_robot_root_frames, get_handover_poses  # noqa: E402
from essay2608.bimanual_physical import ScriptedPhysicalHandover  # noqa: E402
from essay2608.data.handover_schema import HandoverState  # noqa: E402
from essay2608.eval.bimanual_recovery_study import (  # noqa: E402
    BimanualRecoveryIntervention,
    fault_realization,
    score_bimanual_recovery_trace,
    task_outcome_from_trace,
)
from essay2608.eval.bimanual_relation import PhysicalRelationTracker  # noqa: E402
from essay2608.eval.bimanual_relation_study import score_bimanual_relation_trace  # noqa: E402
from essay2608.policy.bimanual_recovery import (  # noqa: E402
    BimanualRecoveryConfig,
    BimanualRelationRecoveryController,
)
from essay2608.policy.bimanual_relation import (  # noqa: E402
    BimanualOnlineRelationEstimator,
    BimanualRelationEstimatorConfig,
    BimanualRelationSample,
)
from essay2608.policy.relation import RelationState  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def row(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().copy()


def value_sequence(values: list[str]) -> list[str]:
    sequence: list[str] = []
    for value in values:
        if not sequence or value != sequence[-1]:
            sequence.append(value)
    return sequence


def object_contact_forces(env, sensor_names: tuple[str, str]) -> np.ndarray:
    forces = []
    for name in sensor_names:
        matrix = env.unwrapped.scene[name].data.force_matrix_w
        forces.append(matrix[0, 0, 0].detach().cpu().numpy().copy())
    return np.stack(forces)


def finger_positions(env, sensor_names: tuple[str, str]) -> np.ndarray:
    positions = np.stack([row(env.unwrapped.scene[name].data.pos_w) for name in sensor_names])
    return positions.reshape(2, -1, 3)[:, 0]


def finger_distance(positions: np.ndarray) -> float:
    if positions.shape != (2, 3):
        raise ValueError(f"指体位置形状错误：{positions.shape}")
    return float(np.linalg.norm(positions[0] - positions[1]))


def relation_state_from_truth(connected: bool) -> RelationState:
    return RelationState.CONNECTED if connected else RelationState.DISCONNECTED


def rebase_expert_after_recovery(
    *,
    expert: ScriptedPhysicalHandover,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    object_pose: torch.Tensor,
    target_pose: torch.Tensor,
    requires_giver_connection: bool,
) -> str:
    """Resume the frozen expert from measured poses without changing its source."""

    from isaaclab.utils import math as math_utils

    expert.state_time = 0.0
    expert.grasp_close_time = None
    expert.right_object_offset = math_utils.subtract_frame_transforms(
        right_pose[:, :3],
        right_pose[:, 3:7],
        object_pose[:, :3],
        object_pose[:, 3:7],
    )
    if requires_giver_connection:
        expert.state = HandoverState.TRANSFER
        expert.left_hold_pose = left_pose.clone()
        expert.right_hold_pose = right_pose.clone()
        return "resume_shared_transfer_from_measured_pose"

    expert.state = HandoverState.RIGHT_TO_TARGET
    expert.right_transport_start_pose = right_pose.clone()
    expert.right_transport_goal = expert._hand_pose_for_object_position(
        target_pose[:, :3],
        expert.right_orientation,
        expert.right_object_offset,
    )
    return "resume_receiver_transport_from_measured_pose"


def main_worker() -> None:
    method = str(args.worker_method)
    condition = str(args.worker_condition)
    seed = int(args.worker_seed)
    result_path = args.result_path.resolve()
    trace_path = args.trace_path.resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    relation_values = json.loads(relation_config_path().read_text(encoding="utf-8"))
    relation_config = BimanualRelationEstimatorConfig.from_dict(relation_values)
    recovery_values = recovery_config_values()
    recovery_config = BimanualRecoveryConfig(**recovery_values)
    if method == "relation_gate":
        recovery_config = replace(recovery_config, enable_recovery=False)
    controller = (
        None
        if method == "clocked_expert"
        else BimanualRelationRecoveryController(recovery_config)
    )

    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=seed)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    expert = ScriptedPhysicalHandover(control_dt, env.unwrapped.device)
    truth_tracker = PhysicalRelationTracker()
    estimator = BimanualOnlineRelationEstimator(relation_config)
    intervention = BimanualRecoveryIntervention(condition, control_dt)
    left_sensor_names = (
        "left_leftfinger_object_contact",
        "left_rightfinger_object_contact",
    )
    right_sensor_names = (
        "right_leftfinger_object_contact",
        "right_rightfinger_object_contact",
    )
    keys = (
        "state",
        "left_ee_pose",
        "right_ee_pose",
        "object_pose",
        "target_pose",
        "left_finger_force",
        "right_finger_force",
        "left_finger_position",
        "right_finger_position",
        "left_finger_distance_m",
        "right_finger_distance_m",
        "left_finger_velocity_m_s",
        "right_finger_velocity_m_s",
        "base_action",
        "supervised_action",
        "applied_action",
        "truth_left_connected",
        "truth_right_connected",
        "truth_label",
        "inferred_left_connected",
        "inferred_right_connected",
        "inferred_left_confidence",
        "inferred_right_confidence",
        "inferred_left_connection_score",
        "inferred_right_connection_score",
        "inferred_left_loss_score",
        "inferred_right_loss_score",
        "inferred_left_state",
        "inferred_right_state",
        "inferred_label",
        "control_left_state",
        "control_right_state",
        "recovery_state",
        "recovery_trigger",
        "recovery_transition",
        "recovery_requires_giver",
        "expert_rebase_event",
        "recovery_action_overridden",
        "transfer_gate_active",
        "regrasp_attempts",
        "phase_clock_held",
        "intervention_active",
        "intervention_event",
    )
    records: dict[str, list] = {key: [] for key in keys}
    previous_left_distance: float | None = None
    previous_right_distance: float | None = None
    environment_done = False

    for step in range(args.max_steps):
        with torch.inference_mode():
            left, right, object_pose, target = get_handover_poses(env)
            target[:, 2] = 0.181
            left_np = row(left)
            right_np = row(right)
            object_np = row(object_pose)
            target_np = row(target)
            left_force = object_contact_forces(env, left_sensor_names)
            right_force = object_contact_forces(env, right_sensor_names)
            left_positions = finger_positions(env, left_sensor_names)
            right_positions = finger_positions(env, right_sensor_names)
            left_distance = finger_distance(left_positions)
            right_distance = finger_distance(right_positions)
            left_velocity = (
                0.0
                if previous_left_distance is None
                else (left_distance - previous_left_distance) / control_dt
            )
            right_velocity = (
                0.0
                if previous_right_distance is None
                else (right_distance - previous_right_distance) / control_dt
            )
            previous_left_distance = left_distance
            previous_right_distance = right_distance
            truth = truth_tracker.update(
                left_ee_position=left_np[:3],
                right_ee_position=right_np[:3],
                object_position=object_np[:3],
                left_finger_forces=left_force,
                right_finger_forces=right_force,
            )
            inferred = estimator.update(
                BimanualRelationSample(
                    left_ee_pose=left_np,
                    right_ee_pose=right_np,
                    object_pose=object_np,
                    left_finger_distance_m=left_distance,
                    right_finger_distance_m=right_distance,
                    left_finger_velocity_m_s=left_velocity,
                    right_finger_velocity_m_s=right_velocity,
                    control_dt_s=control_dt,
                )
            )

            state_before = HandoverState(expert.state)
            expert_snapshot = copy.deepcopy(expert.__dict__)
            base_tensor, finished = expert.compute(left, right, object_pose, target)
            base_action = row(base_tensor)
            resume_rebase = False
            if controller is None:
                supervised_action = base_action.copy()
                recovery_state = "NOT_APPLICABLE"
                recovery_trigger = "NONE"
                recovery_transition = "none"
                requires_giver = False
                expert_rebase_event = "none"
                action_overridden = False
                gate_active = False
                regrasp_attempts = 0
                hold_clock = False
                recovery_failed = False
            else:
                use_oracle = method == "oracle_relation_recovery"
                left_state = (
                    relation_state_from_truth(truth.left.connected)
                    if use_oracle
                    else inferred.left.state
                )
                right_state = (
                    relation_state_from_truth(truth.right.connected)
                    if use_oracle
                    else inferred.right.state
                )
                control_left_state = left_state.value
                control_right_state = right_state.value
                recovery_decision = controller.update(
                    task_state=state_before,
                    normal_action=base_action,
                    left_pose=left_np,
                    right_pose=right_np,
                    object_pose=object_np,
                    right_gripper_opening_m=right_distance,
                    left_relation_state=left_state,
                    right_relation_state=right_state,
                )
                supervised_action = recovery_decision.action
                recovery_state = recovery_decision.state.value
                recovery_trigger = recovery_decision.trigger.value
                recovery_transition = recovery_decision.transition or "none"
                requires_giver = recovery_decision.requires_giver_connection
                expert_rebase_event = "none"
                action_overridden = recovery_decision.action_overridden
                gate_active = recovery_decision.transfer_gate_active
                regrasp_attempts = recovery_decision.regrasp_attempts
                hold_clock = recovery_decision.pause_task_clock
                recovery_failed = controller.failed

                resume_rebase = recovery_decision.state.value == "RESUME_TASK"

            if controller is None:
                control_left_state = "NOT_APPLICABLE"
                control_right_state = "NOT_APPLICABLE"

            if hold_clock:
                if HandoverState(expert.state) != state_before:
                    expert.__dict__.clear()
                    expert.__dict__.update(expert_snapshot)
                    finished = False
                else:
                    expert.state_time = max(0.0, expert.state_time - control_dt)
            if resume_rebase:
                expert_rebase_event = rebase_expert_after_recovery(
                    expert=expert,
                    left_pose=left,
                    right_pose=right,
                    object_pose=object_pose,
                    target_pose=target,
                    requires_giver_connection=requires_giver,
                )
                finished = False

            fault = intervention.apply(
                action=supervised_action,
                task_state=state_before,
                right_pose=right_np,
            )
            applied_action = fault.action
            values = {
                "state": int(state_before),
                "left_ee_pose": left_np,
                "right_ee_pose": right_np,
                "object_pose": object_np,
                "target_pose": target_np,
                "left_finger_force": left_force,
                "right_finger_force": right_force,
                "left_finger_position": left_positions,
                "right_finger_position": right_positions,
                "left_finger_distance_m": left_distance,
                "right_finger_distance_m": right_distance,
                "left_finger_velocity_m_s": left_velocity,
                "right_finger_velocity_m_s": right_velocity,
                "base_action": base_action,
                "supervised_action": supervised_action,
                "applied_action": applied_action,
                "truth_left_connected": truth.left.connected,
                "truth_right_connected": truth.right.connected,
                "truth_label": truth.label,
                "inferred_left_connected": inferred.left_connected,
                "inferred_right_connected": inferred.right_connected,
                "inferred_left_confidence": inferred.left.confidence,
                "inferred_right_confidence": inferred.right.confidence,
                "inferred_left_connection_score": inferred.left.connection_score,
                "inferred_right_connection_score": inferred.right.connection_score,
                "inferred_left_loss_score": inferred.left.loss_score,
                "inferred_right_loss_score": inferred.right.loss_score,
                "inferred_left_state": inferred.left.state.value,
                "inferred_right_state": inferred.right.state.value,
                "inferred_label": inferred.label,
                "control_left_state": control_left_state,
                "control_right_state": control_right_state,
                "recovery_state": recovery_state,
                "recovery_trigger": recovery_trigger,
                "recovery_transition": recovery_transition,
                "recovery_requires_giver": requires_giver,
                "expert_rebase_event": expert_rebase_event,
                "recovery_action_overridden": action_overridden,
                "transfer_gate_active": gate_active,
                "regrasp_attempts": regrasp_attempts,
                "phase_clock_held": hold_clock,
                "intervention_active": fault.active,
                "intervention_event": fault.event,
            }
            for key, value in values.items():
                records[key].append(value)

            action_tensor = torch.as_tensor(
                applied_action,
                dtype=torch.float32,
                device=env.unwrapped.device,
            ).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(
                actions_to_robot_root_frames(env, action_tensor)
            )
            environment_done = bool((terminated | truncated).any().item())
            if step % 100 == 0:
                print(
                    f"[bimanual-recovery-worker] step={step} state={expert.state.name} "
                    f"truth={truth.label} inferred={inferred.label} "
                    f"recovery={recovery_state} fault={fault.event}",
                    flush=True,
                )
            if finished or environment_done or recovery_failed:
                break

    final_left, final_right, final_object, target = get_handover_poses(env)
    target[:, 2] = 0.181
    string_keys = {
        "truth_label",
        "inferred_left_state",
        "inferred_right_state",
        "inferred_label",
        "control_left_state",
        "control_right_state",
        "recovery_state",
        "recovery_trigger",
        "recovery_transition",
        "expert_rebase_event",
        "intervention_event",
    }
    arrays = {
        key: np.asarray(value, dtype="U64" if key in string_keys else None)
        for key, value in records.items()
    }
    arrays.update(
        {
            "control_dt": np.asarray(control_dt, dtype=np.float32),
            "method": np.asarray(method),
            "condition": np.asarray(condition),
            "seed": np.asarray(seed, dtype=np.int64),
            "source_sha256": np.asarray(source_fingerprint()),
            "relation_config_sha256": np.asarray(sha256(relation_config_path())),
            "recovery_config_sha256": np.asarray(recovery_config_sha256()),
            "dataset_sha256": np.asarray(relation_config.dataset_sha256),
            "maximum_steps": np.asarray(args.max_steps, dtype=np.int64),
            "privileged_relation_used_for_control": np.asarray(
                method == "oracle_relation_recovery",
                dtype=bool,
            ),
        }
    )
    final_position = row(final_object)[:3]
    target_position = row(target)[:3]
    recovery_failed = bool(controller is not None and controller.failed)
    arrays.update(
        {
            "terminal_left_pose": row(final_left),
            "terminal_right_pose": row(final_right),
            "terminal_object_pose": row(final_object),
            "terminal_target_pose": row(target),
            "expert_complete": np.asarray(expert.complete, dtype=bool),
            "expert_failed": np.asarray(expert.failed, dtype=bool),
            "expert_failure_reason": np.asarray(expert.failure_reason or "none"),
            "recovery_failed": np.asarray(recovery_failed, dtype=bool),
            "environment_done": np.asarray(environment_done, dtype=bool),
        }
    )
    task_success, failure_reason, outcome = task_outcome_from_trace(
        expert_complete=expert.complete,
        expert_failed=expert.failed,
        expert_failure_reason=expert.failure_reason,
        recovery_failed=recovery_failed,
        environment_done=environment_done,
        object_positions=arrays["object_pose"][:, :3],
        final_position=final_position,
        target_position=target_position,
    )
    metrics = score_bimanual_recovery_trace(
        arrays,
        condition,
        control_dt,
        method=method,
        task_success=task_success,
    )
    relation_metrics = score_bimanual_relation_trace(
        truth_labels=arrays["truth_label"],
        inferred_labels=arrays["inferred_label"],
        truth_left=arrays["truth_left_connected"],
        truth_right=arrays["truth_right_connected"],
        inferred_left=arrays["inferred_left_connected"],
        inferred_right=arrays["inferred_right_connected"],
        control_dt_s=control_dt,
    )
    realization = fault_realization(condition, arrays, control_dt)
    np.savez_compressed(trace_path, **arrays)
    result = {
        "artifact_type": "bimanual_relation_recovery_trial",
        "task_id": TASK_ID,
        "method": method,
        "condition": condition,
        "seed": seed,
        "experiment_fingerprint": experiment_fingerprint(method, condition, seed),
        "source_sha256": source_fingerprint(),
        "relation_config_sha256": sha256(relation_config_path()),
        "recovery_config_sha256": recovery_config_sha256(),
        "dataset_sha256": relation_config.dataset_sha256,
        "privileged_relation_used_for_control": method == "oracle_relation_recovery",
        "privileged_relation_used_by_online_methods": False,
        "control_dt_s": control_dt,
        "maximum_steps": int(args.max_steps),
        "steps": len(arrays["state"]),
        "fault_realization": realization,
        "task_success": task_success,
        "task_failure_reason": failure_reason,
        "expert_complete": expert.complete,
        "expert_failed": expert.failed,
        "recovery_failed": recovery_failed,
        "truth_relation_sequence": value_sequence(list(arrays["truth_label"])),
        "inferred_relation_sequence": value_sequence(list(arrays["inferred_label"])),
        "relation_metrics": relation_metrics,
        "metrics": metrics,
        **outcome,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main_worker()
    finally:
        simulation_app.close()
