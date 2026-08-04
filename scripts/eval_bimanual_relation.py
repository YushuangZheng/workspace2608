"""Evaluate inferred bimanual relations under contact-rich handover interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Essay2608-Bimanual-Physical-Handover-v0"
CONDITIONS = (
    "normal",
    "receiver_miss",
    "receiver_delayed",
    "giver_releases_early",
    "receiver_grasps_then_loses",
    "prolonged_both_hold",
    "one_arm_paused",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
parser.add_argument("--seeds", nargs="+", type=int, default=[8300])
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument(
    "--config",
    type=Path,
    default=Path("configs/experiments/bimanual_relation_offline_v1.json"),
)
parser.add_argument(
    "--output_dir", type=Path, default=Path("outputs/bimanual_relation/online_dev_v1")
)
parser.add_argument("--video_unrealized", action="store_true")
parser.add_argument("--video", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker_seed", type=int, help=argparse.SUPPRESS)
parser.add_argument("--worker_condition", choices=CONDITIONS, help=argparse.SUPPRESS)
parser.add_argument("--result_path", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--trace_path", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--video_dir", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "source/essay2608/essay2608/bimanual.py",
        root / "source/essay2608/essay2608/bimanual_physical.py",
        root / "source/essay2608/essay2608/eval/bimanual_relation.py",
        root / "source/essay2608/essay2608/eval/bimanual_relation_study.py",
        root / "source/essay2608/essay2608/policy/relation.py",
        root / "source/essay2608/essay2608/policy/bimanual_relation.py",
        root
        / "source/essay2608/essay2608/tasks/manager_based/bimanual_physical_handover"
        / "physical_handover_env_cfg.py",
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_path() -> Path:
    path = args.config.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"双臂关系配置不存在：{path}")
    return path


def experiment_fingerprint(seed: int, condition: str) -> str:
    payload = {
        "task_id": TASK_ID,
        "seed": int(seed),
        "condition": condition,
        "max_steps": args.max_steps,
        "config_sha256": sha256(config_path()),
        "source_sha256": source_fingerprint(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def trial_stem(condition: str, seed: int) -> str:
    return f"bimanual_relation__{condition}__seed_{seed}"


def worker_command(condition: str, seed: int, *, video: bool) -> list[str]:
    stem = trial_stem(condition, seed)
    output_dir = args.output_dir.resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--headless",
        "--worker_seed",
        str(seed),
        "--worker_condition",
        condition,
        "--max_steps",
        str(args.max_steps),
        "--config",
        str(config_path()),
        "--output_dir",
        str(output_dir),
        "--result_path",
        str(output_dir / "trials" / f"{stem}.json"),
        "--trace_path",
        str(output_dir / "trials" / f"{stem}.npz"),
    ]
    if args.device:
        command.extend(("--device", args.device))
    if video:
        command.extend(
            (
                "--video",
                "--enable_cameras",
                "--video_dir",
                str(output_dir / "videos" / stem),
            )
        )
    return command


def aggregate_condition(rows: list[dict]) -> dict:
    return {
        "num_trials": len(rows),
        "num_condition_realized": sum(row["condition_realization"]["realized"] for row in rows),
        "num_task_successes": sum(row["task_success"] for row in rows),
        "mean_four_value_accuracy": sum(
            row["relation_metrics"]["four_value_accuracy"] for row in rows
        )
        / len(rows),
        "mean_left_f1": sum(row["relation_metrics"]["left"]["f1"] for row in rows)
        / len(rows),
        "mean_right_f1": sum(row["relation_metrics"]["right"]["f1"] for row in rows)
        / len(rows),
    }


def run_controller() -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"拒绝覆盖已有双臂关系评测目录：{output_dir}")
    (output_dir / "trials").mkdir(parents=True)
    rows = []
    total = len(args.conditions) * len(args.seeds)
    index = 0
    for condition in args.conditions:
        for seed in args.seeds:
            index += 1
            print(
                f"[bimanual-relation] trial={index}/{total} "
                f"condition={condition} seed={seed}",
                flush=True,
            )
            completed = subprocess.run(
                worker_command(condition, seed, video=False), check=False
            )
            result_path = output_dir / "trials" / f"{trial_stem(condition, seed)}.json"
            if not result_path.is_file():
                rows.append(
                    {
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
            if not row["condition_realization"]["realized"] and args.video_unrealized:
                subprocess.run(worker_command(condition, seed, video=True), check=False)

    valid = [row for row in rows if not row.get("worker_result_missing")]
    by_condition = {
        condition: aggregate_condition(
            [row for row in valid if row["condition"] == condition]
        )
        for condition in args.conditions
        if any(row["condition"] == condition for row in valid)
    }
    all_realized = len(valid) == total and all(
        row["condition_realization"]["realized"] for row in valid
    )
    summary = {
        "artifact_type": "bimanual_relation_online_development",
        "task_id": TASK_ID,
        "source_sha256": source_fingerprint(),
        "config_path": str(config_path()),
        "config_sha256": sha256(config_path()),
        "conditions": list(args.conditions),
        "seeds": list(args.seeds),
        "num_expected_trials": total,
        "num_valid_trials": len(valid),
        "all_conditions_physically_realized": all_realized,
        "privileged_contact_used_as_estimator_input": False,
        "by_condition": by_condition,
        "trials": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}, ensure_ascii=False, indent=2))
    return 0 if all_realized else 1


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
from essay2608.eval.bimanual_relation import PhysicalRelationTracker  # noqa: E402
from essay2608.eval.bimanual_relation_study import (  # noqa: E402
    BimanualRelationIntervention,
    condition_realization,
    score_bimanual_relation_trace,
)
from essay2608.policy.bimanual_relation import (  # noqa: E402
    BimanualOnlineRelationEstimator,
    BimanualRelationEstimatorConfig,
    BimanualRelationSample,
)
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


def task_outcome(
    *,
    expert: ScriptedPhysicalHandover,
    environment_done: bool,
    truth_labels: list[str],
    object_positions: np.ndarray,
    final_position: np.ndarray,
    target_position: np.ndarray,
) -> tuple[bool, str, dict]:
    settling = object_positions[-25:]
    settling_displacement = (
        float(np.max(np.linalg.norm(settling - settling[-1], axis=-1)))
        if len(settling)
        else float("inf")
    )
    final_xy_error = float(np.linalg.norm(final_position[:2] - target_position[:2]))
    on_support = bool(abs(final_position[2] - target_position[2]) <= 0.025)
    stable = settling_displacement <= 0.01
    expected = ["none", "left_only", "both", "right_only", "none"]
    relation_sequence = value_sequence(truth_labels)
    if expert.failed:
        reason = expert.failure_reason
    elif not expert.complete:
        reason = "expert_incomplete"
    elif environment_done:
        reason = "environment_done"
    elif relation_sequence != expected:
        reason = "physical_relation_lifecycle_incomplete"
    elif not on_support:
        reason = "object_not_on_support"
    elif not stable:
        reason = "object_not_stable"
    elif final_xy_error >= 0.04:
        reason = "placement_xy_above_threshold"
    else:
        reason = "success"
    return reason == "success", reason, {
        "truth_relation_sequence": relation_sequence,
        "final_xy_error_m": final_xy_error,
        "object_on_support": on_support,
        "stable": stable,
        "settling_displacement_m": settling_displacement,
    }


def main_worker() -> None:
    seed = int(args.worker_seed)
    condition = str(args.worker_condition)
    result_path = args.result_path.resolve()
    trace_path = args.trace_path.resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    config_values = json.loads(config_path().read_text(encoding="utf-8"))
    relation_config = BimanualRelationEstimatorConfig.from_dict(config_values)

    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = seed
    render_mode = "rgb_array" if args.video else None
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=render_mode)
    if args.video:
        args.video_dir.resolve().mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.video_dir.resolve()),
            step_trigger=lambda step: step == 0,
            video_length=args.max_steps,
            name_prefix=f"bimanual_relation_{condition}_seed_{seed}",
            disable_logger=True,
        )
    env.reset(seed=seed)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    expert = ScriptedPhysicalHandover(control_dt, env.unwrapped.device)
    truth_tracker = PhysicalRelationTracker()
    estimator = BimanualOnlineRelationEstimator(relation_config)
    intervention = BimanualRelationIntervention(condition, control_dt)
    left_sensor_names = (
        "left_leftfinger_object_contact",
        "left_rightfinger_object_contact",
    )
    right_sensor_names = (
        "right_leftfinger_object_contact",
        "right_rightfinger_object_contact",
    )
    records: dict[str, list] = {
        key: []
        for key in (
            "state",
            "left_ee_pose",
            "right_ee_pose",
            "object_pose",
            "left_finger_force",
            "right_finger_force",
            "left_finger_position",
            "right_finger_position",
            "left_finger_distance_m",
            "right_finger_distance_m",
            "left_finger_velocity_m_s",
            "right_finger_velocity_m_s",
            "base_action",
            "applied_action",
            "truth_left_connected",
            "truth_right_connected",
            "truth_left_confidence",
            "truth_right_confidence",
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
            "intervention_active",
            "intervention_event",
            "phase_clock_held",
        )
    }
    previous_left_distance: float | None = None
    previous_right_distance: float | None = None
    environment_done = False
    for step in range(args.max_steps):
        with torch.inference_mode():
            left, right, object_pose, target = get_handover_poses(env)
            target[:, 2] = 0.181
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
                left_ee_position=row(left)[:3],
                right_ee_position=row(right)[:3],
                object_position=row(object_pose)[:3],
                left_finger_forces=left_force,
                right_finger_forces=right_force,
            )
            inferred = estimator.update(
                BimanualRelationSample(
                    left_ee_pose=row(left),
                    right_ee_pose=row(right),
                    object_pose=row(object_pose),
                    left_finger_distance_m=left_distance,
                    right_finger_distance_m=right_distance,
                    left_finger_velocity_m_s=left_velocity,
                    right_finger_velocity_m_s=right_velocity,
                    control_dt_s=control_dt,
                )
            )
            state = HandoverState(expert.state)
            base_action, finished = expert.compute(left, right, object_pose, target)
            decision = intervention.apply(
                action=base_action,
                state=state,
                truth_label=truth.label,
                right_pose=right,
            )
            if decision.hold_phase_clock and expert.state == state:
                expert.state_time = max(0.0, expert.state_time - control_dt)

            step_values = {
                "state": int(state),
                "left_ee_pose": row(left),
                "right_ee_pose": row(right),
                "object_pose": row(object_pose),
                "left_finger_force": left_force,
                "right_finger_force": right_force,
                "left_finger_position": left_positions,
                "right_finger_position": right_positions,
                "left_finger_distance_m": left_distance,
                "right_finger_distance_m": right_distance,
                "left_finger_velocity_m_s": left_velocity,
                "right_finger_velocity_m_s": right_velocity,
                "base_action": row(base_action),
                "applied_action": row(decision.action),
                "truth_left_connected": truth.left.connected,
                "truth_right_connected": truth.right.connected,
                "truth_left_confidence": truth.left.confidence,
                "truth_right_confidence": truth.right.confidence,
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
                "intervention_active": decision.active,
                "intervention_event": decision.event,
                "phase_clock_held": decision.hold_phase_clock,
            }
            for key, value in step_values.items():
                records[key].append(value)
            _, _, terminated, truncated, _ = env.step(
                actions_to_robot_root_frames(env, decision.action)
            )
            environment_done = bool((terminated | truncated).any().item())
            if step % 100 == 0:
                print(
                    f"[bimanual-relation-worker] step={step} state={expert.state.name} "
                    f"truth={truth.label} inferred={inferred.label} "
                    f"event={decision.event}",
                    flush=True,
                )
            if finished or environment_done:
                break

    _, _, final_object, target = get_handover_poses(env)
    target[:, 2] = 0.181
    string_keys = {
        "truth_label",
        "inferred_label",
        "inferred_left_state",
        "inferred_right_state",
        "intervention_event",
    }
    arrays = {
        key: np.asarray(value, dtype="U32" if key in string_keys else None)
        for key, value in records.items()
    }
    arrays.update(
        {
            "control_dt": np.asarray(control_dt, dtype=np.float32),
            "seed": np.asarray(seed, dtype=np.int64),
            "condition": np.asarray(condition),
            "source_sha256": np.asarray(source_fingerprint()),
            "config_sha256": np.asarray(sha256(config_path())),
        }
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
    realization = condition_realization(
        condition,
        truth_left=arrays["truth_left_connected"],
        truth_right=arrays["truth_right_connected"],
        truth_labels=arrays["truth_label"],
        intervention_active=arrays["intervention_active"],
        intervention_event=arrays["intervention_event"],
        control_dt_s=control_dt,
    )
    final_position = row(final_object)[:3]
    target_position = row(target)[:3]
    task_success, task_failure_reason, task_diagnostics = task_outcome(
        expert=expert,
        environment_done=environment_done,
        truth_labels=list(arrays["truth_label"]),
        object_positions=arrays["object_pose"][:, :3],
        final_position=final_position,
        target_position=target_position,
    )
    np.savez_compressed(trace_path, **arrays)
    result = {
        "artifact_type": "bimanual_relation_online_trial",
        "task_id": TASK_ID,
        "seed": seed,
        "condition": condition,
        "experiment_fingerprint": experiment_fingerprint(seed, condition),
        "source_sha256": source_fingerprint(),
        "config_sha256": sha256(config_path()),
        "dataset_sha256": relation_config.dataset_sha256,
        "steps": len(arrays["truth_label"]),
        "condition_realization": realization,
        "relation_metrics": relation_metrics,
        "inferred_relation_sequence": value_sequence(list(arrays["inferred_label"])),
        "task_success": task_success,
        "task_failure_reason": task_failure_reason,
        "expert_complete": expert.complete,
        "expert_failed": expert.failed,
        **task_diagnostics,
        "video_requested": bool(args.video),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    env.close()
    if not realization["realized"]:
        raise RuntimeError(f"物理条件未按定义实现：{condition}")


if __name__ == "__main__":
    try:
        main_worker()
    finally:
        simulation_app.close()
