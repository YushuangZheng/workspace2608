"""Evaluate the contact-rich scripted handover in isolated Isaac Lab workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Essay2608-Bimanual-Physical-Handover-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", nargs="+", type=int, default=[7400])
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument(
    "--output_dir", type=Path, default=Path("outputs/physical_handover/dev_v0")
)
parser.add_argument("--success_xy_threshold", type=float, default=0.04)
parser.add_argument("--minimum_both_duration_s", type=float, default=0.20)
parser.add_argument("--video_failures", action="store_true")
parser.add_argument("--video", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker_seed", type=int, help=argparse.SUPPRESS)
parser.add_argument("--result_path", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--trace_path", type=Path, help=argparse.SUPPRESS)
parser.add_argument("--video_dir", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "source/essay2608/essay2608/bimanual.py",
        root / "source/essay2608/essay2608/bimanual_physical.py",
        root / "source/essay2608/essay2608/eval/bimanual_relation.py",
        root
        / "source/essay2608/essay2608/tasks/manager_based/bimanual_physical_handover"
        / "physical_handover_env_cfg.py",
    ]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def experiment_fingerprint(seed: int) -> str:
    payload = {
        "task_id": TASK_ID,
        "seed": int(seed),
        "max_steps": args.max_steps,
        "success_xy_threshold": args.success_xy_threshold,
        "minimum_both_duration_s": args.minimum_both_duration_s,
        "source_sha256": source_fingerprint(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def worker_command(seed: int, *, video: bool) -> list[str]:
    stem = f"scripted_physical_handover__seed_{seed}"
    output_dir = args.output_dir.resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--headless",
        "--worker_seed",
        str(seed),
        "--max_steps",
        str(args.max_steps),
        "--success_xy_threshold",
        str(args.success_xy_threshold),
        "--minimum_both_duration_s",
        str(args.minimum_both_duration_s),
        "--output_dir",
        str(output_dir),
        "--result_path",
        str(output_dir / "trials" / f"{stem}.json"),
        "--trace_path",
        str(output_dir / "trials" / f"{stem}.npz"),
    ]
    if video:
        command.extend(
            [
                "--video",
                "--enable_cameras",
                "--video_dir",
                str(output_dir / "videos" / stem),
            ]
        )
    return command


def run_controller() -> int:
    output_dir = args.output_dir.resolve()
    (output_dir / "trials").mkdir(parents=True, exist_ok=True)
    results = []
    for index, seed in enumerate(args.seeds, start=1):
        print(f"[physical-study] trial={index}/{len(args.seeds)} seed={seed}", flush=True)
        result = subprocess.run(worker_command(seed, video=False), check=False)
        result_path = output_dir / "trials" / f"scripted_physical_handover__seed_{seed}.json"
        if not result_path.is_file():
            results.append(
                {
                    "seed": seed,
                    "success": False,
                    "failure_reason": "worker_result_missing",
                    "worker_returncode": result.returncode,
                }
            )
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(record)
        if not record["success"] and args.video_failures:
            subprocess.run(worker_command(seed, video=True), check=False)

    successes = sum(bool(result.get("success")) for result in results)
    summary = {
        "task_id": TASK_ID,
        "source_sha256": source_fingerprint(),
        "num_trials": len(results),
        "num_successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "seeds": list(args.seeds),
        "trials": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if successes == len(results) else 1


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
from essay2608.eval.bimanual_relation import PhysicalRelationTracker  # noqa: E402
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
    """Read two single-finger force vectors filtered to the dynamic object."""

    forces = []
    for name in sensor_names:
        matrix = env.unwrapped.scene[name].data.force_matrix_w
        forces.append(matrix[0, 0, 0].detach().cpu().numpy().copy())
    return np.stack(forces)


def main_worker() -> None:
    seed = int(args.worker_seed)
    result_path = args.result_path.resolve()
    trace_path = args.trace_path.resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
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
            name_prefix=f"physical_handover_seed_{seed}",
            disable_logger=True,
        )
    env.reset(seed=seed)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    expert = ScriptedPhysicalHandover(control_dt, env.unwrapped.device)
    tracker = PhysicalRelationTracker()
    left_sensor_names = (
        "left_leftfinger_object_contact",
        "left_rightfinger_object_contact",
    )
    right_sensor_names = (
        "right_leftfinger_object_contact",
        "right_rightfinger_object_contact",
    )
    object_asset = env.unwrapped.scene["object"]

    records: dict[str, list] = {
        key: []
        for key in (
            "state",
            "left_ee_position",
            "left_ee_orientation",
            "right_ee_position",
            "right_ee_orientation",
            "object_position",
            "object_orientation",
            "target_position",
            "object_linear_velocity",
            "action",
            "left_finger_force",
            "right_finger_force",
            "left_finger_position",
            "right_finger_position",
            "left_connected",
            "right_connected",
            "left_confidence",
            "right_confidence",
            "relation_label",
        )
    }
    environment_done = False
    for step in range(args.max_steps):
        with torch.inference_mode():
            left, right, object_pose, target = get_handover_poses(env)
            target[:, 2] = 0.181
            left_force = object_contact_forces(env, left_sensor_names)
            right_force = object_contact_forces(env, right_sensor_names)
            left_finger_position = np.stack(
                [row(env.unwrapped.scene[name].data.pos_w) for name in left_sensor_names]
            )
            right_finger_position = np.stack(
                [row(env.unwrapped.scene[name].data.pos_w) for name in right_sensor_names]
            )
            truth = tracker.update(
                left_ee_position=row(left)[:3],
                right_ee_position=row(right)[:3],
                object_position=row(object_pose)[:3],
                left_finger_forces=left_force,
                right_finger_forces=right_force,
            )
            state = int(expert.state)
            action, finished = expert.compute(left, right, object_pose, target)
            records["state"].append(state)
            records["left_ee_position"].append(row(left)[:3])
            records["left_ee_orientation"].append(row(left)[3:7])
            records["right_ee_position"].append(row(right)[:3])
            records["right_ee_orientation"].append(row(right)[3:7])
            records["object_position"].append(row(object_pose)[:3])
            records["object_orientation"].append(row(object_pose)[3:7])
            records["target_position"].append(row(target)[:3])
            records["object_linear_velocity"].append(row(object_asset.data.root_lin_vel_w))
            records["action"].append(row(action))
            records["left_finger_force"].append(left_force)
            records["right_finger_force"].append(right_force)
            records["left_finger_position"].append(left_finger_position)
            records["right_finger_position"].append(right_finger_position)
            records["left_connected"].append(truth.left.connected)
            records["right_connected"].append(truth.right.connected)
            records["left_confidence"].append(truth.left.confidence)
            records["right_confidence"].append(truth.right.confidence)
            records["relation_label"].append(truth.label)
            _, _, terminated, truncated, _ = env.step(
                actions_to_robot_root_frames(env, action)
            )
            environment_done = bool((terminated | truncated).any().item())
            if step % 100 == 0:
                print(
                    f"[physical-worker] step={step} state={expert.state.name} "
                    f"object={row(object_pose)[:3].round(4).tolist()} relation={truth.label} "
                    f"left_force={np.linalg.norm(left_force, axis=-1).round(2).tolist()} "
                    f"right_force={np.linalg.norm(right_force, axis=-1).round(2).tolist()}",
                    flush=True,
                )
            if finished or environment_done:
                break

    left, right, final_object, target = get_handover_poses(env)
    target[:, 2] = 0.181
    object_positions = np.asarray(records["object_position"], dtype=np.float64)
    labels = list(records["relation_label"])
    relation_sequence = value_sequence(labels)
    both_steps = int(sum(label == "both" for label in labels))
    minimum_both_steps = int(np.ceil(args.minimum_both_duration_s / control_dt))
    settling = object_positions[-25:]
    settling_displacement = (
        float(np.max(np.linalg.norm(settling - settling[-1], axis=-1)))
        if len(settling)
        else float("inf")
    )
    final_position = row(final_object)[:3]
    target_position = row(target)[:3]
    final_xy_error = float(np.linalg.norm(final_position[:2] - target_position[:2]))
    on_support = bool(abs(final_position[2] - target_position[2]) <= 0.025)
    stable = settling_displacement <= 0.01
    expected_sequence = ["none", "left_only", "both", "right_only", "none"]
    if expert.failed:
        failure_reason = expert.failure_reason
    elif not expert.complete:
        failure_reason = "expert_incomplete"
    elif environment_done:
        failure_reason = "environment_done"
    elif relation_sequence != expected_sequence:
        failure_reason = "physical_relation_lifecycle_incomplete"
    elif both_steps < minimum_both_steps:
        failure_reason = "both_hold_too_short"
    elif not on_support:
        failure_reason = "object_not_on_support"
    elif not stable:
        failure_reason = "object_not_stable"
    elif final_xy_error >= args.success_xy_threshold:
        failure_reason = "placement_xy_above_threshold"
    else:
        failure_reason = "success"
    success = failure_reason == "success"

    arrays = {
        key: np.asarray(value, dtype="U16" if key == "relation_label" else None)
        for key, value in records.items()
    }
    arrays["control_dt"] = np.asarray(control_dt, dtype=np.float32)
    arrays["terminal_object_position"] = final_position.astype(np.float32)
    arrays["terminal_target_position"] = target_position.astype(np.float32)
    np.savez_compressed(trace_path, **arrays)
    result = {
        "task_id": TASK_ID,
        "seed": seed,
        "experiment_fingerprint": experiment_fingerprint(seed),
        "source_sha256": source_fingerprint(),
        "success": success,
        "failure_reason": failure_reason,
        "expert_complete": expert.complete,
        "expert_failed": expert.failed,
        "steps": len(labels),
        "relation_sequence": relation_sequence,
        "both_duration_s": both_steps * control_dt,
        "maximum_object_height_m": float(np.max(object_positions[:, 2])),
        "final_object_position_m": final_position.tolist(),
        "final_target_position_m": target_position.tolist(),
        "final_xy_error_m": final_xy_error,
        "object_on_support": on_support,
        "stable": stable,
        "settling_displacement_m": settling_displacement,
        "video_requested": bool(args.video),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    env.close()
    if not success:
        raise RuntimeError(f"物理双臂交接失败：{failure_reason}")


if __name__ == "__main__":
    try:
        main_worker()
    finally:
        simulation_app.close()
