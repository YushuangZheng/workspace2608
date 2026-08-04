"""Evaluate bimanual cross-arm policies under execution-time perturbations."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher


METHODS = ("independent_arms", "fixed_handover", "static_cross_arm", "full_dynamac")
CONDITIONS = (
    "static",
    "left_offset",
    "right_offset",
    "handover_shift",
    "left_pause",
    "right_pause",
    "smooth_left",
    "sudden_right",
)
TASK_ID = "Essay2608-Bimanual-Handover-v0"

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/handover_static/v1"))
parser.add_argument("--output_dir", type=Path, default=Path("outputs/handover_minimal/v1"))
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
parser.add_argument("--seeds", nargs="+", type=int, default=[8208])
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--success_threshold", type=float, default=0.06)
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker_method", choices=METHODS, help=argparse.SUPPRESS)
parser.add_argument("--worker_condition", choices=CONDITIONS, help=argparse.SUPPRESS)
parser.add_argument("--worker_seed", type=int, help=argparse.SUPPRESS)
parser.add_argument("--result_path", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float | None]:
    if not trials:
        return [None, None]
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
    return [max(0.0, centre - margin / denominator), min(1.0, centre + margin / denominator)]


def aggregate(trials: list[dict]) -> dict:
    import numpy as np

    result = {}
    for method in args.methods:
        result[method] = {}
        for condition in args.conditions:
            selected = [
                trial["metrics"]
                for trial in trials
                if trial.get("method") == method
                and trial.get("condition") == condition
                and "metrics" in trial
            ]
            successes = sum(bool(item["success"]) for item in selected)
            result[method][condition] = {
                "num_trials": len(selected),
                "success_rate": successes / len(selected) if selected else None,
                "success_rate_ci95_wilson": wilson_interval(successes, len(selected)),
                "mean_final_error_m": float(np.mean([item["final_error_m"] for item in selected]))
                if selected
                else None,
                "mean_coordination_error_m": float(
                    np.mean([item["mean_handover_distance_m"] for item in selected])
                )
                if selected
                else None,
                "mean_path_length_m": float(np.mean([item["path_length_m"] for item in selected]))
                if selected
                else None,
                "mean_inference_ms": float(np.mean([item["mean_inference_ms"] for item in selected]))
                if selected
                else None,
                "failure_reasons": dict(Counter(item["failure_reason"] for item in selected)),
            }
    return result


def run_controller() -> int:
    trial_dir = args.output_dir.resolve() / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    total = len(args.methods) * len(args.conditions) * len(args.seeds)
    index = 0
    for seed in args.seeds:
        for condition in args.conditions:
            for method in args.methods:
                index += 1
                path = trial_dir / f"{method}__{condition}__seed_{seed}.json"
                if args.resume and path.is_file():
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if "metrics" in cached:
                        print(f"[handover-eval] reuse {index}/{total} {method} {condition} {seed}", flush=True)
                        trials.append(cached)
                        continue
                print(f"[handover-eval] trial {index}/{total} {method} {condition} {seed}", flush=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker",
                    "--headless",
                    "--worker_method",
                    method,
                    "--worker_condition",
                    condition,
                    "--worker_seed",
                    str(seed),
                    "--result_path",
                    str(path),
                ]
                completed = subprocess.run(command, check=False)
                if completed.returncode or not path.is_file():
                    trials.append(
                        {"method": method, "condition": condition, "seed": seed, "returncode": completed.returncode}
                    )
                else:
                    trials.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "task_id": TASK_ID,
        "dataset_dir": str(args.data_dir.resolve()),
        "methods": args.methods,
        "conditions": args.conditions,
        "seeds": args.seeds,
        "success_threshold_m": args.success_threshold,
        "ablation": aggregate(trials),
        "trials": trials,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["ablation"], indent=2), flush=True)
    return int(any("metrics" not in trial for trial in trials))


if not args.worker:
    raise SystemExit(run_controller())


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import math as math_utils

import essay2608
from essay2608.bimanual import actions_to_robot_root_frames, get_handover_poses, write_attached_object
from essay2608.data.dataset import load_bimanual_dataset
from essay2608.policy.bimanual import BimanualGaussianPolicy, BimanualPolicyObservation
from isaaclab_tasks.utils import parse_env_cfg


def read_observation(env: gym.Env) -> tuple[BimanualPolicyObservation, tuple[torch.Tensor, ...]]:
    poses = get_handover_poses(env)
    arrays = [pose[0].detach().cpu().numpy().astype(np.float64) for pose in poses]
    return BimanualPolicyObservation(*arrays), poses


def perturb_action(
    action: np.ndarray,
    condition: str,
    phase: int,
    phase_step: int,
    observation: BimanualPolicyObservation,
) -> tuple[np.ndarray, bool]:
    result = action.copy()
    active = False
    if condition == "left_offset" and phase == 4 and 20 <= phase_step < 80:
        result[:3] += np.asarray([0.0, 0.10, 0.04])
        active = True
    elif condition == "right_offset" and phase == 6 and 15 <= phase_step < 65:
        result[8:11] += np.asarray([-0.08, -0.08, 0.04])
        active = True
    elif condition == "handover_shift" and 4 <= phase <= 8:
        result[:3] += np.asarray([0.06, 0.07, 0.03])
        active = True
    elif condition == "left_pause" and phase == 4 and phase_step < 60:
        result[:7] = observation.left_ee_pose
        active = True
    elif condition == "right_pause" and phase == 5 and phase_step < 60:
        result[8:15] = observation.right_ee_pose
        active = True
    elif condition == "smooth_left" and phase == 4:
        progress = min(phase_step / 100.0, 1.0)
        result[:3] += np.asarray([0.0, 0.08 * math.sin(math.pi * progress), 0.03 * math.sin(math.pi * progress)])
        active = True
    elif condition == "sudden_right" and phase == 6 and 20 <= phase_step < 60:
        result[8:11] += np.asarray([-0.10, 0.06, 0.03])
        active = True
    return result, active


def failure_reason(success: bool, complete: bool, environment_done: bool) -> str:
    if success:
        return "success"
    if environment_done:
        return "environment_terminated"
    if not complete:
        return "policy_incomplete"
    return "final_error_above_threshold"


def run_worker() -> None:
    if None in (args.worker_method, args.worker_condition, args.worker_seed) or args.result_path is None:
        raise ValueError("Complete worker arguments are required.")
    demonstrations, manifest = load_bimanual_dataset(args.data_dir, verify_hashes=True)
    policy = BimanualGaussianPolicy(args.worker_method)
    policy.fit(demonstrations)
    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = args.worker_seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args.worker_seed)
    observation, poses = read_observation(env)
    policy.reset(observation)
    left_offset = None
    right_offset = None
    previous_carrier = None
    environment_done = False
    perturbation_seen = False
    inference_times = []
    left_path = [observation.left_ee_pose[:3].copy()]
    right_path = [observation.right_ee_pose[:3].copy()]
    handover_distances = []
    frame_usage = Counter()

    for _ in range(args.max_steps):
        observation, poses = read_observation(env)
        phase_before = policy.phase
        phase_step_before = policy.phase_step
        start = time.perf_counter()
        step = policy.act(observation)
        inference_times.append((time.perf_counter() - start) * 1000.0)
        action, active = perturb_action(
            step.action, args.worker_condition, phase_before, phase_step_before, observation
        )
        perturbation_seen |= active
        for frame in step.diagnostics["left_active_frames"]:
            frame_usage[f"left:{frame}"] += 1
        for frame in step.diagnostics["right_active_frames"]:
            frame_usage[f"right:{frame}"] += 1

        carrier = policy.carrier
        if carrier != previous_carrier:
            if carrier == "left":
                left_offset = math_utils.subtract_frame_transforms(
                    poses[0][:, :3], poses[0][:, 3:7], poses[2][:, :3], poses[2][:, 3:7]
                )
            elif carrier == "right":
                right_offset = math_utils.subtract_frame_transforms(
                    poses[1][:, :3], poses[1][:, 3:7], poses[2][:, :3], poses[2][:, 3:7]
                )
                policy.set_right_attachment_offset(
                    right_offset[0][0].detach().cpu().numpy().astype(np.float64)
                )
            previous_carrier = carrier
        if carrier == "left" and left_offset is not None:
            write_attached_object(env, poses[0], left_offset)
        elif carrier == "right" and right_offset is not None:
            write_attached_object(env, poses[1], right_offset)
        if 5 <= phase_before <= 8:
            handover_distances.append(float(np.linalg.norm(observation.left_ee_pose[:3] - observation.right_ee_pose[:3])))
        left_path.append(observation.left_ee_pose[:3].copy())
        right_path.append(observation.right_ee_pose[:3].copy())
        tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        tensor = actions_to_robot_root_frames(env, tensor)
        _, _, terminated, truncated, _ = env.step(tensor)
        if bool((terminated | truncated).any().item()):
            environment_done = True
            break
        if policy.complete:
            break

    final_observation, _ = read_observation(env)
    final_error = float(np.linalg.norm(final_observation.object_pose[:3] - final_observation.target_pose[:3]))
    success = bool(policy.complete and not environment_done and final_error < args.success_threshold)
    path_length = float(
        np.sum(np.linalg.norm(np.diff(np.asarray(left_path), axis=0), axis=1))
        + np.sum(np.linalg.norm(np.diff(np.asarray(right_path), axis=0), axis=1))
    )
    metrics = {
        "success": success,
        "final_error_m": final_error,
        "policy_complete": policy.complete,
        "environment_done": environment_done,
        "forced_transitions": policy.forced_transitions,
        "path_length_m": path_length,
        "mean_handover_distance_m": float(np.mean(handover_distances)) if handover_distances else None,
        "mean_inference_ms": float(np.mean(inference_times)),
        "perturbation_seen": perturbation_seen,
        "frame_usage_steps": dict(frame_usage),
        "failure_reason": failure_reason(success, policy.complete, environment_done),
    }
    result = {
        "method": args.worker_method,
        "condition": args.worker_condition,
        "seed": args.worker_seed,
        "dataset_sha256": manifest["dataset_sha256"],
        "metrics": metrics,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        run_worker()
    finally:
        simulation_app.close()
