"""Evaluate simultaneous bilateral connection and perturbation recovery."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


METHODS = ("independent_arms", "static_shared_object", "full_dynamac")
CONDITIONS = ("static", "left_offset", "right_offset", "left_pause", "right_pause")
TASK_ID = "Essay2608-Bimanual-Lift-Tray-v0"
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/lift_tray_static/v1"))
parser.add_argument("--output_dir", type=Path, default=Path("outputs/lift_tray_minimal/v1"))
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
parser.add_argument("--seeds", nargs="+", type=int, default=[10208])
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


def run_controller() -> int:
    import numpy as np

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
                        trials.append(cached)
                        print(f"[tray-eval] reuse {index}/{total} {method} {condition}", flush=True)
                        continue
                print(f"[tray-eval] trial {index}/{total} {method} {condition}", flush=True)
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
                    trials.append({"method": method, "condition": condition, "seed": seed})
                else:
                    trials.append(json.loads(path.read_text(encoding="utf-8")))
    ablation = {}
    for method in args.methods:
        ablation[method] = {}
        for condition in args.conditions:
            metrics = [
                trial["metrics"]
                for trial in trials
                if trial.get("method") == method and trial.get("condition") == condition and "metrics" in trial
            ]
            ablation[method][condition] = {
                "num_trials": len(metrics),
                "success_rate": float(np.mean([value["success"] for value in metrics])) if metrics else None,
                "mean_final_error_m": float(np.mean([value["final_error_m"] for value in metrics])) if metrics else None,
                "mean_width_error_m": float(np.mean([value["mean_width_error_m"] for value in metrics])) if metrics else None,
                "mean_path_length_m": float(np.mean([value["path_length_m"] for value in metrics])) if metrics else None,
            }
    summary = {
        "task_id": TASK_ID,
        "methods": args.methods,
        "conditions": args.conditions,
        "seeds": args.seeds,
        "ablation": ablation,
        "trials": trials,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(ablation, indent=2))
    return int(any("metrics" not in trial for trial in trials))


if not args.worker:
    raise SystemExit(run_controller())


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import essay2608
from essay2608.bimanual import actions_to_robot_root_frames
from essay2608.data.dataset import load_bimanual_dataset
from essay2608.policy.bimanual import BimanualPolicyObservation
from essay2608.policy.tray import TrayGaussianPolicy
from essay2608.tray import get_tray_poses, write_bilateral_tray
from isaaclab_tasks.utils import parse_env_cfg


def observe(env):
    poses = get_tray_poses(env)
    arrays = [pose[0].detach().cpu().numpy().astype(np.float64) for pose in poses]
    return BimanualPolicyObservation(*arrays), poses


def perturb(action, condition, phase, phase_step, observation):
    output = action.copy()
    active = False
    if condition == "left_offset" and phase == 4 and 15 <= phase_step < 75:
        output[:3] += np.asarray([0.0, 0.08, 0.06])
        active = True
    elif condition == "right_offset" and phase == 4 and 15 <= phase_step < 75:
        output[8:11] += np.asarray([0.0, -0.08, -0.05])
        active = True
    elif condition == "left_pause" and phase == 3 and phase_step < 55:
        output[:7] = observation.left_ee_pose
        active = True
    elif condition == "right_pause" and phase == 3 and phase_step < 55:
        output[8:15] = observation.right_ee_pose
        active = True
    return output, active


def run_worker() -> None:
    demonstrations, manifest = load_bimanual_dataset(args.data_dir, verify_hashes=True)
    policy = TrayGaussianPolicy(args.worker_method)
    policy.fit(demonstrations)
    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = args.worker_seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args.worker_seed)
    observation, poses = observe(env)
    policy.reset(observation)
    midpoint_offset = None
    environment_done = False
    widths = []
    initial_width = float(np.linalg.norm(observation.left_ee_pose[:3] - observation.right_ee_pose[:3]))
    paths = [[observation.left_ee_pose[:3].copy()], [observation.right_ee_pose[:3].copy()]]
    inference = []
    perturbation_seen = False
    for _ in range(args.max_steps):
        observation, poses = observe(env)
        phase_before, phase_step_before = policy.phase, policy.phase_step
        start = time.perf_counter()
        step = policy.act(observation)
        inference.append((time.perf_counter() - start) * 1000.0)
        action, active = perturb(step.action, args.worker_condition, phase_before, phase_step_before, observation)
        perturbation_seen |= active
        if policy.connected and midpoint_offset is None:
            midpoint_offset = poses[2][:, :3] - 0.5 * (poses[0][:, :3] + poses[1][:, :3])
            initial_width = float(torch.linalg.norm(poses[0][0, :3] - poses[1][0, :3]).item())
        if policy.connected:
            write_bilateral_tray(env, poses[0], poses[1], midpoint_offset)
            widths.append(float(torch.linalg.norm(poses[0][0, :3] - poses[1][0, :3]).item()))
        paths[0].append(observation.left_ee_pose[:3].copy())
        paths[1].append(observation.right_ee_pose[:3].copy())
        tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        _, _, terminated, truncated, _ = env.step(actions_to_robot_root_frames(env, tensor))
        if bool((terminated | truncated).any().item()):
            environment_done = True
            break
        if policy.complete:
            break
    final, _ = observe(env)
    final_error = float(np.linalg.norm(final.object_pose[:3] - final.target_pose[:3]))
    success = bool(policy.complete and not environment_done and final_error < args.success_threshold)
    path_length = sum(
        float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1))) for path in paths
    )
    metrics = {
        "success": success,
        "final_error_m": final_error,
        "mean_width_error_m": float(np.mean(np.abs(np.asarray(widths) - initial_width))) if widths else None,
        "max_width_error_m": float(np.max(np.abs(np.asarray(widths) - initial_width))) if widths else None,
        "path_length_m": path_length,
        "mean_inference_ms": float(np.mean(inference)),
        "forced_transitions": policy.forced_transitions,
        "policy_complete": policy.complete,
        "environment_done": environment_done,
        "perturbation_seen": perturbation_seen,
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
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        run_worker()
    finally:
        simulation_app.close()
