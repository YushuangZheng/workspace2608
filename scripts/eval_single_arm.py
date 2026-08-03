"""Run the minimal single-arm Gaussian/DynaMAC dynamic ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


METHODS = ("world_gaussian", "static_multistream", "mask_only", "full_dynamac")
CONDITIONS = (
    "static",
    "smooth_object",
    "sudden_object",
    "smooth_target",
    "sudden_target",
    "arm_offset",
)
TASK_ID = "Essay2608-Dynamic-Pick-Place-v0"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument("--output_dir", type=Path, default=Path("outputs/single_arm_minimal/v1"))
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
parser.add_argument("--seeds", nargs="+", type=int, default=[6200])
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--success_threshold", type=float, default=0.06)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker_method", choices=METHODS, help=argparse.SUPPRESS)
parser.add_argument("--worker_condition", choices=CONDITIONS, help=argparse.SUPPRESS)
parser.add_argument("--worker_seed", type=int, help=argparse.SUPPRESS)
parser.add_argument("--result_path", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def offline_stream_diagnostics(data_dir: Path) -> dict:
    """Quantify the static PoE precision anomaly without simulation."""

    from essay2608.data.dataset import load_dataset
    from essay2608.policy import StaticMultiStreamPolicy
    from essay2608.policy.base import PolicyObservation

    demonstrations, _ = load_dataset(data_dir, verify_hashes=True)
    policy = StaticMultiStreamPolicy()
    policy.fit(demonstrations)
    demonstration = demonstrations[0]
    phases = {}
    for phase in range(10):
        indices = demonstration.phase_indices(phase)
        source_index = indices[len(indices) // 2]
        observation = PolicyObservation(
            ee_pose=demonstration.ee_pose[source_index],
            object_pose=demonstration.object_pose[source_index],
            target_pose=demonstration.target_pose[source_index],
        )
        policy.reset(observation)
        policy.phase = phase
        policy.phase_step = int(policy.phase_durations[phase] // 2)
        diagnostics = policy._compute_action(observation).diagnostics
        phases[str(phase)] = {
            "phase_name": diagnostics["phase_name"],
            "stream_uncertainty_m": diagnostics["stream_uncertainty_m"],
            "stream_weights": diagnostics["stream_weights"],
            "object_to_target_weight_ratio": diagnostics["stream_weights"]["object"]
            / max(diagnostics["stream_weights"]["target"], 1.0e-12),
        }
    return {"phases": phases}


def aggregate_trials(trials: list[dict]) -> dict:
    """Aggregate one or more seeds by method and condition."""

    import numpy as np

    summary = {}
    for method in args.methods:
        summary[method] = {}
        for condition in args.conditions:
            selected = [
                trial
                for trial in trials
                if trial.get("method") == method and trial.get("condition") == condition and "metrics" in trial
            ]
            if not selected:
                summary[method][condition] = {"num_trials": 0}
                continue
            metrics = [trial["metrics"] for trial in selected]
            summary[method][condition] = {
                "num_trials": len(metrics),
                "success_rate": float(np.mean([item["success"] for item in metrics])),
                "mean_final_error_m": float(np.mean([item["final_error_m"] for item in metrics])),
                "mean_path_length_m": float(np.mean([item["path_length_m"] for item in metrics])),
                "mean_inference_ms": float(np.mean([item["mean_inference_ms"] for item in metrics])),
                "connection_detection_rate": float(np.mean([item["connection_detected"] for item in metrics])),
                "recovery_rate": float(
                    np.mean([item["recovery_success"] for item in metrics if item["recovery_success"] is not None])
                )
                if any(item["recovery_success"] is not None for item in metrics)
                else None,
                "mean_mask_false_positive_rate": float(
                    np.mean([item["mask_false_positive_rate"] for item in metrics])
                ),
                "mean_mask_false_negative_rate": float(
                    np.mean([item["mask_false_negative_rate"] for item in metrics])
                ),
            }
    return summary


def run_controller() -> int:
    """Launch every trial in an isolated Isaac Lab process."""

    output_dir = args.output_dir.resolve()
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trials = []

    total = len(args.methods) * len(args.conditions) * len(args.seeds)
    trial_index = 0
    for seed in args.seeds:
        for condition in args.conditions:
            for method in args.methods:
                trial_index += 1
                result_path = trial_dir / f"{method}__{condition}__seed_{seed}.json"
                print(
                    f"\n[study] trial={trial_index}/{total} method={method} "
                    f"condition={condition} seed={seed}",
                    flush=True,
                )
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
                    str(result_path),
                ]
                result = subprocess.run(command, check=False)
                if result.returncode != 0 or not result_path.is_file():
                    trials.append(
                        {
                            "method": method,
                            "condition": condition,
                            "seed": seed,
                            "worker_returncode": result.returncode,
                        }
                    )
                    continue
                trials.append(json.loads(result_path.read_text(encoding="utf-8")))

    diagnostics = offline_stream_diagnostics(args.data_dir.resolve())
    summary = {
        "task_id": TASK_ID,
        "dataset_dir": str(args.data_dir.resolve()),
        "methods": args.methods,
        "conditions": args.conditions,
        "seeds": args.seeds,
        "success_threshold_m": args.success_threshold,
        "stream_diagnostics": diagnostics,
        "ablation": aggregate_trials(trials),
        "trials": trials,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["ablation"], indent=2), flush=True)
    print(f"[study] summary: {output_dir / 'summary.json'}", flush=True)
    return int(any("metrics" not in trial for trial in trials))


if not args.worker:
    raise SystemExit(run_controller())


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import essay2608
from essay2608.data.dataset import load_dataset
from essay2608.eval import EpisodeTrace, PerturbationController
from essay2608.expert import get_scene_poses
from essay2608.policy import DynaMACPolicy, MaskOnlyPolicy, StaticMultiStreamPolicy, WorldGaussianPolicy
from essay2608.policy.base import PolicyObservation
from isaaclab_tasks.utils import parse_env_cfg


def numpy_observation(env: gym.Env) -> PolicyObservation:
    """Read the first environment into the policy's NumPy observation."""

    ee_pose, object_pose, target_pose = get_scene_poses(env)
    return PolicyObservation(
        ee_pose=ee_pose[0].detach().cpu().numpy().astype(np.float64),
        object_pose=object_pose[0].detach().cpu().numpy().astype(np.float64),
        target_pose=target_pose[0].detach().cpu().numpy().astype(np.float64),
    )


def make_policy(name: str):
    """Construct one ablation policy."""

    return {
        "world_gaussian": WorldGaussianPolicy,
        "static_multistream": StaticMultiStreamPolicy,
        "mask_only": MaskOnlyPolicy,
        "full_dynamac": DynaMACPolicy,
    }[name]()


def run_worker() -> None:
    if args.worker_method is None or args.worker_condition is None or args.worker_seed is None:
        raise ValueError("Worker method, condition, and seed are required.")
    if args.result_path is None:
        raise ValueError("Worker result path is required.")

    demonstrations, manifest = load_dataset(args.data_dir, verify_hashes=True)
    policy = make_policy(args.worker_method)
    policy.fit(demonstrations)

    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = args.worker_seed
    env_cfg.commands.object_pose.debug_vis = False
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args.worker_seed)
    control_dt = env_cfg.sim.dt * env_cfg.decimation

    observation = numpy_observation(env)
    policy.reset(observation)
    perturbation = PerturbationController(args.worker_condition, env)
    trace = EpisodeTrace(control_dt=control_dt)
    environment_done = False
    last_error = float(np.linalg.norm(observation.object_pose[:3] - observation.target_pose[:3]))

    for _ in range(args.max_steps):
        phase_before = policy.phase
        phase_step_before = policy.phase_step
        scene_status = perturbation.update_scene(phase_before, phase_step_before)
        observation = numpy_observation(env)

        start = time.perf_counter()
        policy_step = policy.act(observation)
        inference_ms = (time.perf_counter() - start) * 1000.0
        action, action_status = perturbation.update_action(
            policy_step.action,
            phase_before,
            phase_step_before,
        )
        trace.append(
            observation,
            action,
            policy_step.diagnostics,
            inference_ms,
            scene_status.active or action_status.active,
        )
        last_error = float(np.linalg.norm(observation.object_pose[:3] - observation.target_pose[:3]))

        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        _, _, terminated, truncated, _ = env.step(action_tensor)
        if bool((terminated | truncated).any().item()):
            environment_done = True
            break
        if policy.complete:
            break

    if not environment_done:
        final_observation = numpy_observation(env)
        final_error = float(np.linalg.norm(final_observation.object_pose[:3] - final_observation.target_pose[:3]))
    else:
        final_error = last_error

    metrics = trace.summary(
        final_error=final_error,
        success_threshold=args.success_threshold,
        policy_complete=policy.complete,
        environment_done=environment_done,
        forced_transitions=policy.forced_transitions,
        perturbation_started=perturbation.event_started,
    )
    result = {
        "method": args.worker_method,
        "condition": args.worker_condition,
        "seed": args.worker_seed,
        "dataset_sha256": manifest["dataset_sha256"],
        "perturbation_started": perturbation.event_started,
        "perturbation_finished": perturbation.event_finished,
        "metrics": metrics,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(args.result_path.with_suffix(".npz"), **trace.arrays())
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        run_worker()
    finally:
        simulation_app.close()
