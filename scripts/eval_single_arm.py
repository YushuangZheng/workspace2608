"""Run the minimal single-arm Gaussian/DynaMAC dynamic ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from isaaclab.app import AppLauncher


GEOMETRIC_METHODS = (
    "world_gaussian",
    "static_multistream",
    "skill_dynamac",
    "mask_only",
    "full_dynamac",
    "relation_dynamac",
)
METHODS = (*GEOMETRIC_METHODS, "diffusion_policy")
CONDITIONS = (
    "static",
    "smooth_object",
    "sudden_object",
    "smooth_target",
    "sudden_target",
    "arm_offset",
    "drop_after_grasp",
    "close_without_grasp",
)
TASK_ID = "Essay2608-Dynamic-Pick-Place-v0"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument("--output_dir", type=Path, default=Path("outputs/single_arm_minimal/v1"))
parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(GEOMETRIC_METHODS))
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
parser.add_argument("--seeds", nargs="+", type=int, default=[6200])
parser.add_argument("--max_steps", type=int, default=900)
parser.add_argument("--legacy_success_threshold", type=float, default=0.06)
parser.add_argument("--success_xy_threshold", type=float, default=0.01)
parser.add_argument(
    "--success_xy_sensitivity",
    type=float,
    nargs="+",
    default=[0.005, 0.01, 0.02],
)
parser.add_argument("--support_height_tolerance", type=float, default=0.01)
parser.add_argument("--stability_window", type=int, default=25)
parser.add_argument("--stability_displacement_threshold", type=float, default=0.005)
parser.add_argument("--stability_speed_threshold", type=float, default=0.05)
parser.add_argument(
    "--diffusion_checkpoint",
    type=Path,
    default=Path("outputs/diffusion/v1/checkpoint.pt"),
)
parser.add_argument(
    "--resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Reuse complete per-trial JSON files already present in output_dir.",
)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--worker_method", choices=METHODS, help=argparse.SUPPRESS)
parser.add_argument("--worker_condition", choices=CONDITIONS, help=argparse.SUPPRESS)
parser.add_argument("--worker_seed", type=int, help=argparse.SUPPRESS)
parser.add_argument("--result_path", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


EVALUATION_SCHEMA_VERSION = 5


@lru_cache(maxsize=1)
def calibrated_relation_configuration() -> dict:
    """Load the frozen-data calibration once in the controller process."""

    from essay2608.data.dataset import load_dataset
    from essay2608.policy.relation import calibrate_relation_estimator

    demonstrations, _ = load_dataset(args.data_dir, verify_hashes=True)
    config, _ = calibrate_relation_estimator(demonstrations)
    return config.as_dict()


def policy_configuration(method: str) -> dict:
    """Return the rollout-relevant defaults that are otherwise hidden in classes."""

    common = {
        "bins_per_phase": 25,
        "position_threshold_m": 0.018,
        "maximum_hold_steps": 200,
        "maximum_action_position_step_m": 0.02,
    }
    configurations = {
        "world_gaussian": {**common, "frames": ["world"]},
        "static_multistream": {**common, "frames": ["object", "target"]},
        "skill_dynamac": {
            **common,
            "skill_labels": "scripted_expert_phases_0_to_9",
            "pose_state_dimension": 6,
            "kinematic_scale_threshold": 0.001,
            "linked_bin_fraction": 0.5,
            "frame_selection_threshold": 0.2,
            "selection_mode": "offline_fixed_per_skill",
        },
        "mask_only": {
            **common,
            "detector": {
                "window_steps": 10,
                "relative_rms_std_threshold_m": 0.0015,
                "object_motion_threshold_m": 0.004,
                "release_rule": "open_gripper_only",
            },
        },
        "full_dynamac": {
            **common,
            "detector": {
                "window_steps": 10,
                "relative_rms_std_threshold_m": 0.0015,
                "object_motion_threshold_m": 0.004,
                "release_rule": "open_gripper_only",
            },
            "virtual_frame": "captured_at_phase_4_start",
        },
        "relation_dynamac": {
            **common,
            "estimator": calibrated_relation_configuration(),
            "actual_gripper_joint_feedback": True,
            "contact_sensor_available": False,
            "virtual_frame": "captured_on_connected_transition",
            "frame_activation": "mask_while_connected_virtual_in_phase_4",
        },
        "diffusion_policy": {
            **common,
            "execution_horizon": 4,
            "sampling_seed": 2608,
        },
    }
    return configurations[method]


def perturbation_configuration(condition: str) -> dict:
    """Expose deterministic perturbation magnitudes and trigger rules."""

    configurations = {
        "static": {"kind": "none"},
        "smooth_object": {
            "shift_m": [0.0, 0.08, 0.0],
            "trigger_phase": 1,
            "ramp_phase_steps": [4, 24],
        },
        "sudden_object": {
            "shift_m": [0.0, 0.08, 0.0],
            "trigger_phase": 1,
            "trigger_phase_step": 10,
        },
        "smooth_target": {
            "shift_m": [0.0, -0.10, 0.0],
            "trigger_phase": 4,
            "ramp_phase_steps": [4, 24],
        },
        "sudden_target": {
            "shift_m": [0.0, -0.10, 0.0],
            "trigger_phase": 4,
            "trigger_phase_step": 10,
        },
        "arm_offset": {
            "shift_m": [0.0, 0.06, 0.0],
            "active_phase": 5,
            "active_phase_steps": [5, 25],
        },
        "drop_after_grasp": {
            "shift_m": [0.0, 0.18, 0.0],
            "trigger_phase": 5,
            "trigger_phase_step": 10,
            "place_on_support": True,
            "gripper_command_unchanged": True,
        },
        "close_without_grasp": {
            "shift_m": [0.0, -0.18, 0.0],
            "trigger_phase": 3,
            "place_on_support": True,
            "gripper_command_unchanged": True,
        },
    }
    return configurations[condition]


def sha256_file(path: Path) -> str:
    """Return a streaming content hash for cache identity."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> str:
    """Hash every source file that can change one single-arm rollout."""

    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve()]
    package = root / "source" / "essay2608" / "essay2608"
    for relative in ("policy", "eval", "data"):
        paths.extend(sorted((package / relative).glob("*.py")))
    paths.extend(
        [
            package / "expert.py",
            package
            / "tasks"
            / "manager_based"
            / "dynamic_pick_place"
            / "dynamic_pick_place_env_cfg.py",
        ]
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_git_commit() -> str:
    """Record the repository revision alongside the content-addressed source hash."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def experiment_fingerprint(method: str, condition: str, seed: int) -> tuple[str, dict]:
    """Fingerprint code, frozen data, checkpoint, and all rollout settings."""

    manifest = json.loads((args.data_dir.resolve() / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_git_commit": source_git_commit(),
        "source_sha256": source_fingerprint(),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "method": method,
        "policy_config": policy_configuration(method),
        "condition": condition,
        "perturbation_config": perturbation_configuration(condition),
        "seed": seed,
        "max_steps": args.max_steps,
        "legacy_success_threshold_m": args.legacy_success_threshold,
        "success_xy_threshold_m": args.success_xy_threshold,
        "success_xy_sensitivity_m": sorted(set(args.success_xy_sensitivity)),
        "support_height_tolerance_m": args.support_height_tolerance,
        "stability_window_steps": args.stability_window,
        "stability_displacement_threshold_m": args.stability_displacement_threshold,
        "stability_speed_threshold_m_s": args.stability_speed_threshold,
    }
    if method == "diffusion_policy":
        checkpoint = args.diffusion_checkpoint.resolve()
        payload["diffusion_checkpoint"] = str(checkpoint)
        payload["diffusion_checkpoint_sha256"] = sha256_file(checkpoint)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), payload


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


def bootstrap_mean_interval(values: list[float], seed: int, samples: int = 20000) -> dict:
    """Return a deterministic non-parametric 95% interval for a mean."""

    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": None, "std": None, "ci95": [None, None]}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if len(array) == 1:
        return {"mean": mean, "std": std, "ci95": [mean, mean]}
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    bootstrap_means = np.mean(array[indices], axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {"mean": mean, "std": std, "ci95": [float(low), float(high)]}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float | None]:
    """Return the Wilson 95% confidence interval for a Bernoulli rate."""

    if trials == 0:
        return [None, None]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
    return [max(0.0, centre - margin / denominator), min(1.0, centre + margin / denominator)]


def failure_reason(metrics: dict) -> str:
    """Classify one unsuccessful rollout using mutually exclusive causes."""

    return str(metrics.get("failure_reason", "unknown"))


def aggregate_trials(trials: list[dict]) -> dict:
    """Aggregate one or more seeds by method and condition with uncertainty."""

    import numpy as np

    summary = {}
    for method_index, method in enumerate(args.methods):
        summary[method] = {}
        for condition_index, condition in enumerate(args.conditions):
            selected = [
                trial
                for trial in trials
                if trial.get("method") == method and trial.get("condition") == condition and "metrics" in trial
            ]
            if not selected:
                summary[method][condition] = {"num_trials": 0}
                continue
            metrics = [trial["metrics"] for trial in selected]
            successes = sum(bool(item["success"]) for item in metrics)
            recovery = [item["recovery_success"] for item in metrics if item["recovery_success"] is not None]
            onset_delays = [
                item["connection_onset_delay_s"]
                for item in metrics
                if item["connection_onset_delay_s"] is not None
            ]
            release_delays = [
                item["connection_release_delay_s"]
                for item in metrics
                if item["connection_release_delay_s"] is not None
            ]
            post_event_loss_delays = [
                item["post_event_connection_loss_delay_s"]
                for item in metrics
                if item["post_event_connection_loss_delay_s"] is not None
            ]
            interval_seed = 2608 + 100 * method_index + condition_index
            sensitivity_keys = sorted(metrics[0]["xy_success_sensitivity"])
            phase_keys = sorted(metrics[0]["phase_path_length_m"], key=int)
            summary[method][condition] = {
                "num_trials": len(metrics),
                "success_rate": successes / len(metrics),
                "success_rate_ci95_wilson": wilson_interval(successes, len(metrics)),
                "mean_final_xy_error_m": float(
                    np.mean([item["final_xy_error_m"] for item in metrics])
                ),
                "final_xy_error_m_statistics": bootstrap_mean_interval(
                    [item["final_xy_error_m"] for item in metrics], interval_seed
                ),
                "mean_final_error_3d_m": float(
                    np.mean([item["final_error_3d_m"] for item in metrics])
                ),
                "mean_path_length_m": float(np.mean([item["path_length_m"] for item in metrics])),
                "path_length_m_statistics": bootstrap_mean_interval(
                    [item["path_length_m"] for item in metrics], interval_seed + 1000
                ),
                "mean_phase_path_length_m": {
                    phase: float(
                        np.mean([item["phase_path_length_m"][phase] for item in metrics])
                    )
                    for phase in phase_keys
                },
                "mean_max_ee_speed_m_s": float(
                    np.mean([item["max_ee_speed_m_s"] for item in metrics])
                ),
                "mean_inference_ms": float(np.mean([item["mean_inference_ms"] for item in metrics])),
                "inference_ms_statistics": bootstrap_mean_interval(
                    [item["mean_inference_ms"] for item in metrics], interval_seed + 2000
                ),
                "connection_detection_rate": float(np.mean([item["connection_detected"] for item in metrics])),
                "recovery_rate": float(np.mean(recovery)) if recovery else None,
                "recovery_rate_ci95_wilson": wilson_interval(sum(bool(value) for value in recovery), len(recovery)),
                "mean_mask_false_positive_rate": float(
                    np.mean([item["mask_false_positive_rate"] for item in metrics])
                ),
                "mean_mask_false_negative_rate": float(
                    np.mean([item["mask_false_negative_rate"] for item in metrics])
                ),
                "mean_connection_onset_delay_s": (
                    float(np.mean(onset_delays)) if onset_delays else None
                ),
                "mean_connection_release_delay_s": (
                    float(np.mean(release_delays)) if release_delays else None
                ),
                "mean_post_event_connection_loss_delay_s": (
                    float(np.mean(post_event_loss_delays))
                    if post_event_loss_delays
                    else None
                ),
                "mean_maximum_relation_confidence": float(
                    np.mean([item["maximum_relation_confidence"] for item in metrics])
                ),
                "xy_success_sensitivity": {
                    threshold: float(
                        np.mean([item["xy_success_sensitivity"][threshold] for item in metrics])
                    )
                    for threshold in sensitivity_keys
                },
                "mean_max_raw_policy_action_jump_m": float(
                    np.mean([item["max_raw_policy_action_jump_m"] for item in metrics])
                ),
                "mean_max_rate_limited_policy_action_jump_m": float(
                    np.mean(
                        [item["max_rate_limited_policy_action_jump_m"] for item in metrics]
                    )
                ),
                "failure_reasons": dict(Counter(failure_reason(item) for item in metrics)),
            }
    return summary


def aggregate_by_method(trials: list[dict]) -> dict:
    """Aggregate condition-balanced outcomes using seeds as independent units."""

    paper_summary = {}
    for method_index, method in enumerate(args.methods):
        seed_rows = []
        for seed in args.seeds:
            selected = [
                trial["metrics"]
                for trial in trials
                if trial.get("method") == method and trial.get("seed") == seed and "metrics" in trial
            ]
            if len(selected) != len(args.conditions):
                continue
            dynamic = [item for item in selected if item["recovery_success"] is not None]
            seed_rows.append(
                {
                    "seed": seed,
                    "success_rate": sum(bool(item["success"]) for item in selected) / len(selected),
                    "recovery_rate": (
                        sum(bool(item["recovery_success"]) for item in dynamic) / len(dynamic) if dynamic else None
                    ),
                    "mean_final_xy_error_m": sum(item["final_xy_error_m"] for item in selected)
                    / len(selected),
                    "mean_final_error_3d_m": sum(item["final_error_3d_m"] for item in selected)
                    / len(selected),
                    "mean_path_length_m": sum(item["path_length_m"] for item in selected) / len(selected),
                    "mean_inference_ms": sum(item["mean_inference_ms"] for item in selected) / len(selected),
                }
            )
        interval_seed = 52608 + method_index * 100
        paper_summary[method] = {
            "num_complete_seeds": len(seed_rows),
            "seed_rows": seed_rows,
            "success_rate_statistics": bootstrap_mean_interval(
                [row["success_rate"] for row in seed_rows], interval_seed
            ),
            "recovery_rate_statistics": bootstrap_mean_interval(
                [row["recovery_rate"] for row in seed_rows if row["recovery_rate"] is not None], interval_seed + 1
            ),
            "final_xy_error_m_statistics": bootstrap_mean_interval(
                [row["mean_final_xy_error_m"] for row in seed_rows], interval_seed + 2
            ),
            "final_error_3d_m_statistics": bootstrap_mean_interval(
                [row["mean_final_error_3d_m"] for row in seed_rows], interval_seed + 5
            ),
            "path_length_m_statistics": bootstrap_mean_interval(
                [row["mean_path_length_m"] for row in seed_rows], interval_seed + 3
            ),
            "inference_ms_statistics": bootstrap_mean_interval(
                [row["mean_inference_ms"] for row in seed_rows], interval_seed + 4
            ),
        }
    return paper_summary


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
                fingerprint, fingerprint_payload = experiment_fingerprint(method, condition, seed)
                if args.resume and result_path.is_file():
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if (
                        cached.get("method") == method
                        and cached.get("condition") == condition
                        and cached.get("seed") == seed
                        and cached.get("experiment_fingerprint") == fingerprint
                        and "metrics" in cached
                    ):
                        print(
                            f"[study] reuse trial={trial_index}/{total} method={method} "
                            f"condition={condition} seed={seed}",
                            flush=True,
                        )
                        trials.append(cached)
                        continue
                    print(
                        f"[study] invalidate stale cache method={method} condition={condition} seed={seed}",
                        flush=True,
                    )
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
                            "experiment_fingerprint": fingerprint,
                            "experiment_config": fingerprint_payload,
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
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "success_definition": {
            "legacy_3d_threshold_m": args.legacy_success_threshold,
            "xy_threshold_m": args.success_xy_threshold,
            "xy_sensitivity_thresholds_m": sorted(set(args.success_xy_sensitivity)),
            "support_height": "median final support height in frozen demonstrations",
            "support_height_tolerance_m": args.support_height_tolerance,
            "released_gripper_required": True,
            "stability_window_steps": args.stability_window,
            "stability_displacement_threshold_m": args.stability_displacement_threshold,
            "stability_speed_threshold_m_s": args.stability_speed_threshold,
        },
        "stream_diagnostics": diagnostics,
        "ablation": aggregate_trials(trials),
        "condition_balanced_method_statistics": aggregate_by_method(trials),
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
from essay2608.eval import EpisodeTrace, PerturbationController, SuccessCriteria
from essay2608.expert import get_scene_poses
from essay2608.policy import (
    DiffusionActionPolicy,
    DynaMACPolicy,
    MaskOnlyPolicy,
    RelationDynaMACPolicy,
    SkillDynaMACPolicy,
    StaticMultiStreamPolicy,
    WorldGaussianPolicy,
)
from essay2608.policy.base import PolicyObservation
from isaaclab_tasks.utils import parse_env_cfg


def numpy_observation(env: gym.Env) -> PolicyObservation:
    """Read the first environment into the policy's NumPy observation."""

    ee_pose, object_pose, target_pose = get_scene_poses(env)
    robot = env.unwrapped.scene["robot"]
    gripper_opening = float(torch.sum(robot.data.joint_pos[0, -2:]).item())
    gripper_velocity = float(torch.sum(robot.data.joint_vel[0, -2:]).item())
    return PolicyObservation(
        ee_pose=ee_pose[0].detach().cpu().numpy().astype(np.float64),
        object_pose=object_pose[0].detach().cpu().numpy().astype(np.float64),
        target_pose=target_pose[0].detach().cpu().numpy().astype(np.float64),
        gripper_opening_m=gripper_opening,
        gripper_velocity_m_s=gripper_velocity,
    )


def make_policy(name: str):
    """Construct one ablation policy."""

    if name == "diffusion_policy":
        return DiffusionActionPolicy(args.diffusion_checkpoint)
    return {
        "world_gaussian": WorldGaussianPolicy,
        "static_multistream": StaticMultiStreamPolicy,
        "skill_dynamac": SkillDynaMACPolicy,
        "mask_only": MaskOnlyPolicy,
        "full_dynamac": DynaMACPolicy,
        "relation_dynamac": RelationDynaMACPolicy,
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

    for _ in range(args.max_steps):
        phase_before = policy.phase
        phase_step_before = policy.phase_step
        event_started_before = perturbation.event_started
        scene_status = perturbation.update_scene(phase_before, phase_step_before)
        perturbation_event = perturbation.event_started and not event_started_before
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
            perturbation_event,
        )
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        _, _, terminated, truncated, _ = env.step(action_tensor)
        if bool((terminated | truncated).any().item()):
            environment_done = True
            break
        if policy.complete:
            break

    # Always read the post-step scene state.  The previous implementation used
    # the pre-step error on termination, making terminal trials one control step
    # stale.  The configured horizon ends before the normal time-limit reset.
    final_observation = numpy_observation(env)
    trace.set_terminal_observation(final_observation)
    support_height = float(
        np.median(
            np.concatenate(
                [
                    demonstration.object_pose[-args.stability_window :, 2]
                    for demonstration in demonstrations
                ]
            )
        )
    )
    criteria = SuccessCriteria(
        legacy_3d_threshold_m=args.legacy_success_threshold,
        xy_threshold_m=args.success_xy_threshold,
        xy_sensitivity_thresholds_m=tuple(sorted(set(args.success_xy_sensitivity))),
        support_height_m=support_height,
        support_height_tolerance_m=args.support_height_tolerance,
        stability_window_steps=args.stability_window,
        stability_displacement_m=args.stability_displacement_threshold,
        stability_speed_m_s=args.stability_speed_threshold,
    )

    metrics = trace.summary(
        final_object_position=final_observation.object_pose[:3],
        final_target_position=final_observation.target_pose[:3],
        criteria=criteria,
        policy_complete=policy.complete,
        environment_done=environment_done,
        forced_transitions=policy.forced_transitions,
        perturbation_started=perturbation.event_started,
        relation_loss_expected=args.worker_condition == "drop_after_grasp",
    )
    fingerprint, fingerprint_payload = experiment_fingerprint(
        args.worker_method, args.worker_condition, args.worker_seed
    )
    result = {
        "method": args.worker_method,
        "condition": args.worker_condition,
        "seed": args.worker_seed,
        "dataset_sha256": manifest["dataset_sha256"],
        "experiment_fingerprint": fingerprint,
        "experiment_config": fingerprint_payload,
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
