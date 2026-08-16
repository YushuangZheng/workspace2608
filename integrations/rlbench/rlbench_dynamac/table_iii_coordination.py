"""Local Table III coordination diagnostic for the dynamic HandOver task.

The paper does not publish the arm-trajectory perturbation magnitude, axis,
start time, or demonstration manifest.  This runner therefore uses an explicit
local protocol and marks every result as non-comparable with the paper:

* train from five live ``BimanualHandoverItemDynamic`` demonstrations;
* start the intervention at one third of the fitted policy clock; and
* add a persistent +3 cm world-z offset to one arm's predicted EE targets.

Simulator commands are Python 3.8 compatible.  Policy fitting and serving are
delegated to the existing Python 3.10 implementation.  Data, models, and
results live below separate ``table_iii_coordination`` directories.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import random
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .direct_evaluate import (
    EVALUATION_PROTOCOL_ID,
    PolicyProcess,
    _make_action_mode,
    _noop_action,
)
from .direct_policy import _validate_published_model
from .records import atomic_json, reserve_output

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
POLICY_TASK = "bimanual_handover_item"
TASK_MODULE = "rlbench.bimanual_tasks.bimanual_handover_item_dynamic"
TASK_CLASS = "BimanualHandoverItemDynamic"
DEFAULT_DATA_ROOT = (
    INTEGRATION_ROOT / "data" / "table_iii_coordination" / "g5_seed0"
)
DEFAULT_MODELS_DIR = (
    INTEGRATION_ROOT / "models" / "v1" / "table_iii"
)
DEFAULT_RESULTS_DIR = (
    INTEGRATION_ROOT / "results" / "v1" / "table_iii_coordination"
)
DEFAULT_POLICY_PYTHON = Path(os.environ.get("DYNAMAC_POLICY_PYTHON", "python3.10"))
LOCAL_PROTOCOL_CONFIG = (
    INTEGRATION_ROOT / "configs" / "table_iii_coordination_local.json"
)
PERTURBATION_METERS = (0.0, 0.0, 0.03)
TRIGGER_FRACTION = 1.0 / 3.0


def _task_class():
    return getattr(importlib.import_module(TASK_MODULE), TASK_CLASS)


def _observation_config():
    from rlbench.observation_config import ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.gripper_open = True
    config.gripper_pose = True
    config.task_low_dim_state = True
    return config


def _collection_action_mode():
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

    return BimanualMoveArmThenGripper(
        BimanualJointPosition(),
        BimanualDiscrete(),
    )


def _episode_dir(data_root, episode):
    return (
        Path(data_root)
        / POLICY_TASK
        / "all_variations"
        / "episodes"
        / f"episode{episode}"
    )


def collect(args):
    """Collect the five local dynamic-HandOver expert demonstrations."""

    from rlbench.environment import Environment

    manifest_path = Path(args.data_root) / "collection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing collection: {manifest_path}"
        )
    environment = Environment(
        action_mode=_collection_action_mode(),
        obs_config=_observation_config(),
        headless=args.headless,
        robot_setup="dual_panda",
    )
    records = []
    launched = False
    try:
        environment.launch()
        launched = True
        task_environment = environment.get_task(_task_class())
        variations = task_environment.variation_count()
        for episode in range(args.demonstrations):
            target = _episode_dir(args.data_root, episode)
            if target.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing episode: {target}"
                )
            episode_seed = args.seed + episode
            variation = episode % variations
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            task_environment.set_variation(variation)
            demo, = task_environment.get_demos(amount=1, live_demos=True)
            target.mkdir(parents=True, exist_ok=False)
            for observation in demo:
                perception = getattr(observation, "perception_data", None)
                if isinstance(perception, dict):
                    perception.clear()
            with (target / "low_dim_obs.pkl").open("wb") as stream:
                pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
            with (target / "variation_number.pkl").open("wb") as stream:
                pickle.dump(variation, stream, protocol=pickle.HIGHEST_PROTOCOL)
            record = {
                "episode": episode,
                "seed": episode_seed,
                "variation": variation,
                "observations": len(demo),
                "path": str((target / "low_dim_obs.pkl").resolve()),
            }
            records.append(record)
            print(
                f"dynamic HandOver demo {episode + 1}/{args.demonstrations}: "
                f"{len(demo)} observations",
                flush=True,
            )
    finally:
        if launched:
            environment.shutdown()

    manifest = {
        "schema": "dynamac-table-iii-coordination-demos-v1",
        "task": "bimanual_handover_item_dynamic",
        "task_module": TASK_MODULE,
        "task_class": TASK_CLASS,
        "policy_task_alias": POLICY_TASK,
        "demonstrations": args.demonstrations,
        "base_seed": args.seed,
        "paper_cohort": False,
        "paper_comparable": False,
        "claim_boundary": (
            "Live demonstrations from the public RLBench task; the paper's "
            "demonstration/seed manifest is unpublished."
        ),
        "episodes": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")
    return manifest


def train(args):
    """Fit, authenticate, and atomically publish the coordination policy."""

    output = Path(args.models_dir) / POLICY_TASK
    with reserve_output(output):
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{POLICY_TASK}.staging-",
                dir=str(Path(args.models_dir)),
            )
        )
        try:
            summary = _train_into(args, staging)
            _validate_published_model(POLICY_TASK, staging, summary)
            os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    print(
        f"trained dynamic HandOver: left={summary['left']['durations']} "
        f"right={summary['right']['durations']}",
        flush=True,
    )
    return summary


def _train_into(args, output):
    """Fit all coordination artifacts inside one unpublished staging directory."""

    from essay2608.policy import BimanualDynaMAC

    from .demo_adapter import load_low_dim_obs_pickles, make_bimanual_demonstrations
    from .direct_policy import demonstration_paths, load_policy_config
    from .tapas_segmentation import load_rlbench_segmentation_config
    from .task_specs import get_task_spec

    protocol = json.loads(LOCAL_PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    segment_protocol = protocol["segmentation"]
    base_segmentation = load_rlbench_segmentation_config().for_task(POLICY_TASK)
    segmentation = replace(
        base_segmentation,
        boundary_selection=segment_protocol["boundary_selection"],
        expected_boundary_count=segment_protocol["expected_boundary_count"],
        provenance={
            **{
                key: value
                for key, value in base_segmentation.provenance.items()
                if key != "task_profiles"
            },
            "task_profiles": {},
            "coordination_local_protocol": segment_protocol,
        },
    )
    paths = demonstration_paths(
        Path(args.data_root), POLICY_TASK, args.demonstrations
    )
    names = [path.parent.name for path in paths]
    episodes = load_low_dim_obs_pickles(paths)
    output = Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise RuntimeError(f"coordination staging directory is not empty: {output}")
    debug_plot = output / "segmentation.png"
    converted = make_bimanual_demonstrations(
        episodes,
        get_task_spec(POLICY_TASK),
        names=names,
        config=segmentation,
        debug_plot_path=debug_plot,
    )
    policy_config = load_policy_config(Path(args.config))
    policy = BimanualDynaMAC(config=policy_config)
    policy.fit(converted.left_demonstrations, converted.right_demonstrations)
    policy.left.save(output / "left.npz")
    policy.right.save(output / "right.npz")
    summary = _training_summary(
        policy=policy,
        converted=converted,
        names=names,
        policy_config=policy_config,
        debug_plot=debug_plot,
    )
    atomic_json(output / "training.json", summary)
    return summary


def _training_summary(*, policy, converted, names, policy_config, debug_plot):
    """Build the authenticated manifest shared by staging and tests."""

    return {
        "task": POLICY_TASK,
        "bimanual": True,
        "demonstrations": names,
        "config": asdict(policy_config),
        "adapter": converted.audit,
        "left": {
            "skills": list(policy.left.skill_sequence),
            "durations": [skill.duration for skill in policy.left.skills],
            "config": asdict(policy.left.config),
            "fingerprint": policy.left.fingerprint(),
        },
        "right": {
            "skills": list(policy.right.skill_sequence),
            "durations": [skill.duration for skill in policy.right.skills],
            "config": asdict(policy.right.config),
            "fingerprint": policy.right.fingerprint(),
        },
        "segmentation": converted.segmentation.audit,
        "segmentation_debug_plot": debug_plot.name,
        "local_protocol_config": str(LOCAL_PROTOCOL_CONFIG.resolve()),
        "manifest_schema": "dynamac-direct-training-v2",
        "training_task": "bimanual_handover_item_dynamic",
        "policy_task_alias": POLICY_TASK,
        "paper_cohort": False,
        "paper_comparable": False,
        "claim_boundary": (
            "The dynamic task has the same five object-pose observation schema "
            "as HandOver, but uses locally collected randomized-handover demos."
        ),
    }


def _perturb_action(action, arm, enabled):
    result = np.asarray(action, dtype=np.float64).copy()
    if not enabled or arm == "none":
        return result
    start = 9 if arm == "left" else 0
    result[start : start + 3] += np.asarray(PERTURBATION_METERS)
    return result


def _run_episode(task_environment, worker, *, episode, seed, horizon, arm, trigger):
    from rlbench.backend.exceptions import InvalidActionError

    episode_seed = seed + episode
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    task_environment.set_variation(episode % task_environment.variation_count())
    _, observation = task_environment.reset()
    worker.request("reset", observation)
    invalid_actions = 0
    perturbed_steps = 0
    for step in range(horizon):
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            return {
                "episode": episode,
                "seed": episode_seed,
                "success": False,
                "steps": step,
                "reason": "policy_complete",
                "invalid_actions": invalid_actions,
                "perturbed_steps": perturbed_steps,
            }
        enabled = arm != "none" and step >= trigger
        command = _perturb_action(action, arm, enabled)
        perturbed_steps += int(enabled)
        try:
            observation, reward, terminate = task_environment.step(command)
        except InvalidActionError:
            invalid_actions += 1
            try:
                observation, reward, terminate = task_environment.step(
                    _noop_action(observation)
                )
            except InvalidActionError:
                return {
                    "episode": episode,
                    "seed": episode_seed,
                    "success": False,
                    "steps": step + 1,
                    "reason": "noop_failed",
                    "invalid_actions": invalid_actions,
                    "perturbed_steps": perturbed_steps,
                }
        if reward > 0.0:
            reason = "success"
        elif terminate:
            reason = "terminate"
        elif response.get("complete"):
            reason = "policy_complete"
        else:
            continue
        return {
            "episode": episode,
            "seed": episode_seed,
            "success": bool(reward > 0.0),
            "steps": step + 1,
            "reason": reason,
            "invalid_actions": invalid_actions,
            "perturbed_steps": perturbed_steps,
        }
    return {
        "episode": episode,
        "seed": episode_seed,
        "success": False,
        "steps": horizon,
        "reason": "horizon",
        "invalid_actions": invalid_actions,
        "perturbed_steps": perturbed_steps,
    }


def evaluate(args):
    """Evaluate one explicit local arm-trajectory perturbation condition."""

    output = Path(args.output)
    with reserve_output(output):
        return _evaluate_reserved(args, output)


def _evaluate_reserved(args, output):
    """Run one coordination evaluation while its result path is reserved."""

    from rlbench.environment import Environment

    action_mode = _make_action_mode()
    environment = Environment(
        action_mode=action_mode,
        obs_config=_observation_config(),
        headless=args.headless,
        robot_setup="dual_panda",
    )
    worker = PolicyProcess(
        args.policy_python,
        POLICY_TASK,
        args.models_dir,
        timeout=args.policy_timeout,
    )
    policy_steps = worker.policy_steps
    trigger = min(
        policy_steps - 1,
        int(round(TRIGGER_FRACTION * (policy_steps - 1))),
    )
    launched = False
    episodes = []
    try:
        environment.launch()
        launched = True
        task_environment = environment.get_task(_task_class())
        for episode in range(args.episodes):
            record = _run_episode(
                task_environment,
                worker,
                episode=episode,
                seed=args.seed,
                horizon=args.horizon,
                arm=args.arm,
                trigger=trigger,
            )
            episodes.append(record)
            successes = sum(int(item["success"]) for item in episodes)
            print(
                f"{args.arm} episode {episode + 1}/{args.episodes}: "
                f"{record['reason']} ({successes}/{len(episodes)})",
                flush=True,
            )
    finally:
        worker.close()
        if launched:
            environment.shutdown()

    successes = sum(int(item["success"]) for item in episodes)
    payload = {
        "schema": "dynamac-table-iii-coordination-local-v1",
        "task": "bimanual_handover_item_dynamic",
        "policy_task_alias": POLICY_TASK,
        "result_family": "bimanual_coordination",
        "scenario": (
            f"coordination_hand_{args.arm}" if args.arm != "none" else "local_baseline"
        ),
        "episodes": len(episodes),
        "episodes_requested": args.episodes,
        "episodes_completed": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "seed": args.seed,
        "horizon": args.horizon,
        "learned_policy_steps": policy_steps,
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "controller": {
            "command": "absolute_world_end_effector_pose",
            "primary_ik": "jacobian",
            "fallback_ik": "sampling",
            "sampling_trials": 100,
            "sampling_max_configs": 5,
            "sampling_max_time_ms": 10,
            "sampling_ignore_collisions": True,
            "joint_target_max_steps": 200,
            "failed_action": "one_current_pose_current_gripper_noop",
            "policy_clock_rollback": False,
            "rlbench_commit": "a51b4e609dc5c3e1a8c06046bd87a9da24723da4",
            "pyrep_commit": "b8bd1d7a3182adcd570d001649c0849047ebf197",
        },
        "model_identity": worker.model_identity,
        "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
        "invalid_actions": sum(item["invalid_actions"] for item in episodes),
        "paper_comparable": False,
        "paper_target": (
            0.97 if args.arm in {"left", "right"} else None
        ),
        "claim_boundary": (
            "Diagnostic only: the paper does not publish the perturbation axis, "
            "magnitude, start time, duration, demonstration IDs, or evaluation seeds."
        ),
        "coordination_protocol": {
            "source": "LOCAL_EXPLICIT_DIAGNOSTIC_20260815",
            "paper_comparable": False,
            "protocol_valid": True,
            "perturbed_arm": args.arm,
            "translation_world_m": list(PERTURBATION_METERS),
            "application": "persistent offset on every predicted EE target from trigger",
            "trigger_fraction_of_fitted_policy_clock": TRIGGER_FRACTION,
            "trigger_policy_step": trigger,
            "randomized_handover_location": (
                "public BimanualHandoverItemDynamic; waypoint world-z offset "
                "sampled uniformly in [-0.02, 0.02] m per episode"
            ),
        },
        "results": episodes,
    }
    atomic_json(output, payload)
    print(f"wrote {output}")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collector = subparsers.add_parser("collect")
    collector.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    collector.add_argument("--demonstrations", type=int, default=5)
    collector.add_argument("--seed", type=int, default=0)
    display = collector.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    collector.set_defaults(headless=True)

    trainer = subparsers.add_parser("train")
    trainer.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    trainer.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    trainer.add_argument(
        "--config",
        type=Path,
        default=INTEGRATION_ROOT / "configs" / "dynamac_rlbench_local.json",
    )
    trainer.add_argument("--demonstrations", type=int, default=5)

    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--arm", choices=("left", "right", "none"), required=True)
    evaluator.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    evaluator.add_argument("--episodes", type=int, default=1)
    evaluator.add_argument("--seed", type=int, default=0)
    evaluator.add_argument("--horizon", type=int, default=1000)
    evaluator.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    evaluator.add_argument("--policy-timeout", type=float, default=120.0)
    evaluator.add_argument("--output", type=Path, required=True)
    display = evaluator.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    evaluator.set_defaults(headless=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        if args.demonstrations < 1 or args.seed < 0:
            raise ValueError("demonstrations must be positive and seed non-negative")
        collect(args)
    elif args.command == "train":
        if args.demonstrations < 1:
            raise ValueError("demonstrations must be positive")
        train(args)
    else:
        if args.episodes < 1 or args.seed < 0 or args.horizon < 1:
            raise ValueError("episodes/horizon must be positive and seed non-negative")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
