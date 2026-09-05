"""Run the A1 real-simulator task/interface development gate.

The gate uses the frozen DynaMAC backbone and the shared stage-six executor.
It does not create paper results, calibrate closed-loop thresholds, or expose
any future sealed-test episode.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from integrations.rlbench.iclr2027.asset_audit import audit_live_task
from integrations.rlbench.iclr2027.task_registry import (
    TASKS,
    TASK_SPECS_PATH,
    experiment_task,
)
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import (
    atomic_json,
    reserve_output,
)
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
from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
    PolicyProcess,
    _controller_config,
    _make_action_mode,
    _prepare_low_dim_headless_scene,
)

SCHEMA = "essay2608.iclr2027.a1-development-gate.v1"
DEFAULT_MODELS = INTEGRATION_ROOT / "models" / "iclr2027" / "dynamac"
DEFAULT_OUTPUT_ROOT = INTEGRATION_ROOT / "results" / "iclr2027" / "a1_development_gate"
DEFAULT_SEED = 2_707_200_000
DEFAULT_POLICY_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_POLICY_PYTHON",
        "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python",
    )
)


def _run_episode(
    task_environment, worker, task, *, episode: int, horizon: int
) -> dict[str, Any]:
    from rlbench.backend.exceptions import InvalidActionError

    observation = task_environment.get_observation()
    initial_audit = audit_live_task(task_environment, task)
    worker.request("reset", observation)
    invalid_actions = 0
    joint_hold_commits = 0
    for policy_step in range(horizon):
        response = worker.request("act", observation)
        action = response.get("action")
        if action is None:
            settling = run_final_settling(
                task_environment,
                physics_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
            )
            return {
                "episode": episode,
                "success": bool(settling["success"]),
                "reason": (
                    "success_after_final_settling"
                    if settling["success"]
                    else "policy_complete_without_task_success"
                ),
                "policy_steps": policy_step,
                "invalid_actions": invalid_actions,
                "joint_hold_commits": joint_hold_commits,
                "initial_audit": initial_audit,
                "final_audit": audit_live_task(task_environment, task),
                "final_settling": settling,
            }
        transaction_id = response.get("transaction_id")
        if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
            raise RuntimeError("policy action has no valid transaction id")
        set_policy_gripper_authorization(
            task_environment, response.get("gripper_authorization")
        )
        try:
            observation, reward, terminate = task_environment.step(
                np.asarray(action, dtype=np.float64)
            )
        except InvalidActionError:
            invalid_actions += 1
            observation, reward, terminate, policy_complete = (
                commit_joint_hold_after_primary_failure(
                    task_environment,
                    worker,
                    transaction_id=transaction_id,
                )
            )
            joint_hold_commits += 1
        else:
            committed = worker.request(
                "commit",
                transaction_id=transaction_id,
                primary_action_status=policy_action_execution_status(task_environment),
                primary_action_statuses=policy_action_execution_statuses(
                    task_environment
                ),
            )
            policy_complete = bool(committed.get("complete"))
        if reward > 0.0 or terminate or policy_complete:
            settling = None
            if reward <= 0.0 and policy_complete:
                settling = run_final_settling(
                    task_environment,
                    physics_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
                )
            success = bool(reward > 0.0 or (settling or {}).get("success"))
            reason = (
                "success"
                if reward > 0.0
                else (
                    "success_after_final_settling"
                    if success
                    else (
                        "task_terminated"
                        if terminate
                        else "policy_complete_without_task_success"
                    )
                )
            )
            return {
                "episode": episode,
                "success": success,
                "reason": reason,
                "policy_steps": policy_step + 1,
                "invalid_actions": invalid_actions,
                "joint_hold_commits": joint_hold_commits,
                "initial_audit": initial_audit,
                "final_audit": audit_live_task(task_environment, task),
                **({"final_settling": settling} if settling is not None else {}),
            }
    return {
        "episode": episode,
        "success": False,
        "reason": "development_horizon",
        "policy_steps": horizon,
        "invalid_actions": invalid_actions,
        "joint_hold_commits": joint_hold_commits,
        "initial_audit": initial_audit,
        "final_audit": audit_live_task(task_environment, task),
    }


def run_task(
    task_id: str,
    *,
    episodes: int,
    seed: int,
    horizon: int,
    output: Path,
    policy_python: Path = DEFAULT_POLICY_PYTHON,
    episode_offset: int = 0,
) -> dict[str, Any]:
    task = experiment_task(task_id)
    if task.spec.bimanual:
        raise ValueError(f"{task_id} is validated through reused phase-six evidence")
    task_class = getattr(
        importlib.import_module(task.spec.module), task.spec.class_name
    )
    import rlbench.environment as environment_module
    from rlbench.observation_config import ObservationConfig

    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    action_mode = _make_action_mode(
        GLOBAL_IK_CONTROLLER_PROFILE,
        _controller_config(GLOBAL_IK_CONTROLLER_PROFILE),
    )
    environment = environment_module.Environment(
        action_mode=action_mode,
        obs_config=observation_config,
        headless=True,
    )
    restore_scene, scene_launch = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=True,
        camera_observations_requested=False,
    )
    worker = PolicyProcess(
        policy_python,
        task_id,
        DEFAULT_MODELS,
        task_specs_path=TASK_SPECS_PATH,
    )
    results = []
    launched = False
    try:
        environment.launch()
        launched = True
        variation_count = task_class(
            environment._pyrep, environment._robot
        ).variation_count()
        for local_episode in range(episodes):
            episode = episode_offset + local_episode
            episode_seed = seed + episode
            variation = episode % int(variation_count)
            task_environment, _descriptions, _observation, generation = (
                initialize_fresh_task_generation(
                    environment,
                    task_class,
                    episode_seed=episode_seed,
                    variation=variation,
                    verify_instance=False,
                )
            )
            row = _run_episode(
                task_environment,
                worker,
                task,
                episode=episode,
                horizon=horizon,
            )
            row.update(
                {
                    "seed": episode_seed,
                    "variation": variation,
                    "fresh_task_generation": generation,
                }
            )
            results.append(row)
            print(
                f"{task_id} part {local_episode + 1}/{episodes} "
                f"(episode {episode}): {row['reason']} "
                f"({sum(item['success'] for item in results)}/{len(results)})",
                flush=True,
            )
    finally:
        try:
            worker.close()
            if launched:
                environment.shutdown()
        finally:
            restore_scene()
    infrastructure_errors = sum(
        row["reason"] in {"joint_hold_failed", "infrastructure_error"}
        for row in results
    )
    payload = {
        "schema": SCHEMA,
        "status": (
            "PASS"
            if infrastructure_errors == 0 and len(results) == episodes
            else "FAIL"
        ),
        "purpose": "A1_DEVELOPMENT_ONLY_NOT_PAPER_RESULT",
        "task_id": task_id,
        "episodes": episodes,
        "seed_start": seed,
        "episode_offset": episode_offset,
        "horizon": horizon,
        "scene_launch": scene_launch,
        "policy_model_identity": worker.model_identity,
        "successes": sum(row["success"] for row in results),
        "infrastructure_errors": infrastructure_errors,
        "results": results,
    }
    with reserve_output(output):
        atomic_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    available = sorted(
        task_id for task_id, task in TASKS.items() if not task.spec.bimanual
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=available)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.episodes < 1
        or args.horizon < 1
        or args.seed < 0
        or args.episode_offset < 0
    ):
        raise ValueError("episodes/horizon must be positive and seed non-negative")
    output = args.output or DEFAULT_OUTPUT_ROOT / f"{args.task}.json"
    payload = run_task(
        args.task,
        episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        output=output,
        policy_python=args.policy_python,
        episode_offset=args.episode_offset,
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("status", "task_id", "successes", "episodes")
            }
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
