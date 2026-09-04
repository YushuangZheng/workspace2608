# ruff: noqa: UP007, UP045
"""Run RACER in deterministic live RLBench episodes on server B.

The gate can exercise either the task-goal-only policy path or the released
rich VLM-to-T5 path.  It is intentionally labelled development evidence until
server A supplies the frozen Native-6 manifest and shared audit contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .official_smoke import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COMMIT,
    _git_head,
    _sha256,
)


def _latency_summary(
    values: list[float],
) -> dict[str, Optional[Union[float, int]]]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": round(float(array.mean()), 3),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
    }


def _seed_manifest(
    seed_directory: Path,
    *,
    start_episode: int,
    eval_episodes: int,
) -> tuple[dict[int, str], str]:
    per_episode: dict[int, str] = {}
    manifest_digest = hashlib.sha256()
    for episode in range(start_episode, start_episode + eval_episodes):
        seed_path = seed_directory / f"random_seed{episode}.pkl"
        if not seed_path.is_file():
            raise FileNotFoundError(seed_path)
        digest = _sha256(seed_path)
        per_episode[episode] = digest
        manifest_digest.update(f"{digest}  {seed_path.name}\n".encode())
    return per_episode, manifest_digest.hexdigest()


def _run_episode(
    *,
    env: Any,
    policy: Any,
    evaluator_class: Any,
    task_name: str,
    episode: int,
    episode_length: int,
    llava_api: Optional[Any],
    action_latencies_ms: list[float],
    vlm_latencies_ms: list[float],
) -> dict[str, Any]:
    import torch
    from racer.evaluation.utils import (
        START_ACTION,
        TEMPLATE_first_step,
        TEMPLATE_other_step,
        get_robot_delta_state,
    )

    observation, raw_observation = env.reset(episode)
    previous_raw_observation = None
    current_raw_observation = raw_observation
    last_action = START_ACTION.copy()
    instruction = None
    success = False
    error_status = "success"
    actions: list[list[float]] = []
    language_inputs: list[str] = []

    for step in range(episode_length):
        task_goal = env.task_goal
        if llava_api is not None:
            if instruction is None:
                user_message = TEMPLATE_first_step.format(task_goal=task_goal)
            else:
                robot_delta_state, _ = get_robot_delta_state(
                    previous_raw_observation, current_raw_observation
                )
                user_message = TEMPLATE_other_step.format(
                    task_goal=task_goal,
                    previous_instruction=instruction,
                    robot_delta_state=robot_delta_state,
                )
            vlm_started = time.perf_counter()
            generated = llava_api.get_response(
                user_message, image=observation["front_rgb"]
            )
            vlm_latencies_ms.append((time.perf_counter() - vlm_started) * 1000.0)
            instruction = evaluator_class.parse_vlm_instruction(generated)

        policy_input = evaluator_class.get_input_lang_str_for_policy(
            task_goal, instruction, old_version=False
        )
        torch.cuda.synchronize()
        action_started = time.perf_counter()
        action = policy.act(observation, input_lang_str=policy_input)
        torch.cuda.synchronize()
        action_latencies_ms.append((time.perf_counter() - action_started) * 1000.0)
        action = evaluator_class.action_check(action)
        action = evaluator_class.postprocess(
            task_name, action, last_action, step
        )
        transition = env.step(action)
        success = success or env.is_success()
        actions.append([float(value) for value in action])
        language_inputs.append(policy_input)
        error_status = transition.info["error_status"]
        observation = transition.observation
        if transition.info["obs"] is not None:
            previous_raw_observation = deepcopy(current_raw_observation)
            current_raw_observation = deepcopy(transition.info["obs"])
            current_raw_observation.gripper_open = action[-2]
            current_raw_observation.ignore_collisions = action[-1]
        last_action = action
        if transition.terminal or error_status == "error":
            return {
                "episode": episode,
                "task": task_name,
                "variation": 0,
                "steps": step + 1,
                "success": bool(success),
                "error_status": error_status,
                "language_goal": task_goal,
                "language_inputs": language_inputs,
                "actions": actions,
            }

    return {
        "episode": episode,
        "task": task_name,
        "variation": 0,
        "steps": episode_length,
        "success": bool(success),
        "error_status": error_status,
        "language_goal": env.task_goal,
        "language_inputs": language_inputs,
        "actions": actions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    checkpoint = args.checkpoint.resolve()
    seed_directory = args.seed_directory.resolve()
    commit = _git_head(repository)
    checkpoint_sha256 = _sha256(checkpoint)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"RACER commit mismatch: {commit}")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"RACER checkpoint checksum mismatch: {checkpoint_sha256}")
    if not seed_directory.is_dir():
        raise FileNotFoundError(seed_directory)
    seed_sha256, seed_manifest_sha256 = _seed_manifest(
        seed_directory,
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
    )

    sys.path.insert(0, str(repository))
    os.chdir(repository)

    import torch
    from racer.evaluation.llava_api.api import LlavaAPI
    from racer.evaluation.policy_agent import ModelRVTAgent
    from racer.evaluation.rollout import Evaluator
    from racer.evaluation.simulator import RLBenchSim

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    policy = ModelRVTAgent(
        model_path=str(checkpoint),
        device=args.device,
        use_full_langlen=False,
        lm_addr=args.language_address,
    )
    policy.reset()
    llava_api = LlavaAPI(args.vlm_address) if args.use_vlm else None
    env = RLBenchSim(
        task_name=args.tasks[0],
        dataset_root="",
        episode_length=args.episode_length,
        record_every_n=-1,
        unseen_task=True,
    )
    # The released environment reads these exact 25 simulator RNG states.
    # Keep the source checkout immutable by making its expected relative path
    # available through the pinned repository working directory.
    expected_seed_directory = repository / "racer/gradio_demo/random_seeds"
    if seed_directory != expected_seed_directory:
        raise ValueError(
            "RACER's immutable environment expects its released random_seeds "
            f"directory; got {seed_directory}"
        )

    action_latencies_ms: list[float] = []
    vlm_latencies_ms: list[float] = []
    episode_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for task_name in args.tasks:
            env.set_new_task(task_name)
            for episode in range(
                args.start_episode, args.start_episode + args.eval_episodes
            ):
                retries = 0
                while True:
                    record = _run_episode(
                        env=env,
                        policy=policy,
                        evaluator_class=Evaluator,
                        task_name=task_name,
                        episode=episode,
                        episode_length=args.episode_length,
                        llava_api=llava_api,
                        action_latencies_ms=action_latencies_ms,
                        vlm_latencies_ms=vlm_latencies_ms,
                    )
                    if (
                        record["error_status"] != "error"
                        or retries >= args.invalid_action_retries
                    ):
                        record["invalid_action_retries"] = retries
                        record["seed_state_sha256"] = seed_sha256[episode]
                        episode_records.append(record)
                        break
                    retries += 1
    finally:
        env.close()
    elapsed_seconds = time.perf_counter() - started

    task_success: dict[str, float] = {}
    for task_name in args.tasks:
        task_records = [
            record for record in episode_records if record["task"] == task_name
        ]
        task_success[task_name] = float(
            np.mean([record["success"] for record in task_records]) * 100.0
        )

    return {
        "status": "pass",
        "scope": (
            "live_seeded_rich_vlm_nominal_development_reproduction"
            if args.use_vlm
            else "live_seeded_task_goal_nominal_development_reproduction"
        ),
        "formal_evaluation": False,
        "formal_evaluation_reason": "requires_server_a_frozen_native6_manifest",
        "official_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "tasks": args.tasks,
        "start_episode": args.start_episode,
        "eval_episodes_per_task": args.eval_episodes,
        "episode_length": args.episode_length,
        "base_seed": args.seed,
        "seed_directory": str(seed_directory),
        "seed_manifest_sha256": seed_manifest_sha256,
        "language_address": args.language_address,
        "vlm_address": args.vlm_address if args.use_vlm else None,
        "use_vlm": args.use_vlm,
        "scores_percent": task_success,
        "episodes": episode_records,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "action_latency": _latency_summary(action_latencies_ms),
        "vlm_latency": _latency_summary(vlm_latencies_ms),
        "peak_memory_mib": round(
            torch.cuda.max_memory_allocated(args.device) / 1024**2, 3
        ),
        "device": f"cuda:{args.device}",
        "device_name": torch.cuda.get_device_name(args.device),
        "parameter_count": sum(
            parameter.numel() for parameter in policy.agent._network.parameters()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("/home/ubuntu/workspace/_external/RACER"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/_models/racer/visuomotor-rich/model_17.pth"
        ),
    )
    parser.add_argument(
        "--seed-directory",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/_external/RACER/"
            "racer/gradio_demo/random_seeds"
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=["close_jar"])
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--invalid-action-retries", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument(
        "--language-address", default="http://127.0.0.1:8000/encode/"
    )
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--vlm-address", default="http://127.0.0.1:21002")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
