"""Run a read-only normal subset from the sealed RLBench evaluation set.

The formal evaluators intentionally accept only the complete 200-episode set.
This helper selects existing plans by index without regenerating them and marks
the output as non-paper-comparable diagnostic evidence.  It supports all eight
current tasks and dispatches to the existing uni- or bimanual evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval import (
    direct_evaluate,
    unimanual_evaluate,
)
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_eval_v4 import (
    load_v4_store_intervention_protocol,
    load_v4_store_motion_source_protocol,
)
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_TASK_NAME,
)

ALL_TASKS = tuple(sorted(set(direct_evaluate.TASKS) | set(unimanual_evaluate.TASKS)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument(
        "--episode-index",
        action="append",
        type=int,
        required=True,
        help="Sealed episode index; repeat to run more than one episode.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument(
        "--closed-loop-models-dir",
        type=Path,
        default=Path("integrations/rlbench/models/closed_loop_v1"),
    )
    parser.add_argument("--horizon", type=int, default=1000)
    return parser


def _normalized_episode_indices(values: Iterable[int]) -> tuple[int, ...]:
    indices = tuple(int(value) for value in values)
    if not indices:
        raise ValueError("at least one episode index is required")
    if len(set(indices)) != len(indices):
        raise ValueError("episode indices must be unique")
    if any(not 0 <= index < FIXED_EVAL_EPISODES for index in indices):
        raise ValueError("episode index lies outside the sealed 200-episode set")
    return indices


def _evaluator_for_task(task: str) -> ModuleType:
    if task in direct_evaluate.TASKS:
        return direct_evaluate
    if task in unimanual_evaluate.TASKS:
        return unimanual_evaluate
    raise ValueError(f"unsupported normal diagnostic task: {task}")


def _task_protocol_args(task: str) -> list[str]:
    """Use the task's frozen budgets when evaluator defaults differ."""

    if task != STORE_BOTTLE_TASK_NAME:
        return []
    intervention = load_v4_store_intervention_protocol()
    motion = load_v4_store_motion_source_protocol()
    return [
        "--scenario-max-attempts",
        str(motion["goal_sampling_max_attempts"]),
        "--final-settling-steps",
        str(intervention["final_settling_physics_steps"]),
    ]


def _diagnostic_loader(evaluator: ModuleType, episode_indices: tuple[int, ...]):
    original = evaluator._load_fixed_motion_plans

    def load(args: Any):
        requested_seed = args.seed
        requested_episodes = args.episodes
        args.seed = GLOBAL_EVAL_SEED_START
        args.episodes = FIXED_EVAL_EPISODES
        try:
            manifest, selected = original(args)
        finally:
            args.seed = requested_seed
            args.episodes = requested_episodes
        plans = selected.get("plans")
        if not isinstance(plans, list) or len(plans) != FIXED_EVAL_EPISODES:
            raise RuntimeError("sealed evaluation batch is incomplete")
        subset = dict(selected)
        subset["plans"] = [plans[index] for index in episode_indices]
        return manifest, subset

    return load


def _mark_diagnostic(
    path: Path,
    *,
    task: str,
    episode_indices: tuple[int, ...],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["diagnostic_subset"] = {
        "schema": "rlbench-fixed-eval-read-only-normal-subset-v1",
        "task": task,
        "episode_indices": list(episode_indices),
        "episode_seeds": [GLOBAL_EVAL_SEED_START + value for value in episode_indices],
        "formal_result": False,
        "paper_comparable": False,
        "plan_regenerated": False,
    }
    payload["evaluation_protocol_id"] = (
        f"{payload['evaluation_protocol_id']}+normal-diagnostic-subset-v1"
    )
    payload["fixed_eval_set"][
        "formal_access"
    ] = "canonical_id_read_only_normal_diagnostic_subset"
    scenario_protocol = payload.get("scenario_protocol")
    if isinstance(scenario_protocol, dict):
        scenario_protocol["paper_comparable"] = False
    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        protocol["paper_comparable"] = False
    if "paper_comparable" in payload:
        payload["paper_comparable"] = False
    atomic_json(path, payload)


def main() -> int:
    args = build_parser().parse_args()
    episode_indices = _normalized_episode_indices(args.episode_index)
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    evaluator = _evaluator_for_task(args.task)
    evaluator._load_fixed_motion_plans = _diagnostic_loader(
        evaluator,
        episode_indices,
    )
    evaluator_args = [
        "--task",
        args.task,
        "--models-dir",
        "integrations/rlbench/models/v4",
        "--policy-type",
        "closed_loop_multistream",
        "--closed-loop-models-dir",
        str(args.closed_loop_models_dir),
        "--policy-diagnostics-dir",
        str(args.diagnostics_dir),
        "--controller-profile",
        "auto",
        "--policy-python",
        str(args.policy_python),
        "--episodes",
        str(len(episode_indices)),
        "--seed",
        str(GLOBAL_EVAL_SEED_START + episode_indices[0]),
        "--horizon",
        str(args.horizon),
        "--scenario",
        "static",
        "--eval-set-id",
        "rlbench_eval_v2",
        "--release",
        "v4",
        "--headless",
        "--output",
        str(args.output),
    ]
    evaluator_args.extend(_task_protocol_args(args.task))
    result = evaluator.main(evaluator_args)
    if result == 0:
        _mark_diagnostic(
            args.output,
            task=args.task,
            episode_indices=episode_indices,
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
