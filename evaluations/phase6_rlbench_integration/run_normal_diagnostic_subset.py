"""Run a read-only normal subset from the sealed RLBench evaluation set.

The formal evaluators intentionally accept only the complete 200-episode set.
This helper selects existing plans by index without regenerating them and marks
the output as non-paper-comparable diagnostic evidence.  It supports all eight
current tasks and dispatches to the existing uni- or bimanual evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
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
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    load_staged_motion_plan_batch,
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
        "--policy-type",
        choices=("dynamac", "closed_loop_multistream"),
        default="closed_loop_multistream",
        help=(
            "Policy under diagnostic comparison; auto controller selection "
            "preserves the frozen V4 executor for dynamac and uses the stage-six "
            "executor for closed_loop_multistream."
        ),
    )
    parser.add_argument(
        "--episode-index",
        action="append",
        type=int,
        required=True,
        help=(
            "Sealed episode index; repeat in consecutive ascending order to "
            "run a batch."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument(
        "--closed-loop-models-dir",
        type=Path,
        default=Path("integrations/rlbench/models/closed_loop_phase6_v1"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("integrations/rlbench/models/phase6_v1"),
    )
    parser.add_argument(
        "--motion-plans",
        type=Path,
        default=None,
        help="Optional authenticated fixed qualification batch.",
    )
    parser.add_argument("--controller-profile", default="auto")
    parser.add_argument(
        "--closed-loop-feature-profile",
        choices=("progress_only", "progress_dynamic_roles", "full"),
        default="full",
        help="Closed-loop ablation profile used by this diagnostic subset.",
    )
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--post-success-policy-steps",
        type=int,
        default=0,
        help=(
            "Diagnostic-only extra policy cycles after RLBench first reports "
            "success; this never changes the latched benchmark outcome."
        ),
    )
    return parser


def _normalized_episode_indices(
    values: Iterable[int], *, total_episodes: int = FIXED_EVAL_EPISODES
) -> tuple[int, ...]:
    indices = tuple(int(value) for value in values)
    if not indices:
        raise ValueError("at least one episode index is required")
    if len(set(indices)) != len(indices):
        raise ValueError("episode indices must be unique")
    if any(not 0 <= index < total_episodes for index in indices):
        raise ValueError("episode index lies outside the sealed 200-episode set")
    expected = tuple(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(
            "multiple episode indices must be consecutive and ascending because "
            "the RLBench evaluator advances the formal episode seed by one"
        )
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


def _episode_protocol_args(
    evaluator: ModuleType,
    episode_indices: tuple[int, ...],
) -> list[str]:
    """Keep a direct-evaluation subset on its sealed variation schedule."""

    if evaluator is direct_evaluate:
        return ["--episode-variation-offset", str(episode_indices[0])]
    return []


def _diagnostic_loader(
    evaluator: ModuleType,
    episode_indices: tuple[int, ...],
    *,
    motion_plans: Path | None = None,
):
    if motion_plans is not None:
        payload = json.loads(motion_plans.read_text(encoding="utf-8"))
        plans = load_staged_motion_plan_batch(payload)
        fingerprint = hashlib.sha256(motion_plans.read_bytes()).hexdigest()

        def load_qualification(_args: Any):
            return (
                {
                    "payload": {
                        "evaluation_set_id": "local_fixed_qualification",
                        "spec": {"sha256": fingerprint},
                        "environment_plan_batches": {
                            payload["task_name"]: {"sha256": fingerprint}
                        },
                    },
                    "manifest_sha256": fingerprint,
                },
                {
                    "payload": payload,
                    "plans": [plans[index] for index in episode_indices],
                    "formal_access": "explicit_fixed_qualification_read_only",
                },
            )

        return load_qualification
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


def _install_post_success_continuation(
    evaluator: ModuleType,
    requested_steps: int,
) -> None:
    if requested_steps == 0:
        return
    if requested_steps < 0:
        raise ValueError("post-success policy steps must be non-negative")
    if evaluator is not direct_evaluate:
        raise ValueError(
            "post-success policy continuation currently supports bimanual direct tasks"
        )
    original = evaluator._run_episode

    def run(*args: Any, **kwargs: Any):
        kwargs["post_success_policy_steps"] = requested_steps
        return original(*args, **kwargs)

    evaluator._run_episode = run


def _mark_diagnostic(
    path: Path,
    *,
    task: str,
    episode_indices: tuple[int, ...],
    base_seed: int = GLOBAL_EVAL_SEED_START,
    qualification_batch: Path | None = None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostic_subset = {
        "schema": "rlbench-fixed-eval-read-only-normal-subset-v1",
        "task": task,
        "episode_indices": list(episode_indices),
        "episode_seeds": [base_seed + value for value in episode_indices],
        "formal_result": False,
        "paper_comparable": False,
        "plan_regenerated": False,
    }
    if qualification_batch is not None:
        diagnostic_subset["qualification_batch"] = {
            "path": str(qualification_batch),
            "sha256": hashlib.sha256(qualification_batch.read_bytes()).hexdigest(),
        }
    payload["diagnostic_subset"] = diagnostic_subset
    payload["evaluation_protocol_id"] = (
        f"{payload['evaluation_protocol_id']}+normal-diagnostic-subset-v1"
    )
    payload["fixed_eval_set"]["formal_access"] = (
        "canonical_id_read_only_normal_diagnostic_subset"
        if qualification_batch is None
        else "explicit_fixed_qualification_read_only"
    )
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
    qualification_payload = None
    if args.motion_plans is not None:
        qualification_payload = json.loads(args.motion_plans.read_text(encoding="utf-8"))
        load_staged_motion_plan_batch(qualification_payload)
        if qualification_payload.get("task_name") != args.task:
            raise ValueError("qualification batch task does not match")
    total_episodes = (
        FIXED_EVAL_EPISODES
        if qualification_payload is None
        else int(qualification_payload["episodes"])
    )
    base_seed = (
        GLOBAL_EVAL_SEED_START
        if qualification_payload is None
        else int(qualification_payload["base_seed"])
    )
    episode_indices = _normalized_episode_indices(
        args.episode_index,
        total_episodes=total_episodes,
    )
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    evaluator = _evaluator_for_task(args.task)
    if args.policy_type != "closed_loop_multistream" and args.post_success_policy_steps:
        raise ValueError(
            "post-success continuation is only defined for closed-loop policy"
        )
    if (
        args.policy_type != "closed_loop_multistream"
        and args.closed_loop_feature_profile != "full"
    ):
        raise ValueError(
            "closed-loop feature profiles are only defined for closed-loop policy"
        )
    _install_post_success_continuation(
        evaluator,
        int(args.post_success_policy_steps),
    )
    evaluator._load_fixed_motion_plans = _diagnostic_loader(
        evaluator,
        episode_indices,
        motion_plans=args.motion_plans,
    )
    evaluator_args = [
        "--task",
        args.task,
        "--models-dir",
        str(args.models_dir),
        "--policy-type",
        args.policy_type,
        "--closed-loop-models-dir",
        str(args.closed_loop_models_dir),
        "--closed-loop-feature-profile",
        str(args.closed_loop_feature_profile),
        "--policy-diagnostics-dir",
        str(args.diagnostics_dir),
        "--controller-profile",
        str(args.controller_profile),
        "--policy-python",
        str(args.policy_python),
        "--episodes",
        str(len(episode_indices)),
        "--seed",
        str(base_seed + episode_indices[0]),
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
    evaluator_args.extend(_episode_protocol_args(evaluator, episode_indices))
    if args.motion_plans is not None:
        evaluator_args.extend(("--motion-plans", str(args.motion_plans)))
    result = evaluator.main(evaluator_args)
    if result == 0:
        _mark_diagnostic(
            args.output,
            task=args.task,
            episode_indices=episode_indices,
            base_seed=base_seed,
            qualification_batch=args.motion_plans,
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
