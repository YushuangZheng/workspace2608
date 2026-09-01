"""Run one preregistered Stage-six normal or controlled-fault cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from integrations.rlbench.rlbench_closed_loop.eval.fault_injection import (
    FaultInjectionSpec,
    FaultInjectingTaskEnvironment,
)
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    STAGE6_IK_CONTROLLER_PROFILE,
    Stage6IKControllerConfig,
)
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

PROTOCOL = Path(__file__).with_name("protocol.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_tree_fingerprint(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"formal model tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(REPOSITORY_ROOT).as_posix()
        digest.update(f"{name}\0{_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    """Return whether *path* is contained by *root* on Python 3.8+."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != (
        "essay2608.phase6_formal_protocol.v1"
    ):
        raise ValueError("unsupported Stage-six formal protocol")
    expected_sections = {
        "schema",
        "status",
        "active_implementation",
        "evaluation_set",
        "tasks",
        "methods",
        "shared_execution",
        "faults",
        "statistics",
        "resource_plan",
        "claims",
    }
    if set(value) != expected_sections:
        raise ValueError("formal protocol contains obsolete or unknown sections")
    if value.get("status") != "preregistered_active":
        raise ValueError("formal protocol is not the active preregistered protocol")
    active = value.get("active_implementation")
    expected_active_fields = {
        "controller_profile",
        "protocol_id",
        "validated_mechanism_parent_commit",
        "retained_formal_results_before_activation",
    }
    if not isinstance(active, dict) or set(active) != expected_active_fields:
        raise ValueError("formal active implementation identity is invalid")
    mechanism_commit = active.get("validated_mechanism_parent_commit")
    if (
        not isinstance(mechanism_commit, str)
        or len(mechanism_commit) != 40
        or any(character not in "0123456789abcdef" for character in mechanism_commit)
    ):
        raise ValueError("formal mechanism commit identity is invalid")
    if active.get("retained_formal_results_before_activation") is not False:
        raise ValueError("formal protocol cannot retain pre-activation results")
    shared = value.get("shared_execution")
    expected_protocol_id = Stage6IKControllerConfig().protocol_id
    if (
        not isinstance(shared, dict)
        or shared.get("controller_profile") != STAGE6_IK_CONTROLLER_PROFILE
        or shared.get("protocol_id") != expected_protocol_id
        or active.get("controller_profile") != shared.get("controller_profile")
        or active.get("protocol_id") != shared.get("protocol_id")
    ):
        raise ValueError("formal active implementation and shared executor differ")
    for field in ("base_models_dir", "closed_loop_models_dir"):
        model_root = shared.get(field)
        if not isinstance(model_root, str) or Path(model_root).is_absolute():
            raise ValueError(f"formal {field} must be a repository-relative path")
        resolved = (REPOSITORY_ROOT / model_root).resolve()
        if not _is_within(resolved, REPOSITORY_ROOT) or not resolved.is_dir():
            raise ValueError(f"formal {field} does not identify a model directory")
        fingerprint_field = field.replace("_dir", "_fingerprint")
        expected_fingerprint = shared.get(fingerprint_field)
        if (
            not isinstance(expected_fingerprint, str)
            or len(expected_fingerprint) != 64
            or _model_tree_fingerprint(resolved) != expected_fingerprint
        ):
            raise ValueError(f"formal {field} content differs from the frozen model")
    tasks = value.get("tasks")
    methods = value.get("methods")
    faults = value.get("faults")
    if not isinstance(tasks, list) or len(tasks) != 8 or len(set(tasks)) != 8:
        raise ValueError("formal protocol must contain eight unique tasks")
    if not isinstance(methods, dict) or tuple(methods) != (
        "dynamac_v4",
        "progress_only",
        "progress_dynamic_roles",
        "full",
    ):
        raise ValueError("formal method order or inventory is invalid")
    if not isinstance(faults, dict) or tuple(faults) != (
        "time_stall",
        "grasp_failure",
        "relation_mismatch",
        "unexpected_drop",
    ):
        raise ValueError("formal fault order or inventory is invalid")
    evaluation = value.get("evaluation_set", {})
    if evaluation.get("normal_episode_index_range") != [0, 199]:
        raise ValueError("normal formal range must contain all 200 episodes")
    if evaluation.get("fault_episode_index_range") != [0, 49]:
        raise ValueError("fault formal range must be the preregistered first 50")
    return value


def _evaluator(task: str) -> ModuleType:
    if task in direct_evaluate.TASKS:
        return direct_evaluate
    if task in unimanual_evaluate.TASKS:
        return unimanual_evaluate
    raise ValueError(f"unsupported formal task: {task}")


def _index_range(protocol: Mapping[str, Any], experiment: str) -> tuple[int, ...]:
    key = f"{experiment}_episode_index_range"
    bounds = protocol["evaluation_set"][key]
    start, stop = (int(value) for value in bounds)
    if not 0 <= start <= stop < FIXED_EVAL_EPISODES:
        raise ValueError("formal episode range lies outside sealed evaluation set")
    return tuple(range(start, stop + 1))


def _parse_episode_indices(
    value: str | None,
    *,
    allowed: tuple[int, ...],
) -> tuple[int, ...]:
    """Parse one deterministic contiguous shard inside the sealed range."""

    if value is None:
        return allowed
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("episode indices must be comma-separated integers") from error
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("episode shard must contain unique indices")
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        raise ValueError("episode shard indices must be sorted and contiguous")
    if any(index not in allowed for index in indices):
        raise ValueError("episode shard lies outside the sealed formal range")
    return indices


def _install_preregistered_subset(
    evaluator: ModuleType,
    episode_indices: tuple[int, ...],
) -> None:
    if episode_indices == tuple(range(FIXED_EVAL_EPISODES)):
        return
    original = evaluator._load_fixed_motion_plans

    def load(args: Any):
        requested_seed, requested_episodes = args.seed, args.episodes
        args.seed, args.episodes = GLOBAL_EVAL_SEED_START, FIXED_EVAL_EPISODES
        try:
            manifest, selected = original(args)
        finally:
            args.seed, args.episodes = requested_seed, requested_episodes
        plans = selected.get("plans")
        if not isinstance(plans, list) or len(plans) != FIXED_EVAL_EPISODES:
            raise RuntimeError("sealed evaluation batch is incomplete")
        subset = dict(selected)
        subset["plans"] = [plans[index] for index in episode_indices]
        return manifest, subset

    evaluator._load_fixed_motion_plans = load


def _fault_spec(
    protocol: Mapping[str, Any], task: str, fault: str
) -> FaultInjectionSpec:
    configured = protocol["faults"][fault]
    arm_by_task = configured.get("arm_by_task")
    if not isinstance(arm_by_task, dict) or set(arm_by_task) != set(protocol["tasks"]):
        raise ValueError(f"{fault} must bind an arm for every formal task")
    fields = {key: value for key, value in configured.items() if key != "arm_by_task"}
    fields.update({"kind": fault, "arm": arm_by_task[task]})
    return FaultInjectionSpec.from_mapping(fields)


def _install_fault(evaluator: ModuleType, spec: FaultInjectionSpec) -> None:
    original = evaluator._run_episode

    def run(task_environment: Any, *args: Any, **kwargs: Any):
        wrapped = FaultInjectingTaskEnvironment(task_environment, spec)
        row = original(wrapped, *args, **kwargs)
        if not isinstance(row, dict):
            raise TypeError("RLBench episode result must be an object")
        row["physical_fault"] = wrapped.protocol_metadata()
        return row

    evaluator._run_episode = run


def _task_protocol_args(task: str) -> list[str]:
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


def _method_args(protocol: Mapping[str, Any], method: str) -> list[str]:
    configured = protocol["methods"][method]
    result = ["--policy-type", str(configured["policy_type"])]
    if configured["feature_profile"] is not None:
        result.extend(
            ["--closed-loop-feature-profile", str(configured["feature_profile"])]
        )
    return result


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mark_result(
    path: Path,
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    experiment: str,
    task: str,
    method: str,
    fault: str | None,
    episode_indices: tuple[int, ...],
    fault_spec: FaultInjectionSpec | None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != len(episode_indices):
        raise RuntimeError("formal result episode accounting is incomplete")
    for local_index, (row, formal_index) in enumerate(zip(rows, episode_indices)):
        if not isinstance(row, dict):
            raise RuntimeError("formal episode result must be an object")
        row["shard_local_episode"] = local_index
        row["formal_episode_index"] = formal_index
        row["formal_episode_seed"] = GLOBAL_EVAL_SEED_START + formal_index
    triggered = None
    if fault_spec is not None:
        evidence = [row.get("physical_fault") for row in rows]
        if any(not isinstance(value, dict) for value in evidence):
            raise RuntimeError("formal fault result lacks physical evidence")
        triggered = sum(value.get("triggered") is True for value in evidence)
    payload["stage6_formal_evaluation"] = {
        "schema": "essay2608.phase6_formal_result.v1",
        "formal_result": True,
        "paper_comparable": True,
        "experiment": experiment,
        "task": task,
        "method": method,
        "fault": fault,
        "fault_spec": None if fault_spec is None else fault_spec.to_dict(),
        "episode_indices": list(episode_indices),
        "episode_seeds": [GLOBAL_EVAL_SEED_START + value for value in episode_indices],
        "episodes_completed": len(rows),
        "episodes_fault_triggered": triggered,
        "protocol_path": str(protocol_path.resolve().relative_to(REPOSITORY_ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "git_commit": _git_head(),
        "shared_controller_profile": protocol["shared_execution"]["controller_profile"],
        "policy_internal_state_mutated_by_fault_injector": False,
        "episode_selection_based_on_results": False,
    }
    payload["paper_comparable"] = True
    atomic_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--experiment", choices=("normal", "fault"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fault", default=None)
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Contiguous comma-separated subset of the sealed formal range.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.task not in protocol["tasks"] or args.method not in protocol["methods"]:
        raise ValueError("task or method is outside the frozen formal protocol")
    if args.experiment == "normal" and args.fault is not None:
        raise ValueError("normal formal cell cannot contain a fault")
    if args.experiment == "fault" and args.fault not in protocol["faults"]:
        raise ValueError("fault formal cell requires a frozen fault kind")
    episode_indices = _parse_episode_indices(
        args.episode_indices,
        allowed=_index_range(protocol, args.experiment),
    )
    evaluator = _evaluator(args.task)
    _install_preregistered_subset(evaluator, episode_indices)
    spec = None
    if args.experiment == "fault":
        spec = _fault_spec(protocol, args.task, args.fault)
        _install_fault(evaluator, spec)
    shared = protocol["shared_execution"]
    evaluator_args = [
        "--task",
        args.task,
        "--models-dir",
        str(shared["base_models_dir"]),
        *_method_args(protocol, args.method),
        "--closed-loop-models-dir",
        str(shared["closed_loop_models_dir"]),
        "--policy-diagnostics-dir",
        str(args.diagnostics_dir),
        "--controller-profile",
        str(shared["controller_profile"]),
        "--policy-python",
        str(args.policy_python),
        "--episodes",
        str(len(episode_indices)),
        "--seed",
        str(GLOBAL_EVAL_SEED_START + episode_indices[0]),
        "--horizon",
        str(shared["horizon"]),
        "--policy-timeout",
        str(shared["policy_timeout_seconds"]),
        "--scenario",
        str(shared["scenario"]),
        "--eval-set-id",
        str(protocol["evaluation_set"]["id"]),
        "--release",
        "v4",
        "--headless",
        "--output",
        str(args.output),
    ]
    if evaluator is unimanual_evaluate:
        evaluator_args.extend(["--variation", "0"])
    else:
        evaluator_args.extend(["--episode-variation-offset", str(episode_indices[0])])
    evaluator_args.extend(_task_protocol_args(args.task))
    result = evaluator.main(evaluator_args)
    if result == 0:
        _mark_result(
            args.output,
            protocol_path=args.protocol,
            protocol=protocol,
            experiment=args.experiment,
            task=args.task,
            method=args.method,
            fault=args.fault,
            episode_indices=episode_indices,
            fault_spec=spec,
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
