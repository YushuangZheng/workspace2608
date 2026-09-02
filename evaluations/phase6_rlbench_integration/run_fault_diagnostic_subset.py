"""Run controlled physical faults on sealed RLBench samples.

This is a component-level experiment launcher, not a replacement for the
frozen V4 formal evaluator.  Every compared method receives the same sealed
episode plans, fault trigger predicate, and stage-six executor.  The physical
fault adapter never reads or mutates policy-internal beliefs or StateIds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from integrations.rlbench.rlbench_closed_loop.eval.fault_injection import (
    FaultInjectionSpec,
    FaultInjectingTaskEnvironment,
)
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    STAGE6_IK_CONTROLLER_PROFILE,
)

from . import run_normal_diagnostic_subset as normal_subset


DEFAULT_PROTOCOL = Path(__file__).with_name("fault_protocol.json")
METHODS = (
    "dynamac_v4",
    "progress_only",
    "progress_dynamic_roles",
    "full",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cell(path: Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != (
        "essay2608.phase6_fault_protocol.v1"
    ):
        raise ValueError("unsupported phase-six fault protocol")
    cells = payload.get("cells")
    if not isinstance(cells, Mapping) or name not in cells:
        raise KeyError(f"unknown fault protocol cell: {name}")
    cell = cells[name]
    if not isinstance(cell, dict):
        raise TypeError("fault protocol cell must be an object")
    expected = {"task", "episode_indices", "fault"}
    if set(cell) != expected:
        raise ValueError(
            "fault protocol cell fields must be task/episode_indices/fault"
        )
    return payload, cell


def _install_fault(
    evaluator: ModuleType,
    spec: FaultInjectionSpec,
) -> None:
    original = evaluator._run_episode

    def run(task_environment: Any, *args: Any, **kwargs: Any):
        wrapped = FaultInjectingTaskEnvironment(task_environment, spec)
        row = original(wrapped, *args, **kwargs)
        if not isinstance(row, dict):
            raise TypeError("RLBench episode result must be an object")
        row["physical_fault"] = wrapped.protocol_metadata()
        return row

    evaluator._run_episode = run


def _method_args(method: str) -> list[str]:
    if method not in METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    if method == "dynamac_v4":
        return ["--policy-type", "dynamac"]
    return [
        "--policy-type",
        "closed_loop_multistream",
        "--closed-loop-feature-profile",
        method,
    ]


def _mark_result(
    path: Path,
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    cell_name: str,
    cell: Mapping[str, Any],
    method: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("evaluation result does not contain episode rows")
    faults = [row.get("physical_fault") for row in results]
    if any(not isinstance(value, dict) for value in faults):
        raise RuntimeError("evaluation row is missing physical fault evidence")
    triggered = sum(value.get("triggered") is True for value in faults)
    episode_indices = tuple(int(value) for value in cell["episode_indices"])
    payload["fault_evaluation"] = {
        "schema": "essay2608.phase6_fault_result.v1",
        "cell": cell_name,
        "method": method,
        "task": cell["task"],
        "episode_indices": list(episode_indices),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "fault_spec": dict(cell["fault"]),
        "executor_control": protocol["executor_control"],
        "episodes_completed": len(results),
        "episodes_fault_triggered": triggered,
        "all_completed_episodes_fault_triggered": triggered == len(results),
        "policy_internal_state_mutated_by_injector": False,
        "formal_result": False,
        "paper_comparable": False,
    }
    payload["evaluation_protocol_id"] = (
        f"{payload['evaluation_protocol_id']}+phase6-physical-fault-v1"
    )
    payload["paper_comparable"] = False
    atomic_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument(
        "--closed-loop-models-dir",
        type=Path,
        default=Path("integrations/rlbench/models/closed_loop_phase6_v1"),
    )
    parser.add_argument("--horizon", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol, cell = _load_cell(args.protocol, args.cell)
    task = str(cell["task"])
    episode_indices = normal_subset._normalized_episode_indices(cell["episode_indices"])
    evaluator = normal_subset._evaluator_for_task(task)
    evaluator._load_fixed_motion_plans = normal_subset._diagnostic_loader(
        evaluator, episode_indices
    )
    _install_fault(evaluator, FaultInjectionSpec.from_mapping(cell["fault"]))
    evaluator_args = [
        "--task",
        task,
        "--models-dir",
        "integrations/rlbench/models/v4",
        *_method_args(args.method),
        "--closed-loop-models-dir",
        str(args.closed_loop_models_dir),
        "--policy-diagnostics-dir",
        str(args.diagnostics_dir),
        "--controller-profile",
        STAGE6_IK_CONTROLLER_PROFILE,
        "--policy-python",
        str(args.policy_python),
        "--episodes",
        str(len(episode_indices)),
        "--seed",
        str(normal_subset.GLOBAL_EVAL_SEED_START + episode_indices[0]),
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
    evaluator_args.extend(normal_subset._task_protocol_args(task))
    evaluator_args.extend(
        normal_subset._episode_protocol_args(evaluator, episode_indices)
    )
    result = evaluator.main(evaluator_args)
    if result == 0:
        _mark_result(
            args.output,
            protocol_path=args.protocol,
            protocol=protocol,
            cell_name=args.cell,
            cell=cell,
            method=args.method,
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
