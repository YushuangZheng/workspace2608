"""Validated task-set and semantic registry for the ICLR 2027 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    TaskSpec,
    load_task_specs,
)

CONFIG_ROOT = INTEGRATION_ROOT / "configs" / "iclr2027"
TASK_SPECS_PATH = CONFIG_ROOT / "tasks.json"
EXPERIMENT_REGISTRY_PATH = CONFIG_ROOT / "registry.json"
REGISTRY_SCHEMA = "essay2608.iclr2027.task-registry.v1"
REQUIRED_SETS = frozenset({"main10", "stress4", "horizon3", "native6"})
ALLOWED_FAULTS = frozenset(
    {
        "actuation_delay",
        "missed_interaction",
        "relation_loss",
        "environment_change",
        "coordination_delay",
        "composed_event",
    }
)


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    spec: TaskSpec
    base_task: str
    task_level: int | None
    fixed_base_variation: int | None
    task_sets: tuple[str, ...]
    semantic_entities: dict[str, str]
    success_source: str
    compatible_faults: tuple[str, ...]
    demonstration_source: str


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def load_experiment_registry(
    path: str | Path = EXPERIMENT_REGISTRY_PATH,
    task_specs_path: str | Path = TASK_SPECS_PATH,
) -> dict[str, ExperimentTask]:
    payload = _load_json(Path(path))
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported ICLR 2027 task registry schema")
    specs = load_task_specs(task_specs_path)
    raw_tasks = payload.get("tasks")
    raw_sets = payload.get("task_sets")
    if not isinstance(raw_tasks, dict) or not isinstance(raw_sets, dict):
        raise ValueError("registry requires task_sets and tasks objects")
    if set(raw_sets) != REQUIRED_SETS:
        raise ValueError(f"registry task sets must be {sorted(REQUIRED_SETS)}")
    tasks: dict[str, ExperimentTask] = {}
    for task_id, raw in raw_tasks.items():
        if task_id not in specs or not isinstance(raw, dict):
            raise ValueError(f"invalid task registry entry: {task_id}")
        sets = tuple(raw.get("task_sets", ()))
        if not sets or not set(sets) <= REQUIRED_SETS:
            raise ValueError(f"{task_id} has invalid task_sets")
        entities = raw.get("semantic_entities")
        if not isinstance(entities, dict) or set(entities) != set(
            specs[task_id].frame_names
        ):
            raise ValueError(f"{task_id} semantic entities must cover every task frame")
        faults = tuple(raw.get("compatible_faults", ()))
        if not faults or not set(faults) <= ALLOWED_FAULTS:
            raise ValueError(f"{task_id} has invalid compatible faults")
        success_source = raw.get("success_source")
        demo_source = raw.get("demonstration_source")
        if success_source != "rlbench_task.success" or not isinstance(demo_source, str):
            raise ValueError(f"{task_id} has invalid audit provenance")
        tasks[task_id] = ExperimentTask(
            task_id=task_id,
            spec=specs[task_id],
            base_task=str(raw["base_task"]),
            task_level=raw.get("task_level"),
            fixed_base_variation=raw.get("fixed_base_variation"),
            task_sets=sets,
            semantic_entities={str(k): str(v) for k, v in entities.items()},
            success_source=success_source,
            compatible_faults=faults,
            demonstration_source=demo_source,
        )
    if set(tasks) != set(specs):
        raise ValueError("task-spec and experiment registries must name the same tasks")
    for set_name, members in raw_sets.items():
        if not isinstance(members, list) or len(members) != len(set(members)):
            raise ValueError(f"{set_name} must be a duplicate-free task list")
        expected = {
            task_id for task_id, task in tasks.items() if set_name in task.task_sets
        }
        if set(members) != expected:
            raise ValueError(f"{set_name} membership disagrees with task entries")
    return tasks


TASKS = load_experiment_registry()


def experiment_task(task_id: str) -> ExperimentTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown ICLR 2027 task {task_id!r}") from exc


def experiment_task_set(name: str) -> tuple[ExperimentTask, ...]:
    payload = _load_json(EXPERIMENT_REGISTRY_PATH)
    try:
        task_ids = payload["task_sets"][name]
    except KeyError as exc:
        raise KeyError(f"unknown ICLR 2027 task set {name!r}") from exc
    return tuple(TASKS[task_id] for task_id in task_ids)


__all__ = [
    "CONFIG_ROOT",
    "EXPERIMENT_REGISTRY_PATH",
    "ExperimentTask",
    "REGISTRY_SCHEMA",
    "TASKS",
    "TASK_SPECS_PATH",
    "experiment_task",
    "experiment_task_set",
    "load_experiment_registry",
]
