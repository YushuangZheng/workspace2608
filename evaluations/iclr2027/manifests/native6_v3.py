"""Result-blind manifests for the event-grounded Native-6 E6 rerun."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from evaluations.iclr2027.native6_v3.events import FAULT_FAMILIES
from evaluations.iclr2027.native6_v3.gate import TASK_IDS


DEVELOPMENT_PER_CELL = 10
DEVELOPMENT_SEED_BASE = 2_880_000_000
FORMAL_PER_TASK = 100
PROTOCOL_REVISION = "native6_event_grounded_physics_v3"
TRIGGER_RULE = "first_eligible_physical_event"


def _task_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row["task"])
        if task in TASK_IDS:
            grouped[task].append(dict(row))
    for values in grouped.values():
        values.sort(key=lambda row: str(row["episode_id"]))
    return grouped


def build_development_rows(
    source_nominal_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build 10 result-blind development assignments per task-by-fault cell."""

    grouped = _task_groups(source_nominal_rows)
    output: list[dict[str, Any]] = []
    for task_offset, task in enumerate(TASK_IDS):
        source = grouped.get(task, [])
        if not source:
            raise ValueError(f"no source task metadata for {task}")
        template = source[0]
        variations = sorted({int(row["variation"]) for row in source})
        for fault_offset, family in enumerate(FAULT_FAMILIES):
            for index in range(DEVELOPMENT_PER_CELL):
                episode_id = f"native6_v3_development/{task}/{family}/{index:04d}"
                output.append(
                    {
                        "schema": "essay2608.iclr2027.native6-v3-assignment.v1",
                        "episode_id": episode_id,
                        "pair_id": episode_id,
                        "split": "native6_v3_development",
                        "task": task,
                        "task_level": template.get("task_level"),
                        "variation": variations[(index + fault_offset) % len(variations)],
                        "seed": (
                            DEVELOPMENT_SEED_BASE
                            + task_offset * 100_000
                            + fault_offset * 10_000
                            + index
                        ),
                        "condition": "perturbed",
                        "fault_family": family,
                        "fault_severity": "medium",
                        "trigger_rule": TRIGGER_RULE,
                        "event_ordinal": 1,
                        "horizon": int(template["horizon"]),
                        "recovery_budget": int(template["recovery_budget"]),
                        "protocol_revision": PROTOCOL_REVISION,
                        "development_only": True,
                    }
                )
    return output


def build_formal_perturbed_rows(
    legacy_native6_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse frozen initializations/fault assignments, never legacy outcomes.

    The old early/middle/late label is retained only as provenance and is not
    an input to the v3 executor.  Every v3 row uses the first eligible physical
    event, so all systems must execute these rows anew.
    """

    grouped = _task_groups(legacy_native6_rows)
    output: list[dict[str, Any]] = []
    for task in TASK_IDS:
        source = grouped.get(task, [])
        if len(source) != FORMAL_PER_TASK:
            raise ValueError(f"{task} must provide 100 frozen Native-6 rows")
        for index, row in enumerate(source):
            family = str(row.get("fault_family"))
            if family not in FAULT_FAMILIES:
                raise ValueError(f"{task} contains unsupported fault {family}")
            output.append(
                {
                    "schema": "essay2608.iclr2027.native6-v3-assignment.v1",
                    "episode_id": f"native6_v3_perturbed/{task}/{index:04d}",
                    "source_episode_id": str(row["source_episode_id"]),
                    "source_pair_id": str(row["source_pair_id"]),
                    "legacy_native6_episode_id": str(row["episode_id"]),
                    "legacy_trigger_stage_not_used": row.get("trigger_stage"),
                    "pair_id": f"native6_v3_perturbed/{task}/{index:04d}",
                    "split": "native6_v3_perturbed",
                    "task": task,
                    "task_level": row.get("task_level"),
                    "variation": int(row["variation"]),
                    "seed": int(row["seed"]),
                    "condition": "perturbed",
                    "fault_family": family,
                    "fault_severity": "medium",
                    "trigger_rule": TRIGGER_RULE,
                    "event_ordinal": 1,
                    "horizon": int(row["horizon"]),
                    "recovery_budget": int(row["recovery_budget"]),
                    "protocol_revision": PROTOCOL_REVISION,
                    "readonly_initialization_view": True,
                    "selection_policy": (
                        "reuse_frozen_native6_initializations_and_fault_assignments_"
                        "without_reading_results_v1"
                    ),
                }
            )
    return output


def validate_rows(
    development: Sequence[Mapping[str, Any]],
    formal: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(development) != 180:
        raise ValueError("Native-6 v3 development must contain 180 rows")
    if len(formal) != 600:
        raise ValueError("Native-6 v3 formal perturbed must contain 600 rows")
    for name, rows in (("development", development), ("formal", formal)):
        ids = [str(row["episode_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} contains duplicate episode IDs")
        if {str(row["task"]) for row in rows} != set(TASK_IDS):
            raise ValueError(f"{name} has the wrong task set")
        if any(row.get("protocol_revision") != PROTOCOL_REVISION for row in rows):
            raise ValueError(f"{name} contains another protocol revision")
        if any(row.get("trigger_rule") != TRIGGER_RULE for row in rows):
            raise ValueError(f"{name} contains a legacy trigger rule")
        if any(row.get("event_ordinal") != 1 for row in rows):
            raise ValueError(f"{name} contains a non-first event ordinal")
        if any("trigger_stage" in row for row in rows):
            raise ValueError(f"{name} exposes trigger_stage to the v3 runner")
    development_cells = Counter(
        (str(row["task"]), str(row["fault_family"])) for row in development
    )
    if any(
        development_cells[(task, family)] != DEVELOPMENT_PER_CELL
        for task in TASK_IDS
        for family in FAULT_FAMILIES
    ):
        raise ValueError("development is not 10 rows per task-by-fault cell")
    formal_tasks = Counter(str(row["task"]) for row in formal)
    if any(formal_tasks[task] != FORMAL_PER_TASK for task in TASK_IDS):
        raise ValueError("formal is not 100 rows per task")
    formal_faults = Counter(str(row["fault_family"]) for row in formal)
    if formal_faults != Counter({family: 200 for family in FAULT_FAMILIES}):
        raise ValueError(f"formal faults are not balanced: {formal_faults}")
    development_seeds = {int(row["seed"]) for row in development}
    formal_seeds = {int(row["seed"]) for row in formal}
    if development_seeds & formal_seeds:
        raise ValueError("development and formal seeds overlap")
    return {
        "development_rows": len(development),
        "formal_perturbed_rows": len(formal),
        "development_rows_per_task_fault": DEVELOPMENT_PER_CELL,
        "fault_counts": dict(sorted(formal_faults.items())),
        "selection_read_results": False,
        "legacy_trigger_stage_used": False,
    }


__all__ = [
    "DEVELOPMENT_PER_CELL",
    "DEVELOPMENT_SEED_BASE",
    "FORMAL_PER_TASK",
    "PROTOCOL_REVISION",
    "TRIGGER_RULE",
    "build_development_rows",
    "build_formal_perturbed_rows",
    "validate_rows",
]
