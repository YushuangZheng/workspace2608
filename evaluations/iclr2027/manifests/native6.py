"""Result-blind Native-6 views and development rows for E6.

The formal rows are deterministic subsets of the already-frozen E1 manifests.
This module never reads an episode result and never writes a Main-10 artifact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


NATIVE6_TASK_IDS = (
    "close_jar",
    "open_drawer",
    "insert_onto_square_peg",
    "place_cups_3",
    "stack_cups",
    "sweep_to_dustpan",
)
NATIVE6_FAULTS = (
    "actuation_delay",
    "missed_interaction",
    "relation_loss",
)
TRIGGER_STAGES = ("early", "middle", "late")
FORMAL_PER_TASK = 100
DEVELOPMENT_PER_CONDITION_PER_TASK = 10
DEVELOPMENT_SEED_BASE = 2_770_000_000


def _by_task(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row["task"])
        if task in NATIVE6_TASK_IDS:
            grouped[task].append(dict(row))
    for task_rows in grouped.values():
        task_rows.sort(key=lambda row: str(row["episode_id"]))
    return grouped


def _view_row(
    source: Mapping[str, Any], *, view_name: str, view_index: int
) -> dict[str, Any]:
    task = str(source["task"])
    return {
        **dict(source),
        "source_episode_id": str(source["episode_id"]),
        "source_pair_id": str(source["pair_id"]),
        "episode_id": f"{view_name}/{task}/{view_index:04d}",
        "split": view_name,
        "readonly_view": True,
        "selection_policy": (
            "result_blind_task_and_fault_stratified_deterministic_subset_v1"
        ),
    }


def build_nominal_view(
    source_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the first 100 frozen E1 rows per task without reading results."""

    grouped = _by_task(source_rows)
    output: list[dict[str, Any]] = []
    for task in NATIVE6_TASK_IDS:
        candidates = grouped.get(task, [])
        if len(candidates) < FORMAL_PER_TASK:
            raise ValueError(f"{task} has fewer than 100 E1 nominal rows")
        for index, source in enumerate(candidates[:FORMAL_PER_TASK]):
            if source.get("condition") != "nominal":
                raise ValueError(f"{task} nominal source contains a non-nominal row")
            output.append(
                _view_row(source, view_name="native6_nominal", view_index=index)
            )
    return output


def _fault_stage_quotas(task_offset: int) -> dict[tuple[str, str], int]:
    """Return per-task quotas totalling 100 and globally balancing faults.

    Every task contributes 33 rows for each fault plus one extra row.  The
    extra fault rotates across tasks, producing exactly 200 rows per family
    over Native-6.  Its trigger stage also rotates to avoid a fixed-stage bias.
    """

    extra_fault_index = task_offset % len(NATIVE6_FAULTS)
    quotas: dict[tuple[str, str], int] = {}
    for fault_index, fault in enumerate(NATIVE6_FAULTS):
        for stage in TRIGGER_STAGES:
            quotas[(fault, stage)] = 11
        if fault_index == extra_fault_index:
            extra_stage = TRIGGER_STAGES[(task_offset // 3 + fault_index) % 3]
            quotas[(fault, extra_stage)] += 1
    if sum(quotas.values()) != FORMAL_PER_TASK:
        raise AssertionError("invalid Native-6 perturbed quota table")
    return quotas


def build_perturbed_view(
    source_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select 100 permitted E1 fault rows per task using frozen strata."""

    grouped = _by_task(source_rows)
    output: list[dict[str, Any]] = []
    for task_offset, task in enumerate(NATIVE6_TASK_IDS):
        candidates = grouped.get(task, [])
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        source_order = {
            str(row["episode_id"]): index for index, row in enumerate(candidates)
        }
        for row in candidates:
            fault = row.get("fault_family")
            stage = row.get("trigger_stage")
            if fault in NATIVE6_FAULTS and stage in TRIGGER_STAGES:
                strata[(str(fault), str(stage))].append(row)
        selected: list[dict[str, Any]] = []
        for stratum, count in _fault_stage_quotas(task_offset).items():
            available = strata.get(stratum, [])
            if len(available) < count:
                raise ValueError(
                    f"{task} has {len(available)} rows for {stratum}, needs {count}"
                )
            selected.extend(available[:count])
        selected.sort(key=lambda row: source_order[str(row["episode_id"])])
        for index, source in enumerate(selected):
            if source.get("condition") != "perturbed":
                raise ValueError(f"{task} perturbed source has a non-perturbed row")
            output.append(
                _view_row(source, view_name="native6_perturbed", view_index=index)
            )
    return output


def build_development_rows(
    source_nominal_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a disjoint every-task 10+10 E6 integration manifest."""

    grouped = _by_task(source_nominal_rows)
    output: list[dict[str, Any]] = []
    for task_offset, task in enumerate(NATIVE6_TASK_IDS):
        candidates = grouped.get(task, [])
        if not candidates:
            raise ValueError(f"no source task metadata for {task}")
        template = candidates[0]
        variations = sorted({int(row["variation"]) for row in candidates})
        if not variations:
            raise ValueError(f"no frozen variation values for {task}")
        for condition_offset, condition in enumerate(("nominal", "perturbed")):
            for index in range(DEVELOPMENT_PER_CONDITION_PER_TASK):
                perturbed = condition == "perturbed"
                fault_index = (index + task_offset) % len(NATIVE6_FAULTS)
                fault = NATIVE6_FAULTS[fault_index] if perturbed else None
                stage = (
                    TRIGGER_STAGES[(index // len(NATIVE6_FAULTS) + task_offset) % 3]
                    if perturbed
                    else None
                )
                episode_id = (
                    f"native6_development/{condition}/{task}/{index:04d}"
                )
                output.append(
                    {
                        "schema": template["schema"],
                        "episode_id": episode_id,
                        "split": "native6_development",
                        "task": task,
                        "task_level": template.get("task_level"),
                        "variation": variations[index % len(variations)],
                        "seed": (
                            DEVELOPMENT_SEED_BASE
                            + task_offset * 100_000
                            + condition_offset * 50_000
                            + index
                        ),
                        "condition": condition,
                        "fault_family": fault,
                        "fault_severity": "medium" if perturbed else None,
                        "trigger_stage": stage,
                        "pair_id": episode_id,
                        "horizon": int(template["horizon"]),
                        "recovery_budget": int(template["recovery_budget"]),
                        "development_only": True,
                    }
                )
    return output


def validate_native6_rows(
    development: Sequence[Mapping[str, Any]],
    nominal: Sequence[Mapping[str, Any]],
    perturbed: Sequence[Mapping[str, Any]],
    *,
    source_nominal_ids: set[str] | None = None,
    source_perturbed_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate E6 counts, provenance, disjointness, and fixed semantics."""

    expected_tasks = set(NATIVE6_TASK_IDS)
    if len(development) != 120 or len(nominal) != 600 or len(perturbed) != 600:
        raise ValueError("Native-6 row totals must be 120 development and 600+600 formal")

    for name, rows in (
        ("development", development),
        ("nominal", nominal),
        ("perturbed", perturbed),
    ):
        tasks = {str(row["task"]) for row in rows}
        if tasks != expected_tasks:
            raise ValueError(f"{name} has the wrong task set: {sorted(tasks)}")
        episode_ids = [str(row["episode_id"]) for row in rows]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError(f"{name} contains duplicate episode IDs")

    development_counts = Counter(
        (str(row["task"]), str(row["condition"])) for row in development
    )
    for task in NATIVE6_TASK_IDS:
        if development_counts[(task, "nominal")] != 10:
            raise ValueError(f"{task} does not have 10 nominal development rows")
        if development_counts[(task, "perturbed")] != 10:
            raise ValueError(f"{task} does not have 10 perturbed development rows")

    for name, rows, condition in (
        ("nominal", nominal, "nominal"),
        ("perturbed", perturbed, "perturbed"),
    ):
        counts = Counter(str(row["task"]) for row in rows)
        if any(counts[task] != FORMAL_PER_TASK for task in NATIVE6_TASK_IDS):
            raise ValueError(f"{name} is not 100 rows per task")
        if any(str(row["condition"]) != condition for row in rows):
            raise ValueError(f"{name} contains the wrong condition")
        if any(row.get("readonly_view") is not True for row in rows):
            raise ValueError(f"{name} contains a non-read-only row")

    fault_counts = Counter(str(row["fault_family"]) for row in perturbed)
    if fault_counts != Counter({fault: 200 for fault in NATIVE6_FAULTS}):
        raise ValueError(f"formal fault counts are not balanced: {fault_counts}")
    if any(row.get("fault_severity") != "medium" for row in perturbed):
        raise ValueError("formal Native-6 contains a non-medium fault")
    if any(row.get("fault_family") is not None for row in nominal):
        raise ValueError("formal Native-6 nominal rows contain faults")
    if any(row.get("fault_family") not in NATIVE6_FAULTS for row in development if row["condition"] == "perturbed"):
        raise ValueError("Native-6 development contains a disallowed fault")

    formal_source_ids = {
        str(row["source_episode_id"]) for row in (*nominal, *perturbed)
    }
    if len(formal_source_ids) != 1200:
        raise ValueError("formal Native-6 reuses a source episode more than once")
    if source_nominal_ids is not None and not {
        str(row["source_episode_id"]) for row in nominal
    } <= source_nominal_ids:
        raise ValueError("nominal view contains a non-E1 source episode")
    if source_perturbed_ids is not None and not {
        str(row["source_episode_id"]) for row in perturbed
    } <= source_perturbed_ids:
        raise ValueError("perturbed view contains a non-E1 source episode")

    development_seeds = {int(row["seed"]) for row in development}
    formal_seeds = {int(row["seed"]) for row in (*nominal, *perturbed)}
    if development_seeds & formal_seeds:
        raise ValueError("development and formal Native-6 seeds overlap")
    if any(str(row["task"]) == "place_cups" for row in (*development, *nominal, *perturbed)):
        raise ValueError("one-cup Place Cups leaked into Native-6")

    return {
        "development_rows": len(development),
        "formal_nominal_rows": len(nominal),
        "formal_perturbed_rows": len(perturbed),
        "fault_counts": dict(sorted(fault_counts.items())),
        "task_ids": list(NATIVE6_TASK_IDS),
        "place_cups_task_id": "place_cups_3",
        "e1_results_read_by_selection": False,
    }


__all__ = [
    "DEVELOPMENT_PER_CONDITION_PER_TASK",
    "DEVELOPMENT_SEED_BASE",
    "FORMAL_PER_TASK",
    "NATIVE6_FAULTS",
    "NATIVE6_TASK_IDS",
    "build_development_rows",
    "build_nominal_view",
    "build_perturbed_view",
    "validate_native6_rows",
]
