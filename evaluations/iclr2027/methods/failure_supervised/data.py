"""Deterministic training-only views over A-frozen failure manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REQUIRED_FIELDS = frozenset({"episode_id", "task", "fault_family"})


def _task_pool(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
) -> tuple[dict[str, Any], ...]:
    if not task:
        raise ValueError("task must be non-empty")
    pool: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_index, record in enumerate(records):
        missing = REQUIRED_FIELDS.difference(record)
        if missing:
            raise ValueError(f"failure manifest line {line_index} is missing {sorted(missing)}")
        episode_id = str(record["episode_id"])
        if not episode_id:
            raise ValueError(f"failure manifest line {line_index} has an empty episode_id")
        if episode_id in seen_ids:
            raise ValueError(f"duplicate failure-train episode_id: {episode_id}")
        seen_ids.add(episode_id)
        if record["task"] == task:
            pool.append(dict(record))
    if not pool:
        raise ValueError(f"failure manifest contains no records for task {task!r}")
    return tuple(pool)


def nested_budget_views(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
    budgets: Sequence[int] = (20, 50, 100, 200),
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Use frozen manifest order so every smaller budget is a strict prefix."""

    normalized = tuple(int(value) for value in budgets)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("budgets must contain positive integers")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("budgets must be unique and strictly increasing")
    pool = _task_pool(records, task=task)
    if len(pool) < normalized[-1]:
        raise ValueError(
            f"task {task!r} has {len(pool)} records, fewer than budget {normalized[-1]}"
        )
    return {budget: pool[:budget] for budget in normalized}


def leave_one_family_out_view(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
    held_out_family: str,
) -> tuple[dict[str, Any], ...]:
    """Remove a family from M4 training metadata, without reading test views."""

    if not held_out_family:
        raise ValueError("held_out_family must be non-empty")
    pool = _task_pool(records, task=task)
    if not any(record["fault_family"] == held_out_family for record in pool):
        raise ValueError(f"held-out family {held_out_family!r} is absent from task {task!r}")
    retained = tuple(record for record in pool if record["fault_family"] != held_out_family)
    if not retained:
        raise ValueError("leave-one-family-out view would have no training records")
    return retained


__all__ = ["REQUIRED_FIELDS", "leave_one_family_out_view", "nested_budget_views"]
