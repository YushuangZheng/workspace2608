from __future__ import annotations

import pytest

from evaluations.iclr2027.methods.failure_supervised.data import (
    leave_one_family_out_view,
    nested_budget_views,
)


def _records(count: int = 200) -> list[dict[str, object]]:
    families = ("actuation", "interaction", "relation", "environment")
    return [
        {
            "episode_id": f"drawer-{index:03d}",
            "task": "open_drawer",
            "fault_family": families[index % len(families)],
            "fault_severity": "train-only-metadata",
        }
        for index in range(count)
    ]


def test_budget_views_are_strict_frozen_prefixes() -> None:
    views = nested_budget_views(_records(), task="open_drawer")

    assert tuple(views) == (20, 50, 100, 200)
    for smaller, larger in zip((20, 50, 100), (50, 100, 200)):
        assert views[larger][:smaller] == views[smaller]


def test_leave_one_family_out_filters_training_only() -> None:
    retained = leave_one_family_out_view(
        _records(),
        task="open_drawer",
        held_out_family="relation",
    )

    assert len(retained) == 150
    assert {record["fault_family"] for record in retained} == {
        "actuation",
        "interaction",
        "environment",
    }


def test_manifest_validation_rejects_short_or_duplicate_pools() -> None:
    with pytest.raises(ValueError, match="fewer than budget"):
        nested_budget_views(_records(199), task="open_drawer")

    duplicate = _records()
    duplicate[-1]["episode_id"] = duplicate[0]["episode_id"]
    with pytest.raises(ValueError, match="duplicate"):
        nested_budget_views(duplicate, task="open_drawer")
