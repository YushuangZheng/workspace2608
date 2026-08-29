"""Contracts for the preregistered Stage-six formal matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluations.phase6_formal_evaluation import launch, run_cell, summarize
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    STAGE6_IK_CONTROLLER_PROFILE,
)
from integrations.rlbench.rlbench_closed_loop.eval.fault_injection import (
    FaultInjectionKind,
)


def test_formal_protocol_freezes_full_normal_and_balanced_fault_ranges() -> None:
    protocol = run_cell.load_protocol()
    normal = run_cell._index_range(protocol, "normal")
    fault = run_cell._index_range(protocol, "fault")

    assert normal == tuple(range(200))
    assert fault == tuple(range(50))
    assert [index % 5 for index in fault].count(0) == 10
    assert len(protocol["tasks"]) == 8
    assert tuple(protocol["methods"]) == (
        "dynamac_v4",
        "progress_only",
        "progress_dynamic_roles",
        "full",
    )
    assert protocol["statistics"]["threshold_tuning_from_fault_results"] is False
    assert (
        protocol["shared_execution"]["controller_profile"]
        == STAGE6_IK_CONTROLLER_PROFILE
    )


def test_formal_matrix_has_exact_preregistered_cell_and_episode_counts() -> None:
    protocol = run_cell.load_protocol()
    normal = launch.build_cells(protocol, "normal")
    fault = launch.build_cells(protocol, "fault")
    all_cells = launch.build_cells(protocol, "all")

    assert len(normal) == 32
    assert len(fault) == 128
    assert len(all_cells) == 160
    assert sum(cell.episodes for cell in normal) == 6400
    assert sum(cell.episodes for cell in fault) == 6400
    assert len({cell.cell_id for cell in all_cells}) == len(all_cells)
    assert len({cell.result for cell in all_cells}) == len(all_cells)


@pytest.mark.parametrize(
    ("task", "expected_arm"),
    (
        ("stack_wine", "single"),
        ("bimanual_put_bottle_in_fridge", "left"),
        ("bimanual_handover_item", "right"),
        ("bimanual_lift_tray", "left"),
        ("bimanual_sweep_to_dustpan", "left"),
    ),
)
def test_relation_fault_targets_are_frozen_outside_policy(
    task: str, expected_arm: str
) -> None:
    protocol = run_cell.load_protocol()
    mismatch = run_cell._fault_spec(protocol, task, "relation_mismatch")
    dropped = run_cell._fault_spec(protocol, task, "unexpected_drop")

    assert mismatch.kind is FaultInjectionKind.RELATION_MISMATCH
    assert dropped.kind is FaultInjectionKind.UNEXPECTED_DROP
    assert mismatch.arm == expected_arm
    assert dropped.arm == expected_arm
    assert mismatch.mismatch_translation == (0.04, 0.0, 0.0)


def test_time_stall_freezes_both_arms_without_task_specific_policy_logic() -> None:
    protocol = run_cell.load_protocol()
    for task in protocol["tasks"]:
        spec = run_cell._fault_spec(protocol, task, "time_stall")
        assert spec.kind is FaultInjectionKind.TIME_STALL
        assert spec.arm == "all"
        assert spec.duration_cycles == 12
        assert spec.motion_trigger_distance == 0.01


def test_default_eight_lane_cpu_affinity_is_disjoint_and_complete() -> None:
    affinity = launch._cpu_sets(tuple(range(8)))
    flattened = [cpu for lane in affinity for cpu in lane]

    assert len(affinity) == 8
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(range(128))
    for lane in affinity:
        cores = {cpu % 64 for cpu in lane}
        assert all({core, core + 64}.issubset(lane) for core in cores)


def test_formal_cell_paths_are_separate_from_v4_release_results() -> None:
    protocol = run_cell.load_protocol()
    repository = Path(__file__).resolve().parents[2]
    v4 = repository / "integrations/rlbench/results/v4"
    for cell in launch.build_cells(protocol, "all"):
        assert not cell.result.is_relative_to(v4)
        assert cell.result.is_relative_to(launch.RESULTS_ROOT)


def test_formal_statistics_helpers_match_known_binomial_cases() -> None:
    low, high = summarize._wilson(50, 100)
    assert low == pytest.approx(0.40383153)
    assert high == pytest.approx(0.59616847)
    assert summarize._holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
