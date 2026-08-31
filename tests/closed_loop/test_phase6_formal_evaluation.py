"""Contracts for the preregistered Stage-six formal matrix."""

from __future__ import annotations

import json
from copy import deepcopy
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
    assert protocol["resource_plan"]["parallel_lanes"] == 48
    assert (
        protocol["shared_execution"]["controller_profile"]
        == STAGE6_IK_CONTROLLER_PROFILE
    )


def test_formal_protocol_contains_only_the_active_runtime_identity() -> None:
    protocol = run_cell.load_protocol()

    assert protocol["status"] == "preregistered_active"
    assert not any(
        key.startswith("pre_result_amendment")
        or key.startswith("implementation_correction_after_invalidated")
        for key in protocol
    )
    assert protocol["active_implementation"]["controller_profile"] == (
        protocol["shared_execution"]["controller_profile"]
    )
    assert protocol["active_implementation"]["protocol_id"] == (
        protocol["shared_execution"]["protocol_id"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("obsolete_section", "obsolete or unknown"),
        ("controller_mismatch", "shared executor differ"),
        ("protocol_mismatch", "shared executor differ"),
    ),
)
def test_formal_protocol_rejects_old_or_mixed_runtime_identities(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    protocol = deepcopy(run_cell.load_protocol())
    if mutation == "obsolete_section":
        protocol["pre_result_amendment"] = {"obsolete": True}
    elif mutation == "controller_mismatch":
        protocol["active_implementation"]["controller_profile"] = "old_executor"
    else:
        protocol["active_implementation"]["protocol_id"] = "old_protocol"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_cell.load_protocol(path)


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


def test_default_48_worker_cpu_affinity_is_disjoint_and_complete() -> None:
    specs = launch._lane_specs(tuple(range(8)), 48)
    affinity = [spec.logical_cpus for spec in specs]
    flattened = [cpu for lane in affinity for cpu in lane]

    assert len(affinity) == 48
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


def test_retained_formal_results_are_explicit_content_addressed_and_exclude_handover_closed_loop() -> None:
    protocol = run_cell.load_protocol()
    cells = {cell.cell_id: cell for cell in launch.build_cells(protocol, "normal")}
    records = launch._retained_records()

    assert "normal/bimanual_handover_item/dynamac_v4" in records
    assert "normal/bimanual_handover_item/progress_only" not in records
    assert "normal/bimanual_handover_item/progress_dynamic_roles" not in records
    assert "normal/bimanual_handover_item/full" not in records
    for cell_id in records:
        launch._validate_retained_result(cells[cell_id])


def test_formal_statistics_helpers_match_known_binomial_cases() -> None:
    low, high = summarize._wilson(50, 100)
    assert low == pytest.approx(0.40383153)
    assert high == pytest.approx(0.59616847)
    assert summarize._holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
