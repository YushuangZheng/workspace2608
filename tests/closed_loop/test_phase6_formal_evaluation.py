"""Contracts for the preregistered Stage-six formal matrix."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evaluations.phase6_formal_evaluation import launch, run_cell, summarize
from evaluations.phase6_rlbench_integration import launch_sharded_normal
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
    assert protocol["shared_execution"]["base_models_dir"].endswith("phase6_v1")
    assert protocol["shared_execution"]["closed_loop_models_dir"].endswith(
        "closed_loop_phase6_v1"
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


def test_global_shard_queue_round_robins_cells_and_preserves_exact_indices() -> None:
    protocol = run_cell.load_protocol()
    cells = launch.build_cells(protocol, "normal")[:2]
    shards = launch.build_shards(cells, protocol, shard_size=4)

    assert len(shards) == 100
    assert [shard.cell.cell_id for shard in shards[:4]] == [
        cells[0].cell_id,
        cells[1].cell_id,
        cells[0].cell_id,
        cells[1].cell_id,
    ]
    for cell in cells:
        indices = tuple(
            index
            for shard in shards
            if shard.cell == cell
            for index in shard.episode_indices
        )
        assert indices == tuple(range(200))


def test_qualification_queue_round_robins_methods_and_preserves_coverage(
    tmp_path: Path,
) -> None:
    methods = ("dynamac_v4", "progress_only", "progress_dynamic_roles")
    shards = launch_sharded_normal.build_shards(
        task="wipe_desk",
        methods=methods,
        episodes=50,
        shard_size=1,
        output_root=tmp_path,
    )

    assert [shard.method for shard in shards[:6]] == [
        "dynamac_v4",
        "progress_only",
        "progress_dynamic_roles",
        "dynamac_v4",
        "progress_only",
        "progress_dynamic_roles",
    ]
    for method in methods:
        indices = tuple(
            index
            for shard in shards
            if shard.method == method
            for index in shard.indices
        )
        assert indices == tuple(range(50))


def test_run_cell_accepts_only_contiguous_sealed_episode_shards() -> None:
    allowed = tuple(range(10))
    assert run_cell._parse_episode_indices("4,5,6,7", allowed=allowed) == (
        4,
        5,
        6,
        7,
    )
    with pytest.raises(ValueError, match="contiguous"):
        run_cell._parse_episode_indices("1,3", allowed=allowed)
    with pytest.raises(ValueError, match="outside"):
        run_cell._parse_episode_indices("9,10", allowed=allowed)


def _synthetic_shard_payload(
    cell: launch.FormalCell,
    indices: tuple[int, ...],
    *,
    commit: str,
) -> dict:
    rows = [
        {
            "episode": local,
            "formal_episode_index": index,
            "formal_episode_seed": 2608000000 + index,
            "success": index % 2 == 0,
        }
        for local, index in enumerate(indices)
    ]
    successes = sum(row["success"] for row in rows)
    return {
        "schema": "synthetic-evaluator-v1",
        "release": "v4",
        "policy_type": "closed_loop_multistream",
        "closed_loop_feature_profile": "progress_only",
        "protocol_label": "test",
        "paper_comparable": True,
        "task": cell.task,
        "scenario": "static",
        "episodes": len(indices),
        "episodes_requested": len(indices),
        "episodes_completed": len(indices),
        "seed": 2608000000 + indices[0],
        "variation_schedule": [0] * len(indices),
        "horizon": 1000,
        "evaluation_protocol_id": "test-controller",
        "fixed_eval_set": {"id": "fixed"},
        "controller": {"profile": "test"},
        "model_identity": {"fingerprint": "model"},
        "successes": successes,
        "success_rate": successes / float(len(indices)),
        "episode_accounting": {
            "schema": "test-accounting",
            "planned_episode_denominator": len(indices),
            "completed_episode_count": len(indices),
            "successes_in_planned_denominator": successes,
            "success_rate_all_planned_episodes": successes / float(len(indices)),
            "trigger_reached_count": 0,
            "intervention_complete_count": 0,
            "dynamic_condition_unexercised_count": 0,
            "pre_trigger_success_count": 0,
            "complete_intervention_subset_count": 0,
            "successes_in_complete_intervention_subset": 0,
            "success_rate_in_complete_intervention_subset": None,
        },
        "ik_execution_diagnostics": {
            "attempts": len(indices),
            "residual_max": float(indices[-1]),
            "controller_profile": "test",
            "controller_config": {"fixed": True},
        },
        "gripper_protocol": {"id": "test"},
        "gripper_timing": {"id": "test"},
        "final_settling_protocol": {"id": "test"},
        "motion_plan_batch_fingerprint": "fixed",
        "protocol": {
            "planned_episode_denominator": len(indices),
            "completed_episode_count": len(indices),
            "episodes_with_complete_intervention": 0,
            "successes_in_complete_intervention_subset": 0,
            "success_rate_in_complete_intervention_subset": None,
            "all_episodes_intervened": False,
            "all_interventions_effective": None,
            "all_eligible_interventions_effective": None,
            "protocol_valid": True,
        },
        "results": rows,
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": True,
            "evidence": [{"index": index} for index in indices],
        },
        "stage6_formal_evaluation": {
            "schema": "essay2608.phase6_formal_result.v1",
            "experiment": cell.experiment,
            "task": cell.task,
            "method": cell.method,
            "fault": cell.fault,
            "episode_indices": list(indices),
            "episode_seeds": [2608000000 + index for index in indices],
            "episodes_completed": len(indices),
            "episodes_fault_triggered": None,
            "protocol_sha256": launch._sha256(run_cell.PROTOCOL),
            "git_commit": commit,
        },
    }


def test_shard_merge_is_resumable_and_rejects_incomplete_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(launch, "RESULTS_ROOT", tmp_path)
    cell = launch.FormalCell("normal", "wipe_desk", "progress_only", None, 4)
    shards = (
        launch.FormalShard(cell, (0, 1)),
        launch.FormalShard(cell, (2, 3)),
    )
    commit = "a" * 40
    for shard in shards:
        shard.result.parent.mkdir(parents=True, exist_ok=True)
        shard.result.write_text(
            json.dumps(
                _synthetic_shard_payload(
                    cell,
                    shard.episode_indices,
                    commit=commit,
                )
            ),
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        launch.merge_formal_cell(cell, shards[:1], commit=commit)

    launch.merge_formal_cell(cell, shards, commit=commit)
    merged = json.loads(cell.result.read_text(encoding="utf-8"))
    assert [row["formal_episode_index"] for row in merged["results"]] == [0, 1, 2, 3]
    assert merged["successes"] == 2
    assert merged["success_rate"] == 0.5
    assert merged["ik_execution_diagnostics"]["attempts"] == 4
    assert merged["ik_execution_diagnostics"]["residual_max"] == 3.0
    assert merged["stage6_formal_evaluation"]["shard_merge"][
        "exact_nonoverlapping_coverage"
    ] is True


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


def test_active_executor_revision_retains_no_older_formal_results() -> None:
    run_cell.load_protocol()
    assert launch._retained_records() == {}


def test_formal_statistics_helpers_match_known_binomial_cases() -> None:
    low, high = summarize._wilson(50, 100)
    assert low == pytest.approx(0.40383153)
    assert high == pytest.approx(0.59616847)
    assert summarize._holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
