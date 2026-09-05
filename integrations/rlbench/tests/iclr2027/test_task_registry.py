from __future__ import annotations

import json

from integrations.rlbench.iclr2027.asset_audit import audit_all_task_assets
from integrations.rlbench.iclr2027.finalize_a1 import _unimanual_gate
from integrations.rlbench.iclr2027.task_registry import (
    TASKS,
    experiment_task_set,
    load_experiment_registry,
)
from essay2608.policy.tapas_segmentation import (
    _repeated_pick_place_cycles_subset,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import PolicyServer
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT


def test_experiment_task_sets_are_complete_and_frozen_size() -> None:
    assert len(load_experiment_registry()) == 17
    assert len(experiment_task_set("main10")) == 10
    assert len(experiment_task_set("stress4")) == 4
    assert len(experiment_task_set("horizon3")) == 8
    assert len(experiment_task_set("native6")) == 6


def test_main_place_cups_uses_repeated_three_cup_level() -> None:
    assert "place_cups_3" in {task.task_id for task in experiment_task_set("main10")}
    assert TASKS["place_cups_3"].task_level == 3


def test_articulated_tasks_retain_internal_configuration_fields() -> None:
    assert TASKS["open_drawer"].spec.configuration_schema == {
        "drawer": {"joint_position": 1}
    }
    assert TASKS["push_buttons_3"].spec.configuration_schema == {
        "button0": {"joint_position": 1},
        "button1": {"joint_position": 1},
        "button2": {"joint_position": 1},
    }


def test_sweep_dirt_is_scene_state_not_action_stream() -> None:
    spec = TASKS["sweep_to_dustpan"].spec
    assert spec.action_frame_names == ("broom", "dustpan")
    assert spec.scene_entity_names == ("dirt0", "dirt1", "dirt2", "dirt3", "dirt4")
    assert spec.structural_bindings == {f"dirt{index}": "dustpan" for index in range(5)}


def test_repeated_pick_place_selector_keeps_observed_relation_phases() -> None:
    assert _repeated_pick_place_cycles_subset(
        (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        gripper_row=(20, 60, 80),
        expected=7,
    ) == (10, 20, 50, 60, 70, 80, 100)


def test_frozen_a1_assets_are_readable_and_complete() -> None:
    audit = audit_all_task_assets()
    assert audit["status"] == "PASS"
    assert audit["task_count"] == 17
    assert audit["task_set_counts"] == {
        "main10": 10,
        "stress4": 4,
        "horizon3": 8,
        "native6": 6,
    }
    assert all(task["arms"] for task in audit["tasks"])


def test_generic_static_model_does_not_require_phase6_dynamic_trigger() -> None:
    task = TASKS["push_buttons_1"]
    server = PolicyServer(
        task.task_id,
        INTEGRATION_ROOT / "models" / "iclr2027" / "dynamac",
        task_spec=task.spec,
    )
    assert server.model_identity["phase6_dynamic_trigger_evidence"] is None


def test_a1_gate_separates_backbone_failure_from_infrastructure_failure(
    tmp_path,
) -> None:
    for offset in (0, 10):
        rows = [
            {
                "episode": episode,
                "seed": 1000 + episode,
                "success": False,
                "reason": "policy_complete_without_task_success",
                "invalid_actions": 2,
                "initial_audit": {
                    "task_low_dim_finite": True,
                    "success_conditions": ["ConditionSet"],
                    "relation_observation": {"single": {}},
                },
            }
            for episode in range(offset, offset + 10)
        ]
        (tmp_path / f"task_part{offset}.json").write_text(
            json.dumps(
                {
                    "schema": "essay2608.iclr2027.a1-development-gate.v1",
                    "status": "PASS",
                    "task_id": "task",
                    "results": rows,
                }
            ),
            encoding="utf-8",
        )

    gate = _unimanual_gate("task", tmp_path)
    assert gate["status"] == "PASS"
    assert gate["policy_outcome_classification"] == (
        "FROZEN_BACKBONE_LIMITATION_OBSERVED"
    )
    assert gate["invalid_actions"] == 40
