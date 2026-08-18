from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import paper_comparison
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    evaluation_protocol_id as bimanual_evaluation_protocol_id,
)
from integrations.rlbench.rlbench_dynamac.direct_policy import V3_ADAPTER_PROTOCOL
from integrations.rlbench.rlbench_dynamac.eval_set import (
    INTEGRATION_ROOT,
    validate_formal_artifact_paths,
)
from integrations.rlbench.rlbench_dynamac.paper_comparison import (
    EXPECTED_RELEASE_CONFIGS,
    EXPECTED_TAPAS_COMMIT,
    LocalRun,
    _model_identity_rank,
    _valid_v3_dynamic_protocol,
    _valid_v3_task_semantic_signature,
)
from integrations.rlbench.rlbench_dynamac.runtime import (
    CROSS_INITIALIZATION_JOINT_TOLERANCE,
    CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA,
    CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
    CROSS_INITIALIZATION_SCALAR_TOLERANCE,
    CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
    FORMAL_INTERVENTION_COLLISION_PAIR_POLICY,
    FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA,
    FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
    FRESH_TASK_GENERATION_PROTOCOL_ID,
    QUATERNION_ROTATION_METRIC,
    ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
    ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
    STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
    TASK_SEMANTIC_SIGNATURE_SCHEMA,
    TASK_TREE_STATE_SCHEMA,
    ScenarioController,
    StagedMotionPlan,
    _canonical_json_fingerprint,
    _compact_task_tree_comparison,
    _compare_task_tree_relative_state,
    _staging_source_fingerprint,
    final_settling_metadata,
    staged_motion_plan_batch,
)
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    EXPECTED_UNIMANUAL_BASE_SCENE_SHA256,
    EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT,
    LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
    evaluation_protocol_id,
)
from integrations.rlbench.rlbench_dynamac.v3_protocol import (
    V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA,
    V3_INTERVENTION_SCHEMA,
    build_v3_trigger_anchor_evidence,
    checkpoint_trigger_audit,
    dynamic_trigger_profile,
    load_v3_intervention_protocol,
    load_v3_motion_source_protocol,
    resolve_authenticated_v3_trigger,
)


def _fresh_generation_evidence(
    *,
    generation_index,
    episode_seed,
    variation,
    task_name,
    previous_task_present=False,
):
    body = {
        "schema": FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
        "protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
        "generation_index": generation_index,
        "episode_seed": episode_seed,
        "variation": variation,
        "task_name": task_name,
        "physics_running_before_stop": previous_task_present,
        "physics_stopped_before_task_reload": True,
        "previous_task_present": previous_task_present,
        "previous_task_unloaded_before_stop": previous_task_present,
        "previous_task_unloaded_while_physics_running": previous_task_present,
        "scene_task_absent_before_stop": True,
        "task_model_loaded_fresh": True,
        "fresh_task_python_instance_created": True,
        "task_model_only_reloaded": True,
        "base_scene_reloaded": False,
        "physics_started_by_task_environment": True,
        "rng_seeded_after_reload_immediately_before_reset": True,
        "variation_set_after_seed_before_reset": True,
        "task_environment_reset_calls": 1,
        "reset_verify_instance": True,
    }
    return {**body, "fingerprint": _canonical_json_fingerprint(body)}


def _stack_policy(*, break_required_window: bool = False):
    duration = 72
    raw = np.zeros(duration, dtype=bool)
    raw[68:] = True
    if break_required_window:
        raw[:40] = True
        raw[67:] = True
    gate_enabled = float(np.mean(raw)) > 0.5
    availability = ~raw if gate_enabled else np.ones(duration, dtype=bool)
    selected = np.asarray([True])
    stream = SimpleNamespace(
        availability=availability[None, :],
        active=availability[None, :] & selected[:, None],
        selected_by_eq6=selected,
    )
    skill = SimpleNamespace(
        label=0,
        duration=duration,
        streams={"wine_bottle": stream},
        link_diagnostics={"wine_bottle": {"raw_link_mask": raw.tolist()}},
    )
    return SimpleNamespace(
        skills=[skill],
        skill_sequence=(0,),
        config=SimpleNamespace(
            link_mask_scope="skill_majority_gate_timestep",
            link_filter="none",
        ),
        selection_semantics_id=(
            "eq5_skill_majority_gate_timestep_availability_before_eq6_and_poe_"
            "time_state_position3d_unimodal_v1"
        ),
    )


def test_v3_semantic_signature_validator_is_recursive_and_total() -> None:
    detected = {
        "type": "rlbench.backend.conditions.DetectedCondition",
        "structural_fields": {
            "_obj": {"type": "pyrep.objects.shape.Shape", "name": "cup"},
            "_detector": {
                "type": "pyrep.objects.proximity_sensor.ProximitySensor",
                "name": "success",
            },
            "_negated": False,
        },
        "excluded_runtime_progress_fields": [],
    }
    signature = {
        "schema": TASK_SEMANTIC_SIGNATURE_SCHEMA,
        "task_class": "rlbench.tasks.place_cups.PlaceCups",
        "success_conditions": [
            {
                "type": "rlbench.backend.conditions.OrConditions",
                "structural_fields": {"_conditions": [detected]},
                "excluded_runtime_progress_fields": [
                    "_current_condition_index"
                ],
            }
        ],
        "fail_conditions": [],
        "graspable_objects": [
            {"type": "pyrep.objects.shape.Shape", "name": "cup"}
        ],
    }
    assert _valid_v3_task_semantic_signature(signature) is True

    malformed = copy.deepcopy(signature)
    malformed["success_conditions"] = None
    assert _valid_v3_task_semantic_signature(malformed) is False

    nested_forgery = copy.deepcopy(signature)
    nested_forgery["success_conditions"][0]["structural_fields"][
        "_conditions"
    ] = ["forged"]
    assert _valid_v3_task_semantic_signature(nested_forgery) is False

    generic_top_level = copy.deepcopy(signature)
    generic_top_level["success_conditions"] = [
        {"type": "forged.Condition", "fields": {}}
    ]
    assert _valid_v3_task_semantic_signature(generic_top_level) is False


def test_v3_intervention_registry_freezes_integer_ticks_and_phase_formula() -> None:
    protocol = load_v3_intervention_protocol()

    assert protocol["schema"] == V3_INTERVENTION_SCHEMA
    assert len(protocol["fingerprint"]) == 64
    assert "staging_max_attempts" not in protocol
    assert protocol["provenance"]["frozen_before_v3_formal_evaluation"] is True
    assert (
        protocol["provenance"]["model_weights_retrained_for_trigger_change"]
        is False
    )
    assert protocol["provenance"]["manifests_reauthenticated"] is True
    assert set(protocol["dynamic_environment"]) == {
        "stack_wine",
        "place_cups",
        "open_microwave",
        "wipe_desk",
        "bimanual_put_bottle_in_fridge",
        "bimanual_handover_item",
        "bimanual_lift_tray",
        "bimanual_sweep_to_dustpan",
    }
    for profile in protocol["dynamic_environment"].values():
        assert profile["interaction_arm"] in {"single", "left", "right"}
        assert profile["interaction_object"]
        assert profile["interaction_event"]
        assert profile["expected_gripper_state"] in {"open", "closed"}
        assert profile["phase"] == pytest.approx(
            profile["local_tick"] / (profile["expected_duration"] - 1),
            abs=1.0e-15,
        )
        assert profile["required_active_window"] == [
            profile["local_tick"],
            profile["local_tick"] + protocol["smooth_steps"] - 1,
        ]
    handover = protocol["dynamic_environment"]["bimanual_handover_item"]
    assert handover["interaction_arm"] == "left"
    assert handover["interaction_object"] == "item0"
    assert handover["expected_gripper_state"] == "open"
    assert handover["local_tick"] == 50
    assert handover["required_active_window"] == [50, 59]
    coordination = protocol["coordination"]
    for scenario, perturbed_arm in (
        ("coordination_hand_left", "left"),
        ("coordination_hand_right", "right"),
    ):
        profile = coordination[scenario]
        assert profile["perturbed_arm"] == perturbed_arm
        assert profile["anchor_arm"] == "left"
        assert profile["skill_label"] == 5
        assert profile["evidence_frame"] == "right_ee"
        assert profile["expected_duration"] == 19
        assert profile["local_tick"] == 15
        assert profile["global_tick"] == 235
        assert profile["interaction_object"] == "item0"
        assert profile["expected_gripper_states"] == {
            "left": "closed",
            "right": "open",
        }
        assert profile["handover_stage"] == (
            "receiver_at_giver_held_item_before_gripper_transfer"
        )


def test_v3_intervention_registry_rejects_missing_manual_semantics(
    tmp_path,
) -> None:
    payload = load_v3_intervention_protocol()
    payload.pop("fingerprint")
    del payload["dynamic_environment"]["bimanual_handover_item"][
        "interaction_event"
    ]
    path = tmp_path / "invalid-interventions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields are invalid"):
        load_v3_intervention_protocol(path)


def test_checkpoint_trigger_audit_authenticates_eq5_eq6_intersection() -> None:
    audit = checkpoint_trigger_audit(_stack_policy())

    assert audit["schema"] == V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA
    assert audit["skills"][0]["frames"]["wine_bottle"] == {
        "selected_by_eq6": [True],
        "raw_link_runs": [[[68, 71]]],
        "majority_gate_enabled": [False],
        "availability_runs": [[[0, 71]]],
        "poe_active_runs": [[[0, 71]]],
    }
    assert len(audit["fingerprint"]) == 64


def test_v3_stack_trigger_is_bound_to_checkpoint_window() -> None:
    audit = checkpoint_trigger_audit(_stack_policy())

    evidence = build_v3_trigger_anchor_evidence("stack_wine", audit, {})
    anchor = evidence["anchors"]["stack_wine"]

    assert evidence["validated"] is True
    assert anchor["resolved_global_tick"] == 58
    assert anchor["required_active_window"] == [58, 67]
    assert anchor["selected_by_eq6"] == [True]
    assert anchor["interaction_arm"] == "single"
    assert anchor["interaction_object"] == "wine_bottle"
    assert anchor["expected_gripper_state"] == "open"
    assert anchor["validated"] is True


def test_v3_trigger_fails_if_full_smooth_window_is_not_active() -> None:
    audit = checkpoint_trigger_audit(_stack_policy(break_required_window=True))

    with pytest.raises(RuntimeError, match=r"Equation \(5\)-available"):
        build_v3_trigger_anchor_evidence("stack_wine", audit, {})


def test_dynamic_profile_returns_a_copy() -> None:
    first = dynamic_trigger_profile("stack_wine")
    first["local_tick"] = 0

    assert dynamic_trigger_profile("stack_wine")["local_tick"] == 58


def test_evaluator_resolves_only_an_authenticated_checkpoint_trigger() -> None:
    audit = checkpoint_trigger_audit(_stack_policy())
    envelope = build_v3_trigger_anchor_evidence("stack_wine", audit, {})
    identity = {
        "manifest_authenticated": True,
        "training_manifest_schema": "dynamac-direct-training-v3",
        "checkpoint_trigger_audit_fingerprint": audit["fingerprint"],
        "v3_trigger_anchor_evidence": envelope,
    }

    resolved = resolve_authenticated_v3_trigger(identity, task="stack_wine")

    assert resolved["trigger_step"] == 58
    assert resolved["profile_key"] == "stack_wine"
    assert resolved["evidence"]["validated"] is True

    identity["v3_trigger_anchor_evidence"] = {
        **envelope,
        "intervention_protocol_fingerprint": "forged",
    }
    with pytest.raises(RuntimeError, match="envelope"):
        resolve_authenticated_v3_trigger(identity, task="stack_wine")


def _fixed_loader_run() -> LocalRun:
    return LocalRun(
        path=Path("stack_wine_teleport_seed2608000000_n1_h1000.json"),
        task="stack_wine",
        scenario="teleport",
        seed=2_608_000_000,
        episodes=1,
        horizon=1000,
        variation=0,
        successes=0,
        success_rate=0.0,
        payload={
            "fixed_eval_set": {
                "evaluation_set_id": "rlbench_fixed_v1",
                "manifest_sha256": "a" * 64,
                "spec_sha256": "b" * 64,
                "selected_batch_sha256": "c" * 64,
                "selected_batch_fingerprint": "d" * 64,
                "formal_access": "canonical_id_read_only_no_generation",
            },
            "motion_plan_batch_fingerprint": "d" * 64,
        },
    )


def test_v34_report_resolves_plans_only_through_the_canonical_eval_set(
    monkeypatch,
) -> None:
    plan = object()
    manifest = {
        "manifest_sha256": "a" * 64,
        "payload": {
            "spec": {"sha256": "b" * 64},
            "environment_plan_batches": {
                "stack_wine": {"sha256": "c" * 64}
            },
        },
    }
    selected = {
        "payload": {
            "base_seed": 2_608_000_000,
            "episodes": 1,
            "batch_fingerprint": "d" * 64,
        },
        "plans": [plan],
    }
    monkeypatch.setattr(
        paper_comparison,
        "fixed_environment_plans",
        lambda eval_set_id, task: (manifest, selected),
    )

    run = _fixed_loader_run()
    assert paper_comparison._load_v3_staged_plans(run) == [plan]

    run.payload["fixed_eval_set"]["selected_batch_sha256"] = "e" * 64
    assert paper_comparison._load_v3_staged_plans(run) is None


def test_v34_formal_paths_keep_sealed_inputs_models_and_results_disjoint() -> None:
    results = INTEGRATION_ROOT / "results" / "v3" / "diagnostic.json"
    models = INTEGRATION_ROOT / "models" / "v3"
    validate_formal_artifact_paths(output=results, models_dir=models)

    with pytest.raises(ValueError, match="results root"):
        validate_formal_artifact_paths(
            output=(
                INTEGRATION_ROOT
                / "evaluation_sets"
                / "rlbench_fixed_v1"
                / "forbidden.json"
            ),
            models_dir=models,
        )
    with pytest.raises(ValueError, match="model input"):
        validate_formal_artifact_paths(
            output=results,
            models_dir=INTEGRATION_ROOT / "results" / "v3",
        )


def test_v34_evaluation_protocol_ids_bind_contact_delta_diagnostics() -> None:
    expected_fragment = "formal-root-state-audit2-contact-delta-diagnostic-v3"

    assert expected_fragment in evaluation_protocol_id(3)
    assert expected_fragment in bimanual_evaluation_protocol_id(3)


def test_v34_formal_contact_delta_is_authenticated_but_diagnostic() -> None:
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    task_tree_state = [
        {
            "name": "boundary_root",
            "type": "ObjectType.DUMMY",
            "parent": "task_base",
            "in_boundary_root_subtree": True,
            "world_pose": pose,
            "pose_relative_to_boundary_root": pose,
        }
    ]
    task_tree = _compare_task_tree_relative_state(
        task_tree_state,
        task_tree_state,
        boundary_root_may_move=True,
    )
    table = {"arm": "arm", "external_object_name": "table"}
    contact = {"arm": "arm", "external_object_name": "wine_bottle"}
    audit = {
        "schema": FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA,
        "comparison_class": (
            "same_formal_initialized_task_instance_immediate_pre_to_post_"
            "boundary_root_command"
        ),
        "reference_state": "current_policy_evolved_formal_state",
        "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
        "task_tree": task_tree,
        "task_semantics_matched": True,
        "condition_and_grasp_registry_identity_preserved": True,
        "gripper_grasp_membership_and_parentage_preserved": True,
        "robot_external_collision_pair_policy": (
            FORMAL_INTERVENTION_COLLISION_PAIR_POLICY
        ),
        "before_robot_external_collision_pairs": [table],
        "after_robot_external_collision_pairs": [table, contact],
        "new_robot_external_collision_pairs": [contact],
        "no_new_robot_external_collision_pairs": False,
        "passed": True,
    }
    event = {"formal_intervention_state_audit": audit}

    assert paper_comparison._valid_v3_formal_intervention_state_audit(
        event,
        expected_arms=frozenset({"arm"}),
    )

    inconsistent = copy.deepcopy(event)
    inconsistent["formal_intervention_state_audit"][
        "no_new_robot_external_collision_pairs"
    ] = True
    assert not paper_comparison._valid_v3_formal_intervention_state_audit(
        inconsistent,
        expected_arms=frozenset({"arm"}),
    )

    inconsistent = copy.deepcopy(event)
    inconsistent["formal_intervention_state_audit"][
        "new_robot_external_collision_pairs"
    ] = []
    assert not paper_comparison._valid_v3_formal_intervention_state_audit(
        inconsistent,
        expected_arms=frozenset({"arm"}),
    )


def _v3_teleport_run(tmp_path, *, variation=0):
    pytest.skip(
        "legacy V3.3 free-path/cross-initialization fixture; V3.4 uses the "
        "sealed canonical eval-set and deterministic source reconstruction"
    )
    audit = checkpoint_trigger_audit(_stack_policy())
    envelope = build_v3_trigger_anchor_evidence("stack_wine", audit, {})
    identity = {
        "manifest_authenticated": True,
        "training_manifest_schema": "dynamac-direct-training-v3",
        "training_adapter_protocol": V3_ADAPTER_PROTOCOL,
        "checkpoint_trigger_audit_fingerprint": audit["fingerprint"],
        "v3_trigger_anchor_evidence": envelope,
        "training_config": EXPECTED_RELEASE_CONFIGS["v3"],
        "model_schema_version": 13,
        "selection_semantics_id": _stack_policy().selection_semantics_id,
        "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
        "fingerprint": "stack-v3-checkpoint",
    }
    protocol = load_v3_intervention_protocol()
    motion_source_protocol = load_v3_motion_source_protocol()
    authentication = resolve_authenticated_v3_trigger(identity, task="stack_wine")
    formal_source_fingerprint = "2" * 64
    source_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    goal_pose = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    source_low_dim_state = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    source_task_tree = [
        {
            "name": "boundary_root",
            "type": "shape",
            "parent": "task_base",
            "in_boundary_root_subtree": True,
            "world_pose": list(source_pose),
            "pose_relative_to_boundary_root": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
    ]
    goal_task_tree = copy.deepcopy(source_task_tree)
    goal_task_tree[0]["world_pose"] = list(goal_pose)
    task_descriptions = ["stack the wine bottle"]
    task_semantic_signature = {
        "schema": TASK_SEMANTIC_SIGNATURE_SCHEMA,
        "task_class": "rlbench.tasks.stack_wine.StackWine",
        "success_conditions": [],
        "fail_conditions": [],
        "graspable_objects": [],
    }
    source_collision_records = []
    selected_source_fingerprint = _staging_source_fingerprint(
        task_name="stack_wine",
        variation=variation,
        root_pose=source_pose,
        low_dim_state=source_low_dim_state,
        task_tree_state=source_task_tree,
        semantic_signature=task_semantic_signature,
        descriptions=task_descriptions,
        collision_pair_records=source_collision_records,
    )
    cross_tolerances = {
        "root_translation_m": ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
        "root_rotation_rad": ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
        "task_pose_translation_m": (
            CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
        ),
        "task_pose_rotation_rad": CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        "scalar_state": CROSS_INITIALIZATION_SCALAR_TOLERANCE,
        "joint_position": CROSS_INITIALIZATION_JOINT_TOLERANCE,
    }
    root_cross = {
        "preserved": True,
        "translation_error_m": 0.0,
        "rotation_error_rad": 0.0,
        "translation_tolerance_m": ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
        "rotation_tolerance_rad": ROOT_COMMAND_ROTATION_TOLERANCE_RAD,
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
    }
    low_dim_cross = {
        "preserved": True,
        "comparison_mode": "pose_chunks_sign_invariant",
        "chunk_count": 2,
        "raw_l2": 0.0,
        "raw_max_abs": 0.0,
        "max_translation_m": 0.0,
        "max_rotation_rad": 0.0,
        "translation_tolerance_m": (
            CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
        ),
        "rotation_tolerance_rad": CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        "scalar_tolerance": CROSS_INITIALIZATION_SCALAR_TOLERANCE,
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
    }
    compact_tree_cross = {
        "matched": True,
        "topology_matched": True,
        "comparison_mode": "all_objects_world",
        "expected_object_count": 1,
        "actual_object_count": 1,
        "translation_tolerance_m": (
            CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M
        ),
        "rotation_tolerance_rad": CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        "joint_tolerance": CROSS_INITIALIZATION_JOINT_TOLERANCE,
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
        "all_parents_matched": True,
        "all_subtree_memberships_matched": True,
        "max_translation_error_m": 0.0,
        "max_rotation_error_rad": 0.0,
        "max_joint_position_error": 0.0,
    }
    source_to_goal_tree = _compare_task_tree_relative_state(
        source_task_tree,
        goal_task_tree,
        boundary_root_may_move=True,
    )
    goal_post_validation_tree = _compare_task_tree_relative_state(
        goal_task_tree,
        goal_task_tree,
        boundary_root_may_move=False,
    )
    formal_task_tree_match = _compare_task_tree_relative_state(
        source_task_tree,
        source_task_tree,
        boundary_root_may_move=False,
        translation_tolerance_m=CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
        rotation_tolerance_rad=CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        joint_tolerance=CROSS_INITIALIZATION_JOINT_TOLERANCE,
    )
    assert _compact_task_tree_comparison(formal_task_tree_match) == (
        compact_tree_cross
    )
    velocity_summary = {
        "schema": "rlbench-task-tree-velocity-summary-v1",
        "compared_for_identity": False,
        "diagnostic_only": True,
        "object_count": 1,
        "all_finite": True,
        "max_linear_speed_m_s": 0.0,
        "max_angular_speed_rad_s": 0.0,
    }
    staging_generation = _fresh_generation_evidence(
        generation_index=1,
        episode_seed=0,
        variation=variation,
        task_name="stack_wine",
    )
    formal_generation = _fresh_generation_evidence(
        generation_index=1,
        episode_seed=0,
        variation=variation,
        task_name="stack_wine",
    )
    cross_initialization_audit = {
        "schema": CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA,
        "comparison_class": (
            "same_seed_variation_independent_fresh_task_generation"
        ),
        "fresh_task_generation_protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
        "candidate_source_policy": (
            "each_candidate_uses_its_same_fresh_generation_A"
        ),
        "reference_attempt": 1,
        "selected_attempt": 1,
        "reference_source_fingerprint": selected_source_fingerprint,
        "selected_source_fingerprint": selected_source_fingerprint,
        "all_attempts_passed": True,
        "attempts": [
            {
                "attempt": 1,
                "fresh_task_generation": staging_generation,
                "source_fingerprint": selected_source_fingerprint,
                "root": root_cross,
                "low_dim_state": low_dim_cross,
                "task_tree": compact_tree_cross,
                "task_semantics_matched": True,
                "task_descriptions_matched": True,
                "robot_external_collision_pairs_matched": True,
                "task_object_velocities_finite": True,
                "passed": True,
            }
        ],
        "tolerances": cross_tolerances,
        "quaternion_rotation_metric": QUATERNION_ROTATION_METRIC,
        "worst_observed": {
            "root_translation_m": 0.0,
            "root_rotation_rad": 0.0,
            "low_dim_pose_translation_m": 0.0,
            "low_dim_pose_rotation_rad": 0.0,
            "low_dim_raw_max_abs": 0.0,
            "task_tree_translation_m": 0.0,
            "task_tree_rotation_rad": 0.0,
            "task_tree_joint_position": 0.0,
        },
        "tolerance_role": "fail_closed_cross_generation_watchdog",
    }
    validation = {
        "schema": STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
        "staging_max_attempts": motion_source_protocol[
            "goal_sampling_max_attempts"
        ],
        "environment_role": "independent_disposable_staging",
        "fresh_task_generation_protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
        "selected_source_fresh_task_generation": staging_generation,
        "formal_rollout_sample_or_restore": False,
        "source_waypoint_validated": True,
        "goal_waypoint_validated": True,
        "waypoint_validation_api": "Task.validate",
        "source_validation": "TaskEnvironment.reset_verify_instance_true",
        "task_init_episode_called_by_candidate_sampler": False,
        "task_init_episode_called_by_staging_reset": True,
        "source_and_goal_same_initialized_task_instance": True,
        "candidate_isolation": (
            "prestop_unload_if_present_stop_physics_fresh_task_reload_"
            "single_reset_no_candidate_restore"
        ),
        "task_get_state_called_directly_by_candidate_sampler": False,
        "task_restore_state_called_directly_by_candidate_sampler": False,
        "open_microwave_limit_normalization_false_guard_possible": False,
        "task_descriptions": task_descriptions,
        "task_semantic_signature": task_semantic_signature,
        "task_semantic_fingerprint": _canonical_json_fingerprint(
            task_semantic_signature
        ),
        "selected_source_fingerprint": selected_source_fingerprint,
        "cross_initialization_reproducibility": cross_initialization_audit,
        "selected_source_task_object_velocity_summary": velocity_summary,
        "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
        "source_task_tree_relative_state": source_task_tree,
        "source_task_tree_object_count": 1,
        "source_task_tree_fingerprint": _canonical_json_fingerprint(
            source_task_tree
        ),
        "goal_pre_validation_task_tree_state_preserved": source_to_goal_tree,
        "goal_waypoint_validation_task_tree_state_preserved": (
            goal_post_validation_tree
        ),
        "goal_task_tree_relative_state_preserved": source_to_goal_tree,
        "source_low_dim_size": len(source_low_dim_state),
        "task_frame_rigid_motion": {
            "task_spec": "stack_wine",
            "source_expression": (
                "StackWine.get_low_dim_state: [wine_bottle, success_sensor]"
            ),
            "checked_frames": ["wine_bottle", "success_sensor"],
            "all_pose_chunks_follow_boundary_root_rigid_transform": True,
            "translation_tolerance_m": 1.0e-6,
            "rotation_tolerance_rad": 1.0e-6,
            "frames": [
                {
                    "frame": "wine_bottle",
                    "translation_error_m": 0.0,
                    "rotation_error_rad": 0.0,
                    "preserved": True,
                },
                {
                    "frame": "success_sensor",
                    "translation_error_m": 0.0,
                    "rotation_error_rad": 0.0,
                    "preserved": True,
                },
            ],
        },
        "source_robot_external_collision_pairs": source_collision_records,
        "goal_robot_external_collision_pairs": [],
        "goal_new_robot_external_collision_pairs": [],
        "sampling_attempts": 1,
    }
    plan = StagedMotionPlan(
        task_name="stack_wine",
        source_pose=tuple(source_pose),
        goal_pose=tuple(goal_pose),
        source_low_dim_state=tuple(source_low_dim_state),
        episode_seed=0,
        variation=variation,
        validation=validation,
    )
    batch = staged_motion_plan_batch(
        task_name="stack_wine",
        base_seed=0,
        variations=[variation],
        plans=[plan],
    )
    plan_path = tmp_path / "stack-plans.json"
    plan_path.write_text(json.dumps(batch), encoding="utf-8")
    trigger = authentication["trigger_step"]
    motion = ScenarioController(
        "teleport_task",
        trigger_step=trigger,
        total_steps=10,
        motion_plan=plan,
    ).protocol_metadata()
    settling = final_settling_metadata()
    not_entered = {
        **settling,
        "attempted": False,
        "available": True,
        "steps_executed": 0,
        "first_terminal_step": None,
        "stop_reason": "not_entered",
        "success": False,
        "terminate": False,
    }
    scene_launch = {
        "protocol_id": LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
        "applied": True,
        "source_scene_sha256": EXPECTED_UNIMANUAL_BASE_SCENE_SHA256,
        "derived_scene_sha256": "0" * 64,
        "vision_sensor_count": EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT,
        "vision_sensor_handling": [
            {"name": f"camera_{index}", "before": 0, "after": 1}
            for index in range(EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT)
        ],
        "populated_scene_steps_before_patch": 0,
        "camera_observations_requested": False,
        "task_model_loaded_during_rewrite": False,
        "physics_modified": False,
        "task_modified": False,
        "policy_input_modified": False,
        "qt_qpa_platform": "offscreen",
    }
    event = {
        "kind": "teleport_task",
        "step": trigger,
        "trigger_step": trigger,
        "applied": True,
        "clock_domain": "committed_policy_ticks",
        "motion_protocol": motion,
        "policy_observation_refreshed": True,
        "instance_preservation": validation,
        "planned_root_translation_m": 0.1,
        "planned_root_rotation_rad": 0.0,
        "actual_root_translation_m": 0.1,
        "actual_root_rotation_rad": 0.0,
        "commanded_root_translation_residual_m": 0.0,
        "commanded_root_rotation_residual_rad": 0.0,
        "goal_root_translation_residual_m": 0.0,
        "goal_root_rotation_residual_rad": 0.0,
        "planned_root_motion": True,
        "actual_root_motion": True,
        "commanded_root_pose_reached": True,
        "goal_root_pose_reached": True,
        "protocol_effective": True,
        "formal_intervention_state_audit": {
            "schema": FORMAL_INTERVENTION_STATE_AUDIT_SCHEMA,
            "comparison_class": (
                "same_formal_initialized_task_instance_immediate_pre_to_post_"
                "boundary_root_command"
            ),
            "reference_state": "current_policy_evolved_formal_state",
            "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
            "task_tree": source_to_goal_tree,
            "task_semantics_matched": True,
            "condition_and_grasp_registry_identity_preserved": True,
            "gripper_grasp_membership_and_parentage_preserved": True,
            "robot_external_collision_pair_policy": (
                FORMAL_INTERVENTION_COLLISION_PAIR_POLICY
            ),
            "before_robot_external_collision_pairs": [],
            "after_robot_external_collision_pairs": [],
            "new_robot_external_collision_pairs": [],
            "no_new_robot_external_collision_pairs": True,
            "passed": True,
        },
    }
    row = {
        "episode": 0,
        "success": True,
        "reason": "success",
        "committed_policy_steps": trigger + 1,
        "dynamic_clock_steps": trigger + 1,
        "trigger_step": trigger,
        "interventions": [event],
        "intervention_eligible": True,
        "intervention_reached": True,
        "pre_intervention_terminal": False,
        "pre_intervention_terminal_outcome": None,
        "dynamic_condition_exercised": True,
        "dynamic_condition_unexercised": False,
        "intervention_effective": True,
        "intervention_complete": True,
        "staged_source_binding": {
            "required": True,
            "matched": True,
            "formal_source_bound": True,
            "formal_sampling_or_restore": False,
            "fresh_task_generation_protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
            "selected_source_fresh_task_generation": staging_generation,
            "formal_source_fresh_task_generation": formal_generation,
            "task_name": "stack_wine",
            "task_semantics_matched": True,
            "task_tree_matched": True,
            "task_tree_state_schema": TASK_TREE_STATE_SCHEMA,
            "task_tree_match": formal_task_tree_match,
            "task_descriptions_matched": True,
            "root": root_cross,
            "low_dim_state": low_dim_cross,
            "cross_initialization_reproducibility": {
                "schema": CROSS_INITIALIZATION_REPRODUCIBILITY_SCHEMA,
                "comparison_class": (
                    "selected_staging_fresh_A_to_formal_fresh_"
                    "same_seed_variation_A"
                ),
                "fresh_task_generation_protocol_id": (
                    FRESH_TASK_GENERATION_PROTOCOL_ID
                ),
                "selected_source_fresh_task_generation": staging_generation,
                "formal_source_fresh_task_generation": formal_generation,
                "selected_source_fingerprint": selected_source_fingerprint,
                "formal_source_fingerprint": formal_source_fingerprint,
                "fingerprints_compared_for_identity": False,
                "tolerances": cross_tolerances,
                "root": root_cross,
                "low_dim_state": low_dim_cross,
                "task_tree": compact_tree_cross,
                "task_semantics_matched": True,
                "task_descriptions_matched": True,
                "robot_external_collision_pairs_matched": True,
                "selected_task_object_velocity_summary": velocity_summary,
                "formal_task_object_velocity_summary": velocity_summary,
                "task_object_velocities_compared_for_identity": False,
                "task_object_velocities_finite": True,
                "passed": True,
            },
            "robot_external_collision_pairs_matched": True,
            "selected_source_fingerprint": selected_source_fingerprint,
            "formal_source_fingerprint": formal_source_fingerprint,
            "motion_plan_fingerprint": plan.fingerprint(),
        },
        "motion_plan_fingerprint": plan.fingerprint(),
        "motion_plan_protocol_id": motion["protocol_id"],
        "motion_plan_evidence": {
            "plan_fingerprint": plan.fingerprint(),
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "formal_rollout_sample_or_restore": False,
            "formal_source_bound": True,
            "formal_task_name_bound": "stack_wine",
            "formal_task_semantics_matched": True,
            "formal_task_tree_matched": True,
            "formal_cross_initialization_reproducibility_passed": True,
            "formal_robot_external_collision_pairs_matched": True,
            "selected_source_fingerprint": selected_source_fingerprint,
            "formal_source_fingerprint": formal_source_fingerprint,
        },
        "final_settling": not_entered,
        "fresh_task_generation": formal_generation,
    }
    payload = {
        "schema": "dynamac-table-i-evaluation-v3",
        "task": "stack_wine",
        "scenario": "teleport",
        "seed": 0,
        "variation": variation,
        "variation_count": 3,
        "variation_schedule": [variation],
        "episodes": 1,
        "horizon": 1000,
        "successes": 1,
        "success_rate": 1.0,
        "evaluation_protocol_id": evaluation_protocol_id(3),
        "model_identity": identity,
        "controller": {
            "policy_clock_rollback": True,
            "policy_clock_semantics_id": (
                "policy-tick-transaction-commit-on-primary-action-success-v1"
            ),
            "dynamic_clock_semantics": "advance_only_after_policy_commit",
            "formal_episode_initialization": FRESH_TASK_GENERATION_PROTOCOL_ID,
            "final_settling": settling,
            "scene_launch": scene_launch,
        },
        "final_settling_protocol": settling,
        "episode_accounting": {
            "schema": (
                "planned-denominator-trigger-completion-conditional-success-v3"
            ),
            "planned_episode_denominator": 1,
            "completed_episode_count": 1,
            "successes_in_planned_denominator": 1,
            "success_rate_all_planned_episodes": 1.0,
            "trigger_reached_count": 1,
            "intervention_complete_count": 1,
            "dynamic_condition_unexercised_count": 0,
            "pre_trigger_success_count": 0,
            "complete_intervention_subset_count": 1,
            "successes_in_complete_intervention_subset": 1,
            "success_rate_in_complete_intervention_subset": 1.0,
        },
        "motion_plan_batch_fingerprint": batch["batch_fingerprint"],
        "protocol": {
            "motion_protocol": motion,
            "trigger_reference_domain": "successfully_committed_policy_ticks",
            "trigger_policy_step": trigger,
            "trigger_authentication": authentication,
            "intervention_registry_schema": protocol["schema"],
            "intervention_registry_fingerprint": protocol["fingerprint"],
            "intervention_max_attempts": motion_source_protocol[
                "goal_sampling_max_attempts"
            ],
            "dynamic_episode_accounting_schema": (
                "planned-denominator-trigger-completion-conditional-success-v3"
            ),
            "pre_intervention_failure_policy": (
                "retain_failure_with_null_intervention_effectiveness"
            ),
            "pre_intervention_success_policy": (
                "retain_success_in_planned_denominator_with_unexercised_condition"
            ),
            "smooth_terminal_progress_policy": (
                "strict_effective_prefix_until_episode_terminal"
            ),
            "planned_episode_denominator": 1,
            "completed_episode_count": 1,
            "episodes_intervention_eligible": 1,
            "episodes_pre_intervention_terminal": 0,
            "episodes_dynamic_condition_unexercised": 0,
            "pre_trigger_successes": 0,
            "episodes_with_intervention": 1,
            "episodes_with_effective_intervention": 1,
            "episodes_with_complete_intervention": 1,
            "successes_in_complete_intervention_subset": 1,
            "success_rate_in_complete_intervention_subset": 1.0,
            "all_episodes_intervened": True,
            "all_interventions_effective": True,
            "all_eligible_interventions_effective": True,
            "protocol_valid": True,
            "paper_comparable": False,
            "staged_motion_plan_cache": {
                "path": str(plan_path),
                "schema": batch["schema"],
                "protocol_id": batch["protocol_id"],
                "batch_fingerprint": batch["batch_fingerprint"],
                "plan_fingerprints": [plan.fingerprint()],
                "scenario_independent": True,
                "seed_domain": batch["seed_domain"],
                "staging_max_attempts": motion_source_protocol[
                    "goal_sampling_max_attempts"
                ],
                "cache_key": {
                    "task": "stack_wine",
                    "base_seed": 0,
                    "episodes": 1,
                    "variation_schedule": [variation],
                },
                "atomic_write": True,
                "concurrent_generation": "exclusive_generation_lock_else_wait",
                "staging_shutdown_before_formal_launch": True,
                "fresh_task_generation_per_formal_episode": True,
            },
        },
        "results": [row],
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": True,
            "evidence": [formal_generation],
        },
    }
    return LocalRun(
        path=tmp_path / "stack_wine_teleport_seed0_n1_h1000.json",
        task="stack_wine",
        scenario="teleport",
        seed=0,
        episodes=1,
        horizon=1000,
        variation=variation,
        successes=1,
        success_rate=1.0,
        payload=payload,
    )


def test_v3_report_authenticates_full_staged_dynamic_protocol(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)

    assert _model_identity_rank(run) == 0
    assert _valid_v3_dynamic_protocol(run) is True


def test_v3_report_rejects_resigned_nested_semantic_forgery(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)
    selected_fingerprint = None

    def mutate_plan(plan_payload):
        nonlocal selected_fingerprint
        validation = plan_payload["validation"]
        validation["task_semantic_signature"] = {
            "schema": TASK_SEMANTIC_SIGNATURE_SCHEMA,
            "task_class": "rlbench.tasks.stack_wine.StackWine",
            "success_conditions": [
                {
                    "type": "rlbench.backend.conditions.OrConditions",
                    "structural_fields": {"_conditions": ["forged"]},
                    "excluded_runtime_progress_fields": [
                        "_current_condition_index"
                    ],
                }
            ],
            "fail_conditions": [],
            "graspable_objects": [],
        }
        signature = validation["task_semantic_signature"]
        validation["task_semantic_fingerprint"] = (
            _canonical_json_fingerprint(signature)
        )
        selected_fingerprint = _staging_source_fingerprint(
            task_name=plan_payload["task_name"],
            variation=plan_payload["variation"],
            root_pose=plan_payload["source_pose"],
            low_dim_state=plan_payload["source_low_dim_state"],
            task_tree_state=validation["source_task_tree_relative_state"],
            semantic_signature=signature,
            descriptions=validation["task_descriptions"],
            collision_pair_records=(
                validation["source_robot_external_collision_pairs"]
            ),
        )
        validation["selected_source_fingerprint"] = selected_fingerprint
        cross = validation["cross_initialization_reproducibility"]
        cross["reference_source_fingerprint"] = selected_fingerprint
        cross["selected_source_fingerprint"] = selected_fingerprint
        cross["attempts"][0]["source_fingerprint"] = selected_fingerprint

    _resign_tampered_plan_cache(run, mutate_plan=mutate_plan)
    row = run.payload["results"][0]
    row["staged_source_binding"]["selected_source_fingerprint"] = (
        selected_fingerprint
    )
    row["staged_source_binding"]["cross_initialization_reproducibility"][
        "selected_source_fingerprint"
    ] = selected_fingerprint
    row["motion_plan_evidence"]["selected_source_fingerprint"] = (
        selected_fingerprint
    )

    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_resigned_formal_generation_index(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)
    evidence = run.payload["results"][0]["fresh_task_generation"]
    evidence["generation_index"] = 2
    evidence["fingerprint"] = _canonical_json_fingerprint(
        {key: value for key, value in evidence.items() if key != "fingerprint"}
    )
    run.payload["fresh_task_generation"]["evidence"][0] = evidence
    binding = run.payload["results"][0]["staged_source_binding"]
    binding["formal_source_fresh_task_generation"] = evidence
    binding["cross_initialization_reproducibility"][
        "formal_source_fresh_task_generation"
    ] = evidence

    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_resigned_formal_evidence_numeric_type(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)
    evidence = run.payload["results"][0]["fresh_task_generation"]
    evidence["task_environment_reset_calls"] = True
    evidence["fingerprint"] = _canonical_json_fingerprint(
        {key: value for key, value in evidence.items() if key != "fingerprint"}
    )
    run.payload["fresh_task_generation"]["evidence"][0] = evidence
    binding = run.payload["results"][0]["staged_source_binding"]
    binding["formal_source_fresh_task_generation"] = evidence
    binding["cross_initialization_reproducibility"][
        "formal_source_fresh_task_generation"
    ] = evidence

    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_resigned_staging_generation_gap(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)

    def mutate(validation):
        evidence = validation["cross_initialization_reproducibility"][
            "attempts"
        ][0]["fresh_task_generation"]
        evidence["generation_index"] = 2
        evidence["fingerprint"] = _canonical_json_fingerprint(
            {
                key: value
                for key, value in evidence.items()
                if key != "fingerprint"
            }
        )
        validation["selected_source_fresh_task_generation"] = evidence

    _resign_tampered_plan_cache(run, mutate)

    assert _valid_v3_dynamic_protocol(run) is False


def _resign_tampered_plan_cache(
    run,
    mutate_validation=None,
    *,
    mutate_plan=None,
) -> None:
    cache = run.payload["protocol"]["staged_motion_plan_cache"]
    path = Path(cache["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan_payload = payload["plans"][0]
    if mutate_validation is not None:
        mutate_validation(plan_payload["validation"])
    if mutate_plan is not None:
        mutate_plan(plan_payload)
    unsigned_plan = {
        key: value for key, value in plan_payload.items() if key != "fingerprint"
    }
    plan_fingerprint = hashlib.sha256(
        json.dumps(unsigned_plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan_payload["fingerprint"] = plan_fingerprint
    unsigned_batch = {
        key: value for key, value in payload.items() if key != "batch_fingerprint"
    }
    batch_fingerprint = hashlib.sha256(
        json.dumps(unsigned_batch, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["batch_fingerprint"] = batch_fingerprint
    path.write_text(json.dumps(payload), encoding="utf-8")

    cache["batch_fingerprint"] = batch_fingerprint
    cache["plan_fingerprints"] = [plan_fingerprint]
    run.payload["motion_plan_batch_fingerprint"] = batch_fingerprint
    row = run.payload["results"][0]
    row["motion_plan_fingerprint"] = plan_fingerprint
    row["staged_source_binding"]["motion_plan_fingerprint"] = plan_fingerprint
    row["motion_plan_evidence"]["plan_fingerprint"] = plan_fingerprint
    row["interventions"][0]["instance_preservation"] = copy.deepcopy(
        plan_payload["validation"]
    )


@pytest.mark.parametrize(
    "mutate_validation",
    (
        lambda value: value.__setitem__("task_tree_state_schema", "forged"),
        lambda value: value[
            "goal_pre_validation_task_tree_state_preserved"
        ].__setitem__("comparison_mode", "all_objects_world"),
        lambda value: value[
            "goal_waypoint_validation_task_tree_state_preserved"
        ].__setitem__(
            "comparison_mode",
            "boundary_root_subtree_relative_else_world",
        ),
        lambda value: value[
            "goal_task_tree_relative_state_preserved"
        ].__setitem__("matched", False),
        lambda value: value[
            "cross_initialization_reproducibility"
        ].__setitem__("candidate_source_policy", "first_A_for_every_candidate"),
        lambda value: value[
            "selected_source_task_object_velocity_summary"
        ].__setitem__("compared_for_identity", True),
        lambda value: value[
            "cross_initialization_reproducibility"
        ]["attempts"][0]["root"].__setitem__(
            "translation_error_m",
            2.0 * ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
        ),
        lambda value: value[
            "cross_initialization_reproducibility"
        ]["worst_observed"].__setitem__("low_dim_raw_max_abs", 1.0),
        lambda value: value.__setitem__("source_task_tree_object_count", 2),
        lambda value: value.__setitem__("source_low_dim_size", 7),
        lambda value: value.__setitem__("source_task_tree_fingerprint", "0" * 64),
        lambda value: value.__setitem__("task_semantic_fingerprint", "0" * 64),
        lambda value: value.__setitem__("selected_source_fingerprint", "0" * 64),
        lambda value: value[
            "goal_task_tree_relative_state_preserved"
        ]["objects"][0].__setitem__("name", "forged_same_count_object"),
        lambda value: value.__setitem__(
            "source_robot_external_collision_pairs",
            [{"arm": "forged", "external_object_name": "object"}],
        ),
        lambda value: value[
            "cross_initialization_reproducibility"
        ].__setitem__("attempts", [None]),
    ),
)
def test_v3_report_rejects_resigned_invalid_task_tree_evidence(
    tmp_path,
    mutate_validation,
) -> None:
    run = _v3_teleport_run(tmp_path)
    _resign_tampered_plan_cache(run, mutate_validation)

    assert _valid_v3_dynamic_protocol(run) is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task_tree_state_schema", "forged"),
        (
            "task_tree_match",
            {
                "matched": True,
                "comparison_mode": (
                    "boundary_root_subtree_relative_else_world"
                ),
            },
        ),
    ),
)
def test_v3_report_rejects_invalid_formal_task_tree_binding(
    tmp_path,
    field,
    value,
) -> None:
    run = _v3_teleport_run(tmp_path)
    run.payload["results"][0]["staged_source_binding"][field] = value

    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_formal_velocity_identity_comparison(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)
    binding = run.payload["results"][0]["staged_source_binding"]
    binding["cross_initialization_reproducibility"][
        "task_object_velocities_compared_for_identity"
    ] = True

    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_resigned_zero_motion_plan_or_unbound_event_motion(
    tmp_path,
) -> None:
    run = _v3_teleport_run(tmp_path)
    _resign_tampered_plan_cache(
        run,
        mutate_plan=lambda value: value.__setitem__(
            "goal_pose",
            list(value["source_pose"]),
        ),
    )
    assert _valid_v3_dynamic_protocol(run) is False

    run = _v3_teleport_run(tmp_path)
    run.payload["results"][0]["interventions"][0][
        "planned_root_translation_m"
    ] = 0.2
    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_inconsistent_formal_nested_tree_or_velocity(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)
    binding = run.payload["results"][0]["staged_source_binding"]
    binding["cross_initialization_reproducibility"]["task_tree"][
        "max_translation_error_m"
    ] = 1.0e-7

    assert _valid_v3_dynamic_protocol(run) is False

    run = _v3_teleport_run(tmp_path)
    binding = run.payload["results"][0]["staged_source_binding"]
    binding["cross_initialization_reproducibility"][
        "selected_task_object_velocity_summary"
    ]["max_linear_speed_m_s"] = 1.0

    assert _valid_v3_dynamic_protocol(run) is False


@pytest.mark.parametrize(
    "mutate_audit",
    (
        lambda value: value.__setitem__("task_semantics_matched", False),
        lambda value: value["task_tree"].__setitem__(
            "translation_tolerance_m",
            2.0e-6,
        ),
        lambda value: value["task_tree"]["objects"][0].__setitem__(
            "matched",
            False,
        ),
        lambda value: value.__setitem__(
            "after_robot_external_collision_pairs",
            [{"arm": "arm", "external_object_name": "new_collision"}],
        ),
        lambda value: value.__setitem__(
            "condition_and_grasp_registry_identity_preserved",
            False,
        ),
    ),
)
def test_v3_report_rejects_invalid_formal_intervention_state_audit(
    tmp_path,
    mutate_audit,
) -> None:
    run = _v3_teleport_run(tmp_path)
    audit = run.payload["results"][0]["interventions"][0][
        "formal_intervention_state_audit"
    ]
    mutate_audit(audit)

    assert _valid_v3_dynamic_protocol(run) is False


@pytest.mark.parametrize("success", (True, False))
def test_v3_report_accepts_terminal_at_trigger_clock_before_trigger_application(
    tmp_path,
    success,
) -> None:
    run = _v3_teleport_run(tmp_path)
    payload = copy.deepcopy(run.payload)
    row = payload["results"][0]
    trigger = row["trigger_step"]
    row.update(
        {
            "success": success,
            "reason": "success" if success else "terminate",
            "committed_policy_steps": trigger,
            "dynamic_clock_steps": trigger,
            "interventions": [],
            "intervention_eligible": False,
            "intervention_reached": False,
            "pre_intervention_terminal": True,
            "pre_intervention_terminal_outcome": (
                "success" if success else "failure"
            ),
            "dynamic_condition_exercised": False,
            "dynamic_condition_unexercised": True,
            "intervention_effective": None,
            "intervention_complete": None,
        }
    )
    protocol = payload["protocol"]
    protocol.update(
        {
            "episodes_intervention_eligible": 0,
            "episodes_pre_intervention_terminal": 1,
            "episodes_dynamic_condition_unexercised": 1,
            "pre_trigger_successes": int(success),
            "episodes_with_intervention": 0,
            "episodes_with_effective_intervention": 0,
            "episodes_with_complete_intervention": 0,
            "successes_in_complete_intervention_subset": 0,
            "success_rate_in_complete_intervention_subset": None,
            "all_episodes_intervened": False,
        }
    )
    payload["episode_accounting"].update(
        {
            "successes_in_planned_denominator": int(success),
            "success_rate_all_planned_episodes": float(success),
            "trigger_reached_count": 0,
            "intervention_complete_count": 0,
            "dynamic_condition_unexercised_count": 1,
            "pre_trigger_success_count": int(success),
            "complete_intervention_subset_count": 0,
            "successes_in_complete_intervention_subset": 0,
            "success_rate_in_complete_intervention_subset": None,
        }
    )
    payload["successes"] = int(success)
    payload["success_rate"] = float(success)
    pretrigger = LocalRun(
        **{
            **run.__dict__,
            "successes": int(success),
            "success_rate": float(success),
            "payload": payload,
        }
    )

    assert _valid_v3_dynamic_protocol(pretrigger) is True


def test_v3_report_rejects_staging_attempt_budget_drift(tmp_path) -> None:
    run = _v3_teleport_run(tmp_path)
    plan_path = run.payload["protocol"]["staged_motion_plan_cache"]["path"]
    path = Path(plan_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plans"][0]["validation"]["sampling_attempts"] = 21
    unsigned_plan = {
        key: value for key, value in payload["plans"][0].items() if key != "fingerprint"
    }
    payload["plans"][0]["fingerprint"] = hashlib.sha256(
        json.dumps(unsigned_plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    unsigned_batch = {
        key: value for key, value in payload.items() if key != "batch_fingerprint"
    }
    payload["batch_fingerprint"] = hashlib.sha256(
        json.dumps(unsigned_batch, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _valid_v3_dynamic_protocol(run) is False


def test_v3_report_rejects_unimanual_plan_variation_masquerade(tmp_path) -> None:
    variation_one = _v3_teleport_run(tmp_path, variation=1)
    forged_top_level = LocalRun(
        **{
            **variation_one.__dict__,
            "variation": 0,
            "payload": {**variation_one.payload, "variation": 0},
        }
    )

    assert _valid_v3_dynamic_protocol(variation_one) is True
    assert _valid_v3_dynamic_protocol(forged_top_level) is False


def _v3_coordination_variation_run() -> LocalRun:
    pytest.skip(
        "legacy reset(true) coordination fixture; V3.4 uses the sealed "
        "coordination A-only batch"
    )
    episodes = 7
    trigger = 12
    variation_schedule = [episode % 5 for episode in range(episodes)]
    results = [
        {
            "episode": episode,
            "variation": variation,
            "committed_policy_steps": trigger + 1,
            "perturbed_steps": 1,
            "perturbed_attempts": 1,
            "fresh_task_generation": _fresh_generation_evidence(
                generation_index=episode + 1,
                episode_seed=episode,
                variation=variation,
                task_name="bimanual_handover_item_dynamic",
                previous_task_present=episode > 0,
            ),
        }
        for episode, variation in enumerate(variation_schedule)
    ]
    payload = {
        "episodes_requested": episodes,
        "episodes_completed": episodes,
        "variation_count": 5,
        "variation_schedule": variation_schedule,
        "controller": {
            "coordination_trigger_clock": "successfully_committed_policy_ticks",
            "formal_episode_initialization": FRESH_TASK_GENERATION_PROTOCOL_ID,
        },
        "coordination_protocol": {
            "protocol_valid": True,
            "perturbed_arm": "left",
            "translation_world_m": [0.0, 0.0, 0.03],
            "application": (
                "persistent offset on every predicted EE target from trigger"
            ),
            "trigger_policy_step": trigger,
            "legacy_one_third_trigger_disabled": True,
        },
        "results": results,
        "fresh_task_generation": {
            "required_per_formal_episode": True,
            "all_episodes_recorded": True,
            "evidence": [
                row["fresh_task_generation"] for row in results
            ],
        },
    }
    return LocalRun(
        path=Path("coordination_hand_left_seed0_n7_h1000.json"),
        task="bimanual_handover_item_dynamic",
        scenario="coordination_hand_left",
        seed=0,
        episodes=episodes,
        horizon=1000,
        variation=0,
        successes=0,
        success_rate=0.0,
        payload=payload,
    )


@pytest.mark.parametrize(
    "tamper",
    ("variation_count", "variation_schedule", "row_variation"),
)
def test_v3_coordination_report_binds_each_episode_variation(
    monkeypatch,
    tamper,
) -> None:
    monkeypatch.setattr(
        paper_comparison,
        "_valid_v3_trigger_metadata",
        lambda run: True,
    )
    monkeypatch.setattr(
        paper_comparison,
        "_valid_v3_final_settling",
        lambda run: True,
    )
    monkeypatch.setattr(
        paper_comparison,
        "_v3_trigger_authentication",
        lambda run: {
            "trigger_step": 12,
            "profile": {"perturbed_arm": "left"},
        },
    )
    run = _v3_coordination_variation_run()

    assert paper_comparison._valid_v3_coordination_protocol(run) is True
    if tamper == "variation_count":
        run.payload["variation_count"] = 4
    elif tamper == "variation_schedule":
        run.payload["variation_schedule"][1] = 0
    else:
        run.payload["results"][1]["variation"] = 0

    assert paper_comparison._valid_v3_coordination_protocol(run) is False
