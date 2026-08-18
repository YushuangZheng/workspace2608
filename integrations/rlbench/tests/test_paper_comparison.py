from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest
from essay2608.policy import DynaMACConfig

from integrations.rlbench.rlbench_dynamac.paper_comparison import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RELEASE,
    EXPECTED_LOCAL_CONFIG,
    EXPECTED_RELEASE_CONFIGS,
    EXPECTED_RELEASE_SELECTION_SEMANTICS_IDS,
    EXPECTED_SELECTION_SEMANTICS_ID,
    EXPECTED_TAPAS_COMMIT,
    INTEGRATION_ROOT,
    _expected_v4_root_motion_protocol,
    _v3_derived_motion_metric_matches,
    build_document,
    build_parser,
    build_records,
    discover_runs,
    expected_evaluation_protocol_id,
    markdown,
)
from integrations.rlbench.rlbench_dynamac.runtime import (
    LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    LOW_DIM_STATE_ROUNDTRIP_ATOL,
    PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
    ROOT_COMMAND_TRANSLATION_TOLERANCE_M,
    ScenarioController,
)
from integrations.rlbench.rlbench_dynamac.task_specs import get_task_spec
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    EXPECTED_UNIMANUAL_BASE_SCENE_SHA256,
    EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT,
    LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
)


def test_v4_validator_contract_matches_runtime_metadata() -> None:
    assert _expected_v4_root_motion_protocol() == ScenarioController(
        "teleport_task"
    ).protocol_metadata()


def test_v3_derived_motion_metric_allows_only_serialization_roundoff() -> None:
    assert _v3_derived_motion_metric_matches(
        0.7267024878023849,
        0.7267024878023851,
    )
    assert not _v3_derived_motion_metric_matches(0.1, 0.1 + 1.0e-12)
    assert not _v3_derived_motion_metric_matches(float("nan"), 0.1)
    assert not _v3_derived_motion_metric_matches(True, 1.0)


def _write_run(
    path,
    *,
    task,
    scenario,
    rate,
    family=None,
    paper_comparable=None,
    event=None,
    effective_interventions=None,
    covariance_method=None,
    corrected=True,
    variation=0,
    fingerprint="test-fingerprint",
    training_config=None,
):
    successes = int(rate * 200)
    root_motion = scenario in {"smooth", "teleport"}
    table_i_schema = family == "table_i"
    motion_kind = (
        "smooth_task_motion" if scenario == "smooth" else "teleport_task"
    )
    motion_protocol = ScenarioController(motion_kind).protocol_metadata()
    roundtrip_chunk_count = (
        len(get_task_spec(task).pose_chunks) if root_motion else 1
    )
    preservation = {
        "initialized_episode_preserved": True,
        "task_init_episode_called": False,
        "task_validate_called": False,
        "low_dim_state_roundtrip_preserved": True,
        "low_dim_state_roundtrip_comparison_mode": (
            "pose_chunks_sign_invariant"
        ),
        "low_dim_state_roundtrip_chunk_count": roundtrip_chunk_count,
        "low_dim_state_roundtrip_l2": 0.0,
        "low_dim_state_roundtrip_max_abs": 0.0,
        "low_dim_state_roundtrip_max_translation_m": 0.0,
        "low_dim_state_roundtrip_max_rotation_rad": 0.0,
        "condition_and_grasp_registry_identity_preserved": True,
        "gripper_grasp_membership_and_parentage_preserved": True,
        "configuration_tree_rollback": (
            "task_only_after_each_attempt_and_outer_finally"
        ),
        "task_configuration_tree_restored": True,
        "live_robot_state_untouched": True,
        "live_robot_configuration_trees_accessed": False,
        "robot_collision_pair_policy": (
            "reject_candidate_external_pairs_absent_at_source"
        ),
        "robot_collision_pair_granularity": (
            "named_arm_collection_x_external_collidable_scene_shape"
        ),
        "source_robot_external_collision_pairs": [],
        "goal_robot_external_collision_pairs": [],
        "goal_new_robot_external_collision_pairs": [],
        "sampling_attempts_rejected_for_new_robot_collision_pairs": 0,
        "sampling_attempts": 1,
        "waypoint_cache_identity_preserved": True,
    }
    event_payloads = []
    if root_motion:
        calls = 10 if scenario == "smooth" else 1
        for index in range(1, calls + 1):
            endpoint = index == calls
            fraction = index / calls
            goal_translation_residual = 0.0 if endpoint else 0.1 * (1.0 - fraction)
            event_payload = {
                "kind": motion_kind,
                "step": 59 + index,
                "trigger_step": 60,
                "applied": True,
                "motion_protocol": copy.deepcopy(motion_protocol),
                "instance_preservation": copy.deepcopy(preservation),
                "task_state_l2": 0.1,
                "task_state_changed": True,
                "root_pose_l2": 0.01 if scenario == "smooth" else 0.1,
                "root_pose_changed": True,
                "planned_root_translation_m": 0.1,
                "planned_root_rotation_rad": 0.0,
                "planned_root_motion": True,
                "actual_root_translation_m": (
                    0.01 if scenario == "smooth" else 0.1
                ),
                "actual_root_rotation_rad": 0.0,
                "actual_root_motion": True,
                "commanded_root_translation_residual_m": 0.0,
                "commanded_root_rotation_residual_rad": 0.0,
                "commanded_root_pose_reached": True,
                "goal_root_translation_residual_m": goal_translation_residual,
                "goal_root_rotation_residual_rad": 0.0,
                "goal_root_pose_reached": endpoint,
                "protocol_effective": True,
                "policy_observation_refreshed": True,
            }
            if scenario == "smooth":
                event_payload.update(
                    {
                        "smooth_call": index,
                        "complete": endpoint,
                        "endpoint_applied": endpoint,
                        "endpoint_fraction": fraction,
                    }
                )
            event_payload.update(event or {})
            event_payloads.append(event_payload)
    payload = {
        "task": task,
        "scenario": scenario,
        "seed": 0,
        "episodes": 200,
        "horizon": 1000,
        "variation": variation,
        "successes": successes,
        "success_rate": successes / 200,
        "results": [],
    }
    for episode in range(200):
        succeeded = episode < successes
        row = {
            "episode": episode,
            "success": succeeded,
            "steps": 187,
            "reason": "success" if succeeded else "policy_complete",
            "trigger_step": 60 if root_motion else None,
            "intervention_eligible": root_motion,
            "intervention_reached": root_motion,
            "pre_intervention_terminal": False,
            "intervention_effective": True if root_motion else None,
            "intervention_complete": True if root_motion else None,
        }
        if root_motion:
            event_key = "interventions" if table_i_schema else "scenario_events"
            row[event_key] = copy.deepcopy(event_payloads)
        elif event is not None:
            row["scenario_events"] = [dict(event)]
        payload["results"].append(row)
    if table_i_schema:
        payload["schema"] = "dynamac-table-i-evaluation-v2"
    if corrected:
        selected_training_config = dict(
            training_config or EXPECTED_RELEASE_CONFIGS["v2"]
        )
        selected_release = next(
            (
                release
                for release, config in EXPECTED_RELEASE_CONFIGS.items()
                if config == selected_training_config
            ),
            DEFAULT_RELEASE,
        )
        payload["model_identity"] = {
            "model_schema_version": 13,
            "selection_semantics_id": (
                EXPECTED_RELEASE_SELECTION_SEMANTICS_IDS[selected_release]
            ),
            "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
            "training_config": selected_training_config,
            "fingerprint": fingerprint,
            "manifest_authenticated": True,
        }
        payload["evaluation_protocol_id"] = expected_evaluation_protocol_id(
            task,
            release=selected_release,
        )
        if selected_release == "v2" and task in {
            "stack_wine",
            "place_cups",
            "open_microwave",
            "wipe_desk",
        }:
            payload["controller"] = {
                "scene_launch": {
                    "protocol_id": LOW_DIM_HEADLESS_SCENE_PROTOCOL_ID,
                    "applied": True,
                    "source_scene_sha256": EXPECTED_UNIMANUAL_BASE_SCENE_SHA256,
                    "derived_scene_sha256": "0" * 64,
                    "vision_sensor_count": (
                        EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT
                    ),
                    "vision_sensor_handling": [
                        {"name": f"camera_{index}", "before": 0, "after": 1}
                        for index in range(
                            EXPECTED_UNIMANUAL_BASE_VISION_SENSOR_COUNT
                        )
                    ],
                    "populated_scene_steps_before_patch": 0,
                    "camera_observations_requested": False,
                    "task_model_loaded_during_rewrite": False,
                    "physics_modified": False,
                    "task_modified": False,
                    "policy_input_modified": False,
                    "qt_qpa_platform": "offscreen",
                }
            }
    elif covariance_method is not None:
        payload["model_identity"] = {
            "model_schema_version": 13,
            "selection_semantics_id": EXPECTED_SELECTION_SEMANTICS_ID,
            "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
            "training_config": {
                **EXPECTED_LOCAL_CONFIG,
                "covariance_estimation_method": covariance_method,
            },
            "fingerprint": fingerprint,
            "manifest_authenticated": True,
        }
        payload["evaluation_protocol_id"] = expected_evaluation_protocol_id(task)
    if paper_comparable is not None:
        protocol_key = (
            "protocol"
            if table_i_schema
            else (
                "coordination_protocol"
                if scenario.startswith("coordination")
                else "scenario_protocol"
            )
        )
        payload[protocol_key] = {
            "protocol_valid": True,
            "paper_comparable": paper_comparable,
        }
        if root_motion:
            payload[protocol_key].update(
                {
                    "motion_protocol": motion_protocol,
                    "dynamic_episode_accounting_schema": (
                        "trigger-eligibility-smooth-prefix-v1"
                    ),
                    "pre_intervention_failure_policy": (
                        "retain_failure_with_null_intervention_effectiveness"
                    ),
                    "pre_intervention_success_policy": (
                        "fail_closed_unexercised_dynamic_condition"
                    ),
                    "smooth_terminal_progress_policy": (
                        "strict_effective_prefix_until_episode_terminal"
                    ),
                    "trigger_control_step": 60,
                    "episodes_intervention_eligible": 200,
                    "episodes_pre_intervention_terminal": 0,
                    "episodes_with_intervention": 200,
                    "episodes_with_effective_intervention": (
                        200
                        if effective_interventions is None
                        else effective_interventions
                    ),
                    "all_episodes_intervened": True,
                    "all_interventions_effective": (
                        effective_interventions in {None, 200}
                    ),
                    "all_eligible_interventions_effective": (
                        effective_interventions in {None, 200}
                    ),
                    **(
                        {
                            "smooth_motion_calls": 10,
                            "intervention_max_attempts": 20,
                        }
                        if table_i_schema
                        else {
                            "smooth_interpolation_calls": (
                                10 if scenario == "smooth" else None
                            ),
                            "max_sampling_attempts": 20,
                        }
                    ),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_defaults_select_v3_with_an_explicit_model_identity() -> None:
    args = build_parser().parse_args([])
    document = build_document(
        [],
        [],
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        release=args.release,
    )

    assert DEFAULT_RELEASE == "v3"
    assert args.release == "v3"
    assert args.markdown_output == DEFAULT_OUTPUT_DIR / "paper_comparison.md"
    assert args.csv_output == DEFAULT_OUTPUT_DIR / "paper_comparison.csv"
    assert args.json_output == DEFAULT_OUTPUT_DIR / "paper_comparison.json"
    assert document["schema"] == "dynamac-paper-comparison-v3"
    assert document["selection"]["release"] == "v3"
    expected = document["selection"]["expected_model_identity"]
    assert expected["training_config"]["eq6_covariance_scope"] == (
        "eq5_weighted_subspace"
    )
    assert expected["training_config"]["eq5_position_weight"] == 1.0
    assert expected["training_config"]["eq5_rotation_weight"] == 0.0
    assert set(expected["evaluation_protocol_ids"]) == {
        "unimanual",
        "bimanual",
    }


def test_consolidated_report_separates_reproduction_and_diagnostic(tmp_path):
    _write_run(
        tmp_path / "table_i" / "stack_wine_static_seed0_n200_h1000.json",
        task="stack_wine",
        scenario="static",
        rate=1.0,
        family="table_i",
    )
    _write_run(
        tmp_path / "bimanual_put_bottle_in_fridge_static_seed0_n200_h1000.json",
        task="bimanual_put_bottle_in_fridge",
        scenario="static",
        rate=0.85,
    )
    _write_run(
        tmp_path / "bimanual_put_bottle_in_fridge_teleport_seed0_n200_h1000.json",
        task="bimanual_put_bottle_in_fridge",
        scenario="teleport",
        rate=0.80,
        paper_comparable=False,
    )

    records, warnings = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    document = build_document(
        records, warnings, seed=0, episodes=200, horizon=1000
    )
    text = markdown(document)

    stack_static = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "StackWine"
    )
    store_static = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )
    store_dynamic = next(
        row for row in records if row["table"] == "III" and row["task"] == "StoreBottle"
    )
    hardware = next(row for row in records if row["table"] == "IV")

    assert stack_static["status"] == "local reproduction"
    assert store_static["status"] == "local reproduction"
    assert store_dynamic["status"] == "non-comparable diagnostic"
    assert hardware["status"] == "hardware unavailable"
    assert "0.94" in text
    assert "non-comparable diagnostic" in text


def test_sweepdust_row_links_the_diagnosis_without_changing_selection(tmp_path):
    _write_run(
        tmp_path / "bimanual_sweep_to_dustpan_static_seed0_n200_h1000.json",
        task="bimanual_sweep_to_dustpan",
        scenario="static",
        rate=0.025,
    )

    records, warnings = build_records(
        tmp_path, seed=0, episodes=200, horizon=1000
    )
    document = build_document(
        records, warnings, seed=0, episodes=200, horizon=1000
    )
    sweep = next(
        row for row in records if row["table"] == "II" and row["task"] == "SweepDust"
    )

    assert sweep["local_success_rate"] == 0.025
    assert "[SweepDust diagnosis](sweep_dust_diagnosis.md)" in markdown(document)


def test_table_i_static_is_reproduction_despite_family_level_dynamic_flag(tmp_path):
    _write_run(
        tmp_path / "table_i" / "stack_wine_static_seed0_n200_h1000.json",
        task="stack_wine",
        scenario="static",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_static = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "StackWine"
    )

    assert stack_static["status"] == "local reproduction"


def test_v2_table_i_requires_authenticated_low_dim_scene_launch(tmp_path):
    path = tmp_path / "table_i" / "stack_wine_static_seed0_n200_h1000.json"
    _write_run(
        path,
        task="stack_wine",
        scenario="static",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["controller"]["scene_launch"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_static = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "StackWine"
    )

    assert stack_static["status"] == "invalid diagnostic"


def test_v2_root_motion_requires_configuration_tree_transaction_evidence(
    tmp_path,
):
    path = tmp_path / "table_i_dynamic" / "stack_wine_smooth_seed0_n200.json"
    _write_run(
        path,
        task="stack_wine",
        scenario="smooth",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["results"][73]["interventions"][0]["instance_preservation"][
        "task_configuration_tree_restored"
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_smooth = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Smooth dynamics"
        and row["task"] == "StackWine"
    )

    assert stack_smooth["status"] == "invalid diagnostic"


@pytest.mark.parametrize("scope", ("summary", "event"))
def test_v2_root_motion_rejects_old_motion_protocol(tmp_path, scope):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if scope == "summary":
        motion = payload["scenario_protocol"]["motion_protocol"]
    else:
        motion = payload["results"][73]["scenario_events"][0]["motion_protocol"]
    motion["protocol_id"] = (
        "rlbench-boundary-root-preserve-initialized-episode-v2"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )

    assert lift["status"] == "invalid diagnostic"


def test_v2_table_i_reads_real_unimanual_intervention_event_schema(tmp_path):
    path = tmp_path / "table_i_dynamic" / "stack_wine_teleport_seed0_n200.json"
    _write_run(
        path,
        task="stack_wine",
        scenario="teleport",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "protocol" in payload and "scenario_protocol" not in payload
    assert "interventions" in payload["results"][0]
    assert "scenario_events" not in payload["results"][0]

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_teleport = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Teleportation"
        and row["task"] == "StackWine"
    )

    assert stack_teleport["status"] == "non-comparable diagnostic"


def test_v2_table_i_smooth_reads_real_ten_step_schema_and_exact_endpoint(
    tmp_path,
):
    path = tmp_path / "table_i_dynamic" / "stack_wine_smooth_seed0_n200.json"
    _write_run(
        path,
        task="stack_wine",
        scenario="smooth",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload["results"][0]["interventions"]

    assert "protocol" in payload and "scenario_protocol" not in payload
    assert payload["protocol"]["smooth_motion_calls"] == 10
    assert len(events) == 10
    assert [event["smooth_call"] for event in events] == list(range(1, 11))
    assert [event["endpoint_fraction"] for event in events] == [
        index / 10 for index in range(1, 11)
    ]
    assert all(event["goal_root_pose_reached"] is False for event in events[:-1])
    assert all(event["endpoint_applied"] is False for event in events[:-1])
    assert events[-1]["goal_root_pose_reached"] is True
    assert events[-1]["endpoint_applied"] is True

    # A wide goal tolerance may be entered before the scheduled endpoint. The
    # measured flag may then be true, while endpoint/complete remain false.
    events[4]["goal_root_translation_residual_m"] = 0.0
    events[4]["goal_root_pose_reached"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_smooth = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Smooth dynamics"
        and row["task"] == "StackWine"
    )
    assert stack_smooth["status"] == "non-comparable diagnostic"


def test_v2_dynamic_report_accepts_authenticated_pretrigger_failure(tmp_path):
    path = tmp_path / "table_i_dynamic" / "place_cups_teleport_seed0_n200.json"
    _write_run(
        path,
        task="place_cups",
        scenario="teleport",
        rate=0.9,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["results"][199]
    row.update(
        {
            "success": False,
            "steps": 16,
            "reason": "primary_action_retry_exhausted",
            "interventions": [],
            "intervention_eligible": False,
            "intervention_reached": False,
            "pre_intervention_terminal": True,
            "intervention_effective": None,
            "intervention_complete": None,
        }
    )
    protocol = payload["protocol"]
    protocol.update(
        {
            "episodes_intervention_eligible": 199,
            "episodes_pre_intervention_terminal": 1,
            "episodes_with_intervention": 199,
            "episodes_with_effective_intervention": 199,
            "all_episodes_intervened": False,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    place = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Teleportation"
        and row["task"] == "PlaceCups"
    )
    assert place["status"] == "non-comparable diagnostic"


def test_v2_dynamic_report_accepts_terminal_smooth_prefix(tmp_path):
    path = tmp_path / "table_i_dynamic" / "stack_wine_smooth_seed0_n200.json"
    _write_run(
        path,
        task="stack_wine",
        scenario="smooth",
        rate=0.9,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["results"][199]
    row["interventions"] = row["interventions"][:3]
    row["steps"] = 63
    row["intervention_complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Smooth dynamics"
        and row["task"] == "StackWine"
    )
    assert stack["status"] == "non-comparable diagnostic"


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_accounting_schema",
        "missing_row_eligibility",
        "preterminal_after_trigger",
        "preterminal_success",
        "preterminal_with_event",
        "wrong_summary_count",
    ),
)
def test_v2_dynamic_report_rejects_legacy_or_forged_preterminal_rows(
    tmp_path,
    mutation,
):
    path = tmp_path / "table_i_dynamic" / "place_cups_teleport_seed0_n200.json"
    _write_run(
        path,
        task="place_cups",
        scenario="teleport",
        rate=0.9,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    row = payload["results"][199]
    if mutation == "missing_accounting_schema":
        del protocol["dynamic_episode_accounting_schema"]
    elif mutation == "missing_row_eligibility":
        del row["intervention_eligible"]
    else:
        original_events = row["interventions"]
        row.update(
            {
                "success": False,
                "steps": 16,
                "reason": "primary_action_retry_exhausted",
                "interventions": [],
                "intervention_eligible": False,
                "intervention_reached": False,
                "pre_intervention_terminal": True,
                "intervention_effective": None,
                "intervention_complete": None,
            }
        )
        protocol.update(
            {
                "episodes_intervention_eligible": 199,
                "episodes_pre_intervention_terminal": 1,
                "episodes_with_intervention": 199,
                "episodes_with_effective_intervention": 199,
                "all_episodes_intervened": False,
            }
        )
        if mutation == "preterminal_after_trigger":
            row["steps"] = 61
        elif mutation == "preterminal_success":
            row["success"] = True
            row["reason"] = "success"
            payload["successes"] += 1
            payload["success_rate"] = payload["successes"] / 200
        elif mutation == "preterminal_with_event":
            row["interventions"] = original_events
        elif mutation == "wrong_summary_count":
            protocol["episodes_pre_intervention_terminal"] = 2
        else:  # pragma: no cover - parametrization is exhaustive
            raise AssertionError(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    place = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Teleportation"
        and row["task"] == "PlaceCups"
    )
    assert place["status"] == "invalid diagnostic"


def test_v2_table_iii_reads_real_bimanual_teleport_schema(tmp_path):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    event = payload["results"][0]["scenario_events"][0]

    assert "scenario_protocol" in payload and "protocol" not in payload
    assert "interventions" not in payload["results"][0]
    assert event["actual_root_motion"] is True
    assert event["commanded_root_pose_reached"] is True
    assert event["goal_root_pose_reached"] is True

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )
    assert lift["status"] == "non-comparable diagnostic"


@pytest.mark.parametrize(
    ("scope", "field", "bad_value"),
    (
        (
            "summary",
            "goal_validation",
            "workspace_fit_robot_collision_and_task_validate",
        ),
        (
            "summary",
            "root_command_translation_tolerance_m",
            ROOT_COMMAND_TRANSLATION_TOLERANCE_M * 2.0,
        ),
        (
            "event",
            "robot_collision_validation",
            "reject_any_source_or_goal_collision",
        ),
        (
            "event",
            "grasped_tool_collision_semantics",
            "task_specific_tool_exclusions",
        ),
    ),
)
def test_v4_root_motion_requires_complete_protocol_metadata(
    tmp_path,
    scope,
    field,
    bad_value,
):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = (
        payload["scenario_protocol"]["motion_protocol"]
        if scope == "summary"
        else payload["results"][73]["scenario_events"][0]["motion_protocol"]
    )
    protocol[field] = bad_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )
    assert lift["status"] == "invalid diagnostic"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("task_validate_called", True),
        ("low_dim_state_roundtrip_comparison_mode", "scalar_max_abs"),
        ("low_dim_state_roundtrip_chunk_count", 0),
        (
            "low_dim_state_roundtrip_max_translation_m",
            LOW_DIM_POSE_TRANSLATION_TOLERANCE_M * 2.0,
        ),
        (
            "low_dim_state_roundtrip_max_rotation_rad",
            LOW_DIM_POSE_ROTATION_TOLERANCE_RAD * 2.0,
        ),
        (
            "configuration_tree_rollback",
            "task_then_current_robot_components_after_each_attempt",
        ),
        ("live_robot_state_untouched", False),
        ("live_robot_configuration_trees_accessed", True),
        ("waypoint_cache_identity_preserved", False),
        ("robot_configuration_trees_restored", True),
        ("robot_collision_pair_policy", "reject_all_collision_pairs"),
        ("source_robot_external_collision_pairs", None),
        (
            "goal_robot_external_collision_pairs",
            [
                {
                    "arm": "right_arm",
                    "external_object_handle": 17,
                    "external_object_name": "new_contact",
                }
            ],
        ),
        (
            "goal_new_robot_external_collision_pairs",
            [
                {
                    "arm": "right_arm",
                    "external_object_handle": 17,
                    "external_object_name": "new_contact",
                }
            ],
        ),
        ("sampling_attempts_rejected_for_new_robot_collision_pairs", -1),
        ("sampling_attempts_rejected_for_new_robot_collision_pairs", 1),
    ),
)
def test_v4_root_motion_rejects_incomplete_preservation_evidence(
    tmp_path,
    field,
    bad_value,
):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    preservation = payload["results"][73]["scenario_events"][0][
        "instance_preservation"
    ]
    preservation[field] = bad_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )
    assert lift["status"] == "invalid diagnostic"


def test_v4_root_motion_accepts_sign_invariant_pose_roundtrip_evidence(tmp_path):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    preservation = payload["results"][73]["scenario_events"][0][
        "instance_preservation"
    ]
    preservation["low_dim_state_roundtrip_l2"] = 4.0
    preservation["low_dim_state_roundtrip_max_abs"] = (
        2.0 + LOW_DIM_STATE_ROUNDTRIP_ATOL
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )
    assert lift["status"] == "non-comparable diagnostic"


def test_v4_root_motion_accepts_preserved_source_contact_pair(tmp_path):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    preservation = payload["results"][73]["scenario_events"][0][
        "instance_preservation"
    ]
    pair = {
        "arm": "right_arm",
        "external_object_handle": 17,
        "external_object_name": "already_grasped_tool",
    }
    preservation["source_robot_external_collision_pairs"] = [pair]
    preservation["goal_robot_external_collision_pairs"] = [pair]
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )
    assert lift["status"] == "non-comparable diagnostic"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("actual_root_motion", False),
        ("actual_root_translation_m", 0.0),
        (
            "commanded_root_translation_residual_m",
            ROOT_COMMAND_TRANSLATION_TOLERANCE_M * 2.0,
        ),
        ("commanded_root_pose_reached", False),
        (
            "goal_root_translation_residual_m",
            ROOT_COMMAND_TRANSLATION_TOLERANCE_M * 2.0,
        ),
        ("goal_root_pose_reached", False),
        ("policy_observation_refreshed", False),
    ),
)
def test_v4_teleport_requires_actual_reached_motion(
    tmp_path,
    field,
    bad_value,
):
    path = tmp_path / "bimanual_lift_tray_teleport_seed0_n200_h1000.json"
    _write_run(
        path,
        task="bimanual_lift_tray",
        scenario="teleport",
        rate=0.9,
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    event = payload["results"][73]["scenario_events"][0]
    event[field] = bad_value
    if field == "actual_root_translation_m":
        event["actual_root_rotation_rad"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    lift = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Dynamic environment"
        and row["task"] == "LiftTray"
    )
    assert lift["status"] == "invalid diagnostic"


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_endpoint",
        "summary_calls_not_ten",
        "intermediate_goal_flag_mismatch",
        "endpoint_not_applied",
        "endpoint_goal_not_reached",
    ),
)
def test_v4_smooth_requires_ten_steps_and_final_endpoint(
    tmp_path,
    mutation,
):
    path = tmp_path / "table_i_dynamic" / "stack_wine_smooth_seed0_n200.json"
    _write_run(
        path,
        task="stack_wine",
        scenario="smooth",
        rate=1.0,
        family="table_i",
        paper_comparable=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload["results"][73]["interventions"]
    if mutation == "missing_endpoint":
        events.pop()
    elif mutation == "summary_calls_not_ten":
        payload["protocol"]["smooth_motion_calls"] = 9
    elif mutation == "intermediate_goal_flag_mismatch":
        events[4]["goal_root_pose_reached"] = True
    elif mutation == "endpoint_not_applied":
        events[-1]["endpoint_applied"] = False
    elif mutation == "endpoint_goal_not_reached":
        events[-1]["goal_root_translation_residual_m"] = (
            ROOT_COMMAND_TRANSLATION_TOLERANCE_M * 2.0
        )
        events[-1]["goal_root_pose_reached"] = False
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    stack_smooth = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Smooth dynamics"
        and row["task"] == "StackWine"
    )
    assert stack_smooth["status"] == "invalid diagnostic"


def test_wrong_episode_count_is_not_selected(tmp_path):
    path = tmp_path / "bimanual_handover_item_static_seed0_n100_h1000.json"
    successes = 90
    payload = {
        "task": "bimanual_handover_item",
        "scenario": "static",
        "seed": 0,
        "episodes": 100,
        "horizon": 1000,
        "successes": successes,
        "success_rate": 0.9,
        "results": [
            {"episode": episode, "success": episode < successes}
            for episode in range(100)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    handover = next(
        row for row in records if row["table"] == "II" and row["task"] == "HandOver"
    )
    assert handover["status"] == "pending"
    assert handover["local_success_rate"] is None


def test_table_i_selection_requires_variation_zero(tmp_path):
    _write_run(
        tmp_path / "table_i" / "place_cups_static_variation1_seed0_n200_h1000.json",
        task="place_cups",
        scenario="static",
        rate=1.0,
        family="table_i",
        variation=1,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    place = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "PlaceCups"
    )

    assert place["status"] == "pending"
    assert place["local_success_rate"] is None


def test_corrected_self_identifying_model_is_preferred_over_legacy_result(tmp_path):
    task = "bimanual_put_bottle_in_fridge"
    _write_run(
        tmp_path / f"{task}_static_seed0_n200_h1000.json",
        task=task,
        scenario="static",
        rate=0.85,
        corrected=False,
    )
    _write_run(
        tmp_path / "ridge_local" / f"{task}_static_seed0_n200_h1000.json",
        task=task,
        scenario="static",
        rate=0.80,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    store = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )

    assert store["local_success_rate"] == 0.80
    assert store["source_file"].startswith("ridge_local/")


def test_policy_server_full_config_is_recognized_as_corrected(tmp_path):
    raw_config = json.loads(
        (INTEGRATION_ROOT / "configs" / "dynamac_rlbench_local.json").read_text(
            encoding="utf-8"
        )
    )
    policy_server_config = asdict(DynaMACConfig(**raw_config))
    assert len(policy_server_config) > len(raw_config)

    path = tmp_path / "table_i" / "place_cups_static_seed0_n200_h1000.json"
    _write_run(
        path,
        task="place_cups",
        scenario="static",
        rate=0.99,
        family="table_i",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_identity"]["training_config"] = policy_server_config
    path.write_text(json.dumps(payload), encoding="utf-8")

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    place = next(
        row
        for row in records
        if row["table"] == "I"
        and row["condition"] == "Static"
        and row["task"] == "PlaceCups"
    )

    assert place["status"] == "local reproduction"
    assert place["local_success_rate"] == 0.99


def test_two_corrected_checkpoint_identities_fail_closed(tmp_path):
    task = "bimanual_put_bottle_in_fridge"
    for directory, fingerprint in (("run_a", "fingerprint-a"), ("run_b", "fingerprint-b")):
        _write_run(
            tmp_path / directory / f"{task}_static_seed0_n200_h1000.json",
            task=task,
            scenario="static",
            rate=0.80,
            fingerprint=fingerprint,
        )

    with pytest.raises(RuntimeError, match="multiple corrected results"):
        build_records(tmp_path, seed=0, episodes=200, horizon=1000)


def test_release_filter_allows_versioned_results_to_coexist(tmp_path):
    task = "bimanual_put_bottle_in_fridge"
    for release, fingerprint in (("v1", "fingerprint-v1"), ("v2", "fingerprint-v2")):
        _write_run(
            tmp_path / release / f"{task}_static_seed0_n200_h1000.json",
            task=task,
            scenario="static",
            rate=0.80,
            fingerprint=fingerprint,
            training_config=EXPECTED_RELEASE_CONFIGS[release],
        )

    records, _ = build_records(
        tmp_path,
        seed=0,
        episodes=200,
        horizon=1000,
        release="v1",
    )
    store = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )

    assert store["source_file"].startswith("v1/")
    assert store["status"] == "local reproduction"

    records, _ = build_records(
        tmp_path,
        seed=0,
        episodes=200,
        horizon=1000,
        release="v2",
    )
    store = next(
        row for row in records if row["table"] == "II" and row["task"] == "StoreBottle"
    )
    assert store["source_file"].startswith("v2/")
    assert store["status"] == "local reproduction"


def test_authenticated_coordination_result_is_a_table_iii_diagnostic(tmp_path):
    _write_run(
        tmp_path / "table_iii_coordination" / "hand_left_seed0_n200_h1000.json",
        task="bimanual_handover_item_dynamic",
        scenario="coordination_hand_left",
        rate=0.75,
        paper_comparable=False,
    )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    hand_left = next(
        row
        for row in records
        if row["table"] == "III"
        and row["condition"] == "Coordination"
        and row["task"] == "Hand Left"
    )

    assert hand_left["status"] == "non-comparable diagnostic"
    assert hand_left["local_success_rate"] == 0.75


def test_legacy_coordination_n1_smoke_is_silently_ignored(tmp_path):
    smoke = tmp_path / "table_iii_coordination" / "smoke_left_seed10000_n1.json"
    smoke.parent.mkdir(parents=True)
    smoke.write_text(
        json.dumps(
            {
                "task": "bimanual_handover_item_dynamic",
                "scenario": "coordination_hand_left",
                "seed": 10000,
                "episodes": [{"episode": 0, "success": False}],
                "horizon": 1000,
                "successes": 0,
                "success_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )

    runs, warnings = discover_runs(tmp_path)

    assert runs == []
    assert warnings == []


def test_wipe_desk_dynamic_cells_are_runnable_and_pending_without_results(tmp_path):
    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)

    rows = [
        row
        for row in records
        if row["table"] == "I"
        and row["task"] == "WipeDesk"
        and row["condition"] != "Static"
    ]

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"pending"}
    assert all("preserve-instance" in row["notes"] for row in rows)


def test_place_cups_without_a_complete_corrected_run_is_pending(tmp_path):
    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    rows = [
        row
        for row in records
        if row["table"] == "I" and row["task"] == "PlaceCups"
    ]

    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"pending"}
    assert all(row["local_success_rate"] is None for row in rows)
    assert all(row["local_episodes"] is None for row in rows)


def test_table_iii_uses_teleport_and_ignores_extra_smooth_run(tmp_path):
    task = "bimanual_lift_tray"
    _write_run(
        tmp_path / f"{task}_teleport_seed0_n200_h1000.json",
        task=task,
        scenario="teleport",
        rate=0.90,
        paper_comparable=False,
        event={"applied": True, "task_state_l2": 0.3, "root_pose_l2": 0.2},
        effective_interventions=200,
    )
    _write_run(
        tmp_path / f"{task}_smooth_seed0_n200_h1000.json",
        task=task,
        scenario="smooth",
        rate=0.92,
        paper_comparable=False,
        event={
            "applied": True,
            "task_state_l2": 0.2,
            "root_pose_l2": 0.02,
            "planned_root_translation_m": 0.15,
        },
        effective_interventions=200,
    )

    records, warnings = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    document = build_document(
        records,
        warnings,
        seed=0,
        episodes=200,
        horizon=1000,
    )
    lift = next(
        row
        for row in records
        if row["table"] == "III" and row["task"] == "LiftTray"
    )

    assert lift["local_success_rate"] == 0.90
    assert lift["source_file"].endswith("_teleport_seed0_n200_h1000.json")
    assert "median root-pose L2 0.200" in lift["protocol_note"]
    assert "Additional local diagnostics" not in markdown(document)


def test_table_iii_teleport_rows_use_one_generic_material_note(tmp_path):
    cases = (
        ("bimanual_put_bottle_in_fridge", "StoreBottle", 0.8, 0.0),
        ("bimanual_handover_item", "HandOver", 2.0, 0.0),
        ("bimanual_sweep_to_dustpan", "SweepDust", 1.0e-6, 0.0),
        ("bimanual_lift_tray", "LiftTray", 0.3, 0.2),
    )
    for task, _label, state_l2, root_l2 in cases:
        _write_run(
            tmp_path / f"{task}_teleport_seed0_n200_h1000.json",
            task=task,
            scenario="teleport",
            rate=0.5,
            paper_comparable=False,
            event={
                "applied": True,
                "task_state_l2": state_l2,
                "root_pose_l2": root_l2,
            },
            effective_interventions=200,
        )

    records, _ = build_records(tmp_path, seed=0, episodes=200, horizon=1000)
    notes = {
        row["task"]: row["protocol_note"]
        for row in records
        if row["table"] == "III" and row["condition"] == "Dynamic environment"
    }

    for note in notes.values():
        assert "Task-root intervention" in note
        assert PRESERVE_INSTANCE_MOTION_PROTOCOL_ID in note
    assert "median root-pose L2 0.000" in notes["StoreBottle"]
    assert "median root-pose L2 0.200" in notes["LiftTray"]
