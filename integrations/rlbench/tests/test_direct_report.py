from __future__ import annotations

import hashlib
import json

import pytest

from integrations.rlbench.rlbench_dynamac import direct_report
from integrations.rlbench.rlbench_dynamac.direct_policy import (
    TRAINING_MANIFEST_SCHEMA_V3,
    V3_ADAPTER_PROTOCOL,
)
from integrations.rlbench.rlbench_dynamac.direct_report import TASKS, load_rows, markdown
from integrations.rlbench.rlbench_dynamac.paper_comparison import (
    EXPECTED_LOCAL_CONFIG,
    EXPECTED_SELECTION_SEMANTICS_ID,
    EXPECTED_TAPAS_COMMIT,
    LocalRun,
    _valid_v3_final_settling,
    expected_evaluation_protocol_id,
)
from integrations.rlbench.rlbench_dynamac.runtime import (
    FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
    FRESH_TASK_GENERATION_PROTOCOL_ID,
    final_settling_metadata,
)
from integrations.rlbench.rlbench_dynamac.v3_protocol import (
    dynamic_trigger_profile,
    load_v3_intervention_protocol,
    resolve_authenticated_v3_trigger,
)


def _sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _model_identity(task: str) -> dict[str, object]:
    protocol = load_v3_intervention_protocol()
    profile = dynamic_trigger_profile(task, protocol)
    checkpoint_fingerprint = f"{task}-checkpoint-audit"
    anchor = {
        "anchor_arm": profile["anchor_arm"],
        "skill_label": profile["skill_label"],
        "duration": profile["expected_duration"],
        "evidence_frame": profile["evidence_frame"],
        "local_tick": profile["local_tick"],
        "phase": profile["phase"],
        "phase_formula": "local_tick / (skill_duration - 1)",
        "interaction_arm": profile["interaction_arm"],
        "interaction_object": profile["interaction_object"],
        "interaction_event": profile["interaction_event"],
        "expected_gripper_state": profile["expected_gripper_state"],
        "resolved_global_tick": profile["local_tick"],
        "selected_by_eq6": [True],
        "availability_runs": [[[0, profile["expected_duration"] - 1]]],
        "poe_active_runs": [[[0, profile["expected_duration"] - 1]]],
        "required_active_window": profile["required_active_window"],
        "validated": True,
    }
    envelope = {
        "schema": "dynamac-v3-trigger-anchor-evidence-v1",
        "intervention_protocol_schema": protocol["schema"],
        "intervention_protocol_fingerprint": protocol["fingerprint"],
        "profile_family": "dynamic_environment",
        "checkpoint_trigger_audit_fingerprint": checkpoint_fingerprint,
        "anchors": {task: anchor},
        "validated": True,
    }
    envelope["fingerprint"] = _sha256(envelope)
    return {
        "manifest_authenticated": True,
        "training_manifest_schema": TRAINING_MANIFEST_SCHEMA_V3,
        "training_adapter_protocol": V3_ADAPTER_PROTOCOL,
        "checkpoint_trigger_audit_fingerprint": checkpoint_fingerprint,
        "v3_trigger_anchor_evidence": envelope,
        "training_config": EXPECTED_LOCAL_CONFIG,
        "model_schema_version": 13,
        "selection_semantics_id": EXPECTED_SELECTION_SEMANTICS_ID,
        "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
        "left_fingerprint": f"{task}-left",
        "right_fingerprint": f"{task}-right",
    }


def _fresh_evidence(task, episode, seed, variation):
    body = {
        "schema": FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
        "protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
        "generation_index": episode + 1,
        "episode_seed": seed + episode,
        "variation": variation,
        "task_name": task,
        "physics_running_before_stop": episode > 0,
        "physics_stopped_before_task_reload": True,
        "previous_task_present": episode > 0,
        "previous_task_unloaded_before_stop": episode > 0,
        "previous_task_unloaded_while_physics_running": episode > 0,
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
    return {**body, "fingerprint": _sha256(body)}


def _write_results(tmp_path, *, seed: int = 0) -> None:
    for task, _label, _paper_rate in TASKS:
        identity = _model_identity(task)
        authentication = resolve_authenticated_v3_trigger(identity, task=task)
        settling_protocol = final_settling_metadata()
        not_entered_settling = {
            **settling_protocol,
            "attempted": False,
            "available": True,
            "steps_executed": 0,
            "first_terminal_step": None,
            "stop_reason": "not_entered",
            "success": False,
            "terminate": False,
        }
        variation_count = 5 if task == "bimanual_handover_item" else 1
        variation_schedule = [episode % variation_count for episode in range(2)]
        fresh_generations = [
            _fresh_evidence(task, episode, seed, variation)
            for episode, variation in enumerate(variation_schedule)
        ]
        payload = {
            "task": task,
            "scenario": "static",
            "seed": seed,
            "episodes": 2,
            "horizon": 1000,
            "variation_count": variation_count,
            "variation_schedule": variation_schedule,
            "successes": 1,
            "success_rate": 0.5,
            "evaluation_protocol_id": expected_evaluation_protocol_id(task),
            "model_identity": identity,
            "scenario_protocol": {
                "status": "STATIC_REFERENCE",
                "trigger_reference_domain": "successfully_committed_policy_ticks",
                "trigger_policy_step": None,
                "trigger_authentication": authentication,
                "intervention_registry_schema": (
                    load_v3_intervention_protocol()["schema"]
                ),
                "intervention_registry_fingerprint": (
                    load_v3_intervention_protocol()["fingerprint"]
                ),
                "protocol_valid": True,
                "paper_comparable": True,
            },
            "controller": {
                "policy_clock_rollback": True,
                "policy_clock_semantics_id": (
                    "policy-tick-transaction-commit-on-primary-action-success-v1"
                ),
                "final_settling": settling_protocol,
                "formal_episode_initialization": (
                    FRESH_TASK_GENERATION_PROTOCOL_ID
                ),
            },
            "final_settling_protocol": settling_protocol,
            "episode_accounting": {
                "schema": (
                    "planned-denominator-trigger-completion-conditional-success-v3"
                ),
                "planned_episode_denominator": 2,
                "completed_episode_count": 2,
                "successes_in_planned_denominator": 1,
                "success_rate_all_planned_episodes": 0.5,
                "trigger_reached_count": 0,
                "intervention_complete_count": 0,
                "dynamic_condition_unexercised_count": 0,
                "pre_trigger_success_count": 0,
                "complete_intervention_subset_count": 0,
                "successes_in_complete_intervention_subset": 0,
                "success_rate_in_complete_intervention_subset": None,
            },
            "results": [
                {
                    "episode": 0,
                    "success": True,
                    "reason": "success",
                    "invalid_actions": 0,
                    "scenario_events": [],
                    "trigger_step": None,
                    "intervention_eligible": False,
                    "intervention_reached": False,
                    "pre_intervention_terminal": False,
                    "dynamic_condition_exercised": False,
                    "dynamic_condition_unexercised": None,
                    "intervention_effective": None,
                    "intervention_complete": None,
                    "motion_plan_fingerprint": None,
                    "motion_plan_evidence": None,
                    "final_settling": not_entered_settling,
                    "fresh_task_generation": fresh_generations[0],
                },
                {
                    "episode": 1,
                    "success": False,
                    "reason": "terminate",
                    "invalid_actions": 3,
                    "scenario_events": [],
                    "trigger_step": None,
                    "intervention_eligible": False,
                    "intervention_reached": False,
                    "pre_intervention_terminal": False,
                    "dynamic_condition_exercised": False,
                    "dynamic_condition_unexercised": None,
                    "intervention_effective": None,
                    "intervention_complete": None,
                    "motion_plan_fingerprint": None,
                    "motion_plan_evidence": None,
                    "final_settling": not_entered_settling,
                    "fresh_task_generation": fresh_generations[1],
                },
            ],
            "fresh_task_generation": {
                "required_per_formal_episode": True,
                "all_episodes_recorded": True,
                "evidence": fresh_generations,
            },
        }
        path = tmp_path / f"{task}_static_seed{seed}_n2_h1000.json"
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_direct_report_validates_and_summarizes_runs(tmp_path, monkeypatch) -> None:
    _write_results(tmp_path)
    monkeypatch.setattr(direct_report, "_valid_v3_static_protocol", lambda run: True)

    rows = load_rows(tmp_path, seed=0, episodes=2, horizon=1000)
    report = markdown(rows, seed=0)

    assert len(rows) == 4
    assert rows[0]["invalid_actions"] == 3
    assert rows[0]["termination_reasons"] == {
        "success": 1,
        "terminate": 1,
    }
    assert "1/2" in report
    assert "success=1, terminate=1" in report


def test_direct_report_rejects_identity_mismatch(tmp_path) -> None:
    _write_results(tmp_path)
    task = TASKS[0][0]
    path = tmp_path / f"{task}_static_seed0_n2_h1000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenario"] = "teleport"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        load_rows(tmp_path, seed=0, episodes=2, horizon=1000)


def test_direct_report_rejects_non_v3_manifest(tmp_path) -> None:
    _write_results(tmp_path)
    task = TASKS[0][0]
    path = tmp_path / f"{task}_static_seed0_n2_h1000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_identity"]["training_manifest_schema"] = (
        "dynamac-direct-training-v2"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="v3 protocol/model identity"):
        load_rows(tmp_path, seed=0, episodes=2, horizon=1000)


def test_direct_report_accepts_rlbench_success_terminate_settling_pair(
    tmp_path,
    monkeypatch,
) -> None:
    _write_results(tmp_path)
    monkeypatch.setattr(direct_report, "_valid_v3_static_protocol", lambda run: True)
    for task, _label, _paper_rate in TASKS:
        path = tmp_path / f"{task}_static_seed0_n2_h1000.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        settling = payload["results"][0]["final_settling"]
        settling.update(
            {
                "attempted": True,
                "steps_executed": 2,
                "first_terminal_step": 2,
                "stop_reason": "success",
                "success": True,
                "terminate": True,
            }
        )
        payload["results"][0]["reason"] = "success_after_final_settling"
        path.write_text(json.dumps(payload), encoding="utf-8")
        run = LocalRun(
            path=path,
            task=task,
            scenario="static",
            seed=0,
            episodes=2,
            horizon=1000,
            variation=0,
            successes=1,
            success_rate=0.5,
            payload=payload,
        )
        assert _valid_v3_final_settling(run) is True

    rows = load_rows(tmp_path, seed=0, episodes=2, horizon=1000)
    assert len(rows) == len(TASKS)


def test_direct_report_rejects_forged_top_and_accounting_rate(tmp_path) -> None:
    _write_results(tmp_path)
    task = TASKS[0][0]
    path = tmp_path / f"{task}_static_seed0_n2_h1000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["success_rate"] = 0.75
    payload["episode_accounting"]["success_rate_all_planned_episodes"] = 0.75
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="success rate mismatch"):
        load_rows(tmp_path, seed=0, episodes=2, horizon=1000)
