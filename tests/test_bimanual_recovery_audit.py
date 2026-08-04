from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from essay2608.data.handover_schema import HandoverState
from essay2608.eval.bimanual_recovery_audit import (
    _controller_aggregate,
    _experiment_fingerprint,
    audit_bimanual_recovery_results,
)
from essay2608.eval.bimanual_recovery_study import (
    fault_realization,
    score_bimanual_recovery_trace,
    task_outcome_from_trace,
)
from essay2608.eval.bimanual_relation_study import score_bimanual_relation_trace


def test_bimanual_recovery_audit_recomputes_terminal_and_information_boundary(
    tmp_path: Path,
) -> None:
    repository = tmp_path
    config_dir = repository / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    relation_config = config_dir / "relation.json"
    recovery_config = config_dir / "recovery.json"
    relation_config.write_text("{}\n", encoding="utf-8")
    recovery_config.write_text("{}\n", encoding="utf-8")
    relation_sha = hashlib.sha256(relation_config.read_bytes()).hexdigest()
    recovery_sha = hashlib.sha256(b"{}").hexdigest()
    protocol = {
        "protocol_version": "test",
        "summary_artifact_type": "bimanual_relation_recovery_development",
        "task_id": "test-task",
        "source_sha256": "source",
        "relation_config": "configs/experiments/relation.json",
        "relation_config_sha256": relation_sha,
        "recovery_config": "configs/experiments/recovery.json",
        "recovery_config_sha256": recovery_sha,
        "dataset_sha256": "dataset",
        "methods": ["clocked_expert"],
        "conditions": ["normal"],
        "formal_seeds": [1],
        "maximum_steps": 10,
        "control_dt_s": 0.02,
        "maximum_regrasp_attempts": 2,
        "maximum_recovery_time_s": {},
        "maximum_recovery_action_target_jump_m": 0.08,
        "maximum_recovery_supervised_target_jump_m": 0.08,
        "maximum_recovery_ee_speed_m_s": 0.75,
        "minimum_online_four_value_accuracy": 0.94,
        "minimum_task_successes": {"clocked_expert": {"normal": 1}},
        "strong_fault_conditions": [],
        "minimum_paired_strong_recovery_wins": 0,
        "maximum_paired_strong_recovery_losses": 0,
        "maximum_online_oracle_success_gap": 0,
        "verify_current_sources": False,
    }
    protocol_path = config_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    steps = 5
    states = np.asarray(
        [
            HandoverState.REST,
            HandoverState.LEFT_GRASP,
            HandoverState.TRANSFER,
            HandoverState.RIGHT_TO_TARGET,
            HandoverState.RIGHT_RELEASE,
        ],
        dtype=int,
    )
    left = np.asarray([0, 1, 1, 0, 0], dtype=bool)
    right = np.asarray([0, 0, 1, 1, 0], dtype=bool)
    labels = np.asarray(
        ["none", "left_only", "both", "right_only", "none"], dtype="U32"
    )
    poses = np.zeros((steps, 7), dtype=np.float64)
    poses[:, 2] = 0.181
    poses[:, 3] = 1.0
    forces = np.zeros((steps, 2, 3), dtype=np.float64)
    finger_positions = np.zeros((steps, 2, 3), dtype=np.float64)
    actions = np.zeros((steps, 16), dtype=np.float64)
    no_events = np.full(steps, "none", dtype="U64")
    not_applicable = np.full(steps, "NOT_APPLICABLE", dtype="U64")
    inferred_left_state = np.where(left, "CONNECTED", "DISCONNECTED")
    inferred_right_state = np.where(right, "CONNECTED", "DISCONNECTED")
    arrays = {
        "state": states,
        "left_ee_pose": poses,
        "right_ee_pose": poses,
        "object_pose": poses,
        "target_pose": poses,
        "left_finger_force": forces,
        "right_finger_force": forces,
        "left_finger_position": finger_positions,
        "right_finger_position": finger_positions,
        "left_finger_distance_m": np.zeros(steps),
        "right_finger_distance_m": np.zeros(steps),
        "left_finger_velocity_m_s": np.zeros(steps),
        "right_finger_velocity_m_s": np.zeros(steps),
        "base_action": actions,
        "supervised_action": actions,
        "applied_action": actions,
        "truth_left_connected": left,
        "truth_right_connected": right,
        "truth_label": labels,
        "inferred_left_connected": left,
        "inferred_right_connected": right,
        "inferred_left_confidence": left.astype(float),
        "inferred_right_confidence": right.astype(float),
        "inferred_left_connection_score": left.astype(float),
        "inferred_right_connection_score": right.astype(float),
        "inferred_left_loss_score": np.zeros(steps),
        "inferred_right_loss_score": np.zeros(steps),
        "inferred_left_state": inferred_left_state,
        "inferred_right_state": inferred_right_state,
        "inferred_label": labels,
        "control_left_state": not_applicable,
        "control_right_state": not_applicable,
        "recovery_state": not_applicable,
        "recovery_trigger": np.full(steps, "NONE", dtype="U64"),
        "recovery_transition": no_events,
        "recovery_requires_giver": np.zeros(steps, dtype=bool),
        "expert_rebase_event": no_events,
        "recovery_action_overridden": np.zeros(steps, dtype=bool),
        "transfer_gate_active": np.zeros(steps, dtype=bool),
        "regrasp_attempts": np.zeros(steps, dtype=int),
        "phase_clock_held": np.zeros(steps, dtype=bool),
        "intervention_active": np.zeros(steps, dtype=bool),
        "intervention_event": no_events,
    }
    task_success, failure_reason, outcome = task_outcome_from_trace(
        expert_complete=True,
        expert_failed=False,
        expert_failure_reason=None,
        recovery_failed=False,
        environment_done=False,
        object_positions=poses[:, :3],
        final_position=poses[-1, :3],
        target_position=poses[-1, :3],
    )
    metrics = score_bimanual_recovery_trace(
        arrays,
        "normal",
        0.02,
        method="clocked_expert",
        task_success=task_success,
    )
    relation_metrics = score_bimanual_relation_trace(
        truth_labels=labels,
        inferred_labels=labels,
        truth_left=left,
        truth_right=right,
        inferred_left=left,
        inferred_right=right,
        control_dt_s=0.02,
    )
    realization = fault_realization("normal", arrays, 0.02)

    results = repository / "results"
    trials = results / "trials"
    trials.mkdir(parents=True)
    stem = "bimanual_recovery__clocked_expert__normal__seed_1"
    trace = {
        **arrays,
        "control_dt": np.asarray(0.02, dtype=np.float32),
        "method": np.asarray("clocked_expert"),
        "condition": np.asarray("normal"),
        "seed": np.asarray(1),
        "source_sha256": np.asarray("source"),
        "relation_config_sha256": np.asarray(relation_sha),
        "recovery_config_sha256": np.asarray(recovery_sha),
        "dataset_sha256": np.asarray("dataset"),
        "maximum_steps": np.asarray(10),
        "privileged_relation_used_for_control": np.asarray(False),
        "terminal_left_pose": poses[-1],
        "terminal_right_pose": poses[-1],
        "terminal_object_pose": poses[-1],
        "terminal_target_pose": poses[-1],
        "expert_complete": np.asarray(True),
        "expert_failed": np.asarray(False),
        "expert_failure_reason": np.asarray("none"),
        "recovery_failed": np.asarray(False),
        "environment_done": np.asarray(False),
    }
    np.savez_compressed(trials / f"{stem}.npz", **trace)
    result = {
        "artifact_type": "bimanual_relation_recovery_trial",
        "task_id": "test-task",
        "method": "clocked_expert",
        "condition": "normal",
        "seed": 1,
        "experiment_fingerprint": _experiment_fingerprint(
            protocol, "clocked_expert", "normal", 1
        ),
        "source_sha256": "source",
        "relation_config_sha256": relation_sha,
        "recovery_config_sha256": recovery_sha,
        "dataset_sha256": "dataset",
        "privileged_relation_used_for_control": False,
        "privileged_relation_used_by_online_methods": False,
        "control_dt_s": 0.02,
        "maximum_steps": 10,
        "steps": steps,
        "fault_realization": realization,
        "task_success": task_success,
        "task_failure_reason": failure_reason,
        "expert_complete": True,
        "expert_failed": False,
        "recovery_failed": False,
        "truth_relation_sequence": labels.tolist(),
        "inferred_relation_sequence": labels.tolist(),
        "relation_metrics": relation_metrics,
        "metrics": metrics,
        **outcome,
    }
    (trials / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
    summary_trial = {**result, "worker_returncode": 0}
    summary = {
        "artifact_type": "bimanual_relation_recovery_development",
        "task_id": "test-task",
        "source_sha256": "source",
        "relation_config_sha256": relation_sha,
        "recovery_config": {},
        "recovery_config_sha256": recovery_sha,
        "dataset_sha256": "dataset",
        "maximum_steps": 10,
        "control_dt_s": 0.02,
        "methods": ["clocked_expert"],
        "conditions": ["normal"],
        "seeds": [1],
        "num_expected_trials": 1,
        "num_valid_trials": 1,
        "all_faults_physically_realized": True,
        "by_method_condition": {
            "clocked_expert": {"normal": _controller_aggregate([summary_trial])}
        },
        "trials": [summary_trial],
    }
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = audit_bimanual_recovery_results(results, protocol_path)
    assert report["all_trials_passed"]
    assert report["trial_count"] == 1
