from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from essay2608.eval.bimanual_relation_audit import (
    _experiment_fingerprint,
    audit_bimanual_relation_results,
)
from essay2608.eval.bimanual_relation_study import (
    condition_realization,
    score_bimanual_relation_trace,
)


def test_formal_audit_recomputes_trace_and_exact_membership(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    protocol_dir = tmp_path / "configs" / "experiments"
    protocol_dir.mkdir(parents=True)
    protocol = {
        "protocol_version": "test",
        "task_id": "test-task",
        "source_sha256": "source",
        "estimator_config": "config.json",
        "config_sha256": config_sha,
        "dataset_sha256": "dataset",
        "formal_seeds": [1],
        "conditions": ["normal"],
        "max_steps": 10,
        "control_dt_s": 0.02,
        "minimum_four_value_accuracy": 1.0,
        "minimum_left_f1": 1.0,
        "minimum_right_f1": {"normal": 1.0},
        "maximum_left_transition_delay_s": 0.0,
        "maximum_right_transition_delay_s": {"normal": 0.0},
        "expected_inferred_sequences": {
            "normal": ["none", "left_only", "both", "right_only", "none"]
        },
        "task_success_conditions": ["normal"],
        "minimum_task_successes_per_required_condition": 1,
        "receiver_miss_maximum_right_false_positive_steps": 0,
        "verify_current_sources": False,
    }
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    results = tmp_path / "results"
    trials = results / "trials"
    trials.mkdir(parents=True)
    stem = "bimanual_relation__normal__seed_1"
    truth_label = np.asarray(
        ["none", "left_only", "both", "right_only", "none"], dtype="U32"
    )
    left = np.asarray([0, 1, 1, 0, 0], dtype=bool)
    right = np.asarray([0, 0, 1, 1, 0], dtype=bool)
    intervention_active = np.zeros(5, dtype=bool)
    intervention_event = np.full(5, "none", dtype="U32")
    metrics = score_bimanual_relation_trace(
        truth_labels=truth_label,
        inferred_labels=truth_label,
        truth_left=left,
        truth_right=right,
        inferred_left=left,
        inferred_right=right,
        control_dt_s=0.02,
    )
    realization = condition_realization(
        "normal",
        truth_left=left,
        truth_right=right,
        truth_labels=truth_label,
        intervention_active=intervention_active,
        intervention_event=intervention_event,
        control_dt_s=0.02,
    )
    arrays = {
        "state": np.arange(5),
        "left_ee_pose": np.zeros((5, 7)),
        "right_ee_pose": np.zeros((5, 7)),
        "object_pose": np.zeros((5, 7)),
        "left_finger_force": np.zeros((5, 2, 3)),
        "right_finger_force": np.zeros((5, 2, 3)),
        "left_finger_position": np.zeros((5, 2, 3)),
        "right_finger_position": np.zeros((5, 2, 3)),
        "left_finger_distance_m": np.zeros(5),
        "right_finger_distance_m": np.zeros(5),
        "left_finger_velocity_m_s": np.zeros(5),
        "right_finger_velocity_m_s": np.zeros(5),
        "base_action": np.zeros((5, 16)),
        "applied_action": np.zeros((5, 16)),
        "truth_left_connected": left,
        "truth_right_connected": right,
        "truth_left_confidence": left.astype(float),
        "truth_right_confidence": right.astype(float),
        "truth_label": truth_label,
        "inferred_left_connected": left,
        "inferred_right_connected": right,
        "inferred_left_confidence": left.astype(float),
        "inferred_right_confidence": right.astype(float),
        "inferred_left_connection_score": left.astype(float),
        "inferred_right_connection_score": right.astype(float),
        "inferred_left_loss_score": np.zeros(5),
        "inferred_right_loss_score": np.zeros(5),
        "inferred_left_state": np.full(5, "CONNECTED", dtype="U32"),
        "inferred_right_state": np.full(5, "CONNECTED", dtype="U32"),
        "inferred_label": truth_label,
        "intervention_active": intervention_active,
        "intervention_event": intervention_event,
        "phase_clock_held": np.zeros(5, dtype=bool),
        "control_dt": np.asarray(0.02, dtype=np.float32),
        "seed": np.asarray(1),
        "condition": np.asarray("normal"),
        "source_sha256": np.asarray("source"),
        "config_sha256": np.asarray(config_sha),
    }
    np.savez_compressed(trials / f"{stem}.npz", **arrays)
    result = {
        "artifact_type": "bimanual_relation_online_trial",
        "task_id": "test-task",
        "seed": 1,
        "condition": "normal",
        "experiment_fingerprint": _experiment_fingerprint(protocol, "normal", 1),
        "source_sha256": "source",
        "config_sha256": config_sha,
        "dataset_sha256": "dataset",
        "steps": 5,
        "condition_realization": realization,
        "relation_metrics": metrics,
        "truth_relation_sequence": protocol["expected_inferred_sequences"]["normal"],
        "inferred_relation_sequence": protocol["expected_inferred_sequences"]["normal"],
        "task_success": True,
    }
    (trials / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
    summary = {
        "source_sha256": "source",
        "config_sha256": config_sha,
        "conditions": ["normal"],
        "seeds": [1],
        "num_expected_trials": 1,
        "num_valid_trials": 1,
        "all_conditions_physically_realized": True,
        "trials": [{**result, "worker_returncode": 0}],
    }
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = audit_bimanual_relation_results(results, protocol_path)
    assert report["trial_count"] == 1
    assert report["all_trials_passed"]
    assert report["left"]["micro_f1"] == 1.0
    assert report["right"]["micro_f1"] == 1.0
