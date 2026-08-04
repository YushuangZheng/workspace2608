from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from essay2608.eval.recovery_audit import audit_recovery_run


def _write_minimal_run(root: Path) -> tuple[Path, Path]:
    protocol = {
        "methods": ["method"],
        "conditions": ["drop_lift_early"],
        "held_out_test_seeds": [6500],
        "drop_distances_m": [0.05],
        "drop_directions": ["back"],
        "drop_force_open_steps": [0],
        "maximum_steps": 1000,
        "success_xy_threshold_m": 0.01,
    }
    config = {
        "source_git_commit": "frozen",
        "source_sha256": "source",
        "dataset_sha256": "dataset",
        "schema_version": 7,
        "max_steps": 1000,
        "success_xy_threshold_m": 0.01,
        "perturbation_config": {
            "distance_m": 0.05,
            "direction": "back",
            "force_open_steps": 0,
        },
    }
    trial = {
        "method": "method",
        "condition": "drop_lift_early",
        "seed": 6500,
        "dataset_sha256": "dataset",
        "experiment_fingerprint": "unique",
        "experiment_config": config,
        "metrics": {
            "final_object_position_m": [0.1, 0.2, 0.3],
            "final_target_position_m": [0.4, 0.5, 0.6],
        },
    }
    summary = {"evaluation_schema_version": 7, "trials": [trial]}
    protocol_path = root / "protocol.json"
    summary_path = root / "summary.json"
    trials = root / "trials"
    trials.mkdir()
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    stem = "method__drop_lift_early__seed_6500"
    (trials / f"{stem}.json").write_text(json.dumps(trial), encoding="utf-8")
    n = 2
    scalar = np.zeros(n)
    position = np.zeros((n, 3))
    np.savez_compressed(
        trials / f"{stem}.npz",
        ee_position=position,
        object_position=position,
        target_position=position,
        action=np.zeros((n, 8)),
        phase=scalar,
        connected=scalar.astype(bool),
        perturbation_event=scalar.astype(bool),
        raw_action_position=position,
        policy_action_position=position,
        relation_state=np.asarray(["A", "B"]),
        relation_confidence=scalar,
        active_frames=np.asarray([["world"], ["world"]]),
        recovery_state=np.asarray(["NORMAL", "NORMAL"]),
        recovery_trigger=np.asarray(["NONE", "NONE"]),
        regrasp_attempts=scalar,
        terminal_object_position=np.asarray([0.1, 0.2, 0.3]),
        terminal_target_position=np.asarray([0.4, 0.5, 0.6]),
        terminal_ee_position=np.zeros(3),
    )
    return summary_path, protocol_path


def test_audit_accepts_complete_immutable_run(tmp_path: Path) -> None:
    summary, protocol = _write_minimal_run(tmp_path)
    result = audit_recovery_run(summary, protocol, expected_source_commit="frozen")
    assert result.trial_count == 1
    assert result.fingerprint_count == 1
    assert len(result.summary_sha256) == 64


def test_audit_rejects_terminal_trace_mismatch(tmp_path: Path) -> None:
    summary, protocol = _write_minimal_run(tmp_path)
    npz_path = next((tmp_path / "trials").glob("*.npz"))
    with np.load(npz_path) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["terminal_object_position"] = np.ones(3)
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ValueError, match="终端物体位置"):
        audit_recovery_run(summary, protocol)
