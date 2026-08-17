from __future__ import annotations

import json

import pytest

from integrations.rlbench.rlbench_dynamac.direct_report import TASKS, load_rows, markdown
from integrations.rlbench.rlbench_dynamac.paper_comparison import (
    EXPECTED_LOCAL_CONFIG,
    EXPECTED_SELECTION_SEMANTICS_ID,
    EXPECTED_TAPAS_COMMIT,
    expected_evaluation_protocol_id,
)


def _write_results(tmp_path, *, seed: int = 0) -> None:
    for task, _label, _paper_rate in TASKS:
        payload = {
            "task": task,
            "scenario": "static",
            "seed": seed,
            "episodes": 2,
            "horizon": 1000,
            "successes": 1,
            "success_rate": 0.5,
            "evaluation_protocol_id": expected_evaluation_protocol_id(task),
            "model_identity": {
                "manifest_authenticated": True,
                "training_config": EXPECTED_LOCAL_CONFIG,
                "model_schema_version": 13,
                "selection_semantics_id": EXPECTED_SELECTION_SEMANTICS_ID,
                "tapas_reference_commit": EXPECTED_TAPAS_COMMIT,
                "left_fingerprint": f"{task}-left",
                "right_fingerprint": f"{task}-right",
            },
            "results": [
                {
                    "episode": 0,
                    "success": True,
                    "reason": "success",
                    "invalid_actions": 0,
                },
                {
                    "episode": 1,
                    "success": False,
                    "reason": "policy_complete",
                    "invalid_actions": 3,
                },
            ],
        }
        path = tmp_path / f"{task}_static_seed{seed}_n2_h1000.json"
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_direct_report_validates_and_summarizes_runs(tmp_path) -> None:
    _write_results(tmp_path)

    rows = load_rows(tmp_path, seed=0, episodes=2, horizon=1000)
    report = markdown(rows, seed=0)

    assert len(rows) == 4
    assert rows[0]["invalid_actions"] == 3
    assert rows[0]["termination_reasons"] == {
        "policy_complete": 1,
        "success": 1,
    }
    assert "1/2" in report
    assert "policy_complete=1, success=1" in report


def test_direct_report_rejects_identity_mismatch(tmp_path) -> None:
    _write_results(tmp_path)
    task = TASKS[0][0]
    path = tmp_path / f"{task}_static_seed0_n2_h1000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenario"] = "teleport"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        load_rows(tmp_path, seed=0, episodes=2, horizon=1000)
