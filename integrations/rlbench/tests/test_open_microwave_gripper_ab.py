import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import open_microwave_gripper_ab as ab


def _action(gripper=1.0):
    return np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, gripper, 0.0])


class _Worker:
    policy_steps = 200
    model_identity = {"fingerprint": "fixture"}

    def __init__(self):
        self.transaction = 0
        self.calls = []

    def request(self, command, observation=None, **fields):
        self.calls.append((command, fields))
        if command == "act":
            self.transaction += 1
            return {
                "ok": True,
                "transaction_id": self.transaction,
                "action": _action().tolist(),
            }
        if command == "commit":
            return {"ok": True, "complete": False}
        return {"ok": True}


def test_variant_b_changes_only_gripper_scalar_at_tick_113():
    original = _action()

    a = ab.apply_gripper_timing_variant(
        original,
        variant="A",
        committed_tick=ab.EARLY_CLOSE_TICK,
    )
    b = ab.apply_gripper_timing_variant(
        original,
        variant="B",
        committed_tick=ab.EARLY_CLOSE_TICK,
    )
    b_next = ab.apply_gripper_timing_variant(
        original,
        variant="B",
        committed_tick=ab.EARLY_CLOSE_TICK + 1,
    )

    np.testing.assert_array_equal(a, original)
    np.testing.assert_array_equal(b_next, original)
    assert np.flatnonzero(b != original).tolist() == [ab.GRIPPER_ACTION_INDEX]
    assert b[ab.GRIPPER_ACTION_INDEX] == 0.0
    np.testing.assert_array_equal(b[:7], original[:7])
    assert b[8] == original[8]


@pytest.mark.parametrize("variant", ["A", "B"])
def test_variant_wrapper_rejects_non_9d_actions(variant):
    with pytest.raises(ValueError, match="9D"):
        ab.apply_gripper_timing_variant(
            np.zeros(8),
            variant=variant,
            committed_tick=113,
        )


def test_abort_retries_same_committed_tick_and_only_commit_advances_clock():
    base = _Worker()
    worker = ab.CommittedTickGripperWorker(base, "B", early_close_tick=0)
    worker.request("reset", SimpleNamespace())

    first = worker.request("act", SimpleNamespace())
    assert first["action"][7] == 0.0
    worker.request("abort", transaction_id=first["transaction_id"])
    assert worker.committed_tick == 0

    retry = worker.request("act", SimpleNamespace())
    assert retry["action"][7] == 0.0
    worker.request("commit", transaction_id=retry["transaction_id"])
    assert worker.committed_tick == 1

    next_tick = worker.request("act", SimpleNamespace())
    assert next_tick["action"][7] == 1.0
    metadata = worker.metadata()
    assert metadata["command_close_tick"] == 0
    assert metadata["aborted_attempt_count"] == 1
    assert metadata["target_tick_attempt_count"] == 2
    assert metadata["only_allowed_scalar_changed"] is True


def test_door_and_actual_gripper_transition_metrics_are_measured():
    state = {"door": 0.1, "gripper": 1.0}
    metrics = ab.DoorGripperMetrics(
        lambda: state["door"],
        lambda: [state["gripper"], state["gripper"]],
        observation=SimpleNamespace(gripper_open=1.0),
    )
    state.update(door=1.5, gripper=0.0)
    metrics.sample(
        observation=SimpleNamespace(gripper_open=0.0),
        committed_tick=113,
        phase="primary_action",
    )
    result = metrics.result(committed_tick=114)

    assert result["actual_gripper_close_transition_tick"] == 113
    assert result["door_joint_initial_position_rad"] == pytest.approx(0.1)
    assert result["door_joint_peak_position_rad"] == pytest.approx(1.5)
    assert result["door_joint_peak_displacement_rad"] == pytest.approx(1.4)
    assert result["door_joint_final_displacement_rad"] == pytest.approx(1.4)


def test_pair_order_alternates_ab_ba():
    assert ab.paired_execution_order(0) == ("A", "B")
    assert ab.paired_execution_order(1) == ("B", "A")
    assert ab.paired_execution_order(38) == ("A", "B")
    assert ab.paired_execution_order(39) == ("B", "A")


def test_paired_statistics_reports_rescue_regression_and_exact_mcnemar():
    pairs = [
        {
            "protocol_pair_valid": True,
            "single_variable_exercised": True,
            "outcome": outcome,
        }
        for outcome in (
            "rescue",
            "rescue",
            "rescue",
            "regression",
            "both_success",
            "both_failure",
        )
    ]

    statistics = ab.paired_statistics(pairs)

    assert statistics["paired_rescues_a_fail_b_success"] == 3
    assert statistics["paired_regressions_a_success_b_fail"] == 1
    assert statistics["discordant_pairs"] == 4
    assert statistics["mcnemar_exact_two_sided_p"] == pytest.approx(0.625)
    assert statistics["paired_success_rate_delta_b_minus_a"] == pytest.approx(
        2.0 / 6.0
    )


def _run_record(variant, success, reason):
    target = {
        "original_action_fingerprint": "a" * 64,
        "emitted_action_fingerprint": "b" * 64,
        "differing_indices": [7] if variant == "B" else [],
        "original_gripper": 1.0,
        "emitted_gripper": 0.0 if variant == "B" else 1.0,
    }
    return {
        "variant": variant,
        "success": success,
        "reason": reason,
        "invalid": False,
        "invalid_actions": 0,
        "initial_observation_fingerprint": "c" * 64,
        "command": {
            "target_tick_committed": target,
            "only_allowed_scalar_changed": True,
            "protocol_mutation_exercised": variant == "B",
            "command_close_tick": 113 if variant == "B" else 114,
        },
        "physical": {
            "actual_gripper_close_transition_tick": 113 if variant == "B" else 114,
            "door_joint_initial_position_rad": 0.1,
            "door_joint_peak_position_rad": 1.6,
            "door_joint_peak_displacement_rad": 1.5,
            "door_joint_final_position_rad": 1.5,
            "door_joint_final_displacement_rad": 1.4,
        },
    }


def test_pair_record_and_outputs_are_explicitly_diagnostic(tmp_path):
    pair = ab.build_pair_record(
        episode=0,
        seed=ab.DEV_BASE_SEED,
        order=("A", "B"),
        runs={
            "A": _run_record("A", False, "policy_complete"),
            "B": _run_record("B", True, "success"),
        },
    )
    summary = {
        "schema": ab.DIAGNOSTIC_SCHEMA,
        "status": ab.STATUS,
        "comparability": ab.COMPARABILITY,
        "formal_table_eligible": False,
        "pairs": [pair],
    }
    assert pair["protocol_pair_valid"] is True

    json_path, csv_path = ab.write_outputs(summary, tmp_path)

    loaded = json.loads(json_path.read_text())
    assert loaded["status"] == "PROVISIONAL"
    assert loaded["comparability"] == "NON_COMPARABLE"
    assert loaded["formal_table_eligible"] is False
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "PROVISIONAL"
    assert rows[0]["comparability"] == "NON_COMPARABLE"
    assert rows[0]["outcome"] == "rescue"
    assert rows[0]["execution_order"] == "AB"
    assert rows[0]["b_door_peak_position_rad"] == "1.6"
    assert rows[0]["b_door_final_position_rad"] == "1.5"


def test_cli_has_no_formal_eval_set_or_table_option():
    parser = ab.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--eval-set-id" not in options
    assert "--release" not in options
    assert ab.DEV_BASE_SEED == 4_104_100_000
    assert ab.DEV_EPISODES == 40
    assert ab.DEFAULT_OUTPUT_DIR.parts[-4:] == (
        "results",
        "v4",
        "diagnostics",
        "open_microwave_gripper_timing_ab",
    )
