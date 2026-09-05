from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from evaluations.iclr2027.interfaces.feature_schema import (
    FEATURE_SCHEMA,
    FeatureRecord,
    validate_feature_record,
)
from evaluations.iclr2027.delivery.build_b_handoff import (
    build_handoff_index,
    interface_sources,
)
from evaluations.iclr2027.interfaces.failure_train import (
    causal_violation_labels,
    select_failure_train_rows,
)
from evaluations.iclr2027.interfaces.runtime_monitor import (
    EpisodeContext,
    RuntimeMonitor,
)
from evaluations.iclr2027.audit.faults import (
    CompositeFaultEnvironment,
    build_fault_environment,
)
from evaluations.iclr2027.manifests.build import (
    build_all_manifests,
    validate_all_manifests,
)
from evaluations.iclr2027.recovery.skill_retry import SkillRetry
from evaluations.iclr2027.runners.shadow import shadow_passthrough_action
from evaluations.iclr2027.runners.launch import DynamicQueue


def feature() -> dict:
    return FeatureRecord(
        episode_id="development_nominal/close_jar/0000",
        cycle=2,
        observation_timestamp=2,
        action_timestamp=2,
        arms={
            "single": {
                "ee_pose_xyzw": [0, 0, 0, 0, 0, 0, 1],
                "gripper_open": 1.0,
            }
        },
        task_state=(0.0, 1.0),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        policy_state={"policy_step": 2},
        action_resolution={"aggregate": "reached"},
    ).to_dict()


def test_feature_schema_rejects_evaluator_label_leak() -> None:
    value = feature()
    value["policy_state"]["fault_family"] = "relation_loss"
    with pytest.raises(ValueError, match="leaked"):
        validate_feature_record(value)


def test_feature_schema_rejects_nested_evaluator_label_leak() -> None:
    value = feature()
    value["policy_state"]["metadata"] = {"audit_label": True}
    with pytest.raises(ValueError, match="leaked"):
        validate_feature_record(value)


def test_feature_time_alignment_is_current_cycle() -> None:
    value = feature()
    value["action_timestamp"] = 1
    with pytest.raises(ValueError, match="timestamp"):
        validate_feature_record(value)


class RecordingMonitor(RuntimeMonitor):
    def __init__(self) -> None:
        self.items = []

    def reset(self, episode_context: EpisodeContext) -> None:
        self.items = []

    def observe(self, observation, action, policy_state) -> None:
        action["action"][0] = 99.0
        self.items.append((observation, action, policy_state))

    def score(self):
        return {"value": float(len(self.items))}

    def alarm(self) -> bool:
        return False


def test_shadow_monitor_has_no_action_authority() -> None:
    value = feature()
    expected = list(value["action"])
    monitor = RecordingMonitor()
    action, diagnostic = shadow_passthrough_action(monitor, value)
    assert action == expected
    assert value["action"] == expected
    assert diagnostic == {
        "cycle": 2,
        "scores": {"value": 1.0},
        "alarm": False,
        "threshold": None,
        "persistence_count": 0,
        "metadata": {},
    }
    observation, observed_action, _policy = monitor.items[0]
    assert observation["task_state"] == value["task_state"]
    assert observation["previous_action_resolution"] == value["action_resolution"]
    assert observed_action["action_timestamp"] == value["action_timestamp"]


def test_skill_retry_is_bounded_and_uses_only_confirmed_entry() -> None:
    retry = SkillRetry(recovery_budget=2, maximum_retries=1)
    assert not retry.request(True).requested
    retry.confirm_skill_entry({"skill": 3, "progress": 0})
    first = retry.request(True)
    assert first.requested and first.reference_state == {"skill": 3, "progress": 0}
    retry.consume_cycle()
    retry.consume_cycle()
    assert retry.remaining_budget == 0
    assert retry.request(True).reason == "budget_exhausted"


def test_manifests_have_frozen_counts_and_disjoint_independent_seeds(
    tmp_path: Path,
) -> None:
    index = build_all_manifests(tmp_path)
    counts = validate_all_manifests(tmp_path)
    assert counts["main10_development.jsonl"] == 200
    assert counts["main10_failure_train.jsonl"] == 2000
    assert counts["main10_nominal.jsonl"] == 2000
    assert counts["main10_perturbed.jsonl"] == 2000
    assert counts["main10_normal_calibration_candidates.jsonl"] == 6000
    assert index["sealed_executed"] is False


def test_manifest_build_is_deterministic(tmp_path: Path) -> None:
    first = build_all_manifests(tmp_path)
    second = build_all_manifests(tmp_path)
    assert first == second


def test_development_manifest_is_exactly_ten_plus_ten_per_task() -> None:
    root = Path("evaluations/iclr2027/manifests/main10_development.jsonl")
    rows = [json.loads(line) for line in root.read_text().splitlines()]
    counts = {}
    for row in rows:
        key = (row["task"], row["condition"])
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 20
    assert set(counts.values()) == {10}
    assert all(row["pair_id"] == row["episode_id"] for row in rows)
    assert all("policy_id" not in row for row in rows)


def test_composed_fault_requires_both_physical_components() -> None:
    class Component:
        def __init__(self, family, triggered):
            self.family = family
            self.triggered = triggered
            self.clock = 4

        def protocol_metadata(self):
            return {
                "family": self.family,
                "triggered": self.triggered,
                "events": [{"kind": self.family, "policy_step": 2}],
                "policy_steps_observed": self.clock,
                "physical_effect_observed": self.triggered,
            }

        def record_committed_fallback(self):
            self.clock += 1

    first = Component("actuation_delay", True)
    second = Component("relation_loss", False)
    composed = CompositeFaultEnvironment(second, (first, second))
    assert composed.protocol_metadata()["triggered"] is False
    composed.record_committed_fallback()
    assert first.clock == second.clock == 5
    second.triggered = True
    assert composed.protocol_metadata()["triggered"] is True


def test_severity_selects_frozen_physical_magnitude() -> None:
    class Spec:
        bimanual = False

    class Task:
        task_id = "close_jar"
        spec = Spec()

    config = json.loads(
        Path("evaluations/iclr2027/configs/shared/faults.json").read_text()
    )
    low = build_fault_environment(
        object(),
        Task(),
        family="actuation_delay",
        trigger_stage="middle",
        policy_steps=100,
        config=config,
        severity="low",
    )
    high = build_fault_environment(
        object(),
        Task(),
        family="relation_loss",
        trigger_stage="middle",
        policy_steps=100,
        config=config,
        severity="high",
    )
    assert low._wrapped.spec.duration_cycles == 4
    assert high._wrapped.spec.mismatch_translation == (0.08, 0.0, 0.0)


def test_infrastructure_retry_restores_exhausted_task_to_scheduler_ring() -> None:
    queue = DynamicQueue.__new__(DynamicQueue)
    queue.pending_by_task = {"place_cups_3": deque()}
    queue.task_order = deque()
    row = {"task": "place_cups_3", "episode_id": "failure_train/place_cups_3/0115"}

    queue._requeue_infrastructure_retry(row)

    assert list(queue.task_order) == ["place_cups_3"]
    assert list(queue.pending_by_task["place_cups_3"]) == [row]


def test_b_interface_handoff_indexes_canonical_files_without_private_data() -> None:
    payload = build_handoff_index("interface", interface_sources())
    paths = {item["path"] for item in payload["files"]}
    assert payload["transfer_mode"] == "canonical_paths_no_persistent_copy"
    assert payload["total_files"] == 12
    assert not payload["contains_normal_calibration"]
    assert not payload["contains_sealed_test"]
    assert all(path.startswith("evaluations/iclr2027/") for path in paths)
    assert all(
        "normal_calibration" not in path and "sealed" not in path for path in paths
    )


def test_failure_train_targets_use_previous_post_step_audit() -> None:
    def audit(onset=None, end=None):
        return {
            "schema": "essay2608.iclr2027.physical-event-audit.v1",
            "violation_onset_cycle": onset,
            "violation_end_cycle": end,
        }

    audits = [
        audit(),
        audit(onset=1),
        audit(onset=1),
        audit(onset=1, end=3),
        audit(onset=1, end=3),
    ]
    assert causal_violation_labels(audits) == (0, 0, 1, 1, 0)


def test_failure_train_budget_and_lofo_selection_are_frozen() -> None:
    rows = [
        {
            "task": "task_a",
            "episode_id": f"failure_train/task_a/{index:04d}",
            "fault_family": ("family_a", "family_b")[index % 2],
        }
        for index in range(200)
    ]
    budget = select_failure_train_rows(rows, task="task_a", budget=20)
    assert [row["episode_id"] for row in budget] == [
        f"failure_train/task_a/{index:04d}" for index in range(20)
    ]
    lofo = select_failure_train_rows(rows, task="task_a", held_out_family="family_a")
    assert len(lofo) == 100
    assert {row["fault_family"] for row in lofo} == {"family_b"}
