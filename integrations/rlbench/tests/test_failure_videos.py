from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import failure_videos


def _source_payload(*, success=False):
    return {
        "task": "bimanual_handover_item",
        "scenario": "static",
        "seed": 0,
        "horizon": 1000,
        "model_identity": {
            "manifest_authenticated": True,
            "left_fingerprint": "left",
            "right_fingerprint": "right",
        },
        "results": [
            {
                "episode": 3,
                "success": success,
                "steps": 262,
                "reason": "success" if success else "policy_complete",
            }
        ],
    }


def test_source_selection_accepts_only_original_failures(tmp_path):
    path = tmp_path / "source.json"
    raw = json.dumps(_source_payload(), sort_keys=True).encode()
    path.write_bytes(raw)

    payload, rows, digest, resolved = failure_videos._load_source(
        path, "bimanual_handover_item", (3,)
    )

    assert payload["seed"] == 0
    assert rows[3]["success"] is False
    assert digest == hashlib.sha256(raw).hexdigest()
    assert resolved == path.resolve()

    path.write_text(json.dumps(_source_payload(success=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="successful"):
        failure_videos._load_source(path, "bimanual_handover_item", (3,))


def test_replay_fails_closed_when_source_evaluator_semantics_differ():
    source = _source_payload()
    source["evaluation_protocol_id"] = "legacy-protocol"
    with pytest.raises(RuntimeError, match="does not match"):
        failure_videos._require_current_evaluator_protocol(
            source,
            "bimanual_handover_item",
        )

    source["evaluation_protocol_id"] = (
        failure_videos.direct_evaluate.EVALUATION_PROTOCOL_ID
    )
    assert failure_videos._require_current_evaluator_protocol(
        source,
        "bimanual_handover_item",
    ) == failure_videos.direct_evaluate.EVALUATION_PROTOCOL_ID


def test_episode_indices_are_exact_ordered_and_unique():
    assert failure_videos._normalize_episode_indices([3, 1, 8]) == (3, 1, 8)
    with pytest.raises(ValueError, match="duplicate"):
        failure_videos._normalize_episode_indices([3, 3])
    with pytest.raises(ValueError, match="non-negative"):
        failure_videos._normalize_episode_indices([-1])


def test_composed_rgb_uses_front_and_overhead_with_front_fallback():
    front = np.full((2, 3, 3), 10, dtype=np.uint8)
    overhead = np.full((2, 3, 3), 20, dtype=np.uint8)
    both = SimpleNamespace(perception_data={"front_rgb": front, "overhead_rgb": overhead})
    frame, used = failure_videos._compose_frame(both, ("front", "overhead"))
    assert frame.shape == (2, 6, 3)
    assert used == ("front", "overhead")
    assert np.all(frame[:, :3] == 10)
    assert np.all(frame[:, 3:] == 20)

    front_only = SimpleNamespace(perception_data={"front_rgb": front})
    frame, used = failure_videos._compose_frame(front_only, ("front", "overhead"))
    assert frame.shape == front.shape
    assert used == ("front",)


def test_task_proxy_records_reset_get_and_step_observations():
    observations = [object(), object(), object()]

    class Task:
        def reset(self):
            return ["description"], observations[0]

        def get_observation(self):
            return observations[1]

        def step(self, action):
            assert action == "action"
            return observations[2], 0.0, False

        def variation_count(self):
            return 4

    class Recorder:
        def __init__(self):
            self.values = []

        def capture(self, observation):
            self.values.append(observation)

    recorder = Recorder()
    proxy = failure_videos.RecordingTaskEnvironment(Task(), recorder)

    assert proxy.reset()[1] is observations[0]
    assert proxy.get_observation() is observations[1]
    assert proxy.step("action")[0] is observations[2]
    assert proxy.variation_count() == 4
    assert recorder.values == observations


def test_replay_is_publishable_only_when_model_matches_and_it_still_fails():
    identity = {"manifest_authenticated": True, "fingerprint": "abc"}
    original = {
        "episode": 7,
        "success": False,
        "reason": "policy_complete",
        "steps": 100,
    }
    replay = dict(original)
    confirmation = failure_videos._validate_replay(original, replay, identity, dict(identity))
    assert confirmation == {
        "replay_confirmed_failure": True,
        "same_reason": True,
        "same_steps": True,
        "same_invalid_actions": True,
    }

    successful_replay = dict(replay, success=True, reason="success")
    with pytest.raises(RuntimeError, match="succeeded during replay"):
        failure_videos._validate_replay(original, successful_replay, identity, dict(identity))
    with pytest.raises(RuntimeError, match="model identity"):
        failure_videos._validate_replay(original, replay, identity, {"fingerprint": "different"})

    with pytest.raises(RuntimeError, match="invalid-action count"):
        failure_videos._validate_replay(
            original,
            dict(replay, invalid_actions=1),
            identity,
            dict(identity),
        )


def test_bimanual_episode_offset_is_forwarded_unchanged(monkeypatch):
    seen = {}

    def run_episode(
        task_environment,
        worker,
        episode,
        seed,
        horizon,
        **kwargs,
    ):
        seen.update(
            episode=episode,
            seed=seed,
            horizon=horizon,
            scenario=kwargs["scenario"],
            reference=kwargs["scenario_reference_steps"],
        )
        return {"episode": episode, "success": False}

    monkeypatch.setattr(failure_videos.direct_evaluate, "_run_episode", run_episode)
    worker = SimpleNamespace(policy_steps=262)
    result = failure_videos._run_selected_episode(
        object(),
        worker,
        "bimanual_handover_item",
        {
            "task": "bimanual_handover_item",
            "scenario": "static",
            "seed": 0,
            "horizon": 1000,
        },
        37,
    )

    assert result == {"episode": 37, "success": False}
    assert seen == {
        "episode": 37,
        "seed": 0,
        "horizon": 1000,
        "scenario": "static",
        "reference": 262,
    }


def test_dynamic_bimanual_protocol_is_forwarded_from_source(monkeypatch):
    seen = {}

    def run_episode(task_environment, worker, episode, seed, horizon, **kwargs):
        del task_environment, worker
        seen.update(episode=episode, seed=seed, horizon=horizon, **kwargs)
        return {
            "episode": episode,
            "success": False,
            "scenario_events": [{"applied": True, "protocol_effective": True}],
        }

    monkeypatch.setattr(failure_videos.direct_evaluate, "_run_episode", run_episode)
    source = {
        "task": "bimanual_lift_tray",
        "scenario": "teleport",
        "seed": 4,
        "horizon": 900,
        "scenario_protocol": {
            "trigger_fraction": 0.25,
            "trigger_reference_steps": 116,
            "smooth_interpolation_calls": None,
            "max_sampling_attempts": 17,
        },
    }
    replay = failure_videos._run_selected_episode(
        object(),
        SimpleNamespace(policy_steps=999),
        "bimanual_lift_tray",
        source,
        12,
    )

    assert seen == {
        "episode": 12,
        "seed": 4,
        "horizon": 900,
        "scenario": "teleport",
        "scenario_trigger_fraction": 0.25,
        "scenario_reference_steps": 116,
        "scenario_steps": 10,
        "scenario_max_attempts": 17,
    }
    assert failure_videos._protocol_effective(source, replay) is True


def test_coordination_protocol_uses_alias_arm_and_trigger(monkeypatch):
    seen = {}

    def run_episode(task_environment, worker, **kwargs):
        del task_environment, worker
        seen.update(kwargs)
        return {
            "episode": kwargs["episode"],
            "success": False,
            "perturbed_steps": 10,
        }

    monkeypatch.setattr(failure_videos.table_iii_coordination, "_run_episode", run_episode)
    source = {
        "schema": "dynamac-table-iii-coordination-local-v1",
        "task": "bimanual_handover_item_dynamic",
        "policy_task_alias": "bimanual_handover_item",
        "scenario": "coordination_hand_left",
        "seed": 0,
        "horizon": 1000,
        "coordination_protocol": {
            "protocol_valid": True,
            "perturbed_arm": "left",
            "trigger_policy_step": 87,
        },
    }
    replay = failure_videos._run_selected_episode(
        object(), object(), "bimanual_handover_item", source, 98
    )

    assert seen == {
        "episode": 98,
        "variation": 3,
        "seed": 0,
        "horizon": 1000,
        "arm": "left",
        "trigger": 87,
    }
    assert failure_videos._protocol_effective(source, replay) is True


def test_dynamic_failure_requires_the_source_condition_to_recur():
    source = {
        "task": "place_cups",
        "scenario": "smooth",
    }
    identity = {"manifest_authenticated": True}
    original = {"episode": 2, "success": False}
    replay = {
        "episode": 2,
        "success": False,
        "interventions": [{"applied": True, "protocol_effective": False}],
    }
    with pytest.raises(RuntimeError, match="did not reproduce"):
        failure_videos._validate_replay(original, replay, identity, dict(identity), source)


def test_smooth_failure_requires_the_complete_intervention_sequence():
    source = {
        "task": "place_cups",
        "scenario": "smooth",
        "protocol": {"smooth_motion_calls": 2},
    }
    complete = {
        "interventions": [
            {"applied": True, "protocol_effective": True, "complete": False},
            {
                "applied": True,
                "protocol_effective": True,
                "complete": True,
                "endpoint_applied": True,
            },
        ]
    }
    assert failure_videos._protocol_effective(source, complete) is True
    assert (
        failure_videos._protocol_effective(source, {"interventions": complete["interventions"][:1]})
        is False
    )


def test_output_keys_are_cell_specific_and_path_safe():
    assert (
        failure_videos._default_output_key({"scenario": "teleport"}, "place_cups")
        == "place_cups_teleport"
    )
    assert failure_videos._validate_output_key("table_iii_hand_left") == ("table_iii_hand_left")
    with pytest.raises(ValueError, match="output-key"):
        failure_videos._validate_output_key("../escape")
