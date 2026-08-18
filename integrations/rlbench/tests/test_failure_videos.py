from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import failure_videos


RUNNER_BUDGETS = {
    "smooth_steps": 10,
    "motion_attempts": 100,
    "final_settling_steps": 10,
    "primary_action_attempts": 3,
}


def _formal_runner_inputs(*, variation, trigger_step):
    return {
        "motion_plan": SimpleNamespace(variation=variation),
        "trigger_step": trigger_step,
        "descriptions": ["formal description"],
        "observation": object(),
        "fresh_task_generation": {"formal": True},
        "budgets": dict(RUNNER_BUDGETS),
    }


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


def test_sealed_replay_batch_binds_source_identity_and_episode(monkeypatch):
    class Plan:
        variation = 2
        validation = {"source_seed": 91}

        @staticmethod
        def fingerprint():
            return "plan-fingerprint"

    plan = Plan()
    manifest = {
        "manifest_sha256": "manifest-sha",
        "payload": {
            "spec": {"sha256": "spec-sha"},
            "environment_plan_batches": {
                "bimanual_handover_item": {"sha256": "batch-sha"}
            },
        },
    }
    batch = {
        "payload": {
            "base_seed": 80,
            "episodes": 1,
            "variation_schedule": [2],
            "batch_fingerprint": "batch-fingerprint",
        },
        "plans": [plan],
    }
    monkeypatch.setattr(
        failure_videos,
        "fixed_environment_plans",
        lambda eval_set_id, task: (
            (manifest, batch)
            if eval_set_id == "fixed-v3" and task == "bimanual_handover_item"
            else None
        ),
    )
    source = {
        "task": "bimanual_handover_item",
        "scenario": "static",
        "seed": 80,
        "episodes": 1,
        "variation_schedule": [2],
        "fixed_eval_set": {
            "evaluation_set_id": "fixed-v3",
            "manifest_sha256": "manifest-sha",
            "spec_sha256": "spec-sha",
            "selected_batch_sha256": "batch-sha",
            "selected_batch_fingerprint": "batch-fingerprint",
        },
    }
    original = {
        "motion_plan_fingerprint": "plan-fingerprint",
        "staged_source_binding": {
            "motion_plan_fingerprint": "plan-fingerprint"
        },
        "fresh_task_generation": {"episode_seed": 91, "variation": 2},
    }

    replay_batch = failure_videos._load_sealed_replay_batch(
        source,
        "bimanual_handover_item",
    )
    selected, variation, source_seed = failure_videos._sealed_episode_plan(
        replay_batch,
        original,
        0,
    )

    assert selected is plan
    assert variation == 2
    assert source_seed == 91
    failure_videos._validate_replay_plan(
        replay_batch,
        {
            "motion_plan_fingerprint": "plan-fingerprint",
            "fresh_task_generation": {"episode_seed": 91, "variation": 2},
        },
        plan,
    )
    original["motion_plan_fingerprint"] = "forged"
    with pytest.raises(RuntimeError, match="sealed replay plan"):
        failure_videos._sealed_episode_plan(replay_batch, original, 0)


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


def test_replay_trigger_is_reauthenticated_from_checkpoint(monkeypatch):
    source = {
        "task": "bimanual_handover_item",
        "scenario": "teleport",
        "scenario_protocol": {
            "trigger_reference_steps": 50,
            "trigger_policy_step": 12,
            "trigger_authentication": {"trigger_step": 12},
            "intervention_registry_schema": "registry",
            "intervention_registry_fingerprint": "fingerprint",
        },
    }
    monkeypatch.setattr(
        failure_videos.direct_evaluate,
        "_authenticated_v3_dynamic_trigger",
        lambda args, worker: (
            {"schema": "registry", "fingerprint": "fingerprint"},
            {"trigger_step": 13},
        ),
    )

    with pytest.raises(RuntimeError, match="not authenticated"):
        failure_videos._authenticated_replay_trigger(
            source,
            "bimanual_handover_item",
            SimpleNamespace(policy_steps=50),
            RUNNER_BUDGETS,
        )


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
            motion_plan=kwargs["motion_plan"],
            trigger_step=kwargs["scenario_trigger_step"],
            descriptions=kwargs["descriptions"],
            observation=kwargs["observation"],
            fresh_task_generation=kwargs["fresh_task_generation"],
        )
        return {"episode": episode, "success": False}

    monkeypatch.setattr(failure_videos.direct_evaluate, "_run_episode", run_episode)
    worker = SimpleNamespace(policy_steps=262)
    formal = _formal_runner_inputs(variation=2, trigger_step=None)
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
        **formal,
    )

    assert result == {"episode": 37, "success": False}
    assert seen["episode"] == 37
    assert seen["seed"] == 0
    assert seen["horizon"] == 1000
    assert seen["scenario"] == "static"
    assert seen["reference"] == 262
    assert seen["motion_plan"] is formal["motion_plan"]
    assert seen["trigger_step"] is None
    assert seen["descriptions"] is formal["descriptions"]
    assert seen["observation"] is formal["observation"]
    assert seen["fresh_task_generation"] is formal["fresh_task_generation"]


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
    formal = _formal_runner_inputs(variation=2, trigger_step=29)
    replay = failure_videos._run_selected_episode(
        object(),
        SimpleNamespace(policy_steps=999),
        "bimanual_lift_tray",
        source,
        12,
        **formal,
    )

    assert seen["episode"] == 12
    assert seen["seed"] == 4
    assert seen["horizon"] == 900
    assert seen["scenario"] == "teleport"
    assert seen["scenario_trigger_fraction"] == 0.25
    assert seen["scenario_trigger_step"] == 29
    assert seen["scenario_reference_steps"] == 999
    assert seen["scenario_steps"] == 10
    assert seen["scenario_max_attempts"] == 100
    assert seen["motion_plan"] is formal["motion_plan"]
    assert seen["descriptions"] is formal["descriptions"]
    assert seen["observation"] is formal["observation"]
    assert seen["fresh_task_generation"] is formal["fresh_task_generation"]
    assert failure_videos._protocol_effective(source, replay) is True


def test_unimanual_replay_forwards_sealed_plan_and_fresh_inputs(monkeypatch):
    seen = {}

    def run_episode(
        task_environment,
        worker,
        args,
        episode,
        motion_plan=None,
        **kwargs,
    ):
        del task_environment, worker
        seen.update(
            args=args,
            episode=episode,
            motion_plan=motion_plan,
            **kwargs,
        )
        return {
            "episode": episode,
            "success": False,
            "interventions": [{"applied": True, "protocol_effective": True}],
        }

    monkeypatch.setattr(
        failure_videos.unimanual_evaluate,
        "_run_episode",
        run_episode,
    )
    source = {
        "task": "wipe_desk",
        "scenario": "teleport",
        "seed": 5,
        "variation": 0,
        "horizon": 700,
        "protocol": {"trigger_fraction_of_fitted_policy": 0.2},
    }
    formal = _formal_runner_inputs(variation=0, trigger_step=41)

    replay = failure_videos._run_selected_episode(
        object(),
        object(),
        "wipe_desk",
        source,
        6,
        **formal,
    )

    assert seen["episode"] == 6
    assert seen["motion_plan"] is formal["motion_plan"]
    assert seen["args"].seed == 5
    assert seen["args"].variation == 0
    assert seen["args"].horizon == 700
    assert seen["args"].scenario == "teleport"
    assert seen["args"].trigger_step == 41
    assert seen["args"].smooth_steps == 10
    assert seen["args"].intervention_attempts == 100
    assert seen["descriptions"] is formal["descriptions"]
    assert seen["observation"] is formal["observation"]
    assert seen["fresh_task_generation"] is formal["fresh_task_generation"]
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
        "schema": "dynamac-table-iii-coordination-local-v3",
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
    formal = _formal_runner_inputs(variation=3, trigger_step=87)
    formal["staged_source_binding"] = {"matched": True}
    replay = failure_videos._run_selected_episode(
        object(),
        object(),
        "bimanual_handover_item",
        source,
        98,
        **formal,
    )

    assert seen["episode"] == 98
    assert seen["variation"] == 3
    assert seen["seed"] == 0
    assert seen["horizon"] == 1000
    assert seen["arm"] == "left"
    assert seen["trigger"] == 87
    assert seen["descriptions"] is formal["descriptions"]
    assert seen["observation"] is formal["observation"]
    assert seen["fresh_task_generation"] is formal["fresh_task_generation"]
    assert seen["staged_source_binding"] is formal["staged_source_binding"]
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
