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


def test_source_selection_can_require_original_successes(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(_source_payload(success=True)), encoding="utf-8")

    _payload, rows, _digest, _resolved = failure_videos._load_source(
        path,
        "bimanual_handover_item",
        (3,),
        expected_success=True,
    )

    assert rows[3]["success"] is True
    path.write_text(json.dumps(_source_payload(success=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a successful row"):
        failure_videos._load_source(
            path,
            "bimanual_handover_item",
            (3,),
            expected_success=True,
        )


def test_source_selection_rejects_dynamic_row_before_intervention(tmp_path):
    payload = _source_payload(success=False)
    payload["scenario"] = "teleport"
    payload["results"][0]["scenario_events"] = []
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="did not exercise"):
        failure_videos._load_source(
            path,
            "bimanual_handover_item",
            (3,),
        )


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
        "expected_outcome": "failure",
        "same_reason": True,
        "same_steps": True,
        "same_invalid_actions": True,
    }

    successful_replay = dict(replay, success=True, reason="success")
    with pytest.raises(RuntimeError, match="successful during replay"):
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


def test_success_replay_is_publishable_only_when_it_remains_successful():
    identity = {"manifest_authenticated": True, "fingerprint": "abc"}
    original = {
        "episode": 7,
        "success": True,
        "reason": "success",
        "steps": 100,
        "invalid_actions": 0,
    }

    confirmation = failure_videos._validate_replay(
        original,
        dict(original),
        identity,
        dict(identity),
        expected_success=True,
    )

    assert confirmation == {
        "replay_confirmed_success": True,
        "expected_outcome": "success",
        "same_reason": True,
        "same_steps": True,
        "same_invalid_actions": True,
    }
    with pytest.raises(RuntimeError, match="failed during replay"):
        failure_videos._validate_replay(
            original,
            dict(original, success=False, reason="policy_complete"),
            identity,
            dict(identity),
            expected_success=True,
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


def test_replay_attempt_protocol_is_bounded_and_auditable():
    protocol = failure_videos._replay_attempt_protocol(20)

    assert protocol == {
        "protocol_id": failure_videos.REPLAY_ATTEMPT_PROTOCOL_ID,
        "max_replay_attempts_per_episode": 20,
        "attempt_ordinals_are_one_based": True,
        "fresh_task_generation_per_attempt": True,
        "seed_variation_plan_and_controller_unchanged_between_attempts": True,
        "retryable_completed_result_mismatches": [
            "outcome",
            "source_condition",
            "invalid_actions",
        ],
        "candidate_traversal": "episode_major_in_caller_order",
        "exceptions_fail_closed": True,
        "first_matching_attempt_only": True,
        "one_video_per_source_episode": True,
        "source_evaluation_immutable": True,
        "replays_are_video_confirmation_not_evaluation": True,
    }
    with pytest.raises(ValueError, match="must be positive"):
        failure_videos._replay_attempt_protocol(0)
    with pytest.raises(TypeError, match="must be an integer"):
        failure_videos._replay_attempt_protocol(True)


def test_replay_attempt_assessment_retries_only_three_completed_mismatches():
    original = {
        "episode": 7,
        "success": False,
        "invalid_actions": 0,
    }
    static_source = {"task": "bimanual_handover_item", "scenario": "static"}

    confirmed = failure_videos._assess_replay_attempt(
        original,
        dict(original),
        static_source,
        expected_success=False,
    )
    assert confirmed == {
        "replay_confirmed_outcome": True,
        "source_condition_reproduced": True,
        "same_invalid_actions": True,
        "confirmed_for_publication": True,
        "disposition": "published_first_match",
    }

    outcome = failure_videos._assess_replay_attempt(
        original,
        dict(original, success=True),
        static_source,
        expected_success=False,
    )
    assert outcome["confirmed_for_publication"] is False
    assert outcome["disposition"] == "discarded_outcome_mismatch"

    condition = failure_videos._assess_replay_attempt(
        original,
        dict(original, scenario_events=[]),
        {"task": "bimanual_handover_item", "scenario": "teleport"},
        expected_success=False,
    )
    assert condition["confirmed_for_publication"] is False
    assert condition["disposition"] == "discarded_source_condition_mismatch"

    invalid_actions = failure_videos._assess_replay_attempt(
        original,
        dict(original, invalid_actions=1),
        static_source,
        expected_success=False,
    )
    assert invalid_actions["confirmed_for_publication"] is False
    assert invalid_actions["disposition"] == "discarded_invalid_actions_mismatch"

    with pytest.raises(RuntimeError, match="episode identity"):
        failure_videos._assess_replay_attempt(
            original,
            dict(original, episode=8),
            static_source,
            expected_success=False,
        )


def test_bounded_replay_attempts_stop_at_first_match_and_preserve_ordinals():
    seen = []

    def runner(ordinal):
        seen.append(ordinal)
        return {
            "replay_attempt_ordinal": ordinal,
            "confirmed_for_publication": ordinal == 3,
        }

    accepted, records = failure_videos._run_bounded_replay_attempts(5, runner)

    assert seen == [1, 2, 3]
    assert [record["replay_attempt_ordinal"] for record in records] == [1, 2, 3]
    assert accepted is records[-1]


def test_bounded_replay_attempts_exhaust_and_exceptions_fail_closed():
    accepted, records = failure_videos._run_bounded_replay_attempts(
        2,
        lambda ordinal: {
            "replay_attempt_ordinal": ordinal,
            "confirmed_for_publication": False,
        },
    )
    assert accepted is None
    assert [record["replay_attempt_ordinal"] for record in records] == [1, 2]

    seen = []

    def raises_on_second(ordinal):
        seen.append(ordinal)
        if ordinal == 2:
            raise RuntimeError("infrastructure failure")
        return {
            "replay_attempt_ordinal": ordinal,
            "confirmed_for_publication": False,
        }

    with pytest.raises(RuntimeError, match="infrastructure failure"):
        failure_videos._run_bounded_replay_attempts(5, raises_on_second)
    assert seen == [1, 2]


def test_replay_attempt_cli_defaults_to_one_and_filename_records_ordinal():
    args = failure_videos.build_parser().parse_args(
        ["--task", "wipe_desk", "--episode", "3"]
    )
    assert args.max_replay_attempts_per_episode == 1
    assert (
        failure_videos._replay_output_stem(3, 2_608_000_003, 12)
        == "episode_003_seed_2608000003_attempt_012"
    )


def test_record_replays_same_fixed_episode_fresh_until_first_match(
    monkeypatch,
    tmp_path,
):
    identity = {"manifest_authenticated": True, "fingerprint": "model"}
    original = {
        "episode": 7,
        "success": True,
        "reason": "success",
        "steps": 5,
        "invalid_actions": 0,
    }
    source = {
        "task": "bimanual_handover_item",
        "scenario": "static",
        "seed": 100,
        "model_identity": identity,
        "fixed_eval_set": {"evaluation_set_id": "fixed-v3"},
    }

    class Plan:
        variation = 2

        @staticmethod
        def fingerprint():
            return "sealed-plan"

    class Environment:
        def __init__(self):
            self.launched = False
            self.shutdown_called = False

        def launch(self):
            self.launched = True

        def shutdown(self):
            self.shutdown_called = True

    class Worker:
        model_identity = identity

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Recorder:
        def __init__(self, path, **_kwargs):
            self.path = path
            self.used_cameras = ("front", "overhead")
            self.frame_shape = (360, 1280, 3)
            self.frames = 6

        def capture(self, _observation):
            return None

        def close(self):
            self.path.write_bytes(b"audited-video")

        def abort(self):
            return None

    environment = Environment()
    worker = Worker()
    fresh_calls = []
    replay_calls = []

    monkeypatch.setattr(
        failure_videos,
        "_load_source",
        lambda *_args, **_kwargs: (
            source,
            {7: original},
            "source-sha",
            tmp_path / "source.json",
        ),
    )
    monkeypatch.setattr(
        failure_videos,
        "_require_current_evaluator_protocol",
        lambda *_args: "evaluator-v3",
    )
    monkeypatch.setattr(
        failure_videos,
        "_load_sealed_replay_batch",
        lambda *_args: {"family": "bimanual"},
    )
    monkeypatch.setattr(
        failure_videos,
        "_source_protocol_budgets",
        lambda *_args: dict(RUNNER_BUDGETS),
    )
    monkeypatch.setattr(
        failure_videos,
        "_runtime_components",
        lambda *_args: (environment, worker, object),
    )
    monkeypatch.setattr(
        failure_videos,
        "_authenticated_replay_trigger",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        failure_videos,
        "_sealed_episode_plan",
        lambda *_args: (Plan(), 2, 1234),
    )

    def fresh_generation(_environment, _task_class, **kwargs):
        fresh_calls.append((kwargs["episode_seed"], kwargs["variation"]))
        return object(), ["description"], object(), {
            "episode_seed": kwargs["episode_seed"],
            "variation": kwargs["variation"],
        }

    def run_episode(*_args, **_kwargs):
        replay_calls.append(len(replay_calls) + 1)
        return {
            "episode": 7,
            "success": len(replay_calls) == 2,
            "reason": "success" if len(replay_calls) == 2 else "policy_complete",
            "steps": 5,
            "invalid_actions": 0,
        }

    monkeypatch.setattr(
        failure_videos,
        "initialize_fresh_task_generation",
        fresh_generation,
    )
    monkeypatch.setattr(failure_videos, "_run_selected_episode", run_episode)
    monkeypatch.setattr(
        failure_videos,
        "_validate_replay_plan",
        lambda *_args: None,
    )
    monkeypatch.setattr(failure_videos, "ObservationRecorder", Recorder)

    args = SimpleNamespace(
        task="bimanual_handover_item",
        episode=[7],
        expected_outcome="success",
        minimum_confirmed=1,
        max_replay_attempts_per_episode=3,
        source_result=tmp_path / "source.json",
        output_root=tmp_path / "videos",
        output_key="handover_success",
        cameras=("front", "overhead"),
        fps=12,
        ffmpeg=tmp_path / "ffmpeg",
        resolution=(640, 360),
    )

    target = failure_videos.record(args)

    assert fresh_calls == [(1234, 2), (1234, 2)]
    assert replay_calls == [1, 2]
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert [row["replay_attempt_ordinal"] for row in manifest["attempts"]] == [1, 2]
    assert [row["disposition"] for row in manifest["attempts"]] == [
        "discarded_outcome_mismatch",
        "published_first_match",
    ]
    assert manifest["episodes"][0]["replay_attempt_ordinal"] == 2
    assert manifest["episodes"][0]["replay_attempt_disposition"] == (
        "published_first_match"
    )
    assert manifest["candidate_episode_order"] == [7]
    assert manifest["replay_attempt_protocol"][
        "max_replay_attempts_per_episode"
    ] == 3
    assert [path.name for path in target.glob("*.mp4")] == [
        "episode_007_seed_107_attempt_002.mp4"
    ]
    sidecar = json.loads(
        (target / manifest["episodes"][0]["metadata"]).read_text(encoding="utf-8")
    )
    assert sidecar["replay_attempt_ordinal"] == 2
    assert sidecar["replay_attempt_disposition"] == "published_first_match"
    assert manifest["episodes"][0]["metadata_sha256"] == failure_videos._sha256(
        target / manifest["episodes"][0]["metadata"]
    )
    assert worker.closed is True
    assert environment.shutdown_called is True
