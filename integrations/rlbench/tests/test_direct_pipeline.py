from __future__ import annotations

import json
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from essay2608.policy.dynamac import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACConfig,
    SkillModel,
    StreamModel,
)

from integrations.rlbench.rlbench_dynamac import unimanual_evaluate
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    DEFAULT_MODELS_DIR as DEFAULT_EVALUATION_MODELS_DIR,
)
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    DEFAULT_RESULTS_DIR as DEFAULT_EVALUATION_RESULTS_DIR,
)
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    _apply_scenario,
    _learned_policy_steps,
    _noop_action,
)
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    _finalize_episode_intervention_status as _finalize_bimanual_intervention,
)
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    _run_episode as _run_bimanual_episode,
)
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    build_parser as build_evaluation_parser,
)
from integrations.rlbench.rlbench_dynamac.direct_policy import (
    DEFAULT_CONFIG,
    PolicyServer,
    _validate_published_model,
    _WireObservation,
    demonstration_paths,
)
from integrations.rlbench.rlbench_dynamac.direct_policy import (
    DEFAULT_MODELS_DIR as DEFAULT_TRAINING_MODELS_DIR,
)
from integrations.rlbench.rlbench_dynamac.direct_report import (
    DEFAULT_RESULTS_DIR as DEFAULT_DIRECT_REPORT_RESULTS_DIR,
)
from integrations.rlbench.rlbench_dynamac.records import atomic_json, reserve_output
from integrations.rlbench.rlbench_dynamac.runtime import (
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    PrimaryActionRetryBudget,
    bimanual_observations_from_rlbench,
    execute_joint_target_control,
)
from integrations.rlbench.rlbench_dynamac.table_iii_coordination import (
    _run_episode as _run_coordination_episode,
)
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    DEFAULT_MODELS_DIR as DEFAULT_UNIMANUAL_MODELS_DIR,
)
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    DEFAULT_RESULTS_DIR as DEFAULT_UNIMANUAL_RESULTS_DIR,
)
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    _finalize_episode_intervention_status as _finalize_unimanual_intervention,
)
from integrations.rlbench.rlbench_dynamac.unimanual_evaluate import (
    _run_episode as _run_unimanual_episode,
)


class _InvalidActionForTest(Exception):
    pass


class _JointArm:
    def __init__(self, positions) -> None:
        self._positions = iter(positions)

    def get_joint_positions(self):
        return next(self._positions)


class _StepScene:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _LoadedPolicyForValidation:
    def __init__(self, config, fingerprint, identity) -> None:
        self.config = config
        self._fingerprint = fingerprint
        self._identity = dict(identity)

    def fingerprint(self):
        return self._fingerprint

    def summary(self):
        return dict(self._identity)


class _RuntimeArm:
    def __init__(self, duration=2) -> None:
        self.clock = 0
        self.duration = duration

    @property
    def complete(self):
        return self.clock >= self.duration

    def _capture_runtime_state(self):
        return {"clock": self.clock}

    def _restore_runtime_state(self, state):
        self.clock = state["clock"]


def _core_action(clock):
    return SimpleNamespace(
        pose=np.asarray([float(clock), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        diagnostics={"clock": clock},
    )


class _TransactionalUnimanualPolicy(_RuntimeArm):
    def reset(self, _observation, mode_strategy):
        assert mode_strategy == "map"
        self.clock = 0

    def act(self, _observation):
        self.clock += 1
        return _core_action(self.clock)


class _TransactionalBimanualPolicy:
    def __init__(self) -> None:
        self.left = _RuntimeArm()
        self.right = _RuntimeArm()
        self._last_left_action = None
        self._last_right_action = None

    @property
    def complete(self):
        return self.left.complete and self.right.complete

    def reset(self, _left, _right, mode_strategy):
        assert mode_strategy == "map"
        self.left.clock = 0
        self.right.clock = 0
        self._last_left_action = None
        self._last_right_action = None

    def act(self, _left, _right):
        self.left.clock += 1
        self.right.clock += 1
        left = _core_action(self.left.clock)
        right = _core_action(self.right.clock)
        self._last_left_action = left
        self._last_right_action = right
        return SimpleNamespace(left=left, right=right)


def _transaction_server(task, policy, *, bimanual):
    server = PolicyServer.__new__(PolicyServer)
    server.task = task
    server.bimanual = bimanual
    server.policy = policy
    server._pending_transaction = None
    server._next_transaction_id = 1
    return server


def _wire_pose():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _unimanual_wire_observation():
    pose = _wire_pose()
    return {"gripper_pose": pose, "task_low_dim_state": pose * 2}


def _bimanual_wire_observation():
    pose = _wire_pose()
    return {
        "left": {"gripper_pose": pose},
        "right": {"gripper_pose": pose},
        "task_low_dim_state": pose * 2,
    }


def _install_invalid_action_module(monkeypatch):
    rlbench = ModuleType("rlbench")
    backend = ModuleType("rlbench.backend")
    exceptions = ModuleType("rlbench.backend.exceptions")
    exceptions.InvalidActionError = _InvalidActionForTest
    rlbench.backend = backend
    backend.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "rlbench", rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.backend", backend)
    monkeypatch.setitem(sys.modules, "rlbench.backend.exceptions", exceptions)


class _TransactionalWorker:
    def __init__(self, action) -> None:
        self.action = action
        self.policy_steps = 3
        self.next_transaction_id = 1
        self.requests = []

    def request(self, command, observation=None, **fields):
        self.requests.append((command, fields.get("transaction_id")))
        if command == "reset":
            return {"ok": True, "complete": False}
        if command == "act":
            transaction_id = self.next_transaction_id
            self.next_transaction_id += 1
            return {
                "ok": True,
                "complete": transaction_id >= 2,
                "action": self.action,
                "transaction_id": transaction_id,
            }
        if command == "abort":
            return {"ok": True, "complete": False}
        if command == "commit":
            return {"ok": True, "complete": fields["transaction_id"] >= 2}
        raise AssertionError(command)


class _InvalidThenSuccessfulEnvironment:
    def __init__(self, observation) -> None:
        self.observation = observation
        self.step_calls = 0

    def variation_count(self):
        return 1

    def set_variation(self, _variation):
        return None

    def reset(self):
        return [], self.observation

    def step(self, _action):
        self.step_calls += 1
        if self.step_calls == 1:
            raise _InvalidActionForTest("primary action failed")
        return self.observation, 0.0, False


class _AlwaysInvalidPrimaryEnvironment:
    """Alternate an invalid primary command with a successful no-op."""

    def __init__(self, observation, noop_outcome=None) -> None:
        self.observation = observation
        self.noop_outcome = noop_outcome
        self.step_calls = 0

    def variation_count(self):
        return 1

    def set_variation(self, _variation):
        return None

    def reset(self):
        return [], self.observation

    def step(self, _action):
        self.step_calls += 1
        if self.step_calls % 2:
            raise _InvalidActionForTest("primary action failed")
        attempt = self.step_calls // 2
        if attempt == DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS:
            if self.noop_outcome == "success":
                return self.observation, 1.0, True
            if self.noop_outcome == "terminate":
                return self.observation, 0.0, True
        return self.observation, 0.0, False


def _bimanual_validation_case(monkeypatch):
    base = DynaMACConfig(eq6_empty_selection="keep_argmax")
    derived = BimanualDynaMAC(config=base)
    identity = {
        "model_schema_version": 13,
        "selection_semantics_id": (
            "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
        ),
        "tapas_reference_commit": "52e35214b9baa7b190b87196c36b9e98f4006149",
    }
    policies = {
        "left.npz": _LoadedPolicyForValidation(
            derived.left.config, "left-fingerprint", identity
        ),
        "right.npz": _LoadedPolicyForValidation(
            derived.right.config, "right-fingerprint", identity
        ),
    }
    monkeypatch.setattr(
        DynaMAC,
        "load",
        staticmethod(lambda path: policies[path.name]),
    )
    manifest = {
        "task": "bimanual_handover_item",
        "bimanual": True,
        "config": asdict(base),
        "left": {
            "config": asdict(policies["left.npz"].config),
            "fingerprint": policies["left.npz"].fingerprint(),
        },
        "right": {
            "config": asdict(policies["right.npz"].config),
            "fingerprint": policies["right.npz"].fingerprint(),
        },
    }
    return manifest, policies


def test_direct_training_discovers_episodes_in_numeric_order(tmp_path) -> None:
    root = tmp_path / "task" / "all_variations" / "episodes"
    for number in (10, 2, 0, 1, 11):
        episode = root / f"episode{number}"
        episode.mkdir(parents=True)
        (episode / "low_dim_obs.pkl").touch()

    paths = demonstration_paths(tmp_path, "task", count=4)

    assert [path.parent.name for path in paths] == [
        "episode0",
        "episode1",
        "episode2",
        "episode10",
    ]


def test_v2_is_the_non_overwriting_default_artifact_release() -> None:
    integration_root = DEFAULT_CONFIG.parents[1]

    assert DEFAULT_CONFIG == integration_root / "configs" / "dynamac_rlbench_local.json"
    assert DEFAULT_TRAINING_MODELS_DIR == integration_root / "models" / "v2"
    assert DEFAULT_EVALUATION_MODELS_DIR == integration_root / "models" / "v2"
    assert DEFAULT_EVALUATION_RESULTS_DIR == integration_root / "results" / "v2"
    assert DEFAULT_UNIMANUAL_MODELS_DIR == integration_root / "models" / "v2"
    assert DEFAULT_UNIMANUAL_RESULTS_DIR == integration_root / "results" / "v2"
    assert DEFAULT_DIRECT_REPORT_RESULTS_DIR == (
        integration_root / "results" / "v2" / "table_ii"
    )


def test_result_path_is_exclusively_reserved_and_atomically_published(tmp_path) -> None:
    output = tmp_path / "result.json"

    with reserve_output(output):
        with np.testing.assert_raises(FileExistsError):
            with reserve_output(output):
                pass
        atomic_json(output, {"complete": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"complete": True}
    assert not output.with_name(output.name + ".lock").exists()
    with np.testing.assert_raises(FileExistsError):
        with reserve_output(output):
            pass


def test_wire_observation_converts_rlbench_xyzw_to_core_wxyz() -> None:
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    observation = _WireObservation(
        {
            "left": {"gripper_pose": pose},
            "right": {"gripper_pose": pose},
            "task_low_dim_state": pose * 2,
        }
    )

    left, right = bimanual_observations_from_rlbench(
        observation, "bimanual_lift_tray"
    )

    assert np.allclose(left.ee_pose, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(right.ee_pose, left.ee_pose)
    assert list(left.frames) == ["item", "tray"]


def test_invalid_action_noop_uses_right_first_current_state() -> None:
    right_pose = np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    left_pose = np.asarray([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=right_pose, gripper_open=0.9),
        left=SimpleNamespace(gripper_pose=left_pose, gripper_open=0.1),
    )

    action = _noop_action(observation)

    assert action.shape == (18,)
    assert np.allclose(action[:7], right_pose)
    assert np.allclose(action[9:16], left_pose)
    assert np.allclose(action[[7, 8, 16, 17]], [1.0, 0.0, 0.0, 0.0])


def test_evaluation_defaults_to_200_episodes() -> None:
    args = build_evaluation_parser().parse_args(
        ["--task", "bimanual_handover_item"]
    )

    assert args.episodes == 200
    assert args.headless is True
    assert args.policy_timeout == 120.0
    assert args.scenario == "static"
    assert args.scenario_trigger_fraction == 1.0 / 3.0
    assert args.scenario_steps == 10
    assert args.scenario_reference_steps is None
    assert args.max_primary_action_attempts == 3


def test_primary_action_retry_budget_resets_only_after_success() -> None:
    budget = PrimaryActionRetryBudget(3)

    assert budget.record_failure() is False
    assert budget.record_failure() is False
    budget.record_success()
    assert budget.attempts == 0
    assert budget.record_failure() is False
    assert budget.record_failure() is False
    assert budget.record_failure() is True


@pytest.mark.parametrize("invalid", (True, 3.0, 0, -1))
def test_primary_action_retry_budget_rejects_non_positive_or_non_integer(invalid) -> None:
    error = TypeError if invalid is True or isinstance(invalid, float) else ValueError
    with pytest.raises(error):
        PrimaryActionRetryBudget(invalid)


def test_dynamic_scenario_refreshes_observation_before_policy_action() -> None:
    original = object()
    refreshed = object()

    class Controller:
        def apply(self, task_environment, *, step, horizon):
            assert task_environment is environment
            assert step == 4
            assert horizon == 12
            return {"applied": True, "trigger_step": 4}

    class Environment:
        def get_observation(self):
            return refreshed

    environment = Environment()
    observation, event = _apply_scenario(
        Controller(), environment, original, step=4, horizon=12
    )

    assert observation is refreshed
    assert event["policy_observation_refreshed"] is True


def test_unimanual_dynamic_event_records_refreshed_policy_observation(
    monkeypatch,
) -> None:
    _install_invalid_action_module(monkeypatch)
    original = object()
    refreshed = object()

    class Controller:
        def __init__(self, *_args, **_kwargs):
            pass

        def apply(self, task_environment, *, step, horizon):
            assert task_environment is environment
            assert step == 0
            assert horizon == 12
            return {
                "kind": "smooth_task_motion",
                "step": 0,
                "trigger_step": 0,
                "applied": True,
                "protocol_effective": True,
                "smooth_call": 1,
                "complete": False,
                "endpoint_applied": False,
            }

    class Environment:
        def set_variation(self, variation):
            assert variation == 0

        def reset(self):
            return [], original

        def get_observation(self):
            return refreshed

    class Worker:
        policy_steps = 12

        def __init__(self):
            self.requests = []

        def request(self, command, observation=None, **_fields):
            self.requests.append((command, observation))
            if command == "reset":
                return {"ok": True}
            if command == "act":
                return {"ok": True, "action": None, "complete": True}
            raise AssertionError(command)

    environment = Environment()
    worker = Worker()
    args = SimpleNamespace(
        seed=0,
        variation=0,
        scenario="smooth",
        trigger_fraction=0.0,
        smooth_steps=10,
        intervention_attempts=20,
        horizon=1,
    )
    monkeypatch.setattr(unimanual_evaluate, "ScenarioController", Controller)

    result = _run_unimanual_episode(environment, worker, args, episode=0)

    assert result["interventions"] == [
        {
            "kind": "smooth_task_motion",
            "step": 0,
            "trigger_step": 0,
            "applied": True,
            "protocol_effective": True,
            "smooth_call": 1,
            "complete": False,
            "endpoint_applied": False,
            "policy_observation_refreshed": True,
        }
    ]
    assert result["intervention_eligible"] is True
    assert result["intervention_complete"] is False
    assert worker.requests == [("reset", original), ("act", refreshed)]


@pytest.mark.parametrize(
    ("finalize", "event_key"),
    (
        (_finalize_unimanual_intervention, "interventions"),
        (_finalize_bimanual_intervention, "scenario_events"),
    ),
)
def test_dynamic_failure_before_trigger_is_retained_as_ineligible(
    finalize,
    event_key,
) -> None:
    row = {
        "episode": 41,
        "success": False,
        "steps": 16,
        "reason": "primary_action_retry_exhausted",
        event_key: [],
    }

    result = finalize(
        row,
        scenario="teleport",
        trigger_step=62,
        trigger_reached=False,
        smooth_steps=10,
    )

    assert result["trigger_step"] == 62
    assert result["intervention_eligible"] is False
    assert result["intervention_reached"] is False
    assert result["pre_intervention_terminal"] is True
    assert result["intervention_effective"] is None
    assert result["intervention_complete"] is None


@pytest.mark.parametrize(
    ("finalize", "event_key"),
    (
        (_finalize_unimanual_intervention, "interventions"),
        (_finalize_bimanual_intervention, "scenario_events"),
    ),
)
def test_dynamic_success_before_trigger_fails_closed(finalize, event_key) -> None:
    row = {
        "episode": 0,
        "success": True,
        "steps": 10,
        "reason": "success",
        event_key: [],
    }

    with pytest.raises(RuntimeError, match="succeeded before"):
        finalize(
            row,
            scenario="teleport",
            trigger_step=62,
            trigger_reached=False,
            smooth_steps=10,
        )


@pytest.mark.parametrize(
    ("finalize", "event_key"),
    (
        (_finalize_unimanual_intervention, "interventions"),
        (_finalize_bimanual_intervention, "scenario_events"),
    ),
)
def test_dynamic_trigger_without_effective_event_fails_closed(
    finalize,
    event_key,
) -> None:
    row = {
        "episode": 0,
        "success": False,
        "steps": 63,
        "reason": "terminate",
        event_key: [],
    }

    with pytest.raises(RuntimeError, match="without an effective"):
        finalize(
            row,
            scenario="teleport",
            trigger_step=62,
            trigger_reached=True,
            smooth_steps=10,
        )


@pytest.mark.parametrize(
    ("finalize", "event_key"),
    (
        (_finalize_unimanual_intervention, "interventions"),
        (_finalize_bimanual_intervention, "scenario_events"),
    ),
)
def test_smooth_terminal_after_trigger_accepts_only_a_strict_prefix(
    finalize,
    event_key,
) -> None:
    events = [
        {
            "kind": "smooth_task_motion",
            "step": 62 + index - 1,
            "trigger_step": 62,
            "applied": True,
            "protocol_effective": True,
            "smooth_call": index,
            "complete": False,
            "endpoint_applied": False,
        }
        for index in range(1, 4)
    ]
    row = {
        "episode": 0,
        "success": False,
        "steps": 65,
        "reason": "terminate",
        event_key: events,
    }

    result = finalize(
        row,
        scenario="smooth",
        trigger_step=62,
        trigger_reached=True,
        smooth_steps=10,
    )
    assert result["intervention_effective"] is True
    assert result["intervention_complete"] is False

    forged = {**row, "steps": 66}
    with pytest.raises(RuntimeError, match="next motion tick"):
        finalize(
            forged,
            scenario="smooth",
            trigger_step=62,
            trigger_reached=True,
            smooth_steps=10,
        )


def test_learned_policy_steps_uses_longer_arm_clock(tmp_path) -> None:
    model = tmp_path / "task"
    model.mkdir()
    (model / "training.json").write_text(
        '{"left":{"durations":[10,20]},"right":{"durations":[12,24]}}',
        encoding="utf-8",
    )

    assert _learned_policy_steps(tmp_path, "task") == 36


def test_policy_ping_binds_results_to_the_loaded_checkpoint(tmp_path) -> None:
    config = DynaMACConfig(
        covariance_estimation_method="diagonal_empirical_ridge",
        default_mode_strategy="map",
    )
    policy = DynaMAC(config)
    policy.frame_names = ("wine_bottle",)
    policy.skill_sequence = (7,)
    duration = 3
    mean = np.repeat(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[None, None],
        duration,
        axis=1,
    )
    covariance = np.repeat((np.eye(6) * 1.0e-3)[None, None], duration, axis=1)
    policy.skills = [
        SkillModel(
            label=7,
            duration=duration,
            selected_frames=("virtual_skill_7",),
            mode_priors=np.ones(1),
            streams={
                "virtual_skill_7": StreamModel(
                    "virtual_skill_7",
                    mean,
                    covariance,
                )
            },
            gripper=np.zeros((1, duration, 1)),
            mode_demonstration_indices=((0,),),
        )
    ]
    expected_fingerprint = policy.fingerprint()
    expected_summary = policy.summary()

    model_dir = tmp_path / "models" / "stack_wine"
    model_dir.mkdir(parents=True)
    policy.save(model_dir / "model.npz")
    (model_dir / "training.json").write_text(
        json.dumps(
            {
                "manifest_schema": "dynamac-direct-training-v2",
                "task": "stack_wine",
                "bimanual": False,
                "config": asdict(config),
                "fingerprint": expected_fingerprint,
            }
        ),
        encoding="utf-8",
    )

    response = PolicyServer("stack_wine", tmp_path / "models").handle(
        {"command": "ping"}
    )

    assert response["ok"] is True
    assert response["ready"] is True
    assert response["task"] == "stack_wine"
    assert response["bimanual"] is False
    assert response["policy_steps"] == duration
    identity = response["model_identity"]
    assert set(identity) == {
        "model_schema_version",
        "selection_semantics_id",
        "tapas_reference_commit",
        "config",
        "fingerprint",
        "training_manifest_schema",
        "manifest_authenticated",
        "training_config",
    }
    assert identity["model_schema_version"] == expected_summary["model_schema_version"]
    assert identity["selection_semantics_id"] == expected_summary[
        "selection_semantics_id"
    ]
    assert identity["config"] == asdict(config)
    assert identity["config"]["covariance_estimation_method"] == (
        "diagonal_empirical_ridge"
    )
    assert identity["fingerprint"] == expected_fingerprint
    assert identity["manifest_authenticated"] is True
    assert identity["training_config"] == asdict(config)


def test_unimanual_policy_action_is_only_complete_after_commit() -> None:
    policy = _TransactionalUnimanualPolicy(duration=1)
    server = _transaction_server("stack_wine", policy, bimanual=False)
    observation = _unimanual_wire_observation()
    server.handle({"command": "reset", "observation": observation})

    tentative = server.handle({"command": "act", "observation": observation})

    assert tentative["complete_after_commit"] is True
    assert policy.complete is True
    with pytest.raises(RuntimeError, match="still awaits commit or abort"):
        server.handle({"command": "act", "observation": observation})

    aborted = server.handle(
        {"command": "abort", "transaction_id": tentative["transaction_id"]}
    )
    assert aborted["complete"] is False
    assert policy.clock == 0

    retried = server.handle({"command": "act", "observation": observation})
    committed = server.handle(
        {"command": "commit", "transaction_id": retried["transaction_id"]}
    )
    assert committed["complete"] is True
    assert policy.clock == 1
    assert server.handle({"command": "act", "observation": observation}) == {
        "ok": True,
        "complete": True,
        "action": None,
    }


def test_policy_transaction_rejects_boolean_and_mismatched_ids() -> None:
    policy = _TransactionalUnimanualPolicy()
    server = _transaction_server("stack_wine", policy, bimanual=False)
    observation = _unimanual_wire_observation()
    server.handle({"command": "reset", "observation": observation})
    tentative = server.handle({"command": "act", "observation": observation})

    with pytest.raises(RuntimeError, match="must be an integer"):
        server.handle({"command": "commit", "transaction_id": True})
    with pytest.raises(RuntimeError, match="does not match"):
        server.handle(
            {"command": "commit", "transaction_id": tentative["transaction_id"] + 1}
        )

    server.handle({"command": "abort", "transaction_id": tentative["transaction_id"]})
    assert policy.clock == 0


def test_bimanual_abort_restores_both_clocks_and_last_actions() -> None:
    policy = _TransactionalBimanualPolicy()
    server = _transaction_server("bimanual_lift_tray", policy, bimanual=True)
    observation = _bimanual_wire_observation()
    server.handle({"command": "reset", "observation": observation})

    first = server.handle({"command": "act", "observation": observation})
    server.handle({"command": "commit", "transaction_id": first["transaction_id"]})
    committed_left_pose = policy._last_left_action.pose.copy()
    committed_right_pose = policy._last_right_action.pose.copy()

    second = server.handle({"command": "act", "observation": observation})
    assert policy.left.clock == policy.right.clock == 2
    server.handle({"command": "abort", "transaction_id": second["transaction_id"]})

    assert policy.left.clock == policy.right.clock == 1
    assert np.array_equal(policy._last_left_action.pose, committed_left_pose)
    assert np.array_equal(policy._last_right_action.pose, committed_right_pose)


def test_reset_aborts_a_pending_policy_action_before_starting_new_episode() -> None:
    policy = _TransactionalUnimanualPolicy()
    server = _transaction_server("stack_wine", policy, bimanual=False)
    observation = _unimanual_wire_observation()
    server.handle({"command": "reset", "observation": observation})
    server.handle({"command": "act", "observation": observation})
    assert policy.clock == 1

    response = server.handle({"command": "reset", "observation": observation})

    assert response["complete"] is False
    assert policy.clock == 0
    assert server._pending_transaction is None


def test_bimanual_evaluator_aborts_invalid_target_and_commits_retry(monkeypatch) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _InvalidThenSuccessfulEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(18))

    result = _run_bimanual_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=3,
    )

    assert result["reason"] == "policy_complete"
    assert result["invalid_actions"] == 1
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("abort", 1),
        ("act", None),
        ("commit", 2),
    ]


def test_unimanual_evaluator_aborts_invalid_target_and_commits_retry(monkeypatch) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(gripper_pose=pose, gripper_open=1.0)
    environment = _InvalidThenSuccessfulEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(9))
    args = SimpleNamespace(
        seed=0,
        variation=0,
        scenario="static",
        trigger_fraction=1.0 / 3.0,
        smooth_steps=10,
        intervention_attempts=20,
        horizon=3,
    )

    result = _run_unimanual_episode(environment, worker, args, episode=0)

    assert result["reason"] == "policy_complete"
    assert result["invalid_actions"] == 1
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("abort", 1),
        ("act", None),
        ("commit", 2),
    ]


def test_coordination_evaluator_uses_the_shared_policy_transaction_protocol(
    monkeypatch,
) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _InvalidThenSuccessfulEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(18))

    result = _run_coordination_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=3,
        arm="left",
        trigger=0,
    )

    assert result["reason"] == "policy_complete"
    assert result["invalid_actions"] == 1
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("abort", 1),
        ("act", None),
        ("commit", 2),
    ]


def test_bimanual_retry_budget_fails_closed_without_committing(monkeypatch) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _AlwaysInvalidPrimaryEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(18))

    result = _run_bimanual_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=10,
        max_primary_action_attempts=3,
    )

    assert result["reason"] == "primary_action_retry_exhausted"
    assert result["success"] is False
    assert result["invalid_actions"] == 3
    assert result["primary_action_attempts"] == 3
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("abort", 1),
        ("act", None),
        ("abort", 2),
        ("act", None),
        ("abort", 3),
    ]


def test_unimanual_retry_budget_fails_closed_without_committing(monkeypatch) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(gripper_pose=pose, gripper_open=1.0)
    environment = _AlwaysInvalidPrimaryEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(9))
    args = SimpleNamespace(
        seed=0,
        variation=0,
        scenario="static",
        trigger_fraction=1.0 / 3.0,
        smooth_steps=10,
        intervention_attempts=20,
        max_primary_action_attempts=3,
        horizon=10,
    )

    result = _run_unimanual_episode(environment, worker, args, episode=0)

    assert result["reason"] == "primary_action_retry_exhausted"
    assert result["invalid_actions"] == 3
    assert result["primary_action_attempts"] == 3
    assert all(command != "commit" for command, _ in worker.requests)


def test_dynamic_evaluators_retain_retry_exhaustion_before_trigger(
    monkeypatch,
) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())

    bimanual_observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    bimanual = _run_bimanual_episode(
        _AlwaysInvalidPrimaryEnvironment(bimanual_observation),
        _TransactionalWorker(np.zeros(18)),
        episode=0,
        seed=0,
        horizon=10,
        scenario="teleport",
        scenario_reference_steps=187,
        max_primary_action_attempts=3,
    )

    unimanual_observation = SimpleNamespace(gripper_pose=pose, gripper_open=1.0)
    unimanual_args = SimpleNamespace(
        seed=0,
        variation=0,
        scenario="teleport",
        trigger_fraction=1.0,
        smooth_steps=10,
        intervention_attempts=20,
        max_primary_action_attempts=2,
        horizon=10,
    )
    unimanual = _run_unimanual_episode(
        _AlwaysInvalidPrimaryEnvironment(unimanual_observation),
        _TransactionalWorker(np.zeros(9)),
        unimanual_args,
        episode=0,
    )

    assert bimanual["reason"] == "primary_action_retry_exhausted"
    assert bimanual["pre_intervention_terminal"] is True
    assert bimanual["intervention_effective"] is None
    assert unimanual["reason"] == "primary_action_retry_exhausted"
    assert unimanual["pre_intervention_terminal"] is True
    assert unimanual["intervention_effective"] is None


def test_coordination_retry_budget_fails_closed_without_committing(
    monkeypatch,
) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _AlwaysInvalidPrimaryEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(18))

    result = _run_coordination_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=10,
        arm="left",
        trigger=0,
        max_primary_action_attempts=3,
    )

    assert result["reason"] == "primary_action_retry_exhausted"
    assert result["invalid_actions"] == 3
    assert result["primary_action_attempts"] == 3
    assert all(command != "commit" for command, _ in worker.requests)


@pytest.mark.parametrize(
    ("noop_outcome", "expected_reason", "expected_success"),
    (("success", "success", True), ("terminate", "terminate", False)),
)
def test_noop_terminal_outcome_precedes_retry_exhaustion(
    monkeypatch,
    noop_outcome,
    expected_reason,
    expected_success,
) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _AlwaysInvalidPrimaryEnvironment(
        observation,
        noop_outcome=noop_outcome,
    )
    worker = _TransactionalWorker(np.zeros(18))

    result = _run_bimanual_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=10,
        max_primary_action_attempts=3,
    )

    assert result["reason"] == expected_reason
    assert result["success"] is expected_success
    assert result["invalid_actions"] == 3
    assert all(command != "commit" for command, _ in worker.requests)


def test_bimanual_manifest_base_config_is_bound_to_both_checkpoints(
    tmp_path, monkeypatch
) -> None:
    manifest, _policies = _bimanual_validation_case(monkeypatch)
    _validate_published_model("bimanual_handover_item", tmp_path, manifest)

    manifest["config"] = {**manifest["config"], "tau_omega": 0.25}

    with pytest.raises(RuntimeError, match="not derived from the base config"):
        _validate_published_model("bimanual_handover_item", tmp_path, manifest)


@pytest.mark.parametrize(
    "field",
    ("model_schema_version", "selection_semantics_id", "tapas_reference_commit"),
)
def test_bimanual_manifest_rejects_mixed_checkpoint_identity(
    tmp_path, monkeypatch, field
) -> None:
    manifest, policies = _bimanual_validation_case(monkeypatch)
    policies["right.npz"]._identity[field] = "tampered"

    with pytest.raises(RuntimeError, match="mismatched model identity"):
        _validate_published_model("bimanual_handover_item", tmp_path, manifest)


def test_joint_target_control_ignores_task_terminal_until_combined_action_finishes() -> None:
    scene = _StepScene()
    # No task object is present: consulting scene.task.success() here would fail.
    arm = _JointArm(([0.0], [0.5], [1.0]))

    status = execute_joint_target_control(
        scene,
        ((arm, np.asarray([1.0])),),
        max_steps=3,
        invalid_action_error=_InvalidActionForTest,
    )

    assert status == "reached"
    assert scene.steps == 3


def test_joint_target_control_cap_exhaustion_is_an_invalid_action() -> None:
    scene = _StepScene()
    right = _JointArm(([0.0], [0.1], [0.2], [0.3]))
    left = _JointArm(([0.0], [0.1], [0.2], [0.3]))

    with np.testing.assert_raises_regex(
        _InvalidActionForTest,
        "bimanual command timed out",
    ):
        execute_joint_target_control(
            scene,
            (
                (right, np.asarray([10.0])),
                (left, np.asarray([10.0])),
            ),
            max_steps=4,
            invalid_action_error=_InvalidActionForTest,
            error_message="bimanual command timed out",
        )

    assert scene.steps == 4
