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

from integrations.rlbench.rlbench_dynamac.eval import (
    direct_evaluate,
    unimanual_evaluate,
)
from integrations.rlbench.rlbench_dynamac.core import runtime as rlbench_runtime
from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    DEFAULT_MODELS_DIR as DEFAULT_EVALUATION_MODELS_DIR,
)
from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    DEFAULT_RESULTS_DIR as DEFAULT_EVALUATION_RESULTS_DIR,
)
from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    _apply_scenario,
    _learned_policy_steps,
)
from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    _finalize_episode_intervention_status as _finalize_bimanual_intervention,
)
from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    _run_episode as _run_bimanual_episode,
)
from integrations.rlbench.rlbench_dynamac.eval.direct_evaluate import (
    build_parser as build_evaluation_parser,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    DEFAULT_CONFIG,
    PolicyServer,
    _validate_published_model,
    _WireObservation,
    demonstration_paths,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    DEFAULT_MODELS_DIR as DEFAULT_TRAINING_MODELS_DIR,
)
from integrations.rlbench.rlbench_dynamac.core.records import (
    atomic_json,
    reserve_output,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    FINAL_SETTLING_PROTOCOL_ID,
    FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
    FRESH_TASK_GENERATION_PROTOCOL_ID,
    STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
    PrimaryActionRetryBudget,
    StagedMotionPlan,
    _canonical_json_fingerprint,
    bimanual_observations_from_rlbench,
    execute_joint_target_control,
    run_final_settling,
    staged_motion_plan_batch,
    step_current_joint_hold_noop,
)
from integrations.rlbench.rlbench_dynamac.eval.table_iii_coordination import (
    _run_episode as _run_coordination_episode,
)
from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
    DEFAULT_MODELS_DIR as DEFAULT_UNIMANUAL_MODELS_DIR,
)
from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
    DEFAULT_RESULTS_DIR as DEFAULT_UNIMANUAL_RESULTS_DIR,
)
from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
    _finalize_episode_intervention_status as _finalize_unimanual_intervention,
)
from integrations.rlbench.rlbench_dynamac.eval.unimanual_evaluate import (
    _run_episode as _run_unimanual_episode,
)
from integrations.rlbench.rlbench_dynamac.protocols.v3_protocol import (
    V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA,
    V3_INTERVENTION_SCHEMA,
    build_v3_trigger_anchor_evidence,
    checkpoint_trigger_audit,
    dynamic_trigger_profile,
    load_v3_intervention_protocol,
    resolve_authenticated_v3_trigger,
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

    def preview_next_gripper(self):
        return SimpleNamespace(
            gripper=np.asarray([1.0]),
            crosses_skill_boundary=False,
            repeats_terminal=self.complete,
        )


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

    def preview_next_gripper(self):
        return SimpleNamespace(
            left=self.left.preview_next_gripper(),
            right=self.right.preview_next_gripper(),
        )


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

    # Evaluator transaction tests use a tiny environment without private
    # RLBench scene objects. Production joint-hold behavior is covered by
    # dedicated scene-level tests below; this preserves the fixture's original
    # alternating primary/no-op outcomes.
    def joint_hold_stub(task_environment):
        _observation, reward, terminate = task_environment.step(None)
        get_observation = getattr(task_environment, "get_observation", None)
        observation = (
            get_observation()
            if callable(get_observation)
            else task_environment.observation
        )
        return observation, reward, terminate

    monkeypatch.setattr(
        rlbench_runtime, "step_current_joint_hold_noop", joint_hold_stub
    )
    monkeypatch.setattr(
        direct_evaluate, "step_current_joint_hold_noop", joint_hold_stub
    )
    monkeypatch.setattr(
        unimanual_evaluate, "step_current_joint_hold_noop", joint_hold_stub
    )


class _TransactionalWorker:
    def __init__(self, action, *, complete_after=2) -> None:
        self.action = action
        self.complete_after = int(complete_after)
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
                "complete": transaction_id >= self.complete_after,
                "action": self.action,
                "transaction_id": transaction_id,
            }
        if command == "abort":
            return {"ok": True, "complete": False}
        if command == "commit":
            return {
                "ok": True,
                "complete": fields["transaction_id"] >= self.complete_after,
            }
        raise AssertionError(command)


class _InvalidThenSuccessfulEnvironment:
    def __init__(self, observation) -> None:
        self.observation = observation
        self.step_calls = 0
        self.actions = []

    def variation_count(self):
        return 1

    def set_variation(self, _variation):
        return None

    def reset(self):
        return [], self.observation

    def step(self, action):
        self.step_calls += 1
        self.actions.append(np.asarray(action).copy())
        if self.step_calls == 1:
            raise _InvalidActionForTest("primary action failed")
        return self.observation, 0.0, False


class _SuccessThenContinuingEnvironment(_InvalidThenSuccessfulEnvironment):
    def step(self, action):
        self.step_calls += 1
        self.actions.append(np.asarray(action).copy())
        return self.observation, float(self.step_calls == 1), self.step_calls == 1


def _formal_episode_inputs(environment):
    descriptions, observation = environment.reset()
    return {
        "descriptions": descriptions,
        "observation": observation,
        "fresh_task_generation": {"test_fixture": True},
    }


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
        "manifest_schema": "dynamac-direct-training-v2",
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


def test_inherited_v3_cli_defaults_do_not_overwrite_v4_artifacts() -> None:
    integration_root = DEFAULT_CONFIG.parents[1]

    assert DEFAULT_CONFIG == integration_root / "configs" / "dynamac_rlbench_v3.json"
    assert DEFAULT_TRAINING_MODELS_DIR == integration_root / "models" / "v3"
    assert DEFAULT_EVALUATION_MODELS_DIR == integration_root / "models" / "v3"
    assert DEFAULT_EVALUATION_RESULTS_DIR == integration_root / "results" / "v3"
    assert DEFAULT_UNIMANUAL_MODELS_DIR == integration_root / "models" / "v3"
    assert DEFAULT_UNIMANUAL_RESULTS_DIR == integration_root / "results" / "v3"


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

    left, right = bimanual_observations_from_rlbench(observation, "bimanual_lift_tray")

    assert np.allclose(left.ee_pose, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(right.ee_pose, left.ee_pose)
    assert list(left.frames) == ["item", "tray"]


@pytest.mark.parametrize("bimanual", [False, True])
def test_joint_hold_noop_bypasses_action_mode_and_holds_all_joints(
    bimanual,
) -> None:
    events = []

    class Component:
        def __init__(self, name, positions):
            self.name = name
            self.positions = np.asarray(positions, dtype=np.float64)
            self.targets = []
            self.attachment = object()

        def get_joint_positions(self):
            events.append(("get", self.name))
            return self.positions.copy()

        def set_joint_target_positions(self, positions):
            events.append(("set", self.name))
            self.targets.append(np.asarray(positions, dtype=np.float64))

    names = (
        ("right_arm", "right_gripper", "left_arm", "left_gripper")
        if bimanual
        else ("arm", "gripper")
    )
    components = {
        name: Component(name, [index + 0.1, index + 0.2])
        for index, name in enumerate(names)
    }
    robot = SimpleNamespace(is_bimanual=bimanual, **components)

    class Task:
        def success(self):
            events.append(("success", None))
            return False, False

        def reward(self):
            raise AssertionError("unshaped joint hold must not request reward()")

    class Scene:
        def __init__(self):
            self.robot = robot
            self.task = Task()
            self.steps = 0

        def step(self):
            events.append(("scene_step", None))
            self.steps += 1

    observation = object()

    class Environment:
        _reset_called = True
        _shaped_rewards = False

        def __init__(self):
            self._scene = Scene()
            self.action_steps = 0
            self.observation_calls = 0

        def step(self, _action):
            self.action_steps += 1
            raise AssertionError("joint hold must not call TaskEnvironment.step")

        def get_observation(self):
            events.append(("observation", None))
            self.observation_calls += 1
            return observation

    environment = Environment()
    attachments = {name: component.attachment for name, component in components.items()}

    actual_observation, reward, terminate = step_current_joint_hold_noop(environment)

    assert actual_observation is observation
    assert reward == 0.0
    assert terminate is False
    assert environment.action_steps == 0
    assert environment._scene.steps == 1
    assert environment.observation_calls == 1
    assert [event[0] for event in events] == (
        ["get"] * len(names)
        + ["set"] * len(names)
        + ["scene_step", "success", "observation"]
    )
    for name, component in components.items():
        assert len(component.targets) == 1
        assert np.array_equal(component.targets[0], component.positions)
        assert component.attachment is attachments[name]


def test_joint_hold_noop_propagates_shaped_reward_and_termination() -> None:
    events = []

    class Component:
        def get_joint_positions(self):
            return [0.25]

        def set_joint_target_positions(self, positions):
            assert positions == [0.25]

    class Task:
        def success(self):
            events.append("success")
            return False, True

        def reward(self):
            events.append("reward")
            return 2.5

    scene = SimpleNamespace(
        robot=SimpleNamespace(
            is_bimanual=False,
            arm=Component(),
            gripper=Component(),
        ),
        task=Task(),
        step=lambda: events.append("scene_step"),
    )
    observation = object()
    environment = SimpleNamespace(
        _scene=scene,
        _reset_called=True,
        _shaped_rewards=True,
        get_observation=lambda: (events.append("observation") or observation),
    )

    result = step_current_joint_hold_noop(environment)

    assert result == (observation, 2.5, True)
    assert events == ["scene_step", "success", "reward", "observation"]


def test_evaluation_defaults_to_200_episodes() -> None:
    args = build_evaluation_parser().parse_args(["--task", "bimanual_handover_item"])

    assert args.episodes == 200
    assert args.headless is True
    assert args.policy_timeout == 120.0
    assert args.scenario == "static"
    assert args.scenario_trigger_fraction == 1.0 / 3.0
    assert args.scenario_steps == 10
    assert args.scenario_reference_steps is None
    assert args.max_primary_action_attempts == 1


def test_motion_plan_cache_waits_for_concurrent_staging_writer(
    tmp_path,
    monkeypatch,
) -> None:
    pytest.skip("formal evaluation now requires a sealed canonical eval-set ID")
    cache = tmp_path / "shared-plans.json"
    generation_lock = cache.with_name(cache.name + ".generation.lock")
    generation_lock.write_text("pid=123\n", encoding="utf-8")
    generation_body = {
        "schema": FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
        "protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
        "generation_index": 1,
        "episode_seed": 0,
        "variation": 0,
        "task_name": "bimanual_handover_item",
        "physics_running_before_stop": False,
        "physics_stopped_before_task_reload": True,
        "previous_task_present": False,
        "previous_task_unloaded_before_stop": False,
        "previous_task_unloaded_while_physics_running": False,
        "scene_task_absent_before_stop": True,
        "task_model_loaded_fresh": True,
        "fresh_task_python_instance_created": True,
        "task_model_only_reloaded": True,
        "base_scene_reloaded": False,
        "physics_started_by_task_environment": True,
        "rng_seeded_after_reload_immediately_before_reset": True,
        "variation_set_after_seed_before_reset": True,
        "task_environment_reset_calls": 1,
        "reset_verify_instance": True,
    }
    generation_evidence = {
        **generation_body,
        "fingerprint": _canonical_json_fingerprint(generation_body),
    }
    plan = StagedMotionPlan(
        task_name="bimanual_handover_item",
        source_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        goal_pose=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        source_low_dim_state=(0.0,),
        episode_seed=0,
        variation=0,
        validation={
            "schema": STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "sampling_attempts": 1,
            "staging_max_attempts": 20,
            "fresh_task_generation_protocol_id": (FRESH_TASK_GENERATION_PROTOCOL_ID),
            "selected_source_fresh_task_generation": generation_evidence,
        },
    )
    payload = staged_motion_plan_batch(
        task_name="bimanual_handover_item",
        base_seed=0,
        variations=[0],
        plans=[plan],
    )

    def finish_other_writer(_seconds):
        atomic_json(cache, payload)
        generation_lock.unlink()

    monkeypatch.setattr(direct_evaluate.time, "sleep", finish_other_writer)
    monkeypatch.setattr(
        direct_evaluate.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("waiting reader must not spawn a second staging process")
        ),
    )
    args = SimpleNamespace(
        scenario="teleport",
        motion_plans=cache,
        task="bimanual_handover_item",
        seed=0,
        episodes=1,
        scenario_max_attempts=20,
        motion_plan_wait_timeout=5.0,
        headless=True,
    )

    path, loaded = direct_evaluate._load_or_generate_motion_plans(args)

    assert path == cache
    assert loaded["batch_fingerprint"] == payload["batch_fingerprint"]


def test_primary_action_retry_budget_resets_only_after_success() -> None:
    budget = PrimaryActionRetryBudget(3)

    assert budget.record_failure() is False
    assert budget.record_failure() is False
    budget.record_success()
    assert budget.attempts == 0
    assert budget.record_failure() is False
    assert budget.record_failure() is False
    assert budget.record_failure() is True


@pytest.mark.parametrize(
    ("terminal", "expected_success", "expected_reason"),
    (
        ((False, False), False, "maximum_physics_steps_reached"),
        ((True, True), True, "success"),
        ((False, True), False, "explicit_terminate"),
    ),
)
def test_final_settling_holds_commands_for_up_to_ten_raw_steps(
    terminal,
    expected_success,
    expected_reason,
) -> None:
    class Task:
        def __init__(self):
            self.calls = 0

        def success(self):
            self.calls += 1
            if terminal == (False, False) or self.calls < 3:
                return False, False
            return terminal

    class Scene:
        def __init__(self):
            self.task = Task()
            self.steps = 0

        def step(self):
            self.steps += 1

    scene = Scene()
    environment = SimpleNamespace(_scene=scene)

    result = run_final_settling(
        environment,
        physics_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )

    expected_steps = 10 if terminal == (False, False) else 3
    assert result["protocol_id"] == FINAL_SETTLING_PROTOCOL_ID
    assert result["maximum_physics_steps"] == 10
    assert result["steps_executed"] == expected_steps
    assert result["first_terminal_step"] == (None if expected_steps == 10 else 3)
    assert result["success"] is expected_success
    assert result["terminate"] is terminal[1]
    assert result["stop_reason"] == expected_reason
    assert scene.steps == expected_steps
    assert scene.task.calls == expected_steps


@pytest.mark.parametrize("invalid", (True, 3.0, 0, -1))
def test_primary_action_retry_budget_rejects_non_positive_or_non_integer(
    invalid,
) -> None:
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

    result = _run_unimanual_episode(
        environment,
        worker,
        args,
        episode=0,
        **_formal_episode_inputs(environment),
    )

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
def test_dynamic_success_before_trigger_is_retained_in_planned_denominator(
    finalize,
    event_key,
) -> None:
    row = {
        "episode": 0,
        "success": True,
        "steps": 10,
        "reason": "success",
        event_key: [],
    }

    result = finalize(
        row,
        scenario="teleport",
        trigger_step=62,
        trigger_reached=False,
        smooth_steps=10,
    )

    assert result["success"] is True
    assert result["pre_intervention_terminal"] is True
    assert result["pre_intervention_terminal_outcome"] == "success"
    assert result["dynamic_condition_exercised"] is False
    assert result["dynamic_condition_unexercised"] is True
    assert result["intervention_complete"] is None


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
        "training_adapter_protocol",
        "checkpoint_trigger_audit_fingerprint",
        "v3_trigger_anchor_evidence",
    }
    assert identity["model_schema_version"] == expected_summary["model_schema_version"]
    assert (
        identity["selection_semantics_id"] == expected_summary["selection_semantics_id"]
    )
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


def test_frozen_policy_clock_consumes_applied_action_independent_of_executor_status() -> (
    None
):
    policy = _TransactionalUnimanualPolicy(duration=2)
    server = _transaction_server("stack_wine", policy, bimanual=False)
    observation = _unimanual_wire_observation()
    server.handle({"command": "reset", "observation": observation})

    first = server.handle({"command": "act", "observation": observation})
    assert policy.clock == 1
    assert first["gripper_authorization"] == {"single": True}
    partial = server.handle(
        {
            "command": "commit",
            "transaction_id": first["transaction_id"],
            "primary_action_status": "progressed",
        }
    )

    assert partial["committed"] is True
    assert partial["primary_action_status"] == "progressed"
    assert partial["complete"] is False
    assert policy.clock == 1

    second = server.handle({"command": "act", "observation": observation})
    assert second["gripper_authorization"] == {"single": True}
    stopped = server.handle(
        {
            "command": "commit",
            "transaction_id": second["transaction_id"],
            "primary_action_status": "stopped",
        }
    )
    assert stopped["primary_action_status"] == "stopped"
    assert stopped["complete"] is True
    assert policy.clock == 2


def test_policy_server_applies_global_boundary_gripper_timing_transactionally() -> None:
    class BoundaryPolicy(_TransactionalUnimanualPolicy):
        def preview_next_gripper(self):
            return SimpleNamespace(
                gripper=np.asarray([-1.0]),
                crosses_skill_boundary=True,
                repeats_terminal=False,
            )

    policy = BoundaryPolicy(duration=2)
    server = _transaction_server("stack_wine", policy, bimanual=False)
    observation = _unimanual_wire_observation()
    server.handle({"command": "reset", "observation": observation})

    first = server.handle({"command": "act", "observation": observation})
    assert first["action"][7] == 0.0
    assert first["gripper_timing"]["changed_action_indices"] == [7]
    assert first["gripper_timing"]["crosses_skill_boundary_by_action_index"] == {
        "7": True
    }
    server.handle({"command": "abort", "transaction_id": first["transaction_id"]})
    assert policy.clock == 0

    retry = server.handle({"command": "act", "observation": observation})
    assert retry["action"] == first["action"]
    assert retry["gripper_timing"] == first["gripper_timing"]


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
    assert first["gripper_authorization"] == {"left": True, "right": True}
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


def test_bimanual_evaluator_commits_joint_hold_for_failed_primary(monkeypatch) -> None:
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
        **_formal_episode_inputs(environment),
    )

    assert result["reason"] == "policy_complete"
    assert result["invalid_actions"] == 1
    assert result["steps"] == 2
    assert result["control_attempts"] == 2
    assert environment.step_calls == 3
    assert environment.actions[1].shape == ()
    assert environment.actions[2].shape == (18,)
    assert np.array_equal(environment.actions[0], np.zeros(18))
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("commit", 1),
        ("act", None),
        ("commit", 2),
    ]


def test_bimanual_diagnostic_can_continue_policy_after_latched_success(
    monkeypatch,
) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _SuccessThenContinuingEnvironment(observation)
    worker = _TransactionalWorker(np.zeros(18), complete_after=3)

    result = _run_bimanual_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=6,
        post_success_policy_steps=5,
        **_formal_episode_inputs(environment),
    )

    assert result["success"] is True
    assert result["reason"] == "success_then_policy_complete"
    assert result["steps"] == 3
    assert result["post_success_policy_continuation"] == {
        "diagnostic_only": True,
        "requested_steps": 5,
        "success_latched_policy_step": 1,
        "executed_steps": 2,
        "policy_complete": True,
    }


def test_bimanual_dynamic_motion_does_not_repeat_on_invalid_action_retry(
    monkeypatch,
) -> None:
    _install_invalid_action_module(monkeypatch)
    pose = np.asarray(_wire_pose())
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
        left=SimpleNamespace(gripper_pose=pose, gripper_open=1.0),
    )
    environment = _InvalidThenSuccessfulEnvironment(observation)
    environment.get_observation = lambda: observation
    worker = _TransactionalWorker(np.zeros(18))

    class Controller:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.applied_steps = []
            self.bind_kwargs = None
            self.__class__.instances.append(self)

        def resolved_trigger_step(self, _horizon):
            return 0

        def bind_staged_source(self, *_args, **kwargs):
            self.bind_kwargs = dict(kwargs)
            return {"required": False, "matched": None}

        def apply(self, _task_environment, *, step, horizon):
            assert horizon == 3
            self.applied_steps.append(step)
            applied = step == 0
            return {
                "kind": "teleport_task",
                "step": step,
                "trigger_step": 0,
                "applied": applied,
                "protocol_effective": applied,
                "complete": applied,
                "endpoint_applied": applied,
            }

    monkeypatch.setattr(direct_evaluate, "ScenarioController", Controller)

    result = direct_evaluate._run_episode(
        environment,
        worker,
        episode=0,
        seed=0,
        horizon=3,
        scenario="teleport",
        scenario_trigger_step=0,
        scenario_reference_steps=3,
        episode_variation=2,
        **_formal_episode_inputs(environment),
    )

    assert Controller.instances[0].applied_steps == [0, 1]
    assert Controller.instances[0].bind_kwargs["variation"] == 2
    assert len(result["scenario_events"]) == 1
    assert result["committed_policy_steps"] == 2
    assert result["dynamic_clock_steps"] == 2
    assert result["intervention_complete"] is True


def test_unimanual_evaluator_commits_joint_hold_for_failed_primary(monkeypatch) -> None:
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

    result = _run_unimanual_episode(
        environment,
        worker,
        args,
        episode=0,
        **_formal_episode_inputs(environment),
    )

    assert result["reason"] == "policy_complete"
    assert result["invalid_actions"] == 1
    assert result["steps"] == 2
    assert result["control_attempts"] == 2
    assert environment.step_calls == 3
    assert environment.actions[1].shape == ()
    assert environment.actions[2].shape == (9,)
    assert np.array_equal(environment.actions[0], np.zeros(9))
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("commit", 1),
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
        variation=0,
        seed=0,
        horizon=3,
        arm="left",
        trigger=0,
        **_formal_episode_inputs(environment),
    )

    assert result["reason"] == "policy_complete"
    assert result["invalid_actions"] == 1
    assert worker.requests == [
        ("reset", None),
        ("act", None),
        ("commit", 1),
        ("act", None),
        ("commit", 2),
    ]


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


def test_joint_target_control_ignores_task_terminal_until_combined_action_finishes() -> (
    None
):
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


# V4 inherits this frozen checkpoint-backed trigger contract for tasks without
# a V4-specific trigger override.  It belongs with current evaluator tests,
# not with the retired V3 report validator.
def _inherited_trigger_policy(*, break_required_window: bool = False):
    duration = 72
    raw = np.zeros(duration, dtype=bool)
    raw[68:] = True
    if break_required_window:
        raw[:40] = True
        raw[67:] = True
    gate_enabled = float(np.mean(raw)) > 0.5
    availability = ~raw if gate_enabled else np.ones(duration, dtype=bool)
    selected = np.asarray([True])
    stream = SimpleNamespace(
        availability=availability[None, :],
        active=availability[None, :] & selected[:, None],
        selected_by_eq6=selected,
    )
    skill = SimpleNamespace(
        label=0,
        duration=duration,
        streams={"wine_bottle": stream},
        link_diagnostics={"wine_bottle": {"raw_link_mask": raw.tolist()}},
    )
    return SimpleNamespace(
        skills=[skill],
        skill_sequence=(0,),
        config=SimpleNamespace(
            link_mask_scope="skill_majority_gate_timestep",
            link_filter="none",
        ),
        selection_semantics_id=(
            "eq5_skill_majority_gate_timestep_availability_before_eq6_and_poe_"
            "time_state_position3d_unimodal_v1"
        ),
    )


def test_inherited_intervention_registry_freezes_trigger_semantics() -> None:
    protocol = load_v3_intervention_protocol()

    assert protocol["schema"] == V3_INTERVENTION_SCHEMA
    assert len(protocol["fingerprint"]) == 64
    assert "staging_max_attempts" not in protocol
    assert protocol["provenance"]["frozen_before_v3_formal_evaluation"] is True
    assert protocol["provenance"]["manifests_reauthenticated"] is True
    assert set(protocol["dynamic_environment"]) == {
        "stack_wine",
        "place_cups",
        "open_microwave",
        "wipe_desk",
        "bimanual_put_bottle_in_fridge",
        "bimanual_handover_item",
        "bimanual_lift_tray",
        "bimanual_sweep_to_dustpan",
    }
    for profile in protocol["dynamic_environment"].values():
        assert profile["interaction_arm"] in {"single", "left", "right"}
        assert profile["interaction_object"]
        assert profile["interaction_event"]
        assert profile["expected_gripper_state"] in {"open", "closed"}
        assert profile["phase"] == pytest.approx(
            profile["local_tick"] / (profile["expected_duration"] - 1),
            abs=1.0e-15,
        )
        assert profile["required_active_window"] == [
            profile["local_tick"],
            profile["local_tick"] + protocol["smooth_steps"] - 1,
        ]


def test_inherited_intervention_registry_rejects_missing_manual_semantics(
    tmp_path,
) -> None:
    payload = load_v3_intervention_protocol()
    payload.pop("fingerprint")
    del payload["dynamic_environment"]["bimanual_handover_item"]["interaction_event"]
    path = tmp_path / "invalid-interventions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields are invalid"):
        load_v3_intervention_protocol(path)


def test_inherited_checkpoint_trigger_audit_authenticates_eq5_eq6() -> None:
    audit = checkpoint_trigger_audit(_inherited_trigger_policy())

    assert audit["schema"] == V3_CHECKPOINT_TRIGGER_AUDIT_SCHEMA
    assert audit["skills"][0]["frames"]["wine_bottle"] == {
        "selected_by_eq6": [True],
        "raw_link_runs": [[[68, 71]]],
        "majority_gate_enabled": [False],
        "availability_runs": [[[0, 71]]],
        "poe_active_runs": [[[0, 71]]],
    }
    assert len(audit["fingerprint"]) == 64


def test_inherited_trigger_is_bound_to_authenticated_checkpoint_window() -> None:
    audit = checkpoint_trigger_audit(_inherited_trigger_policy())
    evidence = build_v3_trigger_anchor_evidence("stack_wine", audit, {})
    anchor = evidence["anchors"]["stack_wine"]

    assert evidence["validated"] is True
    assert anchor["resolved_global_tick"] == 58
    assert anchor["required_active_window"] == [58, 67]
    assert anchor["selected_by_eq6"] == [True]
    assert anchor["interaction_arm"] == "single"
    assert anchor["interaction_object"] == "wine_bottle"
    assert anchor["expected_gripper_state"] == "open"


def test_inherited_trigger_rejects_an_inactive_smooth_window() -> None:
    audit = checkpoint_trigger_audit(
        _inherited_trigger_policy(break_required_window=True)
    )

    with pytest.raises(RuntimeError, match=r"Equation \(5\)-available"):
        build_v3_trigger_anchor_evidence("stack_wine", audit, {})


def test_inherited_dynamic_profile_returns_a_copy() -> None:
    first = dynamic_trigger_profile("stack_wine")
    first["local_tick"] = 0

    assert dynamic_trigger_profile("stack_wine")["local_tick"] == 58


def test_current_evaluator_requires_an_authenticated_inherited_trigger() -> None:
    audit = checkpoint_trigger_audit(_inherited_trigger_policy())
    envelope = build_v3_trigger_anchor_evidence("stack_wine", audit, {})
    identity = {
        "manifest_authenticated": True,
        "training_manifest_schema": "dynamac-direct-training-v3",
        "checkpoint_trigger_audit_fingerprint": audit["fingerprint"],
        "v3_trigger_anchor_evidence": envelope,
    }

    resolved = resolve_authenticated_v3_trigger(identity, task="stack_wine")
    assert resolved["trigger_step"] == 58
    assert resolved["profile_key"] == "stack_wine"
    assert resolved["evidence"]["validated"] is True

    identity["v3_trigger_anchor_evidence"] = {
        **envelope,
        "intervention_protocol_fingerprint": "forged",
    }
    with pytest.raises(RuntimeError, match="envelope"):
        resolve_authenticated_v3_trigger(identity, task="stack_wine")
