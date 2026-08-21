from __future__ import annotations

import copy
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.core import runtime
from integrations.rlbench.rlbench_dynamac.eval import direct_evaluate
from integrations.rlbench.rlbench_dynamac.eval import table_iii_coordination as coordination
from integrations.rlbench.rlbench_dynamac.protocols import v4_dynamic_protocol as protocol


def _observation(*, left_pose=None, right_pose=None, left_open=1.0, right_open=0.0):
    left_pose = left_pose or [0.2, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0]
    right_pose = right_pose or [-0.2, 0.1, 0.5, 0.0, 0.0, 0.0, 1.0]
    return SimpleNamespace(
        left=SimpleNamespace(
            gripper_pose=np.asarray(left_pose, dtype=np.float64),
            gripper_open=left_open,
        ),
        right=SimpleNamespace(
            gripper_pose=np.asarray(right_pose, dtype=np.float64),
            gripper_open=right_open,
        ),
    )


def _fake_lift_plan(candidate_seed=1234):
    source = np.asarray([0.1, -0.2, 0.75, 0.1, -0.2, 0.3, 0.92])
    source[3:7] /= np.linalg.norm(source[3:7])
    goal = protocol.sample_v4_lift_goal_pose(source, candidate_seed)
    plan = SimpleNamespace(
        task_name=protocol.V4_LIFT_TASK,
        source_pose=tuple(source),
        goal_pose=tuple(goal),
        validation={"selected_candidate_seed": candidate_seed},
    )
    plan.validation["v4_lift_tray"] = protocol._v4_lift_plan_evidence(plan)
    return plan


def test_v4_protocols_freeze_requested_ticks_motion_and_identity():
    lift = protocol.load_v4_lift_intervention_protocol()
    motion = protocol.load_v4_lift_motion_source_protocol()
    coord = protocol.load_v4_coordination_intervention_protocol()

    assert lift["formal_scenarios"] == ["static", "teleport"]
    assert lift["trigger"]["skill_label"] == 0
    assert lift["trigger"]["local_tick"] == 35
    assert lift["trigger"]["global_tick"] == 35
    assert lift["trigger"]["expected_gripper_states"] == {
        "left": "open",
        "right": "open",
    }
    assert motion["translation"]["radial_min_m"] == pytest.approx(0.03)
    assert motion["translation"]["radial_max_m"] == pytest.approx(0.08)
    assert motion["translation"]["z_delta_m"] == 0.0
    assert motion["rotation"]["yaw_delta_abs_max_rad"] == pytest.approx(0.10)
    assert motion["candidate_generation"]["policy_result_fields_read"] is False
    assert coord["trigger"]["global_tick"] == 235
    assert coord["motion"]["smooth_policy_ticks"] == 10
    assert coord["motion"]["persistent_policy_target_offset"] is True
    assert (
        coord["clock_semantics"]["policy_clock_advances_during_smooth_window"]
        is True
    )

    components = protocol.v4_lift_task_identity_components()
    assert set(components) == {"task_semantics", "motion_source", "intervention"}
    assert all(len(component["fingerprint"]) == 64 for component in components.values())


@pytest.mark.parametrize("candidate_seed", [0, 1, 2608000000, 4294967294])
def test_v4_lift_goal_is_deterministic_source_relative_xy_only(candidate_seed):
    source = np.asarray([0.2, -0.1, 0.8, 0.12, -0.08, 0.31, 0.94])
    source[3:7] /= np.linalg.norm(source[3:7])

    first = protocol.sample_v4_lift_goal_pose(source, candidate_seed)
    second = protocol.sample_v4_lift_goal_pose(source, candidate_seed)
    geometry = protocol.v4_lift_plan_geometry(source, first)

    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    assert 0.03 <= geometry["xy_radius_m"] <= 0.08
    assert geometry["z_delta_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert abs(geometry["yaw_delta_rad"]) <= 0.10
    assert geometry["relative_rotation_xy_norm"] <= 1.0e-12


def test_v4_lift_runtime_loader_authenticates_geometry_and_no_result_selection(
    monkeypatch,
):
    plan = _fake_lift_plan()
    monkeypatch.setattr(
        runtime,
        "load_staged_motion_plan_batch",
        lambda payload: [plan],
    )

    assert protocol.load_v4_lift_motion_plan_batch({"fixture": True}) == [plan]
    evidence = plan.validation["v4_lift_tray"]
    assert evidence["selection_authority"] == "scene_validity_only"
    assert evidence["policy_result_fields_read"] is False

    forged = copy.deepcopy(plan)
    forged.goal_pose = tuple(np.asarray(forged.goal_pose) + np.asarray([0, 0, 0.01, 0, 0, 0, 0]))
    monkeypatch.setattr(
        runtime,
        "load_staged_motion_plan_batch",
        lambda payload: [forged],
    )
    with pytest.raises(ValueError, match="evidence|geometry"):
        protocol.load_v4_lift_motion_plan_batch({"fixture": True})


def test_v4_lift_plan_evidence_accepts_json_roundoff_but_rejects_real_drift():
    plan = _fake_lift_plan()
    geometry = plan.validation["v4_lift_tray"]["accepted_candidate_geometry"]
    geometry["yaw_delta_rad"] += 3.0e-17

    assert protocol.validate_v4_lift_motion_plan(plan)["validated"] is True

    geometry["yaw_delta_rad"] += 1.0e-12
    with pytest.raises(ValueError, match="evidence"):
        protocol.validate_v4_lift_motion_plan(plan)


def test_v4_lift_expected_goal_accepts_one_ulp_but_rejects_real_drift():
    plan = _fake_lift_plan()
    expected_goal = plan.validation["v4_lift_tray"]["expected_goal_pose"]
    expected_goal[6] = float(np.nextafter(expected_goal[6], np.inf))

    assert protocol.validate_v4_lift_motion_plan(plan)["validated"] is True

    expected_goal[6] += 1.0e-12
    with pytest.raises(ValueError, match="evidence"):
        protocol.validate_v4_lift_motion_plan(plan)


@pytest.mark.parametrize(
    "invalid_goal",
    [
        [0.0] * 6,
        [0.0] * 6 + [True],
        [0.0] * 6 + [float("nan")],
    ],
)
def test_v4_lift_expected_goal_requires_finite_numeric_pose7(invalid_goal):
    plan = _fake_lift_plan()
    plan.validation["v4_lift_tray"]["expected_goal_pose"] = invalid_goal

    with pytest.raises(ValueError, match="evidence"):
        protocol.validate_v4_lift_motion_plan(plan)


def test_stage_v4_lift_passes_source_relative_sampler_without_policy_input(
    monkeypatch,
):
    seen = {}
    legacy_plan = object()

    def stage(environment, task_class, **kwargs):
        seen.update(kwargs)
        source = np.asarray([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0])
        goal = kwargs["goal_candidate_sampler"](source, 17)
        assert 0.03 <= np.linalg.norm(goal[:2] - source[:2]) <= 0.08
        return legacy_plan

    monkeypatch.setattr(runtime, "stage_scenario_motion_plan", stage)
    monkeypatch.setattr(
        protocol,
        "attach_v4_lift_plan_evidence",
        lambda plan: ("v4", plan),
    )

    result = protocol.stage_v4_lift_motion_plan(
        object(),
        object(),
        episode_seed=2608000000,
        variation=0,
    )

    assert result == ("v4", legacy_plan)
    assert seen["task_name"] == protocol.V4_LIFT_TASK
    assert seen["episode_seed"] == 2608000000
    assert seen["max_attempts"] == 100
    assert callable(seen["goal_candidate_sampler"])
    assert not any("policy" in key or "result" in key for key in seen)


def test_direct_v4_lift_loader_uses_task_scoped_eval_v2_registry(monkeypatch):
    seen = {}
    plan = SimpleNamespace(validation={"goal_sampling_max_attempts": 100})
    envelope = {
        "runtime_loader": protocol.V4_LIFT_RUNTIME_LOADER_ID,
        "task_identity": {
            "components": protocol.v4_lift_task_identity_components(),
        },
    }

    def fixed(eval_set_id, task, *, runtime_loaders):
        seen["eval_set_id"] = eval_set_id
        seen["task"] = task
        seen["runtime_loaders"] = runtime_loaders
        return {"payload": {}}, {"payload": envelope, "plans": [plan]}

    monkeypatch.setattr(direct_evaluate, "fixed_environment_plans", fixed)
    args = SimpleNamespace(
        release="v4",
        task=protocol.V4_LIFT_TASK,
        eval_set_id="rlbench_eval_v2",
        motion_plans=None,
        seed=direct_evaluate.GLOBAL_EVAL_SEED_START,
        episodes=direct_evaluate.FIXED_EVAL_EPISODES,
        scenario_max_attempts=100,
    )

    _manifest, selected = direct_evaluate._load_fixed_motion_plans(args)

    assert selected["plans"] == [plan]
    assert seen["task"] == protocol.V4_LIFT_TASK
    assert seen["runtime_loaders"] == {
        protocol.V4_LIFT_RUNTIME_LOADER_ID: protocol.load_v4_lift_motion_plan_batch
    }


def test_direct_v4_lift_rejects_smooth_and_authenticates_tick35():
    args = SimpleNamespace(
        release="v4",
        task=protocol.V4_LIFT_TASK,
        scenario="smooth",
        scenario_max_attempts=100,
        final_settling_steps=10,
        scenario_trigger_step=None,
        scenario_reference_steps=None,
    )
    with pytest.raises(ValueError, match="static/teleport"):
        direct_evaluate._validate_v4_lift_protocol_args(args)

    args.scenario = "teleport"
    worker = SimpleNamespace(policy_steps=70)
    registry, authentication = direct_evaluate._authenticated_v4_lift_trigger(
        args,
        worker,
    )
    assert registry["release"] == "v4"
    assert authentication["skill_label"] == 0
    assert authentication["local_tick"] == 35
    assert authentication["trigger_step"] == 35


def test_v4_coordination_offset_changes_only_selected_arm_position():
    action = np.arange(18, dtype=np.float64)
    command = coordination._offset_action(
        action,
        arm="left",
        fraction=0.4,
        translation=protocol.V4_COORDINATION_TRANSLATION_METERS,
    )

    np.testing.assert_allclose(command[:9], action[:9])
    np.testing.assert_allclose(command[9:12], action[9:12] + [0.0, 0.0, 0.012])
    np.testing.assert_allclose(command[12:], action[12:])


def test_v4_coordination_fraction_ramps_then_persists():
    assert [
        coordination._v4_coordination_fraction(tick, 235)
        for tick in range(234, 247)
    ] == [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]


def test_v4_coordination_trigger_is_fixed_at_existing_global_tick():
    authentication = protocol.v4_coordination_trigger_authentication(
        arm="left",
        policy_steps=300,
    )
    assert authentication["trigger_step"] == 235
    assert authentication["smooth_policy_ticks"] == 10
    assert authentication["final_smooth_tick"] == 244
    assert authentication["perturbed_arm"] == "left"
    with pytest.raises(ValueError, match="outside"):
        protocol.v4_coordination_trigger_authentication(
            arm="right",
            policy_steps=244,
        )


def test_v4_coordination_run_uses_ten_normal_ticks_then_persistent_offset(
    monkeypatch,
):
    class InvalidActionError(Exception):
        pass

    rlbench = ModuleType("rlbench")
    rlbench.__path__ = []
    backend = ModuleType("rlbench.backend")
    backend.__path__ = []
    exceptions = ModuleType("rlbench.backend.exceptions")
    exceptions.InvalidActionError = InvalidActionError
    monkeypatch.setitem(sys.modules, "rlbench", rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.backend", backend)
    monkeypatch.setitem(sys.modules, "rlbench.backend.exceptions", exceptions)

    initial = _observation()
    policy_action = direct_evaluate._noop_action(initial)
    policy_action[:3] = [0.6, -0.2, 0.7]
    calls = []
    next_transaction = 0

    class Worker:
        def request(self, command, *args, **kwargs):
            nonlocal next_transaction
            calls.append(command)
            if command == "reset":
                return {"ok": True}
            if command == "act":
                next_transaction += 1
                return {
                    "action": policy_action.copy(),
                    "transaction_id": next_transaction,
                }
            if command == "commit":
                return {"complete": False}
            raise AssertionError(command)

    class TaskEnvironment:
        def __init__(self, observation):
            self.observation = observation
            self.commands = []

        def step(self, command):
            command = np.asarray(command, dtype=np.float64)
            self.commands.append(command.copy())
            self.observation = _observation(
                right_pose=command[:7].tolist(),
                left_pose=command[9:16].tolist(),
                right_open=command[7],
                left_open=command[16],
            )
            reward = 1.0 if len(self.commands) == 12 else 0.0
            return self.observation, reward, False

    task = TaskEnvironment(initial)
    result = coordination._run_episode(
        task,
        Worker(),
        episode=0,
        variation=0,
        seed=2608000000,
        horizon=20,
        arm="right",
        trigger=0,
        max_primary_action_attempts=1,
        observation=initial,
        fresh_task_generation={"fixture": True},
        staged_source_binding={"fixture": True},
        release="v4",
    )

    assert calls == ["reset"] + ["act", "commit"] * 12
    assert len(task.commands) == 12
    np.testing.assert_allclose(
        [command[2] for command in task.commands],
        [0.703, 0.706, 0.709, 0.712, 0.715, 0.718, 0.721, 0.724, 0.727, 0.73, 0.73, 0.73],
    )
    np.testing.assert_allclose(
        [command[11] for command in task.commands],
        [policy_action[11]] * 12,
    )
    assert result["success"] is True
    assert result["committed_policy_steps"] == 12
    assert result["perturbed_steps"] == 10
    audit = result["coordination_intervention"]
    assert audit["smooth_policy_ticks_elapsed"] == 10
    assert audit["policy_requests"] == 10
    assert audit["policy_clock_advances"] == 10
    assert audit["offset_actions_applied"] == 10
    assert audit["persistent_policy_ticks_committed"] == 2
    assert audit["persistent_offset"] is True
    assert audit["completed"] is True


def test_v3_persistent_coordination_action_remains_unchanged():
    action = np.zeros(18, dtype=np.float64)
    np.testing.assert_allclose(
        coordination._perturb_action(action, "left", True)[9:12],
        [0.0, 0.0, 0.03],
    )
    np.testing.assert_allclose(
        coordination._perturb_action(action, "right", True)[:3],
        [0.0, 0.0, 0.03],
    )
    np.testing.assert_allclose(
        coordination._perturb_action(action, "left", False),
        action,
    )
