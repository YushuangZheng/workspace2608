from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_closed_loop.eval.fault_injection import (
    FaultInjectionKind,
    FaultInjectionSpec,
    FaultInjectingTaskEnvironment,
)
from evaluations.phase6_rlbench_integration import run_fault_diagnostic_subset


def _pose(x: float) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class _Object:
    def __init__(self, name: str, position=(0.0, 0.0, 0.0)) -> None:
        self.name = name
        self.position = np.asarray(position, dtype=np.float64)

    def get_name(self):
        return self.name

    def get_position(self):
        return self.position.copy()

    def set_position(self, position):
        self.position = np.asarray(position, dtype=np.float64).copy()


class _Gripper:
    def __init__(self, objects=()) -> None:
        self.objects = list(objects)
        self.release_calls = 0

    def get_grasped_objects(self):
        return list(self.objects)

    def release(self):
        self.objects.clear()
        self.release_calls += 1


class _TaskEnvironment:
    def __init__(self, *, bimanual=False, objects=()) -> None:
        self.bimanual = bimanual
        if bimanual:
            robot = SimpleNamespace(
                right_gripper=_Gripper(objects),
                left_gripper=_Gripper(),
            )
            self.observation = SimpleNamespace(
                right=SimpleNamespace(gripper_pose=_pose(0.1)),
                left=SimpleNamespace(gripper_pose=_pose(-0.1)),
            )
        else:
            robot = SimpleNamespace(gripper=_Gripper(objects))
            self.observation = SimpleNamespace(gripper_pose=_pose(0.1))
        self._scene = SimpleNamespace(robot=robot)
        self.actions = []

    def get_observation(self):
        return self.observation

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        return self.observation, 0.0, False


def test_time_stall_replaces_only_pose_and_records_every_suppressed_cycle() -> None:
    environment = _TaskEnvironment()
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.TIME_STALL,
            duration_cycles=2,
            motion_trigger_distance=0.005,
        ),
    )
    action = np.concatenate((_pose(0.2), [0.0, 1.0]))

    wrapped.step(action)
    wrapped.step(action)
    wrapped.step(action)

    assert wrapped.triggered
    assert np.allclose(environment.actions[0][:7], _pose(0.1))
    assert np.allclose(environment.actions[1][:7], _pose(0.1))
    assert np.allclose(environment.actions[2], action)
    assert environment.actions[0][7:].tolist() == [0.0, 1.0]
    assert [event["kind"] for event in wrapped.events] == [
        "time_stall",
        "time_stall_cycle",
        "time_stall_cycle",
    ]


def test_bimanual_time_stall_uses_right_first_wire_layout() -> None:
    environment = _TaskEnvironment(bimanual=True)
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.TIME_STALL,
            arm="left",
            duration_cycles=1,
        ),
    )
    right = np.concatenate((_pose(0.3), [0.0, 0.0]))
    left = np.concatenate((_pose(-0.3), [1.0, 0.0]))
    action = np.concatenate((right, left))

    wrapped.step(action)

    assert np.allclose(environment.actions[0][:9], right)
    assert np.allclose(environment.actions[0][9:16], _pose(-0.1))
    assert environment.actions[0][16:].tolist() == [1.0, 0.0]


def test_grasp_failure_suppresses_only_selected_close_occurrence() -> None:
    environment = _TaskEnvironment()
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.GRASP_FAILURE,
            close_occurrence=2,
        ),
    )
    open_action = np.concatenate((_pose(0.1), [1.0, 0.0]))
    close_action = np.concatenate((_pose(0.1), [0.0, 0.0]))
    for action in (
        close_action,
        open_action,
        close_action,
        close_action,
        open_action,
        close_action,
    ):
        wrapped.step(action)

    assert [value[7] for value in environment.actions] == [
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert wrapped.events[0]["close_occurrence"] == 2
    assert wrapped.events[-1]["kind"] == "grasp_failure_occurrence_ended"


@pytest.mark.parametrize(
    ("kind", "expected_position", "displaced"),
    (
        (FaultInjectionKind.UNEXPECTED_DROP, [0.0, 0.0, 0.0], False),
        (FaultInjectionKind.RELATION_MISMATCH, [0.04, 0.0, 0.0], True),
    ),
)
def test_relation_fault_releases_after_stable_carriage(
    kind, expected_position, displaced
) -> None:
    item = _Object("item")
    environment = _TaskEnvironment(objects=[item])
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(kind, minimum_grasped_cycles=2),
    )
    action = np.concatenate((_pose(0.1), [0.0, 0.0]))

    wrapped.step(action)
    wrapped.step(action)
    wrapped.step(action)

    assert environment._scene.robot.gripper.release_calls == 1
    assert np.allclose(item.position, expected_position)
    trigger = next(event for event in wrapped.events if event["kind"] == kind.value)
    assert trigger["released_objects"] == ["item"]
    assert trigger["displaced"] is displaced


def test_fault_spec_rejects_unknown_configuration_instead_of_ignoring_it() -> None:
    with pytest.raises(ValueError, match="unknown fault"):
        FaultInjectionSpec.from_mapping({"kind": "time_stall", "typo": 1})


def test_fault_launcher_profiles_share_executor_but_select_distinct_policy_layers() -> (
    None
):
    protocol, cell = run_fault_diagnostic_subset._load_cell(
        run_fault_diagnostic_subset.DEFAULT_PROTOCOL,
        "stack_wine_time_stall",
    )
    assert protocol["executor_control"] == "shared_stage6_executor_for_all_methods"
    assert cell["episode_indices"] == [0, 1, 2]
    assert run_fault_diagnostic_subset._method_args("dynamac_v4") == [
        "--policy-type",
        "dynamac",
    ]
    assert run_fault_diagnostic_subset._method_args("progress_dynamic_roles") == [
        "--policy-type",
        "closed_loop_multistream",
        "--closed-loop-feature-profile",
        "progress_dynamic_roles",
    ]


def test_fault_launcher_attaches_episode_level_physical_evidence() -> None:
    environment = _TaskEnvironment()

    def original(task_environment, action):
        task_environment.step(action)
        return {"success": False}

    module = SimpleNamespace(_run_episode=original)
    run_fault_diagnostic_subset._install_fault(
        module,
        FaultInjectionSpec(
            FaultInjectionKind.TIME_STALL,
            duration_cycles=1,
        ),
    )
    action = np.concatenate((_pose(0.2), [1.0, 0.0]))
    row = module._run_episode(environment, action)

    assert row["physical_fault"]["triggered"] is True
    assert row["physical_fault"]["policy_state_mutated"] is False
    assert row["physical_fault"]["observation_hidden"] is False
