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

    def get_pose(self):
        return np.concatenate((self.position, [0.0, 0.0, 0.0, 1.0]))

    def set_pose(self, pose):
        self.position = np.asarray(pose, dtype=np.float64)[:3].copy()

    def is_dynamic(self):
        return True

    def is_respondable(self):
        return True


class _Sensor:
    def __init__(self, detected=()) -> None:
        self.detected = set(detected)

    def is_detected(self, obj):
        return obj in self.detected


class _Gripper:
    def __init__(self, objects=(), detected=()) -> None:
        self.objects = list(objects)
        self.release_calls = 0
        self._proximity_sensor = _Sensor(detected)
        self.open_amount = 0.0 if objects else 1.0

    def get_grasped_objects(self):
        return list(self.objects)

    def release(self):
        self.objects.clear()
        self.release_calls += 1

    def get_open_amount(self):
        return [self.open_amount, self.open_amount]


class _TaskEnvironment:
    def __init__(self, *, bimanual=False, objects=()) -> None:
        self.bimanual = bimanual
        graspables = list(objects)
        if bimanual:
            robot = SimpleNamespace(
                right_gripper=_Gripper(objects, detected=graspables),
                left_gripper=_Gripper(detected=graspables),
            )
            self.observation = SimpleNamespace(
                right=SimpleNamespace(gripper_pose=_pose(0.1)),
                left=SimpleNamespace(gripper_pose=_pose(-0.1)),
            )
        else:
            robot = SimpleNamespace(gripper=_Gripper(objects, detected=graspables))
            self.observation = SimpleNamespace(gripper_pose=_pose(0.1))
        self._scene = SimpleNamespace(
            robot=robot,
            task=SimpleNamespace(get_graspable_objects=lambda: list(graspables)),
        )
        self._gripper_action_mode = SimpleNamespace(_attach_grasped_objects=True)
        self._action_mode = SimpleNamespace(
            gripper_action_mode=self._gripper_action_mode,
            _policy_gripper_authorization=None,
        )
        self._action_mode.set_policy_gripper_authorization = (
            lambda authorization: setattr(
                self._action_mode,
                "_policy_gripper_authorization",
                authorization,
            )
        )
        self.actions = []
        self.attachment_enabled = []
        self.attachment_suppressed = []

    def get_observation(self):
        return self.observation

    def step(self, action):
        values = np.asarray(action, dtype=np.float64).copy()
        authorization = self._action_mode._policy_gripper_authorization
        if isinstance(authorization, dict):
            if self.bimanual:
                for arm, index in (("right", 7), ("left", 16)):
                    if authorization.get(arm) is False:
                        gripper = getattr(self._scene.robot, f"{arm}_gripper")
                        values[index] = float(
                            all(value > 0.9 for value in gripper.get_open_amount())
                        )
            elif authorization.get("single") is False:
                values[7] = float(
                    all(
                        value > 0.9
                        for value in self._scene.robot.gripper.get_open_amount()
                    )
                )
        self._action_mode._policy_gripper_authorization = None
        self.actions.append(values)
        if self.bimanual:
            self._scene.robot.right_gripper.open_amount = float(values[7] > 0.5)
            self._scene.robot.left_gripper.open_amount = float(values[16] > 0.5)
        else:
            self._scene.robot.gripper.open_amount = float(values[7] > 0.5)
        self.attachment_enabled.append(
            bool(self._gripper_action_mode._attach_grasped_objects)
        )
        self.attachment_suppressed.append(
            frozenset(
                getattr(
                    self._gripper_action_mode,
                    "_dynamac_attachment_suppressed_arms",
                    (),
                )
            )
        )
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
        "time_stall_ended",
    ]
    assert wrapped.protocol_metadata()["physical_audit"] == {
        "effect_observed": True,
        "effect_policy_step": 0,
        "fault_end_policy_step": 1,
        "target_arm": None,
        "target_objects": [],
        "relation_restored": None,
        "relation_restoration_policy_step": None,
        "cycles_to_relation_restoration": None,
    }


def test_fault_waits_for_completed_dynamic_background_without_changing_actions() -> (
    None
):
    environment = _TaskEnvironment()
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.TIME_STALL,
            duration_cycles=1,
            motion_trigger_distance=0.005,
        ),
        background_required=True,
    )
    wrapped.configure_background(scenario="smooth", expected_segments=1)
    action = np.concatenate((_pose(0.2), [0.0, 1.0]))

    wrapped.step(action)
    wrapped.record_background_event(
        {
            "kind": "smooth_task_motion",
            "applied": True,
            "complete": False,
            "protocol_effective": True,
        }
    )
    wrapped.step(action)
    wrapped.record_background_event(
        {
            "kind": "smooth_task_motion",
            "applied": True,
            "complete": True,
            "protocol_effective": True,
        }
    )
    wrapped.step(action)

    assert np.allclose(environment.actions[0], action)
    assert np.allclose(environment.actions[1], action)
    assert np.allclose(environment.actions[2][:7], _pose(0.1))
    metadata = wrapped.protocol_metadata()
    assert metadata["triggered"] is True
    assert metadata["background"]["ready"] is True
    assert metadata["background"]["ready_policy_step"] == 2
    assert metadata["background"]["ready_before_fault"] is True
    assert metadata["background"]["completed_segments"] == ["task_root"]
    assert metadata["background"]["pre_background_policy_steps"] == 2


def test_required_dynamic_background_must_be_configured_before_step() -> None:
    wrapped = FaultInjectingTaskEnvironment(
        _TaskEnvironment(),
        FaultInjectionSpec(FaultInjectionKind.TIME_STALL),
        background_required=True,
    )
    with pytest.raises(RuntimeError, match="did not configure"):
        wrapped.step(np.concatenate((_pose(0.2), [0.0, 1.0])))


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


def test_grasp_failure_suppresses_attachment_not_gripper_close_occurrence() -> None:
    item = _Object("item")
    environment = _TaskEnvironment(objects=[item])
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
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert environment.attachment_enabled == [True] * 6
    assert environment.attachment_suppressed == [
        frozenset(),
        frozenset(),
        frozenset({"single"}),
        frozenset({"single"}),
        frozenset(),
        frozenset(),
    ]
    assert environment._gripper_action_mode._attach_grasped_objects is True
    assert not getattr(
        environment._gripper_action_mode,
        "_dynamac_attachment_suppressed_arms",
        set(),
    )
    assert wrapped.events[0]["close_occurrence"] == 2
    assert wrapped.events[-1]["kind"] == "grasp_failure_occurrence_ended"


def test_bimanual_grasp_failure_suppresses_only_selected_arm() -> None:
    item = _Object("item")
    environment = _TaskEnvironment(bimanual=True, objects=[item])
    # The right gripper already owns the item; a failed left receiver close
    # must leave right-arm attachment available and must not pose-lock it.
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.GRASP_FAILURE,
            arm="left",
            close_occurrence=1,
        ),
    )
    right = np.concatenate((_pose(0.1), [0.0, 0.0]))
    left = np.concatenate((_pose(-0.1), [0.0, 0.0]))

    wrapped.step(np.concatenate((right, left)))

    assert environment.attachment_suppressed == [frozenset({"left"})]
    assert environment._scene.robot.right_gripper.get_grasped_objects() == [item]
    audit = wrapped.protocol_metadata()["physical_audit"]
    assert audit["target_objects"] == ["item"]
    assert audit["effect_observed"] is True


def test_grasp_failure_captures_detected_contact_target_outside_grasp_registry() -> (
    None
):
    tray = _Object("tray")
    environment = _TaskEnvironment(bimanual=True)
    environment._scene.robot.left_gripper._proximity_sensor.detected.add(tray)
    environment._scene.task.get_base = lambda: SimpleNamespace(
        get_objects_in_tree=lambda **_kwargs: [tray]
    )
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.GRASP_FAILURE,
            arm="left",
            close_occurrence=1,
        ),
    )
    right = np.concatenate((_pose(0.1), [1.0, 0.0]))
    left = np.concatenate((_pose(-0.1), [0.0, 0.0]))

    wrapped.step(np.concatenate((right, left)))

    audit = wrapped.protocol_metadata()["physical_audit"]
    assert audit["target_objects"] == ["tray"]
    assert audit["effect_observed"] is True


def test_grasp_occurrence_ignores_close_without_physical_target() -> None:
    item = _Object("item")
    environment = _TaskEnvironment(objects=[item])
    sensor = environment._scene.robot.gripper._proximity_sensor
    sensor.detected.clear()
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.GRASP_FAILURE,
            close_occurrence=1,
        ),
    )
    open_action = np.concatenate((_pose(0.1), [1.0, 0.0]))
    close_action = np.concatenate((_pose(0.1), [0.0, 0.0]))

    wrapped.step(close_action)
    wrapped.step(open_action)
    assert wrapped.triggered is False
    assert wrapped._close_occurrences == 0

    sensor.detected.add(item)
    wrapped.step(close_action)

    assert wrapped.triggered is True
    assert wrapped._close_occurrences == 1
    trigger = wrapped.events[0]
    assert trigger["close_occurrence"] == 1
    assert trigger["target_objects"] == ["item"]


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
    assert environment.actions[1][7] == 1.0
    assert np.allclose(item.position, expected_position)
    trigger = next(event for event in wrapped.events if event["kind"] == kind.value)
    assert trigger["released_objects"] == ["item"]
    assert trigger["displaced"] is displaced
    audit = wrapped.protocol_metadata()["physical_audit"]
    assert audit["effect_observed"] is True
    assert audit["fault_end_policy_step"] == 1
    assert audit["target_objects"] == ["item"]
    assert audit["relation_restored"] is False


def test_relation_fault_audit_observes_later_target_reattachment() -> None:
    item = _Object("item")
    environment = _TaskEnvironment(objects=[item])
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.UNEXPECTED_DROP,
            minimum_grasped_cycles=1,
        ),
    )
    action = np.concatenate((_pose(0.1), [0.0, 0.0]))

    wrapped.step(action)
    environment._scene.robot.gripper.objects = [item]
    wrapped.step(action)

    audit = wrapped.protocol_metadata()["physical_audit"]
    assert audit["effect_observed"] is True
    assert audit["relation_restored"] is True
    assert audit["relation_restoration_policy_step"] == 1
    assert audit["cycles_to_relation_restoration"] == 1
    assert wrapped.events[-1]["kind"] == "relation_restored"


@pytest.mark.parametrize(
    ("kind", "expected_position"),
    (
        (FaultInjectionKind.UNEXPECTED_DROP, [0.0, 0.0, 0.0]),
        (FaultInjectionKind.RELATION_MISMATCH, [0.04, 0.0, 0.0]),
    ),
)
def test_relation_fault_accepts_stable_closed_contact_without_attachment(
    kind, expected_position
) -> None:
    tray = _Object("tray")
    environment = _TaskEnvironment(bimanual=True)
    environment._scene.robot.right_gripper._proximity_sensor.detected.add(tray)
    environment._scene.task.get_base = lambda: SimpleNamespace(
        get_objects_in_tree=lambda **_kwargs: [tray]
    )
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            kind,
            arm="right",
            minimum_grasped_cycles=2,
        ),
    )
    right = np.concatenate((_pose(0.1), [0.0, 0.0]))
    left = np.concatenate((_pose(-0.1), [0.0, 0.0]))
    action = np.concatenate((right, left))

    wrapped.step(action)
    wrapped.step(action)

    assert wrapped.triggered is True
    assert environment.actions[1][7] == 1.0
    assert np.allclose(tray.position, expected_position)
    trigger = next(event for event in wrapped.events if event["kind"] == kind.value)
    assert trigger["interaction_source"] == "maintained_contact"
    assert trigger["stable_interaction_cycles"] == 2
    audit = wrapped.protocol_metadata()["physical_audit"]
    assert audit["effect_observed"] is True
    assert audit["target_objects"] == ["tray"]


def test_relation_fault_release_overrides_executor_gripper_hold_once() -> None:
    tray = _Object("tray")
    environment = _TaskEnvironment(bimanual=True)
    environment._scene.robot.right_gripper._proximity_sensor.detected.add(tray)
    environment._scene.task.get_base = lambda: SimpleNamespace(
        get_objects_in_tree=lambda **_kwargs: [tray]
    )
    environment._action_mode.set_policy_gripper_authorization(
        {"right": False, "left": False}
    )
    wrapped = FaultInjectingTaskEnvironment(
        environment,
        FaultInjectionSpec(
            FaultInjectionKind.UNEXPECTED_DROP,
            arm="right",
            minimum_grasped_cycles=1,
        ),
    )
    right = np.concatenate((_pose(0.1), [0.0, 0.0]))
    left = np.concatenate((_pose(-0.1), [0.0, 0.0]))

    wrapped.step(np.concatenate((right, left)))

    assert environment.actions[0][7] == 1.0
    assert environment.actions[0][16] == 1.0
    assert environment._action_mode._policy_gripper_authorization is None
    assert wrapped.protocol_metadata()["physical_audit"]["effect_observed"] is True
    assert any(
        event["kind"] == "physical_fault_gripper_authorization_override"
        for event in wrapped.events
    )


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
