from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.runtime import (
    DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY,
    DiscreteGripperProtocol,
    make_discrete_gripper_action_mode,
)


class _VendorDiscrete:
    def __init__(
        self,
        attach_grasped_objects: bool = True,
        detach_before_open: bool = True,
    ) -> None:
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open


class _VendorBimanualDiscrete(_VendorDiscrete):
    pass


@pytest.fixture
def fake_vendor_gripper_modes(monkeypatch):
    rlbench = ModuleType("rlbench")
    action_modes = ModuleType("rlbench.action_modes")
    gripper_modes = ModuleType("rlbench.action_modes.gripper_action_modes")
    gripper_modes.Discrete = _VendorDiscrete
    gripper_modes.BimanualDiscrete = _VendorBimanualDiscrete
    rlbench.action_modes = action_modes
    action_modes.gripper_action_modes = gripper_modes
    monkeypatch.setitem(sys.modules, "rlbench", rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes", action_modes)
    monkeypatch.setitem(
        sys.modules,
        "rlbench.action_modes.gripper_action_modes",
        gripper_modes,
    )


class _Stepper:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _Gripper:
    def __init__(self, completion_sequence) -> None:
        self._completion = iter(completion_sequence)
        self.calls = []

    def actuate(self, action, *, velocity):
        self.calls.append((float(action), float(velocity)))
        return next(self._completion)


def _scene(*, gripper=None, right_gripper=None, left_gripper=None):
    robot = SimpleNamespace(
        gripper=gripper,
        right_gripper=right_gripper,
        left_gripper=left_gripper,
    )
    return SimpleNamespace(robot=robot, pyrep=_Stepper(), task=_Stepper())


def test_default_protocol_is_demo_aligned_and_has_stable_identity() -> None:
    protocol = DiscreteGripperProtocol(bimanual=True)

    assert protocol.actuation_velocity == DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY
    assert protocol.protocol_id == (
        "rlbench-discrete-gripper-bimanual-velocity0p04"
        "-attach1-detach-before-open1-v1"
    )
    assert protocol.extend_evaluation_protocol_id("absolute-ee-v3") == (
        "absolute-ee-v3+rlbench-discrete-gripper-bimanual-velocity0p04"
        "-attach1-detach-before-open1-v1"
    )
    assert protocol.extend_evaluation_protocol_id(
        protocol.extend_evaluation_protocol_id("absolute-ee-v3")
    ) == protocol.extend_evaluation_protocol_id("absolute-ee-v3")
    assert protocol.metadata() == {
        "protocol_id": protocol.protocol_id,
        "action_mode": "BimanualDiscrete",
        "arm_layout": "bimanual",
        "actuation_velocity": 0.04,
        "demonstration_actuation_velocity": 0.04,
        "velocity_aligned_with_demonstrations": True,
        "attach_grasped_objects": True,
        "detach_before_open": True,
        "implementation": "project_subclass_preserving_vendor_action_semantics",
    }


@pytest.mark.parametrize("velocity", [0.0, -0.01, np.inf, -np.inf, np.nan])
def test_protocol_rejects_non_positive_or_non_finite_velocity(velocity) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        DiscreteGripperProtocol(bimanual=False, actuation_velocity=velocity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bimanual", 1),
        ("attach_grasped_objects", 1),
        ("detach_before_open", 0),
    ],
)
def test_protocol_requires_boolean_switches(field, value) -> None:
    kwargs = {"bimanual": False, field: value}
    with pytest.raises(TypeError, match=f"{field} must be a bool"):
        DiscreteGripperProtocol(**kwargs)


def test_unimanual_mode_uses_configured_velocity_and_vendor_constructor(
    fake_vendor_gripper_modes,
) -> None:
    mode = make_discrete_gripper_action_mode(
        bimanual=False,
        attach_grasped_objects=False,
        detach_before_open=False,
    )
    gripper = _Gripper([False, True])
    scene = _scene(gripper=gripper)

    mode._actuate(scene, 0.0)

    assert isinstance(mode, _VendorDiscrete)
    assert mode._attach_grasped_objects is False
    assert mode._detach_before_open is False
    assert mode.dynamac_protocol.bimanual is False
    assert gripper.calls == [(0.0, 0.04), (0.0, 0.04)]
    assert scene.pyrep.steps == 2
    assert scene.task.steps == 2


def test_bimanual_mode_stops_each_arm_independently_at_custom_velocity(
    fake_vendor_gripper_modes,
) -> None:
    protocol = DiscreteGripperProtocol(
        bimanual=True,
        actuation_velocity=0.075,
    )
    mode = protocol.make_action_mode()
    right = _Gripper([True])
    left = _Gripper([False, True])
    scene = _scene(right_gripper=right, left_gripper=left)

    mode._actuate(scene, np.asarray([1.0, 0.0]))

    assert isinstance(mode, _VendorBimanualDiscrete)
    assert mode.dynamac_protocol is protocol
    assert right.calls == [(1.0, 0.075)]
    assert left.calls == [(0.0, 0.075), (0.0, 0.075)]
    assert scene.pyrep.steps == 2
    assert scene.task.steps == 2
    assert protocol.metadata()["velocity_aligned_with_demonstrations"] is False
