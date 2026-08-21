from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.core.trac_ik import (
    AlignedTracIKDistanceSolver,
    TracIKDistanceConfig,
    _live_panda_chain_urdf,
)


class _Tip:
    def __init__(self, arm):
        self.arm = arm

    def get_matrix(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = self.arm.current[:3]
        return matrix


class _Arm:
    def __init__(self):
        self.current = np.zeros(7, dtype=np.float64)
        self.tip = _Tip(self)

    def get_joint_positions(self):
        return self.current.copy()

    def get_joint_intervals(self):
        return [False] * 7, [[-1.0, 2.0] for _ in range(7)]

    def get_tip(self):
        return self.tip


class _BoundedSolver:
    def __init__(self, candidate):
        self.candidate = candidate
        self.joint_limits = None
        self.calls = []

    def fk(self, q):
        position = np.zeros(3, dtype=np.float64)
        position[:] = np.asarray(q, dtype=np.float64)[:3]
        return position, np.eye(3, dtype=np.float64)

    def ik_with_bounds(self, position, rotation, seed_jnt_values, bounds):
        self.calls.append((position.copy(), rotation.copy(), seed_jnt_values.copy(), bounds.copy()))
        return None if self.candidate is None else self.candidate.copy()


class _UnboundedSolver:
    def __init__(self):
        self.joint_limits = None

    def fk(self, q):
        del q
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    def ik(self, position, rotation, seed_jnt_values):
        del position, rotation
        return np.asarray(seed_jnt_values, dtype=np.float64)


def _factory_for(solver):
    def factory(**kwargs):
        assert kwargs["solver_type"] == "Distance"
        assert kwargs["base_link_name"] == "robot_base"
        assert kwargs["tip_link_name"] == "Pandatip"
        return solver

    return factory


def _target(x):
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_aligned_trac_ik_uses_current_seed_and_fixed_cartesian_bounds(tmp_path):
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    arm = _Arm()
    candidate = np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    native = _BoundedSolver(candidate)
    config = TracIKDistanceConfig()
    solver = AlignedTracIKDistanceSolver(
        arm,
        config=config,
        urdf_path=urdf,
        factory=_factory_for(native),
    )

    result = solver.solve(_target(0.1))

    assert result is not None
    assert result.bounded_cartesian_api_used is True
    assert result.joint_delta_l2_rad == pytest.approx(0.1)
    assert result.fk_translation_error_m == pytest.approx(0.0)
    assert len(native.calls) == 1
    _position, _rotation, seed, bounds = native.calls[0]
    assert seed == pytest.approx(np.zeros(7))
    assert bounds[:3] == pytest.approx([0.001] * 3)
    assert bounds[3:] == pytest.approx([math.radians(1.0)] * 3)
    lower, upper = native.joint_limits
    assert lower == pytest.approx([-0.35] * 7)
    assert upper == pytest.approx([0.35] * 7)


def test_aligned_trac_ik_rejects_remote_branch(tmp_path):
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    arm = _Arm()
    native = _BoundedSolver(np.asarray([0.3] * 7, dtype=np.float64))
    solver = AlignedTracIKDistanceSolver(
        arm,
        config=TracIKDistanceConfig(),
        urdf_path=urdf,
        factory=_factory_for(native),
    )

    assert solver.solve(_target(0.3)) is None


def test_aligned_trac_ik_requires_bounded_cartesian_api(tmp_path):
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    arm = _Arm()

    with pytest.raises(RuntimeError, match="requires pytracik.ik_with_bounds"):
        AlignedTracIKDistanceSolver(
            arm,
            config=TracIKDistanceConfig(),
            urdf_path=urdf,
            factory=_factory_for(_UnboundedSolver()),
        )


def test_aligned_trac_ik_fails_closed_when_live_chain_drifts(tmp_path):
    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    arm = _Arm()
    native = _BoundedSolver(np.zeros(7, dtype=np.float64))
    solver = AlignedTracIKDistanceSolver(
        arm,
        config=TracIKDistanceConfig(),
        urdf_path=urdf,
        factory=_factory_for(native),
    )
    arm.current[0] = 0.01
    # Deliberately make the live tip disagree with the external FK model.
    arm.tip.get_matrix = lambda: np.eye(4, dtype=np.float64)

    with pytest.raises(RuntimeError, match="no longer matches"):
        solver.solve(_target(0.0))


class _ChainNode:
    def __init__(self, world=None):
        self.world = np.eye(4) if world is None else np.asarray(world)
        self.relative = {}

    def get_matrix(self, relative_to=None):
        if relative_to is None:
            return self.world.copy()
        return self.relative[id(relative_to)].copy()


class _LiveChainArm:
    def __init__(self, joints, tip):
        self.joints = joints
        self.tip = tip

    def get_tip(self):
        return self.tip


class _LiveSolverArm(_LiveChainArm):
    def __init__(self):
        joints = [_ChainNode() for _ in range(7)]
        tip = _ChainNode()
        for index, joint in enumerate(joints):
            downstream = joints[index + 1] if index < 6 else tip
            downstream.relative[id(joint)] = np.eye(4)
        super().__init__(joints, tip)
        self.current = np.zeros(7, dtype=np.float64)

    def get_joint_positions(self):
        return self.current.copy()

    def get_joint_intervals(self):
        return [False] * 7, [[-1.0, 2.0] for _ in range(7)]


def _translated(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_live_chain_uses_coppelia_moving_frame_segments_directly():
    base = _translated(0.4, -0.2, 1.0)
    joints = [_ChainNode(base if index == 0 else None) for index in range(7)]
    tip = _ChainNode()
    expected_segments = []
    for index, joint in enumerate(joints):
        segment = _translated(0.01 * (index + 1), 0.02, 0.03)
        downstream = joints[index + 1] if index < 6 else tip
        downstream.relative[id(joint)] = segment
        expected_segments.append(segment)
    arm = _LiveChainArm(joints, tip)

    urdf, world_from_root = _live_panda_chain_urdf(arm, -np.ones(7), np.ones(7))

    np.testing.assert_allclose(world_from_root, base)
    robot = ET.fromstring(urdf)
    urdf_joints = {node.attrib["name"]: node for node in robot.findall("joint")}
    np.testing.assert_allclose(
        np.fromstring(urdf_joints["joint1"].find("origin").attrib["xyz"], sep=" "),
        np.zeros(3),
    )
    for index in range(1, 7):
        origin = urdf_joints[f"joint{index + 1}"].find("origin")
        np.testing.assert_allclose(
            np.fromstring(origin.attrib["xyz"], sep=" "),
            expected_segments[index - 1][:3, 3],
        )
    tip_origin = urdf_joints["tip_fixed"].find("origin")
    np.testing.assert_allclose(
        np.fromstring(tip_origin.attrib["xyz"], sep=" "),
        expected_segments[-1][:3, 3],
    )


def test_default_live_adapter_loads_temporary_exact_chain_synchronously():
    arm = _LiveSolverArm()
    native = _BoundedSolver(np.zeros(7, dtype=np.float64))
    captured = {}

    def factory(**kwargs):
        path = kwargs["urdf_path"]
        captured["path"] = path
        captured["urdf"] = open(path, encoding="utf-8").read()
        assert kwargs["base_link_name"] == "base"
        assert kwargs["tip_link_name"] == "tip"
        assert kwargs["solver_type"] == "Distance"
        return native

    solver = AlignedTracIKDistanceSolver(
        arm,
        config=TracIKDistanceConfig(),
        factory=factory,
    )

    assert solver.chain_source == "live_coppeliasim_moving_frame_segments"
    assert '<robot name="rlbench_live_panda">' in captured["urdf"]
    assert not Path(captured["path"]).exists()
    result = solver.solve(_target(0.0))
    assert result is not None
    np.testing.assert_allclose(result.joints, np.zeros(7))
