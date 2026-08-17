from __future__ import annotations

import math

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import runtime as runtime_module
from integrations.rlbench.rlbench_dynamac.runtime import (
    LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    LOW_DIM_STATE_ROUNDTRIP_ATOL,
    PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
    ScenarioController,
    _low_dim_roundtrip_metrics,
)


class _Root:
    def __init__(self) -> None:
        self.pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def get_pose(self):
        return self.pose.copy()

    def set_pose(self, pose):
        self.pose = np.asarray(pose, dtype=np.float64).copy()

    def set_orientation(self, orientation):
        assert np.asarray(orientation).shape == (3,)
        # Test goals use translation only.  Resetting to the scene's initial
        # Euler orientation mirrors Scene._place_task before boundary sampling.
        self.pose[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0])


class _ConfigurationTree:
    def __init__(self, component, state) -> None:
        self.component = component
        self.state = np.asarray(state, dtype=np.float64).copy()


class _Component:
    def __init__(self, state, *, target_state=None) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.target_state = np.asarray(
            self.state if target_state is None else target_state,
            dtype=np.float64,
        ).copy()
        self.configuration_tree_reads = 0

    def get_configuration_tree(self):
        self.configuration_tree_reads += 1
        return _ConfigurationTree(self, self.state)


class _CollisionObject:
    def __init__(self, handle, name, *, collidable=True) -> None:
        self.handle = int(handle)
        self.name = str(name)
        self.collidable = bool(collidable)

    def get_handle(self):
        return self.handle

    def get_name(self):
        return self.name

    def is_collidable(self):
        return self.collidable


class _Arm(_Component):
    def __init__(
        self,
        state,
        root,
        colliding_x,
        external_object,
        *,
        target_state=None,
    ) -> None:
        super().__init__(state, target_state=target_state)
        self.root = root
        self.colliding_x = set(colliding_x)
        self.external_object = external_object
        self.collection_member_handles = frozenset({9001})

    def check_arm_collision(self, obj=None):
        colliding = float(self.root.pose[0]) in self.colliding_x
        if obj is None:
            return colliding
        return colliding and obj.get_handle() == self.external_object.get_handle()


class _TrackedObject:
    _next_handle = 1

    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.handle = _TrackedObject._next_handle
        _TrackedObject._next_handle += 1

    def get_handle(self):
        return self.handle

    def get_parent(self):
        return self.parent


class _Gripper(_Component):
    def __init__(self, state) -> None:
        super().__init__(state)
        self._grasped_objects = []
        self._old_parents = []

    def get_grasped_objects(self):
        return self._grasped_objects


class _PyRep:
    def __init__(self, objects) -> None:
        self.restore_order = []
        self.restore_trace = None
        self.objects = list(objects)

    def get_objects_in_tree(self, **_kwargs):
        return list(self.objects)

    def set_configuration_tree(self, tree):
        self.restore_order.append(tree.component)
        if self.restore_trace is not None:
            self.restore_trace.append(tree.component)
        # Model the real force-control failure that motivated motion-v4: a
        # robot configuration-tree restore can snap actual joints to targets.
        tree.component.state = tree.component.target_state.copy()


class _Task:
    def __init__(
        self,
        root,
        *,
        dynamic_drift=0.0,
        restore_pose_delta=None,
        restore_quaternion_sign_flip=False,
    ) -> None:
        self.root = root
        self.local_frames = np.asarray(
            [
                [0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 1.0],
                [-0.30, 0.40, 0.15, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._success_conditions = [object(), object()]
        self._fail_conditions = [object()]
        self.graspable = _TrackedObject(parent=root)
        self._graspable_objects = [self.graspable]
        self._waypoints = object()
        self.original_waypoints = self._waypoints
        self.validate_calls = 0
        self.low_dim_reads = 0
        self.init_episode_calls = 0
        self.restore_calls = 0
        self.dynamic_drift = float(dynamic_drift)
        self.restore_pose_delta = np.zeros(7, dtype=np.float64)
        if restore_pose_delta is not None:
            self.restore_pose_delta = np.asarray(
                restore_pose_delta,
                dtype=np.float64,
            ).copy()
        self.restore_quaternion_sign_flip = bool(restore_quaternion_sign_flip)
        self.robot = None
        self.object_count = 4
        self.restore_trace = None

    def boundary_root(self):
        return self.root

    def base_rotation_bounds(self):
        return (0.0, 0.0, -math.pi), (0.0, 0.0, math.pi)

    def get_low_dim_state(self):
        self.low_dim_reads += 1
        frames = self.local_frames.copy()
        frames[:, :3] += self.root.pose[:3]
        return frames.reshape(-1)

    def get_state(self):
        return (
            {
                "root_pose": self.root.pose.copy(),
                "local_frames": self.local_frames.copy(),
            },
            self.object_count,
        )

    def restore_state(self, state):
        payload, object_count = state
        if self.object_count != object_count:
            raise RuntimeError("task configuration-tree object count changed")
        self.restore_calls += 1
        if self.restore_trace is not None:
            self.restore_trace.append(self)
        # Deliberately bypass Root.set_pose: this models
        # PyRep.set_configuration_tree restoring the saved hierarchy exactly.
        self.root.pose = payload["root_pose"].copy()
        self.local_frames = payload["local_frames"].copy()
        self.local_frames[0] += self.restore_pose_delta
        if self.restore_quaternion_sign_flip:
            self.local_frames[:, 3:7] *= -1.0

    def validate(self):
        self.validate_calls += 1
        raise AssertionError("online sampling must not call Task.validate")

    def init_episode(self, _index):
        self.init_episode_calls += 1
        self.local_frames[:, 0] += 100.0
        self._success_conditions = [object()]
        raise AssertionError("the preserve-instance protocol called init_episode")


class _WorkspaceBoundary:
    def __init__(self, root, task, goals) -> None:
        self.root = root
        self.task = task
        self.goals = [np.asarray(goal, dtype=np.float64) for goal in goals]
        self.sample_calls = 0
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1

    def sample(self, root, *, min_rotation, max_rotation):
        assert root is self.root
        assert len(min_rotation) == len(max_rotation) == 3
        goal = self.goals[min(self.sample_calls, len(self.goals) - 1)]
        self.sample_calls += 1
        root.set_pose(goal)
        # Simulate the float32 drift caused by reset_dynamic_object() on a
        # dynamic descendant. A root-only pose rollback cannot undo this.
        self.task.local_frames[0, 0] += self.task.dynamic_drift


class _Robot:
    def __init__(self, root, external_object, colliding_x=()) -> None:
        self.root = root
        self.colliding_x = set(colliding_x)
        self.is_bimanual = False
        self.arm = _Arm(
            [1.0, 2.00017],
            root,
            colliding_x,
            external_object,
            target_state=[1.0, 2.0],
        )
        self.gripper = _Gripper([3.0, 4.0])

    def is_in_collision(self):
        return self.arm.check_arm_collision()


class _Scene:
    def __init__(
        self,
        goals,
        *,
        colliding_x=(),
        dynamic_drift=0.0,
        restore_pose_delta=None,
        restore_quaternion_sign_flip=False,
    ) -> None:
        self.root = _Root()
        self.task = _Task(
            self.root,
            dynamic_drift=dynamic_drift,
            restore_pose_delta=restore_pose_delta,
            restore_quaternion_sign_flip=restore_quaternion_sign_flip,
        )
        self._workspace_boundary = _WorkspaceBoundary(self.root, self.task, goals)
        self._initial_task_pose = np.zeros(3, dtype=np.float64)
        self.robot_member = _CollisionObject(9001, "robot_member")
        self.obstacle = _CollisionObject(9002, "obstacle")
        self.robot = _Robot(self.root, self.obstacle, colliding_x)
        self.task.robot = self.robot
        self.pyrep = _PyRep([self.robot_member, self.obstacle])
        self.restore_trace = []
        self.task.restore_trace = self.restore_trace
        self.pyrep.restore_trace = self.restore_trace

    def kidnap(self, **_kwargs):
        raise AssertionError("Scene.kidnap must not be used")

    def move_task_smoothly(self, **_kwargs):
        raise AssertionError("Scene.move_task_smoothly must not be used")


class _Environment:
    def __init__(self, scene) -> None:
        self._scene = scene

    def get_observation(self):
        raise AssertionError("ScenarioController must not trigger observation recording")


@pytest.fixture(autouse=True)
def _fake_collision_collection_members(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "_arm_collision_collection_member_handles",
        lambda arm: arm.collection_member_handles,
    )
    monkeypatch.setattr(runtime_module, "_pyrep_shape_object_type", lambda: "shape")


def _goal(x=1.0, y=0.5):
    return np.asarray([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])


def _root_local_positions(task):
    state = task.get_low_dim_state().reshape(-1, 7)
    return state[:, :3] - task.root.pose[:3]


def _pose_state():
    return np.asarray(
        [
            0.1,
            -0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.4,
            0.5,
            0.6,
            0.0,
            0.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
        ],
        dtype=np.float64,
    )


def test_low_dim_pose_comparison_is_quaternion_sign_invariant() -> None:
    source = _pose_state()
    restored = source.copy().reshape(-1, 7)
    restored[:, 3:7] *= -1.0

    metrics = _low_dim_roundtrip_metrics(source, restored.reshape(-1))

    assert metrics["preserved"] is True
    assert metrics["comparison_mode"] == "pose_chunks_sign_invariant"
    assert metrics["chunk_count"] == 2
    assert metrics["max_translation_m"] == 0.0
    assert metrics["max_rotation_rad"] == 0.0
    assert metrics["raw_max_abs"] == 2.0


def test_low_dim_pose_comparison_enforces_physical_error_boundaries() -> None:
    source = _pose_state()
    inside_translation_limit = source.copy()
    inside_translation_limit[0] += 0.5 * LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
    inside_rotation_limit = source.copy()
    angle = 0.5 * LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
    inside_rotation_limit[3:7] = [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]

    assert _low_dim_roundtrip_metrics(source, inside_translation_limit)[
        "preserved"
    ] is True
    assert _low_dim_roundtrip_metrics(source, inside_rotation_limit)[
        "preserved"
    ] is True

    outside_translation = source.copy()
    outside_translation[0] += 2.0 * LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
    outside_rotation = source.copy()
    angle = 2.0 * LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
    outside_rotation[3:7] = [
        0.0,
        0.0,
        math.sin(angle / 2.0),
        math.cos(angle / 2.0),
    ]
    assert _low_dim_roundtrip_metrics(source, outside_translation)[
        "preserved"
    ] is False
    assert _low_dim_roundtrip_metrics(source, outside_rotation)[
        "preserved"
    ] is False


def test_low_dim_non_pose_state_uses_explicit_scalar_fallback() -> None:
    source = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    restored = source.copy()
    restored[1] += 0.5 * LOW_DIM_STATE_ROUNDTRIP_ATOL

    metrics = _low_dim_roundtrip_metrics(source, restored)

    assert metrics["preserved"] is True
    assert metrics["comparison_mode"] == "scalar_max_abs"
    assert metrics["chunk_count"] == 0
    assert metrics["max_translation_m"] is None
    assert metrics["max_rotation_rad"] is None

    restored[1] += 2.0 * LOW_DIM_STATE_ROUNDTRIP_ATOL
    assert _low_dim_roundtrip_metrics(source, restored)["preserved"] is False


def test_teleport_samples_root_without_reinitializing_episode_instance() -> None:
    scene = _Scene([_goal()], dynamic_drift=1.0e-4)
    environment = _Environment(scene)
    task = scene.task
    success_conditions = tuple(task._success_conditions)
    local_positions = _root_local_positions(task)
    arm_state = scene.robot.arm.state.copy()
    gripper_state = scene.robot.gripper.state.copy()
    controller = ScenarioController(
        "teleport_task",
        trigger_fraction=0.0,
        max_attempts=3,
    )

    event = controller.apply(environment, step=0, horizon=10)

    np.testing.assert_allclose(scene.root.pose, _goal(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        _root_local_positions(task), local_positions, rtol=0.0, atol=1.0e-12
    )
    assert task.init_episode_calls == 0
    assert task.validate_calls == 0
    assert task.restore_calls == 2
    assert scene.restore_trace == [task, task]
    assert scene.robot.arm.configuration_tree_reads == 0
    assert scene.robot.gripper.configuration_tree_reads == 0
    assert scene.pyrep.restore_order == []
    np.testing.assert_array_equal(scene.robot.arm.state, arm_state)
    np.testing.assert_array_equal(scene.robot.gripper.state, gripper_state)
    assert task._waypoints is task.original_waypoints
    assert all(
        current is original
        for current, original in zip(task._success_conditions, success_conditions)
    )
    assert event["applied"] is True
    assert event["protocol_effective"] is True
    assert event["planned_root_translation_m"] > 0.0
    assert event["instance_preservation"] == {
        "initialized_episode_preserved": True,
        "task_init_episode_called": False,
        "task_validate_called": False,
        "low_dim_state_roundtrip_preserved": True,
        "low_dim_state_roundtrip_comparison_mode": (
            "pose_chunks_sign_invariant"
        ),
        "low_dim_state_roundtrip_chunk_count": 2,
        "low_dim_state_roundtrip_l2": 0.0,
        "low_dim_state_roundtrip_max_abs": 0.0,
        "low_dim_state_roundtrip_max_translation_m": 0.0,
        "low_dim_state_roundtrip_max_rotation_rad": 0.0,
        "condition_and_grasp_registry_identity_preserved": True,
        "gripper_grasp_membership_and_parentage_preserved": True,
        "configuration_tree_rollback": (
            "task_only_after_each_attempt_and_outer_finally"
        ),
        "task_configuration_tree_restored": True,
        "live_robot_state_untouched": True,
        "live_robot_configuration_trees_accessed": False,
        "robot_collision_pair_policy": (
            "reject_candidate_external_pairs_absent_at_source"
        ),
        "robot_collision_pair_granularity": (
            "named_arm_collection_x_external_collidable_scene_shape"
        ),
        "source_robot_external_collision_pairs": [],
        "goal_robot_external_collision_pairs": [],
        "goal_new_robot_external_collision_pairs": [],
        "sampling_attempts_rejected_for_new_robot_collision_pairs": 0,
        "sampling_attempts": 1,
        "waypoint_cache_identity_preserved": True,
    }
    assert event["motion_protocol"]["protocol_id"] == (
        PRESERVE_INSTANCE_MOTION_PROTOCOL_ID
    )
    assert event["motion_protocol"]["episode_instance_semantics"] == (
        "preserve_initialized_episode"
    )
    assert event["motion_protocol"]["calls_scene_kidnap"] is False
    assert event["motion_protocol"]["calls_scene_move_task_smoothly"] is False
    assert event["motion_protocol"]["task_configuration_tree_restore_api"] == (
        "Task.get_state/restore_state"
    )
    assert event["motion_protocol"]["sampling_rollback"] == (
        "task_configuration_tree_only_live_robot_untouched"
    )
    assert event["motion_protocol"]["live_robot_state_during_goal_sampling"] == (
        "untouched"
    )
    assert event["motion_protocol"]["live_robot_configuration_tree_access"] == (
        "none"
    )
    assert event["motion_protocol"]["calls_task_validate"] is False
    assert event["motion_protocol"]["robot_collision_validation"] == (
        "reject_candidate_external_pairs_absent_at_source"
    )
    assert event["actual_root_motion"] is True
    assert event["commanded_root_pose_reached"] is True
    assert event["goal_root_pose_reached"] is True
    assert event["commanded_root_translation_residual_m"] == 0.0
    assert event["commanded_root_rotation_residual_rad"] == 0.0


def test_sampling_accepts_float_quantized_pose_roundtrip_with_raw_evidence() -> None:
    pose_delta = np.zeros(7, dtype=np.float64)
    pose_delta[0] = 0.5 * LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
    scene = _Scene([_goal()], restore_pose_delta=pose_delta)

    event = ScenarioController("teleport_task", trigger_fraction=0.0).apply(
        _Environment(scene),
        step=0,
        horizon=1,
    )

    evidence = event["instance_preservation"]
    assert evidence["low_dim_state_roundtrip_preserved"] is True
    assert evidence["low_dim_state_roundtrip_comparison_mode"] == (
        "pose_chunks_sign_invariant"
    )
    assert evidence["low_dim_state_roundtrip_max_abs"] > 0.0
    assert evidence["low_dim_state_roundtrip_max_translation_m"] == pytest.approx(
        0.5 * LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    )


def test_sampling_rejects_real_pose_change_beyond_physical_tolerance() -> None:
    pose_delta = np.zeros(7, dtype=np.float64)
    pose_delta[0] = 2.0 * LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
    scene = _Scene([_goal()], restore_pose_delta=pose_delta)

    with pytest.raises(
        RuntimeError,
        match="low-dimensional state beyond pose_chunks_sign_invariant tolerance",
    ):
        ScenarioController("teleport_task", trigger_fraction=0.0).apply(
            _Environment(scene),
            step=0,
            horizon=1,
        )


def test_teleport_rejects_collision_and_preserves_only_one_sampled_goal() -> None:
    scene = _Scene(
        [_goal(0.25), _goal(0.75)],
        colliding_x=(0.25,),
        dynamic_drift=1.0e-4,
    )
    controller = ScenarioController(
        "teleport_task",
        trigger_fraction=0.0,
        max_attempts=2,
    )

    event = controller.apply(_Environment(scene), step=0, horizon=1)

    assert scene._workspace_boundary.sample_calls == 2
    assert event["instance_preservation"]["sampling_attempts"] == 2
    assert event["instance_preservation"][
        "sampling_attempts_rejected_for_new_robot_collision_pairs"
    ] == 1
    assert event["instance_preservation"][
        "goal_new_robot_external_collision_pairs"
    ] == []
    assert scene.task.restore_calls == 3
    np.testing.assert_allclose(scene.root.pose, _goal(0.75), rtol=0.0, atol=0.0)
    assert scene.task.init_episode_calls == 0


def test_existing_source_tool_contact_does_not_reject_unchanged_pair() -> None:
    scene = _Scene(
        [_goal(0.75)],
        colliding_x=(0.0, 0.75),
        dynamic_drift=1.0e-4,
    )

    event = ScenarioController(
        "teleport_task",
        trigger_fraction=0.0,
        max_attempts=1,
    ).apply(_Environment(scene), step=0, horizon=1)

    evidence = event["instance_preservation"]
    expected_pair = [
        {
            "arm": "arm",
            "external_object_handle": scene.obstacle.get_handle(),
            "external_object_name": scene.obstacle.get_name(),
        }
    ]
    assert evidence["source_robot_external_collision_pairs"] == expected_pair
    assert evidence["goal_robot_external_collision_pairs"] == expected_pair
    assert evidence["goal_new_robot_external_collision_pairs"] == []
    assert evidence[
        "sampling_attempts_rejected_for_new_robot_collision_pairs"
    ] == 0
    assert scene._workspace_boundary.sample_calls == 1


def test_smooth_motion_uses_one_goal_and_reaches_exact_endpoint() -> None:
    goal = _goal(0.9, -0.3)
    scene = _Scene([goal])
    environment = _Environment(scene)
    task = scene.task
    local_positions = _root_local_positions(task)
    controller = ScenarioController(
        "smooth_task_motion",
        trigger_fraction=0.0,
        total_steps=3,
    )

    first = controller.apply(environment, step=0, horizon=5)
    np.testing.assert_allclose(scene.root.pose[:3], goal[:3] / 3.0)
    second = controller.apply(environment, step=1, horizon=5)
    np.testing.assert_allclose(scene.root.pose[:3], 2.0 * goal[:3] / 3.0)
    third = controller.apply(environment, step=2, horizon=5)

    assert first["endpoint_fraction"] == 1.0 / 3.0
    assert second["endpoint_fraction"] == 2.0 / 3.0
    assert third["endpoint_fraction"] == 1.0
    assert third["complete"] is True
    assert third["endpoint_applied"] is True
    assert first["actual_root_motion"] is True
    assert first["commanded_root_pose_reached"] is True
    assert first["goal_root_pose_reached"] is False
    assert third["goal_root_pose_reached"] is True
    assert third["protocol_effective"] is True
    np.testing.assert_array_equal(scene.root.pose, goal)
    np.testing.assert_allclose(
        _root_local_positions(task), local_positions, rtol=0.0, atol=1.0e-12
    )
    assert scene._workspace_boundary.sample_calls == 1
    assert task.init_episode_calls == 0


def test_teleport_is_not_effective_when_commanded_pose_is_not_reached() -> None:
    scene = _Scene([_goal()])
    original_set_pose = scene.root.set_pose
    calls = [0]

    def ignore_application_after_sampling(pose):
        calls[0] += 1
        if calls[0] == 1:
            original_set_pose(pose)

    scene.root.set_pose = ignore_application_after_sampling

    event = ScenarioController("teleport_task", trigger_fraction=0.0).apply(
        _Environment(scene), step=0, horizon=1
    )

    assert calls[0] == 2
    assert event["planned_root_motion"] is True
    assert event["actual_root_motion"] is False
    assert event["commanded_root_pose_reached"] is False
    assert event["goal_root_pose_reached"] is False
    assert event["protocol_effective"] is False


def test_smooth_endpoint_requires_actual_final_goal_pose() -> None:
    scene = _Scene([_goal(0.9, -0.3)])
    original_set_pose = scene.root.set_pose
    calls = [0]

    def drop_only_final_endpoint_command(pose):
        calls[0] += 1
        # One sampling call, then three scheduled smooth commands.
        if calls[0] != 4:
            original_set_pose(pose)

    scene.root.set_pose = drop_only_final_endpoint_command
    controller = ScenarioController(
        "smooth_task_motion",
        trigger_fraction=0.0,
        total_steps=3,
    )

    controller.apply(_Environment(scene), step=0, horizon=5)
    controller.apply(_Environment(scene), step=1, horizon=5)
    endpoint = controller.apply(_Environment(scene), step=2, horizon=5)

    assert endpoint["complete"] is True
    assert endpoint["endpoint_applied"] is False
    assert endpoint["actual_root_motion"] is False
    assert endpoint["commanded_root_pose_reached"] is False
    assert endpoint["goal_root_pose_reached"] is False
    assert endpoint["protocol_effective"] is False


def test_completed_motion_is_a_true_noop_without_extra_state_reads() -> None:
    scene = _Scene([_goal()])
    environment = _Environment(scene)
    controller = ScenarioController("teleport_task", trigger_fraction=0.0)
    controller.apply(environment, step=0, horizon=2)
    reads_after_teleport = scene.task.low_dim_reads

    event = controller.apply(environment, step=1, horizon=2)

    assert event["applied"] is False
    assert scene.task.low_dim_reads == reads_after_teleport
    assert scene._workspace_boundary.sample_calls == 1


def test_sampling_error_restores_task_without_touching_live_robot() -> None:
    scene = _Scene([_goal()], dynamic_drift=1.0e-4)
    task = scene.task
    source_pose = scene.root.pose.copy()
    source_state = task.get_low_dim_state().copy()
    source_arm = scene.robot.arm.state.copy()
    source_gripper = scene.robot.gripper.state.copy()
    source_waypoints = task._waypoints
    original_sample = scene._workspace_boundary.sample

    def sample_then_fail(*args, **kwargs):
        original_sample(*args, **kwargs)
        raise ValueError("diagnostic sampling failure")

    scene._workspace_boundary.sample = sample_then_fail

    with pytest.raises(ValueError, match="diagnostic sampling failure"):
        ScenarioController("teleport_task", trigger_fraction=0.0).apply(
            _Environment(scene), step=0, horizon=1
        )

    np.testing.assert_array_equal(scene.root.pose, source_pose)
    np.testing.assert_array_equal(task.get_low_dim_state(), source_state)
    np.testing.assert_array_equal(scene.robot.arm.state, source_arm)
    np.testing.assert_array_equal(scene.robot.gripper.state, source_gripper)
    assert task._waypoints is source_waypoints
    assert task.restore_calls == 2
    assert task.validate_calls == 0
    assert scene.robot.arm.configuration_tree_reads == 0
    assert scene.robot.gripper.configuration_tree_reads == 0
    assert scene.pyrep.restore_order == []


def test_grasp_membership_and_parentage_survive_sampling_transaction() -> None:
    scene = _Scene([_goal()])
    task = scene.task
    grasped = task.graspable
    old_parent = scene.root
    grasped.parent = scene.robot.gripper
    scene.robot.gripper._grasped_objects = [grasped]
    scene.robot.gripper._old_parents = [old_parent]

    event = ScenarioController("teleport_task", trigger_fraction=0.0).apply(
        _Environment(scene), step=0, horizon=1
    )

    assert scene.robot.gripper.get_grasped_objects() == [grasped]
    assert scene.robot.gripper._old_parents == [old_parent]
    assert grasped.get_parent() is scene.robot.gripper
    assert event["instance_preservation"][
        "gripper_grasp_membership_and_parentage_preserved"
    ] is True


def test_parent_change_during_sampling_fails_closed_after_restoration() -> None:
    scene = _Scene([_goal()])
    task = scene.task
    original_sample = scene._workspace_boundary.sample

    def sample_and_reparent(*args, **kwargs):
        original_sample(*args, **kwargs)
        task.graspable.parent = _TrackedObject()

    scene._workspace_boundary.sample = sample_and_reparent
    source_pose = scene.root.pose.copy()
    source_state = task.get_low_dim_state().copy()

    with pytest.raises(
        RuntimeError,
        match="changed gripper grasp membership or object parents",
    ):
        ScenarioController("teleport_task", trigger_fraction=0.0).apply(
            _Environment(scene), step=0, horizon=1
        )

    np.testing.assert_array_equal(scene.root.pose, source_pose)
    np.testing.assert_array_equal(task.get_low_dim_state(), source_state)
