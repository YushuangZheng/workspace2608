from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import runtime as runtime_module
from integrations.rlbench.rlbench_dynamac.runtime import (
    CROSS_INITIALIZATION_JOINT_TOLERANCE,
    CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
    CROSS_INITIALIZATION_SCALAR_TOLERANCE,
    CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
    FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
    FRESH_TASK_GENERATION_PROTOCOL_ID,
    LOW_DIM_POSE_ROTATION_TOLERANCE_RAD,
    LOW_DIM_POSE_TRANSLATION_TOLERANCE_M,
    LOW_DIM_STATE_ROUNDTRIP_ATOL,
    PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
    STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
    TASK_TREE_STATE_SCHEMA,
    ScenarioController,
    StagedMotionPlan,
    _canonical_json_fingerprint,
    _compare_task_tree_relative_state,
    _low_dim_roundtrip_metrics,
    _quaternion_angle_xyzw,
    _stable_collision_pair_records,
    _task_semantic_signature,
    _task_tree_relative_state,
    initialize_fresh_task_generation,
    load_staged_motion_plan_batch,
    stage_scenario_motion_plan,
    staged_motion_plan_batch,
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


class _FrameAwareTreeNode:
    def __init__(self, name, position, *, parent=None, object_type="dummy"):
        self._name = name
        self._type = object_type
        self._parent = parent
        self._children = []
        self._pose = np.asarray(
            [*position, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        if parent is not None:
            parent._children.append(self)

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_parent(self):
        return self._parent

    def get_pose(self, relative_to=None):
        if relative_to is None:
            return self._pose.copy()
        pose = self._pose.copy()
        pose[:3] -= relative_to._pose[:3]
        return pose

    def get_velocity(self):
        return np.zeros(3), np.zeros(3)

    def set_orientation(self, orientation):
        assert np.asarray(orientation).shape == (3,)
        self._pose[3:7] = [0.0, 0.0, 0.0, 1.0]

    def set_pose(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        delta = pose[:3] - self._pose[:3]
        for value in self.get_objects_in_tree(exclude_base=False):
            value._pose[:3] += delta
        self._pose[3:7] = pose[3:7]

    def get_objects_in_tree(self, *, exclude_base=False):
        values = [] if exclude_base else [self]
        for child in self._children:
            values.extend(child.get_objects_in_tree(exclude_base=False))
        return values


class _FrameAwareJoint(_FrameAwareTreeNode):
    def __init__(self, name, position, joint_position, *, parent):
        super().__init__(name, position, parent=parent, object_type="joint")
        self._joint_position = float(joint_position)

    def get_joint_position(self):
        return self._joint_position


class _FrameAwareTask:
    def __init__(
        self,
        *,
        root_x,
        external_x=5.0,
        descendant_local_x=1.0,
        joint_position=0.0,
    ):
        self.base = _FrameAwareTreeNode("task_base", (0.0, 0.0, 0.0))
        self.root = _FrameAwareTreeNode(
            "boundary_root",
            (root_x, 0.0, 0.0),
            parent=self.base,
            object_type="shape",
        )
        self.descendant = _FrameAwareJoint(
            "task_joint",
            (root_x + descendant_local_x, 0.0, 0.0),
            joint_position,
            parent=self.root,
        )
        self.external = _FrameAwareTreeNode(
            "task_metadata_anchor",
            (external_x, 0.0, 0.0),
            parent=self.base,
        )

    def get_base(self):
        return self.base

    def boundary_root(self):
        return self.root


class WaypointError(Exception):
    pass


class _PerAttemptSourceTask:
    def __init__(self, *, reject_candidate=True):
        self.base = _FrameAwareTreeNode("task_base", (0.0, 0.0, 0.0))
        self.root = _FrameAwareTreeNode(
            "stack_wine",
            (0.0, 0.0, 0.0),
            parent=self.base,
            object_type=type("ObjectType", (), {"name": "DUMMY"})(),
        )
        self.frame = _FrameAwareJoint(
            "wine_bottle",
            (1.0, 0.0, 0.0),
            0.0,
            parent=self.root,
        )
        self.success_sensor = _FrameAwareTreeNode(
            "success_sensor",
            (2.0, 0.0, 0.0),
            parent=self.root,
        )
        self.external = _FrameAwareTreeNode(
            "task_metadata_anchor",
            (5.0, 0.0, 0.0),
            parent=self.base,
        )
        self._success_conditions = ["success"]
        self._fail_conditions = []
        self._graspable_objects = []
        self._waypoints = None
        self.candidate_validation_calls = 0
        self.reject_candidate = reject_candidate

    def get_name(self):
        return "stack_wine"

    def get_base(self):
        return self.base

    def boundary_root(self):
        return self.root

    def base_rotation_bounds(self):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self):
        return True

    def get_low_dim_state(self):
        return np.concatenate(
            (self.frame.get_pose(), self.success_sensor.get_pose())
        )

    def reset_source(self, drift):
        self._waypoints = None
        self.root._pose = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.frame._pose = np.asarray(
            [1.0 + drift, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.success_sensor._pose = np.asarray(
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    def move_root_to(self, x):
        delta = float(x) - float(self.root._pose[0])
        for value in self.root.get_objects_in_tree(exclude_base=False):
            value._pose[0] += delta

    def validate(self):
        self.candidate_validation_calls += 1
        if self.reject_candidate and self.candidate_validation_calls == 1:
            raise WaypointError("first candidate is unreachable")
        self._waypoints = [type("Waypoint", (), {"name": "waypoint0"})()]


class _PerAttemptWorkspace:
    def __init__(self, task):
        self.task = task
        self._contained_objects = []

    def clear(self):
        self._contained_objects.clear()

    def sample(self, root, *, min_rotation, max_rotation):
        assert root is self.task.root
        assert len(min_rotation) == len(max_rotation) == 3
        self.task.move_root_to(0.1)
        self._contained_objects.append(root)


class _PerAttemptScene:
    def __init__(self, task):
        self.task = task
        self.robot = type(
            "PerAttemptRobot",
            (),
            {
                "is_bimanual": False,
                "is_in_collision": lambda self: False,
                "arm": type(
                    "PerAttemptArm",
                    (),
                    {
                        "get_joint_positions": lambda self: np.zeros(7),
                        "get_joint_target_positions": lambda self: np.zeros(7),
                        "get_joint_velocities": lambda self: np.zeros(7),
                        "get_tip": lambda self: type(
                            "Tip",
                            (),
                            {
                                "get_pose": lambda self: np.asarray(
                                    [0, 0, 0, 0, 0, 0, 1], dtype=float
                                )
                            },
                        )(),
                    },
                )(),
                "gripper": type(
                    "PerAttemptGripper",
                    (),
                    {
                        "_old_parents": (),
                        "get_grasped_objects": lambda self: (),
                        "get_joint_positions": lambda self: np.zeros(2),
                        "get_joint_target_positions": lambda self: np.zeros(2),
                        "get_joint_velocities": lambda self: np.zeros(2),
                    },
                )(),
            },
        )()
        self._workspace_boundary = _PerAttemptWorkspace(task)
        self._initial_task_pose = np.zeros(3)

    def unload(self):
        self.task = None


class _PerAttemptTaskEnvironment:
    def __init__(self, drifts, *, reject_candidate=True):
        self.task = _PerAttemptSourceTask(reject_candidate=reject_candidate)
        self._scene = _PerAttemptScene(self.task)
        self.drifts = list(drifts)
        self.reset_calls = 0
        self.variation = None

    def set_variation(self, variation):
        self.variation = variation

    def reset(self, *, verify_instance=True):
        assert verify_instance is False
        drift = self.drifts[min(self.reset_calls, len(self.drifts) - 1)]
        self.reset_calls += 1
        self.task.reset_source(drift)
        return ["stack the wine bottle"], object()


class _LifecyclePyRep:
    def __init__(self):
        self.running = False

    def stop(self):
        self.running = False


class _FreshStageEnvironment:
    """Minimal lifecycle wrapper used by pure runtime staging tests."""

    def __init__(self, task_environment):
        self.drifts = list(task_environment.drifts)
        self.generation_index = 0
        self._pyrep = _LifecyclePyRep()
        self._scene = type(
            "FreshScene",
            (),
            {"task": None, "unload": lambda scene: setattr(scene, "task", None)},
        )()
        self._robot = task_environment._scene.robot
        self._action_mode = type(
            "ActionMode",
            (),
            {
                "arm_action_mode": type(
                    "ArmActionMode",
                    (),
                    {"set_control_mode": lambda self, robot: None},
                )(),
            },
        )()

    def get_task(self, _task_class):
        drift = self.drifts[min(self.generation_index, len(self.drifts) - 1)]
        reject_candidate = self.generation_index == 0
        self.generation_index += 1
        task_environment = _PerAttemptTaskEnvironment(
            [drift], reject_candidate=reject_candidate
        )
        self._pyrep.running = True
        self._scene = task_environment._scene
        return task_environment


def _initialize_formal_for_test(task_environment):
    lifecycle = _FreshStageEnvironment(task_environment)
    formal, descriptions, _observation, _evidence = (
        initialize_fresh_task_generation(
            lifecycle,
            object,
            episode_seed=17,
            variation=0,
        )
    )
    return formal, descriptions


def test_fresh_task_generation_uses_strict_lifecycle_and_new_python_instance(
    monkeypatch,
) -> None:
    events = []

    class PyRep:
        running = False

        def stop(self):
            events.append("stop")
            self.running = False

    class Task:
        def get_name(self):
            return "stack_wine"

    class TaskEnvironment:
        def __init__(self, scene):
            self._scene = scene
            self.task = scene.task
            self.reset_calls = 0

        def set_variation(self, variation):
            events.append(("variation", variation))

        def reset(self, *, verify_instance):
            events.append(("reset", verify_instance))
            self.reset_calls += 1
            return ["description"], object()

    class Environment:
        def __init__(self):
            self._pyrep = PyRep()
            self._scene = type(
                "Scene",
                (),
                {
                    "task": None,
                    "unload": lambda scene: (
                        events.append(("unload", self._pyrep.running)),
                        setattr(scene, "task", None),
                    ),
                },
            )()
            self.generations = []

        def get_task(self, task_class):
            assert task_class is Task
            events.append("get_task_load_start")
            self._scene = type(
                "Scene",
                (),
                {
                    "task": Task(),
                    "unload": lambda scene: (
                        events.append(("unload", self._pyrep.running)),
                        setattr(scene, "task", None),
                    ),
                },
            )()
            self._pyrep.running = True
            task_environment = TaskEnvironment(self._scene)
            self.generations.append(task_environment)
            return task_environment

    monkeypatch.setattr(
        runtime_module.random,
        "seed",
        lambda seed: events.append(("random_seed", seed)),
    )
    monkeypatch.setattr(
        runtime_module.np.random,
        "seed",
        lambda seed: events.append(("numpy_seed", seed)),
    )
    environment = Environment()

    first, _, _, first_evidence = initialize_fresh_task_generation(
        environment,
        Task,
        episode_seed=7,
        variation=2,
    )
    first.task.runtime_pollution = True
    second, _, _, second_evidence = initialize_fresh_task_generation(
        environment,
        Task,
        episode_seed=7,
        variation=2,
    )

    expected_generation = [
        "stop",
        "get_task_load_start",
        ("random_seed", 7),
        ("numpy_seed", 7),
        ("variation", 2),
        ("reset", True),
    ]
    assert events == [
        *expected_generation,
        ("unload", True),
        *expected_generation,
    ]
    assert first is not second
    assert first.task is not second.task
    assert not hasattr(second.task, "runtime_pollution")
    assert first.reset_calls == second.reset_calls == 1
    assert first_evidence["generation_index"] == 1
    assert first_evidence["previous_task_present"] is False
    assert first_evidence["previous_task_unloaded_before_stop"] is False
    assert second_evidence["generation_index"] == 2
    assert second_evidence["previous_task_present"] is True
    assert second_evidence["previous_task_unloaded_before_stop"] is True
    assert second_evidence[
        "previous_task_unloaded_while_physics_running"
    ] is True


def test_condition_semantics_exclude_only_typed_runtime_progress() -> None:
    Condition = type("Condition", (), {"reset": lambda self: None})
    Condition.__module__ = "rlbench.backend.conditions"
    OrConditions = type("OrConditions", (Condition,), {})
    OrConditions.__module__ = "rlbench.backend.conditions"
    CustomCondition = type("CustomCondition", (Condition,), {})
    CustomCondition.__module__ = "custom_conditions"

    before = OrConditions()
    before._conditions = [CustomCondition()]
    before._conditions[0].threshold = 0.5
    after = OrConditions()
    after._conditions = [CustomCondition()]
    after._conditions[0].threshold = 0.5
    after._current_condition_index = 99

    class SemanticTask:
        _fail_conditions = []
        _graspable_objects = []

        def __init__(self, condition):
            self._success_conditions = [condition]

    assert _task_semantic_signature(SemanticTask(before)) == (
        _task_semantic_signature(SemanticTask(after))
    )
    after._conditions[0].threshold = 0.6
    assert _task_semantic_signature(SemanticTask(before)) != (
        _task_semantic_signature(SemanticTask(after))
    )

    StatefulCustom = type(
        "StatefulCustom",
        (Condition,),
        {"reset": lambda self: setattr(self, "progress", 0)},
    )
    StatefulCustom.__module__ = "custom_conditions"
    with pytest.raises(RuntimeError, match="unmodeled runtime state"):
        _task_semantic_signature(SemanticTask(StatefulCustom()))


def _comparison_row(comparison, name):
    return next(row for row in comparison["objects"] if row["name"] == name)


def test_task_tree_root_motion_uses_frame_by_subtree_membership() -> None:
    source = _task_tree_relative_state(_FrameAwareTask(root_x=0.0))
    goal = _task_tree_relative_state(_FrameAwareTask(root_x=2.0))

    replay = _compare_task_tree_relative_state(source, source)
    motion = _compare_task_tree_relative_state(
        source,
        goal,
        boundary_root_may_move=True,
    )

    assert replay["matched"] is True
    assert replay["comparison_mode"] == "all_objects_world"
    assert motion["matched"] is True
    assert motion["comparison_mode"] == (
        "boundary_root_subtree_relative_else_world"
    )
    ancestor = _comparison_row(motion, "task_base")
    descendant = _comparison_row(motion, "task_joint")
    assert ancestor["in_boundary_root_subtree"] is False
    assert ancestor["pose_comparison_frame"] == "world"
    assert ancestor["boundary_root_relative_translation_error_m"] == pytest.approx(2.0)
    assert descendant["in_boundary_root_subtree"] is True
    assert descendant["pose_comparison_frame"] == "boundary_root"
    assert descendant["world_translation_error_m"] == pytest.approx(2.0)


def test_task_tree_root_motion_rejects_external_object_or_joint_changes() -> None:
    source = _task_tree_relative_state(_FrameAwareTask(root_x=0.0))
    external_changed = _task_tree_relative_state(
        _FrameAwareTask(root_x=2.0, external_x=5.01)
    )
    joint_changed = _task_tree_relative_state(
        _FrameAwareTask(root_x=2.0, joint_position=0.01)
    )

    external_comparison = _compare_task_tree_relative_state(
        source,
        external_changed,
        boundary_root_may_move=True,
    )
    joint_comparison = _compare_task_tree_relative_state(
        source,
        joint_changed,
        boundary_root_may_move=True,
    )

    assert external_comparison["matched"] is False
    external = _comparison_row(external_comparison, "task_metadata_anchor")
    assert external["pose_comparison_frame"] == "world"
    assert external["translation_error_m"] == pytest.approx(0.01)
    assert joint_comparison["matched"] is False
    assert _comparison_row(joint_comparison, "task_joint")[
        "joint_position_error"
    ] == pytest.approx(0.01)


def test_task_tree_post_validation_comparison_uses_world_pose_for_all_objects() -> None:
    before_validation = _task_tree_relative_state(_FrameAwareTask(root_x=2.0))
    after_validation = _task_tree_relative_state(
        _FrameAwareTask(root_x=2.0, descendant_local_x=1.001)
    )

    comparison = _compare_task_tree_relative_state(
        before_validation,
        after_validation,
        boundary_root_may_move=False,
    )

    assert comparison["matched"] is False
    changed = _comparison_row(comparison, "task_joint")
    assert changed["pose_comparison_frame"] == "world"
    assert changed["translation_error_m"] == pytest.approx(0.001)
    assert TASK_TREE_STATE_SCHEMA == "rlbench-task-tree-dual-frame-state-v1"


def test_staging_rejected_candidate_uses_next_reset_A_as_new_strict_source(
    monkeypatch,
) -> None:
    pytest.skip("superseded by deterministic same-source-seed V3.4 tests")
    monkeypatch.setattr(
        runtime_module,
        "_robot_external_collision_pairs",
        lambda _scene, _robot: (),
    )
    environment = _PerAttemptTaskEnvironment([0.0, 5.0e-6])

    plan = stage_scenario_motion_plan(
        _FreshStageEnvironment(environment),
        object,
        episode_seed=17,
        variation=0,
        task_name="stack_wine",
        max_attempts=20,
    )

    assert plan.validation["sampling_attempts"] == 2
    assert plan.validation["waypoint_rejections"] == 1
    assert plan.source_low_dim_state[0] == pytest.approx(1.0 + 5.0e-6)
    assert plan.goal_pose[0] == pytest.approx(0.1)
    cross = plan.validation["cross_initialization_reproducibility"]
    assert cross["reference_attempt"] == 1
    assert cross["selected_attempt"] == 2
    assert cross["candidate_source_policy"] == (
        "each_candidate_uses_its_same_fresh_generation_A"
    )
    assert len(cross["attempts"]) == 2
    assert cross["attempts"][1]["low_dim_state"]["raw_max_abs"] == pytest.approx(
        5.0e-6
    )
    assert cross["all_attempts_passed"] is True
    assert cross["tolerances"]["joint_position"] == (
        CROSS_INITIALIZATION_JOINT_TOLERANCE
    )
    assert plan.validation["goal_pre_validation_task_tree_state_preserved"][
        "translation_tolerance_m"
    ] == LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
    assert plan.validation["goal_waypoint_validation_task_tree_state_preserved"][
        "rotation_tolerance_rad"
    ] == LOW_DIM_POSE_ROTATION_TOLERANCE_RAD


def test_formal_binding_uses_cross_initialization_caps_but_keeps_topology_strict(
    monkeypatch,
) -> None:
    pytest.skip("superseded by strict deterministic source reconstruction")
    monkeypatch.setattr(
        runtime_module,
        "_robot_external_collision_pairs",
        lambda _scene, _robot: (),
    )
    staged = _PerAttemptTaskEnvironment([0.0, 5.0e-6])
    plan = stage_scenario_motion_plan(
        _FreshStageEnvironment(staged),
        object,
        episode_seed=17,
        variation=0,
        task_name="stack_wine",
        max_attempts=20,
    )
    formal = _PerAttemptTaskEnvironment([15.0e-6])
    formal, descriptions = _initialize_formal_for_test(formal)
    binding = ScenarioController(
        "teleport_task",
        trigger_step=0,
        total_steps=10,
        motion_plan=plan,
    ).bind_staged_source(
        formal,
        episode_seed=17,
        variation=0,
        descriptions=descriptions,
    )

    assert binding["matched"] is True
    assert binding["cross_initialization_reproducibility"]["passed"] is True
    assert binding["selected_source_fingerprint"] != binding[
        "formal_source_fingerprint"
    ]
    assert binding["robot_external_collision_pairs_matched"] is True
    assert binding["cross_initialization_reproducibility"][
        "task_object_velocities_compared_for_identity"
    ] is False

    invalid_formal = _PerAttemptTaskEnvironment([50.0e-6])
    invalid_formal, invalid_descriptions = _initialize_formal_for_test(
        invalid_formal
    )
    with pytest.raises(RuntimeError, match="formal task tree A"):
        ScenarioController(
            "teleport_task",
            trigger_step=0,
            total_steps=10,
            motion_plan=plan,
        ).bind_staged_source(
            invalid_formal,
            episode_seed=17,
            variation=0,
            descriptions=invalid_descriptions,
        )

    root_drift = _PerAttemptTaskEnvironment([15.0e-6])
    root_drift, root_descriptions = _initialize_formal_for_test(root_drift)
    root_drift.task.move_root_to(2.0e-6)
    with pytest.raises(RuntimeError, match="formal source root A"):
        ScenarioController(
            "teleport_task",
            trigger_step=0,
            total_steps=10,
            motion_plan=plan,
        ).bind_staged_source(
            root_drift,
            episode_seed=17,
            variation=0,
            descriptions=root_descriptions,
        )

    topology_drift = _PerAttemptTaskEnvironment([15.0e-6])
    topology_drift, topology_descriptions = _initialize_formal_for_test(
        topology_drift
    )
    topology_drift.task.base._children.remove(topology_drift.task.external)
    topology_drift.task.external._parent = topology_drift.task.root
    topology_drift.task.root._children.append(topology_drift.task.external)
    with pytest.raises(RuntimeError, match="formal task tree A"):
        ScenarioController(
            "teleport_task",
            trigger_step=0,
            total_steps=10,
            motion_plan=plan,
        ).bind_staged_source(
            topology_drift,
            episode_seed=17,
            variation=0,
            descriptions=topology_descriptions,
        )


@pytest.mark.parametrize("kind", ("teleport_task", "smooth_task_motion"))
def test_formal_intervention_hard_audit_failure_does_not_commit_controller(
    monkeypatch,
    kind,
) -> None:
    formal = _PerAttemptTaskEnvironment([0.0], reject_candidate=False)
    root = formal.task.boundary_root()
    source_pose = np.asarray(root.get_pose(), dtype=np.float64)
    goal_pose = source_pose.copy()
    goal_pose[0] += 0.1
    controller = ScenarioController(kind, trigger_step=0, total_steps=2)

    # The transaction-order test does not need a staged-plan fixture. Mark the
    # already-initialized controller as formally bound and supply only the
    # immutable numeric poses consumed by apply().
    controller.motion_plan = object()
    controller._motion_source_pose = source_pose
    controller._motion_goal_pose = goal_pose
    controller._instance_preservation = {
        "motion_plan_fingerprint": "plan",
        "validation_fingerprint": "validation",
    }
    controller._staged_source_bound = True
    monkeypatch.setattr(
        runtime_module,
        "_formal_intervention_state_snapshot",
        lambda _scene: {"snapshot": "before"},
    )

    def fail_hard_audit(_scene, _before):
        raise RuntimeError("formal boundary-root command changed task-tree")

    monkeypatch.setattr(
        runtime_module,
        "_formal_intervention_state_audit",
        fail_hard_audit,
    )

    with pytest.raises(RuntimeError, match="changed task-tree"):
        controller.apply(formal, step=0, horizon=2)

    assert controller._last_motion_policy_step is None
    assert controller._teleported is False
    assert controller._smooth_calls == 0
    assert controller._smooth_complete is False


def test_formal_intervention_records_new_contact_as_diagnostic(
    monkeypatch,
) -> None:
    before = {
        "task_tree": [{"name": "root"}],
        "task_semantics": {"condition": "unchanged"},
        "instance_references": {"success": object()},
        "grasp_state": {"grasped": ()},
        "robot_external_collision_pairs": (("arm", 10, "table"),),
    }
    after = {
        **before,
        "robot_external_collision_pairs": (
            ("arm", 10, "table"),
            ("arm", 20, "microwave_frame_resp"),
        ),
    }
    monkeypatch.setattr(
        runtime_module,
        "_formal_intervention_state_snapshot",
        lambda _scene: after,
    )
    monkeypatch.setattr(
        runtime_module,
        "_compare_task_tree_relative_state",
        lambda *_args, **_kwargs: {"matched": True},
    )
    monkeypatch.setattr(
        runtime_module,
        "_same_instance_references",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        runtime_module,
        "_same_grasp_state",
        lambda *_args: True,
    )

    audit = runtime_module._formal_intervention_state_audit(
        SimpleNamespace(task=object(), robot=object()),
        before,
    )

    assert audit["schema"].endswith("state-audit-v2")
    assert audit["robot_external_collision_pair_policy"] == (
        runtime_module.FORMAL_INTERVENTION_COLLISION_PAIR_POLICY
    )
    assert audit["before_robot_external_collision_pairs"] == [
        {"arm": "arm", "external_object_name": "table"}
    ]
    assert audit["after_robot_external_collision_pairs"] == [
        {"arm": "arm", "external_object_name": "microwave_frame_resp"},
        {"arm": "arm", "external_object_name": "table"},
    ]
    assert audit["new_robot_external_collision_pairs"] == [
        {"arm": "arm", "external_object_name": "microwave_frame_resp"}
    ]
    assert audit["no_new_robot_external_collision_pairs"] is False
    assert audit["passed"] is True


@pytest.mark.parametrize(
    ("failed_guard", "message"),
    (
        ("tree", "task-tree topology or state"),
        ("semantics", "changed task semantics"),
        ("registry", "replaced task condition/grasp registries"),
        ("grasp", "changed gripper membership or parentage"),
    ),
)
def test_formal_intervention_integrity_guards_remain_hard_failures(
    monkeypatch,
    failed_guard,
    message,
) -> None:
    before = {
        "task_tree": [{"name": "root"}],
        "task_semantics": {"condition": "before"},
        "instance_references": {"success": object()},
        "grasp_state": {"grasped": ()},
        "robot_external_collision_pairs": (),
    }
    after = {
        **before,
        "task_semantics": (
            {"condition": "after"}
            if failed_guard == "semantics"
            else before["task_semantics"]
        ),
        "robot_external_collision_pairs": (("arm", 20, "moved_task"),),
    }
    monkeypatch.setattr(
        runtime_module,
        "_formal_intervention_state_snapshot",
        lambda _scene: after,
    )
    monkeypatch.setattr(
        runtime_module,
        "_compare_task_tree_relative_state",
        lambda *_args, **_kwargs: {"matched": failed_guard != "tree"},
    )
    monkeypatch.setattr(
        runtime_module,
        "_same_instance_references",
        lambda *_args: failed_guard != "registry",
    )
    monkeypatch.setattr(
        runtime_module,
        "_same_grasp_state",
        lambda *_args: failed_guard != "grasp",
    )

    with pytest.raises(RuntimeError, match=message):
        runtime_module._formal_intervention_state_audit(
            SimpleNamespace(task=object(), robot=object()),
            before,
        )


def test_staged_motion_plan_batch_is_scenario_independent_and_authenticated() -> None:
    pytest.skip("superseded by the authenticated V3.4 release-plan fixture")
    generation_body = {
        "schema": FRESH_TASK_GENERATION_EVIDENCE_SCHEMA,
        "protocol_id": FRESH_TASK_GENERATION_PROTOCOL_ID,
        "generation_index": 1,
        "episode_seed": 17,
        "variation": 0,
        "task_name": "stack_wine",
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
        task_name="stack_wine",
        source_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        goal_pose=(0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0),
        source_low_dim_state=tuple(_pose_state()),
        episode_seed=17,
        variation=0,
        validation={
            "schema": STAGED_MOTION_PLAN_VALIDATION_SCHEMA,
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "formal_rollout_sample_or_restore": False,
            "formal_source_binding_required": True,
            "sampling_attempts": 1,
            "staging_max_attempts": 20,
            "fresh_task_generation_protocol_id": (
                FRESH_TASK_GENERATION_PROTOCOL_ID
            ),
            "selected_source_fresh_task_generation": generation_evidence,
        },
    )
    payload = staged_motion_plan_batch(
        task_name="stack_wine",
        base_seed=17,
        variations=[0],
        plans=[plan],
    )

    restored = load_staged_motion_plan_batch(payload)

    assert payload["scenario_independent"] is True
    assert "scenario" not in payload
    assert restored[0].fingerprint() == plan.fingerprint()
    assert restored[0].validation["formal_rollout_sample_or_restore"] is False

    legacy = copy.deepcopy(payload)
    legacy_validation = legacy["plans"][0]["validation"]
    legacy_validation["fresh_task_generation_protocol_id"] = (
        "rlbench-stop-reload-start-seed-variation-single-reset-v1"
    )
    legacy_evidence = legacy_validation[
        "selected_source_fresh_task_generation"
    ]
    legacy_evidence["schema"] = (
        "dynamac-rlbench-fresh-task-generation-evidence-v1"
    )
    legacy_evidence["protocol_id"] = (
        "rlbench-stop-reload-start-seed-variation-single-reset-v1"
    )
    legacy_evidence["fingerprint"] = _canonical_json_fingerprint(
        {
            key: value
            for key, value in legacy_evidence.items()
            if key != "fingerprint"
        }
    )
    legacy_plan = legacy["plans"][0]
    legacy_plan["fingerprint"] = _canonical_json_fingerprint(
        {key: value for key, value in legacy_plan.items() if key != "fingerprint"}
    )
    legacy["batch_fingerprint"] = _canonical_json_fingerprint(
        {key: value for key, value in legacy.items() if key != "batch_fingerprint"}
    )
    with pytest.raises(ValueError, match="fresh task-generation protocol"):
        load_staged_motion_plan_batch(legacy)

    tampered = dict(payload)
    tampered["base_seed"] = 18
    with pytest.raises(ValueError, match="fingerprint"):
        load_staged_motion_plan_batch(tampered)

    with pytest.raises(ValueError, match="batch must be a dictionary"):
        load_staged_motion_plan_batch([])

    null_plan = copy.deepcopy(payload)
    null_plan["plans"] = [None]
    null_plan["batch_fingerprint"] = _canonical_json_fingerprint(
        {key: value for key, value in null_plan.items() if key != "batch_fingerprint"}
    )
    with pytest.raises(ValueError, match="plan payload must be a dictionary"):
        load_staged_motion_plan_batch(null_plan)

    legacy_plan = plan.to_json()
    legacy_plan["validation"] = {
        **legacy_plan["validation"],
        "schema": "dynamac-rlbench-staged-motion-plan-validation-v3.2",
    }
    with pytest.raises(ValueError, match="validation schema"):
        StagedMotionPlan.from_json(legacy_plan)

    legacy_plan_schema = plan.to_json()
    legacy_plan_schema["schema"] = "dynamac-rlbench-staged-motion-plan-v3.2"
    with pytest.raises(ValueError, match="plan schema"):
        StagedMotionPlan.from_json(legacy_plan_schema)

    legacy_plan_protocol = plan.to_json()
    legacy_plan_protocol["protocol_id"] = (
        "rlbench-independent-staging-waypoint-validated-boundary-root-v3.2"
    )
    with pytest.raises(ValueError, match="protocol ID"):
        StagedMotionPlan.from_json(legacy_plan_protocol)

    legacy_batch_schema = dict(payload)
    legacy_batch_schema["schema"] = (
        "dynamac-rlbench-staged-motion-plan-batch-v3.2"
    )
    with pytest.raises(ValueError, match="batch schema"):
        load_staged_motion_plan_batch(legacy_batch_schema)

    legacy_batch_protocol = dict(payload)
    legacy_batch_protocol["protocol_id"] = (
        "rlbench-independent-staging-waypoint-validated-boundary-root-v3.2"
    )
    with pytest.raises(ValueError, match="batch protocol"):
        load_staged_motion_plan_batch(legacy_batch_protocol)


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


def test_cross_initialization_caps_accept_store_drift_and_reject_wipe_drift() -> None:
    source = _pose_state()
    store_drift = source.copy()
    store_drift[0] += 5.10073e-6
    angle = 1.130529e-4
    store_drift[3:7] = [
        0.0,
        0.0,
        math.sin(angle / 2.0),
        math.cos(angle / 2.0),
    ]
    wipe_drift = source.copy()
    wipe_drift[0] += 5.0e-5

    store = _low_dim_roundtrip_metrics(
        source,
        store_drift,
        translation_tolerance_m=CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
        rotation_tolerance_rad=CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        scalar_tolerance=CROSS_INITIALIZATION_SCALAR_TOLERANCE,
    )
    wipe = _low_dim_roundtrip_metrics(
        source,
        wipe_drift,
        translation_tolerance_m=CROSS_INITIALIZATION_TRANSLATION_TOLERANCE_M,
        rotation_tolerance_rad=CROSS_INITIALIZATION_ROTATION_TOLERANCE_RAD,
        scalar_tolerance=CROSS_INITIALIZATION_SCALAR_TOLERANCE,
    )

    assert store["preserved"] is True
    assert store["max_rotation_rad"] == pytest.approx(angle, abs=1.0e-12)
    assert wipe["preserved"] is False


def test_stable_collision_records_sort_by_name_not_transient_handle() -> None:
    records = _stable_collision_pair_records(
        (
            ("arm", 1, "z_object"),
            ("arm", 99, "a_object"),
            ("arm", 7, "z_object"),
        )
    )

    assert records == [
        {"arm": "arm", "external_object_name": "a_object"},
        {"arm": "arm", "external_object_name": "z_object"},
    ]


def test_small_quaternion_angle_uses_stable_sign_invariant_chord_metric() -> None:
    angle = 8.135e-6
    source = np.asarray([0.0, 0.0, 0.0, 1.0])
    goal = np.asarray([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])

    assert _quaternion_angle_xyzw(source, goal) == pytest.approx(angle, abs=1.0e-14)
    assert _quaternion_angle_xyzw(source, -goal) == pytest.approx(
        angle,
        abs=1.0e-14,
    )


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
