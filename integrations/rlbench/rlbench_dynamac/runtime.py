"""Pure conversion and intervention helpers for the optional RLBench runtime.

No RLBench/PyRep import happens at module import time.  This lets conversion
and action-layout tests run on machines without CoppeliaSim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .task_specs import TaskSpec, get_task_spec, unwrap_task_low_dim_state
from .task_specs import wxyz_to_xyzw as _wxyz_to_xyzw
from .task_specs import xyzw_to_wxyz as _xyzw_to_wxyz

Array = np.ndarray

PRESERVE_INSTANCE_MOTION_PROTOCOL_ID = (
    "rlbench-boundary-root-preserve-initialized-episode-v4"
)
LOW_DIM_STATE_ROUNDTRIP_ATOL = 1.0e-6
LOW_DIM_POSE_TRANSLATION_TOLERANCE_M = 1.0e-6
LOW_DIM_POSE_ROTATION_TOLERANCE_RAD = 1.0e-6
LOW_DIM_POSE_QUATERNION_NORM_ATOL = 1.0e-3
ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M = 1.0e-9
ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD = 1.0e-9
ROOT_COMMAND_TRANSLATION_TOLERANCE_M = 1.0e-6
ROOT_COMMAND_ROTATION_TOLERANCE_RAD = 1.0e-6


# RLBench's waypoint demonstration generator actuates every gripper at 0.04
# (see ``Scene._handle_extensions_strings`` in the pinned fork).  Evaluation
# must use the same physical command speed: the upstream Discrete modes use
# 0.2, which changes contact dynamics even when the policy trajectory is
# identical to the demonstration.
DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY = 0.04
DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS = 3


@dataclass
class PrimaryActionRetryBudget:
    """Bound consecutive InvalidAction retries for one policy clock tick.

    An invalid primary command is aborted by the policy worker, so the next
    prediction is still the same policy tick.  A successful primary command is
    committed and therefore starts a new tick with a fresh retry budget.
    """

    max_attempts: int = DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
    attempts: int = 0
    peak_attempts: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("max primary action attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max primary action attempts must be positive")

    def record_failure(self) -> bool:
        """Record one aborted primary attempt and report budget exhaustion."""

        self.attempts += 1
        self.peak_attempts = max(self.peak_attempts, self.attempts)
        return self.attempts >= self.max_attempts

    def record_success(self) -> None:
        """Reset only after the primary command is successfully committed."""

        self.attempts = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "max_primary_action_attempts_per_policy_tick": self.max_attempts,
            "exhaustion_reason": "primary_action_retry_exhausted",
            "counter_reset": "after_successful_primary_action_commit",
        }


def _protocol_float_token(value: float) -> str:
    """Return a compact, deterministic float token suitable for protocol IDs."""

    return format(value, ".12g").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class DiscreteGripperProtocol:
    """Task-independent discrete-gripper evaluation protocol.

    The pinned RLBench fork hard-codes velocity ``0.2`` in both ``Discrete``
    action modes, while its demonstration generator hard-codes ``0.04``.
    This project-owned protocol changes only that actuation velocity and
    inherits every other vendor behavior, including grasp attachment,
    release settling, and the bimanual handover logic.

    Keeping construction, metadata, and the protocol-ID fragment on one
    immutable object prevents an evaluator from recording a velocity other
    than the one it actually executes.
    """

    bimanual: bool
    actuation_velocity: float = DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY
    attach_grasped_objects: bool = True
    detach_before_open: bool = True

    def __post_init__(self) -> None:
        for name in ("bimanual", "attach_grasped_objects", "detach_before_open"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        velocity = float(self.actuation_velocity)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError("gripper actuation velocity must be finite and positive")
        object.__setattr__(self, "actuation_velocity", velocity)

    @property
    def protocol_id(self) -> str:
        layout = "bimanual" if self.bimanual else "unimanual"
        velocity = _protocol_float_token(self.actuation_velocity)
        attach = int(self.attach_grasped_objects)
        detach = int(self.detach_before_open)
        return (
            f"rlbench-discrete-gripper-{layout}-velocity{velocity}"
            f"-attach{attach}-detach-before-open{detach}-v1"
        )

    def extend_evaluation_protocol_id(self, base_protocol_id: str) -> str:
        """Include this physical gripper protocol in a stable evaluator ID."""

        base = str(base_protocol_id).strip()
        if not base:
            raise ValueError("base evaluation protocol ID must be non-empty")
        suffix = f"+{self.protocol_id}"
        return base if base.endswith(suffix) else f"{base}{suffix}"

    def metadata(self) -> dict[str, Any]:
        """Return JSON-stable metadata for an evaluation result."""

        return {
            "protocol_id": self.protocol_id,
            "action_mode": "BimanualDiscrete" if self.bimanual else "Discrete",
            "arm_layout": "bimanual" if self.bimanual else "unimanual",
            "actuation_velocity": self.actuation_velocity,
            "demonstration_actuation_velocity": (
                DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY
            ),
            "velocity_aligned_with_demonstrations": bool(
                math.isclose(
                    self.actuation_velocity,
                    DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            "attach_grasped_objects": self.attach_grasped_objects,
            "detach_before_open": self.detach_before_open,
            "implementation": "project_subclass_preserving_vendor_action_semantics",
        }

    def make_action_mode(self) -> Any:
        """Create the configured RLBench mode without importing RLBench eagerly."""

        return _make_discrete_gripper_action_mode(self)


def make_discrete_gripper_action_mode(
    *,
    bimanual: bool,
    actuation_velocity: float = DEMONSTRATION_GRIPPER_ACTUATION_VELOCITY,
    attach_grasped_objects: bool = True,
    detach_before_open: bool = True,
) -> Any:
    """Convenience factory for the aligned single- or dual-arm action mode."""

    return DiscreteGripperProtocol(
        bimanual=bimanual,
        actuation_velocity=actuation_velocity,
        attach_grasped_objects=attach_grasped_objects,
        detach_before_open=detach_before_open,
    ).make_action_mode()


def _make_discrete_gripper_action_mode(protocol: DiscreteGripperProtocol) -> Any:
    """Build a minimal velocity override around the pinned vendor classes."""

    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, Discrete

    if protocol.bimanual:

        class ProtocolBimanualDiscrete(BimanualDiscrete):
            dynamac_protocol = protocol

            def _actuate(self, scene: Any, action: Array) -> None:
                right_action = action[0]
                left_action = action[1]
                right_done = False
                left_done = False
                while not (right_done and left_done):
                    if not right_done:
                        right_done = scene.robot.right_gripper.actuate(
                            right_action,
                            velocity=protocol.actuation_velocity,
                        )
                    if not left_done:
                        left_done = scene.robot.left_gripper.actuate(
                            left_action,
                            velocity=protocol.actuation_velocity,
                        )
                    scene.pyrep.step()
                    scene.task.step()

        mode_class = ProtocolBimanualDiscrete
    else:

        class ProtocolDiscrete(Discrete):
            dynamac_protocol = protocol

            def _actuate(self, scene: Any, action: Array | float) -> None:
                done = False
                while not done:
                    done = scene.robot.gripper.actuate(
                        action,
                        velocity=protocol.actuation_velocity,
                    )
                    scene.pyrep.step()
                    scene.task.step()

        mode_class = ProtocolDiscrete

    return mode_class(
        attach_grasped_objects=protocol.attach_grasped_objects,
        detach_before_open=protocol.detach_before_open,
    )


def execute_joint_target_control(
    scene: Any,
    arm_targets: tuple[tuple[Any, Array], ...],
    *,
    max_steps: int = 200,
    reached_atol: float = 0.01,
    stopped_atol: float = 0.001,
    invalid_action_error: type[Exception] = RuntimeError,
    error_message: str = "absolute end-effector IK execution timed out",
) -> Literal["reached", "stopped"]:
    """Drive one synchronized joint-target command to a terminal arm state.

    RLBench's public IK controller treats either reaching the target or ceasing
    to move (for example after contact) as the end of one high-level arm
    command.  The local controllers add a finite safety bound; exhausting that
    bound is an invalid action and must enter the evaluator's no-op fallback,
    rather than silently continuing with the accompanying gripper command.

    Task success is deliberately not inspected here.  ``MoveArmThenGripper``
    owns one combined action and would otherwise still execute its gripper
    command after this arm helper returned.  Episode termination is therefore
    evaluated once by ``TaskEnvironment.step`` after the combined action.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not arm_targets:
        raise ValueError("at least one arm target is required")
    if reached_atol <= 0.0 or stopped_atol <= 0.0:
        raise ValueError("joint tolerances must be positive")

    normalized = tuple(
        (arm, np.asarray(target, dtype=np.float64).copy())
        for arm, target in arm_targets
    )
    previous: tuple[Array, ...] | None = None
    for _ in range(max_steps):
        scene.step()
        current = tuple(
            np.asarray(arm.get_joint_positions(), dtype=np.float64)
            for arm, _ in normalized
        )
        if all(
            np.allclose(value, target, atol=reached_atol)
            for value, (_, target) in zip(current, normalized)
        ):
            return "reached"
        if previous is not None and all(
            np.allclose(value, prior, atol=stopped_atol)
            for value, prior in zip(current, previous)
        ):
            return "stopped"
        previous = current

    raise invalid_action_error(error_message)


def xyzw_to_wxyz(pose: Array) -> Array:
    """Compatibility export for the canonical, audited convention crossing."""

    return _xyzw_to_wxyz(pose)


def wxyz_to_xyzw(pose: Array) -> Array:
    """Compatibility export for the canonical, audited convention crossing."""

    return _wxyz_to_xyzw(pose)


def unimanual_observation_from_rlbench(observation: Any, task: str | TaskSpec) -> Any:
    """Build a core observation without importing RLBench or PyRep."""

    from essay2608.policy import DynaMACObservation

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if spec.bimanual:
        raise ValueError(f"{spec.task_name} is bimanual")
    return DynaMACObservation(
        ee_pose=xyzw_to_wxyz(observation.gripper_pose),
        frames=spec.extract_pose_chunks(observation.task_low_dim_state),
    )


def bimanual_observations_from_rlbench(
    observation: Any,
    task: str | TaskSpec,
) -> tuple[Any, Any]:
    """Build synchronized left/right core observations from one simulator snapshot."""

    from essay2608.policy import DynaMACObservation

    spec = task if isinstance(task, TaskSpec) else get_task_spec(task)
    if not spec.bimanual:
        raise ValueError(f"{spec.task_name} is unimanual")
    frames = spec.extract_pose_chunks(observation.task_low_dim_state)
    return (
        DynaMACObservation(
            ee_pose=xyzw_to_wxyz(observation.left.gripper_pose),
            frames={name: value.copy() for name, value in frames.items()},
        ),
        DynaMACObservation(
            ee_pose=xyzw_to_wxyz(observation.right.gripper_pose),
            frames={name: value.copy() for name, value in frames.items()},
        ),
    )


def _gripper_to_rlbench(value: Array | float) -> float:
    scalar = float(np.asarray(value, dtype=np.float64).reshape(-1).mean())
    if not np.isfinite(scalar):
        raise ValueError("gripper prediction must be finite")
    # TAPAS stores 2 * gripper_open - 1.  Zero is the deterministic midpoint.
    return float(scalar > 0.0)


def unimanual_action_to_rlbench(action: Any, *, ignore_collisions: bool = False) -> Array:
    """Return the fork's 9D ``pose, gripper, ignore`` action."""

    pose = wxyz_to_xyzw(np.asarray(action.pose, dtype=np.float64))
    return np.concatenate((pose, [_gripper_to_rlbench(action.gripper), float(ignore_collisions)]))


@dataclass(frozen=True)
class ArmActionOffset:
    """Explicitly inferred diagnostic intervention, never an author default."""

    arm: Literal["left", "right"]
    translation: tuple[float, float, float]

    def apply(self, left_pose: Array, right_pose: Array) -> tuple[Array, Array]:
        left = np.asarray(left_pose, dtype=np.float64).copy()
        right = np.asarray(right_pose, dtype=np.float64).copy()
        target = left if self.arm == "left" else right
        target[:3] += np.asarray(self.translation, dtype=np.float64)
        return left, right


def bimanual_action_to_rlbench(
    action: Any,
    *,
    left_ignore_collisions: bool = False,
    right_ignore_collisions: bool = False,
    offset: ArmActionOffset | None = None,
) -> Array:
    """Return the author's right-first 18D bimanual action layout.

    The core action object is left/right named, while the RLBench fork expects
    ``[right pose7, right grip, right ignore, left pose7, left grip, left ignore]``.
    """

    left_pose = np.asarray(action.left.pose, dtype=np.float64)
    right_pose = np.asarray(action.right.pose, dtype=np.float64)
    if offset is not None:
        left_pose, right_pose = offset.apply(left_pose, right_pose)
    right = np.concatenate(
        (
            wxyz_to_xyzw(right_pose),
            [
                _gripper_to_rlbench(action.right.gripper),
                float(right_ignore_collisions),
            ],
        )
    )
    left = np.concatenate(
        (
            wxyz_to_xyzw(left_pose),
            [
                _gripper_to_rlbench(action.left.gripper),
                float(left_ignore_collisions),
            ],
        )
    )
    result = np.concatenate((right, left))
    if result.shape != (18,):
        raise AssertionError(f"invalid RLBench bimanual action shape: {result.shape}")
    return result


def pose_execution_error(command_wxyz: Array, observed_xyzw: Array) -> dict[str, float]:
    command = np.asarray(command_wxyz, dtype=np.float64)
    observed = xyzw_to_wxyz(np.asarray(observed_xyzw, dtype=np.float64))
    position = float(np.linalg.norm(command[:3] - observed[:3]))
    q_command = command[3:7] / np.linalg.norm(command[3:7])
    q_observed = observed[3:7] / np.linalg.norm(observed[3:7])
    dot = float(np.clip(abs(np.dot(q_command, q_observed)), 0.0, 1.0))
    rotation = float(2.0 * math.acos(dot))
    return {"position_m": position, "rotation_rad": rotation}


def _task_low_dim_state(task: Any) -> Array:
    """Read task state directly, without triggering observation recorders."""

    get_state = getattr(task, "get_low_dim_state", None)
    if not callable(get_state):
        raise RuntimeError("RLBench task.get_low_dim_state() is unavailable")
    return unwrap_task_low_dim_state(get_state())


def _instance_reference_snapshot(task: Any) -> dict[str, tuple[Any, ...]]:
    """Capture registries that ``init_episode`` is allowed to replace.

    Comparing object identity (rather than equality) catches a silent rebuild
    of success conditions even when the replacement conditions happen to have
    equal values.
    """

    attributes = (
        "_success_conditions",
        "_fail_conditions",
        "_graspable_objects",
    )
    return {
        name: tuple(getattr(task, name, ()))
        for name in attributes
    }


def _same_instance_references(
    before: dict[str, tuple[Any, ...]],
    after: dict[str, tuple[Any, ...]],
) -> bool:
    if before.keys() != after.keys():
        return False
    return all(
        len(before[name]) == len(after[name])
        and all(left is right for left, right in zip(before[name], after[name]))
        for name in before
    )


def _reference_key(value: Any) -> tuple[str, int] | tuple[str, None]:
    """Return a stable identity key for a PyRep object wrapper or ``None``."""

    if value is None:
        return ("none", None)
    get_handle = getattr(value, "get_handle", None)
    if callable(get_handle):
        return ("handle", int(get_handle()))
    return ("python_id", id(value))


def _robot_collision_arms(robot: Any) -> tuple[tuple[str, Any], ...]:
    """Return named arm collision collections for either robot layout."""

    names = (
        ("right_arm", "left_arm")
        if bool(getattr(robot, "is_bimanual", False))
        else ("arm",)
    )
    arms = []
    for name in names:
        arm = getattr(robot, name, None)
        if arm is None or not callable(getattr(arm, "check_arm_collision", None)):
            raise RuntimeError(f"RLBench robot arm {name!r} cannot check collisions")
        arms.append((name, arm))
    return tuple(arms)


def _arm_collision_collection_member_handles(arm: Any) -> frozenset[int]:
    """Read the current members of a pinned PyRep arm collision collection.

    ``Arm.check_arm_collision(None)`` checks its collection against *all other*
    collidable scene objects. PyRep exposes that check but not the collection's
    current members, so use the already-bound CoppeliaSim API to reproduce the
    exclusion exactly. Current membership matters because a grasped tool can
    become a descendant of the arm and therefore a collection member.
    """

    collection = getattr(arm, "_collision_collection", None)
    if collection is None:
        raise RuntimeError("PyRep arm collision collection handle is unavailable")

    from pyrep.backend import sim

    count = sim.ffi.new("int *")
    values = sim.lib.simGetCollectionObjects(int(collection), count)
    if values == sim.ffi.NULL:
        raise RuntimeError("CoppeliaSim could not enumerate arm collision collection")
    try:
        member_count = int(count[0])
        if member_count < 0:
            raise RuntimeError("CoppeliaSim returned an invalid collection size")
        return frozenset(int(values[index]) for index in range(member_count))
    finally:
        sim.simReleaseBuffer(sim.ffi.cast("char *", values))


def _pyrep_shape_object_type() -> Any:
    """Load the optional PyRep shape enum only when a simulator is active."""

    from pyrep.const import ObjectType

    return ObjectType.SHAPE


def _robot_external_collision_pairs(
    scene: Any,
    robot: Any,
) -> tuple[tuple[str, int, str], ...]:
    """Return concrete arm-to-external-object collision pairs.

    The pair granularity is the named arm collision collection and one current
    external collidable scene shape. Enumerating shapes and invoking
    ``Arm.check_arm_collision(object)`` preserves the vendor collision geometry
    while exposing which pair caused the aggregate boolean. Objects inside an
    arm's current collection are excluded to match CoppeliaSim's
    ``sim_handle_all`` ("all other") semantics; this also handles grasped tools
    without task-specific exclusions.
    """

    pyrep = getattr(scene, "pyrep", None)
    get_objects = getattr(pyrep, "get_objects_in_tree", None)
    if not callable(get_objects):
        raise RuntimeError("PyRep scene object enumeration API is unavailable")

    collidable_objects = []
    for value in get_objects(object_type=_pyrep_shape_object_type()):
        get_handle = getattr(value, "get_handle", None)
        get_name = getattr(value, "get_name", None)
        is_collidable = getattr(value, "is_collidable", None)
        if not all(callable(item) for item in (get_handle, get_name, is_collidable)):
            raise RuntimeError("PyRep scene object collision API is unavailable")
        if bool(is_collidable()):
            collidable_objects.append((int(get_handle()), str(get_name()), value))
    collidable_objects.sort(key=lambda item: item[:2])

    pairs = []
    for arm_name, arm in _robot_collision_arms(robot):
        members = _arm_collision_collection_member_handles(arm)
        arm_pairs = []
        for handle, object_name, value in collidable_objects:
            if handle in members:
                continue
            if bool(arm.check_arm_collision(value)):
                arm_pairs.append((arm_name, handle, object_name))

        # Fail closed if object-level enumeration ever misses a collision type
        # represented by the pinned aggregate PyRep API.
        aggregate_collision = bool(arm.check_arm_collision())
        if aggregate_collision != bool(arm_pairs):
            raise RuntimeError(
                f"could not resolve {arm_name} aggregate collision into "
                "external collidable object pairs"
            )
        pairs.extend(arm_pairs)
    return tuple(sorted(pairs))


def _collision_pair_records(
    pairs: tuple[tuple[str, int, str], ...],
) -> list[dict[str, Any]]:
    """Convert collision-pair identities into JSON-stable evidence rows."""

    return [
        {
            "arm": arm_name,
            "external_object_handle": object_handle,
            "external_object_name": object_name,
        }
        for arm_name, object_handle, object_name in pairs
    ]


def _grasp_state_snapshot(task: Any, robot: Any) -> dict[str, Any]:
    """Capture gripper membership and object parents without changing them."""

    grippers = (
        ("right_gripper", robot.right_gripper),
        ("left_gripper", robot.left_gripper),
    ) if bool(getattr(robot, "is_bimanual", False)) else (
        ("gripper", robot.gripper),
    )
    tracked_objects = list(getattr(task, "_graspable_objects", ()))
    gripper_rows = []
    for name, gripper in grippers:
        get_grasped = getattr(gripper, "get_grasped_objects", None)
        if not callable(get_grasped):
            raise RuntimeError(f"RLBench {name} cannot report grasped objects")
        grasped = tuple(get_grasped())
        tracked_objects.extend(grasped)
        gripper_rows.append(
            {
                "name": name,
                "gripper": gripper,
                "grasped": grasped,
                "old_parent_keys": tuple(
                    _reference_key(parent)
                    for parent in getattr(gripper, "_old_parents", ())
                ),
            }
        )

    unique_objects = []
    seen = set()
    for value in tracked_objects:
        key = _reference_key(value)
        if key in seen:
            continue
        seen.add(key)
        unique_objects.append(value)
    parent_rows = []
    for value in unique_objects:
        get_parent = getattr(value, "get_parent", None)
        if callable(get_parent):
            parent_rows.append((value, _reference_key(get_parent())))
    return {
        "grippers": tuple(gripper_rows),
        "parents": tuple(parent_rows),
    }


def _same_grasp_state(before: dict[str, Any], task: Any, robot: Any) -> bool:
    """Compare grasp lists and object parents after a sampling transaction."""

    after = _grasp_state_snapshot(task, robot)
    before_grippers = before["grippers"]
    after_grippers = after["grippers"]
    if len(before_grippers) != len(after_grippers):
        return False
    for left, right in zip(before_grippers, after_grippers):
        if left["name"] != right["name"] or left["gripper"] is not right["gripper"]:
            return False
        if len(left["grasped"]) != len(right["grasped"]):
            return False
        if any(
            prior is not current
            for prior, current in zip(left["grasped"], right["grasped"])
        ):
            return False
        if left["old_parent_keys"] != right["old_parent_keys"]:
            return False
    if len(before["parents"]) != len(after["parents"]):
        return False
    return all(
        prior_object is current_object and prior_parent == current_parent
        for (prior_object, prior_parent), (current_object, current_parent) in zip(
            before["parents"], after["parents"]
        )
    )


def _is_expected_placement_error(error: Exception) -> bool:
    """Recognize RLBench placement failures without importing RLBench eagerly."""

    expected_names = {"BoundaryError", "WaypointError"}
    return any(base.__name__ in expected_names for base in type(error).__mro__)


def _quaternion_angle_xyzw(left: Array, right: Array) -> float:
    q_left = np.asarray(left, dtype=np.float64)
    q_right = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(q_left))
    right_norm = float(np.linalg.norm(q_right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("pose quaternion must be non-zero")
    dot = float(
        np.clip(abs(np.dot(q_left / left_norm, q_right / right_norm)), 0.0, 1.0)
    )
    return float(2.0 * math.acos(dot))


def _valid_low_dim_pose_chunks(value: Array) -> Array | None:
    """Return normalized ``[N, 7]`` pose chunks or ``None`` for scalar data.

    Every DynaMAC RLBench task schema is pose based, but keeping a deliberate
    scalar fallback makes this preservation guard safe for diagnostic or
    future tasks whose low-dimensional state is not entirely composed of
    world-frame poses.  Requiring approximately unit quaternions prevents an
    arbitrary seven-scalar vector from being misclassified as a pose.
    """

    state = np.asarray(value, dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or state.size % 7 != 0:
        return None
    chunks = state.reshape(-1, 7)
    if not np.all(np.isfinite(chunks)):
        return None
    quaternion_norms = np.linalg.norm(chunks[:, 3:7], axis=1)
    if np.any(quaternion_norms <= 0.0) or not np.allclose(
        quaternion_norms,
        1.0,
        rtol=0.0,
        atol=LOW_DIM_POSE_QUATERNION_NORM_ATOL,
    ):
        return None
    return chunks


def _low_dim_roundtrip_metrics(before: Array, restored: Array) -> dict[str, Any]:
    """Compare a task-state round trip without quaternion-gauge false alarms.

    Complete valid seven-value pose chunks use Euclidean translation and the
    sign-invariant physical quaternion angle.  Other arrays use a clearly
    labelled scalar maximum-absolute-error fallback.  Raw L2/max metrics are
    retained in both modes for forensic reporting, but do not override the
    physical pose decision.
    """

    source = np.asarray(before, dtype=np.float64)
    result = np.asarray(restored, dtype=np.float64)
    if source.shape != result.shape:
        raise ValueError("task low-dimensional state schema changed")
    delta = result - source
    raw_finite = bool(np.all(np.isfinite(source)) and np.all(np.isfinite(result)))
    raw_l2 = float(np.linalg.norm(delta)) if raw_finite else math.inf
    raw_max_abs = (
        float(np.max(np.abs(delta)))
        if raw_finite and delta.size
        else (0.0 if raw_finite else math.inf)
    )
    source_chunks = _valid_low_dim_pose_chunks(source)
    result_chunks = _valid_low_dim_pose_chunks(result)
    if source_chunks is not None and result_chunks is not None:
        translations = np.linalg.norm(
            result_chunks[:, :3] - source_chunks[:, :3],
            axis=1,
        )
        rotations = np.asarray(
            [
                _quaternion_angle_xyzw(left[3:7], right[3:7])
                for left, right in zip(source_chunks, result_chunks)
            ],
            dtype=np.float64,
        )
        max_translation = float(np.max(translations))
        max_rotation = float(np.max(rotations))
        preserved = bool(
            raw_finite
            and max_translation <= LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
            and max_rotation <= LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
        )
        comparison_mode = "pose_chunks_sign_invariant"
        chunk_count = int(source_chunks.shape[0])
    else:
        max_translation = None
        max_rotation = None
        preserved = bool(
            raw_finite and raw_max_abs <= LOW_DIM_STATE_ROUNDTRIP_ATOL
        )
        comparison_mode = "scalar_max_abs"
        chunk_count = 0
    return {
        "preserved": preserved,
        "comparison_mode": comparison_mode,
        "chunk_count": chunk_count,
        "raw_l2": raw_l2,
        "raw_max_abs": raw_max_abs,
        "max_translation_m": max_translation,
        "max_rotation_rad": max_rotation,
    }


def _root_motion_metrics(source: Array, goal: Array) -> dict[str, Any]:
    source_pose = np.asarray(source, dtype=np.float64)
    goal_pose = np.asarray(goal, dtype=np.float64)
    if source_pose.shape != (7,) or goal_pose.shape != (7,):
        raise ValueError("RLBench boundary-root poses must contain seven values")
    translation = float(np.linalg.norm(goal_pose[:3] - source_pose[:3]))
    rotation = _quaternion_angle_xyzw(source_pose[3:7], goal_pose[3:7])
    return {
        "planned_root_translation_m": translation,
        "planned_root_rotation_rad": rotation,
        "planned_root_motion": bool(
            translation > ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
            or rotation > ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
        ),
    }


def _root_application_metrics(
    before: Array,
    commanded: Array,
    applied: Array,
) -> dict[str, Any]:
    """Measure actual root motion and residual to the commanded pose."""

    actual = _root_motion_metrics(before, applied)
    residual = _root_motion_metrics(commanded, applied)
    actual_translation = float(actual["planned_root_translation_m"])
    actual_rotation = float(actual["planned_root_rotation_rad"])
    translation_residual = float(residual["planned_root_translation_m"])
    rotation_residual = float(residual["planned_root_rotation_rad"])
    command_reached = bool(
        translation_residual <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
        and rotation_residual <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
    )
    return {
        "actual_root_translation_m": actual_translation,
        "actual_root_rotation_rad": actual_rotation,
        "actual_root_motion": bool(
            actual_translation > ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
            or actual_rotation > ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
        ),
        "commanded_root_translation_residual_m": translation_residual,
        "commanded_root_rotation_residual_rad": rotation_residual,
        "commanded_root_pose_reached": command_reached,
    }


def _root_goal_reached_metrics(goal: Array, applied: Array) -> dict[str, Any]:
    """Measure an applied pose against the final sampled motion goal."""

    residual = _root_motion_metrics(goal, applied)
    translation = float(residual["planned_root_translation_m"])
    rotation = float(residual["planned_root_rotation_rad"])
    return {
        "goal_root_translation_residual_m": translation,
        "goal_root_rotation_residual_rad": rotation,
        "goal_root_pose_reached": bool(
            translation <= ROOT_COMMAND_TRANSLATION_TOLERANCE_M
            and rotation <= ROOT_COMMAND_ROTATION_TOLERANCE_RAD
        ),
    }


def _interpolate_rlbench_pose(source: Array, goal: Array, fraction: float) -> Array:
    """Interpolate an RLBench ``[xyz, qxyzw]`` pose with shortest-path SLERP."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("pose interpolation fraction must lie in [0, 1]")
    source_pose = np.asarray(source, dtype=np.float64)
    goal_pose = np.asarray(goal, dtype=np.float64)
    if source_pose.shape != (7,) or goal_pose.shape != (7,):
        raise ValueError("RLBench poses must contain seven values")

    position = (1.0 - fraction) * source_pose[:3] + fraction * goal_pose[:3]
    q_source = source_pose[3:7] / np.linalg.norm(source_pose[3:7])
    q_goal = goal_pose[3:7] / np.linalg.norm(goal_pose[3:7])
    dot = float(np.dot(q_source, q_goal))
    if dot < 0.0:
        q_goal = -q_goal
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = q_source + fraction * (q_goal - q_source)
        quaternion /= np.linalg.norm(quaternion)
    else:
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * fraction
        scale_source = math.sin(theta_0 - theta) / sin_theta_0
        scale_goal = math.sin(theta) / sin_theta_0
        quaternion = scale_source * q_source + scale_goal * q_goal
    return np.concatenate((position, quaternion))


def _restore_sampling_configuration(
    *,
    task: Any,
    task_state: Any,
    workspace_boundary: Any,
) -> None:
    """Restore only the temporarily moved task hierarchy, then clear sampling.

    Live force-controlled robot trees are deliberately never read or restored.
    CoppeliaSim configuration-tree restoration can snap a loaded joint from its
    instantaneous physical position to its control target, moving any grasped
    object even when the sampler never moved the robot.
    """

    errors = []
    try:
        task.restore_state(task_state)
    except Exception as error:  # pragma: no cover - defensive aggregation
        errors.append(("task", error))
    try:
        workspace_boundary.clear()
    except Exception as error:  # pragma: no cover - defensive aggregation
        errors.append(("workspace boundary", error))
    if errors:
        scope, error = errors[0]
        raise RuntimeError(
            f"failed to restore {scope} configuration tree after goal sampling"
        ) from error


def _sample_preserving_instance_goal(
    scene: Any,
    *,
    max_attempts: int,
) -> tuple[Array, Array, dict[str, Any]]:
    """Sample a valid root goal without reinitializing the current episode.

    The workspace sampler needs to move the real root in order to evaluate its
    bounding box and collisions. Each attempt is therefore transactional, but
    only the task configuration tree is restored. The live robot is queried for
    source/candidate external collision pairs and grasp auditing, never moved,
    validated through task waypoints, or configuration-tree restored. The
    task's low-dimensional state, condition/grasp registries, grasp membership,
    relevant parent handles, and waypoint-cache identity are checked at the
    restored source pose. A failed preservation check is fatal; silently
    continuing would mix two different episode instances.
    """

    task = getattr(scene, "task", None)
    workspace_boundary = getattr(scene, "_workspace_boundary", None)
    robot = getattr(scene, "robot", None)
    if task is None or workspace_boundary is None or robot is None:
        raise RuntimeError("RLBench scene placement internals are unavailable")

    root = task.boundary_root()
    source_pose = np.asarray(root.get_pose(), dtype=np.float64).copy()
    if source_pose.shape != (7,) or not np.all(np.isfinite(source_pose)):
        raise RuntimeError("boundary root returned an invalid source pose")
    initial_orientation = getattr(scene, "_initial_task_pose", None)
    if initial_orientation is None:
        raise RuntimeError("RLBench Scene._initial_task_pose is unavailable")

    before_state = _task_low_dim_state(task)
    before_references = _instance_reference_snapshot(task)
    get_task_state = getattr(task, "get_state", None)
    restore_task_state = getattr(task, "restore_state", None)
    if not callable(get_task_state) or not callable(restore_task_state):
        raise RuntimeError("RLBench task configuration-tree API is unavailable")
    task_state = get_task_state()
    before_grasp_state = _grasp_state_snapshot(task, robot)
    source_collision_pairs = _robot_external_collision_pairs(scene, robot)
    source_collision_pair_set = frozenset(source_collision_pairs)
    waypoint_sentinel = object()
    before_waypoints = getattr(task, "_waypoints", waypoint_sentinel)
    min_rotation, max_rotation = task.base_rotation_bounds()
    goal_pose = None
    attempts_used = 0
    last_placement_error = None
    goal_collision_pairs = None
    new_collision_pair_rejections = 0

    try:
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            workspace_boundary.clear()
            try:
                # This is the placement part of Scene._place_task, deliberately
                # separated from kidnap()/init_episode().
                root.set_orientation(initial_orientation)
                workspace_boundary.sample(
                    root,
                    min_rotation=min_rotation,
                    max_rotation=max_rotation,
                )
                candidate = np.asarray(root.get_pose(), dtype=np.float64).copy()
                if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                    raise RuntimeError("workspace sampler returned an invalid root pose")
                if not _root_motion_metrics(source_pose, candidate)[
                    "planned_root_motion"
                ]:
                    last_placement_error = "sampled root pose equals its source pose"
                    continue
                candidate_collision_pairs = _robot_external_collision_pairs(
                    scene,
                    robot,
                )
                new_collision_pairs = tuple(
                    sorted(
                        frozenset(candidate_collision_pairs)
                        - source_collision_pair_set
                    )
                )
                if new_collision_pairs:
                    new_collision_pair_rejections += 1
                    last_placement_error = (
                        "sampled task root introduces "
                        f"{len(new_collision_pairs)} new robot collision pair(s)"
                    )
                    continue
                goal_pose = candidate
                goal_collision_pairs = candidate_collision_pairs
                break
            except Exception as error:
                if not _is_expected_placement_error(error):
                    raise
                last_placement_error = str(error) or type(error).__name__
            finally:
                # A 7D root round trip is insufficient: moving the root resets
                # dynamic descendants and can introduce float32 drift. Restore
                # the complete task tree, but never touch live robot trees.
                _restore_sampling_configuration(
                    task=task,
                    task_state=task_state,
                    workspace_boundary=workspace_boundary,
                )
    finally:
        _restore_sampling_configuration(
            task=task,
            task_state=task_state,
            workspace_boundary=workspace_boundary,
        )

    restored_state = _task_low_dim_state(task)
    if before_state.shape != restored_state.shape:
        raise RuntimeError("goal sampling changed task low-dimensional state schema")
    roundtrip = _low_dim_roundtrip_metrics(before_state, restored_state)
    state_preserved = bool(roundtrip["preserved"])
    references_preserved = _same_instance_references(
        before_references,
        _instance_reference_snapshot(task),
    )
    grasp_state_preserved = _same_grasp_state(before_grasp_state, task, robot)
    after_waypoints = getattr(task, "_waypoints", waypoint_sentinel)
    waypoint_cache_preserved = after_waypoints is before_waypoints
    if not state_preserved:
        raise RuntimeError(
            "goal sampling changed the initialized task instance's low-dimensional "
            f"state beyond {roundtrip['comparison_mode']} tolerance "
            f"(raw max {roundtrip['raw_max_abs']:.9g}, "
            f"translation {roundtrip['max_translation_m']}, "
            f"rotation {roundtrip['max_rotation_rad']})"
        )
    if not references_preserved:
        raise RuntimeError(
            "goal sampling replaced task success/failure/grasp registry objects"
        )
    if not grasp_state_preserved:
        raise RuntimeError(
            "goal sampling changed gripper grasp membership or object parents"
        )
    if not waypoint_cache_preserved:
        raise RuntimeError("goal sampling changed the task waypoint cache")
    if goal_pose is None:
        detail = last_placement_error or "no valid root goal was sampled"
        raise RuntimeError(
            "could not sample a preserve-instance task-root goal after "
            f"{max_attempts} attempts: {detail}"
        )
    if goal_collision_pairs is None:  # pragma: no cover - guarded by goal_pose
        raise RuntimeError("goal collision-pair evidence is unavailable")

    selected_new_collision_pairs = tuple(
        sorted(frozenset(goal_collision_pairs) - source_collision_pair_set)
    )
    if selected_new_collision_pairs:  # pragma: no cover - loop rejects these
        raise RuntimeError("selected goal contains an unvalidated collision pair")

    preservation = {
        "initialized_episode_preserved": True,
        "task_init_episode_called": False,
        "task_validate_called": False,
        "low_dim_state_roundtrip_preserved": state_preserved,
        "low_dim_state_roundtrip_comparison_mode": roundtrip["comparison_mode"],
        "low_dim_state_roundtrip_chunk_count": roundtrip["chunk_count"],
        "low_dim_state_roundtrip_l2": roundtrip["raw_l2"],
        "low_dim_state_roundtrip_max_abs": roundtrip["raw_max_abs"],
        "low_dim_state_roundtrip_max_translation_m": roundtrip[
            "max_translation_m"
        ],
        "low_dim_state_roundtrip_max_rotation_rad": roundtrip[
            "max_rotation_rad"
        ],
        "condition_and_grasp_registry_identity_preserved": references_preserved,
        "gripper_grasp_membership_and_parentage_preserved": (
            grasp_state_preserved
        ),
        "configuration_tree_rollback": "task_only_after_each_attempt_and_outer_finally",
        "task_configuration_tree_restored": True,
        "live_robot_state_untouched": True,
        "live_robot_configuration_trees_accessed": False,
        "robot_collision_pair_policy": (
            "reject_candidate_external_pairs_absent_at_source"
        ),
        "robot_collision_pair_granularity": (
            "named_arm_collection_x_external_collidable_scene_shape"
        ),
        "source_robot_external_collision_pairs": _collision_pair_records(
            source_collision_pairs
        ),
        "goal_robot_external_collision_pairs": _collision_pair_records(
            goal_collision_pairs
        ),
        "goal_new_robot_external_collision_pairs": _collision_pair_records(
            selected_new_collision_pairs
        ),
        "sampling_attempts_rejected_for_new_robot_collision_pairs": (
            new_collision_pair_rejections
        ),
        "sampling_attempts": attempts_used,
        "waypoint_cache_identity_preserved": waypoint_cache_preserved,
    }
    return source_pose, goal_pose, preservation


@dataclass
class ScenarioController:
    """Move a task root while preserving the already initialized episode.

    In the pinned fork, both ``Scene.kidnap`` and
    ``Scene.move_task_smoothly`` call ``task.init_episode`` while choosing a
    destination.  That changes task-internal random objects and success
    conditions, so neither method is a valid motion intervention on one
    episode instance. This controller samples a workspace-fitting
    ``boundary_root`` pose, rejects external robot collision pairs absent at
    the source, restores only the task hierarchy transactionally, and then
    applies teleportation or interpolation itself. It never runs waypoint
    validation against, or restores configuration trees for, the live robot.
    """

    kind: Literal["static", "teleport_task", "smooth_task_motion"]
    trigger_fraction: float = 1.0 / 3.0
    total_steps: int = 10
    max_attempts: int = 20
    verify_instance: bool = True
    _teleported: bool = False
    _smooth_calls: int = 0
    _smooth_complete: bool = False
    _motion_source_pose: Any = None
    _motion_goal_pose: Any = None
    _instance_preservation: Any = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must lie in [0, 1]")
        if self.total_steps < 1 or self.max_attempts < 1:
            raise ValueError("total_steps and max_attempts must be positive")
        if self.verify_instance is not True:
            raise ValueError("preserve-instance auditing cannot be disabled")

    def protocol_metadata(self) -> dict[str, Any]:
        """Return JSON-stable semantics shared by both evaluator frontends."""

        return {
            "protocol_id": PRESERVE_INSTANCE_MOTION_PROTOCOL_ID,
            "episode_instance_semantics": "preserve_initialized_episode",
            "goal_object": "task.boundary_root()",
            "goal_sampling": "scene_workspace_boundary_without_task_reinitialization",
            "sampling_rollback": "task_configuration_tree_only_live_robot_untouched",
            "sampling_rollback_frequency": "after_each_attempt_and_outer_finally",
            "task_configuration_tree_restore_api": "Task.get_state/restore_state",
            "task_tree_object_count_guard": True,
            "live_robot_state_during_goal_sampling": "untouched",
            "live_robot_configuration_tree_access": "none",
            "online_task_waypoint_validation": "disabled_to_preserve_live_robot_state",
            "calls_task_validate": False,
            "grasp_membership_and_parentage_audited": True,
            "robot_collision_validation": (
                "reject_candidate_external_pairs_absent_at_source"
            ),
            "robot_collision_pair_granularity": (
                "named_arm_collection_x_external_collidable_scene_shape"
            ),
            "source_robot_contacts_allowed": True,
            "grasped_tool_collision_semantics": (
                "current_arm_collection_membership_without_task_filters"
            ),
            "self_collision_semantics": (
                "current_arm_collection_members_excluded_matching_all_other"
            ),
            "low_dim_state_roundtrip_comparison": (
                "valid_pose_chunks_sign_invariant_else_scalar_max_abs"
            ),
            "low_dim_state_roundtrip_scalar_tolerance": (
                LOW_DIM_STATE_ROUNDTRIP_ATOL
            ),
            "low_dim_state_roundtrip_pose_translation_tolerance_m": (
                LOW_DIM_POSE_TRANSLATION_TOLERANCE_M
            ),
            "low_dim_state_roundtrip_pose_rotation_tolerance_rad": (
                LOW_DIM_POSE_ROTATION_TOLERANCE_RAD
            ),
            "root_application_validation": (
                "planned_motion_and_actual_motion_and_commanded_pose_reached"
            ),
            "root_actual_motion_translation_tolerance_m": (
                ROOT_ACTUAL_MOTION_TRANSLATION_TOLERANCE_M
            ),
            "root_actual_motion_rotation_tolerance_rad": (
                ROOT_ACTUAL_MOTION_ROTATION_TOLERANCE_RAD
            ),
            "root_command_translation_tolerance_m": (
                ROOT_COMMAND_TRANSLATION_TOLERANCE_M
            ),
            "root_command_rotation_tolerance_rad": (
                ROOT_COMMAND_ROTATION_TOLERANCE_RAD
            ),
            "dynamic_state_note": (
                "the task configuration tree restores task poses and joints and "
                "resets task dynamics; live robot trees remain untouched; the "
                "subsequent root-motion intervention resets moved task dynamics"
            ),
            "goal_validation": (
                "workspace_fit_no_new_robot_external_collision_pairs"
            ),
            "calls_task_init_episode": False,
            "calls_scene_kidnap": False,
            "calls_scene_move_task_smoothly": False,
            "smooth_schedule": "fractions_1_over_n_through_n_over_n",
            "smooth_endpoint_validation": "final_goal_pose_reached",
            "smooth_endpoint_guaranteed": True,
        }

    def _ensure_motion_plan(self, scene: Any) -> None:
        if self._motion_goal_pose is not None:
            return
        source, goal, preservation = _sample_preserving_instance_goal(
            scene,
            max_attempts=self.max_attempts,
        )
        self._motion_source_pose = source
        self._motion_goal_pose = goal
        self._instance_preservation = preservation

    def apply(self, task_environment: Any, *, step: int, horizon: int) -> dict[str, Any]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        trigger = min(horizon - 1, int(round(self.trigger_fraction * (horizon - 1))))
        event: dict[str, Any] = {
            "kind": self.kind,
            "step": step,
            "trigger_step": trigger,
            "applied": False,
            "motion_protocol": self.protocol_metadata(),
        }
        if self.kind == "static" or step < trigger:
            return event
        if self.kind == "teleport_task" and self._teleported:
            return event
        if self.kind == "smooth_task_motion" and self._smooth_complete:
            return event
        scene = getattr(task_environment, "_scene", None)
        if scene is None:
            raise RuntimeError("author RLBench TaskEnvironment._scene is unavailable")
        task = scene.task
        root = task.boundary_root()
        before_state = _task_low_dim_state(task)
        before_root = np.asarray(root.get_pose(), dtype=np.float64)
        if self.kind == "teleport_task":
            self._ensure_motion_plan(scene)
            commanded_pose = np.asarray(self._motion_goal_pose, dtype=np.float64)
            root.set_pose(commanded_pose)
            self._teleported = True
            event["applied"] = True
            after_state = _task_low_dim_state(task)
            after_root = np.asarray(root.get_pose(), dtype=np.float64)
            event.update(
                _intervention_change(before_state, after_state, before_root, after_root)
            )
            event.update(
                _root_motion_metrics(self._motion_source_pose, self._motion_goal_pose)
            )
            event.update(
                _root_application_metrics(before_root, commanded_pose, after_root)
            )
            event.update(_root_goal_reached_metrics(commanded_pose, after_root))
            event["protocol_effective"] = bool(
                event["planned_root_motion"]
                and event["actual_root_motion"]
                and event["commanded_root_pose_reached"]
            )
            event["instance_preservation"] = dict(self._instance_preservation)
            return event
        if self.kind == "smooth_task_motion":
            self._ensure_motion_plan(scene)
            self._smooth_calls += 1
            fraction = min(self._smooth_calls / float(self.total_steps), 1.0)
            self._smooth_complete = self._smooth_calls >= self.total_steps
            if self._smooth_complete:
                # Use the sampled value verbatim at the endpoint; interpolation
                # roundoff must not leave the task fractionally short of goal.
                next_pose = np.asarray(self._motion_goal_pose, dtype=np.float64)
            else:
                next_pose = _interpolate_rlbench_pose(
                    self._motion_source_pose,
                    self._motion_goal_pose,
                    fraction,
                )
            root.set_pose(next_pose)
            after_state = _task_low_dim_state(task)
            after_root = np.asarray(root.get_pose(), dtype=np.float64)
            application = _root_application_metrics(
                before_root,
                next_pose,
                after_root,
            )
            goal_reached = _root_goal_reached_metrics(
                self._motion_goal_pose,
                after_root,
            )
            event.update(
                {
                    "applied": True,
                    "smooth_call": self._smooth_calls,
                    "complete": self._smooth_complete,
                    "endpoint_applied": bool(
                        self._smooth_complete
                        and goal_reached["goal_root_pose_reached"]
                    ),
                    "endpoint_fraction": fraction,
                    "instance_preservation": dict(self._instance_preservation),
                }
            )
            event.update(
                _intervention_change(
                    before_state,
                    after_state,
                    before_root,
                    after_root,
                )
            )
            event.update(
                _root_motion_metrics(self._motion_source_pose, self._motion_goal_pose)
            )
            event.update(application)
            event.update(goal_reached)
            event["protocol_effective"] = bool(
                event["planned_root_motion"]
                and event["actual_root_motion"]
                and event["commanded_root_pose_reached"]
            )
            return event
        raise ValueError(f"unsupported scenario kind: {self.kind}")


def _intervention_change(
    before_state: Array,
    after_state: Array,
    before_root: Array,
    after_root: Array,
) -> dict[str, Any]:
    if before_state.shape != after_state.shape:
        raise RuntimeError("dynamic intervention changed task-state schema")
    state_l2 = float(np.linalg.norm(after_state - before_state))
    root_l2 = float(np.linalg.norm(after_root - before_root))
    return {
        "task_state_l2": state_l2,
        "task_state_changed": bool(state_l2 > 1.0e-9),
        "root_pose_l2": root_l2,
        "root_pose_changed": bool(root_l2 > 1.0e-9),
    }
