"""Policy-independent physical fault injection for RLBench evaluation.

The core closed-loop policy never imports this module.  It wraps an RLBench
``TaskEnvironment`` and changes only the action or physical attachment seen by
the benchmark.  Baseline and ablated policies can therefore receive the same
fault protocol without changing their internal state or observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import numpy as np


class FaultInjectionKind(str, Enum):
    TIME_STALL = "time_stall"
    GRASP_FAILURE = "grasp_failure"
    RELATION_MISMATCH = "relation_mismatch"
    UNEXPECTED_DROP = "unexpected_drop"


@dataclass(frozen=True)
class FaultInjectionSpec:
    """One preregistered, one-shot physical fault.

    ``earliest_step`` is only an eligibility floor.  The actual trigger is a
    physical/action predicate shared by every compared policy:

    - time stall: the selected arm requests meaningful Cartesian motion;
    - grasp failure: the selected arm requests the chosen close occurrence;
    - relation faults: the selected gripper has carried an attached object for
      ``minimum_grasped_cycles`` consecutive policy cycles.
    """

    kind: FaultInjectionKind
    arm: Literal["single", "left", "right", "all"] = "single"
    earliest_step: int = 0
    duration_cycles: int = 8
    motion_trigger_distance: float = 0.005
    close_occurrence: int = 1
    minimum_grasped_cycles: int = 3
    mismatch_translation: tuple[float, float, float] = (0.04, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FaultInjectionKind):
            object.__setattr__(self, "kind", FaultInjectionKind(self.kind))
        if self.arm not in {"single", "left", "right", "all"}:
            raise ValueError("fault arm must be single/left/right/all")
        for value in (
            self.earliest_step,
            self.duration_cycles,
            self.close_occurrence,
            self.minimum_grasped_cycles,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("fault cycle and occurrence values must be integers")
        if self.duration_cycles < 1 or self.close_occurrence < 1:
            raise ValueError("fault duration and close occurrence must be positive")
        if self.minimum_grasped_cycles < 1:
            raise ValueError("minimum grasped cycles must be positive")
        if (
            not np.isfinite(self.motion_trigger_distance)
            or self.motion_trigger_distance <= 0
        ):
            raise ValueError("motion trigger distance must be finite and positive")
        translation = np.asarray(self.mismatch_translation, dtype=np.float64)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("mismatch translation must contain three finite values")
        object.__setattr__(
            self, "mismatch_translation", tuple(float(x) for x in translation)
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FaultInjectionSpec":
        known = {
            "kind",
            "arm",
            "earliest_step",
            "duration_cycles",
            "motion_trigger_distance",
            "close_occurrence",
            "minimum_grasped_cycles",
            "mismatch_translation",
        }
        unknown = set(value).difference(known)
        if unknown:
            raise ValueError(f"unknown fault configuration fields: {sorted(unknown)}")
        payload = dict(value)
        payload["kind"] = FaultInjectionKind(payload["kind"])
        if "mismatch_translation" in payload:
            payload["mismatch_translation"] = tuple(payload["mismatch_translation"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["mismatch_translation"] = list(self.mismatch_translation)
        return value


class FaultInjectingTaskEnvironment:
    """Transparent one-shot fault wrapper around an RLBench task environment.

    Formal dynamic-background evaluation can require the external scene-motion
    controller to finish before the physical fault becomes eligible.  The
    evaluator reports those background events through ``configure_background``
    and ``record_background_event``.  This handshake is deliberately outside
    the policy: it never reads or mutates StateId, progress, relation beliefs,
    or execution modes.
    """

    def __init__(
        self,
        task_environment: Any,
        spec: FaultInjectionSpec,
        *,
        background_required: bool = False,
    ) -> None:
        self._environment = task_environment
        self.spec = spec
        self.events: list[dict[str, Any]] = []
        self._policy_step = 0
        self._triggered = False
        self._active_cycles = 0
        self._close_occurrences = 0
        self._previous_close_request = False
        self._current_close_counted = False
        self._current_close_eligible = False
        self._failed_close_active = False
        self._failed_grasp_target_poses: dict[str, tuple[Any, np.ndarray]] = {}
        self._stable_interaction_signature: tuple[str, tuple[str, ...]] | None = None
        self._stable_interaction_cycles = 0
        self._relation_trigger_source: str | None = None
        self._restored_contact_cycles = 0
        self._fault_arm: str | None = None
        self._released_objects: dict[str, Any] = {}
        self._released_object_positions: dict[str, np.ndarray] = {}
        self._effect_observed: bool | None = None
        self._effect_policy_step: int | None = None
        self._fault_end_policy_step: int | None = None
        self._relation_restored: bool | None = None
        self._relation_restoration_policy_step: int | None = None
        self._background_required = bool(background_required)
        self._background_configured = False
        self._background_scenario: str | None = None
        self._background_expected_segments = 0
        self._background_completed_segments: set[str] = set()
        self._background_events: list[dict[str, Any]] = []
        self._background_ready = not self._background_required
        self._background_ready_policy_step: int | None = None
        self._fault_background_ready_at_trigger: bool | None = None
        self._pre_background_policy_steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    @property
    def triggered(self) -> bool:
        return self._triggered

    def configure_background(self, *, scenario: str, expected_segments: int) -> None:
        """Declare the policy-independent dynamic background contract.

        A segment is one independently scheduled scene-root motion.  Ordinary
        tasks have one segment; task-scoped multi-entity plans may have more.
        Configuration must happen before the first action is executed.
        """

        if self._policy_step != 0 or self._background_configured:
            raise RuntimeError(
                "fault background must be configured exactly once before stepping"
            )
        if scenario != "smooth":
            raise ValueError("formal fault background must use the smooth scenario")
        if (
            isinstance(expected_segments, bool)
            or not isinstance(expected_segments, int)
            or expected_segments < 1
        ):
            raise ValueError("dynamic fault background needs at least one segment")
        self._background_configured = True
        self._background_scenario = scenario
        self._background_expected_segments = expected_segments
        self._background_ready = False

    def record_background_event(self, event: Mapping[str, Any]) -> None:
        """Record one externally applied scene-motion tick.

        Only an effective, completed segment opens fault eligibility.  Partial
        interpolation ticks remain auditable but cannot release a fault.
        """

        if not self._background_required:
            return
        if not self._background_configured:
            raise RuntimeError("dynamic background event arrived before configuration")
        if not isinstance(event, Mapping) or event.get("applied") is not True:
            raise ValueError("background notification must describe an applied event")
        record = dict(event)
        self._background_events.append(record)
        if (
            event.get("complete") is not True
            or event.get("protocol_effective") is False
        ):
            return
        segment = str(event.get("entity", "task_root"))
        self._background_completed_segments.add(segment)
        if (
            len(self._background_completed_segments)
            >= self._background_expected_segments
        ):
            self._background_ready = True
            if self._background_ready_policy_step is None:
                self._background_ready_policy_step = self._policy_step

    def protocol_metadata(self) -> dict[str, Any]:
        relation_fault = self.spec.kind in {
            FaultInjectionKind.GRASP_FAILURE,
            FaultInjectionKind.RELATION_MISMATCH,
            FaultInjectionKind.UNEXPECTED_DROP,
        }
        return {
            "schema": "essay2608.rlbench.physical_fault.v2",
            "spec": self.spec.to_dict(),
            "triggered": self._triggered,
            "events": list(self.events),
            "policy_steps_observed": self._policy_step,
            "policy_state_mutated": False,
            "observation_hidden": False,
            "background": {
                "required": self._background_required,
                "configured": self._background_configured,
                "scenario": self._background_scenario,
                "expected_segments": self._background_expected_segments,
                "completed_segments": sorted(self._background_completed_segments),
                "ready": self._background_ready,
                "ready_policy_step": self._background_ready_policy_step,
                "ready_before_fault": self._fault_background_ready_at_trigger,
                "pre_background_policy_steps": self._pre_background_policy_steps,
                "events": list(self._background_events),
            },
            "physical_audit": {
                "effect_observed": self._effect_observed,
                "effect_policy_step": self._effect_policy_step,
                "fault_end_policy_step": self._fault_end_policy_step,
                "target_arm": self._fault_arm,
                "target_objects": sorted(self._released_objects),
                "relation_restored": (
                    self._relation_restored
                    if relation_fault and self._triggered
                    else None
                ),
                "relation_restoration_policy_step": (
                    self._relation_restoration_policy_step
                ),
                "cycles_to_relation_restoration": (
                    None
                    if self._relation_restoration_policy_step is None
                    else self._relation_restoration_policy_step
                    - int(self.events[0]["policy_step"])
                ),
            },
        }

    @staticmethod
    def _is_bimanual_action(action: np.ndarray) -> bool:
        if action.shape == (18,):
            return True
        if action.shape == (9,):
            return False
        raise ValueError(
            f"unsupported RLBench action shape for fault injection: {action.shape}"
        )

    @staticmethod
    def _arm_slice(arm: str, *, bimanual: bool) -> tuple[slice, int]:
        if not bimanual:
            if arm not in {"single", "all"}:
                raise ValueError("unimanual fault requires arm=single/all")
            return slice(0, 7), 7
        # The pinned bimanual fork uses right-first 18D actions.
        if arm == "right":
            return slice(0, 7), 7
        if arm == "left":
            return slice(9, 16), 16
        raise ValueError("bimanual per-arm operation requires left/right")

    @staticmethod
    def _observation_pose(observation: Any, arm: str, *, bimanual: bool) -> np.ndarray:
        if not bimanual:
            return np.asarray(observation.gripper_pose, dtype=np.float64)
        return np.asarray(getattr(observation, arm).gripper_pose, dtype=np.float64)

    def _selected_arms(self, *, bimanual: bool) -> tuple[str, ...]:
        if not bimanual:
            if self.spec.arm not in {"single", "all"}:
                raise ValueError("unimanual fault requires arm=single/all")
            return ("single",)
        if self.spec.arm == "all":
            return ("right", "left")
        if self.spec.arm not in {"left", "right"}:
            raise ValueError("bimanual fault requires arm=left/right/all")
        return (self.spec.arm,)

    def _robot(self) -> Any:
        scene = getattr(self._environment, "_scene", None)
        robot = getattr(scene, "robot", None)
        if robot is None:
            raise RuntimeError("RLBench task environment does not expose scene.robot")
        return robot

    def _gripper(self, arm: str, *, bimanual: bool) -> Any:
        robot = self._robot()
        return getattr(robot, f"{arm}_gripper") if bimanual else robot.gripper

    @staticmethod
    def _object_names(objects: Sequence[Any]) -> list[str]:
        names = []
        for obj in objects:
            getter = getattr(obj, "get_name", None)
            names.append(str(getter()) if callable(getter) else type(obj).__name__)
        return names

    def _trigger_event(self, kind: str, **fields: Any) -> None:
        self._triggered = True
        self._fault_background_ready_at_trigger = self._background_ready
        self.events.append(
            {
                "kind": kind,
                "policy_step": self._policy_step,
                "protocol_effective": True,
                **fields,
            }
        )

    def _apply_time_stall(
        self,
        action: np.ndarray,
        observation: Any,
        *,
        bimanual: bool,
    ) -> np.ndarray:
        selected = self._selected_arms(bimanual=bimanual)
        if not self._triggered and self._policy_step >= self.spec.earliest_step:
            distances = {}
            for arm in selected:
                pose_slice, _ = self._arm_slice(arm, bimanual=bimanual)
                current = self._observation_pose(observation, arm, bimanual=bimanual)
                distances[arm] = float(
                    np.linalg.norm(action[pose_slice][:3] - current[:3])
                )
            if any(
                value >= self.spec.motion_trigger_distance
                for value in distances.values()
            ):
                self._active_cycles = self.spec.duration_cycles
                self._trigger_event(
                    self.spec.kind.value,
                    arms=list(selected),
                    requested_translation_distance=distances,
                    duration_cycles=self.spec.duration_cycles,
                )
        if self._active_cycles <= 0:
            return action
        stalled = action.copy()
        for arm in selected:
            pose_slice, _ = self._arm_slice(arm, bimanual=bimanual)
            stalled[pose_slice] = self._observation_pose(
                observation, arm, bimanual=bimanual
            )
        self._active_cycles -= 1
        self._effect_observed = True
        self._effect_policy_step = (
            self._policy_step
            if self._effect_policy_step is None
            else self._effect_policy_step
        )
        self.events.append(
            {
                "kind": "time_stall_cycle",
                "policy_step": self._policy_step,
                "remaining_cycles": self._active_cycles,
                "protocol_effective": True,
            }
        )
        if self._active_cycles == 0:
            self._fault_end_policy_step = self._policy_step
            self.events.append(
                {
                    "kind": "time_stall_ended",
                    "policy_step": self._policy_step,
                    "protocol_effective": True,
                }
            )
        return stalled

    def _apply_grasp_failure(self, action: np.ndarray, *, bimanual: bool) -> np.ndarray:
        selected = self._selected_arms(bimanual=bimanual)
        if len(selected) != 1:
            raise ValueError("grasp failure requires one selected arm")
        _pose_slice, gripper_index = self._arm_slice(selected[0], bimanual=bimanual)
        close_requested = bool(action[gripper_index] <= 0.5)
        if close_requested and not self._previous_close_request:
            self._current_close_counted = False
            self._current_close_eligible = (
                self._policy_step >= self.spec.earliest_step
            )
        self._previous_close_request = close_requested
        detected_targets: tuple[Any, ...] = ()
        if (
            close_requested
            and not self._current_close_counted
            and self._current_close_eligible
        ):
            detected_targets = self._detected_grasp_targets(
                arm=selected[0],
                bimanual=bimanual,
            )
            if detected_targets:
                # ``close_occurrence`` counts physical interaction attempts,
                # not arbitrary close-valued commands in free space.  A
                # target may first enter the sensor after the fingers have
                # already begun closing, so the occurrence is counted on the
                # first detected cycle of that same close interval.
                self._close_occurrences += 1
                self._current_close_counted = True
        if (
            not self._triggered
            and self._policy_step >= self.spec.earliest_step
            and self._close_occurrences == self.spec.close_occurrence
            and close_requested
        ):
            self._fault_arm = selected[0]
            self._capture_failed_grasp_targets(
                bimanual=bimanual,
                detected_targets=detected_targets or None,
            )
            # A commanded close is not by itself a physical interaction.  Do
            # not mark the formal fault triggered until the selected gripper
            # actually detects a dynamic task target for this occurrence.
            if not self._released_objects:
                return action
            self._failed_close_active = True
            self._relation_restored = False
            self._trigger_event(
                self.spec.kind.value,
                arm=selected[0],
                close_occurrence=self._close_occurrences,
                target_objects=sorted(self._released_objects),
                end_condition="explicit_open_then_new_close_occurrence",
            )
        if not self._failed_close_active:
            if not close_requested:
                self._current_close_counted = False
                self._current_close_eligible = False
            return action
        if not close_requested:
            self._failed_close_active = False
            self._current_close_counted = False
            self._current_close_eligible = False
            self._fault_end_policy_step = self._policy_step
            self.events.append(
                {
                    "kind": "grasp_failure_occurrence_ended",
                    "policy_step": self._policy_step,
                    "arm": selected[0],
                    "protocol_effective": True,
                }
            )
            return action
        # The physically detected targets were frozen on the trigger cycle.
        # Re-scanning the whole task tree on every continued close command is
        # both unnecessary and can dominate simulator runtime.
        if not self._released_objects:
            self._capture_failed_grasp_targets(bimanual=bimanual)
        # Preserve the commanded gripper motion while suppressing only the
        # attachment it is meant to establish.  Replacing the close command
        # with an open command conflates a missed interaction with gripper
        # actuator failure: the task executor then correctly waits for its
        # unapplied discrete command and never reaches the relation evidence
        # that this fault is intended to test.
        self.events.append(
            {
                "kind": "grasp_attachment_suppressed_cycle",
                "policy_step": self._policy_step,
                "arm": selected[0],
                "target_objects": sorted(self._released_objects),
                "protocol_effective": bool(self._released_objects),
            }
        )
        return action

    def _detected_grasp_targets(
        self,
        *,
        arm: str,
        bimanual: bool,
    ) -> tuple[Any, ...]:
        """Return dynamic task objects physically detected by one gripper."""

        scene = getattr(self._environment, "_scene", None)
        task = getattr(scene, "task", None)
        graspables_fn = getattr(task, "get_graspable_objects", None)
        if not callable(graspables_fn):
            raise RuntimeError("grasp failure requires task graspable-object access")
        gripper = self._gripper(arm, bimanual=bimanual)
        sensor = getattr(gripper, "_proximity_sensor", None)
        detects = getattr(sensor, "is_detected", None)
        if not callable(detects):
            raise RuntimeError("grasp failure requires gripper proximity detection")
        candidates = list(graspables_fn())
        # Some RLBench interactions are contact grasps rather than simulator
        # attachments (for example a gripper closing on a tray handle).  Such
        # an object need not appear in ``get_graspable_objects`` even though it
        # is the physical target of the close.  Add only dynamic, respondable
        # task-tree shapes detected by the same gripper proximity sensor.  The
        # rule is task-independent and keeps static tables/fixtures out.
        get_base = getattr(task, "get_base", None)
        if callable(get_base):
            base = get_base()
            get_tree = getattr(base, "get_objects_in_tree", None)
            if callable(get_tree):
                for obj in get_tree(exclude_base=False):
                    is_dynamic = getattr(obj, "is_dynamic", None)
                    is_respondable = getattr(obj, "is_respondable", None)
                    if (
                        callable(is_dynamic)
                        and callable(is_respondable)
                        and bool(is_dynamic())
                        and bool(is_respondable())
                    ):
                        candidates.append(obj)
        detected: list[Any] = []
        seen: set[int] = set()
        for obj in candidates:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if bool(detects(obj)):
                detected.append(obj)
        return tuple(detected)

    def _capture_failed_grasp_targets(
        self,
        *,
        bimanual: bool,
        detected_targets: Sequence[Any] | None = None,
    ) -> None:
        """Record objects physically eligible for the selected close occurrence.

        A missed interaction suppresses the selected gripper's attachment while
        leaving its commanded closing motion intact.  An unattached object in
        the fingers can otherwise be carried by contact friction and create a
        false kinematic LINK.  We therefore preserve the pre-interaction pose
        of detected, currently unowned graspable objects until the policy opens
        and begins a genuinely new close occurrence.  Objects held by the peer
        arm (handover) remain under that arm's physical control and are not
        pose-locked here.
        """

        if self._fault_arm is None:
            raise RuntimeError("grasp failure was activated without a target arm")
        peer_owned_ids: set[int] = set()
        if bimanual:
            peer = "left" if self._fault_arm == "right" else "right"
            peer_owned_ids.update(
                id(obj)
                for obj in self._gripper(
                    peer,
                    bimanual=True,
                ).get_grasped_objects()
            )
        candidates = (
            tuple(detected_targets)
            if detected_targets is not None
            else self._detected_grasp_targets(
                arm=self._fault_arm,
                bimanual=bimanual,
            )
        )
        for obj in candidates:
            name = self._object_names((obj,))[0]
            self._released_objects[name] = obj
            if id(obj) in peer_owned_ids:
                continue
            if name in self._failed_grasp_target_poses:
                continue
            get_pose = getattr(obj, "get_pose", None)
            if not callable(get_pose):
                raise RuntimeError("grasp-failure target lacks pose observation")
            pose = np.asarray(get_pose(), dtype=np.float64)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise RuntimeError("grasp-failure target pose must be finite [7]")
            self._failed_grasp_target_poses[name] = (obj, pose.copy())

    def _restore_failed_grasp_targets(self, *, bimanual: bool) -> bool:
        """Undo contact-only carriage during one deliberately missed grasp."""

        if not self._failed_close_active or self._fault_arm is None:
            return False
        gripper = self._gripper(self._fault_arm, bimanual=bimanual)
        target_ids = {
            id(obj) for obj, _pose in self._failed_grasp_target_poses.values()
        }
        if target_ids.intersection(id(obj) for obj in gripper.get_grasped_objects()):
            gripper.release()
        restored = False
        for obj, pose in self._failed_grasp_target_poses.values():
            setter = getattr(obj, "set_pose", None)
            if not callable(setter):
                raise RuntimeError("grasp-failure target lacks pose mutation API")
            setter(pose)
            restored = True
        if restored:
            self._effect_observed = True
            if self._effect_policy_step is None:
                self._effect_policy_step = self._policy_step
        return restored

    def _set_grasp_attachment_suppressed(
        self,
        arm: str,
        suppressed: bool,
    ) -> frozenset[str]:
        """Temporarily suppress attachment for exactly one selected gripper.

        The formal executor uses ``DiscreteGripperProtocol`` whose concrete
        action mode exposes an episode-local suppression set.  This avoids a
        left-arm fault disabling a simultaneous right-arm grasp while changing
        neither policy state, observations, gripper targets, nor task progress.
        """

        action_mode = getattr(self._environment, "_action_mode", None)
        gripper_mode = getattr(action_mode, "gripper_action_mode", None)
        if gripper_mode is None or not hasattr(gripper_mode, "_attach_grasped_objects"):
            raise RuntimeError(
                "grasp failure requires the aligned gripper attachment protocol"
            )
        previous = frozenset(
            getattr(gripper_mode, "_dynamac_attachment_suppressed_arms", ())
        )
        current = set(previous)
        if suppressed:
            current.add(arm)
        else:
            current.discard(arm)
        gripper_mode._dynamac_attachment_suppressed_arms = current
        return previous

    def _restore_grasp_attachment_suppression(
        self,
        previous: frozenset[str],
    ) -> None:
        action_mode = getattr(self._environment, "_action_mode", None)
        gripper_mode = getattr(action_mode, "gripper_action_mode", None)
        if gripper_mode is None:
            raise RuntimeError("grasp failure lost the gripper action mode")
        gripper_mode._dynamac_attachment_suppressed_arms = set(previous)

    @staticmethod
    def _gripper_is_open(gripper: Any) -> bool:
        getter = getattr(gripper, "get_open_amount", None)
        if not callable(getter):
            raise RuntimeError("relation fault requires gripper opening observation")
        values = np.asarray(getter(), dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise RuntimeError("relation fault gripper opening must be finite")
        return bool(np.all(values > 0.9))

    def _apply_relation_fault(
        self,
        action: np.ndarray,
        *,
        bimanual: bool,
    ) -> bool:
        selected = self._selected_arms(bimanual=bimanual)
        if len(selected) != 1:
            raise ValueError("relation faults require one selected arm")
        gripper = self._gripper(selected[0], bimanual=bimanual)
        grasped = list(gripper.get_grasped_objects())
        _pose_slice, gripper_index = self._arm_slice(selected[0], bimanual=bimanual)
        close_requested = bool(action[gripper_index] <= 0.5)
        if grasped:
            targets = tuple(grasped)
            source = "attachment"
        elif close_requested:
            targets = self._detected_grasp_targets(
                arm=selected[0],
                bimanual=bimanual,
            )
            source = "maintained_contact"
        else:
            targets = ()
            source = "maintained_contact"
        names = tuple(sorted(self._object_names(targets)))
        signature = (source, names) if names else None
        if signature is None:
            self._stable_interaction_signature = None
            self._stable_interaction_cycles = 0
        elif signature == self._stable_interaction_signature:
            self._stable_interaction_cycles += 1
        else:
            self._stable_interaction_signature = signature
            self._stable_interaction_cycles = 1
        if (
            self._triggered
            or self._policy_step < self.spec.earliest_step
            or self._stable_interaction_cycles < self.spec.minimum_grasped_cycles
        ):
            return False
        self._fault_arm = selected[0]
        self._relation_trigger_source = source
        self._released_objects = {self._object_names((obj,))[0]: obj for obj in targets}
        self._released_object_positions = {}
        for name, obj in self._released_objects.items():
            get_position = getattr(obj, "get_position", None)
            if callable(get_position):
                self._released_object_positions[name] = np.asarray(
                    get_position(), dtype=np.float64
                ).copy()
        self._relation_restored = False
        gripper.release()
        displaced = False
        if self.spec.kind == FaultInjectionKind.RELATION_MISMATCH:
            delta = np.asarray(self.spec.mismatch_translation, dtype=np.float64)
            for obj in targets:
                get_position = getattr(obj, "get_position", None)
                set_position = getattr(obj, "set_position", None)
                if not callable(get_position) or not callable(set_position):
                    raise RuntimeError(
                        "relation mismatch object lacks pose mutation API"
                    )
                set_position(np.asarray(get_position(), dtype=np.float64) + delta)
            displaced = True
        self._trigger_event(
            self.spec.kind.value,
            arm=selected[0],
            released_objects=list(names),
            interaction_source=source,
            stable_interaction_cycles=self._stable_interaction_cycles,
            displaced=displaced,
            mismatch_translation=(
                list(self.spec.mismatch_translation) if displaced else None
            ),
        )
        return True

    def _force_release_action(
        self, action: np.ndarray, *, bimanual: bool
    ) -> np.ndarray:
        """Open the selected gripper on the release cycle.

        ``gripper.release()`` removes the simulator attachment, while the open
        command prevents the policy's still-closed command from immediately
        re-attaching the same object in that environment step.  This is the
        executable meaning of the preregistered physical ``release`` and does
        not alter policy state or observations.
        """

        if self._fault_arm is None:
            raise RuntimeError("relation fault was triggered without a target arm")
        _pose_slice, gripper_index = self._arm_slice(self._fault_arm, bimanual=bimanual)
        released = action.copy()
        released[gripper_index] = 1.0
        return released

    def _authorize_forced_release(self, *, bimanual: bool) -> None:
        """Let an exogenous relation fault execute its one-cycle release.

        The shared executor may withhold a policy-supplied gripper command until
        its Cartesian target is complete.  A physical fault intervention is not
        such a policy command: its release must occur at the registered physical
        trigger even when the current task action is not gripper-authorized.
        Preserve the other arm's authorization and override only the selected
        arm for this one action-mode call.  The action mode consumes the value
        immediately, so no authorization leaks into later policy cycles.
        """

        if self._fault_arm is None:
            raise RuntimeError("relation fault was triggered without a target arm")
        action_mode = getattr(self._environment, "_action_mode", None)
        setter = getattr(action_mode, "set_policy_gripper_authorization", None)
        if not callable(setter):
            # Minimal test doubles and native action modes without the shared
            # authorization layer execute the overwritten gripper action
            # directly and therefore need no override.
            return
        current = getattr(action_mode, "_policy_gripper_authorization", None)
        if bimanual:
            if current is None:
                authorization = {"right": None, "left": None}
            elif isinstance(current, Mapping):
                authorization = {
                    "right": current.get("right"),
                    "left": current.get("left"),
                }
            else:
                raise RuntimeError(
                    "bimanual executor exposed invalid gripper authorization"
                )
            authorization[self._fault_arm] = True
        else:
            authorization = {"single": True}
        setter(authorization)
        self.events.append(
            {
                "kind": "physical_fault_gripper_authorization_override",
                "policy_step": self._policy_step,
                "arm": self._fault_arm,
                "protocol_effective": True,
            }
        )

    def _audit_physical_effect(self, *, bimanual: bool) -> None:
        if not self._triggered or self._fault_arm is None:
            return
        gripper = self._gripper(self._fault_arm, bimanual=bimanual)
        grasped = list(gripper.get_grasped_objects())
        grasped_names = set(self._object_names(grasped))

        if self.spec.kind == FaultInjectionKind.GRASP_FAILURE:
            target_names = set(self._released_objects)
            target_grasped = target_names.intersection(grasped_names)
            if self._failed_close_active and target_names and not target_grasped:
                self._effect_observed = True
                if self._effect_policy_step is None:
                    self._effect_policy_step = self._policy_step
            if (
                self._effect_observed
                and not self._failed_close_active
                and self._relation_restored is False
                and grasped_names
            ):
                self._relation_restored = True
                self._relation_restoration_policy_step = self._policy_step
                self.events.append(
                    {
                        "kind": "relation_restored",
                        "policy_step": self._policy_step,
                        "arm": self._fault_arm,
                        "attached_objects": sorted(grasped_names),
                        "protocol_effective": True,
                    }
                )
            return

        if self.spec.kind not in {
            FaultInjectionKind.RELATION_MISMATCH,
            FaultInjectionKind.UNEXPECTED_DROP,
        }:
            return

        target_names = set(self._released_objects)
        still_attached = sorted(target_names.intersection(grasped_names))
        detected_names = set(
            self._object_names(
                self._detected_grasp_targets(
                    arm=self._fault_arm,
                    bimanual=bimanual,
                )
            )
        )
        still_detected = sorted(target_names.intersection(detected_names))
        gripper_open = self._gripper_is_open(gripper)
        if self._effect_observed is None:
            displacement_by_object: dict[str, float | None] = {}
            for name, obj in self._released_objects.items():
                get_position = getattr(obj, "get_position", None)
                before = self._released_object_positions.get(name)
                displacement_by_object[name] = (
                    None
                    if before is None or not callable(get_position)
                    else float(
                        np.linalg.norm(
                            np.asarray(get_position(), dtype=np.float64) - before
                        )
                    )
                )
            detached = (
                not still_attached
                if self._relation_trigger_source == "attachment"
                else gripper_open
            )
            if self.spec.kind == FaultInjectionKind.RELATION_MISMATCH:
                expected = float(np.linalg.norm(self.spec.mismatch_translation))
                displaced = all(
                    value is not None and value >= max(0.0, expected - 0.01)
                    for value in displacement_by_object.values()
                )
            else:
                displaced = True
            self._effect_observed = detached and displaced
            self._effect_policy_step = self._policy_step
            self._fault_end_policy_step = self._policy_step
            self.events.append(
                {
                    "kind": "physical_fault_effect_audit",
                    "policy_step": self._policy_step,
                    "arm": self._fault_arm,
                    "detached": detached,
                    "interaction_source": self._relation_trigger_source,
                    "gripper_open": gripper_open,
                    "still_attached_objects": still_attached,
                    "still_detected_objects": still_detected,
                    "displacement_by_object": displacement_by_object,
                    "protocol_effective": self._effect_observed,
                }
            )
            return

        if self._relation_trigger_source == "attachment":
            restoration_evidence = bool(still_attached)
        else:
            contact_restored = not gripper_open and target_names.issubset(
                detected_names
            )
            self._restored_contact_cycles = (
                self._restored_contact_cycles + 1 if contact_restored else 0
            )
            restoration_evidence = (
                self._restored_contact_cycles >= self.spec.minimum_grasped_cycles
            )
        if (
            self._effect_observed
            and self._relation_restored is False
            and restoration_evidence
        ):
            self._relation_restored = True
            self._relation_restoration_policy_step = self._policy_step
            self.events.append(
                {
                    "kind": "relation_restored",
                    "policy_step": self._policy_step,
                    "arm": self._fault_arm,
                    "attached_objects": still_attached,
                    "contact_objects": still_detected,
                    "interaction_source": self._relation_trigger_source,
                    "protocol_effective": True,
                }
            )

    def step(self, action: Any):
        values = np.asarray(action, dtype=np.float64)
        bimanual = self._is_bimanual_action(values)
        if self._background_required and not self._background_configured:
            raise RuntimeError(
                "formal fault episode did not configure its dynamic background"
            )
        if not self._background_ready:
            result = self._environment.step(values)
            self._pre_background_policy_steps += 1
            self._policy_step += 1
            return result
        relation_fault_triggered = False
        if self.spec.kind in {
            FaultInjectionKind.RELATION_MISMATCH,
            FaultInjectionKind.UNEXPECTED_DROP,
        }:
            relation_fault_triggered = self._apply_relation_fault(
                values,
                bimanual=bimanual,
            )
        current_observation = self._environment.get_observation()
        applied = values
        if self.spec.kind == FaultInjectionKind.TIME_STALL:
            applied = self._apply_time_stall(
                values, current_observation, bimanual=bimanual
            )
        elif self.spec.kind == FaultInjectionKind.GRASP_FAILURE:
            applied = self._apply_grasp_failure(values, bimanual=bimanual)
        elif relation_fault_triggered:
            applied = self._force_release_action(values, bimanual=bimanual)
            self._authorize_forced_release(bimanual=bimanual)
        restore_attachment: frozenset[str] | None = None
        if (
            self.spec.kind == FaultInjectionKind.GRASP_FAILURE
            and self._failed_close_active
        ):
            assert self._fault_arm is not None
            restore_attachment = self._set_grasp_attachment_suppressed(
                self._fault_arm,
                True,
            )
        try:
            result = self._environment.step(applied)
        finally:
            if restore_attachment is not None:
                self._restore_grasp_attachment_suppression(restore_attachment)
        if self.spec.kind == FaultInjectionKind.GRASP_FAILURE:
            restored = self._restore_failed_grasp_targets(bimanual=bimanual)
            if restored:
                observation = self._environment.get_observation()
                result = (observation, result[1], result[2])
        self._audit_physical_effect(bimanual=bimanual)
        self._policy_step += 1
        return result

    def record_committed_fallback(self) -> None:
        """Advance the public policy-cycle clock after a raw joint-hold commit.

        The shared executor resolves an invalid Cartesian target through a raw
        physics hold that intentionally bypasses ``TaskEnvironment.step``.
        That transaction still consumes one policy cycle; without this hook a
        later physical fault would be scheduled against an artificially slow
        injector clock.
        """

        self._policy_step += 1


__all__ = [
    "FaultInjectionKind",
    "FaultInjectionSpec",
    "FaultInjectingTaskEnvironment",
]
