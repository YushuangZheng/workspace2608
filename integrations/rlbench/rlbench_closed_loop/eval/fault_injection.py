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
    """Transparent one-shot fault wrapper around an RLBench task environment."""

    def __init__(self, task_environment: Any, spec: FaultInjectionSpec) -> None:
        self._environment = task_environment
        self.spec = spec
        self.events: list[dict[str, Any]] = []
        self._policy_step = 0
        self._triggered = False
        self._active_cycles = 0
        self._close_occurrences = 0
        self._previous_close_request = False
        self._failed_close_active = False
        self._grasped_cycles = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    @property
    def triggered(self) -> bool:
        return self._triggered

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "schema": "essay2608.rlbench.physical_fault.v1",
            "spec": self.spec.to_dict(),
            "triggered": self._triggered,
            "events": list(self.events),
            "policy_steps_observed": self._policy_step,
            "policy_state_mutated": False,
            "observation_hidden": False,
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
        self.events.append(
            {
                "kind": "time_stall_cycle",
                "policy_step": self._policy_step,
                "remaining_cycles": self._active_cycles,
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
            self._close_occurrences += 1
        self._previous_close_request = close_requested
        if (
            not self._triggered
            and self._policy_step >= self.spec.earliest_step
            and self._close_occurrences == self.spec.close_occurrence
            and close_requested
        ):
            self._failed_close_active = True
            self._trigger_event(
                self.spec.kind.value,
                arm=selected[0],
                close_occurrence=self._close_occurrences,
                end_condition="explicit_open_then_new_close_occurrence",
            )
        if not self._failed_close_active:
            return action
        if not close_requested:
            self._failed_close_active = False
            self.events.append(
                {
                    "kind": "grasp_failure_occurrence_ended",
                    "policy_step": self._policy_step,
                    "arm": selected[0],
                    "protocol_effective": True,
                }
            )
            return action
        suppressed = action.copy()
        suppressed[gripper_index] = 1.0
        self.events.append(
            {
                "kind": "grasp_close_suppressed_cycle",
                "policy_step": self._policy_step,
                "arm": selected[0],
                "protocol_effective": True,
            }
        )
        return suppressed

    def _apply_relation_fault(self, *, bimanual: bool) -> None:
        selected = self._selected_arms(bimanual=bimanual)
        if len(selected) != 1:
            raise ValueError("relation faults require one selected arm")
        gripper = self._gripper(selected[0], bimanual=bimanual)
        grasped = list(gripper.get_grasped_objects())
        self._grasped_cycles = self._grasped_cycles + 1 if grasped else 0
        if (
            self._triggered
            or self._policy_step < self.spec.earliest_step
            or self._grasped_cycles < self.spec.minimum_grasped_cycles
        ):
            return
        names = self._object_names(grasped)
        gripper.release()
        displaced = False
        if self.spec.kind == FaultInjectionKind.RELATION_MISMATCH:
            delta = np.asarray(self.spec.mismatch_translation, dtype=np.float64)
            for obj in grasped:
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
            released_objects=names,
            carried_cycles=self._grasped_cycles,
            displaced=displaced,
            mismatch_translation=(
                list(self.spec.mismatch_translation) if displaced else None
            ),
        )

    def step(self, action: Any):
        values = np.asarray(action, dtype=np.float64)
        bimanual = self._is_bimanual_action(values)
        if self.spec.kind in {
            FaultInjectionKind.RELATION_MISMATCH,
            FaultInjectionKind.UNEXPECTED_DROP,
        }:
            self._apply_relation_fault(bimanual=bimanual)
        current_observation = self._environment.get_observation()
        applied = values
        if self.spec.kind == FaultInjectionKind.TIME_STALL:
            applied = self._apply_time_stall(
                values, current_observation, bimanual=bimanual
            )
        elif self.spec.kind == FaultInjectionKind.GRASP_FAILURE:
            applied = self._apply_grasp_failure(values, bimanual=bimanual)
        result = self._environment.step(applied)
        self._policy_step += 1
        return result


__all__ = [
    "FaultInjectionKind",
    "FaultInjectionSpec",
    "FaultInjectingTaskEnvironment",
]
