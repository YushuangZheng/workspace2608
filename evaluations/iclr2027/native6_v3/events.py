"""Pure event state machine for the Native-6 v3 fault protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import acos, isfinite
from typing import Any, Iterable, Mapping, Optional

import numpy as np


FAULT_FAMILIES = ("actuation_delay", "missed_interaction", "relation_loss")
RELATION_SOURCES = ("attachment", "maintained_contact")


def quaternion_angle_xyzw(first: Iterable[float], second: Iterable[float]) -> float:
    """Return the sign-invariant angular distance between two xyzw quaternions."""

    left = np.asarray(tuple(first), dtype=np.float64)
    right = np.asarray(tuple(second), dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("quaternions must contain four finite xyzw values")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("quaternions must have non-zero norm")
    dot = abs(float(np.dot(left / left_norm, right / right_norm)))
    return float(2.0 * acos(min(1.0, max(-1.0, dot))))


@dataclass(frozen=True)
class FaultDecision:
    family: str
    trigger_rule: str
    eligible: bool
    should_inject: bool
    sim_step: int
    simulation_time_s: float
    event_kind: str
    arm: str
    object_names: tuple[str, ...]
    interaction_source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["object_names"] = list(self.object_names)
        return value


class EventGroundedFaultState:
    """Select the first physically eligible event without policy-state access.

    The adapter is responsible for observing events at their actual execution
    location: movement before dispatch, gripper close after arm motion and
    before discrete attachment, and relation state after completed simulator
    steps.  This class only decides eligibility and never reads a task phase,
    learned belief, method alarm, or high-level cycle fraction.
    """

    def __init__(
        self,
        family: str,
        *,
        arm: str = "single",
        translation_threshold_m: float = 0.005,
        rotation_threshold_rad: float = 0.08726646259971647,
        stable_relation_seconds: float = 0.15,
    ) -> None:
        if family not in FAULT_FAMILIES:
            raise ValueError(f"unsupported Native-6 v3 fault family: {family}")
        if arm not in {"single", "left", "right", "all"}:
            raise ValueError("arm must be single/left/right/all")
        values = (
            translation_threshold_m,
            rotation_threshold_rad,
            stable_relation_seconds,
        )
        if any(not isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("physical thresholds must be finite and positive")
        self.family = family
        self.arm = arm
        self.translation_threshold_m = float(translation_threshold_m)
        self.rotation_threshold_rad = float(rotation_threshold_rad)
        self.stable_relation_seconds = float(stable_relation_seconds)
        self.eligible = False
        self.injection_triggered = False
        self.physical_effect_confirmed = False
        self.eligible_event: Optional[FaultDecision] = None
        self.trigger_event: Optional[FaultDecision] = None
        self.effect_event: Optional[dict[str, Any]] = None
        self._relation_signature: Optional[tuple[str, tuple[str, ...]]] = None
        self._relation_since_s: Optional[float] = None

    @staticmethod
    def _objects(names: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted({str(name) for name in names if str(name)}))

    @staticmethod
    def _validate_clock(sim_step: int, simulation_time_s: float) -> None:
        if isinstance(sim_step, bool) or int(sim_step) < 0:
            raise ValueError("sim_step must be a non-negative integer")
        if not isfinite(float(simulation_time_s)) or float(simulation_time_s) < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")

    def _decision(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        event_kind: str,
        arm: str,
        object_names: Iterable[str] = (),
        interaction_source: Optional[str] = None,
    ) -> FaultDecision:
        self._validate_clock(sim_step, simulation_time_s)
        decision = FaultDecision(
            family=self.family,
            trigger_rule="first_eligible_physical_event",
            eligible=True,
            should_inject=not self.injection_triggered,
            sim_step=int(sim_step),
            simulation_time_s=float(simulation_time_s),
            event_kind=event_kind,
            arm=arm,
            object_names=self._objects(object_names),
            interaction_source=interaction_source,
        )
        self.eligible = True
        if self.eligible_event is None:
            self.eligible_event = decision
        return decision

    def observe_motion_command(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        arm: str,
        translation_m: float,
        rotation_rad: float,
    ) -> Optional[FaultDecision]:
        if self.family != "actuation_delay" or self.injection_triggered:
            return None
        if not all(isfinite(float(value)) and float(value) >= 0.0 for value in (translation_m, rotation_rad)):
            raise ValueError("motion residuals must be finite and non-negative")
        if (
            float(translation_m) < self.translation_threshold_m
            and float(rotation_rad) < self.rotation_threshold_rad
        ):
            return None
        return self._decision(
            sim_step=sim_step,
            simulation_time_s=simulation_time_s,
            event_kind="meaningful_motion_command",
            arm=arm,
        )

    def observe_close_attempt(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        arm: str,
        actual_transition_to_closed: bool,
        detected_task_objects: Iterable[str],
    ) -> Optional[FaultDecision]:
        if self.family != "missed_interaction" or self.injection_triggered:
            return None
        objects = self._objects(detected_task_objects)
        if not actual_transition_to_closed or not objects:
            return None
        return self._decision(
            sim_step=sim_step,
            simulation_time_s=simulation_time_s,
            event_kind="physical_gripper_close_attempt",
            arm=arm,
            object_names=objects,
            interaction_source="pre_attachment_target_detection",
        )

    def observe_relation(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        arm: str,
        source: Optional[str],
        object_names: Iterable[str],
        contact_evidence: Optional[str] = None,
    ) -> Optional[FaultDecision]:
        if self.family != "relation_loss" or self.injection_triggered:
            return None
        self._validate_clock(sim_step, simulation_time_s)
        objects = self._objects(object_names)
        valid = source in RELATION_SOURCES and bool(objects)
        if source == "maintained_contact":
            valid = valid and contact_evidence == "sim_contact_info"
        if not valid:
            self._relation_signature = None
            self._relation_since_s = None
            return None
        signature = (str(source), objects)
        now = float(simulation_time_s)
        if signature != self._relation_signature:
            self._relation_signature = signature
            self._relation_since_s = now
            return None
        assert self._relation_since_s is not None
        if now - self._relation_since_s + 1e-12 < self.stable_relation_seconds:
            return None
        return self._decision(
            sim_step=sim_step,
            simulation_time_s=now,
            event_kind="stable_physical_relation",
            arm=arm,
            object_names=objects,
            interaction_source=str(source),
        )

    def mark_injected(self, decision: FaultDecision) -> None:
        if decision.family != self.family or not decision.should_inject:
            raise ValueError("fault injection does not match an eligible decision")
        if self.injection_triggered:
            raise ValueError("Native-6 v3 faults are one-shot")
        self.injection_triggered = True
        self.trigger_event = decision

    def confirm_physical_effect(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        evidence: Mapping[str, Any],
    ) -> None:
        if not self.injection_triggered or self.trigger_event is None:
            raise ValueError("cannot confirm a physical effect before injection")
        self._validate_clock(sim_step, simulation_time_s)
        if not evidence:
            raise ValueError("physical effect confirmation requires evidence")
        self.physical_effect_confirmed = True
        self.effect_event = {
            "sim_step": int(sim_step),
            "simulation_time_s": float(simulation_time_s),
            "evidence": dict(evidence),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "fault_family": self.family,
            "trigger_rule": "first_eligible_physical_event",
            "eligible": self.eligible,
            "injection_triggered": self.injection_triggered,
            "physical_effect_confirmed": self.physical_effect_confirmed,
            "eligible_event": (
                self.eligible_event.to_dict() if self.eligible_event else None
            ),
            "trigger_event": (
                self.trigger_event.to_dict() if self.trigger_event else None
            ),
            "effect_event": self.effect_event,
        }


__all__ = [
    "EventGroundedFaultState",
    "FAULT_FAMILIES",
    "FaultDecision",
    "quaternion_angle_xyzw",
]
