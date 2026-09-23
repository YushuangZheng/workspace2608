"""Open Drawer missed-interaction state machine introduced by amendment 2.

The original Native-6 v3 files remain byte frozen.  This module isolates the
one corrected event boundary: a real finger--drawer contact can first appear
*during* native gripper closure, so that contact onset is intercepted before a
stable maintained relation is established.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Optional


CONTACT_SOURCE = "simGetContactInfo"
EVENT_KIND = "maintained_contact_onset_during_close"


def _names(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _clock(sim_step: int, simulation_time_s: float) -> tuple[int, float]:
    if isinstance(sim_step, bool) or not isinstance(sim_step, int) or sim_step < 0:
        raise ValueError("sim_step must be a non-negative integer")
    if not isinstance(simulation_time_s, (int, float)) or not isfinite(
        float(simulation_time_s)
    ) or float(simulation_time_s) < 0.0:
        raise ValueError("simulation_time_s must be finite and non-negative")
    return sim_step, float(simulation_time_s)


@dataclass(frozen=True)
class ContactOnsetDecision:
    family: str
    trigger_rule: str
    event_kind: str
    arm: str
    object_names: tuple[str, ...]
    interaction_source: str
    sim_step: int
    simulation_time_s: float
    should_inject: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["object_names"] = list(self.object_names)
        return value


class OpenDrawerMissedInteraction:
    """Intercept first validated active-drawer contact during one real close.

    Object/contact validation stays outside this pure state machine.  The
    adapter must pass only canonical objects produced by the frozen mapping and
    contacts obtained from ``simGetContactInfo``.
    """

    def __init__(self, *, open_threshold: float = 0.9) -> None:
        if not isfinite(float(open_threshold)) or not 0.0 < float(open_threshold) < 1.0:
            raise ValueError("open_threshold must lie strictly between zero and one")
        self.open_threshold = float(open_threshold)
        self.transaction_active = False
        self.arm: Optional[str] = None
        self.active_drawer_objects: tuple[str, ...] = ()
        self.transaction_start: Optional[tuple[int, float]] = None
        self.eligible_event: Optional[ContactOnsetDecision] = None
        self.intervention_started = False
        self.intervention_occurrences = 0
        self.physical_effect_confirmed = False
        self.effect_event: Optional[dict[str, Any]] = None

    def begin_close_transaction(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        arm: str,
        arm_motion_completed: bool,
        actual_open_amount: float,
        native_open_to_close_started: bool,
        active_drawer_objects: Iterable[str],
    ) -> bool:
        """Register an actual close transaction without requiring pre-contact."""

        step, seconds = _clock(sim_step, simulation_time_s)
        objects = _names(active_drawer_objects)
        if self.transaction_active or self.intervention_started:
            raise RuntimeError("an Open Drawer close transaction is already active")
        eligible_transaction = (
            bool(arm_motion_completed)
            and bool(native_open_to_close_started)
            and isfinite(float(actual_open_amount))
            and float(actual_open_amount) > self.open_threshold
            and bool(objects)
        )
        if not eligible_transaction:
            return False
        self.transaction_active = True
        self.arm = str(arm)
        self.active_drawer_objects = objects
        self.transaction_start = (step, seconds)
        return True

    def observe_completed_step(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        actual_open_amount: float,
        native_closure_in_progress: bool,
        validated_contact_objects: Iterable[str],
        contact_source: str,
    ) -> Optional[ContactOnsetDecision]:
        """Return the first real active-drawer contact onset during closure."""

        step, seconds = _clock(sim_step, simulation_time_s)
        if not self.transaction_active or self.eligible_event is not None:
            return None
        if not native_closure_in_progress:
            return None
        if contact_source != CONTACT_SOURCE:
            return None
        if not isfinite(float(actual_open_amount)):
            raise ValueError("actual_open_amount must be finite")
        contacts = set(_names(validated_contact_objects))
        active = tuple(sorted(contacts.intersection(self.active_drawer_objects)))
        if not active:
            return None
        assert self.arm is not None
        decision = ContactOnsetDecision(
            family="missed_interaction",
            trigger_rule="first_eligible_physical_event",
            event_kind=EVENT_KIND,
            arm=self.arm,
            object_names=active,
            interaction_source="maintained_contact",
            sim_step=step,
            simulation_time_s=seconds,
            should_inject=True,
        )
        self.eligible_event = decision
        return decision

    def mark_reverse_open_started(
        self,
        decision: ContactOnsetDecision,
        *,
        sim_step: int,
        simulation_time_s: float,
        applied_open_amount: float,
    ) -> None:
        """Record the single reverse-open intervention at contact onset."""

        step, seconds = _clock(sim_step, simulation_time_s)
        if decision is not self.eligible_event:
            raise ValueError("intervention does not match the first contact-onset event")
        if self.intervention_started or self.intervention_occurrences:
            raise RuntimeError("Open Drawer missed interaction is one-shot")
        if step < decision.sim_step or seconds < decision.simulation_time_s:
            raise ValueError("reverse-open intervention precedes contact onset")
        if not isfinite(float(applied_open_amount)) or abs(float(applied_open_amount) - 1.0) > 1e-9:
            raise ValueError("amendment 2 requires applied open amount 1.0")
        self.intervention_started = True
        self.intervention_occurrences = 1

    def confirm_physical_effect(
        self,
        *,
        sim_step: int,
        simulation_time_s: float,
        actual_open_amount: float,
        validated_contact_objects: Iterable[str],
        attachment_objects: Iterable[str],
        drawer_setter_calls: int,
        arm_target_changes: int,
    ) -> None:
        """Confirm that contact was not maintained and no extra state was reset."""

        step, seconds = _clock(sim_step, simulation_time_s)
        if not self.intervention_started or self.eligible_event is None:
            raise RuntimeError("physical effect cannot precede the intervention")
        if step < self.eligible_event.sim_step or seconds < self.eligible_event.simulation_time_s:
            raise ValueError("physical effect precedes contact onset")
        contacts = set(_names(validated_contact_objects))
        attachments = set(_names(attachment_objects))
        active = set(self.active_drawer_objects)
        failures = []
        if not isfinite(float(actual_open_amount)) or float(actual_open_amount) <= self.open_threshold:
            failures.append("selected gripper is not open above the frozen threshold")
        if contacts.intersection(active):
            failures.append("validated active-drawer contact remains")
        if attachments.intersection(active):
            failures.append("active drawer appears in gripper attachments")
        if self.intervention_occurrences != 1:
            failures.append("intervention occurrence count differs from one")
        if drawer_setter_calls != 0:
            failures.append("drawer pose or joint setter was used")
        if arm_target_changes != 0:
            failures.append("arm target changed during the intervention")
        if failures:
            raise ValueError("; ".join(failures))
        self.physical_effect_confirmed = True
        self.effect_event = {
            "sim_step": step,
            "simulation_time_s": seconds,
            "actual_open_amount": float(actual_open_amount),
            "validated_contact_objects": sorted(contacts),
            "attachment_objects": sorted(attachments),
            "drawer_setter_calls": drawer_setter_calls,
            "arm_target_changes": arm_target_changes,
        }
        self.transaction_active = False

    def finish_without_contact(self) -> None:
        """Close a transaction that never generated a validated contact event."""

        if self.intervention_started:
            raise RuntimeError("an injected transaction requires effect confirmation")
        self.transaction_active = False

    def summary(self) -> Mapping[str, Any]:
        return {
            "eligible": self.eligible_event is not None,
            "injection_triggered": self.intervention_started,
            "physical_effect_confirmed": self.physical_effect_confirmed,
            "intervention_occurrences": self.intervention_occurrences,
            "transaction_start": self.transaction_start,
            "eligible_event": (
                self.eligible_event.to_dict() if self.eligible_event is not None else None
            ),
            "effect_event": self.effect_event,
        }


__all__ = [
    "CONTACT_SOURCE",
    "EVENT_KIND",
    "ContactOnsetDecision",
    "OpenDrawerMissedInteraction",
]
