"""Nested, paired repeated-event protocol for the corrected E4 comparison.

The first intervention is the ordinary frozen single-event intervention: it
uses the same seed, task initialization, family, severity, trigger stage, and
global policy horizon as its paired single-event episode.  No additional
fault environment is opened before that first intervention has physically
triggered.  Fresh opportunities are opened only for later interaction stages.

This makes the paired condition a strict extension of the single-event
condition up to the first event, rather than an independently sampled fault
schedule.  The wrapper observes only public physical interaction conditions
and fault-protocol metadata; it cannot read policy state, alarms, or outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from evaluations.iclr2027.audit.horizon_events import (
    _condition_met,
    ordered_interaction_conditions,
)


NESTED_EVENT_SCHEMA = "essay2608.iclr2027.horizon-nested-events.v1"
NESTED_EVENT_SCHEDULE = "paired_single_event_then_later_interaction_events"


@dataclass
class NestedStageFault:
    interaction_index: Optional[int]
    entry_policy_step: int
    environment: Any
    role: str


@dataclass(frozen=True)
class _ReleaseQualifiedCondition:
    native: Any
    release: Any

    def condition_met(self) -> tuple[bool, bool]:
        native_value = self.native.condition_met()
        release_value = self.release.condition_met()
        if not (
            isinstance(native_value, tuple)
            and len(native_value) == 2
            and isinstance(release_value, tuple)
            and len(release_value) == 2
        ):
            raise TypeError("physical conditions must return (met, terminate)")
        return (
            bool(native_value[0]) and bool(release_value[0]),
            bool(native_value[1]) or bool(release_value[1]),
        )


def _condition_children(condition: Any) -> tuple[Any, ...]:
    for name in ("_conditions", "conditions"):
        value = getattr(condition, name, None)
        if isinstance(value, (list, tuple)):
            return tuple(value)
    return ()


def _public_release_condition(task_environment: Any) -> Any | None:
    """Return the task's existing NothingGrasped predicate, if unique."""

    task = task_environment._scene.task
    pending = list(getattr(task, "_success_conditions", ()))
    matches = []
    while pending:
        condition = pending.pop()
        type_name = f"{type(condition).__module__}.{type(condition).__name__}"
        if type_name.endswith(".NothingGrasped"):
            matches.append(condition)
        pending.extend(_condition_children(condition))
    unique = {id(value): value for value in matches}
    if not unique:
        return None
    if len(unique) != 1:
        raise RuntimeError(
            "nested E4 task exposes multiple public NothingGrasped predicates"
        )
    return next(iter(unique.values()))


class NestedInteractionFaultEnvironment:
    """Preserve one paired event, then add opportunities at later stages."""

    def __init__(
        self,
        task_environment: Any,
        task: Any,
        *,
        family: str,
        severity: str,
        trigger_stage: str,
        paired_policy_steps: int,
        later_stage_policy_steps: int,
        config: Mapping[str, Any],
        fault_builder: Callable[..., Any],
    ) -> None:
        if not family or family == "composed_event":
            raise ValueError("nested E4 scheduling requires one atomic fault family")
        if paired_policy_steps < 1 or later_stage_policy_steps < 1:
            raise ValueError("nested E4 policy-step budgets must be positive")
        self._task = task
        self._family = str(family)
        self._severity = str(severity)
        self._trigger_stage = str(trigger_stage)
        self._later_stage_policy_steps = int(later_stage_policy_steps)
        self._config = config
        self._fault_builder = fault_builder
        native_conditions = ordered_interaction_conditions(task_environment)
        release = _public_release_condition(task_environment)
        self._release_qualified = release is not None
        self._conditions = (
            native_conditions
            if release is None
            else tuple(
                _ReleaseQualifiedCondition(native=value, release=release)
                for value in native_conditions
            )
        )
        expected = getattr(task, "task_level", None)
        if expected is not None and int(expected) != len(self._conditions):
            raise RuntimeError("task level and ordered physical interaction count disagree")
        self._policy_step = 0
        self._completed_interactions: list[dict[str, int]] = []
        self._first_trigger_interaction_index: Optional[int] = None
        self._opened_later_indices: set[int] = set()

        paired = fault_builder(
            task_environment,
            task,
            family=self._family,
            severity=self._severity,
            trigger_stage=self._trigger_stage,
            policy_steps=int(paired_policy_steps),
            config=self._config,
        )
        self._paired = paired
        self._current = paired
        self._stages: list[NestedStageFault] = [
            NestedStageFault(None, 0, paired, "paired_first_event")
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current, name)

    @staticmethod
    def _metadata(environment: Any) -> dict[str, Any]:
        getter = getattr(environment, "protocol_metadata", None)
        if not callable(getter):
            raise RuntimeError("nested stage fault environment lacks protocol metadata")
        value = getter()
        if not isinstance(value, Mapping):
            raise TypeError("nested stage fault metadata must be a mapping")
        return dict(value)

    def _active_interaction_index(self) -> int:
        return min(len(self._completed_interactions), len(self._conditions) - 1)

    def _observe_first_trigger(self, interaction_index: int) -> None:
        if self._first_trigger_interaction_index is not None:
            return
        if self._metadata(self._paired).get("triggered") is True:
            self._first_trigger_interaction_index = int(interaction_index)
            self._stages[0].interaction_index = int(interaction_index)

    def _advance_completed_interactions(self) -> None:
        while len(self._completed_interactions) < len(self._conditions):
            index = len(self._completed_interactions)
            if not _condition_met(self._conditions[index]):
                break
            self._completed_interactions.append(
                {
                    "interaction_index": index,
                    "completion_policy_step": int(self._policy_step),
                }
            )

    def _open_later_stage_if_needed(self) -> None:
        first = self._first_trigger_interaction_index
        index = len(self._completed_interactions)
        if (
            first is None
            or index >= len(self._conditions)
            or index <= first
            or index in self._opened_later_indices
        ):
            return
        wrapped = self._fault_builder(
            self._current,
            self._task,
            family=self._family,
            severity=self._severity,
            trigger_stage=self._trigger_stage,
            policy_steps=self._later_stage_policy_steps,
            config=self._config,
        )
        self._current = wrapped
        self._opened_later_indices.add(index)
        self._stages.append(
            NestedStageFault(index, int(self._policy_step), wrapped, "later_stage_event")
        )

    def step(self, action: Any):
        result = self._current.step(action)
        self._policy_step += 1
        # Before the paired first event, remain a transparent wrapper: even a
        # read-only native success-condition query can perturb simulator-side
        # sensor handling and break the paired physical prefix.  Stage
        # observation therefore begins only after the ordinary one-shot fault
        # has triggered.  A one-stage task never needs stage observation.
        if self._first_trigger_interaction_index is None:
            if len(self._conditions) == 1:
                self._observe_first_trigger(0)
                return result
            if self._metadata(self._paired).get("triggered") is not True:
                return result
            self._advance_completed_interactions()
            self._observe_first_trigger(self._active_interaction_index())
        else:
            self._advance_completed_interactions()
        self._open_later_stage_if_needed()
        return result

    def record_committed_fallback(self) -> None:
        callback = getattr(self._current, "record_committed_fallback", None)
        if not callable(callback):
            raise RuntimeError("active nested fault wrapper lacks fallback accounting")
        callback()
        self._policy_step += 1

    def protocol_metadata(self) -> dict[str, Any]:
        components = []
        events = []
        for stage in self._stages:
            metadata = self._metadata(stage.environment)
            index = -1 if stage.interaction_index is None else int(stage.interaction_index)
            local_restoration = metadata.get("relation_restoration_policy_step")
            global_restoration = (
                None
                if local_restoration is None
                else stage.entry_policy_step + int(local_restoration)
            )
            component_events = []
            for event in metadata.get("events", ()):
                record = dict(event)
                local_step = int(record.get("policy_step", 0))
                record["component_policy_step"] = local_step
                record["policy_step"] = stage.entry_policy_step + local_step
                record["interaction_index"] = index
                record["nested_role"] = stage.role
                component_events.append(record)
                events.append(record)
            components.append(
                {
                    "interaction_index": index,
                    "entry_policy_step": int(stage.entry_policy_step),
                    "nested_role": stage.role,
                    "triggered": metadata.get("triggered") is True,
                    "physical_effect_observed": metadata.get("physical_effect_observed"),
                    "target_arm": metadata.get("target_arm"),
                    "target_objects": list(metadata.get("target_objects", ())),
                    "relation_restored": metadata.get("relation_restored"),
                    "relation_restoration_policy_step": global_restoration,
                    "events": component_events,
                }
            )
        events.sort(
            key=lambda value: (
                int(value.get("policy_step", -1)),
                int(value.get("interaction_index", -1)),
            )
        )
        triggered = [value for value in components if value["triggered"]]
        latest = triggered[-1] if triggered else None
        return {
            "schema": NESTED_EVENT_SCHEMA,
            "family": self._family,
            "event_schedule": NESTED_EVENT_SCHEDULE,
            "triggered": bool(triggered),
            "events": events,
            "policy_steps_observed": int(self._policy_step),
            "policy_state_mutated": False,
            "observation_hidden": False,
            "physical_effect_observed": any(
                value["physical_effect_observed"] is True for value in components
            ),
            "target_arm": None if latest is None else latest["target_arm"],
            "target_objects": [] if latest is None else latest["target_objects"],
            "interaction_count": len(self._conditions),
            "release_qualified_interactions": self._release_qualified,
            "interaction_stages_completed": len(self._completed_interactions),
            "event_opportunity_count": len(components),
            "paired_first_event_triggered": self._first_trigger_interaction_index is not None,
            "paired_first_event_interaction_index": self._first_trigger_interaction_index,
            "later_event_opportunities": max(0, len(components) - 1),
            "triggered_event_count": len(triggered),
            "completed_interactions": list(self._completed_interactions),
            "components": components,
        }


__all__ = [
    "NESTED_EVENT_SCHEMA",
    "NESTED_EVENT_SCHEDULE",
    "NestedInteractionFaultEnvironment",
]
