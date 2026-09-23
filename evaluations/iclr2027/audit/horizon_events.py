"""Policy-independent repeated-fault protocol for Horizon-3.

The ordinary ICLR 2027 fault environments are intentionally one-shot.  E4's
``per-stage`` condition instead offers one such intervention in every
interaction of a repeated task.  This wrapper advances only when the task's
ordered *physical* interaction condition becomes true; it never observes a
policy clock, progress belief, alarm, recovery mode, or episode outcome.

The A6-only auditor reads the already-existing native per-interaction success
conditions exposed by each Horizon task family.  It does not add a policy
variable or modify the shared task adapter, which keeps E4 instrumentation
separate from the frozen E1/E6 task interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence


HORIZON_EVENT_SCHEMA = "essay2608.iclr2027.horizon-events.v1"
EVENT_SCHEDULE = "one_eligible_event_per_interaction_stage"


def _condition_met(condition: Any) -> bool:
    predicate = getattr(condition, "condition_met", None)
    if not callable(predicate):
        raise TypeError("ordered interaction condition lacks condition_met()")
    result = predicate()
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("condition_met() must return the RLBench (met, terminate) pair")
    return bool(result[0])


def ordered_interaction_conditions(task_environment: Any) -> tuple[Any, ...]:
    """Return the native physical interaction sequence for a Horizon task."""

    scene = getattr(task_environment, "_scene", None)
    task = getattr(scene, "task", None)
    provider = getattr(task, "iclr2027_ordered_interaction_conditions", None)
    if callable(provider):
        conditions = tuple(provider())
    else:
        count = int(getattr(task, "_target_count", 0))
        if count < 1:
            raise RuntimeError("Horizon-3 task has no positive native interaction count")
        if hasattr(task, "_on_peg_conditions"):
            conditions = tuple(task._on_peg_conditions[:count])
        elif hasattr(task, "goal_conditions"):
            conditions = tuple(task.goal_conditions[:count])
        elif getattr(task, "_scene_task_name", None) == "remove_cups":
            conditions = tuple(task.success_conditions[-count:])
        else:
            raise RuntimeError(
                "Horizon-3 task family has no audited native interaction sequence"
            )
        if len(conditions) != count:
            raise RuntimeError("native interaction sequence and task level disagree")
    if not conditions:
        raise RuntimeError("Horizon-3 task exposed no interaction conditions")
    if len({id(value) for value in conditions}) != len(conditions):
        raise RuntimeError("ordered interaction conditions must be distinct")
    return conditions


@dataclass(frozen=True)
class StageFault:
    interaction_index: int
    entry_policy_step: int
    environment: Any


class InteractionStageFaultEnvironment:
    """Offer one ordinary one-shot physical fault per reached interaction.

    Each newly reached interaction receives a fresh fault environment.  Older
    environments remain in the transparent wrapper chain so their physical
    bookkeeping can finish, but their one-shot actuator cannot trigger again.
    The local trigger-stage fraction is evaluated against the demonstrated
    per-interaction policy horizon supplied by the caller.
    """

    def __init__(
        self,
        task_environment: Any,
        task: Any,
        *,
        family: str,
        severity: str,
        trigger_stage: str,
        per_interaction_policy_steps: int,
        config: Mapping[str, Any],
        fault_builder: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not family or family == "composed_event":
            raise ValueError("per-interaction scheduling requires one atomic fault family")
        if per_interaction_policy_steps < 1:
            raise ValueError("per-interaction policy horizon must be positive")
        if fault_builder is None:
            # Lazy import avoids a module cycle when the common factory elects
            # to construct this wrapper for an E4 manifest row.
            from evaluations.iclr2027.audit.faults import build_fault_environment

            fault_builder = build_fault_environment
        self._task = task
        self._family = str(family)
        self._severity = str(severity)
        self._trigger_stage = str(trigger_stage)
        self._per_interaction_policy_steps = int(per_interaction_policy_steps)
        self._config = config
        self._fault_builder = fault_builder
        self._conditions = ordered_interaction_conditions(task_environment)
        expected = getattr(task, "task_level", None)
        if expected is not None and int(expected) != len(self._conditions):
            raise RuntimeError(
                "task level and ordered physical interaction count disagree"
            )
        self._policy_step = 0
        self._completed_interactions: list[dict[str, int]] = []
        self._stages: list[StageFault] = []
        self._current = task_environment
        self._open_stage(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current, name)

    def _open_stage(self, interaction_index: int) -> None:
        wrapped = self._fault_builder(
            self._current,
            self._task,
            family=self._family,
            severity=self._severity,
            trigger_stage=self._trigger_stage,
            policy_steps=self._per_interaction_policy_steps,
            config=self._config,
        )
        self._current = wrapped
        self._stages.append(
            StageFault(
                interaction_index=int(interaction_index),
                entry_policy_step=int(self._policy_step),
                environment=wrapped,
            )
        )

    def _advance_completed_interactions(self) -> None:
        # More than one condition may become true in one simulator step (for
        # example after settling).  Consume every newly completed prefix while
        # preserving its exact common physical observation time.
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
            next_index = index + 1
            if next_index < len(self._conditions):
                self._open_stage(next_index)

    def step(self, action: Any):
        result = self._current.step(action)
        self._policy_step += 1
        self._advance_completed_interactions()
        return result

    def record_committed_fallback(self) -> None:
        callback = getattr(self._current, "record_committed_fallback", None)
        if not callable(callback):
            raise RuntimeError("active physical fault wrapper lacks fallback accounting")
        callback()
        self._policy_step += 1

    @staticmethod
    def _metadata(environment: Any) -> dict[str, Any]:
        getter = getattr(environment, "protocol_metadata", None)
        if not callable(getter):
            raise RuntimeError("stage fault environment lacks protocol metadata")
        value = getter()
        if not isinstance(value, Mapping):
            raise TypeError("stage fault metadata must be a mapping")
        return dict(value)

    def protocol_metadata(self) -> dict[str, Any]:
        components = []
        events = []
        for stage in self._stages:
            metadata = self._metadata(stage.environment)
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
                record["interaction_index"] = stage.interaction_index
                component_events.append(record)
                events.append(record)
            components.append(
                {
                    "interaction_index": stage.interaction_index,
                    "entry_policy_step": stage.entry_policy_step,
                    "triggered": metadata.get("triggered") is True,
                    "physical_effect_observed": metadata.get(
                        "physical_effect_observed"
                    ),
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
        triggered_components = [value for value in components if value["triggered"]]
        latest = triggered_components[-1] if triggered_components else None
        return {
            "schema": HORIZON_EVENT_SCHEMA,
            "family": self._family,
            "event_schedule": EVENT_SCHEDULE,
            "triggered": bool(triggered_components),
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
            "interaction_stages_reached": len(self._stages),
            "interaction_stages_completed": len(self._completed_interactions),
            "triggered_event_count": len(triggered_components),
            "completed_interactions": list(self._completed_interactions),
            "components": components,
        }


__all__ = [
    "EVENT_SCHEDULE",
    "HORIZON_EVENT_SCHEMA",
    "InteractionStageFaultEnvironment",
    "ordered_interaction_conditions",
]
