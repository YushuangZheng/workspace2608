"""Policy-independent physical fault layer for the ICLR 2027 protocol.

The injector sees only the public action, simulator objects, and a preregistered
cycle floor.  It cannot read a policy StateId, belief, stream role, alarm, or
recovery decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from integrations.rlbench.rlbench_closed_loop.eval.fault_injection import (
    FaultInjectingTaskEnvironment,
    FaultInjectionKind,
    FaultInjectionSpec,
)

FAULT_SCHEMA = "essay2608.iclr2027.physical-fault.v1"


def default_fault_arm(task_id: str, family: str) -> str:
    """Return the preregistered physical arm, never a policy-selected arm."""

    if not task_id.startswith("bimanual_"):
        return "single"
    if family == "composed_event":
        # The second component of the frozen two-event composition is the
        # relation-loss intervention, so audit the same physical arm.
        family = "relation_loss"
    by_task = {
        "bimanual_handover_item": {
            "missed_interaction": "right",
            "relation_loss": "left",
            "coordination_delay": "right",
        },
        "bimanual_lift_tray": {
            "missed_interaction": "right",
            "relation_loss": "right",
            "coordination_delay": "right",
        },
        "bimanual_sweep_to_dustpan": {
            "missed_interaction": "left",
            "relation_loss": "left",
            "coordination_delay": "right",
        },
        "bimanual_put_bottle_in_fridge": {
            "missed_interaction": "left",
            "relation_loss": "left",
            "coordination_delay": "right",
        },
    }
    if family == "actuation_delay":
        return "all"
    return by_task.get(task_id, {}).get(family, "left")


def _task_state(observation: Any) -> np.ndarray:
    value = observation.task_low_dim_state
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return np.asarray(value, dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class EnvironmentChangeSpec:
    earliest_step: int
    translation: tuple
    articulation: float
    motion_trigger_distance: float


class EnvironmentChangingTaskEnvironment:
    """One-shot movement of a task-relevant physical entity.

    Selection is based on the frozen task schema and simulator geometry rather
    than a task name or policy state.  Articulated tasks use their selected
    public joint; other tasks use the task-tree object nearest a non-operated
    semantic frame and verify the resulting low-dimensional displacement.
    """

    def __init__(self, environment: Any, task: Any, spec: EnvironmentChangeSpec):
        self._environment = environment
        self.task = task
        self.spec = spec
        self._policy_step = 0
        self._triggered = False
        self._events = []
        self._effect_observed = False
        self._effect_policy_step = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    @staticmethod
    def _action_motion(action: np.ndarray, observation: Any, bimanual: bool) -> float:
        if bimanual:
            right = np.linalg.norm(action[:3] - np.asarray(observation.right.gripper_pose)[:3])
            left = np.linalg.norm(action[9:12] - np.asarray(observation.left.gripper_pose)[:3])
            return float(max(right, left))
        return float(np.linalg.norm(action[:3] - np.asarray(observation.gripper_pose)[:3]))

    def _selected_semantic_pose(self, observation: Any) -> np.ndarray:
        poses = self.task.spec.extract_pose_chunks(_task_state(observation), convention="rlbench_xyzw")
        priority = (
            "environment_reference",
            "placement_reference",
            "scene_entity",
            "articulated_object",
            "cooperatively_operated_tool",
            "cooperatively_operated_object",
        )
        chunks = {chunk.name: chunk for chunk in self.task.spec.pose_chunks}
        for role in priority:
            for name, pose in poses.items():
                if chunks[name].role == role:
                    return np.asarray(pose, dtype=np.float64)
        return np.asarray(next(iter(poses.values())), dtype=np.float64)

    def _move_entity(self, observation: Any) -> Mapping[str, Any]:
        live_task = self._environment._scene.task
        selected_index = getattr(live_task, "_current_index", 0)
        joints = getattr(live_task, "_joints", None)
        before_state = _task_state(observation)
        if isinstance(joints, (list, tuple)) and joints:
            joint = joints[int(selected_index)]
            before = float(joint.get_joint_position())
            joint.set_joint_position(before + float(self.spec.articulation))
            target_name = str(joint.get_name())
            intervention = "articulation"
        else:
            target_pose = self._selected_semantic_pose(observation)
            base = live_task.get_base()
            candidates = []
            for obj in base.get_objects_in_tree(exclude_base=False):
                getter = getattr(obj, "get_position", None)
                setter = getattr(obj, "set_position", None)
                if not callable(getter) or not callable(setter):
                    continue
                position = np.asarray(getter(), dtype=np.float64)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    continue
                candidates.append((float(np.linalg.norm(position - target_pose[:3])), obj, position))
            if not candidates:
                raise RuntimeError("environment change found no movable task entity")
            _distance, target, before = min(candidates, key=lambda item: item[0])
            target.set_position(before + np.asarray(self.spec.translation, dtype=np.float64))
            target_name = str(target.get_name())
            intervention = "translation"
        refreshed = self._environment.get_observation()
        displacement = float(np.linalg.norm(_task_state(refreshed) - before_state))
        self._effect_observed = displacement > 1e-8
        self._effect_policy_step = self._policy_step if self._effect_observed else None
        return {
            "kind": "environment_change",
            "policy_step": self._policy_step,
            "target_object": target_name,
            "intervention": intervention,
            "task_state_l2_change": displacement,
            "protocol_effective": self._effect_observed,
        }

    def step(self, action: Any):
        command = np.asarray(action, dtype=np.float64)
        observation = self._environment.get_observation()
        bimanual = command.shape == (18,)
        if command.shape not in {(9,), (18,)}:
            raise ValueError("unsupported action shape for environment change")
        if (
            not self._triggered
            and self._policy_step >= self.spec.earliest_step
            and self._action_motion(command, observation, bimanual)
            >= self.spec.motion_trigger_distance
        ):
            event = dict(self._move_entity(observation))
            self._events.append(event)
            self._triggered = bool(event["protocol_effective"])
        result = self._environment.step(command)
        self._policy_step += 1
        return result

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "schema": FAULT_SCHEMA,
            "family": "environment_change",
            "triggered": self._triggered,
            "events": list(self._events),
            "policy_steps_observed": self._policy_step,
            "policy_state_mutated": False,
            "observation_hidden": False,
            "physical_effect_observed": self._effect_observed,
            "effect_policy_step": self._effect_policy_step,
        }

    def record_committed_fallback(self) -> None:
        self._policy_step += 1


class CommonFaultEnvironment:
    """Normalize legacy physical actuators to the frozen A2 fault schema."""

    def __init__(self, wrapped: Any, family: str):
        self._wrapped = wrapped
        self.family = family

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def step(self, action: Any):
        return self._wrapped.step(action)

    def protocol_metadata(self) -> dict[str, Any]:
        raw = self._wrapped.protocol_metadata()
        audit = raw.get("physical_audit", {})
        return {
            "schema": FAULT_SCHEMA,
            "family": self.family,
            "triggered": raw.get("triggered") is True,
            "events": list(raw.get("events", ())),
            "policy_steps_observed": raw.get("policy_steps_observed"),
            "policy_state_mutated": raw.get("policy_state_mutated"),
            "observation_hidden": raw.get("observation_hidden"),
            "physical_effect_observed": audit.get("effect_observed"),
            "effect_policy_step": audit.get("effect_policy_step"),
            "target_arm": audit.get("target_arm"),
            "target_objects": audit.get("target_objects", []),
            "relation_restored": audit.get("relation_restored"),
            "relation_restoration_policy_step": audit.get(
                "relation_restoration_policy_step"
            ),
        }


class CompositeFaultEnvironment:
    """Expose two separated physical interventions as one audited episode."""

    def __init__(self, wrapped: Any, components: tuple[CommonFaultEnvironment, ...]):
        self._wrapped = wrapped
        self._components = components

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def step(self, action: Any):
        return self._wrapped.step(action)

    def record_committed_fallback(self) -> None:
        # A raw joint-hold commit bypasses every nested ``step`` and therefore
        # advances each component's public policy clock exactly once.
        for component in self._components:
            component.record_committed_fallback()

    def protocol_metadata(self) -> dict[str, Any]:
        components = [component.protocol_metadata() for component in self._components]
        events = []
        for index, component in enumerate(components):
            for event in component.get("events", ()):
                events.append({**event, "component_index": index, "component_family": component["family"]})
        events.sort(key=lambda event: (int(event.get("policy_step", -1)), int(event["component_index"])))
        return {
            "schema": FAULT_SCHEMA,
            "family": "composed_event",
            "triggered": all(component.get("triggered") is True for component in components),
            "events": events,
            "policy_steps_observed": min(
                int(component.get("policy_steps_observed") or 0) for component in components
            ),
            "policy_state_mutated": False,
            "observation_hidden": False,
            "physical_effect_observed": all(
                component.get("physical_effect_observed") is True for component in components
            ),
            "target_arm": components[-1].get("target_arm"),
            "target_objects": components[-1].get("target_objects", []),
            "components": components,
        }


def build_fault_environment(
    task_environment: Any,
    task: Any,
    *,
    family: Optional[str],
    trigger_stage: Optional[str],
    policy_steps: int,
    config: Mapping[str, Any],
    severity: Optional[str] = None,
) -> Any:
    """Construct one frozen physical intervention from a manifest row."""

    if family is None:
        return task_environment
    fractions = config["trigger_stages"]
    if trigger_stage not in fractions:
        raise ValueError("fault row has no frozen trigger stage")
    earliest = max(0, int(round(float(fractions[trigger_stage]) * policy_steps)))
    severity = severity or "medium"
    if severity not in {"low", "medium", "high", "composed"}:
        raise ValueError(f"unsupported frozen fault severity: {severity}")
    medium = config["medium"]
    grid = config["severity_grid"]
    if severity == "low":
        delay_cycles = int(grid["delay_cycles"][0])
        translation = float(grid["translation_m"][0])
        articulation = float(grid["articulation_rad"][0])
    elif severity == "high":
        delay_cycles = int(grid["delay_cycles"][-1])
        translation = float(grid["translation_m"][-1])
        articulation = float(grid["articulation_rad"][-1])
    else:
        delay_cycles = int(medium["actuation_delay_cycles"])
        translation = float(medium["translation_m"])
        articulation = float(medium["articulation_rad"])
    eligibility = config["eligibility"]
    arm = default_fault_arm(task.task_id, family)
    if family == "environment_change":
        return EnvironmentChangingTaskEnvironment(
            task_environment,
            task,
            EnvironmentChangeSpec(
                earliest_step=earliest,
                translation=(translation, 0.0, 0.0),
                articulation=articulation,
                motion_trigger_distance=float(
                    eligibility["motion_trigger_distance_m"]
                ),
            ),
        )
    if family == "composed_event":
        first = build_fault_environment(
            task_environment,
            task,
            family="actuation_delay",
            severity="medium",
            trigger_stage="early",
            policy_steps=policy_steps,
            config=config,
        )
        second = build_fault_environment(
            first,
            task,
            family="relation_loss",
            severity="medium",
            trigger_stage="late",
            policy_steps=policy_steps,
            config=config,
        )
        return CompositeFaultEnvironment(second, (first, second))
    mapping = {
        "actuation_delay": FaultInjectionKind.TIME_STALL,
        "coordination_delay": FaultInjectionKind.TIME_STALL,
        "missed_interaction": FaultInjectionKind.GRASP_FAILURE,
        "relation_loss": FaultInjectionKind.RELATION_MISMATCH,
    }
    if family not in mapping:
        raise ValueError(f"unsupported frozen fault family: {family}")
    duration = (
        int(medium["coordination_delay_cycles"])
        if family == "coordination_delay" and severity == "medium"
        else delay_cycles
    )
    spec = FaultInjectionSpec(
        kind=mapping[family],
        arm=arm,
        earliest_step=earliest,
        duration_cycles=duration,
        motion_trigger_distance=float(eligibility["motion_trigger_distance_m"]),
        close_occurrence=int(medium["missed_interaction_occurrences"]),
        minimum_grasped_cycles=int(eligibility["minimum_stable_relation_cycles"]),
        mismatch_translation=(translation, 0.0, 0.0),
    )
    return CommonFaultEnvironment(
        FaultInjectingTaskEnvironment(task_environment, spec), family
    )


__all__ = [
    "FAULT_SCHEMA",
    "CommonFaultEnvironment",
    "CompositeFaultEnvironment",
    "EnvironmentChangingTaskEnvironment",
    "build_fault_environment",
    "default_fault_arm",
]
