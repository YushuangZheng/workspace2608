"""Policy-independent physical fault layer for the ICLR 2027 protocol.

The injector sees only the public action, simulator objects, and a preregistered
cycle floor.  It cannot read a policy StateId, belief, stream role, alarm, or
recovery decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from integrations.rlbench.rlbench_tsf.eval.fault_injection import (
    FaultInjectingTaskEnvironment,
    FaultInjectionKind,
    FaultInjectionSpec,
)

FAULT_SCHEMA = "essay2608.iclr2027.physical-fault.v1"

# Frozen task-interface roles, not task names or online policy variables.
# Environment change may move a physical reference together with passive
# contents that it supports, but it must not silently move another object the
# robot operates independently.
DIRECTLY_OPERATED_ROLES = frozenset(
    {
        "operated_object",
        "operated_tool",
        "variation_target",
        "cooperatively_operated_object",
        "cooperatively_operated_tool",
        "articulated_object",
    }
)
PASSIVE_SUPPORT_ROLES = frozenset({"carried_object", "scene_entity"})


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
    support_contact_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        if not 0.0 < float(self.support_contact_tolerance_m) <= 0.005:
            raise ValueError("environment support contact tolerance must be in (0, 0.005]")


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
        live_task = self._environment._scene.task
        joints = getattr(live_task, "_joints", None)
        self._semantic_anchors = (
            {}
            if isinstance(joints, (list, tuple)) and joints
            else self._bind_semantic_anchors(
                self._environment.get_observation(), live_task
            )
        )

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

    @staticmethod
    def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
        left = np.asarray(first, dtype=np.float64)
        right = np.asarray(second, dtype=np.float64)
        left /= max(float(np.linalg.norm(left)), 1e-12)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        return float(
            min(np.linalg.norm(left - right), np.linalg.norm(left + right))
        )

    @classmethod
    def _semantic_pose_object(
        cls,
        task_base: Any,
        semantic_pose: np.ndarray,
        *,
        preferred_roots: tuple[Any, ...],
        prefer_preferred_membership: bool,
        translation_tie_tolerance_m: float = 1.0e-4,
    ) -> Any:
        """Bind a semantic pose to one simulator entity using full SE(3).

        Binding occurs once at wrapper construction, before placement can
        colocate an operated object and its destination.  Within the numerical
        translation tie envelope, orientation and the task's existing public
        graspable registration preserve object/reference identity.
        """

        pose = np.asarray(semantic_pose, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("semantic pose must be a finite xyz+xyzw vector")
        preferred_handles = {
            handle
            for root in preferred_roots
            for handle in cls._tree_handles(root)
        }
        candidates = []
        for value in task_base.get_objects_in_tree(exclude_base=False):
            getter = getattr(value, "get_pose", None)
            setter = getattr(value, "set_position", None)
            if not callable(getter) or not callable(setter):
                continue
            candidate = np.asarray(getter(), dtype=np.float64)
            if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                continue
            translation = float(np.linalg.norm(candidate[:3] - pose[:3]))
            rotation = cls._quaternion_distance(candidate[3:], pose[3:])
            preferred = cls._object_handle(value) in preferred_handles
            preferred_rank = int(preferred != prefer_preferred_membership)
            candidates.append(
                (
                    translation,
                    rotation,
                    preferred_rank,
                    cls._object_handle(value),
                    value,
                )
            )
        if not candidates:
            raise RuntimeError("environment change found no pose-resolvable task entity")
        nearest = min(value[0] for value in candidates)
        tied = [
            value
            for value in candidates
            if value[0] <= nearest + float(translation_tie_tolerance_m)
        ]
        return min(
            tied, key=lambda value: (value[1], value[2], value[0], value[3])
        )[-1]

    def _bind_semantic_anchors(
        self,
        observation: Any,
        live_task: Any,
    ) -> dict[str, Any]:
        base = live_task.get_base()
        poses = self.task.spec.extract_pose_chunks(
            _task_state(observation), convention="rlbench_xyzw"
        )
        chunks = {value.name: value for value in self.task.spec.pose_chunks}
        graspables = getattr(live_task, "get_graspable_objects", lambda: ())()
        preferred = tuple(graspables)
        anchors = {}
        for name, pose in poses.items():
            role = str(chunks[name].role)
            anchors[str(name)] = self._semantic_pose_object(
                base,
                np.asarray(pose, dtype=np.float64),
                preferred_roots=preferred,
                prefer_preferred_membership=(
                    role in DIRECTLY_OPERATED_ROLES
                ),
            )
        return anchors

    @staticmethod
    def _object_handle(value: Any) -> int:
        return int(value.get_handle())

    @classmethod
    def _tree_handles(cls, root: Any) -> set[int]:
        return {
            cls._object_handle(value)
            for value in root.get_objects_in_tree(exclude_base=False)
        }

    @staticmethod
    def _nearest_scene_object(base: Any, pose: np.ndarray) -> Any:
        candidates = []
        for obj in base.get_objects_in_tree(exclude_base=False):
            getter = getattr(obj, "get_position", None)
            setter = getattr(obj, "set_position", None)
            if not callable(getter) or not callable(setter):
                continue
            position = np.asarray(getter(), dtype=np.float64)
            if position.shape == (3,) and np.all(np.isfinite(position)):
                candidates.append((float(np.linalg.norm(position - pose[:3])), obj))
        if not candidates:
            raise RuntimeError("environment change found no movable task entity")
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _shape_tree(root: Any) -> list[Any]:
        # Keep PyRep optional at module-import time so protocol/schema tests do
        # not require the simulator environment.
        from pyrep.objects.shape import Shape

        return [
            value
            for value in root.get_objects_in_tree(exclude_base=False)
            if isinstance(value, Shape)
        ]

    @classmethod
    def _minimum_shape_distance(cls, first: Any, others: list[Any]) -> float:
        first_shapes = cls._shape_tree(first)
        other_shapes = [shape for root in others for shape in cls._shape_tree(root)]
        minimum = float("inf")
        for left in first_shapes:
            for right in other_shapes:
                if cls._object_handle(left) == cls._object_handle(right):
                    return 0.0
                try:
                    if left.check_collision(right):
                        return 0.0
                    distance = float(left.check_distance(right))
                except Exception:
                    continue
                if np.isfinite(distance) and distance >= 0.0:
                    minimum = min(minimum, distance)
        return minimum

    @classmethod
    def _support_component_roots(
        cls,
        target: Any,
        candidates: list[tuple[str, str, Any]],
        *,
        grasped_handles: set[int],
        contact_tolerance_m: float,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Return ungrasped semantic roots physically supported with target.

        The connected component is evaluated only from simulator geometry and
        the frozen task schema.  It has no access to policy state, alarms,
        fault identity inside the policy, or the episode outcome.
        """

        component = [target]
        component_handles = cls._tree_handles(target)
        remaining = []
        for semantic_name, role, root in candidates:
            handles = cls._tree_handles(root)
            if handles & component_handles or handles & grasped_handles:
                continue
            remaining.append((semantic_name, role, root, handles))

        preserved: list[Any] = []
        evidence: list[dict[str, Any]] = []
        preserved_handles: set[int] = set()
        while remaining:
            added = False
            next_remaining = []
            for semantic_name, role, root, handles in remaining:
                distance = cls._minimum_shape_distance(root, component)
                if distance <= contact_tolerance_m:
                    component.append(root)
                    component_handles.update(handles)
                    if not handles & preserved_handles:
                        preserved.append(root)
                        preserved_handles.update(handles)
                    evidence.append(
                        {
                            "semantic_name": semantic_name,
                            "role": role,
                            "physical_root": str(root.get_name()),
                            "minimum_shape_distance_m": distance,
                        }
                    )
                    added = True
                else:
                    next_remaining.append(
                        (semantic_name, role, root, handles)
                    )
            remaining = next_remaining
            if not added:
                break
        return preserved, evidence

    @staticmethod
    def _semantic_support_roots(
        records: list[dict[str, Any]],
    ) -> list[tuple[str, str, Any]]:
        """Return only passive contents eligible to follow a moved support."""

        roots = []
        seen = set()
        for record in records:
            role = str(record["semantic_role"])
            if role not in PASSIVE_SUPPORT_ROLES:
                continue
            root = record["physical_root"]
            handle = int(root.get_handle())
            if handle in seen:
                continue
            seen.add(handle)
            roots.append((str(record["semantic_name"]), role, root))
        return roots

    @classmethod
    def _grasped_handles(cls, scene: Any) -> set[int]:
        robot = scene.robot
        grippers = []
        for name in ("gripper", "left_gripper", "right_gripper"):
            gripper = getattr(robot, name, None)
            if gripper is not None and gripper not in grippers:
                grippers.append(gripper)
        handles = set()
        for gripper in grippers:
            for root in gripper.get_grasped_objects():
                handles.update(cls._tree_handles(root))
        return handles

    @staticmethod
    def _physical_entity_root(anchor: Any, task_base: Any) -> Any:
        """Resolve a semantic proxy to the movable physical entity it denotes.

        RLBench commonly exposes a placement reference through a child dummy,
        sensor, or component shape.  Moving that child alone separates the
        task's observation/success proxy from its physical geometry.  Ascend
        the task-local hierarchy until reaching either a model root or the
        direct child of the task base; never move the task base itself.
        """

        current = anchor
        while current != task_base:
            is_model = getattr(current, "is_model", None)
            if callable(is_model) and bool(is_model()):
                return current
            parent = current.get_parent()
            if parent is None or parent == task_base:
                return current
            current = parent
        raise RuntimeError("environment change resolved to the task base")

    @classmethod
    def _minimal_semantic_physical_root(
        cls,
        anchor: Any,
        task_base: Any,
        independent_anchors: tuple[Any, ...],
    ) -> Any:
        """Promote an anchor without absorbing another semantic entity."""

        current = anchor
        anchor_handles = cls._tree_handles(anchor)
        blockers: set[int] = set()
        for root in independent_anchors:
            handles = cls._tree_handles(root)
            if handles & anchor_handles:
                continue
            blockers.update(handles)
        while current != task_base:
            is_model = getattr(current, "is_model", None)
            if callable(is_model) and bool(is_model()):
                return current
            parent = current.get_parent()
            if parent is None or parent == task_base:
                return current
            current_handles = cls._tree_handles(current)
            parent_handles = cls._tree_handles(parent)
            if (parent_handles - current_handles) & blockers:
                return current
            current = parent
        raise RuntimeError("environment change resolved to the task base")

    def _semantic_records(
        self,
        observation: Any,
        base: Any,
    ) -> list[dict[str, Any]]:
        poses = self.task.spec.extract_pose_chunks(
            _task_state(observation), convention="rlbench_xyzw"
        )
        chunks = {value.name: value for value in self.task.spec.pose_chunks}
        independent_names = tuple(
            str(name)
            for name in poses
            if str(chunks[name].role) not in PASSIVE_SUPPORT_ROLES
        )
        independent = tuple(
            self._semantic_anchors[str(name)] for name in independent_names
        )
        records = []
        for name, pose in poses.items():
            semantic_name = str(name)
            anchor = self._semantic_anchors[semantic_name]
            records.append(
                {
                    "semantic_name": semantic_name,
                    "semantic_role": str(chunks[name].role),
                    "pose": np.asarray(pose, dtype=np.float64),
                    "anchor": anchor,
                    "physical_root": self._minimal_semantic_physical_root(
                        anchor, base, independent
                    ),
                }
            )
        operated = [
            record
            for record in records
            if record["semantic_role"] in DIRECTLY_OPERATED_ROLES
        ]
        for record in records:
            root_handles = self._tree_handles(record["physical_root"])
            record["absorbed_operated_semantics"] = sorted(
                other["semantic_name"]
                for other in operated
                if other["semantic_name"] != record["semantic_name"]
                and self._object_handle(other["anchor"]) in root_handles
            )
        return records

    @staticmethod
    def _selected_semantic_record(
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        priority = (
            "environment_reference",
            "placement_reference",
            "scene_entity",
            "articulated_object",
            "cooperatively_operated_tool",
            "cooperatively_operated_object",
        )
        represented_priority_role = False
        for role in priority:
            candidates = [
                record
                for record in records
                if record["semantic_role"] == role
            ]
            represented_priority_role = represented_priority_role or bool(candidates)
            for record in candidates:
                if not record["absorbed_operated_semantics"]:
                    return record
        if represented_priority_role:
            raise RuntimeError(
                "no independently movable task reference is eligible for environment change"
            )
        for record in records:
            if not record["absorbed_operated_semantics"]:
                return record
        raise RuntimeError(
            "no independently movable semantic entity is eligible for environment change"
        )

    def _move_entity(self, observation: Any) -> Mapping[str, Any]:
        live_task = self._environment._scene.task
        selected_index = getattr(live_task, "_current_index", 0)
        joints = getattr(live_task, "_joints", None)
        before_state = _task_state(observation)
        semantic_mapping = []
        selected_semantic_name = None
        if isinstance(joints, (list, tuple)) and joints:
            joint = joints[int(selected_index)]
            before = float(joint.get_joint_position())
            joint.set_joint_position(before + float(self.spec.articulation))
            target_name = str(joint.get_name())
            intervention = "articulation"
            preserved = []
            contact_evidence = []
        else:
            base = live_task.get_base()
            records = self._semantic_records(observation, base)
            selected = self._selected_semantic_record(records)
            selected_semantic_name = str(selected["semantic_name"])
            semantic_anchor = selected["anchor"]
            target = selected["physical_root"]
            preserved, contact_evidence = self._support_component_roots(
                target,
                self._semantic_support_roots(records),
                grasped_handles=self._grasped_handles(
                    self._environment._scene
                ),
                contact_tolerance_m=float(
                    self.spec.support_contact_tolerance_m
                ),
            )
            delta = np.asarray(self.spec.translation, dtype=np.float64)
            before = np.asarray(target.get_position(), dtype=np.float64)
            target.set_position(before + delta)
            for root in preserved:
                root.set_position(
                    np.asarray(root.get_position(), dtype=np.float64) + delta
                )
            target_name = str(target.get_name())
            semantic_anchor_name = str(semantic_anchor.get_name())
            semantic_mapping = [
                {
                    "semantic_name": str(record["semantic_name"]),
                    "semantic_role": str(record["semantic_role"]),
                    "anchor": str(record["anchor"].get_name()),
                    "physical_root": str(record["physical_root"].get_name()),
                    "absorbed_operated_semantics": list(
                        record["absorbed_operated_semantics"]
                    ),
                }
                for record in records
            ]
            intervention = "translation"
        refreshed = self._environment.get_observation()
        displacement = float(np.linalg.norm(_task_state(refreshed) - before_state))
        self._effect_observed = displacement > 1e-8
        self._effect_policy_step = self._policy_step if self._effect_observed else None
        return {
            "kind": "environment_change",
            "policy_step": self._policy_step,
            "target_object": target_name,
            **(
                {
                    "semantic_anchor": semantic_anchor_name,
                    "selected_semantic_name": selected_semantic_name,
                    "semantic_mapping": semantic_mapping,
                    "entity_binding": "initial_full_se3_with_public_graspable_tie_break",
                    "physical_root_rule": "minimal_root_without_independent_operated_entity",
                }
                if intervention == "translation"
                else {}
            ),
            "intervention": intervention,
            "task_state_l2_change": displacement,
            "protocol_effective": self._effect_observed,
            "support_component_rule": (
                "ungrasped_semantic_entities_within_shape_distance"
            ),
            "support_contact_tolerance_m": float(
                self.spec.support_contact_tolerance_m
            ),
            "preserved_support_entities": [
                str(root.get_name()) for root in preserved
            ],
            "support_contact_evidence": contact_evidence,
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
                support_contact_tolerance_m=float(
                    eligibility["environment_support_contact_tolerance_m"]
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
