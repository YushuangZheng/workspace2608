"""Independent simulator-ground-truth audit for physical execution events."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from evaluations.iclr2027.interfaces.feature_schema import AUDIT_SCHEMA


def _task_state(observation: Any) -> np.ndarray:
    value = observation.task_low_dim_state
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return np.asarray(value, dtype=np.float64).reshape(-1)


class PhysicalEventAuditor:
    """Audit physical predicates without reading policy beliefs or alarms."""

    def __init__(
        self,
        task_environment: Any,
        task: Any,
        *,
        family: Optional[str],
        target_arm: str,
        earliest_cycle: int = 0,
        motion_threshold: float = 0.005,
        effect_tolerance: float = 0.001,
    ) -> None:
        self.environment = task_environment
        self.task = task
        self.family = family
        self.target_arm = target_arm
        self.earliest_cycle = int(earliest_cycle)
        if self.earliest_cycle < 0:
            raise ValueError("earliest audit cycle must be non-negative")
        self.motion_threshold = float(motion_threshold)
        self.effect_tolerance = float(effect_tolerance)
        self.eligible = False
        self.physically_triggered = False
        self.violation_onset_cycle = None
        self.violation_end_cycle = None
        self.relation_restored_cycle = None
        self.legal_reentry_cycle = None
        self._stable_relation_cycles = 0
        self._composed_motion_eligible = False
        self._composed_relation_eligible = False
        self._pre = None
        self._last = None
        self._target_objects = []
        self._interaction_candidates = None
        self._previous_close = {arm: False for arm in self._arms()}
        self._close_interval_eligible = {arm: False for arm in self._arms()}
        self._close_interval_counted = {arm: False for arm in self._arms()}
        self._initial_task_state = _task_state(task_environment.get_observation())

    def _robot(self) -> Any:
        return self.environment._scene.robot

    def _arms(self) -> tuple:
        return ("left", "right") if self.task.spec.bimanual else ("single",)

    def _gripper(self, arm: str) -> Any:
        robot = self._robot()
        return getattr(robot, arm + "_gripper") if self.task.spec.bimanual else robot.gripper

    def _arm_pose(self, observation: Any, arm: str) -> np.ndarray:
        value = getattr(observation, arm).gripper_pose if self.task.spec.bimanual else observation.gripper_pose
        return np.asarray(value, dtype=np.float64)

    def _arm_action(self, action: np.ndarray, arm: str) -> tuple:
        if not self.task.spec.bimanual:
            return action[:7], float(action[7])
        return (action[:7], float(action[7])) if arm == "right" else (action[9:16], float(action[16]))

    def _relations(self) -> dict:
        result = {}
        for arm in self._arms():
            gripper = self._gripper(arm)
            grasped = sorted(str(obj.get_name()) for obj in gripper.get_grasped_objects())
            result[arm] = {
                "state": "linked" if grasped else "external",
                "objects": grasped,
            }
        return result

    def _interaction_detected(self, arm: str) -> bool:
        """Independently test the physical target predicate for a close.

        A close-valued command in free space is not an eligible missed
        interaction.  Eligibility requires the selected gripper's proximity
        sensor to detect a graspable or dynamic/respondable task object.
        """

        live_task = self.environment._scene.task
        if self._interaction_candidates is None:
            candidates = []
            graspables = getattr(live_task, "get_graspable_objects", None)
            if callable(graspables):
                candidates.extend(graspables())
            dynamic_candidates = []
            get_base = getattr(live_task, "get_base", None)
            if callable(get_base):
                get_tree = getattr(get_base(), "get_objects_in_tree", None)
                if callable(get_tree):
                    for obj in get_tree(exclude_base=False):
                        dynamic = getattr(obj, "is_dynamic", None)
                        respondable = getattr(obj, "is_respondable", None)
                        if (
                            callable(dynamic)
                            and callable(respondable)
                            and bool(dynamic())
                            and bool(respondable())
                        ):
                            dynamic_candidates.append(obj)
            # Contact-style task targets (for example tray handles) are not
            # always registered as graspables.  Select the dynamic task-tree
            # shapes nearest the frozen semantic entity poses rather than
            # querying every decorative sub-shape on every close cycle.
            semantic_poses = self.task.spec.extract_pose_chunks(
                _task_state(self.environment.get_observation()),
                convention="rlbench_xyzw",
            )
            positioned = []
            for obj in dynamic_candidates:
                getter = getattr(obj, "get_position", None)
                if not callable(getter):
                    continue
                position = np.asarray(getter(), dtype=np.float64)
                if position.shape == (3,) and np.all(np.isfinite(position)):
                    positioned.append((obj, position))
            for pose in semantic_poses.values():
                if positioned:
                    candidates.append(
                        min(
                            positioned,
                            key=lambda item: float(
                                np.linalg.norm(item[1] - np.asarray(pose)[:3])
                            ),
                        )[0]
                    )
            unique = []
            seen = set()
            for obj in candidates:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    unique.append(obj)
            self._interaction_candidates = tuple(unique)
        sensor = getattr(self._gripper(arm), "_proximity_sensor", None)
        detects = getattr(sensor, "is_detected", None)
        if not callable(detects):
            return False
        for obj in self._interaction_candidates:
            if bool(detects(obj)):
                return True
        return False

    def snapshot(self, observation: Any) -> dict:
        success, terminate = self.environment._scene.task.success()
        return {
            "ee": {arm: self._arm_pose(observation, arm) for arm in self._arms()},
            "task_state": _task_state(observation),
            "relations": self._relations(),
            "success": bool(success),
            "terminate": bool(terminate),
        }

    def before_step(self, cycle: int, observation: Any, action: Any) -> None:
        command = np.asarray(action, dtype=np.float64)
        self._pre = self.snapshot(observation)
        selected = self._arms() if self.target_arm == "all" else (self.target_arm,)
        motion = []
        closes = []
        linked = []
        interactions = []
        for arm in selected:
            target, gripper = self._arm_action(command, arm)
            motion.append(float(np.linalg.norm(target[:3] - self._pre["ee"][arm][:3])))
            close = bool(gripper <= 0.5)
            closes.append(close)
            linked.append(self._pre["relations"][arm]["state"] == "linked")
            if self.family == "missed_interaction":
                if close and not self._previous_close[arm]:
                    self._close_interval_counted[arm] = False
                    self._close_interval_eligible[arm] = cycle >= self.earliest_cycle
                if not close:
                    self._close_interval_counted[arm] = False
                    self._close_interval_eligible[arm] = False
                detected = False
                if (
                    close
                    and self._close_interval_eligible[arm]
                    and not self._close_interval_counted[arm]
                ):
                    detected = self._interaction_detected(arm)
                    if detected:
                        self._close_interval_counted[arm] = True
                interactions.append(detected)
                self._previous_close[arm] = close
        if cycle < self.earliest_cycle:
            return
        if self.family in {"actuation_delay", "environment_change"}:
            self.eligible = self.eligible or any(value >= self.motion_threshold for value in motion)
        elif self.family == "coordination_delay":
            self.eligible = self.eligible or (
                self.task.spec.bimanual and any(value >= self.motion_threshold for value in motion)
            )
        elif self.family == "missed_interaction":
            self.eligible = self.eligible or any(
                close and detected
                for close, detected in zip(closes, interactions)
            )
        elif self.family == "relation_loss":
            self._stable_relation_cycles = self._stable_relation_cycles + 1 if any(linked) else 0
            self.eligible = self.eligible or self._stable_relation_cycles >= 3
        elif self.family == "composed_event":
            # The frozen composed protocol is an early actuation delay followed
            # by a late relation loss.  Eligibility requires both independent
            # physical predicates, never merely the scheduled family label.
            all_motion = []
            for arm in self._arms():
                target, _gripper = self._arm_action(command, arm)
                all_motion.append(
                    float(np.linalg.norm(target[:3] - self._pre["ee"][arm][:3]))
                )
            self._composed_motion_eligible = self._composed_motion_eligible or any(
                value >= self.motion_threshold for value in all_motion
            )
            self._stable_relation_cycles = self._stable_relation_cycles + 1 if any(linked) else 0
            self._composed_relation_eligible = (
                self._composed_relation_eligible or self._stable_relation_cycles >= 3
            )
            self.eligible = self._composed_motion_eligible and self._composed_relation_eligible

    def after_step(
        self,
        cycle: int,
        observation: Any,
        injector: Optional[Mapping[str, Any]],
    ) -> dict:
        current = self.snapshot(observation)
        triggered = bool(injector and injector.get("triggered") is True)
        effect_claimed = bool(
            injector and injector.get("physical_effect_observed") is True
        )
        selected = self._arms() if self.target_arm == "all" else (self.target_arm,)
        physical_effect = False
        if triggered and self._pre is not None:
            if self.family in {"actuation_delay", "coordination_delay"}:
                movements = [
                    float(np.linalg.norm(current["ee"][arm][:3] - self._pre["ee"][arm][:3]))
                    for arm in selected
                ]
                physical_effect = effect_claimed and any(
                    movement <= self.effect_tolerance for movement in movements
                )
            elif self.family in {"missed_interaction", "relation_loss"}:
                physical_effect = effect_claimed and any(
                    current["relations"][arm]["state"] == "external"
                    for arm in selected
                )
            elif self.family == "environment_change":
                physical_effect = effect_claimed and float(
                    np.linalg.norm(current["task_state"] - self._pre["task_state"])
                ) > 1e-8
            elif self.family == "composed_event":
                physical_effect = effect_claimed
        if self.eligible and triggered and physical_effect:
            self.physically_triggered = True
            if self.violation_onset_cycle is None:
                self.violation_onset_cycle = cycle
                if injector:
                    self._target_objects = list(injector.get("target_objects", ()))
        if self.violation_onset_cycle is not None and self.violation_end_cycle is None:
            if self.family in {"actuation_delay", "coordination_delay"}:
                ended = any(
                    event.get("kind") == "time_stall_ended"
                    for event in (injector or {}).get("events", ())
                )
                if ended:
                    self.violation_end_cycle = cycle
            elif self.family in {"missed_interaction", "relation_loss"}:
                if any(
                    current["relations"][arm]["state"] == "linked"
                    for arm in selected
                ):
                    self.violation_end_cycle = cycle
                    self.relation_restored_cycle = cycle
        self._last = current
        return self.cycle_record(cycle, current)

    def cycle_record(self, cycle: int, snapshot: Optional[Mapping[str, Any]] = None) -> dict:
        snapshot = snapshot or self._last or self.snapshot(self.environment.get_observation())
        expected_relation = None
        if self.family in {"missed_interaction", "relation_loss"}:
            expected_relation = {arm: "linked" for arm in (self._arms() if self.target_arm == "all" else (self.target_arm,))}
        return {
            "schema": AUDIT_SCHEMA,
            "cycle": int(cycle),
            "eligible": bool(self.eligible),
            "physically_triggered": bool(self.physically_triggered),
            "violation_onset_cycle": self.violation_onset_cycle,
            "violation_end_cycle": self.violation_end_cycle,
            "expected_relation": expected_relation,
            "physical_relation": {
                arm: values["state"] for arm, values in snapshot["relations"].items()
            },
            "relation_restored_cycle": self.relation_restored_cycle,
            "task_boundary_state": None,
            "legal_reentry_cycle": self.legal_reentry_cycle,
            "oracle_recoverable": bool(snapshot["success"] or not snapshot["terminate"]),
            "task_success": bool(snapshot["success"]),
        }

    def summary(self) -> dict:
        return {
            "schema": AUDIT_SCHEMA,
            "eligible": bool(self.eligible),
            "physically_triggered": bool(self.physically_triggered),
            "violation_onset_cycle": self.violation_onset_cycle,
            "violation_end_cycle": self.violation_end_cycle,
            "relation_restored_cycle": self.relation_restored_cycle,
            "legal_reentry_cycle": self.legal_reentry_cycle,
            "target_objects": list(self._target_objects),
            "oracle_recoverable": (
                None
                if self._last is None
                else bool(self._last["success"] or not self._last["terminate"])
            ),
        }


__all__ = ["PhysicalEventAuditor"]
