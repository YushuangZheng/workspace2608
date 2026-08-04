"""Transient receiver faults for relation-gated physical handover recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from essay2608.data.handover_schema import HandoverState


BIMANUAL_RECOVERY_METHODS = (
    "clocked_expert",
    "relation_gate",
    "relation_recovery",
    "oracle_relation_recovery",
)

BIMANUAL_RECOVERY_CONDITIONS = (
    "normal",
    "receiver_miss_once",
    "receiver_brief_loss",
    "receiver_loss_once",
    "receiver_loss_after_release",
)


@dataclass(frozen=True)
class BimanualRecoveryInterventionDecision:
    """One transient fault action and auditable event label."""

    action: np.ndarray
    active: bool
    event: str


class BimanualRecoveryIntervention:
    """Apply each receiver fault once without consulting inferred relation."""

    def __init__(self, condition: str, control_dt_s: float) -> None:
        if condition not in BIMANUAL_RECOVERY_CONDITIONS:
            raise ValueError(f"未知双臂恢复条件：{condition}")
        self.condition = condition
        self.control_dt_s = float(control_dt_s)
        self.started = False
        self.remaining_steps = 0
        self.transfer_steps = 0
        self.fault_target: np.ndarray | None = None

    def _start(
        self,
        *,
        right_pose: np.ndarray,
        duration_s: float,
        displacement_xyz_m: tuple[float, float, float],
    ) -> None:
        self.started = True
        self.remaining_steps = round(duration_s / self.control_dt_s)
        self.fault_target = np.asarray(right_pose[:3], dtype=np.float64) + np.asarray(
            displacement_xyz_m,
            dtype=np.float64,
        )

    def apply(
        self,
        *,
        action: np.ndarray,
        task_state: HandoverState,
        right_pose: np.ndarray,
    ) -> BimanualRecoveryInterventionDecision:
        modified = np.asarray(action, dtype=np.float64).copy()
        if self.condition == "normal":
            return BimanualRecoveryInterventionDecision(modified, False, "none")

        if self.condition == "receiver_miss_once" and not self.started:
            if task_state == HandoverState.RIGHT_GRASP and modified[15] < 0.0:
                self._start(
                    right_pose=right_pose,
                    duration_s=0.80,
                    displacement_xyz_m=(0.050, 0.0, 0.060),
                )
        elif self.condition in {
            "receiver_brief_loss",
            "receiver_loss_once",
            "receiver_loss_after_release",
        } and not self.started:
            expected_state = (
                HandoverState.RIGHT_TO_TARGET
                if self.condition == "receiver_loss_after_release"
                else HandoverState.TRANSFER
            )
            if task_state == expected_state:
                self.transfer_steps += 1
                if self.transfer_steps >= round(0.30 / self.control_dt_s):
                    duration = 0.12 if self.condition == "receiver_brief_loss" else 0.80
                    displacement = (
                        (0.020, 0.0, 0.020)
                        if self.condition == "receiver_brief_loss"
                        else (0.060, 0.0, 0.060)
                    )
                    self._start(
                        right_pose=right_pose,
                        duration_s=duration,
                        displacement_xyz_m=displacement,
                    )

        if self.remaining_steps <= 0:
            return BimanualRecoveryInterventionDecision(modified, False, "none")
        if self.fault_target is None:
            raise RuntimeError("接收故障已启动但缺少冻结动作目标")

        modified[8:11] = self.fault_target
        modified[15] = 1.0
        self.remaining_steps -= 1
        event = {
            "receiver_miss_once": "receiver_initial_grasp_blocked",
            "receiver_brief_loss": "receiver_briefly_forced_open",
            "receiver_loss_once": "receiver_forced_open_and_displaced",
            "receiver_loss_after_release": "receiver_lost_after_giver_release",
        }[self.condition]
        return BimanualRecoveryInterventionDecision(modified, True, event)


def fault_realization(
    condition: str,
    arrays: dict[str, np.ndarray],
    control_dt_s: float,
) -> dict:
    """Verify a declared fault from physical truth rather than its command."""

    if condition not in BIMANUAL_RECOVERY_CONDITIONS:
        raise ValueError(f"未知双臂恢复条件：{condition}")
    active = np.asarray(arrays["intervention_active"], dtype=bool)
    truth_right = np.asarray(arrays["truth_right_connected"], dtype=bool)
    truth_left = np.asarray(arrays["truth_left_connected"], dtype=bool)
    if len({len(active), len(truth_left), len(truth_right)}) != 1 or not len(active):
        raise ValueError("故障动作与双侧物理关系必须非空等长")
    active_indices = np.flatnonzero(active)
    expected_steps = {
        "normal": 0,
        "receiver_miss_once": round(0.80 / control_dt_s),
        "receiver_brief_loss": round(0.12 / control_dt_s),
        "receiver_loss_once": round(0.80 / control_dt_s),
        "receiver_loss_after_release": round(0.80 / control_dt_s),
    }[condition]
    checks = {"exact_intervention_steps": len(active_indices) == expected_steps}
    if condition == "normal":
        checks["no_fault_applied"] = not len(active_indices)
    elif condition == "receiver_miss_once":
        checks["receiver_never_connected_during_fault"] = bool(
            len(active_indices) and not np.any(truth_right[active_indices])
        )
    elif len(active_indices):
        first = int(active_indices[0])
        end = min(len(truth_right), int(active_indices[-1]) + 11)
        checks["receiver_connected_before_fault"] = bool(np.any(truth_right[:first]))
        checks["receiver_physically_lost"] = bool(np.any(~truth_right[first:end]))
        if condition == "receiver_loss_after_release":
            checks["giver_disconnected_before_fault"] = bool(not truth_left[first])
    else:
        checks["receiver_connected_before_fault"] = False
        checks["receiver_physically_lost"] = False
    return {
        "condition": condition,
        "realized": bool(all(checks.values())),
        "checks": checks,
        "intervention_steps": int(len(active_indices)),
        "intervention_duration_s": float(len(active_indices) * control_dt_s),
        "first_intervention_step": int(active_indices[0]) if len(active_indices) else None,
        "last_intervention_step": int(active_indices[-1]) if len(active_indices) else None,
    }


def _maximum_step_norm(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    differences = np.linalg.norm(np.diff(np.asarray(values, dtype=np.float64), axis=0), axis=-1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(values):
            raise ValueError("步进指标 mask 与序列不等长")
        differences = differences[mask[:-1] | mask[1:]]
    return float(np.max(differences)) if len(differences) else None


def score_bimanual_recovery_trace(
    arrays: dict[str, np.ndarray],
    condition: str,
    control_dt_s: float,
    *,
    method: str | None = None,
    task_success: bool | None = None,
) -> dict:
    """Score safety, latency, path, and bounded recovery from step traces."""

    states = np.asarray(arrays["state"], dtype=int)
    truth_left = np.asarray(arrays["truth_left_connected"], dtype=bool)
    truth_right = np.asarray(arrays["truth_right_connected"], dtype=bool)
    inferred_right = np.asarray(arrays["inferred_right_connected"], dtype=bool)
    intervention = np.asarray(arrays["intervention_active"], dtype=bool)
    recovery_state = np.asarray(arrays["recovery_state"]).astype("U64")
    recovery_transition = np.asarray(arrays["recovery_transition"]).astype("U128")
    requires_giver = np.asarray(arrays["recovery_requires_giver"], dtype=bool)
    lengths = {
        len(states),
        len(truth_left),
        len(truth_right),
        len(inferred_right),
        len(intervention),
        len(recovery_state),
        len(recovery_transition),
        len(requires_giver),
    }
    if len(lengths) != 1 or not len(states):
        raise ValueError("双臂恢复评分字段必须非空等长")

    active_recovery = ~np.isin(recovery_state, ["NOT_APPLICABLE", "NORMAL"])
    release_indices = np.flatnonzero(states >= int(HandoverState.LEFT_RELEASE))
    release_attempted = bool(len(release_indices))
    first_release = int(release_indices[0]) if release_attempted else None
    receiver_connected_at_release = bool(
        first_release is not None and truth_right[first_release]
    )
    inferred_receiver_at_release = bool(
        first_release is not None and inferred_right[first_release]
    )
    recovery_onsets = np.flatnonzero(active_recovery)
    resume_indices = np.flatnonzero(recovery_state == "RESUME_TASK")
    cancellation_indices = np.flatnonzero(
        recovery_transition == "RECEIVER_LOSS_DETECTED->NORMAL"
    )
    fault_indices = np.flatnonzero(intervention)
    first_fault = int(fault_indices[0]) if len(fault_indices) else None
    last_fault = int(fault_indices[-1]) if len(fault_indices) else None
    recovery_start = int(recovery_onsets[0]) if len(recovery_onsets) else None
    completion_indices = np.sort(np.concatenate((resume_indices, cancellation_indices)))
    recovery_resume = int(completion_indices[0]) if len(completion_indices) else None
    geometry_states = np.isin(
        recovery_state,
        [
            "SAFE_RETREAT",
            "REAPPROACH",
            "REGRASP",
            "VERIFY_BOTH",
            "VERIFY_RECEIVER",
        ],
    )

    reestablished = None
    if condition != "normal" and last_fault is not None:
        reestablished = bool(np.any(truth_right[last_fault + 1 :]))
    loss_transitions = np.flatnonzero(truth_right[:-1] & ~truth_right[1:]) + 1
    loss_after_fault = (
        loss_transitions[loss_transitions >= first_fault]
        if first_fault is not None
        else np.asarray([], dtype=int)
    )
    truth_loss_step = int(loss_after_fault[0]) if len(loss_after_fault) else None
    intended_release_indices = np.flatnonzero(states >= int(HandoverState.RIGHT_RELEASE))
    intended_release = (
        int(intended_release_indices[0]) if len(intended_release_indices) else len(states)
    )
    post_recovery_receiver = (
        truth_right[recovery_resume:intended_release]
        if recovery_resume is not None and recovery_resume < intended_release
        else np.asarray([], dtype=bool)
    )

    left_path = float(
        np.sum(np.linalg.norm(np.diff(arrays["left_ee_pose"][:, :3], axis=0), axis=-1))
    )
    right_path = float(
        np.sum(np.linalg.norm(np.diff(arrays["right_ee_pose"][:, :3], axis=0), axis=-1))
    )
    action_jump_values = [
        value
        for value in (
            _maximum_step_norm(arrays["applied_action"][:, :3]),
            _maximum_step_norm(arrays["applied_action"][:, 8:11]),
        )
        if value is not None
    ]
    recovery_action_jump_values = [
        value
        for value in (
            _maximum_step_norm(arrays["applied_action"][:, :3], active_recovery),
            _maximum_step_norm(arrays["applied_action"][:, 8:11], active_recovery),
        )
        if value is not None
    ]
    ee_step_values = [
        value
        for value in (
            _maximum_step_norm(arrays["left_ee_pose"][:, :3]),
            _maximum_step_norm(arrays["right_ee_pose"][:, :3]),
        )
        if value is not None
    ]
    recovery_ee_step_values = [
        value
        for value in (
            _maximum_step_norm(arrays["left_ee_pose"][:, :3], active_recovery),
            _maximum_step_norm(arrays["right_ee_pose"][:, :3], active_recovery),
        )
        if value is not None
    ]
    completed = bool(recovery_start is not None and recovery_resume is not None)
    scored_recovery_success = (
        bool(completed and task_success)
        if method in {"relation_recovery", "oracle_relation_recovery"}
        and condition != "normal"
        and task_success is not None
        else None
    )
    required_giver_mask = active_recovery & requires_giver
    return {
        "release_attempted": release_attempted,
        "receiver_connected_at_release": receiver_connected_at_release,
        "inferred_receiver_connected_at_release": inferred_receiver_at_release,
        "unsafe_release": bool(release_attempted and not receiver_connected_at_release),
        "safe_release": bool(release_attempted and receiver_connected_at_release),
        "recovery_triggered": bool(len(recovery_onsets)),
        "geometry_recovery_executed": bool(np.any(geometry_states)),
        "transient_loss_cancelled": bool(len(cancellation_indices)),
        "recovery_start_step": recovery_start,
        "recovery_resume_step": recovery_resume,
        "time_to_recover_s": (
            float((recovery_resume - recovery_start) * control_dt_s)
            if recovery_start is not None and recovery_resume is not None
            else None
        ),
        "recovery_trigger_latency_from_fault_s": (
            float((recovery_start - first_fault) * control_dt_s)
            if recovery_start is not None and first_fault is not None
            else None
        ),
        "truth_receiver_loss_step": truth_loss_step,
        "signed_detection_delay_from_truth_loss_s": (
            float((recovery_start - truth_loss_step) * control_dt_s)
            if recovery_start is not None and truth_loss_step is not None
            else None
        ),
        "relation_reestablished_after_fault": reestablished,
        "recovery_completed": completed,
        "recovery_success": scored_recovery_success,
        "false_recovery_trigger": bool(condition == "normal" and len(recovery_onsets)),
        "giver_retention_required": bool(np.any(required_giver_mask)),
        "giver_retained_during_recovery": (
            bool(np.all(truth_left[required_giver_mask]))
            if np.any(required_giver_mask)
            else None
        ),
        "receiver_retained_until_intended_release": (
            bool(np.all(post_recovery_receiver))
            if len(post_recovery_receiver)
            else None
        ),
        "maximum_regrasp_attempts_observed": int(np.max(arrays["regrasp_attempts"])),
        "task_clock_held_steps": int(np.count_nonzero(arrays["phase_clock_held"])),
        "left_path_length_m": left_path,
        "right_path_length_m": right_path,
        "maximum_action_target_jump_m": max(action_jump_values),
        "maximum_recovery_action_target_jump_m": (
            max(recovery_action_jump_values) if recovery_action_jump_values else None
        ),
        "maximum_ee_speed_m_s": max(ee_step_values) / control_dt_s,
        "maximum_recovery_ee_speed_m_s": (
            max(recovery_ee_step_values) / control_dt_s
            if recovery_ee_step_values
            else None
        ),
    }


def task_outcome_from_trace(
    *,
    expert_complete: bool,
    expert_failed: bool,
    expert_failure_reason: str | None,
    recovery_failed: bool,
    environment_done: bool,
    object_positions: np.ndarray,
    final_position: np.ndarray,
    target_position: np.ndarray,
) -> tuple[bool, str, dict]:
    """Recompute terminal task success from persisted physical evidence."""

    object_positions = np.asarray(object_positions, dtype=np.float64)
    final_position = np.asarray(final_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    if object_positions.ndim != 2 or object_positions.shape[1] != 3 or not len(object_positions):
        raise ValueError("终局评分要求非空的三维物体轨迹")
    if final_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("终局物体和目标位置必须为三维向量")
    settling = object_positions[-25:]
    settling_displacement = float(
        np.max(np.linalg.norm(settling - settling[-1], axis=-1))
    )
    final_xy_error = float(np.linalg.norm(final_position[:2] - target_position[:2]))
    on_support = bool(abs(final_position[2] - target_position[2]) <= 0.025)
    stable = settling_displacement <= 0.01
    if recovery_failed:
        reason = "recovery_failed"
    elif expert_failed:
        reason = expert_failure_reason or "expert_failed_without_reason"
    elif not expert_complete:
        reason = "expert_incomplete"
    elif environment_done:
        reason = "environment_done"
    elif not on_support:
        reason = "object_not_on_support"
    elif not stable:
        reason = "object_not_stable"
    elif final_xy_error >= 0.04:
        reason = "placement_xy_above_threshold"
    else:
        reason = "success"
    return reason == "success", reason, {
        "final_xy_error_m": final_xy_error,
        "object_on_support": on_support,
        "stable": stable,
        "settling_displacement_m": settling_displacement,
    }
