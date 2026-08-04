"""Metrics and semantic success criteria for the single-arm ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuccessCriteria:
    """Placement semantics that avoid conflating support height with XY error."""

    legacy_3d_threshold_m: float = 0.06
    xy_threshold_m: float = 0.01
    xy_sensitivity_thresholds_m: tuple[float, ...] = (0.005, 0.01, 0.02)
    support_height_m: float = 0.021
    support_height_tolerance_m: float = 0.01
    stability_window_steps: int = 25
    stability_displacement_m: float = 0.005
    stability_speed_m_s: float = 0.05


@dataclass
class EpisodeTrace:
    """Compact rollout trace used to compute paper-facing metrics."""

    control_dt: float
    ee_positions: list[np.ndarray] = field(default_factory=list)
    object_positions: list[np.ndarray] = field(default_factory=list)
    target_positions: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    connected: list[bool] = field(default_factory=list)
    perturbation_active: list[bool] = field(default_factory=list)
    perturbation_events: list[bool] = field(default_factory=list)
    active_frames: list[list[str]] = field(default_factory=list)
    stream_weights: list[dict[str, float]] = field(default_factory=list)
    raw_action_positions: list[np.ndarray] = field(default_factory=list)
    policy_action_positions: list[np.ndarray] = field(default_factory=list)
    action_rate_limited: list[bool] = field(default_factory=list)
    relation_states: list[str] = field(default_factory=list)
    relation_confidence: list[float] = field(default_factory=list)
    gripper_opening_m: list[float] = field(default_factory=list)
    gripper_velocity_m_s: list[float] = field(default_factory=list)
    terminal_ee_position: np.ndarray | None = None
    terminal_object_position: np.ndarray | None = None
    terminal_target_position: np.ndarray | None = None
    recovery_states: list[str] = field(default_factory=list)
    recovery_triggers: list[str] = field(default_factory=list)
    regrasp_attempts: list[int] = field(default_factory=list)

    def append(
        self,
        observation,
        action: np.ndarray,
        diagnostics: dict[str, Any],
        inference_ms: float,
        perturbation_active: bool,
        perturbation_event: bool = False,
    ) -> None:
        self.ee_positions.append(observation.ee_pose[:3].copy())
        self.object_positions.append(observation.object_pose[:3].copy())
        self.target_positions.append(observation.target_pose[:3].copy())
        self.actions.append(action.copy())
        self.phases.append(int(diagnostics["phase"]))
        self.inference_ms.append(float(inference_ms))
        self.connected.append(bool(diagnostics.get("connected", False)))
        self.perturbation_active.append(bool(perturbation_active))
        self.perturbation_events.append(bool(perturbation_event))
        self.active_frames.append(list(diagnostics.get("active_frames", [])))
        self.stream_weights.append(dict(diagnostics.get("stream_weights", {})))
        self.raw_action_positions.append(
            np.asarray(diagnostics.get("raw_action_position", action[:3]), dtype=np.float64)
        )
        self.policy_action_positions.append(
            np.asarray(diagnostics.get("policy_action_position", action[:3]), dtype=np.float64)
        )
        self.action_rate_limited.append(bool(diagnostics.get("action_rate_limited", False)))
        self.relation_states.append(str(diagnostics.get("relation_state", "NOT_APPLICABLE")))
        self.relation_confidence.append(float(diagnostics.get("relation_confidence", 0.0)))
        self.gripper_opening_m.append(
            float(observation.gripper_opening_m)
            if observation.gripper_opening_m is not None
            else float("nan")
        )
        self.gripper_velocity_m_s.append(
            float(observation.gripper_velocity_m_s)
            if observation.gripper_velocity_m_s is not None
            else float("nan")
        )
        self.recovery_states.append(str(diagnostics.get("recovery_state", "NOT_APPLICABLE")))
        self.recovery_triggers.append(str(diagnostics.get("recovery_trigger", "NONE")))
        self.regrasp_attempts.append(int(diagnostics.get("regrasp_attempts", 0)))

    def set_terminal_observation(self, observation) -> None:
        """Persist the post-step terminal snapshot separately from action-aligned samples."""

        self.terminal_ee_position = np.asarray(observation.ee_pose[:3], dtype=np.float64).copy()
        self.terminal_object_position = np.asarray(observation.object_pose[:3], dtype=np.float64).copy()
        self.terminal_target_position = np.asarray(observation.target_pose[:3], dtype=np.float64).copy()

    @staticmethod
    def _jumps(positions: np.ndarray) -> np.ndarray:
        if len(positions) <= 1:
            return np.zeros(1, dtype=np.float64)
        return np.linalg.norm(np.diff(positions, axis=0), axis=-1)

    def summary(
        self,
        final_object_position: np.ndarray,
        final_target_position: np.ndarray,
        criteria: SuccessCriteria,
        policy_complete: bool,
        environment_done: bool,
        forced_transitions: int,
        perturbation_started: bool,
        relation_loss_expected: bool = False,
    ) -> dict[str, Any]:
        ee = np.asarray(self.ee_positions)
        actions = np.asarray(self.actions)
        phases = np.asarray(self.phases)
        connected = np.asarray(self.connected, dtype=bool)
        expected_connected = np.isin(phases, [4, 5, 6])

        ee_jumps = self._jumps(ee)
        path_length = float(np.sum(ee_jumps))
        phase_path_length = np.zeros(10, dtype=np.float64)
        if len(phases) > 1:
            np.add.at(phase_path_length, phases[1:], ee_jumps)
        phase_step_counts = np.bincount(phases, minlength=10)
        speed = ee_jumps / self.control_dt
        applied_action_jump = self._jumps(actions[:, :3])
        raw_action_jump = self._jumps(np.asarray(self.raw_action_positions))
        policy_action_jump = self._jumps(np.asarray(self.policy_action_positions))
        false_positive = float(np.mean(connected & ~expected_connected)) if len(connected) else 0.0
        false_negative = float(np.mean(~connected & expected_connected)) if len(connected) else 0.0
        expected_onsets = np.flatnonzero(expected_connected)
        observed_onsets = np.flatnonzero(connected)
        expected_onset = int(expected_onsets[0]) if len(expected_onsets) else None
        observed_onset = int(observed_onsets[0]) if len(observed_onsets) else None
        onset_delay_steps = (
            observed_onset - expected_onset
            if expected_onset is not None and observed_onset is not None
            else None
        )
        expected_releases = np.flatnonzero(expected_connected[:-1] & ~expected_connected[1:]) + 1
        expected_release = int(expected_releases[0]) if len(expected_releases) else None
        observed_release = None
        if expected_release is not None and observed_onset is not None:
            releases = np.flatnonzero(~connected[expected_release:])
            if len(releases):
                observed_release = expected_release + int(releases[0])
        release_delay_steps = (
            observed_release - expected_release
            if expected_release is not None and observed_release is not None
            else None
        )
        relation_transition_steps = [
            index
            for index in range(1, len(self.relation_states))
            if self.relation_states[index] != self.relation_states[index - 1]
        ]
        recovery_transition_steps = [
            index
            for index in range(1, len(self.recovery_states))
            if self.recovery_states[index] != self.recovery_states[index - 1]
        ]
        recovery_active = np.asarray(
            [state not in {"NOT_APPLICABLE", "NORMAL"} for state in self.recovery_states],
            dtype=bool,
        )
        recovery_onsets = np.flatnonzero(recovery_active)
        recovery_start_step = int(recovery_onsets[0]) if len(recovery_onsets) else None
        resume_steps = [index for index, state in enumerate(self.recovery_states) if state == "RESUME_TASK"]
        recovery_resume_step = resume_steps[0] if resume_steps else None
        recovery_failed = "RECOVERY_FAILED" in self.recovery_states
        observed_triggers = [trigger for trigger in self.recovery_triggers if trigger != "NONE"]
        recovery_trigger = observed_triggers[0] if observed_triggers else None
        event_steps = np.flatnonzero(np.asarray(self.perturbation_events, dtype=bool))
        perturbation_event_step = int(event_steps[0]) if len(event_steps) else None
        post_event_loss_step = None
        if (
            relation_loss_expected
            and perturbation_event_step is not None
            and np.any(connected[: perturbation_event_step + 1])
        ):
            losses = np.flatnonzero(~connected[perturbation_event_step:])
            if len(losses):
                post_event_loss_step = perturbation_event_step + int(losses[0])
        post_event_loss_delay_steps = (
            post_event_loss_step - perturbation_event_step
            if perturbation_event_step is not None and post_event_loss_step is not None
            else None
        )

        frame_switch_steps = [
            index
            for index in range(1, len(self.active_frames))
            if self.active_frames[index] != self.active_frames[index - 1]
        ]
        frame_switch_diagnostics = [
            {
                "step": index,
                "before": self.active_frames[index - 1],
                "after": self.active_frames[index],
                "raw_action_jump_m": float(raw_action_jump[index - 1]),
                "policy_action_jump_m": float(policy_action_jump[index - 1]),
                "applied_action_jump_m": float(applied_action_jump[index - 1]),
            }
            for index in frame_switch_steps
        ]
        object_weights = [weights.get("object", 0.0) for weights in self.stream_weights]

        final_object_position = np.asarray(final_object_position, dtype=np.float64)
        final_target_position = np.asarray(final_target_position, dtype=np.float64)
        # Keep metrics and the separately persisted terminal snapshot aligned,
        # including trials that terminate between two action-aligned samples.
        self.terminal_object_position = final_object_position.copy()
        self.terminal_target_position = final_target_position.copy()
        final_xy_error = float(
            np.linalg.norm(final_object_position[:2] - final_target_position[:2])
        )
        final_error_3d = float(np.linalg.norm(final_object_position - final_target_position))
        support_height_error = float(abs(final_object_position[2] - criteria.support_height_m))
        on_support = support_height_error <= criteria.support_height_tolerance_m
        released = bool(len(actions) and actions[-1, 7] > 0.0)

        settling = np.asarray(
            self.object_positions[-criteria.stability_window_steps :], dtype=np.float64
        )
        if len(settling):
            settling = np.concatenate((settling, final_object_position[None]), axis=0)
        settling_displacement = (
            float(np.max(np.linalg.norm(settling - settling[-1], axis=-1)))
            if len(settling)
            else float("inf")
        )
        settling_speed = (
            float(np.max(self._jumps(settling) / self.control_dt))
            if len(settling) > 1
            else float("inf")
        )
        stable = bool(
            len(settling) >= criteria.stability_window_steps
            and settling_displacement <= criteria.stability_displacement_m
            and settling_speed <= criteria.stability_speed_m_s
        )
        semantic_base = bool(
            policy_complete and not environment_done and on_support and released and stable
        )
        legacy_success = bool(
            policy_complete
            and not environment_done
            and final_error_3d < criteria.legacy_3d_threshold_m
        )
        sensitivity = {
            f"{threshold:.6f}": bool(semantic_base and final_xy_error < threshold)
            for threshold in criteria.xy_sensitivity_thresholds_m
        }
        success = bool(semantic_base and final_xy_error < criteria.xy_threshold_m)
        if success:
            failure_reason = "success"
        elif environment_done:
            failure_reason = "environment_terminated"
        elif recovery_failed:
            failure_reason = "recovery_failed"
        elif not policy_complete:
            failure_reason = "policy_incomplete"
        elif not released:
            failure_reason = "not_released"
        elif not on_support:
            failure_reason = "not_on_support"
        elif not stable:
            failure_reason = "unstable_after_release"
        else:
            failure_reason = "placement_xy_above_threshold"

        return {
            "success": success,
            "stable_place_success": success,
            "legacy_success_3d": legacy_success,
            "recovery_success": success if perturbation_started else None,
            "failure_reason": failure_reason,
            "policy_complete": bool(policy_complete),
            "environment_done": bool(environment_done),
            "steps": len(self.actions),
            "final_xy_error_m": final_xy_error,
            "final_error_3d_m": final_error_3d,
            "final_object_height_m": float(final_object_position[2]),
            "final_object_position_m": final_object_position.tolist(),
            "final_target_position_m": final_target_position.tolist(),
            "support_height_error_m": support_height_error,
            "object_on_support": on_support,
            "gripper_released": released,
            "stable_after_release": stable,
            "settling_displacement_m": settling_displacement,
            "settling_max_speed_m_s": settling_speed,
            "xy_success_sensitivity": sensitivity,
            "success_criteria": {
                "legacy_3d_threshold_m": criteria.legacy_3d_threshold_m,
                "xy_threshold_m": criteria.xy_threshold_m,
                "support_height_m": criteria.support_height_m,
                "support_height_tolerance_m": criteria.support_height_tolerance_m,
                "stability_window_steps": criteria.stability_window_steps,
                "stability_displacement_m": criteria.stability_displacement_m,
                "stability_speed_m_s": criteria.stability_speed_m_s,
            },
            "path_length_m": path_length,
            "phase_path_length_m": {
                str(index): float(value) for index, value in enumerate(phase_path_length)
            },
            "phase_step_counts": {
                str(index): int(value) for index, value in enumerate(phase_step_counts)
            },
            "max_ee_speed_m_s": float(np.max(speed)),
            "max_action_position_jump_m": float(np.max(applied_action_jump)),
            "max_raw_policy_action_jump_m": float(np.max(raw_action_jump)),
            "max_rate_limited_policy_action_jump_m": float(np.max(policy_action_jump)),
            "action_rate_limited_steps": int(np.sum(self.action_rate_limited)),
            "mean_inference_ms": float(np.mean(self.inference_ms)),
            "p95_inference_ms": float(np.percentile(self.inference_ms, 95)),
            "frame_switch_steps": frame_switch_steps,
            "frame_switch_diagnostics": frame_switch_diagnostics,
            "connection_detected": bool(np.any(connected)),
            "mask_false_positive_rate": false_positive,
            "mask_false_negative_rate": false_negative,
            "connection_onset_step": observed_onset,
            "expected_connection_onset_step": expected_onset,
            "connection_onset_delay_steps": onset_delay_steps,
            "connection_onset_delay_s": (
                onset_delay_steps * self.control_dt if onset_delay_steps is not None else None
            ),
            "connection_release_step": observed_release,
            "expected_connection_release_step": expected_release,
            "connection_release_delay_steps": release_delay_steps,
            "connection_release_delay_s": (
                release_delay_steps * self.control_dt if release_delay_steps is not None else None
            ),
            "relation_state_transition_steps": relation_transition_steps,
            "maximum_relation_confidence": float(max(self.relation_confidence, default=0.0)),
            "recovery_triggered": bool(len(recovery_onsets)),
            "recovery_trigger": recovery_trigger,
            "recovery_state_transition_steps": recovery_transition_steps,
            "recovery_start_step": recovery_start_step,
            "recovery_resume_step": recovery_resume_step,
            "time_to_recover_s": (
                (recovery_resume_step - recovery_start_step) * self.control_dt
                if recovery_start_step is not None and recovery_resume_step is not None
                else None
            ),
            "recovery_failed": recovery_failed,
            "regrasp_attempt_count": int(max(self.regrasp_attempts, default=0)),
            "false_recovery_trigger": bool(len(recovery_onsets) and not perturbation_started),
            "perturbation_event_step": perturbation_event_step,
            "post_event_relation_loss_expected": relation_loss_expected,
            "post_event_connection_loss_step": post_event_loss_step,
            "post_event_connection_loss_delay_steps": post_event_loss_delay_steps,
            "post_event_connection_loss_delay_s": (
                post_event_loss_delay_steps * self.control_dt
                if post_event_loss_delay_steps is not None
                else None
            ),
            "max_object_stream_weight": float(max(object_weights, default=0.0)),
            "forced_phase_transitions": int(forced_transitions),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """Return numeric arrays suitable for NPZ persistence."""

        arrays = {
            "ee_position": np.asarray(self.ee_positions, dtype=np.float32),
            "object_position": np.asarray(self.object_positions, dtype=np.float32),
            "target_position": np.asarray(self.target_positions, dtype=np.float32),
            "action": np.asarray(self.actions, dtype=np.float32),
            "phase": np.asarray(self.phases, dtype=np.int64),
            "inference_ms": np.asarray(self.inference_ms, dtype=np.float32),
            "connected": np.asarray(self.connected, dtype=np.bool_),
            "perturbation_active": np.asarray(self.perturbation_active, dtype=np.bool_),
            "perturbation_event": np.asarray(self.perturbation_events, dtype=np.bool_),
            "raw_action_position": np.asarray(self.raw_action_positions, dtype=np.float32),
            "policy_action_position": np.asarray(self.policy_action_positions, dtype=np.float32),
            "action_rate_limited": np.asarray(self.action_rate_limited, dtype=np.bool_),
            "relation_state": np.asarray(self.relation_states, dtype="U32"),
            "relation_confidence": np.asarray(self.relation_confidence, dtype=np.float32),
            "gripper_opening_m": np.asarray(self.gripper_opening_m, dtype=np.float32),
            "gripper_velocity_m_s": np.asarray(
                self.gripper_velocity_m_s,
                dtype=np.float32,
            ),
            "active_frames": np.asarray(["|".join(frames) for frames in self.active_frames], dtype="U256"),
            "recovery_state": np.asarray(self.recovery_states, dtype="U32"),
            "recovery_trigger": np.asarray(self.recovery_triggers, dtype="U16"),
            "regrasp_attempts": np.asarray(self.regrasp_attempts, dtype=np.int64),
        }
        if self.terminal_object_position is not None:
            arrays["terminal_object_position"] = np.asarray(self.terminal_object_position, dtype=np.float32)
        if self.terminal_target_position is not None:
            arrays["terminal_target_position"] = np.asarray(self.terminal_target_position, dtype=np.float32)
        if self.terminal_ee_position is not None:
            arrays["terminal_ee_position"] = np.asarray(self.terminal_ee_position, dtype=np.float32)
        return arrays
