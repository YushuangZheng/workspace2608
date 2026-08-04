"""Controlled physical interventions and metrics for bimanual relation studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from essay2608.data.handover_schema import HandoverState


# Same normalized command contract used by the bimanual action manager.  Keep
# this pure evaluation module importable without booting Isaac Sim.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


RELATION_CONDITIONS = (
    "normal",
    "receiver_miss",
    "receiver_delayed",
    "giver_releases_early",
    "receiver_grasps_then_loses",
    "prolonged_both_hold",
    "one_arm_paused",
)


@dataclass(frozen=True)
class InterventionDecision:
    """Action override and phase-clock decision for one control step."""

    action: torch.Tensor
    active: bool
    event: str
    hold_phase_clock: bool


class BimanualRelationIntervention:
    """Apply one declared perturbation without changing relation truth or estimator."""

    def __init__(self, condition: str, control_dt_s: float) -> None:
        if condition not in RELATION_CONDITIONS:
            raise ValueError(f"未知双臂关系条件：{condition}")
        self.condition = condition
        self.dt = float(control_dt_s)
        self.transfer_steps = 0
        self.right_grasp_steps = 0
        self.both_steps = 0
        self.pause_steps = 0
        self.loss_started = False

    def apply(
        self,
        *,
        action: torch.Tensor,
        state: HandoverState,
        truth_label: str,
        right_pose: torch.Tensor,
    ) -> InterventionDecision:
        """Override only commands needed to realize the declared counterfactual."""

        modified = action.clone()
        active = False
        event = "none"
        hold_phase_clock = False

        if self.condition == "receiver_miss":
            if state >= HandoverState.RIGHT_GRASP:
                modified[:, 15] = GRIPPER_OPEN
                active = True
                event = "receiver_forced_open"

        elif self.condition == "receiver_delayed":
            if state == HandoverState.RIGHT_GRASP and float(action[0, 15]) < 0.0:
                self.right_grasp_steps += 1
                hold_phase_clock = self.right_grasp_steps <= round(2.0 / self.dt)
                if self.right_grasp_steps <= round(1.0 / self.dt):
                    active = True
                    modified[:, 15] = GRIPPER_OPEN
                    event = "receiver_delay_open"
                elif hold_phase_clock:
                    active = True
                    modified[:, 15] = GRIPPER_CLOSE
                    event = "receiver_delayed_settle"

        elif self.condition == "giver_releases_early":
            if state >= HandoverState.RIGHT_APPROACH:
                modified[:, 7] = GRIPPER_OPEN
                active = True
                event = "giver_forced_open"

        elif self.condition == "receiver_grasps_then_loses":
            if truth_label == "both" and not self.loss_started:
                self.both_steps += 1
                if self.both_steps >= round(0.40 / self.dt):
                    self.loss_started = True
            if self.loss_started:
                modified[:, 15] = GRIPPER_OPEN
                active = True
                event = "receiver_forced_loss"

        elif self.condition == "prolonged_both_hold":
            if state == HandoverState.TRANSFER:
                self.transfer_steps += 1
                if self.transfer_steps <= round(2.0 / self.dt):
                    active = True
                    event = "both_hold_extended"
                    hold_phase_clock = True

        elif self.condition == "one_arm_paused":
            if state == HandoverState.RIGHT_TO_TARGET:
                self.pause_steps += 1
                if self.pause_steps <= round(2.0 / self.dt):
                    modified[:, 8:15] = right_pose
                    modified[:, 15] = GRIPPER_CLOSE
                    active = True
                    event = "receiver_pose_paused"
                    hold_phase_clock = True

        return InterventionDecision(
            action=modified,
            active=active,
            event=event,
            hold_phase_clock=hold_phase_clock,
        )


def _edge_metrics(truth: np.ndarray, inferred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    inferred = np.asarray(inferred, dtype=bool)
    tp = int(np.count_nonzero(truth & inferred))
    fp = int(np.count_nonzero(~truth & inferred))
    fn = int(np.count_nonzero(truth & ~inferred))
    tn = int(np.count_nonzero(~truth & ~inferred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _transition_delays(
    truth: np.ndarray,
    inferred: np.ndarray,
    control_dt_s: float,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    inferred = np.asarray(inferred, dtype=bool)
    events = []
    for index in np.flatnonzero(truth[1:] != truth[:-1]) + 1:
        target = bool(truth[index])
        candidates = np.flatnonzero(inferred[index:] == target)
        matched = int(index + candidates[0]) if len(candidates) else None
        events.append(
            {
                "truth_step": int(index),
                "transition": "establish" if target else "release_or_loss",
                "matched_step": matched,
                "delay_s": (
                    (matched - int(index)) * control_dt_s if matched is not None else None
                ),
            }
        )
    matched_delays = [event["delay_s"] for event in events if event["delay_s"] is not None]
    return {
        "events": events,
        "num_truth_transitions": len(events),
        "num_matched_transitions": len(matched_delays),
        "mean_delay_s": float(np.mean(matched_delays)) if matched_delays else None,
        "maximum_delay_s": max(matched_delays) if matched_delays else None,
    }


def score_bimanual_relation_trace(
    *,
    truth_labels: np.ndarray,
    inferred_labels: np.ndarray,
    truth_left: np.ndarray,
    truth_right: np.ndarray,
    inferred_left: np.ndarray,
    inferred_right: np.ndarray,
    control_dt_s: float,
) -> dict[str, Any]:
    """Score two independent edges and their composed four-value lifecycle."""

    truth_labels = np.asarray(truth_labels).astype("U16")
    inferred_labels = np.asarray(inferred_labels).astype("U16")
    if len(truth_labels) == 0 or len(truth_labels) != len(inferred_labels):
        raise ValueError("关系真值与推断必须为非空等长序列")
    return {
        "steps": len(truth_labels),
        "four_value_accuracy": float(np.mean(truth_labels == inferred_labels)),
        "left": {
            **_edge_metrics(truth_left, inferred_left),
            "transitions": _transition_delays(
                truth_left, inferred_left, control_dt_s
            ),
        },
        "right": {
            **_edge_metrics(truth_right, inferred_right),
            "transitions": _transition_delays(
                truth_right, inferred_right, control_dt_s
            ),
        },
        "truth_both_steps": int(np.count_nonzero(truth_labels == "both")),
        "inferred_both_steps": int(np.count_nonzero(inferred_labels == "both")),
        "privileged_contact_used_as_estimator_input": False,
    }


def condition_realization(
    condition: str,
    *,
    truth_left: np.ndarray,
    truth_right: np.ndarray,
    truth_labels: np.ndarray,
    intervention_active: np.ndarray,
    intervention_event: np.ndarray,
    control_dt_s: float,
) -> dict[str, Any]:
    """Verify that physics, rather than the intended command alone, realized a condition."""

    left = np.asarray(truth_left, dtype=bool)
    right = np.asarray(truth_right, dtype=bool)
    labels = np.asarray(truth_labels).astype("U16")
    active = np.asarray(intervention_active, dtype=bool)
    events = np.asarray(intervention_event).astype("U32")
    active_steps = np.flatnonzero(active)
    first = int(active_steps[0]) if len(active_steps) else None
    last = int(active_steps[-1]) if len(active_steps) else None
    realized = False
    checks: dict[str, bool] = {}

    if condition == "normal":
        expected = ["none", "left_only", "both", "right_only", "none"]
        sequence = [value for index, value in enumerate(labels) if index == 0 or value != labels[index - 1]]
        checks = {"exact_physical_lifecycle": sequence == expected}
    elif first is not None:
        before = slice(0, first + 1)
        after = slice(first, None)
        if condition == "receiver_miss":
            checks = {
                "left_was_connected": bool(np.any(left[before])),
                "right_never_connected_after_event": not bool(np.any(right[after])),
                "holder_eventually_lost": not bool(left[-1]) and not bool(right[-1]),
            }
        elif condition == "receiver_delayed":
            delay_open = events == "receiver_delay_open"
            forced_open_steps = np.flatnonzero(delay_open)
            first_after_open = (
                int(forced_open_steps[-1]) + 1 if len(forced_open_steps) else len(right)
            )
            checks = {
                "no_right_edge_during_forced_delay": not bool(np.any(right[delay_open])),
                "right_established_after_forced_delay": bool(
                    np.any(right[first_after_open:])
                ),
                "left_retained_during_delay": bool(np.all(left[delay_open])),
            }
        elif condition == "giver_releases_early":
            checks = {
                "left_was_connected": bool(np.any(left[before])),
                "left_lost_after_release": not bool(left[-1]),
                "no_both_after_release": not bool(np.any(labels[after] == "both")),
            }
        elif condition == "receiver_grasps_then_loses":
            checks = {
                "right_was_connected": bool(np.any(right[: first + 1])),
                "right_lost_after_event": not bool(right[-1]),
            }
        elif condition == "prolonged_both_hold":
            checks = {
                "both_throughout_extension": bool(np.all(labels[active] == "both")),
                "extension_at_least_1_8_s": len(active_steps) * control_dt_s >= 1.8,
            }
        elif condition == "one_arm_paused":
            checks = {
                "right_retained_while_paused": bool(np.all(right[active])),
                "pause_at_least_1_8_s": len(active_steps) * control_dt_s >= 1.8,
            }
    realized = bool(checks) and all(checks.values())
    return {
        "condition": condition,
        "realized": realized,
        "checks": checks,
        "intervention_steps": int(len(active_steps)),
        "intervention_duration_s": len(active_steps) * control_dt_s,
        "first_intervention_step": first,
        "last_intervention_step": last,
    }
