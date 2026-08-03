"""Regression tests for the review-driven single-arm corrections."""

from __future__ import annotations

import numpy as np

from essay2608.eval import EpisodeTrace, SuccessCriteria
from essay2608.policy.base import PhaseClockPolicy, PolicyObservation, PolicyStep


def observation(object_position=(0.555, 0.2, 0.021)) -> PolicyObservation:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    return PolicyObservation(
        ee_pose=np.concatenate((np.asarray([0.45, 0.0, 0.35]), identity)),
        object_pose=np.concatenate((np.asarray(object_position), identity)),
        target_pose=np.concatenate((np.asarray([0.55, 0.2, 0.08]), identity)),
    )


def test_semantic_success_ignores_fixed_target_height_residual() -> None:
    trace = EpisodeTrace(control_dt=0.02)
    current = observation()
    action = np.asarray([0.55, 0.2, 0.23, 0.0, 1.0, 0.0, 0.0, 1.0])
    diagnostics = {
        "phase": 9,
        "active_frames": ["target"],
        "raw_action_position": action[:3],
        "policy_action_position": action[:3],
    }
    for _ in range(30):
        trace.append(current, action, diagnostics, inference_ms=0.1, perturbation_active=False)
    metrics = trace.summary(
        final_object_position=current.object_pose[:3],
        final_target_position=current.target_pose[:3],
        criteria=SuccessCriteria(),
        policy_complete=True,
        environment_done=False,
        forced_transitions=0,
        perturbation_started=False,
    )
    assert metrics["success"]
    assert metrics["stable_place_success"]
    assert metrics["legacy_success_3d"]
    assert metrics["final_error_3d_m"] > 0.059
    assert metrics["final_xy_error_m"] < 0.01
    assert not metrics["xy_success_sensitivity"]["0.005000"]
    assert metrics["xy_success_sensitivity"]["0.010000"]


class _JumpPolicy(PhaseClockPolicy):
    def fit(self, demonstrations) -> None:
        del demonstrations

    def _compute_action(self, current: PolicyObservation) -> PolicyStep:
        del current
        action = np.asarray([0.80, 0.40, 0.60, 1.0, 0.0, 0.0, 0.0, 1.0])
        return PolicyStep(action=action, diagnostics={"phase": self.phase})


def test_phase_policy_limits_cartesian_command_jump() -> None:
    current = observation()
    policy = _JumpPolicy()
    policy.reset(current)
    step = policy.act(current)
    displacement = np.linalg.norm(step.action[:3] - current.ee_pose[:3])
    assert displacement <= policy.maximum_action_position_step + 1.0e-12
    assert step.diagnostics["action_rate_limited"]


def test_phase_path_partition_assigns_jump_to_destination_phase() -> None:
    trace = EpisodeTrace(control_dt=0.02)
    action = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    for position, phase in ((0.0, 0), (0.1, 1), (0.3, 1), (0.6, 2)):
        current = observation()
        current = PolicyObservation(
            ee_pose=np.concatenate(([position, 0.0, 0.35], current.ee_pose[3:7])),
            object_pose=current.object_pose,
            target_pose=current.target_pose,
        )
        trace.append(
            current,
            action,
            {"phase": phase},
            inference_ms=0.1,
            perturbation_active=False,
        )
    metrics = trace.summary(
        final_object_position=current.object_pose[:3],
        final_target_position=current.target_pose[:3],
        criteria=SuccessCriteria(),
        policy_complete=True,
        environment_done=False,
        forced_transitions=0,
        perturbation_started=False,
    )
    assert np.isclose(metrics["phase_path_length_m"]["1"], 0.3)
    assert np.isclose(metrics["phase_path_length_m"]["2"], 0.3)
    assert np.isclose(sum(metrics["phase_path_length_m"].values()), metrics["path_length_m"])
