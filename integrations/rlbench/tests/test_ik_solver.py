from __future__ import annotations

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import direct_evaluate, unimanual_evaluate
from integrations.rlbench.rlbench_dynamac.runtime import (
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
    IK_SAMPLING_MAX_CONFIGS,
    IK_SAMPLING_MAX_TIME_MS,
    IK_SAMPLING_TRIALS,
    initialize_ik_solver_diagnostics,
    solve_absolute_ee_ik_with_sampling_fallback,
)


class _IKError(Exception):
    pass


class _ConfigurationError(Exception):
    pass


class _InvalidActionError(Exception):
    pass


class _Arm:
    def __init__(
        self,
        *,
        current,
        jacobian,
        sampling=None,
        cyclics=None,
        intervals=None,
    ) -> None:
        self.current = np.asarray(current, dtype=np.float64)
        self.jacobian = jacobian
        self.sampling = sampling
        joint_count = self.current.size
        self.cyclics = list(cyclics or [False] * joint_count)
        self.intervals = list(intervals or [[-1.0, 2.0]] * joint_count)
        self.jacobian_calls = []
        self.sampling_calls = []

    def get_joint_positions(self):
        return self.current.copy()

    def get_joint_intervals(self):
        return self.cyclics, self.intervals

    def solve_ik_via_jacobian(self, position, **kwargs):
        self.jacobian_calls.append((np.asarray(position), kwargs))
        if isinstance(self.jacobian, BaseException):
            raise self.jacobian
        return self.jacobian

    def solve_ik_via_sampling(self, position, **kwargs):
        self.sampling_calls.append((np.asarray(position), kwargs))
        if isinstance(self.sampling, BaseException):
            raise self.sampling
        return self.sampling


_TARGET = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])


def _solve(arm: _Arm, diagnostics=None):
    diagnostics = diagnostics or initialize_ik_solver_diagnostics()
    result = solve_absolute_ee_ik_with_sampling_fallback(
        arm,
        _TARGET,
        diagnostics=diagnostics,
        ik_error=_IKError,
        configuration_error=_ConfigurationError,
        invalid_action_error=_InvalidActionError,
        error_message="test IK failed",
    )
    return result, diagnostics


def test_valid_jacobian_solution_is_checked_and_selected_without_sampling() -> None:
    arm = _Arm(
        current=[0.0, 0.0],
        jacobian=[0.25, -0.5],
        sampling=AssertionError("sampling should not run"),
    )

    result, diagnostics = _solve(arm)

    np.testing.assert_allclose(result, [0.25, -0.5])
    assert not arm.sampling_calls
    assert diagnostics["selected_via_jacobian"] == 1
    assert diagnostics["selected_via_sampling"] == 0
    assert diagnostics["selected_joint_delta_l2_max"] == pytest.approx(
        np.linalg.norm([0.25, -0.5])
    )


def test_sampling_checks_collisions_and_selects_nearest_valid_current_q() -> None:
    arm = _Arm(
        current=[0.0, 0.0],
        jacobian=_IKError("local solve failed"),
        sampling=np.asarray(
            [
                [0.9, 0.9],
                [0.2, -0.1],
                [np.nan, 0.0],
                [1.2, 0.0],
            ]
        ),
    )

    result, diagnostics = _solve(arm)

    np.testing.assert_allclose(result, [0.2, -0.1])
    assert len(arm.sampling_calls) == 1
    position, kwargs = arm.sampling_calls[0]
    np.testing.assert_allclose(position, _TARGET[:3])
    np.testing.assert_allclose(kwargs["quaternion"], _TARGET[3:])
    assert {key: value for key, value in kwargs.items() if key != "quaternion"} == {
        "ignore_collisions": False,
        "trials": IK_SAMPLING_TRIALS,
        "max_configs": IK_SAMPLING_MAX_CONFIGS,
        "max_time_ms": IK_SAMPLING_MAX_TIME_MS,
        "relative_to": None,
    }
    assert diagnostics["jacobian_failures"] == 1
    assert diagnostics["sampling_candidates_evaluated"] == 4
    assert diagnostics["candidate_rejections_nonfinite"] == 1
    assert diagnostics["candidate_rejections_joint_limits"] == 1
    assert diagnostics["sampling_fallback_successes"] == 1
    assert diagnostics["selected_via_sampling"] == 1


def test_invalid_jacobian_candidate_falls_back_after_joint_limit_check() -> None:
    arm = _Arm(
        current=[0.0],
        jacobian=[1.5],
        sampling=np.asarray([[0.3]]),
    )

    result, diagnostics = _solve(arm)

    np.testing.assert_allclose(result, [0.3])
    assert diagnostics["jacobian_failures"] == 0
    assert diagnostics["jacobian_candidate_rejections"] == 1
    assert diagnostics["candidate_rejections_joint_limits"] == 1
    assert diagnostics["sampling_fallback_successes"] == 1


def test_noncyclic_interval_uses_minimum_plus_range_and_cyclic_is_unbounded() -> None:
    arm = _Arm(
        current=[1.1, 3.0],
        jacobian=[1.4, -3.0],
        sampling=AssertionError("sampling should not run"),
        cyclics=[False, True],
        intervals=[[1.0, 0.5], [0.0, 0.0]],
    )

    result, diagnostics = _solve(arm)

    np.testing.assert_allclose(result, [1.4, -3.0])
    assert diagnostics["selected_via_jacobian"] == 1


def test_all_nonfinite_out_of_limit_or_malformed_samples_are_invalid_action() -> None:
    diagnostics = initialize_ik_solver_diagnostics()
    arm = _Arm(
        current=[0.0],
        jacobian=[np.nan],
        sampling=[[np.inf], [3.0], [0.0, 0.0]],
    )

    with pytest.raises(_InvalidActionError, match="test IK failed"):
        _solve(arm, diagnostics)

    assert diagnostics["jacobian_candidate_rejections"] == 1
    assert diagnostics["candidate_rejections_nonfinite"] == 2
    assert diagnostics["candidate_rejections_joint_limits"] == 1
    assert diagnostics["candidate_rejections_shape"] == 1
    assert diagnostics["sampling_fallback_failures"] == 1


def test_sampling_configuration_failure_remains_an_invalid_action() -> None:
    diagnostics = initialize_ik_solver_diagnostics()
    arm = _Arm(
        current=[0.0],
        jacobian=_IKError("local solve failed"),
        sampling=_ConfigurationError("no collision-free configuration"),
    )

    with pytest.raises(_InvalidActionError, match="test IK failed") as raised:
        _solve(arm, diagnostics)

    assert isinstance(raised.value.__cause__, _ConfigurationError)
    assert diagnostics["jacobian_failures"] == 1
    assert diagnostics["sampling_fallback_failures"] == 1


@pytest.mark.parametrize(
    "module",
    (direct_evaluate, unimanual_evaluate),
)
def test_v4_protocol_is_distinct_while_legacy_v3_id_remains_available(module) -> None:
    current = module.evaluation_protocol_id(DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS)
    legacy = module.legacy_v3_evaluation_protocol_id(
        DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS
    )

    assert (
        "jacobian-then-collision-aware-sampling5-nearest-current-q-"
        "finite-joint-limits"
    ) in current
    assert "contact-delta-diagnostic-v4" in current
    assert "contact-delta-diagnostic-v3" in legacy
    assert "collision-aware" not in legacy
    assert module.LEGACY_V3_EVALUATION_PROTOCOL_ID == legacy
    assert module.EVALUATION_PROTOCOL_ID == current
