from __future__ import annotations

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.eval import direct_evaluate, unimanual_evaluate
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    GLOBAL_IK_CONTROLLER_PROFILE,
    GlobalIKControllerConfig,
    execute_global_ik_ee_control,
    initialize_global_ik_controller_diagnostics,
)


class _IKError(Exception):
    pass


class _ConfigurationError(Exception):
    pass


class _ConfigurationPathError(Exception):
    pass


class _InvalidActionError(Exception):
    pass


class _Tip:
    def get_pose(self):
        return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class _Path:
    def __init__(self, arm):
        self.arm = arm
        self.target = np.asarray([0.1, 0.1])

    def step(self):
        self.arm.set_joint_target_positions(self.target)
        return True

    def get_executed_joint_position_action(self):
        return self.target.copy()


class _Arm:
    def __init__(
        self, *, jacobian_result=None, sampling_result=None, path_fails=False
    ):
        self.current = np.zeros(2, dtype=np.float64)
        self.target = self.current.copy()
        self.jacobian_result = jacobian_result
        self.sampling_result = sampling_result
        self.path_fails = bool(path_fails)
        self.property_calls = []
        self.path_calls = []
        self.target_writes = 0
        self.sampling_calls = []
        self.solver_events = []

    def get_tip(self):
        return _Tip()

    def get_joint_positions(self):
        return self.current.copy()

    def get_joint_intervals(self):
        return [False, False], [[-1.0, 2.0], [-1.0, 2.0]]

    def set_ik_group_properties(self, **kwargs):
        self.property_calls.append(dict(kwargs))

    def solve_ik_via_jacobian(self, position, **kwargs):
        del position, kwargs
        self.solver_events.append("pseudo_inverse")
        if isinstance(self.jacobian_result, Exception):
            raise self.jacobian_result
        return np.asarray(self.jacobian_result, dtype=np.float64)

    def solve_ik_via_sampling(self, *args, **kwargs):
        self.solver_events.append("sampling")
        self.sampling_calls.append((args, dict(kwargs)))
        if self.sampling_result is None:
            raise _ConfigurationError("no sampling solution")
        if isinstance(self.sampling_result, Exception):
            raise self.sampling_result
        return np.asarray(self.sampling_result, dtype=np.float64)

    def get_path(self, position, **kwargs):
        self.path_calls.append((np.asarray(position).copy(), dict(kwargs)))
        if self.path_fails:
            raise _ConfigurationPathError("no path")
        return _Path(self)

    def set_joint_target_positions(self, target):
        self.target_writes += 1
        self.target = np.asarray(target, dtype=np.float64).copy()


class _Scene:
    def __init__(self, *arms):
        self.arms = arms
        self.steps = 0

    def step(self):
        self.steps += 1
        for arm in self.arms:
            arm.current = arm.target.copy()


class _Solver:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.chain_source = "fake_exact_chain"

    def solve(self, target_pose):
        self.calls.append(np.asarray(target_pose, dtype=np.float64).copy())
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is None:
            return None
        return _Result(self.result)


class _Result:
    def __init__(self, joints):
        self.joints = np.asarray(joints, dtype=np.float64)
        self.elapsed_ms = 0.25
        self.fk_translation_error_m = 0.0005
        self.fk_rotation_error_rad = 0.005
        self.bounded_cartesian_api_used = True


class _Factory:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.solvers = []

    def __call__(self, arm):
        arm.solver_events.append("trac_ik_distance")
        self.calls.append(arm)
        solver = _Solver(self.result)
        self.solvers.append(solver)
        return solver


def _target(x):
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _execute(scene, arm_targets, factory, *, config=None):
    diagnostics = initialize_global_ik_controller_diagnostics()
    status = execute_global_ik_ee_control(
        scene,
        tuple(arm_targets),
        config=config or GlobalIKControllerConfig(),
        diagnostics=diagnostics,
        external_solver_factory=factory,
        ik_error=_IKError,
        configuration_error=_ConfigurationError,
        configuration_path_error=_ConfigurationPathError,
        invalid_action_error=_InvalidActionError,
        path_algorithm="RRTConnect",
        error_message="dev IK failed",
    )
    return status, diagnostics


def test_profile_is_formal_global_and_declares_solver_order():
    config = GlobalIKControllerConfig()
    metadata = config.metadata()
    assert metadata["profile"] == GLOBAL_IK_CONTROLLER_PROFILE
    assert metadata["protocol_id"].endswith("-formal-v1")
    assert metadata["formal_default"] is True
    assert metadata["sampling_fallback"] is True
    assert metadata["legacy_frozen_ik_helper_used"] is False
    assert metadata["ik_order"] == (
        "current_seeded_pseudo_inverse_then_bounded_trac_ik_distance_"
        "then_frozen_v4_collision_aware_sampling_then_far_path"
    )
    assert metadata["primary_resolution_method"] == "pseudo_inverse"
    assert metadata["primary_max_iterations"] == 6
    assert metadata["primary_damping"] == pytest.approx(0.1)
    assert metadata["sampling_entry_condition"] == (
        "pseudo_inverse_and_trac_ik_exhausted"
    )
    assert metadata["sampling_trials"] == 100
    assert metadata["sampling_max_configs"] == 5
    assert metadata["sampling_max_time_ms"] == 10
    assert metadata["sampling_ignore_collisions"] is False
    assert metadata["sampling_hard_joint_delta_rejection"] is False

    arm = _Arm(jacobian_result=[0.1, 0.1])
    factory = _Factory([0.2, 0.2])
    status, diagnostics = _execute(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert factory.calls == []
    assert arm.sampling_calls == []
    assert arm.solver_events == ["pseudo_inverse"]
    assert arm.property_calls[0] == {
        "resolution_method": "pseudo_inverse",
        "max_iterations": 6,
        "dls_damping": 0.1,
    }
    assert diagnostics["pseudo_inverse_ik_attempts"] == 1
    assert diagnostics["pseudo_inverse_ik_successes"] == 1
    assert diagnostics["selected_via_pseudo_inverse"] == 1
    assert diagnostics["selected_via_jacobian"] == 1
    assert diagnostics["selected_joint_delta_l2_max"] == pytest.approx(2**0.5 / 10)


def test_evaluators_reject_removed_development_profile_before_importing_pyrep():
    for module in (direct_evaluate, unimanual_evaluate):
        with pytest.raises(ValueError, match="unsupported .* controller profile"):
            module._make_action_mode("trac_ik_distance_dev")


def test_trac_failure_invokes_exact_frozen_sampling_fallback():
    arm = _Arm(
        jacobian_result=_IKError("no local solution"),
        sampling_result=[[0.3, 0.2], [0.1, -0.1]],
    )
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert factory.calls == [arm]
    assert arm.solver_events == [
        "pseudo_inverse",
        "trac_ik_distance",
        "sampling",
    ]
    assert len(arm.sampling_calls) == 1
    _args, kwargs = arm.sampling_calls[0]
    assert kwargs == {
        "quaternion": pytest.approx([0.0, 0.0, 0.0, 1.0]),
        "ignore_collisions": False,
        "trials": 100,
        "max_configs": 5,
        "max_time_ms": 10,
        "relative_to": None,
    }
    assert diagnostics["sampling_candidates_evaluated"] == 2
    assert diagnostics["trac_ik_distance_exhaustions"] == 1
    assert diagnostics["sampling_after_trac_attempts"] == 1
    assert diagnostics["sampling_after_trac_successes"] == 1
    assert diagnostics["sampling_after_trac_failures"] == 0
    assert diagnostics["sampling_fallback_successes"] == 1
    assert diagnostics["selected_via_sampling"] == 1
    assert diagnostics["all_ik_exhaustions"] == 0
    np.testing.assert_allclose(arm.current, [0.1, -0.1])


def test_pseudo_failure_uses_bounded_trac_and_skips_sampling_on_success():
    arm = _Arm(jacobian_result=_IKError("no local solution"))
    factory = _Factory([0.2, -0.1])
    status, diagnostics = _execute(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert factory.calls == [arm]
    assert arm.solver_events == ["pseudo_inverse", "trac_ik_distance"]
    call = factory.solvers[0].calls[0]
    np.testing.assert_allclose(call, _target(0.05))
    assert diagnostics["selected_via_trac_ik_distance"] == 1
    assert diagnostics["trac_ik_distance_solve_time_ms_total"] >= 0.0
    assert diagnostics["trac_ik_distance_chain_sources"] == ["fake_exact_chain"]
    assert diagnostics["trac_ik_distance_bounded_cartesian_api_uses"] == 1
    assert diagnostics["trac_ik_distance_unbounded_cartesian_api_uses"] == 0
    assert diagnostics["trac_ik_distance_fk_translation_error_m_max"] == pytest.approx(
        0.0005
    )
    assert diagnostics["trac_ik_distance_fk_rotation_error_rad_max"] == pytest.approx(
        0.005
    )
    assert diagnostics["pseudo_inverse_ik_failures"] == 1
    assert diagnostics["trac_ik_distance_exhaustions"] == 0
    assert diagnostics["sampling_after_trac_attempts"] == 0
    assert diagnostics["all_ik_exhaustions"] == 0
    assert arm.sampling_calls == []


def test_pseudo_candidate_is_not_subject_to_trac_continuity_gate():
    arm = _Arm(jacobian_result=[0.8, 0.0])
    factory = _Factory([0.2, -0.1])

    status, diagnostics = _execute(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert diagnostics["selected_via_jacobian"] == 1
    assert diagnostics["selected_joint_delta_abs_max"] == pytest.approx(0.8)
    assert arm.sampling_calls == []
    assert factory.calls == []


def test_sampling_large_joint_delta_remains_a_valid_last_ik_fallback():
    arm = _Arm(
        jacobian_result=_IKError("no local solution"),
        sampling_result=[[0.8, 0.0]],
    )
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert arm.solver_events == [
        "pseudo_inverse",
        "trac_ik_distance",
        "sampling",
    ]
    assert diagnostics["selected_via_sampling"] == 1
    assert diagnostics["sampling_after_trac_successes"] == 1
    assert diagnostics["candidate_rejections_joint_delta_abs"] == 0
    assert diagnostics["candidate_rejections_joint_delta_l2"] == 0
    assert diagnostics["selected_joint_delta_abs_max"] == pytest.approx(0.8)
    np.testing.assert_allclose(arm.current, [0.8, 0.0])


def test_external_result_without_bounded_metadata_fails_closed():
    class _UnboundedSolver:
        chain_source = "fake_exact_chain"

        def solve(self, target_pose):
            del target_pose
            return np.asarray([0.1, 0.0])

    arm = _Arm(jacobian_result=_IKError("no local solution"))
    diagnostics = initialize_global_ik_controller_diagnostics()
    with pytest.raises(_InvalidActionError):
        execute_global_ik_ee_control(
            _Scene(arm),
            ((arm, _target(0.05)),),
            config=GlobalIKControllerConfig(),
            diagnostics=diagnostics,
            external_solver_factory=lambda _arm: _UnboundedSolver(),
            ik_error=_IKError,
            configuration_error=_ConfigurationError,
            configuration_path_error=_ConfigurationPathError,
            invalid_action_error=_InvalidActionError,
            path_algorithm="RRTConnect",
            error_message="dev IK failed",
        )
    assert diagnostics["trac_ik_distance_result_metadata_missing"] == 1
    assert diagnostics["trac_ik_distance_bounded_cartesian_api_uses"] == 0
    assert diagnostics["trac_ik_distance_exhaustions"] == 1
    assert diagnostics["sampling_after_trac_attempts"] == 1
    assert diagnostics["sampling_after_trac_failures"] == 1
    assert diagnostics["all_ik_exhaustions"] == 1
    assert len(arm.sampling_calls) == 1


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ([np.nan, 0.0], "nonfinite"),
        ([1.1, 0.0], "joint_limits"),
        ([0.36, 0.0], "joint_delta_abs"),
        ([0.3, 0.3], "joint_delta_l2"),
    ],
)
def test_external_candidates_are_finite_limited_and_continuous(candidate, reason):
    arm = _Arm(jacobian_result=_IKError("no local solution"))
    factory = _Factory(candidate)

    diagnostics = initialize_global_ik_controller_diagnostics()
    with pytest.raises(_InvalidActionError):
        execute_global_ik_ee_control(
            _Scene(arm),
            ((arm, _target(0.05)),),
            config=GlobalIKControllerConfig(max_joint_delta_l2_rad=0.4),
            diagnostics=diagnostics,
            external_solver_factory=factory,
            ik_error=_IKError,
            configuration_error=_ConfigurationError,
            configuration_path_error=_ConfigurationPathError,
            invalid_action_error=_InvalidActionError,
            path_algorithm="RRTConnect",
            error_message="dev IK failed",
        )
    assert diagnostics[f"candidate_rejections_{reason}"] == 1
    assert diagnostics["trac_ik_distance_exhaustions"] == 1
    assert diagnostics["sampling_after_trac_attempts"] == 1
    assert diagnostics["sampling_after_trac_failures"] == 1
    assert diagnostics["all_ik_exhaustions"] == 1
    assert len(arm.sampling_calls) == 1
    assert arm.path_calls == []


def test_path_is_only_after_pseudo_trac_and_sampling_fail_for_far_target():
    arm = _Arm(jacobian_result=_IKError("no local solution"))
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute(
        _Scene(arm), ((arm, _target(0.100001)),), factory
    )

    assert status == "reached"
    assert len(arm.path_calls) == 1
    _position, kwargs = arm.path_calls[0]
    assert kwargs["ignore_collisions"] is False
    assert kwargs["algorithm"] == "RRTConnect"
    assert arm.solver_events[:3] == [
        "pseudo_inverse",
        "trac_ik_distance",
        "sampling",
    ]
    assert diagnostics["all_ik_exhaustions"] == 1
    assert diagnostics["path_after_all_ik_exhaustion"] == 1
    assert len(arm.sampling_calls) == 1


def test_exact_10cm_does_not_trigger_path():
    arm = _Arm(jacobian_result=_IKError("no local solution"))

    with pytest.raises(_InvalidActionError):
        _execute(
            _Scene(arm),
            ((arm, _target(0.10)),),
            _Factory(RuntimeError("no external solution")),
        )

    assert arm.path_calls == []


def test_bimanual_preparation_failure_writes_neither_arm_target():
    right = _Arm(jacobian_result=[0.1, 0.0])
    left = _Arm(jacobian_result=_IKError("no local solution"))
    scene = _Scene(right, left)

    with pytest.raises(_InvalidActionError):
        _execute(
            scene,
            ((right, _target(0.05)), (left, _target(0.05))),
            _Factory(RuntimeError("no external solution")),
        )

    assert scene.steps == 0
    assert right.solver_events == ["pseudo_inverse"]
    assert left.solver_events == [
        "pseudo_inverse",
        "trac_ik_distance",
        "sampling",
    ]
    assert right.target_writes == 0
    assert left.target_writes == 0
