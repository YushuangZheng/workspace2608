from __future__ import annotations

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.eval import (
    direct_evaluate,
    unimanual_evaluate,
)
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    GLOBAL_IK_CONTROLLER_PROFILE,
    STAGE6_IK_CONTROLLER_PROFILE,
    GlobalIKControllerConfig,
    Stage6IKControllerConfig,
    _prepare_collision_aware_path_command,
    execute_global_ik_ee_control,
    execute_stage6_ik_ee_control,
    global_ik_controller_metadata,
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


class _CartesianTip:
    def __init__(self, arm):
        self.arm = arm

    def get_pose(self):
        return np.asarray([self.arm.current[0], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class _Path:
    def __init__(self, arm, target=None):
        self.arm = arm
        self.target = np.asarray(
            [0.1, 0.1] if target is None else target,
            dtype=np.float64,
        )

    def step(self):
        self.arm.set_joint_target_positions(self.target)
        return True

    def get_executed_joint_position_action(self):
        return self.target.copy()


class _Arm:
    def __init__(self, *, jacobian_result=None, sampling_result=None, path_fails=False):
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
        self.velocity = np.zeros(2, dtype=np.float64)
        self.target_history = []

    def get_tip(self):
        return _Tip()

    def get_joint_positions(self):
        return self.current.copy()

    def get_joint_target_positions(self):
        return self.target.copy()

    def get_joint_velocities(self):
        return self.velocity.copy()

    def get_joint_upper_velocity_limits(self):
        return np.ones_like(self.current)

    def get_joint_intervals(self):
        return [False, False], [[-1.0, 2.0], [-1.0, 2.0]]

    def set_ik_group_properties(self, **kwargs):
        self.property_calls.append(dict(kwargs))

    def solve_ik_via_jacobian(self, position, **kwargs):
        del kwargs
        self.solver_events.append("pseudo_inverse")
        if isinstance(self.jacobian_result, Exception):
            raise self.jacobian_result
        if callable(self.jacobian_result):
            return np.asarray(self.jacobian_result(position), dtype=np.float64)
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
        self.target_history.append(self.target.copy())

    def set_joint_positions(self, positions, disable_dynamics=False):
        del disable_dynamics
        self.current = np.asarray(positions, dtype=np.float64).copy()

    def check_arm_collision(self):
        return False


class _CartesianArm(_Arm):
    def get_tip(self):
        return _CartesianTip(self)


class _LinearPathArm(_CartesianArm):
    def __init__(self, *, collision_aware_fails=False, relaxed_fails=False):
        super().__init__(jacobian_result=_IKError("no local solution"))
        self.collision_aware_fails = bool(collision_aware_fails)
        self.relaxed_fails = bool(relaxed_fails)
        self.linear_path_calls = []

    def get_linear_path(self, position, **kwargs):
        self.linear_path_calls.append((np.asarray(position).copy(), dict(kwargs)))
        ignore_collisions = bool(kwargs["ignore_collisions"])
        if (not ignore_collisions and self.collision_aware_fails) or (
            ignore_collisions and self.relaxed_fails
        ):
            raise _ConfigurationPathError("no linear path")
        return _Path(self, [float(position[0]), 0.0])


class _CollisionSelectiveSamplingArm(_CartesianArm):
    def __init__(self):
        super().__init__(jacobian_result=_IKError("no local solution"))

    def solve_ik_via_sampling(self, position, **kwargs):
        self.solver_events.append("sampling")
        self.sampling_calls.append(((position,), dict(kwargs)))
        if not kwargs["ignore_collisions"]:
            raise _ConfigurationError("contact pose rejected by collision filter")
        return np.asarray([[float(position[0]), 0.0]], dtype=np.float64)


class _TaggedPath(_Path):
    def __init__(self, arm, target, kind):
        super().__init__(arm, target)
        self.kind = str(kind)

    def step(self):
        self.arm.last_path_kind = self.kind
        return super().step()


class _PhysicallyStalledPathArm(_LinearPathArm):
    """Expose valid paths while only the final relaxed family can move."""

    def __init__(self):
        super().__init__()
        self.last_path_kind = None
        self.path_family_history = []

    def get_linear_path(self, position, **kwargs):
        self.linear_path_calls.append((np.asarray(position).copy(), dict(kwargs)))
        kind = "relaxed_linear" if kwargs["ignore_collisions"] else "aware_linear"
        self.path_family_history.append(kind)
        return _TaggedPath(self, [float(position[0]), 0.0], kind)

    def get_path(self, position, **kwargs):
        self.path_calls.append((np.asarray(position).copy(), dict(kwargs)))
        self.path_family_history.append("aware_nonlinear")
        return _TaggedPath(
            self,
            [float(position[0]), 0.0],
            "aware_nonlinear",
        )


class _Scene:
    def __init__(self, *arms):
        self.arms = arms
        self.steps = 0

    def step(self):
        self.steps += 1
        for arm in self.arms:
            arm.velocity = (arm.target - arm.current) / 0.05
            arm.current = arm.target.copy()


class _PartiallyStalledScene(_Scene):
    def __init__(self, moving_arm, stalled_arm):
        super().__init__(moving_arm, stalled_arm)
        self.moving_arm = moving_arm

    def step(self):
        self.steps += 1
        self.moving_arm.velocity = (
            self.moving_arm.target - self.moving_arm.current
        ) / 0.05
        self.moving_arm.current = self.moving_arm.target.copy()
        for arm in self.arms:
            if arm is not self.moving_arm:
                arm.velocity = np.zeros_like(arm.current)


class _SlowScene(_Scene):
    def __init__(self, arm, step=0.005):
        super().__init__(arm)
        self.arm = arm
        self.step_size = float(step)

    def step(self):
        self.steps += 1
        delta = self.arm.target - self.arm.current
        change = np.clip(delta, -self.step_size, self.step_size)
        self.arm.velocity = change / 0.05
        self.arm.current += change


class _RelaxedPathOnlyScene(_Scene):
    def step(self):
        self.steps += 1
        for arm in self.arms:
            if arm.last_path_kind == "relaxed_linear":
                arm.velocity = (arm.target - arm.current) / 0.05
                arm.current = arm.target.copy()
            else:
                arm.velocity = np.zeros_like(arm.current)


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
        value = self.result(target_pose) if callable(self.result) else self.result
        return _Result(value)


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


class _LocalStepSolver:
    chain_source = "fake_exact_chain"
    chain_schema = "fake-exact-chain-v1"

    def __init__(self, arm, maximum_cartesian_step):
        self.arm = arm
        self.maximum_cartesian_step = float(maximum_cartesian_step)

    def solve(self, target_pose):
        target = np.asarray(target_pose, dtype=np.float64)
        if abs(float(target[0]) - float(self.arm.current[0])) > (
            self.maximum_cartesian_step + 1.0e-9
        ):
            return None
        return _Result([target[0], 0.0])


class _LocalStepFactory:
    def __init__(self, maximum_cartesian_step):
        self.maximum_cartesian_step = float(maximum_cartesian_step)
        self.calls = []

    def __call__(self, arm):
        arm.solver_events.append("trac_ik_distance")
        self.calls.append(arm)
        return _LocalStepSolver(arm, self.maximum_cartesian_step)


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


def _execute_stage6(
    scene,
    arm_targets,
    factory,
    *,
    config=None,
    per_arm_status_out=None,
):
    diagnostics = initialize_global_ik_controller_diagnostics()
    status = execute_stage6_ik_ee_control(
        scene,
        tuple(arm_targets),
        config=config or Stage6IKControllerConfig(),
        diagnostics=diagnostics,
        external_solver_factory=factory,
        ik_error=_IKError,
        configuration_error=_ConfigurationError,
        configuration_path_error=_ConfigurationPathError,
        invalid_action_error=_InvalidActionError,
        path_algorithm="RRTConnect",
        error_message="stage6 IK failed",
        per_arm_status_out=per_arm_status_out,
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


def test_stage6_profile_is_distinct_and_uses_collision_aware_first_fallbacks():
    config = Stage6IKControllerConfig()
    metadata = config.metadata()
    controller = global_ik_controller_metadata(config)

    assert metadata["profile"] == STAGE6_IK_CONTROLLER_PROFILE
    assert metadata["post_execution_cartesian_verification"] is True
    assert metadata["fallback_collision_policy"] == (
        "collision_aware_first_then_bounded_relaxation"
    )
    assert metadata["task_specific_controller_branches"] is False
    assert metadata["ik_order"].startswith(
        "current_seeded_continuous_pseudo_inverse_then_"
    )
    assert "then_bounded_cartesian_continuation_" in metadata["ik_order"]
    assert metadata["primary_resolution_method"] == (
        "current_seeded_coppeliasim_pseudo_inverse"
    )
    assert metadata["pseudo_inverse_continuity_gate"] is True
    assert metadata["trac_ik_translation_tolerance_m"] == pytest.approx(0.0005)
    assert metadata["trac_ik_rotation_tolerance_rad"] == pytest.approx(np.deg2rad(0.05))
    assert metadata["trac_ik_fk_translation_max_m"] == pytest.approx(0.002)
    assert metadata["trac_ik_fk_rotation_max_rad"] == pytest.approx(np.deg2rad(1.0))
    assert controller["sampling_ik"]["ignore_collisions"] is False
    assert controller["far_path"]["ignore_collisions"] is False
    assert controller["sampling_collision_policy"] == (
        "collision_aware_then_collision_relaxed"
    )
    assert controller["path_fallback"]["order"] == (
        "collision_aware_linear_then_collision_aware_rrt_connect_"
        "after_measured_stall_or_for_far_targets_then_"
        "collision_relaxed_linear"
    )
    assert controller["path_fallback"]["near_nonlinear_entry_condition"] == (
        "same_target_measured_stall_exhausted_local_solver_tiers"
    )
    assert controller["post_execution_cartesian_verification"] is True
    assert controller["cartesian_translation_tolerance_m"] == pytest.approx(0.0005)
    assert controller["cartesian_rotation_tolerance_rad"] == pytest.approx(
        np.deg2rad(0.05)
    )
    assert controller["control_acceptance_translation_tolerance_m"] == (
        pytest.approx(0.001)
    )
    assert controller["control_acceptance_rotation_tolerance_rad"] == (
        pytest.approx(np.deg2rad(0.1))
    )
    assert controller["same_target_alternate_solver_after_primary_stall"] is True
    assert controller["same_target_solver_tier_persistence"] == (
        "per_arm_exact_target_across_closed_loop_cycles"
    )
    assert controller["physical_stall_resolution"] == (
        "same_target_bounded_solver_escalation_then_report_stall"
    )
    assert controller["unreported_hidden_motion_after_physical_stall"] is False
    assert controller["joint_target_execution"] == (
        "shared_clock_joint_target_until_reached_or_stopped"
    )
    assert controller["same_target_ik_solution_cache"] is True
    assert controller["same_target_ik_solution_cache_semantics"] == (
        "reuse_only_after_measured_cartesian_progress_and_"
        "invalidate_before_bounded_stall_escalation"
    )
    assert metadata["cartesian_continuation"] == {
        "entry_condition": "full_target_local_ik_exhausted",
        "translation_step_m": pytest.approx(0.005),
        "rotation_step_rad": pytest.approx(np.deg2rad(2.0)),
        "backoff_factors": [1.0, 0.5, 0.25, 0.125],
        "max_segments_per_policy_action": 8,
        "max_raw_physics_steps_per_policy_action": 8,
        "interpolation": "linear_translation_shortest_xyzw_slerp",
        "progress_feedback": "physical_pose_reobserved_between_segments",
        "commit_semantics": (
            "reach_or_report_bounded_progress_for_closed_loop_reobservation"
        ),
        "task_specific": False,
    }
    assert metadata["sampling_collision_policy"] == (
        "collision_aware_then_collision_relaxed"
    )


def test_stage6_accepts_an_already_reached_target_before_reissuing_ik():
    arm = _CartesianArm(jacobian_result=_IKError("must not solve"))
    arm.current = np.asarray([0.0008, 0.0])
    arm.target = arm.current.copy()
    scene = _Scene(arm)
    factory = _Factory(RuntimeError("must not solve"))

    status, diagnostics = _execute_stage6(
        scene,
        ((arm, _target(0.0)),),
        factory,
    )

    assert status == "reached"
    assert scene.steps == 0
    assert arm.solver_events == []
    assert factory.calls == []
    assert diagnostics["cartesian_pre_execution_control_accepts"] == 1


def test_stage6_continues_toward_a_target_outside_local_ik_basin():
    arm = _CartesianArm(jacobian_result=_IKError("no full-target solution"))
    factory = _LocalStepFactory(maximum_cartesian_step=0.005)
    scene = _Scene(arm)

    first_status, first_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.05)),), factory
    )
    second_status, second_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.05)),), factory
    )

    assert first_status == "progressed"
    assert second_status == "reached"
    assert arm.current[0] == pytest.approx(0.05)
    assert first_diagnostics["cartesian_continuation_attempts"] == 8
    assert first_diagnostics["cartesian_continuation_successes"] == 8
    assert first_diagnostics["cartesian_policy_action_physics_budget_exhaustions"] == 1
    assert second_diagnostics["cartesian_continuation_attempts"] == 1
    assert second_diagnostics["cartesian_multi_pass_goals_completed"] == 1
    assert second_diagnostics["cartesian_continuation_fraction_min"] == pytest.approx(
        0.5
    )
    assert arm.sampling_calls == []


def test_stage6_continuation_backs_off_before_global_sampling():
    arm = _CartesianArm(jacobian_result=_IKError("no full-target solution"))
    factory = _LocalStepFactory(maximum_cartesian_step=0.00125)

    scene = _Scene(arm)
    runs = []
    for _ in range(5):
        runs.append(_execute_stage6(scene, ((arm, _target(0.05)),), factory))

    assert [status for status, _ in runs] == [
        "progressed",
        "progressed",
        "progressed",
        "progressed",
        "reached",
    ]
    assert arm.current[0] == pytest.approx(0.05)
    assert (
        sum(diagnostics["cartesian_continuation_attempts"] for _, diagnostics in runs)
        == 113
    )
    assert (
        sum(diagnostics["cartesian_continuation_successes"] for _, diagnostics in runs)
        == 39
    )
    assert all(
        diagnostics["cartesian_continuation_failures"] == 0 for _, diagnostics in runs
    )
    assert min(
        diagnostics["cartesian_continuation_fraction_min"] for _, diagnostics in runs
    ) == pytest.approx(0.025)
    assert arm.sampling_calls == []


def test_stage6_reobserves_between_bounded_cartesian_continuation_steps():
    arm = _CartesianArm(jacobian_result=_IKError("no full-target solution"))
    scene = _Scene(arm)
    factory = _LocalStepFactory(maximum_cartesian_step=0.005)

    first_status, _first_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.05)),), factory
    )
    second_status, second_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.05)),), factory
    )

    assert first_status == "progressed"
    assert second_status == "reached"
    assert arm.current[0] == pytest.approx(0.05)
    assert len(factory.calls) == 19
    assert second_diagnostics["cartesian_multi_pass_goals_completed"] == 1


def test_stage6_direct_feedback_reaches_a_dense_target():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    factory = _Factory(lambda target: [target[0], 0.0])

    status, diagnostics = _execute_stage6(
        _Scene(arm), ((arm, _target(0.018)),), factory
    )

    assert status == "reached"
    assert factory.calls == []
    assert arm.solver_events == ["pseudo_inverse"]
    assert diagnostics["selected_via_pseudo_inverse"] == 1
    assert diagnostics["cartesian_direct_goal_reaches"] == 1
    np.testing.assert_allclose(arm.current, [0.018, 0.0])


def test_stage6_converged_execution_continues_across_bounded_policy_actions():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    factory = _Factory(lambda _target: [0.2, 0.0])
    scene = _SlowScene(arm)

    runs = []
    for _ in range(10):
        runs.append(_execute_stage6(scene, ((arm, _target(0.2)),), factory))
        if runs[-1][0] == "reached":
            break

    assert runs[0][0] == "progressed"
    assert runs[-1][0] == "reached"
    assert scene.steps > 4
    assert arm.current[0] == pytest.approx(0.2)
    assert (
        sum(
            diagnostics["trac_ik_distance_controller_raw_physics_steps"]
            for _, diagnostics in runs
        )
        == scene.steps
    )
    assert all(
        diagnostics["trac_ik_distance_controller_raw_physics_steps"] <= 8
        for _, diagnostics in runs
    )


def test_stage6_reuses_a_progress_verified_solution_for_the_same_target():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _Scene(arm)
    factory = _Factory(lambda _target: [0.2, 0.0])

    first_status, _first_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.2)),), factory
    )
    # Move outside the pre-execution acceptance envelope while preserving the
    # same absolute target, so this test continues to exercise the verified
    # joint-solution cache rather than the no-op completion path.
    arm.current = np.asarray([0.198, 0.0])
    second_status, second_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.2)),), factory
    )

    assert [first_status, second_status] == ["reached", "reached"]
    assert arm.solver_events == ["pseudo_inverse"]
    assert second_diagnostics["same_target_joint_cache_hits"] == 1
    np.testing.assert_allclose(arm.current, [0.2, 0.0])


def test_stage6_direct_feedback_reports_real_progress_for_a_farther_target():
    arm = _CartesianArm(jacobian_result=lambda position: [0.5 * position[0], 0.0])
    factory = _Factory(lambda target: [0.5 * target[0], 0.0])

    status, diagnostics = _execute_stage6(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "progressed"
    np.testing.assert_allclose(arm.current, [0.025, 0.0])
    assert diagnostics["cartesian_direct_goal_progress_accepts"] == 1
    assert diagnostics["cartesian_goal_directed_progress_accepts"] == 1


def test_stage6_slow_motion_converges_across_bounded_policy_actions():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _SlowScene(arm)
    factory = _Factory(lambda target: [target[0], 0.0])

    first_status, first_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.05)),), factory
    )
    second_status, second_diagnostics = _execute_stage6(
        scene, ((arm, _target(0.05)),), factory
    )

    assert first_status == "progressed"
    assert second_status == "reached"
    assert scene.steps > 4
    assert arm.current[0] == pytest.approx(0.05)
    assert first_diagnostics["cartesian_policy_action_physics_budget_exhaustions"] == 1
    assert second_diagnostics.get("controller_raw_physics_budget_exhaustions", 0) == 0


def test_stage6_small_observed_motion_is_reobserved_between_bounded_actions():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _SlowScene(arm, step=0.0015)
    factory = _Factory(lambda target: [target[0], 0.0])
    config = Stage6IKControllerConfig()

    runs = []
    for _ in range(10):
        runs.append(_execute_stage6(scene, ((arm, _target(0.05)),), factory))
        if runs[-1][0] == "reached":
            break

    assert runs[0][0] == "progressed"
    assert runs[-1][0] == "reached"
    assert scene.steps > 4
    assert abs(arm.current[0] - 0.05) <= (
        config.physical_completion_translation_tolerance_m
    )
    assert all(
        diagnostics["trac_ik_distance_controller_raw_physics_steps"] <= 8
        for _, diagnostics in runs
    )


def test_stage6_bounds_total_raw_physics_before_closed_loop_reobservation():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _SlowScene(arm, step=5.0e-5)
    factory = _Factory(lambda target: [target[0], 0.0])
    config = Stage6IKControllerConfig(
        cartesian_continuation_max_segments=1000,
        cartesian_continuation_max_raw_physics_steps=64,
    )

    status, diagnostics = _execute_stage6(
        scene,
        ((arm, _target(0.05)),),
        factory,
        config=config,
    )

    assert status == "progressed"
    assert scene.steps == 64
    assert 0.0 < arm.current[0] < 0.05
    assert diagnostics["cartesian_policy_action_physics_budget_exhaustions"] == 1
    assert diagnostics["cartesian_policy_action_raw_physics_steps_max"] == 64


def test_stage6_native_primary_avoids_a_coarse_external_model_solution():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _Scene(arm)
    factory = _Factory(lambda target: [target[0] - 0.0015, 0.0])

    status, diagnostics = _execute_stage6(scene, ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert arm.current[0] == pytest.approx(0.05)
    assert arm.solver_events == ["pseudo_inverse"]
    assert factory.calls == []
    assert diagnostics["selected_via_pseudo_inverse"] == 1
    assert diagnostics["selected_via_trac_ik_distance"] == 0


def test_stage6_uses_external_solver_when_native_primary_has_no_solution():
    arm = _CartesianArm(jacobian_result=_IKError("no native solution"))
    scene = _Scene(arm)
    factory = _Factory(lambda target: [target[0], 0.0])

    status, diagnostics = _execute_stage6(scene, ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert arm.current[0] == pytest.approx(0.05)
    assert arm.solver_events == ["pseudo_inverse", "trac_ik_distance"]
    assert diagnostics["pseudo_inverse_ik_failures"] == 1
    assert diagnostics["selected_via_trac_ik_distance"] == 1


def test_stage6_escalates_same_target_after_primary_physically_stalls():
    arm = _CartesianArm(jacobian_result=lambda _position: [0.0, 0.0])
    scene = _Scene(arm)
    factory = _Factory(lambda target: [target[0], 0.0])

    status, diagnostics = _execute_stage6(scene, ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert arm.solver_events == ["pseudo_inverse", "trac_ik_distance"]
    assert arm.current[0] == pytest.approx(0.05)
    assert diagnostics["physical_stall_solver_escalations"] == 1
    assert diagnostics["physical_stall_solver_tier_max"] == 1
    assert diagnostics["selected_via_trac_ik_distance"] == 1


def test_stage6_persists_solver_escalation_across_raw_physics_budgets():
    arm = _CartesianArm(jacobian_result=lambda _position: [0.0, 0.0])
    scene = _Scene(arm)
    factory = _Factory(lambda target: [target[0], 0.0])
    config = Stage6IKControllerConfig(
        cartesian_continuation_max_raw_physics_steps=1,
    )

    first_status, first_diagnostics = _execute_stage6(
        scene,
        ((arm, _target(0.05)),),
        factory,
        config=config,
    )
    second_status, second_diagnostics = _execute_stage6(
        scene,
        ((arm, _target(0.05)),),
        factory,
        config=config,
    )

    assert first_status == "stopped"
    assert first_diagnostics["same_target_cross_cycle_solver_escalations"] == 1
    assert second_status == "reached"
    assert arm.solver_events == ["pseudo_inverse", "trac_ik_distance"]
    assert second_diagnostics["same_target_cross_cycle_solver_resumes"] == 1
    assert arm.current[0] == pytest.approx(0.05)


def test_stage6_resolves_again_when_cartesian_target_changes():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _Scene(arm)
    factory = _Factory(lambda target: [target[0], 0.0])

    _execute_stage6(scene, ((arm, _target(0.02)),), factory)
    _execute_stage6(scene, ((arm, _target(0.03)),), factory)

    assert arm.solver_events == ["pseudo_inverse", "pseudo_inverse"]


def test_global_controller_does_not_use_the_stage6_joint_cache():
    arm = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    scene = _Scene(arm)
    factory = _Factory(lambda target: [target[0], 0.0])

    _execute(scene, ((arm, _target(0.02)),), factory)
    _execute(scene, ((arm, _target(0.02)),), factory)

    assert arm.solver_events == ["pseudo_inverse", "pseudo_inverse"]


def test_stage6_sampling_fallback_is_collision_aware():
    arm = _CartesianArm(
        jacobian_result=_IKError("no local solution"),
        sampling_result=[[0.005, 0.0]],
    )
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute_stage6(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "progressed"
    assert len(arm.sampling_calls) >= 1
    _args, kwargs = arm.sampling_calls[0]
    assert kwargs["ignore_collisions"] is False
    assert diagnostics["selected_via_sampling"] >= 1


def test_stage6_sampling_relaxes_collision_filter_only_after_aware_failure():
    arm = _CollisionSelectiveSamplingArm()
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute_stage6(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert [call[1]["ignore_collisions"] for call in arm.sampling_calls] == [
        False,
        True,
    ]
    assert diagnostics["sampling_collision_relaxed_attempts"] == 1
    assert diagnostics["sampling_collision_relaxed_successes"] == 1
    assert diagnostics["selected_via_collision_relaxed_sampling"] == 1


def test_stage6_near_fallback_uses_fast_linear_path_before_nonlinear_planning():
    arm = _LinearPathArm(collision_aware_fails=True)
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute_stage6(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "reached"
    assert [call[1]["ignore_collisions"] for call in arm.linear_path_calls] == [
        False,
        True,
    ]
    assert arm.path_calls == []
    assert diagnostics["linear_path_collision_aware_failures"] == 1
    assert diagnostics["linear_path_collision_relaxed_successes"] == 1
    assert diagnostics["far_target_planner_attempts"] == 0


def test_stage6_near_fallback_skips_unbounded_nonlinear_planning():
    arm = _LinearPathArm(collision_aware_fails=True, relaxed_fails=True)
    factory = _Factory(RuntimeError("no external solution"))
    diagnostics = initialize_global_ik_controller_diagnostics()

    with pytest.raises(_InvalidActionError):
        execute_stage6_ik_ee_control(
            _Scene(arm),
            ((arm, _target(0.05)),),
            config=Stage6IKControllerConfig(),
            diagnostics=diagnostics,
            external_solver_factory=factory,
            ik_error=_IKError,
            configuration_error=_ConfigurationError,
            configuration_path_error=_ConfigurationPathError,
            invalid_action_error=_InvalidActionError,
            path_algorithm="RRTConnect",
            error_message="stage6 IK failed",
        )

    assert arm.path_calls == []
    assert diagnostics["near_target_nonlinear_planner_skips"] == 1
    assert diagnostics["far_target_planner_attempts"] == 0


def test_stage6_near_fallback_uses_collision_aware_rrt_after_measured_stall():
    arm = _LinearPathArm(collision_aware_fails=True)
    diagnostics = initialize_global_ik_controller_diagnostics()

    command = _prepare_collision_aware_path_command(
        arm,
        _target(0.05),
        config=Stage6IKControllerConfig(),
        diagnostics=diagnostics,
        configuration_path_error=_ConfigurationPathError,
        invalid_action_error=_InvalidActionError,
        path_algorithm="RRTConnect",
        error_message="stage6 IK failed",
        allow_near_nonlinear=True,
    )

    assert command is not None
    assert command.mode == "planned_path"
    assert [call[1]["ignore_collisions"] for call in arm.linear_path_calls] == [False]
    assert len(arm.path_calls) == 1
    assert arm.path_calls[0][1]["ignore_collisions"] is False
    assert diagnostics["near_target_nonlinear_planner_attempts"] == 1
    assert diagnostics["nonlinear_path_successes"] == 1
    assert diagnostics["linear_path_collision_relaxed_attempts"] == 0


def test_stage6_path_families_advance_after_physical_not_only_planning_stall():
    arm = _PhysicallyStalledPathArm()
    scene = _RelaxedPathOnlyScene(arm)
    factory = _Factory(RuntimeError("no external solution"))
    config = Stage6IKControllerConfig(
        cartesian_continuation_max_raw_physics_steps=1,
    )
    statuses = []
    diagnostics = []

    for _ in range(7):
        status, audit = _execute_stage6(
            scene,
            ((arm, _target(0.05)),),
            factory,
            config=config,
        )
        statuses.append(status)
        diagnostics.append(audit)
        if status == "reached":
            break

    assert statuses == ["stopped"] * 6 + ["reached"]
    assert arm.path_family_history == [
        "aware_linear",
        "aware_linear",
        "aware_linear",
        "aware_linear",
        "aware_linear",
        "aware_nonlinear",
        "relaxed_linear",
    ]
    assert diagnostics[-2]["nonlinear_path_successes"] == 1
    assert diagnostics[-1]["linear_path_collision_relaxed_successes"] == 1
    assert max(row["physical_stall_solver_tier_max"] for row in diagnostics) == 6
    assert arm.current[0] == pytest.approx(0.05)


def test_stage6_keeps_all_bimanual_targets_active_on_the_shared_physics_clock():
    right = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    left = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    left.current = np.asarray([0.05, 0.0])
    left.target = left.current.copy()
    factory = _Factory(lambda target: [target[0], 0.0])

    status, diagnostics = _execute_stage6(
        _Scene(right, left),
        ((right, _target(0.05)), (left, _target(0.05))),
        factory,
    )

    assert status == "reached"
    assert right.solver_events == ["pseudo_inverse"]
    assert left.solver_events == ["pseudo_inverse"]
    assert factory.calls == []


def test_stage6_accepts_one_arm_progress_while_the_other_is_stationary():
    moving = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    stalled = _CartesianArm(jacobian_result=lambda position: [position[0], 0.0])
    factory = _Factory(lambda target: [target[0], 0.0])

    per_arm_status = {}
    status, diagnostics = _execute_stage6(
        _PartiallyStalledScene(moving, stalled),
        ((moving, _target(0.05)), (stalled, _target(0.05))),
        factory,
        per_arm_status_out=per_arm_status,
    )

    assert status == "progressed"
    assert moving.current[0] == pytest.approx(0.05)
    assert stalled.current[0] == pytest.approx(0.0)
    assert per_arm_status == {
        id(moving): "reached",
        id(stalled): "stopped",
    }
    assert diagnostics["cartesian_partial_arm_progress_accepts"] == 1


def test_stage6_reports_joint_motion_without_cartesian_progress_as_stopped():
    arm = _CartesianArm(jacobian_result=[-0.2, 0.0])
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute_stage6(_Scene(arm), ((arm, _target(0.05)),), factory)

    assert status == "stopped"
    assert diagnostics["physical_stall_solver_escalations"] >= 1
    assert diagnostics["reached_joint_target_with_cartesian_residual"] >= 1


def test_stage6_rejects_discontinuous_sampling_branch_after_physical_stall():
    class _StalledPrimaryWithLargeSampling(_LinearPathArm):
        def __init__(self):
            super().__init__()
            self.jacobian_result = lambda _position: [0.0, 0.0]
            self.sampling_result = [[0.8, 0.0]]

    arm = _StalledPrimaryWithLargeSampling()
    factory = _Factory(RuntimeError("no external solution"))

    status, diagnostics = _execute_stage6(
        _Scene(arm),
        ((arm, _target(0.05)),),
        factory,
    )

    assert status == "reached"
    assert diagnostics["physical_stall_solver_escalations"] >= 1
    assert diagnostics["candidate_rejections_joint_delta_abs"] >= 1
    assert diagnostics["selected_via_sampling"] == 0
    assert arm.linear_path_calls
    assert diagnostics["linear_path_collision_aware_successes"] >= 1
    assert diagnostics["selected_joint_delta_abs_max"] < 0.8
    np.testing.assert_allclose(arm.current, [0.05, 0.0])


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

    status, diagnostics = _execute(_Scene(arm), ((arm, _target(0.100001)),), factory)

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
