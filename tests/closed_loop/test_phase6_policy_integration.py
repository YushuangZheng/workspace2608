"""Phase-six core/adapter contract tests without launching CoppeliaSim."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from essay2608.policy import DynaMAC, DynaMACObservation
from essay2608.policy.closed_loop import (
    ArmCommand,
    ClosedLoopMultiStreamPolicy,
    ExecutionDecision,
    ExecutionMode,
    PolicyLifecycle,
    RelationEventId,
    RelationVerificationRequest,
    RecoveryTriggerDecision,
    RuntimeObservation,
)
from integrations.rlbench.rlbench_closed_loop.observation_adapter import (
    ClosedLoopObservationAdapter,
    commands_to_rlbench,
)
from integrations.rlbench.rlbench_closed_loop.policy_server import (
    ClosedLoopPolicyServer,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import demonstration_paths


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "integrations/rlbench/models/v4"
BUNDLE_ROOT = ROOT / "integrations/rlbench/models/closed_loop_v1"
DATA_ROOT = ROOT / "integrations/rlbench/data/training/main"


def _pose_xyzw(x: float) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_rlbench_worker_reports_structured_policy_failure_before_next_tick() -> None:
    command = ArmCommand(
        pose=_pose_xyzw(0.1),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        source="structured_failure_hold",
    )
    cycle = SimpleNamespace(
        commands={"single": command},
        lifecycle=PolicyLifecycle.FAILED,
        arms={
            "single": SimpleNamespace(
                failure_reason="no_legal_reentry_state",
                mode_after=ExecutionMode.TASK,
            )
        },
    )
    batch = SimpleNamespace(
        dynamac={"single": object()},
        runtime={"single": SimpleNamespace(ee_pose=_pose_xyzw(0.0))},
    )
    server = object.__new__(ClosedLoopPolicyServer)
    server._pending = None
    server._next_transaction_id = 7
    server._previous_ee = {"single": None}
    server._previous_command = {"single": None}
    server._previous_command_covariance = {"single": None}
    server.arms = ("single",)
    server.bimanual = False
    server._adapt = lambda payload: batch
    server.policy = SimpleNamespace(act=lambda dynamac, runtime: cycle, complete=False)

    response = server._act({})

    assert response["policy_failed"] is True
    assert response["failure_reasons"] == {"single": "no_legal_reentry_state"}
    assert response["transaction_id"] == 7
    assert response["action"] is not None


def test_directional_boundary_verification_request_routes_to_receiver_arm() -> None:
    event_id = RelationEventId("right", "item0", 6, 0, 0, "link_pending")
    request = RelationVerificationRequest(
        arm_id="right",
        frame_id="item0",
        relation="linked",
        pending_event_id=event_id,
    )
    boundary = SimpleNamespace(
        requests={
            "left": SimpleNamespace(verification_requests=(request,)),
            "right": SimpleNamespace(verification_requests=()),
        }
    )
    selected = ClosedLoopMultiStreamPolicy._verification_request(
        "right", None, boundary
    )
    assert selected == request
    assert (
        ClosedLoopMultiStreamPolicy._verification_request("left", None, boundary)
        is None
    )


def test_rlbench_adapter_preserves_shared_snapshot_and_opposite_ee_frames() -> None:
    spec = get_task_spec("bimanual_handover_item")
    task_state = np.concatenate(
        [_pose_xyzw(float(index)) for index in range(len(spec.pose_chunks))]
    )
    payload = {
        "left": {"gripper_pose": _pose_xyzw(10.0), "gripper_open": 1.0},
        "right": {"gripper_pose": _pose_xyzw(20.0), "gripper_open": 0.0},
        "task_low_dim_state": task_state,
    }
    adapter = ClosedLoopObservationAdapter(spec)
    batch = adapter.build(
        payload,
        tick=3,
        previous_ee_pose={"left": None, "right": None},
        previous_command_pose={"left": None, "right": None},
    )
    assert np.allclose(batch.dynamac["left"].frames["right_ee"][:3], [20, 0, 0])
    assert np.allclose(batch.dynamac["right"].frames["left_ee"][:3], [10, 0, 0])
    assert np.allclose(batch.runtime["left"].frame_poses["left_ee"][:3], [10, 0, 0])
    assert np.allclose(batch.runtime["right"].frame_poses["right_ee"][:3], [20, 0, 0])
    assert batch.runtime["left"].gripper_state.tolist() == [1.0]
    assert batch.runtime["right"].gripper_state.tolist() == [-1.0]
    assert all(batch.runtime["left"].frame_visibility.values())
    assert all(
        value == 1.0 for value in batch.runtime["right"].tracking_reliability.values()
    )

    command = ArmCommand(
        pose=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6),
        gripper=np.asarray([1.0]),
        source="test",
    )
    wire = commands_to_rlbench(
        {"left": command, "right": command},
        bimanual=True,
    )
    assert wire.shape == (18,)
    assert wire[7] == wire[16] == 1.0


def test_top_policy_bootstrap_abort_and_rejected_action_do_not_advance() -> None:
    bundle = BUNDLE_ROOT / "stack_wine"
    checkpoint = BASE_ROOT / "stack_wine/model.npz"
    try:
        path = demonstration_paths(DATA_ROOT, "stack_wine", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装 Stage6 StackWine 正常示范")
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装 Stage6 StackWine bundle 或 V4 checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = ClosedLoopMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
    )
    episode = load_low_dim_obs_pickles([path])[0]
    converted = make_unimanual_demonstrations(
        [episode],
        "stack_wine",
        names=[path.parent.name],
    ).demonstrations[0]
    dynamac = DynaMACObservation(
        converted.ee_pose[0],
        {name: values[0] for name, values in converted.frames.items()},
    )
    runtime0 = RuntimeObservation.from_dynamac(
        dynamac,
        tick=0,
        gripper_state=converted.gripper[0],
    )
    policy.reset({"single": dynamac})
    first = policy.act({"single": dynamac}, {"single": runtime0})
    initial = min(policy.task_models["single"].states)
    assert first.arms["single"].execution is not None
    assert first.arms["single"].execution.decision == ExecutionDecision.HOLD
    assert first.arms["single"].belief.progress.nominal_state == initial
    assert first.arms["single"].belief.progress.estimated_state == initial
    policy.abort()
    assert not policy.pending
    assert policy.diagnostics.records == ()

    rejected = policy.act({"single": dynamac}, {"single": runtime0})
    assert rejected.arms["single"].belief.progress.nominal_state == initial
    policy.commit(action_executed=False)
    assert policy.diagnostics.records[-1]["action_commit"] == {
        "action_executed": False,
        "executed_reference_states": None,
    }
    arm_diagnostics = policy.diagnostics.records[-1]["arms"]["single"]
    assert "actual_ee_motion" in arm_diagnostics["motion_features"]
    relation_diagnostics = next(iter(arm_diagnostics["relations"].values()))
    assert "predicted" in relation_diagnostics
    assert "observation_likelihood" in relation_diagnostics
    runtime1 = RuntimeObservation.from_dynamac(
        dynamac,
        tick=1,
        gripper_state=converted.gripper[0],
        previous_ee_pose=dynamac.ee_pose,
        previous_command_pose=dynamac.ee_pose,
    )
    after_hold = policy.act({"single": dynamac}, {"single": runtime1})
    assert after_hold.arms["single"].belief.progress.nominal_state == initial
    assert after_hold.arms["single"].execution is not None
    assert after_hold.arms["single"].execution.cursor_after.reference_state == initial
    policy.abort()
    replay = policy.act({"single": dynamac}, {"single": runtime1})
    assert replay.arms["single"].belief.progress.nominal_state == initial
    policy.commit(executed_reference_states={"single": initial})
    assert policy._last_executed_reference == {"single": initial}


def test_policy_completes_only_after_final_reference_command_is_committed() -> None:
    bundle = BUNDLE_ROOT / "stack_wine"
    checkpoint = BASE_ROOT / "stack_wine/model.npz"
    try:
        path = demonstration_paths(DATA_ROOT, "stack_wine", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装 Stage6 StackWine 正常示范")
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装 Stage6 StackWine bundle 或 V4 checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = ClosedLoopMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
    )
    episode = load_low_dim_obs_pickles([path])[0]
    converted = make_unimanual_demonstrations(
        [episode],
        "stack_wine",
        names=[path.parent.name],
    ).demonstrations[0]
    dynamac = DynaMACObservation(
        converted.ee_pose[-1],
        {name: values[-1] for name, values in converted.frames.items()},
    )
    runtime = RuntimeObservation.from_dynamac(
        dynamac,
        tick=0,
        gripper_state=converted.gripper[-1],
    )
    policy.reset({"single": dynamac})
    model = policy.task_models["single"]
    final_state = max(model.states)
    mode = policy._mode_by_arm_skill["single"][final_state.skill_index]
    node = model.state(final_state)
    policy.belief_updaters["single"].reset(
        initial_progress={final_state: 1.0},
        initial_relations={
            frame: prior[mode].copy()
            for frame, prior in node.demo_relation_priors.items()
        },
    )
    policy.execution_controllers["single"].reset(final_state)

    assert policy.complete is False
    cycle = policy.act({"single": dynamac}, {"single": runtime})
    assert cycle.arms["single"].execution is not None
    assert cycle.arms["single"].execution.weighted_action.state_id == final_state
    assert policy.complete is False
    policy.commit(action_executed=True)
    assert policy.complete is True
    with pytest.raises(RuntimeError, match="已完成"):
        policy.act({"single": dynamac}, {"single": runtime})


def test_no_goal_recovery_servos_frozen_task_target_without_advancing() -> None:
    bundle = BUNDLE_ROOT / "stack_wine"
    checkpoint = BASE_ROOT / "stack_wine/model.npz"
    try:
        path = demonstration_paths(DATA_ROOT, "stack_wine", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装 Stage6 StackWine 正常示范")
    if not bundle.is_dir() or not checkpoint.is_file():
        pytest.skip("本地未安装 Stage6 StackWine bundle 或 V4 checkpoint")

    base = DynaMAC.load(checkpoint)
    policy = ClosedLoopMultiStreamPolicy.load(
        bundle,
        base_policies={"single": base},
    )
    episode = load_low_dim_obs_pickles([path])[0]
    converted = make_unimanual_demonstrations(
        [episode],
        "stack_wine",
        names=[path.parent.name],
    ).demonstrations[0]
    initial = min(policy.task_models["single"].states)
    observed_pose = converted.ee_pose[0].copy()
    observed_pose[0] += 0.30
    dynamac = DynaMACObservation(
        observed_pose,
        {name: values[0] for name, values in converted.frames.items()},
    )
    runtime = RuntimeObservation.from_dynamac(
        dynamac,
        tick=0,
        gripper_state=converted.gripper[0],
    )
    policy.reset({"single": dynamac})
    policy.recovery_managers["single"].begin_recovery(
        RecoveryTriggerDecision(True, ("single:no_plausible_state",), ()),
        source_state=initial,
        mode=policy._mode_by_arm_skill["single"][initial.skill_index],
    )

    cycle = policy.act({"single": dynamac}, {"single": runtime})
    arm = cycle.arms["single"]
    assert arm.mode_before == ExecutionMode.RECOVERY
    assert arm.execution is None
    assert arm.command.source == "recovery_frozen_task_target"
    assert not np.allclose(arm.command.pose, observed_pose)
    assert policy.execution_controllers["single"].cursor.reference_state == initial
    assert policy._last_executed_reference["single"] is None


def test_top_level_auxiliary_mode_freezes_peer_progress_and_boundaries() -> None:
    bundle = BUNDLE_ROOT / "bimanual_handover_item"
    checkpoints = {
        arm: BASE_ROOT / "bimanual_handover_item" / f"{arm}.npz"
        for arm in ("left", "right")
    }
    try:
        path = demonstration_paths(DATA_ROOT, "bimanual_handover_item", 1)[0]
    except FileNotFoundError:
        pytest.skip("本地未安装 Stage6 HandOver 正常示范")
    if not bundle.is_dir() or not all(path.is_file() for path in checkpoints.values()):
        pytest.skip("本地未安装 Stage6 HandOver bundle 或 V4 checkpoint")

    converted = make_bimanual_demonstrations(
        load_low_dim_obs_pickles([path]),
        "bimanual_handover_item",
        names=[path.parent.name],
    )
    demonstrations = {
        "left": converted.left_demonstrations[0],
        "right": converted.right_demonstrations[0],
    }
    observations = {}
    runtimes = {}
    for arm, peer in (("left", "right"), ("right", "left")):
        demonstration = demonstrations[arm]
        frames = {
            name: values[0].copy() for name, values in demonstration.frames.items()
        }
        frames[f"{peer}_ee"] = demonstrations[peer].ee_pose[0].copy()
        observation = DynaMACObservation(demonstration.ee_pose[0], frames)
        observations[arm] = observation
        runtimes[arm] = RuntimeObservation.from_dynamac(
            observation,
            tick=0,
            gripper_state=demonstration.gripper[0],
        )

    policy = ClosedLoopMultiStreamPolicy.load(
        bundle,
        base_policies={arm: DynaMAC.load(path) for arm, path in checkpoints.items()},
    )
    policy.reset(observations)
    initial = {
        arm: policy.execution_controllers[arm].cursor.reference_state
        for arm in policy.arms
    }
    right_mode = policy._mode_by_arm_skill["right"][initial["right"].skill_index]
    policy.recovery_managers["right"].begin_recovery(
        RecoveryTriggerDecision(True, ("right:no_plausible_state",), ()),
        source_state=initial["right"],
        mode=right_mode,
    )

    cycle = policy.act(observations, runtimes)

    left = cycle.arms["left"]
    assert left.mode_before == left.mode_after == ExecutionMode.TASK
    assert left.execution is None
    assert left.belief.update_sequence == ("frozen_progress", "relation_posterior")
    assert left.command.source.startswith("auxiliary_peer_frozen_task_target")
    assert (
        policy.execution_controllers["left"].cursor.reference_state == initial["left"]
    )
    # Recovery may evaluate its own legal reentry, but the TASK peer neither
    # contributes a normal request nor commits a normal boundary transaction.
    if cycle.boundary is not None:
        assert "left" not in cycle.boundary.requests
        assert cycle.boundary.transaction is None
    assert policy._last_executed_reference["left"] is None
