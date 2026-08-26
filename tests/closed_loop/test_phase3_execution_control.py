"""Phase-three acceptance tests for roles, weighted PoE, and task cursor control."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import (
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    GaussianMarginal,
    product_of_experts,
)
from essay2608.policy.closed_loop import (
    ClosedLoopBelief,
    ClosedLoopExecutionConfig,
    ClosedLoopExecutionController,
    ClosedLoopTaskModelBuilder,
    ExecutionDecision,
    FrameRole,
    FrameRoleRouter,
    LinkPendingCandidate,
    MismatchConfig,
    MismatchKind,
    MismatchTracker,
    ProgressEstimate,
    ProgressStatus,
    RelationDecision,
    RelationEstimate,
    RelationEventId,
    RuntimeFeatureBuilder,
    RuntimeObservation,
    StateId,
    weighted_product_of_experts,
)
from essay2608.policy.dynamac import pose_compose, pose_inverse


def pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture()
def phase3_model():
    demonstrations = []
    duration = 4
    grasp_offset = pose(-0.04)
    for demo_index in range(5):
        object_start = pose(0.35 + 0.025 * demo_index, 0.01 * demo_index)
        ee = []
        objects = []
        gripper = []
        for index in range(duration):
            progress = index / (duration - 1)
            ee.append(
                pose_compose(
                    object_start,
                    pose(-0.16 * (1.0 - progress), 0.0, 0.03 * (1.0 - progress)),
                )
            )
            objects.append(object_start)
            gripper.append(1.0)
        carry_start = ee[-1]
        for index in range(duration):
            progress = index / (duration - 1)
            current_ee = pose_compose(
                carry_start,
                pose(0.12 * progress, 0.08 * progress, 0.04 * progress),
            )
            ee.append(current_ee)
            objects.append(pose_compose(current_ee, pose_inverse(grasp_offset)))
            gripper.append(-1.0)
        released_object = objects[-1]
        release_start = ee[-1]
        for index in range(duration):
            progress = index / (duration - 1)
            ee.append(
                pose_compose(
                    release_start,
                    pose(0.02 * progress, -0.03 * progress, 0.10 * progress),
                )
            )
            objects.append(released_object)
            gripper.append(1.0)
        ee_array = np.stack(ee)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee_array,
                action_pose=ee_array.copy(),
                gripper=np.asarray(gripper),
                frames={"object": np.stack(objects)},
                skill=np.repeat([0, 1, 2], duration),
                name=f"phase3_{demo_index}",
            )
        )
    policy = DynaMAC(
        DynaMACConfig(
            tau_m=0.005,
            tau_omega=0.0,
            eq6_empty_selection="keep_argmax",
            kinematic_analysis_enabled=False,
            link_mask_scope="timestep",
            default_mode_strategy="map",
        )
    ).fit(demonstrations)
    model = ClosedLoopTaskModelBuilder().build(
        policy,
        demonstrations,
        recoverable_frames=("object",),
    )
    policy.reset(
        DynaMACObservation(
            demonstrations[0].ee_pose[0],
            {"object": demonstrations[0].frames["object"][0]},
        ),
        mode_strategy="map",
    )
    return model, demonstrations


def relation_estimate(
    frame: str,
    posterior: tuple[float, float],
    decision: RelationDecision,
    *,
    information_weight: float = 1.0,
) -> RelationEstimate:
    values = np.asarray(posterior, dtype=np.float64)
    return RelationEstimate(
        frame_id=frame,
        posterior=values,
        predicted=values,
        demonstration_prior=np.asarray([0.5, 0.5]),
        observation_likelihood=np.ones(2),
        information_weight=information_weight,
        entropy=-float(np.sum(values * np.log(np.maximum(values, 1.0e-12)))),
        informative=decision != RelationDecision.UNKNOWN,
        decision_state=decision,
    )


def runtime_features(
    model,
    demonstrations,
    state_id: StateId,
    *,
    static: bool = False,
):
    demo = demonstrations[0]
    global_index = (
        sum(
            model.base_policy.skills[index].duration
            for index in range(state_id.skill_index)
        )
        + state_id.local_index
    )
    previous_index = max(0, global_index - 1)
    virtual_frames = {
        f"virtual_skill_{skill.label}": demo.ee_pose[
            sum(
                model.base_policy.skills[index].duration for index in range(skill_index)
            )
        ].copy()
        for skill_index, skill in enumerate(model.base_policy.skills)
    }
    previous = RuntimeObservation(
        tick=0,
        ee_pose=demo.ee_pose[previous_index],
        frame_poses={
            "object": demo.frames["object"][previous_index],
            **virtual_frames,
        },
        gripper_state=np.atleast_1d(demo.gripper[previous_index]),
        previous_command_pose=None,
        previous_ee_pose=None,
        tracking_reliability={},
        frame_visibility={},
    )
    current_ee = demo.ee_pose[previous_index if static else global_index]
    current_object = demo.frames["object"][previous_index if static else global_index]
    current = RuntimeObservation(
        tick=1,
        ee_pose=current_ee,
        frame_poses={"object": current_object, **virtual_frames},
        gripper_state=np.atleast_1d(demo.gripper[global_index]),
        previous_command_pose=current_ee,
        previous_ee_pose=demo.ee_pose[previous_index],
        tracking_reliability={},
        frame_visibility={},
    )
    return RuntimeFeatureBuilder().build(current, previous)


def belief_for(
    model,
    demonstrations,
    *,
    tick: int,
    nominal: StateId,
    estimated: StateId,
    posterior: dict[StateId, float] | None = None,
    relation: RelationEstimate | None = None,
    status: ProgressStatus = ProgressStatus.ALIGNED,
    static: bool = False,
) -> ClosedLoopBelief:
    posterior = posterior or {estimated: 1.0}
    confidence = max(posterior.values())
    progress = ProgressEstimate(
        prior=dict(posterior),
        posterior=dict(posterior),
        nominal_state=nominal,
        estimated_state=estimated,
        confidence=confidence,
        entropy=0.0,
        best_explanation_score=1.0,
        status=status,
    )
    return ClosedLoopBelief(
        tick=tick,
        runtime_features=runtime_features(
            model, demonstrations, estimated, static=static
        ),
        relation_estimates={} if relation is None else {"object": relation},
        progress=progress,
        candidate_scores={},
        relation_changes=(),
        local_candidates=tuple(posterior),
        expanded_candidates=(),
    )


def force_relation_prior(model, state_id: StateId, linked: bool) -> None:
    node = model.state(state_id)
    node.demo_relation_priors["object"][:] = (
        np.asarray([0.01, 0.99]) if linked else np.asarray([0.99, 0.01])
    )


def dynamac_observation(model, demonstrations, state_id: StateId) -> DynaMACObservation:
    index = (
        sum(
            model.base_policy.skills[skill].duration
            for skill in range(state_id.skill_index)
        )
        + state_id.local_index
    )
    demo = demonstrations[0]
    virtual_frames = {
        f"virtual_skill_{skill.label}": demo.ee_pose[
            sum(
                model.base_policy.skills[index].duration for index in range(skill_index)
            )
        ]
        for skill_index, skill in enumerate(model.base_policy.skills)
    }
    return DynaMACObservation(
        demo.ee_pose[index],
        {"object": demo.frames["object"][index], **virtual_frames},
    )


def test_dynamic_roles_execute_monitor_recover_and_defer(phase3_model) -> None:
    model, demos = phase3_model
    external_state = StateId(0, 1)
    linked_state = StateId(1, 1)
    force_relation_prior(model, external_state, linked=False)
    force_relation_prior(model, linked_state, linked=True)
    router = FrameRoleRouter(model)

    external_belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=external_state,
        estimated=external_state,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    execute = router.route(external_state, external_belief, mode_by_skill={0: 0})
    assert execute.decisions["object"].role == FrameRole.EXECUTE
    assert execute.execution_weights["object"] > 0.0

    unknown_belief = belief_for(
        model,
        demos,
        tick=2,
        nominal=external_state,
        estimated=external_state,
        relation=relation_estimate(
            "object",
            (0.5, 0.5),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )
    deferred = router.route(external_state, unknown_belief, mode_by_skill={0: 0})
    assert deferred.decisions["object"].role == FrameRole.DEFER
    assert deferred.execution_weights["object"] == pytest.approx(
        execute.execution_weights["object"]
    )
    assert deferred.blocks_advance is True

    linked_belief = belief_for(
        model,
        demos,
        tick=3,
        nominal=linked_state,
        estimated=linked_state,
        relation=relation_estimate("object", (0.05, 0.95), RelationDecision.LINKED),
    )
    monitor = router.route(linked_state, linked_belief, mode_by_skill={1: 0})
    assert monitor.decisions["object"].role == FrameRole.MONITOR
    assert monitor.execution_weights["object"] == 0.0
    assert monitor.decisions["object"].monitor is True
    assert "object" in linked_belief.relation_estimates

    mismatch_belief = belief_for(
        model,
        demos,
        tick=4,
        nominal=linked_state,
        estimated=linked_state,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    recover = router.route(linked_state, mismatch_belief, mode_by_skill={1: 0})
    assert recover.decisions["object"].role == FrameRole.RECOVER
    assert recover.execution_weights["object"] == 0.0
    assert recover.blocks_advance is True
    assert recover.recovery_intents[0].expected_relation == RelationDecision.LINKED


def test_confirmed_unselected_link_is_monitored_without_reenabling_poe(
    phase3_model,
) -> None:
    model, demos = phase3_model
    origin_key = next(
        key for key in sorted(model.link_origins) if key.frame_id == "object"
    )
    state_id = origin_key.state_id
    node = model.state(state_id)
    node.selected_frames = tuple(
        frame for frame in node.selected_frames if frame != "object"
    )
    node.mode_selected_frames = tuple(
        tuple(frame for frame in frames if frame != "object")
        for frames in node.mode_selected_frames
    )
    router = FrameRoleRouter(model)
    modes = {state_id.skill_index: origin_key.mode}

    linked = belief_for(
        model,
        demos,
        tick=1,
        nominal=state_id,
        estimated=state_id,
        relation=relation_estimate("object", (0.02, 0.98), RelationDecision.LINKED),
    )
    monitored = router.route(state_id, linked, mode_by_skill=modes)
    assert monitored.decisions["object"].role == FrameRole.MONITOR
    assert monitored.decisions["object"].selected_offline is False
    assert "object" not in monitored.execution_weights
    assert monitored.blocks_advance is False

    external = belief_for(
        model,
        demos,
        tick=2,
        nominal=state_id,
        estimated=state_id,
        relation=relation_estimate("object", (0.98, 0.02), RelationDecision.EXTERNAL),
    )
    recovered = router.route(state_id, external, mode_by_skill=modes)
    assert recovered.decisions["object"].role == FrameRole.RECOVER
    assert recovered.decisions["object"].selected_offline is False
    assert "object" not in recovered.execution_weights
    assert recovered.blocks_advance is True
    assert recovered.recovery_intents[0].frame_id == "object"


def test_weighted_poe_one_preserves_baseline_and_zero_removes_expert() -> None:
    first = GaussianMarginal("first", pose(0.0), np.eye(6) * 0.02)
    second = GaussianMarginal("second", pose(0.1), np.eye(6) * 0.03)
    baseline_mean, baseline_covariance, _ = product_of_experts((first, second))
    one_mean, one_covariance, _ = weighted_product_of_experts(
        (first, second), (1.0, 1.0)
    )
    assert np.allclose(one_mean, baseline_mean)
    assert np.allclose(one_covariance, baseline_covariance)

    removed_mean, removed_covariance, weights = weighted_product_of_experts(
        (first, second), (1.0, 0.0)
    )
    single_mean, single_covariance, _ = product_of_experts((first,))
    assert np.allclose(removed_mean, single_mean)
    assert np.allclose(removed_covariance, single_covariance)
    assert weights == {"first": pytest.approx(1.0)}
    assert np.allclose(first.mean, pose(0.0))
    assert np.allclose(second.mean, pose(0.1))


def test_controller_holds_and_continues_querying_current_state(phase3_model) -> None:
    model, demos = phase3_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    controller = ClosedLoopExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=current,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
    )
    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert result.weighted_action.action is not None
    assert result.weighted_action.action.diagnostics["time_index"] == 0


def test_unknown_holds_but_reuses_last_trusted_servo_weight(phase3_model) -> None:
    model, demos = phase3_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    controller = ClosedLoopExecutionController(model)
    controller.reset(current)

    trusted = belief_for(
        model,
        demos,
        tick=0,
        nominal=current,
        estimated=current,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    controller.role_router.route(current, trusted, mode_by_skill={0: 0})
    unknown = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate(
            "object",
            (0.95, 0.05),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )
    result = controller.update(
        unknown,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
    )
    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert result.roles.decisions["object"].role == FrameRole.DEFER
    assert result.roles.execution_weights["object"] > 0.0
    assert result.weighted_action.action is not None
    assert result.weighted_action.action.diagnostics["time_index"] == 0


def test_controller_advances_one_direct_successor_only(phase3_model) -> None:
    model, demos = phase3_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, successor, linked=False)
    controller = ClosedLoopExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    result = controller.update(
        belief,
        dynamac_observation(model, demos, successor),
        mode_by_skill={0: 0},
    )
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.cursor_after.reference_state != StateId(0, 2)
    assert result.weighted_action.action is not None
    assert result.weighted_action.action.diagnostics["time_index"] == 1


def test_controller_realigns_index_without_reverse_trajectory_playback(
    phase3_model,
) -> None:
    model, demos = phase3_model
    current = StateId(0, 2)
    nominal = StateId(0, 3)
    estimated = StateId(0, 0)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, estimated, linked=False)
    controller = ClosedLoopExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=nominal,
        estimated=estimated,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    result = controller.update(
        belief,
        dynamac_observation(model, demos, estimated),
        mode_by_skill={0: 0},
    )
    assert result.decision == ExecutionDecision.REALIGN
    assert result.cursor_after.reference_state == estimated
    assert result.weighted_action.action is not None
    assert result.weighted_action.action.diagnostics["time_index"] == 0
    assert result.weighted_action.action.diagnostics["query_advances_clock"] is False


def test_reliable_relation_mismatch_blocks_progress_and_emits_after_persistence(
    phase3_model,
) -> None:
    model, demos = phase3_model
    current = StateId(1, 0)
    successor = StateId(1, 1)
    force_relation_prior(model, current, linked=True)
    controller = ClosedLoopExecutionController(
        model,
        ClosedLoopExecutionConfig(
            mismatch=MismatchConfig(
                no_plausible_cycles=3,
                relation_mismatch_cycles=2,
                persistent_hold_cycles=10,
                stalled_progress_cycles=10,
            )
        ),
    )
    controller.reset(current)
    first = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    first_result = controller.update(
        first,
        dynamac_observation(model, demos, current),
        mode_by_skill={1: 0},
    )
    assert first_result.decision == ExecutionDecision.HOLD
    assert first_result.cursor_after.reference_state == current
    assert first_result.mismatch.events == ()

    second = belief_for(
        model,
        demos,
        tick=2,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    second_result = controller.update(
        second,
        dynamac_observation(model, demos, current),
        mode_by_skill={1: 0},
    )
    assert [event.kind for event in second_result.mismatch.events] == [
        MismatchKind.RELATION_MISMATCH
    ]
    assert second_result.mismatch.events[0].frame_ids == ("object",)


def test_role_router_never_changes_cursor_and_controller_owns_all_commits(
    phase3_model,
) -> None:
    model, demos = phase3_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, successor, linked=False)
    controller = ClosedLoopExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    before = controller.cursor
    controller.role_router.route(current, belief, mode_by_skill={0: 0})
    assert controller.cursor == before
    result = controller.update(
        belief,
        dynamac_observation(model, demos, successor),
        mode_by_skill={0: 0},
    )
    assert controller.cursor == result.cursor_after
    with pytest.raises(ValueError, match="每个递增控制周期只能提交一次"):
        controller.update(
            belief,
            dynamac_observation(model, demos, successor),
            mode_by_skill={0: 0},
        )


def test_pending_unknown_with_observable_low_excitation_requests_verification(
    phase3_model,
) -> None:
    model, demos = phase3_model
    pending_state = StateId(0, 3)
    future_linked = StateId(1, 0)
    force_relation_prior(model, pending_state, linked=False)
    force_relation_prior(model, future_linked, linked=True)
    event_id = RelationEventId(
        model.arm_id,
        "object",
        pending_state.skill_index,
        0,
        0,
        "link_pending",
    )
    model.link_pending_events[event_id] = LinkPendingCandidate(
        event_id=event_id,
        arm_id=model.arm_id,
        frame_id="object",
        candidate_state=pending_state,
        context_state=StateId(0, 0),
        local_means=np.stack([pose(0.0), pose(0.01)]),
        local_covariances=np.stack([np.eye(6) * 0.01] * 2),
        gripper_commands=np.asarray([[1.0], [-1.0]]),
        demonstration_indices=(0, 1, 2, 3, 4),
        event_local_indices=(3, 3, 3, 3, 3),
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=pending_state,
        estimated=pending_state,
        relation=relation_estimate(
            "object",
            (0.5, 0.5),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )
    assert belief.runtime_features.frame_pair_available["object"] is True
    assert belief.runtime_features.action_excitation == pytest.approx(0.0)
    snapshot = FrameRoleRouter(model).route(
        pending_state,
        belief,
        mode_by_skill={0: 0, 1: 0, 2: 0},
    )
    assert snapshot.decisions["object"].role == FrameRole.DEFER
    assert snapshot.recovery_intents == ()
    assert len(snapshot.verification_requests) == 1
    assert snapshot.verification_requests[0].pending_event_id == event_id


def test_pending_without_future_linked_requirement_does_not_request_verification(
    phase3_model,
) -> None:
    model, demos = phase3_model
    terminal = StateId(2, 3)
    force_relation_prior(model, terminal, linked=False)
    event_id = RelationEventId(
        model.arm_id,
        "object",
        terminal.skill_index,
        0,
        0,
        "link_pending",
    )
    model.link_pending_events[event_id] = LinkPendingCandidate(
        event_id=event_id,
        arm_id=model.arm_id,
        frame_id="object",
        candidate_state=terminal,
        context_state=StateId(2, 2),
        local_means=np.stack([pose(0.0), pose(0.01)]),
        local_covariances=np.stack([np.eye(6) * 0.01] * 2),
        gripper_commands=np.asarray([[1.0], [-1.0]]),
        demonstration_indices=(0, 1, 2, 3, 4),
        event_local_indices=(3, 3, 3, 3, 3),
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=terminal,
        estimated=terminal,
        relation=relation_estimate(
            "object",
            (0.5, 0.5),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )
    snapshot = FrameRoleRouter(model).route(
        terminal,
        belief,
        mode_by_skill={0: 0, 1: 0, 2: 0},
    )
    assert snapshot.verification_requests == ()


def test_no_plausible_and_persistent_hold_have_independent_counters(
    phase3_model,
) -> None:
    model, demos = phase3_model
    state = StateId(0, 0)
    force_relation_prior(model, state, linked=False)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=state,
        estimated=state,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.NO_PLAUSIBLE_STATE,
    )
    roles = FrameRoleRouter(model).route(state, belief, mode_by_skill={0: 0})
    tracker = MismatchTracker(
        MismatchConfig(
            no_plausible_cycles=1,
            relation_mismatch_cycles=2,
            persistent_hold_cycles=1,
            stalled_progress_cycles=2,
        )
    )
    update = tracker.update(
        belief,
        ClosedLoopExecutionController(model).cursor,
        ExecutionDecision.HOLD,
        roles,
    )
    assert {event.kind for event in update.events} == {
        MismatchKind.NO_PLAUSIBLE_STATE,
        MismatchKind.PERSISTENT_HOLD,
    }


def test_tracked_phase_three_config_matches_code_defaults() -> None:
    path = Path("configs/closed_loop_execution.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == asdict(ClosedLoopExecutionConfig())
    assert ClosedLoopExecutionConfig.from_json(path).to_dict() == payload
