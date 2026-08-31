"""Phase-five active verification, recovery, and reentry acceptance tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from essay2608.policy import (
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
)
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BoundaryId,
    ClosedLoopBelief,
    ClosedLoopExecutionController,
    ClosedLoopRecoveryConfig,
    ClosedLoopRecoveryManager,
    ClosedLoopTaskModelBuilder,
    EpisodeLinkAnchorRegistry,
    ExecutionMode,
    LinkPendingCandidate,
    LinkRecoveryAnchor,
    MismatchCounters,
    MismatchEvent,
    MismatchKind,
    MismatchUpdate,
    ProbeExitReason,
    ProgressEstimate,
    ProgressStatus,
    RecoveryConfig,
    RecoveryPhase,
    RecoveryTriggerDecision,
    RecoveryTriggerTracker,
    ReentryConfig,
    ReentrySelector,
    RelationDecision,
    RelationEstimate,
    RelationEventId,
    RelationGoalKind,
    RelationGoalPlanner,
    RelationRecoveryController,
    RelationRecoveryIntent,
    RelationStateKey,
    RelationVerificationConfig,
    RelationVerificationController,
    RelationVerificationRequest,
    RuntimeFeatureBuilder,
    RuntimeObservation,
    SafetyConstraintStatus,
    StateId,
    UnlinkMetadataRepository,
    VerificationAttemptSignature,
    VerificationPhase,
)
from essay2608.policy.dynamac import pose_compose, pose_inverse


def pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture()
def phase5_case():
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
                name=f"phase5_{demo_index}",
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
    anchor = next(iter(model.link_anchors.values()))
    pending_id = RelationEventId(
        "single",
        "object",
        anchor.event_id.skill_index,
        anchor.event_id.mode,
        0,
        "link_pending",
    )
    pending = LinkPendingCandidate(
        event_id=pending_id,
        arm_id="single",
        frame_id="object",
        candidate_state=anchor.linked_entry_states[0],
        context_state=anchor.context_state,
        local_means=anchor.local_means,
        local_covariances=anchor.local_covariances,
        gripper_commands=anchor.gripper_commands,
        support_fraction=1.0,
        demonstration_indices=(0, 1, 2, 3, 4),
        event_local_indices=(1, 1, 1, 1, 1),
    )
    model.link_pending_events[pending_id] = pending
    return model, demonstrations, pending


def relation(
    decision: RelationDecision,
    *,
    posterior: tuple[float, float] | None = None,
    information: float = 1.0,
) -> RelationEstimate:
    if posterior is None:
        posterior = {
            RelationDecision.EXTERNAL: (0.9, 0.1),
            RelationDecision.LINKED: (0.1, 0.9),
            RelationDecision.UNKNOWN: (0.5, 0.5),
        }[decision]
    values = np.asarray(posterior, dtype=np.float64)
    return RelationEstimate(
        frame_id="object",
        posterior=values,
        predicted=values,
        demonstration_prior=np.asarray([0.5, 0.5]),
        observation_likelihood=np.ones(2),
        information_weight=information,
        entropy=-float(np.sum(values * np.log(np.maximum(values, 1.0e-12)))),
        informative=information >= 0.1,
        decision_state=decision,
    )


def observation_for(
    model, demonstrations, state: StateId, tick: int
) -> RuntimeObservation:
    demonstration = demonstrations[0]
    index = (
        sum(
            model.base_policy.skills[skill].duration
            for skill in range(state.skill_index)
        )
        + state.local_index
    )
    previous = max(0, index - 1)
    return RuntimeObservation(
        tick=tick,
        ee_pose=demonstration.ee_pose[index],
        frame_poses={"object": demonstration.frames["object"][index]},
        gripper_state=np.atleast_1d(demonstration.gripper[index]),
        previous_command_pose=demonstration.ee_pose[index],
        previous_ee_pose=demonstration.ee_pose[previous],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )


def features_for(model, demonstrations, state: StateId, *, motion: float = 0.01):
    current = observation_for(model, demonstrations, state, 2)
    previous = replace(
        current,
        tick=1,
        ee_pose=pose(current.ee_pose[0] - motion, *current.ee_pose[1:3]),
        previous_ee_pose=None,
        previous_command_pose=None,
    )
    return RuntimeFeatureBuilder().build(current, previous)


def verification_response_features(
    base,
    *,
    ee_translation: float,
    frame_translation: float,
):
    """Return one reliable response to a previously issued verification action."""

    actual = np.asarray([ee_translation, 0.0, 0.0, 0.0, 0.0, 0.0])
    frame = np.asarray([frame_translation, 0.0, 0.0, 0.0, 0.0, 0.0])
    return replace(
        base,
        actual_ee_motion=actual,
        frame_world_motion={"object": frame},
        actual_motion_magnitude=abs(ee_translation),
        frame_pair_available={"object": True},
        paired_tracking_reliability={"object": 1.0},
        relation_information_weight={"object": 1.0},
    )


def belief_for(
    model,
    demonstrations,
    state: StateId,
    estimate: RelationEstimate,
    *,
    tick: int = 2,
    motion: float = 0.01,
) -> ClosedLoopBelief:
    progress = ProgressEstimate(
        prior={state: 1.0},
        posterior={state: 1.0},
        nominal_state=state,
        estimated_state=state,
        confidence=1.0,
        entropy=0.0,
        best_explanation_score=1.0,
        status=ProgressStatus.ALIGNED,
    )
    return ClosedLoopBelief(
        tick=tick,
        runtime_features=features_for(model, demonstrations, state, motion=motion),
        relation_estimates={"object": estimate},
        progress=progress,
        candidate_scores={},
        relation_changes=(),
        local_candidates=(state,),
        expanded_candidates=(),
    )


def test_non_task_belief_update_freezes_progress_and_updates_relation(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    state = StateId(1, 2)
    previous = observation_for(model, demonstrations, state, 5)
    updater = BeliefUpdater(model)
    updater.reset(
        initial_progress={state: 1.0},
        initial_relations={"object": np.asarray([0.5, 0.5])},
        previous_observation=previous,
    )
    current = replace(
        previous,
        tick=6,
        previous_ee_pose=previous.ee_pose,
        previous_command_pose=previous.ee_pose,
    )
    result = updater.update_frozen(current, mode_by_skill={1: 0})
    assert result.progress.posterior == {state: 1.0}
    assert result.progress.estimated_state == state
    assert result.update_sequence == ("frozen_progress", "relation_posterior")
    assert "object" in result.relation_estimates


@pytest.mark.parametrize(
    "initial_relation",
    (RelationDecision.UNKNOWN, RelationDecision.EXTERNAL),
)
def test_verify_link_probes_opposite_approach_returns_and_blocks_repeat(
    phase5_case,
    initial_relation: RelationDecision,
) -> None:
    _, _, pending = phase5_case
    config = RelationVerificationConfig(
        probe_speed=0.10,
        maximum_probe_seconds=0.20,
        minimum_probe_motion=0.001,
        return_position_tolerance=0.001,
    )
    controller = RelationVerificationController(config)
    request = RelationVerificationRequest(
        "single", "object", "linked", pending.event_id
    )
    entry = pose(0.02)
    signature = VerificationAttemptSignature(
        initial_relation,
        pending.candidate_state,
        "grasp-0",
    )
    controller.start(
        request,
        pending,
        task_state=signature.task_state,
        relation_state=signature.relation_state,
        grasp_event=signature.grasp_event,
        entry_pose=entry,
        gripper_command=np.asarray([-1.0]),
        recent_task_poses=(pose(0.0), pose(0.01), entry),
    )
    static_features = replace(
        features_for(*phase5_case[:2], pending.candidate_state),
        actual_motion_magnitude=0.0,
    )
    first = controller.update(
        current_pose=entry,
        features=static_features,
        estimate=relation(RelationDecision.UNKNOWN, information=0.0),
    )
    assert first.phase == VerificationPhase.PROBE
    assert first.action is not None
    assert first.action.pose[0] < entry[0]
    np.testing.assert_array_equal(first.action.gripper_command, [-1.0])
    np.testing.assert_array_equal(first.action.pose[3:], entry[3:])

    informative_features = verification_response_features(
        static_features,
        ee_translation=-0.005,
        frame_translation=-0.005,
    )
    step = first
    for _ in range(config.minimum_response_samples):
        assert step.action is not None
        step = controller.update(
            current_pose=step.action.pose,
            features=informative_features,
            estimate=relation(RelationDecision.LINKED),
        )
    returning = step
    assert returning.phase == VerificationPhase.RETURN
    assert returning.probe_exit_reason == ProbeExitReason.STABLE_RELATION
    assert returning.action is not None
    step = returning
    while step.phase == VerificationPhase.RETURN:
        assert step.action is not None
        step = controller.update(
            current_pose=step.action.pose,
            features=informative_features,
            estimate=relation(RelationDecision.LINKED),
        )
    complete = step
    assert complete.phase == VerificationPhase.COMPLETE
    assert complete.decision == RelationDecision.LINKED
    np.testing.assert_allclose(complete.verified_posterior, [0.1, 0.9])

    with pytest.raises(RuntimeError, match="已经验证过"):
        controller.start(
            request,
            pending,
            task_state=signature.task_state,
            relation_state=signature.relation_state,
            grasp_event=signature.grasp_event,
            entry_pose=entry,
            gripper_command=np.asarray([-1.0]),
            recent_task_poses=(pose(0.0), entry),
        )
    assert controller.attempts.can_attempt(
        pending.event_id,
        replace(signature, task_state=StateId(1, 2)),
    )


def test_verify_link_confirmation_commits_original_filter_posterior(
    phase5_case,
) -> None:
    model, _demonstrations, _pending = phase5_case
    updater = BeliefUpdater(model)
    posterior = np.asarray([0.22, 0.78])

    updater.commit_relation_confirmation(
        "object",
        posterior,
        RelationDecision.LINKED,
    )

    np.testing.assert_allclose(updater._relation_posteriors["object"], posterior)
    assert updater._stable_decisions["object"] == RelationDecision.LINKED
    assert updater._informative_evidence_decisions["object"] == RelationDecision.LINKED
    with pytest.raises(ValueError, match="不一致"):
        updater.commit_relation_confirmation(
            "object",
            posterior,
            RelationDecision.EXTERNAL,
        )


def test_verify_link_timeout_still_returns_and_unsafe_return_fails(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    controller = RelationVerificationController(
        RelationVerificationConfig(
            maximum_probe_seconds=0.05,
            return_position_tolerance=1.0e-5,
            maximum_return_cycles=2,
        )
    )
    request = RelationVerificationRequest(
        "single", "object", "linked", pending.event_id
    )
    entry = pose(0.02)
    controller.start(
        request,
        pending,
        task_state=pending.candidate_state,
        relation_state=RelationDecision.UNKNOWN,
        grasp_event=0,
        entry_pose=entry,
        gripper_command=np.asarray([-1.0]),
        recent_task_poses=(pose(0.0), entry),
    )
    features = replace(
        features_for(model, demonstrations, pending.candidate_state),
        actual_motion_magnitude=0.0,
    )
    probe = controller.update(
        current_pose=entry,
        features=features,
        estimate=relation(RelationDecision.UNKNOWN, information=0.0),
    )
    returning = controller.update(
        current_pose=probe.action.pose,
        features=features,
        estimate=relation(RelationDecision.UNKNOWN, information=0.0),
    )
    assert returning.phase == VerificationPhase.RETURN
    assert returning.probe_exit_reason == ProbeExitReason.TIMEOUT
    failed = controller.update(
        current_pose=probe.action.pose,
        features=features,
        estimate=relation(RelationDecision.UNKNOWN, information=0.0),
        safety=SafetyConstraintStatus(return_safe=False, reason="collision"),
    )
    assert failed.phase == VerificationPhase.FAILED
    assert failed.failure_reason == "collision"


def test_verify_link_can_confirm_during_return_after_probe_timeout(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    config = RelationVerificationConfig(
        maximum_probe_seconds=0.05,
        minimum_response_samples=2,
        minimum_probe_motion=0.001,
        return_position_tolerance=1.0e-5,
    )
    controller = RelationVerificationController(config)
    request = RelationVerificationRequest(
        "single", "object", "linked", pending.event_id
    )
    entry = pose(0.02)
    controller.start(
        request,
        pending,
        task_state=pending.candidate_state,
        relation_state=RelationDecision.UNKNOWN,
        grasp_event=0,
        entry_pose=entry,
        gripper_command=np.asarray([-1.0]),
        recent_task_poses=(pose(0.0), entry),
    )
    static = replace(
        features_for(model, demonstrations, pending.candidate_state),
        actual_motion_magnitude=0.0,
    )
    probe = controller.update(
        current_pose=entry,
        features=static,
        estimate=relation(RelationDecision.UNKNOWN, information=0.0),
    )
    response = verification_response_features(
        static,
        ee_translation=-0.005,
        frame_translation=-0.005,
    )
    returning = controller.update(
        current_pose=probe.action.pose,
        features=response,
        estimate=relation(RelationDecision.LINKED),
    )
    assert returning.phase == VerificationPhase.RETURN
    assert returning.probe_exit_reason == ProbeExitReason.TIMEOUT
    assert returning.verified_posterior is None

    complete = controller.update(
        current_pose=returning.action.pose,
        features=response,
        estimate=relation(
            RelationDecision.LINKED,
            posterior=(0.18, 0.82),
        ),
    )
    assert complete.phase == VerificationPhase.COMPLETE
    assert complete.decision == RelationDecision.LINKED
    np.testing.assert_allclose(complete.verified_posterior, [0.18, 0.82])


def test_verify_link_rejects_single_tick_external_when_probe_window_comoves(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    controller = RelationVerificationController(
        RelationVerificationConfig(
            probe_speed=0.10,
            maximum_probe_seconds=0.50,
            minimum_probe_motion=0.001,
            minimum_response_samples=3,
        )
    )
    request = RelationVerificationRequest(
        "single", "object", "linked", pending.event_id
    )
    entry = pose(0.02)
    controller.start(
        request,
        pending,
        task_state=pending.candidate_state,
        relation_state=RelationDecision.EXTERNAL,
        grasp_event=0,
        entry_pose=entry,
        gripper_command=np.asarray([-1.0]),
        recent_task_poses=(pose(0.0), entry),
    )
    base = features_for(model, demonstrations, pending.candidate_state)
    no_response_yet = replace(base, actual_motion_magnitude=0.0)
    step = controller.update(
        current_pose=entry,
        features=no_response_yet,
        estimate=relation(RelationDecision.EXTERNAL),
    )
    response = verification_response_features(
        base,
        ee_translation=-0.005,
        frame_translation=-0.0045,
    )
    for _ in range(3):
        assert step.action is not None
        step = controller.update(
            current_pose=step.action.pose,
            features=response,
            estimate=relation(RelationDecision.EXTERNAL),
        )
    assert step.phase == VerificationPhase.PROBE
    assert step.response_decision == RelationDecision.LINKED
    assert step.decision == RelationDecision.UNKNOWN

    assert step.action is not None
    step = controller.update(
        current_pose=step.action.pose,
        features=response,
        estimate=relation(RelationDecision.LINKED),
    )
    assert step.phase == VerificationPhase.RETURN
    assert step.decision == RelationDecision.LINKED


def test_verify_link_accepts_external_when_probe_window_does_not_respond(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    controller = RelationVerificationController(
        RelationVerificationConfig(
            probe_speed=0.10,
            maximum_probe_seconds=0.50,
            minimum_probe_motion=0.001,
            minimum_response_samples=3,
        )
    )
    request = RelationVerificationRequest(
        "single", "object", "linked", pending.event_id
    )
    entry = pose(0.02)
    controller.start(
        request,
        pending,
        task_state=pending.candidate_state,
        relation_state=RelationDecision.EXTERNAL,
        grasp_event=0,
        entry_pose=entry,
        gripper_command=np.asarray([-1.0]),
        recent_task_poses=(pose(0.0), entry),
    )
    base = features_for(model, demonstrations, pending.candidate_state)
    step = controller.update(
        current_pose=entry,
        features=replace(base, actual_motion_magnitude=0.0),
        estimate=relation(RelationDecision.EXTERNAL),
    )
    response = verification_response_features(
        base,
        ee_translation=-0.005,
        frame_translation=0.0,
    )
    for _ in range(3):
        assert step.action is not None
        step = controller.update(
            current_pose=step.action.pose,
            features=response,
            estimate=relation(RelationDecision.EXTERNAL),
        )
    assert step.phase == VerificationPhase.RETURN
    assert step.response_decision == RelationDecision.EXTERNAL
    assert step.decision == RelationDecision.EXTERNAL


def test_pending_activation_is_episode_local_and_anchor_reinstantiates(
    phase5_case,
) -> None:
    model, _, pending = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    with pytest.raises(KeyError, match="没有事件级 LINK 来源"):
        registry.resolve("object", StateId(0, 0), 0)
    runtime = registry.activate_pending(pending.event_id)
    assert runtime.source == "verified_pending"
    assert pending.event_id in model.link_pending_events
    assert StateId(0, 0) not in model.link_origins
    resolved = registry.resolve("object", StateId(0, 0), 0)
    assert resolved.origin_event_id == pending.event_id
    world = registry.instantiate(resolved, pose(1.0), 0.001)
    np.testing.assert_allclose(
        world[0].pose,
        pose_compose(pose(1.0), pending.local_means[0]),
    )
    assert np.all(
        np.diag(world[0].covariance)[:3] >= np.diag(pending.local_covariances[0])[:3]
    )
    registry.release("object")
    with pytest.raises(KeyError):
        registry.resolve("object", StateId(0, 0), 0)


def test_pending_failed_link_uses_provisional_template_then_activates_on_success(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    model.link_anchors.clear()
    model.link_origins.clear()
    registry = EpisodeLinkAnchorRegistry(model)
    planner = RelationGoalPlanner(registry, UnlinkMetadataRepository(model))
    goal = planner.plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=pending.candidate_state,
        mode=pending.event_id.mode,
    )[0]
    assert goal.link_anchor is not None
    assert goal.link_anchor.source == "pending_recovery"
    assert registry.active_pending == {}

    recovery = __import__(
        "essay2608.policy.closed_loop", fromlist=["RelationRecoveryController"]
    ).RelationRecoveryController(
        registry,
        RecoveryConfig(
            pose_position_tolerance=1.0e-6,
            maximum_waypoint_cycles=3,
            maximum_relation_verify_cycles=2,
        ),
    )
    recovery.start((goal,))
    frame = demonstrations[0].frames["object"][4]
    current = demonstrations[0].ee_pose[0]
    result = None
    for _ in range(40):
        estimate = relation(RelationDecision.EXTERNAL)
        result = recovery.update(
            current_pose=current,
            current_gripper=np.asarray([-1.0]),
            frame_poses={"object": frame},
            relation_estimates={"object": estimate},
        )
        if result.action is not None:
            current = result.action.pose
        if result.goal_phase is not None and result.goal_phase.value == "verify":
            for _ in range(20):
                result = recovery.update(
                    current_pose=current,
                    current_gripper=np.asarray([-1.0]),
                    frame_poses={"object": frame},
                    relation_estimates={"object": relation(RelationDecision.LINKED)},
                )
                if result.action is not None:
                    current = result.action.pose
                if result.phase != RecoveryPhase.EXECUTING:
                    break
        if result.phase != RecoveryPhase.EXECUTING:
            break

    assert result is not None and result.phase == RecoveryPhase.REENTRY
    assert registry.active_pending["object"].source == "verified_pending"
    assert registry.active_pending["object"].origin_event_id == pending.event_id


def test_relation_goal_planner_uses_event_context_and_unlink_before_link(
    phase5_case,
) -> None:
    model, _, _ = phase5_case
    original = next(iter(model.link_anchors.values()))
    other_id = RelationEventId("single", "other", 1, 0, 0, "link")
    model.relation_frames = (*model.relation_frames, "other")
    model.link_anchors[other_id] = LinkRecoveryAnchor(
        event_id=other_id,
        arm_id="single",
        frame_id="other",
        context_state=original.context_state,
        local_means=original.local_means,
        local_covariances=original.local_covariances,
        gripper_commands=original.gripper_commands,
        linked_entry_states=(StateId(2, 2),),
    )
    model.link_origins[RelationStateKey("single", "other", StateId(2, 2), 0)] = other_id
    planner = RelationGoalPlanner(
        EpisodeLinkAnchorRegistry(model), UnlinkMetadataRepository(model)
    )
    goals = planner.plan(
        (
            RelationRecoveryIntent(
                "single",
                "other",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.EXTERNAL,
                RelationDecision.LINKED,
            ),
        ),
        source_state=StateId(2, 2),
        mode=0,
    )
    assert [goal.kind for goal in goals] == [
        RelationGoalKind.UNLINK,
        RelationGoalKind.LINK,
    ]
    assert goals[1].link_anchor.origin_event_id == other_id
    assert goals[0].unlink_metadata is not None


def test_link_and_unlink_recovery_are_bounded_and_supply_reentry_states(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    planner = RelationGoalPlanner(registry, UnlinkMetadataRepository(model))
    link_goal = planner.plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=StateId(1, 2),
        mode=0,
    )[0]
    recovery = __import__(
        "essay2608.policy.closed_loop", fromlist=["RelationRecoveryController"]
    ).RelationRecoveryController(
        registry,
        RecoveryConfig(
            pose_position_tolerance=1.0e-6,
            maximum_waypoint_cycles=3,
            maximum_relation_verify_cycles=2,
        ),
    )
    recovery.start((link_goal,))
    frame = demonstrations[0].frames["object"][6]
    current = demonstrations[0].ee_pose[4]
    result = None
    for _ in range(30):
        result = recovery.update(
            current_pose=current,
            current_gripper=np.asarray([-1.0]),
            frame_poses={"object": frame},
            relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
        )
        if result.action is not None:
            current = result.action.pose
        if result.goal_phase is not None and result.goal_phase.value == "verify":
            unconfirmed = recovery.update(
                current_pose=current,
                current_gripper=np.asarray([-1.0]),
                frame_poses={"object": frame},
                relation_estimates={
                    "object": relation(RelationDecision.LINKED, information=0.0)
                },
            )
            assert unconfirmed.phase == RecoveryPhase.EXECUTING
            result = unconfirmed
            for _ in range(20):
                if result.action is not None:
                    current = result.action.pose
                result = recovery.update(
                    current_pose=current,
                    current_gripper=np.asarray([-1.0]),
                    frame_poses={"object": frame},
                    relation_estimates={"object": relation(RelationDecision.LINKED)},
                )
                if result.phase != RecoveryPhase.EXECUTING:
                    break
        if result.phase != RecoveryPhase.EXECUTING:
            break
    assert result is not None and result.phase == RecoveryPhase.REENTRY
    assert set(result.legal_reentry_states) == set(link_goal.legal_reentry_states)

    unlink_goal = planner.plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.EXTERNAL,
                RelationDecision.LINKED,
            ),
        ),
        source_state=StateId(2, 2),
        mode=0,
    )[0]
    recovery = __import__(
        "essay2608.policy.closed_loop", fromlist=["RelationRecoveryController"]
    ).RelationRecoveryController(
        registry,
        RecoveryConfig(
            pose_position_tolerance=1.0e-6,
            maximum_waypoint_cycles=3,
            maximum_relation_verify_cycles=2,
        ),
    )
    recovery.start((unlink_goal,))
    current = demonstrations[0].ee_pose[10]
    for _ in range(20):
        result = recovery.update(
            current_pose=current,
            current_gripper=np.asarray([-1.0]),
            frame_poses={"object": demonstrations[0].frames["object"][10]},
            relation_estimates={"object": relation(RelationDecision.LINKED)},
        )
        if result.action is not None:
            assert np.all(result.action.gripper_command == 1.0)
            current = result.action.pose
            if result.goal_phase is not None and result.goal_phase.value == "verify":
                for _ in range(2):
                    result = recovery.update(
                        current_pose=current,
                        current_gripper=np.asarray([1.0]),
                        frame_poses={"object": demonstrations[0].frames["object"][10]},
                        relation_estimates={
                            "object": relation(RelationDecision.EXTERNAL)
                        },
                    )
        if result.phase != RecoveryPhase.EXECUTING:
            break
    assert result.phase == RecoveryPhase.REENTRY
    assert set(result.legal_reentry_states) == set(unlink_goal.legal_reentry_states)


def test_recovery_relation_goal_finishes_before_unnecessary_anchor_timeout(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    goal = RelationGoalPlanner(registry, UnlinkMetadataRepository(model)).plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=StateId(1, 2),
        mode=0,
    )[0]
    recovery = RelationRecoveryController(
        registry,
        RecoveryConfig(
            pose_position_tolerance=1.0e-6,
            maximum_waypoint_cycles=1,
            maximum_attempts_per_goal=1,
        ),
    )
    recovery.start((goal,))

    result = recovery.update(
        current_pose=pose(-10.0),
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": demonstrations[0].frames["object"][6]},
        relation_estimates={"object": relation(RelationDecision.LINKED)},
    )
    assert result.phase == RecoveryPhase.EXECUTING
    result = recovery.update(
        current_pose=pose(-10.0),
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": demonstrations[0].frames["object"][6]},
        relation_estimates={"object": relation(RelationDecision.LINKED)},
    )

    assert result.phase == RecoveryPhase.REENTRY
    assert result.failure is None
    assert result.action is None
    assert result.completed_goals == (goal,)


def test_recovery_does_not_complete_link_goal_from_one_informative_pulse(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    goal = RelationGoalPlanner(registry, UnlinkMetadataRepository(model)).plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=StateId(1, 2),
        mode=0,
    )[0]
    recovery = RelationRecoveryController(registry)
    recovery.start((goal,))
    frame = demonstrations[0].frames["object"][6]
    current = demonstrations[0].ee_pose[4]

    pulse = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.LINKED)},
    )
    assert pulse.phase == RecoveryPhase.EXECUTING
    if pulse.action is not None:
        current = pulse.action.pose
    rejected = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
    )
    assert rejected.phase == RecoveryPhase.EXECUTING
    assert rejected.completed_goals == ()


def test_link_recovery_actively_probes_after_a_static_final_grasp(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    goal = RelationGoalPlanner(registry, UnlinkMetadataRepository(model)).plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=StateId(1, 2),
        mode=0,
    )[0]
    recovery = RelationRecoveryController(
        registry,
        RecoveryConfig(
            pose_position_tolerance=1.0e-6,
            maximum_waypoint_cycles=3,
        ),
        RelationVerificationConfig(
            minimum_probe_motion=5.0e-4,
            return_position_tolerance=1.0e-6,
        ),
    )
    recovery.start((goal,))
    frame = demonstrations[0].frames["object"][6]
    current = demonstrations[0].ee_pose[4]

    probe = None
    for _ in range(40):
        result = recovery.update(
            current_pose=current,
            current_gripper=np.asarray([-1.0]),
            frame_poses={"object": frame},
            relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
        )
        if result.action is not None:
            current = result.action.pose
        if result.action is not None and result.action.source == "recovery_link_probe":
            probe = result
            break

    assert probe is not None
    assert probe.link_probe_phase == "probe"
    assert np.all(probe.action.gripper_command == -1.0)
    confirming = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.LINKED)},
    )
    assert confirming.link_probe_phase == "probe"
    assert confirming.action is not None
    current = confirming.action.pose
    returning = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.LINKED)},
    )
    assert returning.link_probe_phase == "return"
    assert returning.link_probe_exit_reason == ProbeExitReason.STABLE_RELATION
    assert returning.action is not None
    assert returning.action.source == "recovery_link_return"
    assert np.all(returning.action.gripper_command == -1.0)

    current = returning.action.pose
    complete = returning
    for _ in range(20):
        complete = recovery.update(
            current_pose=current,
            current_gripper=np.asarray([-1.0]),
            frame_poses={"object": frame},
            relation_estimates={"object": relation(RelationDecision.LINKED)},
        )
        if complete.action is not None:
            current = complete.action.pose
        if complete.phase != RecoveryPhase.EXECUTING:
            break
    assert complete.phase == RecoveryPhase.REENTRY
    assert complete.link_probe_motion >= 5.0e-4


def test_recovery_failure_has_hard_attempt_limit(phase5_case) -> None:
    model, demonstrations, _ = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    goal = RelationGoalPlanner(registry, UnlinkMetadataRepository(model)).plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=StateId(1, 2),
        mode=0,
    )[0]
    recovery = __import__(
        "essay2608.policy.closed_loop", fromlist=["RelationRecoveryController"]
    ).RelationRecoveryController(
        registry,
        RecoveryConfig(
            maximum_waypoint_cycles=1,
            maximum_attempts_per_goal=1,
            link_frame_stability_confirmation_cycles=1,
        ),
    )
    recovery.start((goal,))
    frame = demonstrations[0].frames["object"][6]
    far = pose(-10.0)
    result = recovery.update(
        current_pose=far,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
    )
    result = recovery.update(
        current_pose=far,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
    )
    assert result.phase == RecoveryPhase.FAILED
    assert result.failure is not None
    assert result.failure.reason == "link_waypoint_timeout"


def test_link_recovery_waits_for_and_locks_a_stable_object_frame(phase5_case) -> None:
    model, demonstrations, _ = phase5_case
    registry = EpisodeLinkAnchorRegistry(model)
    goal = RelationGoalPlanner(registry, UnlinkMetadataRepository(model)).plan(
        (
            RelationRecoveryIntent(
                "single",
                "object",
                RelationDecision.LINKED,
                RelationDecision.EXTERNAL,
            ),
        ),
        source_state=StateId(1, 2),
        mode=0,
    )[0]
    recovery = RelationRecoveryController(
        registry,
        RecoveryConfig(
            link_frame_stability_confirmation_cycles=2,
            link_frame_stability_translation_tolerance=1.0e-4,
            link_frame_restart_translation=0.01,
        ),
    )
    recovery.start((goal,))
    current = demonstrations[0].ee_pose[4]
    frame = demonstrations[0].frames["object"][6].copy()

    first = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([-1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
    )
    assert first.action is not None
    assert first.action.source == "recovery_link_settle"
    np.testing.assert_allclose(first.action.pose, current)
    assert np.all(first.action.gripper_command == 1.0)

    locked = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([1.0]),
        frame_poses={"object": frame},
        relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
    )
    assert locked.action is not None
    assert locked.action.source == "recovery_link_anchor"

    moved_frame = frame.copy()
    moved_frame[0] += 0.02
    restarted = recovery.update(
        current_pose=current,
        current_gripper=np.asarray([1.0]),
        frame_poses={"object": moved_frame},
        relation_estimates={"object": relation(RelationDecision.EXTERNAL)},
    )
    assert restarted.action is not None
    assert restarted.action.source == "recovery_link_settle"
    np.testing.assert_allclose(restarted.action.pose, current)


def test_reentry_uses_full_state_and_requires_cross_skill_guard(phase5_case) -> None:
    model, demonstrations, _ = phase5_case
    candidate = StateId(2, 2)
    belief = belief_for(
        model,
        demonstrations,
        candidate,
        relation(RelationDecision.EXTERNAL),
    )
    selector = ReentrySelector(
        model,
        ReentryConfig(minimum_explanation_score=0.0),
    )
    blocked = selector.select(
        (candidate,),
        belief,
        current_reference=StateId(1, 3),
        mode_by_skill={1: 0, 2: 0},
    )
    assert blocked.decision is None
    assert "cross_skill_guard_not_permitted" in blocked.rejection_reasons[candidate]
    boundary = BoundaryId("single", 1, 2)
    allowed = selector.select(
        (candidate,),
        belief,
        current_reference=StateId(1, 3),
        permitted_boundaries=frozenset({boundary}),
        mode_by_skill={1: 0, 2: 0},
    )
    assert allowed.decision is not None
    assert allowed.decision.state_id == candidate
    updater = BeliefUpdater(model)
    execution = ClosedLoopExecutionController(model)
    current_observation = observation_for(model, demonstrations, candidate, 9)
    selector.apply(
        allowed.decision,
        belief=belief,
        observation=current_observation,
        belief_updater=updater,
        execution_controller=execution,
    )
    assert execution.cursor.reference_state == candidate
    frozen = updater.update_frozen(
        replace(current_observation, tick=10), mode_by_skill={2: 0}
    )
    assert frozen.progress.posterior == {candidate: 1.0}


def test_reentry_reuses_recovery_covariance_without_changing_normal_scores(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    state = next(iter(model.link_anchors.values())).linked_entry_states[0]
    belief = belief_for(
        model,
        demonstrations,
        state,
        relation(RelationDecision.LINKED),
    )
    relative_poses = {
        frame: value.copy()
        for frame, value in belief.runtime_features.relative_poses.items()
    }
    for frame in model.state(state).selected_frames:
        if frame in relative_poses:
            relative_poses[frame][0] += 0.006
    displaced = replace(
        belief,
        runtime_features=replace(
            belief.runtime_features,
            relative_poses=relative_poses,
        ),
    )
    config = ReentryConfig(
        minimum_explanation_score=0.0,
        minimum_robot_compatibility=0.001,
    )
    strict = ReentrySelector(model, config).select(
        (state,),
        displaced,
        current_reference=state,
        mode_by_skill={state.skill_index: 0},
    )
    recovery_inflation = 1.0e-4
    widened = ReentrySelector(
        model,
        config,
        robot_covariance_inflation=recovery_inflation,
    ).select(
        (state,),
        displaced,
        current_reference=state,
        mode_by_skill={state.skill_index: 0},
    )

    assert strict.decision is None
    assert strict.alignment_state == state
    assert strict.scores[state].robot_covariance_inflation == 0.0
    assert "robot_incompatible" in strict.rejection_reasons[state]
    assert widened.decision is not None
    assert widened.alignment_state is None
    assert widened.decision.state_id == state
    assert widened.scores[state].robot_covariance_inflation == recovery_inflation
    assert (
        widened.scores[state].robot_compatibility
        > strict.scores[state].robot_compatibility
    )
    manager = ClosedLoopRecoveryManager(
        model,
        ClosedLoopRecoveryConfig(
            recovery=RecoveryConfig(covariance_inflation=recovery_inflation),
            reentry=config,
        ),
    )
    assert manager.reentry.robot_covariance_inflation == recovery_inflation


def test_reentry_alignment_routes_with_candidate_relation_without_mutating_beta(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    candidate = StateId(1, 0)
    frozen = StateId(1, 2)
    assert model.state(candidate).demo_relation_priors["object"][0, 0] > 0.5
    assert model.state(frozen).demo_relation_priors["object"][0, 1] > 0.5

    frozen_belief = belief_for(
        model,
        demonstrations,
        frozen,
        relation(RelationDecision.EXTERNAL),
    )
    belief = replace(
        frozen_belief,
        runtime_features=features_for(model, demonstrations, candidate),
    )
    controller = ClosedLoopExecutionController(model)
    mode_by_skill = {1: 0}

    # Ordinary TASK routing intentionally keeps using the real frozen beta,
    # so its expected relation remains linked.  The learned direct-successor
    # UNLINK allowance can make that ordinary route non-blocking; the
    # important distinction here is which state's relation expectation owns
    # the read-only recovery-alignment action.
    ordinary = controller._route_roles(
        candidate,
        belief,
        mode_by_skill=mode_by_skill,
        commit=False,
    )
    assert (
        ordinary.decisions["object"].expected_relation
        == RelationDecision.LINKED
    )

    calls = []
    expected = SimpleNamespace(action=object())

    def query(observation, state_id, roles, *, mode_index=None):
        calls.append((observation, state_id, roles, mode_index))
        return expected

    controller.weighted_poe = SimpleNamespace(query=query)
    observation = DynaMACObservation(
        demonstrations[0].ee_pose[4],
        {"object": demonstrations[0].frames["object"][4]},
    )
    result = controller.query_reentry_alignment(
        candidate,
        belief,
        observation,
        mode_by_skill=mode_by_skill,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0][1] == candidate
    assert calls[0][2].blocks_advance is False
    assert (
        calls[0][2].decisions["object"].expected_relation
        == RelationDecision.EXTERNAL
    )
    assert belief.progress.posterior == {frozen: 1.0}
    assert belief.progress.estimated_state == frozen


def test_reentry_relation_threshold_uses_soft_peak_scale_and_physical_direction(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    state = next(iter(model.link_anchors.values())).linked_entry_states[0]
    prior = model.state(state).demo_relation_priors["object"][0]
    assert prior[1] > prior[0]

    matching_belief = belief_for(
        model,
        demonstrations,
        state,
        relation(RelationDecision.LINKED, posterior=(0.46, 0.54)),
    )
    selector = ReentrySelector(
        model,
        ReentryConfig(
            minimum_explanation_score=0.0,
            minimum_relation_compatibility=0.6,
        ),
        robot_covariance_inflation=1.0e-4,
    )
    matching = selector.select(
        (state,),
        matching_belief,
        current_reference=state,
        mode_by_skill={state.skill_index: 0},
    )
    assert matching.scores[state].relation_compatibility < 0.6
    assert matching.scores[state].relation_state_compatibility > 0.6
    assert matching.decision is not None

    opposing_belief = replace(
        matching_belief,
        relation_estimates={
            "object": relation(
                RelationDecision.EXTERNAL,
                posterior=(0.54, 0.46),
            )
        },
    )
    opposing = selector.select(
        (state,),
        opposing_belief,
        current_reference=state,
        mode_by_skill={state.skill_index: 0},
    )
    assert opposing.scores[state].relation_peak_normalized_compatibility > 0.6
    assert opposing.scores[state].relation_state_compatibility == 0.0
    assert opposing.decision is None
    assert "relation_incompatible" in opposing.rejection_reasons[state]


def test_manager_modes_pending_activation_and_persistent_recovery_trigger(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    manager = ClosedLoopRecoveryManager(
        model,
        ClosedLoopRecoveryConfig(
            verification=RelationVerificationConfig(
                probe_speed=0.1,
                minimum_probe_motion=0.001,
            )
        ),
    )
    entry = pose(0.02)
    manager.record_task_pose(pose(0.0))
    manager.record_task_pose(entry)
    unknown = relation(RelationDecision.UNKNOWN, information=0.0)
    belief = belief_for(
        model,
        demonstrations,
        pending.candidate_state,
        unknown,
        motion=0.0,
    )
    belief = replace(
        belief,
        runtime_features=replace(
            belief.runtime_features,
            frame_pair_available={"object": True},
            paired_tracking_reliability={"object": 1.0},
            relation_information_weight={"object": 0.0},
        ),
    )
    request = RelationVerificationRequest(
        "single", "object", "linked", pending.event_id
    )
    manager.begin_verification(
        request,
        belief,
        task_state=pending.candidate_state,
        grasp_event=0,
        current_pose=entry,
        current_gripper=np.asarray([-1.0]),
    )
    assert manager.mode == ExecutionMode.VERIFY_LINK
    first = manager.update_verification(belief, current_pose=entry)
    assert first.verification.action is not None
    linked_belief = replace(
        belief,
        relation_estimates={"object": relation(RelationDecision.LINKED)},
        runtime_features=verification_response_features(
            belief.runtime_features,
            ee_translation=-0.005,
            frame_translation=-0.005,
        ),
    )
    step = first
    while manager.mode == ExecutionMode.VERIFY_LINK:
        assert step.verification is not None
        assert step.verification.action is not None
        step = manager.update_verification(
            linked_belief,
            current_pose=step.verification.action.pose,
        )
    complete = step
    assert complete.mode == ExecutionMode.TASK
    assert "object" in manager.anchor_registry.active_pending

    # A still-emitted request after returning to TASK is a normal no-op in
    # the unchanged event context, not a second verification or an exception.
    external_belief = replace(
        belief,
        relation_estimates={
            "object": relation(RelationDecision.EXTERNAL, information=0.0)
        },
    )
    second_manager = ClosedLoopRecoveryManager(
        model,
        ClosedLoopRecoveryConfig(
            verification=RelationVerificationConfig(
                probe_speed=0.1,
                minimum_probe_motion=0.001,
            )
        ),
    )
    second_manager.record_task_pose(pose(0.0))
    second_manager.record_task_pose(entry)
    assert second_manager.can_begin_verification(
        request,
        external_belief,
        task_state=pending.candidate_state,
        grasp_event=0,
    )
    second_manager.begin_verification(
        request,
        external_belief,
        task_state=pending.candidate_state,
        grasp_event=0,
        current_pose=entry,
        current_gripper=np.asarray([-1.0]),
    )
    while second_manager.mode == ExecutionMode.VERIFY_LINK:
        step = second_manager.update_verification(
            external_belief,
            current_pose=(
                entry
                if second_manager.verification.phase == VerificationPhase.RETURN
                else entry
            ),
        )
        if step.verification is not None and step.verification.action is not None:
            entry = step.verification.action.pose
    assert not second_manager.can_begin_verification(
        request,
        external_belief,
        task_state=pending.candidate_state,
        grasp_event=0,
    )
    assert second_manager.can_begin_verification(
        request,
        external_belief,
        task_state=pending.candidate_state,
        grasp_event=1,
    )


def test_pending_verification_waits_when_task_approach_direction_is_unavailable(
    phase5_case,
) -> None:
    model, demonstrations, pending = phase5_case
    manager = ClosedLoopRecoveryManager(model)
    stationary = pose(0.02)
    manager.record_task_pose(stationary)
    manager.record_task_pose(stationary)
    belief = belief_for(
        model,
        demonstrations,
        pending.candidate_state,
        relation(RelationDecision.UNKNOWN, information=0.0),
        motion=0.0,
    )
    belief = replace(
        belief,
        runtime_features=replace(
            belief.runtime_features,
            frame_pair_available={pending.frame_id: True},
            paired_tracking_reliability={pending.frame_id: 1.0},
            relation_information_weight={pending.frame_id: 0.0},
        ),
    )
    request = RelationVerificationRequest(
        pending.arm_id,
        pending.frame_id,
        "linked",
        pending.event_id,
    )

    assert not manager.can_begin_verification(
        request,
        belief,
        task_state=pending.candidate_state,
        grasp_event=0,
    )
    assert manager.mode == ExecutionMode.TASK

    tracker = RecoveryTriggerTracker({"single": model})
    intent = RelationRecoveryIntent(
        "single",
        "object",
        RelationDecision.LINKED,
        RelationDecision.EXTERNAL,
    )
    mismatch = MismatchUpdate(
        counters=MismatchCounters(relation_mismatch=3),
        events=(
            MismatchEvent(
                MismatchKind.RELATION_MISMATCH,
                tick=4,
                state_id=pending.candidate_state,
                consecutive_cycles=3,
                frame_ids=("object",),
                recovery_intents=(intent,),
            ),
        ),
    )
    trigger = tracker.update({"single": mismatch})
    assert trigger.triggered and trigger.intents == (intent,)
    manager.begin_recovery(trigger, source_state=pending.candidate_state, mode=0)
    assert manager.mode == ExecutionMode.RECOVERY
    assert manager.frozen_reference == pending.candidate_state


def test_manager_reentry_atomically_resets_progress_and_reference(
    phase5_case,
) -> None:
    model, demonstrations, _ = phase5_case
    source = StateId(2, 2)
    manager = ClosedLoopRecoveryManager(
        model,
        ClosedLoopRecoveryConfig(reentry=ReentryConfig(minimum_explanation_score=0.0)),
    )
    manager.begin_recovery(
        RecoveryTriggerDecision(True, ("single:no_plausible_state",), ()),
        source_state=source,
        mode=0,
    )
    belief = belief_for(
        model,
        demonstrations,
        source,
        relation(RelationDecision.EXTERNAL),
    )
    observation = observation_for(model, demonstrations, source, 20)
    updater = BeliefUpdater(model)
    execution = ClosedLoopExecutionController(model)
    result = manager.evaluate_reentry(
        belief,
        observation=observation,
        belief_updater=updater,
        execution_controller=execution,
        mode_by_skill={2: 0},
    )
    assert result.mode == ExecutionMode.TASK
    assert result.reentry is not None
    assert execution.cursor.reference_state == result.reentry.state_id
    frozen = updater.update_frozen(
        replace(observation, tick=21),
        mode_by_skill={result.reentry.state_id.skill_index: 0},
    )
    assert frozen.progress.posterior == {result.reentry.state_id: 1.0}


def test_phase5_config_file_matches_code_defaults() -> None:
    loaded = ClosedLoopRecoveryConfig.from_json("configs/closed_loop_recovery.json")
    assert loaded.to_dict() == ClosedLoopRecoveryConfig().to_dict()
    with pytest.raises(ValueError, match="未知分区"):
        ClosedLoopRecoveryConfig.from_mapping({"unexpected": {}})
