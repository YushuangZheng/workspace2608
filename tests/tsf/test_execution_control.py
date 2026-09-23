"""Phase-three acceptance tests for roles, weighted PoE, and task cursor control."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import (
    DynaMAC,
    DynaMACAction,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    GaussianMarginal,
    product_of_experts,
)
from essay2608.policy.tsf import (
    CandidateScore,
    TSFBelief,
    TSFExecutionConfig,
    TSFExecutionController,
    TSFTaskModelBuilder,
    ExecutionDecision,
    FrameRole,
    FrameRoleConfig,
    FrameRoleRouter,
    FrameRoleSnapshot,
    LinkPendingCandidate,
    MismatchConfig,
    MismatchKind,
    MismatchTracker,
    ProgressEstimate,
    ProgressPriorBuilder,
    ProgressStatus,
    RelationDecision,
    RelationEstimate,
    RelationEventId,
    RelationVerificationRequest,
    RuntimeFeatureBuilder,
    RuntimeObservation,
    StateId,
    WeightedPoEResult,
    weighted_product_of_experts,
)
from essay2608.policy.dynamac import pose_compose, pose_inverse


def pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture()
def execution_model():
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
                name=f"execution_{demo_index}",
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
    model = TSFTaskModelBuilder().build(
        policy,
        demonstrations,
        recoverable_frames=("object",),
    )
    # Most tests in this module exercise online role transitions for a
    # physical action stream, independently of the offline LODO screen.  Keep
    # that stream explicitly eligible in this synthetic fixture; dedicated
    # tests below cover action-relevance exclusion.
    for node in model.states.values():
        node.action_relevant_frames = tuple(
            dict.fromkeys((*node.action_relevant_frames, "object"))
        )
        node.mode_action_relevant_frames = tuple(
            tuple(dict.fromkeys((*frames, "object")))
            for frames in node.mode_action_relevant_frames
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
    observation_likelihood: tuple[float, float] = (1.0, 1.0),
    informative: bool | None = None,
    evidence_direction: RelationDecision = RelationDecision.UNKNOWN,
) -> RelationEstimate:
    values = np.asarray(posterior, dtype=np.float64)
    return RelationEstimate(
        frame_id=frame,
        posterior=values,
        predicted=values,
        demonstration_prior=np.asarray([0.5, 0.5]),
        observation_likelihood=np.asarray(observation_likelihood, dtype=np.float64),
        information_weight=information_weight,
        entropy=-float(np.sum(values * np.log(np.maximum(values, 1.0e-12)))),
        informative=(
            decision != RelationDecision.UNKNOWN if informative is None else informative
        ),
        decision_state=decision,
        informative_evidence_direction=evidence_direction,
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
) -> TSFBelief:
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
    return TSFBelief(
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


def candidate_score(
    state: StateId,
    *,
    robot: float,
    scene: float,
    scene_expected: bool,
) -> CandidateScore:
    return CandidateScore(
        state_id=state,
        robot_log_likelihood=0.0,
        state_log_likelihood=0.0,
        robot_log_support=float(np.log(robot)),
        robot_unadjusted_log_support=float(np.log(robot)),
        state_log_support=float(np.log(scene)),
        relation_log_compatibility=float(np.log(0.95)),
        explanation_log_score=float(np.log(robot * scene * 0.95)),
        normalized_explanation_score=robot * scene * 0.95,
        robot_compatibility=robot,
        state_compatibility=scene,
        relation_compatibility=0.95,
        robot_peak_normalized_compatibility=robot,
        relation_peak_normalized_compatibility=0.95,
        relation_state_compatibility=0.95,
        robot_evidence_available=True,
        scene_evidence_expected=scene_expected,
        scene_evidence_available=scene_expected,
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


def test_dynamic_roles_execute_monitor_recover_and_defer(execution_model) -> None:
    model, demos = execution_model
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
    # A stable role decision authorizes the original expert precision.  The
    # same q(external) evidence must not attenuate it a second time.
    assert execute.execution_weights["object"] == pytest.approx(1.0)

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


def test_action_relevance_only_changes_normal_poe_participation(execution_model) -> None:
    model, demos = execution_model
    state = StateId(0, 1)
    force_relation_prior(model, state, linked=False)
    node = model.state(state)
    assert "object" in node.selected_frames
    node.action_relevant_frames = tuple(
        frame for frame in node.action_relevant_frames if frame != "object"
    )
    node.mode_action_relevant_frames = tuple(
        tuple(frame for frame in frames if frame != "object")
        for frames in node.mode_action_relevant_frames
    )
    external = belief_for(
        model,
        demos,
        tick=1,
        nominal=state,
        estimated=state,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )

    dynamic = FrameRoleRouter(model).route(state, external, mode_by_skill={0: 0})
    assert dynamic.decisions["object"].role == FrameRole.DEFER
    assert dynamic.decisions["object"].execution_weight == 0.0
    assert "object" not in dynamic.execution_weights

    # The progress-only ablation keeps the frozen base-policy candidate mask.
    fixed = FrameRoleRouter(model).route_fixed(state, external, mode_by_skill={0: 0})
    assert fixed.decisions["object"].role == FrameRole.EXECUTE
    assert fixed.execution_weights["object"] == 1.0

    # Action exclusion does not erase relation semantics or recovery.
    force_relation_prior(model, state, linked=True)
    mismatch = FrameRoleRouter(model).route(state, external, mode_by_skill={0: 0})
    assert mismatch.decisions["object"].role == FrameRole.RECOVER
    assert mismatch.decisions["object"].blocks_advance is True


def test_fresh_external_unknown_bootstraps_below_unreachable_soft_prior_peak(
    execution_model,
) -> None:
    model, demos = execution_model
    state = StateId(0, 1)
    force_relation_prior(model, state, linked=False)
    router = FrameRoleRouter(model)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=state,
        estimated=state,
        relation=relation_estimate(
            "object",
            (0.692, 0.308),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )

    routed = router.route(state, belief, mode_by_skill={0: 0})

    decision = routed.decisions["object"]
    assert decision.role == FrameRole.DEFER
    assert decision.actual_relation == RelationDecision.UNKNOWN
    assert decision.execution_weight == pytest.approx(0.692)
    assert decision.blocks_advance is False


def test_unselected_external_unknown_is_soft_evidence_not_a_progress_lock(
    execution_model,
) -> None:
    model, demos = execution_model
    state = StateId(0, 1)
    force_relation_prior(model, state, linked=False)
    node = model.state(state)
    node.action_relevant_frames = tuple(
        frame for frame in node.action_relevant_frames if frame != "object"
    )
    node.mode_action_relevant_frames = tuple(
        tuple(frame for frame in frames if frame != "object")
        for frames in node.mode_action_relevant_frames
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=state,
        estimated=state,
        relation=relation_estimate(
            "object",
            (0.692, 0.308),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )

    routed = FrameRoleRouter(model).route(state, belief, mode_by_skill={0: 0})

    decision = routed.decisions["object"]
    assert decision.selected_offline is False
    assert decision.role == FrameRole.DEFER
    assert decision.execution_weight == 0.0
    assert decision.blocks_advance is False
    assert routed.blocks_advance is False


def test_confirmed_unselected_link_is_monitored_without_reenabling_poe(
    execution_model,
) -> None:
    model, demos = execution_model
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
    node.action_relevant_frames = tuple(
        frame for frame in node.action_relevant_frames if frame != "object"
    )
    node.mode_action_relevant_frames = tuple(
        tuple(frame for frame in frames if frame != "object")
        for frames in node.mode_action_relevant_frames
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


@pytest.mark.parametrize(
    "actual",
    (RelationDecision.UNKNOWN, RelationDecision.EXTERNAL),
)
def test_confirmed_unlink_direct_successor_can_issue_causal_opening(
    execution_model,
    actual: RelationDecision,
) -> None:
    model, demos = execution_model
    event_id, metadata = next(iter(sorted(model.unlink_events.items())))
    predecessor = model.state(metadata.release_state).topology.predecessors[0]
    assert metadata.release_state in model.state(predecessor).topology.successors
    assert any(
        key.frame_id == "object" and key.state_id == predecessor
        for key in model.link_origins
    )
    force_relation_prior(model, predecessor, linked=True)
    posterior = (0.5, 0.5) if actual == RelationDecision.UNKNOWN else (0.95, 0.05)
    snapshot = FrameRoleRouter(model).route(
        predecessor,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=predecessor,
            estimated=predecessor,
            relation=relation_estimate(
                "object",
                posterior,
                actual,
                information_weight=(0.0 if actual == RelationDecision.UNKNOWN else 0.4),
            ),
            static=actual == RelationDecision.UNKNOWN,
        ),
        mode_by_skill={predecessor.skill_index: event_id.mode},
    )
    decision = snapshot.decisions["object"]
    assert decision.role == FrameRole.DEFER
    assert decision.blocks_advance is False
    assert snapshot.recovery_intents == ()

    # The allowance is causal and local.  A reliable early disconnect that is
    # not immediately followed by this learned UNLINK remains a recovery.
    earlier = model.state(predecessor).topology.predecessors[0]
    force_relation_prior(model, earlier, linked=True)
    early = FrameRoleRouter(model).route(
        earlier,
        belief_for(
            model,
            demos,
            tick=2,
            nominal=earlier,
            estimated=earlier,
            relation=relation_estimate(
                "object",
                (0.95, 0.05),
                RelationDecision.EXTERNAL,
            ),
        ),
        mode_by_skill={earlier.skill_index: event_id.mode},
    )
    assert early.decisions["object"].role == FrameRole.RECOVER
    assert early.blocks_advance is True


@pytest.mark.parametrize(
    "actual",
    (RelationDecision.UNKNOWN, RelationDecision.LINKED),
)
def test_confirmed_unlink_release_state_can_issue_detachment_successor(
    execution_model,
    actual: RelationDecision,
) -> None:
    model, demos = execution_model
    event_id, metadata = next(iter(sorted(model.unlink_events.items())))
    release = metadata.release_state
    force_relation_prior(model, release, linked=False)
    posterior = (0.5, 0.5) if actual == RelationDecision.UNKNOWN else (0.05, 0.95)
    router = FrameRoleRouter(model)
    snapshot = router.route(
        release,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=release,
            estimated=release,
            relation=relation_estimate(
                "object",
                posterior,
                actual,
                information_weight=(0.0 if actual == RelationDecision.UNKNOWN else 0.4),
            ),
            static=actual == RelationDecision.UNKNOWN,
        ),
        mode_by_skill={release.skill_index: event_id.mode},
    )
    decision = snapshot.decisions["object"]
    assert decision.role == FrameRole.EXECUTE
    assert decision.execution_weight > 0.0
    assert decision.monitor is True
    assert decision.blocks_advance is False
    assert snapshot.recovery_intents == ()

    # The learned post-release interval supplies the observable detachment
    # samples.  A stale linked decision cannot suppress that normal motion.
    router.commit(snapshot)
    successor = model.state(release).topology.successors[0]
    force_relation_prior(model, successor, linked=False)
    causal_successor = router.route(
        successor,
        belief_for(
            model,
            demos,
            tick=2,
            nominal=successor,
            estimated=successor,
            relation=relation_estimate(
                "object",
                (0.05, 0.95),
                RelationDecision.LINKED,
            ),
        ),
        mode_by_skill={successor.skill_index: event_id.mode},
    )
    assert causal_successor.decisions["object"].role == FrameRole.EXECUTE
    assert causal_successor.decisions["object"].execution_weight > 0.0
    assert causal_successor.blocks_advance is False
    assert causal_successor.recovery_intents == ()

    # The allowance ends with the finite interval supported by all normal
    # demonstrations; it does not disable genuine failed-UNLINK recovery.
    router.commit(causal_successor)
    legal = set(metadata.legal_reentry_states)
    outside = next(
        state
        for state in sorted(model.states)
        if state not in legal
        and metadata.release_state
        not in model.state(state).topology.successors
    )
    force_relation_prior(model, outside, linked=False)
    persistent = router.route(
        outside,
        belief_for(
            model,
            demos,
            tick=3,
            nominal=outside,
            estimated=outside,
            relation=relation_estimate(
                "object",
                (0.05, 0.95),
                RelationDecision.LINKED,
            ),
        ),
        mode_by_skill={outside.skill_index: event_id.mode},
    )
    assert persistent.decisions["object"].role == FrameRole.RECOVER
    assert persistent.blocks_advance is True


@pytest.mark.parametrize(
    "actual",
    [RelationDecision.UNKNOWN, RelationDecision.EXTERNAL],
)
def test_confirmed_link_direct_successor_can_issue_causal_grasp(
    execution_model,
    actual: RelationDecision,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    first_linked = anchor.linked_entry_states[0]
    predecessor = model.state(first_linked).topology.predecessors[0]
    assert first_linked in model.state(predecessor).topology.successors
    force_relation_prior(model, predecessor, linked=True)
    posterior = (0.5, 0.5) if actual == RelationDecision.UNKNOWN else (0.95, 0.05)
    snapshot = FrameRoleRouter(model).route(
        predecessor,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=predecessor,
            estimated=predecessor,
            relation=relation_estimate(
                "object",
                posterior,
                actual,
                information_weight=(0.0 if actual == RelationDecision.UNKNOWN else 0.4),
                observation_likelihood=(1.0, 0.01),
            ),
            static=actual == RelationDecision.UNKNOWN,
        ),
        mode_by_skill={predecessor.skill_index: event_id.mode},
    )
    decision = snapshot.decisions["object"]
    assert decision.role == FrameRole.DEFER
    assert decision.blocks_advance is False
    assert snapshot.recovery_intents == ()

    earlier = model.state(predecessor).topology.predecessors[0]
    force_relation_prior(model, earlier, linked=True)
    early = FrameRoleRouter(model).route(
        earlier,
        belief_for(
            model,
            demos,
            tick=2,
            nominal=earlier,
            estimated=earlier,
            relation=relation_estimate(
                "object",
                (0.95, 0.05),
                RelationDecision.EXTERNAL,
            ),
        ),
        mode_by_skill={earlier.skill_index: event_id.mode},
    )
    assert early.decisions["object"].role == FrameRole.RECOVER
    assert early.blocks_advance is True


def test_formal_link_uses_learned_interval_motion_to_confirm_posterior_lag(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    state_id = anchor.linked_entry_states[0]
    mode = event_id.mode
    router = FrameRoleRouter(model)
    predecessor = model.state(state_id).topology.predecessors[0]
    router.route(
        predecessor,
        belief_for(
            model,
            demos,
            tick=0,
            nominal=predecessor,
            estimated=predecessor,
            relation=relation_estimate(
                "object",
                (0.5, 0.5),
                RelationDecision.UNKNOWN,
                information_weight=0.0,
            ),
            static=True,
        ),
        mode_by_skill={predecessor.skill_index: mode},
    )
    lagging = belief_for(
        model,
        demos,
        tick=1,
        nominal=state_id,
        estimated=state_id,
        relation=relation_estimate(
            "object",
            (0.90, 0.10),
            RelationDecision.EXTERNAL,
            information_weight=0.4,
            observation_likelihood=(0.01, 1.0),
        ),
    )

    snapshot = router.route(
        state_id,
        lagging,
        mode_by_skill={state_id.skill_index: mode},
    )
    decision = snapshot.decisions["object"]
    assert decision.role == FrameRole.DEFER
    assert decision.formal_link_confirmation_pending is True
    assert decision.execution_weight == 0.0
    assert decision.blocks_advance is False
    assert snapshot.recovery_intents == ()


def test_formal_link_confirmation_accepts_legal_skip_over_interval_entry(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    assert len(anchor.linked_entry_states) >= 2
    first = anchor.linked_entry_states[0]
    queried = anchor.linked_entry_states[1]
    predecessor = model.state(first).topology.predecessors[0]
    assert queried in model.state(first).topology.successors
    router = FrameRoleRouter(model)
    modes = {
        predecessor.skill_index: event_id.mode,
        queried.skill_index: event_id.mode,
    }
    router.route(
        predecessor,
        belief_for(
            model,
            demos,
            tick=0,
            nominal=predecessor,
            estimated=predecessor,
            relation=relation_estimate(
                "object",
                (0.95, 0.05),
                RelationDecision.EXTERNAL,
                information_weight=0.0,
            ),
            static=True,
        ),
        mode_by_skill=modes,
    )

    snapshot = router.route(
        queried,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=queried,
            estimated=first,
            posterior={first: 1.0},
            relation=relation_estimate(
                "object",
                (0.90, 0.10),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill=modes,
    )

    decision = snapshot.decisions["object"]
    assert decision.formal_link_confirmation_pending is True
    assert decision.role == FrameRole.DEFER
    assert decision.blocks_advance is False
    assert snapshot.recovery_intents == ()


def test_off_selected_formal_link_entry_is_monitored_before_posterior_catches_up(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    state_id = anchor.linked_entry_states[0]
    predecessor = model.state(state_id).topology.predecessors[0]
    router = FrameRoleRouter(model)
    monkeypatch.setattr(router, "_candidate_frames", lambda *_args, **_kwargs: ())
    router.route(
        predecessor,
        belief_for(
            model,
            demos,
            tick=0,
            nominal=predecessor,
            estimated=predecessor,
            posterior={predecessor: 1.0},
            relation=relation_estimate(
                "object",
                (0.95, 0.05),
                RelationDecision.EXTERNAL,
                information_weight=0.0,
            ),
            static=True,
        ),
        mode_by_skill={predecessor.skill_index: event_id.mode},
    )

    snapshot = router.route(
        state_id,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=state_id,
            estimated=predecessor,
            posterior={predecessor: 1.0},
            relation=relation_estimate(
                "object",
                (0.95, 0.05),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill={state_id.skill_index: event_id.mode},
    )

    decision = snapshot.decisions["object"]
    assert decision.selected_offline is False
    assert decision.formal_link_confirmation_pending is True
    assert decision.role == FrameRole.DEFER
    assert decision.blocks_advance is False


def test_formal_link_confirmation_requires_causal_entry_and_never_masks_drop(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    state_id = anchor.linked_entry_states[0]
    modes = {state_id.skill_index: event_id.mode}

    contradicted = FrameRoleRouter(model).route(
        state_id,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=state_id,
            estimated=state_id,
            relation=relation_estimate(
                "object",
                (0.90, 0.10),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill=modes,
    )
    assert contradicted.decisions["object"].role == FrameRole.RECOVER
    assert contradicted.blocks_advance is True

    router = FrameRoleRouter(model)
    confirmed = belief_for(
        model,
        demos,
        tick=2,
        nominal=state_id,
        estimated=state_id,
        relation=relation_estimate("object", (0.01, 0.99), RelationDecision.LINKED),
    )
    assert (
        router.route(state_id, confirmed, mode_by_skill=modes).blocks_advance is False
    )
    dropped = router.route(
        state_id,
        belief_for(
            model,
            demos,
            tick=3,
            nominal=state_id,
            estimated=state_id,
            relation=relation_estimate(
                "object",
                (0.90, 0.10),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(0.01, 1.0),
            ),
        ),
        mode_by_skill=modes,
    )
    assert dropped.decisions["object"].formal_link_confirmation_pending is False
    assert dropped.decisions["object"].role == FrameRole.RECOVER
    assert dropped.blocks_advance is True


def test_role_commit_can_record_boundary_entry_as_the_causal_state(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    linked_state = anchor.linked_entry_states[0]
    predecessor = model.state(linked_state).topology.predecessors[0]
    source_state = model.state(predecessor).topology.predecessors[0]
    modes = {linked_state.skill_index: event_id.mode}
    router = FrameRoleRouter(model)

    source_belief = belief_for(
        model,
        demos,
        tick=0,
        nominal=source_state,
        estimated=source_state,
        relation=relation_estimate(
            "object",
            (0.95, 0.05),
            RelationDecision.EXTERNAL,
        ),
    )
    source_roles = router.route(
        source_state,
        source_belief,
        mode_by_skill=modes,
        commit=False,
    )
    router.commit(source_roles, source_belief, causal_state=predecessor)

    linked_roles = router.route(
        linked_state,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=linked_state,
            estimated=predecessor,
            posterior={predecessor: 1.0},
            relation=relation_estimate(
                "object",
                (0.95, 0.05),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill=modes,
    )

    assert linked_roles.decisions["object"].formal_link_confirmation_pending is True
    assert linked_roles.decisions["object"].blocks_advance is False


def test_boundary_entry_commit_activates_formal_link_confirmation_lifecycle(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    entry = anchor.linked_entry_states[0]
    predecessor = model.state(entry).topology.predecessors[0]
    modes = {
        predecessor.skill_index: event_id.mode,
        entry.skill_index: event_id.mode,
    }
    router = FrameRoleRouter(model)
    source_belief = belief_for(
        model,
        demos,
        tick=0,
        nominal=predecessor,
        estimated=predecessor,
        posterior={predecessor: 1.0},
        relation=relation_estimate(
            "object",
            (0.95, 0.05),
            RelationDecision.EXTERNAL,
            information_weight=0.0,
        ),
        static=True,
    )
    source_roles = router.route(
        predecessor,
        source_belief,
        mode_by_skill=modes,
        commit=True,
    )

    router.commit_boundary_entry(
        source_roles,
        source_belief,
        entry,
        mode_by_skill=modes,
    )

    next_cycle = router.route(
        entry,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=entry,
            estimated=predecessor,
            posterior={predecessor: 1.0},
            relation=relation_estimate(
                "object",
                (0.90, 0.10),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill=modes,
    )

    decision = next_cycle.decisions["object"]
    assert decision.formal_link_confirmation_pending is True
    assert decision.role == FrameRole.DEFER
    assert decision.blocks_advance is False
    assert next_cycle.recovery_intents == ()


def test_formal_link_rejection_persists_across_later_low_excitation_cycle(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    state_id = anchor.linked_entry_states[0]
    modes = {state_id.skill_index: event_id.mode}
    router = FrameRoleRouter(model)

    contradicted = router.route(
        state_id,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=state_id,
            estimated=state_id,
            relation=relation_estimate(
                "object",
                (0.90, 0.10),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill=modes,
    )
    assert event_id in contradicted.rejected_link_events
    assert contradicted.decisions["object"].role == FrameRole.RECOVER

    unexcited = router.route(
        state_id,
        belief_for(
            model,
            demos,
            tick=2,
            nominal=state_id,
            estimated=state_id,
            relation=relation_estimate(
                "object",
                (0.90, 0.10),
                RelationDecision.EXTERNAL,
                information_weight=0.0,
                observation_likelihood=(0.5, 0.5),
            ),
            static=True,
        ),
        mode_by_skill=modes,
    )
    decision = unexcited.decisions["object"]
    assert decision.formal_link_confirmation_pending is False
    assert decision.role == FrameRole.RECOVER
    assert unexcited.blocks_advance is True


def test_formal_link_natural_confirmation_uses_full_learned_interval(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    assert len(anchor.linked_entry_states) >= 2
    state_id = anchor.linked_entry_states[-1]
    router = FrameRoleRouter(model)
    first = anchor.linked_entry_states[0]
    predecessor = model.state(first).topology.predecessors[0]
    route_states = (predecessor, *anchor.linked_entry_states)
    snapshot = None
    for tick, route_state in enumerate(route_states):
        snapshot = router.route(
            route_state,
            belief_for(
                model,
                demos,
                tick=tick,
                nominal=route_state,
                estimated=route_state,
                relation=relation_estimate(
                    "object",
                    (0.5, 0.5),
                    RelationDecision.UNKNOWN,
                    information_weight=0.0,
                ),
                static=True,
            ),
            mode_by_skill={route_state.skill_index: event_id.mode},
        )
    assert snapshot is not None
    assert snapshot.state_id == state_id
    assert snapshot.decisions["object"].formal_link_confirmation_pending is True
    assert snapshot.decisions["object"].role == FrameRole.DEFER
    assert snapshot.blocks_advance is False


def test_formal_link_confirmation_interval_expires_after_terminal_entry_action(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    router = FrameRoleRouter(model)
    first = anchor.linked_entry_states[0]
    predecessor = model.state(first).topology.predecessors[0]
    route_states = (predecessor, *anchor.linked_entry_states)
    snapshot = None
    for tick, route_state in enumerate(route_states):
        snapshot = router.route(
            route_state,
            belief_for(
                model,
                demos,
                tick=tick,
                nominal=route_state,
                estimated=route_state,
                relation=relation_estimate(
                    "object",
                    (0.9, 0.1),
                    RelationDecision.EXTERNAL,
                    information_weight=0.4,
                    observation_likelihood=(0.01, 1.0),
                ),
            ),
            mode_by_skill={route_state.skill_index: event_id.mode},
        )
    assert snapshot is not None
    assert snapshot.decisions["object"].formal_link_confirmation_pending is True

    terminal = anchor.linked_entry_states[-1]
    predecessor_of_terminal = anchor.linked_entry_states[-2]
    lagging = router.route(
        terminal,
        belief_for(
            model,
            demos,
            tick=len(route_states),
            nominal=terminal,
            estimated=predecessor_of_terminal,
            posterior={predecessor_of_terminal: 1.0},
            relation=relation_estimate(
                "object",
                (0.9, 0.1),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill={terminal.skill_index: event_id.mode},
    )
    assert lagging.decisions["object"].formal_link_confirmation_pending is True

    expired = router.route(
        terminal,
        belief_for(
            model,
            demos,
            tick=len(route_states) + 1,
            nominal=terminal,
            estimated=terminal,
            relation=relation_estimate(
                "object",
                (0.9, 0.1),
                RelationDecision.EXTERNAL,
                information_weight=0.4,
                observation_likelihood=(1.0, 0.01),
            ),
        ),
        mode_by_skill={terminal.skill_index: event_id.mode},
    )
    decision = expired.decisions["object"]
    assert decision.formal_link_confirmation_pending is False
    # The finite close-to-evidence interval has now been consumed.  A still
    # reliable external observation contradicts the persistent formal LINK
    # origin and must immediately enter the ordinary recovery path rather than
    # inheriting grace until the later UNLINK.
    assert decision.role == FrameRole.RECOVER
    assert expired.blocks_advance is True
    assert event_id in expired.rejected_link_events


@pytest.mark.parametrize(
    ("decision_state", "posterior"),
    (
        (RelationDecision.UNKNOWN, (0.35, 0.65)),
        # The formal event must not treat a low-information external decision
        # carried over from before the grasp as fresh contradictory evidence.
        (RelationDecision.EXTERNAL, (0.9, 0.1)),
    ),
)
def test_unresolved_formal_link_requests_verification_after_natural_interval(
    execution_model, decision_state, posterior
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    router = FrameRoleRouter(model)
    first = anchor.linked_entry_states[0]
    predecessor = model.state(first).topology.predecessors[0]

    # Causally execute the learned relation-changing interval.  Its response
    # points toward LINK throughout, but the persistent posterior has not yet
    # crossed the stable-decision threshold.
    positive_unresolved = replace(
        relation_estimate(
            "object",
            (0.4, 0.6),
            RelationDecision.UNKNOWN,
            information_weight=0.4,
            observation_likelihood=(0.01, 1.0),
            informative=True,
        ),
        informative_evidence_direction=RelationDecision.LINKED,
    )
    for tick, route_state in enumerate((predecessor, *anchor.linked_entry_states)):
        snapshot = router.route(
            route_state,
            belief_for(
                model,
                demos,
                tick=tick,
                nominal=route_state,
                estimated=route_state,
                relation=positive_unresolved,
            ),
            mode_by_skill={route_state.skill_index: event_id.mode},
        )
    assert event_id in snapshot.unresolved_formal_link_events

    terminal = anchor.linked_entry_states[-1]
    unresolved = router.route(
        terminal,
        belief_for(
            model,
            demos,
            tick=len(anchor.linked_entry_states) + 1,
            nominal=terminal,
            estimated=terminal,
            relation=relation_estimate(
                "object",
                posterior,
                decision_state,
                information_weight=0.0,
            ),
            static=True,
        ),
        mode_by_skill={terminal.skill_index: event_id.mode},
    )

    assert unresolved.decisions["object"].formal_link_confirmation_pending is False
    assert len(unresolved.verification_requests) == 1
    assert unresolved.blocks_advance is False
    assert unresolved.recovery_intents == ()
    request = unresolved.verification_requests[0]
    assert request.event_id == event_id
    assert request.context_state == terminal


def test_indirect_comotion_link_never_becomes_second_recovery_target(
    execution_model,
) -> None:
    model, demos = execution_model
    direct_id, direct_anchor = next(iter(sorted(model.link_anchors.items())))
    payload_id = replace(direct_id, frame_id="payload")
    payload_anchor = replace(
        direct_anchor,
        event_id=payload_id,
        frame_id="payload",
    )
    model.link_anchors[payload_id] = payload_anchor
    for key, origin in tuple(model.link_origins.items()):
        if origin == direct_id:
            model.link_origins[replace(key, frame_id="payload")] = payload_id
    model.relation_frames = tuple((*model.relation_frames, "payload"))
    for node in model.states.values():
        node.demo_relation_scores["payload"] = node.demo_relation_scores[
            "object"
        ].copy()
        node.demo_relation_priors["payload"] = node.demo_relation_priors[
            "object"
        ].copy()

    state_id = direct_anchor.linked_entry_states[-1]
    external = relation_estimate(
        "object",
        (0.95, 0.05),
        RelationDecision.EXTERNAL,
        information_weight=0.5,
        observation_likelihood=(1.0, 0.01),
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=state_id,
        estimated=state_id,
        relation=external,
    )
    belief = replace(
        belief,
        relation_estimates={
            "object": external,
            "payload": replace(external, frame_id="payload"),
        },
    )
    router = FrameRoleRouter(model)
    assert router._indirect_link_events == frozenset({payload_id})
    snapshot = router.route(
        state_id,
        belief,
        mode_by_skill={state_id.skill_index: direct_id.mode},
    )

    assert [intent.frame_id for intent in snapshot.recovery_intents] == ["object"]
    assert snapshot.decisions["object"].role == FrameRole.RECOVER
    assert snapshot.decisions["payload"].role == FrameRole.DEFER
    assert snapshot.decisions["payload"].monitor is True
    assert snapshot.decisions["payload"].blocks_advance is False
    assert all(
        request.event_id != payload_id for request in snapshot.verification_requests
    )


def test_comotion_link_group_with_two_actionable_frames_does_not_guess_root(
    execution_model,
) -> None:
    model, _ = execution_model
    direct_id, direct_anchor = next(iter(sorted(model.link_anchors.items())))
    payload_id = replace(direct_id, frame_id="payload")
    model.link_anchors[payload_id] = replace(
        direct_anchor,
        event_id=payload_id,
        frame_id="payload",
    )
    earliest = min(direct_anchor.linked_entry_states)
    for state_id, node in model.states.items():
        if state_id >= earliest or "object" not in node.action_relevant_frames:
            continue
        node.mode_action_relevant_frames = tuple(
            tuple((*frames, "payload")) if "object" in frames else frames
            for frames in node.mode_action_relevant_frames
        )
        node.action_relevant_frames = tuple(
            dict.fromkeys((*node.action_relevant_frames, "payload"))
        )

    router = FrameRoleRouter(model)
    assert router._indirect_link_events == frozenset()


def test_terminal_link_response_gets_one_bounded_successor_for_confirmation(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    router = FrameRoleRouter(model)
    first = anchor.linked_entry_states[0]
    predecessor = model.state(first).topology.predecessors[0]
    route_states = (predecessor, *anchor.linked_entry_states)
    for tick, route_state in enumerate(route_states):
        router.route(
            route_state,
            belief_for(
                model,
                demos,
                tick=tick,
                nominal=route_state,
                estimated=route_state,
                relation=relation_estimate(
                    "object",
                    (0.9, 0.1),
                    RelationDecision.EXTERNAL,
                    information_weight=0.4,
                    observation_likelihood=(0.01, 1.0),
                ),
            ),
            mode_by_skill={route_state.skill_index: event_id.mode},
        )

    terminal = anchor.linked_entry_states[-1]
    positive_but_unconfirmed = replace(
        relation_estimate(
            "object",
            (0.4, 0.6),
            RelationDecision.UNKNOWN,
            information_weight=0.4,
            observation_likelihood=(0.01, 1.0),
            informative=True,
        ),
        informative_evidence_direction=RelationDecision.LINKED,
    )
    pending = router.route(
        terminal,
        belief_for(
            model,
            demos,
            tick=len(route_states),
            nominal=terminal,
            estimated=terminal,
            relation=positive_but_unconfirmed,
        ),
        mode_by_skill={terminal.skill_index: event_id.mode},
    )
    decision = pending.decisions["object"]
    assert decision.formal_link_confirmation_pending is True
    assert decision.role == FrameRole.DEFER
    assert decision.blocks_advance is False

    successor = model.state(terminal).topology.successors[0]
    assert successor not in anchor.linked_entry_states
    follow_through = router.route(
        successor,
        belief_for(
            model,
            demos,
            tick=len(route_states) + 1,
            nominal=successor,
            estimated=successor,
            relation=positive_but_unconfirmed,
        ),
        mode_by_skill={successor.skill_index: event_id.mode},
    )
    assert follow_through.decisions["object"].formal_link_confirmation_pending is True
    assert follow_through.blocks_advance is False

    # Once that successor is committed, the allowance is consumed even if the
    # same inconclusive evidence is presented again.
    consumed = router.route(
        successor,
        belief_for(
            model,
            demos,
            tick=len(route_states) + 2,
            nominal=successor,
            estimated=successor,
            relation=positive_but_unconfirmed,
        ),
        mode_by_skill={successor.skill_index: event_id.mode},
    )
    assert consumed.decisions["object"].formal_link_confirmation_pending is False


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


def test_controller_advances_after_current_reference_is_reached(execution_model) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=current,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.weighted_action.action is not None
    assert result.weighted_action.action.diagnostics["time_index"] == 1


def test_observation_confirmed_progress_advances_after_rejected_command(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)

    reset_belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=current,
        relation=relation_estimate(
            "object", (0.95, 0.05), RelationDecision.EXTERNAL
        ),
    )
    reset_cycle = controller.update(
        reset_belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=False,
    )
    assert reset_cycle.decision == ExecutionDecision.HOLD
    assert reset_cycle.cursor_after.reference_state == current

    confirmed_belief = belief_for(
        model,
        demos,
        tick=2,
        nominal=current,
        estimated=current,
        relation=relation_estimate(
            "object", (0.95, 0.05), RelationDecision.EXTERNAL
        ),
    )
    confirmed_cycle = controller.update(
        confirmed_belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=False,
    )

    assert confirmed_cycle.decision == ExecutionDecision.ADVANCE
    assert confirmed_cycle.cursor_after.reference_state == successor
    assert confirmed_cycle.reasons == ("current_reference_reached",)


def test_pending_discrete_action_holds_same_reference_without_executor_gate(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    force_relation_prior(model, current, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=current,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        current_discrete_action_committed=False,
        action_executed=True,
    )

    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert "current_discrete_action_pending" in result.reasons


def _install_control_targets(
    monkeypatch,
    controller: TSFExecutionController,
    targets: dict[StateId, tuple[np.ndarray, float]],
) -> None:
    def query(observation, state_id, roles, *, mode_index=None):
        del observation, mode_index
        target, gripper = targets[state_id]
        return WeightedPoEResult(
            state_id=state_id,
            stream_weights=roles.execution_weights,
            participating_frames=tuple(
                frame
                for frame, weight in roles.execution_weights.items()
                if weight > 0.0
            ),
            action=DynaMACAction(
                pose=target.copy(),
                covariance=np.eye(6, dtype=np.float64) * 0.01,
                gripper=np.asarray([gripper], dtype=np.float64),
                diagnostics={"time_index": state_id.local_index},
            ),
        )

    monkeypatch.setattr(controller.weighted_poe, "query", query)


def test_low_confidence_equivalent_controls_aggregate_and_advance(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    for state in (current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {current: (pose(0.0), 1.0), successor: (pose(0.0), 1.0)},
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=current,
        posterior={current: 0.5, successor: 0.5},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.LOW_CONFIDENCE,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert belief.progress.status == ProgressStatus.LOW_CONFIDENCE
    assert result.control_equivalence.accepted is True
    assert result.control_equivalence.equivalent_states == (current, successor)
    assert result.control_equivalence.aggregated_confidence == pytest.approx(1.0)
    assert result.control_equivalence.normalized_class_entropy == pytest.approx(0.0)
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == (
        "control_equivalent_progress_uncertainty",
        "current_reference_reached",
    )


def test_control_equivalence_ablation_disables_aggregation(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    for state in (current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(
        model,
        control_equivalence_aggregation=False,
    )
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {current: (pose(0.0), 1.0), successor: (pose(0.0), 1.0)},
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=current,
        posterior={current: 0.5, successor: 0.5},
        relation=relation_estimate(
            "object",
            (0.95, 0.05),
            RelationDecision.EXTERNAL,
        ),
        status=ProgressStatus.LOW_CONFIDENCE,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.evaluated is False
    assert result.control_equivalence.accepted is False
    assert result.control_equivalence.reason == "control_equivalence_disabled"


@pytest.mark.parametrize(
    ("successor_pose", "successor_gripper"),
    ((pose(0.5), 1.0), (pose(0.0), -1.0)),
)
def test_low_confidence_control_relevant_ambiguity_still_holds(
    execution_model,
    monkeypatch,
    successor_pose,
    successor_gripper,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    for state in (current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            current: (pose(0.0), 1.0),
            successor: (successor_pose, successor_gripper),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=current,
        posterior={current: 0.5, successor: 0.5},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.LOW_CONFIDENCE,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is False
    assert result.control_equivalence.equivalent_states == (current,)
    assert result.decision == ExecutionDecision.HOLD
    assert "low_progress_confidence" in result.reasons


def test_control_equivalence_never_crosses_skill_boundary(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    terminal = StateId(0, 3)
    entry = StateId(1, 0)
    for state in (terminal, entry):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(terminal)
    _install_control_targets(
        monkeypatch,
        controller,
        {terminal: (pose(0.0), 1.0), entry: (pose(0.0), 1.0)},
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=terminal,
        estimated=terminal,
        posterior={terminal: 0.5, entry: 0.5},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.LOW_CONFIDENCE,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, terminal),
        mode_by_skill={0: 0, 1: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is False
    assert result.control_equivalence.class_count == 2
    assert result.decision == ExecutionDecision.HOLD
    assert "low_progress_confidence" in result.reasons


def test_unknown_holds_but_reuses_last_trusted_servo_weight(execution_model) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    controller = TSFExecutionController(model)
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


def test_unknown_without_history_uses_soft_external_safe_weight(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, successor, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    unknown = belief_for(
        model,
        demos,
        tick=0,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate(
            "object",
            (0.8, 0.2),
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
    decision = result.roles.decisions["object"]
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert decision.role == FrameRole.DEFER
    assert decision.execution_weight == pytest.approx(0.8)
    assert decision.blocks_advance is False
    assert result.weighted_action.available is True

    hidden_controller = TSFExecutionController(model)
    hidden_controller.reset(current)
    hidden = belief_for(
        model,
        demos,
        tick=0,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate(
            "object",
            (0.8, 0.2),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )
    hidden.runtime_features.frame_visibility["object"] = False
    hidden_result = hidden_controller.update(
        hidden,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
    )
    hidden_decision = hidden_result.roles.decisions["object"]
    assert hidden_result.decision == ExecutionDecision.HOLD
    assert hidden_decision.execution_weight == 0.0
    assert hidden_decision.blocks_advance is True


def test_ambiguous_expected_relation_defers_without_blocking_or_false_recovery(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    node = model.state(current)
    node.demo_relation_priors["object"][:] = np.asarray([0.35, 0.65])
    controller = TSFExecutionController(model)
    controller.reset(current)
    linked = belief_for(
        model,
        demos,
        tick=0,
        nominal=current,
        estimated=current,
        relation=relation_estimate("object", (0.2, 0.8), RelationDecision.LINKED),
    )
    result = controller.update(
        linked,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
    )
    decision = result.roles.decisions["object"]
    assert decision.expected_distribution[1] == pytest.approx(0.65)
    assert decision.expected_relation == RelationDecision.UNKNOWN
    assert decision.actual_relation == RelationDecision.LINKED
    assert decision.role == FrameRole.DEFER
    assert decision.execution_weight == 0.0
    assert decision.blocks_advance is False
    assert decision.recovery_intent is None

    controller.reset(current)
    external = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=current,
        relation=relation_estimate("object", (0.8, 0.2), RelationDecision.EXTERNAL),
    )
    external_result = controller.update(
        external,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
    )
    external_decision = external_result.roles.decisions["object"]
    assert external_decision.expected_relation == RelationDecision.UNKNOWN
    assert external_decision.actual_relation == RelationDecision.EXTERNAL
    assert external_decision.role == FrameRole.DEFER
    assert external_decision.execution_weight == pytest.approx(0.8)
    assert external_decision.blocks_advance is False
    assert external_decision.recovery_intent is None


def test_controller_advances_one_direct_successor_only(execution_model) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, successor, linked=False)
    controller = TSFExecutionController(model)
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


def test_committed_reference_is_the_next_cycle_completion_prior_anchor(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, successor, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=successor,
        estimated=successor,
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
    )
    controller.update(
        belief,
        dynamac_observation(model, demos, successor),
        mode_by_skill={0: 0},
    )

    next_prior = ProgressPriorBuilder(model).build(
        {successor: 1.0},
        executed_reference_state=controller.cursor.reference_state,
    )
    assert controller.cursor.reference_state == successor
    assert next_prior.nominal_state == successor
    assert next_prior.probabilities[successor] == pytest.approx(0.85)


def test_controller_holds_current_target_instead_of_replaying_backward(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 2)
    nominal = StateId(0, 3)
    estimated = StateId(0, 0)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, estimated, linked=False)
    controller = TSFExecutionController(model)
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
    assert result.decision == ExecutionDecision.HOLD
    assert result.reasons == ("current_target_incomplete",)
    assert result.cursor_after.reference_state == current
    assert result.weighted_action.action is not None
    assert result.weighted_action.action.diagnostics["time_index"] == 2
    assert result.weighted_action.action.diagnostics["query_advances_clock"] is False


def test_control_equivalent_low_confidence_lag_completes_current_reference(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (pose(0.0), 1.0),
            successor: (pose(0.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.52, current: 0.45, successor: 0.03},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.LOW_CONFIDENCE,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is True
    assert result.control_equivalence.equivalent_states == (
        estimated,
        current,
        successor,
    )
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == (
        "control_equivalent_progress_uncertainty",
        "control_equivalent_current_reference_reached",
    )


def test_control_equivalent_backward_realign_completes_adjacent_reference(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (pose(0.0), 1.0),
            successor: (pose(0.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.94, current: 0.058, successor: 0.002},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.evaluated is True
    assert result.control_equivalence.accepted is True
    assert result.control_equivalence.equivalent_states == (
        estimated,
        current,
        successor,
    )
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == (
        "control_equivalent_reference_lag",
        "control_equivalent_current_reference_reached",
    )


def test_complete_state_lag_does_not_freeze_supported_continuous_reference(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(1.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.94, current: 0.058, successor: 0.002},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    belief = replace(
        belief,
        progress=replace(
            belief.progress,
            prior={estimated: 0.20, current: 0.65, successor: 0.15},
        ),
        candidate_scores={
            estimated: candidate_score(
                estimated, robot=0.10, scene=1.0, scene_expected=False
            ),
            current: candidate_score(
                current, robot=0.99, scene=1.0e-4, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=0.05, scene=0.01, scene_expected=True
            ),
        },
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is False
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == ("continuous_reference_supported",)


def test_tracked_world_command_advances_when_moving_frame_hides_robot_support(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(1.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.99, current: 0.009, successor: 0.001},
        relation=relation_estimate(
            "object", (0.95, 0.05), RelationDecision.EXTERNAL
        ),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    belief = replace(
        belief,
        runtime_features=replace(
            belief.runtime_features,
            command_tracking_available=True,
            command_tracking_compatibility=0.99,
            command_tracking_mahalanobis_squared=0.01,
            action_excitation=1.0,
            paired_tracking_reliability={"object": 1.0},
            relation_information_weight={"object": 0.5},
        ),
        candidate_scores={
            estimated: candidate_score(
                estimated, robot=0.99, scene=1.0, scene_expected=False
            ),
            current: candidate_score(
                current, robot=1.0e-8, scene=1.0e-8, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=1.0e-9, scene=1.0e-8, scene_expected=True
            ),
        },
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == ("continuous_reference_supported",)


def test_tracked_world_command_does_not_override_static_frame_posterior_lag(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(1.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.99, current: 0.009, successor: 0.001},
        relation=relation_estimate(
            "object", (0.95, 0.05), RelationDecision.EXTERNAL
        ),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    belief = replace(
        belief,
        runtime_features=replace(
            belief.runtime_features,
            command_tracking_available=True,
            command_tracking_compatibility=0.99,
            command_tracking_mahalanobis_squared=0.01,
            action_excitation=0.6,
            paired_tracking_reliability={"object": 1.0},
            relation_information_weight={"object": 0.6},
        ),
        candidate_scores={
            estimated: candidate_score(
                estimated, robot=0.99, scene=1.0, scene_expected=False
            ),
            current: candidate_score(
                current, robot=1.0e-8, scene=1.0e-8, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=1.0e-9, scene=1.0e-8, scene_expected=True
            ),
        },
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert result.reasons == ("current_target_incomplete",)


def test_multi_sample_scene_lag_does_not_gate_robot_supported_continuous_reference(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 2)
    successor = StateId(0, 3)
    for state in (estimated, StateId(0, 1), current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            StateId(0, 1): (pose(0.25), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(0.75), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.94, current: 0.058, successor: 0.002},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    belief = replace(
        belief,
        progress=replace(
            belief.progress,
            prior={estimated: 0.10, StateId(0, 1): 0.10, current: 0.65, successor: 0.15},
        ),
        candidate_scores={
            estimated: candidate_score(
                estimated, robot=0.05, scene=1.0, scene_expected=False
            ),
            StateId(0, 1): candidate_score(
                StateId(0, 1), robot=0.10, scene=1.0, scene_expected=False
            ),
            current: candidate_score(
                current, robot=0.99, scene=1.0e-4, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=0.05, scene=0.01, scene_expected=True
            ),
        },
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is False
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == ("continuous_reference_supported",)


def test_robot_supported_link_terminal_hands_successor_verification_to_policy(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 2)
    successor = StateId(0, 3)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            StateId(0, 1): (pose(0.25), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(0.75), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.94, current: 0.058, successor: 0.002},
        relation=relation_estimate(
            "object",
            (0.5, 0.5),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )
    belief = replace(
        belief,
        progress=replace(
            belief.progress,
            prior={
                estimated: 0.10,
                StateId(0, 1): 0.10,
                current: 0.65,
                successor: 0.15,
            },
        ),
        candidate_scores={
            estimated: candidate_score(
                estimated, robot=0.05, scene=1.0, scene_expected=False
            ),
            StateId(0, 1): candidate_score(
                StateId(0, 1), robot=0.10, scene=1.0, scene_expected=False
            ),
            current: candidate_score(
                current, robot=0.99, scene=1.0e-4, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=0.05, scene=0.01, scene_expected=True
            ),
        },
    )
    event_id = next(iter(sorted(model.link_anchors)))
    request = RelationVerificationRequest(
        arm_id=model.arm_id,
        frame_id=event_id.frame_id,
        relation="linked",
        event_id=event_id,
        context_state=current,
    )

    def route(state_id, _belief, **_kwargs):
        return FrameRoleSnapshot(
            state_id=state_id,
            decisions={},
            verification_requests=((request,) if state_id == successor else ()),
            unresolved_formal_link_events=(event_id,),
        )

    monkeypatch.setattr(controller, "_route_roles", route)
    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert result.roles.verification_requests == (request,)
    assert result.reasons == (
        "current_target_incomplete",
        "successor_relation_verification_required",
    )


def test_complete_state_uncertainty_does_not_freeze_supported_continuous_reference(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    previous = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (previous, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            previous: (pose(0.0), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(1.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=current,
        posterior={previous: 0.01, current: 0.52, successor: 0.47},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.LOW_CONFIDENCE,
    )
    belief = replace(
        belief,
        progress=replace(
            belief.progress,
            prior={previous: 0.01, current: 0.75, successor: 0.24},
        ),
        candidate_scores={
            previous: candidate_score(
                previous, robot=0.10, scene=1.0, scene_expected=False
            ),
            current: candidate_score(
                current, robot=0.99, scene=0.20, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=0.60, scene=1.0, scene_expected=False
            ),
        },
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is False
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == ("continuous_reference_supported",)


def test_unexplained_scene_does_not_freeze_supported_continuous_reference(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    previous = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (previous, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            previous: (pose(0.0), 1.0),
            current: (pose(0.5), 1.0),
            successor: (pose(1.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=current,
        posterior={previous: 0.0, current: 0.80, successor: 0.20},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.NO_PLAUSIBLE_STATE,
    )
    belief = replace(
        belief,
        progress=replace(
            belief.progress,
            prior={previous: 0.0, current: 0.80, successor: 0.20},
        ),
        candidate_scores={
            previous: candidate_score(
                previous, robot=0.05, scene=1.0e-12, scene_expected=True
            ),
            current: candidate_score(
                current, robot=0.99, scene=1.0e-12, scene_expected=True
            ),
            successor: candidate_score(
                successor, robot=0.05, scene=1.0e-12, scene_expected=True
            ),
        },
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.accepted is False
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == ("continuous_reference_supported",)


def test_control_equivalent_aligned_reference_lag_uses_reference_control(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (pose(0.0), 1.0),
            successor: (pose(0.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=estimated,
        estimated=estimated,
        posterior={estimated: 1.0},
        relation=relation_estimate(
            "object", (0.95, 0.05), RelationDecision.EXTERNAL
        ),
        status=ProgressStatus.ALIGNED,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=False,
    )

    assert result.control_equivalence.evaluated is True
    assert result.control_equivalence.accepted is True
    assert result.control_equivalence.equivalent_states == (estimated, current)
    assert result.control_equivalence.aggregated_confidence == pytest.approx(1.0)
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.cursor_after.reference_state == successor
    assert result.reasons == (
        "control_equivalent_reference_lag",
        "control_equivalent_current_reference_reached",
    )


@pytest.mark.parametrize(
    ("current_pose", "current_gripper"),
    ((pose(0.5), 1.0), (pose(0.0), -1.0)),
)
def test_control_relevant_backward_realign_still_holds(
    execution_model,
    monkeypatch,
    current_pose,
    current_gripper,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    current = StateId(0, 1)
    successor = StateId(0, 2)
    for state in (estimated, current, successor):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            current: (current_pose, current_gripper),
            successor: (current_pose, current_gripper),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.94, current: 0.058, successor: 0.002},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.evaluated is True
    assert result.control_equivalence.accepted is False
    assert result.control_equivalence.equivalent_states == (estimated,)
    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert result.reasons == ("current_target_incomplete",)


def test_control_equivalent_backward_realign_does_not_absorb_two_state_lag(
    execution_model,
    monkeypatch,
) -> None:
    model, demos = execution_model
    estimated = StateId(0, 0)
    intermediate = StateId(0, 1)
    current = StateId(0, 2)
    for state in (estimated, intermediate, current):
        force_relation_prior(model, state, linked=False)
    controller = TSFExecutionController(model)
    controller.reset(current)
    _install_control_targets(
        monkeypatch,
        controller,
        {
            estimated: (pose(0.0), 1.0),
            intermediate: (pose(0.0), 1.0),
            current: (pose(0.0), 1.0),
        },
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=current,
        estimated=estimated,
        posterior={estimated: 0.94, intermediate: 0.04, current: 0.02},
        relation=relation_estimate("object", (0.95, 0.05), RelationDecision.EXTERNAL),
        status=ProgressStatus.BACKWARD_REALIGNMENT,
    )

    result = controller.update(
        belief,
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )

    assert result.control_equivalence.evaluated is True
    assert result.control_equivalence.accepted is False
    assert result.decision == ExecutionDecision.HOLD
    assert result.cursor_after.reference_state == current
    assert result.reasons == ("current_target_incomplete",)


def test_trusted_progress_not_exact_command_covariance_controls_advancement(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 1)
    successor = StateId(0, 2)
    controller = TSFExecutionController(model)
    controller.reset(current)
    force_relation_prior(model, successor, linked=False)
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
        dynamac_observation(model, demos, current),
        mode_by_skill={0: 0},
        action_executed=True,
    )
    assert result.decision == ExecutionDecision.ADVANCE
    assert result.reasons == ("early_successor_reached",)
    assert result.cursor_after.reference_state == successor


def test_reliable_relation_mismatch_blocks_progress_and_emits_after_persistence(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(1, 0)
    successor = StateId(1, 1)
    # Isolate ordinary mismatch persistence from the separate causal LINK
    # successor allowance exercised above.
    model.link_anchors.clear()
    model.link_origins.clear()
    force_relation_prior(model, current, linked=False)
    controller = TSFExecutionController(
        model,
        TSFExecutionConfig(
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


def test_blocked_proposed_state_contributes_persistent_relation_mismatch(
    execution_model,
) -> None:
    model, demos = execution_model
    event_id, anchor = next(iter(sorted(model.link_anchors.items())))
    assert len(anchor.linked_entry_states) >= 2
    current = anchor.linked_entry_states[-2]
    successor = model.state(current).topology.successors[0]
    assert successor.skill_index == current.skill_index
    # Isolate proposed-state routing from the separately tested formal-event
    # lifecycle: admit the shortened interval's terminal state once, while its
    # direct successor is already outside that interval and must be rejected.
    model.link_anchors[event_id] = replace(
        anchor,
        linked_entry_states=anchor.linked_entry_states[:-1],
    )
    model.unlink_events.clear()
    force_relation_prior(model, current, linked=True)
    force_relation_prior(model, successor, linked=True)
    controller = TSFExecutionController(
        model,
        TSFExecutionConfig(
            mismatch=MismatchConfig(
                no_plausible_cycles=3,
                relation_mismatch_cycles=2,
                persistent_hold_cycles=10,
                stalled_progress_cycles=10,
            )
        ),
    )
    controller.reset(current)
    first_entry = model.link_anchors[event_id].linked_entry_states[0]
    predecessor = model.state(first_entry).topology.predecessors[0]
    for tick, route_state in enumerate((predecessor, first_entry), start=-2):
        controller.role_router.route(
            route_state,
            belief_for(
                model,
                demos,
                tick=tick,
                nominal=route_state,
                estimated=route_state,
                relation=relation_estimate(
                    "object",
                    (0.5, 0.5),
                    RelationDecision.UNKNOWN,
                    information_weight=0.0,
                ),
                static=True,
            ),
            mode_by_skill={route_state.skill_index: event_id.mode},
        )
    relation = relation_estimate(
        "object",
        (0.95, 0.05),
        RelationDecision.EXTERNAL,
        information_weight=0.5,
        observation_likelihood=(0.01, 1.0),
    )

    first = controller.update(
        belief_for(
            model,
            demos,
            tick=1,
            nominal=current,
            estimated=current,
            relation=relation,
        ),
        dynamac_observation(model, demos, current),
        mode_by_skill={current.skill_index: event_id.mode},
        action_executed=True,
    )
    assert first.cursor_after.reference_state == current
    assert "proposed_reference_blocked_by_role" in first.reasons
    assert first.mismatch.counters.relation_mismatch == 1
    assert first.mismatch.events == ()

    second = controller.update(
        belief_for(
            model,
            demos,
            tick=2,
            nominal=current,
            estimated=current,
            relation=relation,
        ),
        dynamac_observation(model, demos, current),
        mode_by_skill={current.skill_index: event_id.mode},
        action_executed=True,
    )
    assert second.cursor_after.reference_state == current
    assert [event.kind for event in second.mismatch.events] == [
        MismatchKind.RELATION_MISMATCH
    ]
    assert second.mismatch.events[0].recovery_intents[0].frame_id == "object"


def test_role_router_never_changes_cursor_and_controller_owns_all_commits(
    execution_model,
) -> None:
    model, demos = execution_model
    current = StateId(0, 0)
    successor = StateId(0, 1)
    force_relation_prior(model, current, linked=False)
    force_relation_prior(model, successor, linked=False)
    controller = TSFExecutionController(model)
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


@pytest.mark.parametrize(
    "decision",
    (RelationDecision.UNKNOWN, RelationDecision.EXTERNAL),
)
def test_pending_unresolved_with_observable_low_excitation_requests_verification(
    execution_model,
    decision: RelationDecision,
) -> None:
    model, demos = execution_model
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
            (0.5, 0.5) if decision == RelationDecision.UNKNOWN else (0.9, 0.1),
            decision,
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
    assert snapshot.decisions["object"].role == (
        FrameRole.DEFER if decision == RelationDecision.UNKNOWN else FrameRole.EXECUTE
    )
    assert snapshot.recovery_intents == ()
    assert len(snapshot.verification_requests) == 1
    assert snapshot.verification_requests[0].event_id == event_id


def test_pending_verification_does_not_require_current_poe_selection(
    execution_model,
) -> None:
    """A newly linked frame normally leaves PoE but can still need verification."""

    model, demos = execution_model
    pending_state = StateId(0, 3)
    future_linked = StateId(1, 0)
    force_relation_prior(model, pending_state, linked=True)
    force_relation_prior(model, future_linked, linked=True)
    node = model.state(pending_state)
    node.selected_frames = tuple(
        frame for frame in node.selected_frames if frame != "object"
    )
    node.mode_selected_frames = tuple(
        tuple(frame for frame in frames if frame != "object")
        for frames in node.mode_selected_frames
    )
    node.action_relevant_frames = tuple(
        frame for frame in node.action_relevant_frames if frame != "object"
    )
    node.mode_action_relevant_frames = tuple(
        tuple(frame for frame in frames if frame != "object")
        for frames in node.mode_action_relevant_frames
    )
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
            (0.9, 0.1),
            RelationDecision.EXTERNAL,
            information_weight=0.0,
        ),
        static=True,
    )

    snapshot = FrameRoleRouter(model).route(
        pending_state,
        belief,
        mode_by_skill={0: 0, 1: 0, 2: 0},
    )

    assert "object" not in snapshot.decisions
    assert len(snapshot.verification_requests) == 1
    assert snapshot.verification_requests[0].event_id == event_id


def test_pending_event_context_remains_active_after_the_closing_state(
    execution_model,
) -> None:
    """Natural motion may become uninformative only after the closing sample."""

    model, demos = execution_model
    pending_state = StateId(0, 1)
    later_state = StateId(0, 3)
    future_linked = StateId(1, 0)
    force_relation_prior(model, pending_state, linked=True)
    force_relation_prior(model, later_state, linked=True)
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
        event_local_indices=(1, 1, 1, 1, 1),
    )
    belief = belief_for(
        model,
        demos,
        tick=1,
        nominal=later_state,
        estimated=later_state,
        relation=relation_estimate(
            "object",
            (0.45, 0.55),
            RelationDecision.UNKNOWN,
            information_weight=0.0,
        ),
        static=True,
    )

    router = FrameRoleRouter(model)
    router.route(
        pending_state,
        belief_for(
            model,
            demos,
            tick=0,
            nominal=pending_state,
            estimated=pending_state,
            relation=relation_estimate(
                "object",
                (0.45, 0.55),
                RelationDecision.UNKNOWN,
                information_weight=0.0,
            ),
            static=True,
        ),
        mode_by_skill={0: 0, 1: 0, 2: 0},
    )
    snapshot = router.route(
        later_state,
        belief,
        mode_by_skill={0: 0, 1: 0, 2: 0},
    )

    assert len(snapshot.verification_requests) == 1
    assert snapshot.verification_requests[0].event_id == event_id


def test_pending_without_future_linked_requirement_does_not_request_verification(
    execution_model,
) -> None:
    model, demos = execution_model
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


def test_pending_event_lifecycle_is_identical_for_first_and_later_occurrences(
    execution_model,
) -> None:
    """Every causally committed occurrence follows the same lifecycle."""

    model, demos = execution_model
    model.link_anchors.clear()
    model.link_origins.clear()
    model.unlink_events.clear()
    model.link_pending_events.clear()
    for state_id in model.states:
        force_relation_prior(model, state_id, linked=False)

    occurrences = (
        (StateId(0, 3), StateId(1, 0), StateId(1, 3), 0),
        (StateId(1, 3), StateId(2, 0), StateId(2, 3), 1),
    )
    event_ids = []
    for source_state, candidate_state, release_state, occurrence in occurrences:
        source_node = model.state(source_state)
        entry_node = model.state(candidate_state)
        source_node.gripper_commands = tuple(
            np.ones_like(command) for command in source_node.gripper_commands
        )
        entry_node.gripper_commands = tuple(
            np.zeros_like(command) for command in entry_node.gripper_commands
        )
        event_id = RelationEventId(
            model.arm_id,
            "object",
            candidate_state.skill_index,
            0,
            occurrence,
            "link_pending",
        )
        event_ids.append(event_id)
        model.link_pending_events[event_id] = LinkPendingCandidate(
            event_id=event_id,
            arm_id=model.arm_id,
            frame_id="object",
            candidate_state=candidate_state,
            context_state=source_state,
            local_means=np.stack([pose(0.0), pose(0.01)]),
            local_covariances=np.stack([np.eye(6) * 0.01] * 2),
            gripper_commands=np.asarray([[1.0], [-1.0]]),
            release_state=release_state,
            demonstration_indices=(0, 1, 2, 3, 4),
            event_local_indices=(1, 1, 1, 1, 1),
        )

    modes = {0: 0, 1: 0, 2: 0}
    reliable_external = relation_estimate(
        "object",
        (0.95, 0.05),
        RelationDecision.EXTERNAL,
        information_weight=0.5,
        observation_likelihood=(1.0, 0.01),
    )
    router = FrameRoleRouter(model)
    requests = []
    for index, (source_state, candidate_state, release_state, _occurrence) in enumerate(
        occurrences
    ):
        source_belief = belief_for(
            model,
            demos,
            tick=4 * index,
            nominal=source_state,
            estimated=source_state,
            relation=reliable_external,
        )
        source_snapshot = router.route(
            source_state,
            source_belief,
            mode_by_skill=modes,
        )
        router.commit_boundary_entry(
            source_snapshot,
            source_belief,
            candidate_state,
            mode_by_skill=modes,
        )

        entry_snapshot = router.route(
            candidate_state,
            belief_for(
                model,
                demos,
                tick=4 * index + 1,
                nominal=candidate_state,
                estimated=candidate_state,
                relation=reliable_external,
            ),
            mode_by_skill=modes,
        )
        assert entry_snapshot.unresolved_pending_link_events == (event_ids[index],)
        assert entry_snapshot.verification_requests == ()

        release_predecessor = StateId(release_state.skill_index, release_state.local_index - 1)
        due_snapshot = router.route(
            release_predecessor,
            belief_for(
                model,
                demos,
                tick=4 * index + 2,
                nominal=release_predecessor,
                estimated=release_predecessor,
                relation=reliable_external,
            ),
            mode_by_skill=modes,
        )
        assert len(due_snapshot.verification_requests) == 1
        request = due_snapshot.verification_requests[0]
        assert request.event_id == event_ids[index]
        assert request.committed_occurrence is True
        requests.append(request.event_id)

        released = router.route(
            release_state,
            belief_for(
                model,
                demos,
                tick=4 * index + 3,
                nominal=release_state,
                estimated=release_state,
                relation=reliable_external,
            ),
            mode_by_skill=modes,
        )
        assert released.unresolved_pending_link_events == ()

    assert requests == event_ids


def test_committed_pending_boundary_occurrence_enters_public_verification(
    execution_model,
) -> None:
    """A permitted boundary close must not hide its exact Pending occurrence."""

    model, demos = execution_model
    source = StateId(0, 3)
    entry = StateId(1, 0)
    model.link_anchors.clear()
    model.link_origins.clear()
    model.link_pending_events.clear()
    event_id = RelationEventId(
        model.arm_id,
        "object",
        entry.skill_index,
        0,
        0,
        "link_pending",
    )
    model.link_pending_events[event_id] = LinkPendingCandidate(
        event_id=event_id,
        arm_id=model.arm_id,
        frame_id="object",
        candidate_state=entry,
        context_state=source,
        local_means=np.stack([pose(0.0), pose(0.01)]),
        local_covariances=np.stack([np.eye(6) * 0.01] * 2),
        gripper_commands=np.asarray([[1.0], [-1.0]]),
        release_state=StateId(2, 0),
        demonstration_indices=(0, 1, 2, 3, 4),
        event_local_indices=(0, 0, 0, 0, 0),
    )
    modes = {0: 0, 1: 0, 2: 0}
    external = relation_estimate(
        "object",
        (0.999, 0.001),
        RelationDecision.EXTERNAL,
        information_weight=0.8,
        observation_likelihood=(1.0, 0.01),
    )
    router = FrameRoleRouter(model)
    source_belief = belief_for(
        model,
        demos,
        tick=0,
        nominal=source,
        estimated=source,
        relation=external,
    )
    source_snapshot = router.route(source, source_belief, mode_by_skill=modes)
    assert source_snapshot.verification_requests == ()

    router.commit_boundary_entry(
        source_snapshot,
        source_belief,
        entry,
        mode_by_skill=modes,
    )
    entry_snapshot = router.route(
        entry,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=entry,
            estimated=entry,
            relation=external,
        ),
        mode_by_skill=modes,
    )

    assert entry_snapshot.unresolved_pending_link_events == (event_id,)
    assert entry_snapshot.verification_requests == ()

    release_predecessor = model.skill_states[1][-1]
    due_snapshot = router.route(
        release_predecessor,
        belief_for(
            model,
            demos,
            tick=2,
            nominal=release_predecessor,
            estimated=release_predecessor,
            relation=external,
        ),
        mode_by_skill=modes,
    )
    assert len(due_snapshot.verification_requests) == 1
    request = due_snapshot.verification_requests[0]
    assert request.event_id == event_id
    assert request.committed_occurrence is True


def test_committed_pending_fresh_external_evidence_uses_existing_mismatch_gate(
    execution_model,
) -> None:
    """Natural post-close evidence rejects the exact occurrence without probing."""

    model, demos = execution_model
    source = StateId(0, 3)
    entry = StateId(1, 0)
    model.link_anchors.clear()
    model.link_origins.clear()
    model.link_pending_events.clear()
    event_id = RelationEventId(
        model.arm_id,
        "object",
        entry.skill_index,
        0,
        0,
        "link_pending",
    )
    model.link_pending_events[event_id] = LinkPendingCandidate(
        event_id=event_id,
        arm_id=model.arm_id,
        frame_id="object",
        candidate_state=entry,
        context_state=source,
        local_means=np.stack([pose(0.0), pose(0.01)]),
        local_covariances=np.stack([np.eye(6) * 0.01] * 2),
        gripper_commands=np.asarray([[1.0], [-1.0]]),
        release_state=StateId(2, 0),
        demonstration_indices=(0, 1, 2, 3, 4),
        event_local_indices=(0, 0, 0, 0, 0),
    )
    modes = {0: 0, 1: 0, 2: 0}
    router = FrameRoleRouter(model)
    source_belief = belief_for(
        model,
        demos,
        tick=0,
        nominal=source,
        estimated=source,
        relation=relation_estimate(
            "object",
            (0.99, 0.01),
            RelationDecision.EXTERNAL,
            information_weight=0.5,
        ),
    )
    source_snapshot = router.route(source, source_belief, mode_by_skill=modes)
    router.commit_boundary_entry(
        source_snapshot,
        source_belief,
        entry,
        mode_by_skill=modes,
    )

    fresh_external = relation_estimate(
        "object",
        (0.99, 0.01),
        RelationDecision.EXTERNAL,
        information_weight=0.5,
        observation_likelihood=(1.0, 0.01),
        evidence_direction=RelationDecision.EXTERNAL,
    )
    snapshot = router.route(
        entry,
        belief_for(
            model,
            demos,
            tick=1,
            nominal=entry,
            estimated=entry,
            relation=fresh_external,
        ),
        mode_by_skill=modes,
    )

    assert snapshot.verification_requests == ()
    assert len(snapshot.recovery_intents) == 1
    intent = snapshot.recovery_intents[0]
    assert intent.origin_event_id == event_id
    assert intent.expected_relation == RelationDecision.LINKED
    assert intent.actual_relation == RelationDecision.EXTERNAL


def test_no_plausible_and_persistent_hold_have_independent_counters(
    execution_model,
) -> None:
    model, demos = execution_model
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
        TSFExecutionController(model).cursor,
        ExecutionDecision.HOLD,
        roles,
    )
    assert {event.kind for event in update.events} == {
        MismatchKind.NO_PLAUSIBLE_STATE,
        MismatchKind.PERSISTENT_HOLD,
    }


def test_rejected_command_does_not_become_task_level_no_plausible_event(
    execution_model,
) -> None:
    model, demos = execution_model
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

    rejected = tracker.update(
        belief,
        TSFExecutionController(model).cursor,
        ExecutionDecision.HOLD,
        roles,
        action_executed=False,
    )

    assert rejected.counters.no_plausible_state == 0
    assert [event.kind for event in rejected.events] == [MismatchKind.PERSISTENT_HOLD]


def test_tracked_phase_three_config_matches_code_defaults() -> None:
    path = Path("configs/tsf_execution.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == asdict(TSFExecutionConfig())
    assert TSFExecutionConfig.from_json(path).to_dict() == payload


def test_phase_three_config_rejects_unknown_top_level_field() -> None:
    with pytest.raises(ValueError, match="未知分区"):
        TSFExecutionConfig.from_mapping({"retired_threshold": 0.75})


def test_phase_three_config_rejects_unknown_frame_role_field() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TSFExecutionConfig.from_mapping(
            {"frame_roles": {"retired_state_cap": 5}}
        )
