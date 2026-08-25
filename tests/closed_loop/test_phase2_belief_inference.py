"""Phase-two acceptance tests for relation-progress belief inference."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import DynaMAC, DynaMACConfig, DynaMACDemonstration
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    BoundaryId,
    CandidateScore,
    ClosedLoopTaskModelBuilder,
    FactorDistribution,
    FactorId,
    ProgressFilter,
    ProgressPriorBuilder,
    ProgressStatus,
    RelationDecision,
    RelationEstimate,
    RelationFilter,
    RelationFilterConfig,
    RuntimeFeatureBuilder,
    RuntimeObservation,
    StateEvaluator,
    StateId,
)
from essay2608.policy.dynamac import pose_compose, pose_inverse


def pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture(scope="module")
def phase2_model():
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
                carry_start, pose(0.12 * progress, 0.08 * progress, 0.04 * progress)
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
                name=f"phase2_{demo_index}",
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
    return ClosedLoopTaskModelBuilder().build(policy, demonstrations), demonstrations


def observation(
    tick: int,
    ee_x: float,
    object_x: float,
    *,
    previous_ee_x: float | None,
    visible: bool = True,
    reliable: float = 1.0,
) -> RuntimeObservation:
    return RuntimeObservation(
        tick=tick,
        ee_pose=pose(ee_x),
        frame_poses={"object": pose(object_x)},
        gripper_state=np.asarray([-1.0]),
        previous_command_pose=None if previous_ee_x is None else pose(ee_x),
        previous_ee_pose=None if previous_ee_x is None else pose(previous_ee_x),
        tracking_reliability={"object": reliable},
        frame_visibility={"object": visible},
    )


def test_runtime_features_distinguish_comotion_external_and_static() -> None:
    builder = RuntimeFeatureBuilder()
    previous = observation(0, 0.0, 0.4, previous_ee_x=None)
    comoving = builder.build(observation(1, 0.1, 0.5, previous_ee_x=0.0), previous)
    external = builder.build(observation(1, 0.1, 0.4, previous_ee_x=0.0), previous)
    static = builder.build(observation(1, 0.0, 0.4, previous_ee_x=0.0), previous)
    assert np.linalg.norm(comoving.relative_motion_residuals["object"]) < 1.0e-12
    assert np.linalg.norm(external.relative_motion_residuals["object"]) > 0.09
    assert comoving.action_excitation == pytest.approx(1.0)
    assert static.action_excitation == pytest.approx(0.0)


def test_relation_filter_has_shared_physical_semantics_without_static_false_link(
    phase2_model,
) -> None:
    model, _ = phase2_model
    feature_builder = RuntimeFeatureBuilder()
    relation_filter = RelationFilter(
        model,
        RelationFilterConfig(demonstration_prior_strength=0.0),
        feature_builder=feature_builder,
    )
    progress = {StateId(0, 1): 1.0}
    previous_observation = observation(0, 0.0, 0.4, previous_ee_x=None)

    synchronous = feature_builder.build(
        observation(1, 0.1, 0.5, previous_ee_x=0.0), previous_observation
    )
    linked = relation_filter.update(
        progress, synchronous, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert np.sum(linked.posterior) == pytest.approx(1.0)
    assert linked.linked > linked.external
    assert linked.decision_state == RelationDecision.LINKED

    nonsynchronous = feature_builder.build(
        observation(1, 0.1, 0.4, previous_ee_x=0.0), previous_observation
    )
    external = relation_filter.update(
        progress, nonsynchronous, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert external.external > external.linked
    assert external.decision_state == RelationDecision.EXTERNAL

    no_motion = feature_builder.build(
        observation(1, 0.0, 0.4, previous_ee_x=0.0), previous_observation
    )
    unknown = relation_filter.update(
        progress, no_motion, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert unknown.decision_state == RelationDecision.UNKNOWN
    assert unknown.informative is False


def test_invisible_or_unreliable_relation_is_unknown(phase2_model) -> None:
    model, _ = phase2_model
    builder = RuntimeFeatureBuilder()
    previous = observation(0, 0.0, 0.4, previous_ee_x=None)
    hidden = builder.build(
        observation(1, 0.1, 0.5, previous_ee_x=0.0, visible=False),
        previous,
    )
    estimate = RelationFilter(model).update(
        {StateId(0, 1): 1.0}, hidden, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert estimate.decision_state == RelationDecision.UNKNOWN
    assert estimate.information_weight == 0.0

    no_previous_frame = builder.build(observation(1, 0.1, 0.5, previous_ee_x=0.0), None)
    first_estimate = RelationFilter(model).update(
        {StateId(0, 1): 1.0},
        no_previous_frame,
        {"object": np.asarray([0.5, 0.5])},
    )["object"]
    assert first_estimate.decision_state == RelationDecision.UNKNOWN
    assert first_estimate.information_weight == 0.0


def test_progress_prior_concentrates_on_successor_and_obeys_boundary_gate(
    phase2_model,
) -> None:
    model, _ = phase2_model
    builder = ProgressPriorBuilder(model)
    internal = builder.build({StateId(0, 1): 1.0})
    assert internal.nominal_state == StateId(0, 2)
    assert internal.probabilities[StateId(0, 2)] == pytest.approx(0.65)

    terminal = StateId(0, len(model.skill_states[0]) - 1)
    blocked = builder.build({terminal: 1.0})
    assert blocked.probabilities == {terminal: 1.0}
    boundary = BoundaryId(model.arm_id, 0, 1)
    released = builder.build(
        {terminal: 1.0}, permitted_boundaries=frozenset({boundary})
    )
    assert released.nominal_state == StateId(1, 0)
    assert set(released.probabilities) == {terminal, StateId(1, 0)}


def candidate(
    state: StateId,
    log_score: float,
    compatibility: float,
    *,
    raw_log_likelihood: float | None = None,
) -> CandidateScore:
    return CandidateScore(
        state_id=state,
        robot_log_likelihood=(
            log_score if raw_log_likelihood is None else raw_log_likelihood
        ),
        state_log_likelihood=0.0,
        robot_log_support=log_score,
        state_log_support=0.0,
        relation_log_compatibility=0.0,
        explanation_log_score=log_score,
        normalized_explanation_score=compatibility,
        robot_compatibility=compatibility,
        state_compatibility=1.0,
        relation_compatibility=1.0,
    )


def test_progress_uses_peak_normalized_support_not_raw_density(
    phase2_model,
) -> None:
    model, _ = phase2_model
    earlier = StateId(0, 1)
    successor = StateId(0, 2)
    scores = {
        # The earlier state has an artificially much higher raw density peak,
        # but both observations sit exactly at their respective state means.
        earlier: candidate(
            earlier,
            log_score=0.0,
            compatibility=1.0,
            raw_log_likelihood=100.0,
        ),
        successor: candidate(
            successor,
            log_score=0.0,
            compatibility=1.0,
            raw_log_likelihood=0.0,
        ),
    }
    estimate = ProgressFilter(model).update(
        {earlier: 0.20, successor: 0.80},
        scores,
        successor,
    )
    assert estimate.estimated_state == successor
    assert estimate.posterior[successor] == pytest.approx(0.80)


def test_progress_posterior_can_realign_backward_and_forward(phase2_model) -> None:
    model, _ = phase2_model
    progress_filter = ProgressFilter(model)
    states = (StateId(0, 1), StateId(0, 2), StateId(0, 3))
    prior = {state: value for state, value in zip(states, (0.2, 0.65, 0.15))}
    backward_scores = {
        states[0]: candidate(states[0], 0.0, 1.0),
        states[1]: candidate(states[1], -20.0, 1.0e-4),
        states[2]: candidate(states[2], -30.0, 1.0e-6),
    }
    backward = progress_filter.update(prior, backward_scores, states[1])
    assert backward.estimated_state == states[0]
    assert backward.status == ProgressStatus.BACKWARD_REALIGNMENT

    forward_scores = {
        states[0]: candidate(states[0], -30.0, 1.0e-6),
        states[1]: candidate(states[1], -20.0, 1.0e-4),
        states[2]: candidate(states[2], 0.0, 1.0),
    }
    forward = progress_filter.update(prior, forward_scores, states[1])
    assert forward.estimated_state == states[2]
    assert forward.status == ProgressStatus.FORWARD_REALIGNMENT


def relation_estimate(
    frame: str,
    posterior: tuple[float, float],
    decision: RelationDecision,
) -> RelationEstimate:
    values = np.asarray(posterior, dtype=np.float64)
    return RelationEstimate(
        frame_id=frame,
        posterior=values,
        predicted=values,
        demonstration_prior=np.asarray([0.5, 0.5]),
        observation_likelihood=np.ones(2),
        information_weight=1.0,
        entropy=-float(np.sum(values * np.log(values))),
        informative=True,
        decision_state=decision,
    )


def test_linked_stream_precision_is_suppressed_in_progress_score(phase2_model) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    runtime = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[1],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    features = RuntimeFeatureBuilder().build(runtime)
    score = StateEvaluator(model).evaluate(
        StateId(0, 1),
        features,
        {
            "object": relation_estimate(
                "object", (0.001, 0.999), RelationDecision.LINKED
            )
        },
        mode_by_skill={0: 0},
    )
    assert score.robot_frame_weights["object"] == pytest.approx(0.001)
    assert math.isfinite(score.robot_log_likelihood)


def test_node_and_edge_scene_factors_use_the_unified_observation(phase2_model) -> None:
    model, _ = phase2_model
    node = model.state(StateId(0, 1))
    original = node.scene_factor_models
    node_factor = FactorId("node", "object", feature="joint_position")
    edge_factor = FactorId("edge", "object", "target")
    node.scene_factor_models = {
        node_factor: {
            0: FactorDistribution(
                mean=np.asarray([0.25]),
                covariance=np.asarray([[0.01]]),
                sample_count=5,
                space="euclidean",
            )
        },
        edge_factor: {
            0: FactorDistribution(
                mean=pose(-0.4),
                covariance=np.eye(6) * 0.01,
                sample_count=5,
            )
        },
    }
    try:
        previous = RuntimeObservation(
            tick=0,
            ee_pose=pose(0.0),
            frame_poses={"object": pose(0.4), "target": pose(0.8)},
            gripper_state=np.asarray([1.0]),
            previous_command_pose=None,
            previous_ee_pose=None,
            tracking_reliability={"object": 1.0, "target": 1.0},
            frame_visibility={"object": True, "target": True},
            entity_configurations={"object": {"joint_position": np.asarray([0.25])}},
        )
        current = RuntimeObservation(
            tick=1,
            ee_pose=pose(0.02),
            frame_poses={"object": pose(0.4), "target": pose(0.8)},
            gripper_state=np.asarray([1.0]),
            previous_command_pose=pose(0.02),
            previous_ee_pose=pose(0.0),
            tracking_reliability={"object": 1.0, "target": 1.0},
            frame_visibility={"object": True, "target": True},
            entity_configurations={"object": {"joint_position": np.asarray([0.25])}},
        )
        features = RuntimeFeatureBuilder().build(current, previous)
        score = StateEvaluator(model).evaluate(
            StateId(0, 1),
            features,
            {
                "object": relation_estimate(
                    "object", (0.999, 0.001), RelationDecision.EXTERNAL
                )
            },
            mode_by_skill={0: 0},
        )
        assert score.scene_evidence_expected is True
        assert score.scene_evidence_available is True
        assert set(score.scene_factor_terms) == {node_factor, edge_factor}
        assert score.state_compatibility == pytest.approx(1.0)
        assert score.state_log_support == pytest.approx(0.0)
        assert score.state_log_likelihood != pytest.approx(0.0)
        for factor_audits in score.scene_factor_gaussian_audits.values():
            assert len(factor_audits) == 1
            audit = factor_audits[0]
            assert audit.normalized_log_support == pytest.approx(
                -0.5 * audit.mahalanobis_squared
            )
            assert audit.raw_log_likelihood == pytest.approx(
                audit.normalized_log_support
                - 0.5
                * (
                    audit.covariance_log_determinant
                    + audit.dimension * math.log(2.0 * math.pi)
                )
            )
    finally:
        node.scene_factor_models = original


def test_belief_updater_is_single_pass_and_auditable(phase2_model) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    previous = RuntimeObservation(
        tick=0,
        ee_pose=demo.ee_pose[0],
        frame_poses={"object": demo.frames["object"][0]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=None,
        previous_ee_pose=None,
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    updater = BeliefUpdater(model)
    updater.reset(previous_observation=previous)
    current = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[1],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    belief = updater.update(current, mode_by_skill={0: 0, 1: 0})
    assert belief.update_sequence == (
        "progress_prior",
        "relation_posterior",
        "progress_posterior",
    )
    assert np.sum(belief.relation_posteriors["object"]) == pytest.approx(1.0)
    assert set(belief.candidate_scores) == set(belief.progress.prior)
    for score in belief.candidate_scores.values():
        assert math.isfinite(score.robot_log_likelihood)
        assert math.isfinite(score.state_log_likelihood)
        assert math.isfinite(score.robot_log_support)
        assert math.isfinite(score.state_log_support)
        assert math.isfinite(score.relation_log_compatibility)
        assert score.explanation_log_score == pytest.approx(
            score.robot_log_support
            + updater.config.state_evaluator.scene_weight * score.state_log_support
            + updater.config.state_evaluator.relation_weight
            * score.relation_log_compatibility
        )
    with pytest.raises(ValueError, match="每个递增控制周期只能更新一次"):
        updater.update(current, mode_by_skill={0: 0, 1: 0})


def test_tracked_phase_two_config_matches_code_defaults() -> None:
    path = Path("configs/closed_loop_belief.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == asdict(BeliefUpdaterConfig())
    assert BeliefUpdaterConfig.from_json(path).to_dict() == payload


def carry_updater(model, demo):
    previous = RuntimeObservation(
        tick=0,
        ee_pose=demo.ee_pose[6],
        frame_poses={"object": demo.frames["object"][6]},
        gripper_state=np.asarray([-1.0]),
        previous_command_pose=None,
        previous_ee_pose=None,
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    updater = BeliefUpdater(model)
    updater.reset(
        initial_progress={StateId(1, 0): 1.0},
        initial_relations={"object": np.asarray([0.001, 0.999])},
        previous_observation=previous,
    )
    current = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[7],
        frame_poses={"object": demo.frames["object"][7]},
        gripper_state=np.asarray([-1.0]),
        previous_command_pose=demo.action_pose[7],
        previous_ee_pose=demo.ee_pose[6],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    belief = updater.update(current, mode_by_skill={0: 0, 1: 0, 2: 0})
    assert belief.relation_estimates["object"].decision_state == RelationDecision.LINKED
    return updater


def test_reliable_early_release_can_expand_to_matching_future_segment(
    phase2_model,
) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    updater = carry_updater(model, demo)
    release = RuntimeObservation(
        tick=2,
        ee_pose=demo.ee_pose[9],
        frame_poses={"object": demo.frames["object"][9]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[9],
        previous_ee_pose=demo.ee_pose[7],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    boundary = BoundaryId(model.arm_id, 1, 2)
    belief = updater.update(
        release,
        permitted_boundaries=frozenset({boundary}),
        mode_by_skill={0: 0, 1: 0, 2: 0},
    )
    assert belief.relation_changes[0].previous == RelationDecision.LINKED
    assert belief.relation_changes[0].current == RelationDecision.EXTERNAL
    assert StateId(2, 1) in belief.expanded_candidates
    assert belief.progress.estimated_state == StateId(2, 1)
    assert belief.progress.status == ProgressStatus.FORWARD_REALIGNMENT


def test_drop_during_carry_does_not_jump_to_release_on_relation_alone(
    phase2_model,
) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    updater = carry_updater(model, demo)
    previous_ee = demo.ee_pose[7]
    last_belief = None
    for tick, height in ((2, 0.2), (3, 0.4)):
        abnormal_ee = pose_compose(demo.ee_pose[7], pose(0.0, 0.0, height))
        dropped = RuntimeObservation(
            tick=tick,
            ee_pose=abnormal_ee,
            frame_poses={"object": demo.frames["object"][7]},
            gripper_state=np.asarray([-1.0]),
            previous_command_pose=abnormal_ee,
            previous_ee_pose=previous_ee,
            tracking_reliability={"object": 1.0},
            frame_visibility={"object": True},
        )
        last_belief = updater.update(
            dropped,
            permitted_boundaries=frozenset(),
            mode_by_skill={0: 0, 1: 0, 2: 0},
        )
        previous_ee = abnormal_ee
    assert last_belief is not None
    assert (
        last_belief.relation_estimates["object"].decision_state
        == RelationDecision.EXTERNAL
    )
    assert last_belief.relation_changes
    assert last_belief.expanded_candidates == ()
    assert last_belief.progress.estimated_state.skill_index == 1
    assert last_belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE
