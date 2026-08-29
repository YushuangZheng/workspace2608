"""Phase-two acceptance tests for relation-progress belief inference."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
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
    ProgressPriorConfig,
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


def test_runtime_features_measure_previous_poe_target_completion() -> None:
    builder = RuntimeFeatureBuilder()
    exact = RuntimeObservation(
        tick=1,
        ee_pose=pose(0.1),
        frame_poses={"object": pose(0.4)},
        gripper_state=np.asarray([-1.0]),
        previous_command_pose=pose(0.1),
        previous_ee_pose=pose(0.0),
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
        previous_command_covariance=np.eye(6) * 0.01,
    )
    missed = RuntimeObservation(
        tick=1,
        ee_pose=pose(0.0),
        frame_poses={"object": pose(0.4)},
        gripper_state=np.asarray([-1.0]),
        previous_command_pose=pose(0.1),
        previous_ee_pose=pose(0.0),
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
        previous_command_covariance=np.eye(6) * 0.01,
    )
    exact_features = builder.build(exact)
    missed_features = builder.build(missed)
    assert exact_features.command_tracking_available is True
    assert exact_features.command_tracking_compatibility == pytest.approx(1.0)
    assert missed_features.command_tracking_mahalanobis_squared == pytest.approx(1.0)
    assert missed_features.command_tracking_compatibility == pytest.approx(
        math.exp(-0.5 / 6.0)
    )


def test_runtime_motion_and_gaussian_support_ignore_quaternion_antipodes(
    phase2_model,
) -> None:
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
    flipped_frame = demo.frames["object"][1].copy()
    flipped_frame[3:7] *= -1.0
    flipped = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": flipped_frame},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    canonical = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    builder = RuntimeFeatureBuilder()
    canonical_features = builder.build(canonical, previous)
    flipped_features = builder.build(flipped, previous)
    np.testing.assert_allclose(
        flipped_features.relative_motion_residuals["object"],
        canonical_features.relative_motion_residuals["object"],
        atol=1.0e-12,
    )
    relation = {
        "object": relation_estimate("object", (0.999, 0.001), RelationDecision.EXTERNAL)
    }
    evaluator = StateEvaluator(model)
    canonical_score = evaluator.evaluate(
        StateId(0, 1), canonical_features, relation, mode_by_skill={0: 0}
    )
    flipped_score = evaluator.evaluate(
        StateId(0, 1), flipped_features, relation, mode_by_skill={0: 0}
    )
    assert flipped_score.robot_log_support == pytest.approx(
        canonical_score.robot_log_support
    )
    assert flipped_score.robot_compatibility == pytest.approx(
        canonical_score.robot_compatibility
    )

    distribution = FactorDistribution(
        mean=pose(0.1), covariance=np.eye(6) * 0.01, sample_count=5
    )
    antipode = distribution.mean.copy()
    antipode[3:7] *= -1.0
    assert distribution.compatibility(antipode) == pytest.approx(1.0)
    assert distribution.log_likelihood(antipode) == pytest.approx(
        distribution.log_likelihood(distribution.mean)
    )


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


def test_relation_filter_bounds_one_cycle_outliers_but_accepts_sustained_motion(
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
    initial = observation(0, 0.0, 0.4, previous_ee_x=None)
    carried_observation = observation(1, 0.1, 0.5, previous_ee_x=0.0)
    carried = feature_builder.build(carried_observation, initial)
    confirmed = relation_filter.update(
        progress, carried, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert confirmed.decision_state == RelationDecision.LINKED

    # A small arm motion coincident with a large relative-pose impulse is a
    # plausible tracking/contact outlier.  It must not erase a confirmed link
    # in one cycle even though its instantaneous evidence favours external.
    contact_observation = observation(2, 0.1025, 0.51, previous_ee_x=0.1)
    contact = feature_builder.build(contact_observation, carried_observation)
    transient = relation_filter.update(
        progress,
        contact,
        {"object": confirmed.posterior},
        previous_decisions={"object": RelationDecision.LINKED},
    )["object"]
    assert transient.observation_likelihood[0] > transient.observation_likelihood[1]
    assert transient.decision_state == RelationDecision.LINKED

    # A subsequent fully excited arm motion without object response remains
    # strong enough to overturn the old relation and detect a real disconnect.
    dropped_observation = observation(3, 0.2025, 0.51, previous_ee_x=0.1025)
    dropped = feature_builder.build(dropped_observation, contact_observation)
    disconnected = relation_filter.update(
        progress,
        dropped,
        {"object": transient.posterior},
        previous_decisions={"object": RelationDecision.LINKED},
    )["object"]
    assert disconnected.decision_state == RelationDecision.EXTERNAL


def test_relation_filter_keeps_mixed_component_contact_wobble_ambiguous(
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
    initial = observation(0, 0.0, 0.4, previous_ee_x=None)
    carried_observation = observation(1, 0.1, 0.5, previous_ee_x=0.0)
    carried = feature_builder.build(carried_observation, initial)
    confirmed = relation_filter.update(
        progress, carried, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert confirmed.decision_state == RelationDecision.LINKED

    # This is a translation-dominant, grasp-like motion with small relative
    # translation residual but a compliant rotational wobble.  Translation
    # supports co-motion while rotation supports a disconnect.  The two
    # action-conditioned components must produce bounded ambiguous evidence,
    # not erase an already motion-confirmed link as one scalar residual did.
    actual_motion = np.asarray([0.00559, 0.0, 0.0, 0.0, 0.0, 0.0123], dtype=np.float64)
    mixed_residual = np.asarray(
        [0.00064, 0.0, 0.0, 0.0, 0.0, 0.02155], dtype=np.float64
    )
    mixed = replace(
        carried,
        actual_ee_motion=actual_motion,
        relative_motion_residuals={"object": mixed_residual},
        actual_motion_magnitude=feature_builder.motion_magnitude(actual_motion),
        action_excitation=0.529,
        relation_information_weight={"object": 0.529},
    )
    previous = confirmed
    for _ in range(4):
        estimate = relation_filter.update(
            progress,
            mixed,
            {"object": previous.posterior},
            previous_decisions={"object": RelationDecision.LINKED},
            previous_evidence_decisions={"object": RelationDecision.LINKED},
        )["object"]
        assert estimate.decision_state != RelationDecision.EXTERNAL
        previous = estimate

    # By contrast, a fully excited translation without frame response remains
    # decisive external evidence; the component split does not hide a drop.
    dropped_observation = observation(2, 0.2, 0.5, previous_ee_x=0.1)
    dropped = feature_builder.build(dropped_observation, carried_observation)
    disconnected = relation_filter.update(
        progress,
        dropped,
        {"object": previous.posterior},
        previous_decisions={"object": RelationDecision.LINKED},
        previous_evidence_decisions={"object": RelationDecision.LINKED},
    )["object"]
    assert disconnected.decision_state == RelationDecision.EXTERNAL


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


def test_confirmed_relation_persists_only_through_reliable_static_cycle(
    phase2_model,
) -> None:
    model, _ = phase2_model
    builder = RuntimeFeatureBuilder()
    relation_filter = RelationFilter(
        model,
        RelationFilterConfig(demonstration_prior_strength=0.0),
        feature_builder=builder,
    )
    progress = {StateId(0, 1): 1.0}
    previous = observation(0, 0.0, 0.4, previous_ee_x=None)
    moving = builder.build(observation(1, 0.1, 0.5, previous_ee_x=0.0), previous)
    confirmed = relation_filter.update(
        progress, moving, {"object": np.asarray([0.5, 0.5])}
    )["object"]
    assert confirmed.decision_state == RelationDecision.LINKED
    assert confirmed.informative is True

    static_previous = observation(1, 0.1, 0.5, previous_ee_x=0.0)
    static = builder.build(observation(2, 0.1, 0.5, previous_ee_x=0.1), static_previous)
    persisted = relation_filter.update(
        progress,
        static,
        {"object": confirmed.posterior},
        previous_decisions={"object": confirmed.decision_state},
    )["object"]
    assert persisted.informative is False
    assert persisted.decision_state == RelationDecision.LINKED

    unconfirmed = relation_filter.update(
        progress, static, {"object": confirmed.posterior}
    )["object"]
    assert unconfirmed.decision_state == RelationDecision.UNKNOWN

    hidden_static = builder.build(
        observation(2, 0.1, 0.5, previous_ee_x=0.1, visible=False),
        static_previous,
    )
    hidden = relation_filter.update(
        progress,
        hidden_static,
        {"object": confirmed.posterior},
        previous_decisions={"object": confirmed.decision_state},
    )["object"]
    assert hidden.decision_state == RelationDecision.UNKNOWN


def test_static_cycle_can_finish_only_an_evidence_supported_confirmation(
    phase2_model,
) -> None:
    model, _ = phase2_model
    builder = RuntimeFeatureBuilder()
    relation_filter = RelationFilter(
        model,
        RelationFilterConfig(demonstration_prior_strength=0.0),
        feature_builder=builder,
    )
    progress = {StateId(0, 1): 1.0}
    previous = observation(0, 0.1, 0.5, previous_ee_x=None)
    static = builder.build(observation(1, 0.1, 0.5, previous_ee_x=0.1), previous)
    posterior = {"object": np.asarray([0.10, 0.90])}

    unsupported = relation_filter.update(progress, static, posterior)["object"]
    assert unsupported.informative is False
    assert unsupported.decision_state == RelationDecision.UNKNOWN

    supported = relation_filter.update(
        progress,
        static,
        posterior,
        previous_evidence_decisions={"object": RelationDecision.LINKED},
    )["object"]
    assert supported.informative is False
    assert supported.decision_state == RelationDecision.LINKED


def test_static_hold_does_not_let_demo_prior_erase_confirmed_relation(
    phase2_model,
) -> None:
    model, _ = phase2_model
    builder = RuntimeFeatureBuilder()
    relation_filter = RelationFilter(model, feature_builder=builder)
    progress = {StateId(1, 1): 1.0}
    previous_observation = observation(0, 0.1, 0.5, previous_ee_x=None)
    static = builder.build(
        observation(1, 0.1, 0.5, previous_ee_x=0.1),
        previous_observation,
    )
    posterior = np.asarray([0.92, 0.08], dtype=np.float64)

    # This fixture state's demo prior supports linked.  During a deliberate
    # HOLD there is nevertheless no new physical evidence that an observed
    # external relation has relinked.  Five quiet recursive updates must keep
    # the confirmed direction instead of repeatedly treating the same demo
    # prior as five fresh observations.
    assert relation_filter.demonstration_prior(progress, "object")[1] > 0.5
    for _ in range(5):
        estimate = relation_filter.update(
            progress,
            static,
            {"object": posterior},
            previous_decisions={"object": RelationDecision.EXTERNAL},
            previous_evidence_decisions={"object": RelationDecision.EXTERNAL},
        )["object"]
        assert estimate.informative is False
        assert estimate.decision_state == RelationDecision.EXTERNAL
        posterior = estimate.posterior


def test_long_static_hold_preserves_motion_confirmed_decision_despite_soft_diffusion(
    phase2_model,
) -> None:
    model, _ = phase2_model
    builder = RuntimeFeatureBuilder()
    relation_filter = RelationFilter(
        model,
        RelationFilterConfig(demonstration_prior_strength=0.0),
        feature_builder=builder,
    )
    progress = {StateId(0, 1): 1.0}
    previous_observation = observation(0, 0.1, 0.5, previous_ee_x=None)
    static = builder.build(
        observation(1, 0.1, 0.5, previous_ee_x=0.1),
        previous_observation,
    )
    posterior = np.asarray([0.05, 0.95], dtype=np.float64)

    # A Markov posterior approaches 0.5 when no new action-conditioned
    # evidence arrives.  That uncertainty must not turn an already
    # motion-confirmed relation into Unknown merely because the task holds.
    for _ in range(300):
        estimate = relation_filter.update(
            progress,
            static,
            {"object": posterior},
            previous_decisions={"object": RelationDecision.LINKED},
            previous_evidence_decisions={"object": RelationDecision.LINKED},
        )["object"]
        assert estimate.informative is False
        assert estimate.decision_state == RelationDecision.LINKED
        posterior = estimate.posterior
    assert posterior[1] < relation_filter.config.decision_probability


def test_only_stable_informative_decision_replaces_relation_confirmation(
    phase2_model,
) -> None:
    model, _ = phase2_model
    updater = BeliefUpdater(model)
    updater._stable_decisions = {"object": RelationDecision.LINKED}
    updater._informative_evidence_decisions = {"object": RelationDecision.LINKED}
    previous = observation(0, 0.0, 0.4, previous_ee_x=None)
    features = RuntimeFeatureBuilder().build(
        observation(1, 0.1, 0.5, previous_ee_x=0.0), previous
    )

    # Contact may transiently favour external even while the persistent
    # posterior still accepts the established linked state.  That raw
    # direction is not a new confirmed relation and must not overwrite the
    # evidence used to bridge a later quiet cycle.
    transient = RelationEstimate(
        frame_id="object",
        posterior=np.asarray([0.2, 0.8]),
        predicted=np.asarray([0.1, 0.9]),
        demonstration_prior=np.asarray([0.3, 0.7]),
        observation_likelihood=np.asarray([0.99, 0.01]),
        information_weight=1.0,
        entropy=-float(np.sum(np.asarray([0.2, 0.8]) * np.log([0.2, 0.8]))),
        informative=True,
        decision_state=RelationDecision.LINKED,
        informative_evidence_direction=RelationDecision.EXTERNAL,
    )
    updater._commit_informative_evidence(features, {"object": transient})
    assert updater._informative_evidence_decisions["object"] == RelationDecision.LINKED

    # A visible and reliable Unknown is posterior uncertainty, not an
    # observation gap.  Preserve the last stable physical state so a future
    # genuinely informative opposite decision can still be reported as a
    # relation change.
    uncertain = RelationEstimate(
        frame_id="object",
        posterior=np.asarray([0.5, 0.5]),
        predicted=np.asarray([0.5, 0.5]),
        demonstration_prior=np.asarray([0.3, 0.7]),
        observation_likelihood=np.ones(2),
        information_weight=0.0,
        entropy=math.log(2.0),
        informative=False,
        decision_state=RelationDecision.UNKNOWN,
    )
    assert updater._relation_changes(features, {"object": uncertain}) == ()
    assert updater._stable_decisions["object"] == RelationDecision.LINKED

    confirmed_external = RelationEstimate(
        frame_id="object",
        posterior=np.asarray([0.9, 0.1]),
        predicted=np.asarray([0.6, 0.4]),
        demonstration_prior=np.asarray([0.7, 0.3]),
        observation_likelihood=np.asarray([0.99, 0.01]),
        information_weight=1.0,
        entropy=-float(np.sum(np.asarray([0.9, 0.1]) * np.log([0.9, 0.1]))),
        informative=True,
        decision_state=RelationDecision.EXTERNAL,
        informative_evidence_direction=RelationDecision.EXTERNAL,
    )
    changes = updater._relation_changes(features, {"object": confirmed_external})
    assert len(changes) == 1
    assert changes[0].previous == RelationDecision.LINKED
    assert changes[0].current == RelationDecision.EXTERNAL


def test_relation_does_not_resurrect_across_visibility_gap_without_new_motion(
    phase2_model,
) -> None:
    model, _ = phase2_model
    updater = BeliefUpdater(
        model,
        BeliefUpdaterConfig(
            relation_filter=RelationFilterConfig(demonstration_prior_strength=0.0)
        ),
    )
    state = StateId(0, 1)
    updater.reset(
        initial_progress={state: 1.0},
        initial_relations={"object": np.asarray([0.05, 0.95])},
        initial_relation_decisions={"object": RelationDecision.LINKED},
        previous_observation=observation(0, 0.1, 0.5, previous_ee_x=None),
    )
    hidden = updater.update(
        observation(1, 0.1, 0.5, previous_ee_x=0.1, visible=False),
        executed_reference_state=state,
        mode_by_skill={0: 0, 1: 0},
    )
    assert (
        hidden.relation_estimates["object"].decision_state == RelationDecision.UNKNOWN
    )

    reappeared = updater.update(
        observation(2, 0.1, 0.5, previous_ee_x=0.1, visible=True),
        executed_reference_state=state,
        mode_by_skill={0: 0, 1: 0},
    )
    assert (
        reappeared.relation_estimates["object"].decision_state
        == RelationDecision.UNKNOWN
    )


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


def test_progress_prior_is_conditioned_on_the_action_reference(phase2_model) -> None:
    model, _ = phase2_model
    builder = ProgressPriorBuilder(
        model,
        ProgressPriorConfig(local_backward_radius=2),
    )
    posterior = {StateId(0, 0): 0.75, StateId(0, 1): 0.25}
    action_reference = StateId(0, 1)
    anchored = builder.build(
        posterior,
        executed_reference_state=action_reference,
    )
    assert anchored.nominal_state == action_reference
    assert anchored.probabilities[StateId(0, 0)] == pytest.approx(0.15)
    assert anchored.probabilities[action_reference] == pytest.approx(0.70)
    assert anchored.probabilities[StateId(0, 2)] == pytest.approx(0.15)


def test_progress_prior_does_not_advance_without_an_executed_action(
    phase2_model,
) -> None:
    model, _ = phase2_model
    state = StateId(0, 0)
    prior = ProgressPriorBuilder(model).build(
        {state: 1.0},
        action_executed=False,
    )
    assert prior.nominal_state == state
    assert prior.probabilities == {state: 1.0}

    with pytest.raises(ValueError, match="未执行动作"):
        ProgressPriorBuilder(model).build(
            {state: 1.0},
            executed_reference_state=state,
            action_executed=False,
        )


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
        robot_unadjusted_log_support=log_score,
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
        previous_command_pose=demo.action_pose[0],
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


def test_unknown_relation_keeps_soft_external_robot_evidence_when_observable(
    phase2_model,
) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    state_id = StateId(0, 1)
    estimate = {
        "object": relation_estimate("object", (0.8, 0.2), RelationDecision.UNKNOWN)
    }
    visible = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 0.5},
        frame_visibility={"object": True},
    )
    visible_score = StateEvaluator(model).evaluate(
        state_id,
        RuntimeFeatureBuilder().build(visible),
        estimate,
        mode_by_skill={0: 0},
    )
    assert visible_score.robot_frame_weights["object"] == pytest.approx(0.4)
    # The hard Unknown decision still cannot masquerade as discrete relation
    # evidence; only the observable robot trajectory uses the soft posterior.
    assert "object" not in visible_score.relation_frame_weights

    hidden = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 0.5},
        frame_visibility={"object": False},
    )
    hidden_score = StateEvaluator(model).evaluate(
        state_id,
        RuntimeFeatureBuilder().build(hidden),
        estimate,
        mode_by_skill={0: 0},
    )
    assert "object" not in hidden_score.robot_frame_weights

    unreliable = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 0.0},
        frame_visibility={"object": True},
    )
    unreliable_score = StateEvaluator(model).evaluate(
        state_id,
        RuntimeFeatureBuilder().build(unreliable),
        estimate,
        mode_by_skill={0: 0},
    )
    assert "object" not in unreliable_score.robot_frame_weights


def test_absolute_explanation_removes_soft_relation_prior_peak_scale(
    phase2_model,
) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    state_id = StateId(0, 1)
    node = model.state(state_id)
    original = node.demo_relation_priors["object"].copy()
    runtime = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1]},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 1.0},
        frame_visibility={"object": True},
    )
    features = RuntimeFeatureBuilder().build(runtime)
    estimate = {
        "object": relation_estimate("object", (0.15, 0.85), RelationDecision.LINKED)
    }
    try:
        node.demo_relation_priors["object"][0] = np.asarray([0.3, 0.7])
        matching = StateEvaluator(model).evaluate(
            state_id, features, estimate, mode_by_skill={0: 0}
        )
        assert matching.relation_compatibility == pytest.approx(0.64)
        assert matching.relation_peak_normalized_compatibility == pytest.approx(
            0.64 / 0.7
        )
        assert matching.normalized_explanation_score == pytest.approx(
            matching.robot_compatibility * matching.state_compatibility * (0.64 / 0.7)
        )
        assert matching.relation_log_compatibility == pytest.approx(math.log(0.64))

        node.demo_relation_priors["object"][0] = np.asarray([0.7, 0.3])
        opposing = StateEvaluator(model).evaluate(
            state_id, features, estimate, mode_by_skill={0: 0}
        )
        assert opposing.relation_compatibility == pytest.approx(0.36)
        assert opposing.relation_peak_normalized_compatibility == pytest.approx(
            0.36 / 0.7
        )
        assert (
            matching.relation_compatibility / opposing.relation_compatibility
        ) == pytest.approx(
            matching.relation_peak_normalized_compatibility
            / opposing.relation_peak_normalized_compatibility
        )

        node.demo_relation_priors["object"][0] = np.asarray([0.5, 0.5])
        ambiguous = StateEvaluator(model).evaluate(
            state_id, features, estimate, mode_by_skill={0: 0}
        )
        assert ambiguous.relation_compatibility == pytest.approx(0.5)
        assert ambiguous.relation_peak_normalized_compatibility == pytest.approx(1.0)
        # The original overlap remains in the posterior score; only the
        # cross-family absolute explanation uses the attainable-peak scale.
        assert ambiguous.relation_log_compatibility == pytest.approx(math.log(0.5))
    finally:
        node.demo_relation_priors["object"] = original


def test_absolute_explanation_uses_jointly_attainable_multiflow_peak(
    phase2_model,
) -> None:
    model, demonstrations = phase2_model
    demo = demonstrations[0]
    state_id = StateId(0, 1)
    node = model.state(state_id)
    # Duplicate the retained stream with an intentionally displaced local
    # mean.  One EE pose cannot sit at both individual expert means, while the
    # weighted PoE still has a well-defined attainable joint optimum.
    node.selected_frames = ("object", "second")
    node.mode_selected_frames = (("object", "second"),)
    node.stream_means["second"] = node.stream_means["object"].copy()
    node.stream_means["second"][0, 0] += 0.03
    node.stream_covariances["second"] = node.stream_covariances["object"].copy()
    second_frame = demo.frames["object"][1].copy()
    runtime = RuntimeObservation(
        tick=1,
        ee_pose=demo.ee_pose[1],
        frame_poses={"object": demo.frames["object"][1], "second": second_frame},
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.action_pose[0],
        previous_ee_pose=demo.ee_pose[0],
        tracking_reliability={"object": 1.0, "second": 1.0},
        frame_visibility={"object": True, "second": True},
    )
    features = RuntimeFeatureBuilder().build(runtime)
    relations = {
        "object": relation_estimate(
            "object", (0.999, 0.001), RelationDecision.EXTERNAL
        ),
        "second": relation_estimate(
            "second", (0.999, 0.001), RelationDecision.EXTERNAL
        ),
    }
    score = StateEvaluator(model).evaluate(
        state_id, features, relations, mode_by_skill={0: 0}
    )
    assert score.robot_compatibility < score.robot_peak_normalized_compatibility
    assert 0.0 < score.robot_peak_normalized_compatibility <= 1.0
    assert score.robot_attainable_peak_log_support < 0.0
    assert score.robot_log_support > score.robot_unadjusted_log_support
    assert score.explanation_log_score == pytest.approx(
        score.robot_log_support
        + score.state_log_support
        + score.relation_log_compatibility
    )


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
        previous_command_pose=demo.action_pose[0],
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
        previous_command_pose=demo.action_pose[6],
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
        previous_command_pose=demo.action_pose[7],
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
    relation_changes = []
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
        relation_changes.extend(last_belief.relation_changes)
        previous_ee = abnormal_ee
    assert last_belief is not None
    assert (
        last_belief.relation_estimates["object"].decision_state
        == RelationDecision.EXTERNAL
    )
    assert relation_changes
    assert last_belief.expanded_candidates == ()
    assert last_belief.progress.estimated_state.skill_index == 1
    assert last_belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE
