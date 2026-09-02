"""Phase-one acceptance tests for the unified offline task model."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
    GaussianMarginal,
    product_of_experts,
)
from essay2608.policy.closed_loop import (
    BoundaryId,
    ClosedLoopTaskModel,
    ClosedLoopTaskModelBuilder,
    ClosedLoopTaskModelConfig,
    FactorId,
    LinkPendingCandidate,
    RelationEventId,
    StateId,
)
from essay2608.policy.closed_loop.task_model_builder import (
    _fit_pose_samples,
    _relation_prior,
)
from essay2608.policy.dynamac import (
    pose_compose,
    pose_inverse,
    synchronized_bimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec


def pose(position: list[float], yaw: float = 0.0) -> np.ndarray:
    return np.asarray(
        [*position, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float64,
    )


def demonstrations(
    *,
    right_arm: bool = False,
    with_scene_metadata: bool = False,
) -> list[DynaMACDemonstration]:
    result = []
    duration = 6
    for demo_index in range(5):
        ee = []
        objects = []
        targets = []
        gripper = []
        object_start = pose(
            [0.42 + 0.018 * demo_index, -0.12 + 0.01 * demo_index, 0.05],
            yaw=0.03 * demo_index,
        )
        target = pose([0.67, 0.18 + 0.006 * demo_index, 0.12])
        for index in range(duration):
            progress = index / (duration - 1)
            local = pose(
                [
                    -0.12 * (1.0 - progress) + 0.014 * demo_index,
                    0.012 * demo_index,
                    0.10 * (1.0 - progress),
                ]
            )
            ee_pose = pose_compose(object_start, local)
            ee.append(ee_pose)
            objects.append(object_start)
            targets.append(target)
            gripper.append(1.0)
        grasp_offset = pose([0.0, 0.0, -0.045])
        carry_start = ee[-1]
        for index in range(duration):
            progress = index / (duration - 1)
            ee_pose = pose_compose(
                carry_start,
                pose([0.08 * progress, 0.10 * progress, 0.08 * progress]),
            )
            ee.append(ee_pose)
            objects.append(pose_compose(ee_pose, pose_inverse(grasp_offset)))
            targets.append(target)
            gripper.append(-1.0)
        released_object = objects[-1]
        release_start = ee[-1]
        for index in range(duration):
            progress = index / (duration - 1)
            ee_pose = pose_compose(
                release_start,
                pose(
                    [
                        0.04 * progress + 0.05 * demo_index * progress,
                        -0.03 * progress + 0.04 * demo_index * progress,
                        0.06 * progress + 0.03 * demo_index * progress,
                    ]
                ),
            )
            ee.append(ee_pose)
            objects.append(released_object)
            targets.append(target)
            gripper.append(1.0)
        ee_array = np.stack(ee)
        if right_arm:
            ee_array = ee_array.copy()
            ee_array[:, 1] += 0.22
        result.append(
            DynaMACDemonstration(
                ee_pose=ee_array,
                action_pose=ee_array.copy(),
                gripper=np.asarray(gripper),
                frames={"object": np.stack(objects), "target": np.stack(targets)},
                skill=np.repeat([0, 1, 2], duration),
                name=("right" if right_arm else "left") + f"_{demo_index}",
                entity_configurations=(
                    {
                        "object": {
                            "joint_position": np.concatenate(
                                (
                                    np.zeros((duration, 1)),
                                    np.linspace(0.0, 1.0, duration)[:, None]
                                    + 0.004 * demo_index,
                                    np.ones((duration, 1)),
                                ),
                                axis=0,
                            )
                        }
                    }
                    if with_scene_metadata
                    else {}
                ),
                scene_entity_poses=(
                    {
                        "structure": np.stack(targets),
                        "root": np.stack(targets),
                    }
                    if with_scene_metadata
                    else {}
                ),
                structural_bindings=(
                    {"object": "structure", "structure": "root"}
                    if with_scene_metadata
                    else {}
                ),
            )
        )
    return result


def config(**changes) -> DynaMACConfig:
    values = {
        "tau_m": 0.005,
        "tau_omega": 0.0,
        "eq6_empty_selection": "keep_argmax",
        "kinematic_analysis_enabled": False,
        "link_mask_scope": "timestep",
        "default_mode_strategy": "map",
    }
    values.update(changes)
    return DynaMACConfig(**values)


def fitted_policy() -> tuple[DynaMAC, list[DynaMACDemonstration]]:
    demos = demonstrations()
    return DynaMAC(config()).fit(demos), demos


def first_observation(demos: list[DynaMACDemonstration]) -> DynaMACObservation:
    first = demos[0]
    return DynaMACObservation(
        ee_pose=first.ee_pose[0],
        frames={name: value[0] for name, value in first.frames.items()},
    )


def test_precision_weights_scale_or_remove_experts() -> None:
    first = GaussianMarginal("first", pose([0.0, 0.0, 0.0]), np.eye(6) * 0.1)
    second = GaussianMarginal("second", pose([0.2, 0.0, 0.0]), np.eye(6) * 0.1)
    baseline = product_of_experts([first, second])
    all_one = product_of_experts([first, second], precision_weights=[1.0, 1.0])
    np.testing.assert_array_equal(all_one[0], baseline[0])
    np.testing.assert_array_equal(all_one[1], baseline[1])

    first_only = product_of_experts([first, second], precision_weights=[1.0, 0.0])
    np.testing.assert_allclose(first_only[0], first.mean)
    np.testing.assert_allclose(first_only[1], first.covariance)
    weighted = product_of_experts([first, second], precision_weights=[3.0, 1.0])
    assert weighted[0][0] < baseline[0][0]


def test_query_state_is_read_only_and_act_retains_the_fixed_clock() -> None:
    policy, demos = fitted_policy()
    observation = first_observation(demos)
    policy.reset(observation, mode_strategy="map")
    before = policy._capture_runtime_state()
    state_id = StateId(0, 0)
    queried = policy.query_state(observation, state_id)
    after = policy._capture_runtime_state()
    assert after["skill_index"] == before["skill_index"]
    assert after["time_index"] == before["time_index"]
    assert after["virtual_frames"].keys() == before["virtual_frames"].keys()
    for name in before["virtual_frames"]:
        np.testing.assert_array_equal(
            after["virtual_frames"][name], before["virtual_frames"][name]
        )

    weighted = policy.query_state(
        observation,
        state_id,
        {name: 1.0 for name in policy.skills[0].selected_frames},
    )
    assert weighted.diagnostics["query_advances_clock"] is False
    assert policy._time_index == 0
    acted = policy.act(observation)
    np.testing.assert_array_equal(acted.pose, queried.pose)
    np.testing.assert_array_equal(acted.covariance, queried.covariance)
    assert policy._time_index == 1

    # Arbitrary later states are queryable when the controller supplies the
    # virtual frames captured in its own runtime snapshot.
    future_skill = policy.skills[2]
    future_frames = dict(observation.frames)
    for skill_index, skill in enumerate(policy.skills):
        future_frames[f"virtual_skill_{skill.label}"] = demos[0].ee_pose[
            skill_index * 6
        ]
    future_observation = DynaMACObservation(observation.ee_pose, future_frames)
    current_clock = (policy._skill_index, policy._time_index)
    future = policy.query_state(future_observation, StateId(2, 3), mode_index=0)
    assert future.diagnostics["skill_index"] == 2
    assert future.diagnostics["time_index"] == 3
    assert (policy._skill_index, policy._time_index) == current_clock
    assert set(future.diagnostics["active_frames"]).issubset(
        future_skill.selected_frames
    )


def test_builder_creates_one_unified_sparse_model() -> None:
    policy, demos = fitted_policy()
    model = ClosedLoopTaskModelBuilder().build(policy, demos)
    assert len(model.states) == sum(skill.duration for skill in policy.skills)
    assert len(model.boundaries) == len(policy.skills) - 1
    assert set(model.relation_frames) == set(policy.frame_names)
    assert all(not name.startswith("virtual_skill_") for name in model.relation_frames)
    assert all(
        np.allclose(np.sum(prior, axis=1), 1.0)
        for node in model.states.values()
        for prior in node.demo_relation_priors.values()
    )
    assert all(
        set(node.demo_relation_scores) == set(node.demo_relation_priors)
        and all(np.all(score >= 0.0) for score in node.demo_relation_scores.values())
        for node in model.states.values()
    )
    assert any(node.scene_factor_models for node in model.states.values())
    assert all(
        factor.kind == "edge"
        for node in model.states.values()
        for factor in node.scene_factor_models
    )
    assert all(
        distribution.neighborhood_radius in {0, 1, 2}
        and distribution.stable_fraction >= 0.8
        and distribution.loo_gain > 1.0e-6
        and distribution.loo_accuracy >= 0.8
        for node in model.states.values()
        for distributions in node.scene_factor_models.values()
        for distribution in distributions.values()
    )
    for boundary in model.boundaries.values():
        duration = policy.skills[boundary.source_skill].duration
        skill = policy.skills[boundary.source_skill]
        expected_indices = tuple(range(max(0, duration - 3), duration))
        assert (
            tuple(state.local_index for state in boundary.terminal_window)
            == expected_indices
        )
        assert (
            boundary.local_completion_model.terminal_states == boundary.terminal_window
        )
        assert set(boundary.scene_condition_thresholds) == set(
            boundary.scene_conditions
        )
        assert all(
            0.0 <= value <= 1.0
            for value in boundary.scene_condition_thresholds.values()
        )
        assert not set(
            boundary.local_completion_model.own_relation_conditions
        ).intersection(boundary.relation_conditions)
        assert all(
            boundary.condition_reliability[key].stable_fraction == 1.0
            for key in (
                *boundary.local_completion_model.own_relation_conditions,
                *boundary.relation_conditions,
            )
        )
        # Local completion supervises the actual frozen skill executor.  Its
        # terminal targets therefore reuse the base model distributions fitted
        # from the normal demonstrations instead of performing a second fit
        # whose quaternion gauge may differ from a released checkpoint.
        for mode in range(len(skill.mode_priors)):
            for frame in skill.selected_frames:
                stream = skill.streams[frame]
                if not stream.is_selected(mode):
                    continue
                goal = boundary.local_completion_model.goal_distributions[
                    f"m{mode}:{frame}"
                ]
                np.testing.assert_array_equal(
                    goal.mean, stream.mean[mode, duration - 1]
                )
                np.testing.assert_array_equal(
                    goal.covariance, stream.covariance[mode, duration - 1]
                )
    second_boundary = next(
        boundary for boundary in model.boundaries.values() if boundary.source_skill == 1
    )
    assert (
        second_boundary.local_completion_model.own_relation_conditions[
            "single/object"
        ].required_state
        == "linked"
    )
    assert "single/object" not in second_boundary.relation_conditions
    assert any(event.frame_id == "object" for event in model.link_anchors)
    assert any(event.frame_id == "object" for event in model.unlink_events)
    assert all(event.frame_id != "target" for event in model.link_anchors)
    assert all(anchor.support_fraction >= 0.8 for anchor in model.link_anchors.values())
    assert all(event.support_fraction >= 0.8 for event in model.unlink_events.values())
    assert model.link_origins
    assert set(model.link_origins.values()).issubset(model.link_anchors)
    for state_id, node in model.states.items():
        skill = policy.skills[state_id.skill_index]
        assert node.mode_priors.shape == skill.mode_priors.shape
        assert node.gripper_commands.shape == (
            len(skill.mode_priors),
            skill.gripper.shape[-1],
        )
        for name, mean in node.stream_means.items():
            assert mean.shape == (len(skill.mode_priors), 7)
            assert np.shares_memory(mean, skill.streams[name].mean)

    object_anchor = next(
        anchor
        for event, anchor in model.link_anchors.items()
        if event.frame_id == "object"
    )
    assert object_anchor.context_state.skill_index == 0
    assert len(object_anchor.local_means) == object_anchor.local_covariances.shape[0]

    first = model.state(StateId(0, 0)).topology
    terminal = model.state(StateId(0, policy.skills[0].duration - 1)).topology
    assert first.predecessors == ()
    assert terminal.skill_terminal is True
    assert terminal.has_cross_skill_successor is True
    assert terminal.successors == (StateId(1, 0),)


def test_relation_event_hysteresis_keeps_transition_band_and_removes_pulses() -> None:
    builder = ClosedLoopTaskModelBuilder()
    assert builder._detect_relation_transitions(
        np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    ) == [("link", 4)]
    assert (
        builder._detect_relation_transitions(np.asarray([0.1, 0.1, 0.8, 0.1, 0.1]))
        == []
    )
    assert builder._detect_relation_transitions(
        np.asarray([0.1, 0.1, 0.9, 0.9, 0.5, 0.5, 0.1, 0.1])
    ) == [("link", 2), ("unlink", 6)]


def test_relation_temporal_evidence_rejects_repeatable_approach_false_link() -> None:
    policy, demos = fitted_policy()
    builder = ClosedLoopTaskModelBuilder()
    aligned = builder._align_demonstrations(policy, demos)

    # Force every demonstration to use the same repeatable pre-grasp relative
    # trajectory. Cross-demo covariance alone then strongly supports linked,
    # even though the object is static and the arm is still approaching it.
    reference_local = np.stack(
        [
            pose_compose(
                pose_inverse(aligned[0].frames["object"][0, index]),
                aligned[0].ee_pose[0, index],
            )
            for index in range(policy.skills[0].duration)
        ]
    )
    for demonstration in range(len(demos)):
        for index, local_pose in enumerate(reference_local):
            aligned[0].ee_pose[demonstration, index] = pose_compose(
                aligned[0].frames["object"][demonstration, index],
                local_pose,
            )

    scales, probabilities, observable = builder._joint_relation_evidence_sequence(
        policy,
        aligned,
        "object",
        np.arange(len(demos)),
        len(policy.skills) - 1,
    )
    approach = slice(1, policy.skills[0].duration)
    raw_linked = np.asarray(
        [
            _relation_prior(scale, builder.config, policy.config.tau_m)[1]
            for scale in scales[approach]
        ]
    )
    assert np.all(raw_linked >= builder.config.relation_link_threshold)
    assert np.all(observable[approach])
    assert np.all(probabilities[approach] <= builder.config.relation_unlink_threshold)


def test_relation_events_require_motion_and_follow_comotion_then_detachment() -> None:
    policy, demos = fitted_policy()
    builder = ClosedLoopTaskModelBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    members = np.arange(len(demos))

    event_probabilities = builder._joint_relation_event_probability_sequence(
        policy,
        aligned,
        "object",
        members,
        len(policy.skills) - 1,
    )
    assert event_probabilities[0] == 0.5
    assert builder._detect_relation_transitions(np.full(8, 0.5)) == []
    assert builder._detect_relation_transitions(event_probabilities) == [
        ("link", 7),
        ("unlink", 13),
    ]

    model = builder.build(policy, demos)
    anchor = next(iter(model.link_anchors.values()))
    assert np.any(anchor.gripper_commands > 0.0)
    assert np.any(anchor.gripper_commands < 0.0)
    release = next(iter(model.unlink_events.values())).release_state
    # Stable relative-motion evidence appears one aligned state after the
    # all-demo opening command in this fixture.  The confirmed UNLINK is
    # therefore anchored at the causal opening state, while kinematic evidence
    # remains mandatory for accepting the event at all.
    assert release == StateId(2, 0)
    release_gripper = aligned[release.skill_index].gripper[members, release.local_index]
    assert np.all(release_gripper > 0.0)

    ordered = tuple(sorted(model.states))
    global_index = {state: index for index, state in enumerate(ordered)}
    entry_indices = [global_index[state] for state in anchor.linked_entry_states]
    assert entry_indices == list(range(entry_indices[0], entry_indices[-1] + 1))
    origin_indices = sorted(
        global_index[key.state_id]
        for key, event_id in model.link_origins.items()
        if event_id == anchor.event_id
    )
    assert set(entry_indices).issubset(origin_indices)
    # Natural confirmation ends after the first stable kinematic evidence,
    # while the relation origin correctly persists until the later UNLINK.
    assert entry_indices[-1] < global_index[release]
    assert any(
        entry_indices[-1] < index < global_index[release] for index in origin_indices
    )


def test_confirmed_relation_events_override_flickering_state_evidence() -> None:
    class FlickeringPriorBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            probabilities = np.zeros(count, dtype=np.float64)
            probabilities[6:] = 1.0
            return probabilities

        def _build_states(self, policy, aligned):
            states, skill_states = super()._build_states(policy, aligned)
            for global_index, state_id in enumerate(sorted(states)):
                linked = 0.95 if global_index % 2 else 0.05
                states[state_id].demo_relation_priors["object"][:] = np.asarray(
                    [1.0 - linked, linked]
                )
            return states, skill_states

    policy, demos = fitted_policy()
    builder = FlickeringPriorBuilder()
    model = builder.build(
        policy,
        demos,
        recoverable_frames=("object",),
    )
    anchor = next(iter(model.link_anchors.values()))
    event_start = min(anchor.linked_entry_states)
    ordered = tuple(sorted(model.states))
    event_index = ordered.index(event_start)
    for index, state_id in enumerate(ordered):
        prior = model.state(state_id).demo_relation_priors["object"][0]
        if index < event_index:
            np.testing.assert_allclose(
                prior,
                np.asarray(
                    [
                        1.0 - builder.config.relation_unlink_threshold,
                        builder.config.relation_unlink_threshold,
                    ]
                ),
            )
            assert not any(
                key.frame_id == "object" and key.state_id == state_id
                for key in model.link_origins
            )
        else:
            np.testing.assert_allclose(
                prior,
                np.asarray(
                    [
                        1.0 - builder.config.relation_link_threshold,
                        builder.config.relation_link_threshold,
                    ]
                ),
            )
            assert any(
                key.frame_id == "object"
                and key.state_id == state_id
                and event_id == anchor.event_id
                for key, event_id in model.link_origins.items()
            )


def test_rejected_relation_pulse_stays_audit_only_not_deployment_expectation() -> None:
    class RejectedPulseBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            return np.full(count, 0.5, dtype=np.float64)

        def _joint_link_pending_evidence_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            return np.zeros(count, dtype=np.float64), np.ones(count, dtype=bool)

        def _build_states(self, policy, aligned):
            states, skill_states = super()._build_states(policy, aligned)
            for state_id in tuple(sorted(states))[7:9]:
                states[state_id].demo_relation_priors["object"][:] = np.asarray(
                    [0.05, 0.95]
                )
            return states, skill_states

    policy, demos = fitted_policy()
    builder = RejectedPulseBuilder()
    model = builder.build(policy, demos, recoverable_frames=("object",))
    assert not model.link_anchors
    assert not model.unlink_events
    assert not model.link_pending_events
    assert not model.link_origins
    expected = np.asarray(
        [
            1.0 - builder.config.relation_unlink_threshold,
            builder.config.relation_unlink_threshold,
        ]
    )
    for state_id in tuple(sorted(model.states))[7:9]:
        np.testing.assert_allclose(
            model.state(state_id).demo_relation_priors["object"][0],
            expected,
        )


def test_link_anchor_starts_after_same_arm_latest_cross_frame_unlink() -> None:
    class CrossFrameEventBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            probabilities = np.zeros(count, dtype=np.float64)
            if frame == "target":
                probabilities[4:8] = 1.0
            elif frame == "object":
                probabilities[12:] = 1.0
            return probabilities

    demos = demonstrations()
    commands = np.ones(18, dtype=np.float64)
    commands[4:8] = -1.0
    commands[8:12] = 1.0
    commands[12:] = -1.0
    for demonstration in demos:
        demonstration.gripper[:, 0] = commands
    policy = DynaMAC(config()).fit(demos)
    model = CrossFrameEventBuilder().build(
        policy,
        demos,
        recoverable_frames=("object", "target"),
    )

    object_anchor = next(
        anchor
        for event, anchor in model.link_anchors.items()
        if event.frame_id == "object"
    )
    assert object_anchor.context_state == StateId(1, 3)
    assert len(object_anchor.local_means) == 4
    assert np.all(object_anchor.gripper_commands[:-1] > 0.0)
    assert np.all(object_anchor.gripper_commands[-1] < 0.0)


def test_link_pending_preserves_unexcited_candidate_without_creating_link() -> None:
    class PendingOnlyBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            return np.full(count, 0.5, dtype=np.float64)

        def _joint_link_pending_evidence_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            probabilities = np.zeros(count, dtype=np.float64)
            probabilities[6:12] = 1.0
            return probabilities, np.zeros(count, dtype=bool)

    policy, demos = fitted_policy()
    builder = PendingOnlyBuilder()
    model = builder.build(
        policy,
        demos,
        recoverable_frames=("object",),
    )
    assert not model.link_anchors
    assert not model.unlink_events
    assert not model.link_origins
    assert len(model.link_pending_events) == 1
    candidate = next(iter(model.link_pending_events.values()))
    assert candidate.event_id.transition == "link_pending"
    assert candidate.frame_id == "object"
    assert candidate.candidate_state == StateId(1, 0)
    assert candidate.support_fraction == 1.0
    assert candidate.demonstration_indices == tuple(range(len(demos)))
    assert candidate.event_local_indices == (0,) * len(demos)
    assert candidate.context_state == StateId(0, 0)
    assert candidate.local_means.shape == (7, 7)
    assert candidate.local_covariances.shape == (7, 6, 6)
    assert candidate.gripper_commands.shape[0] == 7
    assert np.any(candidate.gripper_commands > 0.0)
    assert np.any(candidate.gripper_commands < 0.0)
    np.testing.assert_allclose(
        model.state(candidate.candidate_state).demo_relation_priors["object"][0],
        [0.3, 0.7],
    )
    assert all(
        np.allclose(
            model.state(state).demo_relation_priors["object"][0],
            [0.3, 0.7],
        )
        for state in model.skill_states[candidate.candidate_state.skill_index]
        if state >= candidate.candidate_state
    )

    class ConfirmedLaterBuilder(PendingOnlyBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            probabilities = np.zeros(count, dtype=np.float64)
            probabilities[8:12] = 1.0
            return probabilities

    confirmed = ConfirmedLaterBuilder().build(
        policy,
        demos,
        recoverable_frames=("object",),
    )
    assert confirmed.link_anchors
    assert not confirmed.link_pending_events


def test_link_pending_starts_after_same_arm_latest_cross_frame_unlink() -> None:
    class CrossFramePendingBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            probabilities = np.full(count, 0.5, dtype=np.float64)
            if frame == "target":
                probabilities[:] = 0.0
                probabilities[4:8] = 1.0
            return probabilities

        def _joint_link_pending_evidence_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            count = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            if frame == "target":
                probabilities = np.zeros(count, dtype=np.float64)
                probabilities[4:8] = 1.0
                return probabilities, np.ones(count, dtype=bool)
            probabilities = np.zeros(count, dtype=np.float64)
            probabilities[10:] = 1.0
            return probabilities, np.zeros(count, dtype=bool)

    demos = demonstrations()
    commands = np.ones(18, dtype=np.float64)
    commands[4:8] = -1.0
    commands[8:12] = 1.0
    commands[12:] = -1.0
    for demonstration in demos:
        demonstration.gripper[:, 0] = commands
    policy = DynaMAC(config()).fit(demos)
    model = CrossFramePendingBuilder().build(
        policy,
        demos,
        recoverable_frames=("object", "target"),
    )

    candidate = next(
        candidate
        for event, candidate in model.link_pending_events.items()
        if event.frame_id == "object"
    )
    assert candidate.candidate_state == StateId(2, 0)
    assert candidate.context_state == StateId(1, 3)
    assert len(candidate.local_means) == 4
    assert np.all(candidate.gripper_commands[:-1] > 0.0)
    assert np.all(candidate.gripper_commands[-1] < 0.0)


def test_relation_events_use_joint_prior_lodo_and_all_demo_anchors() -> None:
    class OneMissingFoldBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            if len(members) == 4 and 0 not in members:
                return np.zeros(
                    sum(
                        policy.skills[index].duration
                        for index in range(final_skill + 1)
                    ),
                    dtype=np.float64,
                )
            return super()._joint_relation_event_probability_sequence(
                policy,
                aligned,
                frame,
                members,
                final_skill,
            )

    policy, demos = fitted_policy()
    builder = OneMissingFoldBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    states, _ = builder._build_states(policy, aligned)
    links, unlinks, _, _ = builder._build_relation_events(
        policy,
        states,
        aligned,
        "single",
        frozenset({"object"}),
    )
    assert links and unlinks
    assert all(anchor.support_fraction == 0.8 for anchor in links.values())
    assert all(event.support_fraction == 0.8 for event in unlinks.values())

    anchor = next(iter(links.values()))
    assert 0 not in anchor.demonstration_indices
    context = anchor.context_state
    event = anchor.event_id
    members = np.asarray(
        policy.skills[event.skill_index].mode_demonstration_indices[event.mode]
    )
    context_data = aligned[context.skill_index]
    local_samples = np.stack(
        [
            pose_compose(
                pose_inverse(
                    context_data.frames[event.frame_id][demo, context.local_index]
                ),
                context_data.ee_pose[demo, context.local_index],
            )
            for demo in members
        ]
    )
    expected_mean, expected_covariance = _fit_pose_samples(local_samples, policy)
    np.testing.assert_allclose(anchor.local_means[0], expected_mean)
    np.testing.assert_allclose(anchor.local_covariances[0], expected_covariance)

    baseline_ids = (set(links), set(unlinks))
    for data in aligned.values():
        data.gripper[:] = 1.0
    opened_links, opened_unlinks, _, _ = builder._build_relation_events(
        policy,
        states,
        aligned,
        "single",
        frozenset({"object"}),
    )
    assert baseline_ids[0] and baseline_ids[1]
    assert not opened_links
    assert not opened_unlinks


def test_link_lodo_aligns_delayed_kinematic_evidence_by_shared_close() -> None:
    class DelayedFoldEvidenceBuilder(ClosedLoopTaskModelBuilder):
        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            total = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            probabilities = np.full(total, 0.1, dtype=np.float64)
            link_evidence = 7 if len(members) == 5 else 11
            probabilities[link_evidence:13] = 0.9
            return probabilities

    policy, demos = fitted_policy()
    builder = DelayedFoldEvidenceBuilder()
    model = builder.build(policy, demos, recoverable_frames={"object"})

    anchor = next(
        anchor
        for event, anchor in model.link_anchors.items()
        if event.frame_id == "object"
    )
    assert anchor.support_fraction == 1.0
    assert anchor.event_id.transition == "link"
    assert anchor.event_local_indices == (5, 5, 5, 5, 5)


def test_scene_lodo_robot_baseline_does_not_use_relation_prior() -> None:
    policy, demos = fitted_policy()
    low_alpha = ClosedLoopTaskModelBuilder(
        ClosedLoopTaskModelConfig(relation_alpha=1.0)
    )
    high_alpha = ClosedLoopTaskModelBuilder(
        ClosedLoopTaskModelConfig(relation_alpha=60.0)
    )
    aligned = low_alpha._align_demonstrations(policy, demos)
    state_id = StateId(1, 2)
    training = np.asarray([1, 2, 3, 4])
    candidates = [0, 2, 5]
    low_scores = low_alpha._robot_candidate_scores(
        policy,
        aligned[1],
        state_id,
        0,
        0,
        training,
        candidates,
    )
    high_scores = high_alpha._robot_candidate_scores(
        policy,
        aligned[1],
        state_id,
        0,
        0,
        training,
        candidates,
    )
    assert high_scores == low_scores
    assert all(np.isfinite(score) for score in high_scores.values())


def test_scene_candidates_use_internal_fields_and_direct_structure_only() -> None:
    demos = demonstrations(with_scene_metadata=True)
    policy = DynaMAC(config()).fit(demos)
    builder = ClosedLoopTaskModelBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    state = StateId(1, 2)
    candidates = set(builder._candidate_factors(policy, aligned[1], state, 0, 0))
    assert FactorId("node", "object", feature="joint_position") in candidates
    assert FactorId("edge", "object", "target") in candidates
    # The explicit joint field is the canonical articulation state; the
    # kinematically equivalent child-parent pose must not be duplicated.
    assert FactorId("edge", "object", "structure") not in candidates
    assert FactorId("edge", "target", "structure") not in candidates
    assert FactorId("edge", "structure", "root") not in candidates

    model = builder.build(policy, demos)
    node_distributions = [
        distribution
        for node in model.states.values()
        for factor, distributions in node.scene_factor_models.items()
        if factor.kind == "node"
        for distribution in distributions.values()
    ]
    assert node_distributions
    assert all(distribution.space == "euclidean" for distribution in node_distributions)


def test_scene_factor_selects_smallest_stable_same_mode_radius() -> None:
    policy, demos = fitted_policy()
    skill = policy.skills[1]
    for name, mask in (
        ("object", [True, False, False, False, False, False]),
        ("target", [False, True, False, False, False, False]),
    ):
        stream = skill.streams[name]
        stream.availability[:] = mask
        stream.active[:] = mask

    builder = ClosedLoopTaskModelBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    state = StateId(1, 0)
    edge = FactorId("edge", "object", "target")
    assert edge not in builder._candidate_factors(policy, aligned[1], state, 0, 0)
    assert edge in builder._candidate_factors(policy, aligned[1], state, 0, 1)

    model = builder.build(policy, demos)
    assert model.state(state).scene_factor_models[edge][0].neighborhood_radius == 1


def test_query_state_rejects_mode_inside_progress_state() -> None:
    policy, demos = fitted_policy()
    observation = first_observation(demos)
    policy.reset(observation, mode_strategy="map")
    with pytest.raises(ValueError, match="只包含"):
        policy.query_state(observation, (0, 0, 0))


def test_relation_event_windows_and_metadata_cross_skill_boundaries() -> None:
    class CompleteSequenceBuilder(ClosedLoopTaskModelBuilder):
        probabilities: np.ndarray

        def _joint_relation_event_probability_sequence(
            self,
            policy,
            aligned,
            frame,
            members,
            final_skill,
        ):
            expected = sum(
                policy.skills[index].duration for index in range(final_skill + 1)
            )
            return self.probabilities[:expected].copy()

    policy, demos = fitted_policy()
    builder = CompleteSequenceBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    states, _ = builder._build_states(policy, aligned)
    total = sum(skill.duration for skill in policy.skills)

    # A LINK starts at the final state of skill 0 and obtains its required
    # post-event dwell from skill 1.
    for skill_index, data in aligned.items():
        data.gripper[:] = -1.0 if skill_index > 0 else 1.0
    aligned[0].gripper[:, 5:] = -1.0
    builder.probabilities = np.concatenate((np.zeros(5), np.ones(total - 5)))
    global_index = 0
    for skill_index, skill in enumerate(policy.skills):
        for index in range(skill.duration):
            prior = states[StateId(skill_index, index)].demo_relation_priors["object"]
            prior[:, 1] = builder.probabilities[global_index]
            prior[:, 0] = 1.0 - builder.probabilities[global_index]
            global_index += 1
    links, _, origins, _ = builder._build_relation_events(
        policy,
        states,
        aligned,
        "single",
        frozenset({"object"}),
    )
    boundary_link = next(
        anchor
        for event_id, anchor in links.items()
        if event_id.skill_index == 0 and event_id.transition == "link"
    )
    assert StateId(0, 5) in boundary_link.linked_entry_states
    assert StateId(1, 0) in boundary_link.linked_entry_states
    assert any(
        key.state_id == StateId(1, 0) and event_id == boundary_link.event_id
        for key, event_id in origins.items()
    )

    # The corresponding UNLINK metadata also traverses the boundary: the
    # release state remains in skill 0, while later legal states and the local
    # detachment target may come from skill 1.
    for skill_index, data in aligned.items():
        data.gripper[:] = 1.0 if skill_index > 0 else -1.0
    aligned[0].gripper[:, 5:] = 1.0
    builder.probabilities = np.concatenate((np.ones(5), np.zeros(total - 5)))
    global_index = 0
    for skill_index, skill in enumerate(policy.skills):
        for index in range(skill.duration):
            prior = states[StateId(skill_index, index)].demo_relation_priors["object"]
            prior[:, 1] = builder.probabilities[global_index]
            prior[:, 0] = 1.0 - builder.probabilities[global_index]
            global_index += 1
    _, unlinks, _, _ = builder._build_relation_events(
        policy,
        states,
        aligned,
        "single",
        frozenset({"object"}),
    )
    boundary_unlink = next(
        event
        for event_id, event in unlinks.items()
        if event_id.skill_index == 0 and event_id.transition == "unlink"
    )
    assert boundary_unlink.release_state == StateId(0, 5)
    assert StateId(1, 0) in boundary_unlink.legal_reentry_states


def test_boundary_scene_conditions_include_boundary_only_library_factors() -> None:
    demos = demonstrations(with_scene_metadata=True)
    for demo in demos:
        demo.entity_configurations["object"]["joint_position"][:] = 0.25
    policy = DynaMAC(config()).fit(demos)
    # A selected next-skill reference remains a boundary entity even when its
    # old fixed DynaMAC participation mask is inactive (for example, linked).
    policy.skills[1].streams["object"].active[:] = False
    builder = ClosedLoopTaskModelBuilder()
    model = builder.build(policy, demos)
    boundary_only = FactorId("node", "object", feature="joint_position")
    assert all(
        boundary_only not in node.scene_factor_models for node in model.states.values()
    )
    assert any(
        boundary_only in boundary.scene_conditions
        for boundary in model.boundaries.values()
    )
    assert all(
        boundary.condition_reliability[factor.token].stable_fraction
        == distribution.stable_fraction
        for boundary in model.boundaries.values()
        for factor, distribution in boundary.scene_conditions.items()
    )


def test_boundary_scene_lodo_checks_normal_window_coverage_only() -> None:
    policy, demos = fitted_policy()
    builder = ClosedLoopTaskModelBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    factor = FactorId("edge", "object", "target")
    values = builder._factor_values(aligned[0], factor)
    values[:] = values[:, :1]
    fold_scores = builder._boundary_loo_coverage(
        values,
        policy.skills[0].duration - 3,
        np.arange(len(demos)),
        policy,
        factor,
    )
    assert len(fold_scores) == len(demos)
    assert all(value >= builder._support_compatibility(6) for value in fold_scores)


def test_boundary_relation_candidates_use_only_next_skill_selected_frames() -> None:
    policy, demos = fitted_policy()
    policy.skills[0].selected_frames = ("target",)
    policy.skills[1].selected_frames = ("object",)
    model = ClosedLoopTaskModelBuilder().build(policy, demos)
    boundary = next(
        boundary for boundary in model.boundaries.values() if boundary.source_skill == 0
    )
    assert set(boundary.relation_conditions) == {"single/object"}


def test_boundary_relation_hard_condition_requires_every_lodo_fold() -> None:
    policy, demos = fitted_policy()
    builder = ClosedLoopTaskModelBuilder()
    aligned = builder._align_demonstrations(policy, demos)
    skill = policy.skills[0]
    start = skill.duration - builder.config.boundary_terminal_window
    for local_index in range(start, skill.duration):
        for demonstration in range(5):
            local = pose([0.0 if demonstration < 4 else 1.0, 0.0, 0.0])
            aligned[0].ee_pose[demonstration, local_index] = pose_compose(
                aligned[0].frames["object"][demonstration, local_index],
                local,
            )
    states, _ = builder._build_states(policy, aligned)
    weights = np.arange(1, skill.duration - start + 1, dtype=np.float64)
    weights /= np.sum(weights)
    support = sum(
        weight * states[StateId(0, local_index)].demo_relation_priors["object"][0]
        for weight, local_index in zip(
            weights,
            range(start, skill.duration),
            strict=True,
        )
    )
    assert np.max(support) >= builder.config.boundary_relation_support
    assert (
        builder._fit_boundary_relation_guard(
            policy,
            states,
            aligned,
            0,
            start,
            "object",
        )
        is None
    )


def test_closed_loop_sidecar_roundtrip_and_base_binding(tmp_path: Path) -> None:
    policy, demos = fitted_policy()
    model = ClosedLoopTaskModelBuilder().build(policy, demos)
    template = next(iter(model.link_anchors.values()))
    pending_id = RelationEventId(
        "single",
        "object",
        1,
        0,
        0,
        "link_pending",
    )
    model.link_pending_events[pending_id] = LinkPendingCandidate(
        event_id=pending_id,
        arm_id="single",
        frame_id="object",
        candidate_state=StateId(1, 0),
        context_state=template.context_state,
        local_means=template.local_means,
        local_covariances=template.local_covariances,
        gripper_commands=template.gripper_commands,
        support_fraction=0.8,
        demonstration_indices=(0, 1, 2, 3),
        event_local_indices=(0, 0, 0, 0),
    )
    path = tmp_path / "closed_loop_without_suffix"
    model.save(path)
    assert path.is_file()
    restored = ClosedLoopTaskModel.load(path, policy)
    assert restored.summary() == model.summary()
    for state_id in model.states:
        original = model.states[state_id]
        loaded = restored.states[state_id]
        assert set(loaded.scene_factor_models) == set(original.scene_factor_models)
        for frame in original.demo_relation_priors:
            np.testing.assert_array_equal(
                loaded.demo_relation_scores[frame],
                original.demo_relation_scores[frame],
            )
            np.testing.assert_array_equal(
                loaded.demo_relation_priors[frame],
                original.demo_relation_priors[frame],
            )
    assert restored.link_origins == model.link_origins
    restored_pending = restored.link_pending_events[pending_id]
    original_pending = model.link_pending_events[pending_id]
    assert restored_pending.event_id == original_pending.event_id
    assert restored_pending.candidate_state == original_pending.candidate_state
    assert restored_pending.context_state == original_pending.context_state
    assert restored_pending.support_fraction == original_pending.support_fraction
    assert (
        restored_pending.demonstration_indices == original_pending.demonstration_indices
    )
    assert restored_pending.event_local_indices == original_pending.event_local_indices
    np.testing.assert_array_equal(
        restored_pending.local_means,
        original_pending.local_means,
    )
    np.testing.assert_array_equal(
        restored_pending.local_covariances,
        original_pending.local_covariances,
    )
    np.testing.assert_array_equal(
        restored_pending.gripper_commands,
        original_pending.gripper_commands,
    )
    assert all(
        restored.boundaries[boundary_id].local_completion_model.own_relation_conditions
        == boundary.local_completion_model.own_relation_conditions
        for boundary_id, boundary in model.boundaries.items()
    )
    with np.load(path, allow_pickle=False) as archive:
        assert all("stream" not in key for key in archive.files)
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["base_policy_fingerprint"] == policy.fingerprint()

    incompatible = DynaMAC(config(tau_m=0.004)).fit(demos)
    with pytest.raises(ValueError, match="指纹"):
        ClosedLoopTaskModel.load(path, incompatible)


def test_tracked_phase_one_config_matches_code_defaults() -> None:
    path = Path("configs/closed_loop_task_model.json")
    assert json.loads(path.read_text(encoding="utf-8")) == asdict(
        ClosedLoopTaskModelConfig()
    )


def test_rlbench_roles_limit_recovery_events_without_dropping_frame_fields() -> None:
    stack = get_task_spec("stack_wine")
    handover = get_task_spec("bimanual_handover_item")
    assert stack.frame_names == ("wine_bottle", "success_sensor")
    assert stack.recoverable_relation_frames == ("wine_bottle",)
    assert handover.frame_names == ("item0", "item1", "item2", "item3", "item4")
    assert handover.recoverable_relation_frames == ("item0",)


def test_bimanual_builder_reuses_synchronized_peer_frames() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    left_model, right_model = ClosedLoopTaskModelBuilder().build_bimanual(
        policy,
        left,
        right,
        recoverable_frames=("object",),
    )
    assert "right_ee" in left_model.relation_frames
    assert "left_ee" in right_model.relation_frames
    assert all(event.frame_id == "object" for event in left_model.link_anchors)
    assert all(event.frame_id == "object" for event in right_model.link_anchors)
    left_groups = {
        boundary.transaction_group
        for boundary in left_model.boundaries.values()
        if boundary.transaction_group is not None
    }
    right_groups = {
        boundary.transaction_group
        for boundary in right_model.boundaries.values()
        if boundary.transaction_group is not None
    }
    assert left_groups and left_groups == right_groups
    assert all(
        boundary.affected_arms == ("left", "right")
        for boundary in left_model.boundaries.values()
        if boundary.transaction_group is not None
    )
    for group in left_groups:
        left_boundary = next(
            boundary
            for boundary in left_model.boundaries.values()
            if boundary.transaction_group == group
        )
        right_boundary = next(
            boundary
            for boundary in right_model.boundaries.values()
            if boundary.transaction_group == group
        )
        assert left_boundary.relation_conditions == right_boundary.relation_conditions
        guaranteed = set(
            left_boundary.local_completion_model.own_relation_conditions
        ).union(right_boundary.local_completion_model.own_relation_conditions)
        assert guaranteed.isdisjoint(left_boundary.relation_conditions)


def test_bimanual_peer_execution_dependency_filters_only_redundant_modes(
    tmp_path: Path,
) -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    left_model, right_model = ClosedLoopTaskModelBuilder().build_bimanual(
        policy,
        left,
        right,
        recoverable_frames=("object",),
    )

    for model, peer_frame in (
        (left_model, "right_ee"),
        (right_model, "left_ee"),
    ):
        records = model.builder_config["peer_execution_dependencies"]
        by_skill = {record["skill_index"]: record for record in records}

        # Skill 0 is an independent approach to ordinary physical references.
        # Synchronized demonstrations alone must not turn the other arm into a
        # continuous action reference.
        assert by_skill[0]["eligible"] is False
        assert by_skill[0]["reason"] == "redundant_with_physical_task_reference"
        assert peer_frame not in model.state(model.skill_states[0][0]).selected_frames
        first_boundary = next(
            boundary
            for boundary in model.boundaries.values()
            if boundary.source_skill == 0
        )
        assert all(
            not key.endswith(f":{peer_frame}")
            for key in first_boundary.local_completion_model.goal_distributions
        )

        # Once both arm models contain the same event-confirmed object
        # relation, the same peer stream is retained by the general graph rule.
        assert by_skill[1]["eligible"] is True
        assert by_skill[1]["reason"] == "shared_or_transfer_relation"
        assert peer_frame in model.state(model.skill_states[1][0]).selected_frames

        path = tmp_path / f"{model.arm_id}.npz"
        model.save(path)
        restored = ClosedLoopTaskModel.load(path, model.base_policy)
        assert restored.schema_version == 4
        assert (
            restored.state(restored.skill_states[0][0]).selected_frames
            == model.state(model.skill_states[0][0]).selected_frames
        )
        assert (
            restored.state(restored.skill_states[1][0]).selected_frames
            == model.state(model.skill_states[1][0]).selected_frames
        )


def test_peer_only_directed_geometry_is_not_confused_with_constant_synchrony() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    paired_left, paired_right = synchronized_bimanual_demonstrations(left, right)

    skill = policy.left.skills[0]
    skill.selected_frames = ("right_ee", "virtual_skill_0")
    for name, stream in skill.streams.items():
        selected = name in skill.selected_frames
        stream.selected_by_eq6[:] = selected
        stream.active = (
            stream.availability.copy()
            if selected
            else np.zeros_like(stream.availability, dtype=bool)
        )

    def peer_demonstrations(directed: bool) -> list[DynaMACDemonstration]:
        result = []
        for demonstration in paired_left:
            frames = {
                name: values.copy() for name, values in demonstration.frames.items()
            }
            if directed:
                frames["right_ee"] = np.repeat(
                    pose([0.78, 0.25, 0.25])[None],
                    len(demonstration.ee_pose),
                    axis=0,
                )
            else:
                frames["right_ee"] = np.stack(
                    [
                        pose_compose(current, pose([0.0, 0.22, 0.0]))
                        for current in demonstration.ee_pose
                    ]
                )
            result.append(
                DynaMACDemonstration(
                    ee_pose=demonstration.ee_pose,
                    action_pose=demonstration.action_pose,
                    gripper=demonstration.gripper,
                    frames=frames,
                    skill=demonstration.skill,
                    name=demonstration.name,
                )
            )
        return result

    builder = ClosedLoopTaskModelBuilder()
    right_model = builder.build(
        policy.right,
        paired_right,
        arm_id="right",
        recoverable_frames=(),
    )
    outcomes = {}
    for label, directed in (("directed", True), ("constant", False)):
        values = peer_demonstrations(directed)
        left_model = builder.build(
            policy.left,
            values,
            arm_id="left",
            recoverable_frames=(),
        )
        record = builder._peer_execution_dependency_records(
            left_model,
            right_model,
            values,
        )[0]
        outcomes[label] = record

    assert outcomes["directed"]["eligible"] is True
    assert outcomes["directed"]["reason"] == "exclusive_relative_geometry_target"
    assert outcomes["constant"]["eligible"] is False
    assert outcomes["constant"]["reason"] == "no_cross_arm_execution_dependency"


def test_peer_relative_completion_reuses_standard_boundary_scene_condition() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    paired_left, paired_right = synchronized_bimanual_demonstrations(left, right)
    builder = ClosedLoopTaskModelBuilder()
    left_model = builder.build(
        policy.left,
        paired_left,
        arm_id="left",
        recoverable_frames=("object",),
    )
    right_model = builder.build(
        policy.right,
        paired_right,
        arm_id="right",
        recoverable_frames=("object",),
    )

    # Remove the reciprocal right->left terminal dependency.  The remaining
    # left terminal goal is expressed relative to right_ee, so only the right
    # boundary must wait for the shared physical configuration.
    for boundary_id, boundary in tuple(right_model.boundaries.items()):
        local = boundary.local_completion_model
        kept_goals = {
            key: value
            for key, value in local.goal_distributions.items()
            if not key.endswith(":left_ee")
        }
        kept_thresholds = {
            key: value
            for key, value in local.minimum_goal_log_likelihood.items()
            if key in kept_goals
        }
        right_model.boundaries[boundary_id] = replace(
            boundary,
            local_completion_model=replace(
                local,
                goal_distributions=kept_goals,
                minimum_goal_log_likelihood=kept_thresholds,
            ),
        )

    builder._complete_bimanual_boundary_scene_conditions(
        left_model,
        right_model,
        paired_left,
        paired_right,
    )
    factor = FactorId("edge", "left_ee", "right_ee")
    assert all(
        factor not in boundary.scene_conditions
        and boundary.affected_arms == ("left",)
        and boundary.transaction_group is None
        for boundary in left_model.boundaries.values()
    )
    assert all(
        factor in boundary.scene_conditions
        and boundary.affected_arms == ("left", "right")
        and boundary.transaction_group is None
        for boundary in right_model.boundaries.values()
    )


def test_bimanual_transaction_group_uses_confirmed_boundary_straddling_links() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    builder = ClosedLoopTaskModelBuilder()
    left_model, right_model = builder.build_bimanual(
        policy,
        left,
        right,
        recoverable_frames=("object",),
    )

    # Remove the ordinary boundary guards so the only remaining shared hard
    # evidence is each arm's all-demo LINK trajectory crossing 0 -> 1.
    for model in (left_model, right_model):
        boundary_id = next(
            boundary_id
            for boundary_id in model.boundaries
            if boundary_id.source_skill == 0
        )
        boundary = model.boundaries[boundary_id]
        model.boundaries[boundary_id] = replace(
            boundary,
            relation_conditions={},
            scene_conditions={},
            scene_condition_thresholds={},
            condition_reliability={},
            affected_arms=(model.arm_id,),
            transaction_group=None,
        )

    builder._assign_transaction_groups(left_model, right_model, left, right)
    left_boundary = next(
        boundary
        for boundary in left_model.boundaries.values()
        if boundary.source_skill == 0
    )
    right_boundary = next(
        boundary
        for boundary in right_model.boundaries.values()
        if boundary.source_skill == 0
    )
    assert left_boundary.transaction_group is not None
    assert left_boundary.transaction_group == right_boundary.transaction_group
    assert left_boundary.affected_arms == ("left", "right")
    assert right_boundary.affected_arms == ("left", "right")

    # Replacing the formal events with equally supported Pending candidates
    # must remove the hard transaction evidence.
    for model in (left_model, right_model):
        boundary_id = next(
            boundary_id
            for boundary_id in model.boundaries
            if boundary_id.source_skill == 0
        )
        boundary = model.boundaries[boundary_id]
        anchor = next(iter(model.link_anchors.values()))
        event_id = RelationEventId(
            arm_id=model.arm_id,
            frame_id=anchor.frame_id,
            skill_index=anchor.event_id.skill_index,
            mode=anchor.event_id.mode,
            occurrence=anchor.event_id.occurrence,
            transition="link_pending",
        )
        model.link_pending_events[event_id] = LinkPendingCandidate(
            event_id=event_id,
            arm_id=model.arm_id,
            frame_id=anchor.frame_id,
            candidate_state=anchor.linked_entry_states[0],
            context_state=anchor.context_state,
            local_means=anchor.local_means,
            local_covariances=anchor.local_covariances,
            gripper_commands=anchor.gripper_commands,
            support_fraction=anchor.support_fraction,
            demonstration_indices=anchor.demonstration_indices,
            event_local_indices=anchor.event_local_indices,
        )
        model.link_anchors.clear()
        model.boundaries[boundary_id] = replace(
            boundary,
            affected_arms=(model.arm_id,),
            transaction_group=None,
        )
    builder._assign_transaction_groups(left_model, right_model, left, right)
    assert all(
        boundary.transaction_group is None
        for model in (left_model, right_model)
        for boundary in model.boundaries.values()
        if boundary.source_skill == 0
    )


def test_transaction_group_does_not_promote_directional_relation_guard() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    builder = ClosedLoopTaskModelBuilder()
    left_model, right_model = builder.build_bimanual(
        policy,
        left,
        right,
        recoverable_frames=("object",),
    )
    left_id = next(
        boundary_id
        for boundary_id in left_model.boundaries
        if boundary_id.source_skill == 1
    )
    right_id = next(
        boundary_id
        for boundary_id in right_model.boundaries
        if boundary_id.source_skill == 1
    )
    left_boundary = left_model.boundaries[left_id]
    right_boundary = right_model.boundaries[right_id]
    right_key = "right/object"
    right_condition = right_boundary.local_completion_model.own_relation_conditions[
        right_key
    ]
    right_statistics = right_boundary.condition_reliability[right_key]

    # Left only waits for a relation established by right; it does not itself
    # participate in the shared physical link.  This is directional waiting,
    # not a symmetric atomic boundary.
    left_model.boundaries[left_id] = replace(
        left_boundary,
        local_completion_model=replace(
            left_boundary.local_completion_model,
            own_relation_conditions={},
        ),
        relation_conditions={right_key: right_condition},
        scene_conditions={},
        scene_condition_thresholds={},
        condition_reliability={right_key: right_statistics},
        affected_arms=("left",),
        transaction_group=None,
    )
    right_model.boundaries[right_id] = replace(
        right_boundary,
        relation_conditions={},
        scene_conditions={},
        scene_condition_thresholds={},
        condition_reliability={right_key: right_statistics},
        affected_arms=("right",),
        transaction_group=None,
    )
    left_model.link_anchors.clear()
    right_model.link_anchors.clear()

    builder._assign_transaction_groups(left_model, right_model, left, right)
    assert left_model.boundaries[left_id].transaction_group is None
    assert right_model.boundaries[right_id].transaction_group is None


def test_handover_evidence_adds_only_receiver_linked_directional_guard() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    paired_left, paired_right = synchronized_bimanual_demonstrations(left, right)
    builder = ClosedLoopTaskModelBuilder()
    left_model = builder.build(
        policy.left,
        paired_left,
        arm_id="left",
        recoverable_frames=("object",),
    )
    right_model = builder.build(
        policy.right,
        paired_right,
        arm_id="right",
        recoverable_frames=("object",),
    )
    left_id = BoundaryId("left", 1, 2)
    right_id = BoundaryId("right", 1, 2)

    # Isolate the transfer evidence.  The synthetic left arm already holds and
    # releases object at this boundary.  Make the right command open before the
    # boundary and close at the entry, then attach the all-demo Pending event
    # that an unexcited receiver grasp would produce.
    for model, boundary_id in ((left_model, left_id), (right_model, right_id)):
        boundary = model.boundaries[boundary_id]
        model.boundaries[boundary_id] = replace(
            boundary,
            local_completion_model=replace(
                boundary.local_completion_model,
                own_relation_conditions={},
            ),
            relation_conditions={},
            scene_conditions={},
            scene_condition_thresholds={},
            condition_reliability={},
            affected_arms=(model.arm_id,),
            transaction_group=None,
        )
    right_model.state(right_model.skill_states[1][-1]).gripper_commands[:] = 1.0
    candidate_state = right_model.skill_states[2][0]
    right_model.state(candidate_state).gripper_commands[:] = -1.0
    event_id = RelationEventId("right", "object", 2, 0, 0, "link_pending")
    right_model.link_pending_events[event_id] = LinkPendingCandidate(
        event_id=event_id,
        arm_id="right",
        frame_id="object",
        candidate_state=candidate_state,
        context_state=right_model.skill_states[1][-2],
        local_means=np.stack([pose([0.0, 0.0, 0.0])]),
        local_covariances=np.stack([np.eye(6) * 0.01]),
        gripper_commands=np.asarray([[-1.0]]),
        support_fraction=1.0,
        demonstration_indices=(0, 1, 2, 3, 4),
        event_local_indices=(0, 0, 0, 0, 0),
    )

    builder._complete_bimanual_transfer_relation_conditions(
        left_model,
        right_model,
        paired_left,
        paired_right,
    )
    left_boundary = left_model.boundaries[left_id]
    right_boundary = right_model.boundaries[right_id]
    condition = left_boundary.relation_conditions["right/object"]
    assert condition.required_state == "linked"
    assert condition.external == pytest.approx(0.3)
    assert condition.linked == pytest.approx(0.7)
    assert left_boundary.affected_arms == ("left", "right")
    assert left_boundary.transaction_group is None
    assert right_boundary.relation_conditions == {}
    assert right_boundary.affected_arms == ("right",)

    # A one-way guard remains one-way after the generic transaction pass.
    builder._assign_transaction_groups(
        left_model,
        right_model,
        paired_left,
        paired_right,
    )
    assert left_model.boundaries[left_id].transaction_group is None
    assert right_model.boundaries[right_id].transaction_group is None


def test_bimanual_transaction_group_does_not_force_async_boundaries() -> None:
    left = demonstrations()
    right = demonstrations(right_arm=True)
    for demo in right:
        demo.skill[:] = np.asarray([0] * 8 + [1] * 6 + [2] * 4)
    policy = BimanualDynaMAC(config=config()).fit(left, right)
    left_model, right_model = ClosedLoopTaskModelBuilder().build_bimanual(
        policy,
        left,
        right,
        recoverable_frames=("object",),
    )
    assert all(
        boundary.transaction_group is None
        for boundary in (
            *left_model.boundaries.values(),
            *right_model.boundaries.values(),
        )
    )
    assert all(
        boundary.affected_arms == ("left", "right")
        for boundary in right_model.boundaries.values()
        if boundary.transaction_group is not None
    )
