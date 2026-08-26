"""Phase-four acceptance tests for guards and multi-arm transactions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from essay2608.policy import (
    DynaMAC,
    DynaMACConfig,
    DynaMACDemonstration,
    DynaMACObservation,
)
from essay2608.policy.closed_loop import (
    BoundaryCalibration,
    BoundaryId,
    BoundaryModel,
    BoundaryRuntimeConfig,
    ClosedLoopBelief,
    ClosedLoopExecutionController,
    ClosedLoopTaskModelBuilder,
    ConditionKind,
    EntryGuard,
    FactorDistribution,
    FactorId,
    LinkPendingCandidate,
    LocalCompletionModel,
    MultiArmBoundaryController,
    ProgressEstimate,
    ProgressStatus,
    RelationDecision,
    RelationEstimate,
    RelationEventId,
    RelationGuardDistribution,
    ReliabilityStatistics,
    RuntimeFeatureBuilder,
    RuntimeObservation,
    StateId,
)
from essay2608.policy.dynamac import pose_compose, relative_pose


def pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _make_model(
    arm_id: str,
    *,
    transaction_group: str | None = None,
    affected_arms: tuple[str, ...] | None = None,
):
    demonstrations = []
    for demo_index in range(5):
        object_pose = pose(0.35 + 0.02 * demo_index, 0.01 * demo_index)
        target_pose = pose(0.75, 0.10)
        ee = []
        objects = []
        targets = []
        for skill_index in range(2):
            for local_index in range(4):
                progress = local_index / 3.0
                offset = (
                    pose(-0.12 * (1.0 - progress), 0.0, 0.02 * (1.0 - progress))
                    if skill_index == 0
                    else pose(0.04 * progress, 0.02 * progress)
                )
                ee.append(pose_compose(object_pose, offset))
                objects.append(object_pose)
                targets.append(target_pose)
        ee_values = np.stack(ee)
        demonstrations.append(
            DynaMACDemonstration(
                ee_pose=ee_values,
                action_pose=ee_values.copy(),
                gripper=np.ones(8),
                frames={"object": np.stack(objects), "target": np.stack(targets)},
                skill=np.repeat([0, 1], 4),
                name=f"phase4_{arm_id}_{demo_index}",
            )
        )
    policy = DynaMAC(
        DynaMACConfig(
            kinematic_analysis_enabled=False,
            tau_omega=0.0,
            eq6_empty_selection="keep_argmax",
            link_mask_scope="timestep",
            default_mode_strategy="map",
        )
    ).fit(demonstrations)
    model = ClosedLoopTaskModelBuilder().build(
        policy,
        demonstrations,
        arm_id=arm_id,
        recoverable_frames=("object", "target"),
    )
    boundary_id = BoundaryId(arm_id, 0, 1)
    original = model.boundaries[boundary_id]
    final = model.skill_states[0][-1]
    terminal = model.skill_states[0][-2:]
    goal_distribution = FactorDistribution(
        mean=pose(0.0),
        covariance=np.eye(6) * 0.02,
        sample_count=5,
    )
    scene_factor = FactorId("edge", "object", "target")
    scene_mean = relative_pose(pose(0.75, 0.10), pose(0.35))
    scene_distribution = FactorDistribution(
        mean=scene_mean,
        covariance=np.eye(6) * 0.10,
        sample_count=5,
    )
    own_key = f"{arm_id}/object"
    guard_key = f"{arm_id}/target"
    external = RelationGuardDistribution(0.9, 0.1, "external")
    local = LocalCompletionModel(
        terminal_states=terminal,
        goal_distributions={"m0:object": goal_distribution},
        minimum_goal_log_likelihood={"m0:object": -100.0},
        own_relation_conditions={own_key: external},
    )
    reliability = {
        own_key: ReliabilityStatistics(1.0, 1.0),
        guard_key: ReliabilityStatistics(1.0, 1.0),
        scene_factor.token: ReliabilityStatistics(1.0, 1.0),
    }
    model.boundaries[boundary_id] = BoundaryModel(
        boundary_id=boundary_id,
        source_skill=0,
        target_skill=1,
        terminal_window=terminal,
        local_completion_model=local,
        relation_conditions={guard_key: external},
        scene_conditions={scene_factor: scene_distribution},
        scene_condition_thresholds={scene_factor: 0.20},
        condition_reliability=reliability,
        affected_arms=affected_arms or (arm_id,),
        transaction_group=transaction_group,
    )
    policy.reset(
        DynaMACObservation(
            demonstrations[0].ee_pose[0],
            {
                "object": demonstrations[0].frames["object"][0],
                "target": demonstrations[0].frames["target"][0],
            },
        ),
        mode_strategy="map",
    )
    assert original.boundary_id == boundary_id and final == StateId(0, 3)
    return model, demonstrations


def _relation(
    frame: str,
    posterior: tuple[float, float] = (0.95, 0.05),
    decision: RelationDecision = RelationDecision.EXTERNAL,
    *,
    information_weight: float = 1.0,
) -> RelationEstimate:
    values = np.asarray(posterior, dtype=np.float64)
    return RelationEstimate(
        frame_id=frame,
        posterior=values,
        predicted=values,
        demonstration_prior=np.asarray([0.7, 0.3]),
        observation_likelihood=np.ones(2),
        information_weight=information_weight,
        entropy=-float(np.sum(values * np.log(np.maximum(values, 1.0e-12)))),
        informative=decision != RelationDecision.UNKNOWN,
        decision_state=decision,
    )


def _features(model, demonstrations, tick: int):
    demo = demonstrations[0]
    current_index = 3
    virtual = {
        "virtual_skill_0": demo.ee_pose[0],
        "virtual_skill_1": demo.ee_pose[4],
    }
    frames = {
        "object": demo.frames["object"][current_index],
        "target": demo.frames["target"][current_index],
        **virtual,
    }
    previous = RuntimeObservation(
        tick=tick - 1,
        ee_pose=demo.ee_pose[current_index],
        frame_poses=frames,
        gripper_state=np.asarray([1.0]),
        previous_command_pose=None,
        previous_ee_pose=None,
        tracking_reliability={},
        frame_visibility={},
    )
    current = RuntimeObservation(
        tick=tick,
        ee_pose=demo.ee_pose[current_index],
        frame_poses=frames,
        gripper_state=np.asarray([1.0]),
        previous_command_pose=demo.ee_pose[current_index],
        previous_ee_pose=demo.ee_pose[current_index],
        tracking_reliability={},
        frame_visibility={},
    )
    return RuntimeFeatureBuilder().build(current, previous)


def _belief(
    model,
    demonstrations,
    tick: int,
    *,
    end_probability: float = 1.0,
    target_relation: RelationEstimate | None = None,
    object_relation: RelationEstimate | None = None,
) -> ClosedLoopBelief:
    final = model.skill_states[0][-1]
    early = model.skill_states[0][0]
    posterior = {final: end_probability}
    if end_probability < 1.0:
        posterior[early] = 1.0 - end_probability
    progress = ProgressEstimate(
        prior=dict(posterior),
        posterior=dict(posterior),
        nominal_state=final,
        estimated_state=final if end_probability >= 0.5 else early,
        confidence=max(posterior.values()),
        entropy=0.0,
        best_explanation_score=1.0,
        status=ProgressStatus.ALIGNED,
    )
    return ClosedLoopBelief(
        tick=tick,
        runtime_features=_features(model, demonstrations, tick),
        relation_estimates={
            "object": object_relation or _relation("object"),
            "target": target_relation or _relation("target"),
        },
        progress=progress,
        candidate_scores={},
        relation_changes=(),
        local_candidates=tuple(posterior),
        expanded_candidates=(),
    )


def _config(models, *, cycles: int = 2, local_threshold: float = 0.5):
    return BoundaryRuntimeConfig(
        calibrations={
            boundary_id.token: BoundaryCalibration(local_threshold, cycles)
            for model in models.values()
            for boundary_id in model.boundaries
        }
    )


def _controller_tick(controller, model, demonstrations, belief):
    final = model.skill_states[0][-1]
    if controller.cursor.reference_state != final:
        controller.reset(final)
    demo = demonstrations[0]
    return controller.update(
        belief,
        DynaMACObservation(
            demo.ee_pose[3],
            {
                "object": demo.frames["object"][3],
                "target": demo.frames["target"][3],
                "virtual_skill_0": demo.ee_pose[0],
            },
        ),
        mode_by_skill={0: 0, 1: 0},
    )


def test_local_done_and_guard_conditions_require_stable_cycles() -> None:
    model, demonstrations = _make_model("single")
    config = _config({"single": model}, cycles=2)
    guard = EntryGuard({"single": model}, "single", config)
    boundary_id = BoundaryId("single", 0, 1)
    source = model.skill_states[0][-1]

    first, first_local = guard.evaluate(
        boundary_id,
        {"single": _belief(model, demonstrations, 1)},
        source,
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    assert first_local.score == pytest.approx(0.86)
    assert first_local.consecutive_cycles == 1
    assert not first_local.done
    assert not first.permitted

    second, second_local = guard.evaluate(
        boundary_id,
        {"single": _belief(model, demonstrations, 2)},
        source,
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    assert second_local.done
    assert second.permitted
    assert all(
        result.satisfied
        for condition_id, result in second.condition_results.items()
        if condition_id.kind
        in {ConditionKind.GUARD_RELATION, ConditionKind.GUARD_SCENE}
    )


def test_unknown_own_relation_forces_local_done_false_without_mismatch() -> None:
    model, demonstrations = _make_model("single")
    guard = EntryGuard(
        {"single": model}, "single", _config({"single": model}, cycles=1)
    )
    unknown = _relation("object", (0.8, 0.2), RelationDecision.UNKNOWN)
    request, local = guard.evaluate(
        BoundaryId("single", 0, 1),
        {"single": _belief(model, demonstrations, 1, object_relation=unknown)},
        model.skill_states[0][-1],
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    assert not local.evidence_available
    assert not local.done
    assert not request.permitted
    own = next(
        result
        for condition_id, result in request.condition_results.items()
        if condition_id.kind == ConditionKind.OWN_RELATION
    )
    assert own.reason == "relation_unknown"
    assert request.verification_requests == ()


def test_guard_relation_and_scene_fail_closed_and_reset_streak() -> None:
    model, demonstrations = _make_model("single")
    guard = EntryGuard(
        {"single": model}, "single", _config({"single": model}, cycles=2)
    )
    boundary_id = BoundaryId("single", 0, 1)
    source = model.skill_states[0][-1]
    guard.evaluate(
        boundary_id,
        {"single": _belief(model, demonstrations, 1)},
        source,
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    unknown = _relation("target", (0.9, 0.1), RelationDecision.UNKNOWN)
    blocked, _ = guard.evaluate(
        boundary_id,
        {"single": _belief(model, demonstrations, 2, target_relation=unknown)},
        source,
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    assert not blocked.permitted
    relation = next(
        result
        for condition_id, result in blocked.condition_results.items()
        if condition_id.kind == ConditionKind.GUARD_RELATION
    )
    assert relation.reason == "relation_unknown"
    assert relation.consecutive_cycles == 0

    resumed, _ = guard.evaluate(
        boundary_id,
        {"single": _belief(model, demonstrations, 3)},
        source,
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    assert not resumed.permitted
    relation = next(
        result
        for condition_id, result in resumed.condition_results.items()
        if condition_id.kind == ConditionKind.GUARD_RELATION
    )
    assert relation.consecutive_cycles == 1


def test_pending_link_unknown_requests_verification_without_permit() -> None:
    model, demonstrations = _make_model("single")
    boundary_id = BoundaryId("single", 0, 1)
    boundary = model.boundaries[boundary_id]
    linked = RelationGuardDistribution(0.1, 0.9, "linked")
    model.boundaries[boundary_id] = replace(
        boundary,
        relation_conditions={"single/target": linked},
    )
    final = model.skill_states[0][-1]
    event_id = RelationEventId("single", "target", 0, 0, 0, "link_pending")
    model.link_pending_events[event_id] = LinkPendingCandidate(
        event_id=event_id,
        arm_id="single",
        frame_id="target",
        candidate_state=final,
        context_state=final,
        local_means=np.stack([pose(0.0)]),
        local_covariances=np.stack([np.eye(6) * 0.01]),
        gripper_commands=np.ones((1, 1)),
        demonstration_indices=(0,),
        event_local_indices=(3,),
    )
    unknown = _relation(
        "target",
        (0.3, 0.7),
        RelationDecision.UNKNOWN,
        information_weight=0.0,
    )
    guard = EntryGuard(
        {"single": model}, "single", _config({"single": model}, cycles=1)
    )
    request, local = guard.evaluate(
        boundary_id,
        {"single": _belief(model, demonstrations, 1, target_relation=unknown)},
        final,
        mode_by_arm_skill={"single": {0: 0, 1: 0}},
    )
    assert local.done
    assert not request.permitted
    assert len(request.verification_requests) == 1
    assert request.verification_requests[0].pending_event_id == event_id


def test_cross_arm_relation_guard_uses_other_arm_current_belief() -> None:
    affected = ("left", "right")
    left_model, left_demos = _make_model("left", affected_arms=affected)
    right_model, right_demos = _make_model("right", affected_arms=affected)
    boundary_id = BoundaryId("left", 0, 1)
    boundary = left_model.boundaries[boundary_id]
    old_guard = "left/target"
    cross_guard = "right/target"
    reliability = dict(boundary.condition_reliability)
    reliability[cross_guard] = reliability.pop(old_guard)
    left_model.boundaries[boundary_id] = replace(
        boundary,
        relation_conditions={cross_guard: boundary.relation_conditions[old_guard]},
        condition_reliability=reliability,
    )
    models = {"left": left_model, "right": right_model}
    guard = EntryGuard(models, "left", _config(models, cycles=1))
    right_unknown = _relation("target", (0.9, 0.1), RelationDecision.UNKNOWN)
    blocked, local = guard.evaluate(
        boundary_id,
        {
            "left": _belief(left_model, left_demos, 1),
            "right": _belief(
                right_model, right_demos, 1, target_relation=right_unknown
            ),
        },
        left_model.skill_states[0][-1],
        mode_by_arm_skill={"left": {0: 0, 1: 0}, "right": {0: 0, 1: 0}},
    )
    assert local.done
    assert not blocked.permitted
    cross_result = next(
        result
        for condition_id, result in blocked.condition_results.items()
        if condition_id.kind == ConditionKind.GUARD_RELATION
    )
    assert cross_result.condition_id.arm_id == "right"
    assert cross_result.reason == "relation_unknown"


@pytest.mark.parametrize("transaction_group", ["joint_boundary", None])
def test_multi_arm_transaction_is_joint_or_asynchronous(transaction_group) -> None:
    affected = ("left", "right")
    left_model, left_demos = _make_model(
        "left", transaction_group=transaction_group, affected_arms=affected
    )
    right_model, right_demos = _make_model(
        "right", transaction_group=transaction_group, affected_arms=affected
    )
    models = {"left": left_model, "right": right_model}
    controllers = {
        arm: ClosedLoopExecutionController(model) for arm, model in models.items()
    }
    left_belief = _belief(left_model, left_demos, 1)
    right_belief = _belief(right_model, right_demos, 1, end_probability=0.0)
    left_execution = _controller_tick(
        controllers["left"], left_model, left_demos, left_belief
    )
    right_execution = _controller_tick(
        controllers["right"], right_model, right_demos, right_belief
    )
    boundary = MultiArmBoundaryController(
        models, controllers, _config(models, cycles=1)
    )
    result = boundary.update(
        {"left": left_belief, "right": right_belief},
        mode_by_arm_skill={"left": {0: 0, 1: 0}, "right": {0: 0, 1: 0}},
    )
    assert result.requests["left"].permitted
    assert not result.requests["right"].permitted
    assert result.transaction is not None
    if transaction_group is None:
        assert [request.arm_id for request in result.transaction.committed] == ["left"]
        assert controllers["left"].cursor.reference_state.skill_index == 1
    else:
        assert result.transaction.committed == ()
        assert result.transaction.held_transaction_groups == ("joint_boundary",)
        assert controllers["left"].cursor.reference_state.skill_index == 0
    assert controllers["right"].cursor.reference_state.skill_index == 0
    assert left_execution.weighted_action.available
    assert right_execution.weighted_action.available


def test_joint_transaction_commits_both_arms_in_the_same_tick() -> None:
    affected = ("left", "right")
    left_model, left_demos = _make_model(
        "left", transaction_group="joint", affected_arms=affected
    )
    right_model, right_demos = _make_model(
        "right", transaction_group="joint", affected_arms=affected
    )
    models = {"left": left_model, "right": right_model}
    controllers = {
        arm: ClosedLoopExecutionController(model) for arm, model in models.items()
    }
    beliefs = {
        "left": _belief(left_model, left_demos, 1),
        "right": _belief(right_model, right_demos, 1),
    }
    _controller_tick(controllers["left"], left_model, left_demos, beliefs["left"])
    _controller_tick(controllers["right"], right_model, right_demos, beliefs["right"])
    boundary = MultiArmBoundaryController(
        models, controllers, _config(models, cycles=1)
    )
    result = boundary.update(
        beliefs,
        mode_by_arm_skill={"left": {0: 0, 1: 0}, "right": {0: 0, 1: 0}},
    )
    assert result.transaction is not None
    assert {request.arm_id for request in result.transaction.committed} == {
        "left",
        "right",
    }
    assert all(
        controller.cursor.reference_state.skill_index == 1
        for controller in controllers.values()
    )


def test_multi_arm_evaluation_rejects_mixed_snapshot_ticks() -> None:
    left_model, left_demos = _make_model("left", affected_arms=("left", "right"))
    right_model, right_demos = _make_model("right", affected_arms=("left", "right"))
    models = {"left": left_model, "right": right_model}
    controllers = {
        arm: ClosedLoopExecutionController(model) for arm, model in models.items()
    }
    boundary = MultiArmBoundaryController(
        models, controllers, _config(models, cycles=1)
    )
    with pytest.raises(ValueError, match="pre-action tick"):
        boundary.update(
            {
                "left": _belief(left_model, left_demos, 1),
                "right": _belief(right_model, right_demos, 2),
            }
        )
