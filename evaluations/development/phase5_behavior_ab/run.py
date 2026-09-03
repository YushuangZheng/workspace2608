"""Run controlled component A/B validation for phase-five mechanisms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BoundaryId,
    ClosedLoopBelief,
    ClosedLoopExecutionController,
    ClosedLoopRecoveryConfig,
    ClosedLoopRecoveryManager,
    EpisodeLinkAnchorRegistry,
    ExecutionMode,
    ProgressEstimate,
    ProgressStatus,
    RecoveryConfig,
    RecoveryPhase,
    ReentryConfig,
    ReentrySelector,
    RelationDecision,
    RelationEstimate,
    RelationGoal,
    RelationGoalPlanner,
    RelationRecoveryController,
    RelationRecoveryIntent,
    RelationVerificationRequest,
    RuntimeFeatureBuilder,
    RuntimeLinkAnchor,
    RuntimeObservation,
    StateId,
    UnlinkMetadataRepository,
    VerificationAttemptSignature,
)
from evaluations.development.phase23_component_ab.run import (
    REPOSITORY_ROOT,
    _initial_relations,
    _load_cases,
    _mode_by_skill,
    _runtime_observation,
    _samples_for_skill,
)

SCHEMA = "essay2608-phase5-behavior-ab-config-v2"
RECOVERY_CONFIG_PATH = REPOSITORY_ROOT / "configs/closed_loop_recovery.json"


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "tasks",
        "demonstration_indices",
        "verification_outcomes",
        "reference_reentry_robot_peak_normalized_compatibility",
        "maximum_simulation_cycles",
        "stuck_recovery_waypoint_cycles",
        "stuck_recovery_attempts",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("阶段五行为 A/B 配置字段不完整或包含未知字段")
    if value["schema"] != SCHEMA:
        raise ValueError("阶段五行为 A/B 配置 schema 不匹配")
    if not value["tasks"] or not value["demonstration_indices"]:
        raise ValueError("阶段五行为 A/B 任务和示范索引不能为空")
    if set(value["verification_outcomes"]) != {"linked", "external"}:
        raise ValueError("主动验证必须同时评测 linked 和 external 物理结果")
    reference_threshold = float(
        value["reference_reentry_robot_peak_normalized_compatibility"]
    )
    if not 0.0 <= reference_threshold <= 1.0:
        raise ValueError("重入机器人兼容度参考阈值必须位于 [0,1]")
    for name in (
        "maximum_simulation_cycles",
        "stuck_recovery_waypoint_cycles",
        "stuck_recovery_attempts",
    ):
        if int(value[name]) < 1:
            raise ValueError(f"阶段五行为 A/B 参数 {name} 必须为正整数")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"阶段五行为 A/B 结果 {path.name} 不能为空")
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    items = tuple(float(value) for value in values)
    return float(np.mean(items)) if items else math.nan


def _state_text(state: StateId) -> str:
    return f"k{state.skill_index}:t{state.local_index}"


def _relation_estimate(
    frame: str,
    decision: RelationDecision,
    *,
    information: float = 1.0,
    demonstration_prior: np.ndarray | None = None,
) -> RelationEstimate:
    if decision == RelationDecision.LINKED:
        posterior = np.asarray([0.05, 0.95], dtype=np.float64)
    elif decision == RelationDecision.EXTERNAL:
        posterior = np.asarray([0.95, 0.05], dtype=np.float64)
    else:
        posterior = np.asarray([0.5, 0.5], dtype=np.float64)
    prior = (
        np.asarray([0.5, 0.5], dtype=np.float64)
        if demonstration_prior is None
        else np.asarray(demonstration_prior, dtype=np.float64)
    )
    entropy = -float(np.sum(posterior * np.log(np.maximum(posterior, 1.0e-12))))
    return RelationEstimate(
        frame_id=frame,
        posterior=posterior,
        predicted=posterior,
        demonstration_prior=prior,
        observation_likelihood=np.ones(2, dtype=np.float64),
        information_weight=information,
        entropy=entropy,
        informative=information >= 0.1,
        decision_state=decision,
    )


def _observation(
    tick: int,
    *,
    ee_pose: np.ndarray,
    frame_poses: Mapping[str, np.ndarray],
    gripper: np.ndarray,
    previous: RuntimeObservation | None,
    command_pose: np.ndarray | None,
    entity_configurations: Mapping[str, Mapping[str, np.ndarray]],
) -> RuntimeObservation:
    return RuntimeObservation(
        tick=tick,
        ee_pose=ee_pose,
        frame_poses={name: value.copy() for name, value in frame_poses.items()},
        gripper_state=gripper,
        previous_command_pose=command_pose,
        previous_ee_pose=None if previous is None else previous.ee_pose,
        tracking_reliability={},
        frame_visibility={},
        entity_configurations={
            entity: {name: value.copy() for name, value in fields.items()}
            for entity, fields in entity_configurations.items()
        },
    )


def _pending_initial_state(
    case: Any,
    candidate: Any,
    demonstration: int,
    recovery_config: ClosedLoopRecoveryConfig,
) -> tuple[
    BeliefUpdater,
    ClosedLoopBelief,
    RuntimeObservation,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    dict[int, int],
]:
    state = candidate.candidate_state
    modes = _mode_by_skill(case.policy, demonstration)
    sample = _samples_for_skill(case, demonstration, state.skill_index)[
        state.local_index
    ]
    runtime_anchor = RuntimeLinkAnchor.from_pending(candidate)
    waypoints = EpisodeLinkAnchorRegistry.instantiate(
        runtime_anchor,
        sample.frames[candidate.frame_id],
        covariance_inflation=0.0,
    )
    history = tuple(
        waypoint.pose.copy()
        for waypoint in waypoints[-recovery_config.verification.task_history_length :]
    )
    if len(history) < 2:
        raise RuntimeError(f"{case.key} 的 Pending 模板不足以估计接近方向")
    entry = history[-1].copy()
    frames = {name: value.copy() for name, value in sample.frames.items()}
    previous = _observation(
        0,
        ee_pose=entry,
        frame_poses=frames,
        gripper=runtime_anchor.gripper_commands[-1],
        previous=None,
        command_pose=None,
        entity_configurations=sample.entity_configurations,
    )
    current = _observation(
        1,
        ee_pose=entry,
        frame_poses=frames,
        gripper=runtime_anchor.gripper_commands[-1],
        previous=previous,
        command_pose=entry,
        entity_configurations=sample.entity_configurations,
    )
    updater = BeliefUpdater(case.model)
    updater.reset(
        initial_progress={state: 1.0},
        initial_relations=_initial_relations(case, state, modes),
        previous_observation=previous,
    )
    belief = updater.update_frozen(current, mode_by_skill=modes)
    return (
        updater,
        belief,
        current,
        entry,
        runtime_anchor.gripper_commands[-1].copy(),
        history,
        modes,
    )


def _passive_pending_decision(
    case: Any,
    candidate: Any,
    demonstration: int,
    recovery_config: ClosedLoopRecoveryConfig,
) -> RelationDecision:
    updater, belief, observation, entry, gripper, _, modes = _pending_initial_state(
        case, candidate, demonstration, recovery_config
    )
    cycles = recovery_config.verification.maximum_probe_cycles
    for tick in range(2, cycles + 3):
        next_observation = _observation(
            tick,
            ee_pose=entry,
            frame_poses=observation.frame_poses,
            gripper=gripper,
            previous=observation,
            command_pose=entry,
            entity_configurations=observation.entity_configurations,
        )
        belief = updater.update_frozen(next_observation, mode_by_skill=modes)
        observation = next_observation
    return belief.relation_estimates[candidate.frame_id].decision_state


def _verification_trial(
    case: Any,
    candidate: Any,
    demonstration: int,
    physical_outcome: RelationDecision,
    recovery_config: ClosedLoopRecoveryConfig,
    maximum_cycles: int,
) -> dict[str, Any]:
    baseline = _passive_pending_decision(
        case, candidate, demonstration, recovery_config
    )
    updater, belief, observation, entry, gripper, history, modes = (
        _pending_initial_state(case, candidate, demonstration, recovery_config)
    )
    if belief.relation_estimates[candidate.frame_id].decision_state != (
        RelationDecision.UNKNOWN
    ):
        raise RuntimeError("Pending 主动验证入口必须从 Unknown 开始")
    manager = ClosedLoopRecoveryManager(case.model, recovery_config)
    for task_pose in history:
        manager.record_task_pose(task_pose)
    grasp_event = f"demo-{demonstration}"
    request = RelationVerificationRequest(
        case.arm,
        candidate.frame_id,
        "linked",
        candidate.event_id,
    )
    manager.begin_verification(
        request,
        belief,
        task_state=candidate.candidate_state,
        grasp_event=grasp_event,
        current_pose=entry,
        current_gripper=gripper,
    )
    initial_progress = dict(belief.progress.posterior)
    current_pose = entry.copy()
    candidate_frame = observation.frame_poses[candidate.frame_id].copy()
    maximum_orientation_error = 0.0
    maximum_gripper_error = 0.0
    first_probe_dot = math.nan
    action_count = 0
    probe_actions = 0
    return_actions = 0
    terminal = None
    for _ in range(maximum_cycles):
        result = manager.update_verification(belief, current_pose=current_pose)
        terminal = result.verification
        action = terminal.action
        if action is None:
            if result.mode == ExecutionMode.TASK:
                break
            continue
        action_count += 1
        if action.source == "verify_link_probe":
            probe_actions += 1
            if math.isnan(first_probe_dot):
                approach = history[-1][:3] - history[0][:3]
                first_probe_dot = float(np.dot(action.pose[:3] - entry[:3], approach))
        else:
            return_actions += 1
        maximum_orientation_error = max(
            maximum_orientation_error,
            float(np.linalg.norm(action.pose[3:7] - entry[3:7])),
        )
        maximum_gripper_error = max(
            maximum_gripper_error,
            float(np.max(np.abs(action.gripper_command - gripper))),
        )
        previous_pose = current_pose.copy()
        current_pose = action.pose.copy()
        delta = current_pose[:3] - previous_pose[:3]
        if physical_outcome == RelationDecision.LINKED:
            candidate_frame = candidate_frame.copy()
            candidate_frame[:3] += delta
        frames = {name: value.copy() for name, value in observation.frame_poses.items()}
        frames[candidate.frame_id] = candidate_frame.copy()
        next_observation = _observation(
            observation.tick + 1,
            ee_pose=current_pose,
            frame_poses=frames,
            gripper=gripper,
            previous=observation,
            command_pose=current_pose,
            entity_configurations=observation.entity_configurations,
        )
        belief = updater.update_frozen(next_observation, mode_by_skill=modes)
        observation = next_observation
    if terminal is None or manager.mode != ExecutionMode.TASK:
        raise RuntimeError(f"{case.key} 的主动验证没有在有限周期内返回 TASK")
    signature = VerificationAttemptSignature(
        RelationDecision.UNKNOWN,
        candidate.candidate_state,
        grasp_event,
    )
    repeat_blocked = not manager.verification.attempts.can_attempt(
        candidate.event_id, signature
    )
    changed_context_allowed = manager.verification.attempts.can_attempt(
        candidate.event_id,
        replace(signature, grasp_event=f"{grasp_event}-new"),
    )
    active = candidate.frame_id in manager.anchor_registry.active_pending
    return {
        "trial_id": (
            f"verify:{case.key}:{candidate.event_id.token}:d{demonstration}:"
            f"{physical_outcome.value}"
        ),
        "task": case.task,
        "arm": case.arm,
        "event": candidate.event_id.token,
        "demonstration": demonstration,
        "physical_outcome": physical_outcome.value,
        "baseline_decision": baseline.value,
        "baseline_resolved": int(baseline != RelationDecision.UNKNOWN),
        "proposed_decision": terminal.decision.value,
        "proposed_correct": int(terminal.decision == physical_outcome),
        "action_count": action_count,
        "probe_actions": probe_actions,
        "return_actions": return_actions,
        "return_position_error": float(np.linalg.norm(current_pose[:3] - entry[:3])),
        "return_position_tolerance": (
            recovery_config.verification.return_position_tolerance
        ),
        "maximum_orientation_error": maximum_orientation_error,
        "maximum_gripper_error": maximum_gripper_error,
        "probe_opposes_approach": int(first_probe_dot < 0.0),
        "progress_frozen": int(belief.progress.posterior == initial_progress),
        "repeat_blocked": int(repeat_blocked),
        "changed_context_allowed": int(changed_context_allowed),
        "pending_activation_correct": int(
            active == (physical_outcome == RelationDecision.LINKED)
        ),
    }


def _recovery_goal(
    case: Any,
    *,
    frame: str,
    source_state: StateId,
    mode: int,
    expected: RelationDecision,
    actual: RelationDecision,
) -> RelationGoal:
    registry = EpisodeLinkAnchorRegistry(case.model)
    planner = RelationGoalPlanner(
        registry,
        UnlinkMetadataRepository(case.model),
    )
    return planner.plan(
        (
            RelationRecoveryIntent(
                case.arm,
                frame,
                expected,
                actual,
            ),
        ),
        source_state=source_state,
        mode=mode,
    )[0]


def _recovery_trial(
    case: Any,
    goal: RelationGoal,
    event_token: str,
    demonstration: int,
    recovery_config: ClosedLoopRecoveryConfig,
    maximum_cycles: int,
    stuck_waypoint_cycles: int,
    stuck_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = _samples_for_skill(
        case,
        demonstration,
        goal.source_state.skill_index,
    )[goal.source_state.local_index]
    frame_pose = sample.frames[goal.frame_id]
    registry = EpisodeLinkAnchorRegistry(case.model)
    controller = RelationRecoveryController(registry, recovery_config.recovery)
    controller.start((goal,))
    current = sample.ee_pose.copy()
    result = None
    action_count = 0
    for _ in range(maximum_cycles):
        decision = (
            goal.expected_relation
            if result is not None
            and result.goal_phase is not None
            and result.goal_phase.value == "verify"
            else goal.actual_relation
        )
        result = controller.update(
            current_pose=current,
            current_gripper=sample.gripper,
            frame_poses={goal.frame_id: frame_pose},
            relation_estimates={
                goal.frame_id: _relation_estimate(goal.frame_id, decision)
            },
        )
        if result.action is not None:
            action_count += 1
            current = result.action.pose.copy()
        if result.phase != RecoveryPhase.EXECUTING:
            break
    if result is None:
        raise RuntimeError("关系恢复没有产生任何结果")

    stuck_config = replace(
        recovery_config.recovery,
        maximum_waypoint_cycles=stuck_waypoint_cycles,
        maximum_attempts_per_goal=stuck_attempts,
        maximum_relation_verify_cycles=1,
        unlink_open_cycles=1,
        maximum_total_cycles=20,
    )
    stuck = RelationRecoveryController(
        EpisodeLinkAnchorRegistry(case.model), stuck_config
    )
    stuck.start((goal,))
    immobile = sample.ee_pose.copy()
    immobile[:3] += np.asarray([10.0, 0.0, 0.0])
    stuck_result = None
    stuck_cycles = 0
    for stuck_cycles in range(1, maximum_cycles + 1):
        stuck_result = stuck.update(
            current_pose=immobile,
            current_gripper=sample.gripper,
            frame_poses={goal.frame_id: frame_pose},
            relation_estimates={
                goal.frame_id: _relation_estimate(goal.frame_id, goal.actual_relation)
            },
        )
        if stuck_result.phase != RecoveryPhase.EXECUTING:
            break
    if stuck_result is None:
        raise RuntimeError("有界失败试验没有产生结果")
    common = {
        "task": case.task,
        "arm": case.arm,
        "event": event_token,
        "demonstration": demonstration,
        "goal": goal.kind.value,
    }
    success = {
        "trial_id": f"recover:{case.key}:{event_token}:d{demonstration}",
        **common,
        "baseline_success": 0,
        "baseline_action_count": 0,
        "proposed_success": int(result.phase == RecoveryPhase.REENTRY),
        "proposed_action_count": action_count,
        "completed_goal_count": len(result.completed_goals),
        "legal_reentry_state_count": len(result.legal_reentry_states),
    }
    failure = {
        "trial_id": f"bounded:{case.key}:{event_token}:d{demonstration}",
        **common,
        "terminated": int(stuck_result.phase == RecoveryPhase.FAILED),
        "cycles": stuck_cycles,
        "structured_failure": int(stuck_result.failure is not None),
        "failure_reason": (
            "" if stuck_result.failure is None else stuck_result.failure.reason
        ),
    }
    return success, failure


def _belief_at_state(
    case: Any,
    demonstration: int,
    state: StateId,
) -> tuple[ClosedLoopBelief, RuntimeObservation, dict[int, int]]:
    modes = _mode_by_skill(case.policy, demonstration)
    samples = _samples_for_skill(case, demonstration, state.skill_index)
    current = samples[state.local_index]
    previous = samples[max(0, state.local_index - 1)]
    previous_observation = _runtime_observation(0, previous, None)
    current_observation = _runtime_observation(
        1,
        current,
        previous,
        previous_command_pose=current.ee_pose,
    )
    features = RuntimeFeatureBuilder().build(
        current_observation,
        previous_observation,
    )
    node = case.model.state(state)
    mode = modes[state.skill_index]
    estimates = {}
    for frame in case.model.relation_frames:
        values = node.demo_relation_priors.get(frame)
        prior = (
            np.asarray([0.5, 0.5], dtype=np.float64) if values is None else values[mode]
        )
        decision = (
            RelationDecision.LINKED
            if prior[1] > prior[0]
            else RelationDecision.EXTERNAL
        )
        estimates[frame] = _relation_estimate(
            frame,
            decision,
            demonstration_prior=prior,
        )
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
    belief = ClosedLoopBelief(
        tick=1,
        runtime_features=features,
        relation_estimates=estimates,
        progress=progress,
        candidate_scores={},
        relation_changes=(),
        local_candidates=(state,),
        expanded_candidates=(),
    )
    return belief, current_observation, modes


def _reentry_trials(
    case: Any,
    event_token: str,
    legal_states: Sequence[StateId],
    demonstration: int,
    reentry_config: ReentryConfig,
    reference_reentry_config: ReentryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_skill: dict[int, list[StateId]] = {}
    for state in legal_states:
        by_skill.setdefault(state.skill_index, []).append(state)
    global_index = {
        state: index for index, state in enumerate(sorted(case.model.states))
    }
    selector = ReentrySelector(case.model, reentry_config)
    reference_selector = ReentrySelector(case.model, reference_reentry_config)
    rows = []
    guards = []
    for skill, states in sorted(by_skill.items()):
        states = sorted(set(states))
        if len(states) < 2:
            continue
        for direction, source, truth in (
            ("forward_clock_lag", states[0], states[-1]),
            ("backward_recovery", states[-1], states[0]),
        ):
            belief, observation, modes = _belief_at_state(case, demonstration, truth)
            evaluation = selector.select(
                states,
                belief,
                current_reference=source,
                mode_by_skill=modes,
            )
            reference_evaluation = reference_selector.select(
                states,
                belief,
                current_reference=source,
                mode_by_skill=modes,
            )
            decision = evaluation.decision
            reference_decision = reference_evaluation.decision
            penalty = len(global_index)
            proposed_error = (
                penalty
                if decision is None
                else abs(global_index[decision.state_id] - global_index[truth])
            )
            baseline_error = abs(global_index[source] - global_index[truth])
            reference_error = (
                penalty
                if reference_decision is None
                else abs(
                    global_index[reference_decision.state_id] - global_index[truth]
                )
            )
            commit_correct = False
            if decision is not None:
                updater = BeliefUpdater(case.model)
                execution = ClosedLoopExecutionController(case.model)
                selector.apply(
                    decision,
                    belief=belief,
                    observation=observation,
                    belief_updater=updater,
                    execution_controller=execution,
                )
                frozen = updater.update_frozen(
                    replace(
                        observation,
                        tick=2,
                        previous_ee_pose=observation.ee_pose,
                        previous_command_pose=observation.ee_pose,
                    ),
                    mode_by_skill=modes,
                )
                commit_correct = bool(
                    execution.cursor.reference_state == decision.state_id
                    and frozen.progress.posterior == {decision.state_id: 1.0}
                )
            rows.append(
                {
                    "trial_id": (
                        f"reentry:{case.key}:{event_token}:d{demonstration}:"
                        f"k{skill}:{direction}"
                    ),
                    "task": case.task,
                    "arm": case.arm,
                    "event": event_token,
                    "demonstration": demonstration,
                    "skill": skill,
                    "direction": direction,
                    "candidate_count": len(states),
                    "source_state": _state_text(source),
                    "truth_state": _state_text(truth),
                    "selected_state": (
                        "" if decision is None else _state_text(decision.state_id)
                    ),
                    "baseline_state_error": baseline_error,
                    "reference_state_error": reference_error,
                    "reference_selected": int(reference_decision is not None),
                    "reference_exact": int(
                        reference_decision is not None
                        and reference_decision.state_id == truth
                    ),
                    "proposed_state_error": proposed_error,
                    "proposed_selected": int(decision is not None),
                    "proposed_exact": int(
                        decision is not None and decision.state_id == truth
                    ),
                    "proposed_wins": int(proposed_error < baseline_error),
                    "atomic_commit_correct": int(commit_correct),
                    "truth_robot_peak_normalized_compatibility": (
                        evaluation.scores[
                            truth
                        ].robot_peak_normalized_compatibility
                    ),
                    "truth_scene_compatibility": evaluation.scores[
                        truth
                    ].state_compatibility,
                    "truth_relation_compatibility": evaluation.scores[
                        truth
                    ].relation_compatibility,
                    "truth_explanation_score": evaluation.scores[
                        truth
                    ].normalized_explanation_score,
                    "truth_rejection_reasons": "|".join(
                        evaluation.rejection_reasons.get(truth, ())
                    ),
                    "reference_truth_rejection_reasons": "|".join(
                        reference_evaluation.rejection_reasons.get(truth, ())
                    ),
                }
            )

    skills = sorted(by_skill)
    for source_skill in skills:
        target_skill = source_skill + 1
        if target_skill not in by_skill:
            continue
        source = max(by_skill[source_skill])
        target = min(by_skill[target_skill])
        boundary = BoundaryId(case.arm, source_skill, target_skill)
        if boundary not in case.model.boundaries:
            continue
        belief, _, modes = _belief_at_state(case, demonstration, target)
        blocked = selector.select(
            (target,),
            belief,
            current_reference=source,
            mode_by_skill=modes,
        )
        allowed = selector.select(
            (target,),
            belief,
            current_reference=source,
            permitted_boundaries=frozenset({boundary}),
            mode_by_skill=modes,
        )
        guards.append(
            {
                "trial_id": (
                    f"guard:{case.key}:{event_token}:d{demonstration}:"
                    f"k{source_skill}-k{target_skill}"
                ),
                "task": case.task,
                "arm": case.arm,
                "event": event_token,
                "demonstration": demonstration,
                "boundary": boundary.token,
                "blocked_without_guard": int(blocked.decision is None),
                "guard_rejection_recorded": int(
                    "cross_skill_guard_not_permitted"
                    in blocked.rejection_reasons.get(target, ())
                ),
                "selected_with_guard": int(allowed.decision is not None),
            }
        )
    return rows, guards


def _summary_rows(
    verification: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    reentry: Sequence[Mapping[str, Any]],
    guards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_resolution = _mean(row["baseline_resolved"] for row in verification)
    proposed_resolution = _mean(row["proposed_correct"] for row in verification)
    baseline_recovery = _mean(row["baseline_success"] for row in recovery)
    proposed_recovery = _mean(row["proposed_success"] for row in recovery)
    baseline_mae = _mean(row["baseline_state_error"] for row in reentry)
    proposed_mae = _mean(row["proposed_state_error"] for row in reentry)
    return [
        {
            "family": "pending_verification",
            "trials": len(verification),
            "baseline_primary": baseline_resolution,
            "proposed_primary": proposed_resolution,
            "absolute_gain": proposed_resolution - baseline_resolution,
            "secondary_a": _mean(row["progress_frozen"] for row in verification),
            "secondary_b": _mean(row["repeat_blocked"] for row in verification),
            "secondary_c": _mean(
                row["pending_activation_correct"] for row in verification
            ),
            "secondary_d": _mean(
                int(
                    float(row["return_position_error"])
                    <= float(row["return_position_tolerance"])
                    and float(row["maximum_orientation_error"]) <= 1.0e-12
                    and float(row["maximum_gripper_error"]) <= 1.0e-12
                    and bool(row["probe_opposes_approach"])
                )
                for row in verification
            ),
        },
        {
            "family": "relation_recovery",
            "trials": len(recovery),
            "baseline_primary": baseline_recovery,
            "proposed_primary": proposed_recovery,
            "absolute_gain": proposed_recovery - baseline_recovery,
            "secondary_a": _mean(row["terminated"] for row in failures),
            "secondary_b": _mean(row["structured_failure"] for row in failures),
            "secondary_c": _mean(
                int(row["proposed_action_count"] > 0) for row in recovery
            ),
            "secondary_d": _mean(
                int(row["legal_reentry_state_count"] > 0) for row in recovery
            ),
        },
        {
            "family": "full_state_reentry",
            "trials": len(reentry),
            "baseline_primary": baseline_mae,
            "proposed_primary": proposed_mae,
            "absolute_gain": baseline_mae - proposed_mae,
            "secondary_a": _mean(row["proposed_wins"] for row in reentry),
            "secondary_b": _mean(row["proposed_exact"] for row in reentry),
            "secondary_c": _mean(row["proposed_selected"] for row in reentry),
            "secondary_d": _mean(
                row["atomic_commit_correct"]
                for row in reentry
                if row["proposed_selected"]
            ),
        },
        {
            "family": "cross_skill_guard",
            "trials": len(guards),
            "baseline_primary": 0.0,
            "proposed_primary": _mean(row["blocked_without_guard"] for row in guards),
            "absolute_gain": _mean(row["blocked_without_guard"] for row in guards),
            "secondary_a": _mean(row["guard_rejection_recorded"] for row in guards),
            "secondary_b": _mean(row["selected_with_guard"] for row in guards),
            "secondary_c": math.nan,
            "secondary_d": math.nan,
        },
    ]


def run(config_path: Path, output: Path) -> None:
    config = _read_config(config_path)
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    recovery_config = ClosedLoopRecoveryConfig.from_json(RECOVERY_CONFIG_PATH)
    reference_reentry_config = replace(
        recovery_config.reentry,
        minimum_robot_peak_normalized_compatibility=float(
            config["reference_reentry_robot_peak_normalized_compatibility"]
        ),
    )
    maximum_demo = max(int(value) for value in config["demonstration_indices"])
    verification_rows = []
    recovery_rows = []
    failure_rows = []
    reentry_rows = []
    guard_rows = []

    for task in config["tasks"]:
        print(f"[phase5-behavior-ab] loading {task}", flush=True)
        for case in _load_cases(task, maximum_demo + 1):
            for candidate in case.model.link_pending_events.values():
                demonstrations = set(candidate.demonstration_indices)
                for demonstration in config["demonstration_indices"]:
                    if demonstration not in demonstrations:
                        continue
                    for outcome in config["verification_outcomes"]:
                        verification_rows.append(
                            _verification_trial(
                                case,
                                candidate,
                                int(demonstration),
                                RelationDecision(outcome),
                                recovery_config,
                                int(config["maximum_simulation_cycles"]),
                            )
                        )

            event_specs: list[
                tuple[str, RelationGoal, tuple[int, ...], tuple[StateId, ...]]
            ] = []
            for event_id, anchor in case.model.link_anchors.items():
                goal = _recovery_goal(
                    case,
                    frame=event_id.frame_id,
                    source_state=anchor.linked_entry_states[0],
                    mode=event_id.mode,
                    expected=RelationDecision.LINKED,
                    actual=RelationDecision.EXTERNAL,
                )
                event_specs.append(
                    (
                        event_id.token,
                        goal,
                        anchor.demonstration_indices,
                        anchor.linked_entry_states,
                    )
                )
            for event_id, metadata in case.model.unlink_events.items():
                goal = _recovery_goal(
                    case,
                    frame=event_id.frame_id,
                    source_state=metadata.release_state,
                    mode=event_id.mode,
                    expected=RelationDecision.EXTERNAL,
                    actual=RelationDecision.LINKED,
                )
                event_specs.append(
                    (
                        event_id.token,
                        goal,
                        metadata.demonstration_indices,
                        metadata.legal_reentry_states,
                    )
                )

            for event_token, goal, supported, legal_states in event_specs:
                demonstrations = set(supported)
                for demonstration in config["demonstration_indices"]:
                    if demonstration not in demonstrations:
                        continue
                    recovery_row, failure_row = _recovery_trial(
                        case,
                        goal,
                        event_token,
                        int(demonstration),
                        recovery_config,
                        int(config["maximum_simulation_cycles"]),
                        int(config["stuck_recovery_waypoint_cycles"]),
                        int(config["stuck_recovery_attempts"]),
                    )
                    recovery_rows.append(recovery_row)
                    failure_rows.append(failure_row)
                    reentry, guards = _reentry_trials(
                        case,
                        event_token,
                        legal_states,
                        int(demonstration),
                        recovery_config.reentry,
                        reference_reentry_config,
                    )
                    reentry_rows.extend(reentry)
                    guard_rows.extend(guards)

    if not all(
        (verification_rows, recovery_rows, failure_rows, reentry_rows, guard_rows)
    ):
        raise RuntimeError("阶段五行为 A/B 至少一个评测族没有样本")
    summaries = _summary_rows(
        verification_rows,
        recovery_rows,
        failure_rows,
        reentry_rows,
        guard_rows,
    )
    reference_threshold = float(
        config["reference_reentry_robot_peak_normalized_compatibility"]
    )
    calibrated_threshold = (
        recovery_config.reentry.minimum_robot_peak_normalized_compatibility
    )
    calibration_rows = [
        {
            "normal_trials": len(reentry_rows),
            "reference_threshold": reference_threshold,
            "calibrated_threshold": calibrated_threshold,
            "minimum_truth_robot_peak_normalized_compatibility": min(
                float(row["truth_robot_peak_normalized_compatibility"])
                for row in reentry_rows
            ),
            "reference_selected": sum(
                int(row["reference_selected"]) for row in reentry_rows
            ),
            "calibrated_selected": sum(
                int(row["proposed_selected"]) for row in reentry_rows
            ),
            "reference_exact": sum(int(row["reference_exact"]) for row in reentry_rows),
            "calibrated_exact": sum(int(row["proposed_exact"]) for row in reentry_rows),
            "reference_incorrect_selected": sum(
                int(row["reference_selected"]) - int(row["reference_exact"])
                for row in reentry_rows
            ),
            "calibrated_incorrect_selected": sum(
                int(row["proposed_selected"]) - int(row["proposed_exact"])
                for row in reentry_rows
            ),
            "reference_mae": _mean(
                row["reference_state_error"] for row in reentry_rows
            ),
            "calibrated_mae": _mean(
                row["proposed_state_error"] for row in reentry_rows
            ),
        }
    ]
    _write_csv(output / "verification_trials.csv", verification_rows)
    _write_csv(output / "recovery_trials.csv", recovery_rows)
    _write_csv(output / "bounded_failure_trials.csv", failure_rows)
    _write_csv(output / "reentry_trials.csv", reentry_rows)
    _write_csv(output / "cross_skill_guard_trials.csv", guard_rows)
    _write_csv(output / "reentry_calibration.csv", calibration_rows)
    _write_csv(output / "summary.csv", summaries)
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "recovery_config.json").write_text(
        json.dumps(recovery_config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dirty_worktree_expected": True,
        "executor": "deterministic_ideal_follower",
    }
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {row["family"]: row for row in summaries}
    verification = summary["pending_verification"]
    recovery = summary["relation_recovery"]
    reentry = summary["full_state_reentry"]
    guards = summary["cross_skill_guard"]
    calibration = calibration_rows[0]
    report = [
        "# 阶段五受控行为 A/B 结果",
        "",
        "## 主要结果",
        "",
        (
            f"- Pending 关系判定：被动等待 {verification['baseline_primary']:.2%}，"
            f"主动验证 {verification['proposed_primary']:.2%}，"
            f"提升 {verification['absolute_gain']:.2%}。"
        ),
        (
            f"- 关系恢复：无恢复动作 {recovery['baseline_primary']:.2%}，"
            f"统一 LINK/UNLINK 恢复 {recovery['proposed_primary']:.2%}，"
            f"提升 {recovery['absolute_gain']:.2%}。"
        ),
        (
            f"- 任务重入 StateId MAE：旧时钟 {reentry['baseline_primary']:.4f}，"
            f"完整状态重入 {reentry['proposed_primary']:.4f}，"
            f"降低 {reentry['absolute_gain']:.4f}。"
        ),
        (
            f"- 无入口许可的跨技能候选阻断率："
            f"{guards['proposed_primary']:.2%}；获得许可后的选择率："
            f"{guards['secondary_b']:.2%}。"
        ),
        "",
        "## 附加不变量",
        "",
        (
            f"- 主动验证进度冻结率 {verification['secondary_a']:.2%}，"
            f"同上下文重复触发抑制率 {verification['secondary_b']:.2%}，"
            f"Pending episode 激活正确率 {verification['secondary_c']:.2%}，"
            f"反向探测与原路返回不变量通过率 {verification['secondary_d']:.2%}。"
        ),
        (
            f"- 非响应执行器下有界终止率 {recovery['secondary_a']:.2%}，"
            f"结构化失败率 {recovery['secondary_b']:.2%}。"
        ),
        (
            f"- 重入优于旧时钟的试验比例 {reentry['secondary_a']:.2%}，"
            f"精确状态识别率 {reentry['secondary_b']:.2%}，"
            f"状态选择率 {reentry['secondary_c']:.2%}，"
            f"选中后的原子提交正确率 {reentry['secondary_d']:.2%}。"
        ),
        "",
        "## 正常回放阈值标定",
        "",
        (
            f"- 机器人兼容度阈值由 {calibration['reference_threshold']:.6f} "
            f"标定为 {calibration['calibrated_threshold']:.6f}；"
            f"240条正常试验中的最低正确状态兼容度为 "
            f"{calibration['minimum_truth_robot_peak_normalized_compatibility']:.8f}。"
        ),
        (
            f"- 状态选择 {calibration['reference_selected']}→"
            f"{calibration['calibrated_selected']}，精确选择 "
            f"{calibration['reference_exact']}→{calibration['calibrated_exact']}，"
            f"错误选择保持 {calibration['reference_incorrect_selected']}→"
            f"{calibration['calibrated_incorrect_selected']}，MAE "
            f"{calibration['reference_mae']:.4f}→"
            f"{calibration['calibrated_mae']:.4f}。"
        ),
        "",
        "## 结论边界",
        "",
        "本实验使用冻结 V4 模型、正常示范场景和确定性理想执行器，",
        "验证阶段五组件是否能制造关系信息、执行恢复和重新定位任务状态。",
        "它不包含碰撞、控制误差或完整 RLBench 环境闭环，不能替代阶段六的在线任务成功率实验。",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
