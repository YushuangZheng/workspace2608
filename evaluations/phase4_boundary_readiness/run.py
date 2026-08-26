"""Audit phase-four boundary readiness with normal-demonstration replay."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from essay2608.policy import DynaMACObservation
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    ClosedLoopExecutionConfig,
    ClosedLoopExecutionController,
    ProgressStatus,
    RelationDecision,
    StateId,
)
from essay2608.policy.dynamac import relative_pose
from evaluations.phase23_component_ab.run import (
    MODEL_PATHS,
    REPOSITORY_ROOT,
    _initial_relations,
    _load_cases,
    _mode_by_skill,
    _runtime_observation,
    _samples_for_skill,
)

SCHEMA = "essay2608-phase4-boundary-readiness-config-v1"
BELIEF_CONFIG_PATH = REPOSITORY_ROOT / "configs/closed_loop_belief.json"
EXECUTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/closed_loop_execution.json"


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "claim_boundary",
        "tasks",
        "demonstration_indices",
        "minimum_tracking_reliability",
        "minimum_consecutive_samples_for_h_audit",
        "priority_cases",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("边界就绪性配置根字段不完整或包含未知字段")
    if value["schema"] != SCHEMA:
        raise ValueError("边界就绪性配置 schema 不匹配")
    if not value["tasks"] or not value["demonstration_indices"]:
        raise ValueError("任务和示范索引不能为空")
    if not 0.0 <= float(value["minimum_tracking_reliability"]) <= 1.0:
        raise ValueError("最低跟踪可靠性必须位于 [0,1]")
    if int(value["minimum_consecutive_samples_for_h_audit"]) < 2:
        raise ValueError("H 检查至少需要两个连续样本")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"不能写入空结果 {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    data = tuple(float(value) for value in values)
    return float(np.mean(data)) if data else math.nan


def _minimum(values: Iterable[float], default: float = math.nan) -> float:
    finite = tuple(float(value) for value in values if np.isfinite(value))
    return min(finite) if finite else default


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return float(
        math.exp(np.mean(np.log(np.maximum(values, np.finfo(np.float64).tiny))))
    )


def _worst_robot_frame(score: Any | None) -> tuple[str, float]:
    if score is None or not score.robot_frame_terms:
        return "", math.nan
    supports = {
        frame: float(score.robot_frame_terms[frame])
        / float(score.robot_frame_weights[frame])
        for frame in score.robot_frame_terms
        if score.robot_frame_weights.get(frame, 0.0) > 0.0
    }
    if not supports:
        return "", math.nan
    return min(supports.items(), key=lambda item: item[1])


def _state_text(state_id: StateId) -> str:
    return f"k{state_id.skill_index}:t{state_id.local_index}"


def _factor_observation(factor: Any, features: Any) -> tuple[np.ndarray | None, float]:
    if factor.kind == "node":
        value = features.entity_configurations.get(factor.source, {}).get(
            factor.feature
        )
        if value is None:
            return None, 0.0
        reliability = (
            features.tracking_reliability.get(factor.source, 0.0)
            if factor.source in features.frame_visibility
            else 1.0
        )
        if (
            factor.source in features.frame_visibility
            and not features.frame_visibility[factor.source]
        ):
            reliability = 0.0
        return value, float(reliability)
    assert factor.target is not None
    if (
        factor.source not in features.frame_poses
        or factor.target not in features.frame_poses
    ):
        return None, 0.0
    if not features.frame_visibility.get(
        factor.source, False
    ) or not features.frame_visibility.get(factor.target, False):
        return None, 0.0
    reliability = min(
        features.tracking_reliability.get(factor.source, 0.0),
        features.tracking_reliability.get(factor.target, 0.0),
    )
    return (
        relative_pose(
            features.frame_poses[factor.target],
            features.frame_poses[factor.source],
        ),
        float(reliability),
    )


def _goal_rows(
    case: Any,
    boundary: Any,
    demonstration: int,
    mode: int,
    state_id: StateId,
    sample: Any,
) -> list[dict[str, Any]]:
    rows = []
    prefix = f"m{mode}:"
    local = boundary.local_completion_model
    for key, distribution in sorted(local.goal_distributions.items()):
        if not key.startswith(prefix):
            continue
        frame = key.split(":", 1)[1]
        observed = frame in sample.frames
        log_likelihood = math.nan
        compatibility = math.nan
        threshold = float(local.minimum_goal_log_likelihood[key])
        if observed:
            value = relative_pose(sample.frames[frame], sample.ee_pose)
            log_likelihood = distribution.log_likelihood(value)
            compatibility = distribution.compatibility(value)
        rows.append(
            {
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration,
                "boundary": boundary.boundary_id.token,
                "state": _state_text(state_id),
                "family": "local_goal",
                "condition": key,
                "target": "final_support",
                "observed": int(observed),
                "reliability": 1.0 if observed else 0.0,
                "decision_stable": "",
                "target_decision_match": "",
                "compatibility": compatibility,
                "threshold": threshold,
                "log_likelihood": log_likelihood,
                "log_likelihood_margin": (
                    log_likelihood - threshold if observed else math.nan
                ),
                "threshold_pass": int(observed and log_likelihood >= threshold),
            }
        )
    return rows


def _relation_rows(
    case: Any,
    boundary: Any,
    demonstration: int,
    state_id: StateId,
    belief: Any,
    family: str,
    conditions: Mapping[str, Any],
    minimum_tracking_reliability: float,
) -> list[dict[str, Any]]:
    rows = []
    for key, condition in sorted(conditions.items()):
        arm, frame = key.split("/", 1)
        if arm != case.model.arm_id:
            raise RuntimeError(
                f"{boundary.boundary_id.token} 条件 {key} 不能由本臂关系后验求值"
            )
        estimate = belief.relation_estimates.get(frame)
        visible = belief.runtime_features.frame_visibility.get(frame, False)
        reliability = belief.runtime_features.tracking_reliability.get(frame, 0.0)
        observed = bool(
            estimate is not None
            and visible
            and reliability >= minimum_tracking_reliability
        )
        decision = "missing" if estimate is None else estimate.decision_state.value
        stable = bool(observed and decision != "unknown")
        target_index = 1 if condition.required_state == "linked" else 0
        target_probability = (
            math.nan if estimate is None else float(estimate.posterior[target_index])
        )
        compatibility = (
            math.nan
            if estimate is None
            else float(
                np.dot(
                    estimate.posterior,
                    np.asarray([condition.external, condition.linked]),
                )
            )
        )
        target_match = bool(stable and decision == condition.required_state)
        rows.append(
            {
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration,
                "boundary": boundary.boundary_id.token,
                "state": _state_text(state_id),
                "family": family,
                "condition": key,
                "target": condition.required_state,
                "observed": int(observed),
                "reliability": reliability,
                "decision_stable": int(stable),
                "decision": decision,
                "target_decision_match": int(target_match),
                "target_probability": target_probability,
                "compatibility": compatibility,
                "threshold": "",
                "threshold_pass": "",
            }
        )
    return rows


def _scene_rows(
    case: Any,
    boundary: Any,
    demonstration: int,
    state_id: StateId,
    belief: Any,
    minimum_tracking_reliability: float,
) -> list[dict[str, Any]]:
    rows = []
    for factor, distribution in sorted(boundary.scene_conditions.items()):
        value, reliability = _factor_observation(factor, belief.runtime_features)
        observed = bool(
            value is not None and reliability >= minimum_tracking_reliability
        )
        compatibility = math.nan if value is None else distribution.compatibility(value)
        threshold = float(boundary.scene_condition_thresholds[factor])
        rows.append(
            {
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration,
                "boundary": boundary.boundary_id.token,
                "state": _state_text(state_id),
                "family": "guard_scene",
                "condition": factor.token,
                "target": "boundary_support",
                "observed": int(observed),
                "reliability": reliability,
                "decision_stable": "",
                "target_decision_match": "",
                "compatibility": compatibility,
                "threshold": threshold,
                "threshold_pass": int(observed and float(compatibility) >= threshold),
            }
        )
    return rows


def _family_metrics(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    selected = [row for row in rows if row["family"] == family]
    if not selected:
        return {
            "count": 0,
            "observed_all": True,
            "stable_all": True,
            "match_all": True,
            "threshold_all": True,
            "minimum_compatibility": 1.0,
            "minimum_target_probability": 1.0,
            "minimum_log_margin": math.nan,
        }
    relation = family in {"own_relation", "guard_relation"}
    return {
        "count": len(selected),
        "observed_all": all(bool(row["observed"]) for row in selected),
        "stable_all": (
            all(bool(row["decision_stable"]) for row in selected) if relation else True
        ),
        "match_all": (
            all(bool(row["target_decision_match"]) for row in selected)
            if relation
            else True
        ),
        "threshold_all": (
            all(bool(row["threshold_pass"]) for row in selected)
            if not relation
            else True
        ),
        "minimum_compatibility": _minimum(
            float(row["compatibility"]) for row in selected
        ),
        "minimum_target_probability": _minimum(
            float(row.get("target_probability", math.nan)) for row in selected
        ),
        "minimum_log_margin": _minimum(
            float(row.get("log_likelihood_margin", math.nan)) for row in selected
        ),
    }


def _replay_case(
    case: Any,
    demonstration: int,
    minimum_tracking_reliability: float,
    belief_config: BeliefUpdaterConfig,
    execution_config: ClosedLoopExecutionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mode_by_skill = _mode_by_skill(case.policy, demonstration)
    samples = tuple(
        sample
        for skill_index in range(len(case.policy.skills))
        for sample in _samples_for_skill(case, demonstration, skill_index)
    )
    updater = BeliefUpdater(case.model, belief_config)
    initial = samples[0]
    case.policy.reset(
        DynaMACObservation(initial.ee_pose, initial.frames), mode_strategy="map"
    )
    updater.reset(
        initial_progress={initial.state_id: 1.0},
        initial_relations=_initial_relations(case, initial.state_id, mode_by_skill),
        previous_observation=_runtime_observation(0, initial, None),
    )
    controller = ClosedLoopExecutionController(case.model, execution_config)
    controller.reset(initial.state_id)
    boundary_by_source = {
        boundary.source_skill: boundary for boundary in case.model.boundaries.values()
    }
    previous = initial
    previous_belief: Any | None = None
    tick = 0
    trace_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    final_by_boundary: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for current in samples[1:]:
        tick += 1
        if current.state_id.skill_index != previous.state_id.skill_index:
            updater.reset(
                initial_progress={current.state_id: 1.0},
                initial_relations=(
                    _initial_relations(case, current.state_id, mode_by_skill)
                    if previous_belief is None
                    else previous_belief.relation_posteriors
                ),
                initial_relation_decisions=(
                    {}
                    if previous_belief is None
                    else {
                        frame: estimate.decision_state
                        for frame, estimate in previous_belief.relation_estimates.items()
                        if estimate.decision_state != RelationDecision.UNKNOWN
                    }
                ),
                previous_observation=_runtime_observation(tick, current, previous),
            )
            controller.reset(current.state_id)
            previous = current
            continue
        belief = updater.update(
            _runtime_observation(tick, current, previous),
            # This is a fixed demonstration replay: the observation was
            # generated by the recorded previous-state action, not by the
            # counterfactual action queried below for controller auditing.
            executed_reference_state=previous.state_id,
            mode_by_skill=mode_by_skill,
        )
        execution = controller.update(
            belief,
            DynaMACObservation(current.ee_pose, current.frames),
            mode_by_skill=mode_by_skill,
        )
        previous_belief = belief
        boundary = boundary_by_source.get(current.state_id.skill_index)
        if boundary is not None:
            mode = mode_by_skill[boundary.source_skill]
            state_conditions = _goal_rows(
                case,
                boundary,
                demonstration,
                mode,
                current.state_id,
                current,
            )
            state_conditions.extend(
                _relation_rows(
                    case,
                    boundary,
                    demonstration,
                    current.state_id,
                    belief,
                    "own_relation",
                    boundary.local_completion_model.own_relation_conditions,
                    minimum_tracking_reliability,
                )
            )
            state_conditions.extend(
                _relation_rows(
                    case,
                    boundary,
                    demonstration,
                    current.state_id,
                    belief,
                    "guard_relation",
                    boundary.relation_conditions,
                    minimum_tracking_reliability,
                )
            )
            state_conditions.extend(
                _scene_rows(
                    case,
                    boundary,
                    demonstration,
                    current.state_id,
                    belief,
                    minimum_tracking_reliability,
                )
            )
            condition_rows.extend(state_conditions)
            goal = _family_metrics(state_conditions, "local_goal")
            own = _family_metrics(state_conditions, "own_relation")
            guard_relation = _family_metrics(state_conditions, "guard_relation")
            guard_scene = _family_metrics(state_conditions, "guard_scene")
            terminal = current.state_id in boundary.terminal_window
            progress_supported = (
                belief.progress.status != ProgressStatus.NO_PLAUSIBLE_STATE
            )
            end_probability = float(
                sum(
                    belief.progress.posterior.get(state_id, 0.0)
                    for state_id in boundary.terminal_window
                )
            )
            local_components_ready = bool(
                terminal
                and progress_supported
                and goal["observed_all"]
                and goal["threshold_all"]
                and own["observed_all"]
                and own["stable_all"]
                and own["match_all"]
            )
            all_components_ready = bool(
                local_components_ready
                and guard_relation["observed_all"]
                and guard_relation["stable_all"]
                and guard_relation["match_all"]
                and guard_scene["observed_all"]
                and guard_scene["threshold_all"]
            )
            truth_score = belief.candidate_scores.get(current.state_id)
            best_state, best_score = max(
                belief.candidate_scores.items(),
                key=lambda item: item[1].normalized_explanation_score,
            )
            worst_robot_frame, worst_robot_log_support = _worst_robot_frame(truth_score)
            worst_relation = belief.relation_estimates.get(worst_robot_frame)
            trace = {
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration,
                "boundary": boundary.boundary_id.token,
                "tick": tick,
                "truth_state": _state_text(current.state_id),
                "terminal_truth": int(terminal),
                "estimated_state": _state_text(belief.progress.estimated_state),
                "nominal_state": _state_text(belief.progress.nominal_state),
                "reference_state_before": _state_text(
                    execution.cursor_before.reference_state
                ),
                "reference_state_after": _state_text(
                    execution.cursor_after.reference_state
                ),
                "execution_decision": execution.decision.value,
                "execution_reasons": "|".join(execution.reasons),
                "estimated_in_terminal_window": int(
                    belief.progress.estimated_state in boundary.terminal_window
                ),
                "progress_status": belief.progress.status.value,
                "progress_supported": int(progress_supported),
                "end_probability": end_probability,
                "progress_confidence": belief.progress.confidence,
                "best_explanation_score": belief.progress.best_explanation_score,
                "best_scored_state": _state_text(best_state),
                "best_score_robot_compatibility": best_score.robot_compatibility,
                "best_score_scene_compatibility": best_score.state_compatibility,
                "best_score_relation_compatibility": best_score.relation_compatibility,
                "best_score_relation_peak_normalized_compatibility": (
                    best_score.relation_peak_normalized_compatibility
                ),
                "truth_explanation_score": (
                    math.nan
                    if truth_score is None
                    else truth_score.normalized_explanation_score
                ),
                "truth_robot_compatibility": (
                    math.nan if truth_score is None else truth_score.robot_compatibility
                ),
                "truth_scene_compatibility": (
                    math.nan if truth_score is None else truth_score.state_compatibility
                ),
                "truth_relation_compatibility": (
                    math.nan
                    if truth_score is None
                    else truth_score.relation_compatibility
                ),
                "truth_relation_peak_normalized_compatibility": (
                    math.nan
                    if truth_score is None
                    else truth_score.relation_peak_normalized_compatibility
                ),
                "truth_worst_robot_frame": worst_robot_frame,
                "truth_worst_robot_log_support": worst_robot_log_support,
                "truth_worst_robot_relation_decision": (
                    ""
                    if worst_relation is None
                    else worst_relation.decision_state.value
                ),
                "truth_worst_robot_external_probability": (
                    math.nan if worst_relation is None else worst_relation.external
                ),
                "goal_count": goal["count"],
                "goal_observed_all": int(goal["observed_all"]),
                "goal_support_all": int(goal["threshold_all"]),
                "goal_minimum_compatibility": goal["minimum_compatibility"],
                "goal_geometric_mean_compatibility": _geometric_mean(
                    [
                        float(row["compatibility"])
                        for row in state_conditions
                        if row["family"] == "local_goal" and row["observed"]
                    ]
                ),
                "own_relation_count": own["count"],
                "own_relation_observed_all": int(own["observed_all"]),
                "own_relation_stable_all": int(own["stable_all"]),
                "own_relation_match_all": int(own["match_all"]),
                "own_relation_minimum_compatibility": own["minimum_compatibility"],
                "guard_relation_count": guard_relation["count"],
                "guard_relation_observed_all": int(guard_relation["observed_all"]),
                "guard_relation_stable_all": int(guard_relation["stable_all"]),
                "guard_relation_match_all": int(guard_relation["match_all"]),
                "guard_relation_minimum_compatibility": guard_relation[
                    "minimum_compatibility"
                ],
                "guard_scene_count": guard_scene["count"],
                "guard_scene_observed_all": int(guard_scene["observed_all"]),
                "guard_scene_support_all": int(guard_scene["threshold_all"]),
                "guard_scene_minimum_compatibility": guard_scene[
                    "minimum_compatibility"
                ],
                "local_components_ready": int(local_components_ready),
                "all_components_ready": int(all_components_ready),
            }
            trace_rows.append(trace)
            if current.state_id.local_index == (
                case.policy.skills[boundary.source_skill].duration - 1
            ):
                final_by_boundary[boundary.boundary_id.token] = (
                    trace,
                    state_conditions,
                )
        previous = current

    for boundary in sorted(
        case.model.boundaries.values(), key=lambda item: item.boundary_id
    ):
        token = boundary.boundary_id.token
        final_trace, final_conditions = final_by_boundary[token]
        relevant_trace = [row for row in trace_rows if row["boundary"] == token]
        streak = 0
        for row in reversed(relevant_trace):
            if not row["local_components_ready"]:
                break
            streak += 1
        all_streak = 0
        for row in reversed(relevant_trace):
            if not row["all_components_ready"]:
                break
            all_streak += 1
        terminal_trace = [row for row in relevant_trace if row["terminal_truth"]]
        nonterminal_trace = [row for row in relevant_trace if not row["terminal_truth"]]
        goal = _family_metrics(final_conditions, "local_goal")
        own = _family_metrics(final_conditions, "own_relation")
        guard_relation = _family_metrics(final_conditions, "guard_relation")
        guard_scene = _family_metrics(final_conditions, "guard_scene")
        boundary_rows.append(
            {
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration,
                "boundary": token,
                "source_skill": boundary.source_skill,
                "target_skill": boundary.target_skill,
                "source_duration": case.policy.skills[boundary.source_skill].duration,
                "terminal_window_size": len(boundary.terminal_window),
                "final_progress_status": final_trace["progress_status"],
                "final_progress_supported": final_trace["progress_supported"],
                "final_estimated_in_terminal_window": final_trace[
                    "estimated_in_terminal_window"
                ],
                "final_end_probability": final_trace["end_probability"],
                "final_best_explanation_score": final_trace["best_explanation_score"],
                "terminal_supported_samples": sum(
                    int(row["progress_supported"]) for row in terminal_trace
                ),
                "terminal_sample_count": len(terminal_trace),
                "terminal_minimum_end_probability": _minimum(
                    row["end_probability"] for row in terminal_trace
                ),
                "nonterminal_maximum_end_probability": max(
                    (float(row["end_probability"]) for row in nonterminal_trace),
                    default=0.0,
                ),
                "goal_count": goal["count"],
                "goal_observed_all": int(goal["observed_all"]),
                "goal_support_all": int(goal["threshold_all"]),
                "goal_minimum_compatibility": goal["minimum_compatibility"],
                "goal_minimum_log_margin": goal["minimum_log_margin"],
                "own_relation_count": own["count"],
                "own_relation_observed_all": int(own["observed_all"]),
                "own_relation_stable_all": int(own["stable_all"]),
                "own_relation_match_all": int(own["match_all"]),
                "own_relation_minimum_target_probability": own[
                    "minimum_target_probability"
                ],
                "own_relation_minimum_compatibility": own["minimum_compatibility"],
                "guard_relation_count": guard_relation["count"],
                "guard_relation_observed_all": int(guard_relation["observed_all"]),
                "guard_relation_stable_all": int(guard_relation["stable_all"]),
                "guard_relation_match_all": int(guard_relation["match_all"]),
                "guard_relation_minimum_target_probability": guard_relation[
                    "minimum_target_probability"
                ],
                "guard_relation_minimum_compatibility": guard_relation[
                    "minimum_compatibility"
                ],
                "guard_scene_count": guard_scene["count"],
                "guard_scene_observed_all": int(guard_scene["observed_all"]),
                "guard_scene_support_all": int(guard_scene["threshold_all"]),
                "guard_scene_minimum_compatibility": guard_scene[
                    "minimum_compatibility"
                ],
                "local_ready_streak": streak,
                "all_conditions_ready_streak": all_streak,
            }
        )
    return boundary_rows, condition_rows, trace_rows


def _summarize(
    boundary_rows: Sequence[Mapping[str, Any]],
    minimum_h_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in boundary_rows:
        grouped[(str(row["task"]), str(row["arm"]), str(row["boundary"]))].append(row)
    result = []
    for (task, arm, boundary), rows in sorted(grouped.items()):
        final_progress_ready = all(
            bool(row["final_progress_supported"])
            and bool(row["final_estimated_in_terminal_window"])
            for row in rows
        )
        terminal_min = min(
            float(row["terminal_minimum_end_probability"]) for row in rows
        )
        nonterminal_max = max(
            float(row["nonterminal_maximum_end_probability"]) for row in rows
        )
        progress_separable = terminal_min > nonterminal_max
        goal_ready = all(
            bool(row["goal_observed_all"]) and bool(row["goal_support_all"])
            for row in rows
        )
        own_ready = all(
            bool(row["own_relation_observed_all"])
            and bool(row["own_relation_stable_all"])
            and bool(row["own_relation_match_all"])
            for row in rows
        )
        guard_relation_ready = all(
            bool(row["guard_relation_observed_all"])
            and bool(row["guard_relation_stable_all"])
            and bool(row["guard_relation_match_all"])
            for row in rows
        )
        guard_scene_ready = all(
            bool(row["guard_scene_observed_all"])
            and bool(row["guard_scene_support_all"])
            for row in rows
        )
        minimum_local_streak = min(int(row["local_ready_streak"]) for row in rows)
        theta_inputs_ready = bool(final_progress_ready and goal_ready and own_ready)
        h_inputs_ready = bool(
            theta_inputs_ready and minimum_local_streak >= minimum_h_samples
        )
        guard_inputs_ready = bool(guard_relation_ready and guard_scene_ready)
        if not final_progress_ready:
            readiness = "blocked_progress_endpoint"
        elif not goal_ready or not own_ready:
            readiness = "blocked_local_completion_input"
        elif not guard_inputs_ready:
            readiness = "blocked_guard_input"
        elif not h_inputs_ready:
            readiness = "ready_theta_limited_h"
        else:
            readiness = "ready_for_calibration"
        result.append(
            {
                "task": task,
                "arm": arm,
                "boundary": boundary,
                "demonstrations": len(rows),
                "source_duration": rows[0]["source_duration"],
                "terminal_window_size": rows[0]["terminal_window_size"],
                "terminal_no_plausible_rate": _mean(
                    1.0
                    - float(row["terminal_supported_samples"])
                    / max(1, int(row["terminal_sample_count"]))
                    for row in rows
                ),
                "final_no_plausible_count": sum(
                    not bool(row["final_progress_supported"]) for row in rows
                ),
                "final_estimated_in_terminal_count": sum(
                    bool(row["final_estimated_in_terminal_window"]) for row in rows
                ),
                "final_valid_endpoint_count": sum(
                    bool(row["final_progress_supported"])
                    and bool(row["final_estimated_in_terminal_window"])
                    for row in rows
                ),
                "final_end_probability_minimum": min(
                    float(row["final_end_probability"]) for row in rows
                ),
                "final_end_probability_median": median(
                    float(row["final_end_probability"]) for row in rows
                ),
                "terminal_end_probability_minimum": terminal_min,
                "nonterminal_end_probability_maximum": nonterminal_max,
                "progress_separation_margin": terminal_min - nonterminal_max,
                "progress_endpoint_ready": int(final_progress_ready),
                "progress_separable": int(progress_separable),
                "goal_count": rows[0]["goal_count"],
                "goal_observed_rate": _mean(row["goal_observed_all"] for row in rows),
                "goal_support_rate": _mean(row["goal_support_all"] for row in rows),
                "goal_minimum_compatibility": _minimum(
                    float(row["goal_minimum_compatibility"]) for row in rows
                ),
                "goal_minimum_log_margin": _minimum(
                    float(row["goal_minimum_log_margin"]) for row in rows
                ),
                "own_relation_count": rows[0]["own_relation_count"],
                "own_relation_observed_rate": _mean(
                    row["own_relation_observed_all"] for row in rows
                ),
                "own_relation_stable_rate": _mean(
                    row["own_relation_stable_all"] for row in rows
                ),
                "own_relation_match_rate": _mean(
                    row["own_relation_match_all"] for row in rows
                ),
                "own_relation_minimum_target_probability": _minimum(
                    float(row["own_relation_minimum_target_probability"])
                    for row in rows
                ),
                "guard_relation_count": rows[0]["guard_relation_count"],
                "guard_relation_observed_rate": _mean(
                    row["guard_relation_observed_all"] for row in rows
                ),
                "guard_relation_stable_rate": _mean(
                    row["guard_relation_stable_all"] for row in rows
                ),
                "guard_relation_match_rate": _mean(
                    row["guard_relation_match_all"] for row in rows
                ),
                "guard_relation_minimum_target_probability": _minimum(
                    float(row["guard_relation_minimum_target_probability"])
                    for row in rows
                ),
                "guard_scene_count": rows[0]["guard_scene_count"],
                "guard_scene_observed_rate": _mean(
                    row["guard_scene_observed_all"] for row in rows
                ),
                "guard_scene_support_rate": _mean(
                    row["guard_scene_support_all"] for row in rows
                ),
                "minimum_local_ready_streak": minimum_local_streak,
                "theta_inputs_ready": int(theta_inputs_ready),
                "h_inputs_ready": int(h_inputs_ready),
                "guard_inputs_ready": int(guard_inputs_ready),
                "readiness": readiness,
            }
        )
    return result


def _report(
    config: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    belief_config: BeliefUpdaterConfig,
) -> str:
    priority = set(config["priority_cases"])
    ready = sum(row["readiness"] == "ready_for_calibration" for row in summaries)
    lines = [
        "# 阶段四正常边界就绪性检查结果",
        "",
        "## 1. 结论边界",
        "",
        config["claim_boundary"],
        "",
        "本次回放使用的阶段二最低解释度为 "
        f"`{belief_config.progress_filter.minimum_explanation_score:.10g}`；"
        "完整运行配置保存在同目录 `belief_config.json`。",
        "",
        f"共检查 {len(summaries)} 个技能边界；其中 {ready} 个满足当前严格的标定输入检查。`theta_inputs_ready` 只说明正常边界末端的进度、目标和本臂关系输入均有效；`P_end` 的正负间隔另行保留，供联合完成度标定时判断。`h_inputs_ready` 只说明至少存在配置要求数量的连续正常样本；本检查不自行选择最终阈值。",
        "",
        "## 2. 优先任务",
        "",
        "| 任务/机械臂 | 边界 | 末端有效 | 末端概率最小值 | 最终目标 | 本臂关系 | 边界关系 | 场景 | 连续样本 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(
        summaries,
        key=lambda row: (
            f"{row['task']}/{row['arm']}" not in priority
            and str(row["task"]) not in priority,
            str(row["task"]),
            str(row["arm"]),
            str(row["boundary"]),
        ),
    )
    for row in ordered:
        case = (
            str(row["task"])
            if row["arm"] == "single"
            else f"{row['task']}/{row['arm']}"
        )
        if case not in priority:
            continue
        lines.append(
            "| {case} | {boundary} | {progress}/5 | {end:.4f} | {goal:.0%} | {own:.0%} | {guard:.0%} | {scene:.0%} | {streak} | `{readiness}` |".format(
                case=case,
                boundary=row["boundary"],
                progress=row["final_valid_endpoint_count"],
                end=float(row["final_end_probability_minimum"]),
                goal=float(row["goal_support_rate"]),
                own=float(row["own_relation_match_rate"]),
                guard=float(row["guard_relation_match_rate"]),
                scene=float(row["guard_scene_support_rate"]),
                streak=row["minimum_local_ready_streak"],
                readiness=row["readiness"],
            )
        )
    lines.extend(
        [
            "",
            "## 3. 全部边界",
            "",
            "| 任务/机械臂 | 边界 | 终止窗 NPS | P_end 正负间隔 | theta 输入 | H 输入 | 守卫输入 | 结论 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ordered:
        case = (
            str(row["task"])
            if row["arm"] == "single"
            else f"{row['task']}/{row['arm']}"
        )
        lines.append(
            "| {case} | {boundary} | {nps:.1%} | {margin:.4f} | {theta} | {h} | {guard} | `{readiness}` |".format(
                case=case,
                boundary=row["boundary"],
                nps=float(row["terminal_no_plausible_rate"]),
                margin=float(row["progress_separation_margin"]),
                theta="是" if row["theta_inputs_ready"] else "否",
                h="是" if row["h_inputs_ready"] else "否",
                guard="是" if row["guard_inputs_ready"] else "否",
                readiness=row["readiness"],
            )
        )
    lines.extend(
        [
            "",
            "## 4. 字段解释",
            "",
            "- `NO_PLAUSIBLE_STATE` 时的 `P_end` 只来自传播后的名义先验，本检查不把它计作有效末端证据。",
            "- `P_end 正负间隔` 是全部正常终止窗口样本的最小末端概率减去全部更早样本的最大末端概率；正值表示仅按该分量存在正常完成/未完成分界。",
            "- 目标支持使用阶段一保存的最终状态分布与最低正常训练对数似然，只用于检查输入是否正常，不等同于已经标定 `theta_local`。",
            "- 关系“可观测”与“稳定判定”分开统计：有真值位姿不代表当前运动激励足以把在线关系从 `Unknown` 判为 linked/external。",
            "- `H 输入` 要求每条示范至少有配置数量的连续本地就绪样本；它不直接给出控制周期数 `H`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _environment(
    config_path: Path,
    config: Mapping[str, Any],
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    inputs = [
        config_path,
        BELIEF_CONFIG_PATH,
        EXECUTION_CONFIG_PATH,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "source/policy/dynamac.py",
    ]
    inputs.extend(sorted((REPOSITORY_ROOT / "source/policy/closed_loop").glob("*.py")))
    inputs.extend(
        path
        for task in config["tasks"]
        for path in [MODEL_PATHS[task]]
        if path.is_file()
    )
    inputs.extend(source_paths)
    hashes = {}
    for path in sorted(set(inputs)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            label = str(resolved.relative_to(REPOSITORY_ROOT))
        except ValueError:
            label = str(resolved)
        hashes[label] = _sha256(path)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "input_sha256": hashes,
    }


def _checksums(output: Path) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    content = "".join(
        f"{_sha256(path)}  {path.relative_to(output)}\n" for path in paths
    )
    (output / "SHA256SUMS").write_text(content, encoding="utf-8")


def run(
    config_path: Path,
    output: Path,
    *,
    minimum_explanation_score: float | None = None,
) -> None:
    config = _read_config(config_path)
    belief_config = BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH)
    execution_config = ClosedLoopExecutionConfig.from_json(EXECUTION_CONFIG_PATH)
    if minimum_explanation_score is not None:
        belief_config = replace(
            belief_config,
            progress_filter=replace(
                belief_config.progress_filter,
                minimum_explanation_score=float(minimum_explanation_score),
            ),
        )
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    (output / "belief_config.json").write_text(
        json.dumps(belief_config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "execution_config.json").write_text(
        json.dumps(execution_config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    boundary_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    demonstrations = tuple(int(value) for value in config["demonstration_indices"])
    required_count = max(demonstrations) + 1
    for task in config["tasks"]:
        print(f"[boundary-readiness] loading {task}", flush=True)
        for case in _load_cases(str(task), required_count):
            source_paths.extend(case.demonstration_paths)
            if case.model_path.is_file():
                source_paths.append(case.model_path)
            for demonstration in demonstrations:
                current_boundary, current_conditions, current_trace = _replay_case(
                    case,
                    demonstration,
                    float(config["minimum_tracking_reliability"]),
                    belief_config,
                    execution_config,
                )
                boundary_rows.extend(current_boundary)
                condition_rows.extend(current_conditions)
                trace_rows.extend(current_trace)

    summaries = _summarize(
        boundary_rows,
        int(config["minimum_consecutive_samples_for_h_audit"]),
    )
    _write_csv(output / "boundary_trials.csv", boundary_rows)
    _write_csv(output / "condition_trials.csv", condition_rows)
    _write_csv(output / "boundary_summary.csv", summaries)
    with gzip.open(
        output / "boundary_trace.csv.gz", "wt", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(trace_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(trace_rows)
    (output / "report.md").write_text(
        _report(config, summaries, belief_config), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(
            _environment(config_path, config, source_paths),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _checksums(output)
    print(f"[boundary-readiness] wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--minimum-explanation-score", type=float)
    args = parser.parse_args()
    if args.tasks:
        config = _read_config(args.config)
        unknown = set(args.tasks).difference(config["tasks"])
        if unknown:
            raise ValueError(f"调试任务不在正式配置中：{sorted(unknown)}")
        temporary = args.output.parent / f".{args.output.name}.config.json"
        reduced = dict(config)
        reduced["tasks"] = list(args.tasks)
        temporary.write_text(
            json.dumps(reduced, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            run(
                temporary,
                args.output,
                minimum_explanation_score=args.minimum_explanation_score,
            )
        finally:
            temporary.unlink(missing_ok=True)
    else:
        run(
            args.config,
            args.output,
            minimum_explanation_score=args.minimum_explanation_score,
        )


if __name__ == "__main__":
    main()
