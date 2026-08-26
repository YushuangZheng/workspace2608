"""Calibrate phase-four boundary thresholds from normal control-tick replay."""

from __future__ import annotations

import argparse
import copy
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    BoundaryCalibration,
    BoundaryRuntimeConfig,
    ConditionKind,
    EntryGuard,
    RelationDecision,
    StateId,
)
from evaluations.phase23_component_ab.run import (
    BELIEF_CONFIG_PATH,
    EXECUTION_CONFIG_PATH,
    REPOSITORY_ROOT,
    Sample,
    _initial_relations,
    _load_cases,
    _mode_by_skill,
    _runtime_observation,
)

SCHEMA = "essay2608-phase4-boundary-calibration-config-v1"


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "claim_boundary",
        "tasks",
        "demonstration_indices",
        "control_period_seconds",
        "control_period_source",
        "minimum_confirmation_seconds",
        "terminal_hold_seconds",
        "terminal_settling_seconds",
        "positive_floor_fraction",
        "default_relation_probability",
        "minimum_tracking_reliability",
        "minimum_scene_reliability",
        "minimum_information_weight",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("阶段四标定配置字段不完整或包含未知字段")
    if value["schema"] != SCHEMA:
        raise ValueError("阶段四标定配置 schema 不匹配")
    if not value["tasks"] or not value["demonstration_indices"]:
        raise ValueError("阶段四标定任务和示范索引不能为空")
    period = float(value["control_period_seconds"])
    confirmation = float(value["minimum_confirmation_seconds"])
    hold = float(value["terminal_hold_seconds"])
    settling = float(value["terminal_settling_seconds"])
    if period <= 0.0 or confirmation < period or hold < confirmation:
        raise ValueError("控制周期、最短确认时间或末端保持时间无效")
    if settling < 0.0 or settling >= hold:
        raise ValueError("末端稳定等待时间必须位于 [0, terminal_hold_seconds)")
    if not 0.0 < float(value["positive_floor_fraction"]) < 1.0:
        raise ValueError("正常末端支持保守比例必须位于 (0,1)")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"不能写入空结果 {path.name}")
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("逐控制周期标定轨迹不能为空")
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _state_text(state_id: Any) -> str:
    return f"k{state_id.skill_index}:t{state_id.local_index}"


def _raw_samples(case: Any, demonstration_index: int) -> tuple[Sample, ...]:
    demonstration = case.demonstrations[demonstration_index]
    result: list[Sample | None] = [None] * len(demonstration.skill)
    virtual_starts = {
        skill_index: demonstration.ee_pose[
            np.flatnonzero(demonstration.skill == skill_index)[0]
        ].copy()
        for skill_index in range(len(case.policy.skills))
    }
    for skill_index, skill in enumerate(case.policy.skills):
        indices = np.flatnonzero(demonstration.skill == skill_index)
        if not len(indices):
            raise RuntimeError(f"{case.key} 示范缺少技能 {skill_index}")
        if len(indices) == 1:
            local_indices = np.zeros(1, dtype=np.int64)
        else:
            local_indices = np.rint(
                np.linspace(0, skill.duration - 1, len(indices))
            ).astype(np.int64)
        for source_index, local_index in zip(indices, local_indices, strict=True):
            frames = {
                name: values[source_index].copy()
                for name, values in demonstration.frames.items()
            }
            frames.update(
                {
                    name: values[source_index].copy()
                    for name, values in demonstration.scene_entity_poses.items()
                }
            )
            # DynaMAC captures one fixed virtual execution frame at each skill
            # entry and keeps the previously captured frames.  Raw RLBench
            # observations do not serialize these derived frames, so rebuild
            # the same deterministic runtime snapshot here.
            frames.update(
                {
                    f"virtual_skill_{case.policy.skills[owner].label}": virtual_starts[
                        owner
                    ].copy()
                    for owner in range(skill_index + 1)
                }
            )
            configurations = {
                entity: {
                    field: values[source_index].copy()
                    for field, values in fields.items()
                }
                for entity, fields in demonstration.entity_configurations.items()
            }
            result[source_index] = Sample(
                state_id=StateId(skill_index, int(local_index)),
                ee_pose=demonstration.ee_pose[source_index].copy(),
                action_pose=demonstration.action_pose[source_index].copy(),
                frames=frames,
                gripper=demonstration.gripper[source_index].copy(),
                entity_configurations=configurations,
            )
    if any(sample is None for sample in result):
        raise RuntimeError(f"{case.key} 原始示范包含无法映射的控制周期")
    return tuple(sample for sample in result if sample is not None)


def _base_runtime_config(
    cases: Sequence[Any], config: Mapping[str, Any]
) -> BoundaryRuntimeConfig:
    return BoundaryRuntimeConfig(
        calibrations={
            boundary_id.token: BoundaryCalibration(0.0, 1)
            for case in cases
            for boundary_id in case.model.boundaries
        },
        default_relation_probability=float(config["default_relation_probability"]),
        minimum_tracking_reliability=float(config["minimum_tracking_reliability"]),
        minimum_scene_reliability=float(config["minimum_scene_reliability"]),
        minimum_information_weight=float(config["minimum_information_weight"]),
    )


def _raw_guard_satisfied(request: Any) -> bool:
    return all(
        result.raw_satisfied
        for condition_id, result in request.condition_results.items()
        if condition_id.kind
        in {ConditionKind.GUARD_RELATION, ConditionKind.GUARD_SCENE}
    )


def _trace_row(
    *,
    task: str,
    arm: str,
    demonstration: int,
    boundary: Any,
    tick: int,
    phase: str,
    hold_cycle: int,
    truth_state: Any,
    belief: Any,
    request: Any,
    local: Any,
) -> dict[str, Any]:
    return {
        "task": task,
        "arm": arm,
        "demonstration": demonstration,
        "boundary": boundary.boundary_id.token,
        "transaction_group": boundary.transaction_group or "",
        "tick": tick,
        "phase": phase,
        "hold_cycle": hold_cycle,
        "truth_state": _state_text(truth_state),
        "truth_in_terminal_window": int(truth_state in boundary.terminal_window),
        "estimated_state": _state_text(belief.progress.estimated_state),
        "progress_status": belief.progress.status.value,
        "end_probability": local.end_probability,
        "goal_compatibility": local.goal_compatibility,
        "own_relation_compatibility": local.own_relation_compatibility,
        "local_score": local.score,
        "local_evidence_available": int(local.evidence_available),
        "local_raw_satisfied": int(local.raw_satisfied),
        "local_consecutive_cycles": local.consecutive_cycles,
        "local_done": int(local.done),
        "guard_raw_satisfied": int(_raw_guard_satisfied(request)),
        "transition_permitted": int(request.permitted),
        "verification_request_count": len(request.verification_requests),
    }


def _reset_updater(
    case: Any,
    updater: BeliefUpdater,
    sample: Sample,
    tick: int,
    mode_by_skill: Mapping[int, int],
    previous_belief: Any | None,
) -> None:
    updater.reset(
        initial_progress={sample.state_id: 1.0},
        initial_relations=(
            _initial_relations(case, sample.state_id, mode_by_skill)
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
        previous_observation=_runtime_observation(tick, sample, None),
    )


def _hold_rows(
    *,
    task: str,
    target_case: Any,
    target_boundary: Any,
    demonstration: int,
    tick: int,
    cases_by_arm: Mapping[str, Any],
    samples_by_arm: Mapping[str, tuple[Sample, ...]],
    updaters: Mapping[str, BeliefUpdater],
    beliefs: Mapping[str, Any],
    mode_by_arm_skill: Mapping[str, Mapping[int, int]],
    runtime_config: BoundaryRuntimeConfig,
    hold_cycles: int,
) -> list[dict[str, Any]]:
    cloned = {arm: copy.deepcopy(updater) for arm, updater in updaters.items()}
    current_beliefs = dict(beliefs)
    frozen = {arm: samples[tick] for arm, samples in samples_by_arm.items()}
    guard = EntryGuard(
        cases_by_arm_to_models(cases_by_arm), target_case.arm, runtime_config
    )
    rows = []
    for hold_cycle in range(hold_cycles + 1):
        current_tick = tick + hold_cycle
        if hold_cycle:
            current_beliefs = {
                arm: cloned[arm].update(
                    _runtime_observation(current_tick, sample, sample),
                    executed_reference_state=sample.state_id,
                    mode_by_skill=mode_by_arm_skill[arm],
                )
                for arm, sample in frozen.items()
            }
        request, local = guard.evaluate(
            target_boundary.boundary_id,
            current_beliefs,
            frozen[target_case.arm].state_id,
            mode_by_arm_skill=mode_by_arm_skill,
        )
        rows.append(
            _trace_row(
                task=task,
                arm=target_case.arm,
                demonstration=demonstration,
                boundary=target_boundary,
                tick=current_tick,
                phase="terminal_hold",
                hold_cycle=hold_cycle,
                truth_state=frozen[target_case.arm].state_id,
                belief=current_beliefs[target_case.arm],
                request=request,
                local=local,
            )
        )
    return rows


def cases_by_arm_to_models(cases_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    return {arm: case.model for arm, case in cases_by_arm.items()}


def _replay_task_demo(
    task: str,
    cases: Sequence[Any],
    demonstration: int,
    belief_config: BeliefUpdaterConfig,
    runtime_config: BoundaryRuntimeConfig,
    hold_cycles: int,
) -> list[dict[str, Any]]:
    cases_by_arm = {case.arm: case for case in cases}
    models = cases_by_arm_to_models(cases_by_arm)
    samples_by_arm = {
        arm: _raw_samples(case, demonstration) for arm, case in cases_by_arm.items()
    }
    lengths = {len(samples) for samples in samples_by_arm.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"{task} 双臂原始控制周期没有同步")
    sample_count = lengths.pop()
    mode_by_arm_skill = {
        arm: _mode_by_skill(case.policy, demonstration)
        for arm, case in cases_by_arm.items()
    }
    updaters = {
        arm: BeliefUpdater(case.model, belief_config)
        for arm, case in cases_by_arm.items()
    }
    previous_beliefs: dict[str, Any | None] = {arm: None for arm in cases_by_arm}
    for arm, updater in updaters.items():
        _reset_updater(
            cases_by_arm[arm],
            updater,
            samples_by_arm[arm][0],
            0,
            mode_by_arm_skill[arm],
            None,
        )
    guards = {arm: EntryGuard(models, arm, runtime_config) for arm in cases_by_arm}
    rows: list[dict[str, Any]] = []

    for tick in range(1, sample_count):
        beliefs: dict[str, Any] = {}
        reset_arm = False
        for arm, case in cases_by_arm.items():
            previous = samples_by_arm[arm][tick - 1]
            current = samples_by_arm[arm][tick]
            if current.state_id.skill_index != previous.state_id.skill_index:
                _reset_updater(
                    case,
                    updaters[arm],
                    current,
                    tick,
                    mode_by_arm_skill[arm],
                    previous_beliefs[arm],
                )
                reset_arm = True
                continue
            belief = updaters[arm].update(
                _runtime_observation(tick, current, previous),
                executed_reference_state=previous.state_id,
                mode_by_skill=mode_by_arm_skill[arm],
            )
            beliefs[arm] = belief
            previous_beliefs[arm] = belief
        if reset_arm or set(beliefs) != set(cases_by_arm):
            continue

        for arm, case in cases_by_arm.items():
            current = samples_by_arm[arm][tick]
            boundary = next(
                (
                    candidate
                    for candidate in case.model.boundaries.values()
                    if candidate.source_skill == current.state_id.skill_index
                ),
                None,
            )
            if boundary is None:
                continue
            request, local = guards[arm].evaluate(
                boundary.boundary_id,
                beliefs,
                current.state_id,
                mode_by_arm_skill=mode_by_arm_skill,
            )
            rows.append(
                _trace_row(
                    task=task,
                    arm=arm,
                    demonstration=demonstration,
                    boundary=boundary,
                    tick=tick,
                    phase="recorded",
                    hold_cycle=-1,
                    truth_state=current.state_id,
                    belief=beliefs[arm],
                    request=request,
                    local=local,
                )
            )
            next_skill = (
                None
                if tick + 1 == sample_count
                else samples_by_arm[arm][tick + 1].state_id.skill_index
            )
            if next_skill == current.state_id.skill_index:
                continue
            rows.extend(
                _hold_rows(
                    task=task,
                    target_case=case,
                    target_boundary=boundary,
                    demonstration=demonstration,
                    tick=tick,
                    cases_by_arm=cases_by_arm,
                    samples_by_arm=samples_by_arm,
                    updaters=updaters,
                    beliefs=beliefs,
                    mode_by_arm_skill=mode_by_arm_skill,
                    runtime_config=runtime_config,
                    hold_cycles=hold_cycles,
                )
            )
    return rows


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _calibrate(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["arm"]), str(row["boundary"]))].append(row)
    period = float(config["control_period_seconds"])
    minimum_cycles = math.ceil(float(config["minimum_confirmation_seconds"]) / period)
    settling_cycles = math.ceil(float(config["terminal_settling_seconds"]) / period)
    fraction = float(config["positive_floor_fraction"])
    demonstration_indices = tuple(int(v) for v in config["demonstration_indices"])
    summaries = []
    trial_rows = []
    runtime_by_task: dict[str, dict[str, Any]] = defaultdict(dict)

    for (task, arm, boundary), members in sorted(grouped.items()):
        stable_hold: list[Mapping[str, Any]] = []
        preterminal: list[Mapping[str, Any]] = []
        for row in members:
            if (
                row["phase"] == "terminal_hold"
                and int(row["hold_cycle"]) >= settling_cycles
            ):
                stable_hold.append(row)
            if row["phase"] == "recorded" and not int(row["truth_in_terminal_window"]):
                preterminal.append(row)
        positive_ready = [
            row
            for row in stable_hold
            if int(row["local_evidence_available"]) and int(row["guard_raw_satisfied"])
        ]
        positive_floor = min(
            (float(row["local_score"]) for row in positive_ready), default=0.0
        )
        preterminal_ceiling = max(
            (
                float(row["local_score"])
                for row in preterminal
                if int(row["local_evidence_available"])
                and int(row["guard_raw_satisfied"])
            ),
            default=0.0,
        )
        positive_trials = {int(row["demonstration"]) for row in positive_ready}
        if positive_floor <= 0.0 or positive_trials != set(demonstration_indices):
            threshold = math.nan
            confirmation_cycles = 0
            status = "blocked_normal_terminal_support"
            maximum_false_run = 0
            minimum_hold_run = 0
            separated = False
        else:
            separated = preterminal_ceiling < positive_floor
            threshold = (
                0.5 * (preterminal_ceiling + positive_floor)
                if separated
                else fraction * positive_floor
            )
            false_runs = []
            hold_runs = []
            for demonstration in demonstration_indices:
                recorded = sorted(
                    (
                        row
                        for row in members
                        if row["phase"] == "recorded"
                        and int(row["demonstration"]) == demonstration
                        and not int(row["truth_in_terminal_window"])
                    ),
                    key=lambda row: int(row["tick"]),
                )
                false_runs.append(
                    _longest_true_run(
                        bool(int(row["local_evidence_available"]))
                        and bool(int(row["guard_raw_satisfied"]))
                        and float(row["local_score"]) > threshold
                        for row in recorded
                    )
                )
                held = sorted(
                    (
                        row
                        for row in members
                        if row["phase"] == "terminal_hold"
                        and int(row["demonstration"]) == demonstration
                    ),
                    key=lambda row: int(row["hold_cycle"]),
                )
                hold_run = _longest_true_run(
                    bool(int(row["local_evidence_available"]))
                    and bool(int(row["guard_raw_satisfied"]))
                    and float(row["local_score"]) > threshold
                    for row in held
                )
                hold_runs.append(hold_run)
                trial_rows.append(
                    {
                        "task": task,
                        "arm": arm,
                        "boundary": boundary,
                        "demonstration": demonstration,
                        "maximum_preterminal_ready_run": false_runs[-1],
                        "terminal_hold_ready_run": hold_run,
                    }
                )
            maximum_false_run = max(false_runs)
            confirmation_cycles = max(minimum_cycles, maximum_false_run + 1)
            minimum_hold_run = min(hold_runs)
            status = (
                "calibrated"
                if minimum_hold_run >= confirmation_cycles
                else "blocked_hold_too_short"
            )
        summaries.append(
            {
                "task": task,
                "arm": arm,
                "boundary": boundary,
                "transaction_group": str(members[0]["transaction_group"]),
                "normal_demonstrations": len(demonstration_indices),
                "positive_support_separated": int(separated),
                "normal_terminal_score_floor": positive_floor,
                "normal_preterminal_score_ceiling": preterminal_ceiling,
                "local_score_threshold": threshold,
                "maximum_preterminal_ready_run": maximum_false_run,
                "confirmation_cycles": confirmation_cycles,
                "confirmation_seconds": confirmation_cycles * period,
                "minimum_terminal_hold_ready_run": minimum_hold_run,
                "status": status,
            }
        )
        if status == "calibrated":
            runtime_by_task[task][boundary] = BoundaryCalibration(
                local_score_threshold=float(threshold),
                confirmation_cycles=int(confirmation_cycles),
            )

    serialized = {
        task: BoundaryRuntimeConfig(
            calibrations=calibrations,
            default_relation_probability=float(config["default_relation_probability"]),
            minimum_tracking_reliability=float(config["minimum_tracking_reliability"]),
            minimum_scene_reliability=float(config["minimum_scene_reliability"]),
            minimum_information_weight=float(config["minimum_information_weight"]),
        ).to_dict()
        for task, calibrations in runtime_by_task.items()
    }
    return summaries, trial_rows, serialized


def _report(
    config: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    acceptance: Sequence[Mapping[str, Any]],
    joint_acceptance: Sequence[Mapping[str, Any]],
) -> str:
    calibrated = sum(row["status"] == "calibrated" for row in summaries)
    separated = sum(bool(row["positive_support_separated"]) for row in summaries)
    transaction_groups = {
        (str(row["task"]), str(row["transaction_group"]))
        for row in summaries
        if row["transaction_group"]
    }
    lines = [
        "# 阶段四边界参数标定结果",
        "",
        f"- 控制周期：{float(config['control_period_seconds']):.3f} 秒（`{config['control_period_source']}`）",
        f"- 正常示范：每任务 {len(config['demonstration_indices'])} 条",
        f"- 已标定边界：{calibrated}/{len(summaries)}",
        f"- 无需连续确认消歧即可分离的边界：{separated}/{len(summaries)}",
        f"- 正常运行配置放行复核：{sum(row['accepted'] for row in acceptance)}/{len(acceptance)} 条边界×示范",
        f"- 真实模型联合事务：{len(transaction_groups)} 组，正常联合就绪复核 {sum(row['accepted'] for row in joint_acceptance)}/{len(joint_acceptance)} 组×示范",
        "- 故障数据使用：0",
        "",
        "| 任务/机械臂 | 边界 | 正常末端下界 | 边界前上界 | theta_local | H | 秒 | 结果 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {task}/{arm} | `{boundary}` | {positive:.6g} | {negative:.6g} | {theta:.6g} | {h} | {seconds:.2f} | `{status}` |".format(
                task=row["task"],
                arm=row["arm"],
                boundary=row["boundary"],
                positive=float(row["normal_terminal_score_floor"]),
                negative=float(row["normal_preterminal_score_ceiling"]),
                theta=float(row["local_score_threshold"]),
                h=row["confirmation_cycles"],
                seconds=float(row["confirmation_seconds"]),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "## 联合事务元数据",
            "",
        ]
    )
    if transaction_groups:
        lines.extend(
            [
                "| 任务 | 事务组 | 成员边界 |",
                "|---|---|---|",
            ]
        )
        for task, group in sorted(transaction_groups):
            members = ", ".join(
                f"`{row['boundary']}`"
                for row in summaries
                if row["task"] == task and row["transaction_group"] == group
            )
            lines.append(f"| {task} | `{group}` | {members} |")
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "`theta_local` 与 `H` 只由正常回放确定。若边界前与末端分数重叠，不增加新的特殊门控，而是由既有连续确认机制抑制短暂提前脉冲。末端保持分支使用相同的在线关系与进度更新链，并以真实 0.05 秒控制周期重复当前正常末端观测。",
            "",
        ]
    )
    return "\n".join(lines)


def _acceptance_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["task"]),
                str(row["arm"]),
                str(row["boundary"]),
                int(row["demonstration"]),
            )
        ].append(row)
    result = []
    for (task, arm, boundary, demonstration), members in sorted(grouped.items()):
        premature = sum(
            int(row["transition_permitted"])
            for row in members
            if row["phase"] == "recorded" and not int(row["truth_in_terminal_window"])
        )
        held = sorted(
            (row for row in members if row["phase"] == "terminal_hold"),
            key=lambda row: int(row["hold_cycle"]),
        )
        permitted_cycles = [
            int(row["hold_cycle"]) for row in held if int(row["transition_permitted"])
        ]
        first_permit = permitted_cycles[0] if permitted_cycles else -1
        final_permitted = bool(held and int(held[-1]["transition_permitted"]))
        accepted = premature == 0 and final_permitted
        result.append(
            {
                "task": task,
                "arm": arm,
                "boundary": boundary,
                "transaction_group": str(members[0]["transaction_group"]),
                "demonstration": demonstration,
                "premature_preterminal_permits": premature,
                "first_terminal_hold_permit_cycle": first_permit,
                "final_terminal_hold_permitted": int(final_permitted),
                "accepted": int(accepted),
            }
        )
    return result


def _joint_acceptance_rows(
    summaries: Sequence[Mapping[str, Any]],
    acceptance: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in summaries:
        group = str(row["transaction_group"])
        if group:
            expected[(str(row["task"]), group)].add(str(row["boundary"]))
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in acceptance:
        group = str(row["transaction_group"])
        if group:
            grouped[(str(row["task"]), group, int(row["demonstration"]))].append(row)
    result = []
    for (task, group, demonstration), members in sorted(grouped.items()):
        expected_members = expected[(task, group)]
        received = {str(row["boundary"]) for row in members}
        all_final = all(bool(row["final_terminal_hold_permitted"]) for row in members)
        accepted = (
            received == expected_members
            and all_final
            and all(bool(row["accepted"]) for row in members)
        )
        result.append(
            {
                "task": task,
                "transaction_group": group,
                "demonstration": demonstration,
                "expected_members": "|".join(sorted(expected_members)),
                "received_members": "|".join(sorted(received)),
                "first_joint_hold_permit_cycle": max(
                    int(row["first_terminal_hold_permit_cycle"]) for row in members
                ),
                "all_members_final_permitted": int(all_final),
                "accepted": int(accepted),
            }
        )
    return result


def run(config_path: Path, output: Path) -> None:
    config = _read_config(config_path)
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    runtime_directory = output / "runtime_configs"
    runtime_directory.mkdir()
    belief_config = BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH)
    hold_cycles = math.ceil(
        float(config["terminal_hold_seconds"]) / float(config["control_period_seconds"])
    )
    maximum_demo = max(int(value) for value in config["demonstration_indices"])
    all_rows: list[dict[str, Any]] = []
    for task in config["tasks"]:
        print(f"[phase4-calibration] loading {task}", flush=True)
        cases = _load_cases(task, maximum_demo + 1)
        runtime_config = _base_runtime_config(cases, config)
        for demonstration in config["demonstration_indices"]:
            print(
                f"[phase4-calibration] replay {task} demo {demonstration}",
                flush=True,
            )
            all_rows.extend(
                _replay_task_demo(
                    task,
                    cases,
                    int(demonstration),
                    belief_config,
                    runtime_config,
                    hold_cycles,
                )
            )
    summaries, trials, runtime_configs = _calibrate(all_rows, config)
    if set(runtime_configs) != set(config["tasks"]):
        missing = set(config["tasks"]).difference(runtime_configs)
        raise RuntimeError(f"存在未完成全部边界标定的任务：{sorted(missing)}")
    if any(row["status"] != "calibrated" for row in summaries):
        raise RuntimeError("至少一个边界未通过正常标定，请检查结果")

    # Re-run through the final runtime objects.  This checks EntryGuard's
    # actual strict inequalities and streak counters rather than emulating
    # them from the calibration table.
    acceptance_trace: list[dict[str, Any]] = []
    for task in config["tasks"]:
        print(f"[phase4-calibration] validating {task}", flush=True)
        cases = _load_cases(task, maximum_demo + 1)
        final_runtime = BoundaryRuntimeConfig.from_mapping(runtime_configs[task])
        for demonstration in config["demonstration_indices"]:
            acceptance_trace.extend(
                _replay_task_demo(
                    task,
                    cases,
                    int(demonstration),
                    belief_config,
                    final_runtime,
                    hold_cycles,
                )
            )
    acceptance = _acceptance_rows(acceptance_trace)
    if len(acceptance) != len(summaries) * len(config["demonstration_indices"]):
        raise RuntimeError("正式运行配置复核没有覆盖每个边界×示范")
    if any(not row["accepted"] for row in acceptance):
        raise RuntimeError("正式运行配置在正常回放中提前放行或无法最终放行")
    joint_acceptance = _joint_acceptance_rows(summaries, acceptance)
    expected_joint_trials = len(
        {
            (str(row["task"]), str(row["transaction_group"]))
            for row in summaries
            if row["transaction_group"]
        }
    ) * len(config["demonstration_indices"])
    if len(joint_acceptance) != expected_joint_trials or any(
        not row["accepted"] for row in joint_acceptance
    ):
        raise RuntimeError("联合事务成员未在全部正常示范中共同达到入口许可")

    shutil.copy2(config_path, output / "config.json")
    shutil.copy2(BELIEF_CONFIG_PATH, output / "belief_config.json")
    shutil.copy2(EXECUTION_CONFIG_PATH, output / "execution_config.json")
    _write_csv(output / "boundary_calibration.csv", summaries)
    _write_csv(output / "normal_trials.csv", trials)
    _write_csv(output / "runtime_acceptance.csv", acceptance)
    if joint_acceptance:
        _write_csv(output / "joint_transaction_acceptance.csv", joint_acceptance)
    _write_trace(output / "control_tick_trace.csv.gz", all_rows)
    for task, value in sorted(runtime_configs.items()):
        (runtime_directory / f"{task}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output / "report.md").write_text(
        _report(config, summaries, acceptance, joint_acceptance), encoding="utf-8"
    )
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status_porcelain": subprocess.run(
            ["git", "status", "--short"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
        "numpy": np.__version__,
        "control_period_source_sha256": _sha256(
            REPOSITORY_ROOT / "RLBench/rlbench/task_environment.py"
        ),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output)}\n" for path in files),
        encoding="utf-8",
    )
    print(f"[phase4-calibration] wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
