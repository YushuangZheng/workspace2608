"""Run the deterministic phase-two/three offline component A/B benchmark."""

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from essay2608.policy import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACObservation,
    synchronized_bimanual_demonstrations,
)
from essay2608.policy.dynamac import (
    pose_compose,
    pose_inverse,
    pose_log_nearest,
    relative_pose,
)
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    ClosedLoopExecutionConfig,
    ClosedLoopExecutionController,
    ClosedLoopTaskModel,
    ClosedLoopTaskModelBuilder,
    FrameRole,
    ProgressStatus,
    RelationStateKey,
    RuntimeFeatureBuilder,
    RuntimeObservation,
    StateId,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import get_task_spec
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import demonstration_paths
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    make_store_bottle_semantic_demonstrations,
    store_bottle_semantic_task_spec,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "integrations/rlbench/data/training/main"
MODEL_ROOT = REPOSITORY_ROOT / "integrations/rlbench/models/v4"
STORE_BOTTLE = "bimanual_put_bottle_in_fridge"
SCHEMA_RESULT = "essay2608-phase23-component-ab-result-v1"
BELIEF_CONFIG_PATH = REPOSITORY_ROOT / "configs/closed_loop_belief.json"
EXECUTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/closed_loop_execution.json"

MODEL_PATHS = {
    "stack_wine": MODEL_ROOT / "stack_wine/model.npz",
    "place_cups": MODEL_ROOT / "place_cups/model.npz",
    "open_microwave": MODEL_ROOT / "open_microwave/model.npz",
    "wipe_desk": MODEL_ROOT / "wipe_desk/model.npz",
    "bimanual_handover_item": MODEL_ROOT / "bimanual_handover_item",
    "bimanual_lift_tray": MODEL_ROOT / "bimanual_lift_tray",
    STORE_BOTTLE: MODEL_ROOT / STORE_BOTTLE,
    "bimanual_sweep_to_dustpan": MODEL_ROOT / "bimanual_sweep_to_dustpan",
}


@dataclass(frozen=True)
class Sample:
    state_id: StateId
    ee_pose: np.ndarray
    action_pose: np.ndarray
    frames: dict[str, np.ndarray]
    gripper: np.ndarray
    entity_configurations: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class ArmCase:
    task: str
    arm: str
    policy: DynaMAC
    model: ClosedLoopTaskModel
    demonstrations: Sequence[Any]
    aligned: Mapping[int, Any]
    recoverable_frames: tuple[str, ...]
    demonstration_paths: tuple[Path, ...]
    model_path: Path

    @property
    def key(self) -> str:
        return self.task if self.arm == "single" else f"{self.task}/{self.arm}"


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "seed",
        "claim_boundary",
        "tasks",
        "demonstration_indices",
        "progress",
        "relation",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("评测配置根字段不完整或包含未知字段")
    if value["schema"] not in {
        "essay2608-phase23-component-ab-config-v1",
        "essay2608-phase23-component-ab-config-v2",
        "essay2608-phase23-component-ab-config-v3",
    }:
        raise ValueError("评测配置 schema 不匹配")
    if not value["tasks"] or not value["demonstration_indices"]:
        raise ValueError("任务和示范索引不能为空")
    if len(set(value["tasks"])) != len(value["tasks"]):
        raise ValueError("评测任务不能重复")
    if len(set(value["demonstration_indices"])) != len(value["demonstration_indices"]):
        raise ValueError("评测示范索引不能重复")
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


def _mode_by_skill(policy: DynaMAC, demonstration_index: int) -> dict[int, int]:
    return {
        skill_index: next(
            mode
            for mode, members in enumerate(skill.mode_demonstration_indices)
            if demonstration_index in members
        )
        for skill_index, skill in enumerate(policy.skills)
    }


def _samples_for_skill(
    case: ArmCase,
    demonstration_index: int,
    skill_index: int,
) -> tuple[Sample, ...]:
    data = case.aligned[skill_index]
    samples = []
    for local_index in range(case.policy.skills[skill_index].duration):
        frames = {
            name: values[demonstration_index, local_index].copy()
            for name, values in data.frames.items()
        }
        frames.update(
            {
                name: values[demonstration_index, local_index].copy()
                for name, values in data.scene_entity_poses.items()
            }
        )
        configurations = {
            entity: {
                field: values[demonstration_index, local_index].copy()
                for field, values in fields.items()
            }
            for entity, fields in data.entity_configurations.items()
        }
        samples.append(
            Sample(
                StateId(skill_index, local_index),
                data.ee_pose[demonstration_index, local_index].copy(),
                data.action_pose[demonstration_index, local_index].copy(),
                frames,
                np.atleast_1d(data.gripper[demonstration_index, local_index]).copy(),
                configurations,
            )
        )
    return tuple(samples)


def _runtime_observation(
    tick: int,
    current: Sample,
    previous: Sample | None,
    *,
    previous_command_pose: np.ndarray | None = None,
) -> RuntimeObservation:
    return RuntimeObservation(
        tick=tick,
        ee_pose=current.ee_pose,
        frame_poses=current.frames,
        gripper_state=current.gripper,
        previous_command_pose=(
            None
            if previous is None
            else (
                previous.action_pose
                if previous_command_pose is None
                else previous_command_pose
            )
        ),
        previous_ee_pose=None if previous is None else previous.ee_pose,
        tracking_reliability={},
        frame_visibility={},
        entity_configurations=current.entity_configurations,
    )


def _replace_frame(sample: Sample, frame: str, pose: np.ndarray) -> Sample:
    frames = {name: value.copy() for name, value in sample.frames.items()}
    frames[frame] = np.asarray(pose, dtype=np.float64).copy()
    return Sample(
        sample.state_id,
        sample.ee_pose.copy(),
        sample.action_pose.copy(),
        frames,
        sample.gripper.copy(),
        {
            entity: {name: value.copy() for name, value in fields.items()}
            for entity, fields in sample.entity_configurations.items()
        },
    )


def _initial_relations(
    case: ArmCase,
    state_id: StateId,
    mode_by_skill: Mapping[int, int],
) -> dict[str, np.ndarray]:
    node = case.model.state(state_id)
    mode = mode_by_skill[state_id.skill_index]
    return {
        frame: values[mode].copy()
        for frame, values in node.demo_relation_priors.items()
    }


def _state_global_indices(model: ClosedLoopTaskModel) -> dict[StateId, int]:
    return {state: index for index, state in enumerate(sorted(model.states))}


def _state_text(state_id: StateId) -> str:
    return f"k{state_id.skill_index}:t{state_id.local_index}"


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.mean(array)) if len(array) else math.nan


def _progress_trace(
    case: ArmCase,
    demonstration_index: int,
    scenario: str,
    magnitude: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    durations = [skill.duration for skill in case.policy.skills]
    skill_index = max(range(len(durations)), key=lambda index: durations[index])
    samples = _samples_for_skill(case, demonstration_index, skill_index)
    warmup = min(int(config["warmup_states"]), max(3, len(samples) // 4))
    aftermath = min(int(config["aftermath_states"]), max(4, len(samples) // 3))
    maximum_shift = max(
        [0]
        + list(config["stutter_extra_observations"])
        + list(config["skip_observations"])
    )
    lower = warmup + 1
    upper = len(samples) - aftermath - maximum_shift - 2
    if upper < lower:
        lower = max(2, len(samples) // 3)
        upper = min(len(samples) - 4, 2 * len(samples) // 3)
    if upper < lower:
        raise RuntimeError(f"{case.key} 最长技能仍不足以构造时间扰动窗口")
    anchor = (lower + upper) // 2
    start = max(0, anchor - warmup)
    stop = min(len(samples), anchor + aftermath + maximum_shift + 2)
    source_indices = list(range(start, stop))
    if scenario == "stutter":
        insertion = source_indices.index(anchor) + 1
        source_indices[insertion:insertion] = [anchor] * magnitude
    elif scenario == "skip":
        source_indices = [
            index
            for index in source_indices
            if not anchor < index <= anchor + magnitude
        ]
    elif scenario != "normal":
        raise ValueError(f"未知时间扰动 {scenario}")

    mode_by_skill = _mode_by_skill(case.policy, demonstration_index)
    initial = samples[source_indices[0]]
    case.policy.reset(
        DynaMACObservation(initial.ee_pose, initial.frames), mode_strategy="map"
    )
    updater = BeliefUpdater(
        case.model,
        BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH),
    )
    updater.reset(
        initial_progress={initial.state_id: 1.0},
        initial_relations=_initial_relations(case, initial.state_id, mode_by_skill),
        previous_observation=_runtime_observation(0, initial, None),
    )
    controller = ClosedLoopExecutionController(
        case.model,
        ClosedLoopExecutionConfig.from_json(EXECUTION_CONFIG_PATH),
    )
    controller.reset(initial.state_id)
    global_indices = _state_global_indices(case.model)
    baseline_errors = []
    proposed_errors = []
    baseline_exact = []
    proposed_exact = []
    no_plausible = 0
    trace = []
    previous = initial
    for tick, source_index in enumerate(source_indices[1:], start=1):
        current = samples[source_index]
        belief = updater.update(
            _runtime_observation(tick, current, previous),
            # Offline observations follow the recorded demonstration action.
            # The controller query is audited but is not applied to this
            # fixed counterfactual trajectory.
            executed_reference_state=previous.state_id,
            mode_by_skill=mode_by_skill,
        )
        execution = controller.update(
            belief,
            DynaMACObservation(current.ee_pose, current.frames),
            mode_by_skill=mode_by_skill,
        )
        baseline_local = min(start + tick, len(samples) - 1)
        baseline_state = StateId(skill_index, baseline_local)
        baseline_error = abs(
            global_indices[baseline_state] - global_indices[current.state_id]
        )
        proposed_error = abs(
            global_indices[belief.progress.estimated_state]
            - global_indices[current.state_id]
        )
        baseline_errors.append(float(baseline_error))
        proposed_errors.append(float(proposed_error))
        baseline_exact.append(float(baseline_error == 0))
        proposed_exact.append(float(proposed_error == 0))
        no_plausible += int(belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE)
        trace.append(
            {
                "family": "progress",
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration_index,
                "scenario": scenario,
                "magnitude": magnitude,
                "frame": "",
                "tick": tick,
                "truth_state": _state_text(current.state_id),
                "baseline_state": _state_text(baseline_state),
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
                "baseline_abs_error": baseline_error,
                "proposed_abs_error": proposed_error,
                "progress_status": belief.progress.status.value,
                "relation_decision": "",
                "frame_role": "",
                "execution_weight": "",
                "blocks_advance": "",
                "in_perturbation": int(
                    scenario != "normal"
                    and (
                        source_index == anchor
                        if scenario == "stutter"
                        else tick == source_indices.index(anchor) + 1
                    )
                ),
            }
        )
        previous = current

    trial_id = f"progress:{case.key}:d{demonstration_index}:{scenario}:m{magnitude}"
    result = {
        "trial_id": trial_id,
        "task": case.task,
        "arm": case.arm,
        "demonstration": demonstration_index,
        "scenario": scenario,
        "magnitude": magnitude,
        "skill_index": skill_index,
        "anchor_local_index": anchor,
        "ticks": len(baseline_errors),
        "baseline_mae": _mean(baseline_errors),
        "proposed_mae": _mean(proposed_errors),
        "mae_improvement": _mean(baseline_errors) - _mean(proposed_errors),
        "baseline_exact_rate": _mean(baseline_exact),
        "proposed_exact_rate": _mean(proposed_exact),
        "baseline_final_error": baseline_errors[-1],
        "proposed_final_error": proposed_errors[-1],
        "proposed_no_plausible_cycles": no_plausible,
        "proposed_wins": int(_mean(proposed_errors) < _mean(baseline_errors)),
    }
    for row in trace:
        row["trial_id"] = trial_id
    return result, trace


def _motion_magnitude(sequence: Sequence[Sample], start: int, stop: int) -> float:
    builder = RuntimeFeatureBuilder()
    return float(
        sum(
            builder.motion_magnitude(
                pose_log_nearest(
                    sequence[index - 1].ee_pose,
                    sequence[index].ee_pose,
                )
            )
            for index in range(start + 1, stop)
        )
    )


def _relation_candidate(
    case: ArmCase,
    demonstration_index: int,
    target: str,
    config: Mapping[str, Any],
) -> tuple[int, int, str] | None:
    mode_by_skill = _mode_by_skill(case.policy, demonstration_index)
    probability_index = 1 if target == "linked" else 0
    threshold = float(config["expected_relation_probability"])
    warmup = int(config["warmup_states"])
    horizon = int(config["perturbation_states"]) + int(config["aftermath_states"])
    candidates = []
    for skill_index, skill in enumerate(case.policy.skills):
        samples = _samples_for_skill(case, demonstration_index, skill_index)
        mode = mode_by_skill[skill_index]
        lower = warmup + 1
        # A relation window may end exactly at the skill's final state.  No
        # post-window observation is required: terminal LINK evidence is one
        # of the cases this benchmark is intended to cover.
        upper = len(samples) - horizon
        for anchor in range(lower, max(lower, upper + 1)):
            stop = anchor + horizon
            if stop > len(samples):
                continue
            for frame in case.recoverable_frames:
                stable = True
                formal_origin = None
                for local_index in range(anchor, stop):
                    state_id = StateId(skill_index, local_index)
                    node = case.model.state(state_id)
                    probability_supported = (
                        node.demo_relation_priors[frame][mode, probability_index]
                        >= threshold
                    )
                    if target == "linked":
                        origin = case.model.link_origins.get(
                            RelationStateKey(case.model.arm_id, frame, state_id, mode)
                        )
                        if formal_origin is None:
                            formal_origin = origin
                        event_confirmed = origin is not None and origin == formal_origin
                        eligible = probability_supported and event_confirmed
                    else:
                        eligible = (
                            probability_supported
                            and frame in node.mode_selected_frames[mode]
                        )
                    if not eligible:
                        stable = False
                        break
                if not stable:
                    continue
                motion = _motion_magnitude(samples, anchor - 1, stop)
                if motion < float(config["minimum_cumulative_ee_motion"]):
                    continue
                minimum_probability = min(
                    float(
                        case.model.state(
                            StateId(skill_index, local_index)
                        ).demo_relation_priors[frame][mode, probability_index]
                    )
                    for local_index in range(anchor, stop)
                )
                candidates.append(
                    (minimum_probability + motion, skill_index, anchor, frame)
                )
    if not candidates:
        return None
    _, skill_index, anchor, frame = max(candidates)
    return skill_index, anchor, frame


def _relation_trace(
    case: ArmCase,
    demonstration_index: int,
    intervention: str,
    perturbed: bool,
    candidate: tuple[int, int, str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    skill_index, anchor, frame = candidate
    samples = list(_samples_for_skill(case, demonstration_index, skill_index))
    warmup = int(config["warmup_states"])
    perturbation = int(config["perturbation_states"])
    aftermath = int(config["aftermath_states"])
    start = anchor - warmup
    stop = anchor + perturbation + aftermath
    samples = samples[start:stop]
    local_anchor = warmup
    expected = "linked" if intervention == "break_link" else "external"
    if perturbed:
        if intervention == "break_link":
            frozen = samples[local_anchor - 1].frames[frame].copy()
            for index in range(local_anchor, len(samples)):
                samples[index] = _replace_frame(samples[index], frame, frozen)
        elif intervention == "false_coupling":
            relation = relative_pose(
                samples[local_anchor - 1].frames[frame],
                samples[local_anchor - 1].ee_pose,
            )
            for index in range(local_anchor, len(samples)):
                following = pose_compose(samples[index].ee_pose, pose_inverse(relation))
                samples[index] = _replace_frame(samples[index], frame, following)
        else:
            raise ValueError(f"未知关系扰动 {intervention}")

    mode_by_skill = _mode_by_skill(case.policy, demonstration_index)
    initial = samples[0]
    case.policy.reset(
        DynaMACObservation(initial.ee_pose, initial.frames), mode_strategy="map"
    )
    updater = BeliefUpdater(
        case.model,
        BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH),
    )
    updater.reset(
        initial_progress={initial.state_id: 1.0},
        initial_relations=_initial_relations(case, initial.state_id, mode_by_skill),
        previous_observation=_runtime_observation(0, initial, None),
    )
    controller = ClosedLoopExecutionController(
        case.model,
        ClosedLoopExecutionConfig.from_json(EXECUTION_CONFIG_PATH),
    )
    controller.reset(initial.state_id)
    roles = []
    blocks = []
    decisions = []
    weights = []
    baseline_active = []
    detection_ticks = []
    trace = []
    previous = initial
    for tick, current in enumerate(samples[1:], start=1):
        belief = updater.update(
            _runtime_observation(tick, current, previous),
            executed_reference_state=previous.state_id,
            mode_by_skill=mode_by_skill,
        )
        execution = controller.update(
            belief,
            DynaMACObservation(current.ee_pose, current.frames),
            mode_by_skill=mode_by_skill,
        )
        snapshot = execution.roles
        decision = snapshot.decisions.get(frame)
        role = None if decision is None else decision.role
        weight = 0.0 if decision is None else decision.execution_weight
        relation_decision = belief.relation_estimates[frame].decision_state.value
        stream = case.policy.skills[skill_index].streams.get(frame)
        active = bool(
            stream is not None
            and stream.is_active(
                mode_by_skill[skill_index],
                current.state_id.local_index,
            )
        )
        in_horizon = tick >= local_anchor
        if in_horizon:
            roles.append("unselected" if role is None else role.value)
            blocks.append(bool(snapshot.blocks_advance))
            decisions.append(relation_decision)
            weights.append(float(weight))
            baseline_active.append(bool(active))
            if role == FrameRole.RECOVER:
                detection_ticks.append(tick - local_anchor)
        trace.append(
            {
                "family": "relation",
                "task": case.task,
                "arm": case.arm,
                "demonstration": demonstration_index,
                "scenario": intervention + ("_perturbed" if perturbed else "_control"),
                "magnitude": perturbation,
                "frame": frame,
                "tick": tick,
                "truth_state": _state_text(current.state_id),
                "baseline_state": _state_text(current.state_id),
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
                "baseline_abs_error": 0,
                "proposed_abs_error": abs(
                    _state_global_indices(case.model)[belief.progress.estimated_state]
                    - _state_global_indices(case.model)[current.state_id]
                ),
                "progress_status": belief.progress.status.value,
                "relation_decision": relation_decision,
                "frame_role": "unselected" if role is None else role.value,
                "execution_weight": weight,
                "blocks_advance": int(snapshot.blocks_advance),
                "in_perturbation": int(in_horizon),
            }
        )
        previous = current

    scenario = intervention + ("_perturbed" if perturbed else "_control")
    trial_id = f"relation:{case.key}:d{demonstration_index}:{scenario}:{frame}"
    recover_count = sum(role == FrameRole.RECOVER.value for role in roles)
    defer_count = sum(role == FrameRole.DEFER.value for role in roles)
    result = {
        "trial_id": trial_id,
        "task": case.task,
        "arm": case.arm,
        "demonstration": demonstration_index,
        "scenario": scenario,
        "intervention": intervention,
        "perturbed": int(perturbed),
        "frame": frame,
        "expected_relation": expected,
        "skill_index": skill_index,
        "anchor_local_index": anchor,
        "horizon_ticks": len(roles),
        "baseline_detection": 0,
        "proposed_detection": int(bool(detection_ticks)),
        "proposed_detection_latency": (detection_ticks[0] if detection_ticks else ""),
        "proposed_recover_rate": recover_count / max(1, len(roles)),
        "proposed_defer_rate": defer_count / max(1, len(roles)),
        "proposed_block_rate": _mean(float(value) for value in blocks),
        "baseline_active_rate": _mean(float(value) for value in baseline_active),
        "proposed_mean_weight": _mean(weights),
        "proposed_final_relation": decisions[-1],
        "baseline_active_expert_suppressed": int(
            intervention == "false_coupling"
            and any(baseline_active)
            and bool(detection_ticks)
        ),
    }
    for row in trace:
        row["trial_id"] = trial_id
    return result, trace


def _load_cases(task: str, demonstration_count: int) -> list[ArmCase]:
    paths = tuple(demonstration_paths(DATA_ROOT, task, demonstration_count))
    episodes = load_low_dim_obs_pickles(paths)
    names = [path.parent.name for path in paths]
    builder = ClosedLoopTaskModelBuilder()
    model_path = MODEL_PATHS[task]
    if task == STORE_BOTTLE:
        spec = store_bottle_semantic_task_spec()
        converted = make_store_bottle_semantic_demonstrations(episodes, names=names)
    else:
        spec = get_task_spec(task)
        converted = (
            make_bimanual_demonstrations(episodes, spec, names=names)
            if spec.bimanual
            else make_unimanual_demonstrations(episodes, spec, names=names)
        )
    recoverable = spec.recoverable_relation_frames
    if not spec.bimanual:
        policy = DynaMAC.load(model_path)
        model = builder.build(
            policy,
            converted.demonstrations,
            recoverable_frames=recoverable,
        )
        return [
            ArmCase(
                task,
                "single",
                policy,
                model,
                converted.demonstrations,
                builder._align_demonstrations(policy, converted.demonstrations),
                recoverable,
                paths,
                model_path,
            )
        ]

    policy = BimanualDynaMAC(
        left=DynaMAC.load(model_path / "left.npz"),
        right=DynaMAC.load(model_path / "right.npz"),
    )
    left_model, right_model = builder.build_bimanual(
        policy,
        converted.left_demonstrations,
        converted.right_demonstrations,
        recoverable_frames=recoverable,
    )
    left_demos, right_demos = synchronized_bimanual_demonstrations(
        converted.left_demonstrations,
        converted.right_demonstrations,
    )
    return [
        ArmCase(
            task,
            "left",
            policy.left,
            left_model,
            left_demos,
            builder._align_demonstrations(policy.left, left_demos),
            recoverable,
            paths,
            model_path / "left.npz",
        ),
        ArmCase(
            task,
            "right",
            policy.right,
            right_model,
            right_demos,
            builder._align_demonstrations(policy.right, right_demos),
            recoverable,
            paths,
            model_path / "right.npz",
        ),
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"不能写入空结果 {path.name}")
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("逐周期轨迹不能为空")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _progress_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scenario"])].append(row)
    result = []
    for scenario, members in sorted(groups.items()):
        baseline = _mean(float(row["baseline_mae"]) for row in members)
        proposed = _mean(float(row["proposed_mae"]) for row in members)
        result.append(
            {
                "family": "progress",
                "scenario": scenario,
                "trials": len(members),
                "baseline_mae": baseline,
                "proposed_mae": proposed,
                "absolute_mae_reduction": baseline - proposed,
                "relative_mae_reduction": (
                    0.0 if baseline == 0.0 else (baseline - proposed) / baseline
                ),
                "proposed_win_rate": _mean(
                    float(row["proposed_wins"]) for row in members
                ),
                "baseline_exact_rate": _mean(
                    float(row["baseline_exact_rate"]) for row in members
                ),
                "proposed_exact_rate": _mean(
                    float(row["proposed_exact_rate"]) for row in members
                ),
            }
        )
    return result


def _relation_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["scenario"])].append(row)
    result = []
    for scenario, members in sorted(groups.items()):
        latencies = [
            float(row["proposed_detection_latency"])
            for row in members
            if row["proposed_detection_latency"] != ""
        ]
        result.append(
            {
                "family": "relation",
                "scenario": scenario,
                "trials": len(members),
                "baseline_detection_rate": _mean(
                    float(row["baseline_detection"]) for row in members
                ),
                "proposed_detection_rate": _mean(
                    float(row["proposed_detection"]) for row in members
                ),
                "proposed_mean_block_rate": _mean(
                    float(row["proposed_block_rate"]) for row in members
                ),
                "median_detection_latency": (
                    float(median(latencies)) if latencies else ""
                ),
                "active_expert_suppression_rate": _mean(
                    float(row["baseline_active_expert_suppressed"]) for row in members
                ),
            }
        )
    return result


def _task_summary(
    progress_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep task/arm-level aggregates so pooled results remain auditable."""
    result: list[dict[str, Any]] = []
    progress_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in progress_rows:
        key = (str(row["task"]), str(row["arm"]), str(row["scenario"]))
        progress_groups[key].append(row)
    for (task, arm, scenario), members in sorted(progress_groups.items()):
        baseline = _mean(float(row["baseline_mae"]) for row in members)
        proposed = _mean(float(row["proposed_mae"]) for row in members)
        total_ticks = sum(int(row["ticks"]) for row in members)
        result.append(
            {
                "family": "progress",
                "task": task,
                "arm": arm,
                "scenario": scenario,
                "trials": len(members),
                "baseline_mae": baseline,
                "proposed_mae": proposed,
                "relative_mae_reduction": (
                    0.0 if baseline == 0.0 else (baseline - proposed) / baseline
                ),
                "proposed_win_rate": _mean(
                    float(row["proposed_wins"]) for row in members
                ),
                "proposed_no_plausible_cycle_rate": (
                    sum(int(row["proposed_no_plausible_cycles"]) for row in members)
                    / max(1, total_ticks)
                ),
            }
        )

    relation_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in relation_rows:
        key = (str(row["task"]), str(row["arm"]), str(row["scenario"]))
        relation_groups[key].append(row)
    for (task, arm, scenario), members in sorted(relation_groups.items()):
        latencies = [
            float(row["proposed_detection_latency"])
            for row in members
            if row["proposed_detection_latency"] != ""
        ]
        result.append(
            {
                "family": "relation",
                "task": task,
                "arm": arm,
                "scenario": scenario,
                "trials": len(members),
                "baseline_detection_rate": _mean(
                    float(row["baseline_detection"]) for row in members
                ),
                "proposed_detection_rate": _mean(
                    float(row["proposed_detection"]) for row in members
                ),
                "proposed_block_rate": _mean(
                    float(row["proposed_block_rate"]) for row in members
                ),
                "median_detection_latency": (
                    float(median(latencies)) if latencies else ""
                ),
                "active_expert_suppression_rate": _mean(
                    float(row["baseline_active_expert_suppressed"]) for row in members
                ),
            }
        )
    return result


def _normalize_generated_text(path: Path) -> None:
    """Keep generated text artifacts stable and clean in version control."""

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _plots(
    directory: Path,
    progress_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    directory.mkdir()
    progress_summary = [row for row in summary if row["family"] == "progress"]
    labels = [str(row["scenario"]) for row in progress_summary]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(
        positions - width / 2,
        [float(row["baseline_mae"]) for row in progress_summary],
        width,
        label="DynaMAC fixed clock",
    )
    axis.bar(
        positions + width / 2,
        [float(row["proposed_mae"]) for row in progress_summary],
        width,
        label="Online progress estimate",
    )
    axis.set_xticks(positions, labels, rotation=20)
    axis.set_ylabel("Mean absolute StateId error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    progress_figure = directory / "progress_mae_by_scenario.svg"
    figure.savefig(progress_figure)
    plt.close(figure)
    _normalize_generated_text(progress_figure)

    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    baseline = np.asarray([float(row["baseline_mae"]) for row in progress_rows])
    proposed = np.asarray([float(row["proposed_mae"]) for row in progress_rows])
    axis.scatter(baseline, proposed, s=12, alpha=0.55)
    maximum = max(1.0, float(np.max(np.concatenate((baseline, proposed)))))
    axis.plot([0.0, maximum], [0.0, maximum], "--", color="black", linewidth=1)
    axis.set_xlim(0.0, maximum)
    axis.set_ylim(0.0, maximum)
    axis.set_xlabel("DynaMAC fixed-clock MAE")
    axis.set_ylabel("Online-estimator MAE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    scatter_figure = directory / "progress_trial_scatter.svg"
    figure.savefig(scatter_figure)
    plt.close(figure)
    _normalize_generated_text(scatter_figure)

    relation_summary = [row for row in summary if row["family"] == "relation"]
    labels = [str(row["scenario"]) for row in relation_summary]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(
        positions - width / 2,
        [float(row["baseline_detection_rate"]) for row in relation_summary],
        width,
        label="DynaMAC static mask",
    )
    axis.bar(
        positions + width / 2,
        [float(row["proposed_detection_rate"]) for row in relation_summary],
        width,
        label="Dynamic roles",
    )
    axis.set_xticks(positions, labels, rotation=20)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Mismatch detection rate")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    relation_figure = directory / "relation_detection.svg"
    figure.savefig(relation_figure)
    plt.close(figure)
    _normalize_generated_text(relation_figure)


def _environment(
    cases: Sequence[ArmCase],
    config_path: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    source_files = sorted(
        (REPOSITORY_ROOT / "source/policy/closed_loop").glob("*.py")
    ) + [
        REPOSITORY_ROOT / "source/policy/dynamac.py",
        BELIEF_CONFIG_PATH,
        EXECUTION_CONFIG_PATH,
        Path(__file__).resolve(),
        config_path,
    ]
    model_paths = sorted({case.model_path for case in cases})
    demo_paths = sorted({path for case in cases for path in case.demonstration_paths})
    return {
        "schema": "essay2608-phase23-component-ab-environment-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_porcelain": _git("status", "--porcelain").splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in source_files
        },
        "model_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in model_paths
        },
        "demonstration_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path) for path in demo_paths
        },
        "policy_fingerprints": {case.key: case.policy.fingerprint() for case in cases},
    }


def _report(
    path: Path,
    config: Mapping[str, Any],
    cases: Sequence[ArmCase],
    progress_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> None:
    progress_summary = [row for row in summary if row["family"] == "progress"]
    relation_summary = [row for row in summary if row["family"] == "relation"]
    lines = [
        "# 阶段二—三组件级 A/B 评测报告",
        "",
        "## 评测边界",
        "",
        str(config["claim_boundary"]),
        "",
        f"覆盖 {len(set(case.task for case in cases))} 个任务、{len(cases)} 套机械臂模型、"
        f"{len(config['demonstration_indices'])} 条正常示范索引。时间试验 {len(progress_rows)} 组，"
        f"关系试验 {len(relation_rows)} 组。",
        "",
        "## 时间扰动结果",
        "",
        "| 场景 | 试验数 | 固定时钟 MAE | 在线估计 MAE | 相对降低 | 新方法胜率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in progress_summary:
        lines.append(
            f"| {row['scenario']} | {row['trials']} | {row['baseline_mae']:.4f} | "
            f"{row['proposed_mae']:.4f} | {100*row['relative_mae_reduction']:.2f}% | "
            f"{100*row['proposed_win_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 关系扰动结果",
            "",
            "固定流掩码基线没有在线关系失配判定，因此其检测率按机制定义为0；对照组用于衡量动态角色误报。",
            "",
            "| 场景 | 试验数 | 基线检测率 | 动态角色检测率 | 平均阻断率 | 检测延迟中位数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in relation_summary:
        latency = (
            "—"
            if row["median_detection_latency"] == ""
            else f"{row['median_detection_latency']:.1f}"
        )
        lines.append(
            f"| {row['scenario']} | {row['trials']} | "
            f"{100*row['baseline_detection_rate']:.2f}% | "
            f"{100*row['proposed_detection_rate']:.2f}% | "
            f"{100*row['proposed_mean_block_rate']:.2f}% | {latency} |"
        )
    lines.extend(
        [
            "",
            "## 可复现性",
            "",
            "配置、逐试验指标、按任务/机械臂聚合的统计、压缩逐周期轨迹、源码/模型/"
            "示范哈希、环境信息、图表和 SHA256SUMS 均与本报告同目录保存。",
            "",
            "## 解释限制",
            "",
            "该结果只支持在线进度估计、动态角色和推进阻断的组件级结论。输入由训练用正常示范派生，且离线动作不会改变后续环境；不得将本结果表述为独立测试集泛化、恢复成功率或完整任务成功率。",
        ]
    )
    if skipped:
        counts = Counter(str(row["reason"]) for row in skipped)
        lines.extend(["", "## 未构造的关系条件", ""])
        for reason, count in sorted(counts.items()):
            lines.append(f"- {reason}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checksums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    text = "\n".join(
        f"{_sha256(path)}  {path.relative_to(directory)}" for path in files
    )
    (directory / "SHA256SUMS").write_text(text + "\n", encoding="utf-8")


def run(
    config_path: Path,
    output: Path,
    *,
    selected_tasks: Sequence[str] | None = None,
    selected_demonstrations: Sequence[int] | None = None,
) -> None:
    config = _read_config(config_path)
    tasks = list(config["tasks"])
    configured_demonstrations = list(config["demonstration_indices"])
    demonstrations = list(configured_demonstrations)
    if selected_tasks:
        unknown_tasks = set(selected_tasks).difference(tasks)
        if unknown_tasks:
            raise ValueError(f"命令行任务不在固定配置中：{sorted(unknown_tasks)}")
        tasks = [task for task in tasks if task in selected_tasks]
    if selected_demonstrations is not None:
        unknown_demonstrations = set(selected_demonstrations).difference(demonstrations)
        if unknown_demonstrations:
            raise ValueError(
                "命令行示范索引不在固定配置中：" f"{sorted(unknown_demonstrations)}"
            )
        demonstrations = [
            index for index in demonstrations if index in selected_demonstrations
        ]
    if output.exists():
        raise FileExistsError(f"评测输出目录已存在，拒绝覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"评测暂存目录已存在：{staging}")
    staging.mkdir()
    try:
        effective_config = json.loads(json.dumps(config))
        effective_config["tasks"] = tasks
        effective_config["demonstration_indices"] = demonstrations
        (staging / "config.json").write_text(
            json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "belief_config.json").write_text(
            json.dumps(
                BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH).to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "execution_config.json").write_text(
            json.dumps(
                ClosedLoopExecutionConfig.from_json(EXECUTION_CONFIG_PATH).to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        cases = []
        for task in tasks:
            print(f"loading {task}", flush=True)
            cases.extend(_load_cases(task, max(configured_demonstrations) + 1))

        progress_rows: list[dict[str, Any]] = []
        relation_rows: list[dict[str, Any]] = []
        tick_rows: list[dict[str, Any]] = []
        skipped = []
        progress_config = effective_config["progress"]
        relation_config = effective_config["relation"]
        for case in cases:
            print(f"evaluating {case.key}", flush=True)
            for demonstration_index in demonstrations:
                if progress_config["include_normal_control"]:
                    trial, trace = _progress_trace(
                        case,
                        demonstration_index,
                        "normal",
                        0,
                        progress_config,
                    )
                    progress_rows.append(trial)
                    tick_rows.extend(trace)
                for magnitude in progress_config["stutter_extra_observations"]:
                    trial, trace = _progress_trace(
                        case,
                        demonstration_index,
                        "stutter",
                        int(magnitude),
                        progress_config,
                    )
                    progress_rows.append(trial)
                    tick_rows.extend(trace)
                for magnitude in progress_config["skip_observations"]:
                    trial, trace = _progress_trace(
                        case,
                        demonstration_index,
                        "skip",
                        int(magnitude),
                        progress_config,
                    )
                    progress_rows.append(trial)
                    tick_rows.extend(trace)

                for intervention in relation_config["interventions"]:
                    target = "linked" if intervention == "break_link" else "external"
                    candidate = _relation_candidate(
                        case,
                        demonstration_index,
                        target,
                        relation_config,
                    )
                    if candidate is None:
                        skipped.append(
                            {
                                "task": case.task,
                                "arm": case.arm,
                                "demonstration": demonstration_index,
                                "intervention": intervention,
                                "reason": (
                                    "no stable formal linked interval"
                                    if target == "linked"
                                    else "no stable selected external segment"
                                ),
                            }
                        )
                        continue
                    controls = (
                        [False, True]
                        if relation_config["include_matched_controls"]
                        else [True]
                    )
                    for perturbed in controls:
                        trial, trace = _relation_trace(
                            case,
                            demonstration_index,
                            intervention,
                            perturbed,
                            candidate,
                            relation_config,
                        )
                        relation_rows.append(trial)
                        tick_rows.extend(trace)

        summary = _progress_summary(progress_rows) + _relation_summary(relation_rows)
        task_summary = _task_summary(progress_rows, relation_rows)
        _write_csv(staging / "progress_trials.csv", progress_rows)
        _write_csv(staging / "relation_trials.csv", relation_rows)
        _write_csv(staging / "task_summary.csv", task_summary)
        _write_trace(staging / "tick_trace.csv.gz", tick_rows)
        _write_csv(staging / "summary.csv", summary)
        (staging / "summary.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA_RESULT,
                    "progress_trials": len(progress_rows),
                    "relation_trials": len(relation_rows),
                    "tick_rows": len(tick_rows),
                    "skipped_relation_candidates": skipped,
                    "summary": summary,
                    "task_summary": task_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "environment.json").write_text(
            json.dumps(
                _environment(cases, config_path, sys.argv),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _plots(staging / "figures", progress_rows, relation_rows, summary)
        _report(
            staging / "report.md",
            effective_config,
            cases,
            progress_rows,
            relation_rows,
            summary,
            skipped,
        )
        _checksums(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--demonstrations", nargs="*", type=int)
    arguments = parser.parse_args()
    run(
        arguments.config.resolve(),
        arguments.output.resolve(),
        selected_tasks=arguments.tasks,
        selected_demonstrations=arguments.demonstrations,
    )


if __name__ == "__main__":
    main()
