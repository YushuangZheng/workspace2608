"""Offline phase-level attribution for saved single-arm rollout traces."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from essay2608.policy.base import PHASE_NAMES


def _jumps(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(values, axis=0), axis=-1)


def analyze_phase_trace(
    arrays: dict[str, np.ndarray],
    result: dict[str, Any],
    control_dt: float = 0.02,
) -> dict[str, Any]:
    """Partition a rollout exactly by the destination phase of every step jump."""

    required = {
        "ee_position",
        "object_position",
        "target_position",
        "action",
        "phase",
        "raw_action_position",
        "policy_action_position",
        "action_rate_limited",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Trace is missing arrays: {sorted(missing)}")
    phases = np.asarray(arrays["phase"], dtype=np.int64)
    if not len(phases):
        raise ValueError("Cannot analyze an empty trace.")
    for name in required - {"phase"}:
        if len(arrays[name]) != len(phases):
            raise ValueError(f"Trace array {name!r} has inconsistent length.")
    if np.any((phases < 0) | (phases >= len(PHASE_NAMES))):
        raise ValueError("Trace contains an unknown phase index.")

    ee_jump = _jumps(arrays["ee_position"])
    raw_jump = _jumps(arrays["raw_action_position"])
    policy_jump = _jumps(arrays["policy_action_position"])
    applied_jump = _jumps(np.asarray(arrays["action"])[:, :3])
    destination_phases = phases[1:]
    object_position = np.asarray(arrays["object_position"], dtype=np.float64)
    target_position = np.asarray(arrays["target_position"], dtype=np.float64)
    object_target_xy = np.linalg.norm(object_position[:, :2] - target_position[:, :2], axis=-1)
    object_target_3d = np.linalg.norm(object_position - target_position, axis=-1)
    limited = np.asarray(arrays["action_rate_limited"], dtype=bool)

    phase_rows: dict[str, dict[str, Any]] = {}
    for phase, phase_name in enumerate(PHASE_NAMES):
        state_indices = np.flatnonzero(phases == phase)
        jump_indices = np.flatnonzero(destination_phases == phase)
        if not len(state_indices):
            continue
        first = int(state_indices[0])
        entry_jump_index = first - 1
        phase_rows[phase_name] = {
            "phase": phase,
            "steps": int(len(state_indices)),
            "duration_s": float(len(state_indices) * control_dt),
            # Assigning each jump to its destination phase makes these rows
            # exactly additive to the rollout's saved total path length.
            "path_length_m": float(np.sum(ee_jump[jump_indices])),
            "mean_ee_speed_m_s": (
                float(np.mean(ee_jump[jump_indices] / control_dt)) if len(jump_indices) else 0.0
            ),
            "max_ee_speed_m_s": (
                float(np.max(ee_jump[jump_indices] / control_dt)) if len(jump_indices) else 0.0
            ),
            "max_raw_action_jump_m": (
                float(np.max(raw_jump[jump_indices])) if len(jump_indices) else 0.0
            ),
            "max_policy_action_jump_m": (
                float(np.max(policy_jump[jump_indices])) if len(jump_indices) else 0.0
            ),
            "max_applied_action_jump_m": (
                float(np.max(applied_jump[jump_indices])) if len(jump_indices) else 0.0
            ),
            "entry_raw_action_jump_m": (
                float(raw_jump[entry_jump_index]) if entry_jump_index >= 0 else 0.0
            ),
            "entry_policy_action_jump_m": (
                float(policy_jump[entry_jump_index]) if entry_jump_index >= 0 else 0.0
            ),
            "rate_limited_steps": int(np.sum(limited[state_indices])),
            "mean_object_target_xy_error_m": float(np.mean(object_target_xy[state_indices])),
            "final_object_target_xy_error_m": float(object_target_xy[state_indices[-1]]),
            "final_object_target_3d_error_m": float(object_target_3d[state_indices[-1]]),
        }

    frame_switches = []
    for switch in result["metrics"].get("frame_switch_diagnostics", []):
        step = int(switch["step"])
        frame_switches.append(
            {
                **switch,
                "phase": int(phases[min(step, len(phases) - 1)]),
                "phase_name": PHASE_NAMES[int(phases[min(step, len(phases) - 1)])],
            }
        )
    attributed_path = sum(row["path_length_m"] for row in phase_rows.values())
    saved_path = float(result["metrics"]["path_length_m"])
    return {
        "method": result["method"],
        "condition": result["condition"],
        "seed": int(result["seed"]),
        "steps": int(len(phases)),
        "path_length_m": saved_path,
        "attributed_path_length_m": attributed_path,
        "path_partition_residual_m": attributed_path - saved_path,
        "forced_phase_transitions": int(result["metrics"]["forced_phase_transitions"]),
        "frame_switches": frame_switches,
        "phases": phase_rows,
    }


def summarize_success_metrics(
    results: Iterable[dict[str, Any]],
    legacy_3d_threshold_m: float = 0.06,
) -> dict[str, Any]:
    """Reconcile legacy 3-D, XY, and composite stable-place indicators."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["method"]].append(result["metrics"])
    if not grouped:
        raise ValueError("No trial results were supplied.")
    summary = {}
    for method, metrics in sorted(grouped.items()):
        legacy = [
            bool(
                row["policy_complete"]
                and not row["environment_done"]
                and row["final_error_3d_m"] < legacy_3d_threshold_m
            )
            for row in metrics
        ]
        stable_place = [bool(row["success"]) for row in metrics]
        summary[method] = {
            "num_trials": len(metrics),
            "legacy_3d_success_rate": float(np.mean(legacy)),
            "stable_place_success_rate": float(np.mean(stable_place)),
            "mean_legacy_3d_error_m": float(
                np.mean([row["final_error_3d_m"] for row in metrics])
            ),
            "mean_xy_position_error_m": float(
                np.mean([row["final_xy_error_m"] for row in metrics])
            ),
            "support_rate": float(np.mean([row["object_on_support"] for row in metrics])),
            "release_rate": float(np.mean([row["gripper_released"] for row in metrics])),
            "stability_rate": float(np.mean([row["stable_after_release"] for row in metrics])),
        }
    return {
        "legacy_3d_threshold_m": legacy_3d_threshold_m,
        "methods": summary,
    }


def compare_paired_methods(
    analyses: Iterable[dict[str, Any]],
    baseline: str = "mask_only",
    candidate: str = "full_dynamac",
) -> dict[str, Any]:
    """Return paired candidate-minus-baseline path and duration attribution."""

    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for analysis in analyses:
        if analysis["method"] in {baseline, candidate}:
            by_key[(analysis["condition"], analysis["seed"])][analysis["method"]] = analysis
    incomplete = [key for key, rows in by_key.items() if set(rows) != {baseline, candidate}]
    if incomplete:
        raise ValueError(f"Unpaired method trials: {incomplete}")
    if not by_key:
        raise ValueError("No paired trials were supplied.")

    pairs = []
    phase_path_differences: dict[str, list[float]] = defaultdict(list)
    phase_step_differences: dict[str, list[int]] = defaultdict(list)
    for (condition, seed), rows in sorted(by_key.items()):
        baseline_row = rows[baseline]
        candidate_row = rows[candidate]
        phase_rows = {}
        for phase_name in PHASE_NAMES:
            baseline_phase = baseline_row["phases"].get(phase_name, {})
            candidate_phase = candidate_row["phases"].get(phase_name, {})
            path_difference = float(candidate_phase.get("path_length_m", 0.0)) - float(
                baseline_phase.get("path_length_m", 0.0)
            )
            step_difference = int(candidate_phase.get("steps", 0)) - int(
                baseline_phase.get("steps", 0)
            )
            phase_path_differences[phase_name].append(path_difference)
            phase_step_differences[phase_name].append(step_difference)
            phase_rows[phase_name] = {
                "candidate_minus_baseline_path_m": path_difference,
                "candidate_minus_baseline_steps": step_difference,
            }
        total_difference = candidate_row["path_length_m"] - baseline_row["path_length_m"]
        pairs.append(
            {
                "condition": condition,
                "seed": seed,
                "baseline_path_m": baseline_row["path_length_m"],
                "candidate_path_m": candidate_row["path_length_m"],
                "candidate_minus_baseline_path_m": total_difference,
                "candidate_minus_baseline_percent": 100.0
                * total_difference
                / baseline_row["path_length_m"],
                "candidate_minus_baseline_steps": candidate_row["steps"] - baseline_row["steps"],
                "forced_transitions": {
                    baseline: baseline_row["forced_phase_transitions"],
                    candidate: candidate_row["forced_phase_transitions"],
                },
                "phases": phase_rows,
            }
        )

    total_differences = np.asarray(
        [pair["candidate_minus_baseline_path_m"] for pair in pairs], dtype=np.float64
    )
    total_step_differences = np.asarray(
        [pair["candidate_minus_baseline_steps"] for pair in pairs], dtype=np.float64
    )
    duration_path_correlation = (
        float(np.corrcoef(total_step_differences, total_differences)[0, 1])
        if np.std(total_step_differences) > 0.0 and np.std(total_differences) > 0.0
        else None
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "num_pairs": len(pairs),
        "mean_candidate_minus_baseline_path_m": float(np.mean(total_differences)),
        "mean_candidate_minus_baseline_percent": float(
            np.mean([pair["candidate_minus_baseline_percent"] for pair in pairs])
        ),
        "num_candidate_shorter": int(np.sum(total_differences < 0.0)),
        "duration_path_correlation": duration_path_correlation,
        "phase_mean_differences": {
            phase_name: {
                "candidate_minus_baseline_path_m": float(np.mean(phase_path_differences[phase_name])),
                "candidate_minus_baseline_steps": float(np.mean(phase_step_differences[phase_name])),
            }
            for phase_name in PHASE_NAMES
        },
        "pairs": pairs,
    }
