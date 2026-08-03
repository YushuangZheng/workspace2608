"""Velocity-based skill-boundary diagnostics for frozen demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import numpy as np

from essay2608.data.dataset import Demonstration
from essay2608.data.transforms import quaternion_distance_radians


@dataclass(frozen=True)
class SegmentationConfig:
    """Training-only parameters for low-speed interval detection."""

    smoothing_duration_s: float = 0.10
    minimum_low_speed_duration_s: float = 0.12
    maximum_merge_gap_s: float = 0.08
    endpoint_tolerance_s: float = 0.06
    threshold_quantile: float = 0.40
    alignment_tolerance_normalized: float = 0.05
    minimum_alignment_support_fraction: float = 0.60


@dataclass(frozen=True)
class SegmentationTrace:
    """Per-sample diagnostic arrays and their serializable summary."""

    demonstration_name: str
    time: np.ndarray
    linear_speed_m_s: np.ndarray
    angular_speed_rad_s: np.ndarray
    low_speed_mask: np.ndarray
    low_speed_intervals: tuple[tuple[int, int], ...]
    boundary_indices: np.ndarray
    manual_boundary_indices: np.ndarray
    summary: dict[str, Any]


def _odd_window(duration_s: float, control_dt: float) -> int:
    samples = max(int(round(duration_s / control_dt)), 1)
    return samples if samples % 2 else samples + 1


def _duration_steps(duration_s: float, control_dt: float) -> int:
    return max(int(ceil(duration_s / control_dt - 1.0e-9)), 1)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values.astype(np.float64, copy=True)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def end_effector_speeds(
    demonstration: Demonstration,
    smoothing_duration_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed end-effector linear and angular speed at every sample."""

    dt = demonstration.control_dt
    position_delta = np.linalg.norm(np.diff(demonstration.ee_pose[:, :3], axis=0), axis=-1)
    angular_delta = quaternion_distance_radians(
        demonstration.ee_pose[1:, 3:7],
        demonstration.ee_pose[:-1, 3:7],
    )
    linear = np.concatenate((position_delta / dt, position_delta[-1:] / dt))
    angular = np.concatenate((angular_delta / dt, angular_delta[-1:] / dt))
    window = _odd_window(smoothing_duration_s, dt)
    return _moving_average(linear, window), _moving_average(angular, window)


def calibrate_speed_thresholds(
    demonstrations: list[Demonstration],
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[float, float]:
    """Calibrate one pair of thresholds from all frozen training trajectories."""

    if not demonstrations:
        raise ValueError("At least one demonstration is required.")
    speeds = [
        end_effector_speeds(demonstration, config.smoothing_duration_s)
        for demonstration in demonstrations
    ]
    linear = np.concatenate([item[0] for item in speeds])
    angular = np.concatenate([item[1] for item in speeds])
    return (
        float(np.quantile(linear, config.threshold_quantile)),
        float(np.quantile(angular, config.threshold_quantile)),
    )


def _true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def _merge_intervals(
    intervals: list[tuple[int, int]],
    maximum_gap_steps: int,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= maximum_gap_steps:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _boundary_candidates(
    intervals: list[tuple[int, int]],
    steps: int,
    endpoint_tolerance_steps: int,
) -> np.ndarray:
    boundaries = []
    for start, end in intervals:
        if start <= endpoint_tolerance_steps and end < steps:
            boundary = end
        elif steps - end <= endpoint_tolerance_steps and start > 0:
            boundary = start
        elif start > 0 and end < steps:
            boundary = (start + end) // 2
        else:
            continue
        if 0 < boundary < steps:
            boundaries.append(int(boundary))
    return np.asarray(sorted(set(boundaries)), dtype=np.int64)


def _manual_boundaries(demonstration: Demonstration) -> np.ndarray:
    return np.flatnonzero(np.diff(demonstration.state) != 0).astype(np.int64) + 1


def _segment_state_composition(
    states: np.ndarray,
    boundaries: np.ndarray,
) -> list[dict[str, Any]]:
    edges = np.concatenate(([0], boundaries, [len(states)]))
    result = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        values, counts = np.unique(states[start:end], return_counts=True)
        result.append(
            {
                "segment": index,
                "start_index": int(start),
                "end_index_exclusive": int(end),
                "manual_state_fractions": {
                    str(int(value)): float(count / max(end - start, 1))
                    for value, count in zip(values, counts)
                },
            }
        )
    return result


def segment_demonstration(
    demonstration: Demonstration,
    linear_speed_threshold_m_s: float,
    angular_speed_threshold_rad_s: float,
    config: SegmentationConfig = SegmentationConfig(),
) -> SegmentationTrace:
    """Find persistent low-speed intervals and derive candidate skill boundaries."""

    linear, angular = end_effector_speeds(demonstration, config.smoothing_duration_s)
    low_speed = (linear <= linear_speed_threshold_m_s) & (
        angular <= angular_speed_threshold_rad_s
    )
    gap_steps = _duration_steps(config.maximum_merge_gap_s, demonstration.control_dt)
    minimum_steps = _duration_steps(
        config.minimum_low_speed_duration_s,
        demonstration.control_dt,
    )
    endpoint_steps = _duration_steps(config.endpoint_tolerance_s, demonstration.control_dt)
    intervals = _merge_intervals(_true_intervals(low_speed), gap_steps)
    intervals = [item for item in intervals if item[1] - item[0] >= minimum_steps]
    boundaries = _boundary_candidates(intervals, demonstration.steps, endpoint_steps)
    manual = _manual_boundaries(demonstration)

    candidate_rows = []
    for boundary in boundaries:
        nearest_index = int(manual[np.argmin(np.abs(manual - boundary))])
        candidate_rows.append(
            {
                "index": int(boundary),
                "time_s": float(demonstration.time[boundary]),
                "normalized_time": float(boundary / max(demonstration.steps - 1, 1)),
                "manual_state_at_boundary": int(demonstration.state[boundary]),
                "nearest_manual_boundary_index": nearest_index,
                "nearest_manual_boundary_time_s": float(demonstration.time[nearest_index]),
                "nearest_manual_boundary_absolute_deviation_s": float(
                    abs(boundary - nearest_index) * demonstration.control_dt
                ),
            }
        )

    summary = {
        "demonstration": demonstration.path.name,
        "steps": demonstration.steps,
        "control_dt_s": demonstration.control_dt,
        "duration_s": float(demonstration.time[-1]),
        "smoothing_window_steps": _odd_window(
            config.smoothing_duration_s,
            demonstration.control_dt,
        ),
        "minimum_low_speed_steps": minimum_steps,
        "maximum_merge_gap_steps": gap_steps,
        "thresholds": {
            "linear_speed_m_s": linear_speed_threshold_m_s,
            "angular_speed_rad_s": angular_speed_threshold_rad_s,
        },
        "low_speed_fraction": float(np.mean(low_speed)),
        "low_speed_intervals": [
            {
                "start_index": start,
                "end_index_exclusive": end,
                "start_time_s": float(demonstration.time[start]),
                "end_time_s": float(
                    demonstration.time[min(end, demonstration.steps - 1)]
                ),
                "duration_s": float((end - start) * demonstration.control_dt),
                "manual_states": sorted(
                    int(value) for value in np.unique(demonstration.state[start:end])
                ),
            }
            for start, end in intervals
        ],
        "candidate_boundaries": candidate_rows,
        "automatic_segment_count": int(len(boundaries) + 1),
        "manual_segment_count": int(len(manual) + 1),
        "manual_boundaries": [
            {
                "index": int(index),
                "time_s": float(demonstration.time[index]),
                "new_state": int(demonstration.state[index]),
            }
            for index in manual
        ],
        "automatic_segment_state_composition": _segment_state_composition(
            demonstration.state,
            boundaries,
        ),
    }
    return SegmentationTrace(
        demonstration_name=demonstration.path.name,
        time=demonstration.time.copy(),
        linear_speed_m_s=linear,
        angular_speed_rad_s=angular,
        low_speed_mask=low_speed,
        low_speed_intervals=tuple(intervals),
        boundary_indices=boundaries,
        manual_boundary_indices=manual,
        summary=summary,
    )


def align_candidate_boundaries(
    traces: list[SegmentationTrace],
    config: SegmentationConfig = SegmentationConfig(),
) -> list[dict[str, Any]]:
    """Reference-free normalized-time clustering with at most one point per demo."""

    observations = []
    for demo_index, trace in enumerate(traces):
        for boundary in trace.summary["candidate_boundaries"]:
            observations.append(
                {
                    "demo_index": demo_index,
                    "demonstration": trace.demonstration_name,
                    **boundary,
                }
            )
    clusters: list[list[dict[str, Any]]] = []
    for observation in sorted(observations, key=lambda item: item["normalized_time"]):
        eligible = []
        for index, cluster in enumerate(clusters):
            if any(item["demo_index"] == observation["demo_index"] for item in cluster):
                continue
            centre = float(np.mean([item["normalized_time"] for item in cluster]))
            distance = abs(observation["normalized_time"] - centre)
            if distance <= config.alignment_tolerance_normalized:
                eligible.append((distance, index))
        if eligible:
            clusters[min(eligible)[1]].append(observation)
        else:
            clusters.append([observation])

    minimum_support = max(
        int(ceil(config.minimum_alignment_support_fraction * len(traces))),
        1,
    )
    result = []
    for cluster in clusters:
        if len(cluster) < minimum_support:
            continue
        normalized = np.asarray([item["normalized_time"] for item in cluster])
        times = np.asarray([item["time_s"] for item in cluster])
        result.append(
            {
                "cluster": len(result),
                "support": len(cluster),
                "support_fraction": len(cluster) / len(traces),
                "mean_normalized_time": float(np.mean(normalized)),
                "std_normalized_time": float(np.std(normalized)),
                "mean_time_s": float(np.mean(times)),
                "std_time_s": float(np.std(times)),
                "members": [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "demo_index"
                    }
                    for item in sorted(cluster, key=lambda row: row["demonstration"])
                ],
            }
        )
    return sorted(result, key=lambda item: item["mean_normalized_time"])


def analyze_segmentation(
    demonstrations: list[Demonstration],
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[dict[str, Any], list[SegmentationTrace]]:
    """Run the complete training-only segmentation diagnostic."""

    linear_threshold, angular_threshold = calibrate_speed_thresholds(
        demonstrations,
        config,
    )
    traces = [
        segment_demonstration(
            demonstration,
            linear_threshold,
            angular_threshold,
            config,
        )
        for demonstration in demonstrations
    ]
    alignment = align_candidate_boundaries(traces, config)
    counts = np.asarray(
        [trace.summary["automatic_segment_count"] for trace in traces],
        dtype=np.float64,
    )
    deviations = [
        boundary["nearest_manual_boundary_absolute_deviation_s"]
        for trace in traces
        for boundary in trace.summary["candidate_boundaries"]
    ]
    result = {
        "method": "velocity_low_speed_diagnostic",
        "training_only": True,
        "replaces_training_segmentation": False,
        "config": asdict(config),
        "calibrated_thresholds": {
            "linear_speed_m_s": linear_threshold,
            "angular_speed_rad_s": angular_threshold,
        },
        "num_demonstrations": len(traces),
        "per_demonstration": [trace.summary for trace in traces],
        "aligned_boundaries": alignment,
        "consistency": {
            "automatic_segment_counts": [int(value) for value in counts],
            "mean_automatic_segment_count": float(np.mean(counts)),
            "std_automatic_segment_count": float(np.std(counts)),
            "fully_supported_boundary_clusters": sum(
                cluster["support"] == len(traces) for cluster in alignment
            ),
            "mean_aligned_boundary_time_std_s": float(
                np.mean([cluster["std_time_s"] for cluster in alignment])
            )
            if alignment
            else None,
            "mean_nearest_manual_boundary_absolute_deviation_s": float(
                np.mean(deviations)
            )
            if deviations
            else None,
            "max_nearest_manual_boundary_absolute_deviation_s": float(
                np.max(deviations)
            )
            if deviations
            else None,
        },
    }
    return result, traces
