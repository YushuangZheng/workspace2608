"""Environment-independent NumPy skill segmentation used by DynaMAC.

The source anchor is ``robot-learning-freiburg/TAPAS`` commit
``52e35214b9baa7b190b87196c36b9e98f4006149``, principally
``tapas_gmm.dataset.demos.Demos.segment`` and its distance-, gripper-, and
velocity-based boundary helpers.  The code here is an independently written,
data-only NumPy port. Inputs are normalized pose, task-frame, and gripper-state
trajectories; this module has no RLBench or simulator dependency. Environment
adapters remain responsible for extracting those canonical signals and choosing
task-specific protocols.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray
TAPAS_REFERENCE_COMMIT = "52e35214b9baa7b190b87196c36b9e98f4006149"
TAPAS_CONFIG_DEFAULTS_SOURCE_STATUS = (
    "TAPAS_HELPER_DEFAULTS_PLUS_AUTHOR_EMAIL_SIGNAL_UNION_20260814"
)
TAPAS_NUMPY_PORT_SOURCE_STATUS = "INDEPENDENT_CODE_ALIGNED_NUMPY_PORT"
TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS = (
    "AUTHOR_EMAIL_TASK_DEPENDENT_BIMANUAL_COORDINATION_20260814"
)
TAPAS_VELOCITY_THRESHOLD = 0.005
TAPAS_DISTANCE_THRESHOLD = 0.06
TAPAS_GRIPPER_THRESHOLD = 0.2
TAPAS_MAX_INDEX_DISTANCE = 4
TAPAS_MIN_CLUSTER_LENGTH = 1
TAPAS_MIN_END_DISTANCE = 10
@dataclass(frozen=True)
class TAPASSegmentationConfig:
    """Author-clarified defaults plus preserved TAPAS parameter provenance.

    ``from_mapping`` intentionally ignores no fields: known operational fields
    are parsed and all remaining metadata is retained in ``provenance``.  This
    allows evidence sidecars to travel into the run manifest without affecting
    numerical behavior.
    """

    min_len: int = TAPAS_MIN_CLUSTER_LENGTH
    distance_based: bool = False
    gripper_based: bool = True
    velocity_based: bool = True
    distance_threshold: float = TAPAS_DISTANCE_THRESHOLD
    repeat_first_step: int = 0
    repeat_final_step: int = 0
    fix_frames: bool = True
    min_end_distance: int = TAPAS_MIN_END_DISTANCE
    velocity_threshold: float = TAPAS_VELOCITY_THRESHOLD
    gripper_threshold: float = TAPAS_GRIPPER_THRESHOLD
    max_idx_distance: int = TAPAS_MAX_INDEX_DISTANCE
    gripper_min_end_distance: int | None = None
    candidate_merge_fraction: float = 0.0
    boundary_selection: str = "all"
    expected_boundary_count: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.distance_based and (self.gripper_based or self.velocity_based):
            raise ValueError("distance segmentation cannot be combined with velocity/gripper")
        if not any((self.distance_based, self.gripper_based, self.velocity_based)):
            raise ValueError("at least one TAPAS segmentation signal must be enabled")
        if self.min_len < 1:
            raise ValueError("min_len must be at least one")
        if self.repeat_first_step < 0 or self.repeat_final_step < 0:
            raise ValueError("repeat padding must be non-negative")
        if self.min_end_distance < 0 or self.max_idx_distance < 0:
            raise ValueError("index-distance parameters must be non-negative")
        if self.gripper_min_end_distance is not None and self.gripper_min_end_distance < 0:
            raise ValueError("gripper_min_end_distance must be non-negative or null")
        if not 0.0 <= self.candidate_merge_fraction <= 0.5:
            raise ValueError("candidate_merge_fraction must be in [0, 0.5]")
        if self.boundary_selection not in {
            "all",
            "gripper_preferred_temporal_consensus",
            "single_grasp_contact_cycle",
            "temporal_consensus",
            "temporal_consensus_require_gripper",
        }:
            raise ValueError("unsupported boundary_selection")
        for name, value in (("expected_boundary_count", self.expected_boundary_count),):
            if value is not None and (
                isinstance(value, bool) or int(value) != value or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null")
        thresholds = (
            self.distance_threshold,
            self.velocity_threshold,
            self.gripper_threshold,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("TAPAS segmentation thresholds must be finite and non-negative")

    @property
    def strategy(self) -> str:
        if self.distance_based:
            return "distance"
        if self.velocity_based and self.gripper_based:
            return "velocity_gripper_union"
        if self.gripper_based:
            return "gripper"
        return "velocity"

    @classmethod
    def velocity_defaults(cls, **overrides: Any) -> TAPASSegmentationConfig:
        """Return the public TAPAS velocity-segmentation parameterization."""

        return cls(
            distance_based=False,
            gripper_based=False,
            velocity_based=True,
            repeat_first_step=0,
            repeat_final_step=0,
            **overrides,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TAPASSegmentationConfig:
        """Parse operational fields while retaining arbitrary evidence fields."""

        if not isinstance(value, Mapping):
            raise TypeError("TAPAS segmentation config must be a mapping")
        raw = dict(value)
        reference = raw.get("reference_commit")
        if reference is not None and reference != TAPAS_REFERENCE_COMMIT:
            raise ValueError(
                f"segmentation config references {reference}, expected {TAPAS_REFERENCE_COMMIT}"
            )
        if "max_idx_dist" in raw and "max_idx_distance" not in raw:
            raw["max_idx_distance"] = raw["max_idx_dist"]

        operational = {
            "min_len",
            "distance_based",
            "gripper_based",
            "velocity_based",
            "distance_threshold",
            "repeat_first_step",
            "repeat_final_step",
            "fix_frames",
            "min_end_distance",
            "velocity_threshold",
            "gripper_threshold",
            "max_idx_distance",
            "gripper_min_end_distance",
            "candidate_merge_fraction",
            "boundary_selection",
            "expected_boundary_count",
        }
        kwargs = {name: raw[name] for name in operational if name in raw}
        strategy_fields = {"distance_based", "gripper_based", "velocity_based"}
        if strategy_fields.intersection(raw) and any(
            bool(raw.get(name, False)) for name in strategy_fields
        ):
            # A concise strategy mapping must not inherit other enabled class
            # defaults; callers that want the union set both flags explicitly.
            kwargs.update({name: bool(raw.get(name, False)) for name in strategy_fields})
        extras = {
            name: item
            for name, item in value.items()
            if name not in operational and name != "max_idx_dist"
        }
        kwargs["provenance"] = extras
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> TAPASSegmentationConfig:
        """Load a segmentation configuration from an explicit environment-owned path."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_len": self.min_len,
            "distance_based": self.distance_based,
            "gripper_based": self.gripper_based,
            "velocity_based": self.velocity_based,
            "distance_threshold": self.distance_threshold,
            "repeat_first_step": self.repeat_first_step,
            "repeat_final_step": self.repeat_final_step,
            "fix_frames": self.fix_frames,
            "min_end_distance": self.min_end_distance,
            "velocity_threshold": self.velocity_threshold,
            "gripper_threshold": self.gripper_threshold,
            "max_idx_distance": self.max_idx_distance,
            "gripper_min_end_distance": self.gripper_min_end_distance,
            "candidate_merge_fraction": self.candidate_merge_fraction,
            "boundary_selection": self.boundary_selection,
            "expected_boundary_count": self.expected_boundary_count,
            "provenance": dict(self.provenance),
        }

    def for_task(self, task_name: str) -> TAPASSegmentationConfig:
        """Apply one named profile supplied by an environment-owned config.

        Profiles only override numerical/post-processing fields; the core does
        not define which environment or task selects a particular profile.
        """

        profiles = self.provenance.get("task_profiles", {})
        if not isinstance(profiles, Mapping):
            raise ValueError("task_profiles must be a mapping")
        profile = profiles.get(task_name)
        if profile is None:
            return self
        if not isinstance(profile, Mapping):
            raise ValueError(f"task profile for {task_name} must be a mapping")
        operational = {
            "min_len",
            "distance_based",
            "gripper_based",
            "velocity_based",
            "distance_threshold",
            "repeat_first_step",
            "repeat_final_step",
            "fix_frames",
            "min_end_distance",
            "velocity_threshold",
            "gripper_threshold",
            "max_idx_distance",
            "gripper_min_end_distance",
            "candidate_merge_fraction",
            "boundary_selection",
            "expected_boundary_count",
        }
        unknown = set(profile).difference(operational | {"source_status", "note"})
        if unknown:
            raise ValueError(
                f"task profile for {task_name} has unknown fields: {sorted(unknown)}"
            )
        overrides = {name: profile[name] for name in operational if name in profile}
        provenance = {
            **dict(self.provenance),
            "active_task_profile": task_name,
            "active_task_profile_source_status": profile.get("source_status"),
            "active_task_profile_note": profile.get("note"),
        }
        return replace(self, **overrides, provenance=provenance)


def _pose_trajectory(value: Any, *, label: str) -> Array:
    pose = np.asarray(value, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7 or len(pose) < 1:
        raise ValueError(f"{label} must have shape [T, 7] with T >= 1")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(np.linalg.norm(pose[:, 3:7], axis=1) <= 0.0):
        raise ValueError(f"{label} contains a zero quaternion")
    return pose


def translation_action_magnitude(ee_pose: Any) -> Array:
    """Reconstruct TAPAS' forward translational action magnitude.

    TAPAS represents the action in the current EE frame.  Rotation preserves
    its Euclidean norm, so the norm of successive world translations is
    identical.  The final next-observation action is the repeated terminal pose
    and therefore has zero magnitude.
    """

    pose = _pose_trajectory(ee_pose, label="EE pose")
    magnitude = np.zeros(len(pose), dtype=np.float64)
    if len(pose) > 1:
        magnitude[:-1] = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    return magnitude


def tapas_velocity_boundaries(
    action_magnitude: Any,
    *,
    velocity_threshold: float = TAPAS_VELOCITY_THRESHOLD,
    max_idx_distance: int = TAPAS_MAX_INDEX_DISTANCE,
    min_cluster_len: int = TAPAS_MIN_CLUSTER_LENGTH,
    min_end_distance: int = TAPAS_MIN_END_DISTANCE,
) -> tuple[int, ...]:
    """Port TAPAS' velocity stop-cluster centers for one trajectory."""

    magnitude = np.asarray(action_magnitude, dtype=np.float64)
    if magnitude.ndim != 1 or len(magnitude) < 1:
        raise ValueError("action magnitude must be a non-empty vector")
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("action magnitude contains non-finite values")
    if velocity_threshold < 0.0 or not np.isfinite(velocity_threshold):
        raise ValueError("velocity_threshold must be finite and non-negative")
    if max_idx_distance < 0 or min_cluster_len < 1 or min_end_distance < 0:
        raise ValueError("invalid TAPAS velocity segmentation parameters")

    stop_indices = np.flatnonzero(np.abs(magnitude) < velocity_threshold)
    if not len(stop_indices):
        return ()
    splits = np.flatnonzero(np.diff(stop_indices) > max_idx_distance) + 1
    clusters = np.split(stop_indices, splits)
    # torch.mean(...).int() truncates these non-negative means toward zero.
    centers = tuple(
        int(np.mean(cluster)) for cluster in clusters if len(cluster) >= min_cluster_len
    )
    return tuple(
        center
        for center in centers
        if center > min_end_distance and center < len(magnitude) - min_end_distance
    )


def tapas_gripper_boundaries(
    gripper_state: Any,
    *,
    closed_threshold: float = TAPAS_GRIPPER_THRESHOLD,
    min_len: int = TAPAS_MIN_CLUSTER_LENGTH,
    max_idx_distance: int = TAPAS_MAX_INDEX_DISTANCE,
    min_end_distance: int = TAPAS_MIN_END_DISTANCE,
) -> tuple[int, ...]:
    """Port TAPAS' closed-gripper segment start/end boundary extraction."""

    state = np.asarray(gripper_state, dtype=np.float64)
    if state.ndim == 2 and state.shape[1] == 1:
        state = state[:, 0]
    if state.ndim != 1 or len(state) < 1 or not np.all(np.isfinite(state)):
        raise ValueError("gripper state must be a non-empty finite vector")
    closed = np.flatnonzero(state < closed_threshold)
    if not len(closed):
        return ()
    splits = np.flatnonzero(np.diff(closed) > max_idx_distance) + 1
    clusters = np.split(closed, splits)
    retained = [cluster for cluster in clusters if cluster[-1] - cluster[0] > min_len]
    candidates = tuple(int(cluster[0]) for cluster in retained) + tuple(
        int(cluster[-1]) for cluster in retained
    )
    return tuple(
        index
        for index in candidates
        if index > min_end_distance and index < len(state) - min_end_distance
    )


def gripper_change_boundaries(
    gripper_state: Any,
    *,
    change_threshold: float = TAPAS_GRIPPER_THRESHOLD,
    min_end_distance: int = TAPAS_MIN_END_DISTANCE,
) -> tuple[int, ...]:
    """Return sample indices where the measured gripper state changes.

    The pinned TAPAS closed-segment helper above remains available unchanged.
    DynaMAC's author clarification instead says to add *gripper changes* as
    candidate skill boundaries, so the corrected default detects transitions
    directly and uses the first sample of the new gripper state as the boundary.
    """

    state = np.asarray(gripper_state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    if state.ndim != 2 or len(state) < 1 or not np.all(np.isfinite(state)):
        raise ValueError("gripper state must have shape [T] or [T, D] and be finite")
    if change_threshold < 0.0 or not np.isfinite(change_threshold):
        raise ValueError("gripper change threshold must be finite and non-negative")
    if min_end_distance < 0:
        raise ValueError("min_end_distance must be non-negative")
    changed = np.any(np.abs(np.diff(state, axis=0)) > change_threshold, axis=1)
    candidates = np.flatnonzero(changed) + 1
    return tuple(
        int(index)
        for index in candidates
        if index > min_end_distance and index < len(state) - min_end_distance
    )


def _frame_trajectory(value: Any, *, steps: int, label: str) -> Array:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{label} has no task frames")
        frames = np.stack(
            [_pose_trajectory(item, label=f"{label}.{name}") for name, item in value.items()],
            axis=1,
        )
    else:
        frames = np.asarray(value, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] != steps or frames.shape[2] != 7:
        raise ValueError(
            f"{label} must have shape [T, F, 7] (or be a frame mapping), got {frames.shape}"
        )
    if frames.shape[1] < 1 or not np.all(np.isfinite(frames)):
        raise ValueError(f"{label} must contain at least one finite task frame")
    return frames


def tapas_distance_boundaries(
    ee_poses: Sequence[Any],
    frame_trajectories: Sequence[Any],
    *,
    distance_threshold: float = TAPAS_DISTANCE_THRESHOLD,
    min_end_distance: int = TAPAS_MIN_END_DISTANCE,
    fix_frames: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Port TAPAS' current static-frame distance segmentation.

    TAPAS assumes that a retained frame is approached once and is approached in
    every demonstration.  This port raises a descriptive ``ValueError`` where
    the upstream implementation would reach an assertion or fail while sorting
    ``None``; it does not invent or impute a boundary.
    """

    if len(ee_poses) != len(frame_trajectories) or not ee_poses:
        raise ValueError("EE and frame trajectory collections must be non-empty and paired")
    poses = [
        _pose_trajectory(value, label=f"EE pose trajectory {index}")
        for index, value in enumerate(ee_poses)
    ]
    frames = [
        _frame_trajectory(value, steps=len(pose), label=f"frame trajectory {index}")
        for index, (value, pose) in enumerate(zip(frame_trajectories, poses, strict=True))
    ]
    frame_count = frames[0].shape[1]
    if any(item.shape[1] != frame_count for item in frames):
        raise ValueError("all demonstrations must have the same ordered task frames")
    if distance_threshold < 0.0 or not np.isfinite(distance_threshold):
        raise ValueError("distance_threshold must be finite and non-negative")

    per_frame_means: list[tuple[int, ...] | None] = []
    for frame_index in range(frame_count):
        means: list[int | None] = []
        for pose, trajectory in zip(poses, frames, strict=True):
            reference = (
                trajectory[0:1, frame_index, :3] if fix_frames else trajectory[:, frame_index, :3]
            )
            distance = np.linalg.norm(pose[:, :3] - reference, axis=1)
            indices = np.flatnonzero(distance < distance_threshold)
            indices = indices[
                (indices > min_end_distance) & (indices < len(pose) - min_end_distance)
            ]
            means.append(int(np.mean(indices)) if len(indices) else None)
        if means[0] is None:
            if any(item is not None for item in means):
                raise ValueError(
                    "TAPAS distance segmentation requires an omitted frame to be "
                    "omitted in every demonstration"
                )
            per_frame_means.append(None)
        else:
            if any(item is None for item in means):
                raise ValueError(
                    "TAPAS distance segmentation requires each retained frame to be "
                    "approached in every demonstration"
                )
            per_frame_means.append(tuple(int(item) for item in means if item is not None))

    boundaries: list[tuple[int, ...]] = []
    for demo_index in range(len(poses)):
        values = [frame[demo_index] for frame in per_frame_means if frame is not None]
        boundaries.append(tuple(sorted(values)))
    return tuple(boundaries)


@dataclass(frozen=True)
class TAPASSegmentation:
    """Boundaries, half-open skill slices, and machine-readable evidence."""

    raw_boundaries: tuple[tuple[int, ...], ...]
    boundaries: tuple[tuple[int, ...], ...]
    skill_bounds: tuple[tuple[tuple[int, int], ...], ...]
    strategy: str = "velocity"
    alignment_method: str = "equal_boundary_count_fail_closed_no_truncation"
    config: TAPASSegmentationConfig | None = None
    boundary_components: Mapping[str, tuple[tuple[int, ...], ...]] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        count = len(self.raw_boundaries)
        if count < 1 or len(self.boundaries) != count or len(self.skill_bounds) != count:
            raise ValueError("TAPAS segmentation collections must be non-empty and paired")
        skill_counts = {len(item) for item in self.skill_bounds}
        if len(skill_counts) != 1:
            raise ValueError("TAPAS demonstrations must have a shared skill count")
        for index, (raw, aligned, bounds) in enumerate(
            zip(self.raw_boundaries, self.boundaries, self.skill_bounds, strict=True)
        ):
            if tuple(sorted(raw)) != raw or tuple(sorted(aligned)) != aligned:
                raise ValueError(f"demonstration {index} boundaries are not sorted")
            if aligned != raw:
                raise ValueError(
                    f"demonstration {index} boundaries may not be truncated or reordered"
                )
            if any(stop <= start for start, stop in bounds):
                raise ValueError(f"demonstration {index} contains an empty skill slice")
            starts = tuple(start for start, _ in bounds)
            stops = tuple(stop for _, stop in bounds)
            if starts[0] != 0 or starts[1:] != stops[:-1]:
                raise ValueError(f"demonstration {index} skill slices are not contiguous")
            if aligned != stops[:-1]:
                raise ValueError(f"demonstration {index} boundaries and slices disagree")
        for name, rows in self.boundary_components.items():
            if not name or len(rows) != count:
                raise ValueError("boundary components must be named and paired by demonstration")

    @property
    def skill_count(self) -> int:
        return len(self.skill_bounds[0])

    @property
    def trajectory_lengths(self) -> tuple[int, ...]:
        return tuple(bounds[-1][1] for bounds in self.skill_bounds)

    @property
    def skill_labels(self) -> tuple[Array, ...]:
        """Return positional integer labels; no semantic skill names are claimed."""

        labels: list[Array] = []
        for bounds in self.skill_bounds:
            values = np.full(bounds[-1][1], -1, dtype=np.int64)
            for label, (start, stop) in enumerate(bounds):
                values[start:stop] = label
            if np.any(values < 0):
                raise AssertionError("internal TAPAS bounds do not cover the trajectory")
            labels.append(values)
        return tuple(labels)

    def labels_for(self, demonstration_index: int) -> Array:
        return self.skill_labels[demonstration_index].copy()

    @property
    def audit(self) -> dict[str, Any]:
        """Return JSON-ready boundary and label provenance."""

        raw_counts = [len(item) for item in self.raw_boundaries]
        aligned_counts = [len(item) for item in self.boundaries]
        records: list[dict[str, Any]] = []
        label_records: list[dict[str, Any]] = []
        for demo_index, (raw, aligned, bounds) in enumerate(
            zip(self.raw_boundaries, self.boundaries, self.skill_bounds, strict=True)
        ):
            for raw_rank, sample_index in enumerate(raw):
                records.append(
                    {
                        "demonstration_index": demo_index,
                        "raw_boundary_rank": raw_rank,
                        "sample_index": sample_index,
                        "retained": True,
                        "aligned_boundary_rank": raw_rank,
                        "label_before": raw_rank,
                        "label_after": raw_rank + 1,
                        "status": "RETAINED_BY_FAIL_CLOSED_ALIGNMENT",
                    }
                )
            for label, (start, stop) in enumerate(bounds):
                label_records.append(
                    {
                        "demonstration_index": demo_index,
                        "label": label,
                        "start_inclusive": start,
                        "stop_exclusive": stop,
                        "source": "POSITIONAL_LABEL_FROM_ALIGNED_BOUNDARY_ORDER",
                        "semantic_skill_name": None,
                    }
                )
        return {
            "schema_version": 2,
            "strategy": self.strategy,
            "reference_commit": TAPAS_REFERENCE_COMMIT,
            "algorithm_source_status": TAPAS_NUMPY_PORT_SOURCE_STATUS,
            "alignment_method": self.alignment_method,
            "raw_boundaries": [list(item) for item in self.raw_boundaries],
            "aligned_boundaries": [list(item) for item in self.boundaries],
            "skill_bounds": [[[start, stop] for start, stop in item] for item in self.skill_bounds],
            "raw_boundary_counts": raw_counts,
            "aligned_boundary_counts": aligned_counts,
            "boundary_components": {
                name: [list(row) for row in rows]
                for name, rows in self.boundary_components.items()
            },
            "candidate_merge_fraction": (
                self.config.candidate_merge_fraction if self.config is not None else 0.0
            ),
            "boundary_selection": (
                self.config.boundary_selection if self.config is not None else "all"
            ),
            "expected_boundary_count": (
                self.config.expected_boundary_count if self.config is not None else None
            ),
            "candidate_filtering_is_rank_truncation": False,
            "minimum_count_truncation_applied": False,
            "no_boundary_truncation": True,
            "cross_demo_count_policy": "FAIL_CLOSED_ON_UNEQUAL_BOUNDARY_COUNTS",
            "semantic_alignment_risk": False,
            "boundary_records": records,
            "label_records": label_records,
            "semantic_labels_used": False,
            "repeat_padding_applied_to_labels": False,
            "config": self.config.to_dict() if self.config is not None else None,
            "claim_boundary": (
                "Velocity/gripper candidates follow the pinned TAPAS helpers. Declared "
                "task profiles merge nearby signals and, where configured, choose an "
                "explicit deterministic subset before fail-closed alignment; they never "
                "drop a suffix by rank. Positional labels remain adapter outputs."
            ),
        }


@dataclass(frozen=True)
class BimanualTAPASSegmentation:
    """Task-coordinated applications of the TAPAS single-arm boundary signals."""

    left: TAPASSegmentation
    right: TAPASSegmentation
    coordination: str = "independent"
    coordination_source_status: str = TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS
    debug_plots_required: bool = False
    pre_coordination_left_boundaries: tuple[tuple[int, ...], ...] | None = None
    pre_coordination_right_boundaries: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.left.skill_bounds) != len(self.right.skill_bounds):
            raise ValueError("left/right segmentations must contain the same demonstrations")
        if self.left.trajectory_lengths != self.right.trajectory_lengths:
            raise ValueError("left/right trajectories must remain sample-aligned")
        if self.coordination not in {"independent", "shared_union"}:
            raise ValueError("bimanual coordination must be independent or shared_union")
        if self.coordination == "shared_union" and (
            self.left.boundaries != self.right.boundaries
            or self.left.skill_bounds != self.right.skill_bounds
        ):
            raise ValueError("shared_union must apply identical boundaries to both arms")
        paired = (
            self.pre_coordination_left_boundaries,
            self.pre_coordination_right_boundaries,
        )
        if (paired[0] is None) != (paired[1] is None):
            raise ValueError("pre-coordination boundary evidence must be paired")
        if paired[0] is not None and (
            len(paired[0]) != len(self.left.skill_bounds)
            or len(paired[1]) != len(self.right.skill_bounds)
        ):
            raise ValueError("pre-coordination boundary evidence has the wrong demo count")

    @property
    def audit(self) -> dict[str, Any]:
        # Keep the v1 claim text byte-for-byte stable for manifest verification.
        # It is inert provenance: task/profile selection remains adapter-owned.
        return {
            "schema_version": 2,
            "application_source_status": self.coordination_source_status,
            "coordination": self.coordination,
            "shared_boundaries_between_arms": self.coordination == "shared_union",
            "debug_plots_required": self.debug_plots_required,
            "pre_coordination_boundaries": {
                "left": (
                    [list(row) for row in self.pre_coordination_left_boundaries]
                    if self.pre_coordination_left_boundaries is not None
                    else None
                ),
                "right": (
                    [list(row) for row in self.pre_coordination_right_boundaries]
                    if self.pre_coordination_right_boundaries is not None
                    else None
                ),
            },
            "left": self.left.audit,
            "right": self.right.audit,
            "claim_boundary": (
                "StoreBottle independent-arm and HandOver shared-union coordination follow "
                "the 2026-08-14 author clarification. LiftTray/SweepDust shared-union is "
                "a local interaction inference and requires debug plots."
            ),
        }


def align_tapas_boundaries(
    raw_boundaries: Sequence[Sequence[int]],
    trajectory_lengths: Sequence[int],
    *,
    strategy: str = "velocity",
    truncate_to_minimum: bool = False,
    config: TAPASSegmentationConfig | None = None,
    boundary_components: Mapping[str, Sequence[Sequence[int]]] | None = None,
) -> TAPASSegmentation:
    """Build half-open slices, failing closed rather than truncating boundaries.

    ``truncate_to_minimum`` remains in the signature for legacy callers, but no
    value authorizes truncation.  Unequal counts always raise because an extra
    boundary may have occurred in the middle rather than at the trajectory end.
    """

    if len(raw_boundaries) != len(trajectory_lengths) or not raw_boundaries:
        raise ValueError("boundaries and trajectory lengths must be non-empty and paired")
    normalized: list[tuple[int, ...]] = []
    for index, (items, length) in enumerate(zip(raw_boundaries, trajectory_lengths, strict=True)):
        if isinstance(length, bool) or int(length) != length or length < 1:
            raise ValueError(f"trajectory {index} length must be a positive integer")
        values = tuple(int(item) for item in items)
        if any(item != original for item, original in zip(values, items, strict=True)):
            raise ValueError(f"trajectory {index} boundaries must be integers")
        if any(value <= 0 or value >= length for value in values):
            raise ValueError(f"trajectory {index} has an out-of-range boundary")
        if any(current <= previous for previous, current in zip(values, values[1:])):
            raise ValueError(f"trajectory {index} boundaries must be strictly increasing")
        normalized.append(values)

    counts = {len(item) for item in normalized}
    if len(counts) != 1:
        raise ValueError(
            "TAPAS demonstrations have unequal boundary counts; author clarification "
            "forbids prefix/min-count truncation"
        )
    aligned = normalized.copy()
    alignment_method = "equal_boundary_count_fail_closed_no_truncation"

    normalized_components: dict[str, tuple[tuple[int, ...], ...]] = {}
    for name, rows in (boundary_components or {}).items():
        if not isinstance(name, str) or not name or len(rows) != len(normalized):
            raise ValueError("boundary components must be named and paired by demonstration")
        normalized_components[name] = tuple(tuple(int(value) for value in row) for row in rows)

    skill_bounds: list[tuple[tuple[int, int], ...]] = []
    for length, boundaries in zip(trajectory_lengths, aligned, strict=True):
        starts = (0, *boundaries)
        stops = (*boundaries, int(length))
        bounds = tuple(zip(starts, stops, strict=True))
        if any(stop <= start for start, stop in bounds):
            raise ValueError("TAPAS boundaries produced an empty skill slice")
        skill_bounds.append(bounds)
    return TAPASSegmentation(
        raw_boundaries=tuple(normalized),
        boundaries=tuple(aligned),
        skill_bounds=tuple(skill_bounds),
        strategy=strategy,
        alignment_method=alignment_method,
        config=config,
        boundary_components=normalized_components,
    )


def _coerce_config(
    config: TAPASSegmentationConfig | Mapping[str, Any] | None,
) -> TAPASSegmentationConfig:
    if config is None:
        return TAPASSegmentationConfig()
    if isinstance(config, TAPASSegmentationConfig):
        return config
    return TAPASSegmentationConfig.from_mapping(config)


def _raw_segment_boundaries(
    poses: Sequence[Array],
    *,
    frame_trajectories: Sequence[Any] | None,
    gripper_states: Sequence[Any] | None,
    config: TAPASSegmentationConfig,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, tuple[tuple[int, ...], ...]]]:
    """Compute boundary candidates without aligning or dropping any of them."""

    if config.distance_based:
        if frame_trajectories is None:
            raise ValueError("distance segmentation requires task-frame trajectories")
        raw = tapas_distance_boundaries(
            poses,
            frame_trajectories,
            distance_threshold=config.distance_threshold,
            min_end_distance=config.min_end_distance,
            fix_frames=config.fix_frames,
        )
        return raw, {"distance": raw}

    components: dict[str, tuple[tuple[int, ...], ...]] = {}
    if config.velocity_based:
        components["ee_translation_velocity"] = tuple(
            tapas_velocity_boundaries(
                translation_action_magnitude(item),
                velocity_threshold=config.velocity_threshold,
                max_idx_distance=config.max_idx_distance,
                min_cluster_len=config.min_len,
                min_end_distance=config.min_end_distance,
            )
            for item in poses
        )
    if config.gripper_based:
        if gripper_states is None or len(gripper_states) != len(poses):
            raise ValueError("gripper segmentation requires one state trajectory per demo")
        components["gripper_change"] = tuple(
            gripper_change_boundaries(
                item,
                change_threshold=config.gripper_threshold,
                min_end_distance=(
                    config.min_end_distance
                    if config.gripper_min_end_distance is None
                    else config.gripper_min_end_distance
                ),
            )
            for item in gripper_states
        )

    raw = tuple(
        tuple(
            sorted(
                {
                    boundary
                    for rows in components.values()
                    for boundary in rows[demo_index]
                }
            )
        )
        for demo_index in range(len(poses))
    )
    return raw, components


def _merge_candidate_rows(
    rows: Sequence[Sequence[int]],
    lengths: Sequence[int],
    *,
    fraction: float,
    preferred_rows: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Merge temporally adjacent candidates without rank/tail truncation.

    A gripper transition wins when it shares a neighborhood with a velocity
    stop; otherwise the integer median is used.  The threshold is relative to
    each demonstration length so the same profile works before TAPAS' common-
    length subsampling.
    """

    if len(rows) != len(lengths):
        raise ValueError("candidate rows and lengths must be paired")
    if preferred_rows is None:
        preferred_rows = [()] * len(rows)
    if len(preferred_rows) != len(rows):
        raise ValueError("preferred candidate rows must be paired")
    merged: list[tuple[int, ...]] = []
    for row, length, preferred in zip(rows, lengths, preferred_rows, strict=True):
        values = sorted(set(int(value) for value in row))
        radius = int(round(fraction * int(length)))
        groups: list[list[int]] = []
        for value in values:
            if not groups or value - groups[-1][-1] > radius:
                groups.append([value])
            else:
                groups[-1].append(value)
        preferred_set = set(int(value) for value in preferred)
        selected: list[int] = []
        for group in groups:
            preferred_group = [value for value in group if value in preferred_set]
            source = preferred_group or group
            selected.append(int(np.median(source)))
        merged.append(tuple(selected))
    return tuple(merged)


def _best_temporal_subset(
    row: Sequence[int],
    *,
    length: int,
    anchors: Array,
    required: Sequence[int] = (),
) -> tuple[int, ...]:
    """Choose an ordered candidate subset closest to normalized anchors."""

    from itertools import combinations

    count = len(anchors)
    values = tuple(sorted(set(int(value) for value in row)))
    required_set = set(int(value) for value in required)
    if len(values) < count or len(required_set) > count:
        raise ValueError(
            f"need {count} boundaries but only have {len(values)} candidates "
            f"({len(required_set)} required)"
        )
    choices = [item for item in combinations(values, count) if required_set.issubset(item)]
    if not choices:
        raise ValueError("no ordered temporal subset retains every required gripper change")
    return min(
        choices,
        key=lambda item: float(
            np.sum((np.asarray(item, dtype=np.float64) / float(length) - anchors) ** 2)
        ),
    )


def _single_grasp_contact_cycle_subset(
    row: Sequence[int],
    *,
    gripper_row: Sequence[int],
    expected: int,
) -> tuple[int, ...]:
    """Align a pick/place cycle by contact phase instead of boundary rank.

    The common four-boundary cycle is: stop above the object, close the
    gripper, finish the post-grasp lift, and reach the final pre-release stop.
    Extra velocity stops inside the long transport phase are discarded.  This
    selector intentionally fails closed for any other contact pattern; it is a
    declared local task-profile rule, not an inferred author default.
    """

    if expected != 4:
        raise ValueError(
            "single_grasp_contact_cycle requires exactly four boundaries"
        )
    values = tuple(sorted(set(int(value) for value in row)))
    contacts = tuple(sorted(set(int(value) for value in gripper_row)))
    if len(contacts) != 1:
        raise ValueError(
            "single_grasp_contact_cycle requires exactly one retained gripper "
            f"change, got {contacts}"
        )
    contact = contacts[0]
    if contact not in values:
        raise ValueError(
            "the retained gripper change must be present after candidate merging"
        )
    before = tuple(value for value in values if value < contact)
    after = tuple(value for value in values if value > contact)
    if not before or len(after) < 2:
        raise ValueError(
            "single_grasp_contact_cycle needs an approach stop before contact and "
            f"at least two velocity stops after contact, got {values}"
        )
    selected = (before[-1], contact, after[0], after[-1])
    if len(set(selected)) != expected:
        raise ValueError(
            f"single_grasp_contact_cycle produced non-distinct boundaries: {selected}"
        )
    return selected


def _select_consistent_boundaries(
    rows: Sequence[Sequence[int]],
    lengths: Sequence[int],
    *,
    config: TAPASSegmentationConfig,
    gripper_rows: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Apply one declared, deterministic cohort-level candidate selector."""

    expected = config.expected_boundary_count
    counts = [len(row) for row in rows]
    if expected is None:
        if len(set(counts)) != 1:
            raise ValueError(
                "unequal boundary counts after candidate merging; author clarification "
                "forbids prefix/min-count truncation. Configure an explicit task profile "
                f"or inspect debug plots (counts={counts}, rows={list(rows)})"
            )
        return tuple(tuple(row) for row in rows)
    if config.boundary_selection == "all":
        if any(count != expected for count in counts):
            raise ValueError(
                f"task profile expects {expected} boundaries after merging, got "
                f"counts={counts}, rows={list(rows)}"
            )
        return tuple(tuple(row) for row in rows)

    if config.boundary_selection == "single_grasp_contact_cycle":
        if gripper_rows is None or len(gripper_rows) != len(rows):
            raise ValueError(
                "single-grasp contact alignment requires paired gripper candidates"
            )
        return tuple(
            _single_grasp_contact_cycle_subset(
                row,
                gripper_row=gripper,
                expected=expected,
            )
            for row, gripper in zip(rows, gripper_rows, strict=True)
        )

    if config.boundary_selection == "gripper_preferred_temporal_consensus":
        if gripper_rows is None or len(gripper_rows) != len(rows):
            raise ValueError("gripper-preferred selection requires paired gripper candidates")
        reference_rows = [
            tuple(row) for row in gripper_rows if len(tuple(row)) == expected
        ]
        if not reference_rows:
            raise ValueError(
                "cannot estimate attachment-guided anchors: no demonstration has the "
                f"expected {expected} gripper changes"
            )
        anchors = np.median(
            np.asarray(
                [
                    np.asarray(row, dtype=np.float64) / float(length)
                    for row, length in zip(gripper_rows, lengths, strict=True)
                    if len(tuple(row)) == expected
                ]
            ),
            axis=0,
        )
        return tuple(
            tuple(gripper)
            if len(tuple(gripper)) == expected
            else _best_temporal_subset(
                row,
                length=int(length),
                anchors=anchors,
                required=gripper,
            )
            for row, gripper, length in zip(rows, gripper_rows, lengths, strict=True)
        )

    reference = [
        np.asarray(row, dtype=np.float64) / float(length)
        for row, length in zip(rows, lengths, strict=True)
        if len(row) == expected
    ]
    if not reference:
        raise ValueError(
            f"cannot estimate temporal anchors: no row has expected count {expected}"
        )
    anchors = np.median(np.stack(reference), axis=0)
    require_gripper = config.boundary_selection == "temporal_consensus_require_gripper"
    if require_gripper and (gripper_rows is None or len(gripper_rows) != len(rows)):
        raise ValueError("gripper-required selection requires paired gripper candidates")
    selected = []
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        required = tuple(gripper_rows[index]) if require_gripper else ()
        if len(row) == expected and set(required).issubset(row):
            selected.append(tuple(row))
        else:
            selected.append(
                _best_temporal_subset(
                    row,
                    length=int(length),
                    anchors=anchors,
                    required=required,
                )
            )
    return tuple(selected)


def _postprocess_boundaries(
    raw: Sequence[Sequence[int]],
    lengths: Sequence[int],
    *,
    config: TAPASSegmentationConfig,
    gripper_rows: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    merged = _merge_candidate_rows(
        raw,
        lengths,
        fraction=config.candidate_merge_fraction,
        preferred_rows=gripper_rows,
    )
    return _select_consistent_boundaries(
        merged,
        lengths,
        config=config,
        gripper_rows=gripper_rows,
    )


def segment_trajectories(
    ee_poses: Sequence[Any],
    *,
    frame_trajectories: Sequence[Any] | None = None,
    gripper_states: Sequence[Any] | None = None,
    config: TAPASSegmentationConfig | Mapping[str, Any] | None = None,
) -> TAPASSegmentation:
    """Segment one arm from velocity/gripper candidates without count truncation."""

    cfg = _coerce_config(config)
    if not ee_poses:
        raise ValueError("at least one EE trajectory is required")
    poses = [
        _pose_trajectory(item, label=f"EE pose trajectory {index}")
        for index, item in enumerate(ee_poses)
    ]
    lengths = [len(item) for item in poses]
    raw, components = _raw_segment_boundaries(
        poses,
        frame_trajectories=frame_trajectories,
        gripper_states=gripper_states,
        config=cfg,
    )
    raw = _postprocess_boundaries(
        raw,
        lengths,
        config=cfg,
        gripper_rows=components.get("gripper_change"),
    )
    return align_tapas_boundaries(
        raw,
        lengths,
        strategy=cfg.strategy,
        truncate_to_minimum=False,
        config=cfg,
        boundary_components=components,
    )


def segment_bimanual_trajectories(
    left_ee_poses: Sequence[Any],
    right_ee_poses: Sequence[Any],
    *,
    frame_trajectories: Sequence[Any] | None = None,
    left_gripper_states: Sequence[Any] | None = None,
    right_gripper_states: Sequence[Any] | None = None,
    config: TAPASSegmentationConfig | Mapping[str, Any] | None = None,
    coordination: str = "independent",
    coordination_source_status: str = TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS,
    debug_plots_required: bool = False,
) -> BimanualTAPASSegmentation:
    """Apply task-level independent or shared-union bimanual segmentation."""

    if len(left_ee_poses) != len(right_ee_poses) or not left_ee_poses:
        raise ValueError("left/right EE trajectory collections must be non-empty and paired")
    left_poses = [
        _pose_trajectory(item, label=f"left EE pose trajectory {index}")
        for index, item in enumerate(left_ee_poses)
    ]
    right_poses = [
        _pose_trajectory(item, label=f"right EE pose trajectory {index}")
        for index, item in enumerate(right_ee_poses)
    ]
    for index, (left, right) in enumerate(zip(left_poses, right_poses, strict=True)):
        if len(left) != len(right):
            raise ValueError(f"demonstration {index} left/right trajectories are not aligned")
    if coordination not in {"independent", "shared_union"}:
        raise ValueError("bimanual coordination must be independent or shared_union")
    cfg = _coerce_config(config)
    lengths = [len(item) for item in left_poses]
    left_raw, left_components = _raw_segment_boundaries(
        left_poses,
        frame_trajectories=frame_trajectories,
        gripper_states=left_gripper_states,
        config=cfg,
    )
    right_raw, right_components = _raw_segment_boundaries(
        right_poses,
        frame_trajectories=frame_trajectories,
        gripper_states=right_gripper_states,
        config=cfg,
    )
    left_gripper = left_components.get("gripper_change")
    right_gripper = right_components.get("gripper_change")
    left_candidates = _merge_candidate_rows(
        left_raw,
        lengths,
        fraction=cfg.candidate_merge_fraction,
        preferred_rows=left_gripper,
    )
    right_candidates = _merge_candidate_rows(
        right_raw,
        lengths,
        fraction=cfg.candidate_merge_fraction,
        preferred_rows=right_gripper,
    )
    if coordination == "shared_union":
        coordinated = tuple(
            tuple(sorted(set(left_row).union(right_row)))
            for left_row, right_row in zip(left_candidates, right_candidates, strict=True)
        )
        joint_gripper = (
            tuple(
                tuple(sorted(set(left_row).union(right_row)))
                for left_row, right_row in zip(left_gripper, right_gripper, strict=True)
            )
            if left_gripper is not None and right_gripper is not None
            else None
        )
        coordinated = _postprocess_boundaries(
            coordinated,
            lengths,
            config=cfg,
            gripper_rows=joint_gripper,
        )
        joint_components = {
            **{f"left_{name}": rows for name, rows in left_components.items()},
            **{f"right_{name}": rows for name, rows in right_components.items()},
        }
        left = align_tapas_boundaries(
            coordinated,
            lengths,
            strategy=f"{cfg.strategy}_bimanual_shared_union",
            config=cfg,
            boundary_components=joint_components,
        )
        right = align_tapas_boundaries(
            coordinated,
            lengths,
            strategy=f"{cfg.strategy}_bimanual_shared_union",
            config=cfg,
            boundary_components=joint_components,
        )
    else:
        left_selected = _select_consistent_boundaries(
            left_candidates,
            lengths,
            config=cfg,
            gripper_rows=left_gripper,
        )
        right_selected = _select_consistent_boundaries(
            right_candidates,
            lengths,
            config=cfg,
            gripper_rows=right_gripper,
        )
        left = align_tapas_boundaries(
            left_selected,
            lengths,
            strategy=cfg.strategy,
            config=cfg,
            boundary_components=left_components,
        )
        right = align_tapas_boundaries(
            right_selected,
            lengths,
            strategy=cfg.strategy,
            config=cfg,
            boundary_components=right_components,
        )
    return BimanualTAPASSegmentation(
        left=left,
        right=right,
        coordination=coordination,
        coordination_source_status=coordination_source_status,
        debug_plots_required=debug_plots_required,
        pre_coordination_left_boundaries=left_candidates,
        pre_coordination_right_boundaries=right_candidates,
    )


# Compatibility aliases emphasize that pose trajectories, rather than simulator
# objects, are the only required input for velocity segmentation.
segment_pose_trajectories = segment_trajectories
segment_bimanual_pose_trajectories = segment_bimanual_trajectories


__all__ = [
    "BimanualTAPASSegmentation",
    "TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS",
    "TAPAS_CONFIG_DEFAULTS_SOURCE_STATUS",
    "TAPAS_DISTANCE_THRESHOLD",
    "TAPAS_GRIPPER_THRESHOLD",
    "TAPAS_MAX_INDEX_DISTANCE",
    "TAPAS_MIN_CLUSTER_LENGTH",
    "TAPAS_MIN_END_DISTANCE",
    "TAPAS_NUMPY_PORT_SOURCE_STATUS",
    "TAPAS_REFERENCE_COMMIT",
    "TAPAS_VELOCITY_THRESHOLD",
    "TAPASSegmentation",
    "TAPASSegmentationConfig",
    "align_tapas_boundaries",
    "gripper_change_boundaries",
    "segment_bimanual_pose_trajectories",
    "segment_bimanual_trajectories",
    "segment_pose_trajectories",
    "segment_trajectories",
    "tapas_distance_boundaries",
    "tapas_gripper_boundaries",
    "tapas_velocity_boundaries",
    "translation_action_magnitude",
]
