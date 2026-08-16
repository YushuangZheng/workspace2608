"""Audited low-dimensional task-frame schemas for the pinned RLBench fork.

The schemas in this module are transcriptions of ``get_low_dim_state()`` in
``vonHartz/RLBench:tapas@a51b4e609dc5c3e1a8c06046bd87a9da24723da4``.
They name and split the concatenated RLBench poses without importing RLBench or
PyRep.  The 2026-08-14 author clarification confirms that every frame returned
by ``get_low_dim_state`` is a candidate, in source order.  The same clarification
defines StoreBottle/HandOver arm coordination; inferred choices remain labelled.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Array = np.ndarray
RLBenchPoseConvention = "world [x, y, z, qx, qy, qz, qw]"
CorePoseConvention = "world [x, y, z, qw, qx, qy, qz]"
RLBENCH_REFERENCE_COMMIT = "a51b4e609dc5c3e1a8c06046bd87a9da24723da4"
TASK_SCHEMA_SOURCE_STATUS = "EXACT_FROM_PUBLIC_RLBENCH_GET_LOW_DIM_STATE"
CANDIDATE_FRAME_POLICY = "ALL_GET_LOW_DIM_STATE_FRAMES_IN_SOURCE_ORDER"
CANDIDATE_FRAME_POLICY_SOURCE_STATUS = "AUTHOR_EMAIL_EXPLICIT_20260814"
TASK_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tasks.json"

SegmentationCoordination = Literal["single", "independent", "shared_union"]


def _pose_array(value: Any, *, convention: str) -> Array:
    """Return a finite float64 pose array whose final dimension is seven."""

    pose = np.asarray(value, dtype=np.float64)
    if pose.ndim < 1 or pose.shape[-1] != 7:
        raise ValueError(f"{convention} pose must end in 7 values, got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{convention} pose contains non-finite values")
    quaternion_norm = np.linalg.norm(pose[..., 3:7], axis=-1)
    if np.any(quaternion_norm <= 0.0):
        raise ValueError(f"{convention} pose contains a zero quaternion")
    return pose


def xyzw_to_wxyz(pose: Any) -> Array:
    """Convert one or more RLBench ``xyzw`` poses to core ``wxyz`` order.

    Only component order changes.  Quaternion magnitude and sign are preserved
    so the conversion is lossless and provenance hashes remain meaningful.
    """

    value = _pose_array(pose, convention="RLBench xyzw")
    return np.concatenate((value[..., :3], value[..., 6:7], value[..., 3:6]), axis=-1)


def wxyz_to_xyzw(pose: Any) -> Array:
    """Convert one or more core ``wxyz`` poses to RLBench ``xyzw`` order."""

    value = _pose_array(pose, convention="core wxyz")
    return np.concatenate((value[..., :3], value[..., 4:7], value[..., 3:4]), axis=-1)


# Descriptive aliases make convention crossings visible at call sites.
rlbench_pose_xyzw_to_core_wxyz = xyzw_to_wxyz
core_pose_wxyz_to_rlbench_xyzw = wxyz_to_xyzw


def unwrap_task_low_dim_state(value: Any) -> Array:
    """Unwrap and validate the fork's accidental ``(task_state,)`` value.

    In the pinned fork, ``Scene.get_observation`` assigns
    ``task_low_dim_state = (...),`` (note the trailing comma).  Saved
    observations therefore contain a single-element tuple rather than the
    array returned by the task.  This function accepts either representation,
    but rejects wider/nested tuples instead of silently flattening them.
    """

    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError("RLBench task_low_dim_state tuple must contain exactly one value")
        value = value[0]
        if isinstance(value, tuple):
            raise ValueError("nested task_low_dim_state tuples are not supported")
    state = np.asarray(value, dtype=np.float64)
    if state.ndim == 2 and state.shape[0] == 1:
        # Some downstream serializers materialize the tuple bug as [1, N].
        state = state[0]
    if state.ndim != 1 or state.size == 0:
        raise ValueError(f"task_low_dim_state must be a non-empty flat vector, got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("task_low_dim_state contains non-finite values")
    return state.copy()


@dataclass(frozen=True)
class TaskPoseChunk:
    """One seven-value pose in a task's public low-dimensional state."""

    name: str
    role: str
    index: int

    @property
    def start(self) -> int:
        return 7 * self.index

    @property
    def stop(self) -> int:
        return self.start + 7

    @property
    def source_slice(self) -> tuple[int, int]:
        return self.start, self.stop


@dataclass(frozen=True)
class TaskSpec:
    """Data-only specification for one paper-relevant RLBench task."""

    task_name: str
    paper_task_name: str
    module: str
    class_name: str
    bimanual: bool
    paper_evaluation_group: str
    paper_scenarios: tuple[str, ...]
    pose_chunks: tuple[TaskPoseChunk, ...]
    source_expression: str
    source_status: str = TASK_SCHEMA_SOURCE_STATUS
    candidate_frame_policy: str = CANDIDATE_FRAME_POLICY
    candidate_frame_policy_source_status: str = CANDIDATE_FRAME_POLICY_SOURCE_STATUS
    segmentation_coordination: SegmentationCoordination = "single"
    segmentation_coordination_source_status: str = "SINGLE_ARM_NOT_APPLICABLE"
    segmentation_debug_plots_required: bool = False

    @property
    def frame_names(self) -> tuple[str, ...]:
        return tuple(chunk.name for chunk in self.pose_chunks)

    @property
    def expected_low_dim_size(self) -> int:
        return 7 * len(self.pose_chunks)

    @property
    def arm_count(self) -> int:
        return 2 if self.bimanual else 1

    def extract_pose_chunks(
        self,
        task_low_dim_state: Any,
        *,
        convention: str = "core_wxyz",
    ) -> dict[str, Array]:
        """Split a saved task state into ordered, named pose copies."""

        state = unwrap_task_low_dim_state(task_low_dim_state)
        if state.size != self.expected_low_dim_size:
            raise ValueError(
                f"{self.task_name} task_low_dim_state has {state.size} values; "
                f"expected {self.expected_low_dim_size} from {self.source_expression}"
            )
        if convention not in {"rlbench_xyzw", "core_wxyz"}:
            raise ValueError("convention must be 'rlbench_xyzw' or 'core_wxyz'")
        chunks = {chunk.name: state[chunk.start : chunk.stop].copy() for chunk in self.pose_chunks}
        if convention == "core_wxyz":
            chunks = {name: xyzw_to_wxyz(pose) for name, pose in chunks.items()}
        return chunks


def _parse_task_spec(task_name: str, raw: Mapping[str, Any]) -> TaskSpec:
    required = {
        "paper_task_name",
        "module",
        "class_name",
        "bimanual",
        "paper_evaluation_group",
        "paper_scenarios",
        "pose_chunks",
        "source_expression",
        "source_status",
        "candidate_frame_policy",
        "candidate_frame_policy_source_status",
        "segmentation_coordination",
        "segmentation_coordination_source_status",
        "segmentation_debug_plots_required",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"{task_name} task spec is missing {sorted(missing)}")
    raw_chunks = raw["pose_chunks"]
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError(f"{task_name}.pose_chunks must be a non-empty list")
    chunks: list[TaskPoseChunk] = []
    for index, item in enumerate(raw_chunks):
        if not isinstance(item, Mapping):
            raise ValueError(f"{task_name}.pose_chunks[{index}] must be an object")
        name = item.get("name")
        role = item.get("role")
        if not isinstance(name, str) or not name or not isinstance(role, str) or not role:
            raise ValueError(f"{task_name}.pose_chunks[{index}] needs name and role")
        chunks.append(TaskPoseChunk(name=name, role=role, index=index))
    if len({chunk.name for chunk in chunks}) != len(chunks):
        raise ValueError(f"{task_name} has duplicate task-frame names")
    raw_scenarios = raw["paper_scenarios"]
    if (
        not isinstance(raw_scenarios, list)
        or not raw_scenarios
        or any(
            item not in {"static", "smooth", "teleport", "dynamic_unspecified"}
            for item in raw_scenarios
        )
    ):
        raise ValueError(f"{task_name}.paper_scenarios is invalid")
    bimanual = bool(raw["bimanual"])
    coordination = raw["segmentation_coordination"]
    allowed_coordination = {"independent", "shared_union"} if bimanual else {"single"}
    if coordination not in allowed_coordination:
        raise ValueError(
            f"{task_name}.segmentation_coordination must be one of "
            f"{sorted(allowed_coordination)}"
        )
    if raw["candidate_frame_policy"] != CANDIDATE_FRAME_POLICY:
        raise ValueError(f"{task_name} must retain every get_low_dim_state candidate frame")
    if raw["candidate_frame_policy_source_status"] != CANDIDATE_FRAME_POLICY_SOURCE_STATUS:
        raise ValueError(f"{task_name} candidate-frame provenance is not author-confirmed")
    source_status = raw["segmentation_coordination_source_status"]
    if not isinstance(source_status, str) or not source_status:
        raise ValueError(f"{task_name}.segmentation_coordination_source_status is invalid")
    debug_plots_required = raw["segmentation_debug_plots_required"]
    if not isinstance(debug_plots_required, bool):
        raise ValueError(f"{task_name}.segmentation_debug_plots_required must be boolean")
    return TaskSpec(
        task_name=task_name,
        paper_task_name=str(raw["paper_task_name"]),
        module=str(raw["module"]),
        class_name=str(raw["class_name"]),
        bimanual=bimanual,
        paper_evaluation_group=str(raw["paper_evaluation_group"]),
        paper_scenarios=tuple(raw_scenarios),
        pose_chunks=tuple(chunks),
        source_expression=str(raw["source_expression"]),
        source_status=str(raw["source_status"]),
        candidate_frame_policy=str(raw["candidate_frame_policy"]),
        candidate_frame_policy_source_status=str(
            raw["candidate_frame_policy_source_status"]
        ),
        segmentation_coordination=coordination,
        segmentation_coordination_source_status=source_status,
        segmentation_debug_plots_required=debug_plots_required,
    )


def load_task_specs(path: str | Path = TASK_CONFIG_PATH) -> dict[str, TaskSpec]:
    """Load and validate the frozen JSON task registry."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "rlbench-dynamac-task-specs-v2":
        raise ValueError("unsupported RLBench DynaMAC task-spec schema")
    source = payload.get("source", {})
    if source.get("commit") != RLBENCH_REFERENCE_COMMIT:
        raise ValueError("task specs are not bound to the pinned RLBench commit")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, Mapping) or not raw_tasks:
        raise ValueError("task registry must contain a non-empty tasks object")
    return {
        str(name): _parse_task_spec(str(name), raw)
        for name, raw in raw_tasks.items()
        if isinstance(raw, Mapping)
    }


def _snake_case(value: str) -> str:
    normalized = value.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", normalized).lower()
    return re.sub(r"_+", "_", normalized).strip("_")


TASK_SPECS = load_task_specs()
_TASK_ALIASES = {
    _snake_case(alias): name
    for name, spec in TASK_SPECS.items()
    for alias in (name, spec.class_name, spec.paper_task_name)
}


def get_task_spec(task_name: str) -> TaskSpec:
    """Return a task specification by snake-case, class, or paper name."""

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string")
    normalized = _snake_case(task_name)
    canonical = _TASK_ALIASES.get(normalized, normalized)
    try:
        return TASK_SPECS[canonical]
    except KeyError as exc:
        raise KeyError(
            f"unsupported RLBench DynaMAC task {task_name!r}; available tasks: {sorted(TASK_SPECS)}"
        ) from exc


def task_pose_chunks(
    task_name: str | TaskSpec,
    task_low_dim_state: Any,
    *,
    convention: str = "core_wxyz",
) -> dict[str, Array]:
    """Convenience wrapper around :meth:`TaskSpec.extract_pose_chunks`."""

    spec = task_name if isinstance(task_name, TaskSpec) else get_task_spec(task_name)
    return spec.extract_pose_chunks(task_low_dim_state, convention=convention)


split_task_pose_chunks = task_pose_chunks


__all__ = [
    "CANDIDATE_FRAME_POLICY",
    "CANDIDATE_FRAME_POLICY_SOURCE_STATUS",
    "CorePoseConvention",
    "RLBENCH_REFERENCE_COMMIT",
    "RLBenchPoseConvention",
    "SegmentationCoordination",
    "TASK_CONFIG_PATH",
    "TASK_SCHEMA_SOURCE_STATUS",
    "TASK_SPECS",
    "TaskPoseChunk",
    "TaskSpec",
    "core_pose_wxyz_to_rlbench_xyzw",
    "get_task_spec",
    "load_task_specs",
    "rlbench_pose_xyzw_to_core_wxyz",
    "split_task_pose_chunks",
    "task_pose_chunks",
    "unwrap_task_low_dim_state",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
