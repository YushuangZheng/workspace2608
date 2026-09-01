"""Versioned StoreBottle scene semantics and adapter entry points.

This module is intentionally independent of PyRep so training-data adapters,
configuration audits, and unit tests can use the corrected semantics without
launching CoppeliaSim.  The live RLBench class is isolated in the tracked
``store_bottle_live_v4`` module and imported only by simulator processes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    CANDIDATE_FRAME_POLICY,
    CANDIDATE_FRAME_POLICY_SOURCE_STATUS,
    TaskPoseChunk,
    TaskSpec,
)
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT

STORE_BOTTLE_SEMANTIC_SCHEMA = "rlbench-store-bottle-semantic-scene-v4"
STORE_BOTTLE_SEMANTIC_VERSION = "store_bottle_clean_v4"
STORE_BOTTLE_POLICY_SPEC_SCHEMA = "rlbench-store-bottle-policy-spec-v4"
STORE_BOTTLE_TASK_NAME = "bimanual_put_bottle_in_fridge"
STORE_BOTTLE_CONFIG_PATH = (
    INTEGRATION_ROOT
    / "configs"
    / "v4"
    / "store_bottle_semantics.json"
)
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT
_V4_IDENTITY_TRAINING_DATA_ROOT = "integrations/rlbench/data/v4/store_bottle"


def _canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def store_bottle_semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    """Keep the released semantic identity stable across data relocation."""

    identity = json.loads(json.dumps(payload))
    identity["release"]["training_data_root"] = _V4_IDENTITY_TRAINING_DATA_ROOT
    return _canonical_fingerprint(identity)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StoreBottleEntityGroup:
    """One independently movable physical entity and its audited subtree."""

    name: str
    semantic_root_name: str
    scene_root_name: str
    scene_root_type: str
    frame_name: str
    frame_object_name: str
    parents: tuple[tuple[str, str], ...]

    @property
    def parent_by_object(self) -> dict[str, str]:
        return dict(self.parents)

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.parents)


@dataclass(frozen=True)
class StoreBottleSemanticSpec:
    """Authenticated StoreBottle-only task, observation, and group identity."""

    schema: str
    semantic_version: str
    release_name: str
    training_data_root: str
    models_root: str
    evaluation_set_id: str
    other_task_models: str
    task_name: str
    paper_task_name: str
    module: str
    class_name: str
    legacy_ttm_task_name: str
    legacy_ttm_path: str
    legacy_ttm_sha256: str
    source_expression: str
    source_status: str
    pose_chunks: tuple[TaskPoseChunk, ...]
    frame_objects: tuple[tuple[str, str], ...]
    entity_groups: tuple[StoreBottleEntityGroup, ...]
    legacy_compatibility: tuple[tuple[str, Any], ...]
    fingerprint: str

    @property
    def group_by_name(self) -> dict[str, StoreBottleEntityGroup]:
        return {group.name: group for group in self.entity_groups}

    @property
    def frame_object_by_name(self) -> dict[str, str]:
        return dict(self.frame_objects)

    @property
    def task_spec(self) -> TaskSpec:
        """Return the existing adapter's data-only view of corrected frames."""

        return TaskSpec(
            task_name=self.task_name,
            paper_task_name=self.paper_task_name,
            module=self.module,
            class_name=self.class_name,
            bimanual=True,
            paper_evaluation_group="Bimanual",
            paper_scenarios=("static", "dynamic_unspecified"),
            pose_chunks=self.pose_chunks,
            configuration_chunks=(),
            structural_bindings={},
            source_expression=self.source_expression,
            source_status=self.source_status,
            candidate_frame_policy=CANDIDATE_FRAME_POLICY,
            candidate_frame_policy_source_status=(
                CANDIDATE_FRAME_POLICY_SOURCE_STATUS
            ),
            segmentation_coordination="independent",
            segmentation_coordination_source_status=(
                "AUTHOR_EMAIL_EXPLICIT_STOREBOTTLE_INDEPENDENT_20260814"
            ),
            segmentation_debug_plots_required=False,
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _parse_entity_group(
    name: str,
    raw: Any,
    *,
    task_name: str,
) -> StoreBottleEntityGroup:
    value = _require_mapping(raw, f"entity_groups.{name}")
    expected = {
        "semantic_root_name",
        "scene_root_name",
        "scene_root_type",
        "frame_name",
        "frame_object_name",
        "parents",
    }
    if set(value) != expected:
        raise ValueError(f"entity_groups.{name} fields are invalid")
    parents_raw = _require_mapping(value["parents"], f"entity_groups.{name}.parents")
    parents: list[tuple[str, str]] = []
    for object_name, parent_name in parents_raw.items():
        if (
            not isinstance(object_name, str)
            or not object_name
            or not isinstance(parent_name, str)
            or not parent_name
        ):
            raise ValueError(f"entity_groups.{name}.parents contains an invalid edge")
        parents.append((object_name, parent_name))
    fields = {
        key: value[key]
        for key in (
            "semantic_root_name",
            "scene_root_name",
            "scene_root_type",
            "frame_name",
            "frame_object_name",
        )
    }
    if any(not isinstance(item, str) or not item for item in fields.values()):
        raise ValueError(f"entity_groups.{name} names must be non-empty strings")
    if fields["scene_root_type"] != "SHAPE":
        raise ValueError(f"entity_groups.{name} root must be a SHAPE")
    parent_by_object = dict(parents)
    scene_root = fields["scene_root_name"]
    if parent_by_object.get(scene_root) != task_name:
        raise ValueError(f"entity_groups.{name} root must be a direct task child")
    if fields["frame_object_name"] not in parent_by_object:
        raise ValueError(f"entity_groups.{name} frame object is outside its group")
    members = set(parent_by_object)
    if any(
        object_name != scene_root and parent_name not in members
        for object_name, parent_name in parents
    ):
        raise ValueError(f"entity_groups.{name} is not a closed subtree")
    return StoreBottleEntityGroup(
        name=name,
        semantic_root_name=fields["semantic_root_name"],
        scene_root_name=scene_root,
        scene_root_type=fields["scene_root_type"],
        frame_name=fields["frame_name"],
        frame_object_name=fields["frame_object_name"],
        parents=tuple(parents),
    )


def load_store_bottle_semantic_spec(
    path: str | Path = STORE_BOTTLE_CONFIG_PATH,
    *,
    verify_ttm: bool = True,
) -> StoreBottleSemanticSpec:
    """Load the V4 StoreBottle-only contract without changing the V1 registry."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "semantic_version",
        "release",
        "task",
        "observation",
        "entity_groups",
        "legacy_compatibility",
    }:
        raise ValueError("StoreBottle semantic config fields are invalid")
    if (
        payload.get("schema") != STORE_BOTTLE_SEMANTIC_SCHEMA
        or payload.get("semantic_version") != STORE_BOTTLE_SEMANTIC_VERSION
    ):
        raise ValueError("unsupported StoreBottle semantic config")

    release = _require_mapping(payload["release"], "release")
    if (
        set(release)
        != {
            "name",
            "training_data_root",
            "models_root",
            "evaluation_set_id",
            "other_task_models",
        }
        or release.get("name") != "v4"
        or release.get("training_data_root")
        != (
            "integrations/rlbench/data/training/main/"
            "bimanual_put_bottle_in_fridge"
        )
        or release.get("models_root") != "integrations/rlbench/models/v4"
        or release.get("evaluation_set_id") != "rlbench_eval_v2"
        or release.get("other_task_models")
        != "inherit_v3_byte_for_byte_and_record_in_release_manifest"
    ):
        raise ValueError("StoreBottle V4 release paths are invalid")

    task = _require_mapping(payload["task"], "task")
    if set(task) != {
        "task_name",
        "paper_task_name",
        "module",
        "class_name",
        "legacy_ttm_task_name",
        "legacy_ttm_path",
        "legacy_ttm_sha256",
    } or (
        task.get("task_name") != STORE_BOTTLE_TASK_NAME
        or task.get("module")
        != "integrations.rlbench.rlbench_dynamac.store_bottle_live_v4"
        or task.get("class_name") != "BimanualPutBottleInFridgeSemanticV4"
        or task.get("legacy_ttm_task_name") != STORE_BOTTLE_TASK_NAME
    ):
        raise ValueError("StoreBottle semantic task identity is invalid")
    if any(not isinstance(value, str) or not value for value in task.values()):
        raise ValueError("StoreBottle semantic task fields must be strings")

    observation = _require_mapping(payload["observation"], "observation")
    if set(observation) != {"source_expression", "source_status", "pose_chunks"}:
        raise ValueError("StoreBottle semantic observation fields are invalid")
    raw_chunks = observation["pose_chunks"]
    if not isinstance(raw_chunks, list) or len(raw_chunks) != 2:
        raise ValueError("StoreBottle semantic observation must contain two poses")
    chunks: list[TaskPoseChunk] = []
    frame_objects: list[tuple[str, str]] = []
    for index, raw_chunk in enumerate(raw_chunks):
        chunk = _require_mapping(raw_chunk, f"observation.pose_chunks[{index}]")
        if set(chunk) != {"name", "role", "scene_object_name"} or any(
            not isinstance(chunk[key], str) or not chunk[key]
            for key in ("name", "role", "scene_object_name")
        ):
            raise ValueError("StoreBottle semantic pose chunk is invalid")
        chunks.append(TaskPoseChunk(chunk["name"], chunk["role"], index))
        frame_objects.append((chunk["name"], chunk["scene_object_name"]))
    if [chunk.name for chunk in chunks] != ["bottle", "fridge"]:
        raise ValueError("StoreBottle semantic frame order must be bottle, fridge")

    groups_raw = _require_mapping(payload["entity_groups"], "entity_groups")
    if set(groups_raw) != {"bottle", "fridge"}:
        raise ValueError("StoreBottle must define bottle and fridge groups")
    groups = tuple(
        _parse_entity_group(name, groups_raw[name], task_name=task["task_name"])
        for name in ("bottle", "fridge")
    )
    bottle_members = set(groups[0].members)
    fridge_members = set(groups[1].members)
    if bottle_members & fridge_members:
        raise ValueError("StoreBottle semantic groups must be disjoint")
    if {group.frame_name: group.frame_object_name for group in groups} != dict(
        frame_objects
    ):
        raise ValueError("StoreBottle observation frames do not match entity groups")
    waypoint_members = {
        member
        for group in groups
        for member in group.members
        if member.startswith("waypoint")
    }
    if waypoint_members != {f"waypoint{index}" for index in range(9)}:
        raise ValueError("StoreBottle semantic groups must assign every waypoint")
    if "success" not in fridge_members or "success" in bottle_members:
        raise ValueError("StoreBottle success sensor must follow the fridge")

    compatibility = _require_mapping(
        payload["legacy_compatibility"], "legacy_compatibility"
    )
    expected_compatibility = {
        "legacy_class_unchanged",
        "legacy_low_dim_frames",
        "legacy_boundary_root_unchanged",
        "semantic_v4_must_not_infer_dynamic_entities_from_boundary_root",
        "sealed_v1_artifacts_remain_read_only",
    }
    if (
        set(compatibility) != expected_compatibility
        or compatibility.get("legacy_class_unchanged")
        != "BimanualPutBottleInFridge"
        or compatibility.get("legacy_low_dim_frames")
        != ["bottle", "fridge_root"]
        or compatibility.get("legacy_boundary_root_unchanged") != "fridge_root"
        or compatibility.get(
            "semantic_v4_must_not_infer_dynamic_entities_from_boundary_root"
        )
        is not True
        or compatibility.get("sealed_v1_artifacts_remain_read_only") is not True
    ):
        raise ValueError("StoreBottle legacy-compatibility contract is invalid")

    ttm_path = (_REPOSITORY_ROOT / task["legacy_ttm_path"]).resolve()
    if verify_ttm and (
        not ttm_path.is_file() or _file_sha256(ttm_path) != task["legacy_ttm_sha256"]
    ):
        raise ValueError("StoreBottle semantic task is not bound to the pinned TTM")

    return StoreBottleSemanticSpec(
        schema=payload["schema"],
        semantic_version=payload["semantic_version"],
        release_name=release["name"],
        training_data_root=release["training_data_root"],
        models_root=release["models_root"],
        evaluation_set_id=release["evaluation_set_id"],
        other_task_models=release["other_task_models"],
        task_name=task["task_name"],
        paper_task_name=task["paper_task_name"],
        module=task["module"],
        class_name=task["class_name"],
        legacy_ttm_task_name=task["legacy_ttm_task_name"],
        legacy_ttm_path=task["legacy_ttm_path"],
        legacy_ttm_sha256=task["legacy_ttm_sha256"],
        source_expression=observation["source_expression"],
        source_status=observation["source_status"],
        pose_chunks=tuple(chunks),
        frame_objects=tuple(frame_objects),
        entity_groups=groups,
        legacy_compatibility=tuple(compatibility.items()),
        fingerprint=store_bottle_semantic_fingerprint(payload),
    )


def store_bottle_semantic_task_spec() -> TaskSpec:
    """Return the corrected StoreBottle spec for existing generic adapters."""

    return load_store_bottle_semantic_spec().task_spec


def store_bottle_policy_spec_identity(
    spec: StoreBottleSemanticSpec | None = None,
) -> dict[str, Any]:
    """Return the complete training/online identity for the corrected frames."""

    selected = load_store_bottle_semantic_spec() if spec is None else spec
    task_spec = selected.task_spec
    return {
        "schema": STORE_BOTTLE_POLICY_SPEC_SCHEMA,
        "semantic_schema": selected.schema,
        "semantic_version": selected.semantic_version,
        "semantic_fingerprint": selected.fingerprint,
        "task": task_spec.task_name,
        "paper_task_name": task_spec.paper_task_name,
        "module": task_spec.module,
        "class_name": task_spec.class_name,
        "bimanual": task_spec.bimanual,
        "frame_names": list(task_spec.frame_names),
        "frame_objects": [
            {"frame": frame, "scene_object": scene_object}
            for frame, scene_object in selected.frame_objects
        ],
        "pose_chunks": [
            {
                "name": chunk.name,
                "role": chunk.role,
                "index": chunk.index,
                "source_slice": list(chunk.source_slice),
            }
            for chunk in task_spec.pose_chunks
        ],
        "expected_low_dim_size": task_spec.expected_low_dim_size,
        "source_expression": task_spec.source_expression,
        "source_status": task_spec.source_status,
        "candidate_frame_policy": task_spec.candidate_frame_policy,
        "candidate_frame_policy_source_status": (
            task_spec.candidate_frame_policy_source_status
        ),
        "segmentation_coordination": task_spec.segmentation_coordination,
        "segmentation_coordination_source_status": (
            task_spec.segmentation_coordination_source_status
        ),
        "segmentation_debug_plots_required": (
            task_spec.segmentation_debug_plots_required
        ),
    }


def extract_store_bottle_semantic_episode(episode: Any) -> Any:
    """Extract one paired episode using bottle and true-fridge frames."""

    from integrations.rlbench.rlbench_dynamac.data.demo_adapter import extract_bimanual_episode

    return extract_bimanual_episode(episode, store_bottle_semantic_task_spec())


def make_store_bottle_semantic_demonstrations(
    episodes: Sequence[Any],
    **kwargs: Any,
) -> Any:
    """Build StoreBottle V4 demonstrations through the existing adapter."""

    from integrations.rlbench.rlbench_dynamac.data.demo_adapter import make_bimanual_demonstrations

    return make_bimanual_demonstrations(
        episodes,
        store_bottle_semantic_task_spec(),
        **kwargs,
    )


def store_bottle_semantic_observations_from_rlbench(
    observation: Any,
) -> tuple[Any, Any]:
    """Build synchronized policy observations with the corrected frame names."""

    from integrations.rlbench.rlbench_dynamac.core.runtime import bimanual_observations_from_rlbench

    return bimanual_observations_from_rlbench(
        observation,
        store_bottle_semantic_task_spec(),
    )


def validate_store_bottle_scene_hierarchy(
    rows: Sequence[Mapping[str, Any]],
    spec: StoreBottleSemanticSpec | None = None,
) -> dict[str, Any]:
    """Validate a task-tree snapshot against the two semantic group contracts."""

    selected = load_store_bottle_semantic_spec() if spec is None else spec
    actual: dict[str, str | None] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("StoreBottle task-tree rows must be objects")
        name = row.get("name")
        parent = row.get("parent")
        if (
            not isinstance(name, str)
            or not name
            or (parent is not None and not isinstance(parent, str))
            or name in actual
        ):
            raise ValueError("StoreBottle task-tree row is invalid")
        actual[name] = parent

    expected = {
        member: parent
        for group in selected.entity_groups
        for member, parent in group.parents
    }
    missing = sorted(set(expected).difference(actual))
    parent_mismatches = {
        name: {"expected": parent, "actual": actual.get(name)}
        for name, parent in expected.items()
        if name in actual and actual[name] != parent
    }
    return {
        "schema": "rlbench-store-bottle-semantic-hierarchy-audit-v1",
        "semantic_version": selected.semantic_version,
        "semantic_fingerprint": selected.fingerprint,
        "groups": {
            group.name: {
                "scene_root_name": group.scene_root_name,
                "frame_name": group.frame_name,
                "frame_object_name": group.frame_object_name,
                "members": list(group.members),
            }
            for group in selected.entity_groups
        },
        "missing": missing,
        "parent_mismatches": parent_mismatches,
        "passed": not missing and not parent_mismatches,
    }


__all__ = [
    "STORE_BOTTLE_CONFIG_PATH",
    "STORE_BOTTLE_POLICY_SPEC_SCHEMA",
    "STORE_BOTTLE_SEMANTIC_SCHEMA",
    "STORE_BOTTLE_SEMANTIC_VERSION",
    "STORE_BOTTLE_TASK_NAME",
    "StoreBottleEntityGroup",
    "StoreBottleSemanticSpec",
    "extract_store_bottle_semantic_episode",
    "load_store_bottle_semantic_spec",
    "make_store_bottle_semantic_demonstrations",
    "store_bottle_semantic_observations_from_rlbench",
    "store_bottle_policy_spec_identity",
    "store_bottle_semantic_fingerprint",
    "store_bottle_semantic_task_spec",
    "validate_store_bottle_scene_hierarchy",
]
