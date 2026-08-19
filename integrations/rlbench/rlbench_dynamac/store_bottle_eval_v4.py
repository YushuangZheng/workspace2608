"""StoreBottle V4 task-scoped multi-entity evaluation protocol.

The protocol is deliberately independent of PyRep at import time.  Simulator
objects are imported or inspected only by the staging/binding/controller entry
points, while plan authentication can run in the policy and seal processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .store_bottle_semantics import (
    STORE_BOTTLE_SEMANTIC_SCHEMA,
    STORE_BOTTLE_SEMANTIC_VERSION,
    STORE_BOTTLE_TASK_NAME,
    load_store_bottle_semantic_spec,
)


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
V4_STORE_INTERVENTION_CONFIG = (
    INTEGRATION_ROOT / "configs" / "v4" / "store_bottle_intervention.json"
)
V4_STORE_MOTION_SOURCE_CONFIG = (
    INTEGRATION_ROOT / "configs" / "v4" / "store_bottle_motion_source.json"
)
V4_STORE_INTERVENTION_SCHEMA = "rlbench-dynamac-store-bottle-intervention-v4"
V4_STORE_MOTION_SOURCE_SCHEMA = "rlbench-dynamac-store-bottle-motion-source-v4"
V4_STORE_PLAN_SCHEMA = "dynamac-store-bottle-multi-entity-motion-plan-v4"
V4_STORE_PLAN_VALIDATION_SCHEMA = (
    "dynamac-store-bottle-multi-entity-plan-validation-v4"
)
V4_STORE_BATCH_SCHEMA = "dynamac-store-bottle-multi-entity-plan-batch-v4"
V4_STORE_RUNTIME_LOADER_ID = "store-bottle-multi-entity-motion-plan-batch-v4"
V4_STORE_MOTION_PROTOCOL_ID = (
    "store-bottle-independent-entity-source-relative-teleport-v4"
)
V4_STORE_TRIGGER_AUTHENTICATION_SCHEMA = (
    "dynamac-store-bottle-multi-entity-trigger-authentication-v4"
)

V4_STORE_BOTTLE_TRIGGER_STEP = 60
V4_STORE_FRIDGE_TRIGGER_STEP = 45
V4_STORE_ENTITY_ORDER = ("bottle", "fridge")
V4_STORE_MODE_ORDER = ("bottle_only", "fridge_only", "both")
V4_STORE_MODE_MOVED_ENTITIES = {
    "bottle_only": frozenset({"bottle"}),
    "fridge_only": frozenset({"fridge"}),
    "both": frozenset({"bottle", "fridge"}),
}
V4_STORE_ENTITY_ROOTS = {"bottle": "fridge_root", "fridge": "fridge_base"}
V4_STORE_ENTITY_FRAMES = {"bottle": "bottle", "fridge": "fridge"}
V4_STORE_TRIGGER_STEPS = {
    "bottle": V4_STORE_BOTTLE_TRIGGER_STEP,
    "fridge": V4_STORE_FRIDGE_TRIGGER_STEP,
}
V4_STORE_POSE_ATOL = 5.0e-6
V4_STORE_ROTATION_ATOL_RAD = 5.0e-5
_UINT32_MODULUS = 2**32 - 1


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(relative: str) -> Path:
    path = (REPOSITORY_ROOT / relative).resolve()
    path.relative_to(REPOSITORY_ROOT.resolve())
    return path


def _load_config(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"unsupported StoreBottle V4 protocol schema in {path}")
    return {**payload, "fingerprint": canonical_fingerprint(payload)}


def load_v4_store_intervention_protocol(
    path: Path = V4_STORE_INTERVENTION_CONFIG,
    *,
    verify_evidence_files: bool = True,
) -> dict[str, Any]:
    payload = _load_config(path, V4_STORE_INTERVENTION_SCHEMA)
    triggers = payload.get("triggers")
    evidence = payload.get("evidence")
    if (
        payload.get("release") != "v4"
        or payload.get("task") != STORE_BOTTLE_TASK_NAME
        or payload.get("formal_scenarios") != ["static", "teleport"]
        or payload.get("clock") != "successfully_committed_policy_ticks"
        or payload.get("application_timing")
        != "before_requesting_policy_action_at_global_tick"
        or not isinstance(triggers, dict)
        or set(triggers) != set(V4_STORE_ENTITY_ORDER)
        or payload.get("teleport")
        != {
            "applications_per_moved_entity": 1,
            "entities_trigger_independently": True,
            "observation_refreshed_before_policy_action": True,
        }
        or payload.get("final_settling_physics_steps") != 10
        or not isinstance(evidence, dict)
        or evidence.get("selection_authority")
        != "manual_pre_interaction_skill_semantics_from_v4_checkpoint_and_five_static_expert_demonstrations"
        or evidence.get("integer_ticks_authoritative") is not True
        or evidence.get("expert_gripper_open_at_all_projected_triggers") is not True
        or evidence.get("policy_evaluation_results_read") is not False
        or evidence.get("result_based_retuning_forbidden") is not True
        or evidence.get("legacy_v3_profile_unchanged") is not True
    ):
        raise ValueError("StoreBottle V4 intervention protocol is invalid")
    expected_triggers = {
        "bottle": {
            "arm": "left",
            "frame": "bottle",
            "skill_label": 0,
            "local_tick": V4_STORE_BOTTLE_TRIGGER_STEP,
            "global_tick": V4_STORE_BOTTLE_TRIGGER_STEP,
            "expected_gripper_state": "open",
            "interaction_event": "left gripper approaching the bottle before grasp",
        },
        "fridge": {
            "arm": "right",
            "frame": "fridge",
            "skill_label": 0,
            "local_tick": V4_STORE_FRIDGE_TRIGGER_STEP,
            "global_tick": V4_STORE_FRIDGE_TRIGGER_STEP,
            "expected_gripper_state": "open",
            "interaction_event": (
                "right gripper approaching the fridge handle before door grasp"
            ),
        },
    }
    if triggers != expected_triggers:
        raise ValueError("StoreBottle V4 entity triggers are invalid")
    if verify_evidence_files:
        for path_key, sha_key in (
            ("training_manifest_path", "training_manifest_sha256"),
            ("collection_manifest_path", "collection_manifest_sha256"),
        ):
            evidence_path = _repository_path(evidence.get(path_key, ""))
            if not evidence_path.is_file() or _file_sha256(evidence_path) != evidence.get(
                sha_key
            ):
                raise ValueError(f"StoreBottle V4 trigger evidence changed: {path_key}")
        training = json.loads(
            _repository_path(evidence["training_manifest_path"]).read_text(
                encoding="utf-8"
            )
        )
        audit = training.get("checkpoint_trigger_audit", {}).get("arms", {})
        if (
            training.get("manifest_schema") != "dynamac-direct-training-v4"
            or training.get("training_identity", {}).get("fingerprint")
            != evidence.get("training_identity_fingerprint")
            or training.get("adapter", {}).get("frame_names") != ["bottle", "fridge"]
            or training.get("left", {}).get("durations", [None])[0] != 115
            or training.get("right", {}).get("durations", [None])[0] != 113
            or audit.get("left", {}).get("skills", [{}])[0].get("duration") != 115
            or audit.get("right", {}).get("skills", [{}])[0].get("duration") != 113
        ):
            raise ValueError("StoreBottle V4 trigger/checkpoint evidence is invalid")
    return payload


def load_v4_store_motion_source_protocol(
    path: Path = V4_STORE_MOTION_SOURCE_CONFIG,
    *,
    verify_semantics_file: bool = True,
) -> dict[str, Any]:
    payload = _load_config(path, V4_STORE_MOTION_SOURCE_SCHEMA)
    semantics = payload.get("task_semantics")
    schedule = payload.get("episode_mode_schedule")
    entities = payload.get("entities")
    generation = payload.get("candidate_generation")
    if (
        payload.get("release") != "v4"
        or payload.get("task") != STORE_BOTTLE_TASK_NAME
        or payload.get("source_selection_max_attempts") != 20
        or payload.get("goal_sampling_max_attempts") != 100
        or payload.get("runtime_loader") != V4_STORE_RUNTIME_LOADER_ID
        or not isinstance(semantics, dict)
        or semantics.get("schema") != STORE_BOTTLE_SEMANTIC_SCHEMA
        or semantics.get("semantic_version") != STORE_BOTTLE_SEMANTIC_VERSION
        or semantics.get("low_dim_frames") != ["bottle", "fridge"]
        or semantics.get("independent_roots") != V4_STORE_ENTITY_ROOTS
        or schedule
        != {
            "formula": "episode_index_mod_3",
            "mapping": {
                "0": "bottle_only",
                "1": "fridge_only",
                "2": "both",
            },
            "formal_n200_counts": {
                "bottle_only": 67,
                "fridge_only": 67,
                "both": 66,
            },
        }
        or not isinstance(entities, dict)
        or set(entities) != set(V4_STORE_ENTITY_ORDER)
        or not isinstance(generation, dict)
        or generation.get("protocol_id")
        != "store-bottle-independent-entity-source-relative-candidate-v4"
        or generation.get("selection_authority") != "scene_validity_only"
        or generation.get("policy_result_fields_read") is not False
        or generation.get("result_based_candidate_selection_forbidden") is not True
        or generation.get("minimum_paired_arm_waypoint_distance_m") != 0.18
        or generation.get(
            "minimum_bottle_to_fridge_door_sweep_proxy_clearance_m"
        )
        != 0.2
    ):
        raise ValueError("StoreBottle V4 motion-source protocol is invalid")
    expected_limits = {
        "bottle": ("fridge_root", 0.03, 0.10, 0.10, 1_013_904_223),
        "fridge": ("fridge_base", 0.02, 0.05, 0.05, 2_654_435_761),
    }
    for entity, (root, low, high, yaw, salt) in expected_limits.items():
        value = entities[entity]
        if (
            value.get("frame") != entity
            or value.get("root") != root
            or value.get("candidate_seed_salt") != salt
            or value.get("translation")
            != {
                "reference": "entity_source_A_root",
                "frame": "world_xy",
                "radial_min_m": low,
                "radial_max_m": high,
                "z_delta_m": 0.0,
            }
            or value.get("rotation")
            != {
                "composition": "world_z_yaw_left_multiply_source_quaternion",
                "yaw_delta_abs_max_rad": yaw,
                "roll_pitch": "unchanged_from_source_A",
            }
        ):
            raise ValueError(f"StoreBottle V4 {entity} motion limits are invalid")
    if verify_semantics_file:
        semantic_path = _repository_path(semantics.get("semantic_config_path", ""))
        if (
            not semantic_path.is_file()
            or _file_sha256(semantic_path) != semantics.get("semantic_config_sha256")
        ):
            raise ValueError("StoreBottle V4 semantic config changed")
        load_store_bottle_semantic_spec(semantic_path)
    return payload


def store_mode_for_episode(episode_index: int) -> str:
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise ValueError("StoreBottle episode index must be non-negative")
    return V4_STORE_MODE_ORDER[episode_index % 3]


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quaternion_angle_xyzw(left: Any, right: Any) -> float:
    # Copy before normalization: callers commonly pass a quaternion view into
    # a serialized pose array, and mutating that view would make round-tripping
    # depend on tiny cross-Python normalization differences.
    lhs = np.array(left, dtype=np.float64, copy=True)
    rhs = np.array(right, dtype=np.float64, copy=True)
    lhs /= np.linalg.norm(lhs)
    rhs /= np.linalg.norm(rhs)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(lhs, rhs)), 0.0, 1.0))))


def store_entity_candidate_seed(candidate_seed: int, entity: str) -> int:
    protocol = load_v4_store_motion_source_protocol(verify_semantics_file=False)
    if entity not in V4_STORE_ENTITY_ORDER:
        raise ValueError("unknown StoreBottle entity")
    if (
        isinstance(candidate_seed, bool)
        or not isinstance(candidate_seed, int)
        or candidate_seed < 0
    ):
        raise ValueError("StoreBottle candidate seed must be non-negative")
    salt = protocol["entities"][entity]["candidate_seed_salt"]
    return int((candidate_seed * 1_664_525 + salt) % _UINT32_MODULUS)


def sample_v4_store_entity_goal_pose(
    source_pose: Any,
    candidate_seed: int,
    *,
    entity: str,
) -> np.ndarray:
    """Sample one entity B from only its A and preregistered seed."""

    protocol = load_v4_store_motion_source_protocol(verify_semantics_file=False)
    if entity not in V4_STORE_ENTITY_ORDER:
        raise ValueError("unknown StoreBottle entity")
    source = np.asarray(source_pose, dtype=np.float64)
    if source.shape != (7,) or not np.all(np.isfinite(source)):
        raise ValueError("StoreBottle source pose must be finite 7D")
    norm = float(np.linalg.norm(source[3:7]))
    if norm <= 1.0e-12 or not math.isfinite(norm):
        raise ValueError("StoreBottle source quaternion is invalid")
    quaternion = source[3:7] / norm
    entity_seed = store_entity_candidate_seed(candidate_seed, entity)
    generator = random.Random(entity_seed)
    limits = protocol["entities"][entity]
    radius = generator.uniform(
        limits["translation"]["radial_min_m"],
        limits["translation"]["radial_max_m"],
    )
    angle = generator.uniform(-math.pi, math.pi)
    yaw = generator.uniform(
        -limits["rotation"]["yaw_delta_abs_max_rad"],
        limits["rotation"]["yaw_delta_abs_max_rad"],
    )
    position = source[:3].copy()
    position[0] += radius * math.cos(angle)
    position[1] += radius * math.sin(angle)
    half = 0.5 * yaw
    yaw_quaternion = np.asarray([0.0, 0.0, math.sin(half), math.cos(half)])
    goal_quaternion = _quaternion_multiply_xyzw(yaw_quaternion, quaternion)
    goal_quaternion /= np.linalg.norm(goal_quaternion)
    return np.concatenate((position, goal_quaternion))


def store_entity_geometry(source_pose: Any, goal_pose: Any) -> dict[str, float]:
    source = np.asarray(source_pose, dtype=np.float64)
    goal = np.asarray(goal_pose, dtype=np.float64)
    if source.shape != (7,) or goal.shape != (7,) or not (
        np.all(np.isfinite(source)) and np.all(np.isfinite(goal))
    ):
        raise ValueError("StoreBottle plan poses must be finite 7D")
    delta = goal[:3] - source[:3]
    source_q = source[3:7] / np.linalg.norm(source[3:7])
    goal_q = goal[3:7] / np.linalg.norm(goal[3:7])
    inverse = np.asarray([-source_q[0], -source_q[1], -source_q[2], source_q[3]])
    relative = _quaternion_multiply_xyzw(goal_q, inverse)
    if relative[3] < 0.0:
        relative = -relative
    relative /= np.linalg.norm(relative)
    return {
        "xy_radius_m": float(np.linalg.norm(delta[:2])),
        "z_delta_m": float(delta[2]),
        "yaw_delta_rad": float(2.0 * math.atan2(relative[2], relative[3])),
        "relative_rotation_xy_norm": float(np.linalg.norm(relative[:2])),
    }


@dataclass(frozen=True)
class StoreBottleEntityMotion:
    name: str
    root_name: str
    frame_name: str
    source_pose: tuple[float, ...]
    goal_pose: tuple[float, ...]
    moved: bool
    candidate_seed: int | None

    def __post_init__(self) -> None:
        if self.name not in V4_STORE_ENTITY_ORDER:
            raise ValueError("unknown StoreBottle entity")
        if (
            self.root_name != V4_STORE_ENTITY_ROOTS[self.name]
            or self.frame_name != V4_STORE_ENTITY_FRAMES[self.name]
        ):
            raise ValueError("StoreBottle entity root/frame identity is invalid")
        source = np.asarray(self.source_pose, dtype=np.float64)
        goal = np.asarray(self.goal_pose, dtype=np.float64)
        if source.shape != (7,) or goal.shape != (7,) or not (
            np.all(np.isfinite(source)) and np.all(np.isfinite(goal))
        ):
            raise ValueError("StoreBottle entity poses must be finite 7D")
        if not isinstance(self.moved, bool):
            raise ValueError("StoreBottle entity moved flag is invalid")
        if self.moved:
            if (
                isinstance(self.candidate_seed, bool)
                or not isinstance(self.candidate_seed, int)
                or self.candidate_seed < 0
            ):
                raise ValueError("moved StoreBottle entity needs a candidate seed")
            expected = sample_v4_store_entity_goal_pose(
                source,
                self.candidate_seed,
                entity=self.name,
            )
            geometry = store_entity_geometry(source, goal)
            limits = load_v4_store_motion_source_protocol(
                verify_semantics_file=False
            )["entities"][self.name]
            if (
                np.linalg.norm(expected[:3] - goal[:3]) > V4_STORE_POSE_ATOL
                or _quaternion_angle_xyzw(expected[3:], goal[3:])
                > V4_STORE_ROTATION_ATOL_RAD
                or geometry["xy_radius_m"]
                < limits["translation"]["radial_min_m"] - V4_STORE_POSE_ATOL
                or geometry["xy_radius_m"]
                > limits["translation"]["radial_max_m"] + V4_STORE_POSE_ATOL
                or abs(geometry["z_delta_m"]) > V4_STORE_POSE_ATOL
                or abs(geometry["yaw_delta_rad"])
                > limits["rotation"]["yaw_delta_abs_max_rad"]
                + V4_STORE_ROTATION_ATOL_RAD
                or geometry["relative_rotation_xy_norm"]
                > V4_STORE_ROTATION_ATOL_RAD
            ):
                raise ValueError("StoreBottle entity A-to-B geometry is invalid")
        elif self.candidate_seed is not None or not np.allclose(
            source,
            goal,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("unmoved StoreBottle entity must encode exact A=B")
        object.__setattr__(self, "source_pose", tuple(float(v) for v in source))
        object.__setattr__(self, "goal_pose", tuple(float(v) for v in goal))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_name": self.root_name,
            "frame_name": self.frame_name,
            "source_pose": list(self.source_pose),
            "goal_pose": list(self.goal_pose),
            "moved": self.moved,
            "candidate_seed": self.candidate_seed,
            "trigger_step": V4_STORE_TRIGGER_STEPS[self.name],
        }

    @classmethod
    def from_json(cls, payload: Any) -> "StoreBottleEntityMotion":
        if not isinstance(payload, dict):
            raise ValueError("StoreBottle entity plan must be an object")
        base_fields = {
            "name",
            "root_name",
            "frame_name",
            "source_pose",
            "goal_pose",
            "moved",
            "candidate_seed",
            "trigger_step",
        }
        if set(payload) not in {frozenset(base_fields), frozenset(base_fields | {"geometry"})}:
            raise ValueError("StoreBottle entity plan fields are invalid")
        entity = cls(
            name=payload.get("name", ""),
            root_name=payload.get("root_name", ""),
            frame_name=payload.get("frame_name", ""),
            source_pose=tuple(payload.get("source_pose", ())),
            goal_pose=tuple(payload.get("goal_pose", ())),
            moved=payload.get("moved"),
            candidate_seed=payload.get("candidate_seed"),
        )
        structural = {key: value for key, value in payload.items() if key != "geometry"}
        if structural != entity.to_json():
            raise ValueError("StoreBottle entity plan fields are inconsistent")
        # Early smoke batches carried this redundant derived diagnostic.  It is
        # accepted only within a tight numeric tolerance so Python 3.8/3.12 libm
        # roundoff cannot invalidate an otherwise byte-authenticated batch; the
        # canonical V4 representation omits it to keep fingerprints portable.
        if "geometry" in payload:
            raw_geometry = payload["geometry"]
            expected_geometry = store_entity_geometry(
                entity.source_pose,
                entity.goal_pose,
            )
            if (
                not isinstance(raw_geometry, dict)
                or set(raw_geometry) != set(expected_geometry)
                or any(
                    not isinstance(raw_geometry[key], (int, float))
                    or isinstance(raw_geometry[key], bool)
                    or not math.isfinite(float(raw_geometry[key]))
                    or abs(float(raw_geometry[key]) - expected_geometry[key]) > 1.0e-10
                    for key in expected_geometry
                )
            ):
                raise ValueError("StoreBottle derived entity geometry is inconsistent")
        return entity


@dataclass(frozen=True)
class StoreBottleMultiEntityPlan:
    task_name: str
    episode_index: int
    episode_seed: int
    variation: int
    mode: str
    entities: tuple[StoreBottleEntityMotion, ...]
    source_low_dim_state: tuple[float, ...]
    validation: dict[str, Any]

    def __post_init__(self) -> None:
        if self.task_name != STORE_BOTTLE_TASK_NAME:
            raise ValueError("StoreBottle plan task is invalid")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
            or isinstance(self.episode_seed, bool)
            or not isinstance(self.episode_seed, int)
            or self.episode_seed < 0
            or isinstance(self.variation, bool)
            or not isinstance(self.variation, int)
            or self.variation < 0
        ):
            raise ValueError("StoreBottle plan schedule is invalid")
        if self.mode != store_mode_for_episode(self.episode_index):
            raise ValueError("StoreBottle plan mode does not follow episode_index%3")
        entities = tuple(self.entities)
        if tuple(entity.name for entity in entities) != V4_STORE_ENTITY_ORDER:
            raise ValueError("StoreBottle plan must contain bottle then fridge")
        moved = frozenset(entity.name for entity in entities if entity.moved)
        if moved != V4_STORE_MODE_MOVED_ENTITIES[self.mode]:
            raise ValueError("StoreBottle plan moved entities do not match its mode")
        low_dim = np.asarray(self.source_low_dim_state, dtype=np.float64)
        if low_dim.shape != (14,) or not np.all(np.isfinite(low_dim)):
            raise ValueError("StoreBottle source low-dimensional state must be 14D")
        validation = self.validation
        motion = load_v4_store_motion_source_protocol(verify_semantics_file=False)
        intervention = load_v4_store_intervention_protocol(
            verify_evidence_files=False
        )
        if (
            not isinstance(validation, dict)
            or validation.get("schema") != V4_STORE_PLAN_VALIDATION_SCHEMA
            or validation.get("source_seed") is None
            or validation.get("source_waypoint_validated") is not True
            or validation.get("goal_waypoint_validated") is not True
            or validation.get("goal_sampling_max_attempts")
            != motion["goal_sampling_max_attempts"]
            or not 1
            <= validation.get("sampling_attempts", 0)
            <= motion["goal_sampling_max_attempts"]
            or validation.get("motion_source_fingerprint") != motion["fingerprint"]
            or validation.get("intervention_fingerprint")
            != intervention["fingerprint"]
            or validation.get("policy_result_fields_read") is not False
        ):
            raise ValueError("StoreBottle plan validation evidence is invalid")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(
            self,
            "source_low_dim_state",
            tuple(float(v) for v in low_dim),
        )
        object.__setattr__(self, "validation", dict(validation))

    @property
    def moved_entities(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.entities if entity.moved)

    def entity(self, name: str) -> StoreBottleEntityMotion:
        for entity in self.entities:
            if entity.name == name:
                return entity
        raise KeyError(name)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": V4_STORE_PLAN_SCHEMA,
            "protocol_id": V4_STORE_MOTION_PROTOCOL_ID,
            "task_name": self.task_name,
            "episode_index": self.episode_index,
            "episode_seed": self.episode_seed,
            "variation": self.variation,
            "mode": self.mode,
            "entities": {entity.name: entity.to_json() for entity in self.entities},
            "source_low_dim_state": list(self.source_low_dim_state),
            "validation": dict(self.validation),
        }

    def fingerprint(self) -> str:
        return canonical_fingerprint(self.metadata())

    def to_json(self) -> dict[str, Any]:
        return {**self.metadata(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_json(cls, payload: Any) -> "StoreBottleMultiEntityPlan":
        if not isinstance(payload, dict):
            raise ValueError("StoreBottle plan must be an object")
        expected_fields = {
            "schema",
            "protocol_id",
            "task_name",
            "episode_index",
            "episode_seed",
            "variation",
            "mode",
            "entities",
            "source_low_dim_state",
            "validation",
            "fingerprint",
        }
        body = {key: value for key, value in payload.items() if key != "fingerprint"}
        if (
            set(payload) != expected_fields
            or payload.get("schema") != V4_STORE_PLAN_SCHEMA
            or payload.get("protocol_id") != V4_STORE_MOTION_PROTOCOL_ID
            or payload.get("fingerprint") != canonical_fingerprint(body)
        ):
            raise ValueError("StoreBottle plan authentication failed")
        raw_entities = payload.get("entities")
        if not isinstance(raw_entities, dict) or set(raw_entities) != set(
            V4_STORE_ENTITY_ORDER
        ):
            raise ValueError("StoreBottle plan entities are invalid")
        plan = cls(
            task_name=payload.get("task_name", ""),
            episode_index=payload.get("episode_index"),
            episode_seed=payload.get("episode_seed"),
            variation=payload.get("variation"),
            mode=payload.get("mode", ""),
            entities=tuple(
                StoreBottleEntityMotion.from_json(raw_entities[name])
                for name in V4_STORE_ENTITY_ORDER
            ),
            source_low_dim_state=tuple(payload.get("source_low_dim_state", ())),
            validation=(
                payload.get("validation")
                if isinstance(payload.get("validation"), dict)
                else {}
            ),
        )
        return plan


def store_bottle_motion_plan_batch(
    *,
    base_seed: int,
    variations: list[int],
    plans: list[StoreBottleMultiEntityPlan],
) -> dict[str, Any]:
    if len(variations) != len(plans) or not plans:
        raise ValueError("StoreBottle batch schedule length is invalid")
    counts = {mode: 0 for mode in V4_STORE_MODE_ORDER}
    for episode, (variation, plan) in enumerate(zip(variations, plans)):
        if (
            plan.episode_index != episode
            or plan.episode_seed != base_seed + episode
            or plan.variation != variation
            or plan.mode != store_mode_for_episode(episode)
        ):
            raise ValueError("StoreBottle plan does not match batch schedule")
        counts[plan.mode] += 1
    if len(plans) == 200 and counts != {
        "bottle_only": 67,
        "fridge_only": 67,
        "both": 66,
    }:
        raise ValueError("StoreBottle formal n200 mode counts are invalid")
    body = {
        "schema": V4_STORE_BATCH_SCHEMA,
        "protocol_id": V4_STORE_MOTION_PROTOCOL_ID,
        "task_name": STORE_BOTTLE_TASK_NAME,
        "base_seed": int(base_seed),
        "episodes": len(plans),
        "variation_schedule": [int(value) for value in variations],
        "scenario_independent": True,
        "seed_domain": (
            "logical_episode=base_seed+episode;A=certified_source_seed;"
            "B=attempt_seed_then_independent_entity_seed;scenario_and_policy_results_excluded"
        ),
        "mode_schedule": "episode_index_mod_3",
        "mode_counts": counts,
        "plans": [plan.to_json() for plan in plans],
    }
    return {**body, "batch_fingerprint": canonical_fingerprint(body)}


def load_v4_store_motion_plan_batch(payload: dict[str, Any]) -> list[Any]:
    """Stable runtime loader for an ``rlbench_eval_v2`` task envelope."""

    if not isinstance(payload, dict):
        raise ValueError("StoreBottle plan batch must be an object")
    body = {key: value for key, value in payload.items() if key != "batch_fingerprint"}
    expected_fields = {
        "schema",
        "protocol_id",
        "task_name",
        "base_seed",
        "episodes",
        "variation_schedule",
        "scenario_independent",
        "seed_domain",
        "mode_schedule",
        "mode_counts",
        "plans",
        "batch_fingerprint",
    }
    if (
        set(payload) != expected_fields
        or
        payload.get("schema") != V4_STORE_BATCH_SCHEMA
        or payload.get("protocol_id") != V4_STORE_MOTION_PROTOCOL_ID
        or payload.get("task_name") != STORE_BOTTLE_TASK_NAME
        or payload.get("scenario_independent") is not True
        or payload.get("mode_schedule") != "episode_index_mod_3"
        or payload.get("batch_fingerprint") != canonical_fingerprint(body)
        or not isinstance(payload.get("plans"), list)
    ):
        raise ValueError("StoreBottle plan batch authentication failed")
    plans = [StoreBottleMultiEntityPlan.from_json(row) for row in payload["plans"]]
    expected = store_bottle_motion_plan_batch(
        base_seed=payload.get("base_seed"),
        variations=payload.get("variation_schedule"),
        plans=plans,
    )
    if any(
        payload.get(key) != expected.get(key)
        for key in expected_fields - {"plans", "batch_fingerprint"}
    ):
        raise ValueError("StoreBottle plan batch fields are inconsistent")
    return plans


def v4_store_runtime_loaders() -> dict[str, Any]:
    return {V4_STORE_RUNTIME_LOADER_ID: load_v4_store_motion_plan_batch}


def v4_store_task_identity_components() -> dict[str, dict[str, str]]:
    semantics = load_store_bottle_semantic_spec()
    motion = load_v4_store_motion_source_protocol()
    intervention = load_v4_store_intervention_protocol()
    return {
        "task_semantics": {
            "schema": semantics.schema,
            "fingerprint": semantics.fingerprint,
        },
        "motion_source": {
            "schema": motion["schema"],
            "fingerprint": motion["fingerprint"],
        },
        "intervention": {
            "schema": intervention["schema"],
            "fingerprint": intervention["fingerprint"],
        },
    }


def build_v4_store_task_scoped_plan_batch(
    *,
    base_seed: int,
    variations: list[int],
    plans: list[StoreBottleMultiEntityPlan],
) -> dict[str, Any]:
    from .eval_set import build_task_scoped_identity, build_task_scoped_plan_batch

    runtime_batch = store_bottle_motion_plan_batch(
        base_seed=base_seed,
        variations=variations,
        plans=plans,
    )
    identity = build_task_scoped_identity(
        task_name=STORE_BOTTLE_TASK_NAME,
        components=v4_store_task_identity_components(),
    )
    return build_task_scoped_plan_batch(
        task_name=STORE_BOTTLE_TASK_NAME,
        task_identity=identity,
        runtime_loader=V4_STORE_RUNTIME_LOADER_ID,
        runtime_batch=runtime_batch,
    )


def v4_store_trigger_authentication(policy_steps: int) -> dict[str, Any]:
    protocol = load_v4_store_intervention_protocol()
    if (
        isinstance(policy_steps, bool)
        or not isinstance(policy_steps, int)
        or policy_steps <= max(V4_STORE_TRIGGER_STEPS.values())
    ):
        raise ValueError("StoreBottle V4 triggers lie outside the policy clock")
    return {
        "schema": V4_STORE_TRIGGER_AUTHENTICATION_SCHEMA,
        "protocol_schema": protocol["schema"],
        "protocol_fingerprint": protocol["fingerprint"],
        "clock": protocol["clock"],
        "triggers": {
            name: {
                "arm": protocol["triggers"][name]["arm"],
                "frame": protocol["triggers"][name]["frame"],
                "skill_label": 0,
                "local_tick": V4_STORE_TRIGGER_STEPS[name],
                "global_tick": V4_STORE_TRIGGER_STEPS[name],
                "trigger_step": V4_STORE_TRIGGER_STEPS[name],
                "expected_gripper_state": "open",
            }
            for name in V4_STORE_ENTITY_ORDER
        },
        "validated_against_policy_horizon": policy_steps,
        "policy_result_fields_read": False,
        "result_based_retuning": False,
    }


def v4_store_task_class() -> Any:
    """Import the tracked live semantic task only in a simulator process."""

    from .store_bottle_live_v4 import BimanualPutBottleInFridgeSemanticV4

    return BimanualPutBottleInFridgeSemanticV4


def _semantic_roots(task: Any) -> dict[str, Any]:
    getter = getattr(task, "semantic_motion_roots", None)
    if not callable(getter):
        raise RuntimeError("StoreBottle V4 task does not expose semantic roots")
    roots = getter()
    if not isinstance(roots, dict) or set(roots) != set(V4_STORE_ENTITY_ORDER):
        raise RuntimeError("StoreBottle V4 semantic roots are incomplete")
    handles = []
    for entity in V4_STORE_ENTITY_ORDER:
        root = roots[entity]
        get_name = getattr(root, "get_name", None)
        if not callable(get_name) or str(get_name()) != V4_STORE_ENTITY_ROOTS[entity]:
            raise RuntimeError(f"StoreBottle V4 {entity} root identity is invalid")
        get_handle = getattr(root, "get_handle", None)
        handles.append(int(get_handle()) if callable(get_handle) else id(root))
    if len(set(handles)) != 2:
        raise RuntimeError("StoreBottle V4 roots must be independent objects")
    return roots


def _semantic_tree_state(task: Any, roots: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Capture each task object in its owning entity frame (or world)."""

    spec = load_store_bottle_semantic_spec()
    base = task.get_base()
    objects = list(base.get_objects_in_tree(exclude_base=False))
    membership: dict[str, str] = {}
    for group in spec.entity_groups:
        for name in group.members:
            if name in membership:
                raise RuntimeError("StoreBottle semantic entity subtrees overlap")
            membership[name] = group.name
    rows = []
    seen = set()
    for value in objects:
        name = str(value.get_name())
        value_type = str(value.get_type())
        key = (name, value_type)
        if key in seen:
            raise RuntimeError("StoreBottle task tree has duplicate stable identities")
        seen.add(key)
        parent = value.get_parent() if callable(getattr(value, "get_parent", None)) else None
        parent_name = (
            str(parent.get_name())
            if parent is not None and callable(getattr(parent, "get_name", None))
            else None
        )
        entity = membership.get(name)
        pose = np.asarray(
            value.get_pose(relative_to=roots[entity]) if entity else value.get_pose(),
            dtype=np.float64,
        )
        row = {
            "name": name,
            "type": value_type,
            "parent": parent_name,
            "entity": entity,
            "pose_frame": entity or "world",
            "pose": pose.tolist(),
        }
        joint = getattr(value, "get_joint_position", None)
        if callable(joint):
            row["joint_position"] = float(joint())
        rows.append(row)
    rows.sort(key=lambda row: (row["name"], row["type"], row["parent"] or ""))
    expected_members = {
        name for group in spec.entity_groups for name in group.members
    }
    actual_members = {row["name"] for row in rows if row["entity"] is not None}
    if expected_members != actual_members:
        missing = sorted(expected_members - actual_members)
        raise RuntimeError(f"StoreBottle semantic subtree members missing: {missing}")
    return rows


def _compare_semantic_tree(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(expected) != len(actual):
        return {"matched": False, "reason": "row_count"}
    maximum_translation = 0.0
    maximum_rotation = 0.0
    maximum_joint = 0.0
    for lhs, rhs in zip(expected, actual):
        structural = ("name", "type", "parent", "entity", "pose_frame")
        if any(lhs.get(key) != rhs.get(key) for key in structural):
            return {"matched": False, "reason": "topology"}
        left_pose = np.asarray(lhs.get("pose"), dtype=np.float64)
        right_pose = np.asarray(rhs.get("pose"), dtype=np.float64)
        if left_pose.shape != (7,) or right_pose.shape != (7,):
            return {"matched": False, "reason": "pose_shape"}
        maximum_translation = max(
            maximum_translation,
            float(np.linalg.norm(left_pose[:3] - right_pose[:3])),
        )
        maximum_rotation = max(
            maximum_rotation,
            _quaternion_angle_xyzw(left_pose[3:], right_pose[3:]),
        )
        left_joint = lhs.get("joint_position")
        right_joint = rhs.get("joint_position")
        if (left_joint is None) != (right_joint is None):
            return {"matched": False, "reason": "joint_schema"}
        if left_joint is not None:
            maximum_joint = max(maximum_joint, abs(left_joint - right_joint))
    matched = bool(
        maximum_translation <= V4_STORE_POSE_ATOL
        and maximum_rotation <= V4_STORE_ROTATION_ATOL_RAD
        and maximum_joint <= V4_STORE_POSE_ATOL
    )
    return {
        "matched": matched,
        "reason": None if matched else "numeric_state",
        "maximum_translation_error_m": maximum_translation,
        "maximum_rotation_error_rad": maximum_rotation,
        "maximum_joint_error": maximum_joint,
    }


def _low_dim_frame_audit(task: Any) -> dict[str, Any]:
    state = np.asarray(task.get_low_dim_state(), dtype=np.float64)
    roots = _semantic_roots(task)
    bottle = np.asarray(task.bottle.get_pose(), dtype=np.float64)
    fridge = np.asarray(roots["fridge"].get_pose(), dtype=np.float64)
    expected = np.concatenate((bottle, fridge))
    matched = bool(
        state.shape == (14,)
        and np.all(np.isfinite(state))
        and np.allclose(state, expected, atol=V4_STORE_POSE_ATOL, rtol=0.0)
    )
    return {
        "frames": ["bottle", "fridge"],
        "state": state.tolist(),
        "expected": expected.tolist(),
        "matched": matched,
    }


def _waypoint_positions(task: Any) -> dict[str, np.ndarray]:
    waypoints = getattr(task, "_waypoints", None)
    if not isinstance(waypoints, list) or not waypoints:
        raise RuntimeError("StoreBottle Task.validate() did not materialize waypoints")
    positions: dict[str, np.ndarray] = {}
    for waypoint in waypoints:
        name = getattr(waypoint, "name", None)
        getter = getattr(waypoint, "get_waypoint_object", None)
        if not isinstance(name, str) or not callable(getter):
            raise RuntimeError("StoreBottle waypoint identity is unavailable")
        positions[name] = np.asarray(getter().get_position(), dtype=np.float64)
    expected = {f"waypoint{index}" for index in range(9)}
    if set(positions) != expected:
        raise RuntimeError("StoreBottle waypoint set differs from waypoint0..8")
    return positions


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-16:
        return float(np.linalg.norm(point - start))
    amount = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + amount * delta)))


def _scene_safety_audit(task: Any) -> dict[str, Any]:
    motion = load_v4_store_motion_source_protocol(verify_semantics_file=False)
    positions = _waypoint_positions(task)
    pairs = (("waypoint0", "waypoint4"), ("waypoint1", "waypoint5"),
             ("waypoint2", "waypoint6"), ("waypoint3", "waypoint7"),
             ("waypoint3", "waypoint8"))
    paired = {
        f"{right}:{left}": float(np.linalg.norm(positions[right] - positions[left]))
        for right, left in pairs
    }
    bottle_position = np.asarray(task.bottle.get_position(), dtype=np.float64)
    door_polyline = [positions[f"waypoint{index}"] for index in range(4)]
    clearance = min(
        _point_segment_distance(bottle_position, start, end)
        for start, end in zip(door_polyline[:-1], door_polyline[1:])
    )
    min_pair = min(paired.values())
    pair_threshold = motion["candidate_generation"][
        "minimum_paired_arm_waypoint_distance_m"
    ]
    door_threshold = motion["candidate_generation"][
        "minimum_bottle_to_fridge_door_sweep_proxy_clearance_m"
    ]
    return {
        "waypoint_names": sorted(positions),
        "task_validate_reachability": True,
        "paired_arm_waypoint_distances_m": paired,
        "minimum_paired_arm_waypoint_distance_m": min_pair,
        "minimum_required_paired_arm_waypoint_distance_m": pair_threshold,
        "bottle_to_fridge_door_sweep_proxy_clearance_m": clearance,
        "minimum_required_door_sweep_proxy_clearance_m": door_threshold,
        "obvious_arm_crossing_free": min_pair >= pair_threshold,
        "door_sweep_proxy_clear": clearance >= door_threshold,
        "passed": min_pair >= pair_threshold and clearance >= door_threshold,
    }


def stage_v4_store_motion_plan(
    environment: Any,
    task_class: Any,
    *,
    episode_index: int,
    episode_seed: int,
    variation: int,
    max_attempts: int = 100,
    source_max_attempts: int = 20,
) -> StoreBottleMultiEntityPlan:
    """Stage one independently movable StoreBottle A/B plan without policy data."""

    motion = load_v4_store_motion_source_protocol()
    intervention = load_v4_store_intervention_protocol()
    if task_class is not v4_store_task_class():
        raise ValueError("StoreBottle V4 staging requires the SemanticV4 task class")
    if (
        max_attempts != motion["goal_sampling_max_attempts"]
        or source_max_attempts != motion["source_selection_max_attempts"]
    ):
        raise ValueError("StoreBottle V4 staging budgets differ from protocol")
    from .runtime import (
        _certify_goal_b,
        _certify_source_a,
        _is_expected_placement_error,
        _robot_external_collision_pairs,
        _source_reconstruction_audit,
        _source_seed,
        _source_state_snapshot,
        _stable_collision_pair_records,
        _waypoint_cache_evidence,
        _workspace_boundary_accepts_current_root,
        initialize_fresh_task_generation,
    )

    source_attempt_rows = []
    certified = None
    source_certification = None
    source_seed = None
    for attempt in range(1, source_max_attempts + 1):
        candidate_source_seed = _source_seed(episode_seed, variation, attempt)
        task_environment, descriptions, _observation, generation = (
            initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=candidate_source_seed,
                variation=variation,
                verify_instance=False,
            )
        )
        scene = task_environment._scene
        try:
            roots = _semantic_roots(scene.task)
            low_dim_audit = _low_dim_frame_audit(scene.task)
            if not low_dim_audit["matched"]:
                raise RuntimeError("StoreBottle V4 low-dimensional frames are invalid")
            snapshot, certification = _certify_source_a(
                scene,
                descriptions,
                generation,
            )
            tree = _semantic_tree_state(scene.task, roots)
        except Exception as error:
            expected = _is_expected_placement_error(error) or str(error) in {
                "reset(false) source workspace placement did not succeed",
                "reset(false) source robot is in collision",
                "reset(false) source placement collision audit failed",
            }
            if not expected:
                raise
            source_attempt_rows.append(
                {
                    "attempt": attempt,
                    "source_seed": candidate_source_seed,
                    "accepted": False,
                    "reason": str(error) or type(error).__name__,
                    "fresh_task_generation": generation,
                }
            )
            continue
        source_attempt_rows.append(
            {
                "attempt": attempt,
                "source_seed": candidate_source_seed,
                "accepted": True,
                "reason": None,
                "fresh_task_generation": generation,
            }
        )
        certified = snapshot
        certified["store_semantic_tree"] = tree
        certified["store_root_poses"] = {
            name: np.asarray(roots[name].get_pose(), dtype=np.float64).tolist()
            for name in V4_STORE_ENTITY_ORDER
        }
        source_certification = certification
        source_seed = candidate_source_seed
        break
    if certified is None or source_seed is None:
        raise RuntimeError("could not certify a StoreBottle V4 source A")

    task_environment, descriptions, _observation, selected_generation = (
        initialize_fresh_task_generation(
            environment,
            task_class,
            episode_seed=source_seed,
            variation=variation,
            verify_instance=False,
        )
    )
    selected_scene = task_environment._scene
    selected_roots = _semantic_roots(selected_scene.task)
    selected = _source_state_snapshot(selected_scene, descriptions)
    reconstruction = _source_reconstruction_audit(certified, selected)
    selected_tree = _semantic_tree_state(selected_scene.task, selected_roots)
    selected_root_poses = {
        name: np.asarray(selected_roots[name].get_pose(), dtype=np.float64).tolist()
        for name in V4_STORE_ENTITY_ORDER
    }
    if (
        not reconstruction["passed"]
        or _compare_semantic_tree(certified["store_semantic_tree"], selected_tree)[
            "matched"
        ]
        is not True
        or any(
            not np.allclose(
                certified["store_root_poses"][name],
                selected_root_poses[name],
                atol=V4_STORE_POSE_ATOL,
                rtol=0.0,
            )
            for name in V4_STORE_ENTITY_ORDER
        )
        or _waypoint_cache_evidence(selected_scene.task)["state"] != "none"
    ):
        raise RuntimeError("StoreBottle selected source did not reconstruct A")

    mode = store_mode_for_episode(episode_index)
    moved_entities = V4_STORE_MODE_MOVED_ENTITIES[mode]
    goal_attempt_rows = []
    accepted = None
    for attempt in range(1, max_attempts + 1):
        task_environment, descriptions, _observation, generation = (
            initialize_fresh_task_generation(
                environment,
                task_class,
                episode_seed=source_seed,
                variation=variation,
                verify_instance=False,
            )
        )
        scene = task_environment._scene
        task = scene.task
        roots = _semantic_roots(task)
        current = _source_state_snapshot(scene, descriptions)
        current_reconstruction = _source_reconstruction_audit(selected, current)
        current_tree = _semantic_tree_state(task, roots)
        current_root_poses = {
            name: np.asarray(roots[name].get_pose(), dtype=np.float64)
            for name in V4_STORE_ENTITY_ORDER
        }
        if (
            not current_reconstruction["passed"]
            or not _compare_semantic_tree(selected_tree, current_tree)["matched"]
            or _waypoint_cache_evidence(task)["state"] != "none"
        ):
            raise RuntimeError("StoreBottle B retry did not reconstruct selected A")
        candidate_seed = int(
            (episode_seed * 1_000_003 + variation * 9_176 + attempt * 7_919)
            % _UINT32_MODULUS
        )
        goals = {}
        for entity in V4_STORE_ENTITY_ORDER:
            source_pose = current_root_poses[entity]
            goals[entity] = (
                sample_v4_store_entity_goal_pose(
                    source_pose,
                    candidate_seed,
                    entity=entity,
                )
                if entity in moved_entities
                else source_pose.copy()
            )
        source_collisions = _robot_external_collision_pairs(scene, scene.robot)
        try:
            for entity in V4_STORE_ENTITY_ORDER:
                roots[entity].set_pose(goals[entity])
                # RLBench's StoreBottle placement boundary was authored for the
                # bottle shape.  The fixed fridge intentionally extends beyond
                # that spawn boundary even at its valid nominal pose, so applying
                # the boundary test to ``fridge_base`` would reject every legal
                # small fridge displacement.  Bottle candidates use the original
                # geometry gate; fridge candidates use the preregistered 2--5 cm
                # source-local envelope followed by full Task.validate().
                if (
                    entity == "bottle"
                    and entity in moved_entities
                    and not _workspace_boundary_accepts_current_root(
                        scene,
                        task.bottle,
                    )
                ):
                    raise RuntimeError("bottle goal does not fit workspace")
            candidate_tree = _semantic_tree_state(task, roots)
            tree_audit = _compare_semantic_tree(current_tree, candidate_tree)
            if not tree_audit["matched"]:
                raise RuntimeError("entity-root motion did not preserve semantic subtrees")
            low_dim_goal = _low_dim_frame_audit(task)
            if not low_dim_goal["matched"]:
                raise RuntimeError("goal low-dimensional frames are inconsistent")
            goal_collisions = _robot_external_collision_pairs(scene, scene.robot)
            if frozenset(goal_collisions) - frozenset(source_collisions):
                raise RuntimeError("goal introduces new robot collision pairs")
            certification = _certify_goal_b(scene, descriptions)
            safety = _scene_safety_audit(task)
            if not safety["passed"]:
                raise RuntimeError("goal fails StoreBottle scene-safety gates")
            goal_collisions = _robot_external_collision_pairs(scene, scene.robot)
            if frozenset(goal_collisions) - frozenset(source_collisions):
                raise RuntimeError("validated goal leaves new robot collision pairs")
        except Exception as error:
            if not (_is_expected_placement_error(error) or isinstance(error, RuntimeError)):
                raise
            goal_attempt_rows.append(
                {
                    "attempt": attempt,
                    "candidate_seed": candidate_seed,
                    "accepted": False,
                    "reason": str(error) or type(error).__name__,
                    "fresh_task_generation": generation,
                }
            )
            continue
        accepted = {
            "attempt": attempt,
            "candidate_seed": candidate_seed,
            "goals": goals,
            "goal_certification": certification,
            "scene_safety": safety,
            "semantic_tree_motion": tree_audit,
            "source_collisions": _stable_collision_pair_records(source_collisions),
            "goal_collisions": _stable_collision_pair_records(goal_collisions),
            "fresh_task_generation": generation,
        }
        goal_attempt_rows.append(
            {
                "attempt": attempt,
                "candidate_seed": candidate_seed,
                "accepted": True,
                "reason": None,
                "fresh_task_generation": generation,
            }
        )
        break
    if accepted is None:
        reason_counts: dict[str, int] = {}
        for row in goal_attempt_rows:
            reason = str(row.get("reason"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        raise RuntimeError(
            "could not certify a StoreBottle V4 multi-entity B; "
            f"rejections={reason_counts}"
        )

    entities = tuple(
        StoreBottleEntityMotion(
            name=name,
            root_name=V4_STORE_ENTITY_ROOTS[name],
            frame_name=V4_STORE_ENTITY_FRAMES[name],
            source_pose=tuple(selected_root_poses[name]),
            goal_pose=tuple(accepted["goals"][name]),
            moved=name in moved_entities,
            candidate_seed=(accepted["candidate_seed"] if name in moved_entities else None),
        )
        for name in V4_STORE_ENTITY_ORDER
    )
    validation = {
        "schema": V4_STORE_PLAN_VALIDATION_SCHEMA,
        "source_seed": source_seed,
        "source_selection_max_attempts": source_max_attempts,
        "source_seed_attempts": source_attempt_rows,
        "selected_source_fresh_task_generation": selected_generation,
        "selected_source_reconstruction": reconstruction,
        "source_certification": source_certification,
        "source_waypoint_validated": True,
        "goal_waypoint_validated": True,
        "goal_sampling_max_attempts": max_attempts,
        "sampling_attempts": accepted["attempt"],
        "selected_candidate_seed": accepted["candidate_seed"],
        "goal_candidate_attempts": goal_attempt_rows,
        "source_root_poses": selected_root_poses,
        "source_task_tree_relative_state": selected["task_tree"],
        "source_store_semantic_tree": selected_tree,
        "source_robot_numeric_state": selected["robot_numeric_state"],
        "source_stable_grasp_state": selected["stable_grasp_state"],
        "source_robot_external_collision_pairs": selected[
            "robot_external_collision_pairs"
        ],
        "task_semantic_signature": selected["task_semantics"],
        "task_descriptions": selected["descriptions"],
        "goal_certification": accepted["goal_certification"],
        "scene_safety": accepted["scene_safety"],
        "semantic_tree_motion": accepted["semantic_tree_motion"],
        "goal_robot_external_collision_pairs": accepted["goal_collisions"],
        "motion_source_schema": motion["schema"],
        "motion_source_fingerprint": motion["fingerprint"],
        "intervention_schema": intervention["schema"],
        "intervention_fingerprint": intervention["fingerprint"],
        "selection_authority": "scene_validity_only",
        "policy_result_fields_read": False,
    }
    return StoreBottleMultiEntityPlan(
        task_name=STORE_BOTTLE_TASK_NAME,
        episode_index=episode_index,
        episode_seed=episode_seed,
        variation=variation,
        mode=mode,
        entities=entities,
        source_low_dim_state=tuple(selected["low_dim_state"]),
        validation=validation,
    )


def bind_v4_store_source_plan(
    task_environment: Any,
    plan: StoreBottleMultiEntityPlan,
    *,
    descriptions: Any,
    fresh_task_generation: dict[str, Any],
) -> dict[str, Any]:
    """Bind a formal fresh reset to both independently registered source roots."""

    from .runtime import (
        _source_reconstruction_audit,
        _source_state_snapshot,
        _validate_fresh_task_generation_evidence,
        _waypoint_cache_evidence,
    )

    _validate_fresh_task_generation_evidence(
        fresh_task_generation,
        episode_seed=plan.validation["source_seed"],
        variation=plan.variation,
        task_name=plan.task_name,
        verify_instance=False,
    )
    scene = task_environment._scene
    task = scene.task
    roots = _semantic_roots(task)
    actual = _source_state_snapshot(scene, descriptions)
    expected = {
        "task_name": plan.task_name,
        "root_pose": np.asarray(plan.entity("bottle").source_pose),
        "low_dim_state": np.asarray(plan.source_low_dim_state),
        "task_tree": plan.validation["source_task_tree_relative_state"],
        "task_semantics": plan.validation["task_semantic_signature"],
        "descriptions": plan.validation["task_descriptions"],
        "robot_numeric_state": plan.validation["source_robot_numeric_state"],
        "stable_grasp_state": plan.validation["source_stable_grasp_state"],
        "robot_external_collision_pairs": plan.validation[
            "source_robot_external_collision_pairs"
        ],
        "velocity_summary": actual["velocity_summary"],
    }
    reconstruction = _source_reconstruction_audit(expected, actual)
    root_matches = {
        name: bool(
            np.allclose(
                roots[name].get_pose(),
                plan.entity(name).source_pose,
                atol=V4_STORE_POSE_ATOL,
                rtol=0.0,
            )
        )
        for name in V4_STORE_ENTITY_ORDER
    }
    tree = _compare_semantic_tree(
        plan.validation["source_store_semantic_tree"],
        _semantic_tree_state(task, roots),
    )
    low_dim = _low_dim_frame_audit(task)
    if (
        not reconstruction["passed"]
        or not all(root_matches.values())
        or not tree["matched"]
        or not low_dim["matched"]
        or _waypoint_cache_evidence(task)["state"] != "none"
    ):
        raise RuntimeError("formal StoreBottle V4 source A binding failed")
    return {
        "schema": "dynamac-store-bottle-formal-source-a-binding-v4",
        "required": True,
        "formal_source_bound": True,
        "task_name": plan.task_name,
        "source_seed": plan.validation["source_seed"],
        "root_matches": root_matches,
        "task_semantics_matched": True,
        "task_tree_matched": True,
        "low_dim_frames_matched": True,
        "deterministic_source_reconstruction": reconstruction,
        "formal_task_validate_calls": 0,
        "plan_fingerprint": plan.fingerprint(),
        "fresh_task_generation": fresh_task_generation,
    }


class StoreBottleMultiEntityController:
    """Apply each moved StoreBottle root once at its own frozen trigger."""

    def __init__(self, *, plan: StoreBottleMultiEntityPlan, scenario: str):
        if scenario not in {"static", "teleport"}:
            raise ValueError("StoreBottle V4 supports only static/teleport")
        self.plan = plan
        self.scenario = scenario
        self._applied: set[str] = set()
        self._source_binding: dict[str, Any] | None = None

    @property
    def required_entities(self) -> tuple[str, ...]:
        return self.plan.moved_entities if self.scenario == "teleport" else ()

    def bind_source(
        self,
        task_environment: Any,
        *,
        descriptions: Any,
        fresh_task_generation: dict[str, Any],
    ) -> dict[str, Any]:
        self._source_binding = bind_v4_store_source_plan(
            task_environment,
            self.plan,
            descriptions=descriptions,
            fresh_task_generation=fresh_task_generation,
        )
        return dict(self._source_binding)

    def apply(
        self,
        task_environment: Any,
        observation: Any,
        *,
        policy_step: int,
    ) -> tuple[Any, list[dict[str, Any]]]:
        if self._source_binding is None:
            raise RuntimeError("StoreBottle controller source was not bound")
        if self.scenario == "static":
            return observation, []
        due = [
            entity
            for entity in self.required_entities
            if entity not in self._applied
            and V4_STORE_TRIGGER_STEPS[entity] == policy_step
        ]
        missed = [
            entity
            for entity in self.required_entities
            if entity not in self._applied
            and V4_STORE_TRIGGER_STEPS[entity] < policy_step
        ]
        if missed:
            raise RuntimeError(f"StoreBottle entity trigger was skipped: {missed}")
        if not due:
            return observation, []
        from .runtime import _robot_external_collision_pairs

        scene = task_environment._scene
        task = scene.task
        roots = _semantic_roots(task)
        before_tree = _semantic_tree_state(task, roots)
        before_collisions = _robot_external_collision_pairs(scene, scene.robot)
        before_root_poses = {
            name: np.asarray(roots[name].get_pose(), dtype=np.float64)
            for name in V4_STORE_ENTITY_ORDER
        }
        events = []
        for entity in due:
            expected_before = (
                self.plan.entity(entity).source_pose
                if entity not in self._applied
                else self.plan.entity(entity).goal_pose
            )
            if not np.allclose(
                before_root_poses[entity],
                expected_before,
                atol=V4_STORE_POSE_ATOL,
                rtol=0.0,
            ):
                raise RuntimeError(f"StoreBottle {entity} drifted before intervention")
            roots[entity].set_pose(self.plan.entity(entity).goal_pose)
            self._applied.add(entity)
        after_tree = _semantic_tree_state(task, roots)
        tree_audit = _compare_semantic_tree(before_tree, after_tree)
        after_collisions = _robot_external_collision_pairs(scene, scene.robot)
        new_collisions = sorted(
            set(after_collisions) - set(before_collisions),
            key=lambda row: tuple(row),
        )
        low_dim = _low_dim_frame_audit(task)
        if not tree_audit["matched"] or not low_dim["matched"] or new_collisions:
            raise RuntimeError(
                "StoreBottle intervention failed subtree/frame/current-collision audit"
            )
        observation = task_environment.get_observation()
        for entity in due:
            entity_plan = self.plan.entity(entity)
            actual = np.asarray(roots[entity].get_pose(), dtype=np.float64)
            position_error = float(
                np.linalg.norm(actual[:3] - np.asarray(entity_plan.goal_pose)[:3])
            )
            rotation_error = _quaternion_angle_xyzw(
                actual[3:], np.asarray(entity_plan.goal_pose)[3:]
            )
            effective = bool(
                position_error <= V4_STORE_POSE_ATOL
                and rotation_error <= V4_STORE_ROTATION_ATOL_RAD
            )
            if not effective:
                raise RuntimeError("StoreBottle entity did not reach registered B")
            events.append(
                {
                    "kind": "teleport_store_entity",
                    "applied": True,
                    "complete": True,
                    "protocol_effective": True,
                    "entity": entity,
                    "step": policy_step,
                    "trigger_step": V4_STORE_TRIGGER_STEPS[entity],
                    "source_pose": list(entity_plan.source_pose),
                    "goal_pose": list(entity_plan.goal_pose),
                    "actual_pose": actual.tolist(),
                    "geometry": store_entity_geometry(
                        entity_plan.source_pose,
                        entity_plan.goal_pose,
                    ),
                    "position_error_m": position_error,
                    "rotation_error_rad": rotation_error,
                    "semantic_tree_preserved": tree_audit,
                    "new_robot_collision_pairs": [],
                    "policy_observation_refreshed": True,
                }
            )
        return observation, events

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "schema": "dynamac-store-bottle-multi-entity-controller-v4",
            "protocol_id": V4_STORE_MOTION_PROTOCOL_ID,
            "scenario": self.scenario,
            "episode_mode": self.plan.mode,
            "required_entities": list(self.required_entities),
            "triggers": {
                name: V4_STORE_TRIGGER_STEPS[name] for name in V4_STORE_ENTITY_ORDER
            },
            "independent_roots": dict(V4_STORE_ENTITY_ROOTS),
            "applications_per_moved_entity": 1,
        }
