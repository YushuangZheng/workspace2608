"""A1 audit of frozen ICLR 2027 task and learned-model assets.

This module validates data/model readability before the public evaluation
layer exists.  It is deliberately not the runtime ``PhysicalEventAuditor``
defined by stage A2 and produces no monitor labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from integrations.rlbench.iclr2027.task_registry import TASKS, ExperimentTask
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT

SCHEMA = "essay2608.iclr2027.a1-asset-audit.v1"
NEW_MODEL_ROOT = INTEGRATION_ROOT / "models" / "iclr2027"
REUSED_BASE_ROOT = INTEGRATION_ROOT / "models" / "phase6_v1"
DEMONSTRATION_ROOT = INTEGRATION_ROOT / "data" / "iclr2027" / "demonstrations"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_roots(task: ExperimentTask) -> tuple[Path, Path]:
    if task.spec.bimanual:
        return (
            REUSED_BASE_ROOT / task.task_id,
            NEW_MODEL_ROOT / "closed_loop" / task.task_id,
        )
    return (
        NEW_MODEL_ROOT / "dynamac" / task.task_id,
        NEW_MODEL_ROOT / "closed_loop" / task.task_id,
    )


def _load_policy(task: ExperimentTask):
    from essay2608.policy import DynaMAC
    from essay2608.policy.closed_loop import ClosedLoopMultiStreamPolicy

    base_root, closed_root = _model_roots(task)
    if task.spec.bimanual:
        base = {
            "left": DynaMAC.load(base_root / "left.npz"),
            "right": DynaMAC.load(base_root / "right.npz"),
        }
    else:
        base = {"single": DynaMAC.load(base_root / "model.npz")}
    policy = ClosedLoopMultiStreamPolicy.load(closed_root, base_policies=base)
    return base_root, closed_root, base, policy


def _assert_finite_mapping(values: dict[str, Any], *, label: str) -> None:
    for key, value in values.items():
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise RuntimeError(f"{label} {key!r} is empty or non-finite")


def audit_task_asset(task: ExperimentTask) -> dict[str, Any]:
    """Validate one task's registry, demonstrations, base policy and sidecar."""

    base_root, closed_root, base, policy = _load_policy(task)
    if tuple(policy.arms) != tuple(base):
        raise RuntimeError(f"{task.task_id}: arm identities disagree")
    per_arm: dict[str, Any] = {}
    for arm, model in policy.task_models.items():
        durations = tuple(int(skill.duration) for skill in base[arm].skills)
        expected_state_count = sum(durations)
        if len(model.states) != expected_state_count:
            raise RuntimeError(
                f"{task.task_id}/{arm}: state count disagrees with durations"
            )
        if len(model.boundaries) != max(0, len(durations) - 1):
            raise RuntimeError(f"{task.task_id}/{arm}: boundary count is not K-1")
        expected_relation_frames = set(task.spec.action_frame_names)
        if task.spec.bimanual:
            expected_relation_frames.add("right_ee" if arm == "left" else "left_ee")
        if set(model.relation_frames) != expected_relation_frames:
            raise RuntimeError(
                f"{task.task_id}/{arm}: relation frames disagree with task spec"
            )
        action_relevance = model.builder_config.get("action_stream_relevance")
        fold_partition = model.builder_config.get("lodo_fold_partition")
        if model.schema_version != 5 or not isinstance(action_relevance, list):
            raise RuntimeError(
                f"{task.task_id}/{arm}: action-relevance sidecar is not schema v5"
            )
        if not isinstance(fold_partition, list) or len(fold_partition) != 5:
            raise RuntimeError(
                f"{task.task_id}/{arm}: action-relevance LODO folds are incomplete"
            )
        for state_id, node in model.states.items():
            if state_id.skill_index >= len(durations):
                raise RuntimeError(f"{task.task_id}/{arm}: invalid state skill index")
            if not 0 <= state_id.local_index < durations[state_id.skill_index]:
                raise RuntimeError(f"{task.task_id}/{arm}: invalid local state index")
            skill = base[arm].skills[state_id.skill_index]
            for mode, candidates in enumerate(node.mode_selected_frames):
                relevant = node.mode_action_relevant_frames[mode]
                if not set(relevant).issubset(candidates):
                    raise RuntimeError(
                        f"{task.task_id}/{arm}: action relevance broadens Eq. 6"
                    )
                active_candidates = {
                    frame
                    for frame in candidates
                    if skill.streams[frame].is_active(mode, state_id.local_index)
                }
                active_relevant = {
                    frame
                    for frame in relevant
                    if skill.streams[frame].is_active(mode, state_id.local_index)
                }
                if active_candidates and not active_relevant:
                    raise RuntimeError(
                        f"{task.task_id}/{arm}: action relevance leaves PoE undefined"
                    )
            if set(node.demo_relation_priors) != set(model.relation_frames):
                raise RuntimeError(
                    f"{task.task_id}/{arm}: relation prior coverage is incomplete"
                )
            _assert_finite_mapping(node.demo_relation_priors, label="relation prior")
            for prior in node.demo_relation_priors.values():
                probabilities = np.asarray(prior, dtype=np.float64)
                if probabilities.ndim != 2 or probabilities.shape[1] != 2:
                    raise RuntimeError(
                        f"{task.task_id}/{arm}: invalid relation prior shape"
                    )
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
                    raise RuntimeError(
                        f"{task.task_id}/{arm}: relation prior is not normalized"
                    )
        for boundary_id, boundary in model.boundaries.items():
            if boundary_id.source_skill + 1 != boundary_id.target_skill:
                raise RuntimeError(f"{task.task_id}/{arm}: non-adjacent skill boundary")
            if not boundary.terminal_window:
                raise RuntimeError(f"{task.task_id}/{arm}: empty terminal window")
            if not boundary.local_completion_model.goal_distributions:
                raise RuntimeError(
                    f"{task.task_id}/{arm}: unreadable local completion goal"
                )
        per_arm[arm] = {
            "durations": list(durations),
            "state_count": len(model.states),
            "boundary_count": len(model.boundaries),
            "relation_frames": list(model.relation_frames),
            "link_events": len(model.link_anchors),
            "link_pending_events": len(model.link_pending_events),
            "unlink_events": len(model.unlink_events),
            "scene_factor_count": sum(
                len(node.scene_factor_models) for node in model.states.values()
            ),
            "action_relevance": {
                "schema_version": model.schema_version,
                "folds": len(fold_partition),
                "skill_modes": len(action_relevance),
                "pruned_skill_modes": sum(
                    set(record["retained_frames"]) != set(record["candidate_frames"])
                    for record in action_relevance
                ),
                "pruned_frame_decisions": sum(
                    not metrics["retained"]
                    for record in action_relevance
                    for metrics in record["frame_metrics"].values()
                ),
            },
        }

    if task.spec.bimanual:
        demonstration = {
            "source": task.demonstration_source,
            "reused_phase6": True,
        }
        model_files = [base_root / "left.npz", base_root / "right.npz"]
    else:
        manifest_path = DEMONSTRATION_ROOT / task.task_id / "collection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("demonstrations")
        if not isinstance(records, list) or len(records) != 5:
            raise RuntimeError(f"{task.task_id}: expected exactly five demonstrations")
        for record in records:
            path = DEMONSTRATION_ROOT / str(record["path"])
            if _sha256(path) != record["sha256"]:
                raise RuntimeError(f"{task.task_id}: demonstration hash mismatch")
        demonstration = {
            "source": task.demonstration_source,
            "reused_phase6": False,
            "count": len(records),
            "manifest_sha256": _sha256(manifest_path),
        }
        model_files = [base_root / "model.npz"]

    model_files.extend(closed_root / f"{arm}.npz" for arm in policy.arms)
    if not all(path.is_file() for path in model_files):
        raise RuntimeError(f"{task.task_id}: one or more learned artifacts are missing")
    return {
        "task_id": task.task_id,
        "base_task": task.base_task,
        "task_level": task.task_level,
        "task_sets": list(task.task_sets),
        "semantic_entities": dict(task.semantic_entities),
        "success_source": task.success_source,
        "compatible_faults": list(task.compatible_faults),
        "demonstration": demonstration,
        "base_model_root": str(base_root.relative_to(INTEGRATION_ROOT)),
        "closed_loop_model_root": str(closed_root.relative_to(INTEGRATION_ROOT)),
        "model_sha256": {
            str(path.relative_to(INTEGRATION_ROOT)): _sha256(path)
            for path in model_files
        },
        "arms": per_arm,
    }


def audit_all_task_assets() -> dict[str, Any]:
    tasks = [audit_task_asset(task) for task in TASKS.values()]
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "task_count": len(tasks),
        "task_set_counts": {
            name: sum(name in task.task_sets for task in TASKS.values())
            for name in ("main10", "stress4", "horizon3", "native6")
        },
        "tasks": tasks,
    }


def audit_live_task(task_environment: Any, task: ExperimentTask) -> dict[str, Any]:
    """Read task success, relation-observation and semantic APIs in one snapshot."""

    scene = getattr(task_environment, "_scene", None)
    live_task = getattr(scene, "task", None)
    robot = getattr(scene, "robot", None)
    if live_task is None or robot is None:
        raise RuntimeError("live RLBench task/robot is unavailable")
    state = np.asarray(live_task.get_low_dim_state(), dtype=np.float64).reshape(-1)
    poses = task.spec.extract_pose_chunks(state)
    configurations = task.spec.extract_entity_configurations(state)
    success, terminate = live_task.success()
    conditions = tuple(getattr(live_task, "_success_conditions", ()))
    if not conditions:
        raise RuntimeError(f"{task.task_id}: no readable RLBench success conditions")
    boundary_root = live_task.boundary_root()

    if task.spec.bimanual:
        grippers = {
            "left": getattr(robot, "left_gripper", None),
            "right": getattr(robot, "right_gripper", None),
        }
    else:
        grippers = {"single": getattr(robot, "gripper", None)}
    relation_observation = {}
    for arm, gripper in grippers.items():
        if gripper is None or not callable(
            getattr(gripper, "get_grasped_objects", None)
        ):
            raise RuntimeError(
                f"{task.task_id}/{arm}: gripper relation API unavailable"
            )
        grasped = tuple(gripper.get_grasped_objects())
        relation_observation[arm] = {
            "grasped_objects": sorted(str(value.get_name()) for value in grasped),
            "proximity_observation_available": getattr(
                gripper, "_proximity_sensor", None
            )
            is not None,
        }
    return {
        "task_low_dim_size": int(state.size),
        "task_low_dim_finite": bool(np.all(np.isfinite(state))),
        "entity_poses": {
            entity: np.asarray(value, dtype=np.float64).tolist()
            for entity, value in poses.items()
        },
        "entity_configurations": {
            entity: {
                name: np.asarray(value, dtype=np.float64).tolist()
                for name, value in fields.items()
            }
            for entity, fields in configurations.items()
        },
        "success": bool(success),
        "terminate": bool(terminate),
        "success_conditions": [type(value).__name__ for value in conditions],
        "boundary_root": str(boundary_root.get_name()),
        "relation_observation": relation_observation,
    }


__all__ = [
    "SCHEMA",
    "audit_all_task_assets",
    "audit_live_task",
    "audit_task_asset",
]
