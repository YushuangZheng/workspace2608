"""Reconstruct omitted M2 stream marginals from frozen policy state and models."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from essay2608.policy import BimanualDynaMAC, DynaMAC, transform_marginal
from integrations.rlbench.iclr2027.task_registry import experiment_task
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    bimanual_observations_from_rlbench,
    unimanual_observation_from_rlbench,
)


SINGLE_MODELS = INTEGRATION_ROOT / "models" / "iclr2027" / "dynamac"
BIMANUAL_MODELS = INTEGRATION_ROOT / "models" / "v4"


def _reference_state(
    policy_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    arm: str,
    bimanual: bool,
) -> Mapping[str, Any]:
    value = metadata.get("reference_state")
    if isinstance(value, Mapping):
        return value
    value = policy_state.get("reference_state_if_available")
    if bimanual and isinstance(value, Mapping):
        value = value.get(arm)
    if not isinstance(value, Mapping):
        raise ValueError("recorded stream metadata has no reference state")
    return value


def _observations(feature: Mapping[str, Any], task: Any) -> dict[str, Any]:
    arms = feature["arms"]
    wire = SimpleNamespace(
        task_low_dim_state=np.asarray(feature["task_state"], dtype=np.float64)
    )
    if not task.spec.bimanual:
        wire.gripper_pose = np.asarray(
            arms["single"]["ee_pose_xyzw"], dtype=np.float64
        )
        return {
            "single": unimanual_observation_from_rlbench(wire, task.spec)
        }
    wire.left = SimpleNamespace(
        gripper_pose=np.asarray(arms["left"]["ee_pose_xyzw"], dtype=np.float64)
    )
    wire.right = SimpleNamespace(
        gripper_pose=np.asarray(arms["right"]["ee_pose_xyzw"], dtype=np.float64)
    )
    left, right = bimanual_observations_from_rlbench(wire, task.spec)
    left, right = BimanualDynaMAC._synchronous_observations(left, right)
    return {"left": left, "right": right}


@lru_cache(maxsize=None)
def _cached_policies(task_id: str) -> tuple[dict[str, DynaMAC], dict[str, str]]:
    task = experiment_task(task_id)
    if task.spec.bimanual:
        root = BIMANUAL_MODELS / task.task_id
        policies = {
            "left": DynaMAC.load(root / "left.npz"),
            "right": DynaMAC.load(root / "right.npz"),
        }
    else:
        root = SINGLE_MODELS / task.task_id
        policies = {"single": DynaMAC.load(root / "model.npz")}
    return policies, {
        arm: policy.fingerprint() for arm, policy in policies.items()
    }


def _stream_diagnostic(
    policy: DynaMAC,
    observation: Any,
    *,
    skill_index: int,
    progress: int,
    mode: int,
) -> dict[str, Any]:
    """Rebuild only the stream statistics persisted by the current runner.

    Full ``query_state`` also solves the iterative manifold PoE pose.  M2 does
    not consume that joint pose, so invoking the solver for every historical
    cycle would add substantial computation without reconstructing any field
    used by the monitor.
    """

    if skill_index < 0 or skill_index >= len(policy.skills):
        raise IndexError("recorded skill index is outside the frozen model")
    skill = policy.skills[skill_index]
    if progress < 0 or progress >= skill.duration:
        raise IndexError("recorded progress is outside the frozen skill")
    if mode < 0 or mode >= len(skill.mode_priors):
        raise IndexError("recorded mode is outside the frozen skill")
    marginals = []
    for name in skill.selected_frames:
        stream = skill.streams[name]
        mask_index = 0 if policy.config.link_mask_scope == "skill_majority" else progress
        if not stream.is_selected(mode) or not stream.is_active(mode, mask_index):
            continue
        marginals.append(
            transform_marginal(
                name,
                policy._frame_pose(name, observation),
                stream.mean[mode, progress],
                stream.covariance[mode, progress],
                diagonalize=policy.config.diagonalize_transformed_covariance,
            )
        )
    if not marginals:
        raise RuntimeError("recorded state has no active frozen motion stream")
    log_scores = np.asarray(
        [-np.linalg.slogdet(item.covariance)[1] for item in marginals],
        dtype=np.float64,
    )
    determinant_scores = np.exp(log_scores - np.max(log_scores))
    normalized = determinant_scores / np.sum(determinant_scores)
    return {
        "selected_frames": list(skill.selected_frames),
        "active_frames": [item.frame for item in marginals],
        "poe_weights": {
            item.frame: float(weight)
            for item, weight in zip(marginals, normalized, strict=True)
        },
        "marginal_means": {item.frame: item.mean.tolist() for item in marginals},
        "marginal_covariances": {
            item.frame: item.covariance.tolist() for item in marginals
        },
    }


def reconstruct_stream_marginals(
    task_id: str,
    cycles: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return copied features with authenticated world-frame marginals.

    Old A-only calibration logs already contain the emitted StateId, active
    mask, PoE weights, robot state, and task frames, but predate persistence of
    the corresponding world-frame Gaussian means/covariances.  The omitted
    deterministic quantities are reconstructed from the same frozen policy
    checkpoint.  Every cycle must reproduce the recorded active mask and PoE
    weights before its marginals are accepted.
    """

    if not cycles:
        raise ValueError("cannot reconstruct an empty episode")
    task = experiment_task(task_id)
    policies, fingerprints = _cached_policies(task_id)
    last_skill: dict[str, int] = {}
    rebuilt: list[dict[str, Any]] = []
    reconstructed_arm_cycles = 0
    maximum_weight_error = 0.0
    for cycle_index, row in enumerate(cycles):
        feature = deepcopy(row["feature"])
        observations = _observations(feature, task)
        policy_state = feature["policy_state"]
        stream_root = policy_state.get("stream_metadata")
        if not isinstance(stream_root, Mapping):
            raise ValueError("cycle has no stream metadata")
        for arm, policy in policies.items():
            metadata = stream_root.get(arm) if task.spec.bimanual else stream_root
            if not isinstance(metadata, Mapping):
                raise ValueError(f"cycle has no {arm} stream metadata")
            metadata = dict(metadata)
            state = _reference_state(
                policy_state, metadata, arm, task.spec.bimanual
            )
            skill = int(state["skill"])
            progress = int(state["progress"])
            mode = int(state["mode"])
            if last_skill.get(arm) != skill:
                virtual = f"virtual_skill_{policy.skills[skill].label}"
                policy._virtual_frames[virtual] = observations[arm].ee_pose.copy()
                last_skill[arm] = skill
            diagnostic = _stream_diagnostic(
                policy,
                observations[arm],
                skill_index=skill,
                progress=progress,
                mode=mode,
            )
            if tuple(diagnostic["selected_frames"]) != tuple(
                metadata.get("selected_streams", ())
            ):
                raise ValueError(
                    f"recorded selected streams differ from frozen model at cycle {cycle_index}"
                )
            if tuple(diagnostic["active_frames"]) != tuple(
                metadata.get("active_streams", ())
            ):
                raise ValueError(
                    f"recorded active mask differs from frozen model at cycle {cycle_index}"
                )
            recorded_weights = metadata.get("poe_weights", {})
            if set(recorded_weights) != set(diagnostic["poe_weights"]):
                raise ValueError(
                    f"recorded PoE frames differ from frozen model at cycle {cycle_index}"
                )
            for frame, recorded in recorded_weights.items():
                error = abs(
                    float(recorded) - float(diagnostic["poe_weights"][frame])
                )
                maximum_weight_error = max(maximum_weight_error, error)
                if not np.isclose(
                    float(recorded),
                    float(diagnostic["poe_weights"][frame]),
                    rtol=1.0e-9,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        f"recorded PoE weight differs from frozen model at cycle {cycle_index}"
                    )
            means = diagnostic["marginal_means"]
            covariances = diagnostic["marginal_covariances"]
            if "marginal_means" in metadata:
                for frame in means:
                    if not np.allclose(
                        metadata["marginal_means"][frame],
                        means[frame],
                        rtol=1.0e-9,
                        atol=1.0e-12,
                    ):
                        raise ValueError("stored and reconstructed means differ")
            else:
                metadata["marginal_means"] = means
                reconstructed_arm_cycles += 1
            if "marginal_covariances" in metadata:
                for frame in covariances:
                    if not np.allclose(
                        metadata["marginal_covariances"][frame],
                        covariances[frame],
                        rtol=1.0e-9,
                        atol=1.0e-12,
                    ):
                        raise ValueError("stored and reconstructed covariances differ")
            else:
                metadata["marginal_covariances"] = covariances
            if task.spec.bimanual:
                policy_state["stream_metadata"][arm] = metadata
            else:
                policy_state["stream_metadata"] = metadata
        rebuilt.append(feature)
    return rebuilt, {
        "schema": "essay2608.iclr2027.m2-stream-reconstruction.v1",
        "task": task_id,
        "cycles": len(cycles),
        "reconstructed_arm_cycles": reconstructed_arm_cycles,
        "maximum_poe_weight_absolute_error": maximum_weight_error,
        "model_fingerprints": fingerprints,
        "recorded_active_masks_verified": True,
        "recorded_poe_weights_verified": True,
        "source_cycle_rows_modified": False,
    }


__all__ = ["reconstruct_stream_marginals"]
