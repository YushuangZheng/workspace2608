"""RLBench protocol layer for the environment-independent TAPAS segmenter.

The numerical boundary generation, candidate merging, alignment, and bimanual
coordination algorithms live in :mod:`essay2608.policy.tapas_segmentation`.
This module retains only RLBench-owned configuration location, action timing,
signed gripper encoding, reproduction diagnostics, and compatibility exports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from essay2608.policy.tapas_segmentation import (
    TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS,
    TAPAS_CONFIG_DEFAULTS_SOURCE_STATUS,
    TAPAS_DISTANCE_THRESHOLD,
    TAPAS_GRIPPER_THRESHOLD,
    TAPAS_MAX_INDEX_DISTANCE,
    TAPAS_MIN_CLUSTER_LENGTH,
    TAPAS_MIN_END_DISTANCE,
    TAPAS_NUMPY_PORT_SOURCE_STATUS,
    TAPAS_REFERENCE_COMMIT,
    TAPAS_VELOCITY_THRESHOLD,
    BimanualTAPASSegmentation,
    TAPASSegmentation,
    _pose_trajectory,
    align_tapas_boundaries,
    gripper_change_boundaries,
    segment_bimanual_pose_trajectories,
    segment_bimanual_trajectories,
    segment_pose_trajectories,
    segment_trajectories,
    tapas_distance_boundaries,
    tapas_gripper_boundaries,
    tapas_velocity_boundaries,
    translation_action_magnitude,
)
from essay2608.policy.tapas_segmentation import (
    TAPASSegmentationConfig as CoreTAPASSegmentationConfig,
)
from essay2608.policy.tapas_segmentation import (
    _single_grasp_contact_cycle_subset as _single_grasp_contact_cycle_subset,
)

Array = np.ndarray
TAPAS_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "tapas_segmentation.json"
)
TAPAS_ACTION_TIMING = "obs[t] -> obs[t+1], terminal observation repeated"


@dataclass(frozen=True)
class TAPASSegmentationConfig(CoreTAPASSegmentationConfig):
    """Core config with the legacy RLBench-owned default JSON location."""

    @classmethod
    def from_json(
        cls,
        path: str | Path = TAPAS_DEFAULT_CONFIG_PATH,
    ) -> TAPASSegmentationConfig:
        return super().from_json(path)


def load_rlbench_segmentation_config(
    path: str | Path = TAPAS_DEFAULT_CONFIG_PATH,
) -> TAPASSegmentationConfig:
    """Load the RLBench-owned segmentation protocol into the generic config type."""

    return TAPASSegmentationConfig.from_json(path)


def forward_pose_action(ee_pose: Any) -> Array:
    """Pair each observation with the next world pose and repeat the terminal."""

    pose = _pose_trajectory(ee_pose, label="EE pose")
    return np.concatenate((pose[1:], pose[-1:]), axis=0).copy()


def forward_gripper_action(gripper_state: Any, *, signed: bool = True) -> Array:
    """Pair gripper state with ``next_obs`` and repeat the terminal sample.

    RLBench/TAPAS stores a signed action, ``2 * gripper_open - 1``.  Set
    ``signed=False`` only when the downstream consumer intentionally keeps the
    simulator's native ``[0, 1]`` state convention.
    """

    state = np.asarray(gripper_state, dtype=np.float64)
    if state.ndim not in {1, 2} or len(state) < 1:
        raise ValueError("gripper state must have shape [T] or [T, D], T >= 1")
    if not np.all(np.isfinite(state)):
        raise ValueError("gripper state contains non-finite values")
    if signed and (np.any(state < 0.0) or np.any(state > 1.0)):
        raise ValueError("signed RLBench conversion requires gripper states in [0, 1]")
    action = np.concatenate((state[1:], state[-1:]), axis=0).copy()
    return 2.0 * action - 1.0 if signed else action


def next_observation_actions(
    ee_pose: Any,
    gripper_state: Any,
    *,
    signed_gripper: bool = True,
) -> tuple[Array, Array]:
    """Return sample-aligned next-observation pose and gripper actions."""

    pose_action = forward_pose_action(ee_pose)
    gripper_action = forward_gripper_action(gripper_state, signed=signed_gripper)
    if len(pose_action) != len(gripper_action):
        raise ValueError("EE and gripper trajectories must have the same length")
    return pose_action, gripper_action


def save_bimanual_segmentation_debug_plot(
    left_ee_poses: Sequence[Any],
    right_ee_poses: Sequence[Any],
    left_gripper_states: Sequence[Any],
    right_gripper_states: Sequence[Any],
    segmentation: BimanualTAPASSegmentation,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save the RLBench reproduction's velocity/gripper boundary plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("debug plotting requires matplotlib") from exc

    count = len(left_ee_poses)
    collections = (
        (left_ee_poses, left_gripper_states, segmentation.left.boundaries, "left"),
        (right_ee_poses, right_gripper_states, segmentation.right.boundaries, "right"),
    )
    figure, axes = plt.subplots(2, count, squeeze=False, figsize=(3.2 * count, 5.2))
    for row_index, (poses, grippers, boundaries, arm) in enumerate(collections):
        if len(poses) != count or len(grippers) != count:
            raise ValueError("debug plot inputs must match the segmentation demo count")
        for demo_index, (pose, gripper, selected) in enumerate(
            zip(poses, grippers, boundaries, strict=True)
        ):
            velocity = translation_action_magnitude(pose)
            state = np.asarray(gripper, dtype=np.float64).reshape(len(velocity), -1)[:, 0]
            x = np.linspace(0.0, 1.0, len(velocity))
            axis = axes[row_index, demo_index]
            axis.plot(x, velocity, color="tab:blue", linewidth=1.0, label="EE step")
            scale = max(float(np.max(velocity)), 1.0e-9)
            axis.plot(x, state * scale, color="tab:orange", linewidth=1.0, label="gripper")
            for boundary in selected:
                axis.axvline(
                    boundary / max(len(velocity) - 1, 1),
                    color="0.35",
                    linestyle=":",
                )
            axis.set_title(f"{arm} demo {demo_index}")
            axis.set_xlim(0.0, 1.0)
            if demo_index == 0:
                axis.set_ylabel("translation step / scaled gripper")
            if row_index == 1:
                axis.set_xlabel("normalized time")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


__all__ = [
    "BimanualTAPASSegmentation",
    "TAPAS_ACTION_TIMING",
    "TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS",
    "TAPAS_CONFIG_DEFAULTS_SOURCE_STATUS",
    "TAPAS_DEFAULT_CONFIG_PATH",
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
    "forward_gripper_action",
    "forward_pose_action",
    "gripper_change_boundaries",
    "load_rlbench_segmentation_config",
    "next_observation_actions",
    "save_bimanual_segmentation_debug_plot",
    "segment_bimanual_pose_trajectories",
    "segment_bimanual_trajectories",
    "segment_pose_trajectories",
    "segment_trajectories",
    "tapas_distance_boundaries",
    "tapas_gripper_boundaries",
    "tapas_velocity_boundaries",
    "translation_action_magnitude",
]
