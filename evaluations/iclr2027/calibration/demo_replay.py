"""Shared data structures for replaying demonstrations during calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from essay2608.policy import DynaMAC
from essay2608.policy.tsf import RuntimeObservation, StateId, TSFTaskModel
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT


BELIEF_CONFIG_PATH = REPOSITORY_ROOT / "configs/tsf_inference.json"
EXECUTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/tsf_execution.json"


@dataclass(frozen=True)
class Sample:
    """One aligned demonstration sample used by offline calibration."""

    state_id: StateId
    ee_pose: np.ndarray
    action_pose: np.ndarray
    frames: dict[str, np.ndarray]
    gripper: np.ndarray
    entity_configurations: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class ArmCase:
    """Motion policy, task model, and demonstrations for one controlled arm."""

    task: str
    arm: str
    policy: DynaMAC
    model: TSFTaskModel
    demonstrations: Sequence[Any]
    aligned: Mapping[int, Any]
    recoverable_frames: tuple[str, ...]
    demonstration_paths: tuple[Path, ...]
    model_path: Path

    @property
    def key(self) -> str:
        return self.task if self.arm == "single" else f"{self.task}/{self.arm}"


def _mode_by_skill(policy: DynaMAC, demonstration_index: int) -> dict[int, int]:
    return {
        skill_index: next(
            mode
            for mode, members in enumerate(skill.mode_demonstration_indices)
            if demonstration_index in members
        )
        for skill_index, skill in enumerate(policy.skills)
    }


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


__all__ = [
    "ArmCase",
    "BELIEF_CONFIG_PATH",
    "EXECUTION_CONFIG_PATH",
    "Sample",
]
