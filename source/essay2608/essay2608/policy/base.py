"""Common policy interface and phase clock."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from essay2608.data.dataset import Demonstration


PHASE_NAMES = [
    "rest",
    "approach_above_object",
    "approach_object",
    "grasp_object",
    "lift_object",
    "move_above_target",
    "lower_to_target",
    "release_object",
    "retreat",
    "complete",
]
MOVEMENT_PHASES = {1, 2, 4, 5, 6, 8}


@dataclass(frozen=True)
class PolicyObservation:
    """State required by the geometric single-arm policies."""

    ee_pose: np.ndarray
    object_pose: np.ndarray
    target_pose: np.ndarray


@dataclass(frozen=True)
class PolicyStep:
    """One action and its scientific diagnostics."""

    action: np.ndarray
    diagnostics: dict[str, Any]


class SingleArmPolicy(ABC):
    """Minimal fit/reset/act policy contract."""

    name = "abstract"

    @abstractmethod
    def fit(self, demonstrations: list[Demonstration]) -> None:
        """Fit policy parameters from demonstrations."""

    @abstractmethod
    def reset(self, observation: PolicyObservation) -> None:
        """Reset online state for one rollout."""

    @abstractmethod
    def act(self, observation: PolicyObservation) -> PolicyStep:
        """Return one absolute Cartesian pose and gripper command."""

    @property
    @abstractmethod
    def complete(self) -> bool:
        """Whether the learned phase sequence is complete."""


class PhaseClockPolicy(SingleArmPolicy):
    """Data-derived phase timing with endpoint reach gating."""

    position_threshold = 0.018
    maximum_hold_steps = 200

    def __init__(self, bins: int = 25) -> None:
        self.bins = int(bins)
        self.phase_durations = np.ones(len(PHASE_NAMES), dtype=np.int64)
        self.phase = 0
        self.phase_step = 0
        self.total_step = 0
        self._complete = False
        self.forced_transitions = 0

    def _fit_phase_durations(self, demonstrations: list[Demonstration]) -> None:
        durations = []
        for phase in range(len(PHASE_NAMES)):
            durations.append([len(demonstration.phase_indices(phase)) for demonstration in demonstrations])
        self.phase_durations = np.asarray([round(np.median(values)) for values in durations], dtype=np.int64)

    def reset(self, observation: PolicyObservation) -> None:
        self.phase = 0
        self.phase_step = 0
        self.total_step = 0
        self._complete = False
        self.forced_transitions = 0
        self._on_reset(observation)

    def _on_reset(self, observation: PolicyObservation) -> None:
        del observation

    def _on_transition(self, new_phase: int, observation: PolicyObservation) -> None:
        del new_phase, observation

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES[self.phase]

    def profile_index(self) -> int:
        duration = max(int(self.phase_durations[self.phase]), 1)
        progress = min(self.phase_step, duration - 1) / max(duration - 1, 1)
        return min(round(progress * (self.bins - 1)), self.bins - 1)

    def _advance_clock(self, observation: PolicyObservation, desired_position: np.ndarray) -> None:
        duration = int(self.phase_durations[self.phase])
        self.phase_step += 1
        self.total_step += 1
        if self.phase_step < duration:
            return

        reached = np.linalg.norm(observation.ee_pose[:3] - desired_position) < self.position_threshold
        should_advance = self.phase not in MOVEMENT_PHASES or reached
        if self.phase_step >= duration + self.maximum_hold_steps:
            should_advance = True
            self.forced_transitions += 1
        if not should_advance:
            return

        if self.phase == len(PHASE_NAMES) - 1:
            self._complete = True
            return
        self.phase += 1
        self.phase_step = 0
        self._on_transition(self.phase, observation)

    def act(self, observation: PolicyObservation) -> PolicyStep:
        if self._complete:
            raise RuntimeError("Cannot act after policy completion.")
        step = self._compute_action(observation)
        self._advance_clock(observation, step.action[:3])
        return step

    @abstractmethod
    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        """Compute an action without advancing the phase clock."""
