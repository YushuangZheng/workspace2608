"""Simplified paper-faithful skill-level DynaMAC Gaussian baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from essay2608.data.dataset import Demonstration

from .base import PHASE_NAMES, PolicyObservation, PolicyStep
from .gaussian import fit_frame_gaussian
from .multistream import StaticMultiStreamPolicy


class SkillDynaMACPolicy(StaticMultiStreamPolicy):
    """Select fixed frame sets per labelled skill using training precision only.

    This baseline follows the structure of DynaMAC Algorithm 1 while retaining
    this project's phase-aligned Gaussian and translational PoE. The scripted
    expert phases are used as provisional skill labels until the independent
    segmentation diagnostic is validated.
    """

    name = "skill_dynamac"
    method_scope = "paper_faithful_simplified_gaussian"

    def __init__(
        self,
        bins: int = 25,
        kinematic_scale_threshold: float = 0.001,
        linked_bin_fraction: float = 0.5,
        frame_selection_threshold: float = 0.2,
    ) -> None:
        super().__init__(bins=bins)
        self.kinematic_scale_threshold = float(kinematic_scale_threshold)
        self.linked_bin_fraction = float(linked_bin_fraction)
        self.frame_selection_threshold = float(frame_selection_threshold)
        self.selected_frames: dict[int, list[str]] = {}
        self.skill_diagnostics: dict[int, dict[str, Any]] = {}
        self.virtual_frame_poses: dict[int, np.ndarray] = {}

    @staticmethod
    def _precision_log_determinant(covariance: np.ndarray) -> np.ndarray:
        sign, log_determinant = np.linalg.slogdet(covariance)
        if np.any(sign <= 0.0):
            raise ValueError("Pose covariance must be positive definite.")
        return -log_determinant

    def _kinematic_diagnostics(self, frame_name: str, phase: int) -> dict[str, Any]:
        covariance = self.models[frame_name].pose_covariance[phase]
        _, covariance_log_determinant = np.linalg.slogdet(covariance)
        # Eq. (5): det(precision)^(-1 / (2d)), with d = 6.
        geometric_mean_scale = np.exp(covariance_log_determinant / 12.0)
        linked_fraction = float(
            np.mean(geometric_mean_scale < self.kinematic_scale_threshold)
        )
        return {
            "minimum_geometric_mean_scale": float(np.min(geometric_mean_scale)),
            "median_geometric_mean_scale": float(np.median(geometric_mean_scale)),
            "linked_bin_fraction": linked_fraction,
            "linked": linked_fraction >= self.linked_bin_fraction,
        }

    def _selection_scores(self, candidates: list[str], phase: int) -> dict[str, float]:
        log_precision = np.stack(
            [
                self._precision_log_determinant(
                    self.models[frame_name].pose_covariance[phase]
                )
                for frame_name in candidates
            ]
        )
        maximum = np.max(log_precision, axis=0, keepdims=True)
        normalized = np.exp(log_precision - maximum)
        normalized /= np.sum(normalized, axis=0, keepdims=True)
        return {
            frame_name: float(np.max(normalized[index]))
            for index, frame_name in enumerate(candidates)
        }

    def fit(self, demonstrations: list[Demonstration]) -> None:
        self._fit_phase_durations(demonstrations)
        virtual_frames = [f"virtual_skill_{phase}" for phase in range(len(PHASE_NAMES))]
        frame_names = ["world", "object", "target", *virtual_frames]
        self.models = {
            frame_name: fit_frame_gaussian(demonstrations, frame_name, bins=self.bins)
            for frame_name in frame_names
        }
        self.selected_frames = {}
        self.skill_diagnostics = {}
        for phase in range(len(PHASE_NAMES)):
            # The goal marker is exogenous by task definition. The physical
            # object is the only frame that can become robot-controlled here.
            object_link = self._kinematic_diagnostics("object", phase)
            valid_dynamic_frames = ["target"]
            if not object_link["linked"]:
                valid_dynamic_frames.insert(0, "object")
            candidates = [
                *valid_dynamic_frames,
                *(f"virtual_skill_{index}" for index in range(phase + 1)),
            ]
            scores = self._selection_scores(candidates, phase)
            selected = [
                frame_name
                for frame_name in candidates
                if scores[frame_name] > self.frame_selection_threshold
            ]
            if not selected:
                selected = [max(scores, key=scores.get)]
            self.selected_frames[phase] = selected
            self.skill_diagnostics[phase] = {
                "phase": phase,
                "phase_name": PHASE_NAMES[phase],
                "object_link": object_link,
                "candidate_frames": candidates,
                "selection_scores": scores,
                "selected_frames": selected,
                "thresholds": {
                    "kinematic_scale": self.kinematic_scale_threshold,
                    "linked_bin_fraction": self.linked_bin_fraction,
                    "frame_selection": self.frame_selection_threshold,
                },
            }

    def _on_reset(self, observation: PolicyObservation) -> None:
        super()._on_reset(observation)
        self.virtual_frame_poses = {0: observation.ee_pose.copy()}

    def _on_transition(self, new_phase: int, observation: PolicyObservation) -> None:
        self.virtual_frame_poses[new_phase] = observation.ee_pose.copy()

    def _frame_pose(self, frame_name: str, observation: PolicyObservation) -> np.ndarray:
        if frame_name.startswith("virtual_skill_"):
            phase = int(frame_name.removeprefix("virtual_skill_"))
            if phase not in self.virtual_frame_poses:
                raise RuntimeError(f"Virtual frame for skill {phase} has not been captured.")
            return self.virtual_frame_poses[phase]
        return super()._frame_pose(frame_name, observation)

    def _active_frames(self, observation: PolicyObservation) -> list[str]:
        del observation
        return list(self.selected_frames[self.phase])

    def _connection_state(self) -> bool:
        return bool(self.skill_diagnostics[self.phase]["object_link"]["linked"])

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        step = super()._compute_action(observation)
        diagnostics = {
            **step.diagnostics,
            "policy_family": self.method_scope,
            "selection_mode": "offline_skill_fixed",
            "skill_frame_diagnostics": self.skill_diagnostics[self.phase],
        }
        return PolicyStep(action=step.action, diagnostics=diagnostics)
