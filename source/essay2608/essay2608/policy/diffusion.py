"""Low-dimensional conditional diffusion action-chunk baseline.

This is intentionally described as a diffusion *baseline*, not a reproduction of
the image-based Diffusion Policy architecture.  It isolates the five-demo data
efficiency comparison while using exactly the geometric observations available to
the Gaussian policies.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from essay2608.data.dataset import Demonstration

from .base import PHASE_NAMES, PhaseClockPolicy, PolicyObservation, PolicyStep


def condition_vector(observation: PolicyObservation, phase: int, progress: float) -> np.ndarray:
    one_hot = np.zeros(len(PHASE_NAMES), dtype=np.float32)
    one_hot[phase] = 1.0
    return np.concatenate(
        (
            observation.ee_pose[:3],
            observation.object_pose[:3],
            observation.target_pose[:3],
            one_hot,
            [progress],
        )
    ).astype(np.float32)


class ConditionalDenoiser(nn.Module):
    """Compact MLP epsilon predictor for one Cartesian action chunk."""

    def __init__(self, condition_dim: int, action_dim: int, horizon: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.time_dim = 32
        input_dim = condition_dim + action_dim * horizon + self.time_dim
        output_dim = action_dim * horizon
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        frequency = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angle = timestep.float()[:, None] * frequency[None]
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)

    def forward(self, noisy_action: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        flattened = noisy_action.reshape(len(noisy_action), -1)
        output = self.network(torch.cat((flattened, condition, self._time_embedding(timestep)), dim=-1))
        return output.reshape_as(noisy_action)


class DiffusionActionPolicy(PhaseClockPolicy):
    """Checkpoint-backed DDPM action-chunk policy for single-arm evaluation."""

    name = "diffusion_policy"

    def __init__(self, checkpoint: str | Path, execution_horizon: int = 4) -> None:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.horizon = int(payload["horizon"])
        self.action_dim = int(payload["action_dim"])
        self.diffusion_steps = int(payload["diffusion_steps"])
        super().__init__(bins=int(payload.get("bins", 25)))
        self.model = ConditionalDenoiser(
            int(payload["condition_dim"]), self.action_dim, self.horizon, int(payload["hidden_dim"])
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.condition_mean = np.asarray(payload["condition_mean"], dtype=np.float32)
        self.condition_std = np.asarray(payload["condition_std"], dtype=np.float32)
        self.action_mean = np.asarray(payload["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(payload["action_std"], dtype=np.float32)
        self.dataset_sha256 = str(payload["dataset_sha256"])
        self.execution_horizon = min(int(execution_horizon), self.horizon)
        # A 32-step schedule needs a larger terminal beta than the common
        # 1000-step schedule so q(x_T) is actually close to unit Gaussian.
        self.betas = torch.linspace(1.0e-4, 0.20, self.diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self._generator = torch.Generator(device="cpu")

    def fit(self, demonstrations: list[Demonstration]) -> None:
        self._fit_phase_durations(demonstrations)

    def _on_reset(self, observation: PolicyObservation) -> None:
        del observation
        self._chunk = None
        self._chunk_index = 0
        self._generator.manual_seed(2608)

    def _on_transition(self, new_phase: int, observation: PolicyObservation) -> None:
        del new_phase, observation
        self._chunk = None
        self._chunk_index = 0

    @torch.inference_mode()
    def _sample_chunk(self, observation: PolicyObservation) -> np.ndarray:
        duration = max(int(self.phase_durations[self.phase]), 1)
        progress = min(self.phase_step, duration - 1) / max(duration - 1, 1)
        raw_condition = condition_vector(observation, self.phase, progress)
        normalized_condition = (raw_condition - self.condition_mean) / self.condition_std
        condition = torch.from_numpy(normalized_condition).unsqueeze(0)
        sample = torch.randn(
            (1, self.horizon, self.action_dim), generator=self._generator, dtype=torch.float32
        )
        for index in reversed(range(self.diffusion_steps)):
            timestep = torch.full((1,), index, dtype=torch.long)
            predicted_noise = self.model(sample, timestep, condition)
            alpha = self.alphas[index]
            alpha_bar = self.alpha_bars[index]
            sample = (sample - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if index:
                noise = torch.randn(sample.shape, generator=self._generator, dtype=sample.dtype)
                sample += torch.sqrt(self.betas[index]) * noise
        result = sample[0].numpy() * self.action_std + self.action_mean
        return result

    def _compute_action(self, observation: PolicyObservation) -> PolicyStep:
        if self._chunk is None or self._chunk_index >= self.execution_horizon:
            self._chunk = self._sample_chunk(observation)
            self._chunk_index = 0
        action = self._chunk[self._chunk_index].astype(np.float64, copy=True)
        self._chunk_index += 1
        action[:3] = np.clip(action[:3], [0.15, -0.55, 0.12], [0.85, 0.55, 0.75])
        quaternion_norm = np.linalg.norm(action[3:7])
        if quaternion_norm < 1.0e-6:
            action[3:7] = observation.ee_pose[3:7]
        else:
            action[3:7] /= quaternion_norm
        action[7] = -1.0 if action[7] < 0.0 else 1.0
        diagnostics = {
            "method": self.name,
            "phase": self.phase,
            "phase_name": self.phase_name,
            "profile_index": self.profile_index(),
            "active_frames": ["learned_condition"],
            "stream_weights": {},
            "connected": False,
            "chunk_index": self._chunk_index - 1,
            "denoising_steps": self.diffusion_steps,
        }
        return PolicyStep(action=action, diagnostics=diagnostics)
