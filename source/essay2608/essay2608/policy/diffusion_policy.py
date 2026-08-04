"""低维 Diffusion Policy 风格对照。

它使用与 DynaMAC 相同的位姿/任务参数输入，不是论文表格中的图像版官方 Diffusion
Policy。保留它只为同一数据接口下的工程对照，所有报告必须使用 ``low_dimensional`` 标签。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .dynamac import (
    DynaMACAction,
    DynaMACDemonstration,
    DynaMACObservation,
    _compressed_skill_sequence,
    normalize_quaternion,
)


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    horizon: int = 16
    execution_horizon: int = 4
    diffusion_steps: int = 32
    hidden_dimension: int = 256
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1.0e-4
    random_seed: int = 2608


class DiffusionPolicy:
    """条件 DDPM action-chunk 基线；Torch 在实例化时才加载。"""

    name = "diffusion_policy_low_dimensional"

    def __init__(self, config: DiffusionPolicyConfig = DiffusionPolicyConfig()) -> None:
        import torch
        from torch import nn

        self.config = config
        self.torch = torch
        self.nn = nn
        self.model = None
        self.frame_names: tuple[str, ...] = ()
        self.skill_sequence: tuple[int, ...] = ()
        self.skill_durations = np.empty(0, dtype=np.int64)
        self._skill_index = 0
        self._time_index = 0
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self._complete = False

    def _condition(
        self,
        observation: DynaMACObservation,
        skill_index: int,
        progress: float,
    ) -> np.ndarray:
        one_hot = np.zeros(len(self.skill_sequence), dtype=np.float32)
        one_hot[skill_index] = 1.0
        return np.concatenate(
            (
                observation.ee_pose,
                *(observation.frames[name] for name in self.frame_names),
                one_hot,
                [progress],
            )
        ).astype(np.float32)

    def _build_model(self, condition_dimension: int, action_dimension: int):
        torch = self.torch
        nn = self.nn
        time_dimension = 32
        horizon = self.config.horizon

        class Denoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                input_dimension = condition_dimension + action_dimension * horizon + time_dimension
                self.network = nn.Sequential(
                    nn.Linear(input_dimension, self_hidden),
                    nn.SiLU(),
                    nn.Linear(self_hidden, self_hidden),
                    nn.SiLU(),
                    nn.Linear(self_hidden, self_hidden),
                    nn.SiLU(),
                    nn.Linear(self_hidden, action_dimension * horizon),
                )

            def forward(self, noisy, timestep, condition):
                half = time_dimension // 2
                frequency = torch.exp(
                    torch.arange(half, device=timestep.device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half - 1, 1))
                )
                angle = timestep.float()[:, None] * frequency[None]
                embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
                value = torch.cat((noisy.flatten(1), condition, embedding), dim=-1)
                return self.network(value).reshape_as(noisy)

        self_hidden = self.config.hidden_dimension
        return Denoiser()

    def fit(self, demonstrations: Sequence[DynaMACDemonstration]) -> DiffusionPolicy:
        if not demonstrations:
            raise ValueError("至少需要一条演示")
        self.frame_names = tuple(sorted(demonstrations[0].frames))
        self.skill_sequence = tuple(_compressed_skill_sequence(demonstrations[0].skill))
        for demonstration in demonstrations:
            if tuple(sorted(demonstration.frames)) != self.frame_names:
                raise ValueError("所有演示必须具有相同任务参数")
            if tuple(_compressed_skill_sequence(demonstration.skill)) != self.skill_sequence:
                raise ValueError("所有演示必须具有相同技能顺序")
        self.skill_durations = np.asarray(
            [
                round(
                    np.mean(
                        [np.sum(demonstration.skill == label) for demonstration in demonstrations]
                    )
                )
                for label in self.skill_sequence
            ],
            dtype=np.int64,
        )

        conditions = []
        chunks = []
        action_dimension = 7 + demonstrations[0].gripper.shape[1]
        for demonstration in demonstrations:
            actions = np.concatenate((demonstration.action_pose, demonstration.gripper), axis=1)
            for skill_index, label in enumerate(self.skill_sequence):
                indices = np.flatnonzero(demonstration.skill == label)
                for local_index, step in enumerate(indices):
                    progress = local_index / max(len(indices) - 1, 1)
                    observation = DynaMACObservation(
                        demonstration.ee_pose[step],
                        {name: demonstration.frames[name][step] for name in self.frame_names},
                    )
                    conditions.append(self._condition(observation, skill_index, progress))
                    chosen = indices[
                        np.minimum(local_index + np.arange(self.config.horizon), len(indices) - 1)
                    ]
                    chunks.append(actions[chosen])
        condition_values = np.stack(conditions).astype(np.float32)
        action_values = np.stack(chunks).astype(np.float32)
        self.condition_mean = np.mean(condition_values, axis=0)
        self.condition_std = np.maximum(np.std(condition_values, axis=0), 1.0e-6)
        self.action_mean = np.mean(action_values, axis=(0, 1))
        self.action_std = np.maximum(np.std(action_values, axis=(0, 1)), 1.0e-6)
        condition_values = (condition_values - self.condition_mean) / self.condition_std
        action_values = (action_values - self.action_mean) / self.action_std

        torch = self.torch
        torch.manual_seed(self.config.random_seed)
        self.model = self._build_model(condition_values.shape[1], action_dimension)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        condition_tensor = torch.from_numpy(condition_values)
        action_tensor = torch.from_numpy(action_values)
        betas = torch.linspace(1.0e-4, 0.20, self.config.diffusion_steps)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        generator = torch.Generator().manual_seed(self.config.random_seed)
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(action_tensor), generator=generator)
            for start in range(0, len(permutation), self.config.batch_size):
                batch = permutation[start : start + self.config.batch_size]
                clean = action_tensor[batch]
                condition = condition_tensor[batch]
                timestep = torch.randint(
                    0,
                    self.config.diffusion_steps,
                    (len(batch),),
                    generator=generator,
                )
                noise = torch.randn(clean.shape, generator=generator)
                alpha_bar = alpha_bars[timestep, None, None]
                noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
                loss = torch.mean((self.model(noisy, timestep, condition) - noise) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = alpha_bars
        self.model.eval()
        return self

    @property
    def complete(self) -> bool:
        return self._complete

    def reset(self, observation: DynaMACObservation) -> None:
        del observation
        if self.model is None:
            raise RuntimeError("Diffusion Policy 尚未拟合")
        self._skill_index = 0
        self._time_index = 0
        self._chunk = None
        self._chunk_index = 0
        self._complete = False
        self._generator = self.torch.Generator().manual_seed(self.config.random_seed)

    def _sample(self, observation: DynaMACObservation) -> np.ndarray:
        torch = self.torch
        duration = max(int(self.skill_durations[self._skill_index]), 1)
        progress = min(self._time_index, duration - 1) / max(duration - 1, 1)
        condition = self._condition(observation, self._skill_index, progress)
        condition = torch.from_numpy(
            ((condition - self.condition_mean) / self.condition_std).astype(np.float32)
        )[None]
        action_dimension = len(self.action_mean)
        sample = torch.randn((1, self.config.horizon, action_dimension), generator=self._generator)
        with torch.inference_mode():
            for index in reversed(range(self.config.diffusion_steps)):
                timestep = torch.full((1,), index, dtype=torch.long)
                predicted = self.model(sample, timestep, condition)
                sample = (
                    sample
                    - (1.0 - self.alphas[index])
                    / torch.sqrt(1.0 - self.alpha_bars[index])
                    * predicted
                ) / torch.sqrt(self.alphas[index])
                if index:
                    sample += torch.sqrt(self.betas[index]) * torch.randn(
                        sample.shape, generator=self._generator
                    )
        return sample[0].numpy() * self.action_std + self.action_mean

    def act(self, observation: DynaMACObservation) -> DynaMACAction:
        if self._complete:
            raise RuntimeError("Diffusion Policy 已完成")
        if self._chunk is None or self._chunk_index >= self.config.execution_horizon:
            self._chunk = self._sample(observation)
            self._chunk_index = 0
        raw = self._chunk[self._chunk_index].copy()
        self._chunk_index += 1
        pose = raw[:7]
        pose[3:7] = normalize_quaternion(pose[3:7])
        gripper = raw[7:]
        diagnostics = {
            "method": self.name,
            "scope": "low_dimensional_not_official_image_baseline",
            "skill_index": self._skill_index,
            "time_index": self._time_index,
        }
        self._time_index += 1
        if self._time_index >= self.skill_durations[self._skill_index]:
            if self._skill_index == len(self.skill_sequence) - 1:
                self._complete = True
            else:
                self._skill_index += 1
                self._time_index = 0
                self._chunk = None
                self._chunk_index = 0
        return DynaMACAction(pose, gripper, diagnostics)


__all__ = ["DiffusionPolicy", "DiffusionPolicyConfig"]
