"""真值状态输入的 Diffusion Policy 独立复现。

MiDiGaP/DynaMAC 论文为公平比较向所有方法提供真值物体位姿，并采用 state-based U-Net
Diffusion Policy。本实现保持相同输入语义、16 步 receding horizon 和条件时序 U-Net，
但不是官方仓库代码；实验报告必须标注为独立复现。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from essay2608.policy.dynamac import (
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
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.horizon < 4 or self.horizon % 4:
            raise ValueError("U-Net horizon 必须是 4 的正倍数")
        if not 1 <= self.execution_horizon <= self.horizon:
            raise ValueError("execution_horizon 必须位于 [1, horizon]")
        if self.diffusion_steps < 2 or self.hidden_dimension < 16:
            raise ValueError("扩散步数或隐藏维数过小")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("训练参数必须为正")


class DiffusionPolicy:
    """条件 DDPM action-chunk 基线；Torch 在实例化时才加载。"""

    name = "diffusion_policy_state_unet"

    def __init__(self, config: DiffusionPolicyConfig = DiffusionPolicyConfig()) -> None:
        import torch
        from torch import nn

        self.config = config
        self.torch = torch
        self.nn = nn
        self.device = torch.device(
            "cuda"
            if config.device == "auto" and torch.cuda.is_available()
            else ("cpu" if config.device == "auto" else config.device)
        )
        self.model = None
        self.frame_names: tuple[str, ...] = ()
        self.skill_sequence: tuple[int, ...] = ()
        self.duration = 0
        self._time_index = 0
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self._complete = False

    def _condition(
        self,
        observation: DynaMACObservation,
    ) -> np.ndarray:
        return np.concatenate(
            (
                observation.ee_pose,
                *(observation.frames[name] for name in self.frame_names),
            )
        ).astype(np.float32)

    def _build_model(self, condition_dimension: int, action_dimension: int):
        torch = self.torch
        nn = self.nn
        time_dimension = 32

        class FiLMBlock(nn.Module):
            def __init__(self, inputs: int, outputs: int, context_dimension: int) -> None:
                super().__init__()
                self.first = nn.Conv1d(inputs, outputs, kernel_size=3, padding=1)
                self.second = nn.Conv1d(outputs, outputs, kernel_size=3, padding=1)
                self.context = nn.Linear(context_dimension, outputs * 2)
                self.residual = (
                    nn.Identity() if inputs == outputs else nn.Conv1d(inputs, outputs, 1)
                )

            def forward(self, value, context):
                scale, shift = self.context(context).chunk(2, dim=-1)
                hidden = torch.nn.functional.silu(self.first(value))
                hidden = hidden * (1.0 + scale[..., None]) + shift[..., None]
                hidden = self.second(torch.nn.functional.silu(hidden))
                return torch.nn.functional.silu(hidden + self.residual(value))

        class Denoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                context_dimension = self_hidden * 2
                self.context = nn.Sequential(
                    nn.Linear(condition_dimension + time_dimension, context_dimension),
                    nn.SiLU(),
                    nn.Linear(context_dimension, context_dimension),
                    nn.SiLU(),
                )
                self.input = nn.Conv1d(action_dimension, self_hidden, kernel_size=3, padding=1)
                self.down_one = FiLMBlock(self_hidden, self_hidden, context_dimension)
                self.down_two = FiLMBlock(self_hidden, self_hidden * 2, context_dimension)
                self.middle = FiLMBlock(self.hidden_two, self.hidden_two, context_dimension)
                self.up_two = nn.ConvTranspose1d(self.hidden_two, self_hidden * 2, 2, stride=2)
                self.decode_two = FiLMBlock(self_hidden * 4, self_hidden, context_dimension)
                self.up_one = nn.ConvTranspose1d(self_hidden, self_hidden, 2, stride=2)
                self.decode_one = FiLMBlock(self_hidden * 2, self_hidden, context_dimension)
                self.output = nn.Conv1d(self_hidden, action_dimension, kernel_size=1)

            @property
            def hidden_two(self) -> int:
                return self_hidden * 2

            def forward(self, noisy, timestep, condition):
                half = time_dimension // 2
                frequency = torch.exp(
                    torch.arange(half, device=timestep.device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(half - 1, 1))
                )
                angle = timestep.float()[:, None] * frequency[None]
                embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
                context = self.context(torch.cat((condition, embedding), dim=-1))
                value = self.input(noisy.transpose(1, 2))
                skip_one = self.down_one(value, context)
                value = torch.nn.functional.avg_pool1d(skip_one, 2)
                skip_two = self.down_two(value, context)
                value = torch.nn.functional.avg_pool1d(skip_two, 2)
                value = self.middle(value, context)
                value = self.up_two(value)
                value = self.decode_two(torch.cat((value, skip_two), dim=1), context)
                value = self.up_one(value)
                value = self.decode_one(torch.cat((value, skip_one), dim=1), context)
                return self.output(value).transpose(1, 2)

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
        self.duration = max(round(np.mean([len(item.ee_pose) for item in demonstrations])), 1)

        conditions = []
        chunks = []
        action_dimension = 7 + demonstrations[0].gripper.shape[1]
        for demonstration in demonstrations:
            actions = np.concatenate((demonstration.action_pose, demonstration.gripper), axis=1)
            for step in range(len(demonstration.ee_pose)):
                observation = DynaMACObservation(
                    demonstration.ee_pose[step],
                    {name: demonstration.frames[name][step] for name in self.frame_names},
                )
                conditions.append(self._condition(observation))
                chosen = np.minimum(
                    step + np.arange(self.config.horizon), len(demonstration.ee_pose) - 1
                )
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
        self.model = self._build_model(condition_values.shape[1], action_dimension).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        condition_tensor = torch.from_numpy(condition_values).to(self.device)
        action_tensor = torch.from_numpy(action_values).to(self.device)
        betas = torch.linspace(1.0e-4, 0.20, self.config.diffusion_steps, device=self.device)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        generator = torch.Generator(device=self.device).manual_seed(self.config.random_seed)
        for _ in range(self.config.epochs):
            permutation = torch.randperm(
                len(action_tensor), generator=generator, device=self.device
            )
            for start in range(0, len(permutation), self.config.batch_size):
                batch = permutation[start : start + self.config.batch_size]
                clean = action_tensor[batch]
                condition = condition_tensor[batch]
                timestep = torch.randint(
                    0,
                    self.config.diffusion_steps,
                    (len(batch),),
                    generator=generator,
                    device=self.device,
                )
                noise = torch.randn(clean.shape, generator=generator, device=self.device)
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
        self._time_index = 0
        self._chunk = None
        self._chunk_index = 0
        self._complete = False
        self._generator = self.torch.Generator(device=self.device).manual_seed(
            self.config.random_seed
        )

    def _sample(self, observation: DynaMACObservation) -> np.ndarray:
        torch = self.torch
        condition = self._condition(observation)
        condition = torch.from_numpy(
            ((condition - self.condition_mean) / self.condition_std).astype(np.float32)
        )[None].to(self.device)
        action_dimension = len(self.action_mean)
        sample = torch.randn(
            (1, self.config.horizon, action_dimension),
            generator=self._generator,
            device=self.device,
        )
        with torch.inference_mode():
            for index in reversed(range(self.config.diffusion_steps)):
                timestep = torch.full((1,), index, dtype=torch.long, device=self.device)
                predicted = self.model(sample, timestep, condition)
                sample = (
                    sample
                    - (1.0 - self.alphas[index])
                    / torch.sqrt(1.0 - self.alpha_bars[index])
                    * predicted
                ) / torch.sqrt(self.alphas[index])
                if index:
                    sample += torch.sqrt(self.betas[index]) * torch.randn(
                        sample.shape, generator=self._generator, device=self.device
                    )
        return sample[0].cpu().numpy() * self.action_std + self.action_mean

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
            "scope": "independent_state_unet_reproduction_not_official_code",
            "time_index": self._time_index,
        }
        self._time_index += 1
        if self._time_index >= self.duration:
            self._complete = True
        return DynaMACAction(pose, gripper, diagnostics)

    def save(self, path: str | Path) -> None:
        """保存无 pickle 的独立复现 checkpoint。"""

        if self.model is None:
            raise RuntimeError("Diffusion Policy 尚未拟合")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_items = list(self.model.state_dict().items())
        metadata = {
            "schema": "essay2608.diffusion_policy.state_unet.v2",
            "config": asdict(self.config),
            "frame_names": list(self.frame_names),
            "skill_sequence": list(self.skill_sequence),
            "duration": self.duration,
            "model_state_keys": [name for name, _ in state_items],
            "scope": "independent_state_unet_reproduction_not_official_code",
        }
        arrays = {
            f"model_{index:04d}": value.detach().cpu().numpy()
            for index, (_, value) in enumerate(state_items)
        }
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            condition_mean=self.condition_mean,
            condition_std=self.condition_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> DiffusionPolicy:
        """严格读取由 :meth:`save` 写出的无 pickle checkpoint。"""

        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema") != "essay2608.diffusion_policy.state_unet.v2":
                raise ValueError("不支持的 Diffusion Policy checkpoint schema")
            policy = cls(DiffusionPolicyConfig(**metadata["config"]))
            policy.frame_names = tuple(metadata["frame_names"])
            policy.skill_sequence = tuple(int(value) for value in metadata["skill_sequence"])
            policy.duration = int(metadata["duration"])
            policy.condition_mean = archive["condition_mean"].copy()
            policy.condition_std = archive["condition_std"].copy()
            policy.action_mean = archive["action_mean"].copy()
            policy.action_std = archive["action_std"].copy()
            policy.model = policy._build_model(
                len(policy.condition_mean), len(policy.action_mean)
            ).to(policy.device)
            state = {
                name: policy.torch.from_numpy(archive[f"model_{index:04d}"].copy())
                for index, name in enumerate(metadata["model_state_keys"])
            }
            policy.model.load_state_dict(state, strict=True)

        policy.betas = policy.torch.linspace(
            1.0e-4, 0.20, policy.config.diffusion_steps, device=policy.device
        )
        policy.alphas = 1.0 - policy.betas
        policy.alpha_bars = policy.torch.cumprod(policy.alphas, dim=0)
        policy.model.eval()
        return policy


__all__ = ["DiffusionPolicy", "DiffusionPolicyConfig"]
