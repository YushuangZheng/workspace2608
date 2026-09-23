"""Pinned FAIL-Detect conditional 1-D U-Net used by the Main-10 scorer.

The architecture mirrors ``CFM.net_CFM.get_unet(10)`` at the pinned public
FAIL-Detect revision b758e55f7c0c988188f2e4876ffc03ae8a3c30ed.  It is kept
locally so A can train and load the project-domain normal-data scorer without
depending on an unversioned external checkout.  The only adaptation is the
upstream causal feature representation; network and logpZO semantics remain
unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / (half - 1)
        frequency = torch.exp(
            torch.arange(half, device=value.device, dtype=torch.float32) * -scale
        )
        phase = value[:, None].to(dtype=frequency.dtype) * frequency[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
        *,
        kernel_size: int = 3,
        groups: int = 8,
        condition_predicts_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    input_channels, output_channels, kernel_size, groups=groups
                ),
                Conv1dBlock(
                    output_channels, output_channels, kernel_size, groups=groups
                ),
            ]
        )
        condition_channels = output_channels * (2 if condition_predicts_scale else 1)
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, condition_channels)
        )
        self.cond_predict_scale = bool(condition_predicts_scale)
        self.out_channels = int(output_channels)
        self.residual_conv = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](value)
        embedding = self.cond_encoder(condition).unsqueeze(-1)
        if self.cond_predict_scale:
            embedding = embedding.reshape(
                embedding.shape[0], 2, self.out_channels, 1
            )
            out = embedding[:, 0] * out + embedding[:, 1]
        else:
            out = out + embedding
        return self.blocks[1](out) + self.residual_conv(value)


class ConditionalUnet1D(nn.Module):
    """Official diffusion-policy U-Net interface: ``[B,T,D] -> [B,T,D]``."""

    def __init__(
        self,
        input_dim: int,
        *,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        groups: int = 8,
    ) -> None:
        super().__init__()
        all_dims = (int(input_dim), *tuple(int(value) for value in down_dims))
        condition_dim = int(diffusion_step_embed_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(condition_dim),
            nn.Linear(condition_dim, condition_dim * 4),
            nn.Mish(),
            nn.Linear(condition_dim * 4, condition_dim),
        )
        in_out = tuple(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            condition_dim,
                            kernel_size=kernel_size,
                            groups=groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            condition_dim,
                            kernel_size=kernel_size,
                            groups=groups,
                        ),
                        nn.Identity() if last else Downsample1d(dim_out),
                    ]
                )
            )
        middle = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    middle,
                    middle,
                    condition_dim,
                    kernel_size=kernel_size,
                    groups=groups,
                ),
                ConditionalResidualBlock1D(
                    middle,
                    middle,
                    condition_dim,
                    kernel_size=kernel_size,
                    groups=groups,
                ),
            ]
        )
        self.up_modules = nn.ModuleList()
        reversed_pairs = tuple(reversed(in_out[1:]))
        for index, (dim_in, dim_out) in enumerate(reversed_pairs):
            last = index == len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            condition_dim,
                            kernel_size=kernel_size,
                            groups=groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            condition_dim,
                            kernel_size=kernel_size,
                            groups=groups,
                        ),
                        nn.Identity() if last else Upsample1d(dim_in),
                    ]
                )
            )
        start_dim = down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.input_dim = int(input_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        **_: Any,
    ) -> torch.Tensor:
        value = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            times = torch.tensor([timestep], dtype=torch.long, device=value.device)
        elif timestep.ndim == 0:
            times = timestep[None].to(value.device)
        else:
            times = timestep.to(value.device)
        condition = self.diffusion_step_encoder(times.expand(value.shape[0]))
        skips = []
        for first, second, downsample in self.down_modules:
            value = second(first(value, condition), condition)
            skips.append(value)
            value = downsample(value)
        for middle in self.mid_modules:
            value = middle(value, condition)
        for first, second, upsample in self.up_modules:
            value = torch.cat((value, skips.pop()), dim=1)
            value = upsample(second(first(value, condition), condition))
        return self.final_conv(value).transpose(1, 2)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "fail_detect_conditional_unet1d",
            "input_dim": self.input_dim,
            "diffusion_step_embed_dim": 128,
            "down_dims": [256, 512, 1024],
            "kernel_size": 5,
            "n_groups": 8,
            "cond_predict_scale": False,
        }


def build_official_velocity_model(input_dim: int = 10) -> ConditionalUnet1D:
    if input_dim != 10:
        raise ValueError("the frozen M3 backend requires ten input channels")
    return ConditionalUnet1D(input_dim)


__all__ = ["ConditionalUnet1D", "build_official_velocity_model"]
