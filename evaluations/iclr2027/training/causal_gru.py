"""Lightweight unidirectional GRU used by the M4 supervised baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class CausalGRUClassifier(nn.Module):
    """Emit a violation logit for every prefix of a feature trajectory."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("GRU dimensions and layer count must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have shape [batch, time, input_dim]")
        encoded, next_hidden = self.gru(features, hidden)
        return self.output(encoded).squeeze(-1), next_hidden

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "causal_gru_v1",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": False,
        }


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_weight: float | None = None,
) -> torch.Tensor:
    """Cycle-level BCE that excludes right-padding, never future labels."""

    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have equal shape")
    if logits.ndim != 2:
        raise ValueError("cycle-level tensors must have shape [batch, time]")
    mask = valid_mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise ValueError("valid_mask must select at least one cycle")
    pos_weight = None
    if positive_weight is not None:
        if positive_weight <= 0.0:
            raise ValueError("positive_weight must be positive")
        pos_weight = logits.new_tensor(float(positive_weight))
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        reduction="none",
        pos_weight=pos_weight,
    )
    return losses[mask].mean()


__all__ = ["CausalGRUClassifier", "masked_binary_cross_entropy"]
