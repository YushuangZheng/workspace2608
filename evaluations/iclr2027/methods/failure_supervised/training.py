"""Training and checkpoint utilities for the causal M4 classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import CausalGRUClassifier, masked_binary_cross_entropy

CHECKPOINT_SCHEMA = "failure-supervised-causal-gru-v1"


@dataclass(frozen=True)
class TrainingStepMetrics:
    loss: float
    valid_cycles: int
    gradient_norm: float


def train_step(
    model: CausalGRUClassifier,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positive_weight: float | None = None,
    max_gradient_norm: float | None = None,
) -> TrainingStepMetrics:
    """Run one causal, padding-aware optimization step."""

    if max_gradient_norm is not None and max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(features)
    loss = masked_binary_cross_entropy(
        logits,
        targets,
        valid_mask,
        positive_weight=positive_weight,
    )
    loss.backward()
    clip_value = float("inf") if max_gradient_norm is None else max_gradient_norm
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    if not bool(torch.isfinite(gradient_norm).item()):
        raise ValueError("training step produced non-finite gradients")
    optimizer.step()
    return TrainingStepMetrics(
        loss=float(loss.detach().cpu().item()),
        valid_cycles=int(valid_mask.to(dtype=torch.bool).sum().item()),
        gradient_norm=float(gradient_norm.detach().cpu().item()),
    )


def save_training_checkpoint(
    path: str | Path,
    model: CausalGRUClassifier,
    *,
    training_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model identity and optional optimizer state."""

    if isinstance(training_step, bool) or not isinstance(training_step, int):
        raise TypeError("training_step must be an integer")
    if training_step < 0:
        raise ValueError("training_step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.checkpoint_metadata(),
        "model_state_dict": model.state_dict(),
        "training_step": training_step,
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CausalGRUClassifier, dict[str, Any]]:
    """Load a versioned checkpoint without assuming a frozen feature schema."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported failure-supervised checkpoint schema")
    model_metadata = payload.get("model")
    if not isinstance(model_metadata, dict):
        raise ValueError("checkpoint is missing model metadata")
    expected_architecture = "causal_gru_v1"
    if model_metadata.get("architecture") != expected_architecture:
        raise ValueError("unsupported failure-supervised model architecture")
    model = CausalGRUClassifier(
        input_dim=int(model_metadata["input_dim"]),
        hidden_dim=int(model_metadata["hidden_dim"]),
        num_layers=int(model_metadata["num_layers"]),
        dropout=float(model_metadata["dropout"]),
    )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing model parameters")
    model.load_state_dict(state_dict, strict=True)
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "TrainingStepMetrics",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "train_step",
]
