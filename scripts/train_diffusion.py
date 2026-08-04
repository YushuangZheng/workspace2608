"""Train the low-dimensional conditional DDPM baseline on frozen demonstrations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from essay2608.data.dataset import load_dataset
from essay2608.policy.base import PHASE_NAMES, PolicyObservation
from essay2608.policy.diffusion import ConditionalDenoiser, condition_vector


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument("--output", type=Path, default=Path("outputs/diffusion/v1/checkpoint.pt"))
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--diffusion_steps", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--seed", type=int, default=2608)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


def build_training_arrays(demonstrations, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    conditions = []
    chunks = []
    for demonstration in demonstrations:
        for phase in range(len(PHASE_NAMES)):
            indices = demonstration.phase_indices(phase)
            for local_index, index in enumerate(indices):
                progress = local_index / max(len(indices) - 1, 1)
                observation = PolicyObservation(
                    demonstration.ee_pose[index],
                    demonstration.object_pose[index],
                    demonstration.target_pose[index],
                )
                conditions.append(condition_vector(observation, phase, progress))
                end = min(local_index + horizon, len(indices))
                chunk = demonstration.action[indices[local_index:end]]
                if len(chunk) < horizon:
                    chunk = np.concatenate((chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)))
                chunks.append(chunk)
    return np.asarray(conditions, dtype=np.float32), np.asarray(chunks, dtype=np.float32)


def main() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    demonstrations, manifest = load_dataset(args.data_dir, verify_hashes=True)
    raw_conditions, raw_actions = build_training_arrays(demonstrations, args.horizon)
    condition_mean = raw_conditions.mean(axis=0)
    condition_std = np.maximum(raw_conditions.std(axis=0), 1.0e-3)
    action_mean = raw_actions.reshape(-1, raw_actions.shape[-1]).mean(axis=0)
    action_std = np.maximum(raw_actions.reshape(-1, raw_actions.shape[-1]).std(axis=0), 1.0e-3)
    conditions = torch.from_numpy((raw_conditions - condition_mean) / condition_std)
    actions = torch.from_numpy((raw_actions - action_mean) / action_std)
    device = torch.device(args.device)
    model = ConditionalDenoiser(
        conditions.shape[1], actions.shape[2], args.horizon, args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    betas = torch.linspace(1.0e-4, 0.20, args.diffusion_steps, device=device)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    losses = []
    model.train()
    for step in range(args.steps):
        indices = torch.randint(len(actions), (args.batch_size,), generator=generator)
        clean = actions[indices].to(device)
        condition = conditions[indices].to(device)
        timestep = torch.randint(args.diffusion_steps, (args.batch_size,), generator=generator).to(device)
        noise = torch.randn(clean.shape, device=device)
        alpha_bar = alpha_bars[timestep, None, None]
        noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
        predicted = model(noisy, timestep, condition)
        loss = torch.mean((predicted - noise) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 500 == 0:
            print(f"[diffusion] step={step + 1}/{args.steps} loss={np.mean(losses[-100:]):.6f}", flush=True)
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "condition_dim": conditions.shape[1],
        "action_dim": actions.shape[2],
        "horizon": args.horizon,
        "diffusion_steps": args.diffusion_steps,
        "hidden_dim": args.hidden_dim,
        "bins": 25,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "dataset_sha256": manifest["dataset_sha256"],
        "num_demonstrations": len(demonstrations),
        "training_steps": args.steps,
        "seed": args.seed,
        "final_loss": float(np.mean(losses[-100:])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metrics = {
        "checkpoint": str(args.output.resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "num_demonstrations": len(demonstrations),
        "num_training_samples": len(actions),
        "training_steps": args.steps,
        "final_noise_prediction_loss": payload["final_loss"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
