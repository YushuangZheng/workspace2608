"""Export the official Square policy features consumed by logpZO."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import dill
import torch
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/_external/FAIL-Detect"),
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)

    import hydra
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
        TrainDiffusionUnetHybridWorkspace,
    )
    from omegaconf import OmegaConf

    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cfg.task.dataset.dataset_path = str(args.dataset.resolve())
    cfg.task.dataset_path = str(args.dataset.resolve())
    cfg.task.env_runner.dataset_path = str(args.dataset.resolve())
    cfg.training.device = args.device
    OmegaConf.resolve(cfg)

    workspace = TrainDiffusionUnetHybridWorkspace(cfg, output_dir=str(checkpoint.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )
    normalizer = dataset.get_normalizer()
    policy = workspace.ema_model
    if policy is None:
        raise ValueError("official checkpoint does not contain an EMA policy")
    policy.set_normalizer(normalizer)
    device = torch.device(args.device)
    policy.to(device).eval()

    all_features: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
            normalized_observations = policy.normalizer.normalize(batch["obs"])
            normalized_actions = policy.normalizer["action"].normalize(batch["action"])
            batch_size = normalized_actions.shape[0]
            observations = dict_apply(
                normalized_observations,
                lambda value: value[:, : policy.n_obs_steps].reshape(-1, *value.shape[2:]),
            )
            features = policy.obs_encoder(observations).reshape(batch_size, -1)
            trajectory = normalized_actions.reshape(batch_size, -1)
            all_features.append(features.cpu())
            all_actions.append(trajectory.cpu())

    features = torch.cat(all_features, dim=0)
    actions = torch.cat(all_actions, dim=0)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "X": features,
            "Y": actions,
            "reproduction": {
                "official_commit": "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed",
                "task": "square",
                "policy_type": "flow",
                "policy_checkpoint": str(checkpoint),
                "samples": len(features),
                "feature_shape": list(features.shape),
                "action_shape": list(actions.shape),
            },
        },
        temporary,
    )
    temporary.replace(destination)
    print(
        f"saved {len(features)} samples: X={tuple(features.shape)} "
        f"Y={tuple(actions.shape)} to {destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
