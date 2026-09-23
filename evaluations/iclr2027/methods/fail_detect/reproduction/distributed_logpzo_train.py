"""Eight-GPU training for the pinned FAIL-Detect logpZO score network."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/_external/FAIL-Detect"),
    )
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--workers-per-rank", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-epochs-this-run", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _adjust_xshape(features: torch.Tensor, input_dim: int) -> torch.Tensor:
    total_dim = features.shape[1]
    remainder = total_dim % input_dim
    if remainder:
        features = torch.cat(
            [features, features.new_zeros(features.shape[0], input_dim - remainder)],
            dim=1,
        )
        total_dim = features.shape[1]
    reshaped_dim = total_dim // input_dim
    if reshaped_dim % 4:
        extra = (4 - reshaped_dim % 4) * input_dim
        features = torch.cat([features, features.new_zeros(features.shape[0], extra)], dim=1)
    return features.reshape(features.shape[0], -1, input_dim)


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_save(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by WORLD_SIZE")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_primary = rank == 0

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    sys.path.insert(0, str(official_root / "UQ_baselines"))
    os.chdir(official_root)
    from CFM.net_CFM import get_unet

    feature_payload = torch.load(args.features, map_location="cpu", weights_only=True)
    if not isinstance(feature_payload, dict) or "X" not in feature_payload:
        raise ValueError("feature file must contain the upstream X tensor")
    features = feature_payload["X"].to(dtype=torch.float32)
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("X must be a finite rank-two tensor")
    features = _adjust_xshape(features, input_dim=10)
    dataset = TensorDataset(features)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // world_size,
        sampler=sampler,
        num_workers=args.workers_per_rank,
        persistent_workers=args.workers_per_rank > 0,
        pin_memory=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = get_unet(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    start_epoch = 0
    losses: list[float] = []
    checkpoint_path = args.checkpoint.resolve()
    if not args.no_resume and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        _move_optimizer(optimizer, device)
        start_epoch = int(payload["epoch"])
        losses = [float(value) for value in payload.get("losses", [])]

    distributed_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        static_graph=True,
        gradient_as_bucket_view=True,
    )
    stop_epoch = args.epochs
    if args.max_epochs_this_run is not None:
        stop_epoch = min(stop_epoch, start_epoch + args.max_epochs_this_run)
    metrics_path = args.metrics.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    try:
        for epoch in range(start_epoch, stop_epoch):
            sampler.set_epoch(epoch)
            epoch_seed = args.seed + rank + epoch * world_size
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            random.seed(epoch_seed)
            model.train()
            epoch_started = time.monotonic()
            loss_sum = 0.0
            count = 0
            for (observation,) in loader:
                observation = observation.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                x0 = observation
                x1 = torch.randn_like(x0)
                true_velocity = x1 - x0
                continuous_time = torch.rand(len(x1), device=device, dtype=observation.dtype).view(
                    -1, *([1] * (observation.ndim - 1))
                )
                current = x0 + continuous_time * true_velocity
                discrete_time = (continuous_time.reshape(-1) * 100).long()
                predicted_velocity = distributed_model(current, discrete_time)
                loss = (predicted_velocity - true_velocity).pow(2).mean()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                count += 1

            aggregate = torch.tensor([loss_sum, float(count)], device=device, dtype=torch.float64)
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
            epoch_loss = float((aggregate[0] / aggregate[1]).item())
            losses.append(epoch_loss)
            completed_epoch = epoch + 1
            should_save = (
                completed_epoch % args.checkpoint_every == 0
                or completed_epoch == args.epochs
                or completed_epoch == stop_epoch
            )
            dist.barrier(device_ids=[local_rank])
            if is_primary and should_save:
                _atomic_save(
                    checkpoint_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": completed_epoch,
                        "losses": losses,
                        "reproduction": {
                            "official_commit": ("b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"),
                            "task": "square",
                            "policy_type": "flow",
                            "world_size": world_size,
                            "global_batch_size": args.global_batch_size,
                            "precision": "float32",
                            "seed": args.seed,
                        },
                    },
                )
            dist.barrier(device_ids=[local_rank])
            metric = {
                "epoch": completed_epoch,
                "loss": epoch_loss,
                "epoch_seconds": time.monotonic() - epoch_started,
                "elapsed_seconds": time.monotonic() - started,
                "max_memory_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
            }
            if is_primary:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metric, sort_keys=True) + "\n")
                print(json.dumps(metric, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
