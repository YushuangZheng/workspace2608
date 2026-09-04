"""Distributed, resumable training for the pinned FAIL-Detect Square policy.

The upstream workspace is single-process.  This launcher keeps its model,
loss, optimizer, EMA, dataset split, global batch size, scheduler horizon and
checkpoint format, while splitting each global batch over one process per
GPU.  Rollout evaluation is intentionally run after training in a separate
official evaluation job; changing evaluation cadence does not change weights.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


class _LossModule(torch.nn.Module):
    """Expose the upstream policy's ``compute_loss`` through ``forward``."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.policy.compute_loss(batch)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/_external/FAIL-Detect"),
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/ubuntu/workspace/_datasets/robomimic-v0.1/square/ph/image_abs.hdf5"
        ),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--workers-per-rank", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-epochs-this-run", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _atomic_checkpoint(workspace: Any, output_dir: pathlib.Path) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_dir / "latest.ckpt.tmp"
    final = checkpoint_dir / "latest.ckpt"
    workspace.save_checkpoint(path=temporary, use_thread=False)
    os.replace(temporary, final)


def _all_reduce_mean(value_sum: float, count: int, device: torch.device) -> float:
    payload = torch.tensor([value_sum, float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return float((payload[0] / payload[1].clamp_min(1)).item())


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
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(official_root)
    sys.path.insert(0, str(official_root))

    import hydra
    from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
    from diffusion_policy.dataset.base_dataset import BaseImageDataset
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
        TrainDiffusionUnetHybridWorkspace,
    )
    from omegaconf import OmegaConf

    config_path = (
        official_root / "diffusion_policy/configs_robomimic/"
        "image_square_ph_visual_flow_policy_cnn.yaml"
    )
    cfg = OmegaConf.load(config_path)
    cfg.training.seed = args.seed
    cfg.training.device = f"cuda:{local_rank}"
    cfg.training.num_epochs = args.epochs
    cfg.training.resume = not args.no_resume
    cfg.dataloader.batch_size = args.global_batch_size // world_size
    cfg.dataloader.num_workers = args.workers_per_rank
    cfg.dataloader.shuffle = False
    cfg.dataloader.persistent_workers = args.workers_per_rank > 0
    cfg.val_dataloader.batch_size = args.global_batch_size // world_size
    cfg.val_dataloader.num_workers = args.workers_per_rank
    cfg.val_dataloader.shuffle = False
    cfg.val_dataloader.persistent_workers = args.workers_per_rank > 0
    cfg.task.dataset.dataset_path = str(dataset_path)
    cfg.task.dataset_path = str(dataset_path)
    cfg.task.env_runner.dataset_path = str(dataset_path)
    cfg.logging.mode = "disabled"
    OmegaConf.resolve(cfg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    workspace = TrainDiffusionUnetHybridWorkspace(cfg, output_dir=str(output_dir))
    latest = output_dir / "checkpoints/latest.ckpt"
    if cfg.training.resume and latest.is_file():
        if is_primary:
            print(f"Resuming from {latest}", flush=True)
        workspace.load_checkpoint(path=latest, map_location="cpu")

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if not isinstance(dataset, BaseImageDataset):
        raise TypeError(f"unexpected dataset type: {type(dataset)}")
    validation_dataset = dataset.get_validation_dataset()
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader_kwargs = {
        "batch_size": args.global_batch_size // world_size,
        "num_workers": args.workers_per_rank,
        "pin_memory": True,
        "persistent_workers": args.workers_per_rank > 0,
    }
    train_loader = DataLoader(dataset, sampler=train_sampler, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, sampler=validation_sampler, **loader_kwargs)
    normalizer = dataset.get_normalizer()
    workspace.model.set_normalizer(normalizer)
    if workspace.ema_model is not None:
        workspace.ema_model.set_normalizer(normalizer)

    workspace.model.to(device)
    if workspace.ema_model is not None:
        workspace.ema_model.to(device)
    optimizer_to(workspace.optimizer, device)
    loss_module = _LossModule(workspace.model)
    distributed_model = DistributedDataParallel(
        loss_module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        # The pinned policy registers two compatibility parameters that its
        # flow-matching loss does not consume.  Upstream single-GPU training
        # simply leaves their gradients unset, so preserve that behavior.
        find_unused_parameters=False,
        static_graph=True,
        gradient_as_bucket_view=True,
    )
    scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=workspace.optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(len(train_loader) * args.epochs)
        // cfg.training.gradient_accumulate_every,
        last_epoch=workspace.global_step - 1,
    )
    ema: EMAModel | None = None
    if workspace.ema_model is not None:
        ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
        ema.optimization_step = workspace.global_step

    # The model initialization must be identical, but stochastic crops and
    # flow noise should not be duplicated across ranks during training.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    metadata = {
        "scope": "official_square_flow_policy_distributed_training",
        "official_commit": "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed",
        "config": config_path.name,
        "dataset": str(dataset_path),
        "epochs": args.epochs,
        "seed": args.seed,
        "world_size": world_size,
        "global_batch_size": args.global_batch_size,
        "local_batch_size": args.global_batch_size // world_size,
        "train_examples": len(dataset),
        "validation_examples": len(validation_dataset),
        "train_steps_per_epoch": len(train_loader),
        "precision": "float32",
        "rollout_during_training": False,
        "notes": [
            "Upstream model, loss, optimizer, EMA, split, scheduler horizon and "
            "global batch size are preserved.",
            "Official nominal rollout evaluation is executed after training so "
            "that simulator evaluation does not idle seven GPUs.",
        ],
    }
    if is_primary:
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    start_epoch = int(workspace.epoch)
    stop_epoch = args.epochs
    if args.max_epochs_this_run is not None:
        stop_epoch = min(stop_epoch, start_epoch + args.max_epochs_this_run)
    log_path = output_dir / "training_metrics.jsonl"
    run_started = time.monotonic()

    try:
        for epoch in range(start_epoch, stop_epoch):
            epoch_started = time.monotonic()
            train_sampler.set_epoch(epoch)
            epoch_seed = args.seed + rank + epoch * world_size
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            random.seed(epoch_seed)
            distributed_model.train()
            workspace.optimizer.zero_grad(set_to_none=True)
            train_sum = 0.0
            train_count = 0
            sample_batch = None
            for batch_index, batch in enumerate(train_loader):
                batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                if sample_batch is None:
                    sample_batch = batch
                should_sync = (
                    batch_index + 1
                ) % cfg.training.gradient_accumulate_every == 0 or batch_index + 1 == len(
                    train_loader
                )
                sync_context = nullcontext() if should_sync else distributed_model.no_sync()
                with sync_context:
                    raw_loss = distributed_model(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                if should_sync:
                    workspace.optimizer.step()
                    workspace.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if ema is not None:
                        ema.step(workspace.model)
                train_sum += float(raw_loss.detach())
                train_count += 1
                workspace.global_step += 1

            train_loss = _all_reduce_mean(train_sum, train_count, device)
            metrics: dict[str, Any] = {
                "epoch": epoch,
                "global_step": workspace.global_step,
                "train_loss": train_loss,
                "lr": scheduler.get_last_lr()[0],
            }

            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                workspace.model.eval()
                validation_sum = 0.0
                validation_count = 0
                with torch.no_grad():
                    for batch in validation_loader:
                        batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                        value = workspace.model.compute_loss(batch)
                        validation_sum += float(value)
                        validation_count += 1
                metrics["validation_loss"] = _all_reduce_mean(
                    validation_sum, validation_count, device
                )

            if (
                is_primary
                and sample_batch is not None
                and ((epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs)
            ):
                policy = workspace.ema_model or workspace.model
                policy.eval()
                with torch.no_grad():
                    result = policy.predict_action(sample_batch["obs"])
                    metrics["train_action_mse_error"] = float(
                        torch.nn.functional.mse_loss(result["action_pred"], sample_batch["action"])
                    )

            workspace.epoch = epoch + 1
            dist.barrier()
            should_checkpoint = (
                workspace.epoch % args.checkpoint_every == 0
                or workspace.epoch == args.epochs
                or workspace.epoch == stop_epoch
            )
            if is_primary and should_checkpoint:
                _atomic_checkpoint(workspace, output_dir)
            dist.barrier()

            metrics["epoch_seconds"] = time.monotonic() - epoch_started
            metrics["elapsed_seconds"] = time.monotonic() - run_started
            metrics["max_memory_allocated_mib_per_rank"] = float(
                torch.cuda.max_memory_allocated(device) / 1024**2
            )
            max_memory = torch.tensor(metrics["max_memory_allocated_mib_per_rank"], device=device)
            dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)
            metrics["max_memory_allocated_mib"] = float(max_memory)
            if is_primary:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
