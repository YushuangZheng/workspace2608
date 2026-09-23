"""Run one real-data optimizer step with the pinned FAIL-Detect Square policy.

This is a reproduction readiness check, not a trained-policy result.  It loads
the official Square configuration and absolute-action dataset, instantiates the
official policy, and verifies one finite forward/backward/update cycle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

PINNED_COMMIT = "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"
DEFAULT_OFFICIAL_ROOT = Path("/home/ubuntu/workspace/_external/FAIL-Detect")
DEFAULT_DATASET = Path("/home/ubuntu/workspace/_datasets/robomimic-v0.1/square/ph/image_abs.hdf5")
CONFIG_NAME = "image_square_ph_visual_flow_policy_cnn.yaml"


def _official_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_smoke(official_root: Path, dataset_path: Path, device: str) -> dict[str, object]:
    import sys

    import torch
    from torch.utils.data import DataLoader

    commit = _official_commit(official_root)
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"official checkout drifted: expected {PINNED_COMMIT}, got {commit}")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    torch_device = torch.device(device)
    cuda_device_index = None
    if torch_device.type == "cuda":
        cuda_device_index = (
            torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        )

    sys.path.insert(0, str(official_root))
    from diffusion_policy.common.pytorch_util import dict_apply

    torch.manual_seed(2608)
    np.random.seed(2608)
    if cuda_device_index is not None:
        torch.cuda.set_device(cuda_device_index)

    cfg = OmegaConf.load(official_root / "diffusion_policy" / "configs_robomimic" / CONFIG_NAME)
    cfg.task.dataset.dataset_path = str(dataset_path)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)
    model.to(torch_device)
    model.train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    batch = dict_apply(batch, lambda value: value.to(torch_device))

    first_parameter = next(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.numel()
    )
    before = first_parameter.detach().clone()
    loss = model.compute_loss(batch)
    if not bool(torch.isfinite(loss).item()):
        raise AssertionError("official Square policy returned a non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    if not bool(torch.isfinite(grad_norm).item()):
        raise AssertionError("official Square policy returned non-finite gradients")
    optimizer.step()
    max_parameter_update = float((first_parameter.detach() - before).abs().max().cpu().item())
    if max_parameter_update <= 0:
        raise AssertionError("optimizer step did not update the first policy parameter")

    observation_shapes = {key: list(value.shape) for key, value in batch["obs"].items()}
    report: dict[str, object] = {
        "status": "pass",
        "scope": "official_square_real_data_one_optimizer_step",
        "official_commit": commit,
        "official_config": CONFIG_NAME,
        "dataset": str(dataset_path),
        "dataset_samples": len(dataset),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": device,
        "observation_shapes": observation_shapes,
        "action_shape": list(batch["action"].shape),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(loss.detach().cpu().item()),
        "gradient_norm": float(grad_norm.detach().cpu().item()),
        "max_first_parameter_update": max_parameter_update,
    }
    if cuda_device_index is not None:
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(cuda_device_index)
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(cuda_device_index)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_smoke(args.official_root.resolve(), args.dataset.resolve(), args.device)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
