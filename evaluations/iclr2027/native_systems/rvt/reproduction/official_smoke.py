"""Validate the pinned RVT-2 checkpoint on server B.

This script deliberately stops at a nominal-policy smoke.  It does not read
the shared Main-10, Stress-4, or Native-6 manifests owned by server A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "367995a1a2169b6352bf4e8b0ed405890462a3a0"
EXPECTED_CHECKPOINT_SHA256 = (
    "25b4cdeae72ac98ecf7eac79604d6553f50d4f8bef6cb2610631875b42cf437f"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _describe(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all().item()),
        }
    if isinstance(value, dict):
        return {key: _describe(item) for key, item in value.items()}
    return type(value).__name__


def _task_reset() -> dict[str, Any]:
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import CloseJar

    observation_config = ObservationConfig()
    observation_config.set_all(True)
    action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
    environment = Environment(
        action_mode,
        dataset_root="",
        obs_config=observation_config,
        headless=True,
    )
    try:
        environment.launch()
        task = environment.get_task(CloseJar)
        descriptions, observation = task.reset()
        return {
            "status": "pass",
            "task": "close_jar",
            "descriptions": descriptions,
            "front_rgb_shape": list(observation.front_rgb.shape),
            "front_point_cloud_shape": list(observation.front_point_cloud.shape),
            "low_dim_shape": list(observation.get_low_dim_data().shape),
        }
    finally:
        environment.shutdown()


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    checkpoint = args.checkpoint.resolve()
    commit = _git_head(repository)
    checkpoint_sha256 = _sha256(checkpoint)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"RVT commit mismatch: {commit}")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"RVT-2 checkpoint checksum mismatch: {checkpoint_sha256}")

    # Upstream imports `mvt` as a top-level package and therefore expects its
    # rvt directory to be the working directory, exactly as its README states.
    upstream_workdir = repository / "rvt"
    sys.path.insert(0, str(upstream_workdir))
    sys.path.insert(0, str(repository))
    os.chdir(upstream_workdir)

    import torch
    from rvt.eval import load_agent

    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}"
    agent = load_agent(model_path=str(checkpoint), device=args.device)
    point_cloud = [
        torch.rand(args.points, 3, device=device, dtype=torch.float32) * 1.8 - 0.9
    ]
    image_features = [
        torch.rand(args.points, 3, device=device, dtype=torch.float32) * 2.0 - 1.0
    ]
    proprioception = torch.zeros(1, 4, device=device)
    language = torch.zeros(1, 77, 512, device=device)
    with torch.inference_mode():
        output = agent._network(
            pc=point_cloud,
            img_feat=image_features,
            proprio=proprioception,
            lang_emb=language,
            img_aug=0,
        )

    result: dict[str, Any] = {
        "status": "pass",
        "scope": "official_checkpoint_synthetic_tensor_gpu_forward",
        "official_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": args.seed,
        "synthetic_points": args.points,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": device,
        "device_name": torch.cuda.get_device_name(args.device),
        "compute_capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(args.device)
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in agent._network.parameters()
        ),
        "outputs": _describe(output),
        "peak_memory_mib": round(
            torch.cuda.max_memory_allocated(args.device) / 1024**2, 3
        ),
        "formal_evaluation": "not_run_requires_server_a_native6_manifest",
    }
    if args.task_reset:
        result["official_task_reset"] = _task_reset()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("/home/ubuntu/workspace/_external/RVT"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/ubuntu/workspace/_models/rvt/rvt2/model_99.pth"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--task-reset", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
