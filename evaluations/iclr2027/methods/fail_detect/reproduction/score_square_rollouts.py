"""Run targeted official Square rollouts and record only logpZO scores.

The public evaluation runner computes every paper baseline in one process.
This reproduction keeps its policy, simulator, environment modification and
logpZO implementations, while replacing unrelated baseline calculations and
STAC's 256 repeated samples with no-op placeholders.  Those values are not
written to the result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time
import types
from typing import Any

import dill
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/_external/FAIL-Detect"),
    )
    parser.add_argument("--policy-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--logpzo-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=500)
    parser.add_argument("--parallel-envs", type=int, default=25)
    parser.add_argument("--modify", action="store_true")
    return parser.parse_args()


def _zero_for_batch(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros(tensor.shape[0], device=tensor.device)


def main() -> None:
    args = _parse_args()
    if args.num_rollouts <= 0 or args.parallel_envs <= 0:
        raise ValueError("rollout and environment counts must be positive")
    if args.num_rollouts % args.parallel_envs:
        raise ValueError("num-rollouts must be divisible by parallel-envs")

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    sys.path.insert(0, str(official_root / "UQ_test"))
    sys.path.insert(0, str(official_root / "UQ_baselines"))
    os.chdir(official_root / "UQ_test")

    # Import through importlib because the pinned filename contains a hyphen.
    import eval_load_baseline as baseline
    import hydra

    runner_module = importlib.import_module(
        "diffusion_policy.env_runner.robomimic_image_runner_FAIL-Detect"
    )
    from CFM.net_CFM import get_unet
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
        TrainDiffusionUnetHybridWorkspace,
    )

    device = torch.device(args.device)
    policy_payload = torch.load(
        args.policy_checkpoint.resolve(), map_location="cpu", pickle_module=dill
    )
    cfg = policy_payload["cfg"]
    cfg.training.device = args.device
    cfg.policy.num_inference_steps = 1
    cfg.policy._target_ = (
        "diffusion_policy.policy.flow_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy"
    )
    cfg.task.dataset.dataset_path = str(args.dataset.resolve())
    cfg.task.dataset_path = str(args.dataset.resolve())
    cfg.task.env_runner.dataset_path = str(args.dataset.resolve())
    cfg.task.env_runner._target_ = (
        "diffusion_policy.env_runner.robomimic_image_runner_FAIL-Detect.RobomimicImageRunner"
    )
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = args.num_rollouts
    cfg.task.env_runner.n_test_vis = 0
    cfg.task.env_runner.test_start_seed = args.start_seed
    cfg.task.env_runner.n_envs = args.parallel_envs

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_dir = output_path.parent / f"{output_path.stem}_media"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    workspace = TrainDiffusionUnetHybridWorkspace(cfg, output_dir=str(rollout_dir))
    workspace.load_payload(policy_payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model
    if policy is None:
        raise ValueError("policy checkpoint has no EMA model")
    policy.to(device).eval()

    logpzo_payload = torch.load(
        args.logpzo_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    logpzo = get_unet(10).to(device).eval()
    logpzo.load_state_dict(logpzo_payload["model"], strict=True)
    logpzo.global_eps = None

    # The runner imports this module again by its short name; patch functions
    # in-place so all unrelated methods return correctly shaped placeholders.
    baseline.DER_UQ = lambda _model, observation, _task: _zero_for_batch(observation)
    baseline.RND_UQ = lambda _model, action, _observation: _zero_for_batch(action)
    baseline.CFM_UQ = lambda _model, observation, task_name=None: _zero_for_batch(observation)
    baseline.logpO_UQ = lambda _model, observation, task_name=None: _zero_for_batch(observation)
    baseline.NatPN_UQ = lambda _model, observation: _zero_for_batch(observation)
    baseline.PCA_kmeans_UQ = lambda _model, observation: _zero_for_batch(observation)
    baseline.STAC_UQ = lambda _previous, current: _zero_for_batch(current)
    # The upstream runner saves every observation frame even when videos are
    # disabled.  Suppress only this diagnostic side effect.
    runner_module.plot_and_save_images = lambda *unused_args, **unused_kwargs: None

    original_predict = policy.predict_action
    cached_action_prediction: torch.Tensor | None = None

    def efficient_predict(
        self: Any, observations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        nonlocal cached_action_prediction
        if self.num_rep == 1:
            result = original_predict(observations)
            cached_action_prediction = result["action_pred"].detach()
            return result
        if self.num_rep == 256 and cached_action_prediction is not None:
            return {"action_pred": cached_action_prediction.repeat_interleave(256, dim=0)}
        raise RuntimeError(f"unexpected upstream sampling multiplicity {self.num_rep}")

    policy.predict_action = types.MethodType(efficient_predict, policy)
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(rollout_dir))
    runner.curr_shape = 84
    runner.baseline_model = object()
    runner.baseline_model_RND = object()
    runner.baseline_model_CFM = object()
    runner.baseline_model_logpZO = logpzo
    runner.baseline_model_natpn = object()
    runner.baseline_model_PCA_kmeans = object()
    runner.task_name = "square"
    runner.modify_t = 50

    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        with torch.no_grad():
            raw = runner.run(policy, modify=args.modify)
    finally:
        runner.env.close()
    elapsed = time.monotonic() - started

    episodes = []
    prefix = "test/sim_max_reward_"
    for key, value in raw.items():
        if not key.startswith(prefix):
            continue
        seed = int(key.removeprefix(prefix))
        if not isinstance(value, list) or len(value) < 5:
            raise ValueError(f"unexpected upstream result for seed {seed}")
        scores = [float(item) for item in value[4].split("/")]
        episodes.append(
            {
                "seed": seed,
                "success": int(float(value[0]) > 0.0),
                "logpzo": scores,
            }
        )
    episodes.sort(key=lambda item: item["seed"])
    if len(episodes) != args.num_rollouts:
        raise ValueError(f"expected {args.num_rollouts} episodes, got {len(episodes)}")
    result = {
        "status": "pass",
        "scope": "official_square_flow_policy_logpzo_rollouts",
        "official_commit": "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed",
        "condition": "ood_modify" if args.modify else "id_nominal",
        "start_seed": args.start_seed,
        "num_rollouts": args.num_rollouts,
        "parallel_envs": args.parallel_envs,
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "successes": sum(item["success"] for item in episodes),
        "episodes": episodes,
        "notes": [
            "Unrelated public-code baselines were disabled; policy, simulator, "
            "environment modification and logpZO score paths are pinned upstream code.",
            "The public runner's unconditional PNG diagnostic was disabled; it does "
            "not affect observations, actions, scores or success conditions.",
        ],
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        f"saved {len(episodes)} {result['condition']} episodes, "
        f"successes={result['successes']}, elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
