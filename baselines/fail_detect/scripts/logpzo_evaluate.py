#!/usr/bin/env python3
"""Thin Transport evaluator retaining only FAIL-Detect's released logpZO score."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_LABEL = "upstream_release_external_dp_checkpoint"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_existing(path, modify, start_seed, episodes):
    if not path.exists():
        return {}, []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "protocol_label": PROTOCOL_LABEL,
        "task": "transport",
        "policy_type": "diffusion",
        "modify": modify,
        "start_seed": start_seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("existing output mismatch for {}".format(key))
    records = payload.get("episodes", [])
    seeds = [record["seed"] for record in records]
    expected_prefix = list(range(start_seed, start_seed + len(records)))
    if seeds != expected_prefix:
        raise RuntimeError("existing output is not a contiguous resumable prefix")
    previous_target = payload.get("requested_episodes")
    if not isinstance(previous_target, int) or previous_target > episodes:
        raise RuntimeError("existing output target cannot be reduced")
    if len(records) > episodes:
        raise RuntimeError("existing output has more records than requested")
    if payload.get("complete") and len(records) != previous_target:
        raise RuntimeError("complete output has wrong episode count")
    if payload.get("complete") and len(records) == episodes:
        print("evaluation already complete: {}".format(path))
    return payload, records


def load_policy(upstream, checkpoint, device, output_dir, episodes, start_seed, parallel_envs):
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    with open_dict(cfg):
        cfg.task.env_runner._target_ = "diffusion_policy.env_runner.robomimic_image_runner_FAIL-Detect.RobomimicImageRunner"
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_train_vis = 0
        cfg.task.env_runner.n_test = episodes
        cfg.task.env_runner.n_test_vis = 0
        cfg.task.env_runner.test_start_seed = start_seed
        cfg.task.env_runner.n_envs = min(episodes, parallel_envs)
        cfg.policy.num_inference_steps = 70

    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg, output_dir=str(output_dir))
    workspace.model.load_state_dict(payload["state_dicts"]["model"], strict=True)
    workspace.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    policy.num_rep = 1
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    runner.curr_shape = 84
    return policy, runner


def load_logpzo(upstream, checkpoint, device):
    import torch

    from logpzo_network import build_logpzo_network

    network = build_logpzo_network(upstream, 20).to(device)
    payload = torch.load(str(checkpoint), map_location=device)
    result = network.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("logpZO strict load reported incompatible keys")
    network.eval()
    return network


def evaluate_missing(policy, runner, logpzo, modify, output_path, base_payload, existing):
    import numpy as np
    import torch
    import tqdm
    from diffusion_policy.common.pytorch_util import dict_apply

    upstream = Path(base_payload["upstream"])
    sys.path.insert(0, str(upstream / "UQ_test"))
    import eval_load_baseline as elb

    env = runner.env
    n_envs = len(runner.env_fns)
    n_inits = len(runner.env_init_fn_dills)
    n_chunks = int(math.ceil(n_inits / n_envs))
    records = list(existing)
    try:
        for chunk_idx in range(n_chunks):
            started = time.monotonic()
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            active = end - start
            init_functions = list(runner.env_init_fn_dills[start:end])
            if active < n_envs:
                init_functions.extend([runner.env_init_fn_dills[0]] * (n_envs - active))
            env.call_each("run_dill_function", args_list=[(value,) for value in init_functions])
            obs = env.reset()
            policy.reset()
            scores = []
            done = False
            modified = False
            action_steps = None
            progress = tqdm.tqdm(total=runner.max_steps, desc="Transport logpZO {}/{}".format(chunk_idx + 1, n_chunks), leave=False)
            while not done:
                actual_t = progress.n
                if modify and not modified and actual_t >= 50:
                    env.call_each("modify_environment", args_list=[(0.1, runner.render_obs_key)] * n_envs)
                    modified = True
                obs_dict = dict_apply(dict(obs), lambda value: torch.from_numpy(value).to(device=policy.device))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    score = elb.logpZO_UQ(logpzo, action_dict["global_cond"], task_name="transport")
                scores.append(score.detach().cpu())
                numpy_action = action_dict["action"].detach().cpu().numpy()
                if not np.all(np.isfinite(numpy_action)):
                    raise RuntimeError("policy produced NaN or Inf action")
                env_action = runner.undo_transform_action(numpy_action) if runner.abs_action else numpy_action
                obs, _, done_array, _ = env.step(env_action)
                done = bool(np.all(done_array))
                action_steps = int(numpy_action.shape[1])
                progress.update(action_steps)
            progress.close()
            if modify and modified:
                env.call_each("modify_environment", args_list=[(-0.1, runner.render_obs_key)] * n_envs)

            score_matrix = torch.stack(scores, dim=1).numpy()
            rewards = env.call("get_attr", "reward")[:active]
            for local_index in range(active):
                records.append({
                    "seed": int(runner.env_seeds[start + local_index]),
                    "success": bool(np.max(rewards[local_index]) >= 1),
                    "max_reward": float(np.max(rewards[local_index])),
                    "action_steps": action_steps,
                    "logpzo": [float(value) for value in score_matrix[local_index]],
                })
            base_payload["episodes"] = records
            base_payload["completed_episodes"] = len(records)
            base_payload["complete"] = len(records) == base_payload["requested_episodes"]
            base_payload["last_chunk_seconds"] = time.monotonic() - started
            base_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output_path, base_payload)
        env.reset()
    finally:
        try:
            env.close()
        except Exception:
            pass
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--logpzo-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--parallel-envs", type=int, default=10)
    parser.add_argument("--modify", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.episodes <= 0 or args.parallel_envs <= 0:
        parser.error("episodes and parallel-envs must be positive")

    import torch

    repo_root = args.repo_root.resolve()
    upstream = repo_root / "baselines/fail_detect/upstream"
    sys.path.insert(0, str(upstream))
    # The released configs keep dataset paths relative to their repository.
    os.chdir(str(upstream))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    with args.artifact_lock.open("r", encoding="utf-8") as handle:
        artifact_lock = json.load(handle)
    detector_sha = sha256_file(args.logpzo_checkpoint)
    previous_payload, existing = load_existing(
        args.output, args.modify, args.start_seed, args.episodes
    )
    expected_hashes = {
        "policy_checkpoint_sha256": artifact_lock["artifacts"]["transport_ph_dp_checkpoint"]["sha256"],
        "dataset_sha256": artifact_lock["artifacts"]["transport_image_abs"]["sha256"],
        "logpzo_checkpoint_sha256": detector_sha,
    }
    for key, expected in expected_hashes.items():
        if previous_payload and previous_payload.get(key) != expected:
            raise RuntimeError("existing output provenance mismatch for {}".format(key))
    if len(existing) == args.episodes:
        return
    first_missing_seed = args.start_seed + len(existing)
    missing = args.episodes - len(existing)

    payload = {
        "schema": "dynamac-fail-detect-logpzo-rollouts-v1",
        "protocol_label": PROTOCOL_LABEL,
        "claim_boundary": "Official external Diffusion Policy checkpoint; not the FAIL-Detect paper's 300-epoch policy checkpoint.",
        "task": "transport",
        "policy_type": "diffusion",
        "modify": args.modify,
        "start_seed": args.start_seed,
        "requested_episodes": args.episodes,
        "completed_episodes": len(existing),
        "complete": False,
        "num_inference_steps": 70,
        "parallel_envs": min(missing, args.parallel_envs),
        "upstream_commit": "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed",
        **expected_hashes,
        "created_at": previous_payload.get("created_at", datetime.now(timezone.utc).isoformat()),
        "episodes": existing,
        "upstream": str(upstream),
    }
    atomic_json(args.output, payload)

    policy, runner = load_policy(
        upstream, args.checkpoint, device, args.output.parent / "workspace",
        missing, first_missing_seed, args.parallel_envs,
    )
    logpzo = load_logpzo(upstream, args.logpzo_checkpoint, device)
    records = evaluate_missing(policy, runner, logpzo, args.modify, args.output, payload, existing)
    if len(records) != args.episodes:
        raise RuntimeError("evaluation completed with wrong episode count")


if __name__ == "__main__":
    main()
