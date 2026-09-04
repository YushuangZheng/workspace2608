# ruff: noqa: UP007, UP045
"""Run the pinned RVT checkpoint in live, seeded RLBench episodes.

This is a development reproduction gate.  It deliberately does not consume
server A's frozen Native-6 manifest or claim a formal E6 result.  The released
RACER seed bank is used only to make simulator resets repeatable when it is
available; otherwise a deterministic NumPy seed is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .official_smoke import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COMMIT,
    _git_head,
    _sha256,
)


def _percentiles(
    values: list[float],
) -> dict[str, Optional[Union[float, int]]]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": round(float(array.mean()), 3),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
    }


def _seed_file(seed_directory: Optional[Path], episode: int) -> Optional[Path]:
    if seed_directory is None:
        return None
    candidate = seed_directory / f"random_seed{episode}.pkl"
    return candidate if candidate.is_file() else None


def _configure_upstream(repository: Path) -> Path:
    upstream_workdir = repository / "rvt"
    sys.path.insert(0, str(upstream_workdir))
    sys.path.insert(0, str(repository))
    os.chdir(upstream_workdir)
    return upstream_workdir


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    checkpoint = args.checkpoint.resolve()
    seed_directory = (
        args.seed_directory.resolve() if args.seed_directory is not None else None
    )
    commit = _git_head(repository)
    checkpoint_sha256 = _sha256(checkpoint)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"RVT commit mismatch: {commit}")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"RVT-2 checkpoint checksum mismatch: {checkpoint_sha256}")
    if seed_directory is not None and not seed_directory.is_dir():
        raise FileNotFoundError(seed_directory)

    _configure_upstream(repository)

    import rvt.eval as upstream_eval
    import torch
    from rvt.utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2
    from yarr.utils.process_str import change_case

    reset_records: list[dict[str, Any]] = []

    class LiveSeededMultiTaskEnv(CustomMultiTaskRLBenchEnv2):
        """Replace only demo loading with a deterministic live task reset."""

        def reset_to_demo(self, i: int, variation_number: int = -1) -> dict:
            if self._episodes_this_task == self._swap_task_every:
                self._set_new_task()
                self._episodes_this_task = 0
            self._episodes_this_task += 1
            self._i = 0

            selected_variation = 0 if variation_number < 0 else variation_number
            self._task.set_variation(selected_variation)
            state_file = _seed_file(seed_directory, i)
            if state_file is None:
                effective_seed = args.seed + i
                np.random.seed(effective_seed)
                random.seed(effective_seed)
                state_sha256 = None
                seed_source = "generated_numpy_seed"
            else:
                with state_file.open("rb") as handle:
                    np.random.set_state(pickle.load(handle))
                random.seed(args.seed + i)
                state_sha256 = hashlib.sha256(state_file.read_bytes()).hexdigest()
                effective_seed = None
                seed_source = "released_racer_seed_bank"

            descriptions, observation = self._task.reset()
            self._lang_goal = descriptions[0]
            self._previous_obs_dict = self.extract_obs(observation)
            self._record_current_episode = (
                self.eval
                and self._record_every_n > 0
                and self._episode_index % self._record_every_n == 0
            )
            self._episode_index += 1
            self._recorded_images.clear()
            reset_records.append(
                {
                    "episode": i,
                    "task": change_case(self._task._task.__class__.__name__),
                    "variation": selected_variation,
                    "seed_source": seed_source,
                    "effective_seed": effective_seed,
                    "seed_state_sha256": state_sha256,
                    "language_goal": self._lang_goal,
                    "steps": 0,
                    "success": False,
                    "terminal": False,
                }
            )
            return self._previous_obs_dict

        def step(self, act_result: Any) -> Any:
            transition = super().step(act_result)
            if reset_records:
                record = reset_records[-1]
                record["steps"] = self._i
                record["success"] = bool(transition.reward >= 100.0)
                record["terminal"] = bool(transition.terminal)
            return transition

    upstream_eval.CustomMultiTaskRLBenchEnv = LiveSeededMultiTaskEnv
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    agent = upstream_eval.load_agent(model_path=str(checkpoint), device=args.device)
    action_latencies_ms: list[float] = []
    original_act = agent.act

    def timed_act(*act_args: Any, **act_kwargs: Any) -> Any:
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        output = original_act(*act_args, **act_kwargs)
        torch.cuda.synchronize(args.device)
        action_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return output

    agent.act = timed_act
    started = time.perf_counter()
    scores = upstream_eval.eval(
        agent=agent,
        tasks=args.tasks,
        eval_datafolder="",
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        replay_ground_truth=False,
        device=args.device,
        headless=True,
        logging=False,
        verbose=args.verbose,
        save_video=False,
    )
    elapsed_seconds = time.perf_counter() - started

    return {
        "status": "pass",
        "scope": "live_seeded_nominal_development_reproduction",
        "formal_evaluation": False,
        "formal_evaluation_reason": "requires_server_a_frozen_native6_manifest",
        "official_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "tasks": args.tasks,
        "start_episode": args.start_episode,
        "eval_episodes_per_task": args.eval_episodes,
        "episode_length": args.episode_length,
        "base_seed": args.seed,
        "seed_directory": str(seed_directory) if seed_directory is not None else None,
        "scores_percent": {
            task: float(score) for task, score in zip(args.tasks, scores)
        },
        "episodes": reset_records,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "action_latency": _percentiles(action_latencies_ms),
        "peak_memory_mib": round(
            torch.cuda.max_memory_allocated(args.device) / 1024**2, 3
        ),
        "device": f"cuda:{args.device}",
        "device_name": torch.cuda.get_device_name(args.device),
    }


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
    parser.add_argument(
        "--seed-directory",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/_external/RACER/"
            "racer/gradio_demo/random_seeds"
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=["close_jar"])
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
