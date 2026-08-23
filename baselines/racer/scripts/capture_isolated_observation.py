#!/usr/bin/env python3
"""Capture the same reset after loading RACER policy code in the parent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))
from isolated_simulator_proxy import SpawnIsolatedRLBenchSim
from observation_fingerprint import snapshot
from racer.evaluation import rollout as _rollout  # noqa: F401: conflict-side import is intentional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="place_cups")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    simulator = None
    payload = None
    try:
        simulator = SpawnIsolatedRLBenchSim(
            task_name=args.task,
            dataset_root=args.dataset_root,
            episode_length=args.episode_length,
        )
        obs_dict, observation = simulator.reset(args.episode)
        payload = {
            "schema": "racer_initial_observation_v1",
            "capture_mode": "policy-parent-spawn-isolated-simulator",
            "task": args.task,
            "episode": args.episode,
            "snapshot": snapshot(obs_dict, observation),
            "worker_exit_requirement": "explicit-close-natural-status-0",
        }
    finally:
        if simulator is not None:
            simulator.close()
    # proxy.close() above requires a natural worker status 0 before evidence is
    # published for the comparator.
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
