"""Collect five low-dimensional demonstrations for DynaMAC Table I tasks.

This module is intentionally Python 3.8 compatible.  The task variation and
random seed are explicit local protocol choices because the paper does not
publish its demonstration manifest.  Existing episode files are never
overwritten.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
from pathlib import Path

import numpy as np

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
DEFAULT_DATA_ROOT = INTEGRATION_ROOT / "data" / "training" / "main"
TASKS = {
    "stack_wine": ("rlbench.tasks.stack_wine", "StackWine"),
    "place_cups": ("rlbench.tasks.place_cups", "PlaceCups"),
    "open_microwave": ("rlbench.tasks.open_microwave", "OpenMicrowave"),
    "wipe_desk": ("rlbench.tasks.wipe_desk", "WipeDesk"),
}


def _make_environment(headless):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    return Environment(
        action_mode=MoveArmThenGripper(JointPosition(), Discrete()),
        obs_config=observation_config,
        headless=headless,
    )


def _episode_path(data_root, task, episode):
    return (
        data_root
        / task
        / "all_variations"
        / "episodes"
        / f"episode{episode}"
    )


def collect_task(environment, task, data_root, demonstrations, seed, variation):
    module_name, class_name = TASKS[task]
    task_class = getattr(importlib.import_module(module_name), class_name)
    task_environment = environment.get_task(task_class)
    if variation < 0 or variation >= task_environment.variation_count():
        raise ValueError(
            f"{task} variation {variation} outside "
            f"[0, {task_environment.variation_count()})"
        )

    records = []
    for episode in range(demonstrations):
        output = _episode_path(data_root, task, episode)
        low_dim = output / "low_dim_obs.pkl"
        if output.exists() or low_dim.exists():
            raise FileExistsError(f"refusing to overwrite existing episode: {output}")
        episode_seed = seed + episode
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        task_environment.set_variation(variation)
        descriptions, _ = task_environment.reset()
        demo, = task_environment.get_demos(amount=1, live_demos=True)
        output.mkdir(parents=True, exist_ok=False)
        with low_dim.open("wb") as stream:
            pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (output / "variation_number.pkl").open("wb") as stream:
            pickle.dump(variation, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (output / "variation_descriptions.pkl").open("wb") as stream:
            pickle.dump(descriptions, stream, protocol=pickle.HIGHEST_PROTOCOL)
        record = {
            "episode": episode,
            "seed": episode_seed,
            "variation": variation,
            "observations": len(demo),
            "path": low_dim.relative_to(data_root).as_posix(),
        }
        records.append(record)
        print(
            f"{task} demo {episode + 1}/{demonstrations}: {len(demo)} observations",
            flush=True,
        )
    return records


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=sorted(TASKS) + ["all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--demonstrations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variation", type=int, default=0)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.demonstrations < 1 or args.seed < 0:
        raise ValueError("demonstrations must be positive and seed non-negative")
    tasks = list(TASKS) if not args.task or args.task == ["all"] else args.task
    environment = _make_environment(args.headless)
    launched = False
    manifest = {
        "schema": "dynamac-table-i-live-demos-v1",
        "protocol_label": "local_table_i_v1",
        "paper_comparable": False,
        "claim_boundary": (
            "Live demonstrations from the pinned public RLBench fork; not the "
            "paper cohort. Seed and fixed variation are local explicit choices."
        ),
        "seed": args.seed,
        "variation": args.variation,
        "demonstrations_per_task": args.demonstrations,
        "tasks": {},
    }
    try:
        environment.launch()
        launched = True
        for task in tasks:
            manifest["tasks"][task] = collect_task(
                environment,
                task,
                args.data_root,
                args.demonstrations,
                args.seed,
                args.variation,
            )
    finally:
        if launched:
            environment.shutdown()
    manifest_path = args.data_root / "collection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
