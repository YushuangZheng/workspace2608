"""Collect the five development demonstrations for ICLR 2027 task assets.

This is an A1-only collection utility.  It writes task-scoped manifests and
never overwrites a completed episode or manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import random
from pathlib import Path

import numpy as np

from integrations.rlbench.iclr2027.task_registry import TASKS, experiment_task
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT

DEFAULT_ROOT = INTEGRATION_ROOT / "data" / "iclr2027" / "demonstrations"
DEFAULT_SEED = 2_707_100_000
SCHEMA = "essay2608.iclr2027.demonstrations.v1"


def _environment(headless: bool):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    observation = ObservationConfig()
    observation.set_all(False)
    observation.gripper_open = True
    observation.gripper_pose = True
    observation.task_low_dim_state = True
    return Environment(
        action_mode=MoveArmThenGripper(JointPosition(), Discrete()),
        obs_config=observation,
        headless=headless,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_dir(root: Path, task_id: str, episode: int) -> Path:
    return root / task_id / "all_variations" / "episodes" / f"episode{episode}"


def _variation(task_environment, episode: int, fixed: int | None) -> int:
    if fixed is not None:
        # Fixed-level wrappers expose their selected base variation as wrapper
        # variation zero; the base variation remains explicit in the manifest.
        return 0
    return episode % int(task_environment.variation_count())


def collect_task(
    environment,
    task_id: str,
    *,
    root: Path,
    demonstrations: int,
    seed: int,
) -> Path:
    task = experiment_task(task_id)
    if task.spec.bimanual:
        raise ValueError(f"{task_id} is a reused bimanual asset, not collected here")
    module = importlib.import_module(task.spec.module)
    task_class = getattr(module, task.spec.class_name)
    task_environment = environment.get_task(task_class)
    task_root = root / task_id
    manifest_path = task_root / "collection_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite completed task: {manifest_path}")
    records = []
    for episode in range(demonstrations):
        output = _episode_dir(root, task_id, episode)
        low_dim = output / "low_dim_obs.pkl"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite partial episode: {output}")
        episode_seed = seed + episode
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        wrapper_variation = _variation(
            task_environment, episode, task.fixed_base_variation
        )
        task_environment.set_variation(wrapper_variation)
        descriptions, reset_observation = task_environment.reset()
        raw = reset_observation.task_low_dim_state
        task.spec.extract_pose_chunks(raw)
        task.spec.extract_entity_configurations(raw)
        (demo,) = task_environment.get_demos(amount=1, live_demos=True)
        for observation in demo:
            task.spec.extract_pose_chunks(observation.task_low_dim_state)
            task.spec.extract_entity_configurations(observation.task_low_dim_state)
        output.mkdir(parents=True, exist_ok=False)
        with low_dim.open("wb") as stream:
            pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (output / "variation_number.pkl").open("wb") as stream:
            pickle.dump(wrapper_variation, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (output / "variation_descriptions.pkl").open("wb") as stream:
            pickle.dump(descriptions, stream, protocol=pickle.HIGHEST_PROTOCOL)
        records.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "wrapper_variation": wrapper_variation,
                "base_variation": (
                    task.fixed_base_variation
                    if task.fixed_base_variation is not None
                    else wrapper_variation
                ),
                "observations": len(demo),
                "low_dim_size": task.spec.expected_low_dim_size,
                "path": low_dim.relative_to(root).as_posix(),
                "sha256": _sha256(low_dim),
            }
        )
        print(
            f"{task_id}: demo {episode + 1}/{demonstrations} ({len(demo)} states)",
            flush=True,
        )
    task_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "status": "A1_DEVELOPMENT_DEMONSTRATIONS",
        "task_id": task_id,
        "base_task": task.base_task,
        "task_level": task.task_level,
        "task_spec": {
            "module": task.spec.module,
            "class_name": task.spec.class_name,
            "frame_names": list(task.spec.frame_names),
            "action_frame_names": list(task.spec.action_frame_names),
            "scene_entity_names": list(task.spec.scene_entity_names),
            "configuration_schema": task.spec.configuration_schema,
        },
        "demonstrations": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    available = sorted(
        task_id for task_id, task in TASKS.items() if not task.spec.bimanual
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=available + ["all"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--demonstrations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.demonstrations != 5:
        raise ValueError("A1 freezes exactly five demonstrations per task")
    available = sorted(
        task_id for task_id, task in TASKS.items() if not task.spec.bimanual
    )
    selected = available if not args.task or args.task == ["all"] else args.task
    environment = _environment(args.headless)
    launched = False
    try:
        environment.launch()
        launched = True
        for offset, task_id in enumerate(selected):
            path = collect_task(
                environment,
                task_id,
                root=args.root,
                demonstrations=args.demonstrations,
                seed=args.seed + 1000 * offset,
            )
            print(f"wrote {path}", flush=True)
    finally:
        if launched:
            environment.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
