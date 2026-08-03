"""Collect five static bilateral lift-tray demonstrations in isolated workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Essay2608-Bimanual-Lift-Tray-v0"
parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--seed", type=int, default=9208)
parser.add_argument("--output_dir", type=Path, default=Path("data/lift_tray_static/v1"))
parser.add_argument("--success_threshold", type=float, default=0.06)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def require_mutable(path: Path) -> None:
    if (path / "FROZEN").exists():
        raise RuntimeError(f"Refusing to overwrite frozen dataset: {path}")


def run_controller() -> int:
    output_dir = args.output_dir.resolve()
    require_mutable(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    attempts = 0
    with tempfile.TemporaryDirectory(prefix=".tray_", dir=output_dir.parent) as temporary:
        while len(entries) < args.num_demos and attempts < args.max_attempts:
            attempts += 1
            seed = args.seed + attempts - 1
            worker_dir = Path(temporary) / f"attempt_{attempts:03d}"
            print(f"[tray] attempt={attempts} saved={len(entries)}/{args.num_demos} seed={seed}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
                "--headless",
                "--num_demos",
                "1",
                "--max_attempts",
                "1",
                "--seed",
                str(seed),
                "--output_dir",
                str(worker_dir),
            ]
            completed = subprocess.run(command, check=False)
            demo = worker_dir / "demo_000.npz"
            worker_manifest = worker_dir / "manifest.json"
            if completed.returncode or not demo.is_file() or not worker_manifest.is_file():
                continue
            payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
            if len(payload.get("demos", [])) != 1:
                continue
            destination = output_dir / f"demo_{len(entries):03d}.npz"
            demo.replace(destination)
            entry = dict(payload["demos"][0])
            entry.update(file=destination.name, attempt=attempts, seed=seed)
            entries.append(entry)
    manifest = {
        "task_id": TASK_ID,
        "num_demos": len(entries),
        "attempts": attempts,
        "base_seed": args.seed,
        "quaternion_order": "wxyz",
        "coordinate_frame": "local_environment",
        "demos": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if len(entries) != args.num_demos:
        raise RuntimeError(f"Collected only {len(entries)}/{args.num_demos} tray demonstrations.")
    return 0


if not args.worker:
    raise SystemExit(run_controller())


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import essay2608
from essay2608.bimanual import actions_to_robot_root_frames
from essay2608.tray import ScriptedLiftTray, get_tray_poses, write_bilateral_tray
from isaaclab_tasks.utils import parse_env_cfg


def row(value: torch.Tensor) -> np.ndarray:
    return value[0].detach().cpu().numpy().copy()


def run_worker() -> None:
    output_dir = args.output_dir.resolve()
    require_mutable(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args.seed)
    dt = env_cfg.sim.dt * env_cfg.decimation
    expert = ScriptedLiftTray(dt, env.unwrapped.device)
    keys = (
        "state",
        "left_ee_pose",
        "right_ee_pose",
        "object_pose",
        "target_pose",
        "action",
        "joint_pos",
        "joint_vel",
        "carrier",
    )
    records = {key: [] for key in keys}
    midpoint_offset = None
    environment_done = False
    complete = False
    for _ in range(args.max_steps):
        with torch.inference_mode():
            left, right, tray, target = get_tray_poses(env)
            state = int(expert.state)
            action, complete = expert.compute(left, right, tray, target)
            if expert.connected and midpoint_offset is None:
                midpoint_offset = tray[:, :3] - 0.5 * (left[:, :3] + right[:, :3])
            if expert.connected:
                write_bilateral_tray(env, left, right, midpoint_offset)
            left_robot = env.unwrapped.scene["left_robot"]
            right_robot = env.unwrapped.scene["right_robot"]
            records["state"].append(state)
            records["left_ee_pose"].append(row(left))
            records["right_ee_pose"].append(row(right))
            records["object_pose"].append(row(tray))
            records["target_pose"].append(row(target))
            records["action"].append(row(action))
            records["joint_pos"].append(np.concatenate((row(left_robot.data.joint_pos), row(right_robot.data.joint_pos))))
            records["joint_vel"].append(np.concatenate((row(left_robot.data.joint_vel), row(right_robot.data.joint_vel))))
            records["carrier"].append(3 if expert.connected else 0)
            simulator_action = actions_to_robot_root_frames(env, action)
            _, _, terminated, truncated, _ = env.step(simulator_action)
            environment_done = bool((terminated | truncated).any().item())
            if complete or environment_done:
                break
    _, _, final_tray, target = get_tray_poses(env)
    final_error = float(torch.linalg.norm(final_tray[0, :3] - target[0, :3]).item())
    success = bool(complete and not environment_done and final_error < args.success_threshold)
    entries = []
    if success:
        path = output_dir / "demo_000.npz"
        np.savez_compressed(
            path,
            time=np.arange(len(records["state"]), dtype=np.float32) * dt,
            **{key: np.asarray(value) for key, value in records.items()},
            control_dt=np.asarray(dt, dtype=np.float32),
            final_error=np.asarray(final_error, dtype=np.float32),
            quaternion_order=np.asarray("wxyz"),
            coordinate_frame=np.asarray("local_environment"),
        )
        entries.append(
            {
                "file": path.name,
                "steps": len(records["state"]),
                "final_error": final_error,
                "initial_object_pose": records["object_pose"][0].astype(float).tolist(),
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps({"task_id": TASK_ID, "num_demos": len(entries), "demos": entries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"success": success, "final_error": final_error, "steps": len(records["state"])}), flush=True)
    env.close()
    if not success:
        raise RuntimeError("Lift-tray expert did not complete successfully.")


if __name__ == "__main__":
    try:
        run_worker()
    finally:
        simulation_app.close()
