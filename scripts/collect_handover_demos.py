"""Collect bimanual handover demonstrations in isolated Isaac Lab workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Essay2608-Bimanual-Handover-v0"

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--seed", type=int, default=7208)
parser.add_argument("--output_dir", type=Path, default=Path("data/handover_static/v1"))
parser.add_argument("--success_threshold", type=float, default=0.06)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def require_mutable_output(output_dir: Path) -> None:
    if (output_dir / "FROZEN").exists():
        raise RuntimeError(f"Refusing to overwrite frozen dataset: {output_dir}")


def run_controller() -> int:
    """Use a fresh simulator process per attempt to avoid autoreset deadlocks."""

    output_dir = args.output_dir.resolve()
    require_mutable_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    attempts = 0
    with tempfile.TemporaryDirectory(prefix=".handover_", dir=output_dir.parent) as temporary:
        while len(entries) < args.num_demos and attempts < args.max_attempts:
            attempts += 1
            seed = args.seed + attempts - 1
            worker_dir = Path(temporary) / f"attempt_{attempts:03d}"
            print(f"[handover] attempt={attempts} saved={len(entries)}/{args.num_demos} seed={seed}", flush=True)
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
            result = subprocess.run(command, check=False)
            demo_path = worker_dir / "demo_000.npz"
            manifest_path = worker_dir / "manifest.json"
            if result.returncode or not demo_path.is_file() or not manifest_path.is_file():
                continue
            worker_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(worker_manifest.get("demos", [])) != 1:
                continue
            final_path = output_dir / f"demo_{len(entries):03d}.npz"
            demo_path.replace(final_path)
            entry = dict(worker_manifest["demos"][0])
            entry.update({"file": final_path.name, "attempt": attempts, "seed": seed})
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
        raise RuntimeError(f"Collected only {len(entries)}/{args.num_demos} successful demonstrations.")
    return 0


if not args.worker:
    raise SystemExit(run_controller())


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import essay2608
from essay2608.bimanual import (
    ScriptedHandover,
    actions_to_robot_root_frames,
    get_handover_poses,
    write_attached_object,
)
from isaaclab_tasks.utils import parse_env_cfg


def _row(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().copy()


def run_worker() -> None:
    print("[handover] worker initializing", flush=True)
    output_dir = args.output_dir.resolve()
    require_mutable_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    print("[handover] creating environment", flush=True)
    env = gym.make(TASK_ID, cfg=env_cfg)
    print("[handover] resetting environment", flush=True)
    env.reset(seed=args.seed)
    print(f"[handover] action_shape={env.unwrapped.action_space.shape}", flush=True)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    expert = ScriptedHandover(control_dt, env.unwrapped.device)
    records = {
        key: []
        for key in (
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
    }
    complete = False
    environment_done = False
    for step_index in range(args.max_steps):
        with torch.inference_mode():
            if step_index == 0:
                print("[handover] reading initial poses", flush=True)
            left, right, object_pose, target = get_handover_poses(env)
            if step_index == 0:
                print(
                    f"[handover] left={_row(left)[:3].tolist()} right={_row(right)[:3].tolist()} "
                    f"object={_row(object_pose)[:3].tolist()}",
                    flush=True,
                )
            elif step_index % 250 == 0:
                print(
                    f"[handover] step={step_index} state={expert.state.name} "
                    f"left={_row(left)[:3].tolist()} right={_row(right)[:3].tolist()}",
                    flush=True,
                )
            state = int(expert.state)
            action, complete = expert.compute(left, right, object_pose, target)
            carrier = expert.carrier
            carrier_offset = expert.carrier_offset
            if carrier == "left":
                write_attached_object(env, left, carrier_offset)
            elif carrier == "right":
                write_attached_object(env, right, carrier_offset)
            simulator_action = actions_to_robot_root_frames(env, action)
            left_robot = env.unwrapped.scene["left_robot"]
            right_robot = env.unwrapped.scene["right_robot"]
            records["state"].append(state)
            records["left_ee_pose"].append(_row(left))
            records["right_ee_pose"].append(_row(right))
            records["object_pose"].append(_row(object_pose))
            records["target_pose"].append(_row(target))
            records["action"].append(_row(action))
            records["joint_pos"].append(
                np.concatenate((_row(left_robot.data.joint_pos), _row(right_robot.data.joint_pos)))
            )
            records["joint_vel"].append(
                np.concatenate((_row(left_robot.data.joint_vel), _row(right_robot.data.joint_vel)))
            )
            records["carrier"].append({None: 0, "left": 1, "right": 2}[carrier])
            if step_index == 0:
                print("[handover] stepping initial action", flush=True)
            _, _, terminated, truncated, _ = env.step(simulator_action)
            if step_index == 0:
                print("[handover] initial action complete", flush=True)
            environment_done = bool((terminated | truncated).any().item())
            if complete or environment_done:
                break

    final_left, final_right, final_object, target = get_handover_poses(env)
    final_error = float(torch.linalg.norm(final_object[0, :3] - target[0, :3]).item())
    success = bool(complete and not environment_done and final_error < args.success_threshold)
    entries = []
    if success:
        output_path = output_dir / "demo_000.npz"
        np.savez_compressed(
            output_path,
            time=np.arange(len(records["state"]), dtype=np.float32) * control_dt,
            state=np.asarray(records["state"], dtype=np.int64),
            left_ee_pose=np.asarray(records["left_ee_pose"], dtype=np.float32),
            right_ee_pose=np.asarray(records["right_ee_pose"], dtype=np.float32),
            object_pose=np.asarray(records["object_pose"], dtype=np.float32),
            target_pose=np.asarray(records["target_pose"], dtype=np.float32),
            action=np.asarray(records["action"], dtype=np.float32),
            joint_pos=np.asarray(records["joint_pos"], dtype=np.float32),
            joint_vel=np.asarray(records["joint_vel"], dtype=np.float32),
            carrier=np.asarray(records["carrier"], dtype=np.int64),
            control_dt=np.asarray(control_dt, dtype=np.float32),
            final_error=np.asarray(final_error, dtype=np.float32),
            quaternion_order=np.asarray("wxyz"),
            coordinate_frame=np.asarray("local_environment"),
        )
        entries.append(
            {
                "file": output_path.name,
                "steps": len(records["state"]),
                "final_error": final_error,
                "initial_object_pose": records["object_pose"][0].astype(float).tolist(),
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps({"task_id": TASK_ID, "num_demos": len(entries), "demos": entries}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "success": success,
                "final_error": final_error,
                "steps": len(records["state"]),
                "state": expert.state.name,
                "left_position": _row(final_left)[:3].tolist(),
                "right_position": _row(final_right)[:3].tolist(),
            }
        ),
        flush=True,
    )
    env.close()
    if not success:
        raise RuntimeError("Handover expert did not complete successfully.")


if __name__ == "__main__":
    try:
        run_worker()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
