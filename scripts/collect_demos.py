"""Collect static demonstrations with a fresh Isaac Lab process per attempt."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Essay2608-Dynamic-Pick-Place-v0"


parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--seed", type=int, default=2608)
parser.add_argument(
    "--output_dir",
    type=str,
    default="data/static_demos",
)
parser.add_argument(
    "--success_threshold",
    type=float,
    default=0.06,
)
parser.add_argument(
    "--collection_worker",
    action="store_true",
    help=argparse.SUPPRESS,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def run_collection_workers() -> int:
    """Collect each attempt in a fresh process to avoid reset deadlocks."""

    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_demos = 0
    attempts = 0
    manifest: list[dict] = []

    with tempfile.TemporaryDirectory(
        prefix=".collect_demos_",
        dir=output_dir.parent,
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)

        while (
            saved_demos < args_cli.num_demos
            and attempts < args_cli.max_attempts
        ):
            attempts += 1
            attempt_seed = args_cli.seed + attempts - 1
            worker_dir = temporary_root_path / f"attempt_{attempts:03d}"

            print(
                f"\n[collect] starting worker: attempt={attempts}, "
                f"saved={saved_demos}/{args_cli.num_demos}, "
                f"seed={attempt_seed}",
                flush=True,
            )

            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--collection_worker",
                "--num_demos",
                "1",
                "--max_attempts",
                "1",
                "--seed",
                str(attempt_seed),
                "--output_dir",
                str(worker_dir),
            ]
            result = subprocess.run(command, check=False)

            worker_demo = worker_dir / "demo_000.npz"
            worker_manifest_path = worker_dir / "manifest.json"
            if (
                result.returncode != 0
                or not worker_demo.is_file()
                or not worker_manifest_path.is_file()
            ):
                print(
                    f"[collect] worker attempt {attempts} failed; retrying.",
                    flush=True,
                )
                continue

            worker_manifest = json.loads(
                worker_manifest_path.read_text(encoding="utf-8")
            )
            worker_entries = worker_manifest.get("demos", [])
            if len(worker_entries) != 1:
                print(
                    f"[collect] worker attempt {attempts} produced an "
                    "invalid manifest; retrying.",
                    flush=True,
                )
                continue

            output_path = output_dir / f"demo_{saved_demos:03d}.npz"
            worker_demo.replace(output_path)

            entry = dict(worker_entries[0])
            entry.update(
                {
                    "file": output_path.name,
                    "attempt": attempts,
                    "seed": attempt_seed,
                }
            )
            manifest.append(entry)
            saved_demos += 1

            print(f"[collect] saved: {output_path}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "num_demos": saved_demos,
                "attempts": attempts,
                "seed": args_cli.seed,
                "quaternion_order": "wxyz",
                "coordinate_frame": "local_environment",
                "demos": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[collect] manifest: {manifest_path}", flush=True)
    print(f"[collect] saved demos: {saved_demos}", flush=True)

    if saved_demos < args_cli.num_demos:
        raise RuntimeError(
            f"Saved only {saved_demos}/{args_cli.num_demos} "
            f"demos after {attempts} attempts."
        )

    return 0


if not args_cli.collection_worker:
    raise SystemExit(run_collection_workers())


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import essay2608
from essay2608.expert import (
    GRIPPER_OPEN,
    ScriptedPickPlace,
    get_scene_poses,
    object_target_error,
)
from isaaclab_tasks.utils import parse_env_cfg


def tensor_row(tensor: torch.Tensor) -> np.ndarray:
    """Copy the first environment row to CPU NumPy."""

    return tensor[0].detach().cpu().numpy().copy()


def new_records() -> dict[str, list]:
    """Create an empty episode record."""

    return {
        "state": [],
        "ee_pose": [],
        "object_pose": [],
        "target_pose": [],
        "action": [],
        "joint_pos": [],
        "joint_vel": [],
    }


def save_demo(
    output_dir: Path,
    demo_index: int,
    records: dict[str, list],
    control_dt: float,
    final_error: float,
) -> Path:
    """Save one successful demonstration."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"demo_{demo_index:03d}.npz"

    np.savez_compressed(
        output_path,
        time=np.arange(
            len(records["state"]),
            dtype=np.float32,
        )
        * control_dt,
        state=np.asarray(records["state"], dtype=np.int64),
        ee_pose=np.stack(records["ee_pose"]).astype(np.float32),
        object_pose=np.stack(records["object_pose"]).astype(np.float32),
        target_pose=np.stack(records["target_pose"]).astype(np.float32),
        action=np.stack(records["action"]).astype(np.float32),
        joint_pos=np.stack(records["joint_pos"]).astype(np.float32),
        joint_vel=np.stack(records["joint_vel"]).astype(np.float32),
        control_dt=np.asarray(control_dt, dtype=np.float32),
        final_error=np.asarray(final_error, dtype=np.float32),
        quaternion_order=np.asarray("wxyz"),
        coordinate_frame=np.asarray("local_environment"),
    )

    return output_path


def make_hold_action(env: gym.Env) -> torch.Tensor:
    """Hold the current EE pose with the gripper open."""

    ee_pose, _, _ = get_scene_poses(env)

    gripper = torch.full(
        (env.unwrapped.num_envs, 1),
        GRIPPER_OPEN,
        device=env.unwrapped.device,
    )

    return torch.cat([ee_pose, gripper], dim=-1)


def force_same_step_autoreset(env: gym.Env) -> None:
    """Trigger Isaac Lab's supported Same-Step autoreset path.

    ManagerBasedRLEnv should only be explicitly reset once. To start a new
    episode, mark the current episode as one step before timeout and call
    env.step(). The step method detects the timeout and resets the environment
    internally.
    """

    unwrapped = env.unwrapped

    # step() increments this buffer before computing the timeout term.
    unwrapped.episode_length_buf[:] = (
        unwrapped.max_episode_length - 1
    )

    hold_action = make_hold_action(env)

    _, _, terminated, truncated, _ = env.step(hold_action)
    done = terminated | truncated

    if not bool(done.all().item()):
        raise RuntimeError(
            "Failed to trigger Same-Step autoreset. "
            f"terminated={terminated}, truncated={truncated}"
        )

    # The returned state already belongs to the newly reset episode.
    print(
        "[collect] Same-Step autoreset completed",
        flush=True,
    )


def main() -> None:
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.seed = args_cli.seed

    # Debug markers are unnecessary during headless collection.
    if args_cli.headless:
        env_cfg.commands.object_pose.debug_vis = False

    env = gym.make(TASK_ID, cfg=env_cfg)
    control_dt = env_cfg.sim.dt * env_cfg.decimation

    # IMPORTANT: ManagerBasedRLEnv is explicitly reset only once.
    env.reset(seed=args_cli.seed)

    saved_demos = 0
    attempts = 0
    manifest: list[dict] = []

    while (
        saved_demos < args_cli.num_demos
        and attempts < args_cli.max_attempts
        and simulation_app.is_running()
    ):
        attempts += 1

        print(
            f"\n[collect] attempt={attempts}, "
            f"saved={saved_demos}/{args_cli.num_demos}",
            flush=True,
        )

        expert = ScriptedPickPlace(
            dt=control_dt,
            device=env.unwrapped.device,
        )

        records = new_records()
        sequence_complete = False
        environment_done = False

        for _ in range(args_cli.max_steps):
            with torch.inference_mode():
                ee_pose, object_pose, target_pose = get_scene_poses(env)
                state_before_action = int(expert.state)

                action, request_reset = expert.compute(
                    ee_pose,
                    object_pose,
                    target_pose,
                )

                robot = env.unwrapped.scene["robot"]

                records["state"].append(state_before_action)
                records["ee_pose"].append(tensor_row(ee_pose))
                records["object_pose"].append(tensor_row(object_pose))
                records["target_pose"].append(tensor_row(target_pose))
                records["action"].append(tensor_row(action))
                records["joint_pos"].append(
                    tensor_row(robot.data.joint_pos)
                )
                records["joint_vel"].append(
                    tensor_row(robot.data.joint_vel)
                )

                _, _, terminated, truncated, _ = env.step(action)

                if request_reset:
                    sequence_complete = True
                    break

                if bool((terminated | truncated).any().item()):
                    # step() has already reset the environment here.
                    environment_done = True
                    break

        if sequence_complete:
            final_error = object_target_error(env)
        else:
            final_error = float("inf")

        success = (
            sequence_complete
            and not environment_done
            and final_error < args_cli.success_threshold
        )

        print(
            f"[collect] complete={sequence_complete}, "
            f"environment_done={environment_done}, "
            f"final_error={final_error:.4f} m, "
            f"success={success}",
            flush=True,
        )

        if success:
            output_path = save_demo(
                output_dir=output_dir,
                demo_index=saved_demos,
                records=records,
                control_dt=control_dt,
                final_error=final_error,
            )

            manifest.append(
                {
                    "file": output_path.name,
                    "steps": len(records["state"]),
                    "control_dt": control_dt,
                    "final_error": final_error,
                    "attempt": attempts,
                }
            )

            saved_demos += 1
            print(f"[collect] saved: {output_path}", flush=True)

        # Do not reset after the last requested attempt.  Resetting an Isaac
        # Lab environment after the scripted episode currently blocks inside
        # env.step(), while closing this one-demo worker is safe.  The
        # collection controller starts a fresh worker for each attempt.
        if (
            saved_demos >= args_cli.num_demos
            or attempts >= args_cli.max_attempts
            or not simulation_app.is_running()
        ):
            break

        # If env.step() terminated the episode, it already auto-reset.
        # Otherwise, force a timeout so step() performs the supported reset.
        if not environment_done:
            force_same_step_autoreset(env)

    env.close()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "num_demos": saved_demos,
                "attempts": attempts,
                "seed": args_cli.seed,
                "quaternion_order": "wxyz",
                "coordinate_frame": "local_environment",
                "demos": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[collect] manifest: {manifest_path}")
    print(f"[collect] saved demos: {saved_demos}")

    if saved_demos < args_cli.num_demos:
        raise RuntimeError(
            f"Saved only {saved_demos}/{args_cli.num_demos} "
            f"demos after {attempts} attempts."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
