"""Run the essay2608 scripted pick-and-place expert."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import essay2608
from essay2608.expert import ScriptedPickPlace, get_scene_poses, object_target_error
from isaaclab_tasks.utils import parse_env_cfg


TASK_ID = "Essay2608-Dynamic-Pick-Place-v0"


def main() -> None:
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset()

    if env.unwrapped.action_space.shape[-1] != 8:
        raise RuntimeError(
            f"Expected 8-dimensional IK action, got {env.unwrapped.action_space.shape}."
        )

    expert = ScriptedPickPlace(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        device=env.unwrapped.device,
    )

    completed_episodes = 0
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            ee_pose, object_pose, target_pose = get_scene_poses(env)
            action, request_reset = expert.compute(
                ee_pose,
                object_pose,
                target_pose,
            )

            _, _, terminated, truncated, _ = env.step(action)

            if step % 100 == 0:
                print(
                    f"[step {step}] state={expert.state.name} "
                    f"object={object_pose[0, :3].cpu().tolist()}"
                )

            if request_reset:
                print(
                    f"[episode] final object-target error: "
                    f"{object_target_error(env):.4f} m"
                )

                completed_episodes += 1

                if args_cli.headless and completed_episodes >= args_cli.episodes:
                    break

                env.reset()
                expert.reset()

            elif bool((terminated | truncated).any().item()):
                print("[episode] ended early; resetting.")
                env.reset()
                expert.reset()

            step += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
