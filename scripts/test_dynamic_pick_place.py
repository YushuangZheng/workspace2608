"""Smoke test for Essay2608 dynamic pick-and-place."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import essay2608
from isaaclab_tasks.utils import parse_env_cfg


TASK_ID = "Essay2608-Dynamic-Pick-Place-v0"


def main() -> None:
    if TASK_ID not in gym.registry:
        raise RuntimeError(f"Task is not registered: {TASK_ID}")

    env_cfg = parse_env_cfg(
        TASK_ID,
        device="cuda:0",
        num_envs=1,
    )

    env = gym.make(TASK_ID, cfg=env_cfg)
    obs, _ = env.reset()

    print(f"Task registered: {TASK_ID}")
    print("Environment created successfully")
    print("Action space:", env.action_space)

    if isinstance(obs, dict):
        print("Observation groups:", list(obs.keys()))
        if "policy" in obs and isinstance(obs["policy"], dict):
            print("Policy observations:", list(obs["policy"].keys()))

    actions = torch.zeros(
        env.action_space.shape,
        device=env.unwrapped.device,
    )

    # 无头模式运行100步后自动退出；
    # GUI模式持续运行，直到关闭窗口或按Ctrl+C。
    max_steps = 100 if args_cli.headless else None
    step = 0

    while simulation_app.is_running():
        if max_steps is not None and step >= max_steps:
            break

        obs, reward, terminated, truncated, info = env.step(actions)

        if step == 5:
            object_asset = env.unwrapped.scene["object"]
            ee_sensor = env.unwrapped.scene["ee_frame"]
            target_command = env.unwrapped.command_manager.get_command(
                "object_pose"
            )

            print(
                "Object position:",
                object_asset.data.root_pos_w[0].cpu().tolist(),
            )
            print(
                "End-effector position:",
                ee_sensor.data.target_pos_w[0, 0].cpu().tolist(),
            )
            print(
                "Target command:",
                target_command[0].cpu().tolist(),
            )

        if step % 100 == 0:
            print(f"Simulation step: {step}")

        step += 1

    print(f"Simulation finished after {step} steps")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
