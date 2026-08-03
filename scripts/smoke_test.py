"""Smoke test for essay2608."""

from isaaclab.app import AppLauncher

# 必须先启动 Isaac Sim，再导入任务相关模块
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import essay2608
from isaaclab_tasks.utils import parse_env_cfg


TASK_ID = "Template-Essay2608-v0"


def main():
    assert TASK_ID in gym.registry, f"Task not registered: {TASK_ID}"
    print(f"Task registered: {TASK_ID}")

    env_cfg = parse_env_cfg(
        TASK_ID,
        device="cuda:0",
        num_envs=1,
    )

    env = gym.make(TASK_ID, cfg=env_cfg)
    obs, info = env.reset()

    print("Environment created successfully")
    print("Action space:", env.action_space)
    print("Observation type:", type(obs))

    actions = torch.zeros(
        env.action_space.shape,
        device=env.unwrapped.device,
    )

    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(actions)

        if step % 20 == 0:
            print(f"Simulation step: {step}")

    print("100 simulation steps completed successfully")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
