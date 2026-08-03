"""Scripted pick-and-place expert for essay2608."""

import argparse
from enum import IntEnum

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run a scripted Franka pick-and-place expert."
)
parser.add_argument(
    "--episodes",
    type=int,
    default=1,
    help="Number of successful episodes in headless mode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import essay2608
from isaaclab_tasks.utils import parse_env_cfg


TASK_ID = "Essay2608-Dynamic-Pick-Place-v0"

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


class PickPlaceState(IntEnum):
    REST = 0
    APPROACH_ABOVE_OBJECT = 1
    APPROACH_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4
    MOVE_ABOVE_TARGET = 5
    LOWER_TO_TARGET = 6
    RELEASE_OBJECT = 7
    RETREAT = 8
    COMPLETE = 9


class ScriptedPickPlace:
    """Simple Cartesian-space finite-state machine."""

    def __init__(
        self,
        dt: float,
        device: str,
        position_threshold: float = 0.02,
    ) -> None:
        self.dt = float(dt)
        self.device = device
        self.position_threshold = float(position_threshold)

        # Downward-facing Franka gripper quaternion in wxyz order.
        self.down_quat = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]],
            device=device,
        )

        self.lift_position = torch.zeros((1, 3), device=device)
        self.reset()

    def reset(self) -> None:
        self.state = PickPlaceState.REST
        self.state_time = 0.0
        self.lift_position.zero_()
        print("\n[state] REST")

    def _transition(self, new_state: PickPlaceState) -> None:
        print(f"[state] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_time = 0.0

    def _reached(
        self,
        current_position: torch.Tensor,
        desired_position: torch.Tensor,
    ) -> bool:
        error = torch.linalg.norm(
            current_position - desired_position,
            dim=-1,
        )
        return bool((error < self.position_threshold).all().item())

    def compute(
        self,
        ee_pose: torch.Tensor,
        object_pose: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Return absolute EE pose plus binary gripper action."""

        ee_position = ee_pose[:, :3]
        object_position = object_pose[:, :3]
        target_position = target_pose[:, :3]

        desired_pose = ee_pose.clone()
        gripper = GRIPPER_OPEN
        request_reset = False

        if self.state == PickPlaceState.REST:
            # Initially hold the current pose.
            if self.state_time >= 0.5:
                self._transition(
                    PickPlaceState.APPROACH_ABOVE_OBJECT
                )

        elif self.state == PickPlaceState.APPROACH_ABOVE_OBJECT:
            desired_pose[:, :3] = object_position
            desired_pose[:, 2] += 0.12
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if (
                self._reached(ee_position, desired_pose[:, :3])
                and self.state_time >= 0.3
            ):
                self._transition(PickPlaceState.APPROACH_OBJECT)

        elif self.state == PickPlaceState.APPROACH_OBJECT:
            desired_pose[:, :3] = object_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if (
                self._reached(ee_position, desired_pose[:, :3])
                and self.state_time >= 0.3
            ):
                self._transition(PickPlaceState.GRASP_OBJECT)

        elif self.state == PickPlaceState.GRASP_OBJECT:
            desired_pose[:, :3] = object_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_CLOSE

            if self.state_time >= 0.8:
                self.lift_position[:] = object_position
                self.lift_position[:, 2] += 0.15
                self._transition(PickPlaceState.LIFT_OBJECT)

        elif self.state == PickPlaceState.LIFT_OBJECT:
            desired_pose[:, :3] = self.lift_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_CLOSE

            if (
                self._reached(ee_position, desired_pose[:, :3])
                and self.state_time >= 0.3
            ):
                self._transition(
                    PickPlaceState.MOVE_ABOVE_TARGET
                )

        elif self.state == PickPlaceState.MOVE_ABOVE_TARGET:
            desired_pose[:, :3] = target_position
            desired_pose[:, 2] += 0.15
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_CLOSE

            if (
                self._reached(ee_position, desired_pose[:, :3])
                and self.state_time >= 0.3
            ):
                self._transition(PickPlaceState.LOWER_TO_TARGET)

        elif self.state == PickPlaceState.LOWER_TO_TARGET:
            desired_pose[:, :3] = target_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_CLOSE

            if (
                self._reached(ee_position, desired_pose[:, :3])
                and self.state_time >= 0.3
            ):
                self._transition(PickPlaceState.RELEASE_OBJECT)

        elif self.state == PickPlaceState.RELEASE_OBJECT:
            desired_pose[:, :3] = target_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if self.state_time >= 0.8:
                self._transition(PickPlaceState.RETREAT)

        elif self.state == PickPlaceState.RETREAT:
            desired_pose[:, :3] = target_position
            desired_pose[:, 2] += 0.15
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if (
                self._reached(ee_position, desired_pose[:, :3])
                and self.state_time >= 0.3
            ):
                self._transition(PickPlaceState.COMPLETE)

        elif self.state == PickPlaceState.COMPLETE:
            desired_pose[:, :3] = target_position
            desired_pose[:, 2] += 0.15
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if self.state_time >= 1.0:
                request_reset = True

        self.state_time += self.dt

        gripper_action = torch.full(
            (ee_pose.shape[0], 1),
            gripper,
            device=self.device,
        )

        action = torch.cat(
            [desired_pose, gripper_action],
            dim=-1,
        )
        return action, request_reset


def get_scene_poses(
    env: gym.Env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read EE, object and target poses in the local environment frame."""

    unwrapped = env.unwrapped
    env_origins = unwrapped.scene.env_origins

    ee_sensor = unwrapped.scene["ee_frame"]
    ee_position = (
        ee_sensor.data.target_pos_w[:, 0, :]
        - env_origins
    )
    ee_orientation = ee_sensor.data.target_quat_w[:, 0, :]

    object_asset = unwrapped.scene["object"]
    object_position = object_asset.data.root_pos_w - env_origins
    object_orientation = object_asset.data.root_quat_w

    target_command = unwrapped.command_manager.get_command(
        "object_pose"
    )
    target_position = target_command[:, :3]
    target_orientation = target_command[:, 3:7]

    ee_pose = torch.cat(
        [ee_position, ee_orientation],
        dim=-1,
    )
    object_pose = torch.cat(
        [object_position, object_orientation],
        dim=-1,
    )
    target_pose = torch.cat(
        [target_position, target_orientation],
        dim=-1,
    )

    return ee_pose, object_pose, target_pose


def make_hold_action(env: gym.Env) -> torch.Tensor:
    """Create an initial action that holds the current EE pose."""

    ee_pose, _, _ = get_scene_poses(env)

    gripper = torch.full(
        (env.unwrapped.num_envs, 1),
        GRIPPER_OPEN,
        device=env.unwrapped.device,
    )
    return torch.cat([ee_pose, gripper], dim=-1)


def report_episode(env: gym.Env) -> None:
    """Print final object-to-target error."""

    _, object_pose, target_pose = get_scene_poses(env)

    position_error = torch.linalg.norm(
        object_pose[:, :3] - target_pose[:, :3],
        dim=-1,
    )

    print(
        "[episode] object-target error:",
        f"{position_error[0].item():.4f} m",
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset()

    action_shape = env.unwrapped.action_space.shape
    print("Action shape:", action_shape)

    if action_shape[-1] != 8:
        raise RuntimeError(
            "Expected 8 actions: "
            "position(3) + quaternion(4) + gripper(1), "
            f"but received {action_shape}."
        )

    control_dt = env_cfg.sim.dt * env_cfg.decimation

    state_machine = ScriptedPickPlace(
        dt=control_dt,
        device=env.unwrapped.device,
    )

    actions = make_hold_action(env)
    successful_episodes = 0
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            (
                _,
                _,
                terminated,
                truncated,
                _,
            ) = env.step(actions)

            ee_pose, object_pose, target_pose = get_scene_poses(env)

            actions, request_reset = state_machine.compute(
                ee_pose,
                object_pose,
                target_pose,
            )

            if step % 100 == 0:
                print(
                    f"[step {step}] "
                    f"state={state_machine.state.name} "
                    f"ee={ee_pose[0, :3].cpu().tolist()} "
                    f"object={object_pose[0, :3].cpu().tolist()}"
                )

            environment_done = bool(
                (terminated | truncated).any().item()
            )

            if request_reset:
                report_episode(env)
                successful_episodes += 1
                print(
                    "[episode] scripted sequence completed:",
                    successful_episodes,
                )

                if (
                    args_cli.headless
                    and successful_episodes >= args_cli.episodes
                ):
                    break

                env.reset()
                state_machine.reset()
                actions = make_hold_action(env)

            elif environment_done:
                print(
                    "[episode] environment ended before "
                    "the scripted sequence completed; resetting."
                )
                env.reset()
                state_machine.reset()
                actions = make_hold_action(env)

            step += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
