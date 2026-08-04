"""Scripted pick-and-place expert shared by collection and evaluation scripts."""

from enum import IntEnum

import gymnasium as gym
import torch


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
    """Cartesian-space finite-state-machine expert."""

    def __init__(
        self,
        dt: float,
        device: str,
        position_threshold: float = 0.02,
    ) -> None:
        self.dt = float(dt)
        self.device = device
        self.position_threshold = float(position_threshold)

        # Quaternion order: w, x, y, z.
        # This makes the Franka gripper point downward.
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
        """Return an absolute EE pose and a binary gripper command."""

        ee_position = ee_pose[:, :3]
        object_position = object_pose[:, :3]
        target_position = target_pose[:, :3]

        desired_pose = ee_pose.clone()
        gripper = GRIPPER_OPEN
        request_reset = False

        if self.state == PickPlaceState.REST:
            if self.state_time >= 0.5:
                self._transition(PickPlaceState.APPROACH_ABOVE_OBJECT)

        elif self.state == PickPlaceState.APPROACH_ABOVE_OBJECT:
            desired_pose[:, :3] = object_position
            desired_pose[:, 2] += 0.12
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if self._reached(ee_position, desired_pose[:, :3]) and self.state_time >= 0.3:
                self._transition(PickPlaceState.APPROACH_OBJECT)

        elif self.state == PickPlaceState.APPROACH_OBJECT:
            desired_pose[:, :3] = object_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_OPEN

            if self._reached(ee_position, desired_pose[:, :3]) and self.state_time >= 0.3:
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

            if self._reached(ee_position, desired_pose[:, :3]) and self.state_time >= 0.3:
                self._transition(PickPlaceState.MOVE_ABOVE_TARGET)

        elif self.state == PickPlaceState.MOVE_ABOVE_TARGET:
            desired_pose[:, :3] = target_position
            desired_pose[:, 2] += 0.15
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_CLOSE

            if self._reached(ee_position, desired_pose[:, :3]) and self.state_time >= 0.3:
                self._transition(PickPlaceState.LOWER_TO_TARGET)

        elif self.state == PickPlaceState.LOWER_TO_TARGET:
            desired_pose[:, :3] = target_position
            desired_pose[:, 3:7] = self.down_quat
            gripper = GRIPPER_CLOSE

            if self._reached(ee_position, desired_pose[:, :3]) and self.state_time >= 0.3:
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

            if self._reached(ee_position, desired_pose[:, :3]) and self.state_time >= 0.3:
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

        return torch.cat([desired_pose, gripper_action], dim=-1), request_reset


def get_scene_poses(
    env: gym.Env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read EE, object and target poses in the local environment frame."""

    unwrapped = env.unwrapped
    env_origins = unwrapped.scene.env_origins

    ee_sensor = unwrapped.scene["ee_frame"]
    ee_position = ee_sensor.data.target_pos_w[:, 0, :] - env_origins
    ee_orientation = ee_sensor.data.target_quat_w[:, 0, :]

    object_asset = unwrapped.scene["object"]
    object_position = object_asset.data.root_pos_w - env_origins
    object_orientation = object_asset.data.root_quat_w

    target_command = unwrapped.command_manager.get_command("object_pose")
    target_position = target_command[:, :3]
    target_orientation = target_command[:, 3:7]

    ee_pose = torch.cat([ee_position, ee_orientation], dim=-1)
    object_pose = torch.cat([object_position, object_orientation], dim=-1)
    target_pose = torch.cat([target_position, target_orientation], dim=-1)

    return ee_pose, object_pose, target_pose


def object_target_error(env: gym.Env) -> float:
    """Return object-to-target position error in metres."""

    _, object_pose, target_pose = get_scene_poses(env)
    error = torch.linalg.norm(
        object_pose[:, :3] - target_pose[:, :3],
        dim=-1,
    )
    return float(error[0].item())
