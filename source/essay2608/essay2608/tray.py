"""Scripted bilateral expert and shared-tray pose helpers."""

from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
import torch

from essay2608.bimanual import GRIPPER_CLOSE, GRIPPER_OPEN, _asset_body_pose


TRAY_TARGET_POSE = torch.tensor([0.60, 0.0, 0.34, 1.0, 0.0, 0.0, 0.0])


class TrayState(IntEnum):
    REST = 0
    APPROACH = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT = 4
    LOWER = 5
    RELEASE = 6
    RETREAT = 7
    COMPLETE = 8


def get_tray_poses(env: gym.Env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left = _asset_body_pose(env, "left_robot", "panda_hand")
    right = _asset_body_pose(env, "right_robot", "panda_hand")
    origins = env.unwrapped.scene.env_origins
    asset = env.unwrapped.scene["object"]
    tray = torch.cat((asset.data.root_pos_w - origins, asset.data.root_quat_w), dim=-1)
    target = TRAY_TARGET_POSE.to(device=left.device, dtype=left.dtype).repeat(left.shape[0], 1)
    return left, right, tray, target


def write_bilateral_tray(
    env: gym.Env,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    midpoint_offset: torch.Tensor,
) -> None:
    """Keep the tray center attached to the midpoint of both grippers."""

    tray = env.unwrapped.scene["object"]
    midpoint = 0.5 * (left_pose[:, :3] + right_pose[:, :3])
    position = midpoint + midpoint_offset
    quaternion = torch.zeros((len(position), 4), device=position.device, dtype=position.dtype)
    quaternion[:, 0] = 1.0
    world_pose = torch.cat((position + env.unwrapped.scene.env_origins, quaternion), dim=-1)
    tray.write_root_pose_to_sim(world_pose)
    tray.write_root_velocity_to_sim(torch.zeros((len(position), 6), device=position.device))


class ScriptedLiftTray:
    """Symmetric expert that maintains simultaneous bilateral connection."""

    def __init__(self, dt: float, device: str, threshold: float = 0.04) -> None:
        self.dt = float(dt)
        self.device = device
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self.state = TrayState.REST
        self.state_time = 0.0
        self.left_orientation = None
        self.right_orientation = None
        self.left_grasp_pose = None
        self.right_grasp_pose = None

    @property
    def connected(self) -> bool:
        return TrayState.GRASP <= self.state <= TrayState.RELEASE

    def _ready(self, left: torch.Tensor, right: torch.Tensor, left_desired: torch.Tensor, right_desired: torch.Tensor) -> bool:
        reached = (
            torch.linalg.norm(left[:, :3] - left_desired[:, :3], dim=-1) < self.threshold
        ) & (torch.linalg.norm(right[:, :3] - right_desired[:, :3], dim=-1) < self.threshold)
        return bool(reached.all()) or self.state_time >= 2.5

    def _transition(self, state: TrayState) -> None:
        print(f"[tray] {self.state.name} -> {state.name}", flush=True)
        self.state = state
        self.state_time = 0.0

    def compute(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        tray: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if self.left_orientation is None:
            self.left_orientation = left[:, 3:7].clone()
            self.right_orientation = right[:, 3:7].clone()
        left_desired = left.clone()
        right_desired = right.clone()
        left_gripper = right_gripper = GRIPPER_OPEN
        done = False
        left_edge = tray.clone()
        right_edge = tray.clone()
        left_edge[:, 1] += 0.145
        right_edge[:, 1] -= 0.145
        left_edge[:, 2] += 0.035
        right_edge[:, 2] += 0.035

        if self.left_grasp_pose is None:
            left_lift = left_edge.clone()
            right_lift = right_edge.clone()
        else:
            left_lift = self.left_grasp_pose.clone()
            right_lift = self.right_grasp_pose.clone()
            left_lift[:, 2] += 0.20
            right_lift[:, 2] += 0.20
        left_transport = left_lift.clone()
        right_transport = right_lift.clone()
        left_transport[:, 0] += 0.12
        right_transport[:, 0] += 0.12
        left_lower = left_transport.clone()
        right_lower = right_transport.clone()
        left_lower[:, 2] -= 0.08
        right_lower[:, 2] -= 0.08

        if self.state == TrayState.REST:
            if self.state_time >= 0.4:
                self._transition(TrayState.APPROACH)
        elif self.state == TrayState.APPROACH:
            left_desired, right_desired = left_edge, right_edge
            if self._ready(left, right, left_desired, right_desired):
                self.left_grasp_pose = left.clone()
                self.right_grasp_pose = right.clone()
                self._transition(TrayState.GRASP)
        elif self.state == TrayState.GRASP:
            left_desired, right_desired = left_edge, right_edge
            left_gripper = right_gripper = GRIPPER_CLOSE
            if self.state_time >= 0.5:
                self._transition(TrayState.LIFT)
        elif self.state == TrayState.LIFT:
            left_desired, right_desired = left_lift, right_lift
            left_gripper = right_gripper = GRIPPER_CLOSE
            if self._ready(left, right, left_desired, right_desired):
                self._transition(TrayState.TRANSPORT)
        elif self.state == TrayState.TRANSPORT:
            left_desired, right_desired = left_transport, right_transport
            left_gripper = right_gripper = GRIPPER_CLOSE
            if self._ready(left, right, left_desired, right_desired):
                self._transition(TrayState.LOWER)
        elif self.state == TrayState.LOWER:
            left_desired, right_desired = left_lower, right_lower
            left_gripper = right_gripper = GRIPPER_CLOSE
            if self._ready(left, right, left_desired, right_desired):
                self._transition(TrayState.RELEASE)
        elif self.state == TrayState.RELEASE:
            left_desired, right_desired = left_lower, right_lower
            if self.state_time >= 0.5:
                self._transition(TrayState.RETREAT)
        elif self.state == TrayState.RETREAT:
            left_desired, right_desired = left_lower.clone(), right_lower.clone()
            left_desired[:, 1] += 0.12
            right_desired[:, 1] -= 0.12
            left_desired[:, 2] += 0.08
            right_desired[:, 2] += 0.08
            if self._ready(left, right, left_desired, right_desired):
                self._transition(TrayState.COMPLETE)
        else:
            done = self.state_time >= 0.5

        left_desired[:, 3:7] = self.left_orientation
        right_desired[:, 3:7] = self.right_orientation
        left_action = torch.cat(
            (left_desired, torch.full((len(left), 1), left_gripper, device=self.device)), dim=-1
        )
        right_action = torch.cat(
            (right_desired, torch.full((len(right), 1), right_gripper, device=self.device)), dim=-1
        )
        self.state_time += self.dt
        return torch.cat((left_action, right_action), dim=-1), done
