"""Shared bimanual scripted experts and simulator pose accessors."""

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.utils import math as math_utils

from essay2608.data.handover_schema import HandoverState, handover_relation_label


GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
HANDOVER_POSE = torch.tensor([0.48, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0])
HANDOVER_TARGET_POSE = torch.tensor([0.48, -0.20, 0.22, 1.0, 0.0, 0.0, 0.0])


def _asset_body_pose(env: gym.Env, asset_name: str, body_name: str) -> torch.Tensor:
    robot = env.unwrapped.scene[asset_name]
    body_ids, _ = robot.find_bodies(body_name)
    body_id = body_ids[0]
    origins = env.unwrapped.scene.env_origins
    offset = torch.tensor([0.0, 0.0, 0.107], device=robot.device).repeat(robot.data.body_pos_w.shape[0], 1)
    position = robot.data.body_pos_w[:, body_id] + math_utils.quat_apply(robot.data.body_quat_w[:, body_id], offset)
    position -= origins
    orientation = robot.data.body_quat_w[:, body_id]
    return torch.cat((position, orientation), dim=-1)


def get_handover_poses(env: gym.Env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return left EE, right EE, object, and target poses in environment coordinates."""

    left = _asset_body_pose(env, "left_robot", "panda_hand")
    right = _asset_body_pose(env, "right_robot", "panda_hand")
    origins = env.unwrapped.scene.env_origins
    object_asset = env.unwrapped.scene["object"]
    object_pose = torch.cat((object_asset.data.root_pos_w - origins, object_asset.data.root_quat_w), dim=-1)
    target = HANDOVER_TARGET_POSE.to(device=left.device, dtype=left.dtype).repeat(left.shape[0], 1)
    return left, right, object_pose, target


def actions_to_robot_root_frames(env: gym.Env, actions: torch.Tensor) -> torch.Tensor:
    """Convert two environment-frame pose commands to each Franka root frame."""

    converted = actions.clone()
    origins = env.unwrapped.scene.env_origins
    for start, asset_name in ((0, "left_robot"), (8, "right_robot")):
        robot = env.unwrapped.scene[asset_name]
        desired_position_w = actions[:, start : start + 3] + origins
        desired_quaternion_w = actions[:, start + 3 : start + 7]
        position_b, quaternion_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            desired_position_w,
            desired_quaternion_w,
        )
        converted[:, start : start + 3] = position_b
        converted[:, start + 3 : start + 7] = quaternion_b
    return converted


def write_attached_object(
    env: gym.Env,
    carrier_pose: torch.Tensor,
    object_offset: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Write the lightweight object pose to the current carrier EE."""

    object_asset = env.unwrapped.scene["object"]
    position, quaternion = math_utils.combine_frame_transforms(
        carrier_pose[:, :3],
        carrier_pose[:, 3:7],
        object_offset[0],
        object_offset[1],
    )
    world_pose = torch.cat((position, quaternion), dim=-1)
    world_pose[:, :3] += env.unwrapped.scene.env_origins
    object_asset.write_root_pose_to_sim(world_pose)
    object_asset.write_root_velocity_to_sim(torch.zeros((carrier_pose.shape[0], 6), device=carrier_pose.device))


class ScriptedHandover:
    """Symmetric Cartesian handover expert with explicit carrier state."""

    def __init__(self, dt: float, device: str, position_threshold: float = 0.035) -> None:
        self.dt = float(dt)
        self.device = device
        self.position_threshold = float(position_threshold)
        self.handover_pose = HANDOVER_POSE.to(device=device).unsqueeze(0)
        self.target_pose = HANDOVER_TARGET_POSE.to(device=device).unsqueeze(0)
        self.reset()

    def reset(self) -> None:
        self.state = HandoverState.REST
        self.state_time = 0.0
        self.left_orientation: torch.Tensor | None = None
        self.right_orientation: torch.Tensor | None = None
        self.lift_pose: torch.Tensor | None = None
        self.left_retreat_pose: torch.Tensor | None = None
        self.left_object_offset: tuple[torch.Tensor, torch.Tensor] | None = None
        self.right_object_offset: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def carrier(self) -> str | None:
        if HandoverState.LEFT_GRASP <= self.state <= HandoverState.RIGHT_GRASP:
            return "left"
        if HandoverState.TRANSFER <= self.state <= HandoverState.RIGHT_RELEASE:
            return "right"
        return None

    @property
    def relation_label(self) -> str:
        """Script supervision label, independent of the kinematic carrier."""

        return handover_relation_label(self.state)

    @property
    def carrier_offset(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.carrier == "left":
            return self.left_object_offset
        if self.carrier == "right":
            return self.right_object_offset
        return None

    def _transition(self, state: HandoverState) -> None:
        print(f"[handover] {self.state.name} -> {state.name}", flush=True)
        self.state = state
        self.state_time = 0.0

    def _reached(self, current: torch.Tensor, desired: torch.Tensor) -> bool:
        return bool((torch.linalg.norm(current[:, :3] - desired[:, :3], dim=-1) < self.position_threshold).all())

    def _ready(self, current: torch.Tensor, desired: torch.Tensor, timeout: float = 2.0) -> bool:
        return self._reached(current, desired) or self.state_time >= timeout

    def compute(
        self,
        left_pose: torch.Tensor,
        right_pose: torch.Tensor,
        object_pose: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Return concatenated left/right pose-gripper commands."""

        if self.left_orientation is None:
            self.left_orientation = left_pose[:, 3:7].clone()
            self.right_orientation = right_pose[:, 3:7].clone()

        left_desired = left_pose.clone()
        right_desired = right_pose.clone()
        left_gripper = GRIPPER_OPEN
        right_gripper = GRIPPER_OPEN
        done = False

        left_object = object_pose.clone()
        left_object[:, 2] += 0.015
        left_lift = self.lift_pose if self.lift_pose is not None else left_object
        left_handover = self.handover_pose.repeat(left_pose.shape[0], 1)
        # Rendezvous in the observed object frame instead of assuming that the
        # left arm exactly reached the nominal handover point.  This is the
        # cross-arm coupling the later policy baselines must reproduce.
        right_handover = object_pose.clone()
        right_handover[:, 1] -= 0.015
        right_pre_handover = right_handover.clone()
        right_pre_handover[:, 1] -= 0.10
        right_target = target_pose.clone()
        if self.right_object_offset is not None:
            offset_position, _ = self.right_object_offset
            # The Cartesian controller intentionally keeps the demonstrated hand
            # orientation fixed.  Solve only the translational attachment equation
            # p_object = p_hand + R_hand p_offset in that same orientation.
            target_position = target_pose[:, :3] - math_utils.quat_apply(
                self.right_orientation, offset_position
            )
            right_target = torch.cat((target_position, self.right_orientation), dim=-1)

        if self.state == HandoverState.REST:
            if self.state_time >= 0.4:
                self._transition(HandoverState.LEFT_APPROACH)
        elif self.state == HandoverState.LEFT_APPROACH:
            left_desired = left_object
            if self._ready(left_pose, left_desired):
                self.left_object_offset = math_utils.subtract_frame_transforms(
                    left_pose[:, :3],
                    left_pose[:, 3:7],
                    object_pose[:, :3],
                    object_pose[:, 3:7],
                )
                self._transition(HandoverState.LEFT_GRASP)
        elif self.state == HandoverState.LEFT_GRASP:
            left_desired = left_object
            left_gripper = GRIPPER_CLOSE
            if self.state_time >= 0.5:
                self.lift_pose = left_pose.clone()
                self.lift_pose[:, 2] += 0.12
                self._transition(HandoverState.LEFT_LIFT)
        elif self.state == HandoverState.LEFT_LIFT:
            left_desired = left_lift
            left_gripper = GRIPPER_CLOSE
            if self._ready(left_pose, left_desired):
                self._transition(HandoverState.LEFT_TO_HANDOVER)
        elif self.state == HandoverState.LEFT_TO_HANDOVER:
            left_desired = left_handover
            left_gripper = GRIPPER_CLOSE
            if self._ready(left_pose, left_desired):
                self._transition(HandoverState.RIGHT_APPROACH)
        elif self.state == HandoverState.RIGHT_APPROACH:
            left_desired = left_handover
            left_gripper = GRIPPER_CLOSE
            right_desired = right_pre_handover
            if self._ready(right_pose, right_desired):
                self._transition(HandoverState.RIGHT_GRASP)
        elif self.state == HandoverState.RIGHT_GRASP:
            left_desired = left_handover
            left_gripper = GRIPPER_CLOSE
            right_desired = right_handover
            right_gripper = GRIPPER_CLOSE
            if self._ready(right_pose, right_desired) and self.state_time >= 0.4:
                # Preserve the world-space object pose at the carrier switch.  The
                # resulting right-hand offset is also inverted above when forming
                # ``right_target``, so continuity does not trade away placement
                # accuracy.
                self.right_object_offset = math_utils.subtract_frame_transforms(
                    right_pose[:, :3],
                    right_pose[:, 3:7],
                    object_pose[:, :3],
                    object_pose[:, 3:7],
                )
                self._transition(HandoverState.TRANSFER)
        elif self.state == HandoverState.TRANSFER:
            left_desired = left_handover
            left_gripper = GRIPPER_CLOSE
            right_desired = right_handover
            right_gripper = GRIPPER_CLOSE
            if self.state_time >= 0.3:
                self._transition(HandoverState.LEFT_RELEASE)
        elif self.state == HandoverState.LEFT_RELEASE:
            left_desired = left_handover
            right_desired = right_handover
            right_gripper = GRIPPER_CLOSE
            if self.state_time >= 0.4:
                self.left_retreat_pose = left_pose.clone()
                self.left_retreat_pose[:, 1] += 0.12
                self._transition(HandoverState.RIGHT_TO_TARGET)
        elif self.state == HandoverState.RIGHT_TO_TARGET:
            left_desired = self.left_retreat_pose
            right_desired = right_target
            right_gripper = GRIPPER_CLOSE
            if self._ready(right_pose, right_desired):
                self._transition(HandoverState.RIGHT_RELEASE)
        elif self.state == HandoverState.RIGHT_RELEASE:
            right_desired = right_target
            if self.state_time >= 0.4:
                self._transition(HandoverState.RETREAT)
        elif self.state == HandoverState.RETREAT:
            left_desired = self.left_retreat_pose
            right_desired = right_target
            right_desired[:, 2] += 0.10
            if self._ready(right_pose, right_desired):
                self._transition(HandoverState.COMPLETE)
        elif self.state == HandoverState.COMPLETE:
            done = self.state_time >= 0.5

        left_desired[:, 3:7] = self.left_orientation
        right_desired[:, 3:7] = self.right_orientation
        left_action = torch.cat(
            (left_desired, torch.full((left_pose.shape[0], 1), left_gripper, device=self.device)), dim=-1
        )
        right_action = torch.cat(
            (right_desired, torch.full((right_pose.shape[0], 1), right_gripper, device=self.device)), dim=-1
        )
        self.state_time += self.dt
        return torch.cat((left_action, right_action), dim=-1), done
