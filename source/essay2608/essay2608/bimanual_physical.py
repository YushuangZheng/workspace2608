"""Physical handover expert and phase-independent privileged relation truth."""

from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils

from essay2608.bimanual import GRIPPER_CLOSE, GRIPPER_OPEN
from essay2608.data.handover_schema import HandoverState


PHYSICAL_HANDOVER_OBJECT_POSITION = torch.tensor([0.50, 0.0, 0.32])
LEFT_GRASP_OFFSET = torch.tensor([0.0, 0.0, -0.002])
RIGHT_GRASP_OFFSET = torch.tensor([0.100, 0.0, -0.002])
# Keep command targets inside the well-conditioned workspace found by the
# measured-pose development traces.  Relation truth never uses these commands.
LEFT_TABLE_COMMAND_OFFSET = torch.tensor([0.0, 0.015, 0.0])
RIGHT_HANDOVER_COMMAND_OFFSET = torch.zeros(3)
RIGHT_GRASP_FINAL_COMMAND_OFFSET = torch.zeros(3)


class ScriptedPhysicalHandover:
    """Pose-feedback expert that never writes the object pose or relation label."""

    def __init__(self, dt: float, device: str, position_threshold_m: float = 0.050) -> None:
        self.dt = float(dt)
        self.device = device
        self.position_threshold_m = float(position_threshold_m)
        self.left_alignment_threshold_m = 0.015
        self.handover_object_position = PHYSICAL_HANDOVER_OBJECT_POSITION.to(device=device)
        self.left_grasp_offset = LEFT_GRASP_OFFSET.to(device=device)
        self.right_grasp_offset = RIGHT_GRASP_OFFSET.to(device=device)
        self.left_table_command_offset = LEFT_TABLE_COMMAND_OFFSET.to(device=device)
        self.right_handover_command_offset = RIGHT_HANDOVER_COMMAND_OFFSET.to(device=device)
        self.right_grasp_final_command_offset = RIGHT_GRASP_FINAL_COMMAND_OFFSET.to(
            device=device
        )
        self.reset()

    def reset(self) -> None:
        self.state = HandoverState.REST
        self.state_time = 0.0
        self.left_orientation: torch.Tensor | None = None
        self.right_orientation: torch.Tensor | None = None
        self.left_object_offset: tuple[torch.Tensor, torch.Tensor] | None = None
        self.right_object_offset: tuple[torch.Tensor, torch.Tensor] | None = None
        self.right_grasp_start_pose: torch.Tensor | None = None
        self.right_grasp_goal: torch.Tensor | None = None
        self.right_grasp_command: torch.Tensor | None = None
        self.left_grasp_start_pose: torch.Tensor | None = None
        self.left_grasp_goal: torch.Tensor | None = None
        self.left_grasp_command: torch.Tensor | None = None
        self.right_approach_staged = False
        self.left_hold_pose: torch.Tensor | None = None
        self.right_hold_pose: torch.Tensor | None = None
        self.right_transport_start_pose: torch.Tensor | None = None
        self.right_transport_goal: torch.Tensor | None = None
        self.left_lift_pose: torch.Tensor | None = None
        self.left_lift_start_pose: torch.Tensor | None = None
        self.left_retreat_pose: torch.Tensor | None = None
        self.right_retreat_pose: torch.Tensor | None = None
        self.grasp_close_time: float | None = None
        self.failed = False
        self.failure_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.state == HandoverState.COMPLETE and not self.failed

    def _transition(self, state: HandoverState) -> None:
        print(f"[physical-handover] {self.state.name} -> {state.name}", flush=True)
        self.state = state
        self.state_time = 0.0
        self.grasp_close_time = None

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.failure_reason = reason
        print(f"[physical-handover] FAILED: {reason}", flush=True)

    def _reached(
        self,
        current: torch.Tensor,
        desired: torch.Tensor,
        threshold_m: float | None = None,
    ) -> bool:
        error = torch.linalg.norm(current[:, :3] - desired[:, :3], dim=-1)
        threshold = self.position_threshold_m if threshold_m is None else threshold_m
        return bool((error < threshold).all())

    def _pose_at_object_site(
        self,
        object_pose: torch.Tensor,
        offset: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        position = object_pose[:, :3] + offset.unsqueeze(0)
        return torch.cat((position, orientation), dim=-1)

    def _pose_at_object_yaw_site(
        self,
        object_pose: torch.Tensor,
        offset: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate only the grasp-site offset with object yaw, not the wrist."""

        local_offset = offset.unsqueeze(0).expand(object_pose.shape[0], -1)
        position = object_pose[:, :3] + math_utils.quat_apply_yaw(
            object_pose[:, 3:7], local_offset
        )
        return torch.cat((position, orientation), dim=-1)

    def _hand_pose_for_object_position(
        self,
        object_position: torch.Tensor,
        orientation: torch.Tensor,
        object_offset: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        relative_position, _ = object_offset
        hand_position = object_position - math_utils.quat_apply(orientation, relative_position)
        return torch.cat((hand_position, orientation), dim=-1)

    def compute(
        self,
        left_pose: torch.Tensor,
        right_pose: torch.Tensor,
        object_pose: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Return two absolute pose/gripper actions from observable geometry only."""

        if self.left_orientation is None:
            self.left_orientation = left_pose[:, 3:7].clone()
            # The receiver base is mirrored by 180 degrees, which reverses but
            # does not change the world-space finger closing axis.  Holding the
            # measured initial orientation avoids waiting-stage motion.
            self.right_orientation = right_pose[:, 3:7].clone()

        left_desired = left_pose.clone()
        right_desired = right_pose.clone()
        left_gripper = GRIPPER_OPEN
        right_gripper = GRIPPER_OPEN

        left_site_goal = self._pose_at_object_site(
            object_pose, self.left_grasp_offset, self.left_orientation
        )
        left_site = left_site_goal.clone()
        left_site[:, :3] += self.left_table_command_offset.unsqueeze(0)
        left_pregrasp_goal = left_site_goal.clone()
        left_pregrasp_goal[:, 2] += 0.06
        left_pregrasp = left_pregrasp_goal.clone()
        left_pregrasp[:, :3] += self.left_table_command_offset.unsqueeze(0)
        right_site_goal = self._pose_at_object_yaw_site(
            object_pose, self.right_grasp_offset, self.right_orientation
        )
        right_site = right_site_goal.clone()
        right_site[:, :3] += self.right_handover_command_offset.unsqueeze(0)
        right_pregrasp_goal = right_site_goal.clone()
        right_pregrasp_goal[:, 2] += 0.08
        right_pregrasp = right_pregrasp_goal.clone()
        right_pregrasp[:, :3] += self.right_handover_command_offset.unsqueeze(0)

        if self.state == HandoverState.REST:
            if self.state_time >= 0.6:
                self._transition(HandoverState.LEFT_APPROACH)
        elif self.state == HandoverState.LEFT_APPROACH:
            left_desired = left_pregrasp
            # Finish horizontal alignment while there is still vertical
            # clearance.  The old 5 cm state threshold let the hand descend
            # with a large lateral error, so the palm could push the baton
            # before the fingers closed.
            if self._reached(
                left_pose,
                left_pregrasp,
                self.left_alignment_threshold_m,
            ):
                self.left_grasp_start_pose = left_pose.clone()
                self.left_grasp_goal = left_site_goal.clone()
                self.left_grasp_command = left_site.clone()
                self._transition(HandoverState.LEFT_GRASP)
            elif self.state_time >= 4.0:
                self._fail("left_pregrasp_timeout")
        elif self.state == HandoverState.LEFT_GRASP:
            approach_fraction = min(self.state_time / 1.0, 1.0)
            left_desired = self.left_grasp_start_pose + approach_fraction * (
                self.left_grasp_command - self.left_grasp_start_pose
            )
            if (
                self.grasp_close_time is None
                and approach_fraction >= 1.0
                and self._reached(
                    left_pose,
                    self.left_grasp_command,
                    self.left_alignment_threshold_m,
                )
            ):
                self.grasp_close_time = self.state_time
            if self.grasp_close_time is not None:
                left_gripper = GRIPPER_CLOSE
                if self.state_time - self.grasp_close_time >= 0.8:
                    self.left_object_offset = math_utils.subtract_frame_transforms(
                        left_pose[:, :3],
                        left_pose[:, 3:7],
                        object_pose[:, :3],
                        object_pose[:, 3:7],
                    )
                    self.left_lift_pose = left_pose.clone()
                    self.left_lift_start_pose = left_pose.clone()
                    self.left_lift_pose[:, 2] += 0.14
                    self._transition(HandoverState.LEFT_LIFT)
            elif self.state_time >= 4.0:
                self._fail("left_grasp_pose_timeout")
        elif self.state == HandoverState.LEFT_LIFT:
            lift_fraction = min(self.state_time / 1.5, 1.0)
            left_desired = self.left_lift_start_pose + lift_fraction * (
                self.left_lift_pose - self.left_lift_start_pose
            )
            left_gripper = GRIPPER_CLOSE
            if lift_fraction >= 1.0 and self._reached(left_pose, self.left_lift_pose):
                if float(object_pose[0, 2]) < 0.24:
                    self._fail("left_pick_failed")
                else:
                    self._transition(HandoverState.LEFT_TO_HANDOVER)
            elif self.state_time >= 4.0:
                self._fail("left_lift_timeout")
        elif self.state == HandoverState.LEFT_TO_HANDOVER:
            left_desired = self._hand_pose_for_object_position(
                self.handover_object_position.unsqueeze(0),
                self.left_orientation,
                self.left_object_offset,
            )
            left_gripper = GRIPPER_CLOSE
            if self._reached(left_pose, left_desired) or self.state_time >= 3.0:
                self._transition(HandoverState.RIGHT_APPROACH)
        elif self.state == HandoverState.RIGHT_APPROACH:
            left_desired = self._hand_pose_for_object_position(
                self.handover_object_position.unsqueeze(0),
                self.left_orientation,
                self.left_object_offset,
            )
            left_gripper = GRIPPER_CLOSE
            if not self.right_approach_staged:
                # First align outside the baton end while retaining vertical
                # clearance.  Descending along the tool axis lets both open
                # fingers straddle the short side before either touches it.
                staging_goal = right_pregrasp_goal.clone()
                staging_goal[:, 0] += 0.07
                right_desired = staging_goal.clone()
                right_desired[:, :3] += self.right_handover_command_offset.unsqueeze(0)
                if self._reached(right_pose, staging_goal):
                    self.right_approach_staged = True
                    self.state_time = 0.0
                    print("[physical-handover] RIGHT_APPROACH staged", flush=True)
                elif self.state_time >= 5.0:
                    self._fail("right_staging_timeout")
            else:
                right_desired = right_pregrasp
                if self._reached(right_pose, right_pregrasp_goal):
                    self.right_grasp_start_pose = right_pose.clone()
                    self.right_grasp_goal = right_site_goal.clone()
                    self.right_grasp_command = right_site_goal.clone()
                    self.right_grasp_command[:, :3] += (
                        self.right_grasp_final_command_offset.unsqueeze(0)
                    )
                    self._transition(HandoverState.RIGHT_GRASP)
                elif self.state_time >= 4.0:
                    self._fail("right_pregrasp_timeout")
        elif self.state == HandoverState.RIGHT_GRASP:
            left_desired = self._hand_pose_for_object_position(
                self.handover_object_position.unsqueeze(0),
                self.left_orientation,
                self.left_object_offset,
            )
            left_gripper = GRIPPER_CLOSE
            approach_fraction = min(self.state_time / 1.5, 1.0)
            right_desired = self.right_grasp_start_pose + approach_fraction * (
                self.right_grasp_command - self.right_grasp_start_pose
            )
            if (
                self.grasp_close_time is None
                and approach_fraction >= 1.0
                and self._reached(right_pose, self.right_grasp_goal)
            ):
                self.grasp_close_time = self.state_time
            if self.grasp_close_time is not None:
                right_gripper = GRIPPER_CLOSE
                if self.state_time - self.grasp_close_time >= 0.8:
                    self.right_object_offset = math_utils.subtract_frame_transforms(
                        right_pose[:, :3],
                        right_pose[:, 3:7],
                        object_pose[:, :3],
                        object_pose[:, 3:7],
                    )
                    self.left_hold_pose = left_pose.clone()
                    self.right_hold_pose = right_pose.clone()
                    self._transition(HandoverState.TRANSFER)
            elif self.state_time >= 5.0:
                self._fail("right_grasp_pose_timeout")
        elif self.state == HandoverState.TRANSFER:
            load_fraction = min(self.state_time / 1.0, 1.0)
            left_desired = self.left_hold_pose
            right_desired = self.right_hold_pose
            left_gripper = GRIPPER_CLOSE
            right_gripper = GRIPPER_CLOSE
            if load_fraction >= 1.0:
                self._transition(HandoverState.LEFT_RELEASE)
        elif self.state == HandoverState.LEFT_RELEASE:
            left_desired = self.left_hold_pose
            right_desired = self.right_hold_pose
            right_gripper = GRIPPER_CLOSE
            if self.state_time >= 0.7:
                # The object settles into the receiver while the giver opens.
                # Re-measure this observable transform instead of transporting
                # with the stale offset captured during the shared hold.
                self.right_object_offset = math_utils.subtract_frame_transforms(
                    right_pose[:, :3],
                    right_pose[:, 3:7],
                    object_pose[:, :3],
                    object_pose[:, 3:7],
                )
                self.left_retreat_pose = left_pose.clone()
                self.left_retreat_pose[:, 0] -= 0.15
                self.left_retreat_pose[:, 1] += 0.12
                self.right_transport_start_pose = right_pose.clone()
                self.right_transport_goal = self._hand_pose_for_object_position(
                    target_pose[:, :3], self.right_orientation, self.right_object_offset
                )
                self._transition(HandoverState.RIGHT_TO_TARGET)
        elif self.state == HandoverState.RIGHT_TO_TARGET:
            left_desired = self.left_retreat_pose
            travel_duration = 6.0
            lower_duration = 2.0
            travel_pose = self.right_transport_goal.clone()
            travel_pose[:, 2] = self.right_transport_start_pose[:, 2]
            if self.state_time < travel_duration:
                fraction = min(self.state_time / travel_duration, 1.0)
                smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                right_desired = self.right_transport_start_pose + smooth * (
                    travel_pose - self.right_transport_start_pose
                )
            else:
                fraction = min(
                    (self.state_time - travel_duration) / lower_duration,
                    1.0,
                )
                smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                right_desired = travel_pose + smooth * (
                    self.right_transport_goal - travel_pose
                )
            right_gripper = GRIPPER_CLOSE
            if (
                self.state_time >= travel_duration + lower_duration
                and fraction >= 1.0
                and self._reached(
                right_pose, self.right_transport_goal
                )
            ):
                self._transition(HandoverState.RIGHT_RELEASE)
            elif self.state_time >= 10.5:
                self._fail("right_target_timeout")
        elif self.state == HandoverState.RIGHT_RELEASE:
            left_desired = self.left_retreat_pose
            right_desired = self._hand_pose_for_object_position(
                target_pose[:, :3], self.right_orientation, self.right_object_offset
            )
            if self.state_time >= 0.8:
                self.right_retreat_pose = right_pose.clone()
                self.right_retreat_pose[:, 2] += 0.10
                self._transition(HandoverState.RETREAT)
        elif self.state == HandoverState.RETREAT:
            left_desired = self.left_retreat_pose
            right_desired = self.right_retreat_pose
            if self._reached(right_pose, right_desired):
                self._transition(HandoverState.COMPLETE)
            elif self.state_time >= 4.0:
                self._fail("right_retreat_timeout")

        left_desired[:, 3:7] = self.left_orientation
        right_desired[:, 3:7] = self.right_orientation
        left_action = torch.cat(
            (
                left_desired,
                torch.full(
                    (left_pose.shape[0], 1), left_gripper, device=self.device
                ),
            ),
            dim=-1,
        )
        right_action = torch.cat(
            (
                right_desired,
                torch.full(
                    (right_pose.shape[0], 1), right_gripper, device=self.device
                ),
            ),
            dim=-1,
        )
        self.state_time += self.dt
        return torch.cat((left_action, right_action), dim=-1), self.complete or self.failed
