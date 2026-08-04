"""Dynamic pick-and-place environment for essay2608."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import (
    FrankaCubeLiftEnvCfg,
)


@configclass
class DynamicPickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    """Minimal Franka dynamic pick-and-place environment."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Start with a single environment for debugging.
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Longer episode for later scripted demonstrations.
        self.episode_length_s = 20.0

        # Deterministic observations during initial development.
        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = False

        # Keep the target fixed throughout one episode.
        self.commands.object_pose.resampling_time_range = (1000.0, 1000.0)
        self.commands.object_pose.debug_vis = True

        # Initial target region on the right side of the table.
        self.commands.object_pose.ranges.pos_x = (0.55, 0.55)
        self.commands.object_pose.ranges.pos_y = (0.20, 0.20)
        self.commands.object_pose.ranges.pos_z = (0.08, 0.08)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # Viewer.
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.15)
