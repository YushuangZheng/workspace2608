"""OpenArm bimanual handover environment for dynamic cross-arm coordination."""

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


TARGET_POSE = (0.48, -0.20, 0.22, 1.0, 0.0, 0.0, 0.0)


def ee_pose(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the configured Cartesian tool pose in local environment coordinates."""

    robot = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    orientation = robot.data.body_quat_w[:, body_id]
    offset = torch.tensor((0.0, 0.0, 0.107), device=robot.device).repeat(env.num_envs, 1)
    position = robot.data.body_pos_w[:, body_id] + math_utils.quat_apply(orientation, offset)
    position -= env.scene.env_origins
    return torch.cat((position, orientation), dim=-1)


def object_pose(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return object pose in local environment coordinates."""

    asset = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w - env.scene.env_origins
    return torch.cat((position, asset.data.root_quat_w), dim=-1)


def target_pose(env) -> torch.Tensor:
    """Return the fixed placement target in local environment coordinates."""

    return torch.tensor(TARGET_POSE, device=env.device).repeat(env.num_envs, 1)


def gripper_state(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return both measured finger joint positions for one gripper."""

    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids]


@configclass
class HandoverSceneCfg(InteractiveSceneCfg):
    """One bimanual robot and a lightweight transfer object."""

    left_robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftRobot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=(0.0, 0.40, 0.0)),
    )
    right_robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightRobot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=(0.0, -0.40, 0.0)),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, 0.20, 0.22), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.045, 0.045, 0.045),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.95)),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.48, 0.0, 0.14)),
        spawn=sim_utils.CuboidCfg(
            size=(0.90, 1.10, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Two independent Cartesian actions and gripper commands."""

    left_arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="left_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    left_gripper_action = BinaryJointPositionActionCfg(
        asset_name="left_robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )
    right_arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="right_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    right_gripper_action = BinaryJointPositionActionCfg(
        asset_name="right_robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Explicit geometric and measured-gripper handover observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        left_ee_pose = ObsTerm(
            func=ee_pose,
            params={"asset_cfg": SceneEntityCfg("left_robot", body_names=["panda_hand"])},
        )
        right_ee_pose = ObsTerm(
            func=ee_pose,
            params={"asset_cfg": SceneEntityCfg("right_robot", body_names=["panda_hand"])},
        )
        object_pose = ObsTerm(func=object_pose, params={"asset_cfg": SceneEntityCfg("object")})
        target_pose = ObsTerm(func=target_pose)
        left_gripper_state = ObsTerm(
            func=gripper_state,
            params={
                "asset_cfg": SceneEntityCfg(
                    "left_robot", joint_names=["panda_finger_joint.*"]
                )
            },
        )
        right_gripper_state = ObsTerm(
            func=gripper_state,
            params={
                "asset_cfg": SceneEntityCfg(
                    "right_robot", joint_names=["panda_finger_joint.*"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.01, 0.01)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    left_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("left_robot")},
    )
    right_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("right_robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BimanualHandoverEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal dynamic handover benchmark with a 16-D absolute action."""

    scene: HandoverSceneCfg = HandoverSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.4, 1.4, 1.1)
        self.viewer.lookat = (0.48, 0.0, 0.32)
