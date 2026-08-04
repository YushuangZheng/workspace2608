"""Contact-rich bimanual handover without kinematic object attachment."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from essay2608.tasks.manager_based.bimanual_handover.handover_env_cfg import (
    ActionsCfg,
    BimanualHandoverEnvCfg,
    RewardsCfg,
    TerminationsCfg,
    ee_pose,
    gripper_state,
    object_pose,
)


PHYSICAL_OBJECT_HEIGHT_M = 0.181
PHYSICAL_TARGET_POSE = (0.48, -0.20, PHYSICAL_OBJECT_HEIGHT_M, 1.0, 0.0, 0.0, 0.0)


def physical_target_pose(env) -> torch.Tensor:
    """Return the placement pose on the physical table surface."""

    return torch.tensor(PHYSICAL_TARGET_POSE, device=env.device).repeat(env.num_envs, 1)


def _physical_franka(
    prim_path: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    static_friction: float = 1.2,
    dynamic_friction: float = 1.0,
) -> ArticulationCfg:
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path=prim_path,
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=position,
            rot=rotation,
        ),
    )
    robot.spawn.activate_contact_sensors = True
    robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    return robot


@configclass
class PhysicalHandoverSceneCfg(InteractiveSceneCfg):
    """Two independent Frankas, one dynamic cube, and actual finger contacts."""

    left_robot: ArticulationCfg = _physical_franka(
        "{ENV_REGEX_NS}/LeftRobot", (0.0, 0.40, 0.0)
    )
    right_robot: ArticulationCfg = _physical_franka(
        # The receiver mirrors the sender from the east side of the table.
        # A 180-degree yaw preserves the finger closing axis in world space,
        # while putting the grasp point well inside the receiver workspace.
        "{ENV_REGEX_NS}/RightRobot",
        (1.16, -0.25, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.48, 0.20, PHYSICAL_OBJECT_HEIGHT_M),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            # A short baton leaves two distinct grasp sites without extending
            # backward into the sender palm during table pickup.
            size=(0.24, 0.055, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
                max_linear_velocity=2.0,
                max_angular_velocity=8.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.45, 0.12)
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.48, 0.0, 0.14)),
        spawn=sim_utils.CuboidCfg(
            size=(0.90, 1.10, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.25, 0.28)
            ),
        ),
    )

    left_leftfinger_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeftRobot/panda_leftfinger",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        track_pose=True,
        debug_vis=False,
    )
    left_rightfinger_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeftRobot/panda_rightfinger",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        track_pose=True,
        debug_vis=False,
    )
    right_leftfinger_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RightRobot/panda_leftfinger",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        track_pose=True,
        debug_vis=False,
    )
    right_rightfinger_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RightRobot/panda_rightfinger",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        track_pose=True,
        debug_vis=False,
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
class PhysicalObservationsCfg:
    """Policy observations deliberately exclude privileged contact forces."""

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
        target_pose = ObsTerm(func=physical_target_pose)
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
        object_linear_velocity = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_angular_velocity = ObsTerm(
            func=mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class PhysicalEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.012, 0.012),
                "y": (-0.012, 0.012),
                "z": (0.0, 0.0),
                "yaw": (-0.10, 0.10),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PhysicalActionsCfg(ActionsCfg):
    """Keep the 16-D interface while avoiding an impulsive receiver squeeze."""

    right_gripper_action = BinaryJointPositionActionCfg(
        asset_name="right_robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.024},
    )


@configclass
class BimanualPhysicalHandoverEnvCfg(BimanualHandoverEnvCfg):
    """Physical handover with a 16-D action and privileged evaluation sensors."""

    scene: PhysicalHandoverSceneCfg = PhysicalHandoverSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PhysicalObservationsCfg = PhysicalObservationsCfg()
    actions: PhysicalActionsCfg = PhysicalActionsCfg()
    events: PhysicalEventCfg = PhysicalEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 45.0
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.viewer.eye = (1.35, 1.35, 1.0)
        self.viewer.lookat = (0.48, 0.0, 0.30)
