"""Lift-tray variant of the validated two-Franka handover scene."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..bimanual_handover.handover_env_cfg import (
    ActionsCfg,
    BimanualHandoverEnvCfg,
    EventCfg,
    HandoverSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class LiftTraySceneCfg(HandoverSceneCfg):
    """Two arms surrounding a lightweight shared tray."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, 0.0, 0.22), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.24, 0.34, 0.025),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.62, 0.12)),
        ),
    )


@configclass
class BimanualLiftTrayEnvCfg(BimanualHandoverEnvCfg):
    """Shared-object transport task with the same 16-D action contract."""

    scene: LiftTraySceneCfg = LiftTraySceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.lookat = (0.54, 0.0, 0.35)
