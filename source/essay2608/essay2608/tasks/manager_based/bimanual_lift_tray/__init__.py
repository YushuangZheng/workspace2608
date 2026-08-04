"""Bimanual lift-tray task registration."""

import gymnasium as gym


gym.register(
    id="Essay2608-Bimanual-Lift-Tray-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_tray_env_cfg:BimanualLiftTrayEnvCfg",
    },
)
