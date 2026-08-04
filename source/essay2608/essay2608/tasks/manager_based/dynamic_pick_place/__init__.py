"""Dynamic pick-and-place task registration."""

import gymnasium as gym


gym.register(
    id="Essay2608-Dynamic-Pick-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_pick_place_env_cfg:"
            "DynamicPickPlaceEnvCfg"
        ),
    },
)
