"""Bimanual handover task registration."""

import gymnasium as gym


gym.register(
    id="Essay2608-Bimanual-Handover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:BimanualHandoverEnvCfg",
    },
)
