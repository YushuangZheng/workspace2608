"""Registration for the contact-rich bimanual handover task."""

import gymnasium as gym


gym.register(
    id="Essay2608-Bimanual-Physical-Handover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.physical_handover_env_cfg:BimanualPhysicalHandoverEnvCfg"
        ),
    },
)
