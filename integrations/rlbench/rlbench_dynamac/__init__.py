"""Small public API for the author-aligned RLBench DynaMAC adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_DEMO_ADAPTER_EXPORTS = (
    "ADAPTER_CLAIM_BOUNDARY",
    "ArmEpisodeArrays",
    "BimanualDemonstrationResult",
    "BimanualEpisodeArrays",
    "DYNAMAC_GRIPPER_TARGET_TIMING",
    "DYNAMAC_POSE_TARGET_TIMING",
    "UnimanualDemonstrationResult",
    "UnsafeLowDimPickleError",
    "adapt_bimanual_demonstrations",
    "adapt_unimanual_demonstrations",
    "extract_bimanual_episode",
    "extract_unimanual_episode",
    "load_low_dim_obs_pickle",
    "load_low_dim_obs_pickles",
    "make_bimanual_demonstrations",
    "make_unimanual_demonstrations",
)
_SEGMENTATION_EXPORTS = (
    "DYNAMAC_CURRENT_STATE_TIMING",
    "TAPAS_ACTION_TIMING",
    "TAPAS_BIMANUAL_APPLICATION_SOURCE_STATUS",
    "TAPAS_DEFAULT_CONFIG_PATH",
    "TAPAS_NUMPY_PORT_SOURCE_STATUS",
    "TAPAS_REFERENCE_COMMIT",
    "BimanualTAPASSegmentation",
    "TAPASSegmentation",
    "TAPASSegmentationConfig",
    "align_tapas_boundaries",
    "current_gripper_state",
    "forward_gripper_action",
    "forward_pose_action",
    "load_rlbench_segmentation_config",
    "next_observation_actions",
    "segment_bimanual_pose_trajectories",
    "segment_bimanual_trajectories",
    "segment_pose_trajectories",
    "segment_trajectories",
    "tapas_distance_boundaries",
    "tapas_gripper_boundaries",
    "tapas_velocity_boundaries",
    "translation_action_magnitude",
)
_TASK_SPEC_EXPORTS = (
    "RLBENCH_REFERENCE_COMMIT",
    "TASK_SPECS",
    "TaskPoseChunk",
    "TaskSpec",
    "core_pose_wxyz_to_rlbench_xyzw",
    "get_task_spec",
    "load_task_specs",
    "rlbench_pose_xyzw_to_core_wxyz",
    "split_task_pose_chunks",
    "task_pose_chunks",
    "unwrap_task_low_dim_state",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
)
_STORE_BOTTLE_EXPORTS = (
    "STORE_BOTTLE_CONFIG_PATH",
    "STORE_BOTTLE_POLICY_SPEC_SCHEMA",
    "STORE_BOTTLE_SEMANTIC_SCHEMA",
    "STORE_BOTTLE_SEMANTIC_VERSION",
    "STORE_BOTTLE_TASK_NAME",
    "StoreBottleEntityGroup",
    "StoreBottleSemanticSpec",
    "extract_store_bottle_semantic_episode",
    "load_store_bottle_semantic_spec",
    "make_store_bottle_semantic_demonstrations",
    "store_bottle_semantic_observations_from_rlbench",
    "store_bottle_policy_spec_identity",
    "store_bottle_semantic_task_spec",
    "validate_store_bottle_scene_hierarchy",
)

_EXPORT_MODULES = {
    **{name: ".data.demo_adapter" for name in _DEMO_ADAPTER_EXPORTS},
    **{name: ".data.tapas_segmentation" for name in _SEGMENTATION_EXPORTS},
    **{name: ".core.task_specs" for name in _TASK_SPEC_EXPORTS},
    **{
        name: ".protocols.store_bottle_semantics"
        for name in _STORE_BOTTLE_EXPORTS
    },
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
