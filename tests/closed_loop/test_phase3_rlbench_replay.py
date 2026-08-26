"""Read-only V4 replay checks for phase-three role and weighted-PoE control."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import DynaMAC, DynaMACObservation
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    ClosedLoopTaskModelBuilder,
    FrameRole,
    FrameRoleRouter,
    RuntimeObservation,
    StateId,
    WeightedPoEExecutor,
)
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import demonstration_paths

DATA_ROOT = Path("integrations/rlbench/data/training/main")
MODEL_PATH = Path("integrations/rlbench/models/v4/stack_wine/model.npz")


def test_stack_wine_normal_replay_routes_roles_and_keeps_actions_available() -> None:
    try:
        paths = demonstration_paths(DATA_ROOT, "stack_wine", 5)
    except FileNotFoundError:
        pytest.skip("本地未安装 StackWine 五条正常训练示范")
    if not MODEL_PATH.is_file():
        pytest.skip("本地未安装 StackWine V4 模型")

    converted = make_unimanual_demonstrations(
        load_low_dim_obs_pickles(paths),
        "stack_wine",
        names=[path.parent.name for path in paths],
    )
    policy = DynaMAC.load(MODEL_PATH)
    builder = ClosedLoopTaskModelBuilder()
    model = builder.build(
        policy,
        converted.demonstrations,
        recoverable_frames=("wine_bottle",),
    )
    aligned = builder._align_demonstrations(policy, converted.demonstrations)

    demonstration_index = 0
    mode_by_skill = {
        skill_index: next(
            mode
            for mode, members in enumerate(skill.mode_demonstration_indices)
            if demonstration_index in members
        )
        for skill_index, skill in enumerate(policy.skills)
    }
    sequence = []
    virtual_frames: dict[str, np.ndarray] = {}
    for skill_index, skill in enumerate(policy.skills):
        data = aligned[skill_index]
        virtual_frames[f"virtual_skill_{skill.label}"] = data.ee_pose[
            demonstration_index, 0
        ].copy()
        for local_index in range(skill.duration):
            frames = {
                name: values[demonstration_index, local_index].copy()
                for name, values in data.frames.items()
            }
            frames.update(
                {name: value.copy() for name, value in virtual_frames.items()}
            )
            sequence.append(
                (
                    StateId(skill_index, local_index),
                    data.ee_pose[demonstration_index, local_index].copy(),
                    frames,
                    data.gripper[demonstration_index, local_index].copy(),
                )
            )

    def runtime_observation(tick: int, item, previous_item=None) -> RuntimeObservation:
        _, ee_pose, frames, gripper = item
        return RuntimeObservation(
            tick=tick,
            ee_pose=ee_pose,
            frame_poses=frames,
            gripper_state=gripper,
            previous_command_pose=None if previous_item is None else ee_pose,
            previous_ee_pose=None if previous_item is None else previous_item[1],
            tracking_reliability={},
            frame_visibility={},
        )

    policy.reset(
        DynaMACObservation(sequence[0][1], sequence[0][2]),
        mode_strategy="map",
    )
    updater = BeliefUpdater(model)
    updater.reset(previous_observation=runtime_observation(0, sequence[0]))
    router = FrameRoleRouter(model)
    executor = WeightedPoEExecutor(model)

    role_counts: Counter[FrameRole] = Counter()
    unavailable = []
    recoveries = []
    selected_unknown_weights = []
    for tick in range(1, len(sequence)):
        truth, ee_pose, frames, _ = sequence[tick]
        belief = updater.update(
            runtime_observation(tick, sequence[tick], sequence[tick - 1]),
            permitted_boundaries=frozenset(model.boundaries),
            mode_by_skill=mode_by_skill,
        )
        roles = router.route(truth, belief, mode_by_skill=mode_by_skill)
        result = executor.query(
            DynaMACObservation(ee_pose, frames),
            truth,
            roles,
            mode_index=mode_by_skill[truth.skill_index],
        )
        role_counts.update(decision.role for decision in roles.decisions.values())
        if not result.available:
            unavailable.append((tick, truth, roles.execution_weights))
        recoveries.extend(roles.recovery_intents)
        for frame, decision in roles.decisions.items():
            if decision.role == FrameRole.DEFER:
                selected_unknown_weights.append(
                    (truth, frame, decision.execution_weight)
                )

    assert unavailable == []
    assert recoveries == []
    assert role_counts[FrameRole.EXECUTE] > 0
    assert selected_unknown_weights
    assert any(weight > 0.0 for _, _, weight in selected_unknown_weights)
