"""Normal V4 RLBench sidecar replay for phase-two scale validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import (
    BimanualDynaMAC,
    DynaMAC,
    synchronized_bimanual_demonstrations,
)
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    ClosedLoopTaskModelBuilder,
    ProgressStatus,
    RuntimeObservation,
    StateId,
)
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import demonstration_paths

DATA_ROOT = Path("integrations/rlbench/data/training/main")
MODEL_PATH = Path("integrations/rlbench/models/v4/stack_wine/model.npz")
HANDOVER_MODEL_ROOT = Path("integrations/rlbench/models/v4/bimanual_handover_item")


def test_stack_wine_five_normal_demonstrations_track_in_sidecar_replay() -> None:
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
    ordered_states = tuple(sorted(model.states))
    global_index = {state: index for index, state in enumerate(ordered_states)}
    offsets: list[int] = []
    no_plausible = 0
    update_count = 0

    for demonstration_index in range(5):
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
                    name: values[demonstration_index, local_index]
                    for name, values in data.frames.items()
                }
                frames.update(virtual_frames)
                sequence.append(
                    (
                        StateId(skill_index, local_index),
                        data.ee_pose[demonstration_index, local_index],
                        {name: value.copy() for name, value in frames.items()},
                        data.gripper[demonstration_index, local_index],
                        data.action_pose[demonstration_index, local_index],
                    )
                )

        def runtime_observation(
            tick: int,
            item,
            previous_item=None,
        ) -> RuntimeObservation:
            _, ee_pose, frames, gripper, _ = item
            return RuntimeObservation(
                tick=tick,
                ee_pose=ee_pose,
                frame_poses=frames,
                gripper_state=gripper,
                previous_command_pose=(
                    None if previous_item is None else previous_item[4]
                ),
                previous_ee_pose=None if previous_item is None else previous_item[1],
                tracking_reliability={},
                frame_visibility={},
            )

        updater = BeliefUpdater(model)
        updater.reset(previous_observation=runtime_observation(0, sequence[0]))
        for tick in range(1, len(sequence)):
            belief = updater.update(
                runtime_observation(tick, sequence[tick], sequence[tick - 1]),
                executed_reference_state=sequence[tick - 1][0],
                permitted_boundaries=frozenset(model.boundaries),
                mode_by_skill=mode_by_skill,
            )
            truth = sequence[tick][0]
            offsets.append(
                global_index[belief.progress.estimated_state] - global_index[truth]
            )
            no_plausible += int(
                belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE
            )
            update_count += 1
            for estimate in belief.relation_estimates.values():
                assert np.sum(estimate.posterior) == pytest.approx(1.0)
            assert set(belief.candidate_scores) == set(belief.progress.prior)

    absolute = np.abs(np.asarray(offsets, dtype=np.int64))
    assert float(np.mean(absolute)) <= 0.60
    assert int(np.max(absolute)) <= 2
    assert float(np.mean(absolute == 0)) >= 0.55
    assert no_plausible / update_count <= 0.01


def test_handover_right_static_segment_does_not_stick_at_old_boundary() -> None:
    try:
        paths = demonstration_paths(DATA_ROOT, "bimanual_handover_item", 5)
    except FileNotFoundError:
        pytest.skip("本地未安装 HandOver 五条正常训练示范")
    left_path = HANDOVER_MODEL_ROOT / "left.npz"
    right_path = HANDOVER_MODEL_ROOT / "right.npz"
    if not left_path.is_file() or not right_path.is_file():
        pytest.skip("本地未安装 HandOver V4 双臂模型")

    converted = make_bimanual_demonstrations(
        load_low_dim_obs_pickles(paths),
        "bimanual_handover_item",
        names=[path.parent.name for path in paths],
    )
    policy = BimanualDynaMAC(
        left=DynaMAC.load(left_path),
        right=DynaMAC.load(right_path),
    )
    builder = ClosedLoopTaskModelBuilder()
    _, model = builder.build_bimanual(
        policy,
        converted.left_demonstrations,
        converted.right_demonstrations,
        recoverable_frames=("item0",),
    )
    _, right_demonstrations = synchronized_bimanual_demonstrations(
        converted.left_demonstrations,
        converted.right_demonstrations,
    )
    aligned = builder._align_demonstrations(policy.right, right_demonstrations)

    demonstration_index = 0
    mode_by_skill = {
        skill_index: next(
            mode
            for mode, members in enumerate(skill.mode_demonstration_indices)
            if demonstration_index in members
        )
        for skill_index, skill in enumerate(policy.right.skills)
    }
    sequence = []
    virtual_frames: dict[str, np.ndarray] = {}
    for skill_index, skill in enumerate(policy.right.skills):
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
                    data.action_pose[demonstration_index, local_index].copy(),
                )
            )

    def runtime_observation(tick: int, item, previous_item=None) -> RuntimeObservation:
        _, ee_pose, frames, gripper, _ = item
        return RuntimeObservation(
            tick=tick,
            ee_pose=ee_pose,
            frame_poses=frames,
            gripper_state=gripper,
            previous_command_pose=None if previous_item is None else previous_item[4],
            previous_ee_pose=None if previous_item is None else previous_item[1],
            tracking_reliability={},
            frame_visibility={},
        )

    ordered_states = tuple(sorted(model.states))
    global_index = {state: index for index, state in enumerate(ordered_states)}
    updater = BeliefUpdater(model)
    updater.reset(previous_observation=runtime_observation(0, sequence[0]))
    offsets = []
    no_plausible = 0
    for tick in range(1, len(sequence)):
        belief = updater.update(
            runtime_observation(tick, sequence[tick], sequence[tick - 1]),
            executed_reference_state=sequence[tick - 1][0],
            permitted_boundaries=frozenset(model.boundaries),
            mode_by_skill=mode_by_skill,
        )
        truth = sequence[tick][0]
        offsets.append(
            global_index[belief.progress.estimated_state] - global_index[truth]
        )
        no_plausible += int(belief.progress.status == ProgressStatus.NO_PLAUSIBLE_STATE)

    absolute = np.abs(np.asarray(offsets, dtype=np.int64))
    assert float(np.mean(absolute)) <= 0.60
    assert int(np.max(absolute)) <= 3
    assert float(np.mean(absolute == 0)) >= 0.60
    assert no_plausible == 0
