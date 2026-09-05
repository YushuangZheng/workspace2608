"""Read-only V4 replay checks for phase-three role and weighted-PoE control."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from essay2608.policy import DynaMAC, DynaMACObservation
from essay2608.policy.closed_loop import (
    BeliefUpdater,
    ClosedLoopExecutionController,
    ClosedLoopTaskModelBuilder,
    FrameRole,
    RelationDecision,
    RuntimeObservation,
    StateId,
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
    skill_sequences = []
    virtual_frames: dict[str, np.ndarray] = {}
    for skill_index, skill in enumerate(policy.skills):
        data = aligned[skill_index]
        virtual_frames[f"virtual_skill_{skill.label}"] = data.ee_pose[
            demonstration_index, 0
        ].copy()
        sequence = []
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
        skill_sequences.append(tuple(sequence))

    def runtime_observation(
        tick: int,
        item,
        previous_item=None,
        previous_command_pose=None,
    ) -> RuntimeObservation:
        _, ee_pose, frames, gripper, _ = item
        return RuntimeObservation(
            tick=tick,
            ee_pose=ee_pose,
            frame_poses=frames,
            gripper_state=gripper,
            previous_command_pose=(
                None
                if previous_item is None
                else (
                    previous_item[4]
                    if previous_command_pose is None
                    else previous_command_pose
                )
            ),
            previous_ee_pose=None if previous_item is None else previous_item[1],
            tracking_reliability={},
            frame_visibility={},
        )

    first = skill_sequences[0][0]
    policy.reset(DynaMACObservation(first[1], first[2]), mode_strategy="map")
    role_counts: Counter[FrameRole] = Counter()
    unavailable = []
    recoveries = []
    selected_unknown_weights = []
    tick = 0
    carried_relations = None
    carried_decisions = None
    carried_evidence_decisions = None
    controller = ClosedLoopExecutionController(model)
    controller.reset(first[0])
    previous_roles = None
    previous_belief = None
    for sequence in skill_sequences:
        initial = sequence[0]
        skill_index = initial[0].skill_index
        mode = mode_by_skill[skill_index]
        initial_relations = {
            frame: values[mode].copy()
            for frame, values in model.state(initial[0]).demo_relation_priors.items()
        }
        updater = BeliefUpdater(model)
        updater.reset(
            initial_progress={initial[0]: 1.0},
            initial_relations=(
                initial_relations if carried_relations is None else carried_relations
            ),
            initial_relation_decisions=carried_decisions,
            initial_relation_evidence_decisions=carried_evidence_decisions,
            previous_observation=runtime_observation(tick, initial),
        )
        if previous_roles is not None:
            assert previous_belief is not None
            # The replay resets its per-skill belief window, but the real
            # executor crosses this boundary transactionally. Preserve that
            # causal entry so a demonstrated LINK receives its ordinary
            # post-entry confirmation interval instead of being mistaken for
            # an arbitrary reset directly into a linked state.
            controller.role_router.commit_boundary_entry(
                previous_roles,
                previous_belief,
                initial[0],
                mode_by_skill=mode_by_skill,
            )
            controller.commit_reentry(initial[0])
        previous = initial
        for current in sequence[1:]:
            tick += 1
            truth, ee_pose, frames, _, _ = current
            belief = updater.update(
                runtime_observation(tick, current, previous),
                # The fixed trace supplies the recorded target state whose
                # action produced this observation; the queried controller
                # action is not applied to the demonstration.
                executed_reference_state=current[0],
                mode_by_skill=mode_by_skill,
            )
            cycle = controller.update(
                belief,
                DynaMACObservation(ee_pose, frames),
                mode_by_skill=mode_by_skill,
            )
            roles = cycle.roles
            result = cycle.weighted_action
            role_counts.update(decision.role for decision in roles.decisions.values())
            if not result.available:
                unavailable.append(
                    (
                        tick,
                        truth,
                        controller.cursor,
                        belief.progress,
                        roles.decisions,
                        roles.execution_weights,
                    )
                )
            recoveries.extend(roles.recovery_intents)
            for frame, decision in roles.decisions.items():
                if decision.role == FrameRole.DEFER:
                    selected_unknown_weights.append(
                        (truth, frame, decision.execution_weight)
                    )
            previous = current
        carried_relations = belief.relation_posteriors
        carried_decisions = {
            frame: estimate.decision_state
            for frame, estimate in belief.relation_estimates.items()
            if estimate.decision_state != RelationDecision.UNKNOWN
        }
        carried_evidence_decisions = updater.informative_evidence_decisions
        previous_roles = roles
        previous_belief = belief

    assert unavailable == []
    assert recoveries == []
    assert role_counts[FrameRole.EXECUTE] > 0
    assert selected_unknown_weights
    assert all(weight >= 0.0 for _, _, weight in selected_unknown_weights)
