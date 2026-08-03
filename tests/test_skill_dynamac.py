"""Training-only tests for the skill-level DynaMAC baseline."""

from __future__ import annotations

import numpy as np

from essay2608.data.dataset import load_dataset
from essay2608.data.transforms import quaternion_residual_vector
from essay2608.policy import (
    DynaMACPolicy,
    OnlineDynaMACPrototype,
    SkillDynaMACPolicy,
)
from essay2608.policy.base import PHASE_NAMES, PolicyObservation


def test_quaternion_residual_handles_scalar_and_batch_inputs() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    quarter_turn = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    scalar = quaternion_residual_vector(identity, quarter_turn)
    batch = quaternion_residual_vector(identity, np.stack((identity, quarter_turn)))
    np.testing.assert_allclose(scalar, [0.0, 0.0, np.pi / 2.0])
    np.testing.assert_allclose(batch[0], np.zeros(3))
    np.testing.assert_allclose(batch[1], scalar)


def test_online_policy_compatibility_alias_is_explicit() -> None:
    assert DynaMACPolicy is OnlineDynaMACPrototype
    assert OnlineDynaMACPrototype.name == "full_dynamac"


def test_skill_policy_selects_fixed_explainable_frames_from_training_data() -> None:
    demonstrations, _ = load_dataset("data/pick_place_static/v1", verify_hashes=True)
    policy = SkillDynaMACPolicy()
    policy.fit(demonstrations)
    assert set(policy.selected_frames) == set(range(len(PHASE_NAMES)))
    assert all(policy.selected_frames[phase] for phase in range(len(PHASE_NAMES)))
    assert policy.skill_diagnostics[4]["object_link"]["linked"]
    assert "object" not in policy.selected_frames[4]
    assert policy.models["object"].pose_covariance.shape[-2:] == (6, 6)

    first = demonstrations[0]
    observation = PolicyObservation(
        ee_pose=first.ee_pose[0],
        object_pose=first.object_pose[0],
        target_pose=first.target_pose[0],
    )
    policy.reset(observation)
    step = policy.act(observation)
    assert step.diagnostics["selection_mode"] == "offline_skill_fixed"
    assert step.diagnostics["active_frames"] == policy.selected_frames[0]
    assert not hasattr(policy, "detector")
