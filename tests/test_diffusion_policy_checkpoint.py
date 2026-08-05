from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from essay2608.policy import (  # noqa: E402
    DiffusionPolicy,
    DiffusionPolicyConfig,
    DynaMACDemonstration,
    DynaMACObservation,
)


def test_diffusion_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    poses = np.tile(np.asarray([0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]), (8, 1))
    poses[:, 0] += np.linspace(0.0, 0.1, len(poses))
    object_poses = np.tile(np.asarray([0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]), (8, 1))
    demonstrations = [
        DynaMACDemonstration(
            ee_pose=poses,
            action_pose=poses,
            gripper=np.zeros((8, 1)),
            frames={"object": object_poses},
            skill=np.zeros(8, dtype=np.int64),
            name=f"demo_{index}",
        )
        for index in range(2)
    ]
    config = DiffusionPolicyConfig(
        horizon=4,
        execution_horizon=2,
        diffusion_steps=2,
        hidden_dimension=16,
        epochs=1,
        batch_size=32,
        device="cpu",
    )
    observation = DynaMACObservation(poses[0], {"object": object_poses[0]})
    policy = DiffusionPolicy(config).fit(demonstrations)
    assert policy.condition_mean.shape == (14,)
    policy.reset(observation)
    expected = policy.act(observation)
    checkpoint = tmp_path / "dp.npz"
    policy.save(checkpoint)

    loaded = DiffusionPolicy.load(checkpoint)
    loaded.reset(observation)
    actual = loaded.act(observation)
    np.testing.assert_allclose(actual.pose, expected.pose)
    np.testing.assert_allclose(actual.gripper, expected.gripper)
