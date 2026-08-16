from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac import failure_videos


def _source_payload(*, success=False):
    return {
        "task": "bimanual_handover_item",
        "scenario": "static",
        "seed": 0,
        "horizon": 1000,
        "model_identity": {
            "manifest_authenticated": True,
            "left_fingerprint": "left",
            "right_fingerprint": "right",
        },
        "results": [
            {
                "episode": 3,
                "success": success,
                "steps": 262,
                "reason": "success" if success else "policy_complete",
            }
        ],
    }


def test_source_selection_accepts_only_original_failures(tmp_path):
    path = tmp_path / "source.json"
    raw = json.dumps(_source_payload(), sort_keys=True).encode()
    path.write_bytes(raw)

    payload, rows, digest, resolved = failure_videos._load_source(
        path, "bimanual_handover_item", (3,)
    )

    assert payload["seed"] == 0
    assert rows[3]["success"] is False
    assert digest == hashlib.sha256(raw).hexdigest()
    assert resolved == path.resolve()

    path.write_text(json.dumps(_source_payload(success=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="successful"):
        failure_videos._load_source(path, "bimanual_handover_item", (3,))


def test_episode_indices_are_exact_ordered_and_unique():
    assert failure_videos._normalize_episode_indices([3, 1, 8]) == (3, 1, 8)
    with pytest.raises(ValueError, match="duplicate"):
        failure_videos._normalize_episode_indices([3, 3])
    with pytest.raises(ValueError, match="non-negative"):
        failure_videos._normalize_episode_indices([-1])


def test_composed_rgb_uses_front_and_overhead_with_front_fallback():
    front = np.full((2, 3, 3), 10, dtype=np.uint8)
    overhead = np.full((2, 3, 3), 20, dtype=np.uint8)
    both = SimpleNamespace(
        perception_data={"front_rgb": front, "overhead_rgb": overhead}
    )
    frame, used = failure_videos._compose_frame(both, ("front", "overhead"))
    assert frame.shape == (2, 6, 3)
    assert used == ("front", "overhead")
    assert np.all(frame[:, :3] == 10)
    assert np.all(frame[:, 3:] == 20)

    front_only = SimpleNamespace(perception_data={"front_rgb": front})
    frame, used = failure_videos._compose_frame(
        front_only, ("front", "overhead")
    )
    assert frame.shape == front.shape
    assert used == ("front",)


def test_task_proxy_records_reset_get_and_step_observations():
    observations = [object(), object(), object()]

    class Task:
        def reset(self):
            return ["description"], observations[0]

        def get_observation(self):
            return observations[1]

        def step(self, action):
            assert action == "action"
            return observations[2], 0.0, False

        def variation_count(self):
            return 4

    class Recorder:
        def __init__(self):
            self.values = []

        def capture(self, observation):
            self.values.append(observation)

    recorder = Recorder()
    proxy = failure_videos.RecordingTaskEnvironment(Task(), recorder)

    assert proxy.reset()[1] is observations[0]
    assert proxy.get_observation() is observations[1]
    assert proxy.step("action")[0] is observations[2]
    assert proxy.variation_count() == 4
    assert recorder.values == observations


def test_replay_is_publishable_only_when_model_matches_and_it_still_fails():
    identity = {"manifest_authenticated": True, "fingerprint": "abc"}
    original = {
        "episode": 7,
        "success": False,
        "reason": "policy_complete",
        "steps": 100,
    }
    replay = dict(original)
    confirmation = failure_videos._validate_replay(
        original, replay, identity, dict(identity)
    )
    assert confirmation == {
        "replay_confirmed_failure": True,
        "same_reason": True,
        "same_steps": True,
    }

    successful_replay = dict(replay, success=True, reason="success")
    with pytest.raises(RuntimeError, match="succeeded during replay"):
        failure_videos._validate_replay(
            original, successful_replay, identity, dict(identity)
        )
    with pytest.raises(RuntimeError, match="model identity"):
        failure_videos._validate_replay(
            original, replay, identity, {"fingerprint": "different"}
        )


def test_bimanual_episode_offset_is_forwarded_unchanged(monkeypatch):
    seen = {}

    def run_episode(
        task_environment,
        worker,
        episode,
        seed,
        horizon,
        **kwargs,
    ):
        seen.update(
            episode=episode,
            seed=seed,
            horizon=horizon,
            scenario=kwargs["scenario"],
            reference=kwargs["scenario_reference_steps"],
        )
        return {"episode": episode, "success": False}

    monkeypatch.setattr(failure_videos.direct_evaluate, "_run_episode", run_episode)
    worker = SimpleNamespace(policy_steps=262)
    result = failure_videos._run_selected_episode(
        object(),
        worker,
        "bimanual_handover_item",
        {"seed": 0, "horizon": 1000},
        37,
    )

    assert result == {"episode": 37, "success": False}
    assert seen == {
        "episode": 37,
        "seed": 0,
        "horizon": 1000,
        "scenario": "static",
        "reference": 262,
    }
