from __future__ import annotations

from pathlib import Path

from evaluations.iclr2027.interfaces.feature_schema import EPISODE_SCHEMA, FeatureRecord
from evaluations.iclr2027.runners.episode_io import (
    EpisodeWriter,
    load_cycles,
    load_episode,
    resolve_cycle_file,
)


def feature() -> dict:
    """Return the minimal valid public record used by the storage tests."""

    return FeatureRecord(
        episode_id="development_nominal/close_jar/0000",
        cycle=2,
        observation_timestamp=2,
        action_timestamp=2,
        arms={
            "single": {
                "ee_pose_xyzw": [0, 0, 0, 0, 0, 0, 1],
                "gripper_open": 1.0,
            }
        },
        task_state=(0.0, 1.0),
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        policy_state={"policy_step": 2},
        action_resolution={"aggregate": "reached"},
    ).to_dict()


def test_cycle_and_episode_records_commit_atomically(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, "development_nominal/close_jar/0000")
    writer.write_cycle(
        feature(),
        {"cycle": 2, "schema": "audit"},
        execution={"reward": 0.0},
    )
    path = writer.finalize(
        {
            "schema": EPISODE_SCHEMA,
            "episode_id": "development_nominal/close_jar/0000",
            "split": "development_nominal",
            "task": "close_jar",
            "method_id": "m0_dynamac",
            "condition": "nominal",
            "success": False,
            "cycles": 1,
        }
    )
    episode = load_episode(path)
    assert episode["cycle_file_location"] == "episode_relative"
    assert not Path(episode["cycle_file"]).is_absolute()
    cycles = load_cycles(resolve_cycle_file(path, episode))
    assert episode["cycle_records"] == 1
    assert cycles[0]["feature"]["cycle"] == cycles[0]["audit"]["cycle"] == 2
