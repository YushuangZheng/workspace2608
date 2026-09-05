from __future__ import annotations

from pathlib import Path

from evaluations.iclr2027.interfaces.feature_schema import EPISODE_SCHEMA
from evaluations.iclr2027.runners.episode_io import (
    EpisodeWriter,
    load_cycles,
    load_episode,
    resolve_cycle_file,
)
from evaluations.iclr2027.tests.test_a2_infrastructure import feature


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
