from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.evaluation_videos import (
    CAPTURE_CONFIG_SCHEMA,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELECTION_SEED,
    EpisodeVideo,
    LightweightCaptureConfig,
    ObservationVideoRecorder,
    RecordingTaskEnvironment,
    SELECTION_PROTOCOL_ID,
    SELECTION_SCHEMA,
    finalize_cell_videos,
    retention_quota,
)


def _recordings(cell_dir: Path, successes: int, failures: int):
    cell_dir.mkdir(parents=True, exist_ok=True)
    records = []
    outcomes = [True] * successes + [False] * failures
    for episode, success in enumerate(outcomes):
        video = cell_dir / f"episode_{episode:03d}.mp4"
        sidecar = cell_dir / f"episode_{episode:03d}.json"
        video.write_bytes(f"video-{episode}".encode("ascii"))
        sidecar.write_text(
            json.dumps({"episode": episode, "success": success}),
            encoding="utf-8",
        )
        records.append(
            EpisodeVideo(
                episode=episode,
                success=success,
                video=video,
                companions=(sidecar,),
                episode_seed=2_608_000_000 + episode,
            )
        )
    return records


class _MemoryWriter:
    instances = []

    def __init__(self, path, *, width, height, config):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.config = config
        self.frames = 0
        self.values = []
        self.closed = False
        self.aborted = False
        self.__class__.instances.append(self)

    def write(self, frame):
        self.frames += 1
        self.values.append(np.asarray(frame).copy())

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


def _observation(value):
    return SimpleNamespace(
        perception_data={"front_rgb": np.full((4, 6, 3), value, dtype=np.float32)}
    )


def test_lightweight_recorder_streams_without_buffering_and_audits_v4():
    _MemoryWriter.instances.clear()
    config = LightweightCaptureConfig(capture_every_n_observations=2)
    recorder = ObservationVideoRecorder(
        Path("episode.mp4"),
        config=config,
        writer_factory=_MemoryWriter,
    )

    assert recorder.capture(_observation(0.5)) is True
    assert recorder.capture(_observation(0.6)) is False
    assert recorder.capture(_observation(1.0)) is True
    metadata = recorder.close()

    writer = _MemoryWriter.instances[0]
    assert writer.frames == 2
    assert writer.closed is True
    assert writer.values[0].dtype == np.uint8
    assert writer.values[0][0, 0, 0] == 127
    assert metadata["frames"] == 2
    assert metadata["observations_seen"] == 3
    assert metadata["capture_config"]["schema"] == CAPTURE_CONFIG_SCHEMA
    assert metadata["capture_config"]["streamed_without_frame_buffer"] is True


def test_recording_task_environment_captures_only_returned_observations():
    _MemoryWriter.instances.clear()

    class Environment:
        marker = "delegated"

        def reset(self):
            return ["description"], _observation(0.1)

        def get_observation(self):
            return _observation(0.2)

        def step(self, action):
            assert action == "action"
            return _observation(0.3), 0.0, False

    recorder = ObservationVideoRecorder(
        Path("episode.mp4"),
        writer_factory=_MemoryWriter,
    )
    proxy = RecordingTaskEnvironment(Environment(), recorder)

    proxy.reset()
    proxy.get_observation()
    proxy.step("action")

    assert proxy.marker == "delegated"
    assert recorder.frames == 3


@pytest.mark.parametrize(
    ("success_rate", "paper_rate", "tier", "successes", "failures"),
    [
        (0.85, 0.84, "sr_ge_80_paper_diff_lt_2pp", 0, 0),
        (0.82, 0.80, "sr_ge_80", 3, 3),
        (0.80, None, "sr_ge_80", 3, 3),
        (0.799, 0.799, "sr_50_to_lt_80", 5, 10),
        (0.50, 0.50, "sr_50_to_lt_80", 5, 10),
        (0.499, 0.499, "sr_lt_50", 5, 20),
    ],
)
def test_retention_quota_boundaries(
    success_rate,
    paper_rate,
    tier,
    successes,
    failures,
):
    quota = retention_quota(success_rate, paper_rate)

    assert quota.tier == tier
    assert quota.successes == successes
    assert quota.failures == failures


@pytest.mark.parametrize(
    ("successes", "failures", "paper_rate", "retained_successes", "retained_failures"),
    [
        (17, 3, 0.84, 0, 0),
        (16, 4, 0.50, 3, 3),
        (15, 10, 0.50, 5, 10),
        (10, 20, 0.50, 5, 20),
        # Both classes are below their requested quota, so every file remains.
        (4, 4, 0.90, 4, 4),
    ],
)
def test_finalize_applies_each_tier_and_class_shortfall(
    tmp_path,
    successes,
    failures,
    paper_rate,
    retained_successes,
    retained_failures,
):
    cell = tmp_path / "cell"
    records = _recordings(cell, successes, failures)

    manifest = finalize_cell_videos(
        cell,
        records,
        cell_key="task/scenario",
        successes=successes,
        episodes=successes + failures,
        paper_success_rate=paper_rate,
    )

    retained = manifest["selection"]["retained"]
    assert retained == {
        "successes": retained_successes,
        "failures": retained_failures,
    }
    assert len(list(cell.glob("*.mp4"))) == retained_successes + retained_failures
    assert len(list(cell.glob("episode_*.json"))) == (
        retained_successes + retained_failures
    )
    assert len(manifest["selected"]) == retained_successes + retained_failures
    assert len(manifest["deleted"]) == (
        successes + failures - retained_successes - retained_failures
    )
    on_disk = json.loads(
        (cell / "video_selection_manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk == manifest


def test_fixed_seed_selection_is_independent_of_inventory_order(tmp_path):
    first_cell = tmp_path / "first"
    second_cell = tmp_path / "second"
    first = _recordings(first_cell, 16, 4)
    second = _recordings(second_cell, 16, 4)

    first_manifest = finalize_cell_videos(
        first_cell,
        first,
        cell_key="open_microwave/static",
        successes=16,
        episodes=20,
        paper_success_rate=0.50,
        selection_seed=12345,
    )
    second_manifest = finalize_cell_videos(
        second_cell,
        reversed(second),
        cell_key="open_microwave/static",
        successes=16,
        episodes=20,
        paper_success_rate=0.50,
        selection_seed=12345,
    )

    assert [row["episode"] for row in first_manifest["selected"]] == [
        row["episode"] for row in second_manifest["selected"]
    ]
    assert first_manifest["selection"]["seed"] == 12345
    assert first_manifest["selection"]["protocol_id"] == SELECTION_PROTOCOL_ID
    assert first_manifest["schema"] == SELECTION_SCHEMA


def test_selected_hashes_and_deleted_companions_are_auditable(tmp_path):
    cell = tmp_path / "cell"
    records = _recordings(cell, 16, 4)
    original_bytes = {
        record.episode: record.video.read_bytes() for record in records
    }

    manifest = finalize_cell_videos(
        cell,
        records,
        cell_key="open_microwave/teleport",
        successes=16,
        episodes=20,
        paper_success_rate=0.50,
    )

    selected = {row["episode"] for row in manifest["selected"]}
    deleted = {row["episode"] for row in manifest["deleted"]}
    assert selected.isdisjoint(deleted)
    assert selected | deleted == set(range(20))
    for row in manifest["selected"]:
        path = cell / row["video"]
        assert row["video_sha256"] == hashlib.sha256(
            original_bytes[row["episode"]]
        ).hexdigest()
        assert path.is_file()
        assert all((cell / companion).is_file() for companion in row["companions"])
    for row in manifest["deleted"]:
        assert not (cell / row["video"]).exists()
        assert all(not (cell / companion).exists() for companion in row["companions"])


def test_inventory_mismatch_fails_before_deleting_anything(tmp_path):
    cell = tmp_path / "cell"
    records = _recordings(cell, 1, 1)

    with pytest.raises(ValueError, match="every evaluated episode"):
        finalize_cell_videos(
            cell,
            records,
            cell_key="task/static",
            successes=1,
            episodes=3,
            paper_success_rate=0.50,
        )

    assert all(record.video.is_file() for record in records)
    assert all(record.companions[0].is_file() for record in records)
    assert not (cell / "video_selection_manifest.json").exists()


def test_outcome_mismatch_fails_before_deleting_anything(tmp_path):
    cell = tmp_path / "cell"
    records = _recordings(cell, 1, 1)

    with pytest.raises(ValueError, match="outcomes disagree"):
        finalize_cell_videos(
            cell,
            records,
            cell_key="task/static",
            successes=2,
            episodes=2,
            paper_success_rate=0.50,
        )

    assert all(record.video.is_file() for record in records)


def test_artifacts_outside_cell_are_rejected_without_mutation(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    record = EpisodeVideo(episode=0, success=True, video=outside)

    with pytest.raises(ValueError, match="outside the cell directory"):
        finalize_cell_videos(
            cell,
            [record],
            cell_key="task/static",
            successes=1,
            episodes=1,
            paper_success_rate=0.50,
        )

    assert outside.read_bytes() == b"outside"


def test_unserializable_metadata_fails_before_deleting_anything(tmp_path):
    cell = tmp_path / "cell"
    records = _recordings(cell, 1, 1)

    with pytest.raises(TypeError, match="cannot serialize"):
        finalize_cell_videos(
            cell,
            records,
            cell_key="task/static",
            successes=1,
            episodes=2,
            paper_success_rate=0.50,
            cell_metadata={"unsupported": object()},
        )

    assert all(record.video.is_file() for record in records)
    assert all(record.companions[0].is_file() for record in records)


def test_default_seed_and_missing_paper_rate_are_recorded(tmp_path):
    cell = tmp_path / "cell"
    records = _recordings(cell, 8, 2)

    manifest = finalize_cell_videos(
        cell,
        records,
        cell_key="task/static",
        successes=8,
        episodes=10,
        paper_success_rate=None,
    )

    assert manifest["selection"]["seed"] == DEFAULT_SELECTION_SEED
    assert manifest["selection"]["tier"] == "sr_ge_80"
    assert manifest["selection"]["retained"] == {"successes": 3, "failures": 2}
    assert manifest["cell_result"]["paper_success_rate"] is None
    assert manifest["cell_result"]["absolute_difference_percentage_points"] is None


def test_default_output_root_is_v4_and_not_the_replay_directory():
    assert DEFAULT_OUTPUT_ROOT.as_posix().endswith("/results/v4/evaluation_videos")


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_rates_are_rejected(value):
    with pytest.raises(ValueError):
        retention_quota(value, 0.5)
