from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.rlbench.rlbench_dynamac import (
    direct_evaluate,
    table_iii_coordination,
    unimanual_evaluate,
)
from integrations.rlbench.rlbench_dynamac.evaluation_videos import (
    DEFAULT_MANIFEST_NAME,
    RecordingTaskEnvironment,
)


class _Recorder:
    instances = []

    def __init__(self, path, *, config) -> None:
        self.path = Path(path)
        self.config = config
        self.observations = []
        self.aborted = False
        self.__class__.instances.append(self)

    def capture(self, observation):
        self.observations.append(observation)
        return True

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"streamed-video")
        return {
            "file": self.path.name,
            "frames": len(self.observations),
            "observations_seen": len(self.observations),
            "capture_config": dict(self.config.audit()),
        }

    def abort(self):
        self.aborted = True
        self.path.unlink(missing_ok=True)


class _TaskEnvironment:
    marker = "original-environment"

    def __init__(self, returned_observation) -> None:
        self.returned_observation = returned_observation

    def step(self, action):
        assert action == "formal-action"
        return self.returned_observation, 0.0, False


def test_actual_episode_run_receives_recording_proxy_and_writes_sidecar(tmp_path):
    _Recorder.instances.clear()
    initial = object()
    returned = object()
    task_environment = _TaskEnvironment(returned)

    def run_episode(formal_environment):
        assert isinstance(formal_environment, RecordingTaskEnvironment)
        assert formal_environment.marker == "original-environment"
        observation, reward, terminate = formal_environment.step("formal-action")
        assert observation is returned
        assert reward == 0.0
        assert terminate is False
        return {
            "episode": 4,
            "success": True,
            "reason": "success",
            "steps": 1,
            "invalid_actions": 0,
        }

    result, recording = direct_evaluate._run_episode_with_optional_v4_video(
        task_environment,
        initial,
        enabled=True,
        cell_dir=tmp_path,
        cell_key="task/static",
        episode=4,
        episode_seed=2_608_000_004,
        run_episode=run_episode,
        recorder_factory=_Recorder,
    )

    assert result["success"] is True
    assert recording.episode == 4
    assert recording.success is True
    assert recording.video.read_bytes() == b"streamed-video"
    assert _Recorder.instances[0].observations == [initial, returned]
    sidecar = json.loads(recording.companions[0].read_text(encoding="utf-8"))
    assert sidecar["formal_episode_completed"] is True
    assert sidecar["success"] is True
    assert sidecar["capture"]["observations_seen"] == 2


def test_mid_episode_failure_keeps_unpruned_video_without_selection_manifest(tmp_path):
    _Recorder.instances.clear()
    initial = object()
    returned = object()

    def fail_after_one_step(formal_environment):
        formal_environment.step("formal-action")
        raise RuntimeError("formal rollout stopped")

    with pytest.raises(RuntimeError, match="formal rollout stopped"):
        direct_evaluate._run_episode_with_optional_v4_video(
            _TaskEnvironment(returned),
            initial,
            enabled=True,
            cell_dir=tmp_path,
            cell_key="task/teleport",
            episode=0,
            episode_seed=2_608_000_000,
            run_episode=fail_after_one_step,
            recorder_factory=_Recorder,
        )

    assert (tmp_path / "episode_000_seed_2608000000.mp4").is_file()
    sidecar_path = tmp_path / "episode_000_seed_2608000000.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["formal_episode_completed"] is False
    assert sidecar["error_type"] == "RuntimeError"
    assert not (tmp_path / DEFAULT_MANIFEST_NAME).exists()


def test_v3_disabled_path_uses_original_environment_and_never_constructs_recorder(
    tmp_path,
):
    task_environment = _TaskEnvironment(object())
    calls = []

    class ForbiddenRecorder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("V3 must not construct a recorder")

    def run_episode(environment):
        calls.append(environment)
        return {"success": False, "reason": "horizon"}

    result, recording = direct_evaluate._run_episode_with_optional_v4_video(
        task_environment,
        object(),
        enabled=False,
        cell_dir=None,
        cell_key=None,
        episode=0,
        episode_seed=0,
        run_episode=run_episode,
        recorder_factory=ForbiddenRecorder,
    )

    assert result == {"success": False, "reason": "horizon"}
    assert recording is None
    assert calls == [task_environment]
    assert not tmp_path.joinpath(DEFAULT_MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    ("parser", "base_argv"),
    (
        (
            direct_evaluate.build_parser,
            ["--task", "bimanual_lift_tray"],
        ),
        (
            unimanual_evaluate.build_parser,
            ["--task", "open_microwave", "--scenario", "static"],
        ),
        (
            table_iii_coordination.build_parser,
            ["evaluate", "--arm", "left", "--output", "result.json"],
        ),
    ),
)
def test_all_formal_clis_gate_capture_strictly_to_v4(parser, base_argv):
    default = parser().parse_args(base_argv)
    assert default.release == "v3"
    assert default.record_v4_evaluation_videos is False
    assert direct_evaluate._v4_video_capture_enabled(default) is False

    v4_without_capture = parser().parse_args([*base_argv, "--release", "v4"])
    with pytest.raises(ValueError, match="requires episode video capture"):
        direct_evaluate._v4_video_capture_enabled(v4_without_capture)

    capture_on_v3 = parser().parse_args(
        [*base_argv, "--record-v4-evaluation-videos"]
    )
    with pytest.raises(ValueError, match="requires --release v4"):
        direct_evaluate._v4_video_capture_enabled(capture_on_v3)

    v4 = parser().parse_args(
        [*base_argv, "--release", "v4", "--record-v4-evaluation-videos"]
    )
    assert direct_evaluate._v4_video_capture_enabled(v4) is True


def test_video_selection_finishes_before_formal_result_commit(monkeypatch, tmp_path):
    events = []
    output = tmp_path / "result.json"
    payload = {"successes": 82, "episodes": 100}
    selection = {
        "path": "selection.json",
        "sha256": "a" * 64,
        "schema": "selection-v1",
    }

    def finalize():
        events.append("finalize")
        return selection

    def commit(path, value):
        events.append("commit")
        assert path == output
        assert value["evaluation_video_capture"]["selection_manifest"] == selection

    monkeypatch.setattr(direct_evaluate, "atomic_json", commit)
    direct_evaluate._commit_formal_result_with_optional_v4_videos(
        output,
        payload,
        enabled=True,
        capture_metadata={"paper_success_rate": 0.82},
        finalize_videos=finalize,
    )

    assert events == ["finalize", "commit"]
    assert payload["evaluation_video_capture"][
        "formal_result_committed_after_video_selection"
    ] is True


def test_video_finalization_failure_never_commits_formal_result(monkeypatch, tmp_path):
    committed = []
    monkeypatch.setattr(
        direct_evaluate,
        "atomic_json",
        lambda *args, **kwargs: committed.append((args, kwargs)),
    )

    def fail():
        raise RuntimeError("manifest failed")

    with pytest.raises(RuntimeError, match="manifest failed"):
        direct_evaluate._commit_formal_result_with_optional_v4_videos(
            tmp_path / "result.json",
            {"results": []},
            enabled=True,
            capture_metadata={"paper_success_rate": 1.0},
            finalize_videos=fail,
        )

    assert committed == []


def test_requested_paper_targets_drive_cell_quota_inputs():
    assert direct_evaluate.V4_VIDEO_PAPER_TARGETS[
        ("bimanual_put_bottle_in_fridge", "static")
    ] == pytest.approx(0.82)
    assert direct_evaluate.V4_VIDEO_PAPER_TARGETS[
        ("bimanual_put_bottle_in_fridge", "teleport")
    ] == pytest.approx(0.82)
    assert direct_evaluate.V4_VIDEO_PAPER_TARGETS[
        ("bimanual_lift_tray", "static")
    ] == pytest.approx(1.0)
    assert direct_evaluate.V4_VIDEO_PAPER_TARGETS[
        ("bimanual_lift_tray", "teleport")
    ] == pytest.approx(1.0)
    assert table_iii_coordination.V4_VIDEO_PAPER_TARGET == pytest.approx(0.97)
