"""Deterministic retention for videos recorded during formal evaluation.

This module deliberately knows nothing about RLBench, cameras, or ffmpeg.  An
evaluator can stream a lightweight video for every episode with its existing
recorder, register the resulting files as :class:`EpisodeVideo`, and call
:func:`finalize_cell_videos` exactly once after the cell result is known.  The
finalizer verifies that every evaluated episode has a recording before it
deletes anything, retains a fixed-seed outcome-stratified sample, removes all
unselected artifacts, and writes an auditable manifest.

The intended evaluator integration is intentionally small::

    videos = []
    for episode in range(episodes):
        # Run the formal episode while streaming returned RGB observations.
        videos.append(EpisodeVideo(episode, success, video_path, sidecars))
    manifest = finalize_cell_videos(
        cell_dir,
        videos,
        cell_key="open_microwave/static",
        successes=successes,
        episodes=episodes,
        paper_success_rate=0.99,
    )

The module is Python 3.8 compatible because the formal simulator evaluators
run in that environment.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .records import atomic_json, json_ready


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = INTEGRATION_ROOT / "results" / "v4" / "evaluation_videos"
CAPTURE_CONFIG_SCHEMA = "dynamac-v4-formal-evaluation-video-capture-config-v1"
EPISODE_VIDEO_SCHEMA = "dynamac-v4-formal-evaluation-episode-video-v1"
SELECTION_SCHEMA = "dynamac-v4-formal-cell-video-selection-v1"
SELECTION_PROTOCOL_ID = (
    "dynamac-v4-fixed-seed-sha256-ranked-outcome-stratified-retention-v1"
)
DEFAULT_SELECTION_SEED = 2_608_000_000
DEFAULT_MANIFEST_NAME = "video_selection_manifest.json"


@dataclass(frozen=True)
class LightweightCaptureConfig:
    """V4 settings for streamed, observation-granularity episode capture."""

    camera: str = "front"
    fps: int = 12
    capture_every_n_observations: int = 1
    ffmpeg: Path = Path("/usr/bin/ffmpeg")
    preset: str = "veryfast"
    crf: int = 28

    def __post_init__(self) -> None:
        if not isinstance(self.camera, str) or not self.camera:
            raise ValueError("camera must be a non-empty string")
        _validate_count(self.fps, "fps", positive=True)
        _validate_count(
            self.capture_every_n_observations,
            "capture_every_n_observations",
            positive=True,
        )
        _validate_count(self.crf, "crf")
        if self.crf > 51:
            raise ValueError("crf must be in [0, 51]")
        if not isinstance(self.preset, str) or not self.preset:
            raise ValueError("preset must be a non-empty string")
        object.__setattr__(self, "ffmpeg", Path(self.ffmpeg))

    def audit(self) -> Mapping[str, Any]:
        return {
            "schema": CAPTURE_CONFIG_SCHEMA,
            "camera": self.camera,
            "fps": self.fps,
            "capture_every_n_observations": self.capture_every_n_observations,
            "capture_granularity": "returned_high_level_observations",
            "streamed_without_frame_buffer": True,
            "codec": "H.264/yuv420p",
            "preset": self.preset,
            "crf": self.crf,
        }


class StreamingFFmpegWriter:
    """Write RGB frames directly to ffmpeg without retaining them in memory."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        config: LightweightCaptureConfig,
    ) -> None:
        self.path = Path(path)
        self.frames = 0
        self._closed = False
        if width < 1 or height < 1:
            raise ValueError("video dimensions must be positive")
        if width % 2 or height % 2:
            raise ValueError("H.264/yuv420p video dimensions must be even")
        if not config.ffmpeg.is_file():
            raise FileNotFoundError(f"ffmpeg executable is unavailable: {config.ffmpeg}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(config.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(config.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._closed or self._process.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            error = b""
            if self._process.stderr is not None:
                error = self._process.stderr.read()
            raise RuntimeError(
                f"ffmpeg stopped while recording: {error.decode(errors='replace')}"
            ) from exc
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error = b""
        if self._process.stderr is not None:
            error = self._process.stderr.read()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}: {error.decode(errors='replace')}"
            )
        if self.frames < 1 or not self.path.is_file():
            raise RuntimeError("ffmpeg did not produce an episode video")

    def abort(self) -> None:
        if not self._closed and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._closed = True
        self.path.unlink(missing_ok=True)


def _observation_rgb(observation: Any, camera: str) -> np.ndarray:
    key = f"{camera}_rgb"
    perception = getattr(observation, "perception_data", None)
    value = perception.get(key) if isinstance(perception, dict) else None
    if value is None:
        value = getattr(observation, key, None)
    if value is None:
        raise ValueError(f"observation has no {key} frame")
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{key} frame must have shape HxWx3")
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


class ObservationVideoRecorder:
    """Lazily stream one camera from returned observations to an MP4 file."""

    def __init__(
        self,
        path: Path,
        *,
        config: Optional[LightweightCaptureConfig] = None,
        writer_factory: Callable[..., Any] = StreamingFFmpegWriter,
    ) -> None:
        self.path = Path(path)
        self.config = config or LightweightCaptureConfig()
        self.writer_factory = writer_factory
        self.observations_seen = 0
        self.frame_shape = None
        self._writer = None
        self._closed = False

    @property
    def frames(self) -> int:
        return 0 if self._writer is None else int(self._writer.frames)

    def capture(self, observation: Any) -> bool:
        if self._closed:
            raise RuntimeError("episode recorder is closed")
        ordinal = self.observations_seen
        self.observations_seen += 1
        if ordinal % self.config.capture_every_n_observations:
            return False
        frame = _observation_rgb(observation, self.config.camera)
        if self._writer is None:
            self.frame_shape = tuple(int(value) for value in frame.shape)
            self._writer = self.writer_factory(
                self.path,
                width=frame.shape[1],
                height=frame.shape[0],
                config=self.config,
            )
        elif tuple(frame.shape) != self.frame_shape:
            raise ValueError("RGB frame shape changed during an episode")
        self._writer.write(frame)
        return True

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("episode recorder is already closed")
        if self._writer is None:
            raise RuntimeError("episode produced no captured RGB frames")
        self._writer.close()
        self._closed = True
        return {
            "schema": EPISODE_VIDEO_SCHEMA,
            "file": self.path.name,
            "frames": self.frames,
            "frame_shape": list(self.frame_shape),
            "observations_seen": self.observations_seen,
            "capture_config": dict(self.config.audit()),
        }

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.abort()
        else:
            self.path.unlink(missing_ok=True)
        self._closed = True


class RecordingTaskEnvironment:
    """Transparent task-environment proxy that captures returned observations."""

    def __init__(self, task_environment: Any, recorder: ObservationVideoRecorder) -> None:
        self._task_environment = task_environment
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task_environment, name)

    def reset(self) -> Any:
        descriptions, observation = self._task_environment.reset()
        self._recorder.capture(observation)
        return descriptions, observation

    def get_observation(self) -> Any:
        observation = self._task_environment.get_observation()
        self._recorder.capture(observation)
        return observation

    def step(self, action: Any) -> Any:
        observation, reward, terminate = self._task_environment.step(action)
        self._recorder.capture(observation)
        return observation, reward, terminate


@dataclass(frozen=True)
class EpisodeVideo:
    """Files produced for one completed formal-evaluation episode.

    ``companions`` normally contains a JSON sidecar.  Every companion follows
    the video's retention decision so pruning never leaves stale sidecars.
    All paths must be regular files below the cell directory at finalization.
    """

    episode: int
    success: bool
    video: Path
    companions: Tuple[Path, ...] = ()
    episode_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.episode, bool) or not isinstance(self.episode, int):
            raise TypeError("episode must be an integer")
        if self.episode < 0:
            raise ValueError("episode must be non-negative")
        if not isinstance(self.success, bool):
            raise TypeError("success must be boolean")
        if self.episode_seed is not None and (
            isinstance(self.episode_seed, bool)
            or not isinstance(self.episode_seed, int)
            or self.episode_seed < 0
        ):
            raise ValueError("episode_seed must be a non-negative integer or None")
        object.__setattr__(self, "video", Path(self.video))
        object.__setattr__(
            self,
            "companions",
            tuple(Path(path) for path in self.companions),
        )


@dataclass(frozen=True)
class RetentionQuota:
    """Outcome-specific video counts selected by the declared SR tier."""

    tier: str
    successes: int
    failures: int
    paper_close_enough_for_zero: bool


def _rate(value: Any, name: str, *, optional: bool = False) -> Optional[float]:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def retention_quota(
    success_rate: float,
    paper_success_rate: Optional[float],
) -> RetentionQuota:
    """Return the fixed retention tier for one completed evaluation cell.

    Rates use the repository's native ``[0, 1]`` convention.  The two-point
    comparison is strict: an absolute difference of exactly 2 percentage
    points does *not* enter the zero-video tier.
    """

    observed = _rate(success_rate, "success_rate")
    paper = _rate(
        paper_success_rate,
        "paper_success_rate",
        optional=True,
    )
    # Decimal(str(...)) makes the strict two-point boundary insensitive to a
    # binary-float representation such as 0.82 - 0.80 == 0.0199999....
    difference = (
        None
        if paper is None
        else abs(Decimal(str(observed)) - Decimal(str(paper)))
    )
    close_to_paper = bool(
        observed >= 0.80
        and difference is not None
        and difference < Decimal("0.02")
    )
    if close_to_paper:
        return RetentionQuota("sr_ge_80_paper_diff_lt_2pp", 0, 0, True)
    if observed >= 0.80:
        return RetentionQuota("sr_ge_80", 3, 3, False)
    if observed >= 0.50:
        return RetentionQuota("sr_50_to_lt_80", 5, 10, False)
    return RetentionQuota("sr_lt_50", 5, 20, False)


def _validate_count(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_manifest_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("manifest_name must be a non-empty filename")
    path = Path(name)
    if path.name != name or name in {".", ".."}:
        raise ValueError("manifest_name must not contain a directory")
    return name


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _relative_regular_file(cell_dir: Path, value: Path, label: str) -> Tuple[Path, str]:
    unresolved = value if value.is_absolute() else cell_dir / value
    if unresolved.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {value}")
    resolved = unresolved.resolve()
    if not _inside(cell_dir, resolved):
        raise ValueError(f"{label} is outside the cell directory: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {value}")
    return resolved, resolved.relative_to(cell_dir).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rank(
    *,
    selection_seed: int,
    cell_key: str,
    success: bool,
    episode: int,
) -> str:
    category = "success" if success else "failure"
    payload = (
        f"{SELECTION_PROTOCOL_ID}\n{selection_seed}\n{cell_key}\n"
        f"{category}\n{episode}\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select(
    records: Sequence[EpisodeVideo],
    *,
    count: int,
    selection_seed: int,
    cell_key: str,
    success: bool,
) -> Tuple[EpisodeVideo, ...]:
    candidates = [record for record in records if record.success is success]
    ranked = sorted(
        candidates,
        key=lambda record: (
            _rank(
                selection_seed=selection_seed,
                cell_key=cell_key,
                success=success,
                episode=record.episode,
            ),
            record.episode,
        ),
    )
    # min() implements the explicit "missing class keeps everything available"
    # rule without borrowing unused quota from the other outcome class.
    return tuple(ranked[: min(count, len(ranked))])


def _normalize_recordings(
    recordings: Iterable[EpisodeVideo],
    *,
    cell_dir: Path,
    manifest_path: Path,
) -> Tuple[Tuple[EpisodeVideo, ...], Mapping[int, Mapping[str, Any]]]:
    normalized = tuple(recordings)
    if any(not isinstance(record, EpisodeVideo) for record in normalized):
        raise TypeError("recordings must contain only EpisodeVideo values")

    by_episode = {}
    claimed_paths = set()
    resolved = {}
    for record in normalized:
        if record.episode in by_episode:
            raise ValueError(f"duplicate episode recording: {record.episode}")
        by_episode[record.episode] = record

        video_path, video_relative = _relative_regular_file(
            cell_dir,
            record.video,
            f"episode {record.episode} video",
        )
        artifacts = [(video_path, video_relative)]
        for index, companion in enumerate(record.companions):
            artifacts.append(
                _relative_regular_file(
                    cell_dir,
                    companion,
                    f"episode {record.episode} companion {index}",
                )
            )
        for absolute, relative in artifacts:
            if absolute == manifest_path:
                raise ValueError("an episode artifact conflicts with the selection manifest")
            if absolute in claimed_paths:
                raise ValueError(f"episode artifacts reuse the same file: {relative}")
            claimed_paths.add(absolute)
        resolved[record.episode] = {
            "video_path": video_path,
            "video": video_relative,
            "companions": [relative for _, relative in artifacts[1:]],
            "artifact_paths": [absolute for absolute, _ in artifacts],
        }
    return normalized, resolved


def finalize_cell_videos(
    cell_dir: Path,
    recordings: Iterable[EpisodeVideo],
    *,
    cell_key: str,
    successes: int,
    episodes: int,
    paper_success_rate: Optional[float],
    selection_seed: int = DEFAULT_SELECTION_SEED,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    cell_metadata: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Prune one completed cell's episode videos and write its manifest.

    The function fails before deletion unless the inventory has exactly one
    recording for every evaluated episode and its success labels agree with
    the cell aggregate.  This makes a missing encoder output visible rather
    than silently biasing the retained sample.
    """

    if not isinstance(cell_key, str) or not cell_key.strip():
        raise ValueError("cell_key must be a non-empty string")
    successes = _validate_count(successes, "successes")
    episodes = _validate_count(episodes, "episodes", positive=True)
    selection_seed = _validate_count(selection_seed, "selection_seed")
    if successes > episodes:
        raise ValueError("successes must not exceed episodes")
    manifest_name = _validate_manifest_name(manifest_name)
    paper = _rate(
        paper_success_rate,
        "paper_success_rate",
        optional=True,
    )
    if cell_metadata is not None and not isinstance(cell_metadata, Mapping):
        raise TypeError("cell_metadata must be a mapping or None")

    cell_root = Path(cell_dir).resolve()
    if not cell_root.is_dir():
        raise FileNotFoundError(f"cell directory does not exist: {cell_dir}")
    manifest_path = cell_root / manifest_name
    normalized, resolved = _normalize_recordings(
        recordings,
        cell_dir=cell_root,
        manifest_path=manifest_path,
    )
    if len(normalized) != episodes:
        raise ValueError(
            "recording inventory must contain every evaluated episode: "
            f"expected {episodes}, found {len(normalized)}"
        )
    recorded_successes = sum(record.success for record in normalized)
    if recorded_successes != successes:
        raise ValueError(
            "recording outcomes disagree with the cell aggregate: "
            f"expected {successes} successes, found {recorded_successes}"
        )

    success_rate = successes / episodes
    quota = retention_quota(success_rate, paper)
    selected_successes = _select(
        normalized,
        count=quota.successes,
        selection_seed=selection_seed,
        cell_key=cell_key,
        success=True,
    )
    selected_failures = _select(
        normalized,
        count=quota.failures,
        selection_seed=selection_seed,
        cell_key=cell_key,
        success=False,
    )
    selected_episodes = {
        record.episode for record in (*selected_successes, *selected_failures)
    }

    selected_rows = []
    deleted_rows = []
    for record in sorted(normalized, key=lambda item: item.episode):
        paths = resolved[record.episode]
        base = {
            "episode": record.episode,
            "episode_seed": record.episode_seed,
            "outcome": "success" if record.success else "failure",
            "video": paths["video"],
            "companions": list(paths["companions"]),
        }
        if record.episode in selected_episodes:
            selected_rows.append(
                {
                    **base,
                    "video_sha256": _sha256(paths["video_path"]),
                    "video_bytes": paths["video_path"].stat().st_size,
                }
            )
        else:
            deleted_rows.append(base)

    manifest = {
        "schema": SELECTION_SCHEMA,
        "cell_key": cell_key,
        "cell_metadata": dict(cell_metadata or {}),
        "cell_result": {
            "successes": successes,
            "episodes": episodes,
            "success_rate": success_rate,
            "paper_success_rate": paper,
            "absolute_difference_percentage_points": (
                None if paper is None else abs(success_rate - paper) * 100.0
            ),
        },
        "selection": {
            "protocol_id": SELECTION_PROTOCOL_ID,
            "seed": selection_seed,
            "sampler": "ascending SHA-256 rank within success/failure strata",
            "rank_identity": [
                "protocol_id",
                "selection_seed",
                "cell_key",
                "outcome",
                "episode",
            ],
            "thresholds": {
                "paper_close_zero_video_rule": (
                    "success_rate >= 0.80 and absolute paper difference < 0.02"
                ),
                "sr_ge_80_otherwise": {"successes": 3, "failures": 3},
                "sr_50_to_lt_80": {"successes": 5, "failures": 10},
                "sr_lt_50": {"successes": 5, "failures": 20},
                "class_shortfall": "retain every available item; do not transfer quota",
            },
            "tier": quota.tier,
            "requested": {
                "successes": quota.successes,
                "failures": quota.failures,
            },
            "available": {
                "successes": recorded_successes,
                "failures": episodes - recorded_successes,
            },
            "retained": {
                "successes": len(selected_successes),
                "failures": len(selected_failures),
            },
            "paper_close_enough_for_zero": quota.paper_close_enough_for_zero,
        },
        "selected": selected_rows,
        "deleted": deleted_rows,
        "all_episodes_recorded_before_selection": True,
        "unselected_artifacts_deleted": True,
    }
    # Convert caller metadata before deletion so unsupported values fail while
    # the complete recording inventory is still intact.
    manifest = json_ready(manifest)

    # Validation and selected-video hashing above are deliberately complete
    # before the first destructive operation.
    for row in deleted_rows:
        record = resolved[row["episode"]]
        for path in record["artifact_paths"]:
            path.unlink()
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = [
    "CAPTURE_CONFIG_SCHEMA",
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SELECTION_SEED",
    "EPISODE_VIDEO_SCHEMA",
    "EpisodeVideo",
    "LightweightCaptureConfig",
    "ObservationVideoRecorder",
    "RecordingTaskEnvironment",
    "RetentionQuota",
    "SELECTION_PROTOCOL_ID",
    "SELECTION_SCHEMA",
    "StreamingFFmpegWriter",
    "finalize_cell_videos",
    "retention_quota",
]
