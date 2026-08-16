"""Replay and record audited failures without changing policy semantics.

Existing evaluation files contain low-dimensional episode records only.  This
tool replays explicitly selected failed episodes with the same evaluator,
model, seed, variation and controller, while enabling RGB observations solely
for video capture.  A replay is published only when it is still a failure and
the loaded model identity exactly matches the source evaluation.

The simulator side remains Python 3.8 compatible.  Frames are passed directly
to the system ``ffmpeg`` executable, so OpenCV and imageio are not required in
the pinned RLBench environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import direct_evaluate, unimanual_evaluate
from .records import atomic_json

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = INTEGRATION_ROOT / "models" / "v1"
DEFAULT_OUTPUT_ROOT = (
    INTEGRATION_ROOT / "results" / "failure_videos" / "v1"
)
DEFAULT_SOURCE_RESULTS = {
    "bimanual_handover_item": (
        INTEGRATION_ROOT
        / "results"
        / "v1"
        / "table_ii"
        / "bimanual_handover_item_static_seed0_n200_h1000.json"
    ),
    "bimanual_sweep_to_dustpan": (
        INTEGRATION_ROOT
        / "results"
        / "v1"
        / "table_ii"
        / "bimanual_sweep_to_dustpan_static_seed0_n200_h1000.json"
    ),
    "wipe_desk": (
        INTEGRATION_ROOT
        / "results"
        / "v1"
        / "table_i"
        / "wipe_desk_static_variation0_seed0_n200_h1000.json"
    ),
}
BIMANUAL_TASKS = {
    "bimanual_handover_item",
    "bimanual_sweep_to_dustpan",
}
DEFAULT_CAMERAS = ("front", "overhead")
SCHEMA = "dynamac-confirmed-failure-videos-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_episode_indices(values):
    if not values:
        raise ValueError("at least one --episode is required")
    result = []
    seen = set()
    for value in values:
        episode = int(value)
        if episode < 0:
            raise ValueError("episode indices must be non-negative")
        if episode in seen:
            raise ValueError(f"duplicate episode index: {episode}")
        seen.add(episode)
        result.append(episode)
    return tuple(result)


def _load_source(path, task, episode_indices):
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("task") != task:
        raise ValueError(
            f"source task {payload.get('task')!r} does not match requested {task!r}"
        )
    if payload.get("scenario") != "static":
        raise ValueError("failure-video replay currently accepts static results only")
    if not isinstance(payload.get("seed"), int) or payload["seed"] < 0:
        raise ValueError("source evaluation has an invalid seed")
    if not isinstance(payload.get("horizon"), int) or payload["horizon"] < 1:
        raise ValueError("source evaluation has an invalid horizon")
    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, dict) or not model_identity.get(
        "manifest_authenticated"
    ):
        raise ValueError("source evaluation is not bound to an authenticated model")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("source evaluation has no episode rows")
    by_episode = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("episode"), int):
            raise ValueError("source evaluation contains an invalid episode row")
        if row["episode"] in by_episode:
            raise ValueError("source evaluation contains duplicate episode rows")
        by_episode[row["episode"]] = row
    selected = {}
    for episode in episode_indices:
        if episode not in by_episode:
            raise ValueError(f"episode {episode} is absent from the source evaluation")
        row = by_episode[episode]
        if bool(row.get("success")):
            raise ValueError(f"episode {episode} was successful in the source evaluation")
        selected[episode] = row
    return payload, selected, hashlib.sha256(raw).hexdigest(), source_path


def _rgb_from_observation(observation, camera):
    perception = getattr(observation, "perception_data", None)
    if not isinstance(perception, dict):
        return None
    value = perception.get(f"{camera}_rgb")
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{camera} RGB frame must have shape HxWx3")
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _compose_frame(observation, cameras):
    frames = []
    used = []
    for camera in cameras:
        frame = _rgb_from_observation(observation, camera)
        if frame is not None:
            frames.append(frame)
            used.append(camera)
    if not frames:
        raise ValueError("observation contains no requested RGB camera")
    height = frames[0].shape[0]
    if any(frame.shape[0] != height for frame in frames):
        raise ValueError("camera frames must have equal heights")
    return np.concatenate(frames, axis=1), tuple(used)


class _FFmpegWriter:
    def __init__(self, path, *, width, height, fps, ffmpeg):
        self.path = Path(path)
        self.frames = 0
        command = [
            str(ffmpeg),
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
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
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

    def write(self, frame):
        if self._process.stdin is None:
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

    def close(self):
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error = b""
        if self._process.stderr is not None:
            error = self._process.stderr.read()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}: "
                f"{error.decode(errors='replace')}"
            )
        if self.frames < 1 or not self.path.is_file():
            raise RuntimeError("ffmpeg did not produce a video")

    def abort(self):
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()


class ObservationRecorder:
    """Lazy RGB recorder with a deterministic front-only fallback."""

    def __init__(self, path, *, cameras, fps, ffmpeg, writer_class=_FFmpegWriter):
        self.path = Path(path)
        self.requested_cameras = tuple(cameras)
        self.fps = int(fps)
        self.ffmpeg = Path(ffmpeg)
        self.writer_class = writer_class
        self.used_cameras = None
        self.frame_shape = None
        self._writer = None

    @property
    def frames(self):
        return 0 if self._writer is None else self._writer.frames

    def capture(self, observation):
        cameras = self.requested_cameras if self.used_cameras is None else self.used_cameras
        frame, used = _compose_frame(observation, cameras)
        if self.used_cameras is None:
            # Front is the stable common view.  Overhead is included when the
            # scene exposes it; missing optional views do not prevent capture.
            if "front" not in used:
                raise ValueError("front RGB camera is unavailable")
            self.used_cameras = used
            self.frame_shape = tuple(int(value) for value in frame.shape)
            self._writer = self.writer_class(
                self.path,
                width=frame.shape[1],
                height=frame.shape[0],
                fps=self.fps,
                ffmpeg=self.ffmpeg,
            )
        elif tuple(frame.shape) != self.frame_shape:
            raise ValueError("RGB frame shape changed during an episode")
        self._writer.write(frame)

    def close(self):
        if self._writer is None:
            raise RuntimeError("episode produced no RGB frames")
        self._writer.close()

    def abort(self):
        if self._writer is not None:
            self._writer.abort()


class RecordingTaskEnvironment:
    """Transparent TaskEnvironment proxy that observes returned snapshots."""

    def __init__(self, task_environment, recorder):
        self._task_environment = task_environment
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._task_environment, name)

    def reset(self):
        descriptions, observation = self._task_environment.reset()
        self._recorder.capture(observation)
        return descriptions, observation

    def get_observation(self):
        observation = self._task_environment.get_observation()
        self._recorder.capture(observation)
        return observation

    def step(self, action):
        observation, reward, terminate = self._task_environment.step(action)
        self._recorder.capture(observation)
        return observation, reward, terminate


def _validate_replay(original, replay, source_identity, loaded_identity):
    if loaded_identity != source_identity:
        raise RuntimeError("loaded model identity differs from the source evaluation")
    if bool(original.get("success")):
        raise ValueError("the selected source row is not a failure")
    if bool(replay.get("success")):
        raise RuntimeError(
            f"episode {original.get('episode')} succeeded during replay; "
            "no failure video will be published"
        )
    if replay.get("episode") != original.get("episode"):
        raise RuntimeError("replay episode identity differs from the source row")
    return {
        "replay_confirmed_failure": True,
        "same_reason": replay.get("reason") == original.get("reason"),
        "same_steps": replay.get("steps") == original.get("steps"),
    }


def _observation_config(cameras, resolution):
    from rlbench.observation_config import CameraConfig, ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.gripper_open = True
    config.gripper_pose = True
    config.task_low_dim_state = True
    config.camera_configs = {
        camera: CameraConfig(
            rgb=True,
            depth=False,
            point_cloud=False,
            mask=False,
            image_size=tuple(resolution),
        )
        for camera in cameras
    }
    return config


def _run_selected_episode(task_environment, worker, task, source, episode):
    if task in BIMANUAL_TASKS:
        return direct_evaluate._run_episode(
            task_environment,
            worker,
            episode,
            source["seed"],
            source["horizon"],
            scenario="static",
            scenario_reference_steps=worker.policy_steps,
        )
    run_args = SimpleNamespace(
        seed=source["seed"],
        variation=source.get("variation", 0),
        horizon=source["horizon"],
        scenario="static",
        trigger_fraction=1.0 / 3.0,
        smooth_steps=10,
        intervention_attempts=20,
    )
    return unimanual_evaluate._run_episode(
        task_environment, worker, run_args, episode
    )


def _runtime_components(args):
    from rlbench.environment import Environment

    if args.task in BIMANUAL_TASKS:
        module_name, class_name = direct_evaluate.TASKS[args.task]
        action_mode = direct_evaluate._make_action_mode()
        environment = Environment(
            action_mode=action_mode,
            obs_config=_observation_config(args.cameras, args.resolution),
            headless=args.headless,
            robot_setup="dual_panda",
        )
        worker_class = direct_evaluate.PolicyProcess
        default_python = direct_evaluate.DEFAULT_POLICY_PYTHON
    else:
        module_name, class_name = unimanual_evaluate.TASKS[args.task]
        action_mode = unimanual_evaluate._make_action_mode()
        environment = Environment(
            action_mode=action_mode,
            obs_config=_observation_config(args.cameras, args.resolution),
            headless=args.headless,
        )
        worker_class = unimanual_evaluate.PolicyProcess
        default_python = unimanual_evaluate.DEFAULT_POLICY_PYTHON
    task_class = getattr(importlib.import_module(module_name), class_name)
    policy_python = args.policy_python or default_python
    worker = worker_class(
        policy_python,
        args.task,
        args.models_dir,
        timeout=args.policy_timeout,
    )
    return environment, worker, task_class


def record(args):
    episodes = _normalize_episode_indices(args.episode)
    minimum_confirmed = (
        len(episodes) if args.minimum_confirmed is None else args.minimum_confirmed
    )
    if minimum_confirmed < 1 or minimum_confirmed > len(episodes):
        raise ValueError(
            "--minimum-confirmed must be between one and the number of candidates"
        )
    source_path = args.source_result or DEFAULT_SOURCE_RESULTS[args.task]
    source, originals, source_sha, resolved_source = _load_source(
        source_path, args.task, episodes
    )
    target = Path(args.output_root).resolve() / args.task
    if target.exists():
        raise FileExistsError(f"refusing to overwrite failure videos: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.task}.staging-", dir=str(target.parent))
    )
    environment = worker = None
    launched = False
    manifest_rows = []
    attempts = []
    try:
        environment, worker, task_class = _runtime_components(args)
        if worker.model_identity != source["model_identity"]:
            raise RuntimeError("loaded model identity differs from the source evaluation")
        environment.launch()
        launched = True
        task_environment = environment.get_task(task_class)
        for episode in episodes:
            episode_seed = source["seed"] + episode
            stem = f"episode_{episode:03d}_seed_{episode_seed:03d}"
            video_path = staging / f"{stem}.mp4"
            recorder = ObservationRecorder(
                video_path,
                cameras=args.cameras,
                fps=args.fps,
                ffmpeg=args.ffmpeg,
            )
            proxy = RecordingTaskEnvironment(task_environment, recorder)
            try:
                replay = _run_selected_episode(
                    proxy, worker, args.task, source, episode
                )
                recorder.close()
            except Exception:
                recorder.abort()
                raise
            attempt = {
                "episode": episode,
                "episode_seed": episode_seed,
                "original_result": originals[episode],
                "replay_result": replay,
                "replay_confirmed_failure": not bool(replay.get("success")),
            }
            attempts.append(attempt)
            if bool(replay.get("success")):
                video_path.unlink()
                print(
                    f"{args.task} episode {episode}: replay succeeded; "
                    "discarded from the failure-video set",
                    flush=True,
                )
                continue
            confirmation = _validate_replay(
                originals[episode],
                replay,
                source["model_identity"],
                worker.model_identity,
            )
            sidecar = {
                "schema": SCHEMA,
                "task": args.task,
                "scenario": "static",
                "episode": episode,
                "episode_seed": episode_seed,
                "variation": (
                    episode % task_environment.variation_count()
                    if args.task in BIMANUAL_TASKS
                    else source.get("variation", 0)
                ),
                "source_result": str(resolved_source),
                "source_result_sha256": source_sha,
                "evaluation_protocol_id": source.get("evaluation_protocol_id"),
                "model_identity": worker.model_identity,
                "original_result": originals[episode],
                "replay_result": replay,
                **confirmation,
                "video": {
                    "file": video_path.name,
                    "sha256": _sha256(video_path),
                    "requested_cameras": list(args.cameras),
                    "used_cameras": list(recorder.used_cameras),
                    "layout": "horizontal_left_to_right",
                    "source_resolution": list(args.resolution),
                    "encoded_resolution": [
                        recorder.frame_shape[1],
                        recorder.frame_shape[0],
                    ],
                    "fps": args.fps,
                    "frames": recorder.frames,
                    "codec": "H.264/yuv420p",
                },
                "policy_inputs_unchanged": True,
                "capture_granularity": "returned_high_level_observations",
            }
            sidecar_path = staging / f"{stem}.json"
            atomic_json(sidecar_path, sidecar)
            manifest_rows.append(
                {
                    "episode": episode,
                    "episode_seed": episode_seed,
                    "video": video_path.name,
                    "video_sha256": sidecar["video"]["sha256"],
                    "metadata": sidecar_path.name,
                    "replay_confirmed_failure": True,
                }
            )
            print(
                f"{args.task} episode {episode}: confirmed failure, "
                f"wrote {recorder.frames} frames",
                flush=True,
            )
            if len(manifest_rows) == minimum_confirmed:
                break
        if len(manifest_rows) < minimum_confirmed:
            raise RuntimeError(
                f"only {len(manifest_rows)} of {minimum_confirmed} required "
                "failure replays were confirmed; nothing will be published"
            )
        manifest = {
            "schema": SCHEMA,
            "task": args.task,
            "scenario": "static",
            "source_result": str(resolved_source),
            "source_result_sha256": source_sha,
            "model_identity": worker.model_identity,
            "evaluation_protocol_id": source.get("evaluation_protocol_id"),
            "policy_controller_semantics": "identical_to_source_evaluator",
            "policy_inputs_unchanged": True,
            "required_confirmed_failures": minimum_confirmed,
            "attempts": attempts,
            "episodes": manifest_rows,
        }
        atomic_json(staging / "manifest.json", manifest)
        os.replace(str(staging), str(target))
        print(f"published {target}", flush=True)
        return target
    finally:
        if worker is not None:
            worker.close()
        if launched:
            environment.shutdown()
        if staging.exists():
            shutil.rmtree(staging)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(DEFAULT_SOURCE_RESULTS))
    parser.add_argument(
        "--episode",
        action="append",
        type=int,
        required=True,
        help="Exact source-evaluation episode index; repeat for multiple videos.",
    )
    parser.add_argument(
        "--minimum-confirmed",
        type=int,
        default=None,
        help=(
            "Publish after this many candidates replay as failures. By default, "
            "every selected episode must remain a failure."
        ),
    )
    parser.add_argument("--source-result", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-python", type=Path, default=None)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 360),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("front", "overhead"),
        default=DEFAULT_CAMERAS,
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fps < 1 or any(value < 1 for value in args.resolution):
        raise ValueError("fps and resolution must be positive")
    if args.policy_timeout <= 0.0:
        raise ValueError("policy timeout must be positive")
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg executable is unavailable: {args.ffmpeg}")
    return 0 if record(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
