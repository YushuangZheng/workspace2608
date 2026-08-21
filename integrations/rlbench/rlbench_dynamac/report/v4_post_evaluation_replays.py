"""Plan and launch outcome-stratified V4 replay videos after evaluation.

Formal evaluation stays low-dimensional and immutable.  This launcher first
admits every completed result through :mod:`v4_formal_launch`, derives the
success/failure retention quota and exact SHA-256 candidate order used by
``evaluation_videos``, then delegates each outcome to the existing
``failure_videos`` CLI.  The delegated recorder owns all simulator execution,
outcome-drift handling, two-camera encoding, and atomic directory publication.

The default ``plan`` command is read-only and never starts RLBench.  ``execute``
uses at most eight reusable GPU/Xvfb lanes and writes only below
``results/v4/replay_video``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch
from integrations.rlbench.rlbench_dynamac.report import (
    evaluation_videos,
    failure_videos,
)
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json

from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
RESULTS_ROOT = INTEGRATION_ROOT / "results" / "v4"
REPLAY_ROOT = RESULTS_ROOT / "replay_video"
LAUNCH_ROOT = REPLAY_ROOT / "_launch" / "post_evaluation"
DEFAULT_SIM_PYTHON = v4_formal_launch.DEFAULT_SIM_PYTHON
DEFAULT_POLICY_PYTHON = v4_formal_launch.DEFAULT_POLICY_PYTHON
DEFAULT_XVFB_RUN = v4_formal_launch.DEFAULT_XVFB_RUN
DEFAULT_FFMPEG = Path("/usr/bin/ffmpeg")
DEFAULT_GPUS = tuple(v4_formal_launch.DEFAULT_GPUS[:8])
DEFAULT_CAMERAS = ("front", "overhead")
DEFAULT_FPS = 12
DEFAULT_RESOLUTION = (640, 360)
DEFAULT_MAX_REPLAY_ATTEMPTS_PER_EPISODE = 1
MAX_GPU_LANES = 8
XVFB_SERVER_BASE = 110
PLAN_SCHEMA = "dynamac-v4-post-evaluation-replay-plan-v1"
LAUNCH_SCHEMA = "dynamac-v4-post-evaluation-replay-launch-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _safe_output_key(cell_name: str, outcome: str) -> str:
    value = f"{cell_name}_{outcome}".lower()
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value
    ):
        raise ValueError(f"formal cell name cannot form a replay key: {cell_name!r}")
    return value


@dataclass(frozen=True)
class ReplayJob:
    """One isolated invocation of ``failure_videos`` for one outcome class."""

    cell_name: str
    cell_id: str
    source_task: str
    scenario: str
    policy_task: str
    outcome: str
    quota_tier: str
    requested_quota: int
    required_confirmed: int
    candidate_episodes: tuple[int, ...]
    source_result: Path
    source_result_sha256: str
    models_dir: Path
    output_key: str

    @property
    def target(self) -> Path:
        return REPLAY_ROOT / self.output_key

    def recorder_command(
        self,
        *,
        sim_python: Path,
        policy_python: Path,
        ffmpeg: Path,
        fps: int,
        resolution: tuple[int, int],
        maximum_attempts: int,
        overwrite: bool,
    ) -> tuple[str, ...]:
        command = [
            str(sim_python),
            "-m",
            "integrations.rlbench.rlbench_dynamac.report.failure_videos",
            "--task",
            self.policy_task,
        ]
        for episode in self.candidate_episodes:
            command.extend(("--episode", str(episode)))
        command.extend(
            (
                "--expected-outcome",
                self.outcome,
                "--minimum-confirmed",
                str(self.required_confirmed),
                "--max-replay-attempts-per-episode",
                str(maximum_attempts),
                "--source-result",
                str(self.source_result),
                "--models-dir",
                str(self.models_dir),
                "--output-root",
                str(REPLAY_ROOT),
                "--output-key",
                self.output_key,
                "--overwrite" if overwrite else "--no-overwrite",
                "--policy-python",
                str(policy_python),
                "--ffmpeg",
                str(ffmpeg),
                "--fps",
                str(fps),
                "--resolution",
                str(resolution[0]),
                str(resolution[1]),
                "--cameras",
                *DEFAULT_CAMERAS,
                "--headless",
            )
        )
        return tuple(command)


@dataclass(frozen=True)
class CellSelection:
    """Auditable quota and source inventory for one formal cell."""

    cell_name: str
    cell_id: str
    success_rate: float
    paper_success_rate: float
    quota_tier: str
    requested_successes: int
    requested_failures: int
    available_successes: int
    available_failures: int
    required_successes: int
    required_failures: int


@dataclass(frozen=True)
class ReplayPlan:
    evaluation_set: Mapping[str, Any]
    selections: tuple[CellSelection, ...]
    jobs: tuple[ReplayJob, ...]
    selection_seed: int

    def audit(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "release": "v4",
            "selection_protocol_id": evaluation_videos.SELECTION_PROTOCOL_ID,
            "selection_seed": self.selection_seed,
            "evaluation_set": dict(self.evaluation_set),
            "cells": [selection.__dict__ for selection in self.selections],
            "jobs": [
                {
                    "cell_name": job.cell_name,
                    "cell_id": job.cell_id,
                    "source_task": job.source_task,
                    "scenario": job.scenario,
                    "policy_task": job.policy_task,
                    "outcome": job.outcome,
                    "quota_tier": job.quota_tier,
                    "requested_quota": job.requested_quota,
                    "required_confirmed": job.required_confirmed,
                    "candidate_episodes": list(job.candidate_episodes),
                    "source_result": str(job.source_result),
                    "source_result_sha256": job.source_result_sha256,
                    "models_dir": str(job.models_dir),
                    "target": str(job.target),
                }
                for job in self.jobs
            ],
        }


def _ranked_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_key: str,
    success: bool,
    selection_seed: int,
) -> tuple[int, ...]:
    candidates = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("formal result contains a non-object episode row")
        episode = row.get("episode")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise ValueError("formal result contains an invalid episode index")
        if episode in seen:
            raise ValueError(f"formal result repeats episode {episode}")
        seen.add(episode)
        if bool(row.get("success")) is success:
            candidates.append(episode)
    return tuple(
        sorted(
            candidates,
            key=lambda episode: (
                evaluation_videos._rank(
                    selection_seed=selection_seed,
                    cell_key=cell_key,
                    success=success,
                    episode=episode,
                ),
                episode,
            ),
        )
    )


def _plan_cell(
    cell: v4_formal_launch.FormalCell,
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    selection_seed: int,
) -> tuple[CellSelection, tuple[ReplayJob, ...]]:
    if payload.get("release") != "v4":
        raise RuntimeError(f"formal result is not V4: {cell.result}")
    if payload.get("task") != cell.task or payload.get("scenario") != cell.scenario:
        raise RuntimeError(f"formal result identity differs from {cell.cell_id}")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise RuntimeError(f"formal result has no episode rows: {cell.result}")
    success_rate = payload.get("success_rate")
    if isinstance(success_rate, bool) or not isinstance(success_rate, (int, float)):
        raise RuntimeError(f"formal result has no numeric success rate: {cell.result}")
    paper_rate = cell.paper_success_rate
    quota = evaluation_videos.retention_quota(success_rate, paper_rate)
    successes = _ranked_candidates(
        rows,
        cell_key=cell.cell_id,
        success=True,
        selection_seed=selection_seed,
    )
    failures = _ranked_candidates(
        rows,
        cell_key=cell.cell_id,
        success=False,
        selection_seed=selection_seed,
    )
    if len(successes) + len(failures) != len(rows):
        raise RuntimeError("formal result episode accounting is incomplete")
    required_successes = min(quota.successes, len(successes))
    required_failures = min(quota.failures, len(failures))
    selection = CellSelection(
        cell_name=cell.name,
        cell_id=cell.cell_id,
        success_rate=float(success_rate),
        paper_success_rate=float(paper_rate),
        quota_tier=quota.tier,
        requested_successes=quota.successes,
        requested_failures=quota.failures,
        available_successes=len(successes),
        available_failures=len(failures),
        required_successes=required_successes,
        required_failures=required_failures,
    )
    policy_task = payload.get("policy_task_alias") or payload.get("task")
    if policy_task not in failure_videos.SUPPORTED_TASKS:
        raise RuntimeError(f"failure_videos does not support task {policy_task!r}")
    jobs = []
    for outcome, requested, required, candidates in (
        ("success", quota.successes, required_successes, successes),
        ("failure", quota.failures, required_failures, failures),
    ):
        if required == 0:
            continue
        jobs.append(
            ReplayJob(
                cell_name=cell.name,
                cell_id=cell.cell_id,
                source_task=str(payload["task"]),
                scenario=str(payload["scenario"]),
                policy_task=str(policy_task),
                outcome=outcome,
                quota_tier=quota.tier,
                requested_quota=requested,
                required_confirmed=required,
                # Every matching source row is a finite ordered backup pool.
                # failure_videos stops as soon as ``required`` are confirmed.
                candidate_episodes=candidates,
                source_result=Path(cell.result).resolve(),
                source_result_sha256=source_sha256,
                models_dir=Path(cell.models_dir).resolve(),
                output_key=_safe_output_key(cell.name, outcome),
            )
        )
    return selection, tuple(jobs)


def build_validated_plan(
    *,
    selection_seed: int = evaluation_videos.DEFAULT_SELECTION_SEED,
) -> ReplayPlan:
    """Authenticate all 22 formal results and derive replay jobs."""

    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("selection seed must be an integer")
    if selection_seed < 0:
        raise ValueError("selection seed must be non-negative")
    canonical = v4_formal_launch._validate_evaluation_set()
    selections = []
    jobs = []
    output_keys = set()
    for cell in v4_formal_launch.FORMAL_CELLS:
        state = v4_formal_launch.cell_state(cell, canonical)
        if state != "COMPLETED_VALIDATED":
            raise RuntimeError(
                f"post-evaluation replay requires all formal results; {cell.cell_id} is {state}"
            )
        payload = _load_json(cell.result, f"validated result {cell.cell_id}")
        selection, cell_jobs = _plan_cell(
            cell,
            payload,
            source_sha256=_sha256(cell.result),
            selection_seed=selection_seed,
        )
        selections.append(selection)
        for job in cell_jobs:
            if job.output_key in output_keys:
                raise RuntimeError(f"duplicate replay output key: {job.output_key}")
            output_keys.add(job.output_key)
            jobs.append(job)
    return ReplayPlan(
        evaluation_set=canonical,
        selections=tuple(selections),
        jobs=tuple(jobs),
        selection_seed=selection_seed,
    )


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must contain integers") from error
    if (
        not 1 <= len(result) <= MAX_GPU_LANES
        or len(set(result)) != len(result)
        or any(gpu < 0 for gpu in result)
    ):
        raise argparse.ArgumentTypeError("provide 1..8 distinct non-negative GPUs")
    return result


def _validate_runtime(
    *,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
) -> tuple[Path, Path, Path, Path, tuple[int, ...], Mapping[str, object]]:
    sim_python = v4_formal_launch._regular_executable(
        sim_python, "Python 3.8 simulator interpreter"
    )
    policy_python = v4_formal_launch._regular_executable(
        policy_python, "Python 3.10 policy interpreter"
    )
    xvfb_run = v4_formal_launch._regular_executable(xvfb_run, "xvfb-run")
    ffmpeg = v4_formal_launch._regular_executable(ffmpeg, "ffmpeg")
    if len(gpus) > MAX_GPU_LANES:
        raise RuntimeError("post-evaluation replay supports at most eight GPU lanes")
    gpus = v4_formal_launch._validate_gpu_assignment(gpus)
    environment = v4_formal_launch._launch_environment(policy_python, gpus[0])
    v4_formal_launch._validate_python_runtime(
        sim_python,
        expected=(3, 8),
        imports=(
            "numpy",
            "pyrep",
            "rlbench",
            "integrations.rlbench.rlbench_dynamac.report.failure_videos",
            "integrations.rlbench.rlbench_dynamac.core.trac_ik",
        ),
        checks=(
            "from integrations.rlbench.rlbench_dynamac.core.pytracik_dependency "
            "import assert_formal_pytracik_build",
            "assert_formal_pytracik_build()",
        ),
        environment=environment,
        label="simulator Python",
    )
    v4_formal_launch._validate_python_runtime(
        policy_python,
        expected=(3, 10),
        imports=(
            "numpy",
            "scipy",
            "sklearn",
            "integrations.rlbench.rlbench_dynamac.data.direct_policy",
        ),
        environment=environment,
        label="policy Python",
    )
    model_release = v4_formal_launch._validate_model_release()
    return (
        sim_python,
        policy_python,
        xvfb_run,
        ffmpeg,
        tuple(gpus),
        model_release,
    )


def _xvfb_command(
    job: ReplayJob,
    *,
    lane: int,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    xvfb_log: Path,
    fps: int,
    resolution: tuple[int, int],
    maximum_attempts: int,
    overwrite: bool,
) -> tuple[str, ...]:
    return (
        str(xvfb_run),
        "--auto-servernum",
        "--server-num",
        str(XVFB_SERVER_BASE + lane),
        "--error-file",
        str(xvfb_log),
        "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
        *job.recorder_command(
            sim_python=sim_python,
            policy_python=policy_python,
            ffmpeg=ffmpeg,
            fps=fps,
            resolution=resolution,
            maximum_attempts=maximum_attempts,
            overwrite=overwrite,
        ),
    )


def render_plan(
    plan: ReplayPlan,
    *,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
    fps: int,
    resolution: tuple[int, int],
    maximum_attempts: int,
    overwrite: bool,
) -> str:
    lines = [
        "V4 post-evaluation replay plan (PLAN ONLY; no simulator is started):",
        f"cells={len(plan.selections)} jobs={len(plan.jobs)} "
        f"lanes={len(gpus)} cameras={','.join(DEFAULT_CAMERAS)}",
    ]
    for index, job in enumerate(plan.jobs):
        lane = index % len(gpus)
        command = _xvfb_command(
            job,
            lane=lane,
            sim_python=sim_python,
            policy_python=policy_python,
            xvfb_run=xvfb_run,
            ffmpeg=ffmpeg,
            xvfb_log=LAUNCH_ROOT / "<run-id>" / f"{job.output_key}.xvfb.log",
            fps=fps,
            resolution=resolution,
            maximum_attempts=maximum_attempts,
            overwrite=overwrite,
        )
        lines.extend(
            (
                "",
                f"[{index + 1}/{len(plan.jobs)}] {job.cell_id} {job.outcome} "
                f"required={job.required_confirmed}/"
                f"candidates={len(job.candidate_episodes)} "
                f"GPU-lane={lane} GPU={gpus[lane]}",
                "CUDA_VISIBLE_DEVICES={gpu} {command}".format(
                    gpu=gpus[lane],
                    command=" ".join(shlex.quote(value) for value in command),
                ),
                f"target={job.target}",
            )
        )
    return "\n".join(lines)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_job_output(job: ReplayJob) -> Mapping[str, Any]:
    target = job.target
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError(f"replay job did not publish a real directory: {target}")
    manifest_path = target / "manifest.json"
    manifest = _load_json(manifest_path, f"replay manifest {job.output_key}")
    if (
        manifest.get("schema") != failure_videos.SCHEMA
        or manifest.get("release") != "v4"
        or manifest.get("task") != job.policy_task
        or manifest.get("source_task") != job.source_task
        or manifest.get("scenario") != job.scenario
        or manifest.get("output_key") != job.output_key
        or manifest.get("source_result") != str(job.source_result)
        or manifest.get("source_result_sha256") != job.source_result_sha256
        or manifest.get("expected_outcome") != job.outcome
        or manifest.get("required_confirmed_trajectories") != job.required_confirmed
        or manifest.get("candidate_episode_order") != list(job.candidate_episodes)
    ):
        raise RuntimeError(f"replay manifest identity is invalid: {manifest_path}")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != job.required_confirmed:
        raise RuntimeError(f"replay manifest quota is incomplete: {manifest_path}")
    candidate_positions = {
        episode: index for index, episode in enumerate(job.candidate_episodes)
    }
    seen_episodes = set()
    previous_position = -1
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("replay manifest contains a non-object outcome row")
        episode = row.get("episode")
        position = (
            candidate_positions.get(episode)
            if isinstance(episode, int) and not isinstance(episode, bool)
            else None
        )
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode in seen_episodes
            or position is None
            or position <= previous_position
            or row.get("expected_outcome") != job.outcome
            or row.get("confirmed_for_publication") is not True
            or row.get("replay_confirmed_outcome") is not True
            or row.get("release") != "v4"
        ):
            raise RuntimeError("replay manifest contains an invalid outcome row")
        seen_episodes.add(episode)
        previous_position = position
        metadata_name = row.get("metadata")
        video_name = row.get("video")
        if not isinstance(metadata_name, str) or not isinstance(video_name, str):
            raise RuntimeError("replay manifest row has no artifact paths")
        metadata_path = (target / metadata_name).resolve()
        video_path = (target / video_name).resolve()
        if (
            not _inside(target.resolve(), metadata_path)
            or not _inside(target.resolve(), video_path)
            or not metadata_path.is_file()
            or not video_path.is_file()
        ):
            raise RuntimeError("replay manifest row escapes or lacks its target")
        sidecar = _load_json(metadata_path, "replay sidecar")
        video = sidecar.get("video")
        if (
            row.get("metadata_sha256") != _sha256(metadata_path)
            or row.get("video_sha256") != _sha256(video_path)
            or sidecar.get("release") != "v4"
            or sidecar.get("task") != job.policy_task
            or sidecar.get("source_task") != job.source_task
            or sidecar.get("scenario") != job.scenario
            or sidecar.get("episode") != episode
            or sidecar.get("source_result") != str(job.source_result)
            or sidecar.get("source_result_sha256") != job.source_result_sha256
            or sidecar.get("expected_outcome") != job.outcome
            or sidecar.get("confirmed_for_publication") is not True
            or not isinstance(video, dict)
            or video.get("file") != video_name
            or video.get("requested_cameras") != list(DEFAULT_CAMERAS)
            or video.get("used_cameras") != list(DEFAULT_CAMERAS)
            or video.get("sha256") != _sha256(video_path)
        ):
            raise RuntimeError("replay sidecar does not prove a two-view V4 video")
    return manifest


def _terminate_process_groups(processes: Iterable[subprocess.Popen]) -> None:
    v4_formal_launch._terminate_process_groups(processes)


def _run_queue(
    jobs: Sequence[ReplayJob],
    *,
    run_root: Path,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
    fps: int,
    resolution: tuple[int, int],
    maximum_attempts: int,
    overwrite: bool,
) -> tuple[Mapping[str, Any], ...]:
    queue = list(jobs)
    available_lanes = list(range(len(gpus)))
    unfinished = {}
    processes = {}
    logs = {}
    assignments = []

    def launch_job(job: ReplayJob, lane: int) -> None:
        gpu = gpus[lane]
        log_path = run_root / f"{job.output_key}.log"
        xvfb_log = run_root / f"{job.output_key}.xvfb.log"
        stream = log_path.open("xb")
        logs[job.output_key] = stream
        command = _xvfb_command(
            job,
            lane=lane,
            sim_python=sim_python,
            policy_python=policy_python,
            xvfb_run=xvfb_run,
            ffmpeg=ffmpeg,
            xvfb_log=xvfb_log,
            fps=fps,
            resolution=resolution,
            maximum_attempts=maximum_attempts,
            overwrite=overwrite,
        )
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=v4_formal_launch._launch_environment(policy_python, gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[job.output_key] = process
        unfinished[job.output_key] = (process, job, lane)
        assignments.append(
            {
                "cell_id": job.cell_id,
                "outcome": job.outcome,
                "output_key": job.output_key,
                "lane": lane,
                "gpu": gpu,
                "pid": process.pid,
                "started_unix": time.time(),
            }
        )

    try:
        while queue or unfinished:
            while queue and available_lanes:
                launch_job(queue.pop(0), available_lanes.pop(0))
            for name, (process, job, lane) in list(unfinished.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del unfinished[name]
                available_lanes.append(lane)
                available_lanes.sort()
                logs[name].flush()
                if return_code != 0:
                    _terminate_process_groups(item[0] for item in unfinished.values())
                    raise RuntimeError(
                        f"post-evaluation replay {name} exited {return_code}; see {run_root}"
                    )
                _validate_job_output(job)
                assignment = next(row for row in assignments if row["output_key"] == name)
                assignment["finished_unix"] = time.time()
            if queue or unfinished:
                time.sleep(1.0)
    except BaseException:
        _terminate_process_groups(processes.values())
        raise
    finally:
        for stream in logs.values():
            stream.close()
    return tuple(assignments)


def execute(
    *,
    selection_seed: int,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
    fps: int,
    resolution: tuple[int, int],
    maximum_attempts: int,
    overwrite: bool,
) -> Mapping[str, Any]:
    plan = build_validated_plan(selection_seed=selection_seed)
    (
        sim_python,
        policy_python,
        xvfb_run,
        ffmpeg,
        gpus,
        model_release,
    ) = _validate_runtime(
        sim_python=sim_python,
        policy_python=policy_python,
        xvfb_run=xvfb_run,
        ffmpeg=ffmpeg,
        gpus=gpus,
    )
    pending_jobs = []
    skipped_valid = []
    for job in plan.jobs:
        if not overwrite and (job.target.exists() or job.target.is_symlink()):
            manifest = _validate_job_output(job)
            skipped_valid.append(
                {
                    "cell_id": job.cell_id,
                    "outcome": job.outcome,
                    "output_key": job.output_key,
                    "target": str(job.target),
                    "manifest_sha256": _sha256(job.target / "manifest.json"),
                    "confirmed_trajectories": len(manifest["episodes"]),
                }
            )
        else:
            pending_jobs.append(job)
    if not plan.jobs:
        return {
            "schema": LAUNCH_SCHEMA,
            "status": "no_replays_required",
            "plan": plan.audit(),
            "model_release": model_release,
            "skipped_valid": [],
        }
    if not pending_jobs:
        return {
            "schema": LAUNCH_SCHEMA,
            "status": "all_replays_already_valid",
            "plan": plan.audit(),
            "model_release": model_release,
            "assignments": [],
            "skipped_valid": skipped_valid,
        }
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LAUNCH_ROOT / "execute.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"another post-evaluation replay launcher owns {lock}") from error
    started = time.time()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-pid{os.getpid()}"
        run_root = LAUNCH_ROOT / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        assignments = _run_queue(
            pending_jobs,
            run_root=run_root,
            sim_python=sim_python,
            policy_python=policy_python,
            xvfb_run=xvfb_run,
            ffmpeg=ffmpeg,
            gpus=gpus,
            fps=fps,
            resolution=resolution,
            maximum_attempts=maximum_attempts,
            overwrite=overwrite,
        )
        summary = {
            "schema": LAUNCH_SCHEMA,
            "status": "completed",
            "run_id": run_id,
            "started_unix": started,
            "finished_unix": time.time(),
            "gpu_lanes": list(gpus),
            "cameras": list(DEFAULT_CAMERAS),
            "model_release": model_release,
            "plan": plan.audit(),
            "assignments": list(assignments),
            "skipped_valid": skipped_valid,
        }
        atomic_json(run_root / "launch_summary.json", summary)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("plan", "preflight", "execute"),
        default="plan",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=evaluation_videos.DEFAULT_SELECTION_SEED,
    )
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--xvfb-run", type=Path, default=DEFAULT_XVFB_RUN)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--gpus", type=_parse_gpus, default=DEFAULT_GPUS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_RESOLUTION,
    )
    parser.add_argument(
        "--max-replay-attempts-per-episode",
        type=int,
        default=DEFAULT_MAX_REPLAY_ATTEMPTS_PER_EPISODE,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_cli_values(args: argparse.Namespace) -> None:
    if args.selection_seed < 0:
        raise ValueError("selection seed must be non-negative")
    if args.fps < 1 or any(value < 1 for value in args.resolution):
        raise ValueError("fps and resolution must be positive")
    failure_videos._validate_max_replay_attempts(args.max_replay_attempts_per_episode)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli_values(args)
    if args.command == "plan":
        plan = build_validated_plan(selection_seed=args.selection_seed)
        print(
            render_plan(
                plan,
                sim_python=args.sim_python,
                policy_python=args.policy_python,
                xvfb_run=args.xvfb_run,
                ffmpeg=args.ffmpeg,
                gpus=args.gpus,
                fps=args.fps,
                resolution=tuple(args.resolution),
                maximum_attempts=args.max_replay_attempts_per_episode,
                overwrite=args.overwrite,
            )
        )
        return 0
    if args.command == "preflight":
        plan = build_validated_plan(selection_seed=args.selection_seed)
        runtime = _validate_runtime(
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            xvfb_run=args.xvfb_run,
            ffmpeg=args.ffmpeg,
            gpus=args.gpus,
        )
        print(
            json.dumps(
                {
                    "status": "ready",
                    "plan": plan.audit(),
                    "model_release": runtime[-1],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = execute(
        selection_seed=args.selection_seed,
        sim_python=args.sim_python,
        policy_python=args.policy_python,
        xvfb_run=args.xvfb_run,
        ffmpeg=args.ffmpeg,
        gpus=args.gpus,
        fps=args.fps,
        resolution=tuple(args.resolution),
        maximum_attempts=args.max_replay_attempts_per_episode,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CellSelection",
    "ReplayJob",
    "ReplayPlan",
    "build_parser",
    "build_validated_plan",
    "execute",
    "main",
    "render_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
