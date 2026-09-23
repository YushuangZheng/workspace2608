"""Launch a bounded queue of post-evaluation qualitative video replays."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluations.development.formal_evaluation.resources import build_lane_specs
from evaluations.iclr2027.runners import replay_video, shared_episode
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval.v4_formal_launch import _launch_environment


DEFAULT_SIM_PYTHON = Path(
    "/home/zhengyushuang/.conda/envs-migrated-20260816/dynamac-paper/bin/python"
)
DEFAULT_POLICY_PYTHON = Path(
    "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python"
)


@dataclass
class Running:
    process: subprocess.Popen
    stream: Any
    lane: int
    job: dict[str, Any]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("selection root must be an object")
    return value


def _resolve(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    path.relative_to(REPOSITORY_ROOT.resolve())
    return path


def _paths(root: Path, job: dict[str, Any]) -> tuple[Path, Path, Path]:
    index = str(job["episode_id"]).rsplit("/", 1)[-1]
    output_relpath = job.get("output_relpath")
    if output_relpath is None:
        relative_cell = Path(job["task"]) / job["method_dir"] / job["outcome"]
    else:
        relative_cell = Path(str(output_relpath))
        if relative_cell.is_absolute() or ".." in relative_cell.parts:
            raise ValueError(f"unsafe output_relpath: {output_relpath}")
    cell = root / relative_cell
    video = cell / f"episode_{index}.mp4"
    records = root / "_records" / relative_cell / index
    log_key = "__".join((*relative_cell.parts, index))
    log = root / "_logs" / f"{log_key}.log"
    return video, records, log


def _valid_existing(video: Path, job: dict[str, Any]) -> bool:
    sidecar = video.with_suffix(".json")
    if not video.is_file() or not sidecar.is_file():
        return False
    value = _load(sidecar)
    return bool(
        value.get("outcome_reproduced") is True
        and value.get("identity_reproduced") is True
        and value.get("perturbation_reproduced") is True
        and value.get("episode_id") == job["episode_id"]
        and value.get("expected_outcome") == job["outcome"]
    )


def launch(
    selection_path: Path,
    *,
    workers: int,
    sim_python: Path,
    policy_python: Path,
    only_episode: str | None = None,
    only_method_dir: str | None = None,
    validate_only: bool = False,
) -> int:
    selection_path = selection_path.resolve()
    selection = _load(selection_path)
    manifest = _resolve(selection["manifest"])
    root = _resolve(selection["output_root"])
    jobs = list(selection["jobs"])
    if only_episode is not None:
        jobs = [job for job in jobs if job["episode_id"] == only_episode]
    if only_method_dir is not None:
        jobs = [job for job in jobs if job["method_dir"] == only_method_dir]
    if not jobs:
        raise ValueError("selection filters matched no replay jobs")
    if workers < 1 or workers > 8:
        raise ValueError("replay video workers must be in [1, 8]")
    if len({job["episode_id"] + ":" + job["method"] for job in jobs}) != len(jobs):
        raise ValueError("selection contains duplicate episode/method jobs")
    root.mkdir(parents=True, exist_ok=True)
    pending = deque()
    completed = []
    for job in jobs:
        video, records, log = _paths(root, job)
        source = _resolve(job["source_result"])
        if not source.is_file():
            raise FileNotFoundError(source)
        source_value = shared_episode._load_json(source)
        row = shared_episode._load_manifest_row(manifest, job["episode_id"])
        replay_video._validate_source_identity(source_value, row)
        replay_video._validate_perturbation_reproduction(
            source_value, source_value
        )
        expected_outcome = "success" if source_value.get("success") else "fail"
        if source_value.get("method_id") != job["method"]:
            raise ValueError("selection method differs from retained source")
        method_identity = source_value.get("method_config_identity")
        if not isinstance(method_identity, dict):
            raise ValueError("retained source has no method configuration identity")
        method_config = _resolve(str(method_identity.get("path")))
        if not method_config.is_file():
            raise FileNotFoundError(method_config)
        if replay_video._sha256(method_config) != method_identity.get("sha256"):
            raise ValueError(
                "retained method configuration no longer matches its audited hash"
            )
        if expected_outcome != job["outcome"]:
            raise ValueError("selection outcome differs from retained source")
        if _valid_existing(video, job):
            completed.append(job)
            continue
        if video.exists() or video.with_suffix(".json").exists():
            raise RuntimeError(f"incomplete or unauthenticated replay exists: {video}")
        pending.append((job, source, method_config, video, records, log))

    if validate_only:
        print(
            f"validated {len(jobs)} replay jobs; "
            f"existing={len(completed)}, pending={len(pending)}",
            flush=True,
        )
        return 0

    lanes = build_lane_specs(tuple(range(8)), workers)
    free = deque(range(len(lanes)))
    running: dict[int, Running] = {}
    failures = []
    while pending or running:
        while pending and free:
            lane_index = free.popleft()
            lane = lanes[lane_index]
            job, source, method_config, video, records, log = pending.popleft()
            video.parent.mkdir(parents=True, exist_ok=True)
            records.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w", encoding="utf-8")
            command = [
                "/usr/bin/xvfb-run",
                "--auto-servernum",
                "--server-num",
                str(280 + lane_index),
                "--error-file",
                str(log.with_suffix(".xvfb.log")),
                "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
                str(sim_python),
                "-m",
                "evaluations.iclr2027.runners.replay_video",
                "--manifest",
                str(manifest),
                "--episode-id",
                job["episode_id"],
                "--method",
                str(method_config),
                "--source-result",
                str(source),
                "--expected-outcome",
                job["outcome"],
                "--output-video",
                str(video),
                "--output-records",
                str(records),
                "--policy-python",
                str(policy_python),
            ]
            environment = _launch_environment(policy_python, lane.gpu)
            tmpdir = root / "_tmp" / f"lane_{lane_index:02d}"
            tmpdir.mkdir(parents=True, exist_ok=True)
            environment["TMPDIR"] = str(tmpdir)

            def affinity(cpus=tuple(lane.logical_cpus)):
                os.setsid()
                os.sched_setaffinity(0, set(cpus))

            process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_ROOT),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                preexec_fn=affinity,
            )
            running[lane_index] = Running(process, stream, lane_index, job)
        time.sleep(0.25)
        for lane_index, item in list(running.items()):
            code = item.process.poll()
            if code is None:
                continue
            item.stream.close()
            video, _records, _log = _paths(root, item.job)
            if code == 0 and _valid_existing(video, item.job):
                completed.append(item.job)
                state = "verified"
            else:
                failures.append({"job": item.job, "return_code": code})
                state = f"failed({code})"
            print(
                f"[{len(running)-1} active] {item.job['episode_id']} "
                f"{item.job['method_dir']}/{item.job['outcome']} {state}; "
                f"complete={len(completed)}/{len(jobs)}",
                flush=True,
            )
            del running[lane_index]
            free.append(lane_index)

    summary = {
        "schema": "essay2608.iclr2027.qualitative-replay-launch.v1",
        "selection": str(selection_path),
        "jobs": len(jobs),
        "verified": len(completed),
        "failures": failures,
        "workers": workers,
        "formal_results_modified": False,
    }
    atomic_json(root / "REPLAY_SUMMARY.json", summary)
    return 0 if not failures and len(completed) == len(jobs) else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--only-episode")
    parser.add_argument("--only-method-dir")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    return launch(
        args.selection,
        workers=args.workers,
        sim_python=args.sim_python.resolve(),
        policy_python=args.policy_python.resolve(),
        only_episode=args.only_episode,
        only_method_dir=args.only_method_dir,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
