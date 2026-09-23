"""Dynamic queue for the corrected nested-paired E4 episode entry point."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluations.iclr2027.audit.horizon_nested_events import NESTED_EVENT_SCHEDULE
from evaluations.iclr2027.runners import launch


class NestedHorizonQueue(launch.DynamicQueue):
    def _launch(self, row: dict[str, Any], lane_index: int) -> None:
        lane = self.lanes[lane_index]
        identifier = launch._safe(row["episode_id"])
        attempt = self.attempts[row["episode_id"]]
        log_path = self.log_root / (identifier + ".attempt%d.log" % attempt)
        stream = log_path.open("w", encoding="utf-8")
        command = [
            str(self.xvfb_run),
            "--auto-servernum",
            "--server-num",
            str(180 + lane_index),
            "--error-file",
            str(self.log_root / (identifier + ".xvfb.log")),
            "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
            str(self.sim_python),
            "-m",
            "evaluations.iclr2027.runners.a6_horizon_nested_episode",
            "--manifest",
            str(self.manifest),
            "--episode-id",
            str(row["episode_id"]),
            "--output-root",
            str(self.output_root),
            "--policy-python",
            str(self.policy_python),
            "--method",
            self.method,
        ]
        if self.policy_diagnostics_dir is not None:
            command.extend(["--policy-diagnostics-dir", str(self.policy_diagnostics_dir)])
        if self.calibration_artifact is not None:
            command.extend(["--calibration-artifact", str(self.calibration_artifact)])
        environment = launch._launch_environment(self.policy_python, lane.gpu)
        lane_tmp = self.tmp_root / ("lane_%02d" % lane_index)
        lane_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(lane_tmp)

        def affinity() -> None:
            os.setsid()
            os.sched_setaffinity(0, set(lane.logical_cpus))

        process = subprocess.Popen(
            command,
            cwd=str(launch.REPOSITORY_ROOT),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            preexec_fn=affinity,
        )
        self.running[lane_index] = launch.Running(
            process, row, lane_index, stream, time.monotonic()
        )
        self.peak_active = max(self.peak_active, len(self.running))


def main(argv: list[str] | None = None) -> int:
    args = launch._parser().parse_args(argv)
    rows = launch._rows(args.manifest)
    if args.task:
        wanted = set(args.task)
        rows = [row for row in rows if row["task"] in wanted]
    if args.episode_id:
        wanted_episodes = set(args.episode_id)
        rows = [row for row in rows if row["episode_id"] in wanted_episodes]
    if args.condition:
        rows = [row for row in rows if row["condition"] == args.condition]
    if args.limit_per_task is not None:
        counts: defaultdict[str, int] = defaultdict(int)
        limited = []
        for row in rows:
            if counts[row["task"]] >= args.limit_per_task:
                continue
            counts[row["task"]] += 1
            limited.append(row)
        rows = limited
    if not rows:
        raise ValueError("no nested E4 manifest rows selected")
    if any(row.get("event_schedule") != NESTED_EVENT_SCHEDULE for row in rows):
        raise ValueError("nested E4 queue accepts only nested-paired rows")
    if any(
        int(row.get("task_level") or 0) < 1
        or (
            int(row.get("task_level") or 0) < 2
            and row.get("development_only") is not True
        )
        for row in rows
    ):
        raise ValueError(
            "one-stage rows are allowed only in the development equivalence gate"
        )
    queue = NestedHorizonQueue(
        rows,
        manifest=args.manifest,
        output_root=args.output_root,
        workers=args.workers,
        sim_python=args.sim_python,
        policy_python=args.policy_python,
        xvfb_run=args.xvfb_run,
        success_target=args.until_successes_per_task,
        retry_infrastructure=args.retry_infrastructure,
        episode_timeout_seconds=args.episode_timeout_seconds,
        method=args.method,
        calibration_artifact=args.calibration_artifact,
        policy_diagnostics_dir=args.policy_diagnostics_dir,
    )
    metadata_path = queue.output_root / "RUN_METADATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["episode_module"] = (
        "evaluations.iclr2027.runners.a6_horizon_nested_episode"
    )
    metadata["event_schedule"] = NESTED_EVENT_SCHEDULE
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
