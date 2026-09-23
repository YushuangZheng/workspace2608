"""Global dynamic one-episode job queue for ICLR 2027 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from evaluations.iclr2027.runners.resources import (
    build_lane_specs,
)
from evaluations.iclr2027.runners.episode_io import (
    EpisodeWriter,
    completed_episode_ids,
    load_episode,
)
from evaluations.iclr2027.interfaces.feature_schema import EPISODE_SCHEMA
from evaluations.iclr2027.methods.registry import load_method_spec
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.eval.v4_formal_launch import (
    _launch_environment,
)

DEFAULT_SIM_PYTHON = Path(os.environ.get("DYNAMAC_SIM_PYTHON", sys.executable))
DEFAULT_POLICY_PYTHON = Path(os.environ.get("DYNAMAC_POLICY_PYTHON", sys.executable))
DEFAULT_XVFB_RUN = Path("/usr/bin/xvfb-run")
FAULT_CONFIG = (
    REPOSITORY_ROOT / "evaluations" / "iclr2027" / "configs" / "shared" / "faults.json"
)


def _safe(episode_id: str) -> str:
    return episode_id.replace("/", "__")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class Running:
    process: subprocess.Popen
    row: dict[str, Any]
    lane: int
    stream: Any
    started: float
    timed_out: bool = False


class DynamicQueue:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        manifest: Path,
        output_root: Path,
        workers: int,
        sim_python: Path,
        policy_python: Path,
        xvfb_run: Path,
        success_target: Optional[int],
        retry_infrastructure: int,
        episode_timeout_seconds: float,
        method: str,
        calibration_artifact: Path | None,
        policy_diagnostics_dir: Path | None,
    ) -> None:
        self.manifest = manifest.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_root = self.output_root / "logs"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root = self.output_root / "tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.sim_python = sim_python.resolve()
        self.policy_python = policy_python.resolve()
        self.xvfb_run = xvfb_run.resolve()
        self.success_target = success_target
        self.retry_infrastructure = int(retry_infrastructure)
        self.episode_timeout_seconds = float(episode_timeout_seconds)
        self.method = str(method)
        self.calibration_artifact = (
            None
            if calibration_artifact is None
            else calibration_artifact.resolve()
        )
        self.policy_diagnostics_dir = (
            None
            if policy_diagnostics_dir is None
            else policy_diagnostics_dir.resolve()
        )
        if self.episode_timeout_seconds <= 0:
            raise ValueError("episode timeout must be positive")
        self.attempts = defaultdict(int)
        self.successes = defaultdict(int)
        self.finished = defaultdict(int)
        self.infrastructure_errors = 0
        self.selected_episode_count = len(rows)
        done = completed_episode_ids(self.output_root)
        self.pending_by_task = {}
        for row in rows:
            if row["episode_id"] in done:
                result_path = self.output_root / "episodes" / (_safe(row["episode_id"]) + ".json")
                result = load_episode(result_path)
                self.finished[row["task"]] += 1
                self.successes[row["task"]] += int(result["success"])
                self.infrastructure_errors += int(
                    result.get("reason") == "infrastructure_error"
                )
                continue
            self.pending_by_task.setdefault(row["task"], deque()).append(row)
        self.task_order = deque(sorted(self.pending_by_task))
        self.lanes = build_lane_specs(tuple(range(8)), workers)
        self.free_lanes = deque(lane.lane for lane in self.lanes)
        self.running = {}
        self.cancelled = False
        self.peak_active = 0
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_run_metadata(rows)

    def _write_run_metadata(self, rows: list[dict[str, Any]]) -> None:
        manifest_hash = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        payload = {
            "schema": "essay2608.iclr2027.queue-run.v1",
            "started_utc": self.started_utc,
            "manifest": str(self.manifest),
            "manifest_sha256": manifest_hash,
            "selected_episode_count": len(rows),
            "workers": len(self.lanes),
            "one_episode_per_job": True,
            "dynamic_global_queue": True,
            "renderer": "headless_xvfb",
            "simulator_api": "in_process_pyrep",
            "policy_worker_transport": "private_stdio_pipe_per_episode",
            "network_control_port": None,
            "episode_timeout_seconds": self.episode_timeout_seconds,
            "method": self.method,
            "calibration_artifact": (
                None
                if self.calibration_artifact is None
                else str(self.calibration_artifact)
            ),
            "policy_diagnostics_dir": (
                None
                if self.policy_diagnostics_dir is None
                else str(self.policy_diagnostics_dir)
            ),
            "lanes": [
                {
                    "lane": lane.lane,
                    "gpu": lane.gpu,
                    "numa_node": lane.numa_node,
                    "physical_cores": list(lane.physical_cores),
                    "logical_cpus": list(lane.logical_cpus),
                    "display_floor": 180 + lane.lane,
                    "tmpdir": str(self.tmp_root / ("lane_%02d" % lane.lane)),
                }
                for lane in self.lanes
            ],
        }
        (self.output_root / "RUN_METADATA.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _task_done(self, task: str) -> bool:
        return (
            self.success_target is not None
            and self.successes[task] >= self.success_target
        )

    def _next(self) -> Optional[dict[str, Any]]:
        if not self.task_order:
            return None
        for _ in range(len(self.task_order)):
            task = self.task_order[0]
            self.task_order.rotate(-1)
            queue = self.pending_by_task[task]
            if self._task_done(task):
                queue.clear()
                continue
            if queue:
                return queue.popleft()
        self.task_order = deque(
            task
            for task in self.task_order
            if self.pending_by_task[task] and not self._task_done(task)
        )
        return None

    def _launch(self, row: dict[str, Any], lane_index: int) -> None:
        lane = self.lanes[lane_index]
        identifier = _safe(row["episode_id"])
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
            "evaluations.iclr2027.runners.shared_episode",
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
            command.extend(
                ["--policy-diagnostics-dir", str(self.policy_diagnostics_dir)]
            )
        if self.calibration_artifact is not None:
            command.extend(
                ["--calibration-artifact", str(self.calibration_artifact)]
            )
        environment = _launch_environment(self.policy_python, lane.gpu)
        lane_tmp = self.tmp_root / ("lane_%02d" % lane_index)
        lane_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(lane_tmp)

        def affinity() -> None:
            os.setsid()
            os.sched_setaffinity(0, set(lane.logical_cpus))

        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            preexec_fn=affinity,
        )
        self.running[lane_index] = Running(process, row, lane_index, stream, time.monotonic())
        self.peak_active = max(self.peak_active, len(self.running))

    def _requeue_infrastructure_retry(self, row: dict[str, Any]) -> None:
        """Return a failed episode to both the task queue and its scheduler ring."""

        task = row["task"]
        self.pending_by_task[task].appendleft(row)
        if task not in self.task_order:
            self.task_order.append(task)

    def _discard_episode_outputs(self, row: dict[str, Any]) -> None:
        """Remove outputs from an attempt that exceeded the wall-time limit.

        A simulator process can take a few seconds to unwind after its
        ``xvfb-run`` parent receives SIGTERM.  Such an attempt is invalid even
        if it manages to write a result during that grace interval; removing
        all episode-scoped files before requeueing prevents a late attempt from
        being mistaken for its deterministic retry.
        """

        identifier = _safe(row["episode_id"])
        for path in (
            self.output_root / "episodes" / f"{identifier}.json",
            self.output_root / "episodes" / f"{identifier}.json.tmp",
            self.output_root / "cycles" / f"{identifier}.jsonl.gz",
            self.output_root / "cycles" / f"{identifier}.jsonl.gz.tmp",
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _write_exhausted_infrastructure_failure(
        self, row: dict[str, Any], *, timed_out: bool, elapsed: float
    ) -> dict[str, Any]:
        """Commit an ITT failure after the frozen infrastructure retry is exhausted."""

        spec = load_method_spec(self.method)
        writer = EpisodeWriter(self.output_root, str(row["episode_id"]))
        summary = {
            "schema": EPISODE_SCHEMA,
            "episode_id": str(row["episode_id"]),
            "split": str(row["split"]),
            "task": str(row["task"]),
            "task_level": row.get("task_level"),
            "variation": int(row["variation"]),
            "seed": int(row["seed"]),
            "condition": str(row["condition"]),
            "fault_family": row.get("fault_family"),
            "fault_severity": row.get("fault_severity"),
            "trigger_stage": row.get("trigger_stage"),
            "method_id": spec.method_id,
            "method_config_identity": {
                "path": str(spec.config_path.relative_to(REPOSITORY_ROOT)),
                "sha256": spec.config_sha256,
                "fault_config_sha256": _sha256(FAULT_CONFIG),
                "policy_model": None,
                "monitor_calibration": (
                    None
                    if self.calibration_artifact is None
                    else {
                        "path": str(
                            self.calibration_artifact.relative_to(REPOSITORY_ROOT)
                        ),
                        "sha256": _sha256(self.calibration_artifact),
                    }
                ),
            },
            "success": False,
            "final_success": False,
            "reason": "infrastructure_error",
            "termination_reason": "infrastructure_error",
            "cycles": 0,
            "recovery_cycles": 0,
            "generic_retry_cycles": 0,
            "policy_recovery_cycles": 0,
            "first_alarm_cycle": None,
            "false_interventions": 0,
            "relation_restored_cycle": None,
            "legal_reentry_cycle": None,
            "post_reentry_completion": None,
            "invalid_actions": 0,
            "wall_seconds": float(elapsed),
            "peak_memory_kib": 0,
            "audit": {
                "eligible": False,
                "physically_triggered": False,
                "violation_onset_cycle": None,
                "violation_end_cycle": None,
                "relation_restored_cycle": None,
                "legal_reentry_cycle": None,
            },
            "fault_protocol": None,
            "fresh_task_generation": None,
            "error": {
                "type": "EpisodeProcessTimeout" if timed_out else "EpisodeProcessFailure",
                "message": "episode infrastructure retry budget exhausted",
                "attempts": int(self.attempts[row["episode_id"]]) + 1,
            },
        }
        writer.finalize(summary)
        return load_episode(writer.episode_path)

    def _collect(self, lane_index: int, running: Running) -> None:
        code = running.process.poll()
        if code is None:
            return
        if (
            running.timed_out
            and time.monotonic() - running.started
            <= self.episode_timeout_seconds + 10.0
        ):
            # Keep the lane reserved for the full termination grace period.
            # Otherwise an exited xvfb-run wrapper can cause the same episode
            # to be relaunched while a simulator grandchild is still writing.
            return
        if running.timed_out:
            try:
                os.killpg(running.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        running.stream.close()
        del self.running[lane_index]
        self.free_lanes.append(lane_index)
        row = running.row
        if running.timed_out:
            self._discard_episode_outputs(row)
        path = self.output_root / "episodes" / (_safe(row["episode_id"]) + ".json")
        result = load_episode(path) if path.is_file() else None
        infrastructure = (
            running.timed_out
            or result is None
            or result.get("reason") == "infrastructure_error"
        )
        if infrastructure and self.attempts[row["episode_id"]] < self.retry_infrastructure:
            self.attempts[row["episode_id"]] += 1
            self._requeue_infrastructure_retry(row)
            return
        elapsed = time.monotonic() - running.started
        if infrastructure and result is None:
            result = self._write_exhausted_infrastructure_failure(
                row, timed_out=running.timed_out, elapsed=elapsed
            )
        if infrastructure:
            self.infrastructure_errors += 1
        self.finished[row["task"]] += 1
        if result is not None:
            self.successes[row["task"]] += int(result["success"])
            audit = result.get("audit", {})
            trigger = "%d/%d" % (
                int(bool(audit.get("physically_triggered"))),
                int(bool(audit.get("eligible"))),
            )
            reason = result["reason"]
        else:
            trigger = "0/0"
            reason = "episode_timeout" if running.timed_out else "missing_result"
        print(
            "[%d active, %d free] %s %s %.1fs; task=%d done %d success; trigger=%s"
            % (
                len(self.running),
                len(self.free_lanes),
                row["episode_id"],
                reason,
                elapsed,
                self.finished[row["task"]],
                self.successes[row["task"]],
                trigger,
            ),
            flush=True,
        )

    def run(self) -> int:
        def cancel(_signal, _frame) -> None:
            self.cancelled = True

        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        try:
            while not self.cancelled:
                now = time.monotonic()
                for running in self.running.values():
                    elapsed = now - running.started
                    if elapsed > self.episode_timeout_seconds and not running.timed_out:
                        running.timed_out = True
                        try:
                            os.killpg(running.process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    elif running.timed_out and elapsed > self.episode_timeout_seconds + 10.0:
                        try:
                            os.killpg(running.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                for lane_index, running in list(self.running.items()):
                    self._collect(lane_index, running)
                while self.free_lanes:
                    row = self._next()
                    if row is None:
                        break
                    self._launch(row, self.free_lanes.popleft())
                if not self.running:
                    if self._next() is None:
                        break
                    # Put it back: this branch is defensive because available
                    # lanes normally consume pending work above.
                    raise RuntimeError("pending work exists without a running lane")
                time.sleep(0.25)
        finally:
            if self.cancelled:
                process_groups = []
                for running in self.running.values():
                    try:
                        process_groups.append(os.getpgid(running.process.pid))
                        os.killpg(running.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                for running in self.running.values():
                    try:
                        running.process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                    running.stream.close()
                # ``xvfb-run`` can exit before its simulator grandchild.  Kill
                # any surviving members of the episode-local process groups so
                # cancellation cannot leak CoppeliaSim or policy workers.
                for process_group in process_groups:
                    try:
                        os.killpg(process_group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        completed_episode_count = sum(self.finished.values())
        missing_selected_count = (
            max(0, self.selected_episode_count - completed_episode_count)
            if self.success_target is None
            else 0
        )
        status = {
            "cancelled": self.cancelled,
            "success_target": self.success_target,
            "workers": len(self.lanes),
            "peak_active": self.peak_active,
            "one_episode_per_job": True,
            "episode_timeout_seconds": self.episode_timeout_seconds,
            "selected_episode_count": self.selected_episode_count,
            "completed_episode_count": completed_episode_count,
            "missing_selected_count": missing_selected_count,
            "finished": dict(self.finished),
            "successes": dict(self.successes),
            "infrastructure_errors": self.infrastructure_errors,
        }
        (self.output_root / "QUEUE_STATUS.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(status, sort_keys=True), flush=True)
        if self.cancelled:
            return 130
        if self.infrastructure_errors:
            return 2
        if missing_selected_count:
            return 4
        if self.success_target is not None and any(
            self.successes[task] < self.success_target for task in self.pending_by_task
        ):
            return 3
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--task", action="append")
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--condition", choices=("nominal", "perturbed"))
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--until-successes-per-task", type=int)
    parser.add_argument("--retry-infrastructure", type=int, default=1)
    parser.add_argument("--episode-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--xvfb-run", type=Path, default=DEFAULT_XVFB_RUN)
    parser.add_argument("--method", default="m0_dynamac")
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--policy-diagnostics-dir", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    rows = _rows(args.manifest)
    if args.task:
        wanted = set(args.task)
        rows = [row for row in rows if row["task"] in wanted]
    if args.episode_id:
        wanted_episodes = set(args.episode_id)
        rows = [row for row in rows if row["episode_id"] in wanted_episodes]
    if args.condition:
        rows = [row for row in rows if row["condition"] == args.condition]
    if args.limit_per_task is not None:
        counts = defaultdict(int)
        limited = []
        for row in rows:
            if counts[row["task"]] >= args.limit_per_task:
                continue
            counts[row["task"]] += 1
            limited.append(row)
        rows = limited
    if not rows:
        raise ValueError("no manifest rows selected")
    queue = DynamicQueue(
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
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
