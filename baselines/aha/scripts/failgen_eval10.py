#!/usr/bin/env python3
"""Bounded, sequential runner for AHA's released ten-task FailGen eval list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO


OFFICIAL_EVAL_TASKS = (
    "stack_chairs",
    "place_hanger_on_rack",
    "place_cups",
    "light_bulb_out",
    "light_bulb_in",
    "lamp_on",
    "hockey",
    "open_oven",
    "meat_on_grill",
    "pick_up_cup",
)
CAMERAS = ("front", "overhead", "wrist")
MAX_TOTAL_SECONDS = 2 * 60 * 60
MAX_TASK_SECONDS = 10 * 60
MAX_ATTEMPT_SECONDS = 5 * 60
PNG_NAME = re.compile(r"^(front|overhead|wrist)_(\d+)\.png$")
SIMULATOR_CRASH = re.compile(
    r"(?:coppeliasim|remote.?api).*(?:crash|died|lost|closed|disconnect|"
    r"terminated|stopped)|failed to connect.*(?:coppeliasim|remote.?api)",
    flags=re.IGNORECASE,
)
RETRYABLE_FAILURE_CLASSES = frozenset(("simulator_crash", "worker_signal"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--official-eval-script", required=True, type=Path)
    parser.add_argument("--xvfb-bin", required=True, type=Path)
    parser.add_argument("--cop-root", required=True, type=Path)
    parser.add_argument("--total-timeout-seconds", type=int, default=MAX_TOTAL_SECONDS)
    parser.add_argument("--task-timeout-seconds", type=int, default=MAX_TASK_SECONDS)
    parser.add_argument(
        "--attempt-timeout-seconds", type=int, default=MAX_ATTEMPT_SECONDS
    )
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--max-tries", type=int, default=1)
    parser.add_argument("--display-min", type=int, default=120)
    parser.add_argument("--display-max", type=int, default=199)
    parser.add_argument("--static-check", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"AHA_TASK_YIELD {line}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tries != 1:
        raise ValueError("the fixed protocol requires --max-tries 1")
    if args.max_restarts != 1:
        raise ValueError("the fixed protocol requires --max-restarts 1")
    if not 0 < args.total_timeout_seconds <= MAX_TOTAL_SECONDS:
        raise ValueError("total timeout must be in (0, 7200]")
    if not 0 < args.task_timeout_seconds <= MAX_TASK_SECONDS:
        raise ValueError("task timeout must be in (0, 600]")
    if not 0 < args.attempt_timeout_seconds <= MAX_ATTEMPT_SECONDS:
        raise ValueError("attempt timeout must be in (0, 300]")
    if args.display_min < 1 or args.display_max < args.display_min:
        raise ValueError("invalid X display range")


def parse_official_tasks(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"tasks=\((.*?)\)", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"cannot find tasks array in {path}")
    return tuple(re.findall(r'"([a-z0-9_]+)"', match.group(1)))


def verify_official_tasks(path: Path) -> None:
    observed = parse_official_tasks(path)
    if observed != OFFICIAL_EVAL_TASKS:
        raise ValueError(
            "released eval list differs from the pinned bounded list: "
            f"expected={OFFICIAL_EVAL_TASKS}, observed={observed}"
        )


def process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, IndexError, ValueError, PermissionError):
        return None


def stop_owned_process(process: subprocess.Popen[Any], start_ticks: int | None) -> None:
    if process.poll() is not None:
        return
    if start_ticks is None or process_start_ticks(process.pid) != start_ticks:
        raise RuntimeError(f"refusing to signal PID {process.pid}: identity changed")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if process_start_ticks(process.pid) == start_ticks:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def display_is_free(number: int) -> bool:
    if Path(f"/tmp/.X{number}-lock").exists():
        return False
    if Path(f"/tmp/.X11-unix/X{number}").exists():
        return False
    probe = subprocess.run(
        ["xdpyinfo"],
        env={**os.environ, "DISPLAY": f":{number}"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3,
        check=False,
    )
    return probe.returncode != 0


def choose_display(args: argparse.Namespace) -> int:
    for number in range(args.display_min, args.display_max + 1):
        if display_is_free(number):
            return number
    raise RuntimeError("no unused X display is available in the configured range")


def launch_xvfb(
    args: argparse.Namespace, log_stream: IO[bytes], timeout_seconds: float
) -> tuple[subprocess.Popen[Any], int, int | None]:
    display = choose_display(args)
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    process = subprocess.Popen(
        [
            str(args.xvfb_bin),
            f":{display}",
            "-screen",
            "0",
            "1400x900x24",
            "-ac",
            "+extension",
            "GLX",
            "+iglx",
            "+render",
            "-noreset",
        ],
        env=clean_env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    start_ticks = process_start_ticks(process.pid)
    ready_budget = min(10.0, timeout_seconds)
    deadline = time.monotonic() + ready_budget
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Xvfb exited with rc={process.returncode}")
            probe = subprocess.run(
                ["xdpyinfo"],
                env={**os.environ, "DISPLAY": f":{display}"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=min(3.0, max(0.1, deadline - time.monotonic())),
                check=False,
            )
            if probe.returncode == 0:
                return process, display, start_ticks
            time.sleep(0.1)
        raise RuntimeError(
            f"Xvfb did not become ready within {ready_budget:.1f} seconds"
        )
    except BaseException:
        if process.poll() is None:
            stop_owned_process(process, start_ticks)
        raise


def worker_environment(args: argparse.Namespace, display: int) -> dict[str, str]:
    environment = os.environ.copy()
    cop_root = str(args.cop_root.resolve())
    environment.update(
        {
            "DISPLAY": f":{display}",
            "XDG_CACHE_HOME": str(args.xvfb_bin.resolve().parents[3]),
            "COPPELIASIM_ROOT": cop_root,
            "LD_LIBRARY_PATH": cop_root,
            "QT_PLUGIN_PATH": cop_root,
            "QT_QPA_PLATFORM_PLUGIN_PATH": str(args.cop_root / "platforms"),
            "QT_QPA_PLATFORM": "xcb",
            "QT_XCB_GL_INTEGRATION": "xcb_glx",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for name in (
        "CUDA_DEVICE_ORDER",
        "LIBGL_DRIVERS_PATH",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "GALLIUM_DRIVER",
    ):
        environment.pop(name, None)
    return environment


def classify_worker_failure(
    return_code: int, worker_payload: dict[str, Any]
) -> str:
    if return_code < 0:
        return "worker_signal"
    error_type = worker_payload.get("error_type", "")
    if error_type == "FailGenNotProduced":
        return "failgen_not_produced"
    if error_type == "ReleasedConfigurationError":
        return "released_configuration"
    error = str(worker_payload.get("error", ""))
    if SIMULATOR_CRASH.search(error):
        return "simulator_crash"
    return "simulator_or_worker_error"


def run_worker(
    args: argparse.Namespace,
    task: str,
    attempt_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    worker_result = attempt_dir / "worker_result.json"
    worker_log = attempt_dir / "worker.log"
    xvfb_log = attempt_dir / "xvfb.log"
    xvfb: subprocess.Popen[Any] | None = None
    xvfb_ticks: int | None = None
    child: subprocess.Popen[Any] | None = None
    child_ticks: int | None = None
    started = time.monotonic()
    outcome: dict[str, Any] = {
        "status": "failed",
        "failure_class": "runner_error",
        "started_at_utc": utc_now(),
    }
    attempt_deadline = started + timeout_seconds
    try:
        with xvfb_log.open("wb") as xvfb_stream:
            xvfb, display, xvfb_ticks = launch_xvfb(
                args, xvfb_stream, max(0.1, attempt_deadline - time.monotonic())
            )
            with worker_log.open("wb") as worker_stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(args.worker),
                        "--task",
                        task,
                        "--output",
                        str(attempt_dir / "artifacts"),
                        "--result-json",
                        str(worker_result),
                        "--max-tries",
                        "1",
                    ],
                    env=worker_environment(args, display),
                    stdout=worker_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child_ticks = process_start_ticks(child.pid)
                atomic_json(
                    attempt_dir / "process_meta.json",
                    {
                        "display": f":{display}",
                        "xvfb_pid": xvfb.pid,
                        "xvfb_start_ticks": xvfb_ticks,
                        "worker_pid": child.pid,
                        "worker_start_ticks": child_ticks,
                    },
                )
                try:
                    worker_budget = attempt_deadline - time.monotonic()
                    if worker_budget <= 0:
                        raise subprocess.TimeoutExpired(child.args, timeout_seconds)
                    return_code = child.wait(timeout=worker_budget)
                except subprocess.TimeoutExpired:
                    stop_owned_process(child, child_ticks)
                    outcome["failure_class"] = "timeout"
                    outcome["error"] = f"worker exceeded {timeout_seconds:.1f}s"
                    return outcome

        worker_payload: dict[str, Any] = {}
        if worker_result.is_file():
            worker_payload = json.loads(worker_result.read_text(encoding="utf-8"))
        outcome["worker_return_code"] = return_code
        outcome["worker"] = worker_payload
        if return_code == 0 and worker_payload.get("status") == "success":
            outcome["status"] = "worker_success"
            outcome["failure_class"] = None
            return outcome
        outcome["failure_class"] = classify_worker_failure(
            return_code, worker_payload
        )
        return outcome
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        outcome["failure_class"] = "xvfb_or_runner_startup"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        return outcome
    finally:
        if child is not None and child.poll() is None:
            stop_owned_process(child, child_ticks)
        if xvfb is not None and xvfb.poll() is None:
            stop_owned_process(xvfb, xvfb_ticks)
        outcome["duration_seconds"] = round(time.monotonic() - started, 3)
        outcome["finished_at_utc"] = utc_now()


def verify_pngs(artifact_dir: Path, attempt_dir: Path) -> dict[str, Any]:
    from PIL import Image

    resolved = artifact_dir.resolve()
    if not resolved.is_relative_to(attempt_dir.resolve()):
        raise ValueError("worker artifact directory escaped its attempt directory")
    if not resolved.is_dir():
        raise ValueError(f"artifact directory does not exist: {resolved}")

    by_camera: dict[str, list[int]] = {camera: [] for camera in CAMERAS}
    files: list[dict[str, Any]] = []
    for path in sorted(resolved.glob("*.png")):
        match = PNG_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected PNG name: {path.name}")
        camera, index_text = match.groups()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"empty image dimensions: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        index = int(index_text)
        by_camera[camera].append(index)
        files.append(
            {
                "name": path.name,
                "camera": camera,
                "index": index,
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    counts = {camera: len(indices) for camera, indices in by_camera.items()}
    if not files or any(count == 0 for count in counts.values()):
        raise ValueError(f"missing one or more camera streams: {counts}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"camera stream counts differ: {counts}")
    for camera, indices in by_camera.items():
        expected = list(range(len(indices)))
        if sorted(indices) != expected:
            raise ValueError(
                f"non-contiguous frame indices for {camera}: {sorted(indices)}"
            )
    return {
        "status": "complete",
        "png_count": len(files),
        "camera_counts": counts,
        "files": files,
    }


def run_task(
    args: argparse.Namespace,
    task: str,
    task_index: int,
    global_deadline: float,
) -> dict[str, Any]:
    task_dir = args.output_root / f"{task_index:02d}_{task}"
    task_dir.mkdir(parents=True, exist_ok=False)
    task_started = time.monotonic()
    task_deadline = min(global_deadline, task_started + args.task_timeout_seconds)
    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    for attempt_number in range(1, args.max_restarts + 2):
        remaining = task_deadline - time.monotonic()
        if remaining <= 0:
            break
        timeout_seconds = min(float(args.attempt_timeout_seconds), remaining)
        attempt_dir = task_dir / f"attempt_{attempt_number}"
        attempt = run_worker(args, task, attempt_dir, timeout_seconds)
        attempt["attempt"] = attempt_number
        attempt["coppelia_restart_count"] = attempt_number - 1
        attempts.append(attempt)

        if attempt["status"] == "worker_success":
            try:
                worker = attempt["worker"]
                integrity = verify_pngs(Path(worker["artifact_dir"]), attempt_dir)
                result = {
                    "schema_version": 1,
                    "task": task,
                    "episode_count": 1,
                    "max_tries": 1,
                    "status": "success",
                    "failure_class": None,
                    "failure_type": worker["failure_type"],
                    "waypoint": worker["waypoint"],
                    "renderer": worker["renderer"],
                    "attempts_used": attempt_number,
                    "coppelia_restart_count": attempt_number - 1,
                    "image_integrity": integrity,
                    "attempts": attempts,
                }
                break
            except (KeyError, OSError, ValueError) as exc:
                attempt["status"] = "failed"
                attempt["failure_class"] = "image_integrity"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                break
        if attempt["failure_class"] not in RETRYABLE_FAILURE_CLASSES:
            break
    if result is None:
        failure_class = (
            attempts[-1]["failure_class"] if attempts else "task_timeout"
        )
        result = {
            "schema_version": 1,
            "task": task,
            "episode_count": 1,
            "max_tries": 1,
            "status": "failed",
            "failure_class": failure_class,
            "failure_type": (
                attempts[-1].get("worker", {}).get("failure_type")
                if attempts
                else None
            ),
            "waypoint": (
                attempts[-1].get("worker", {}).get("waypoint")
                if attempts
                else None
            ),
            "attempts_used": len(attempts),
            "coppelia_restart_count": max(0, len(attempts) - 1),
            "image_integrity": None,
            "attempts": attempts,
        }
    result["duration_seconds"] = round(time.monotonic() - task_started, 3)
    result["finished_at_utc"] = utc_now()
    atomic_json(task_dir / "task_result.json", result)
    return result


def write_summary(
    output_root: Path, results: list[dict[str, Any]], started: str
) -> dict[str, Any]:
    successes = sum(result["status"] == "success" for result in results)
    summary = {
        "schema_version": 1,
        "protocol": "aha_released_failgen_eval10_bounded_v1",
        "official_tasks": list(OFFICIAL_EVAL_TASKS),
        "episodes_per_task": 1,
        "max_tries": 1,
        "max_coppelia_restarts_per_task": 1,
        "total_timeout_seconds": MAX_TOTAL_SECONDS,
        "renderer": "opengl3",
        "compute": "CPU/llvmpipe; CUDA_VISIBLE_DEVICES is empty",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "tasks_attempted": len(results),
        "tasks_succeeded": successes,
        "tasks_failed": len(results) - successes,
        "all_tasks_succeeded": successes == len(OFFICIAL_EVAL_TASKS),
        "results": results,
    }
    atomic_json(output_root / "summary.json", summary)
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "task",
                "status",
                "failure_class",
                "failure_type",
                "waypoint",
                "attempts_used",
                "coppelia_restart_count",
                "png_count",
                "front_pngs",
                "overhead_pngs",
                "wrist_pngs",
                "duration_seconds",
            ),
        )
        writer.writeheader()
        for result in results:
            integrity = result.get("image_integrity") or {}
            counts = integrity.get("camera_counts") or {}
            writer.writerow(
                {
                    "task": result["task"],
                    "status": result["status"],
                    "failure_class": result.get("failure_class"),
                    "failure_type": result.get("failure_type"),
                    "waypoint": result.get("waypoint"),
                    "attempts_used": result["attempts_used"],
                    "coppelia_restart_count": result["coppelia_restart_count"],
                    "png_count": integrity.get("png_count", 0),
                    "front_pngs": counts.get("front", 0),
                    "overhead_pngs": counts.get("overhead", 0),
                    "wrist_pngs": counts.get("wrist", 0),
                    "duration_seconds": result["duration_seconds"],
                }
            )
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    verify_official_tasks(args.official_eval_script)
    if args.static_check:
        print("Static protocol check passed: official 10 tasks, 1 episode, max_tries=1")
        return 0
    if not args.worker.is_file():
        raise SystemExit(f"worker not found: {args.worker}")
    if not args.xvfb_bin.is_file() or not os.access(args.xvfb_bin, os.X_OK):
        raise SystemExit(f"Xvfb is not executable: {args.xvfb_bin}")
    if not args.cop_root.is_dir():
        raise SystemExit(f"CoppeliaSim root not found: {args.cop_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    started = utc_now()
    atomic_json(
        args.output_root / "run_config.json",
        {
            "official_tasks": list(OFFICIAL_EVAL_TASKS),
            "episodes_per_task": 1,
            "max_tries": 1,
            "max_restarts": 1,
            "total_timeout_seconds": args.total_timeout_seconds,
            "task_timeout_seconds": args.task_timeout_seconds,
            "attempt_timeout_seconds": args.attempt_timeout_seconds,
            "cuda_visible_devices": "",
            "libgl_always_software": "1",
            "started_at_utc": started,
        },
    )
    events = args.output_root / "events.jsonl"
    deadline = time.monotonic() + args.total_timeout_seconds
    results: list[dict[str, Any]] = []

    try:
        for task_index, task in enumerate(OFFICIAL_EVAL_TASKS, start=1):
            if time.monotonic() >= deadline:
                result = {
                    "schema_version": 1,
                    "task": task,
                    "episode_count": 1,
                    "max_tries": 1,
                    "status": "failed",
                    "failure_class": "total_timeout_not_started",
                    "failure_type": None,
                    "waypoint": None,
                    "attempts_used": 0,
                    "coppelia_restart_count": 0,
                    "image_integrity": None,
                    "attempts": [],
                    "duration_seconds": 0.0,
                    "finished_at_utc": utc_now(),
                }
            else:
                result = run_task(args, task, task_index, deadline)
            results.append(result)
            append_event(events, result)
    finally:
        summary = write_summary(args.output_root, results, started)
    return 0 if summary["all_tasks_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
