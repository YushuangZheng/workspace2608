"""Run isolated short RLBench jobs to admit a safe formal worker count."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch

from .launch import (
    DEFAULT_POLICY_PYTHON,
    DEFAULT_SIM_PYTHON,
    PROTOCOL,
    _gpu_inventory,
    _sha256,
)
from .resources import MAX_WORKERS, LaneSpec, build_lane_specs


DEFAULT_ROOT = REPOSITORY_ROOT / "integrations/rlbench/results/phase6_concurrency_probe"
PROBE_MODULE = "evaluations.phase6_rlbench_integration.run_normal_diagnostic_subset"
TASK_PRIORITY = (
    "bimanual_handover_item",
    "bimanual_lift_tray",
    "bimanual_sweep_to_dustpan",
    "bimanual_put_bottle_in_fridge",
    "place_cups",
    "open_microwave",
    "wipe_desk",
    "stack_wine",
)
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "policy worker error",
    "RuntimeError:",
    "XIO:  fatal IO error",
    "No space left on device",
    "CUDA out of memory",
)


@dataclass
class ProbeProcess:
    spec: LaneSpec
    task: str
    episode_index: int
    process: subprocess.Popen[bytes]
    stream: Any
    log: Path
    xvfb_log: Path
    result: Path
    diagnostics: Path
    started_unix: float


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must contain integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU list must be non-empty and distinct")
    return result


def _environment(policy_python: Path, spec: LaneSpec) -> dict[str, str]:
    environment = v4_formal_launch._launch_environment(policy_python, spec.gpu)
    environment.update(
        {
            "OMP_NUM_THREADS": str(max(1, len(spec.physical_cores))),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ESSAY2608_FORMAL_RENDER_BACKEND": "xvfb_software_gl",
        }
    )
    return environment


def _command(
    *,
    sim_python: Path,
    policy_python: Path,
    spec: LaneSpec,
    task: str,
    episode_index: int,
    horizon: int,
    result: Path,
    diagnostics: Path,
    xvfb_log: Path,
) -> tuple[str, ...]:
    return (
        str(v4_formal_launch.DEFAULT_XVFB_RUN),
        "--auto-servernum",
        "--server-num",
        str(220 + spec.lane),
        "--error-file",
        str(xvfb_log),
        "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
        str(sim_python),
        "-m",
        PROBE_MODULE,
        "--task",
        task,
        "--policy-type",
        "closed_loop_multistream",
        "--closed-loop-feature-profile",
        "full",
        "--episode-index",
        str(episode_index),
        "--output",
        str(result),
        "--diagnostics-dir",
        str(diagnostics),
        "--policy-python",
        str(policy_python),
        "--closed-loop-models-dir",
        str(REPOSITORY_ROOT / "integrations/rlbench/models/closed_loop_v1"),
        "--horizon",
        str(horizon),
    )


def _memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _gpu_sample() -> list[dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = []
    for line in completed.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        result.append(
            {
                "index": index,
                "memory_used_mib": memory,
                "utilization_percent": utilization,
            }
        )
    return result


def _resource_sample(
    entries: Sequence[ProbeProcess], *, stage: str, include_gpu: bool
) -> dict[str, Any]:
    return {
        "unix": time.time(),
        "stage": stage,
        "load_average": list(os.getloadavg()),
        "memory_available_bytes": _memory_available_bytes(),
        "gpus": _gpu_sample() if include_gpu else [],
        "active_workers": sum(entry.process.poll() is None for entry in entries),
    }


def _terminate(entries: Sequence[ProbeProcess]) -> None:
    active = [entry for entry in entries if entry.process.poll() is None]
    for entry in active:
        try:
            os.killpg(entry.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15.0
    while active and time.monotonic() < deadline:
        active = [entry for entry in active if entry.process.poll() is None]
        if active:
            time.sleep(0.25)
    for entry in active:
        try:
            os.killpg(entry.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def execute_probe(args: argparse.Namespace) -> dict[str, Any]:
    sim_python = v4_formal_launch._regular_executable(
        args.sim_python, "simulator Python"
    )
    policy_python = v4_formal_launch._regular_executable(
        args.policy_python, "policy Python"
    )
    inventory = {row["index"] for row in _gpu_inventory()}
    missing = sorted(set(args.gpus).difference(inventory))
    if missing:
        raise RuntimeError(f"requested GPUs are unavailable: {missing}")
    specs = build_lane_specs(args.gpus, args.workers)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-w{args.workers}"
    run_root = args.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    entries: list[ProbeProcess] = []
    samples: list[dict[str, Any]] = []
    started = time.time()
    try:
        for spec in specs:
            task = TASK_PRIORITY[spec.lane % len(TASK_PRIORITY)]
            episode_index = (
                args.episode_index_base + spec.lane // len(TASK_PRIORITY)
            ) % 200
            stem = f"lane{spec.lane:02d}__gpu{spec.gpu}__{task}__episode{episode_index:03d}"
            result = run_root / "results" / f"{stem}.json"
            diagnostics = run_root / "diagnostics" / stem
            log = run_root / "logs" / f"{stem}.log"
            xvfb_log = run_root / "logs" / f"{stem}.xvfb.log"
            result.parent.mkdir(parents=True, exist_ok=True)
            diagnostics.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("xb")
            command = _command(
                sim_python=sim_python,
                policy_python=policy_python,
                spec=spec,
                task=task,
                episode_index=episode_index,
                horizon=args.horizon,
                result=result,
                diagnostics=diagnostics,
                xvfb_log=xvfb_log,
            )

            def set_affinity(cpus: tuple[int, ...] = spec.logical_cpus) -> None:
                os.sched_setaffinity(0, set(cpus))

            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=_environment(policy_python, spec),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=set_affinity,
            )
            entries.append(
                ProbeProcess(
                    spec=spec,
                    task=task,
                    episode_index=episode_index,
                    process=process,
                    stream=stream,
                    log=log,
                    xvfb_log=xvfb_log,
                    result=result,
                    diagnostics=diagnostics,
                    started_unix=time.time(),
                )
            )
            if len(entries) % 8 == 0 or len(entries) == len(specs):
                # Sampling process state during launch captures the true peak
                # without making every spawn wait for an nvidia-smi query.
                samples.append(
                    _resource_sample(entries, stage="launching", include_gpu=False)
                )

        while any(entry.process.poll() is None for entry in entries):
            samples.append(_resource_sample(entries, stage="running", include_gpu=True))
            time.sleep(args.sample_interval)
    except BaseException:
        _terminate(entries)
        raise
    finally:
        for entry in entries:
            entry.stream.close()

    rows = []
    fatal_hits = []
    stale_locks = []
    for entry in entries:
        combined = ""
        for path in (entry.log, entry.xvfb_log):
            if path.exists():
                combined += path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern for pattern in FATAL_PATTERNS if pattern in combined]
        if hits:
            fatal_hits.append({"lane": entry.spec.lane, "patterns": hits})
        lock = entry.result.with_name(entry.result.name + ".lock")
        if lock.exists():
            stale_locks.append(str(lock.relative_to(REPOSITORY_ROOT)))
        payload_ok = False
        if entry.result.exists():
            try:
                payload = json.loads(entry.result.read_text(encoding="utf-8"))
                marker = payload.get("diagnostic_subset", {})
                payload_ok = bool(
                    marker.get("formal_result") is False
                    and marker.get("episode_indices") == [entry.episode_index]
                )
            except (OSError, json.JSONDecodeError):
                payload_ok = False
        rows.append(
            {
                **entry.spec.to_dict(),
                "task": entry.task,
                "episode_index": entry.episode_index,
                "pid": entry.process.pid,
                "return_code": entry.process.returncode,
                "result_valid": payload_ok,
                "runtime_seconds": time.time() - entry.started_unix,
                "log": str(entry.log.relative_to(REPOSITORY_ROOT)),
                "result": str(entry.result.relative_to(REPOSITORY_ROOT)),
            }
        )
    all_successful = all(
        row["return_code"] == 0 and row["result_valid"] for row in rows
    )
    minimum_available = min(
        (sample["memory_available_bytes"] for sample in samples),
        default=_memory_available_bytes(),
    )
    total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    memory_safe = minimum_available >= int(total_memory * 0.10)
    peak_active_workers = max(
        (int(sample["active_workers"]) for sample in samples), default=0
    )
    full_concurrency_observed = peak_active_workers == args.workers
    admitted = bool(
        all_successful
        and not fatal_hits
        and not stale_locks
        and memory_safe
        and full_concurrency_observed
    )
    summary = {
        "schema": "essay2608.phase6_concurrency_probe.v1",
        "run_id": run_id,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "protocol_sha256": _sha256(PROTOCOL),
        "workers": args.workers,
        "horizon": args.horizon,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "admitted": admitted,
        "criteria": {
            "all_workers_exit_zero_with_valid_diagnostic_result": all_successful,
            "fatal_log_patterns_absent": not fatal_hits,
            "stale_result_locks_absent": not stale_locks,
            "minimum_system_memory_available_at_least_10_percent": memory_safe,
            "all_workers_observed_concurrently": full_concurrency_observed,
        },
        "fatal_hits": fatal_hits,
        "stale_locks": stale_locks,
        "minimum_memory_available_bytes": minimum_available,
        "peak_active_workers": peak_active_workers,
        "lanes": rows,
        "resource_samples": samples,
    }
    atomic_json(run_root / "summary.json", summary)
    atomic_json(args.output_root / "latest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--gpus", type=_parse_gpus, default=tuple(range(8)))
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--episode-index-base", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"workers must lie in [1,{MAX_WORKERS}]")
    if args.horizon < 1 or args.sample_interval <= 0:
        raise ValueError("horizon and sample interval must be positive")
    summary = execute_probe(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
