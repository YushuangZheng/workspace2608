"""Preflight, plan, and run the preregistered Stage-six evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch

from .run_cell import PROTOCOL, load_protocol
from .resources import LaneSpec, build_lane_specs

RESULTS_ROOT = REPOSITORY_ROOT / "integrations/rlbench/results/phase6_formal_v1"
LAUNCH_ROOT = RESULTS_ROOT / "_launch"
CLOSED_LOOP_MODELS = REPOSITORY_ROOT / "integrations/rlbench/models/closed_loop_v1"
RUNNER_MODULE = "evaluations.phase6_formal_evaluation.run_cell"
DEFAULT_SIM_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_SIM_PYTHON",
        "/home/zhengyushuang/.conda/envs-migrated-20260816/dynamac-paper/bin/python",
    )
)
DEFAULT_POLICY_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_POLICY_PYTHON",
        "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python",
    )
)
DEFAULT_GPUS = tuple(range(8))
DEFAULT_WORKERS = 48


@dataclass(frozen=True)
class FormalCell:
    experiment: str
    task: str
    method: str
    fault: Optional[str]
    episodes: int

    @property
    def cell_id(self) -> str:
        parts = [self.experiment]
        if self.fault is not None:
            parts.append(self.fault)
        parts.extend((self.task, self.method))
        return "/".join(parts)

    @property
    def name(self) -> str:
        return self.cell_id.replace("/", "__")

    @property
    def result(self) -> Path:
        folder = RESULTS_ROOT / self.experiment
        if self.fault is not None:
            folder /= self.fault
        return folder / self.task / f"{self.method}_n{self.episodes}.json"

    @property
    def diagnostics(self) -> Path:
        return RESULTS_ROOT / "diagnostics" / self.name

    def command(self, sim_python: Path, policy_python: Path) -> tuple[str, ...]:
        values = [
            str(sim_python),
            "-m",
            RUNNER_MODULE,
            "--protocol",
            str(PROTOCOL),
            "--experiment",
            self.experiment,
            "--task",
            self.task,
            "--method",
            self.method,
            "--output",
            str(self.result),
            "--diagnostics-dir",
            str(self.diagnostics),
            "--policy-python",
            str(policy_python),
            "--closed-loop-models-dir",
            str(CLOSED_LOOP_MODELS),
        ]
        if self.fault is not None:
            values.extend(("--fault", self.fault))
        return tuple(values)


def build_cells(protocol: Mapping[str, Any], section: str) -> tuple[FormalCell, ...]:
    normal_bounds = protocol["evaluation_set"]["normal_episode_index_range"]
    fault_bounds = protocol["evaluation_set"]["fault_episode_index_range"]
    normal_n = int(normal_bounds[1]) - int(normal_bounds[0]) + 1
    fault_n = int(fault_bounds[1]) - int(fault_bounds[0]) + 1
    cells = []
    if section in {"normal", "all"}:
        cells.extend(
            FormalCell("normal", task, method, None, normal_n)
            for task in protocol["tasks"]
            for method in protocol["methods"]
        )
    if section in {"fault", "all"}:
        cells.extend(
            FormalCell("fault", task, method, fault, fault_n)
            for fault in protocol["faults"]
            for task in protocol["tasks"]
            for method in protocol["methods"]
        )
    return tuple(cells)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_identity(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"model tree is empty: {root}")
    records = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path) for path in files
    }
    digest = hashlib.sha256()
    for name, value in records.items():
        digest.update(f"{name}\0{value}\n".encode("utf-8"))
    return {
        "root": root.relative_to(REPOSITORY_ROOT).as_posix(),
        "file_count": len(records),
        "fingerprint": digest.hexdigest(),
        "files": records,
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_clean_commit() -> str:
    dirty = _git("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise RuntimeError("formal execution requires a clean committed worktree")
    return _git("rev-parse", "HEAD")


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must contain integers") from error
    if not 1 <= len(result) <= 8 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("provide 1..8 distinct GPU indices")
    return result


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = []
    for line in completed.stdout.splitlines():
        index, name, memory, bus = (item.strip() for item in line.split(",", 3))
        result.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory),
                "pci_bus_id": bus,
            }
        )
    return result


def _validate_gpus(gpus: Sequence[int]) -> tuple[int, ...]:
    values = tuple(gpus)
    available = {row["index"] for row in _gpu_inventory()}
    if not 1 <= len(values) <= 8 or len(set(values)) != len(values):
        raise RuntimeError("formal evaluation requires 1..8 distinct GPU lanes")
    missing = sorted(set(values).difference(available))
    if missing:
        raise RuntimeError(f"requested GPU indices are unavailable: {missing}")
    return values


def _lane_specs(gpus: Sequence[int], workers: int) -> tuple[LaneSpec, ...]:
    try:
        return build_lane_specs(gpus, workers)
    except (ValueError, RuntimeError) as error:
        raise RuntimeError(f"invalid formal worker allocation: {error}") from error


def _environment(policy_python: Path, gpu: int, cpus: Sequence[int]) -> dict[str, str]:
    environment = v4_formal_launch._launch_environment(policy_python, gpu)
    # Eight simulators must not each create a full-machine BLAS thread pool.
    environment.update(
        {
            "OMP_NUM_THREADS": str(max(1, len(cpus) // 2)),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ESSAY2608_FORMAL_RENDER_BACKEND": "xvfb_software_gl",
        }
    )
    return environment


def _xvfb_command(
    cell: FormalCell, lane: int, sim: Path, policy: Path
) -> tuple[str, ...]:
    xvfb_log = LAUNCH_ROOT / "active" / f"{cell.name}.xvfb.log"
    return (
        str(v4_formal_launch.DEFAULT_XVFB_RUN),
        "--auto-servernum",
        "--server-num",
        str(140 + lane),
        "--error-file",
        str(xvfb_log),
        "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
        *cell.command(sim, policy),
    )


def _validate_result(cell: FormalCell, commit: Optional[str] = None) -> None:
    try:
        payload = json.loads(cell.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read formal cell result: {cell.result}") from error
    metadata = payload.get("stage6_formal_evaluation")
    if not isinstance(metadata, dict) or metadata.get("schema") != (
        "essay2608.phase6_formal_result.v1"
    ):
        raise RuntimeError(f"result has no Stage-six formal identity: {cell.result}")
    expected = {
        "experiment": cell.experiment,
        "task": cell.task,
        "method": cell.method,
        "fault": cell.fault,
        "episodes_completed": cell.episodes,
        "protocol_sha256": _sha256(PROTOCOL),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"result {name} differs from protocol: {cell.result}")
    if commit is not None and metadata.get("git_commit") != commit:
        raise RuntimeError(
            f"result commit differs from current formal commit: {cell.result}"
        )
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != cell.episodes:
        raise RuntimeError(f"formal result has incomplete episode rows: {cell.result}")


def _states(cells: Sequence[FormalCell], commit: Optional[str]) -> dict[str, str]:
    result = {}
    for cell in cells:
        lock = cell.result.with_name(cell.result.name + ".lock")
        if lock.exists():
            raise RuntimeError(f"formal result has an active/stale lock: {lock}")
        if cell.result.exists():
            _validate_result(cell, commit)
            result[cell.cell_id] = "COMPLETED_VALIDATED"
        else:
            result[cell.cell_id] = "PENDING"
    return result


def preflight(
    *,
    protocol: Mapping[str, Any],
    cells: Sequence[FormalCell],
    sim_python: Path,
    policy_python: Path,
    gpus: Sequence[int],
    require_clean: bool,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    sim = v4_formal_launch._regular_executable(sim_python, "simulator Python")
    policy = v4_formal_launch._regular_executable(policy_python, "policy Python")
    v4_formal_launch._regular_executable(v4_formal_launch.DEFAULT_XVFB_RUN, "xvfb-run")
    gpus = _validate_gpus(gpus)
    configured_workers = int(protocol["resource_plan"]["parallel_lanes"])
    if workers != configured_workers:
        raise RuntimeError(
            f"formal workers {workers} differ from frozen protocol {configured_workers}"
        )
    specs = _lane_specs(gpus, workers)
    environment = _environment(policy, specs[0].gpu, specs[0].logical_cpus)
    v4_formal_launch._validate_python_runtime(
        sim,
        expected=(3, 8),
        imports=(
            "numpy",
            "pyrep",
            "rlbench",
            RUNNER_MODULE,
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
        policy,
        expected=(3, 10),
        imports=("numpy", "scipy", "sklearn", "essay2608.policy.closed_loop"),
        environment=environment,
        label="policy Python",
    )
    commit = _validate_clean_commit() if require_clean else None
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "git_commit": commit or _git("rev-parse", "HEAD"),
        "worktree_clean_required": require_clean,
        "evaluation_set": v4_formal_launch._validate_evaluation_set(),
        "v4_models": v4_formal_launch._validate_model_release(),
        "closed_loop_models": _tree_identity(CLOSED_LOOP_MODELS),
        "gpus": _gpu_inventory(),
        "selected_gpu_lanes": list(gpus),
        "parallel_workers": workers,
        "worker_lanes": [spec.to_dict() for spec in specs],
        "cpu_affinity": [list(spec.logical_cpus) for spec in specs],
        "render_backend": protocol["resource_plan"]["render_backend"],
        "cells": _states(cells, commit),
    }


def render_plan(
    protocol: Mapping[str, Any],
    cells: Sequence[FormalCell],
    sim_python: Path,
    policy_python: Path,
    gpus: Sequence[int],
    workers: int = DEFAULT_WORKERS,
) -> str:
    specs = _lane_specs(gpus, workers)
    lines = [
        "Stage-six preregistered formal plan (no simulator started)",
        f"protocol={PROTOCOL.relative_to(REPOSITORY_ROOT)} sha256={_sha256(PROTOCOL)}",
        f"cells={len(cells)} episodes={sum(cell.episodes for cell in cells)} workers={workers}",
        "renderer=xvfb_software_gl; CUDA identity is isolated per lane",
    ]
    for index, cell in enumerate(cells):
        lane = index % len(specs)
        spec = specs[lane]
        command = _xvfb_command(cell, lane, sim_python, policy_python)
        lines.extend(
            (
                "",
                f"[{index + 1}/{len(cells)}] {cell.cell_id}",
                f"lane={lane} gpu={spec.gpu} cpus={','.join(map(str, spec.logical_cpus))}",
                " ".join(shlex.quote(value) for value in command),
                f"result={cell.result}",
            )
        )
    return "\n".join(lines)


def _terminate(processes: Iterable[subprocess.Popen]) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15.0
    while active and time.monotonic() < deadline:
        active = [process for process in active if process.poll() is None]
        if active:
            time.sleep(0.25)
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def execute(
    *,
    protocol: Mapping[str, Any],
    cells: Sequence[FormalCell],
    sim_python: Path,
    policy_python: Path,
    gpus: Sequence[int],
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    identity = preflight(
        protocol=protocol,
        cells=cells,
        sim_python=sim_python,
        policy_python=policy_python,
        gpus=gpus,
        require_clean=True,
        workers=workers,
    )
    pending = [cell for cell in cells if identity["cells"][cell.cell_id] == "PENDING"]
    if not pending:
        return {"status": "nothing_to_run", **identity}
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LAUNCH_ROOT / "execute.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"another formal launcher owns {lock}") from error
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-pid{os.getpid()}"
    run_root = LAUNCH_ROOT / "runs" / run_id
    active_root = LAUNCH_ROOT / "active"
    run_root.mkdir(parents=True, exist_ok=False)
    active_root.mkdir(parents=True, exist_ok=True)
    processes: dict[str, tuple[subprocess.Popen, FormalCell, int]] = {}
    streams: dict[str, Any] = {}
    assignments: list[dict[str, Any]] = []
    specs = _lane_specs(gpus, workers)
    available = list(range(len(specs)))
    queue = list(pending)
    started = time.time()
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} run_id={run_id}\n")
        stream.flush()
        os.fsync(stream.fileno())

    def launch(cell: FormalCell, lane: int) -> None:
        spec = specs[lane]
        cell.result.parent.mkdir(parents=True, exist_ok=True)
        cell.diagnostics.mkdir(parents=True, exist_ok=True)
        log_path = run_root / f"{cell.name}.log"
        log_stream = log_path.open("xb")
        streams[cell.cell_id] = log_stream
        command = _xvfb_command(cell, lane, sim_python, policy_python)

        def set_affinity() -> None:
            os.sched_setaffinity(0, set(spec.logical_cpus))

        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=_environment(policy_python, spec.gpu, spec.logical_cpus),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=set_affinity,
        )
        processes[cell.cell_id] = (process, cell, lane)
        assignments.append(
            {
                "cell_id": cell.cell_id,
                "lane": lane,
                "gpu": spec.gpu,
                "numa_node": spec.numa_node,
                "physical_cores": list(spec.physical_cores),
                "cpus": list(spec.logical_cpus),
                "pid": process.pid,
                "command": list(command),
                "started_unix": time.time(),
            }
        )

    try:
        while queue or processes:
            while queue and available:
                launch(queue.pop(0), available.pop(0))
            for cell_id, (process, cell, lane) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del processes[cell_id]
                available.append(lane)
                available.sort()
                streams[cell_id].flush()
                if return_code != 0:
                    _terminate(entry[0] for entry in processes.values())
                    raise RuntimeError(
                        f"formal cell {cell_id} exited {return_code}; see {run_root}"
                    )
                _validate_result(cell, identity["git_commit"])
                identity["cells"][cell_id] = "COMPLETED_VALIDATED"
                row = next(
                    value for value in assignments if value["cell_id"] == cell_id
                )
                row["finished_unix"] = time.time()
            if queue or processes:
                time.sleep(1.0)
    except BaseException:
        _terminate(entry[0] for entry in processes.values())
        raise
    finally:
        for stream in streams.values():
            stream.close()
        lock.unlink(missing_ok=True)
        for path in active_root.glob("*.xvfb.log"):
            target = run_root / path.name
            if not target.exists():
                shutil.move(str(path), target)

    summary = {
        "schema": "essay2608.phase6_formal_launch.v1",
        "status": "completed",
        "run_id": run_id,
        "started_unix": started,
        "finished_unix": time.time(),
        "assignments": assignments,
        **identity,
    }
    atomic_json(run_root / "launch_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("plan", "preflight", "execute"), default="plan"
    )
    parser.add_argument(
        "--section", choices=("normal", "fault", "all"), default="normal"
    )
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--gpus", type=_parse_gpus, default=DEFAULT_GPUS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(PROTOCOL)
    cells = build_cells(protocol, args.section)
    if args.command == "plan":
        print(
            render_plan(
                protocol,
                cells,
                args.sim_python,
                args.policy_python,
                args.gpus,
                args.workers,
            )
        )
        return 0
    if args.command == "preflight":
        result = preflight(
            protocol=protocol,
            cells=cells,
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            gpus=args.gpus,
            require_clean=False,
            workers=args.workers,
        )
    else:
        result = execute(
            protocol=protocol,
            cells=cells,
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            gpus=args.gpus,
            workers=args.workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
