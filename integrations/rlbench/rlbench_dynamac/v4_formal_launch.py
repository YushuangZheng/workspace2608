"""Preflight and launch the six preregistered V4 formal evaluation cells.

The default ``plan`` command is read-only and never starts RLBench.  ``execute``
first authenticates the sealed evaluation set and V4 model inventory, then
starts only missing cells.  Each cell gets a distinct CUDA device and Xvfb
server.  A process/protocol failure terminates every still-running sibling.

This launcher does not stage evaluation inputs, modify V3 results, or replay
videos.  Formal result publication remains the evaluator's atomic
``reserve_output``/``atomic_json`` transaction, after video selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .records import atomic_json


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = INTEGRATION_ROOT / "results" / "v4"
VIDEO_ROOT = RESULTS_ROOT / "evaluation_videos"
EVAL_SET_ROOT = INTEGRATION_ROOT / "evaluation_sets" / "rlbench_eval_v2"
MODELS_ROOT = INTEGRATION_ROOT / "models" / "v4"
MODEL_RELEASE_MANIFEST = MODELS_ROOT / "release_manifest.json"
COPPELIASIM_ROOT = REPOSITORY_ROOT / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
LAUNCH_ROOT = RESULTS_ROOT / "_launch" / "formal_six_cells"
EVAL_SET_ID = "rlbench_eval_v2"
EPISODES = 200
BASE_SEED = 2_608_000_000
HORIZON = 1_000
POLICY_TIMEOUT_SECONDS = 120.0
MAX_PRIMARY_ACTION_ATTEMPTS = 3
FINAL_SETTLING_STEPS = 10
SCENARIO_STEPS = 10
SCENARIO_MAX_ATTEMPTS = 100
DEFAULT_SIM_PYTHON = Path(os.environ.get("DYNAMAC_SIM_PYTHON", sys.executable))
DEFAULT_POLICY_PYTHON = Path(
    os.environ.get("DYNAMAC_POLICY_PYTHON", "python3.10")
)
DEFAULT_XVFB_RUN = Path("/usr/bin/xvfb-run")
DEFAULT_FFMPEG = Path("/usr/bin/ffmpeg")
DEFAULT_GPUS = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class FormalCell:
    """One immutable V4 result/video/model/command mapping."""

    name: str
    task: str
    scenario: str
    evaluator_module: str
    evaluator_arguments: Tuple[str, ...]
    models_dir: Path
    result: Path
    video_cell: Path

    @property
    def cell_id(self) -> str:
        return f"{self.task}/{self.scenario}"

    def command(self, sim_python: Path, policy_python: Path) -> Tuple[str, ...]:
        return (
            str(sim_python),
            "-m",
            self.evaluator_module,
            *self.evaluator_arguments,
            "--models-dir",
            str(self.models_dir),
            "--policy-python",
            str(policy_python),
            "--episodes",
            str(EPISODES),
            "--seed",
            str(BASE_SEED),
            "--eval-set-id",
            EVAL_SET_ID,
            "--horizon",
            str(HORIZON),
            "--policy-timeout",
            str(POLICY_TIMEOUT_SECONDS),
            "--max-primary-action-attempts",
            str(MAX_PRIMARY_ACTION_ATTEMPTS),
            "--final-settling-steps",
            str(FINAL_SETTLING_STEPS),
            "--release",
            "v4",
            "--record-v4-evaluation-videos",
            "--v4-evaluation-video-root",
            str(VIDEO_ROOT),
            "--headless",
            "--output",
            str(self.result),
        )


def _direct_cell(task: str, scenario: str) -> FormalCell:
    family = "table_ii" if scenario == "static" else "table_iii_environment"
    result = RESULTS_ROOT / family / (
        f"{task}_{scenario}_seed{BASE_SEED}_n{EPISODES}_h{HORIZON}.json"
    )
    return FormalCell(
        name=f"{task}_{scenario}",
        task=task,
        scenario=scenario,
        evaluator_module="integrations.rlbench.rlbench_dynamac.direct_evaluate",
        evaluator_arguments=(
            "--task",
            task,
            "--scenario",
            scenario,
            "--scenario-steps",
            str(SCENARIO_STEPS),
            "--scenario-max-attempts",
            str(SCENARIO_MAX_ATTEMPTS),
        ),
        models_dir=MODELS_ROOT,
        result=result,
        video_cell=VIDEO_ROOT / task / scenario,
    )


def _coordination_cell(arm: str) -> FormalCell:
    scenario = f"coordination_hand_{arm}"
    result = RESULTS_ROOT / "table_iii_coordination" / (
        f"{scenario}_v4_cartesian_tick235_"
        f"seed{BASE_SEED}_n{EPISODES}_h{HORIZON}.json"
    )
    return FormalCell(
        name=scenario,
        task="bimanual_handover_item_dynamic",
        scenario=scenario,
        evaluator_module=(
            "integrations.rlbench.rlbench_dynamac.table_iii_coordination"
        ),
        evaluator_arguments=(
            "evaluate",
            "--arm",
            arm,
            "--trigger-step",
            "235",
        ),
        # PolicyProcess appends bimanual_handover_item.  The released dynamic
        # checkpoint is models/v4/table_iii/bimanual_handover_item.
        models_dir=MODELS_ROOT / "table_iii",
        result=result,
        video_cell=VIDEO_ROOT / "bimanual_handover_item_dynamic" / scenario,
    )


FORMAL_CELLS: Tuple[FormalCell, ...] = (
    _direct_cell("bimanual_put_bottle_in_fridge", "static"),
    _direct_cell("bimanual_put_bottle_in_fridge", "teleport"),
    _direct_cell("bimanual_lift_tray", "static"),
    _direct_cell("bimanual_lift_tray", "teleport"),
    _coordination_cell("left"),
    _coordination_cell("right"),
)


def _regular_executable(path: Path, label: str) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute() and expanded.parent == Path("."):
        discovered = shutil.which(str(expanded))
        if discovered is not None:
            expanded = Path(discovered)
    resolved = expanded.resolve()
    if not resolved.is_file() or not os.access(str(resolved), os.X_OK):
        raise RuntimeError(f"{label} is not an executable file: {resolved}")
    return resolved


def _validate_python_runtime(
    executable: Path,
    *,
    expected: Tuple[int, int],
    imports: Sequence[str],
    environment: Mapping[str, str],
    label: str,
) -> None:
    statements = [
        "import sys",
        (
            "assert sys.version_info[:2] == {expected!r}, "
            "'unexpected Python ' + sys.version"
        ).format(expected=expected),
    ]
    statements.extend(f"import {module}" for module in imports)
    try:
        subprocess.run(
            [str(executable), "-c", "; ".join(statements)],
            cwd=str(REPOSITORY_ROOT),
            env=dict(environment),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(f"{label} import/version preflight failed: {detail}") from error


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_model_release() -> Dict[str, object]:
    """Re-inventory every V4 checkpoint and require exact manifest equality."""

    from .store_bottle_release_v4 import build_model_release_manifest

    published = _load_json(MODEL_RELEASE_MANIFEST, "V4 model release manifest")
    current = build_model_release_manifest(require_complete=True)
    if published != current:
        raise RuntimeError(
            "models/v4 no longer matches its verified release_manifest.json"
        )
    fingerprint = published.get("fingerprint")
    if not _is_sha256(fingerprint):
        raise RuntimeError("V4 model release manifest fingerprint is invalid")
    return {
        "schema": published.get("schema"),
        "release": published.get("release"),
        "manifest_path": MODEL_RELEASE_MANIFEST.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "manifest_sha256": _file_sha256(MODEL_RELEASE_MANIFEST),
        "manifest_fingerprint": fingerprint,
    }


def _validate_evaluation_set() -> Dict[str, Any]:
    """Authenticate all V2 references and regenerated task-scoped batches."""

    from .eval_set import load_fixed_eval_set_manifest
    from .v4_partial_report import canonical_eval_identity_from_loaded_manifest

    locks = sorted(EVAL_SET_ROOT.rglob("*.lock")) if EVAL_SET_ROOT.is_dir() else []
    if locks:
        raise RuntimeError(
            "rlbench_eval_v2 still has an active/stale build lock: "
            + ", ".join(str(path) for path in locks)
        )
    loaded = load_fixed_eval_set_manifest(
        EVAL_SET_ID,
        full_preflight=True,
        verify_training_files=True,
    )
    if loaded["payload"].get("evaluation_set_id") != EVAL_SET_ID:
        raise RuntimeError("loaded evaluation-set identity is not rlbench_eval_v2")
    return canonical_eval_identity_from_loaded_manifest(loaded)


def _available_gpu_indices() -> Tuple[int, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi GPU inventory failed") from error
    try:
        values = tuple(int(line.strip()) for line in completed.stdout.splitlines())
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned a malformed GPU index") from error
    if not values:
        raise RuntimeError("no NVIDIA GPU is available")
    return values


def _validate_gpu_assignment(gpus: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(gpus)
    if len(values) != len(FORMAL_CELLS) or len(set(values)) != len(values):
        raise RuntimeError("six formal cells require six distinct GPU indices")
    available = set(_available_gpu_indices())
    missing = sorted(set(values) - available)
    if missing:
        raise RuntimeError(f"requested GPUs are unavailable: {missing}")
    return values


def _validate_completed_result(
    cell: FormalCell,
    canonical_eval_identity: Mapping[str, Any],
) -> None:
    """Use the formal six-cell report validator as the skip admission gate."""

    from .v4_partial_report import TARGET_BY_ID, _validate_v4_result

    specification = TARGET_BY_ID.get(cell.cell_id)
    if specification is None:
        raise RuntimeError(f"cell is outside the formal six-cell scope: {cell.cell_id}")
    _validate_v4_result(
        cell.result,
        specification,
        RESULTS_ROOT,
        canonical_eval_identity=canonical_eval_identity,
    )


def cell_state(
    cell: FormalCell,
    canonical_eval_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return COMPLETED_VALIDATED or PENDING; reject ambiguous partial state."""

    result_lock = cell.result.with_name(cell.result.name + ".lock")
    if result_lock.exists():
        raise RuntimeError(f"result path is currently reserved: {result_lock}")
    if cell.result.exists():
        if canonical_eval_identity is None:
            from .v4_partial_report import load_canonical_eval_identity

            canonical_eval_identity = load_canonical_eval_identity()
        _validate_completed_result(cell, canonical_eval_identity)
        return "COMPLETED_VALIDATED"
    if cell.video_cell.exists() and any(cell.video_cell.iterdir()):
        raise RuntimeError(
            "video artifacts exist without an admitted atomic result; "
            f"quarantine/inspect them before retrying: {cell.video_cell}"
        )
    return "PENDING"


def _preflight_with_identity(
    *,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, object]]:
    """Run launch gates and return cells plus immutable release snapshots."""

    sim_python = _regular_executable(
        sim_python, "Python 3.8 simulator interpreter"
    )
    policy_python = _regular_executable(
        policy_python, "Python 3.10 policy interpreter"
    )
    _regular_executable(xvfb_run, "xvfb-run")
    ffmpeg = _regular_executable(ffmpeg, "ffmpeg")
    if ffmpeg != DEFAULT_FFMPEG.resolve():
        raise RuntimeError("V4 lightweight capture is frozen to /usr/bin/ffmpeg")
    _regular_executable(COPPELIASIM_ROOT / "coppeliaSim", "CoppeliaSim")
    if not (COPPELIASIM_ROOT / "libcoppeliaSim.so").is_file():
        raise RuntimeError("CoppeliaSim shared library is missing")
    gpus = _validate_gpu_assignment(gpus)
    launch_environment = _launch_environment(policy_python, gpus[0])
    _validate_python_runtime(
        sim_python,
        expected=(3, 8),
        imports=(
            "numpy",
            "pyrep",
            "rlbench",
            "integrations.rlbench.rlbench_dynamac.direct_evaluate",
        ),
        environment=launch_environment,
        label="simulator Python",
    )
    _validate_python_runtime(
        policy_python,
        expected=(3, 10),
        imports=(
            "numpy",
            "scipy",
            "sklearn",
            "integrations.rlbench.rlbench_dynamac.direct_policy",
        ),
        environment=launch_environment,
        label="policy Python",
    )
    model_release_identity = _validate_model_release()
    canonical_eval_identity = _validate_evaluation_set()
    states = {
        cell.name: cell_state(cell, canonical_eval_identity)
        for cell in FORMAL_CELLS
    }
    return states, canonical_eval_identity, model_release_identity


def preflight(
    *,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
) -> Dict[str, str]:
    """Run every non-simulator launch gate and classify the six result cells."""

    states, _, _ = _preflight_with_identity(
        sim_python=sim_python,
        policy_python=policy_python,
        xvfb_run=xvfb_run,
        ffmpeg=ffmpeg,
        gpus=gpus,
    )
    return states


def _parse_gpus(value: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must contain integers") from error
    if any(value < 0 for value in result):
        raise argparse.ArgumentTypeError("GPU indices must be non-negative")
    if len(result) != len(FORMAL_CELLS) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("provide exactly six distinct GPU indices")
    return result


def _launch_environment(policy_python: Path, gpu: int) -> Dict[str, str]:
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    required_pythonpath = [
        str(REPOSITORY_ROOT / "RLBench"),
        str(REPOSITORY_ROOT),
    ]
    if inherited_pythonpath:
        required_pythonpath.append(inherited_pythonpath)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(required_pythonpath),
            "COPPELIASIM_ROOT": str(COPPELIASIM_ROOT),
            "QT_QPA_PLATFORM_PLUGIN_PATH": str(COPPELIASIM_ROOT),
            "LD_LIBRARY_PATH": os.pathsep.join(
                value
                for value in (
                    str(COPPELIASIM_ROOT),
                    environment.get("LD_LIBRARY_PATH", ""),
                )
                if value
            ),
            # RGB capture uses the Xvfb display created for each lane.  The
            # offscreen Qt plugin cannot create the OpenGL3 renderer context
            # used by CoppeliaSim vision sensors and segfaults on first frame.
            "QT_QPA_PLATFORM": "xcb",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "DYNAMAC_POLICY_PYTHON": str(policy_python),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _xvfb_command(
    cell: FormalCell,
    *,
    index: int,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    xvfb_log: Path,
) -> Tuple[str, ...]:
    return (
        str(xvfb_run),
        "--auto-servernum",
        "--server-num",
        str(90 + index),
        "--error-file",
        str(xvfb_log),
        "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
        *cell.command(sim_python, policy_python),
    )


def render_plan(
    *,
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    gpus: Sequence[int],
) -> str:
    """Render all exact commands without reading artifacts or starting work."""

    lines = [
        "V4 formal six-cell plan (PLAN ONLY; no simulator is started):",
        f"evaluation_set={EVAL_SET_ID} episodes={EPISODES} "
        f"seed={BASE_SEED} horizon={HORIZON} release=v4",
    ]
    for index, (cell, gpu) in enumerate(zip(FORMAL_CELLS, gpus)):
        command = _xvfb_command(
            cell,
            index=index,
            sim_python=sim_python,
            policy_python=policy_python,
            xvfb_run=xvfb_run,
            xvfb_log=LAUNCH_ROOT / "<run-id>" / f"{cell.name}.xvfb.log",
        )
        lines.extend(
            (
                "",
                f"[{index + 1}/6] {cell.cell_id} GPU={gpu}",
                "CUDA_VISIBLE_DEVICES={gpu} {command}".format(
                    gpu=gpu,
                    command=" ".join(shlex.quote(value) for value in command),
                ),
                f"result={cell.result}",
                f"videos={cell.video_cell}",
            )
        )
    return "\n".join(lines)


def _terminate_process_groups(processes: Iterable[subprocess.Popen]) -> None:
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
    sim_python: Path,
    policy_python: Path,
    xvfb_run: Path,
    ffmpeg: Path,
    gpus: Sequence[int],
) -> Mapping[str, object]:
    """Launch pending cells in parallel and fail fast as one process group."""

    states, canonical_eval_identity, model_release_identity = (
        _preflight_with_identity(
            sim_python=sim_python,
            policy_python=policy_python,
            xvfb_run=xvfb_run,
            ffmpeg=ffmpeg,
            gpus=gpus,
        )
    )
    pending = [cell for cell in FORMAL_CELLS if states[cell.name] == "PENDING"]
    if not pending:
        return {
            "status": "nothing_to_run",
            "cells": states,
            "evaluation_set": canonical_eval_identity,
            "model_release": model_release_identity,
        }

    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LAUNCH_ROOT / "execute.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"another V4 six-cell launcher owns {lock}") from error
    processes: Dict[str, subprocess.Popen] = {}
    logs = {}
    started = time.time()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        run_id = (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + f"-pid{os.getpid()}"
        )
        run_root = LAUNCH_ROOT / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        for index, cell in enumerate(FORMAL_CELLS):
            if cell not in pending:
                continue
            gpu = gpus[index]
            log_path = run_root / f"{cell.name}.log"
            xvfb_log = run_root / f"{cell.name}.xvfb.log"
            log_stream = log_path.open("xb")
            logs[cell.name] = log_stream
            command = _xvfb_command(
                cell,
                index=index,
                sim_python=sim_python,
                policy_python=policy_python,
                xvfb_run=xvfb_run,
                xvfb_log=xvfb_log,
            )
            processes[cell.name] = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_ROOT),
                env=_launch_environment(policy_python, gpu),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        unfinished = dict(processes)
        while unfinished:
            for name, process in list(unfinished.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del unfinished[name]
                logs[name].flush()
                if return_code != 0:
                    _terminate_process_groups(unfinished.values())
                    raise RuntimeError(
                        f"formal cell {name} exited {return_code}; see {run_root}"
                    )
                cell = next(item for item in FORMAL_CELLS if item.name == name)
                try:
                    _validate_completed_result(cell, canonical_eval_identity)
                except Exception:
                    _terminate_process_groups(unfinished.values())
                    raise
                states[name] = "COMPLETED_VALIDATED"
            if unfinished:
                time.sleep(1.0)
    except BaseException:
        _terminate_process_groups(processes.values())
        raise
    finally:
        for stream in logs.values():
            stream.close()
        lock.unlink(missing_ok=True)

    summary = {
        "schema": "dynamac-v4-formal-six-cell-launch-v1",
        "status": "completed",
        "evaluation_set_id": EVAL_SET_ID,
        "evaluation_set": canonical_eval_identity,
        "model_release": model_release_identity,
        "release": "v4",
        "episodes": EPISODES,
        "seed": BASE_SEED,
        "horizon": HORIZON,
        "run_id": run_id,
        "started_unix": started,
        "finished_unix": time.time(),
        "cells": states,
    }
    atomic_json(run_root / "launch_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("plan", "preflight", "execute"),
        default="plan",
    )
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--xvfb-run", type=Path, default=DEFAULT_XVFB_RUN)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=DEFAULT_GPUS,
        help="Exactly six distinct GPU indices, in formal-cell order.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print(
            render_plan(
                sim_python=args.sim_python,
                policy_python=args.policy_python,
                xvfb_run=args.xvfb_run,
                gpus=args.gpus,
            )
        )
        return 0
    if args.command == "preflight":
        states = preflight(
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            xvfb_run=args.xvfb_run,
            ffmpeg=args.ffmpeg,
            gpus=args.gpus,
        )
        print(json.dumps(states, indent=2, sort_keys=True))
        return 0
    summary = execute(
        sim_python=args.sim_python,
        policy_python=args.policy_python,
        xvfb_run=args.xvfb_run,
        ffmpeg=args.ffmpeg,
        gpus=args.gpus,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BASE_SEED",
    "EVAL_SET_ID",
    "FORMAL_CELLS",
    "FormalCell",
    "HORIZON",
    "EPISODES",
    "build_parser",
    "cell_state",
    "execute",
    "main",
    "preflight",
    "render_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
