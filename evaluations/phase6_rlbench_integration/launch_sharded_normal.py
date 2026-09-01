"""Work-conserving sharded launcher for fixed normal qualification batches.

This runner is deliberately outside the core policy.  It schedules immutable
episode-index shards from all requested methods through one global lane pool,
persists every shard atomically, resumes completed shards, and writes one
strictly index-checked compact result per method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluations.phase6_formal_evaluation.launch import (
    DEFAULT_POLICY_PYTHON,
    DEFAULT_SIM_PYTHON,
    _environment,
)
from evaluations.phase6_formal_evaluation.resources import build_lane_specs
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.core.runtime import (
    STAGE6_IK_CONTROLLER_PROFILE,
    load_staged_motion_plan_batch,
)
from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch

RUNNER_MODULE = (
    "evaluations.phase6_rlbench_integration.run_normal_diagnostic_subset"
)
METHODS = {
    "dynamac_v4": ("dynamac", "full"),
    "progress_only": ("closed_loop_multistream", "progress_only"),
    "progress_dynamic_roles": (
        "closed_loop_multistream",
        "progress_dynamic_roles",
    ),
    "full": ("closed_loop_multistream", "full"),
}


@dataclass(frozen=True)
class QualificationShard:
    task: str
    method: str
    indices: tuple[int, ...]
    output_root: Path

    @property
    def shard_id(self) -> str:
        return (
            f"{self.task}/{self.method}/"
            f"episodes_{self.indices[0]:04d}_{self.indices[-1]:04d}"
        )

    @property
    def name(self) -> str:
        return self.shard_id.replace("/", "__")

    @property
    def result(self) -> Path:
        return self.output_root / "shards" / self.method / f"{self.name}.json"

    @property
    def diagnostics(self) -> Path:
        return self.output_root / "diagnostics" / self.method / self.name


def build_shards(
    *,
    task: str,
    methods: Sequence[str],
    episodes: int,
    shard_size: int,
    output_root: Path,
) -> tuple[QualificationShard, ...]:
    if episodes < 1 or shard_size < 1:
        raise ValueError("episodes and shard size must be positive")
    unknown = set(methods).difference(METHODS)
    if unknown:
        raise ValueError(f"unknown qualification methods: {sorted(unknown)}")
    per_method = {
        method: [
            QualificationShard(
                task,
                method,
                tuple(range(offset, min(episodes, offset + shard_size))),
                output_root,
            )
            for offset in range(0, episodes, shard_size)
        ]
        for method in methods
    }
    result = []
    rounds = max(len(values) for values in per_method.values())
    for index in range(rounds):
        result.extend(
            per_method[method][index]
            for method in methods
            if index < len(per_method[method])
        )
    return tuple(result)


def _command(
    shard: QualificationShard,
    *,
    sim_python: Path,
    policy_python: Path,
    models_dir: Path,
    closed_loop_models_dir: Path,
    motion_plans: Path,
    horizon: int,
) -> tuple[str, ...]:
    policy_type, feature_profile = METHODS[shard.method]
    values = [
        str(sim_python),
        "-m",
        RUNNER_MODULE,
        "--task",
        shard.task,
        "--policy-type",
        policy_type,
        "--closed-loop-feature-profile",
        feature_profile,
        "--models-dir",
        str(models_dir),
        "--closed-loop-models-dir",
        str(closed_loop_models_dir),
        "--motion-plans",
        str(motion_plans),
        "--controller-profile",
        STAGE6_IK_CONTROLLER_PROFILE,
        "--output",
        str(shard.result),
        "--diagnostics-dir",
        str(shard.diagnostics),
        "--policy-python",
        str(policy_python),
        "--horizon",
        str(horizon),
    ]
    for index in shard.indices:
        values.extend(("--episode-index", str(index)))
    return tuple(values)


def _validate_shard(
    shard: QualificationShard,
    *,
    batch_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(shard.result.read_text(encoding="utf-8"))
    metadata = payload.get("diagnostic_subset")
    if (
        not isinstance(metadata, dict)
        or metadata.get("task") != shard.task
        or metadata.get("episode_indices") != list(shard.indices)
        or metadata.get("qualification_batch", {}).get("sha256") != batch_sha256
    ):
        raise RuntimeError(f"qualification shard identity differs: {shard.result}")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != len(shard.indices):
        raise RuntimeError(f"qualification shard rows are incomplete: {shard.result}")
    return payload


def merge_method(
    *,
    task: str,
    method: str,
    episodes: int,
    shards: Sequence[QualificationShard],
    motion_plans: Path,
    output_root: Path,
) -> Path:
    batch_sha256 = hashlib.sha256(motion_plans.read_bytes()).hexdigest()
    ordered = sorted(shards, key=lambda shard: shard.indices[0])
    payloads = [
        _validate_shard(shard, batch_sha256=batch_sha256) for shard in ordered
    ]
    indices = tuple(index for shard in ordered for index in shard.indices)
    if indices != tuple(range(episodes)) or len(indices) != len(set(indices)):
        raise RuntimeError(f"qualification coverage differs for {task}/{method}")
    identity_fields = (
        "task",
        "scenario",
        "policy_type",
        "closed_loop_feature_profile",
        "evaluation_protocol_id",
        "fixed_eval_set",
        "controller",
        "model_identity",
    )
    for field in identity_fields:
        values = [payload.get(field) for payload in payloads]
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"qualification runtime identity differs at {field}")
    rows = []
    for shard, payload in zip(ordered, payloads, strict=True):
        for index, row in zip(shard.indices, payload["results"], strict=True):
            value = dict(row)
            value["shard_local_episode"] = value.get("episode")
            value["episode"] = index
            value["qualification_episode_index"] = index
            rows.append(value)
    successes = sum(bool(row.get("success")) for row in rows)
    steps = np.asarray([int(row.get("steps", 0)) for row in rows], dtype=np.int64)
    result = {
        "schema": "essay2608.task_frame_qualification_result.v1",
        "task": task,
        "method": method,
        "paper_comparable": False,
        "benchmark_scope": "standard_task_frame_interface",
        "fixed_batch": {
            "path": str(motion_plans.relative_to(REPOSITORY_ROOT)),
            "sha256": batch_sha256,
            "batch_fingerprint": json.loads(
                motion_plans.read_text(encoding="utf-8")
            )["batch_fingerprint"],
        },
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / float(episodes),
        "reason_counts": dict(Counter(str(row.get("reason")) for row in rows)),
        "steps": {
            "mean": float(np.mean(steps)),
            "median": float(np.median(steps)),
            "max": int(np.max(steps)),
        },
        "runtime_identity": {
            field: payloads[0].get(field) for field in identity_fields
        },
        "results": rows,
        "shard_results": [
            str(shard.result.relative_to(REPOSITORY_ROOT)) for shard in ordered
        ],
        "exact_nonoverlapping_coverage": True,
    }
    output = output_root / f"{method}_n{episodes}.json"
    atomic_json(output, result)
    return output


def _terminate(processes: Sequence[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def execute(args: argparse.Namespace) -> dict[str, Any]:
    # Canonicalize path identity once so launch, resume and merge cannot differ
    # merely because the caller used repository-relative command-line paths.
    for name in (
        "motion_plans",
        "models_dir",
        "closed_loop_models_dir",
        "output_root",
        "sim_python",
        "policy_python",
    ):
        setattr(args, name, getattr(args, name).resolve())
    payload = json.loads(args.motion_plans.read_text(encoding="utf-8"))
    plans = load_staged_motion_plan_batch(payload)
    if payload.get("task_name") != args.task:
        raise ValueError("qualification batch task differs")
    episodes = len(plans)
    methods = tuple(dict.fromkeys(args.method or METHODS))
    all_shards = build_shards(
        task=args.task,
        methods=methods,
        episodes=episodes,
        shard_size=args.shard_size,
        output_root=args.output_root,
    )
    batch_sha256 = hashlib.sha256(args.motion_plans.read_bytes()).hexdigest()
    queue = []
    resumed = []
    for shard in all_shards:
        if shard.result.exists():
            _validate_shard(shard, batch_sha256=batch_sha256)
            resumed.append(shard.shard_id)
        else:
            queue.append(shard)
    specs = build_lane_specs(tuple(range(8)), args.workers)
    available = list(range(len(specs)))
    active: dict[str, tuple[subprocess.Popen, QualificationShard, int, Any]] = {}
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_root = args.output_root / "launch" / run_id
    log_root.mkdir(parents=True, exist_ok=False)
    assignments = []

    def launch(shard: QualificationShard, lane: int) -> None:
        spec = specs[lane]
        shard.result.parent.mkdir(parents=True, exist_ok=True)
        shard.diagnostics.mkdir(parents=True, exist_ok=True)
        log = (log_root / f"{shard.name}.log").open("xb")
        command = _command(
            shard,
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            models_dir=args.models_dir,
            closed_loop_models_dir=args.closed_loop_models_dir,
            motion_plans=args.motion_plans,
            horizon=args.horizon,
        )
        xvfb = (
            str(v4_formal_launch.DEFAULT_XVFB_RUN),
            "--auto-servernum",
            "--server-num",
            str(340 + lane),
            f"--error-file={log_root / (shard.name + '.xvfb.log')}",
            "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
            *command,
        )

        def affinity() -> None:
            os.sched_setaffinity(0, set(spec.logical_cpus))

        process = subprocess.Popen(
            xvfb,
            cwd=REPOSITORY_ROOT,
            env=_environment(args.policy_python, spec.gpu, spec.logical_cpus),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=affinity,
        )
        active[shard.shard_id] = (process, shard, lane, log)
        assignments.append(
            {
                "shard_id": shard.shard_id,
                "lane": lane,
                "gpu": spec.gpu,
                "pid": process.pid,
                "started_unix": time.time(),
            }
        )

    try:
        while queue or active:
            while queue and available:
                launch(queue.pop(0), available.pop(0))
            for shard_id, (process, shard, lane, log) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                del active[shard_id]
                available.append(lane)
                available.sort()
                if return_code != 0:
                    _terminate([value[0] for value in active.values()])
                    raise RuntimeError(
                        f"qualification shard failed: {shard_id}; see {log_root}"
                    )
                _validate_shard(shard, batch_sha256=batch_sha256)
                next(row for row in assignments if row["shard_id"] == shard_id)[
                    "finished_unix"
                ] = time.time()
            if queue or active:
                time.sleep(0.5)
    except BaseException:
        _terminate([value[0] for value in active.values()])
        raise
    finally:
        for process, _shard, _lane, log in active.values():
            if not log.closed:
                log.close()

    outputs = {}
    for method in methods:
        outputs[method] = str(
            merge_method(
                task=args.task,
                method=method,
                episodes=episodes,
                shards=[shard for shard in all_shards if shard.method == method],
                motion_plans=args.motion_plans,
                output_root=args.output_root,
            )
        )
    summary = {
        "schema": "essay2608.work_conserving_qualification_launch.v1",
        "workers": args.workers,
        "shard_size": args.shard_size,
        "work_conserving_global_queue": True,
        "methods": list(methods),
        "resumed_shards": resumed,
        "assignments": assignments,
        "outputs": outputs,
    }
    atomic_json(log_root / "launch_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="wipe_desk")
    parser.add_argument("--method", action="append", choices=tuple(METHODS))
    parser.add_argument("--motion-plans", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--closed-loop-models-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = execute(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
