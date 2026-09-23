"""Parallel episode-sharded shadow replay with persistent per-shard monitors."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
POLICY_PYTHON = Path(
    "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _task_counts(
    path: Path, condition: str, limit_per_task: int | None
) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("condition") == condition:
                task = str(row["task"])
                values[task] = values.get(task, 0) + 1
    if limit_per_task is not None:
        values = {task: min(count, limit_per_task) for task, count in values.items()}
    return dict(sorted(values.items()))


def _work_items(
    counts: dict[str, int], workers: int
) -> list[tuple[str, int, int]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    shard_counts = {task: 1 for task, count in counts.items() if count > 0}
    remaining = max(0, workers - len(shard_counts))
    tasks = list(shard_counts)
    while remaining:
        changed = False
        for task in tasks:
            if remaining == 0:
                break
            if shard_counts[task] >= counts[task]:
                continue
            shard_counts[task] += 1
            remaining -= 1
            changed = True
        if not changed:
            break
    return [
        (task, shard_index, shard_count)
        for task, shard_count in shard_counts.items()
        for shard_index in range(shard_count)
    ]


def _run_task(
    manifest: Path,
    result_root: Path,
    output_root: Path,
    method: str,
    calibration: Path | None,
    condition: str,
    task: str,
    shard_index: int,
    shard_count: int,
    limit_per_task: int | None,
    threads_per_worker: int,
) -> None:
    task_root = (
        output_root
        / "tasks"
        / task
        / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    )
    command = [
        str(POLICY_PYTHON),
        "-m",
        "evaluations.iclr2027.runners.shadow_replay",
        "--manifest",
        str(manifest),
        "--result-root",
        str(result_root),
        "--output-root",
        str(task_root),
        "--method",
        method,
        "--condition",
        condition,
        "--task",
        task,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]
    if limit_per_task is not None:
        command.extend(["--limit-per-task", str(limit_per_task)])
    if calibration is not None:
        command.extend(["--calibration-artifact", str(calibration)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "source:."
    environment["OMP_NUM_THREADS"] = str(threads_per_worker)
    environment["MKL_NUM_THREADS"] = str(threads_per_worker)
    log = (
        output_root
        / "logs"
        / f"{task}.shard_{shard_index:02d}_of_{shard_count:02d}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"shadow replay failed for {task} shard {shard_index}/{shard_count}; "
            f"see {log}"
        )


def run_parallel(
    *,
    manifest: Path,
    result_root: Path,
    output_root: Path,
    method: str,
    calibration: Path | None,
    condition: str,
    workers: int,
    limit_per_task: int | None = None,
) -> dict[str, Any]:
    counts = _task_counts(manifest, condition, limit_per_task)
    items = _work_items(counts, workers)
    active_workers = min(workers, len(items))
    threads_per_worker = max(1, min(8, 96 // active_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=active_workers) as pool:
        futures = {
            pool.submit(
                _run_task,
                manifest,
                result_root,
                output_root,
                method,
                calibration,
                condition,
                task,
                shard_index,
                shard_count,
                limit_per_task,
                threads_per_worker,
            ): (task, shard_index, shard_count)
            for task, shard_index, shard_count in items
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
    fragments = []
    for task, shard_index, shard_count in items:
        index = (
            output_root
            / "tasks"
            / task
            / f"shard_{shard_index:02d}_of_{shard_count:02d}"
            / "score_index.json"
        )
        fragments.append(json.loads(index.read_text(encoding="utf-8")))
    first = fragments[0]
    entries = [entry for fragment in fragments for entry in fragment["files"]]
    artifact = {
        **{key: value for key, value in first.items() if key != "files"},
        "schema": "essay2608.iclr2027.parallel-shadow-score-index.v1",
        "task_filter": None,
        "episode_shard": None,
        "task_shards": sorted(counts),
        "episode_shards": {
            task: sum(item[0] == task for item in items) for task in counts
        },
        "parallel_task_workers": active_workers,
        "parallel_episode_workers": active_workers,
        "threads_per_worker": threads_per_worker,
        "episodes": sum(fragment["episodes"] for fragment in fragments),
        "cycles": sum(fragment["cycles"] for fragment in fragments),
        "alarm_episodes": sum(fragment["alarm_episodes"] for fragment in fragments),
        "action_passthrough_verified": all(
            fragment["action_passthrough_verified"] for fragment in fragments
        ),
        "audit_fields_used_by_monitor": False,
        "source_files_modified": False,
        "shadow_code_identity": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "evaluations/iclr2027/runners/shadow.py",
                ROOT / "evaluations/iclr2027/runners/shadow_replay.py",
                ROOT / "evaluations/iclr2027/runners/shadow_parallel.py",
                ROOT / "evaluations/iclr2027/runners/ours_shadow.py",
            )
        },
        "files": entries,
    }
    _write_json(output_root / "score_index.json", artifact)
    files = [
        path
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output_root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output_root)}\n" for path in files),
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--condition", choices=("nominal", "perturbed"), required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit-per-task", type=int)
    args = parser.parse_args(argv)
    artifact = run_parallel(
        manifest=args.manifest.resolve(),
        result_root=args.result_root.resolve(),
        output_root=args.output_root.resolve(),
        method=args.method,
        calibration=(
            None
            if args.calibration_artifact is None
            else args.calibration_artifact.resolve()
        ),
        condition=args.condition,
        workers=args.workers,
        limit_per_task=args.limit_per_task,
    )
    print(
        json.dumps(
            {
                "method_id": artifact["method_id"],
                "episodes": artifact["episodes"],
                "cycles": artifact["cycles"],
                "alarm_episodes": artifact["alarm_episodes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
