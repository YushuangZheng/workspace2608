"""Resume-safe supervisor for the complete public Square reproduction chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

MODULE_ROOT = "evaluations.iclr2027.methods.fail_detect.reproduction"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/ubuntu/workspace/_runs/fail_detect/square_flow_seed1103_full_20260904"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/essay2608"),
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/ubuntu/workspace/_datasets/robomimic-v0.1/square/ph/image_abs.hdf5"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--rollouts-per-condition", type=int, default=2000)
    parser.add_argument("--rollouts-per-shard", type=int, default=500)
    parser.add_argument("--parallel-envs", type=int, default=25)
    return parser.parse_args()


def _last_jsonl(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return None if not lines else json.loads(lines[-1])


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _run(
    command: list[str],
    *,
    cwd: pathlib.Path,
    log: pathlib.Path,
    env: dict[str, str] | None = None,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"COMMAND {json.dumps(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}: {command}")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    if args.rollouts_per_condition % args.rollouts_per_shard:
        raise ValueError("rollouts-per-condition must be divisible by rollouts-per-shard")
    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = run_dir / "supervisor.log"
    status_path = run_dir / "reproduction_status.json"

    def status(stage: str, state: str, **extra: Any) -> None:
        payload = {"stage": stage, "state": state, "updated_unix": time.time(), **extra}
        temporary = status_path.with_name(f".{status_path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(status_path)
        with supervisor_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    policy_metrics = run_dir / "training_metrics.jsonl"
    policy_checkpoint = run_dir / "checkpoints/latest.ckpt"
    launcher_pid = int((run_dir / "launcher.pid").read_text(encoding="utf-8"))
    status("policy_training", "waiting", launcher_pid=launcher_pid)
    while True:
        latest = _last_jsonl(policy_metrics)
        if latest is not None and int(latest["epoch"]) == 799:
            break
        if not _pid_exists(launcher_pid):
            raise RuntimeError(
                f"policy trainer {launcher_pid} exited before epoch 800; latest={latest}"
            )
        time.sleep(args.poll_seconds)
    if not policy_checkpoint.is_file():
        raise FileNotFoundError(policy_checkpoint)
    status("policy_training", "complete", final_metrics=latest)

    features = run_dir / "square_data_flow.pt"
    if not features.is_file():
        status("feature_export", "running")
        export_env = dict(os.environ)
        export_env["CUDA_VISIBLE_DEVICES"] = "0"
        _run(
            [
                sys.executable,
                "-m",
                f"{MODULE_ROOT}.export_square_features",
                "--checkpoint",
                str(policy_checkpoint),
                "--dataset",
                str(args.dataset.resolve()),
                "--output",
                str(features),
                "--device",
                "cuda:0",
            ],
            cwd=project_root,
            log=run_dir / "feature_export.log",
            env=export_env,
        )
    status("feature_export", "complete", bytes=features.stat().st_size)

    logpzo_dir = run_dir / "logpzo"
    logpzo_checkpoint = logpzo_dir / "square_flow.ckpt"
    logpzo_metrics = logpzo_dir / "training_metrics.jsonl"
    logpzo_last = _last_jsonl(logpzo_metrics)
    if logpzo_last is None or int(logpzo_last["epoch"]) < 200:
        status("logpzo_training", "running", latest=logpzo_last)
        train_env = dict(os.environ)
        train_env.update({"OMP_NUM_THREADS": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        _run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "-m",
                f"{MODULE_ROOT}.distributed_logpzo_train",
                "--features",
                str(features),
                "--checkpoint",
                str(logpzo_checkpoint),
                "--metrics",
                str(logpzo_metrics),
                "--epochs",
                "200",
                "--seed",
                "1103",
                "--global-batch-size",
                "128",
            ],
            cwd=project_root,
            log=logpzo_dir / "launcher.log",
            env=train_env,
        )
    logpzo_last = _last_jsonl(logpzo_metrics)
    if logpzo_last is None or int(logpzo_last["epoch"]) != 200:
        raise RuntimeError(f"incomplete logpZO training: {logpzo_last}")
    status("logpzo_training", "complete", final_metrics=logpzo_last)

    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    shards = args.rollouts_per_condition // args.rollouts_per_shard
    jobs: list[tuple[subprocess.Popen[bytes], pathlib.Path, Any]] = []
    for condition_index, modify in enumerate((False, True)):
        condition = "ood" if modify else "nominal"
        for shard in range(shards):
            gpu = condition_index * shards + shard
            if gpu >= 8:
                raise ValueError("the requested rollout matrix needs more than eight GPUs")
            output = rollout_dir / f"{condition}_shard{shard:02d}.json"
            if output.is_file():
                continue
            log_path = rollout_dir / f"{condition}_shard{shard:02d}.log"
            stream = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                f"{MODULE_ROOT}.score_square_rollouts",
                "--policy-checkpoint",
                str(policy_checkpoint),
                "--logpzo-checkpoint",
                str(logpzo_checkpoint),
                "--dataset",
                str(args.dataset.resolve()),
                "--output",
                str(output),
                "--device",
                "cuda:0",
                "--start-seed",
                str(100000 + shard * args.rollouts_per_shard),
                "--num-rollouts",
                str(args.rollouts_per_shard),
                "--parallel-envs",
                str(args.parallel_envs),
            ]
            if modify:
                command.append("--modify")
            job_env = dict(os.environ)
            job_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "MUJOCO_GL": "osmesa",
                    "CC": "/usr/bin/gcc-12",
                    "CXX": "/usr/bin/g++-12",
                    "OMP_NUM_THREADS": "1",
                }
            )
            stream.write(f"COMMAND {json.dumps(command)}\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=job_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            jobs.append((process, output, stream))
    status("official_rollouts", "running", launched=len(jobs))
    failures = []
    for process, output, stream in jobs:
        returncode = process.wait()
        stream.close()
        if returncode or not output.is_file():
            failures.append({"pid": process.pid, "returncode": returncode, "output": str(output)})
    if failures:
        raise RuntimeError(f"rollout shard failures: {failures}")
    status("official_rollouts", "complete")

    nominal = sorted(rollout_dir.glob("nominal_shard*.json"))
    ood = sorted(rollout_dir.glob("ood_shard*.json"))
    calibration_result = run_dir / "square_logpzo_cp_alarm_result.json"
    status("calibration_alarm", "running")
    _run(
        [
            sys.executable,
            "-m",
            f"{MODULE_ROOT}.calibrate_square_logpzo",
            "--nominal",
            *[str(path) for path in nominal],
            "--ood",
            *[str(path) for path in ood],
            "--output",
            str(calibration_result),
        ],
        cwd=project_root,
        log=run_dir / "calibration_alarm.log",
    )

    hashes = {
        str(path.relative_to(run_dir)): _sha256(path)
        for path in [policy_checkpoint, features, logpzo_checkpoint, calibration_result]
    }
    final = {
        "status": "pass",
        "scope": "complete_official_square_flow_logpzo_cp_alarm_reproduction",
        "official_commit": "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed",
        "artifacts": hashes,
        "rollout_files": [str(path.relative_to(run_dir)) for path in nominal + ood],
    }
    final_path = run_dir / "final_reproduction_result.json"
    temporary = final_path.with_name(f".{final_path.name}.tmp")
    temporary.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    temporary.replace(final_path)
    status("complete", "pass", result=str(final_path))


if __name__ == "__main__":
    main()
