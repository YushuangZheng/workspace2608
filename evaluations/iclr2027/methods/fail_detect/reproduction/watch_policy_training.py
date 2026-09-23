"""Restart the 8-GPU Square policy trainer only after a verified process exit."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, TextIO

MODULE = (
    "evaluations.iclr2027.methods.fail_detect.reproduction."
    "distributed_policy_train"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/ubuntu/workspace/_runs/fail_detect/"
            "square_flow_seed1103_full_20260904"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/workspace/essay2608"),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=3)
    return parser.parse_args()


def _last_metric(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return None if not lines else json.loads(lines[-1])


def _pid_state(pid: int) -> str | None:
    stat = pathlib.Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return None
    fields = stat.read_text(encoding="utf-8").split()
    return fields[2] if len(fields) > 2 else None


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _launch(
    *, run_dir: pathlib.Path, project_root: pathlib.Path, stream: TextIO
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        MODULE,
        "--output-dir",
        str(run_dir),
        "--epochs",
        "800",
        "--seed",
        "1103",
        "--global-batch-size",
        "64",
        "--workers-per-rank",
        "2",
        "--checkpoint-every",
        "50",
        "--val-every",
        "1",
        "--sample-every",
        "5",
    ]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    stream.write(f"COMMAND {json.dumps(command)}\n")
    stream.flush()
    return subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def main() -> None:
    args = _parse_args()
    if args.poll_seconds <= 0 or args.max_restarts < 0:
        raise ValueError("invalid watchdog limits")
    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    metrics_path = run_dir / "training_metrics.jsonl"
    launcher_pid_path = run_dir / "launcher.pid"
    status_path = run_dir / "training_watchdog_status.json"
    restart_log = run_dir / "training_restarts.log"
    restart_count = 0
    child: subprocess.Popen[str] | None = None

    with restart_log.open("a", encoding="utf-8") as stream:
        while True:
            latest = _last_metric(metrics_path)
            if latest is not None and int(latest["epoch"]) == 799:
                _write_json(
                    status_path,
                    {
                        "state": "complete",
                        "restart_count": restart_count,
                        "latest": latest,
                        "updated_unix": time.time(),
                    },
                )
                return

            launcher_pid = int(launcher_pid_path.read_text(encoding="utf-8"))
            state = _pid_state(launcher_pid)
            if state is not None and state != "Z":
                _write_json(
                    status_path,
                    {
                        "state": "monitoring",
                        "launcher_pid": launcher_pid,
                        "launcher_state": state,
                        "restart_count": restart_count,
                        "latest": latest,
                        "updated_unix": time.time(),
                    },
                )
                time.sleep(args.poll_seconds)
                continue

            if child is not None:
                child.wait()
                child = None
            if restart_count >= args.max_restarts:
                _write_json(
                    status_path,
                    {
                        "state": "failed",
                        "reason": "restart_limit_exhausted",
                        "dead_launcher_pid": launcher_pid,
                        "restart_count": restart_count,
                        "latest": latest,
                        "updated_unix": time.time(),
                    },
                )
                raise RuntimeError("policy training restart limit exhausted")

            restart_count += 1
            child = _launch(
                run_dir=run_dir, project_root=project_root, stream=stream
            )
            temporary_pid = launcher_pid_path.with_name(".launcher.pid.tmp")
            temporary_pid.write_text(f"{child.pid}\n", encoding="utf-8")
            temporary_pid.replace(launcher_pid_path)
            _write_json(
                status_path,
                {
                    "state": "restarted",
                    "previous_launcher_pid": launcher_pid,
                    "launcher_pid": child.pid,
                    "restart_count": restart_count,
                    "resumed_from_checkpoint": (
                        run_dir / "checkpoints/latest.ckpt"
                    ).is_file(),
                    "latest": latest,
                    "updated_unix": time.time(),
                },
            )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
