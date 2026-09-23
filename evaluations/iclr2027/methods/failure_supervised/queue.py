"""Detached, bounded GPU queue: 30 Main-10 checkpoints first, then 84 E3 jobs."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import subprocess
import sys
import time

from evaluations.iclr2027.interfaces.failure_train import load_failure_train_manifest

from .runtime import sha256
from .train_cli import CONFIG, MANIFEST, ROOT, TRAINING, code_identity, config, write_json

MODULE = "evaluations.iclr2027.methods.failure_supervised.train_cli"
QUEUE_RUN = "queue_20260909_current_executor"


def jobs() -> tuple[list[dict], list[dict]]:
    cfg = config()
    main = [
        {"task": task, "seed": seed, "budget": 200}
        for task in cfg["main10"]
        for seed in cfg["training_seeds"]
    ]
    e3 = [
        {"task": task, "seed": seed, "budget": budget}
        for task in cfg["stress4"]
        for budget in (20, 50, 100)
        for seed in cfg["training_seeds"]
    ]
    rows = load_failure_train_manifest(MANIFEST)
    for task in cfg["stress4"]:
        families = sorted({r["fault_family"] for r in rows if r["task"] == task})
        for family in families:
            for seed in cfg["training_seeds"]:
                e3.append({"task": task, "seed": seed, "held_out_family": family})
    if len(main) != 30 or len(e3) != 84:
        raise ValueError("manifest/config job counts differ from frozen execution plan")
    return main, e3


def arguments(job: dict) -> list[str]:
    return [
        item for key, value in job.items() for item in ("--" + key.replace("_", "-"), str(value))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--main10-only", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        parser.error("gpus must be unique numeric IDs")
    run = TRAINING / QUEUE_RUN
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = {
        "config_sha256": sha256(CONFIG),
        "code_sha256": code_identity(),
        "manifest_sha256": sha256(MANIFEST),
    }
    main_jobs, e3_jobs = jobs()
    write_json(
        run / "schedule.json",
        {**frozen, "main10": main_jobs, "e3": [] if args.main10_only else e3_jobs, "gpus": gpus},
    )
    write_json(
        run / "packages.json",
        {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    )
    phases = [("prepare", [{"task": t} for t in config()["main10"]]), ("main10", main_jobs)]
    if not args.main10_only:
        phases.append(("e3", e3_jobs))
    status = {
        "status": "running",
        "pid": os.getpid(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": {},
        "failed": [],
        "running": {},
        **frozen,
    }
    for phase, pending in phases:
        status["phase"] = phase
        status["completed"][phase] = []
        next_job = 0
        active = {}
        while next_job < len(pending) or active:
            if (
                sha256(CONFIG) != frozen["config_sha256"]
                or code_identity() != frozen["code_sha256"]
            ):
                raise RuntimeError(
                    "training source/config changed while queue active; stop launching jobs"
                )
            for gpu in gpus:
                if gpu in active or next_job == len(pending):
                    continue
                job = pending[next_job]
                index = next_job
                next_job += 1
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="2",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    PYTHONHASHSEED=str(job.get("seed", 0)),
                )
                path = run / f"{phase}_{index:03d}.log"
                log = path.open("a")
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        MODULE,
                        "prepare" if phase == "prepare" else "train",
                        *arguments(job),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                active[gpu] = (proc, log, job, path)
            for gpu, (proc, log, job, path) in list(active.items()):
                result = proc.poll()
                if result is None:
                    continue
                log.close()
                del active[gpu]
                record = {**job, "exit_code": result, "log": str(path.relative_to(ROOT))}
                if result:
                    status["failed"].append({"phase": phase, **record})
                else:
                    if phase != "prepare":
                        golden = subprocess.run(
                            [sys.executable, "-u", "-m", MODULE, "golden", *arguments(job)],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1"),
                        )
                        if golden.returncode:
                            record["golden_error"] = golden.stderr[-4000:]
                            status["failed"].append({"phase": phase + "_golden", **record})
                    status["completed"][phase].append(record)
            status["running"] = {gpu: {"pid": p.pid, **j} for gpu, (p, _, j, _) in active.items()}
            status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json(run / "status.json", status)
            if active:
                time.sleep(5)
        if status["failed"]:
            status["status"] = "failed_requires_inspection"
            write_json(run / "status.json", status)
            raise SystemExit(1)
    status["status"] = "pass"
    status["formal_calibration"] = "pending_A_only"
    status["A_integration_acceptance"] = "not_yet_run"
    write_json(run / "status.json", status)
    print(
        json.dumps(
            {
                "status": status["status"],
                "completed": {k: len(v) for k, v in status["completed"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
