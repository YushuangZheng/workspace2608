#!/data/yukun/miniconda3/envs/dynamac-spr/bin/python
"""Validate and aggregate all ten official SPR LIBERO-Long task runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

SPR_BASE = Path("/data/yukun/essay2608/baselines/spr")
DEFAULT_RESULTS = SPR_BASE / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "libero10_full"
EXPECTED_COMMIT = "d57e4b81ebdcacea574b68be29d61ba04cdc7051"
EXPECTED_CHECKPOINT_REVISION = "b5838d84d462abd41a45c2b3e7258fa11ec0ed0f"
EXPECTED_EVALUATOR_HASH = "f8785337c4711f5f40fe5961a788f06587f366117d252b89c60b6fec1c90f4fb"
EXPECTED_PARSER_HASH = "72f2da77c1d2a3145765221fa3b22dae569a7d715a48fa46a4876be93dc80681"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header_value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def latest_log(run_root: Path) -> Path:
    logs = sorted(run_root.glob("*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise RuntimeError(f"task run has no log: {run_root}")
    return logs[-1]


def output_files(run_root: Path, log_path: Path, kind: str, suffix: str) -> list[Path]:
    run_start = datetime.strptime(log_path.stem, "%Y%m%d_%H%M%S").timestamp()
    return sorted(
        path
        for path in (run_root / "rollouts").glob(
            f"*/libero_10/libero_10/{kind}/*{suffix}"
        )
        if path.stat().st_mtime >= run_start
    )


def validate_task(results_root: Path, task_id: int) -> dict[str, object]:
    run_root = results_root / f"released_code_libero10_task{task_id}"
    log_path = latest_log(run_root)
    text = log_path.read_text(errors="replace")
    completed_values = re.findall(r"# episodes completed so far: (\d+)", text)
    success_values = re.findall(r"# successes: (\d+) \(([0-9.]+)%\)", text)
    outcomes = [
        value == "True"
        for value in re.findall(r"^Success: (True|False)$", text, re.MULTILINE)
    ]
    if not completed_values or not success_values:
        raise RuntimeError(f"task {task_id}: official log has no completed result")
    completed = int(completed_values[-1])
    successes = int(success_values[-1][0])

    annotation_files = output_files(run_root, log_path, "annotations", ".json")
    records = [json.loads(path.read_text()) for path in annotation_files]
    records.sort(key=lambda record: int(record["episode"]))
    annotation_ids = [int(record["episode"]) for record in records]
    annotation_outcomes = [bool(record["success"]) for record in records]
    descriptions = sorted({str(record["task"]) for record in records})
    raw_videos = output_files(run_root, log_path, "raw", ".mp4")
    annotated_videos = output_files(run_root, log_path, "annotated", ".mp4")

    checks = {
        "official_episode_count_is_50": completed == 50,
        "official_outcomes_count_is_50": len(outcomes) == 50,
        "official_success_count_matches_outcomes": sum(outcomes) == successes,
        "annotation_count_is_50": len(records) == 50,
        "annotation_episode_ids_are_1_to_50": annotation_ids == list(range(1, 51)),
        "annotation_outcomes_match_official_log": annotation_outcomes == outcomes,
        "raw_video_count_is_50": len(raw_videos) == 50,
        "annotated_video_count_is_50": len(annotated_videos) == 50,
        "single_task_description": len(descriptions) == 1,
        "upstream_commit_matches": header_value(text, "upstream_commit") == EXPECTED_COMMIT,
        "checkpoint_revision_matches": header_value(text, "checkpoint_revision")
        == EXPECTED_CHECKPOINT_REVISION,
        "four_visible_gpus": len((header_value(text, "cuda_visible_devices") or "").split(","))
        == 4,
    }
    logged_task_id = header_value(text, "task_id")
    if logged_task_id is not None:
        checks["logged_task_id_matches"] = int(logged_task_id) == task_id
    logged_evaluator_hash = header_value(text, "evaluator_sha256")
    if logged_evaluator_hash is not None:
        checks["logged_evaluator_hash_matches"] = (
            logged_evaluator_hash == EXPECTED_EVALUATOR_HASH
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"task {task_id}: independent validation failed: {failed}")

    return {
        "task_id": task_id,
        "task_description": descriptions[0],
        "episodes": completed,
        "successes": successes,
        "success_rate_percent": successes / completed * 100.0,
        "official_log": str(log_path),
        "checks": checks,
    }


def wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return (center - half_width) * 100.0, (center + half_width) * 100.0


def validate_identity() -> dict[str, object]:
    upstream = SPR_BASE / "upstream"
    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""
    evaluator_hash = sha256(upstream / "experiments/libero/run_libero_eval_vllm.py")
    parser_hash = sha256(upstream / "experiments/libero/sprvla.py")

    checkpoint = SPR_BASE / "checkpoints/libero_10"
    checkpoint_record = json.loads((SPR_BASE / "metadata/checkpoint.json").read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    actual_shard_hashes = {name: sha256(checkpoint / name) for name in shards}
    checks = {
        "upstream_commit_matches": commit == EXPECTED_COMMIT,
        "upstream_worktree_clean": clean,
        "evaluator_hash_matches": evaluator_hash == EXPECTED_EVALUATOR_HASH,
        "parser_hash_matches": parser_hash == EXPECTED_PARSER_HASH,
        "checkpoint_has_614_indexed_tensors": len(index["weight_map"]) == 614,
        "checkpoint_declared_size_matches": index["metadata"]["total_size"]
        == 16_238_835_616,
        "checkpoint_shard_hashes_match": actual_shard_hashes
        == checkpoint_record["sha256"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"source/checkpoint identity validation failed: {failed}")
    return {
        "upstream_commit": commit,
        "evaluator_sha256": evaluator_hash,
        "parser_sha256": parser_hash,
        "checkpoint_revision": EXPECTED_CHECKPOINT_REVISION,
        "checkpoint_shard_sha256": actual_shard_hashes,
        "checks": checks,
    }


def write_csv(path: Path, task_rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "task_id",
                "task_description",
                "episodes",
                "successes",
                "success_rate_percent",
            ],
        )
        writer.writeheader()
        for row in task_rows:
            writer.writerow({name: row[name] for name in writer.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tasks = [validate_task(args.results_root, task_id) for task_id in range(10)]
    identity = validate_identity()
    total_episodes = sum(int(task["episodes"]) for task in tasks)
    total_successes = sum(int(task["successes"]) for task in tasks)
    if total_episodes != 500:
        raise RuntimeError(f"aggregate denominator must be 500, got {total_episodes}")
    ci_low, ci_high = wilson_interval(total_successes, total_episodes)
    summary = {
        "schema": "dynamac-baseline-aggregate-result-v1",
        "method": "SPR",
        "checkpoint": "SPRVLA/libero_10",
        "protocol_label": "released-code evaluator reproduction",
        "suite": "LIBERO-Long",
        "tasks": tasks,
        "aggregate": {
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate_percent": total_successes / total_episodes * 100.0,
            "wilson_95_ci_percent": [ci_low, ci_high],
        },
        "paper_reference": {
            "label": "SPR / Ours (separately trained)",
            "libero_long_success_rate_percent": 82.8,
            "scope": "ten-task aggregate",
            "ours_star_excluded": True,
            "exclusion_reason": "The public libero_10 checkpoint is not identified as the joint all-suite Ours* checkpoint."
        },
        "identity": identity,
        "verification": {
            "each_task_has_50_independently_matched_log_annotation_and_video_outcomes": True,
            "aggregate_denominator_is_500": True,
            "source_and_checkpoint_identity_match": True,
            "pass": True,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    task_csv_path = args.output_root / "task_results.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_csv(task_csv_path, tasks)
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"summary={summary_path}")
    print(f"tasks={task_csv_path}")


if __name__ == "__main__":
    main()
