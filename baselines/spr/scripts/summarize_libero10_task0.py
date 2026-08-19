#!/data/yukun/miniconda3/envs/dynamac-spr/bin/python
"""Summarize an unmodified SPR LIBERO evaluator log without changing its protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_RUN_ROOT = Path(
    "/data/yukun/essay2608/baselines/spr/results/released_code_libero10_task0"
)
SPR_BASE = Path("/data/yukun/essay2608/baselines/spr")
EXPECTED_UPSTREAM_COMMIT = "d57e4b81ebdcacea574b68be29d61ba04cdc7051"
EXPECTED_CHECKPOINT_REVISION = "b5838d84d462abd41a45c2b3e7258fa11ec0ed0f"
EXPECTED_CODE_HASHES = {
    "upstream/experiments/libero/run_libero_eval_vllm.py": "f8785337c4711f5f40fe5961a788f06587f366117d252b89c60b6fec1c90f4fb",
    "upstream/experiments/libero/sprvla.py": "72f2da77c1d2a3145765221fa3b22dae569a7d715a48fa46a4876be93dc80681",
    "scripts/run_libero10_task0.sh": "1876c30d16030db41e811c4360df20365681b6bbead4fa2d40bffe7e24275e41",
}


def latest_log(run_root: Path) -> Path:
    logs = sorted(run_root.glob("*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"no evaluator logs found below {run_root}")
    return logs[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_header(text: str) -> dict[str, str | None]:
    def value(key: str) -> str | None:
        match = re.search(rf"^{re.escape(key)}=(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    return {
        "run_id": value("run_id"),
        "upstream_commit": value("upstream_commit"),
        "checkpoint_revision": value("checkpoint_revision"),
        "cuda_visible_devices": value("cuda_visible_devices"),
    }


def artifact_evidence(run_root: Path, log_path: Path) -> dict[str, object]:
    run_start = datetime.strptime(log_path.stem, "%Y%m%d_%H%M%S").timestamp()
    rollout_root = run_root / "rollouts"

    def files(kind: str, suffix: str) -> list[Path]:
        return sorted(
            path
            for path in rollout_root.glob(f"*/libero_10/libero_10/{kind}/*{suffix}")
            if path.stat().st_mtime >= run_start
        )

    annotation_files = files("annotations", ".json")
    raw_videos = files("raw", ".mp4")
    annotated_videos = files("annotated", ".mp4")
    records = [json.loads(path.read_text()) for path in annotation_files]
    records.sort(key=lambda record: int(record["episode"]))
    return {
        "annotation_json_count": len(records),
        "annotation_episode_ids": [int(record["episode"]) for record in records],
        "annotation_outcomes": [bool(record["success"]) for record in records],
        "raw_video_count": len(raw_videos),
        "annotated_video_count": len(annotated_videos),
    }


def identity_evidence(full_verify_checkpoint: bool) -> dict[str, object]:
    actual_hashes = {
        relative: sha256(SPR_BASE / relative) for relative in EXPECTED_CODE_HASHES
    }
    upstream_commit = subprocess.run(
        ["git", "-C", str(SPR_BASE / "upstream"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    upstream_clean = subprocess.run(
        ["git", "-C", str(SPR_BASE / "upstream"), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""

    checkpoint = SPR_BASE / "checkpoints/libero_10"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    checkpoint_evidence: dict[str, object] = {
        "indexed_tensors": len(index["weight_map"]),
        "declared_total_size_bytes": index["metadata"]["total_size"],
        "shards": shards,
        "missing_shards": [name for name in shards if not (checkpoint / name).is_file()],
        "full_sha256_verified": full_verify_checkpoint,
    }
    if full_verify_checkpoint:
        expected = json.loads((SPR_BASE / "metadata/checkpoint.json").read_text())["sha256"]
        actual = {name: sha256(checkpoint / name) for name in shards}
        checkpoint_evidence["sha256"] = actual
        checkpoint_evidence["sha256_matches_record"] = actual == expected

    return {
        "upstream_commit": upstream_commit,
        "upstream_worktree_clean": upstream_clean,
        "code_sha256": actual_hashes,
        "code_sha256_matches_record": actual_hashes == EXPECTED_CODE_HASHES,
        "checkpoint": checkpoint_evidence,
    }


def parse_log(log_path: Path, full_verify_checkpoint: bool = False) -> dict[str, object]:
    text = log_path.read_text(errors="replace")
    header = parse_header(text)
    episode_matches = re.findall(r"# episodes completed so far: (\d+)", text)
    success_matches = re.findall(r"# successes: (\d+) \(([0-9.]+)%\)", text)
    completed = int(episode_matches[-1]) if episode_matches else 0
    successes = int(success_matches[-1][0]) if success_matches else 0
    rate = float(success_matches[-1][1]) if success_matches else 0.0
    episode_outcomes = [value == "True" for value in re.findall(r"^Success: (True|False)$", text, re.MULTILINE)]
    artifacts = artifact_evidence(log_path.parent, log_path)
    identity = identity_evidence(full_verify_checkpoint)
    evidence_agrees = (
        len(episode_outcomes) == completed
        and sum(episode_outcomes) == successes
        and artifacts["annotation_json_count"] == completed
        and artifacts["annotation_episode_ids"] == list(range(1, completed + 1))
        and artifacts["annotation_outcomes"] == episode_outcomes
        and artifacts["raw_video_count"] == completed
        and artifacts["annotated_video_count"] == completed
    )
    identity_agrees = (
        header["upstream_commit"] == EXPECTED_UPSTREAM_COMMIT
        and header["checkpoint_revision"] == EXPECTED_CHECKPOINT_REVISION
        and identity["upstream_commit"] == EXPECTED_UPSTREAM_COMMIT
        and identity["upstream_worktree_clean"] is True
        and identity["code_sha256_matches_record"] is True
        and not identity["checkpoint"]["missing_shards"]
        and identity["checkpoint"]["indexed_tensors"] == 614
        and (
            not full_verify_checkpoint
            or identity["checkpoint"].get("sha256_matches_record") is True
        )
    )
    return {
        "schema": "dynamac-baseline-result-v1",
        "method": "SPR",
        "protocol_label": "released-code evaluator reproduction",
        "suite": "LIBERO-Long",
        "task_id": 0,
        "task_description": "put both the alphabet soup and the tomato sauce in the basket",
        "seed": 7,
        "expected_episodes": 50,
        "completed_episodes": completed,
        "successes": successes,
        "success_rate_percent": rate,
        "episode_outcomes": episode_outcomes,
        "complete": completed == 50,
        "source_log": str(log_path),
        "official_log_header": header,
        "independent_artifact_evidence": artifacts,
        "identity_evidence": identity,
        "verification": {
            "official_counts_match_saved_annotations_and_videos": evidence_agrees,
            "source_checkpoint_and_launcher_identity_match": identity_agrees,
            "pass": evidence_agrees and identity_agrees,
        },
        "paper_full_suite_spr_percent": 82.8,
        "comparison_warning": "Task-0 success is not directly comparable with the paper's ten-task suite average."
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-verify-checkpoint",
        action="store_true",
        help="Recompute SHA256 for all four checkpoint shards (about 16 GB).",
    )
    args = parser.parse_args()

    log_path = args.log or latest_log(args.run_root)
    summary = parse_log(log_path, full_verify_checkpoint=args.full_verify_checkpoint)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
