"""Materialize the preregistered 50-success-per-task calibration view."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluations.iclr2027.manifests.build import (
    INDEX_SCHEMA,
    MANIFEST_SCHEMA,
    ROOT,
    validate_all_manifests,
)
from evaluations.iclr2027.runners.episode_io import load_episode, resolve_cycle_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def materialize(
    candidate_manifest: Path,
    result_root: Path,
    output_manifest: Path,
    *,
    successes_per_task: int = 50,
) -> list[dict[str, Any]]:
    """Select the first successful candidates in preregistered manifest order.

    The output is a read-only view.  It points to immutable source artifacts;
    selection never rewrites causal cycle records or relabels a failed rollout.
    """

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    result_root = result_root.resolve()
    for source in _rows(candidate_manifest):
        task = str(source["task"])
        if counts[task] >= successes_per_task:
            continue
        safe = str(source["episode_id"]).replace("/", "__")
        result_path = result_root / "episodes" / (safe + ".json")
        if not result_path.is_file():
            continue
        result = load_episode(result_path)
        if result.get("reason") == "infrastructure_error" or not result.get("success"):
            continue
        cycle_path = resolve_cycle_file(result_path, result)
        if not cycle_path.is_file():
            raise FileNotFoundError(cycle_path)
        retained_index = counts[task]
        selected.append(
            {
                **source,
                "schema": MANIFEST_SCHEMA,
                "episode_id": f"normal_calibration/{task}/{retained_index:04d}",
                "pair_id": f"normal_calibration/{task}/{retained_index:04d}",
                "split": "normal_calibration",
                "readonly_view": True,
                "source_episode_id": source["episode_id"],
                "source_result": str(result_path.relative_to(result_root)),
                "source_result_sha256": _sha256(result_path),
                "source_cycle_file": str(cycle_path.relative_to(result_root)),
                "source_cycle_sha256": _sha256(cycle_path),
            }
        )
        counts[task] += 1
    tasks = sorted({str(row["task"]) for row in _rows(candidate_manifest)})
    missing = {task: successes_per_task - counts[task] for task in tasks if counts[task] < successes_per_task}
    if missing:
        raise RuntimeError(f"insufficient successful calibration candidates: {missing}")
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    validate_all_manifests(output_manifest.parent)
    index_path = output_manifest.parent / "MANIFEST_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("unknown manifest index schema")
    index["manifests"][output_manifest.name] = {
        "rows": len(selected),
        "sha256": _sha256(output_manifest),
        "materialized_from": candidate_manifest.name,
        "successes_per_task": successes_per_task,
    }
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT / "main10_normal_calibration_candidates.jsonl",
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=ROOT / "main10_normal_calibration.jsonl",
    )
    parser.add_argument("--successes-per-task", type=int, default=50)
    args = parser.parse_args(argv)
    rows = materialize(
        args.candidate_manifest,
        args.result_root,
        args.output_manifest,
        successes_per_task=args.successes_per_task,
    )
    print(json.dumps({"rows": len(rows), "manifest": str(args.output_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
