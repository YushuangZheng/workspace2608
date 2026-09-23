"""A-only normal calibration for the representative Main-10 M4 checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluations.iclr2027.calibration.monitor import (
    episode_nonconformity,
    split_conformal_upper_threshold,
)
from evaluations.iclr2027.interfaces.feature_schema import validate_feature_record
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.registry import build_monitor, load_method_spec
from evaluations.iclr2027.runners.episode_io import load_cycles, load_episode

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "evaluations/iclr2027/configs/shared/monitor_calibration.json"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "evaluations/iclr2027/manifests/main10_normal_calibration.jsonl"
DEFAULT_RESULTS = REPOSITORY_ROOT / "evaluations/iclr2027/datasets/normal_calibration_candidates"
DEFAULT_METHOD = "m4_failure_supervised_runtime"
ARTIFACT_SCHEMA = "essay2608.iclr2027.monitor-calibration.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    raw = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream:
        stream.write(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def calibrate_m4(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    result_root: Path = DEFAULT_RESULTS,
    output_root: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    method: str = DEFAULT_METHOD,
) -> dict[str, Any]:
    config = _json(config_path)
    if (
        config.get("schema") != "essay2608.iclr2027.monitor-calibration-config.v1"
        or config.get("calibration_authority") != "server_a_normal_only"
        or config.get("model_weight_updates_allowed") is not False
    ):
        raise ValueError("invalid A-only normal calibration contract")
    expected = int(config["minimum_episodes_per_task"])
    alpha = float(config["normal_episode_false_alarm_budget"])
    spec = load_method_spec(method)
    training_seed = int(spec.monitor["training_seed"])
    training_budget = int(spec.monitor["training_budget"])
    held_out_family = spec.monitor.get("held_out_family")
    if output_root is None:
        if training_budget != 200 or held_out_family is not None:
            raise ValueError("non-E1 M4 calibration requires an explicit output path")
        output_root = (
            REPOSITORY_ROOT
            / f"evaluations/iclr2027/artifacts/calibration/monitors/m4/seed_{training_seed}/v1"
        )
    else:
        output_root = output_root.resolve()
    persistence = int(spec.monitor.get("persistence_cycles", 3))
    # The backend config is the frozen source of M4 persistence.
    backend = _json(REPOSITORY_ROOT / spec.monitor["backend_config"])
    persistence = int(backend["persistence"])
    manifest_rows = _rows(manifest_path)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        if (
            row.get("condition") != "nominal"
            or row.get("fault_family") is not None
            or row.get("split") != "normal_calibration"
        ):
            raise ValueError("M4 calibration manifest contains a non-normal row")
        by_task[str(row["task"])].append(row)
    if not by_task or set(map(len, by_task.values())) != {expected}:
        raise ValueError("M4 calibration requires exactly 50 episodes per task")

    task_artifacts = {}
    score_index = []
    for task, rows in sorted(by_task.items()):
        monitor = build_monitor(spec, task_id=task)
        if monitor is None:
            raise RuntimeError("M4 monitor factory returned no monitor")
        statistics = []
        total_cycles = 0
        for row in rows:
            result_path = result_root / str(row["source_result"])
            cycle_path = result_root / str(row["source_cycle_file"])
            result = load_episode(result_path)
            if (
                _sha256(result_path) != row["source_result_sha256"]
                or _sha256(cycle_path) != row["source_cycle_sha256"]
                or result.get("method_id") != "m0_dynamac"
                or result.get("condition") != "nominal"
                or not result.get("success")
            ):
                raise ValueError(f"invalid M4 calibration source: {row['episode_id']}")
            cycles = load_cycles(cycle_path)
            source_episode_id = str(row["source_episode_id"])
            if cycles[0]["feature"]["episode_id"] != source_episode_id:
                raise ValueError("M4 calibration source episode identity mismatch")
            monitor.reset(
                EpisodeContext(
                    episode_id=source_episode_id,
                    task_id=task,
                    method_id=spec.method_id,
                    bimanual=len(cycles[0]["feature"]["arms"]) == 2,
                    horizon=int(row["horizon"]),
                    feature_schema=str(cycles[0]["feature"]["schema"]),
                    method_config_hash=spec.config_sha256,
                    checkpoint_hash=getattr(monitor, "checkpoint_hash", None),
                )
            )
            scores = []
            score_rows = []
            for expected_cycle, cycle in enumerate(cycles):
                feature = validate_feature_record(cycle["feature"])
                if feature["cycle"] != expected_cycle:
                    raise ValueError("M4 calibration cycles are not contiguous")
                monitor.observe_record(feature)
                score = float(monitor.score()[backend["score_name"]])
                scores.append(score)
                score_rows.append(
                    {
                        "episode_id": row["episode_id"],
                        "cycle": expected_cycle,
                        "score": score,
                    }
                )
            statistic = episode_nonconformity(
                scores, [True] * len(scores), persistence
            )
            if statistic is None:
                raise ValueError("M4 calibration episode is shorter than persistence")
            statistics.append(statistic)
            total_cycles += len(scores)
            safe = str(row["episode_id"]).replace("/", "__")
            score_path = output_root / "scores" / task / f"{safe}.jsonl.gz"
            _write_gzip_jsonl(score_path, score_rows)
            score_index.append(
                {
                    "episode_id": row["episode_id"],
                    "task": task,
                    "cycles": len(scores),
                    "episode_nonconformity": statistic,
                    "path": str(score_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(score_path),
                    "source_result_sha256": row["source_result_sha256"],
                    "source_cycle_sha256": row["source_cycle_sha256"],
                }
            )
        threshold, rank = split_conformal_upper_threshold(statistics, alpha)
        false_alarms = sum(value > threshold for value in statistics)
        task_artifacts[task] = {
            "threshold": threshold,
            "persistence_cycles": persistence,
            "calibration_episodes": len(statistics),
            "calibration_cycles": total_cycles,
            "conformal_rank_one_indexed": rank,
            "calibration_episode_false_alarms": false_alarms,
            "calibration_episode_false_alarm_rate": false_alarms / len(statistics),
            "checkpoint_sha256": getattr(monitor, "checkpoint_hash"),
        }

    score_index_path = output_root / "score_index.json"
    _write_json(
        score_index_path,
        {
            "schema": "essay2608.iclr2027.monitor-calibration-score-index.v1",
            "method_id": spec.method_id,
            "episodes": len(score_index),
            "files": score_index,
        },
    )
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "method_id": spec.method_id,
        "formal": True,
        "calibration_authority": "server_a_normal_only",
        "normal_episode_false_alarm_budget": alpha,
        "threshold_scope": "per_task",
        "threshold_rule": config["threshold_rule"],
        "method_config_identity": {
            "path": str(spec.config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": spec.config_sha256,
        },
        "calibration_config_identity": {
            "path": str(config_path.resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(config_path),
        },
        "calibration_manifest_identity": {
            "path": str(manifest_path.resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(manifest_path),
            "episodes": len(manifest_rows),
        },
        "score_index_identity": {
            "path": str(score_index_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(score_index_path),
        },
        "training_seed": training_seed,
        "training_budget": training_budget,
        "held_out_family": held_out_family,
        "tasks": task_artifacts,
        "model_weights_updated": False,
        "development_fault_labels_read": False,
        "sealed_test_read": False,
    }
    _write_json(output_root / "calibration.json", artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    args = parser.parse_args(argv)
    artifact = calibrate_m4(
        manifest_path=args.manifest,
        result_root=args.results,
        output_root=args.output,
        config_path=args.config,
        method=args.method,
    )
    print(
        json.dumps(
            {
                "tasks": len(artifact["tasks"]),
                "maximum_false_alarm_rate": max(
                    value["calibration_episode_false_alarm_rate"]
                    for value in artifact["tasks"].values()
                ),
                "output": artifact["score_index_identity"]["path"].replace(
                    "score_index.json", "calibration.json"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate_m4"]
