"""Normal-only split-conformal calibration for the M2 trajectory monitor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.interfaces.feature_schema import validate_feature_record
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.registry import build_monitor, load_method_spec
from evaluations.iclr2027.runners.episode_io import (
    load_cycles,
    load_episode,
    resolve_cycle_file,
)
from evaluations.iclr2027.runners.shadow import shadow_observe
from evaluations.iclr2027.calibration.m2_streams import reconstruct_stream_marginals


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/configs/shared/monitor_calibration.json"
)
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/manifests/main10_normal_calibration.jsonl"
)
DEFAULT_RESULT_ROOT = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/datasets/normal_calibration_candidates"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/artifacts/calibration/monitors/m2/v1"
)
ARTIFACT_SCHEMA = "essay2608.iclr2027.monitor-calibration.v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_deterministic_gzip_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream:
        stream.write(raw)
    path.write_bytes(buffer.getvalue())


def episode_nonconformity(
    scores: Iterable[float],
    available: Iterable[bool],
    persistence_cycles: int,
) -> float | None:
    """Highest threshold that one persisted high-score run could exceed."""

    if persistence_cycles < 1:
        raise ValueError("persistence must be positive")
    window: deque[float] = deque(maxlen=persistence_cycles)
    maximum: float | None = None
    for score, usable in zip(scores, available, strict=True):
        if not usable:
            window.clear()
            continue
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("calibration score must be finite")
        window.append(value)
        if len(window) == persistence_cycles:
            candidate = min(window)
            maximum = candidate if maximum is None else max(maximum, candidate)
    return maximum


def split_conformal_upper_threshold(
    episode_statistics: Iterable[float],
    false_alarm_budget: float,
) -> tuple[float, int]:
    """Return the finite-sample upper split-conformal order statistic."""

    values = sorted(float(value) for value in episode_statistics)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("conformal calibration requires finite episode statistics")
    if not 0.0 < false_alarm_budget < 1.0:
        raise ValueError("false-alarm budget must lie in (0, 1)")
    rank = math.ceil((len(values) + 1) * (1.0 - false_alarm_budget))
    if rank > len(values):
        raise ValueError(
            "calibration set is too small for the requested finite threshold"
        )
    return values[rank - 1], rank


def calibrate_m2(
    *,
    manifest_path: Path,
    result_root: Path,
    output_root: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = _json(config_path)
    if config.get("schema") != "essay2608.iclr2027.monitor-calibration-config.v1":
        raise ValueError("unsupported monitor calibration config")
    if config.get("calibration_authority") != "server_a_normal_only":
        raise ValueError("formal monitor calibration must remain on server A")
    if config.get("model_weight_updates_allowed") is not False:
        raise ValueError("monitor calibration cannot update model weights")
    spec = load_method_spec("m2_trajectory_likelihood")
    persistence = int(spec.monitor["persistence_cycles"])
    alpha = float(config["normal_episode_false_alarm_budget"])
    expected = int(config["minimum_episodes_per_task"])
    manifest_rows = _rows(manifest_path)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        if (
            row.get("condition") != "nominal"
            or row.get("fault_family") is not None
            or row.get("split") != "normal_calibration"
        ):
            raise ValueError("calibration manifest contains a non-normal row")
        by_task[str(row["task"])].append(row)
    if not by_task or set(map(len, by_task.values())) != {expected}:
        raise ValueError("calibration manifest must contain exactly 50 rows per task")

    output_root.mkdir(parents=True, exist_ok=True)
    score_index = []
    task_artifacts: dict[str, Any] = {}
    total_cycles = 0
    total_available = 0
    for task, rows in sorted(by_task.items()):
        statistics = []
        task_available = 0
        task_cycles = 0
        for row in rows:
            source_episode_id = str(row.get("source_episode_id", row["episode_id"]))
            safe = source_episode_id.replace("/", "__")
            result_reference = row.get("source_result")
            result_path = (
                result_root / str(result_reference)
                if result_reference is not None
                else result_root / "episodes" / f"{safe}.json"
            )
            result = load_episode(result_path)
            expected_result_hash = row.get("source_result_sha256")
            if (
                expected_result_hash is not None
                and _sha256(result_path) != expected_result_hash
            ):
                raise ValueError(f"manifest result hash mismatch: {row['episode_id']}")
            if (
                result.get("episode_id") != source_episode_id
                or result.get("method_id") not in {"m0_dynamac", spec.method_id}
                or not result.get("success")
                or result.get("condition") != "nominal"
            ):
                raise ValueError(f"invalid M2 calibration replay: {row['episode_id']}")
            cycle_reference = row.get("source_cycle_file")
            cycle_path = (
                result_root / str(cycle_reference)
                if cycle_reference is not None
                else resolve_cycle_file(result_path, result)
            )
            if _sha256(cycle_path) != result["cycle_file_sha256"]:
                raise ValueError(f"cycle hash mismatch: {row['episode_id']}")
            expected_cycle_hash = row.get("source_cycle_sha256")
            if (
                expected_cycle_hash is not None
                and _sha256(cycle_path) != expected_cycle_hash
            ):
                raise ValueError(f"manifest cycle hash mismatch: {row['episode_id']}")
            cycles = load_cycles(cycle_path)
            reconstructed_features, reconstruction = reconstruct_stream_marginals(
                task, cycles
            )
            monitor = build_monitor(spec)
            if monitor is None:
                raise RuntimeError("M2 monitor factory returned no monitor")
            monitor.reset(
                EpisodeContext(
                    episode_id=str(row["episode_id"]),
                    task_id=task,
                    method_id=spec.method_id,
                    bimanual=len(cycles[0]["feature"]["arms"]) == 2,
                    horizon=int(row["horizon"]),
                    feature_schema=str(cycles[0]["feature"]["schema"]),
                    method_config_hash=spec.config_sha256,
                )
            )
            scores = []
            available = []
            score_rows = []
            for expected_cycle, reconstructed in enumerate(reconstructed_features):
                feature = validate_feature_record(reconstructed)
                if int(feature["cycle"]) != expected_cycle:
                    raise ValueError(f"non-contiguous cycles: {row['episode_id']}")
                diagnostic = shadow_observe(monitor, feature)
                score = float(diagnostic["scores"]["standardized_nll"])
                usable = diagnostic["scores"]["available_streams"] > 0.0
                scores.append(score)
                available.append(usable)
                score_rows.append(
                    {
                        "episode_id": row["episode_id"],
                        "cycle": expected_cycle,
                        "score": score,
                        "available": bool(usable),
                    }
                )
            statistic = episode_nonconformity(scores, available, persistence)
            if statistic is None:
                raise ValueError(
                    f"M2 has no {persistence}-cycle usable window: {row['episode_id']}"
                )
            statistics.append(statistic)
            task_cycles += len(scores)
            task_available += sum(available)
            score_path = output_root / "scores" / task / f"{safe}.jsonl.gz"
            _write_deterministic_gzip_jsonl(score_path, score_rows)
            score_index.append(
                {
                    "episode_id": row["episode_id"],
                    "task": task,
                    "cycles": len(scores),
                    "available_cycles": sum(available),
                    "episode_nonconformity": statistic,
                    "path": str(score_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(score_path),
                    "source_result_sha256": _sha256(result_path),
                    "source_cycle_sha256": _sha256(cycle_path),
                    "stream_reconstruction": reconstruction,
                }
            )
        threshold, rank = split_conformal_upper_threshold(statistics, alpha)
        false_alarms = sum(value > threshold for value in statistics)
        task_artifacts[task] = {
            "threshold": threshold,
            "persistence_cycles": persistence,
            "calibration_episodes": len(statistics),
            "calibration_cycles": task_cycles,
            "available_cycles": task_available,
            "conformal_rank_one_indexed": rank,
            "calibration_episode_false_alarms": false_alarms,
            "calibration_episode_false_alarm_rate": false_alarms / len(statistics),
            "minimum_episode_nonconformity": min(statistics),
            "maximum_episode_nonconformity": max(statistics),
        }
        total_cycles += task_cycles
        total_available += task_available

    index_path = output_root / "score_index.json"
    _write_json(
        index_path,
        {
            "schema": "essay2608.iclr2027.monitor-calibration-score-index.v1",
            "method_id": spec.method_id,
            "episodes": len(score_index),
            "cycles": total_cycles,
            "available_cycles": total_available,
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
            "path": str(index_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(index_path),
        },
        "implementation_identity": {
            "calibrator": {
                "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "stream_reconstruction": {
                "path": str(
                    Path(reconstruct_stream_marginals.__code__.co_filename)
                    .resolve()
                    .relative_to(REPOSITORY_ROOT)
                ),
                "sha256": _sha256(
                    Path(reconstruct_stream_marginals.__code__.co_filename).resolve()
                ),
            },
        },
        "tasks": task_artifacts,
        "model_weights_updated": False,
        "development_fault_labels_read": False,
        "sealed_test_read": False,
    }
    artifact_path = output_root / "calibration.json"
    _write_json(artifact_path, artifact)
    report_lines = [
        "# M2 正常样本正式标定",
        "",
        "仅使用服务器 A 保存的 500 条成功 nominal 回放；未更新模型权重，未读取故障标签或 sealed test。",
        "",
        "| 任务 | 阈值 | 连续周期 | 正常回放误报警 | 可用/总周期 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, value in task_artifacts.items():
        report_lines.append(
            "| %s | %.9g | %d | %d/%d | %d/%d |"
            % (
                task,
                value["threshold"],
                value["persistence_cycles"],
                value["calibration_episode_false_alarms"],
                value["calibration_episodes"],
                value["available_cycles"],
                value["calibration_cycles"],
            )
        )
    (output_root / "RESULTS.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    hashed = [
        path
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output_root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output_root)}\n" for path in hashed
        ),
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    artifact = calibrate_m2(
        manifest_path=args.manifest.resolve(),
        result_root=args.result_root.resolve(),
        output_root=args.output_root.resolve(),
        config_path=args.config.resolve(),
    )
    print(
        json.dumps(
            {
                "method_id": artifact["method_id"],
                "tasks": len(artifact["tasks"]),
                "episodes": artifact["calibration_manifest_identity"]["episodes"],
                "output": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
