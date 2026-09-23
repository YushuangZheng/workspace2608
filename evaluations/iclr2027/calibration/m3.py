"""A-only normal calibration for the Main-10 FAIL-Detect monitor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluations.iclr2027.interfaces.feature_schema import validate_feature_record
from evaluations.iclr2027.methods.fail_detect.conformal import TimeVaryingConformalBand
from evaluations.iclr2027.methods.fail_detect.model import build_official_velocity_model
from evaluations.iclr2027.methods.fail_detect.preprocessing import FeatureLayout
from evaluations.iclr2027.methods.fail_detect.training import _prepared_features
from evaluations.iclr2027.methods.registry import load_method_spec
from evaluations.iclr2027.runners.episode_io import load_cycles, load_episode

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "evaluations/iclr2027/configs/shared/monitor_calibration.json"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "evaluations/iclr2027/manifests/main10_normal_calibration.jsonl"
DEFAULT_RESULTS = REPOSITORY_ROOT / "evaluations/iclr2027/datasets/normal_calibration_candidates"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "evaluations/iclr2027/artifacts/calibration/monitors/m3/v1"
BACKEND_CONFIG = REPOSITORY_ROOT / "evaluations/iclr2027/configs/methods/m3_fail_detect_main10.json"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "evaluations/iclr2027/artifacts/checkpoints/m3"
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


def _score_vectors(
    vectors: np.ndarray,
    *,
    model: Any,
    device: str,
    input_dim: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    prepared = _prepared_features(vectors.astype(np.float32, copy=False), input_dim)
    output = []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(prepared), batch_size):
            batch = torch.from_numpy(prepared[start : start + batch_size]).to(device)
            times = torch.zeros(len(batch), dtype=torch.long, device=device)
            velocity = model(batch, times)
            score = (batch + velocity).reshape(len(batch), -1).square().sum(dim=1)
            output.append(score.cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def _fixed_progress_trajectory(
    policy_steps: Sequence[int],
    scores: Sequence[float],
    *,
    horizon: int,
) -> np.ndarray:
    result = np.full(horizon, np.nan, dtype=np.float64)
    for step, score in zip(policy_steps, scores, strict=True):
        if not 0 <= int(step) < horizon:
            raise ValueError("calibration policy_step is outside the demonstration horizon")
        previous = result[int(step)]
        result[int(step)] = float(score) if np.isnan(previous) else max(previous, float(score))
    present = np.flatnonzero(np.isfinite(result))
    if not len(present) or present[0] != 0:
        raise ValueError("calibration rollout does not start at policy step zero")
    last = int(present[-1])
    if np.any(~np.isfinite(result[: last + 1])):
        raise ValueError("calibration rollout skips an interior DynaMAC policy step")
    # A normal episode may terminate as soon as RLBench reports task success,
    # before the final nominal policy tick.  Forward-filling only that
    # unreachable suffix defines a complete threshold schedule without adding
    # observations or affecting any pre-success alarm decision.
    result[last + 1 :] = result[last]
    return result


def calibrate_m3(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    result_root: Path = DEFAULT_RESULTS,
    output_root: Path = DEFAULT_OUTPUT,
    config_path: Path = DEFAULT_CONFIG,
    device: str = "cuda:0",
    batch_size: int = 512,
) -> dict[str, Any]:
    import torch

    calibration_config = _json(config_path)
    backend_config = _json(BACKEND_CONFIG)
    if (
        calibration_config.get("schema")
        != "essay2608.iclr2027.monitor-calibration-config.v1"
        or calibration_config.get("calibration_authority") != "server_a_normal_only"
        or calibration_config.get("model_weight_updates_allowed") is not False
    ):
        raise ValueError("invalid A-only normal calibration contract")
    expected = int(calibration_config["minimum_episodes_per_task"])
    alpha = float(calibration_config["normal_episode_false_alarm_budget"])
    spec = load_method_spec("m3_fail_detect_runtime")
    manifest_rows = _rows(manifest_path)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        if (
            row.get("condition") != "nominal"
            or row.get("fault_family") is not None
            or row.get("split") != "normal_calibration"
        ):
            raise ValueError("M3 calibration manifest contains a non-normal row")
        by_task[str(row["task"])].append(row)
    if not by_task or set(map(len, by_task.values())) != {expected}:
        raise ValueError("M3 calibration requires exactly 50 episodes per task")

    task_artifacts = {}
    score_index = []
    for task, rows in sorted(by_task.items()):
        checkpoint = CHECKPOINT_ROOT / task / "seed_1103" / "model.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = checkpoint_payload["metadata"]
        if (
            checkpoint_payload.get("schema") != "essay2608.iclr2027.m3-logpzo.v1"
            or metadata.get("task") != task
            or metadata.get("config_sha256") != _sha256(BACKEND_CONFIG)
            or metadata.get("fit_split") != "normal_demonstrations"
        ):
            raise ValueError(f"invalid M3 checkpoint identity for {task}")
        layout = FeatureLayout.from_dict(metadata["layout"])
        mean = np.asarray(metadata["normalizer"]["mean"], dtype=np.float32)
        std = np.asarray(metadata["normalizer"]["std"], dtype=np.float32)
        clip = float(backend_config["normalizer"]["clip"])
        feature_index = _json(REPOSITORY_ROOT / metadata["feature_index"]["path"])
        horizons = {int(value["selected_steps"]) for value in feature_index["episodes"]}
        if len(horizons) != 1:
            raise ValueError("demonstration feature horizons are not fixed")
        policy_horizon = horizons.pop()

        all_vectors = []
        boundaries = []
        episode_cycles = []
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
                raise ValueError(f"invalid M3 calibration source: {row['episode_id']}")
            cycles = load_cycles(cycle_path)
            start = len(all_vectors)
            steps = []
            for expected_cycle, cycle in enumerate(cycles):
                feature = validate_feature_record(cycle["feature"])
                if feature["cycle"] != expected_cycle:
                    raise ValueError("M3 calibration cycles are not contiguous")
                current_layout = FeatureLayout.from_record(feature)
                if current_layout != layout:
                    raise ValueError("M3 calibration/checkpoint feature layouts differ")
                all_vectors.append(layout.encode_validated(feature))
                step = feature["policy_state"].get("policy_step")
                if isinstance(step, bool) or not isinstance(step, int):
                    raise ValueError("M3 calibration record lacks integer policy_step")
                steps.append(step)
            boundaries.append((start, len(all_vectors)))
            episode_cycles.append(steps)
        normalized = np.clip(
            (np.stack(all_vectors).astype(np.float32) - mean) / std,
            -clip,
            clip,
        )
        model = build_official_velocity_model(int(backend_config["unet_input_channels"]))
        model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
        all_scores = _score_vectors(
            normalized,
            model=model,
            device=device,
            input_dim=int(backend_config["unet_input_channels"]),
            batch_size=batch_size,
        )
        trajectories = []
        for row, steps, (start, stop) in zip(rows, episode_cycles, boundaries, strict=True):
            scores = all_scores[start:stop]
            trajectories.append(
                _fixed_progress_trajectory(steps, scores, horizon=policy_horizon)
            )
            safe = str(row["episode_id"]).replace("/", "__")
            score_path = output_root / "scores" / task / f"{safe}.jsonl.gz"
            _write_gzip_jsonl(
                score_path,
                (
                    {
                        "episode_id": row["episode_id"],
                        "cycle": cycle,
                        "policy_step": int(step),
                        "score": float(score),
                    }
                    for cycle, (step, score) in enumerate(zip(steps, scores, strict=True))
                ),
            )
            score_index.append(
                {
                    "episode_id": row["episode_id"],
                    "task": task,
                    "cycles": len(scores),
                    "path": str(score_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(score_path),
                    "source_result_sha256": row["source_result_sha256"],
                    "source_cycle_sha256": row["source_cycle_sha256"],
                }
            )
        matrix = np.stack(trajectories)
        split = len(matrix) // 2
        band = TimeVaryingConformalBand.fit(
            matrix[:split], matrix[split:], alpha=alpha, modulation_kind="tfunc"
        )
        persistence = int(backend_config["persistence"])
        episode_alarms = 0
        for trajectory in matrix:
            streak = 0
            alarm = False
            for score, threshold in zip(trajectory, band.upper, strict=True):
                streak = streak + 1 if score > threshold else 0
                alarm = alarm or streak >= persistence
            episode_alarms += int(alarm)
        task_artifacts[task] = {
            "threshold_schedule": band.to_dict(),
            "schedule_axis": "dynamac_policy_step",
            "successful_suffix_rule": "forward_fill_last_observed_score",
            "persistence_cycles": persistence,
            "calibration_episodes": len(matrix),
            "mean_split_episodes": split,
            "width_split_episodes": len(matrix) - split,
            "policy_horizon": policy_horizon,
            "calibration_episode_false_alarms": episode_alarms,
            "calibration_episode_false_alarm_rate": episode_alarms / len(matrix),
            "checkpoint_sha256": _sha256(checkpoint),
        }
        del model, checkpoint_payload
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

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
        "threshold_scope": "per_task_time_varying",
        "threshold_rule": {
            "name": "official_functional_conformal_upper_band",
            "axis": "dynamac_policy_step",
            "strict_exceedance": True,
            "persistence_cycles": int(backend_config["persistence"]),
        },
        "method_config_identity": {
            "path": str(spec.config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": spec.config_sha256,
        },
        "backend_config_identity": {
            "path": str(BACKEND_CONFIG.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(BACKEND_CONFIG),
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
        "tasks": task_artifacts,
        "model_weights_updated": False,
        "development_fault_labels_read": False,
        "sealed_test_read": False,
    }
    artifact_path = output_root / "calibration.json"
    _write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    artifact = calibrate_m3(
        manifest_path=args.manifest,
        result_root=args.results,
        output_root=args.output,
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "tasks": len(artifact["tasks"]),
                "maximum_false_alarm_rate": max(
                    value["calibration_episode_false_alarm_rate"]
                    for value in artifact["tasks"].values()
                ),
                "output": str(Path(args.output) / "calibration.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate_m3"]
