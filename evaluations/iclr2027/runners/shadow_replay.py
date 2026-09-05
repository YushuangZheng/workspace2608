"""Replay a frozen causal trajectory through one runtime monitor.

The monitor receives only ``cycle.feature``.  Physical-event audit fields are
deliberately neither read nor copied into the score artifact; label alignment
is a later analysis step.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import defaultdict
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
from evaluations.iclr2027.runners.shadow import shadow_passthrough_action


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "essay2608.iclr2027.shadow-score-index.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


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


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_shadow(
    *,
    manifest_path: Path,
    result_root: Path,
    output_root: Path,
    method: str,
    calibration_path: Path | None = None,
    condition: str | None = None,
    limit_per_task: int | None = None,
) -> dict[str, Any]:
    spec = load_method_spec(method)
    calibration = (
        None
        if calibration_path is None
        else json.loads(calibration_path.read_text(encoding="utf-8"))
    )
    rows = _manifest_rows(manifest_path)
    if condition is not None:
        rows = [row for row in rows if row.get("condition") == condition]
    if limit_per_task is not None:
        counts: dict[str, int] = defaultdict(int)
        selected = []
        for row in rows:
            task = str(row["task"])
            if counts[task] >= limit_per_task:
                continue
            counts[task] += 1
            selected.append(row)
        rows = selected
    if not rows:
        raise ValueError("no manifest rows selected for shadow replay")

    output_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for manifest_row in rows:
        episode_id = str(manifest_row["episode_id"])
        safe = episode_id.replace("/", "__")
        result_path = result_root / "episodes" / f"{safe}.json"
        result = load_episode(result_path)
        if result.get("episode_id") != episode_id:
            raise ValueError(f"result identity mismatch: {episode_id}")
        cycle_path = resolve_cycle_file(result_path, result)
        if _sha256(cycle_path) != result["cycle_file_sha256"]:
            raise ValueError(f"cycle hash mismatch: {episode_id}")
        cycles = load_cycles(cycle_path)
        monitor = build_monitor(
            spec,
            calibration=calibration,
            task_id=str(manifest_row["task"]),
        )
        if monitor is None:
            raise ValueError(f"method has no runtime monitor: {spec.method_id}")
        first_feature = validate_feature_record(cycles[0]["feature"])
        monitor.reset(
            EpisodeContext(
                episode_id=episode_id,
                task_id=str(manifest_row["task"]),
                method_id=spec.method_id,
                bimanual=len(first_feature["arms"]) == 2,
                horizon=int(manifest_row["horizon"]),
                feature_schema=str(first_feature["schema"]),
                method_config_hash=spec.config_sha256,
            )
        )
        score_rows = []
        input_actions = []
        passthrough_actions = []
        for expected_cycle, cycle in enumerate(cycles):
            feature = validate_feature_record(cycle["feature"])
            if int(feature["cycle"]) != expected_cycle:
                raise ValueError(f"non-contiguous cycles: {episode_id}")
            input_actions.append(feature["action"])
            passthrough, diagnostic = shadow_passthrough_action(monitor, feature)
            passthrough_actions.append(passthrough)
            score_rows.append({"episode_id": episode_id, **diagnostic})
        input_digest = _canonical_digest(input_actions)
        passthrough_digest = _canonical_digest(passthrough_actions)
        if input_digest != passthrough_digest:
            raise RuntimeError(f"shadow monitor changed an action: {episode_id}")
        score_path = output_root / "scores" / f"{safe}.jsonl.gz"
        _write_gzip_jsonl(score_path, score_rows)
        alarm_cycles = [row["cycle"] for row in score_rows if row["alarm"]]
        entries.append(
            {
                "episode_id": episode_id,
                "task": manifest_row["task"],
                "condition": manifest_row["condition"],
                "cycles": len(score_rows),
                "alarm_cycles": len(alarm_cycles),
                "first_alarm_cycle": alarm_cycles[0] if alarm_cycles else None,
                "source_result_sha256": _sha256(result_path),
                "source_cycle_sha256": _sha256(cycle_path),
                "input_action_sha256": input_digest,
                "passthrough_action_sha256": passthrough_digest,
                "action_passthrough_verified": True,
                "score_path": str(score_path.relative_to(REPOSITORY_ROOT)),
                "score_sha256": _sha256(score_path),
            }
        )
    artifact = {
        "schema": SCHEMA,
        "method_id": spec.method_id,
        "method_config_identity": {
            "path": str(spec.config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": spec.config_sha256,
        },
        "calibration_identity": (
            None
            if calibration_path is None
            else {
                "path": str(calibration_path.resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(calibration_path),
            }
        ),
        "manifest_identity": {
            "path": str(manifest_path.resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(manifest_path),
        },
        "episodes": len(entries),
        "cycles": sum(entry["cycles"] for entry in entries),
        "alarm_episodes": sum(entry["first_alarm_cycle"] is not None for entry in entries),
        "action_passthrough_verified": all(
            entry["action_passthrough_verified"] for entry in entries
        ),
        "audit_fields_used_by_monitor": False,
        "source_files_modified": False,
        "files": entries,
    }
    index_path = output_root / "score_index.json"
    _write_json(index_path, artifact)
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--condition", choices=("nominal", "perturbed"))
    parser.add_argument("--limit-per-task", type=int)
    args = parser.parse_args(argv)
    artifact = replay_shadow(
        manifest_path=args.manifest.resolve(),
        result_root=args.result_root.resolve(),
        output_root=args.output_root.resolve(),
        method=args.method,
        calibration_path=(
            None
            if args.calibration_artifact is None
            else args.calibration_artifact.resolve()
        ),
        condition=args.condition,
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


__all__ = ["replay_shadow"]
