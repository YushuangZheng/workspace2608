"""Summarize the frozen stage-six RLBench component pilot.

The script is intentionally deterministic and reads only already-produced
normal/fault results.  It does not run the simulator or tune policy values.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPOSITORY_ROOT / "evaluations/phase6_rlbench_integration"
RAW_RESULT_ROOT = REPOSITORY_ROOT / "integrations/rlbench/results/diagnostics"
OUTPUT_ROOT = EVALUATION_ROOT / "results/v1"
RAW_ARCHIVE_ROOT = OUTPUT_ROOT / "raw_results"
METHODS = (
    "dynamac_v4",
    "progress_only",
    "progress_dynamic_roles",
    "full",
)
FAULTS = (
    "time_stall",
    "grasp_failure",
    "relation_mismatch",
    "unexpected_drop",
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_token(value: str) -> tuple[int, int]:
    skill, local = value.removeprefix("k").split(":t", maxsplit=1)
    return int(skill), int(local)


def _jsonl(path: Path) -> list[dict]:
    rows = []
    if path.suffix == ".gz":
        stream_context = gzip.open(path, "rt", encoding="utf-8")
    else:
        stream_context = path.open("r", encoding="utf-8")
    with stream_context as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _result_source(name: str) -> Path:
    runtime_path = RAW_RESULT_ROOT / name
    archive_path = RAW_ARCHIVE_ROOT / name
    if runtime_path.is_file():
        return runtime_path
    if archive_path.is_file():
        return archive_path
    raise FileNotFoundError(name)


def _archive_result(path: Path) -> Path:
    RAW_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_ARCHIVE_ROOT / path.name
    if path != archive_path:
        shutil.copyfile(path, archive_path)
    return archive_path


def _fault_result_path(fault: str, method: str) -> Path:
    return _result_source(f"phase6_fault_stack_wine_{fault}_{method}_final_v21.json")


def _diagnostic_dir(fault: str, method: str) -> Path:
    return EVALUATION_ROOT / "diagnostics" / f"fault_{fault}_{method}_final_v21"


def _trace_paths(diagnostic_dir: Path) -> list[Path]:
    plain = list(diagnostic_dir.glob("episode_*.jsonl"))
    compressed = list(diagnostic_dir.glob("episode_*.jsonl.gz"))
    return sorted([*plain, *compressed])


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    normal_source = _result_source("phase6_normal_stack_wine_n3_final_v21.json")
    normal_path = _archive_result(normal_source)
    normal = _load_json(normal_path)
    normal_rows = normal["results"]
    if len(normal_rows) != 3 or not all(row["success"] for row in normal_rows):
        raise RuntimeError("final normal gate must contain three successful episodes")

    protocol_path = EVALUATION_ROOT / "fault_protocol.json"
    protocol_sha256 = _sha256(protocol_path)
    normal_traces = _trace_paths(
        EVALUATION_ROOT / "diagnostics/normal_stack_wine_n3_final_v21/stack_wine"
    )
    if len(normal_traces) != 3:
        raise RuntimeError("expected three final normal diagnostic traces")
    raw_paths = [normal_path, protocol_path, *normal_traces]
    fault_summaries = []
    csv_rows = []
    time_stall_rows = []
    recovery_action_counts: dict[str, dict[str, int]] = {}

    for fault in FAULTS:
        for method in METHODS:
            source_path = _fault_result_path(fault, method)
            path = _archive_result(source_path)
            raw_paths.append(path)
            result = _load_json(path)
            evaluation = result["fault_evaluation"]
            rows = result["results"]
            if evaluation["protocol_sha256"] != protocol_sha256:
                raise RuntimeError(f"fault protocol hash mismatch: {path}")
            if len(rows) != 3 or evaluation["episodes_fault_triggered"] != 3:
                raise RuntimeError(f"fault cell is incomplete: {path}")
            successes = sum(bool(row["success"]) for row in rows)
            summary = {
                "fault": fault,
                "method": method,
                "episodes": len(rows),
                "fault_triggered": evaluation["episodes_fault_triggered"],
                "successes": successes,
                "success_rate": successes / len(rows),
                "steps": [int(row["steps"]) for row in rows],
                "mean_steps": sum(int(row["steps"]) for row in rows) / len(rows),
                "reasons": dict(Counter(str(row["reason"]) for row in rows)),
                "result_path": str(path.relative_to(REPOSITORY_ROOT)),
                "result_sha256": _sha256(path),
            }
            fault_summaries.append(summary)
            csv_rows.append(summary)

            if method != "dynamac_v4":
                diagnostic_dir = _diagnostic_dir(fault, method) / "stack_wine"
                traces = _trace_paths(diagnostic_dir)
                if len(traces) != 3:
                    raise RuntimeError(
                        f"expected three diagnostic traces: {diagnostic_dir}"
                    )
                raw_paths.extend(traces)
                if method == "full":
                    sources = Counter()
                    for trace in traces:
                        sources.update(
                            row["arms"]["single"]["action"]["source"]
                            for row in _jsonl(trace)
                        )
                    recovery_action_counts[fault] = dict(sorted(sources.items()))

            if fault != "time_stall":
                continue
            for episode_index, row in enumerate(rows):
                events = row["physical_fault"]["events"]
                trigger = events[0]
                duration = int(trigger["duration_cycles"])
                if method == "dynamac_v4":
                    time_stall_rows.append(
                        {
                            "method": method,
                            "episode": episode_index,
                            "trigger_tick": int(trigger["policy_step"]),
                            "duration_cycles": duration,
                            "start_reference_state": "fixed_clock",
                            "end_reference_state": "fixed_clock",
                            "state_advancement": duration,
                            "measurement": "fixed_clock_semantics",
                        }
                    )
                    continue
                candidates = _trace_paths(_diagnostic_dir(fault, method) / "stack_wine")
                trace_path = next(
                    path
                    for path in candidates
                    if path.name.startswith(f"episode_{episode_index:04d}.jsonl")
                )
                trace_by_tick = {int(item["tick"]): item for item in _jsonl(trace_path)}
                start_tick = int(trigger["policy_step"])
                end_tick = start_tick + duration - 1
                start_state = trace_by_tick[start_tick]["arms"]["single"][
                    "reference_state"
                ]
                end_state = trace_by_tick[end_tick]["arms"]["single"]["reference_state"]
                start_skill, start_local = _state_token(start_state)
                end_skill, end_local = _state_token(end_state)
                if start_skill != end_skill:
                    raise RuntimeError("time-stall window unexpectedly crossed a skill")
                time_stall_rows.append(
                    {
                        "method": method,
                        "episode": episode_index,
                        "trigger_tick": start_tick,
                        "duration_cycles": duration,
                        "start_reference_state": start_state,
                        "end_reference_state": end_state,
                        "state_advancement": end_local - start_local,
                        "measurement": "diagnostic_reference_state",
                    }
                )

    summary = {
        "schema": "essay2608.phase6_rlbench_component_pilot.v1",
        "status": "diagnostic_component_pilot",
        "formal_result": False,
        "paper_comparable": False,
        "task": "stack_wine",
        "sealed_evaluation_set": "rlbench_eval_v2",
        "controller": {
            "profile": normal["controller"]["profile"],
            "protocol_id": normal["controller"]["protocol_id"],
        },
        "normal_gate": {
            "episodes": len(normal_rows),
            "successes": sum(bool(row["success"]) for row in normal_rows),
            "success_rate": 1.0,
            "steps": [int(row["steps"]) for row in normal_rows],
            "result_path": str(normal_path.relative_to(REPOSITORY_ROOT)),
            "result_sha256": _sha256(normal_path),
            "false_auxiliary_mode_cycles": 0,
        },
        "fault_protocol": {
            "path": str(protocol_path.relative_to(REPOSITORY_ROOT)),
            "sha256": protocol_sha256,
            "fault_data_used_for_core_parameter_tuning": False,
        },
        "fault_results": fault_summaries,
        "time_stall_state_advancement": time_stall_rows,
        "full_method_action_source_counts": recovery_action_counts,
        "interpretation_limits": [
            "Three sealed StackWine episodes per cell are insufficient for a formal success-rate claim.",
            "The pilot validates controlled component behavior, not statistical significance across all tasks and variations.",
            "Fault samples were not used to calibrate relation, progress, boundary, or recovery thresholds.",
        ],
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (OUTPUT_ROOT / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "fault",
                "method",
                "episodes",
                "fault_triggered",
                "successes",
                "success_rate",
                "mean_steps",
                "result_path",
                "result_sha256",
            ),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    with (OUTPUT_ROOT / "time_stall_state_advancement.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(time_stall_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(time_stall_rows)

    manifest_paths = {
        *raw_paths,
        summary_path,
        OUTPUT_ROOT / "summary.csv",
        OUTPUT_ROOT / "time_stall_state_advancement.csv",
        EVALUATION_ROOT / "RESULTS.md",
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "configs/closed_loop_recovery.json",
        REPOSITORY_ROOT / "source/policy/closed_loop/recovery.py",
        REPOSITORY_ROOT / "integrations/rlbench/rlbench_dynamac/core/runtime.py",
        *(REPOSITORY_ROOT / "integrations/rlbench/models/closed_loop_v1").glob("*/*"),
    }
    manifest_lines = [
        f"{_sha256(path)}  {path.relative_to(REPOSITORY_ROOT)}"
        for path in sorted(manifest_paths)
        if path.is_file()
    ]
    (OUTPUT_ROOT / "SHA256SUMS").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
