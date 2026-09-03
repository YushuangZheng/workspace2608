"""Preflight, plan, and run the preregistered Stage-six evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval import v4_formal_launch
from integrations.rlbench.rlbench_dynamac.eval.eval_set import GLOBAL_EVAL_SEED_START

from .run_cell import PROTOCOL, load_protocol
from .resources import LaneSpec, build_lane_specs

RESULTS_ROOT = REPOSITORY_ROOT / "integrations/rlbench/results/phase6_formal_v1"
LAUNCH_ROOT = RESULTS_ROOT / "_launch"
RETAINED_RESULTS = Path(__file__).with_name("retained_results.json")
BASE_MODELS = REPOSITORY_ROOT / "integrations/rlbench/models/phase6_v1"
CLOSED_LOOP_MODELS = (
    REPOSITORY_ROOT / "integrations/rlbench/models/closed_loop_phase6_v1"
)
RUNNER_MODULE = "evaluations.development.phase6_formal_evaluation.run_cell"
DEFAULT_SIM_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_SIM_PYTHON",
        "/home/zhengyushuang/.conda/envs-migrated-20260816/dynamac-paper/bin/python",
    )
)
DEFAULT_POLICY_PYTHON = Path(
    os.environ.get(
        "DYNAMAC_POLICY_PYTHON",
        "/home/zhengyushuang/.conda/envs-migrated-20260816/RoboTwin/bin/python",
    )
)
DEFAULT_GPUS = tuple(range(8))
DEFAULT_WORKERS = 48
DEFAULT_SHARD_SIZE = 4


@dataclass(frozen=True)
class FormalCell:
    experiment: str
    task: str
    method: str
    fault: Optional[str]
    episodes: int

    @property
    def cell_id(self) -> str:
        parts = [self.experiment]
        if self.fault is not None:
            parts.append(self.fault)
        parts.extend((self.task, self.method))
        return "/".join(parts)

    @property
    def name(self) -> str:
        return self.cell_id.replace("/", "__")

    @property
    def result(self) -> Path:
        folder = RESULTS_ROOT / self.experiment
        if self.fault is not None:
            folder /= self.fault
        return folder / self.task / f"{self.method}_n{self.episodes}.json"

    @property
    def diagnostics(self) -> Path:
        return RESULTS_ROOT / "diagnostics" / self.name

    def command(self, sim_python: Path, policy_python: Path) -> tuple[str, ...]:
        values = [
            str(sim_python),
            "-m",
            RUNNER_MODULE,
            "--protocol",
            str(PROTOCOL),
            "--experiment",
            self.experiment,
            "--task",
            self.task,
            "--method",
            self.method,
            "--output",
            str(self.result),
            "--diagnostics-dir",
            str(self.diagnostics),
            "--policy-python",
            str(policy_python),
        ]
        if self.fault is not None:
            values.extend(("--fault", self.fault))
        return tuple(values)


@dataclass(frozen=True)
class FormalShard:
    """One resumable contiguous episode shard from a formal logical cell."""

    cell: FormalCell
    episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.episode_indices or self.episode_indices != tuple(
            range(self.episode_indices[0], self.episode_indices[-1] + 1)
        ):
            raise ValueError("formal shard indices must be non-empty and contiguous")

    @property
    def shard_id(self) -> str:
        return (
            f"{self.cell.cell_id}/episodes_"
            f"{self.episode_indices[0]:04d}_{self.episode_indices[-1]:04d}"
        )

    @property
    def name(self) -> str:
        return self.shard_id.replace("/", "__")

    @property
    def result(self) -> Path:
        return (
            RESULTS_ROOT
            / "_shards"
            / self.cell.name
            / f"episodes_{self.episode_indices[0]:04d}_{self.episode_indices[-1]:04d}.json"
        )

    @property
    def diagnostics(self) -> Path:
        return RESULTS_ROOT / "diagnostics" / "_shards" / self.name

    def command(self, sim_python: Path, policy_python: Path) -> tuple[str, ...]:
        values = list(self.cell.command(sim_python, policy_python))
        output_index = values.index("--output") + 1
        diagnostics_index = values.index("--diagnostics-dir") + 1
        values[output_index] = str(self.result)
        values[diagnostics_index] = str(self.diagnostics)
        values.extend(("--episode-indices", ",".join(map(str, self.episode_indices))))
        return tuple(values)


def build_shards(
    cells: Sequence[FormalCell],
    protocol: Mapping[str, Any],
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[FormalShard, ...]:
    """Partition every cell into deterministic contiguous queue jobs."""

    if shard_size < 1:
        raise ValueError("formal shard size must be positive")
    by_cell: list[list[FormalShard]] = []
    for cell in cells:
        key = f"{cell.experiment}_episode_index_range"
        first, last = (int(value) for value in protocol["evaluation_set"][key])
        expected = tuple(range(first, last + 1))
        if len(expected) != cell.episodes:
            raise ValueError(f"formal cell episode count differs: {cell.cell_id}")
        by_cell.append(
            [
                FormalShard(cell, expected[offset : offset + shard_size])
                for offset in range(0, len(expected), shard_size)
            ]
        )
    # Round-robin cells so the initial 48 slots cover every pending logical
    # cell.  Once launched, all freed slots still draw from this one global
    # queue; no lane is reserved for a task or method.
    shards = []
    for round_index in range(max((len(value) for value in by_cell), default=0)):
        shards.extend(
            value[round_index] for value in by_cell if round_index < len(value)
        )
    return tuple(shards)


def build_cells(protocol: Mapping[str, Any], section: str) -> tuple[FormalCell, ...]:
    normal_bounds = protocol["evaluation_set"]["normal_episode_index_range"]
    dynamic_bounds = protocol["evaluation_set"]["dynamic_episode_index_range"]
    fault_bounds = protocol["evaluation_set"]["fault_episode_index_range"]
    normal_n = int(normal_bounds[1]) - int(normal_bounds[0]) + 1
    dynamic_n = int(dynamic_bounds[1]) - int(dynamic_bounds[0]) + 1
    fault_n = int(fault_bounds[1]) - int(fault_bounds[0]) + 1
    cells = []
    if section in {"normal", "all"}:
        cells.extend(
            FormalCell("normal", task, method, None, normal_n)
            for task in protocol["tasks"]
            for method in protocol["methods"]
        )
    if section in {"dynamic", "all"}:
        cells.extend(
            FormalCell("dynamic", task, method, None, dynamic_n)
            for task in protocol["tasks"]
            for method in protocol["methods"]
        )
    if section in {"fault", "all"}:
        cells.extend(
            FormalCell("fault", task, method, fault, fault_n)
            for fault in protocol["faults"]
            for task in protocol["tasks"]
            for method in protocol["methods"]
        )
    return tuple(cells)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_identity(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"model tree is empty: {root}")
    records = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path) for path in files
    }
    digest = hashlib.sha256()
    for name, value in records.items():
        digest.update(f"{name}\0{value}\n".encode("utf-8"))
    return {
        "root": root.relative_to(REPOSITORY_ROOT).as_posix(),
        "file_count": len(records),
        "fingerprint": digest.hexdigest(),
        "files": records,
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_clean_commit() -> str:
    dirty = _git("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise RuntimeError("formal execution requires a clean committed worktree")
    return _git("rev-parse", "HEAD")


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU list must contain integers") from error
    if not 1 <= len(result) <= 8 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("provide 1..8 distinct GPU indices")
    return result


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = []
    for line in completed.stdout.splitlines():
        index, name, memory, bus = (item.strip() for item in line.split(",", 3))
        result.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory),
                "pci_bus_id": bus,
            }
        )
    return result


def _validate_gpus(gpus: Sequence[int]) -> tuple[int, ...]:
    values = tuple(gpus)
    available = {row["index"] for row in _gpu_inventory()}
    if not 1 <= len(values) <= 8 or len(set(values)) != len(values):
        raise RuntimeError("formal evaluation requires 1..8 distinct GPU lanes")
    missing = sorted(set(values).difference(available))
    if missing:
        raise RuntimeError(f"requested GPU indices are unavailable: {missing}")
    return values


def _lane_specs(gpus: Sequence[int], workers: int) -> tuple[LaneSpec, ...]:
    try:
        return build_lane_specs(gpus, workers)
    except (ValueError, RuntimeError) as error:
        raise RuntimeError(f"invalid formal worker allocation: {error}") from error


def _environment(policy_python: Path, gpu: int, cpus: Sequence[int]) -> dict[str, str]:
    environment = v4_formal_launch._launch_environment(policy_python, gpu)
    # Eight simulators must not each create a full-machine BLAS thread pool.
    environment.update(
        {
            "OMP_NUM_THREADS": str(max(1, len(cpus) // 2)),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ESSAY2608_FORMAL_RENDER_BACKEND": "xvfb_software_gl",
        }
    )
    return environment


def _xvfb_command(
    cell: FormalCell | FormalShard, lane: int, sim: Path, policy: Path
) -> tuple[str, ...]:
    xvfb_log = LAUNCH_ROOT / "active" / f"{cell.name}.xvfb.log"
    return (
        str(v4_formal_launch.DEFAULT_XVFB_RUN),
        "--auto-servernum",
        "--server-num",
        str(140 + lane),
        "--error-file",
        str(xvfb_log),
        "--server-args=-screen 0 1280x1024x24 -nolisten tcp",
        *cell.command(sim, policy),
    )


def _validate_result(
    cell: FormalCell,
    commit: Optional[str] = None,
    *,
    protocol_sha256: Optional[str] = None,
) -> None:
    try:
        payload = json.loads(cell.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read formal cell result: {cell.result}") from error
    metadata = payload.get("stage6_formal_evaluation")
    if not isinstance(metadata, dict) or metadata.get("schema") != (
        "essay2608.phase6_formal_result.v1"
    ):
        raise RuntimeError(f"result has no Stage-six formal identity: {cell.result}")
    expected = {
        "experiment": cell.experiment,
        "task": cell.task,
        "method": cell.method,
        "fault": cell.fault,
        "episodes_completed": cell.episodes,
        "protocol_sha256": protocol_sha256 or _sha256(PROTOCOL),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"result {name} differs from protocol: {cell.result}")
    if commit is not None and metadata.get("git_commit") != commit:
        raise RuntimeError(
            f"result commit differs from current formal commit: {cell.result}"
        )
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != cell.episodes:
        raise RuntimeError(f"formal result has incomplete episode rows: {cell.result}")


def _validate_shard(shard: FormalShard, commit: Optional[str] = None) -> None:
    try:
        payload = json.loads(shard.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read formal shard result: {shard.result}"
        ) from error
    metadata = payload.get("stage6_formal_evaluation")
    if not isinstance(metadata, dict) or metadata.get("schema") != (
        "essay2608.phase6_formal_result.v1"
    ):
        raise RuntimeError(f"formal shard has no identity: {shard.result}")
    expected = {
        "experiment": shard.cell.experiment,
        "task": shard.cell.task,
        "method": shard.cell.method,
        "fault": shard.cell.fault,
        "episode_indices": list(shard.episode_indices),
        "episodes_completed": len(shard.episode_indices),
        "protocol_sha256": _sha256(PROTOCOL),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"formal shard {name} differs: {shard.result}")
    if commit is not None and metadata.get("git_commit") != commit:
        raise RuntimeError(f"formal shard commit differs: {shard.result}")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != len(shard.episode_indices):
        raise RuntimeError(f"formal shard rows are incomplete: {shard.result}")
    observed = [row.get("formal_episode_index") for row in rows]
    if observed != list(shard.episode_indices):
        raise RuntimeError(f"formal shard row indices differ: {shard.result}")


def _merge_ik_diagnostics(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Runtime counters are emitted lazily: a shard that never exercises a
    # diagnostic path can legitimately omit that counter.  Configuration and
    # identity fields, in contrast, must be present and identical everywhere.
    keys = set().union(*(set(payload) for payload in payloads))
    result: dict[str, Any] = {}
    for key in sorted(keys):
        values = [payload[key] for payload in payloads if key in payload]
        missing = len(values) != len(payloads)
        first = values[0]
        if isinstance(first, bool) or isinstance(first, str) or first is None:
            if missing:
                raise RuntimeError(f"formal shard IK identity is missing at {key}")
            if any(value != first for value in values[1:]):
                raise RuntimeError(f"formal shard IK identity differs at {key}")
            result[key] = first
        elif isinstance(first, dict):
            if missing:
                raise RuntimeError(f"formal shard IK config is missing at {key}")
            if any(value != first for value in values[1:]):
                raise RuntimeError(f"formal shard IK config differs at {key}")
            result[key] = first
        elif isinstance(first, list):
            if any(not isinstance(value, list) for value in values[1:]):
                raise RuntimeError(f"formal shard IK type differs at {key}")
            result[key] = sorted({item for value in values for item in value})
        elif isinstance(first, (int, float)):
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values[1:]
            ):
                raise RuntimeError(f"formal shard IK type differs at {key}")
            if key.endswith("_max") or key.endswith("_tier_max"):
                result[key] = max(values)
            elif key.endswith("_min"):
                result[key] = min(values)
            else:
                result[key] = sum(values)
        else:
            raise RuntimeError(f"unsupported IK diagnostic type at {key}")
    return result


def _merge_episode_accounting(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    schemas = {payload.get("schema") for payload in payloads}
    if len(schemas) != 1:
        raise RuntimeError("formal shard episode-accounting schemas differ")
    result: dict[str, Any] = {"schema": next(iter(schemas))}
    rate_keys = {
        "success_rate_all_planned_episodes",
        "success_rate_in_complete_intervention_subset",
    }
    keys = set(payloads[0])
    if any(set(payload) != keys for payload in payloads[1:]):
        raise RuntimeError("formal shard episode-accounting fields differ")
    for key in sorted(keys.difference({"schema", *rate_keys})):
        values = [payload[key] for payload in payloads]
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise RuntimeError(f"episode-accounting field {key} is not a count")
        result[key] = sum(values)
    denominator = result["planned_episode_denominator"]
    result["success_rate_all_planned_episodes"] = result[
        "successes_in_planned_denominator"
    ] / float(denominator)
    complete = result["complete_intervention_subset_count"]
    result["success_rate_in_complete_intervention_subset"] = (
        result["successes_in_complete_intervention_subset"] / float(complete)
        if complete
        else None
    )
    return result


_PROTOCOL_COUNT_FIELDS = frozenset(
    {
        "episodes_intervention_eligible",
        "episodes_pre_intervention_terminal",
        "episodes_dynamic_condition_unexercised",
        "pre_trigger_successes",
        "planned_episode_denominator",
        "completed_episode_count",
        "episodes_with_intervention",
        "episodes_with_effective_intervention",
        "episodes_with_complete_intervention",
        "successes_in_complete_intervention_subset",
    }
)


def _merge_protocol_summary(
    payloads: Sequence[Mapping[str, Any]],
    episode_indices: tuple[int, ...],
) -> dict[str, Any]:
    result = json.loads(json.dumps(payloads[0]))
    for field in _PROTOCOL_COUNT_FIELDS:
        if field in result:
            result[field] = sum(int(payload[field]) for payload in payloads)
    complete = int(result.get("episodes_with_complete_intervention", 0))
    if "success_rate_in_complete_intervention_subset" in result:
        result["success_rate_in_complete_intervention_subset"] = (
            int(result["successes_in_complete_intervention_subset"]) / float(complete)
            if complete
            else None
        )
    for field in (
        "all_episodes_intervened",
        "all_interventions_effective",
        "all_eligible_interventions_effective",
        "protocol_valid",
    ):
        if field not in result:
            continue
        values = [payload[field] for payload in payloads]
        result[field] = (
            None
            if all(value is None for value in values)
            else all(value is not False for value in values)
        )
    cache = result.get("staged_motion_plan_cache")
    if isinstance(cache, dict):
        caches = [payload["staged_motion_plan_cache"] for payload in payloads]
        cache["plan_fingerprints"] = [
            item for value in caches for item in value.get("plan_fingerprints", [])
        ]
        key = cache.get("cache_key")
        if isinstance(key, dict):
            key["base_seed"] = GLOBAL_EVAL_SEED_START + episode_indices[0]
            key["episodes"] = len(episode_indices)
            if "variation_schedule" in key:
                key["variation_schedule"] = [
                    item
                    for value in caches
                    for item in value.get("cache_key", {}).get("variation_schedule", [])
                ]
    return result


def _validate_merge_identity(payloads: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "schema",
        "release",
        "policy_type",
        "closed_loop_feature_profile",
        "protocol_label",
        "task",
        "scenario",
        "horizon",
        "evaluation_protocol_id",
        "fixed_eval_set",
        "controller",
        "model_identity",
        "gripper_protocol",
        "gripper_timing",
        "final_settling_protocol",
        "motion_plan_batch_fingerprint",
    )
    for field in fields:
        values = [payload.get(field) for payload in payloads]
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"formal shard runtime identity differs at {field}")


def merge_formal_cell(
    cell: FormalCell,
    shards: Sequence[FormalShard],
    *,
    commit: str,
) -> None:
    """Strictly merge complete shard coverage into the canonical cell result."""

    ordered = tuple(sorted(shards, key=lambda shard: shard.episode_indices[0]))
    for shard in ordered:
        _validate_shard(shard, commit)
    indices = tuple(index for shard in ordered for index in shard.episode_indices)
    if indices != tuple(range(cell.episodes)):
        raise RuntimeError(f"formal shard coverage is incomplete for {cell.cell_id}")
    payloads = [
        json.loads(shard.result.read_text(encoding="utf-8")) for shard in ordered
    ]
    _validate_merge_identity(payloads)
    rows = [row for payload in payloads for row in payload["results"]]
    observed = [row.get("formal_episode_index") for row in rows]
    if observed != list(indices) or len(observed) != len(set(observed)):
        raise RuntimeError(f"formal shard rows overlap or have gaps for {cell.cell_id}")
    for row, index in zip(rows, indices, strict=True):
        row["episode"] = index
    result = json.loads(json.dumps(payloads[0]))
    result["results"] = rows
    result["episodes"] = cell.episodes
    result["episodes_requested"] = cell.episodes
    result["episodes_completed"] = cell.episodes
    result["seed"] = GLOBAL_EVAL_SEED_START + indices[0]
    result["variation_schedule"] = [
        item for payload in payloads for item in payload.get("variation_schedule", [])
    ]
    successes = sum(bool(row.get("success")) for row in rows)
    result["successes"] = successes
    result["success_rate"] = successes / float(cell.episodes)
    result["episode_accounting"] = _merge_episode_accounting(
        [payload["episode_accounting"] for payload in payloads]
    )
    result["ik_execution_diagnostics"] = _merge_ik_diagnostics(
        [payload["ik_execution_diagnostics"] for payload in payloads]
    )
    protocol_key = "protocol" if "protocol" in result else "scenario_protocol"
    result[protocol_key] = _merge_protocol_summary(
        [payload[protocol_key] for payload in payloads], indices
    )
    result["fresh_task_generation"] = {
        "required_per_formal_episode": True,
        "all_episodes_recorded": all(
            payload["fresh_task_generation"]["all_episodes_recorded"]
            for payload in payloads
        ),
        "evidence": [
            item
            for payload in payloads
            for item in payload["fresh_task_generation"]["evidence"]
        ],
    }
    if "store_mode_subgroups" in result and result["store_mode_subgroups"] is not None:
        merged_groups = {}
        for mode in result["store_mode_subgroups"]:
            values = [payload["store_mode_subgroups"][mode] for payload in payloads]
            counts = {
                key: sum(int(value[key]) for value in values)
                for key in values[0]
                if key != "success_rate"
            }
            counts["success_rate"] = (
                counts["successes"] / float(counts["completed"])
                if counts["completed"]
                else None
            )
            merged_groups[mode] = counts
        result["store_mode_subgroups"] = merged_groups
    metadata = result["stage6_formal_evaluation"]
    metadata["episode_indices"] = list(indices)
    metadata["episode_seeds"] = [GLOBAL_EVAL_SEED_START + index for index in indices]
    metadata["episodes_completed"] = cell.episodes
    triggered_values = [
        payload["stage6_formal_evaluation"].get("episodes_fault_triggered")
        for payload in payloads
    ]
    metadata["episodes_fault_triggered"] = (
        None
        if all(value is None for value in triggered_values)
        else sum(int(value or 0) for value in triggered_values)
    )
    metadata["shard_merge"] = {
        "schema": "essay2608.phase6_formal_shard_merge.v1",
        "shard_count": len(ordered),
        "shard_results": [
            shard.result.relative_to(RESULTS_ROOT).as_posix() for shard in ordered
        ],
        "exact_nonoverlapping_coverage": True,
    }
    cell.result.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(cell.result, result)
    _validate_result(cell, commit)


def _retained_records() -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(RETAINED_RESULTS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read retained formal-result manifest: {RETAINED_RESULTS}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "active_protocol_sha256",
        "reason",
        "records",
    }:
        raise RuntimeError("retained formal-result manifest has invalid fields")
    if payload.get("schema") != "essay2608.phase6_retained_results.v2":
        raise RuntimeError("retained formal-result manifest has invalid schema")
    if payload.get("active_protocol_sha256") != _sha256(PROTOCOL):
        raise RuntimeError("retained formal results belong to a different protocol")
    if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        raise RuntimeError("retained formal-result manifest must state its reason")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise RuntimeError("retained formal-result records must be an object")
    for cell_id, record in records.items():
        if (
            not isinstance(cell_id, str)
            or not isinstance(record, dict)
            or set(record) != {"path", "git_commit", "protocol_sha256", "sha256"}
            or any(not isinstance(record[name], str) for name in record)
        ):
            raise RuntimeError("retained formal-result record is invalid")
    return records


def _validate_retained_result(cell: FormalCell) -> None:
    record = _retained_records().get(cell.cell_id)
    if record is None:
        raise RuntimeError(
            f"old-commit result is not explicitly retained: {cell.result}"
        )
    expected_path = cell.result.relative_to(RESULTS_ROOT).as_posix()
    if record["path"] != expected_path:
        raise RuntimeError(f"retained result path differs: {cell.result}")
    if record["sha256"] != _sha256(cell.result):
        raise RuntimeError(f"retained result content differs: {cell.result}")
    _validate_result(
        cell,
        record["git_commit"],
        protocol_sha256=record["protocol_sha256"],
    )


def _validate_available_result(cell: FormalCell, commit: Optional[str] = None) -> str:
    """Validate either the active result identity or an explicit retention.

    A targeted rerun may leave unaffected cells byte-for-byte frozen while
    replacing only the cells touched by a general mechanism correction.  The
    retention manifest authenticates those exact files; it never rewrites
    their embedded provenance or silently accepts an arbitrary old result.
    """

    try:
        _validate_result(cell, commit)
    except RuntimeError:
        _validate_retained_result(cell)
        return "COMPLETED_RETAINED"
    return "COMPLETED_VALIDATED"


def _states(cells: Sequence[FormalCell], commit: Optional[str]) -> dict[str, str]:
    result = {}
    for cell in cells:
        lock = cell.result.with_name(cell.result.name + ".lock")
        if lock.exists():
            raise RuntimeError(f"formal result has an active/stale lock: {lock}")
        if cell.result.exists():
            result[cell.cell_id] = _validate_available_result(cell, commit)
        else:
            result[cell.cell_id] = "PENDING"
    return result


def preflight(
    *,
    protocol: Mapping[str, Any],
    cells: Sequence[FormalCell],
    sim_python: Path,
    policy_python: Path,
    gpus: Sequence[int],
    require_clean: bool,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    sim = v4_formal_launch._regular_executable(sim_python, "simulator Python")
    policy = v4_formal_launch._regular_executable(policy_python, "policy Python")
    v4_formal_launch._regular_executable(v4_formal_launch.DEFAULT_XVFB_RUN, "xvfb-run")
    gpus = _validate_gpus(gpus)
    configured_workers = int(protocol["resource_plan"]["parallel_lanes"])
    if workers != configured_workers:
        raise RuntimeError(
            f"formal workers {workers} differ from frozen protocol {configured_workers}"
        )
    specs = _lane_specs(gpus, workers)
    environment = _environment(policy, specs[0].gpu, specs[0].logical_cpus)
    v4_formal_launch._validate_python_runtime(
        sim,
        expected=(3, 8),
        imports=(
            "numpy",
            "pyrep",
            "rlbench",
            RUNNER_MODULE,
            "integrations.rlbench.rlbench_dynamac.core.trac_ik",
        ),
        checks=(
            "from integrations.rlbench.rlbench_dynamac.core.pytracik_dependency "
            "import assert_formal_pytracik_build",
            "assert_formal_pytracik_build()",
        ),
        environment=environment,
        label="simulator Python",
    )
    v4_formal_launch._validate_python_runtime(
        policy,
        expected=(3, 10),
        imports=("numpy", "scipy", "sklearn", "essay2608.policy.closed_loop"),
        environment=environment,
        label="policy Python",
    )
    commit = _validate_clean_commit() if require_clean else None
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "git_commit": commit or _git("rev-parse", "HEAD"),
        "worktree_clean_required": require_clean,
        "evaluation_set": v4_formal_launch._validate_evaluation_set(),
        "frozen_v4_release": v4_formal_launch._validate_model_release(),
        "phase6_base_models": _tree_identity(BASE_MODELS),
        "closed_loop_models": _tree_identity(CLOSED_LOOP_MODELS),
        "gpus": _gpu_inventory(),
        "selected_gpu_lanes": list(gpus),
        "parallel_workers": workers,
        "worker_lanes": [spec.to_dict() for spec in specs],
        "cpu_affinity": [list(spec.logical_cpus) for spec in specs],
        "render_backend": protocol["resource_plan"]["render_backend"],
        "cells": _states(cells, commit),
    }


def render_plan(
    protocol: Mapping[str, Any],
    cells: Sequence[FormalCell],
    sim_python: Path,
    policy_python: Path,
    gpus: Sequence[int],
    workers: int = DEFAULT_WORKERS,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> str:
    specs = _lane_specs(gpus, workers)
    shards = build_shards(cells, protocol, shard_size=shard_size)
    lines = [
        "Stage-six preregistered formal plan (no simulator started)",
        f"protocol={PROTOCOL.relative_to(REPOSITORY_ROOT)} sha256={_sha256(PROTOCOL)}",
        f"cells={len(cells)} shards={len(shards)} episodes={sum(cell.episodes for cell in cells)} workers={workers}",
        "renderer=xvfb_software_gl; CUDA identity is isolated per lane",
    ]
    for index, shard in enumerate(shards):
        lane = index % len(specs)
        spec = specs[lane]
        command = _xvfb_command(shard, lane, sim_python, policy_python)
        lines.extend(
            (
                "",
                f"[{index + 1}/{len(shards)}] {shard.shard_id}",
                f"lane={lane} gpu={spec.gpu} cpus={','.join(map(str, spec.logical_cpus))}",
                " ".join(shlex.quote(value) for value in command),
                f"result={shard.result}",
            )
        )
    return "\n".join(lines)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate(processes: Iterable[subprocess.Popen]) -> None:
    active = [process for process in processes if process.poll() is None]
    process_groups = [process.pid for process in active]
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15.0
    remaining_groups = process_groups
    while remaining_groups and time.monotonic() < deadline:
        for process in active:
            process.poll()
        remaining_groups = [
            process_group
            for process_group in remaining_groups
            if _process_group_exists(process_group)
        ]
        if remaining_groups:
            time.sleep(0.25)
    for process_group in remaining_groups:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in active:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def execute(
    *,
    protocol: Mapping[str, Any],
    cells: Sequence[FormalCell],
    sim_python: Path,
    policy_python: Path,
    gpus: Sequence[int],
    workers: int = DEFAULT_WORKERS,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    identity = preflight(
        protocol=protocol,
        cells=cells,
        sim_python=sim_python,
        policy_python=policy_python,
        gpus=gpus,
        require_clean=True,
        workers=workers,
    )
    pending_cells = [
        cell for cell in cells if identity["cells"][cell.cell_id] == "PENDING"
    ]
    if not pending_cells:
        return {"status": "nothing_to_run", **identity}
    all_shards = build_shards(
        pending_cells,
        protocol,
        shard_size=shard_size,
    )
    shards_by_cell = {
        cell.cell_id: tuple(
            shard for shard in all_shards if shard.cell.cell_id == cell.cell_id
        )
        for cell in pending_cells
    }
    pending_shards = []
    resumed_shards = []
    for shard in all_shards:
        lock_path = shard.result.with_name(shard.result.name + ".lock")
        if lock_path.exists():
            raise RuntimeError(f"formal shard has an active/stale lock: {lock_path}")
        if shard.result.exists():
            _validate_shard(shard, identity["git_commit"])
            resumed_shards.append(shard.shard_id)
        else:
            pending_shards.append(shard)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LAUNCH_ROOT / "execute.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"another formal launcher owns {lock}") from error
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-pid{os.getpid()}"
    run_root = LAUNCH_ROOT / "runs" / run_id
    active_root = LAUNCH_ROOT / "active"
    run_root.mkdir(parents=True, exist_ok=False)
    active_root.mkdir(parents=True, exist_ok=True)
    processes: dict[str, tuple[subprocess.Popen, FormalShard, int]] = {}
    streams: dict[str, Any] = {}
    assignments: list[dict[str, Any]] = []
    specs = _lane_specs(gpus, workers)
    available = list(range(len(specs)))
    queue = list(pending_shards)
    started = time.time()
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} run_id={run_id}\n")
        stream.flush()
        os.fsync(stream.fileno())

    def launch(shard: FormalShard, lane: int) -> None:
        spec = specs[lane]
        shard.result.parent.mkdir(parents=True, exist_ok=True)
        shard.diagnostics.mkdir(parents=True, exist_ok=True)
        log_path = run_root / f"{shard.name}.log"
        log_stream = log_path.open("xb")
        streams[shard.shard_id] = log_stream
        command = _xvfb_command(shard, lane, sim_python, policy_python)

        def set_affinity() -> None:
            os.sched_setaffinity(0, set(spec.logical_cpus))

        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=_environment(policy_python, spec.gpu, spec.logical_cpus),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=set_affinity,
        )
        processes[shard.shard_id] = (process, shard, lane)
        assignments.append(
            {
                "cell_id": shard.cell.cell_id,
                "shard_id": shard.shard_id,
                "episode_indices": list(shard.episode_indices),
                "lane": lane,
                "gpu": spec.gpu,
                "numa_node": spec.numa_node,
                "physical_cores": list(spec.physical_cores),
                "cpus": list(spec.logical_cpus),
                "pid": process.pid,
                "command": list(command),
                "started_unix": time.time(),
            }
        )

    try:
        while queue or processes:
            while queue and available:
                launch(queue.pop(0), available.pop(0))
            for shard_id, (process, shard, lane) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del processes[shard_id]
                available.append(lane)
                available.sort()
                streams[shard_id].flush()
                if return_code != 0:
                    _terminate(entry[0] for entry in processes.values())
                    raise RuntimeError(
                        f"formal shard {shard_id} exited {return_code}; see {run_root}"
                    )
                _validate_shard(shard, identity["git_commit"])
                row = next(
                    value for value in assignments if value["shard_id"] == shard_id
                )
                row["finished_unix"] = time.time()
            if queue or processes:
                time.sleep(1.0)
    except BaseException:
        _terminate(entry[0] for entry in processes.values())
        raise
    finally:
        for stream in streams.values():
            stream.close()
        lock.unlink(missing_ok=True)
        for path in active_root.glob("*.xvfb.log"):
            target = run_root / path.name
            if not target.exists():
                shutil.move(str(path), target)

    for cell in pending_cells:
        merge_formal_cell(
            cell,
            shards_by_cell[cell.cell_id],
            commit=identity["git_commit"],
        )
        identity["cells"][cell.cell_id] = "COMPLETED_VALIDATED"

    summary = {
        "schema": "essay2608.phase6_formal_launch.v1",
        "status": "completed",
        "run_id": run_id,
        "started_unix": started,
        "finished_unix": time.time(),
        "assignments": assignments,
        "scheduler": {
            "schema": "essay2608.phase6_global_shard_queue.v1",
            "workers": workers,
            "shard_size": shard_size,
            "work_conserving_global_queue": True,
            "resumed_shards": resumed_shards,
            "new_shards_completed": len(assignments),
        },
        **identity,
    }
    atomic_json(run_root / "launch_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("plan", "preflight", "execute"), default="plan"
    )
    parser.add_argument(
        "--section",
        choices=("normal", "dynamic", "fault", "all"),
        default="normal",
    )
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--gpus", type=_parse_gpus, default=DEFAULT_GPUS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(PROTOCOL)
    cells = build_cells(protocol, args.section)
    if args.command == "plan":
        print(
            render_plan(
                protocol,
                cells,
                args.sim_python,
                args.policy_python,
                args.gpus,
                args.workers,
                args.shard_size,
            )
        )
        return 0
    if args.command == "preflight":
        result = preflight(
            protocol=protocol,
            cells=cells,
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            gpus=args.gpus,
            require_clean=False,
            workers=args.workers,
        )
    else:
        result = execute(
            protocol=protocol,
            cells=cells,
            sim_python=args.sim_python,
            policy_python=args.policy_python,
            gpus=args.gpus,
            workers=args.workers,
            shard_size=args.shard_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
