"""Canonical, causally aligned reader for M4 failure-training sequences."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .feature_schema import AUDIT_SCHEMA, EPISODE_SCHEMA, validate_feature_record

MANIFEST_SCHEMA = "essay2608.iclr2027.episode-manifest.v1"
NESTED_BUDGETS = (20, 50, 100, 200)


@dataclass(frozen=True)
class FailureTrainSequence:
    """One variable-length M4 input sequence and its causal binary targets."""

    episode_id: str
    task: str
    features: tuple[dict[str, Any], ...]
    labels: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_failure_train_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identities = set()
    for row in rows:
        if row.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("unsupported failure-train manifest schema")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("failure-train row requires an episode_id")
        if episode_id in identities:
            raise ValueError(f"duplicate failure-train episode: {episode_id}")
        identities.add(episode_id)
    return rows


def select_failure_train_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    budget: Optional[int] = None,
    held_out_family: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Select a frozen per-task nested budget or a full-pool LOFO view.

    Budget subsets preserve manifest order and take the first N task rows.
    LOFO removes the named family from the complete 200-row task pool without
    replacing it with later or resampled episodes.
    """

    task_rows = [dict(row) for row in rows if row.get("task") == task]
    if len(task_rows) != 200:
        raise ValueError(f"{task} has {len(task_rows)} rows, expected 200")
    if held_out_family is not None:
        if budget is not None:
            raise ValueError("budget and held_out_family are mutually exclusive")
        if not any(row.get("fault_family") == held_out_family for row in task_rows):
            raise ValueError(f"{task} has no {held_out_family} examples to hold out")
        return [row for row in task_rows if row.get("fault_family") != held_out_family]
    selected_budget = 200 if budget is None else int(budget)
    if selected_budget not in NESTED_BUDGETS:
        raise ValueError(f"budget must be one of {NESTED_BUDGETS}")
    return task_rows[:selected_budget]


def violation_active(audit: Mapping[str, Any]) -> bool:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unsupported physical-event audit schema")
    onset = audit.get("violation_onset_cycle")
    end = audit.get("violation_end_cycle")
    if end is not None and onset is None:
        raise ValueError("violation end cannot precede onset")
    return onset is not None and end is None


def causal_violation_labels(audits: Iterable[Mapping[str, Any]]) -> tuple[int, ...]:
    """Align post-step audits with the next cycle's pre-step feature.

    Row t stores feature_t before physical step t and audit_t after that step.
    Consequently y_0 is zero and y_t reflects whether audit_(t-1) reported an
    active physical violation.  The current row's audit is never an input.
    """

    labels = []
    previous_active = False
    for audit in audits:
        labels.append(int(previous_active))
        previous_active = violation_active(audit)
    return tuple(labels)


def load_failure_train_sequence(
    dataset_root: Path,
    manifest_row: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> FailureTrainSequence:
    dataset_root = Path(dataset_root)
    episode_id = str(manifest_row["episode_id"])
    safe = episode_id.replace("/", "__")
    result_path = dataset_root / "episodes" / (safe + ".json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != EPISODE_SCHEMA:
        raise ValueError(f"unsupported episode schema: {episode_id}")
    for key in (
        "episode_id",
        "task",
        "variation",
        "seed",
        "condition",
        "fault_family",
        "fault_severity",
        "trigger_stage",
    ):
        if result.get(key) != manifest_row.get(key):
            raise ValueError(f"manifest/result mismatch for {episode_id}: {key}")
    if result.get("cycle_file_location") != "episode_relative":
        raise ValueError(f"non-portable cycle reference: {episode_id}")
    reference = Path(str(result["cycle_file"]))
    if reference.is_absolute():
        raise ValueError(f"absolute cycle reference: {episode_id}")
    cycle_path = (result_path.parent / reference).resolve()
    expected_cycle_root = (dataset_root / "cycles").resolve()
    if cycle_path.parent != expected_cycle_root or not cycle_path.is_file():
        raise ValueError(f"cycle file escaped the canonical directory: {episode_id}")
    if verify_hash and _sha256(cycle_path) != result.get("cycle_file_sha256"):
        raise ValueError(f"cycle hash mismatch: {episode_id}")

    features = []
    audits = []
    with gzip.open(cycle_path, "rt", encoding="utf-8") as stream:
        for expected_cycle, line in enumerate(stream):
            record = json.loads(line)
            feature = validate_feature_record(record["feature"])
            audit = record["audit"]
            if (
                record.get("cycle") != expected_cycle
                or feature["cycle"] != expected_cycle
                or audit.get("cycle") != expected_cycle
                or feature["episode_id"] != episode_id
            ):
                raise ValueError(f"non-contiguous or mismatched cycle: {episode_id}")
            violation_active(audit)
            features.append(feature)
            audits.append(audit)
    if len(features) != result.get("cycle_records") or len(features) != result.get(
        "cycles"
    ):
        raise ValueError(f"cycle count mismatch: {episode_id}")
    return FailureTrainSequence(
        episode_id=episode_id,
        task=str(manifest_row["task"]),
        features=tuple(features),
        labels=causal_violation_labels(audits),
    )


__all__ = [
    "FailureTrainSequence",
    "NESTED_BUDGETS",
    "causal_violation_labels",
    "load_failure_train_manifest",
    "load_failure_train_sequence",
    "select_failure_train_rows",
    "violation_active",
]
