"""Audit the current A6 composite Main-10 endpoint data.

The historical A5 acceptance remains immutable evidence for the original
36,000 endpoint runs.  A6 replaces only complete, outcome-independent M5/M6
task cells after the post-E4 method freeze.  This audit proves that the
resulting composite still has complete identities, paired initial states,
cycle hashes, and a physical re-audit derived from the promoted bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.analysis.rebuild_a5_audit import (
    AUDITOR_PATH,
    INDEX_PATH as REAUDIT_INDEX,
    load_current_reaudit,
)
from evaluations.iclr2027.runners.a6_endpoint_compatibility import (
    assert_frozen_m5_current,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
RESULT = EVAL / "results" / "controlled" / "e1_e2"
ENDPOINT = RESULT / "end_to_end"
PLAN = RESULT / "A5_RUN_PLAN.json"
HISTORICAL_ACCEPTANCE = RESULT / "A5_ACCEPTANCE.json"
MAIN10_PROMOTION = RESULT / "post_e4_m5_refresh" / "PROMOTION.json"
LIFT_TRAY_PROMOTION = (
    RESULT / "m5_lift_tray_perturbed_refresh" / "PROMOTION.json"
)
HANDOVER_PROMOTION = (
    EVAL
    / "results"
    / "controlled"
    / "post_e4_handover_refresh"
    / "PROMOTION.json"
)
OUTPUT = EVAL / "results" / "a6_execution" / "A6_CURRENT_ENDPOINT_INTEGRITY.json"
POST_E4_FREEZE = (
    EVAL / "results" / "a6_execution" / "M5_POST_E4_FREEZE.json"
)
CURRENT_M5_FREEZE = (
    EVAL
    / "results"
    / "a6_execution"
    / "M5_PENDING_DIRECT_IDENTITY_FREEZE.json"
)
CURRENT_E1_PROMOTION = (
    EVAL
    / "results"
    / "controlled"
    / "pending_direct_identity_refresh"
    / "promotions"
    / "E1_PROMOTION.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe(episode_id: str) -> str:
    return episode_id.replace("/", "__")


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _first_observation(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        row = json.loads(next(stream))
    feature = row["feature"]
    return {"arms": feature["arms"], "task_state": feature["task_state"]}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit() -> dict[str, Any]:
    errors: list[str] = []
    plan = _json(PLAN)
    historical = _json(HISTORICAL_ACCEPTANCE)
    if not str(historical.get("status", "")).startswith("PASS"):
        errors.append("historical A5 acceptance is not PASS")

    # The post-E4 freeze remains immutable provenance for historical cells,
    # but the live model/source tree was superseded by the accepted
    # pending-direct-identity freeze.  Verify the historical record as a
    # record and the current freeze against the live bytes; requiring the
    # live tree to equal both freezes is impossible by construction.
    post_e4_freeze = _json(POST_E4_FREEZE)
    if post_e4_freeze.get("status") != "PASS":
        errors.append("historical post-E4 M5 freeze is not PASS")
    freeze = assert_frozen_m5_current()
    for label, path in (
        ("main10", MAIN10_PROMOTION),
        ("lift_tray", LIFT_TRAY_PROMOTION),
        ("handover", HANDOVER_PROMOTION),
    ):
        if not path.is_file():
            errors.append(f"missing post-E4 {label} promotion")
            continue
        promotion = _json(path)
        if promotion.get("status") != "PASS":
            errors.append(f"post-E4 {label} promotion is not PASS")
        if promotion.get("outcome_selective_replacement") is True:
            errors.append(f"post-E4 {label} promotion is outcome-selective")

    if not CURRENT_E1_PROMOTION.is_file():
        errors.append("missing pending-direct-identity E1 promotion")
    else:
        current_promotion = _json(CURRENT_E1_PROMOTION)
        if current_promotion.get("status") != "PASS_PROMOTED":
            errors.append("pending-direct-identity E1 promotion is not PASS")
        if current_promotion.get("selection_is_outcome_independent") is not True:
            errors.append("pending-direct-identity E1 promotion is outcome-selective")
        if current_promotion.get("current_m5_freeze_sha256") != _sha256(CURRENT_M5_FREEZE):
            errors.append("pending-direct-identity E1 promotion uses the wrong M5 freeze")

    manifests: dict[str, list[dict[str, Any]]] = {}
    for condition, record in plan["manifests"].items():
        path = ROOT / str(record["path"])
        rows = _rows(path)
        if len(rows) != 2000 or _sha256(path) != record["sha256"]:
            errors.append(f"sealed manifest identity mismatch: {condition}")
        manifests[str(condition)] = rows

    config_hashes = {
        str(item["key"]): str(item["config_sha256"])
        for item in plan["methods"]
    }
    for item in plan["methods"]:
        path = ROOT / str(item["config"])
        if not path.is_file() or _sha256(path) != item["config_sha256"]:
            errors.append(f"method config changed: {item['key']}")

    totals = {
        "episode_results": 0,
        "cycle_files": 0,
        "cycle_hash_errors": 0,
        "identity_errors": 0,
        "pairing_errors": 0,
        "pairing_unverifiable_baseline_infrastructure": 0,
        "infrastructure_failures": 0,
        "temporary_files": 0,
    }
    method_cells: dict[str, Any] = {}
    baseline_initial: dict[tuple[str, str], str] = {}
    method_keys = tuple(config_hashes)
    if not method_keys or method_keys[0] != "m0":
        raise RuntimeError("A5 plan must enumerate m0 before paired methods")

    current_episode_paths: dict[tuple[str, str], Path] = {}
    for condition in ("nominal", "perturbed"):
        expected = {str(row["episode_id"]): row for row in manifests[condition]}
        for method in method_keys:
            root = ENDPOINT / method / condition
            queue = _json(root / "QUEUE_STATUS.json")
            episodes = sorted((root / "episodes").glob("*.json"))
            cycles = sorted((root / "cycles").glob("*.jsonl.gz"))
            method_cells[f"{method}/{condition}"] = {
                "episodes": len(episodes),
                "cycles": len(cycles),
                "infrastructure_errors": int(queue.get("infrastructure_errors", -1)),
                "missing": int(queue.get("missing_selected_count", -1)),
            }
            if (
                len(episodes) != 2000
                or len(cycles) != 2000
                or int(queue.get("completed_episode_count", -1)) != 2000
                or int(queue.get("missing_selected_count", -1)) != 0
            ):
                totals["identity_errors"] += 1
            observed_names = {path.stem for path in episodes}
            if observed_names != {_safe(value) for value in expected}:
                totals["identity_errors"] += 1

            for episode_id, manifest_row in expected.items():
                episode_path = root / "episodes" / f"{_safe(episode_id)}.json"
                if not episode_path.is_file():
                    totals["identity_errors"] += 1
                    continue
                episode = _json(episode_path)
                totals["episode_results"] += 1
                current_episode_paths[(method, episode_id)] = episode_path
                if (
                    episode.get("episode_id") != episode_id
                    or episode.get("task") != manifest_row["task"]
                    or episode.get("variation") != manifest_row["variation"]
                    or episode.get("seed") != manifest_row["seed"]
                    or episode.get("condition") != condition
                    or (episode.get("method_config_identity") or {}).get("sha256")
                    != config_hashes[method]
                    or int(episode.get("cycle_records", -1))
                    != int(episode.get("cycles", -2))
                ):
                    totals["identity_errors"] += 1
                if episode.get("reason") == "infrastructure_error":
                    totals["infrastructure_failures"] += 1
                cycle_path = (episode_path.parent / str(episode["cycle_file"])).resolve()
                if not cycle_path.is_file():
                    totals["cycle_hash_errors"] += 1
                    continue
                totals["cycle_files"] += 1
                if _sha256(cycle_path) != episode.get("cycle_file_sha256"):
                    totals["cycle_hash_errors"] += 1
                if int(episode.get("cycle_records", 0)) > 0:
                    initial = _canonical(_first_observation(cycle_path))
                    pair = (condition, episode_id)
                    if method == "m0":
                        baseline_initial[pair] = initial
                    elif pair not in baseline_initial:
                        totals["pairing_unverifiable_baseline_infrastructure"] += 1
                    elif baseline_initial[pair] != initial:
                        totals["pairing_errors"] += 1
            totals["temporary_files"] += len(list(root.rglob("*.tmp")))

    reaudit_errors = 0
    index = _json(REAUDIT_INDEX)
    reaudit = load_current_reaudit()
    if (
        index.get("status") != "CURRENT"
        or int(index.get("episodes", -1)) != 18000
        or len(reaudit) != 18000
        or (index.get("auditor_implementation") or {}).get("sha256")
        != _sha256(AUDITOR_PATH)
    ):
        reaudit_errors += 1
    expected_reaudit = {
        (method, str(row["episode_id"]))
        for method in method_keys
        for row in manifests["perturbed"]
    }
    if set(reaudit) != expected_reaudit:
        reaudit_errors += 1
    for key, row in reaudit.items():
        episode_path = current_episode_paths.get(key)
        source = row.get("source") or {}
        if (
            episode_path is None
            or source.get("episode_sha256") != _sha256(episode_path)
        ):
            reaudit_errors += 1
            continue
        episode = _json(episode_path)
        if source.get("cycle_file_sha256") != episode.get("cycle_file_sha256"):
            reaudit_errors += 1

    passed = (
        totals["episode_results"] == 36000
        and totals["cycle_files"] == 36000
        and not any(
            totals[key]
            for key in (
                "cycle_hash_errors",
                "identity_errors",
                "pairing_errors",
                "temporary_files",
            )
        )
        and reaudit_errors == 0
        and not errors
    )
    record = {
        "schema": "essay2608.iclr2027.a6-current-endpoint-integrity.v2",
        "status": "PASS" if passed else "FAIL",
        "historical_a5_acceptance": {
            "path": str(HISTORICAL_ACCEPTANCE.relative_to(ROOT)),
            "sha256": _sha256(HISTORICAL_ACCEPTANCE),
        },
        "post_e4_m5_freeze": {
            "path": str(POST_E4_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(POST_E4_FREEZE),
            "model_identity": post_e4_freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": post_e4_freeze["algorithm_source_identity"]["aggregate_sha256"],
            "role": "historical_provenance",
        },
        "current_m5_freeze": {
            "path": str(CURRENT_M5_FREEZE.relative_to(ROOT)),
            "sha256": _sha256(CURRENT_M5_FREEZE),
            "model_identity": freeze["model_tree_identity"]["aggregate_sha256"],
            "algorithm_identity": freeze["algorithm_source_identity"]["aggregate_sha256"],
            "affected_tasks": freeze["affected_tasks"],
            "role": "live_endpoint_identity",
        },
        "promotions": {
            "main10": {
                "path": str(MAIN10_PROMOTION.relative_to(ROOT)),
                "sha256": _sha256(MAIN10_PROMOTION) if MAIN10_PROMOTION.is_file() else None,
            },
            "lift_tray": {
                "path": str(LIFT_TRAY_PROMOTION.relative_to(ROOT)),
                "sha256": _sha256(LIFT_TRAY_PROMOTION) if LIFT_TRAY_PROMOTION.is_file() else None,
            },
            "handover": {
                "path": str(HANDOVER_PROMOTION.relative_to(ROOT)),
                "sha256": _sha256(HANDOVER_PROMOTION) if HANDOVER_PROMOTION.is_file() else None,
            },
            "pending_direct_identity_e1": {
                "path": str(CURRENT_E1_PROMOTION.relative_to(ROOT)),
                "sha256": _sha256(CURRENT_E1_PROMOTION) if CURRENT_E1_PROMOTION.is_file() else None,
            },
        },
        "end_to_end": totals,
        "method_cells": method_cells,
        "re_audit": {
            "path": str(REAUDIT_INDEX.relative_to(ROOT)),
            "sha256": _sha256(REAUDIT_INDEX),
            "episodes": len(reaudit),
            "errors": reaudit_errors,
        },
        "errors": errors,
        "infrastructure_failures_remain_intention_to_treat": True,
        "outcome_selective_replacement": False,
    }
    _atomic_json(OUTPUT, record)
    if not passed:
        raise RuntimeError(
            "A6 current endpoint integrity failed:\n- "
            + "\n- ".join(errors + [f"reaudit_errors={reaudit_errors}", str(totals)])
        )
    return record


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
