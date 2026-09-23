"""Development-only differential gate for the nested-paired E4 protocol."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.audit.horizon_nested_events import NESTED_EVENT_SCHEDULE
from evaluations.iclr2027.runners.a6_e4_nested import (
    DEVELOPMENT_GATE,
    MANIFEST as FORMAL_MANIFEST,
    OUTPUT,
    PLAN as FORMAL_PLAN,
    SOURCE as SINGLE_EVENT_SOURCE,
    prepare as prepare_formal,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "evaluations" / "iclr2027"
ROOT_OUT = OUTPUT / "development"
SINGLE_MANIFEST = ROOT_OUT / "manifests" / "paired_single.jsonl"
NESTED_MANIFEST = ROOT_OUT / "manifests" / "paired_nested.jsonl"
PLAN = ROOT_OUT / "E4_NESTED_DEVELOPMENT_PLAN.json"
METHOD = EVAL / "configs" / "methods" / "m0_dynamac.json"
TASK_LEVELS = {
    "place_cups_1": 1,
    "place_cups_2": 2,
    "place_cups_3": 3,
    "push_buttons_1": 1,
    "push_buttons_2": 2,
    "push_buttons_3": 3,
    "remove_cups_1": 1,
    "remove_cups_2": 2,
}
BASE_FAMILIES = ("actuation_delay", "environment_change")
EXTRA_CONTACT_FAMILY_TASKS = {"place_cups_2", "remove_cups_2"}
EXTRA_CONTACT_FAMILIES = ("missed_interaction", "relation_loss")
WORKERS = 8
TIMEOUT_SECONDS = 900.0


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _development_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    single = []
    nested = []
    source_rows = _rows(SINGLE_EVENT_SOURCE)
    for task in TASK_LEVELS:
        level = TASK_LEVELS[task]
        families = BASE_FAMILIES + (
            EXTRA_CONTACT_FAMILIES if task in EXTRA_CONTACT_FAMILY_TASKS else ()
        )
        for family in families:
            source = next(
                (
                    row
                    for row in source_rows
                    if row.get("task") == task
                    and row.get("fault_family") == family
                ),
                None,
            )
            if source is None:
                raise RuntimeError(
                    f"frozen single-event manifest has no {task}/{family} row"
                )
            pair = f"e4_nested_development/{task}/{family}"
            base = {
                **source,
                "development_only": True,
                "pair_id": pair,
            }
            first = {
                **base,
                "episode_id": f"e4_nested_development_single/{task}/{family}",
                "split": "e4_nested_development_single",
            }
            second = {
                **base,
                "episode_id": f"e4_nested_development_nested/{task}/{family}",
                "split": "e4_nested_development_nested",
                "event_schedule": NESTED_EVENT_SCHEDULE,
                "paired_single_event_episode_id": first["episode_id"],
            }
            single.append(first)
            nested.append(second)
    _atomic_jsonl(SINGLE_MANIFEST, single)
    _atomic_jsonl(NESTED_MANIFEST, nested)
    return single, nested


def prepare() -> dict[str, Any]:
    formal = prepare_formal()
    single, nested = _development_rows()
    plan = {
        "schema": "essay2608.iclr2027.a6-e4-nested-development-plan.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_only": True,
        "sealed_results_read": False,
        "pairs": len(single),
        "single_manifest": {
            "path": str(SINGLE_MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(SINGLE_MANIFEST),
        },
        "nested_manifest": {
            "path": str(NESTED_MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(NESTED_MANIFEST),
        },
        "method": {"path": str(METHOD.relative_to(ROOT)), "sha256": _sha256(METHOD)},
        "formal_plan": {"path": str(FORMAL_PLAN.relative_to(ROOT)), "sha256": _sha256(FORMAL_PLAN)},
        "formal_manifest_sha256": formal["manifest"]["sha256"],
        "protocol_files": formal["protocol_files"],
        "workers": WORKERS,
        "episode_timeout_seconds": TIMEOUT_SECONDS,
    }
    if PLAN.is_file():
        previous = _json(PLAN)
        left, right = dict(previous), dict(plan)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        if left != right:
            retained = list((ROOT_OUT / "single" / "episodes").glob("*.json"))
            retained += list((ROOT_OUT / "nested" / "episodes").glob("*.json"))
            if retained:
                raise RuntimeError("existing nested development plan disagrees with inputs")
            _atomic_json(PLAN, plan)
            return plan
        return previous
    _atomic_json(PLAN, plan)
    return plan


def _run(manifest: Path, output: Path, *, nested: bool) -> None:
    expected = len(_rows(manifest))
    module = (
        "evaluations.iclr2027.runners.a6_horizon_nested_launch"
        if nested
        else "evaluations.iclr2027.runners.launch"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--manifest",
        str(manifest),
        "--output-root",
        str(output),
        "--workers",
        str(WORKERS),
        "--episode-timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--retry-infrastructure",
        "1",
        "--method",
        str(METHOD),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    status = _json(output / "QUEUE_STATUS.json")
    if (
        completed.returncode not in (0, 2)
        or int(status.get("completed_episode_count", -1)) != expected
        or int(status.get("missing_selected_count", -1)) != 0
    ):
        raise SystemExit(completed.returncode or 4)


def run() -> None:
    prepare()
    _run(SINGLE_MANIFEST, ROOT_OUT / "single", nested=False)
    _run(NESTED_MANIFEST, ROOT_OUT / "nested", nested=True)


def _safe(value: str) -> str:
    return value.replace("/", "__")


def _cycles(root: Path, episode_id: str) -> list[dict[str, Any]]:
    path = root / "cycles" / (_safe(episode_id) + ".jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    return _json(root / "episodes" / (_safe(episode_id) + ".json"))


def _normal_cycle(row: Mapping[str, Any]) -> dict[str, Any]:
    feature = copy.deepcopy(dict(row["feature"]))
    feature.pop("episode_id", None)
    execution = row["execution"]
    return {
        "cycle": int(row["cycle"]),
        "feature": feature,
        "execution": {
            key: copy.deepcopy(execution.get(key))
            for key in (
                "action_resolution",
                "applied_action",
                "policy_complete",
                "reward",
                "terminate",
            )
        },
    }


def _first_event(cycles: list[Mapping[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    for row in cycles:
        events = ((row.get("execution") or {}).get("injector") or {}).get("events") or ()
        if events:
            return int(row["cycle"]), dict(events[0])
    return None


def _normal_event(event: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(event))
    for key in ("nested_role", "interaction_index", "component_policy_step"):
        value.pop(key, None)
    return value


def validate() -> dict[str, Any]:
    plan = prepare()
    single_rows = [json.loads(line) for line in SINGLE_MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    nested_rows = [json.loads(line) for line in NESTED_MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    by_pair = {row["pair_id"]: row for row in nested_rows}
    records = []
    one_stage_exact = 0
    paired_prefix_exact = 0
    untriggered_full_execution_exact = 0
    later_opportunities = 0
    for first in single_rows:
        second = by_pair[first["pair_id"]]
        left_episode = _episode(ROOT_OUT / "single", first["episode_id"])
        right_episode = _episode(ROOT_OUT / "nested", second["episode_id"])
        if left_episode["reason"] == "infrastructure_error" or right_episode["reason"] == "infrastructure_error":
            raise RuntimeError("development equivalence cannot accept infrastructure errors")
        left = _cycles(ROOT_OUT / "single", first["episode_id"])
        right = _cycles(ROOT_OUT / "nested", second["episode_id"])
        left_event = _first_event(left)
        right_event = _first_event(right)
        if (left_event is None) != (right_event is None):
            raise RuntimeError(f"paired first-event trigger disagreement: {first['pair_id']}")
        if left_event is None:
            prefix = min(len(left), len(right))
            if (
                [_normal_cycle(row) for row in left]
                != [_normal_cycle(row) for row in right]
                or left_episode["final_success"] != right_episode["final_success"]
                or left_episode["reason"] != right_episode["reason"]
                or left_episode["cycles"] != right_episode["cycles"]
            ):
                raise RuntimeError(
                    f"untriggered paired execution diverged: {first['pair_id']}"
                )
            untriggered_full_execution_exact += 1
        else:
            if left_event[0] != right_event[0] or _normal_event(left_event[1]) != _normal_event(right_event[1]):
                raise RuntimeError(f"paired first event changed: {first['pair_id']}")
            prefix = left_event[0] + 1
        if [_normal_cycle(row) for row in left[:prefix]] != [
            _normal_cycle(row) for row in right[:prefix]
        ]:
            raise RuntimeError(f"trajectory changed before paired first event: {first['pair_id']}")
        paired_prefix_exact += 1
        level = int(first["task_level"])
        if level == 1:
            if (
                [_normal_cycle(row) for row in left] != [_normal_cycle(row) for row in right]
                or left_episode["final_success"] != right_episode["final_success"]
                or left_episode["reason"] != right_episode["reason"]
                or left_episode["cycles"] != right_episode["cycles"]
            ):
                raise RuntimeError(f"one-stage nested execution is not equivalent: {first['pair_id']}")
            one_stage_exact += 1
        audit = right_episode.get("horizon_audit") or {}
        later_opportunities += int(audit.get("later_event_opportunities") or 0)
        records.append(
            {
                "pair_id": first["pair_id"],
                "task": first["task"],
                "task_level": level,
                "paired_prefix_cycles": prefix,
                "paired_first_event_triggered": left_event is not None,
                "later_event_opportunities": int(audit.get("later_event_opportunities") or 0),
                "one_stage_full_execution_equal": level == 1,
                "untriggered_full_execution_equal": left_event is None,
            }
        )
    expected_pairs = len(single_rows)
    expected_one_stage = sum(int(row["task_level"]) == 1 for row in single_rows)
    if one_stage_exact != expected_one_stage or paired_prefix_exact != expected_pairs:
        raise RuntimeError("nested development equivalence coverage is incomplete")
    if later_opportunities < 1:
        raise RuntimeError("development run did not exercise any later event opportunity")
    gate = {
        "schema": "essay2608.iclr2027.a6-e4-nested-development-gate.v1",
        "status": "PASS",
        "development_only": True,
        "sealed_results_read": False,
        "pairs": len(records),
        "paired_prefix_exact": paired_prefix_exact,
        "one_stage_full_execution_exact": one_stage_exact,
        "untriggered_full_execution_exact": untriggered_full_execution_exact,
        "later_event_opportunities_exercised": later_opportunities,
        "formal_manifest_sha256": plan["formal_manifest_sha256"],
        "protocol_files": plan["protocol_files"],
        "records": records,
    }
    _atomic_json(DEVELOPMENT_GATE, gate)
    return gate


def status() -> dict[str, Any]:
    plan = prepare()
    cells = {}
    for key in ("single", "nested"):
        count = len(list((ROOT_OUT / key / "episodes").glob("*.json")))
        cells[key] = {"completed": count, "expected": int(plan["pairs"])}
    return {"cells": cells, "gate": _json(DEVELOPMENT_GATE) if DEVELOPMENT_GATE.is_file() else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "validate", "status"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare()
    elif args.command == "run":
        run()
        value = status()
    elif args.command == "validate":
        value = validate()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
