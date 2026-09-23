"""Validate and aggregate the seven appendix-level E5 ablations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.analysis.rebuild_a5_audit import (
    AUDITOR_PATH,
    reaudit_episode,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
E5_ROOT = EVAL_ROOT / "results" / "controlled" / "e5"
A5_ROOT = EVAL_ROOT / "results" / "controlled" / "e1_e2" / "end_to_end"
PLAN = E5_ROOT / "A6_E5_FINE_RUN_PLAN.json"
OUTPUT = E5_ROOT / "derived"
STRESS4 = ("open_drawer", "place_cups_3", "bimanual_handover_item", "bimanual_lift_tray")
BIMANUAL = frozenset(("bimanual_handover_item", "bimanual_lift_tray"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe(value: str) -> str:
    return str(value).replace("/", "__")


def _episode(root: Path, episode_id: str) -> dict[str, Any]:
    path = root / "episodes" / f"{_safe(episode_id)}.json"
    if not path.is_file():
        raise RuntimeError(f"missing E5 fine episode: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_source_episode_path"] = str(path)
    return value


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_new(
    key: str,
    condition: str,
    manifest: list[Mapping[str, Any]],
    config_sha256: str,
) -> list[dict[str, Any]]:
    root = E5_ROOT / "fine" / key / condition
    expected_ids = {_safe(str(row["episode_id"])) for row in manifest}
    actual_ids = {path.stem for path in (root / "episodes").glob("*.json")}
    if actual_ids != expected_ids:
        raise RuntimeError(f"E5 fine {key} result set mismatch: missing={len(expected_ids-actual_ids)}, extra={len(actual_ids-expected_ids)}")
    output = []
    for row in manifest:
        result = _episode(root, str(row["episode_id"]))
        if (
            result.get("episode_id") != row["episode_id"]
            or result.get("task") != row["task"]
            or int(result.get("seed")) != int(row["seed"])
            or int(result.get("variation")) != int(row["variation"])
            or result.get("method_config_identity", {}).get("sha256") != config_sha256
        ):
            raise RuntimeError(f"E5 fine identity mismatch: {key}/{row['episode_id']}")
        output.append(result)
    return output


def _load_full(manifest: list[Mapping[str, Any]], condition: str) -> list[dict[str, Any]]:
    root = A5_ROOT / "m5" / condition
    output = []
    for row in manifest:
        source = str(row.get("source_episode_id") or row["episode_id"])
        result = _episode(root, source)
        if result.get("episode_id") != source or result.get("task") != row["task"] or int(result.get("seed")) != int(row["seed"]) or int(result.get("variation")) != int(row["variation"]):
            raise RuntimeError(f"E5 fine Full reuse mismatch: {source}")
        output.append(result)
    return output


def _macro(rows: Iterable[Mapping[str, Any]], tasks: Iterable[str]) -> float:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    rates = []
    for task in tasks:
        values = grouped[task]
        if not values:
            raise RuntimeError(f"empty E5 fine task cell: {task}")
        rates.append(sum(bool(value["final_success"]) for value in values) / len(values))
    return mean(rates)


def generate() -> dict[str, Any]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    manifests = {condition: _rows(ROOT / str(plan["manifests"][condition]["path"])) for condition in ("nominal", "perturbed")}
    methods = [("full", "Full method", None)] + [(str(item["key"]), str(item["paper_name"]), str(item["sha256"])) for item in plan["methods"]]
    loaded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, _name, sha in methods:
        for condition in ("nominal", "perturbed"):
            loaded[(key, condition)] = (
                _load_full(manifests[condition], condition)
                if key == "full"
                else _load_new(key, condition, manifests[condition], str(sha))
            )

    current_audits = {
        (key, str(row["episode_id"])): reaudit_episode(
            key, Path(str(row["_source_episode_path"])), row
        )
        for key, _name, _sha in methods
        for row in loaded[(key, "perturbed")]
    }

    def physically_triggered(key: str, row: Mapping[str, Any]) -> bool:
        return bool(current_audits[(key, str(row["episode_id"]))]["physically_triggered"])

    summary_rows = []
    complete = []
    for key, name, _sha in methods:
        nominal = loaded[(key, "nominal")]
        perturbed = loaded[(key, "perturbed")]
        triggered = [row for row in perturbed if physically_triggered(key, row)]
        summary_rows.append({
            "variant_key": key,
            "variant": name,
            "nominal_task_macro_success": _macro(nominal, STRESS4),
            "perturbed_task_macro_success": _macro(perturbed, STRESS4),
            "bimanual_perturbed_task_macro_success": _macro(perturbed, BIMANUAL),
            "physically_triggered_episodes": len(triggered),
            "recovery_completion": sum(bool(row["final_success"]) for row in triggered) / len(triggered) if triggered else 0.0,
        })
        for condition, values in (("nominal", nominal), ("perturbed", perturbed)):
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in values:
                grouped[str(row["task"])].append(row)
            for task in STRESS4:
                cell = grouped[task]
                successes = sum(bool(row["final_success"]) for row in cell)
                complete.append({
                    "variant_key": key,
                    "variant": name,
                    "condition": condition,
                    "task": task,
                    "successes": successes,
                    "episodes": len(cell),
                    "success_rate": successes / len(cell),
                    "physically_triggered": (
                        sum(physically_triggered(key, row) for row in cell)
                        if condition == "perturbed"
                        else 0
                    ),
                    "infrastructure_errors": sum(row.get("termination_reason") == "infrastructure_error" for row in cell),
                })
    _atomic_csv(OUTPUT / "appendix_fine_ablation.csv", summary_rows)
    _atomic_csv(OUTPUT / "appendix_fine_ablation_complete.csv", complete)
    summary = {
        "schema": "essay2608.iclr2027.a6-e5-fine-analysis.v1",
        "status": "PASS",
        "new_episodes": int(plan["expected_new_episodes"]),
        "evaluated_episodes": int(plan["expected_evaluated_episodes"]),
        "reused_from_e2_episodes": int(plan["reused_from_e2"]["episodes"]),
        "full_reused_from_e1": True,
        "physical_audit": {
            "path": str(AUDITOR_PATH.relative_to(ROOT)),
            "sha256": _sha256(AUDITOR_PATH),
            "source": "reconstructed_from_retained_fault_protocol_and_cycle_evidence",
        },
        "outputs": {
            "appendix_fine_ablation.csv": _sha256(OUTPUT / "appendix_fine_ablation.csv"),
            "appendix_fine_ablation_complete.csv": _sha256(OUTPUT / "appendix_fine_ablation_complete.csv"),
        },
    }
    _atomic_json(OUTPUT / "E5_FINE_ANALYSIS.json", summary)
    return summary


def main() -> int:
    print(json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
