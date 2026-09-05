"""Build small causal-only adapter fixtures from development rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluations.iclr2027.interfaces.feature_schema import validate_feature_record
from evaluations.iclr2027.runners.episode_io import (
    load_cycles,
    load_episode,
    resolve_cycle_file,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "development_examples"
DEVELOPMENT = Path("evaluations/iclr2027/results/development").resolve()
EPISODES = (
    "development_nominal/close_jar/0000",
    "development_perturbed/close_jar/0000",
    "development_nominal/bimanual_handover_item/0000",
    "development_perturbed/bimanual_handover_item/0000",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    sources = []
    for episode_id in EPISODES:
        safe = episode_id.replace("/", "__")
        result_path = DEVELOPMENT / "episodes" / (safe + ".json")
        result = load_episode(result_path)
        cycles = load_cycles(resolve_cycle_file(result_path, result))
        trigger_cycle = result.get("audit", {}).get("violation_onset_cycle")
        wanted = {0, min(1, len(cycles) - 1), len(cycles) - 1}
        if trigger_cycle is not None:
            wanted.update(
                cycle
                for cycle in (trigger_cycle - 1, trigger_cycle, trigger_cycle + 1)
                if 0 <= cycle < len(cycles)
            )
        for cycle in sorted(wanted):
            feature = validate_feature_record(cycles[cycle]["feature"])
            records.append(feature)
        sources.append(
            {
                "episode_id": episode_id,
                "condition": result["condition"],
                "task": result["task"],
                "selected_feature_cycles": sorted(wanted),
            }
        )
    record_path = OUTPUT / "causal_records.jsonl"
    record_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    index = {
        "schema": "essay2608.iclr2027.development-fixtures.v1",
        "feature_schema": records[0]["schema"],
        "records": len(records),
        "causal_records_sha256": _sha256(record_path),
        "sources": sources,
        "contains_evaluator_labels": False,
        "contains_calibration_or_sealed_records": False,
    }
    (OUTPUT / "FIXTURE_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(records), "output": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
