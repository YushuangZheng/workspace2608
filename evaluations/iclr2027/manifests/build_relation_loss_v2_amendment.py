"""Build the source-conditioned maintained-contact relation-loss amendments.

The episode assignments remain unchanged.  This result-blind view selects the
two task-by-fault cells whose runtime interaction source is maintained contact;
the protocol revision lives in the public fault configuration and injector.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_ROOT = ROOT / "evaluations" / "iclr2027" / "manifests"
AMENDMENT_ROOT = MANIFEST_ROOT / "amendments"
TASKS = ("open_drawer", "bimanual_lift_tray")
FAULT_FAMILY = "relation_loss"
REVISION = "relation_loss_source_conditioned_physics_v2"
SOURCES = {
    "main10_perturbed": MANIFEST_ROOT / "main10_perturbed.jsonl",
    "main10_failure_train": MANIFEST_ROOT / "main10_failure_train.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["task"] in TASKS and row["fault_family"] == FAULT_FAMILY:
                rows.append(row)
    counts = {task: sum(row["task"] == task for row in rows) for task in TASKS}
    if counts != {task: 50 for task in TASKS}:
        raise RuntimeError(f"unexpected amendment counts for {path}: {counts}")
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def build() -> dict:
    AMENDMENT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, source in SOURCES.items():
        output = AMENDMENT_ROOT / f"{REVISION}_{name}.jsonl"
        rows = _selected(source)
        _write_jsonl(output, rows)
        outputs[name] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": _sha256(source),
            "path": str(output.relative_to(ROOT)),
            "rows": len(rows),
            "counts_by_task": {task: 50 for task in TASKS},
            "sha256": _sha256(output),
        }
    index = {
        "schema": "essay2608.iclr2027.relation-loss-amendment.v1",
        "revision": REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_read_episode_results": False,
        "episode_assignments_changed": False,
        "tasks": list(TASKS),
        "fault_family": FAULT_FAMILY,
        "outputs": outputs,
    }
    index_path = AMENDMENT_ROOT / f"{REVISION}_INDEX.json"
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(index_path)
    return index


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
