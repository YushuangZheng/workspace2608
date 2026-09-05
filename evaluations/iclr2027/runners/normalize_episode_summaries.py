"""Normalize pre-freeze A2 episode summaries to the final logger fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from evaluations.iclr2027.runners.episode_io import (
    load_cycles,
    load_episode,
    resolve_cycle_file,
)
from integrations.rlbench.rlbench_dynamac.core.paths import INTEGRATION_ROOT, REPOSITORY_ROOT

FAULT_CONFIG = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "configs" / "shared" / "faults.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_identity(task: str) -> dict:
    root = (
        INTEGRATION_ROOT / "models" / "v4" / task
        if task.startswith("bimanual_")
        else INTEGRATION_ROOT / "models" / "iclr2027" / "dynamac" / task
    )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return {
        "model_root": str(root.relative_to(REPOSITORY_ROOT)),
        "artifact_sha256": digest.hexdigest(),
        "files": [str(path.relative_to(root)) for path in files],
    }


def normalize(root: Path) -> int:
    identities = {}
    count = 0
    for path in sorted((root / "episodes").glob("*.json")):
        value = load_episode(path)
        task = value["task"]
        identities.setdefault(task, _tree_identity(task))
        audit = value.get("audit", {})
        if "fault_protocol" not in value and value.get("fault_family") is not None:
            final_protocol = None
            events = []
            seen_events = set()
            for cycle in load_cycles(resolve_cycle_file(path, value)):
                injector = cycle.get("execution", {}).get("injector")
                if not isinstance(injector, dict):
                    continue
                final_protocol = {**injector, "events": []}
                for event in injector.get("events", ()):
                    identity = json.dumps(event, sort_keys=True, separators=(",", ":"))
                    if identity not in seen_events:
                        seen_events.add(identity)
                        events.append(event)
            if final_protocol is not None:
                final_protocol["events"] = events
            value["fault_protocol"] = final_protocol
        value.update(
            {
                "method_config_identity": {
                    "policy_model": identities[task],
                    "fault_config_sha256": _sha256(FAULT_CONFIG),
                },
                "final_success": bool(value["success"]),
                "termination_reason": value["reason"],
                "recovery_cycles": int(value.get("recovery_cycles", 0)),
                "first_alarm_cycle": value.get("first_alarm_cycle"),
                "false_interventions": int(value.get("false_interventions", 0)),
                "relation_restored_cycle": audit.get("relation_restored_cycle"),
                "legal_reentry_cycle": audit.get("legal_reentry_cycle"),
                "post_reentry_completion": value.get("post_reentry_completion"),
            }
        )
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(str(temporary), str(path))
        count += 1
    metadata_path = root / "RUN_METADATA.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "simulator_api": "in_process_pyrep",
                "policy_worker_transport": "private_stdio_pipe_per_episode",
                "network_control_port": None,
            }
        )
        temporary = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(str(temporary), str(metadata_path))
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="+", type=Path)
    args = parser.parse_args(argv)
    counts = {str(root): normalize(root) for root in args.root}
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
