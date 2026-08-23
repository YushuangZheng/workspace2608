#!/usr/bin/env python3
"""Record at most one reactive compatibility repair for the bounded run."""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_file", type=Path)
    parser.add_argument("command", choices=("check", "register", "show"))
    parser.add_argument("description", nargs="?")
    args = parser.parse_args()
    payload = {"schema": "dynamac-fail-detect-compatibility-repair-v1", "repairs": []}
    if args.state_file.exists():
        with args.state_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    repairs = payload.get("repairs", [])
    if len(repairs) > 1:
        raise RuntimeError("more than one compatibility repair is recorded")
    if args.command == "show":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "check":
        return
    if not args.description:
        parser.error("register requires a description")
    if repairs:
        raise RuntimeError("one compatibility repair was already used; stop the run")
    repairs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": args.description,
    })
    payload["repairs"] = repairs
    atomic_json(args.state_file, payload)


if __name__ == "__main__":
    main()
