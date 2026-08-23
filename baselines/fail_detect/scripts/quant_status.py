#!/usr/bin/env python3
"""Atomically update or print the ignored FAIL-Detect pipeline status file."""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


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


def load_status(path):
    if not path.exists():
        return {"schema": "dynamac-fail-detect-quant-status-v1", "history": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("status_file", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update")
    update.add_argument("--state", required=True)
    update.add_argument("--stage", required=True)
    update.add_argument("--detail", default="")
    update.add_argument("--pid", type=int, default=os.getppid())
    update.add_argument("--extra-json")

    sub.add_parser("show")
    args = parser.parse_args()

    if args.command == "show":
        print(json.dumps(load_status(args.status_file), indent=2, sort_keys=True))
        return

    status = load_status(args.status_file)
    timestamp = now_iso()
    if "started_at" not in status:
        status["started_at"] = timestamp
    event = {
        "timestamp": timestamp,
        "state": args.state,
        "stage": args.stage,
        "detail": args.detail,
        "pid": args.pid,
    }
    if args.extra_json:
        event["extra"] = json.loads(args.extra_json)
    status.update(event)
    status["updated_at"] = timestamp
    history = status.setdefault("history", [])
    history.append(event)
    status["history"] = history[-100:]
    atomic_json(args.status_file, status)


if __name__ == "__main__":
    main()
