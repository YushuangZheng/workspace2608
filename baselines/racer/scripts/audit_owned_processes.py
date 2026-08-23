#!/usr/bin/env python3
"""Record and reject processes that retain an exact RACER ownership marker."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _redacted_command(payload: bytes | None) -> list[str]:
    if not payload:
        return []
    arguments = [item.decode("utf-8", errors="replace") for item in payload.split(b"\0") if item]
    for index, argument in enumerate(arguments[:-1]):
        if argument == "--authkey-hex":
            arguments[index + 1] = "<redacted>"
    return arguments


def _classify(arguments: list[str]) -> list[str]:
    text = " ".join(arguments).lower()
    categories = []
    patterns = {
        "coppeliasim": r"coppeliasim|libcoppeliasim",
        "simulator_worker": r"isolated_simulator_worker",
        "language_service": r"racer_lm_server",
        "vision_language_service": r"racer_llava_server",
        "evaluator": r"rollout\.py|instrumented_rollout|isolated_rollout",
        "xvfb": r"(?:^|/)xvfb(?:\s|$)",
    }
    for category, pattern in patterns.items():
        if re.search(pattern, text):
            categories.append(category)
    return categories or ["other"]


def scan(environment_key: str, value: str, ignored: set[int]):
    marker = f"{environment_key}={value}".encode()
    processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in ignored or pid == os.getpid():
            continue
        environ = _read_bytes(entry / "environ")
        if environ is None or marker not in environ.split(b"\0"):
            continue
        command = _redacted_command(_read_bytes(entry / "cmdline"))
        processes.append(
            {"pid": pid, "command": command, "categories": _classify(command)}
        )
    return sorted(processes, key=lambda item: item["pid"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-key", default="RACER_RUN_ID")
    parser.add_argument("--value", required=True)
    parser.add_argument("--ignore-pid", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.environment_key):
        raise SystemExit("environment key must be an uppercase identifier")
    if not args.value or "\0" in args.value:
        raise SystemExit("ownership marker value is invalid")
    processes = scan(args.environment_key, args.value, set(args.ignore_pid))
    payload = {
        "schema": "racer_owned_process_audit_v1",
        "environment_key": args.environment_key,
        "value": args.value,
        "residual_count": len(processes),
        "ok": not processes,
        "processes": processes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
