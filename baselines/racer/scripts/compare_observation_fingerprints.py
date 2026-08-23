#!/usr/bin/env python3
"""Require exact initialization-observation equality across RACER backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def first_differences(left: Any, right: Any, path: str = "snapshot", limit: int = 20):
    differences: list[str] = []

    def visit(a: Any, b: Any, current: str) -> None:
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append(f"{current}: type {type(a).__name__} != {type(b).__name__}")
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    differences.append(f"{current}.{key}: missing on direct side")
                elif key not in b:
                    differences.append(f"{current}.{key}: missing on isolated side")
                else:
                    visit(a[key], b[key], f"{current}.{key}")
                if len(differences) >= limit:
                    return
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(f"{current}: length {len(a)} != {len(b)}")
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{current}[{index}]")
                if len(differences) >= limit:
                    return
        elif a != b:
            differences.append(f"{current}: {a!r} != {b!r}")

    visit(left, right, path)
    return differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--isolated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    direct = json.loads(args.direct.read_text(encoding="utf-8"))
    isolated = json.loads(args.isolated.read_text(encoding="utf-8"))
    required = ("schema", "task", "episode", "snapshot")
    for name, payload in (("direct", direct), ("isolated", isolated)):
        missing = [key for key in required if key not in payload]
        if missing:
            raise SystemExit(f"{name} capture is missing keys: {missing}")
    metadata_match = all(direct[key] == isolated[key] for key in required[:-1])
    differences = first_differences(direct["snapshot"], isolated["snapshot"])
    report = {
        "ok": metadata_match and not differences,
        "schema": "racer_initial_observation_comparison_v1",
        "direct_capture": str(args.direct),
        "isolated_capture": str(args.isolated),
        "metadata_match": metadata_match,
        "differences": differences,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
