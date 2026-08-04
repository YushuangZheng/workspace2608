"""Read-only hard audit of pre-registered bimanual online-relation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from essay2608.eval.bimanual_relation_audit import audit_bimanual_relation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/bimanual_relation_protocol_v1.json"),
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/bimanual_relation/formal_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_bimanual_relation_results(args.results_dir, args.protocol)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "entries"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
