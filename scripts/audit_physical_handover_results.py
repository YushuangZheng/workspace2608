"""只读审计预注册的真实物理双臂交接 v2 正式结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from essay2608.eval.physical_handover_audit import (
    V2_SEEDS,
    V2_SOURCE_SHA256,
    V3_SEEDS,
    V3_SOURCE_SHA256,
    audit_physical_handover_run,
)


FROZEN_RUNS = {
    "v2": {
        "summary": Path("outputs/physical_handover/formal_v2/summary.json"),
        "seeds": V2_SEEDS,
        "source_sha256": V2_SOURCE_SHA256,
        "successes": 18,
    },
    "v3": {
        "summary": Path("outputs/physical_handover/formal_v3/summary.json"),
        "seeds": V3_SEEDS,
        "source_sha256": V3_SOURCE_SHA256,
        "successes": 20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=tuple(FROZEN_RUNS), default="v2")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = FROZEN_RUNS[args.version]
    result = audit_physical_handover_run(
        args.summary or frozen["summary"],
        expected_seeds=frozen["seeds"],
        expected_source_sha256=frozen["source_sha256"],
        expected_successes=frozen["successes"],
    )
    print(f"真实物理双臂交接 {args.version} 正式结果审计通过。")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
