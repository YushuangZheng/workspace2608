"""只读审计预注册的真实物理双臂交接 v2 正式结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from essay2608.eval.physical_handover_audit import audit_physical_handover_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/physical_handover/formal_v2/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    result = audit_physical_handover_run(parse_args().summary)
    print("真实物理双臂交接 v2 正式结果审计通过。")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
