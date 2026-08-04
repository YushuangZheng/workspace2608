"""对预注册的单臂关系恢复结果执行硬完整性审计。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from essay2608.eval.recovery_audit import audit_recovery_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/recovery_scientific/v1/summary.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/recovery_protocol_v1.json"),
    )
    parser.add_argument("--protocol_tag", default="recovery-protocol-v1")
    return parser.parse_args()


def tag_commit(tag: str) -> str:
    return subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    result = audit_recovery_run(
        args.summary,
        args.protocol,
        expected_source_commit=tag_commit(args.protocol_tag),
    )
    print("单臂关系恢复正式结果审计通过。")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
