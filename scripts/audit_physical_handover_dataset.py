"""Audit and optionally freeze a contact-rich physical handover dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from essay2608.data.physical_handover import audit_physical_handover_dataset
from essay2608.eval.physical_handover_audit import V3_SOURCE_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data/handover_physical/v1")
    )
    parser.add_argument("--expected_seeds", nargs="+", type=int)
    parser.add_argument("--expected_source_sha256", default=V3_SOURCE_SHA256)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dataset_version", default="handover_physical_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    audit = audit_physical_handover_dataset(
        data_dir,
        expected_seeds=args.expected_seeds,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "entries"}, indent=2))
    if not args.freeze:
        return
    frozen_path = data_dir / "FROZEN"
    if frozen_path.exists():
        raise RuntimeError(f"拒绝覆盖已冻结物理数据集：{data_dir}")
    manifest_path = data_dir / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        **original,
        "frozen": True,
        "dataset_version": args.dataset_version,
        "dataset_sha256": audit["dataset_sha256"],
        "source_git_commit": commit,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance": {key: value for key, value in audit.items() if key != "entries"},
        "demos": audit["entries"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frozen_path.write_text(
        f"{args.dataset_version}\nsha256={audit['dataset_sha256']}\n",
        encoding="utf-8",
    )
    verification = audit_physical_handover_dataset(
        data_dir,
        expected_seeds=args.expected_seeds,
        expected_source_sha256=args.expected_source_sha256,
    )
    if verification["dataset_sha256"] != audit["dataset_sha256"]:
        raise RuntimeError("冻结后物理数据集哈希发生变化")
    print(f"物理交接数据已冻结：{audit['dataset_sha256']}")


if __name__ == "__main__":
    main()
