"""Audit, fingerprint, and optionally freeze the handover dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from essay2608.data import audit_bimanual_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/handover_static/v1"))
parser.add_argument("--freeze", action="store_true")
args = parser.parse_args()


def main() -> None:
    data_dir = args.data_dir.resolve()
    result = audit_bimanual_dataset(data_dir)
    print(json.dumps(result, indent=2))
    if not args.freeze:
        return
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
        "dataset_version": "handover_static_v1",
        "dataset_sha256": result["dataset_sha256"],
        "source_git_commit": commit,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance": {key: value for key, value in result.items() if key not in {"entries", "dataset_sha256"}},
        "demos": result["entries"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (data_dir / "FROZEN").write_text(
        f"handover_static_v1\nsha256={result['dataset_sha256']}\n",
        encoding="utf-8",
    )
    print(f"Frozen {data_dir} as {result['dataset_sha256']}")


if __name__ == "__main__":
    main()
