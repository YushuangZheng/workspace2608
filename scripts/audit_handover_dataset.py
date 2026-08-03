"""Audit, fingerprint, and optionally freeze the handover dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from essay2608.data import audit_bimanual_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/handover_static/v2"))
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--dataset_version", type=str)
args = parser.parse_args()


def main() -> None:
    data_dir = args.data_dir.resolve()
    result = audit_bimanual_dataset(data_dir)
    print(json.dumps(result, indent=2))
    if not args.freeze:
        return
    if (data_dir / "FROZEN").exists():
        raise RuntimeError(f"Refusing to overwrite frozen dataset: {data_dir}")
    manifest_path = data_dir / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dataset_version = args.dataset_version or f"handover_static_{data_dir.name}"
    manifest = {
        **original,
        "frozen": True,
        "dataset_version": dataset_version,
        "dataset_sha256": result["dataset_sha256"],
        "source_git_commit": commit,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance": {key: value for key, value in result.items() if key not in {"entries", "dataset_sha256"}},
        "demos": result["entries"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (data_dir / "FROZEN").write_text(
        f"{dataset_version}\nsha256={result['dataset_sha256']}\n",
        encoding="utf-8",
    )
    print(f"Frozen {data_dir} as {result['dataset_sha256']}")


if __name__ == "__main__":
    main()
