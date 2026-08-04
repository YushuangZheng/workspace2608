"""Audit and freeze the versioned single-arm demonstration dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from essay2608.data.dataset import audit_dataset, load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--seeds", type=int, nargs="*", default=[2608, 2609, 2610, 2611, 2612])
args = parser.parse_args()


def current_commit() -> str:
    """Return the source commit used for dataset acceptance."""

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    dataset_dir = args.data_dir.resolve()
    result = audit_dataset(dataset_dir, seeds=args.seeds)

    if args.freeze:
        frozen_marker = dataset_dir / "FROZEN"
        existing_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        if frozen_marker.exists() or existing_manifest.get("frozen", False):
            raise RuntimeError(f"Dataset is already frozen: {dataset_dir}")

        manifest = {
            "dataset_name": "pick_place_static",
            "dataset_version": "v1",
            "frozen": True,
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_git_commit": current_commit(),
            "task_id": existing_manifest.get("task_id", "Essay2608-Dynamic-Pick-Place-v0"),
            "num_demos": len(result["entries"]),
            "success_threshold_m": 0.06,
            "quaternion_order": "wxyz",
            "coordinate_frame": "local_environment",
            "dataset_sha256": result["dataset_sha256"],
            "acceptance": {
                "status": "passed",
                "required_state_sequence": list(range(10)),
                "minimum_initial_object_distance_m": result["minimum_initial_object_distance_m"],
                "max_final_error_m": result["max_final_error_m"],
                "max_step_position_jump_m": result["max_step_position_jump_m"],
                "max_postgrasp_position_rms_std_m": result["max_postgrasp_position_rms_std_m"],
            },
            "demos": result["entries"],
        }
        (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        frozen_marker.write_text(
            f"pick_place_static/v1\nsha256={result['dataset_sha256']}\n",
            encoding="utf-8",
        )

    demonstrations, manifest = load_dataset(dataset_dir, verify_hashes=True)
    print(
        json.dumps(
            {
                "dataset": f"{manifest.get('dataset_name', 'unversioned')}/{manifest.get('dataset_version', '?')}",
                "frozen": manifest.get("frozen", False),
                "num_demos": len(demonstrations),
                "dataset_sha256": manifest.get("dataset_sha256", result["dataset_sha256"]),
                "acceptance": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
