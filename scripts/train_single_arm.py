"""Fit and persist deterministic single-arm Gaussian policy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from essay2608.data.dataset import load_dataset
from essay2608.policy import (
    DynaMACPolicy,
    MaskOnlyPolicy,
    SkillDynaMACPolicy,
    StaticMultiStreamPolicy,
    WorldGaussianPolicy,
)


POLICIES = {
    "world_gaussian": WorldGaussianPolicy,
    "static_multistream": StaticMultiStreamPolicy,
    "skill_dynamac": SkillDynaMACPolicy,
    "mask_only": MaskOnlyPolicy,
    "full_dynamac": DynaMACPolicy,
}


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, default=Path("data/pick_place_static/v1"))
parser.add_argument("--output_dir", type=Path, default=Path("outputs/single_arm_models/v1"))
parser.add_argument("--bins", type=int, default=25)
args = parser.parse_args()


def artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    demonstrations, dataset_manifest = load_dataset(args.data_dir, verify_hashes=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    for name, policy_type in POLICIES.items():
        policy = policy_type(bins=args.bins)
        policy.fit(demonstrations)
        arrays = {"phase_durations": policy.phase_durations}
        if isinstance(policy, WorldGaussianPolicy):
            models = {"world": policy.model}
        else:
            models = policy.models
        for frame_name, model in models.items():
            if model is None:
                continue
            arrays[f"{frame_name}__mean_pose"] = model.mean_pose
            arrays[f"{frame_name}__position_covariance"] = model.position_covariance
            arrays[f"{frame_name}__pose_covariance"] = model.pose_covariance
            arrays[f"{frame_name}__mean_gripper"] = model.mean_gripper

        artifact_path = output_dir / f"{name}.npz"
        np.savez_compressed(artifact_path, **arrays)
        artifacts.append(
            {
                "method": name,
                "file": artifact_path.name,
                "sha256": artifact_hash(artifact_path),
                "frames": sorted(models),
                **(
                    {"skill_diagnostics": policy.skill_diagnostics}
                    if isinstance(policy, SkillDynaMACPolicy)
                    else {}
                ),
            }
        )

    manifest = {
        "dataset_name": dataset_manifest["dataset_name"],
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "bins_per_phase": args.bins,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved fitted policies to {output_dir}")


if __name__ == "__main__":
    main()
