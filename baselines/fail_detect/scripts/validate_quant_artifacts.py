#!/usr/bin/env python3
"""Validate Transport data and strict-load the external official DP checkpoint."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CAMERA_KEYS = (
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
    "shouldercamera0_image",
    "shouldercamera1_image",
)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_git(upstream, expected_commit):
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_commit:
        raise RuntimeError("upstream commit mismatch: {} != {}".format(actual, expected_commit))
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        proc = subprocess.run(["git", "-C", str(upstream)] + list(args), check=False)
        if proc.returncode != 0:
            raise RuntimeError("tracked changes exist in the pinned upstream checkout")
    return actual


def validate_dataset(dataset_path, upstream):
    import h5py
    import numpy as np

    if not dataset_path.is_file():
        raise RuntimeError("missing dataset: {}".format(dataset_path))
    with dataset_path.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError("invalid HDF5 magic: {}".format(dataset_path))

    sys.path.insert(0, str(upstream))
    from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_actions
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    with h5py.File(str(dataset_path), "r") as handle:
        if "data" not in handle:
            raise RuntimeError("HDF5 has no data group")
        demos = sorted(handle["data"].keys())
        if not demos:
            raise RuntimeError("HDF5 contains no demonstrations")
        demo = handle["data"][demos[0]]
        raw_actions = np.asarray(demo["actions"][:2])
        if raw_actions.ndim != 2 or raw_actions.shape[-1] != 14:
            raise RuntimeError("expected dual-arm raw action dimension 14; found {}".format(raw_actions.shape))
        converted = _convert_actions(raw_actions, True, RotationTransformer("axis_angle", "rotation_6d"))
        if converted.shape[-1] != 20:
            raise RuntimeError("absolute-action conversion did not produce dimension 20")
        if "obs" not in demo:
            raise RuntimeError("first demonstration has no obs group")
        observations = demo["obs"]
        camera_shapes = {}
        for key in CAMERA_KEYS:
            if key not in observations:
                raise RuntimeError("missing camera observation: {}".format(key))
            shape = tuple(observations[key].shape)
            if len(shape) != 4 or shape[1:] != (84, 84, 3):
                raise RuntimeError("unexpected {} shape: {}".format(key, shape))
            camera_shapes[key] = shape
        env_args = handle["data"].attrs.get("env_args")
        if env_args is None:
            raise RuntimeError("dataset is missing data.env_args")
    return {
        "demonstrations": len(demos),
        "first_demo": demos[0],
        "raw_action_dimension": 14,
        "converted_action_dimension": int(converted.shape[-1]),
        "camera_shapes": {key: list(value) for key, value in camera_shapes.items()},
    }


def strict_load_checkpoint(checkpoint, config_path, upstream):
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    sys.path.insert(0, str(upstream))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill)
    for key in ("cfg", "state_dicts"):
        if key not in payload:
            raise RuntimeError("checkpoint payload missing {}".format(key))
    for key in ("model", "ema_model"):
        if key not in payload["state_dicts"]:
            raise RuntimeError("checkpoint state_dicts missing {}".format(key))

    checkpoint_cfg = payload["cfg"]
    checkpoint_action_dim = int(checkpoint_cfg.shape_meta.action.shape[0])
    if checkpoint_action_dim != 20:
        raise RuntimeError("checkpoint action dimension is not 20")

    local_cfg = OmegaConf.load(str(config_path))
    OmegaConf.resolve(local_cfg)
    if int(local_cfg.shape_meta.action.shape[0]) != 20:
        raise RuntimeError("local FAIL-Detect config action dimension is not 20")
    workspace_class = hydra.utils.get_class(local_cfg._target_)
    workspace = workspace_class(local_cfg, output_dir=str(config_path.parent / ".strict_load_tmp"))
    model_result = workspace.model.load_state_dict(payload["state_dicts"]["model"], strict=True)
    ema_result = workspace.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    if model_result.missing_keys or model_result.unexpected_keys or ema_result.missing_keys or ema_result.unexpected_keys:
        raise RuntimeError("strict checkpoint load reported incompatible keys")
    parameters = sum(parameter.numel() for parameter in workspace.model.parameters())
    return {
        "checkpoint_workspace": str(checkpoint_cfg._target_),
        "local_workspace": str(local_cfg._target_),
        "action_dimension": checkpoint_action_dim,
        "model_parameters": int(parameters),
        "strict_model_load": True,
        "strict_ema_load": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    upstream = repo_root / "baselines/fail_detect/upstream"
    manifest_path = repo_root / "baselines/fail_detect/quant_artifacts.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with args.artifact_lock.open("r", encoding="utf-8") as handle:
        artifact_lock = json.load(handle)
    if artifact_lock.get("protocol_label") != manifest["protocol_label"]:
        raise RuntimeError("artifact lock protocol mismatch")

    dataset_path = Path(artifact_lock["artifacts"]["transport_image_abs"]["path"])
    checkpoint_path = Path(artifact_lock["artifacts"]["transport_ph_dp_checkpoint"]["path"])
    if sha256_file(dataset_path) != artifact_lock["artifacts"]["transport_image_abs"]["sha256"]:
        raise RuntimeError("dataset SHA-256 no longer matches artifact lock")
    if sha256_file(checkpoint_path) != artifact_lock["artifacts"]["transport_ph_dp_checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint SHA-256 no longer matches artifact lock")

    result = {
        "schema": "dynamac-fail-detect-quant-validation-v1",
        "protocol_label": manifest["protocol_label"],
        "upstream_commit": validate_git(upstream, manifest["fail_detect_source"]["commit"]),
        "dataset": validate_dataset(dataset_path, upstream),
        "checkpoint": strict_load_checkpoint(
            checkpoint_path,
            upstream / "diffusion_policy/configs_robomimic/image_transport_ph_visual_diffusion_policy_cnn.yaml",
            upstream,
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
